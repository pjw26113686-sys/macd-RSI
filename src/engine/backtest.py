"""단일 이벤트기반 백테스트 엔진 (strategy_spec_v2 §0, §3, §4).

체결 규율(무결성 §0):
  - 신호는 종가 확정 봉(t)에서만 평가한다.
  - 진입 체결가 = 신호 다음 봉(t+1)의 시가. (미래참조 방지)
  - 신호기반 청산(데드크로스 / RSI<50)도 동일 규율: 종가 t 확정 → t+1 시가 체결.
  - 손절·목표가는 지정가(resting) 레벨 → 닿는 봉에서 그 레벨로 즉시 체결(갭 시 시가).
  - 한 봉 안에서 손절·목표가 동시 도달 시 손절 우선(고저 순서 불명 → 최악 가정, §0-4).

진입 셋업 추적(§3.2/3.3/3.4):
  - ENTRY_BASE 불발한 골든크로스에서 RSI<50 → low 셋업, RSI>=70 → high 셋업 무장.
  - 이후 delay_window 봉 내에서 지연진입 조건 충족 시 진입, 초과 시 셋업 폐기.
  - 동일봉 우선순위: ENTRY_BASE > DELAY_HIGH > DELAY_LOW. 무포지션일 때만 진입.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import signals_core
from src.engine import position as pos


@dataclass
class BacktestResult:
    trades: list            # list[pos.Trade]
    equity: pd.Series       # 봉별 자산(현금 + 보유평가액)
    signals: pd.DataFrame   # 지표·신호 컬럼이 붙은 입력 df
    initial_capital: float


def _delay_low_ok(row) -> bool:
    return bool(
        row["macd_above_signal"]
        and row["rsi_cross_up_low"]
        and row["is_bull_candle"]
        and row["hma_uptrend"]
        and row["volume_ok"]
    )


def _delay_high_ok(row) -> bool:
    return bool(
        row["macd_above_signal"]
        and row["rsi_in_support_band"]
        and row["is_bull_candle"]
        and row["hma_uptrend"]
        and row["volume_ok"]
    )


def _warmup_bars(params: dict) -> int:
    """지표가 안정화되는 최소 봉수. 이 구간에서는 진입하지 않는다(재귀편향 방지)."""
    return max(
        params["macd_slow"] + params["macd_signal"],
        params["rsi_period"] + 1,
        params["hma_period"] + int(np.sqrt(params["hma_period"])) + 1,
        params["vol_ma_period"],
        params["swing_lookback_M"],
    )


def run_backtest(
    df: pd.DataFrame,
    params: dict,
    costs: pos.Costs,
    initial_capital: float = 10000.0,
    position_pct: float = 1.0,
) -> BacktestResult:
    """OHLCV df(시간 오름차순)로 지표·신호를 계산한 뒤 백테스트."""
    sig = signals_core.add_signals(df, params)
    sig = sig.reset_index(drop=False)
    if "time" not in sig.columns:
        sig = sig.rename(columns={sig.columns[0]: "time"})
    return run_on_signals(sig, params, costs, initial_capital, position_pct)


def run_on_signals(
    sig: pd.DataFrame,
    params: dict,
    costs: pos.Costs,
    initial_capital: float = 10000.0,
    position_pct: float = 1.0,
) -> BacktestResult:
    """이미 신호 컬럼이 계산된 프레임(`sig`)으로 백테스트.

    `sig`는 open/high/low/close/volume + signals_core.SIGNAL_COLUMNS + 'time'
    컬럼을 가진 RangeIndex DataFrame이어야 한다. (테스트에서 신호를 직접 주입할 때 사용)
    """
    sig = sig.reset_index(drop=True)
    n = len(sig)
    warmup = _warmup_bars(params)
    M = params["swing_lookback_M"]
    lows = sig["low"].to_numpy()

    cash = float(initial_capital)
    position: pos.Position | None = None
    setup = None            # {"kind": "low"|"high", "expires": int}
    pending_entry = False   # 직전 종가에서 확정된 진입을 이번 봉 시가에 체결
    pending_exit = False    # 직전 종가에서 확정된 신호청산을 이번 봉 시가에 체결

    trades: list = []
    equity = np.full(n, np.nan, dtype=float)

    for t in range(n):
        row = sig.iloc[t]
        o, c = row["open"], row["close"]

        # ---- 1) 직전 종가 확정 신호청산 → 이번 봉 시가 체결 ----
        if pending_exit and position is not None:
            cash += _close_all(position, pos.sell_fill(o, costs), costs, t, row["time"],
                               position.trade.last_reason, trades)
            position = None
        pending_exit = False

        # ---- 2) 직전 종가 확정 진입 → 이번 봉 시가 체결 ----
        if pending_entry and position is None:
            entry_fill = pos.buy_fill(o, costs)
            swing_low = float(lows[max(0, t - M):t].min()) if t > 0 else float(lows[t])
            stop_loss, target, _ = pos.compute_stop_and_target(entry_fill, swing_low, params)
            if stop_loss < entry_fill:  # 유효한 손절폭만
                qty = (cash * position_pct) / (entry_fill * (1.0 + costs.fee))
                if qty > 0:
                    cost = pos.buy_notional_cost(entry_fill, qty, costs)
                    cash -= cost
                    trade = pos.Trade(
                        entry_time=row["time"], entry_index=t, entry_price=entry_fill,
                        qty=qty, stop_loss=stop_loss, target=target, cost=cost,
                    )
                    position = pos.Position(entry_fill, qty, stop_loss, target, pos.FULL, trade)
        pending_entry = False

        # ---- 3) 보유 포지션 봉내 관리 ----
        if position is not None:
            cash_delta, status = _manage_open_bar(position, row, costs, t, trades)
            cash += cash_delta
            if status == "closed":
                position = None
            elif status == "signal_pending":
                pending_exit = True  # 데드크로스/RSI<50 확정 → 다음 봉 시가 청산

        # ---- 4) 무포지션 & 워밍업 이후: 종가 t 진입신호 평가 → 다음 봉 시가 체결 ----
        if position is None and not pending_exit and warmup <= t < n - 1:
            decision, setup = _evaluate_entry(row, setup, t, params)
            if decision is not None:
                pending_entry = True

        # ---- 5) 봉별 자산 기록 (현금 + 보유평가액) ----
        equity[t] = cash + (position.qty * c if position is not None else 0.0)

    # 종료 시 잔여 포지션은 마지막 종가로 청산
    if position is not None:
        last = sig.iloc[n - 1]
        cash += _close_all(position, pos.sell_fill(last["close"], costs), costs, n - 1,
                           last["time"], "EOD", trades)
        equity[n - 1] = cash

    equity_series = pd.Series(equity, index=sig["time"], name="equity").ffill().fillna(initial_capital)
    return BacktestResult(trades=trades, equity=equity_series, signals=sig,
                          initial_capital=initial_capital)


# --------------------------------------------------------------------------- #
# 진입 평가 (상태 추적)
# --------------------------------------------------------------------------- #
def _evaluate_entry(row, setup, t: int, params: dict):
    """(decision, setup) 반환. decision in {None,'base','delay_high','delay_low'}."""
    low_th = params["rsi_entry_low"]
    dwin = params["delay_window"]

    # 1) 기본 진입 (최우선)
    if bool(row["entry_base"]):
        return "base", None

    # 2) 무장된 셋업(이전 봉에서 무장)의 만료/발동
    if setup is not None:
        if t > setup["expires"]:
            setup = None
        elif setup["kind"] == "high" and _delay_high_ok(row):
            return "delay_high", None
        elif setup["kind"] == "low" and _delay_low_ok(row):
            return "delay_low", None

    # 3) base 불발한 신규 골든크로스 → 셋업 무장
    if bool(row["macd_golden_cross"]):
        if bool(row["rsi_overheated"]):           # RSI>=70 → 눌림목(high) 셋업
            setup = {"kind": "high", "expires": t + dwin}
        elif row["rsi"] < low_th:                 # RSI<50 → 50돌파(low) 셋업
            setup = {"kind": "low", "expires": t + dwin}
        # 50<=RSI<70 인데 base 불발(hma/volume 미충족)은 셋업 무장 안 함
    return None, setup


# --------------------------------------------------------------------------- #
# 봉내 포지션 관리 — 반환 (cash_delta, status)
# status: 'closed' | 'signal_pending' | 'hold'
# --------------------------------------------------------------------------- #
def _manage_open_bar(position: pos.Position, row, costs: pos.Costs, t: int, trades: list):
    o, h, l = row["open"], row["high"], row["low"]
    cash_delta = 0.0

    # (a) 손절 우선 (§0-4). 갭하락 시 시가 체결.
    if l <= position.stop_loss:
        fill = pos.sell_fill(min(o, position.stop_loss), costs)
        cash_delta += _close_all(position, fill, costs, t, row["time"], "STOP", trades)
        return cash_delta, "closed"

    # (b) 목표가 도달 → FULL이면 50% 분할익절 + 본전스탑 이동
    if position.state == pos.FULL and h >= position.target:
        fill = pos.sell_fill(max(o, position.target), costs)  # 갭상승 시 시가
        cash_delta += _close_partial(position, fill, position.qty * 0.5, costs, t,
                                     row["time"], "TP1_50%")
        position.state = pos.HALF
        position.stop_loss = position.entry_price  # 본전 보존
        # 같은 봉에서 신호청산도 이어 점검 (아래)

    # (c) 신호청산: 데드크로스(전량, 모든 상태) / RSI<50(HALF 잔량) — 종가 확정 → 다음봉 시가
    dead = bool(row["macd_dead_cross"])
    rsi_exit = bool(row["rsi_exit_below_low"]) and position.state == pos.HALF
    if dead or rsi_exit:
        position.trade.last_reason = "DEAD_CROSS" if dead else "RSI_EXIT"
        return cash_delta, "signal_pending"

    return cash_delta, "hold"


# --------------------------------------------------------------------------- #
# 청산 실행 헬퍼 — 현금 증가분(proceeds) 반환
# --------------------------------------------------------------------------- #
def _close_partial(position, price, qty, costs, t, time, reason) -> float:
    proceeds = pos.sell_proceeds(price, qty, costs)
    position.qty -= qty
    position.trade.exits.append(pos.Fill(time, t, "sell", price, qty, reason))
    position.trade.proceeds += proceeds
    return proceeds


def _close_all(position, price, costs, t, time, reason, trades) -> float:
    qty = position.qty
    proceeds = pos.sell_proceeds(price, qty, costs)
    position.qty = 0.0
    position.trade.exits.append(pos.Fill(time, t, "sell", price, qty, reason))
    position.trade.proceeds += proceeds
    position.trade.exit_time = time
    position.trade.exit_index = t
    position.trade.last_reason = reason
    trades.append(position.trade)
    return proceeds
