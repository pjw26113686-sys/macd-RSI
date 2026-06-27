"""전략-불가지론 이벤트기반 백테스트 엔진.

엔진은 특정 전략을 모른다. 입력으로 받는 것:
  - 신호 df (전략이 만든 enter_long/enter_short [+exit_*/exit_target] 컬럼 포함)
  - ExitModel (청산 정책 선언)
  - PnLModel (노셔널/선물 손익 계산)
  - SizingConfig (포지션 사이징)

체결 무결성:
  - 신호는 종가확정 봉(t)에서만 평가. 진입 = 다음 봉(t+1) 시가 체결.
  - 손절·목표는 지정가 → 닿는 봉에서 그 레벨로 즉시 체결(갭 시 시가).
  - 한 봉에서 손절·목표 동시 도달 시 손절 우선(최악 가정).
  - 신호청산·시간청산은 종가확정 → 다음 봉 시가 체결.

프랍룰을 위해 봉별 equity를 종가(close)뿐 아니라 봉 고/저(보유 미실현 포함)로도
기록한다 → equity_high/equity_low (인트라데이 트레일링DD 정확도).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.engine import position as pos
from src.engine.pnl import LONG, SHORT, SizingConfig


@dataclass
class BacktestResult:
    trades: list                # list[pos.Trade]
    equity: pd.Series           # 봉별 종가기준 자산
    equity_high: pd.Series      # 봉내 우호적 극단 자산(미실현 최대)
    equity_low: pd.Series       # 봉내 비우호적 극단 자산(미실현 최소)
    signals: pd.DataFrame
    initial_capital: float


def run_backtest(df, strategy, params, pnl_model, sizing,
                 initial_capital=10000.0, instrument=None, warmup=None):
    """OHLCV df로 전략 신호를 계산한 뒤 백테스트."""
    sig = strategy.add_signals(df, params)
    sig = sig.reset_index(drop=False)
    if "time" not in sig.columns:
        sig = sig.rename(columns={sig.columns[0]: "time"})
    if warmup is None:
        warmup = _default_warmup(params)
    return run_on_signals(sig, params, strategy.exit_model, pnl_model, sizing,
                          initial_capital, instrument, warmup)


def _default_warmup(params: dict) -> int:
    """지표 안정화 봉수 추정(보수적 max). 이 구간에서는 진입하지 않는다."""
    candidates = [
        params.get("macd_slow", 0) + params.get("macd_signal", 0),
        params.get("rsi_period", 0) + 1,
        params.get("hma_period", 0),
        params.get("bb_period", 0),
        params.get("slow_ma", 0),
        params.get("swing_lookback_M", 0),
        params.get("vol_ma_period", 0),
    ]
    return max([c for c in candidates] + [1])


def run_on_signals(sig, params, exit_model, pnl_model, sizing: SizingConfig,
                   initial_capital=10000.0, instrument=None, warmup=1):
    """신호 컬럼이 계산된 프레임으로 백테스트.

    sig: open/high/low/close/volume + enter_long/enter_short(+exit_*/exit_target)
         + 'time' 컬럼을 가진 RangeIndex DataFrame.
    """
    sig = sig.reset_index(drop=True)
    n = len(sig)
    M = params.get("swing_lookback_M", 10)
    highs = sig["high"].to_numpy()
    lows = sig["low"].to_numpy()
    has_target_col = "exit_target" in sig.columns

    cash = float(initial_capital)
    position: pos.Position | None = None
    pending_entry: str | None = None   # LONG | SHORT | None
    pending_exit = False

    trades: list = []
    equity = np.full(n, np.nan)
    eq_high = np.full(n, np.nan)
    eq_low = np.full(n, np.nan)

    for t in range(n):
        row = sig.iloc[t]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]

        # 1) 직전 종가확정 신호/시간 청산 → 이번 봉 시가 체결
        if pending_exit and position is not None:
            fill = pnl_model.exit_fill(o, position.direction)
            cash += _close_all(position, fill, pnl_model, t, row["time"],
                               position.trade.last_reason, trades)
            position = None
        pending_exit = False

        # 2) 직전 종가확정 진입 → 이번 봉 시가 체결
        if pending_entry is not None and position is None:
            position, d_cash = _open_position(
                pending_entry, o, t, row, sig, params, exit_model, pnl_model,
                sizing, cash, M, highs, lows, has_target_col, instrument)
            cash += d_cash
        pending_entry = None

        # 3) 보유 포지션 봉내 관리
        if position is not None:
            d_cash, status = _manage_open_bar(position, row, pnl_model, exit_model,
                                              t, trades, has_target_col)
            cash += d_cash
            if status == "closed":
                position = None
            elif status == "signal_pending":
                pending_exit = True

        # 4) 무포지션 & 워밍업 이후: 진입후보 → 다음 봉 시가 체결
        if position is None and not pending_exit and warmup <= t < n - 1:
            if bool(row.get("enter_long", False)):
                pending_entry = LONG
            elif bool(row.get("enter_short", False)):
                pending_entry = SHORT

        # 5) 봉별 자산 기록(종가 + 봉 고저 미실현)
        if position is not None:
            mc = pnl_model.mark(c, position.entry_price, position.qty, position.direction)
            mh = pnl_model.mark(h, position.entry_price, position.qty, position.direction)
            ml = pnl_model.mark(l, position.entry_price, position.qty, position.direction)
            equity[t] = cash + mc
            eq_high[t] = cash + max(mh, ml)
            eq_low[t] = cash + min(mh, ml)
        else:
            equity[t] = eq_high[t] = eq_low[t] = cash

    # 종료 시 잔여 포지션은 마지막 종가로 청산
    if position is not None:
        last = sig.iloc[n - 1]
        fill = pnl_model.exit_fill(last["close"], position.direction)
        cash += _close_all(position, fill, pnl_model, n - 1, last["time"], "EOD", trades)
        equity[n - 1] = eq_high[n - 1] = eq_low[n - 1] = cash

    idx = sig["time"]
    return BacktestResult(
        trades=trades,
        equity=pd.Series(equity, index=idx, name="equity").ffill().fillna(initial_capital),
        equity_high=pd.Series(eq_high, index=idx, name="equity_high").ffill().fillna(initial_capital),
        equity_low=pd.Series(eq_low, index=idx, name="equity_low").ffill().fillna(initial_capital),
        signals=sig,
        initial_capital=initial_capital,
    )


# --------------------------------------------------------------------------- #
# 진입 (다음 봉 시가)
# --------------------------------------------------------------------------- #
def _open_position(direction, o, t, row, sig, params, exit_model, pnl_model,
                   sizing, cash, M, highs, lows, has_target_col, instrument):
    """(position|None, cash_delta) 반환."""
    entry_fill = pnl_model.entry_fill(o, direction)
    # swing 참조 극단 (체결봉 t 직전 M봉)
    if direction == LONG:
        ref = float(lows[max(0, t - M):t].min()) if t > 0 else float(lows[t])
    else:
        ref = float(highs[max(0, t - M):t].max()) if t > 0 else float(highs[t])
    exit_target = float(row["exit_target"]) if has_target_col and pd.notna(row.get("exit_target")) else None

    stop, target = pos.compute_stop_and_target(
        entry_fill, ref, params, direction, exit_model, instrument, exit_target)
    if stop is None:
        return None, 0.0

    qty = pnl_model.size(cash, entry_fill, stop, direction, sizing)
    if qty <= 0:
        return None, 0.0

    risk_basis = pnl_model.risk_basis(entry_fill, stop, qty, direction)
    d_cash = pnl_model.entry_cash_delta(entry_fill, qty, direction)
    trade = pos.Trade(
        entry_time=row["time"], entry_index=t, entry_price=entry_fill, qty=qty,
        direction=direction, stop_loss=stop, target=target, risk_basis=risk_basis,
        realized=d_cash,
    )
    position = pos.Position(entry_fill, qty, direction, stop, target, pos.FULL, trade)
    return position, d_cash


# --------------------------------------------------------------------------- #
# 봉내 포지션 관리 — (cash_delta, status)  status: closed|signal_pending|hold
# --------------------------------------------------------------------------- #
def _manage_open_bar(position, row, pnl_model, exit_model, t, trades, has_target_col):
    o, h, l = row["open"], row["high"], row["low"]
    d = position.direction
    cash_delta = 0.0

    # target_mode='mid'면 목표가를 봉마다 갱신(예: 볼린저 중심선 이동)
    if exit_model.target_mode == "mid" and has_target_col and pd.notna(row.get("exit_target")):
        position.target = float(row["exit_target"])

    # (a) 손절 우선
    hit_stop = (l <= position.stop_loss) if d == LONG else (h >= position.stop_loss)
    if hit_stop:
        raw = min(o, position.stop_loss) if d == LONG else max(o, position.stop_loss)
        fill = pnl_model.exit_fill(raw, d)
        cash_delta += _close_all(position, fill, pnl_model, t, row["time"], "STOP", trades)
        return cash_delta, "closed"

    # (b) 목표 도달
    if position.target is not None:
        hit_tgt = (h >= position.target) if d == LONG else (l <= position.target)
        if hit_tgt:
            raw = max(o, position.target) if d == LONG else min(o, position.target)
            fill = pnl_model.exit_fill(raw, d)
            if exit_model.partial_tp_pct and position.state == pos.FULL:
                _close_partial(position, fill, position.qty * exit_model.partial_tp_pct,
                               pnl_model, t, row["time"], "TP1")
                position.state = pos.HALF
                if exit_model.breakeven_after_tp:
                    position.stop_loss = position.entry_price
            else:
                cash_delta += _close_all(position, fill, pnl_model, t, row["time"], "TARGET", trades)
                return cash_delta, "closed"

    # (c) 신호청산
    if exit_model.signal_exit:
        ex = bool(row.get("exit_long", False)) if d == LONG else bool(row.get("exit_short", False))
        if ex:
            position.trade.last_reason = "SIGNAL"
            return cash_delta, "signal_pending"

    # (d) 시간청산
    if exit_model.max_hold_bars is not None:
        if (t - position.trade.entry_index) >= exit_model.max_hold_bars:
            position.trade.last_reason = "TIME"
            return cash_delta, "signal_pending"

    return cash_delta, "hold"


# --------------------------------------------------------------------------- #
# 청산 실행 헬퍼 — 현금 변화($) 반환
# --------------------------------------------------------------------------- #
def _close_partial(position, fill, qty, pnl_model, t, time, reason) -> float:
    delta = pnl_model.exit_cash_delta(position.entry_price, fill, qty, position.direction)
    position.qty -= qty
    position.trade.realized += delta
    position.trade.exits.append(pos.Fill(time, t, "exit", fill, qty, reason))
    return delta


def _close_all(position, fill, pnl_model, t, time, reason, trades) -> float:
    qty = position.qty
    delta = pnl_model.exit_cash_delta(position.entry_price, fill, qty, position.direction)
    position.qty = 0.0
    position.trade.realized += delta
    position.trade.exits.append(pos.Fill(time, t, "exit", fill, qty, reason))
    position.trade.exit_time = time
    position.trade.exit_index = t
    position.trade.last_reason = reason
    trades.append(position.trade)
    return delta
