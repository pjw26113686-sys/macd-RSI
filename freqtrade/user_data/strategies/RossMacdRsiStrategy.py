"""freqtrade 전략 — 자체 엔진과 동일한 두뇌(signals_core)를 공유하는 독립 검증용.

목적(무결성 우선):
  - 지표·진입신호를 `src.signals_core`에서 그대로 가져와 자체 엔진과 일치시킨다.
  - freqtrade `lookahead-analysis` / `recursive-analysis`로 미래참조·재귀편향을
    독립 검증한다.
  - 청산(손절/목표 분할익절/데드크로스/RSI)은 합리적 수준으로 재현하되, 두 엔진의
    체결모델 차이(슬리피지·봉내 체결)로 정확한 P&L 일치는 목표가 아니다.

주의: 이 파일은 freqtrade 런타임에서만 동작한다(IStrategy 상속). 실제 백테스트/
lookahead-analysis는 freqtrade가 설치되고 BTC/USDT 1h 데이터가 있는 환경에서
실행한다(레포 README 참고).
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from pandas import DataFrame

from freqtrade.strategy import IStrategy, stoploss_from_absolute

# 레포 루트를 sys.path에 추가해 src.* 를 import (단일 두뇌 공유)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import signals_core  # noqa: E402
from src.engine.backtest import _warmup_bars  # noqa: E402

_PARAMS_PATH = _REPO_ROOT / "config" / "params.yaml"


class RossMacdRsiStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False

    # 청산은 전부 커스텀 콜백/시그널이 담당 → ROI 비활성, 손절은 custom_stoploss.
    minimal_roi = {"0": 10}
    stoploss = -0.99
    use_custom_stoploss = True
    position_adjustment_enable = True  # 50% 분할익절(adjust_trade_position)
    use_exit_signal = True
    process_only_new_candles = True

    # hma_period=100 기준 워밍업(≈111)보다 넉넉히. bot_start에서 재설정.
    startup_candle_count = 200

    def bot_start(self, **kwargs) -> None:
        with open(_PARAMS_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.params = cfg["strategy"]
        self.startup_candle_count = _warmup_bars(self.params)

    # ------------------------------------------------------------------ #
    # 지표 + 진입후보 — signals_core 단일 두뇌
    # ------------------------------------------------------------------ #
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return signals_core.add_signals(dataframe, self.params)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        mask = dataframe["enter_long"].astype(bool)
        dataframe["enter_long"] = mask.astype(int)
        dataframe.loc[mask, "enter_tag"] = dataframe.loc[mask, "entry_kind"]
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dead = dataframe["macd_dead_cross"].astype(bool)
        dataframe.loc[dead, "exit_long"] = 1          # 데드크로스 전량 청산(§4.4)
        dataframe.loc[dead, "exit_tag"] = "dead_cross"
        return dataframe

    # ------------------------------------------------------------------ #
    # 진입 시 손절/목표가 산정 후 trade 커스텀데이터에 저장
    # ------------------------------------------------------------------ #
    def _ensure_levels(self, trade, pair: str) -> None:
        if trade.get_custom_data("stop_price") is not None:
            return
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return
        # 진입 체결 봉의 위치(시간 기준) 탐색
        pos = df["date"].searchsorted(trade.open_date_utc, side="right") - 1
        pos = max(0, int(pos))
        M = self.params["swing_lookback_M"]
        swing_low = float(df["low"].iloc[max(0, pos - M):pos].min())
        entry = trade.open_rate
        stop = swing_low * (1.0 - self.params["stop_buffer"])
        if not (stop < entry):
            stop = entry * (1.0 - 0.01)  # 비정상 시 보호용 1% 손절
        R = (entry - stop) / entry
        target = entry * (1.0 + self.params["reward_ratio"] * R)
        trade.set_custom_data("stop_price", float(stop))
        trade.set_custom_data("target_price", float(target))
        trade.set_custom_data("half_done", False)

    def custom_stoploss(self, pair: str, trade, current_time, current_rate: float,
                        current_profit: float, after_fill: bool = False, **kwargs) -> float:
        self._ensure_levels(trade, pair)
        stop = trade.get_custom_data("stop_price")
        if stop is None:
            return self.stoploss
        if trade.get_custom_data("half_done"):
            stop = trade.open_rate  # 1차 익절 후 본전 보존(§4.3)
        return stoploss_from_absolute(
            stop, current_rate, is_short=trade.is_short, leverage=trade.leverage
        )

    # 목표가 도달 → 50% 분할익절(§4.3.1)
    def adjust_trade_position(self, trade, current_time, current_rate: float,
                              current_profit: float, min_stake, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs):
        self._ensure_levels(trade, trade.pair)
        if trade.get_custom_data("half_done"):
            return None
        target = trade.get_custom_data("target_price")
        if target is not None and current_rate >= target:
            trade.set_custom_data("half_done", True)
            return -(trade.stake_amount * 0.5)  # 음수 = 포지션 50% 축소
        return None

    # HALF 상태에서 RSI<50 이탈 시 잔량 청산(§4.3.3)
    def custom_exit(self, pair: str, trade, current_time, current_rate: float,
                    current_profit: float, **kwargs):
        if not trade.get_custom_data("half_done"):
            return None
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return None
        if bool(df["rsi_exit_below_low"].iloc[-1]):
            return "rsi_exit"
        return None
