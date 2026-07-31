"""[momentum] MACD 골든크로스 + Wilder RSI 모멘텀 (기존 단일 두뇌 이식).

기존 signals_core를 그대로 신호 생성기로 감싸 레지스트리에 올린다. 청산은 데드크로스
(전량) / RSI<50(잔량). 이 전략의 세부 규율은 strategy_spec_v2 / signals_core 참조.
"""
from __future__ import annotations

import pandas as pd

from src import signals_core
from src.engine.backtest import _warmup_bars
from src.strategies.base import StrategySpec, register

DEFAULT_PARAMS = {
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "rsi_period": 14, "rsi_entry_low": 50, "rsi_entry_high": 70,
    "rsi_support_band": 5, "hma_period": 100,
    "vol_ma_period": 20, "vol_ma_mult": 1.0,
    "delay_window": 5, "min_body_ratio": 0.5,
    "swing_lookback_M": 10, "stop_buffer": 0.001, "reward_ratio": 2.0,
}

PARAM_GRID = {
    "macd_fast": [8, 12],
    "rsi_period": [7, 9, 14],
    "hma_period": [50, 100],
    "rsi_entry_low": [45, 50],
}


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    return signals_core.add_signals(df, params)


SPEC = register(StrategySpec(
    name="momentum_macd_rsi",
    category="momentum",
    description="MACD 골든크로스 + Wilder RSI + HMA/거래량 필터. 지연진입·분할익절·본전스탑.",
    default_params=DEFAULT_PARAMS,
    param_grid=PARAM_GRID,
    generate_signals=generate_signals,
    warmup_bars=_warmup_bars,
    exit_col="macd_dead_cross",
    exit_half_col="rsi_exit_below_low",
    tags=("trend", "crossover"),
))
