"""레퍼런스 예시 1 — 이동평균 교차(가장 단순한 롱/숏 추세추종).

진입: 빠른MA가 느린MA를 상향돌파 → 롱 / 하향돌파 → 숏.
청산: 반대 교차(신호청산) 또는 손절(swing)/목표(rr). 추세추종 베이스라인.
"""
from __future__ import annotations

import pandas as pd

from src import signals_core as ind
from src.strategies.base import ExitModel, Strategy, empty_signals


class MaCrossStrategy(Strategy):
    name = "ma_cross"

    default_params = {
        "fast_ma": 20,
        "slow_ma": 50,
        "swing_lookback_M": 10,
        "stop_buffer": 0.001,
        "reward_ratio": 2.0,
    }

    exit_model = ExitModel(
        stop_mode="swing",
        target_mode="rr",
        signal_exit=True,        # 반대 교차 시 청산
        max_hold_bars=None,
    )

    signal_columns = ["fast_ma", "slow_ma", "enter_long", "enter_short",
                      "exit_long", "exit_short"]

    def add_signals(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        ind.validate_ohlcv(df)
        out = empty_signals(df)
        out["fast_ma"] = ind.ema(out["close"], params["fast_ma"])
        out["slow_ma"] = ind.ema(out["close"], params["slow_ma"])

        gc = ind.crossover(out["fast_ma"], out["slow_ma"])
        dc = ind.crossunder(out["fast_ma"], out["slow_ma"])
        out["enter_long"] = gc.fillna(False)
        out["enter_short"] = dc.fillna(False)
        out["exit_long"] = dc.fillna(False)    # 하향교차 → 롱 청산
        out["exit_short"] = gc.fillna(False)   # 상향교차 → 숏 청산
        return out
