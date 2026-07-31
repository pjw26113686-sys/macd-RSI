"""[breakout] 돈치안 채널 브레이크아웃 (터틀 계열).

가정: 가격이 최근 N봉 고점을 돌파하면 추세가 이어진다.
  진입: 종가가 직전 N봉의 최고 고점(현재봉 제외)을 상향 돌파.
  청산(전량): 종가가 직전 M봉의 최저 저점(현재봉 제외) 하향 이탈.
  청산(잔량): 종가가 채널 중앙선 하회.
손절/목표는 엔진 공통(swing-low stop + reward_ratio)이 관리.

미래참조 0: 채널은 rolling(...).shift(1)로 **현재봉을 제외한 과거만** 사용하고,
현재 종가와 비교한다. 진입은 종가 확정 → 엔진이 다음 봉 시가에 체결.
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base import StrategySpec, register

DEFAULT_PARAMS = {
    "entry_lookback": 20,   # 돌파 기준 고점 봉수 N
    "exit_lookback": 10,    # 이탈 기준 저점 봉수 M
    # 엔진 공통 손절/목표
    "swing_lookback_M": 10,
    "stop_buffer": 0.001,
    "reward_ratio": 2.0,
}

PARAM_GRID = {
    "entry_lookback": [20, 30, 55],
    "exit_lookback": [10, 20],
    "reward_ratio": [2.0, 3.0],
}


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    n = params["entry_lookback"]
    m = params["exit_lookback"]

    # 현재봉 제외 과거 N/M봉 채널 (shift(1)).
    upper = out["high"].rolling(window=n).max().shift(1)
    lower_exit = out["low"].rolling(window=m).min().shift(1)
    mid = (out["high"].rolling(window=n).max().shift(1)
           + out["low"].rolling(window=n).min().shift(1)) / 2.0
    out["dc_upper"], out["dc_lower"], out["dc_mid"] = upper, lower_exit, mid

    close = out["close"]
    out["enter_long"] = close > upper
    out["exit_all"] = close < lower_exit
    out["exit_half"] = close < mid

    for col in ["enter_long", "exit_all", "exit_half"]:
        out[col] = out[col].fillna(False)
    return out


def warmup_bars(params: dict) -> int:
    return max(params["entry_lookback"], params["exit_lookback"]) + 1


SPEC = register(StrategySpec(
    name="breakout_donchian",
    category="breakout",
    description="직전 N봉 고점 돌파 진입, M봉 저점 이탈 청산. 추세추종(터틀 계열).",
    default_params=DEFAULT_PARAMS,
    param_grid=PARAM_GRID,
    generate_signals=generate_signals,
    warmup_bars=warmup_bars,
    exit_col="exit_all",
    exit_half_col="exit_half",
    tags=("trend", "channel"),
))
