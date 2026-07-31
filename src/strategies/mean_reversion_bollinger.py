"""[mean_reversion] 볼린저 밴드 평균회귀.

가정: 추세가 없을 때 가격은 밴드 밖으로 벗어나면 중심선(SMA)으로 되돌아온다.
  진입: 하단 밴드 아래로 이탈했다가 종가가 하단 밴드 위로 복귀(과매도 반등 확정).
  청산(전량): 종가가 중심선(SMA) 도달 → 평균 회귀 완료.
  청산(잔량): 종가가 상단 밴드 도달 → 과열, 잔량 정리.
손절/목표는 엔진 공통(swing-low stop + reward_ratio)이 관리.

미래참조 0: 모든 컬럼은 현재/과거 봉만 사용(shift(+1)만). 진입은 종가 확정 → 엔진이
다음 봉 시가에 체결.
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base import StrategySpec, register

DEFAULT_PARAMS = {
    "bb_period": 20,
    "bb_std": 2.0,
    # 엔진 공통 손절/목표
    "swing_lookback_M": 10,
    "stop_buffer": 0.001,
    "reward_ratio": 2.0,
}

PARAM_GRID = {
    "bb_period": [10, 20, 30],
    "bb_std": [1.5, 2.0, 2.5],
    "reward_ratio": [1.5, 2.0],
}


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    n = params["bb_period"]
    k = params["bb_std"]

    mid = out["close"].rolling(window=n).mean()
    sd = out["close"].rolling(window=n).std(ddof=0)
    lower = mid - k * sd
    upper = mid + k * sd
    out["bb_mid"], out["bb_lower"], out["bb_upper"] = mid, lower, upper

    close = out["close"]
    # 진입: 직전 봉은 하단 밴드 아래, 현재 봉 종가는 하단 밴드 위로 복귀(반등 확정).
    out["enter_long"] = (close.shift(1) < lower.shift(1)) & (close >= lower)
    # 청산(전량): 중심선 회귀. 청산(잔량): 상단 밴드 과열.
    out["exit_all"] = close >= mid
    out["exit_half"] = close >= upper

    for col in ["enter_long", "exit_all", "exit_half"]:
        out[col] = out[col].fillna(False)
    return out


def warmup_bars(params: dict) -> int:
    return params["bb_period"] + 1


SPEC = register(StrategySpec(
    name="mean_reversion_bollinger",
    category="mean_reversion",
    description="볼린저 하단 이탈 후 복귀 진입, 중심선 회귀 청산. 횡보장 역추세.",
    default_params=DEFAULT_PARAMS,
    param_grid=PARAM_GRID,
    generate_signals=generate_signals,
    warmup_bars=warmup_bars,
    exit_col="exit_all",
    exit_half_col="exit_half",
    tags=("counter-trend", "band"),
))
