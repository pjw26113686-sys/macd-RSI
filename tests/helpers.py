"""테스트 공용 헬퍼: 기본 파라미터 + 신호 프레임 빌더."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import signals_core


def default_params() -> dict:
    return {
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "rsi_period": 14, "rsi_entry_low": 50, "rsi_entry_high": 70,
        "rsi_support_band": 5, "hma_period": 20,
        "vol_ma_period": 20, "vol_ma_mult": 1.0,
        "delay_window": 5, "min_body_ratio": 0.5,
        "swing_lookback_M": 10, "stop_buffer": 0.001, "reward_ratio": 2.0,
    }


def empty_signal_frame(n: int) -> pd.DataFrame:
    """모든 신호 컬럼이 False인 OHLCV 프레임. 테스트에서 필요한 봉만 켠다."""
    base = {
        "time": pd.RangeIndex(n),
        "open": np.full(n, 100.0),
        "high": np.full(n, 100.0),
        "low": np.full(n, 100.0),
        "close": np.full(n, 100.0),
        "volume": np.full(n, 1000.0),
    }
    df = pd.DataFrame(base)
    for col in signals_core.SIGNAL_COLUMNS:
        df[col] = False
    df["rsi"] = 55.0  # 셋업 무장 로직이 참조
    return df


def random_walk_ohlcv(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1, n).cumsum()
    close = 100.0 + steps
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC", name="time")
    high = close + rng.random(n)
    low = close - rng.random(n)
    open_ = close - rng.normal(0, 0.5, n)
    vol = rng.integers(500, 2000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )
