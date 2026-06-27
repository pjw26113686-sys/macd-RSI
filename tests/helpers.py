"""테스트 공용 헬퍼: 신호 프레임 빌더 + 합성 데이터."""
from __future__ import annotations

import numpy as np
import pandas as pd


def signal_frame(n: int) -> pd.DataFrame:
    """OHLCV + 진입/청산 신호 컬럼이 비어있는 프레임. 필요한 봉만 켠다.

    run_on_signals가 기대하는 컬럼: time, open/high/low/close/volume,
    enter_long/enter_short(+선택 exit_long/exit_short/exit_target).
    """
    df = pd.DataFrame({
        "time": pd.RangeIndex(n),
        "open": np.full(n, 100.0),
        "high": np.full(n, 100.0),
        "low": np.full(n, 100.0),
        "close": np.full(n, 100.0),
        "volume": np.full(n, 1000.0),
        "enter_long": False,
        "enter_short": False,
        "exit_long": False,
        "exit_short": False,
    })
    return df


def set_bar(df, i, o, h, l, c):
    df.loc[i, ["open", "high", "low", "close"]] = [o, h, l, c]


def random_walk_ohlcv(n: int, seed: int = 0, freq: str = "1h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0, 1, n).cumsum()
    idx = pd.date_range("2021-01-01", periods=n, freq=freq, tz="UTC", name="time")
    high = close + rng.random(n)
    low = close - rng.random(n)
    open_ = close - rng.normal(0, 0.5, n)
    vol = rng.integers(500, 2000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )
