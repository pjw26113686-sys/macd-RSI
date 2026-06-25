"""지표·신호 컬럼 단위테스트."""
import numpy as np
import pandas as pd

from src import signals_core as sc
from tests.helpers import default_params


def test_wma_known_value():
    s = pd.Series([1.0, 2.0, 3.0])
    # WMA period 3 = (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6
    assert abs(sc._wma(s, 3).iloc[-1] - 14.0 / 6.0) < 1e-9


def test_rsi_all_gains_is_100():
    close = pd.Series(np.arange(1, 50, dtype=float))  # 단조 증가
    rsi = sc.rsi_wilder(close, 14)
    assert rsi.dropna().iloc[-1] == 100.0


def test_rsi_bounded_0_100():
    rng = np.random.default_rng(1)
    close = pd.Series(100 + rng.normal(0, 1, 500).cumsum())
    rsi = sc.rsi_wilder(close, 14).dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_golden_and_dead_cross_are_exclusive():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "open": 100 + rng.normal(0, 1, 400).cumsum(),
    })
    df["close"] = df["open"] + rng.normal(0, 0.3, 400)
    df["high"] = df[["open", "close"]].max(axis=1) + 0.5
    df["low"] = df[["open", "close"]].min(axis=1) - 0.5
    df["volume"] = 1000.0
    out = sc.add_signals(df, default_params())
    # 같은 봉에서 골든·데드 동시 발생 불가
    assert not (out["macd_golden_cross"] & out["macd_dead_cross"]).any()


def test_is_bull_candle_body_ratio():
    df = pd.DataFrame({
        "open":   [100.0, 100.0, 100.0],
        "close":  [110.0, 100.5, 95.0],   # 큰양봉 / 작은양봉 / 음봉
        "high":   [111.0, 101.0, 100.0],
        "low":    [99.0,  99.5,  94.0],
        "volume": [1000.0, 1000.0, 1000.0],
    })
    out = sc.add_signals(df, default_params())  # min_body_ratio=0.5
    assert out["is_bull_candle"].tolist() == [True, False, False]


def test_volume_ok_threshold():
    n = 30
    df = pd.DataFrame({
        "open": np.full(n, 100.0), "high": np.full(n, 101.0),
        "low": np.full(n, 99.0), "close": np.full(n, 100.0),
        "volume": np.full(n, 1000.0),
    })
    df.loc[25, "volume"] = 2000.0  # 평균 위
    df.loc[26, "volume"] = 100.0   # 평균 아래
    out = sc.add_signals(df, default_params())
    assert bool(out.loc[25, "volume_ok"]) is True
    assert bool(out.loc[26, "volume_ok"]) is False
