"""지표 라이브러리(signals_core) 단위테스트."""
import numpy as np
import pandas as pd

from src import signals_core as sc


def test_wma_known_value():
    s = pd.Series([1.0, 2.0, 3.0])
    # WMA(3) = (1*1 + 2*2 + 3*3)/(1+2+3) = 14/6
    assert abs(sc.wma(s, 3).iloc[-1] - 14.0 / 6.0) < 1e-9


def test_rsi_all_gains_is_100():
    close = pd.Series(np.arange(1, 50, dtype=float))
    assert sc.rsi_wilder(close, 14).dropna().iloc[-1] == 100.0


def test_rsi_bounded_0_100():
    rng = np.random.default_rng(1)
    close = pd.Series(100 + rng.normal(0, 1, 500).cumsum())
    rsi = sc.rsi_wilder(close, 14).dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_bollinger_midpoint_is_sma():
    close = pd.Series(np.arange(1, 100, dtype=float))
    mid, up, lo = sc.bollinger(close, 20, 2.0)
    assert np.isclose(mid.iloc[-1], close.iloc[-20:].mean())
    m = mid.notna()
    assert (up[m] >= mid[m]).all() and (lo[m] <= mid[m]).all()


def test_crossover_and_crossunder_exclusive():
    a = pd.Series([1.0, 2.0, 1.0, 2.0])
    b = pd.Series([1.5, 1.5, 1.5, 1.5])
    co = sc.crossover(a, b)
    cu = sc.crossunder(a, b)
    assert not (co & cu).any()
    assert co.iloc[1]   # 1→2 상향 돌파
    assert cu.iloc[2]   # 2→1 하향 돌파


def test_atr_positive():
    rng = np.random.default_rng(3)
    close = 100 + rng.normal(0, 1, 200).cumsum()
    df = pd.DataFrame({
        "high": close + 1.0, "low": close - 1.0, "close": close,
        "open": close, "volume": 1000.0,
    })
    assert (sc.atr(df, 14).dropna() > 0).all()
