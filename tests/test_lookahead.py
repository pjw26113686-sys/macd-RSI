"""미래참조(look-ahead) 검증 — 자체엔진이 freqtrade 내장검증을 못 받으므로 직접 구현.

방법: signals_core를 전체 데이터로 계산한 값과, t시점까지 자른 부분집합으로 계산한
값의 t시점 신호가 동일해야 한다. 다르면 미래 데이터가 새어든 것이다.
"""
import numpy as np

from src import signals_core as sc
from tests.helpers import default_params, random_walk_ohlcv


def test_no_future_leak_in_signal_columns():
    params = default_params()
    df = random_walk_ohlcv(600, seed=7)
    full = sc.add_signals(df, params)

    rng = np.random.default_rng(123)
    test_points = rng.integers(200, len(df) - 1, size=25)

    for t in sorted(set(int(x) for x in test_points)):
        trunc = sc.add_signals(df.iloc[: t + 1], params)
        for col in sc.SIGNAL_COLUMNS:
            full_val = bool(full.iloc[t][col])
            trunc_val = bool(trunc.iloc[t][col])
            assert full_val == trunc_val, (
                f"미래참조 감지: col={col}, t={t}, full={full_val}, trunc={trunc_val}"
            )


def test_indicators_match_at_t_under_truncation():
    """연속 지표값도 t시점 절단과 전체가 (워밍업 이후) 동일해야 한다."""
    params = default_params()
    df = random_walk_ohlcv(400, seed=11)
    full = sc.compute_indicators(df, params)
    t = 350
    trunc = sc.compute_indicators(df.iloc[: t + 1], params)
    for col in ["macd_line", "signal_line", "rsi", "hma"]:
        a, b = full.iloc[t][col], trunc.iloc[t][col]
        assert np.isclose(a, b, rtol=1e-9, atol=1e-9), f"{col}: {a} != {b}"
