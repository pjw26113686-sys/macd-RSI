"""미래참조(look-ahead) 검증 — 등록된 모든 전략에 대해 자동 강제.

방법: 전략 신호를 전체 데이터로 계산한 값과, t시점까지 자른 부분으로 계산한 값의
t시점 신호가 동일해야 한다. 다르면 미래 데이터가 새어든 것이다. 새 전략을 등록하면
이 테스트가 자동으로 그 전략도 검사한다.
"""
import numpy as np
import pytest

from src.strategies.registry import STRATEGIES, get_strategy
from tests.helpers import random_walk_ohlcv


def _equal(a, b) -> bool:
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    fa, fb = float(a), float(b)
    if np.isnan(fa) and np.isnan(fb):
        return True
    return np.isclose(fa, fb, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_no_future_leak(name):
    strat = get_strategy(name)
    params = strat.params()
    df = random_walk_ohlcv(500, seed=7, freq="5min")
    full = strat.add_signals(df, params)

    rng = np.random.default_rng(123)
    for t in sorted(set(int(x) for x in rng.integers(200, len(df) - 1, size=20))):
        trunc = strat.add_signals(df.iloc[: t + 1], params)
        for col in strat.signal_columns:
            assert _equal(full.iloc[t][col], trunc.iloc[t][col]), (
                f"미래참조 감지: 전략={name}, col={col}, t={t}, "
                f"full={full.iloc[t][col]}, trunc={trunc.iloc[t][col]}"
            )
