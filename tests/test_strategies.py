"""카테고리 전략 레지스트리 + 각 전략의 무결성/엔진연동 검증.

핵심: 어떤 카테고리의 전략이든 (1) lookahead-safe이고 (2) 동일 엔진·검증 레이어로
돌아가야 한다. 새 전략을 추가하면 이 테스트가 자동으로 그 계약을 강제한다.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import strategies as strat
from src.engine import backtest as bt
from src.engine.position import Costs
from src.validation import cscv_pbo, expand_grid, run_sweep
from tests.helpers import random_walk_ohlcv

ALL_SPECS = strat.all_specs()
COSTS = Costs(fee=0.001, slippage=0.001, sell_tax=0.0)
BPY = 24 * 365


# --------------------------------------------------------------------------- #
# 레지스트리
# --------------------------------------------------------------------------- #
def test_registry_has_three_categories():
    cats = strat.categories()
    assert {"momentum", "mean_reversion", "breakout"} <= set(cats)


def test_get_and_by_category_consistent():
    for spec in ALL_SPECS:
        assert strat.get(spec.name) is spec
        assert spec in strat.by_category(spec.category)


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError):
        strat.register(ALL_SPECS[0])  # 이미 등록됨


# --------------------------------------------------------------------------- #
# 각 전략: 신호 컬럼 계약 + 미래참조 0
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", ALL_SPECS, ids=[s.name for s in ALL_SPECS])
def test_signal_contract_columns(spec):
    df = random_walk_ohlcv(500, seed=3)
    sig = spec.generate_signals(df, spec.default_params)
    for col in ["enter_long", spec.exit_col, spec.exit_half_col]:
        assert col in sig.columns, f"{spec.name}에 {col} 없음"
        assert sig[col].dtype == bool


@pytest.mark.parametrize("spec", ALL_SPECS, ids=[s.name for s in ALL_SPECS])
def test_no_future_leak(spec):
    """전체계산 값과 t시점 절단계산 값이 t에서 동일해야 한다(미래참조 0)."""
    params = spec.default_params
    df = random_walk_ohlcv(600, seed=7)
    full = spec.generate_signals(df, params)
    cols = ["enter_long", spec.exit_col, spec.exit_half_col]

    rng = np.random.default_rng(123)
    warm = spec.warmup_bars(params)
    for t in sorted({int(x) for x in rng.integers(warm + 5, len(df) - 1, size=20)}):
        trunc = spec.generate_signals(df.iloc[: t + 1], params)
        for col in cols:
            assert bool(full.iloc[t][col]) == bool(trunc.iloc[t][col]), (
                f"미래참조: {spec.name}.{col} @ t={t}"
            )


# --------------------------------------------------------------------------- #
# 각 전략: 엔진 연동 + 검증 레이어 연동
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", ALL_SPECS, ids=[s.name for s in ALL_SPECS])
def test_runs_through_engine(spec):
    df = random_walk_ohlcv(800, seed=5)
    res = bt.run_backtest(df, spec.default_params, COSTS, strategy=spec)
    assert len(res.equity) == len(df)
    assert np.isfinite(res.equity.to_numpy()).all()
    assert (res.equity > 0).all()  # 파산/음수자산 없음


@pytest.mark.parametrize("spec", ALL_SPECS, ids=[s.name for s in ALL_SPECS])
def test_sweep_and_pbo_per_strategy(spec):
    df = random_walk_ohlcv(1500, seed=9)
    params_list = expand_grid(spec.default_params, spec.param_grid)
    assert len(params_list) >= 4  # CSCV 순위매김에 충분한 설정 수
    sw = run_sweep(df, params_list, COSTS, bars_per_year=BPY, strategy=spec)
    assert sw["returns"].shape == (len(df), len(params_list))
    res = cscv_pbo(sw["returns"].to_numpy(), n_blocks=10)
    assert 0.0 <= res["pbo"] <= 1.0


def test_strategies_produce_distinct_signals():
    """서로 다른 카테고리는 서로 다른 진입을 낸다(같은 전략을 두 번 등록한 게 아님)."""
    df = random_walk_ohlcv(800, seed=1)
    entries = {}
    for spec in ALL_SPECS:
        sig = spec.generate_signals(df, spec.default_params)
        entries[spec.name] = sig["enter_long"].to_numpy()
    names = list(entries)
    # 최소 한 쌍은 진입 패턴이 달라야 한다.
    assert any(
        not np.array_equal(entries[a], entries[b])
        for i, a in enumerate(names) for b in names[i + 1:]
    )
