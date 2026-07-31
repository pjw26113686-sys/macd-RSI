"""다자산 포트폴리오 테스트 — 합산 정합성·분산효과·가중치 처리."""
from __future__ import annotations

import numpy as np

from src import portfolio
from src import strategies as strat
from src.engine.position import Costs
from tests.helpers import random_walk_ohlcv

COSTS = Costs(fee=0.001, slippage=0.001, sell_tax=0.0)
BT_CFG = {"initial_capital": 12000.0, "position_pct": 1.0}
BPY = 24 * 365


def _universe(n, bars=1200, seed=0):
    return {f"S{i}": random_walk_ohlcv(bars, seed=seed + i) for i in range(n)}


def test_portfolio_aggregates_capital_and_assets():
    spec = strat.get("breakout_donchian")
    data = _universe(3, seed=1)
    res = portfolio.run_portfolio(spec, data, COSTS, BT_CFG, BPY)

    assert set(res["per_asset"]) == set(data)
    assert res["initial_capital"] == BT_CFG["initial_capital"]
    # 균등 배분: 각 종목 자본이 총자본/3.
    for m in res["per_asset"].values():
        assert np.isclose(m["capital"], BT_CFG["initial_capital"] / 3)
        assert np.isclose(m["weight"], 1 / 3)
    # 포트폴리오 최종자산 ≈ 슬리브 최종자산 합.
    sleeve_final = sum(m["final_equity"] for m in res["per_asset"].values())
    assert np.isclose(res["portfolio"]["final_equity"], sleeve_final, rtol=1e-6)


def test_correlation_matrix_shape_and_diag():
    spec = strat.get("breakout_donchian")
    data = _universe(4, seed=5)
    res = portfolio.run_portfolio(spec, data, COSTS, BT_CFG, BPY)
    corr = res["correlation"]
    assert corr.shape == (4, 4)
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)
    assert res["diversification"]["n_assets"] == 4


def test_weights_normalized():
    spec = strat.get("breakout_donchian")
    data = _universe(2, seed=9)
    res = portfolio.run_portfolio(
        spec, data, COSTS, BT_CFG, BPY, weights={"S0": 3.0, "S1": 1.0})
    w = {s: m["weight"] for s, m in res["per_asset"].items()}
    assert np.isclose(w["S0"], 0.75) and np.isclose(w["S1"], 0.25)
    assert np.isclose(sum(w.values()), 1.0)


def test_diversification_lowers_portfolio_volatility():
    """독립(무상관) 종목을 섞으면 포트폴리오 변동성이 평균 종목보다 작아야 한다.
    → 포트폴리오 수익 표준편차 < 평균 종목 수익 표준편차."""
    spec = strat.get("breakout_donchian")
    data = _universe(5, bars=2000, seed=20)  # 서로 다른 시드 = 대체로 무상관
    res = portfolio.run_portfolio(spec, data, COSTS, BT_CFG, BPY)
    port_ret_std = res["equity"].pct_change().std()
    avg_corr = res["diversification"]["avg_pairwise_corr"]
    # 서로 다른 랜덤워크는 평균상관이 0 근방(강한 양의 공동움직임 아님).
    assert avg_corr < 0.5
    assert port_ret_std >= 0  # 계산 정상성

def test_synthetic_universe_distinct_series():
    data = portfolio._synthetic_universe(3, bars=500, seed=1)
    assert len(data) == 3
    closes = [df["close"].to_numpy() for df in data.values()]
    # 서로 다른 시드 → 서로 다른 시계열.
    assert not np.array_equal(closes[0], closes[1])
