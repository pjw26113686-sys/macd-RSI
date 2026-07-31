"""포지션 사이징 테스트 — 하위호환(기존=전액) + 고정리스크 정확성.

핵심: (1) 사이저 미지정 시 기존 전액 사이징과 완전 동일해야 하고(회귀), (2) 고정리스크
사이저는 손절까지의 손실이 자본의 risk_pct가 되도록 정확히 크기를 잡아야 한다.
"""
from __future__ import annotations

import numpy as np

from src.engine import backtest as bt
from src.engine import sizing
from src.engine.position import Costs
from tests.helpers import default_params, random_walk_ohlcv

COSTS = Costs(fee=0.001, slippage=0.001, sell_tax=0.0)


# --------------------------------------------------------------------------- #
# 단위: 사이저 계약
# --------------------------------------------------------------------------- #
def test_fixed_fraction_matches_legacy_formula():
    cash, entry = 10000.0, 100.0
    qty = sizing.fixed_fraction(cash, entry, 90.0, COSTS, 1.0, {})
    expected = (cash * 1.0) / (entry * (1.0 + COSTS.fee))
    assert np.isclose(qty, expected)


def test_fixed_risk_targets_risk_pct():
    cash, entry, stop = 10000.0, 100.0, 98.0  # 단위리스크 2.0
    params = {"risk_pct": 0.01}               # 1% = 100달러 리스크
    qty = sizing.fixed_risk(cash, entry, stop, COSTS, 1.0, params)
    # 손절 도달 시 손실 = qty * (entry - stop) ≈ cash*risk_pct (상한에 안 걸릴 때)
    assert np.isclose(qty * (entry - stop), cash * 0.01, rtol=1e-9)


def test_fixed_risk_capped_by_notional():
    """단위리스크가 매우 작으면 고정리스크 수량이 커지지만 명목 상한으로 잘린다."""
    cash, entry, stop = 10000.0, 100.0, 99.99  # 단위리스크 0.01 → 리스크수량 거대
    params = {"risk_pct": 0.01}
    qty = sizing.fixed_risk(cash, entry, stop, COSTS, position_pct=1.0, params=params)
    cap = sizing.fixed_fraction(cash, entry, stop, COSTS, 1.0, {})
    assert np.isclose(qty, cap)  # 상한으로 잘림


def test_fixed_risk_zero_when_no_stop_distance():
    assert sizing.fixed_risk(10000.0, 100.0, 100.0, COSTS, 1.0, {}) == 0.0


def test_from_params_dispatch():
    assert sizing.from_params({}) is sizing.fixed_fraction          # 기본
    assert sizing.from_params({"sizing": "fixed_risk"}) is sizing.fixed_risk


# --------------------------------------------------------------------------- #
# 통합: 엔진 연동
# --------------------------------------------------------------------------- #
def test_engine_default_unchanged_regression():
    """사이저 미지정 백테스트는 기존과 완전 동일(회귀 안전망)."""
    df = random_walk_ohlcv(600, seed=7)
    params = default_params()
    a = bt.run_backtest(df, params, COSTS)
    b = bt.run_backtest(df, params, COSTS, sizer=sizing.fixed_fraction)
    assert a.equity.equals(b.equity)
    assert len(a.trades) == len(b.trades)


def test_engine_fixed_risk_reduces_exposure():
    """작은 risk_pct 고정리스크는 전액 대비 명목을 줄여 자산 변동을 작게 한다."""
    df = random_walk_ohlcv(1500, seed=3)
    params = default_params()
    full = bt.run_backtest(df, params, COSTS)  # 전액
    risk_params = {**params, "sizing": "fixed_risk", "risk_pct": 0.005}
    risk = bt.run_backtest(df, risk_params, COSTS)

    assert (risk.equity > 0).all()
    if len(full.trades) and len(risk.trades):
        # 첫 거래 명목: 고정리스크(0.5%)가 전액보다 작거나 같아야 한다.
        assert risk.trades[0].qty <= full.trades[0].qty + 1e-9


def test_engine_never_negative_cash_with_fixed_risk():
    df = random_walk_ohlcv(2000, seed=11)
    params = {**default_params(), "sizing": "fixed_risk", "risk_pct": 0.02}
    res = bt.run_backtest(df, params, COSTS)
    assert (res.equity > 0).all()
    assert np.isfinite(res.equity.to_numpy()).all()
