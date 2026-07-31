"""블록 부트스트랩 신뢰구간 테스트 — 구간 포함성·손익확률 방향성."""
from __future__ import annotations

import numpy as np

from src.validation import bootstrap


def test_ci_brackets_point_estimate():
    """점추정치가 신뢰구간 [lo, hi] 안에 있어야 한다."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 2000)
    res = bootstrap.bootstrap_confidence_intervals(r, 24 * 365, n_resamples=300, seed=1)
    for k, m in res["metrics"].items():
        assert m["lo"] <= m["point"] <= m["hi"], f"{k}: point가 구간 밖"


def test_prob_positive_high_for_strong_uptrend():
    rng = np.random.default_rng(1)
    r = rng.normal(0.002, 0.005, 2000)   # 뚜렷한 양의 드리프트
    res = bootstrap.bootstrap_confidence_intervals(r, 24 * 365, n_resamples=300, seed=2)
    assert res["prob_positive"] > 0.9


def test_prob_positive_low_under_volatility_drag():
    """산술평균 0인 노이즈는 변동성 드래그로 복리수익이 음의 편향(손익확률<0.5)."""
    rng = np.random.default_rng(7)
    r = rng.normal(0.0, 0.015, 3000)
    res = bootstrap.bootstrap_confidence_intervals(r, 24 * 365, n_resamples=400, seed=8)
    assert res["prob_positive"] < 0.5


def test_wider_ci_for_noisier_series():
    """변동성이 크면 총수익 신뢰구간이 넓어야 한다."""
    rng = np.random.default_rng(3)
    calm = rng.normal(0.0005, 0.005, 2000)
    wild = rng.normal(0.0005, 0.02, 2000)
    c = bootstrap.bootstrap_confidence_intervals(calm, 24 * 365, n_resamples=300, seed=4)
    w = bootstrap.bootstrap_confidence_intervals(wild, 24 * 365, n_resamples=300, seed=4)
    cw = c["metrics"]["total_return"]
    ww = w["metrics"]["total_return"]
    assert (ww["hi"] - ww["lo"]) > (cw["hi"] - cw["lo"])


def test_too_short_returns_nan_gracefully():
    res = bootstrap.bootstrap_confidence_intervals([0.01, 0.02], 24 * 365)
    assert res["n_resamples"] == 0
    assert res["prob_positive"] != res["prob_positive"]  # NaN


def test_deterministic_with_seed():
    rng = np.random.default_rng(5)
    r = rng.normal(0.0003, 0.01, 1500)
    a = bootstrap.bootstrap_confidence_intervals(r, 24 * 365, n_resamples=200, seed=42)
    b = bootstrap.bootstrap_confidence_intervals(r, 24 * 365, n_resamples=200, seed=42)
    assert a["metrics"]["Sharpe"]["lo"] == b["metrics"]["Sharpe"]["lo"]
    assert a["prob_positive"] == b["prob_positive"]
