"""검증 레이어 테스트 — 분할 인과성, PBO/DSR 수학 성질, 스윕 정합성.

원칙(기존 무결성 규율 연장):
  - 워크포워드/purged 분할은 train이 test보다 과거이거나 purge된 원거리여야 한다.
  - PBO는 알려진 극단(순수노이즈 vs 진짜 우수설정)에서 이론값에 수렴해야 한다.
  - DSR은 시도 수가 늘수록(다중검정) 낮아져야 한다.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.validation import splits, pbo, sweep
from tests.helpers import default_params, random_walk_ohlcv


# --------------------------------------------------------------------------- #
# splits — 인과성 / 누수 방지
# --------------------------------------------------------------------------- #
def test_walk_forward_train_precedes_test():
    folds = splits.walk_forward_splits(1000, n_splits=5, train_min_frac=0.5,
                                       mode="anchored")
    assert len(folds) == 5
    for train, test in folds:
        assert train.max() < test.min()          # train은 test보다 과거
        assert len(np.intersect1d(train, test)) == 0


def test_walk_forward_rolling_fixed_window():
    folds = splits.walk_forward_splits(1000, n_splits=5, train_min_frac=0.4,
                                       mode="rolling")
    lengths = {len(train) for train, _ in folds}
    # rolling은 학습창 길이가 (거의) 일정. anchored라면 누적 증가.
    assert max(lengths) - min(lengths) <= 1


def test_walk_forward_embargo_gap():
    emb = 10
    folds = splits.walk_forward_splits(1000, n_splits=4, train_min_frac=0.5,
                                       mode="anchored", embargo=emb)
    for train, test in folds:
        assert test.min() - train.max() - 1 >= emb - 1  # 완충 구간 존재


def test_purged_kfold_removes_horizon_and_embargo():
    n, k, h, e = 500, 5, 20, 15
    folds = splits.purged_kfold_splits(n, n_splits=k, embargo=e, horizon=h)
    assert len(folds) == k
    for train, test in folds:
        assert len(np.intersect1d(train, test)) == 0
        ts, te = test.min(), test.max()
        # purge 구간 [ts-h, te+e) 안에는 학습표본이 없어야 한다.
        forbidden = set(range(max(0, ts - h), min(n, te + e + 1)))
        assert forbidden.isdisjoint(set(train.tolist()))


def test_walk_forward_rejects_bad_fracs():
    with pytest.raises(ValueError):
        splits.walk_forward_splits(100, train_min_frac=0.0)
    with pytest.raises(ValueError):
        splits.walk_forward_splits(100, train_min_frac=1.0)


# --------------------------------------------------------------------------- #
# PBO — 알려진 극단에서의 수렴
# --------------------------------------------------------------------------- #
def test_pbo_pure_noise_is_near_half():
    """순수 노이즈 설정들(진짜 우열 없음)의 PBO는 0.5 근방이어야 한다."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 1, size=(2000, 20))
    res = pbo.cscv_pbo(returns, n_blocks=14)
    assert 0.30 <= res["pbo"] <= 0.70


def test_pbo_one_genuinely_best_is_low():
    """한 설정이 IS·OOS 모두에서 일관되게 우수하면 PBO는 낮아야 한다."""
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 1, size=(2000, 20))
    returns[:, 0] += 0.5  # 0번 설정에 지속적 우위(전 구간 균질)
    res = pbo.cscv_pbo(returns, n_blocks=14)
    assert res["pbo"] < 0.10
    # 지속 우위 설정은 OOS에서 손실로 뒤집히는 일이 드물다.
    # (perf_degradation 회귀기울기는 단일 지배설정 시 상보분할 반상관 아티팩트를
    #  타므로 부호를 단정하지 않는다 — 대신 OOS 손실확률로 견고성을 확인.)
    assert res["prob_oos_loss"] < 0.05


def test_pbo_requires_even_blocks_and_multiple_configs():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        pbo.cscv_pbo(rng.normal(0, 1, (100, 5)), n_blocks=7)  # 홀수
    with pytest.raises(ValueError):
        pbo.cscv_pbo(rng.normal(0, 1, (100, 1)), n_blocks=4)  # 설정 1개


# --------------------------------------------------------------------------- #
# Deflated Sharpe — 다중검정 성질
# --------------------------------------------------------------------------- #
def test_expected_max_sharpe_grows_with_trials():
    """시도 수가 많을수록 우연 기대 최대 Sharpe(문턱)는 커진다."""
    e10 = pbo.expected_max_sharpe_ratio(sr_variance=1.0, n_trials=10)
    e100 = pbo.expected_max_sharpe_ratio(sr_variance=1.0, n_trials=100)
    assert e10 < e100
    assert pbo.expected_max_sharpe_ratio(1.0, n_trials=1) == 0.0


def test_dsr_decreases_with_more_trials():
    """같은 수익률이라도 '더 많이 시도해 얻은 것'이면 DSR은 낮아진다."""
    rng = np.random.default_rng(3)
    r = rng.normal(0.02, 1.0, 1000)  # 약한 양의 Sharpe
    trials_few = np.array([0.02, 0.01, 0.0])
    trials_many = np.concatenate([[0.02], rng.normal(0, 0.03, 200)])
    dsr_few = pbo.deflated_sharpe_ratio(r, sr_trials=trials_few)["dsr"]
    dsr_many = pbo.deflated_sharpe_ratio(r, sr_trials=trials_many)["dsr"]
    assert dsr_many <= dsr_few


def test_dsr_strong_signal_passes():
    """다중검정을 감안해도 강한 신호는 높은 DSR."""
    rng = np.random.default_rng(4)
    r = rng.normal(0.15, 1.0, 2000)
    res = pbo.deflated_sharpe_ratio(r, n_trials=1)
    assert res["dsr"] > 0.95


# --------------------------------------------------------------------------- #
# sweep — 엔진 연동 정합성
# --------------------------------------------------------------------------- #
def test_expand_grid_cardinality():
    base = default_params()
    grid = {"macd_fast": [8, 12], "rsi_period": [7, 9, 14]}
    out = sweep.expand_grid(base, grid)
    assert len(out) == 6
    assert all(p["macd_slow"] == base["macd_slow"] for p in out)  # base 유지
    assert {(p["macd_fast"], p["rsi_period"]) for p in out} == {
        (8, 7), (8, 9), (8, 14), (12, 7), (12, 9), (12, 14)
    }
    assert sweep.expand_grid(base, {}) == [base]


def test_run_sweep_returns_aligned_matrix():
    from src.engine.position import Costs
    df = random_walk_ohlcv(600, seed=11)
    base = default_params()
    params_list = sweep.expand_grid(base, {"macd_fast": [8, 12], "rsi_period": [9, 14]})
    costs = Costs(fee=0.001, slippage=0.001, sell_tax=0.0)
    res = sweep.run_sweep(df, params_list, costs, bars_per_year=24 * 365)

    assert res["returns"].shape == (len(df), 4)      # T x N 정렬
    assert len(res["metrics"]) == 4
    assert res["sr_trials"].shape == (4,)
    assert np.isfinite(res["returns"].to_numpy()).all()


def test_walk_forward_analysis_survives_no_trade_strategy():
    """어떤 설정도 거래 0건이면 전 폴드 Sharpe가 NaN → 예전엔 best_i=None 크래시.
    회귀 방지: 폴드가 유효한 설정을 갖고 정상 반환해야 한다."""
    from src.engine.position import Costs
    from src.strategies.base import StrategySpec

    def _never_enter(df, params):
        out = df.copy()
        out["enter_long"] = False
        out["exit_all"] = False
        out["exit_half"] = False
        return out

    spec = StrategySpec(
        name="_no_trade", category="test", description="진입 없음",
        default_params={"swing_lookback_M": 10, "stop_buffer": 0.001, "reward_ratio": 2.0},
        param_grid={"reward_ratio": [1.5, 2.0]},
        generate_signals=_never_enter, warmup_bars=lambda p: 5,
        exit_col="exit_all", exit_half_col="exit_half",
    )
    df = random_walk_ohlcv(800, seed=13)
    params_list = sweep.expand_grid(spec.default_params, spec.param_grid)
    costs = Costs(fee=0.001, slippage=0.001, sell_tax=0.0)
    res = sweep.walk_forward_analysis(
        df, params_list, costs, bars_per_year=24 * 365, n_splits=3, strategy=spec,
    )
    assert res["n_folds"] == 3
    for f in res["folds"]:
        assert f["chosen_index"] is not None
        assert f["chosen_params"] in params_list


def test_walk_forward_analysis_runs_and_reports_degradation():
    from src.engine.position import Costs
    df = random_walk_ohlcv(1200, seed=12)
    base = default_params()
    params_list = sweep.expand_grid(base, {"macd_fast": [8, 12], "hma_period": [20, 50]})
    costs = Costs(fee=0.001, slippage=0.001, sell_tax=0.0)
    res = sweep.walk_forward_analysis(
        df, params_list, costs, bars_per_year=24 * 365,
        n_splits=4, mode="rolling",
    )
    assert res["n_folds"] == 4
    assert len(res["folds"]) == 4
    for f in res["folds"]:
        assert f["chosen_params"] in params_list
    assert "degradation" in res
