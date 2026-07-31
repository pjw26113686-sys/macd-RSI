"""전략 평가 API 테스트 — UI가 소비하는 구조화 결과의 정합성.

app.py(Streamlit)는 렌더링만 하므로 로직 회귀는 여기서 evaluate_strategy로 잡는다.
"""
from __future__ import annotations

import numpy as np

from src import analysis
from src import strategies as strat
from src.engine.position import Costs
from src.strategies.base import StrategySpec
from tests.helpers import random_walk_ohlcv

COSTS = Costs(fee=0.001, slippage=0.001, sell_tax=0.0)
BT_CFG = {"initial_capital": 10000.0, "position_pct": 1.0}
BPY = 24 * 365


def test_buy_hold_equity_tracks_price():
    df = random_walk_ohlcv(300, seed=1)
    eq = analysis.buy_hold_equity(df, 10000.0)
    assert np.isclose(eq.iloc[0], 10000.0)
    # 최종/초기 비율이 종가 비율과 같아야 한다.
    assert np.isclose(eq.iloc[-1] / eq.iloc[0], df["close"].iloc[-1] / df["close"].iloc[0])


def test_evaluate_registered_strategy_full_result():
    spec = strat.get("breakout_donchian")
    df = random_walk_ohlcv(1500, seed=5)
    res = analysis.evaluate_strategy(
        spec, df, COSTS, BT_CFG, BPY, blocks=10, folds=3)

    assert res["gate_ok"] is True
    for key in ["pbo", "dsr", "wfa", "equity", "benchmark", "best_metrics",
                "verdict_level", "n_configs"]:
        assert key in res
    assert res["verdict_level"] in {"신뢰", "주의", "기각"}
    # 자산곡선과 벤치마크가 데이터 인덱스에 정렬돼 있어야(그래프용).
    assert len(res["equity"]) == len(df)
    assert 0.0 <= res["pbo"]["pbo"] <= 1.0


def test_evaluate_skips_performance_when_integrity_fails():
    """미래참조 전략은 gate_ok False, 성과 키는 없어야 한다."""
    def _leaky(df, params):
        out = df.copy()
        out["enter_long"] = out["close"] < out["close"].shift(-1)  # 미래참조
        out["exit_all"] = False
        out["exit_half"] = False
        return out

    leaky = StrategySpec(
        name="_leaky_analysis", category="test", description="누수",
        default_params={"swing_lookback_M": 10, "stop_buffer": 0.001, "reward_ratio": 2.0},
        param_grid={"reward_ratio": [1.5, 2.0]},
        generate_signals=_leaky, warmup_bars=lambda p: 5,
        exit_col="exit_all", exit_half_col="exit_half",
    )
    df = random_walk_ohlcv(600, seed=2)
    res = analysis.evaluate_strategy(leaky, df, COSTS, BT_CFG, BPY, do_wfa=False)
    assert res["gate_ok"] is False
    assert "pbo" not in res


def test_evaluate_sizing_changes_result():
    """fixed_risk 사이징이 최우수 설정 지표를 바꾼다(사이징이 실제 반영되는지)."""
    spec = strat.get("breakout_donchian")
    df = random_walk_ohlcv(1500, seed=8)
    a = analysis.evaluate_strategy(spec, df, COSTS, BT_CFG, BPY, blocks=10,
                                   do_wfa=False, sizing="fixed_fraction")
    b = analysis.evaluate_strategy(spec, df, COSTS, BT_CFG, BPY, blocks=10,
                                   do_wfa=False, sizing="fixed_risk", risk_pct=0.005)
    # 사이징이 다르면 최종자산도 달라야 한다(동일하면 사이징 미반영).
    assert a["best_metrics"]["final_equity"] != b["best_metrics"]["final_equity"]
