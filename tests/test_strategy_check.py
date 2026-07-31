"""전략 합격 판정 하네스 테스트.

핵심: 하네스가 (1) 정상 전략을 통과시키고, (2) 미래참조/계약위반 전략을 확실히
FAIL로 잡고, (3) 임의의 .py 파일에서 전략을 로드할 수 있어야 한다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import strategy_check as sc
from src import strategies as strat
from src.engine.position import Costs
from src.strategies.base import StrategySpec
from tests.helpers import random_walk_ohlcv

COSTS = Costs(fee=0.001, slippage=0.001, sell_tax=0.0)
BT_CFG = {"initial_capital": 10000.0, "position_pct": 1.0}
DF = random_walk_ohlcv(800, seed=4)


# --------------------------------------------------------------------------- #
# 정상 전략은 무결성 게이트를 통과한다
# --------------------------------------------------------------------------- #
def test_builtin_strategies_pass_integrity():
    for spec in strat.all_specs():
        checks = sc.run_integrity_gauntlet(spec, DF, COSTS, BT_CFG)
        fails = [c for c in checks if c.is_gate_failure]
        assert not fails, f"{spec.name} 무결성 실패: {[(c.name, c.detail) for c in fails]}"


# --------------------------------------------------------------------------- #
# 미래참조 전략은 lookahead 게이트가 잡는다
# --------------------------------------------------------------------------- #
def _leaky_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    # 미래참조: 다음 봉 종가가 더 높으면 진입 (shift(-1) = 미래).
    out["enter_long"] = out["close"] < out["close"].shift(-1)
    out["exit_all"] = out["close"] > out["close"].shift(-1)
    out["exit_half"] = out["exit_all"]
    for c in ["enter_long", "exit_all", "exit_half"]:
        out[c] = out[c].fillna(False)
    return out


LEAKY_SPEC = StrategySpec(
    name="_leaky_test", category="test", description="미래참조 일부러 넣은 전략",
    default_params={"swing_lookback_M": 10, "stop_buffer": 0.001, "reward_ratio": 2.0},
    param_grid={"reward_ratio": [1.5, 2.0]},
    generate_signals=_leaky_signals, warmup_bars=lambda p: 5,
    exit_col="exit_all", exit_half_col="exit_half",
)


def test_lookahead_gate_catches_future_reference():
    res = sc.check_lookahead(LEAKY_SPEC, DF)
    assert res.status == sc.FAIL
    assert "누수" in res.detail


def test_gauntlet_marks_leaky_strategy_as_failed():
    checks = sc.run_integrity_gauntlet(LEAKY_SPEC, DF, COSTS, BT_CFG)
    _, gate_ok = sc.format_gauntlet(LEAKY_SPEC, checks, None)
    assert gate_ok is False


# --------------------------------------------------------------------------- #
# 계약 위반 (컬럼 누락)도 잡는다
# --------------------------------------------------------------------------- #
def test_contract_gate_catches_missing_column():
    bad = StrategySpec(
        name="_bad_test", category="test", description="청산 컬럼 누락",
        default_params={"swing_lookback_M": 10, "stop_buffer": 0.001, "reward_ratio": 2.0},
        param_grid={"reward_ratio": [1.5, 2.0]},
        generate_signals=lambda df, p: df.assign(enter_long=False),  # exit 컬럼 없음
        warmup_bars=lambda p: 5, exit_col="exit_all", exit_half_col="exit_half",
    )
    res = sc.check_contract(bad, DF)
    assert res.status == sc.FAIL
    assert "누락" in res.detail


# --------------------------------------------------------------------------- #
# 임의의 .py 파일에서 전략 로드 ("가져오면")
# --------------------------------------------------------------------------- #
def test_load_spec_from_template_module():
    template = Path(__file__).resolve().parents[1] / "src" / "strategies" / "_template.py"
    spec = sc.load_spec_from_module(str(template))
    assert isinstance(spec, StrategySpec)
    assert spec.name == "template_ma_cross"
    # 로드한 전략이 무결성 게이트를 통과하는지까지 확인.
    checks = sc.run_integrity_gauntlet(spec, DF, COSTS, BT_CFG)
    assert not [c for c in checks if c.is_gate_failure]
