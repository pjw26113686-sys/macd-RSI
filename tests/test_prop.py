"""프랍 룰 평가기 — 트레일링DD·수익목표·일일손실·일관성 판정 검증."""
import pandas as pd

from src.prop.evaluator import evaluate_prop
from src.prop.rules import get_ruleset


def _series(values, freq="1D"):
    idx = pd.date_range("2024-01-02 18:00", periods=len(values), freq=freq,
                        tz="UTC", name="time")
    s = pd.Series(values, index=idx, dtype=float)
    return s


def test_trailing_dd_fail():
    rules = get_ruleset("apex_50k")          # trailing 2500
    eq = _series([50000, 51000, 48400])      # 고점 51000 → 임계 48500, 48400 하회
    r = evaluate_prop(eq, eq, eq, rules)
    assert r.failed and r.fail_reason == "TRAILING_DD"
    assert not r.passed


def test_trailing_dd_uses_intraday_low():
    rules = get_ruleset("apex_50k")
    close = _series([50000, 51000, 48600])   # 종가만 보면 통과(48600>48500)
    low = _series([50000, 51000, 48400])     # 봉 저가(wick)는 임계 하회
    r = evaluate_prop(close, close, low, rules)
    assert r.failed and r.fail_reason == "TRAILING_DD"   # wick에서 실격


def test_profit_target_pass():
    rules = get_ruleset("apex_50k")          # target 3000
    eq = _series([50000, 51000, 52000, 53000])
    r = evaluate_prop(eq, eq, eq, rules)
    assert r.passed and not r.failed
    assert r.days_to_pass == 4


def test_daily_loss_fail_topstep():
    rules = get_ruleset("topstep_50k")       # daily_loss 1000, eod trailing 2000
    eq = _series([49800, 48700], freq="1h")  # 같은 세션 내 -1100
    r = evaluate_prop(eq, eq, eq, rules)
    assert r.failed and r.fail_reason == "DAILY_LOSS"


def test_consistency_violation():
    rules = get_ruleset("apex_50k")          # consistency 0.30
    # 일별 +500, +2000 → 총 +2500, 최고일 2000 = 80% > 30%
    eq = _series([50000, 50500, 52500])
    r = evaluate_prop(eq, eq, eq, rules)
    assert not r.passed and not r.failed     # 목표(3000) 미달, 실격도 아님
    assert r.consistency_ratio > 0.30
    assert r.passed_consistency is False


def test_clean_pass_consistency_ok():
    rules = get_ruleset("apex_50k")
    # 균등 상승으로 목표 달성 + 단일일 비중 낮음
    eq = _series([50000, 50800, 51600, 52400, 53200])
    r = evaluate_prop(eq, eq, eq, rules)
    assert r.passed
    assert r.passed_consistency is True
