"""판정카드 — PBO/DSR/성능감쇠를 일반인이 읽을 수 있는 신뢰/주의/기각으로.

기관은 "Sharpe 2.0!"만 보지 않는다. 그 숫자가 다중검정·과최적화·비정규성을 견디는지
본다. 이 모듈은 그 판정을 세 축의 신호등으로 요약한다.
"""
from __future__ import annotations


# 문턱값 — 보수적 기본. 필요 시 호출측에서 재정의.
PBO_WARN = 0.20     # 이 이상이면 주의
PBO_FAIL = 0.50     # 이 이상이면 사실상 과최적화(무작위)
DSR_PASS = 0.95     # 이 이상이면 통계적으로 유의
DSR_WARN = 0.90


def _verdict_level(pbo: float, dsr: float, degradation: float) -> tuple[str, str]:
    """(레벨, 한줄사유). 레벨 ∈ {"신뢰", "주의", "기각"}."""
    reasons = []
    fail = False
    warn = False

    if pbo != pbo:  # NaN
        warn = True
        reasons.append("PBO 계산불가(설정/표본 부족)")
    elif pbo >= PBO_FAIL:
        fail = True
        reasons.append(f"PBO {pbo:.0%} ≥ {PBO_FAIL:.0%} (과최적화)")
    elif pbo >= PBO_WARN:
        warn = True
        reasons.append(f"PBO {pbo:.0%} 다소 높음")

    if dsr == dsr:
        if dsr < DSR_WARN:
            fail = True
            reasons.append(f"DSR {dsr:.0%} < {DSR_WARN:.0%} (우연 넘지 못함)")
        elif dsr < DSR_PASS:
            warn = True
            reasons.append(f"DSR {dsr:.0%} 경계")

    if degradation == degradation and degradation > 0:
        # OOS에서 select 지표가 IS 대비 하락. 절대 하락폭이 크면 경고.
        if degradation >= 1.0:
            warn = True
            reasons.append(f"IS→OOS 감쇠 {degradation:.2f}")

    if fail:
        return "기각", "; ".join(reasons)
    if warn:
        return "주의", "; ".join(reasons) if reasons else "경계값"
    return "신뢰", "PBO 낮고 DSR 유의, 감쇠 경미"


def format_validation_report(
    title: str,
    pbo_result: dict,
    dsr_result: dict,
    wfa_result: dict | None = None,
    ci_result: dict | None = None,
) -> str:
    """검증 산출물을 사람이 읽는 판정카드 문자열로."""
    pbo = pbo_result["pbo"]
    dsr = dsr_result["dsr"]
    degradation = wfa_result["degradation"] if wfa_result else float("nan")
    level, reason = _verdict_level(pbo, dsr, degradation)

    icon = {"신뢰": "✅", "주의": "⚠️", "기각": "⛔"}[level]

    def pct(x, d=1):
        return "n/a" if x != x else f"{x*100:.{d}f}%"

    def num(x, d=2):
        return "n/a" if x != x else f"{x:.{d}f}"

    lines = [
        f"╔══════ 오버피팅 검증 · {title} ══════",
        f"║  판정            : {icon} {level}  ({reason})",
        "║  ────────────────────────────────────",
        f"║  PBO(과최적화확률) : {pct(pbo)}   (낮을수록 견고, {PBO_WARN:.0%}↑ 주의)",
        f"║    └ OOS 손실확률  : {pct(pbo_result['prob_oos_loss'])}",
        f"║    └ IS→OOS 회귀   : {num(pbo_result['perf_degradation'])} (1=감쇠없음, <0=뒤집힘)",
        f"║    └ 평가 조합수   : {pbo_result['n_combos']}",
        f"║  Deflated Sharpe  : {pct(dsr)}   (참Sharpe>우연 문턱일 확률, {DSR_PASS:.0%}↑ 유의)",
        f"║    └ 관측 Sharpe   : {num(dsr_result['sr_observed'], 3)} (관측단위)",
        f"║    └ 우연 문턱 SR₀ : {num(dsr_result['sr0'], 3)}  (시도 {dsr_result['n_trials']}회 기준)",
        f"║    └ 왜도/첨도     : {num(dsr_result['skew'])} / {num(dsr_result['kurtosis'])}",
    ]
    if wfa_result:
        lines += [
            "║  ────────────────────────────────────",
            f"║  워크포워드({wfa_result['n_folds']}폴드, {wfa_result['select_by']} 기준)",
            f"║    └ IS 평균      : {num(wfa_result['is_mean'])}",
            f"║    └ OOS 평균     : {num(wfa_result['oos_mean'])}",
            f"║    └ 감쇠         : {num(wfa_result['degradation'])} (양수=OOS 하락)",
        ]
        oos = wfa_result.get("oos_metrics", {})
        if "total_return" in oos:
            lines.append(f"║    └ OOS 총수익률 : {pct(oos['total_return'])} (폴드평균)")
        if "MDD" in oos:
            lines.append(f"║    └ OOS MDD      : {pct(oos['MDD'])}")
    if ci_result and ci_result.get("n_resamples"):
        cim = ci_result["metrics"]
        lines += [
            "║  ────────────────────────────────────",
            f"║  신뢰구간(부트스트랩 {ci_result['n_resamples']}회, {ci_result['ci']:.0%})",
            f"║    └ 총수익률   : {pct(cim['total_return']['point'])}  "
            f"[{pct(cim['total_return']['lo'])}, {pct(cim['total_return']['hi'])}]",
            f"║    └ Sharpe    : {num(cim['Sharpe']['point'])}  "
            f"[{num(cim['Sharpe']['lo'])}, {num(cim['Sharpe']['hi'])}]",
            f"║    └ MDD       : {pct(cim['MDD']['point'])}  "
            f"[{pct(cim['MDD']['lo'])}, {pct(cim['MDD']['hi'])}]",
            f"║    └ 손익확률   : {pct(ci_result['prob_positive'])}",
        ]
    lines.append("╚═══════════════════════════════════════")
    return "\n".join(lines)
