"""프랍 평가 룰셋 + 프리셋 (Apex 위주, Topstep 확장 가능).

> 주의: 아래 수치는 **출발값**이다. 프랍사 약관은 자주 바뀌므로 실제 챌린지
> 규정과 대조해 조정할 것. 구조(필드)는 두 회사를 모두 표현하도록 설계했다.

핵심 개념:
  - trailing_drawdown : 계좌 고점(인트라데이 또는 EOD)에서 이만큼 빠지면 실격.
  - trailing_type     : "intraday"(봉 고저로 추적, Apex) | "eod"(종가 기준).
  - trail_lock_at     : 트레일링 임계가 이 잔고에 도달하면 고정(None=계속 추적).
  - daily_loss_limit  : 하루 손실이 이만큼이면 실격(Topstep). Apex 평가=None.
  - profit_target     : 누적이익이 이만큼이면 통과.
  - consistency_pct   : 단일일 이익이 총이익의 이 비율을 넘으면 일관성 위반.
  - reset_tz/reset_time : 일경계(세션 리셋) 기준 시각(예: 미국 17:00 CT).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PropRuleSet:
    name: str
    start_balance: float
    profit_target: float
    trailing_drawdown: float
    trailing_type: str = "intraday"     # "intraday" | "eod"
    trail_lock_at: float | None = None  # 잔고 도달 시 트레일링 고정
    daily_loss_limit: float | None = None
    consistency_pct: float | None = None
    max_contracts: int = 10
    reset_tz: str = "America/Chicago"
    reset_time: int = 17                 # 17:00 CT 일경계


# Apex 프리셋 (평가단계 기준 출발값). 일관성 30%는 PA(펀딩) 출금 조건.
APEX_PRESETS: dict[str, PropRuleSet] = {
    "apex_25k": PropRuleSet("apex_25k", 25000, 1500, 1500,
                            consistency_pct=0.30, max_contracts=4),
    "apex_50k": PropRuleSet("apex_50k", 50000, 3000, 2500,
                            consistency_pct=0.30, max_contracts=10),
    "apex_100k": PropRuleSet("apex_100k", 100000, 6000, 3000,
                             consistency_pct=0.30, max_contracts=14),
    "apex_150k": PropRuleSet("apex_150k", 150000, 9000, 5000,
                             consistency_pct=0.30, max_contracts=17),
}

# Topstep 자리(daily_loss_limit 사용 · EOD 트레일링). 출발값.
TOPSTEP_PRESETS: dict[str, PropRuleSet] = {
    "topstep_50k": PropRuleSet("topstep_50k", 50000, 3000, 2000,
                               trailing_type="eod", daily_loss_limit=1000,
                               max_contracts=5),
}

ALL_PRESETS: dict[str, PropRuleSet] = {**APEX_PRESETS, **TOPSTEP_PRESETS}


def get_ruleset(name: str) -> PropRuleSet:
    if name not in ALL_PRESETS:
        raise KeyError(f"알 수 없는 프랍 룰셋: {name}. 사용 가능: {sorted(ALL_PRESETS)}")
    return ALL_PRESETS[name]
