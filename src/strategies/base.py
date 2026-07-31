"""전략 플러그인 계약 + 카테고리 레지스트리.

플랫폼의 척추. 모든 전략은 카테고리(momentum/mean_reversion/breakout/…)에 속하고,
동일한 계약을 따르므로 **어떤 전략이든 같은 엔진·검증 레이어로 돌릴 수 있다**.

계약(무결성 규율 연장):
  generate_signals(df, params) -> DataFrame
      OHLCV 인덱스(time)를 유지한 채, 최소한 다음 불리언 컬럼을 추가해 반환한다.
        - enter_long      : 진입 후보(종가 확정). 엔진이 다음 봉 시가에 체결.
        - <exit_col>      : 전량 신호청산 트리거.
        - <exit_half_col> : 분할익절 후 잔량(HALF) 신호청산 트리거.
      모든 컬럼은 t시점까지의 데이터만 사용해야 한다(미래참조 0). shift(+k)만 허용.
  warmup_bars(params) -> int
      지표 안정화 전 진입을 금지할 봉수(재귀편향 방지).

손절/목표(swing-low stop + reward_ratio)는 엔진이 공통으로 관리하므로, 전략
default_params에 stop_buffer / reward_ratio / swing_lookback_M를 포함시킨다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    name: str
    category: str
    description: str
    default_params: dict
    param_grid: dict                                    # 스윕용 {param: [values]}
    generate_signals: Callable[[pd.DataFrame, dict], pd.DataFrame]
    warmup_bars: Callable[[dict], int]
    exit_col: str = "exit_all"
    exit_half_col: str = "exit_half"
    tags: tuple = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# 레지스트리
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, StrategySpec] = {}


def register(spec: StrategySpec) -> StrategySpec:
    """전략을 전역 레지스트리에 등록. 이름 중복은 거부."""
    if spec.name in _REGISTRY:
        raise ValueError(f"전략 이름 중복: {spec.name!r}")
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> StrategySpec:
    if name not in _REGISTRY:
        raise KeyError(
            f"미등록 전략: {name!r}. 사용 가능: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def all_specs() -> list[StrategySpec]:
    return list(_REGISTRY.values())


def categories() -> list[str]:
    return sorted({s.category for s in _REGISTRY.values()})


def by_category(category: str) -> list[StrategySpec]:
    return [s for s in _REGISTRY.values() if s.category == category]
