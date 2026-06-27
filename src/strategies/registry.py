"""전략 레지스트리 — 새 전략 추가 = 여기 STRATEGIES dict에 한 줄.

compare/run이 이 매핑에서 전략을 찾아 순회한다.
"""
from __future__ import annotations

from src.strategies.base import Strategy
from src.strategies.example_bollinger import BollingerReversionStrategy
from src.strategies.example_ma_cross import MaCrossStrategy

# 이름 → 전략 클래스
STRATEGIES: dict[str, type[Strategy]] = {
    MaCrossStrategy.name: MaCrossStrategy,
    BollingerReversionStrategy.name: BollingerReversionStrategy,
}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f"알 수 없는 전략: {name}. 사용 가능: {sorted(STRATEGIES)}")
    return STRATEGIES[name]()


def all_strategies() -> list[Strategy]:
    return [cls() for cls in STRATEGIES.values()]
