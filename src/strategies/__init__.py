"""카테고리별 전략 레지스트리.

이 패키지를 import하면 내장 전략들이 자동 등록된다. 새 전략은 모듈을 추가하고
아래 import 목록에 한 줄 넣으면(또는 register 호출) 즉시 스윕·검증 대상이 된다.

카테고리(현재): momentum / mean_reversion / breakout.
"""
from src.strategies.base import (
    StrategySpec,
    all_specs,
    by_category,
    categories,
    get,
    register,
)

# 내장 전략 자동 등록 (import 시 register 실행).
from src.strategies import breakout_donchian  # noqa: E402,F401
from src.strategies import mean_reversion_bollinger  # noqa: E402,F401
from src.strategies import momentum_macd_rsi  # noqa: E402,F401

__all__ = [
    "StrategySpec",
    "register",
    "get",
    "all_specs",
    "categories",
    "by_category",
]
