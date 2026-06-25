"""포지션 상태머신 + 비용/체결 헬퍼 (strategy_spec_v2 §4, §5).

상태: FLAT(미보유) / FULL(전량 보유) / HALF(1차 익절 후 본전스탑 잔량).
실제 봉 순회/신호 평가는 backtest.py가 담당하고, 여기서는
가격 레벨 계산·체결가(슬리피지/수수료)·포지션 객체를 제공한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

FULL = "FULL"
HALF = "HALF"


@dataclass
class Costs:
    fee: float          # 편도 수수료율
    slippage: float     # 편도 슬리피지율
    sell_tax: float = 0.0  # 매도 거래세 (한국주식용; 1차 미국/암호화폐는 0)


@dataclass
class Fill:
    """단일 체결 기록."""
    time: object
    bar_index: int
    side: str           # "buy" | "sell"
    price: float        # 슬리피지 반영 후 체결가
    qty: float
    reason: str


@dataclass
class Trade:
    """하나의 포지션(진입~완전청산)을 집계한 거래 기록."""
    entry_time: object
    entry_index: int
    entry_price: float          # 슬리피지 반영 체결가
    qty: float                  # 최초 진입 수량
    stop_loss: float
    target: float
    cost: float                 # 진입 총비용(수수료 포함)
    exits: list = field(default_factory=list)  # list[Fill]
    proceeds: float = 0.0       # 청산 총수령액(수수료/세금 차감 후)
    exit_time: object = None
    exit_index: int = None
    last_reason: str = ""

    @property
    def pnl(self) -> float:
        return self.proceeds - self.cost

    @property
    def ret(self) -> float:
        return self.pnl / self.cost if self.cost else 0.0

    @property
    def bars_held(self) -> int:
        return (self.exit_index - self.entry_index) if self.exit_index is not None else 0

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class Position:
    """진행 중 포지션의 가변 상태."""
    entry_price: float
    qty: float          # 현재 남은 수량
    stop_loss: float
    target: float
    state: str          # FULL | HALF
    trade: Trade


# --------------------------------------------------------------------------- #
# 가격 레벨 계산 (§4.1, §4.2)
# --------------------------------------------------------------------------- #
def compute_stop_and_target(entry_price: float, swing_low: float, params: dict):
    """손절가 = swing_low*(1-buffer), 목표가 = entry*(1 + reward*R)."""
    stop_loss = swing_low * (1.0 - params["stop_buffer"])
    R = (entry_price - stop_loss) / entry_price
    target = entry_price * (1.0 + params["reward_ratio"] * R)
    return stop_loss, target, R


# --------------------------------------------------------------------------- #
# 체결가 (슬리피지)
# --------------------------------------------------------------------------- #
def buy_fill(raw_price: float, costs: Costs) -> float:
    return raw_price * (1.0 + costs.slippage)


def sell_fill(raw_price: float, costs: Costs) -> float:
    return raw_price * (1.0 - costs.slippage)


def buy_notional_cost(price: float, qty: float, costs: Costs) -> float:
    """매수 총비용(수수료 포함)."""
    return price * qty * (1.0 + costs.fee)


def sell_proceeds(price: float, qty: float, costs: Costs) -> float:
    """매도 총수령액(수수료·거래세 차감)."""
    return price * qty * (1.0 - costs.fee - costs.sell_tax)
