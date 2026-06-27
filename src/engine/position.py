"""포지션/거래 기록 + 손절·목표 레벨 계산 (방향 일반화).

손익 계산·체결가·사이징은 `src.engine.pnl`의 PnLModel이 담당한다. 여기서는
포지션 상태(가변)와 거래 집계, 그리고 청산 레벨(손절/목표) 계산만 책임진다.

상태: FULL(전량 보유) / HALF(1차 익절 후 잔량). FLAT는 position=None으로 표현.
방향: LONG / SHORT.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.engine.pnl import LONG, SHORT

FULL = "FULL"
HALF = "HALF"


@dataclass
class Fill:
    """단일 체결 기록."""
    time: object
    bar_index: int
    side: str           # "buy" | "sell" | "cover" 등(로깅용)
    price: float        # 슬리피지 반영 체결가
    qty: float
    reason: str


@dataclass
class Trade:
    """하나의 포지션(진입~완전청산) 집계."""
    entry_time: object
    entry_index: int
    entry_price: float
    qty: float                  # 최초 진입 수량(계약수 또는 명목수량)
    direction: str              # LONG | SHORT
    stop_loss: float
    target: float | None
    risk_basis: float           # R-multiple/수익률 정규화 기준($)
    realized: float = 0.0       # 진입+청산 누적 현금변화(=순손익$)
    exits: list = field(default_factory=list)   # list[Fill]
    exit_time: object = None
    exit_index: int = None
    last_reason: str = ""

    @property
    def pnl(self) -> float:
        return self.realized

    @property
    def ret(self) -> float:
        return self.realized / self.risk_basis if self.risk_basis else 0.0

    @property
    def bars_held(self) -> int:
        return (self.exit_index - self.entry_index) if self.exit_index is not None else 0

    @property
    def is_win(self) -> bool:
        return self.realized > 0


@dataclass
class Position:
    """진행 중 포지션의 가변 상태."""
    entry_price: float
    qty: float          # 현재 남은 수량
    direction: str
    stop_loss: float
    target: float | None
    state: str          # FULL | HALF
    trade: Trade


# --------------------------------------------------------------------------- #
# 손절 / 목표 레벨 계산 (방향 일반화)
# --------------------------------------------------------------------------- #
def compute_stop_and_target(entry_fill, ref_extreme, params, direction, exit_model,
                            instrument=None, exit_target=None):
    """(stop_loss, target) 반환. 무효(손절폭<=0)면 stop=None.

    ref_extreme : swing 모드의 직전 극단(롱=swing_low, 숏=swing_high).
    exit_target : target_mode='mid'일 때 사용할 목표가(예: 볼린저 중심선).
    """
    buf = params.get("stop_buffer", 0.0)

    # ---- 손절 ----
    if exit_model.stop_mode == "swing":
        if direction == LONG:
            stop = ref_extreme * (1.0 - buf)
        else:
            stop = ref_extreme * (1.0 + buf)
    elif exit_model.stop_mode == "fixed":
        if instrument is None:
            raise ValueError("stop_mode='fixed'는 instrument가 필요합니다.")
        dist = params["stop_ticks"] * instrument.tick_size
        stop = entry_fill - dist if direction == LONG else entry_fill + dist
    else:
        raise ValueError(f"지원하지 않는 stop_mode: {exit_model.stop_mode}")

    # 유효성: 손절은 진입가 반대편에 있어야 함
    if direction == LONG and stop >= entry_fill:
        return None, None
    if direction == SHORT and stop <= entry_fill:
        return None, None

    # ---- 목표 ----
    if exit_model.target_mode == "none":
        target = None
    elif exit_model.target_mode == "mid":
        target = exit_target   # 엔진이 봉마다 갱신할 수도 있으나 진입시점 값 사용
    elif exit_model.target_mode == "rr":
        R = abs(entry_fill - stop)
        rr = params.get("reward_ratio", 2.0)
        target = entry_fill + rr * R if direction == LONG else entry_fill - rr * R
    else:
        raise ValueError(f"지원하지 않는 target_mode: {exit_model.target_mode}")

    return stop, target
