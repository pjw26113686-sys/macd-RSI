"""손익(P&L) 모델 — 엔진이 시장종류와 무관하게 동작하도록 손익 계산을 위임.

두 구현:
  - NotionalPnL : crypto/stock. % 비용, 롱 전용(기존 동작 보존).
  - FuturesPnL  : CME 선물. 계약·틱·달러 손익, 롱/숏 양방향.

엔진(backtest.py)은 아래 공통 인터페이스만 호출한다:
  entry_fill / exit_fill   : 슬리피지 반영 체결가
  size                     : 진입 수량(노셔널=명목수량, 선물=계약수)
  entry_cash_delta         : 진입 시 현금 변화(음수=유출)
  mark                     : 보유 포지션이 equity에 더하는 평가액
  exit_cash_delta          : 청산 시 현금 변화

equity = cash + mark(price). 두 모델 모두 이 항등식을 만족하도록 정의한다.
  - 노셔널 롱: 진입에 명목금액을 cash에서 지불, mark=수량×현재가.
  - 선물    : 진입은 수수료만 cash 차감, mark=방향×(현재가-진입가)×포인트가치×계약수.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.instruments import Instrument

LONG = "LONG"
SHORT = "SHORT"


@dataclass
class SizingConfig:
    """포지션 사이징 설정(config의 sizing 블록)."""
    mode: str = "fixed"          # "fixed" | "risk"
    contracts: int = 1           # mode=fixed (선물)
    position_pct: float = 1.0    # 노셔널(crypto/stock) 명목 비중
    risk_dollars: float = 250.0  # mode=risk: 거래당 손절폭 리스크($)
    max_contracts: int = 10      # mode=risk 상한 (선물)


# --------------------------------------------------------------------------- #
# 노셔널 (crypto / stock) — 롱 전용, % 비용
# --------------------------------------------------------------------------- #
@dataclass
class NotionalPnL:
    fee: float = 0.0
    slippage: float = 0.0
    sell_tax: float = 0.0

    def entry_fill(self, raw: float, direction: str = LONG) -> float:
        return raw * (1.0 + self.slippage)   # 롱 매수: 불리하게 위로

    def exit_fill(self, raw: float, direction: str = LONG) -> float:
        return raw * (1.0 - self.slippage)   # 롱 매도: 불리하게 아래로

    def size(self, equity, entry_fill, stop, direction, cfg: SizingConfig) -> float:
        return (equity * cfg.position_pct) / (entry_fill * (1.0 + self.fee))

    def entry_cash_delta(self, entry_fill, qty, direction) -> float:
        return -(entry_fill * qty * (1.0 + self.fee))   # 명목금액 지불

    def mark(self, price, entry_fill, qty, direction) -> float:
        return qty * price                              # 보유 평가액

    def exit_cash_delta(self, entry_fill, exit_fill, qty, direction) -> float:
        return exit_fill * qty * (1.0 - self.fee - self.sell_tax)

    def risk_basis(self, entry_fill, stop, qty, direction) -> float:
        return entry_fill * qty * (1.0 + self.fee)   # 명목 원가 기준


# --------------------------------------------------------------------------- #
# 선물 (CME) — 롱/숏, 계약·틱·달러
# --------------------------------------------------------------------------- #
@dataclass
class FuturesPnL:
    instrument: Instrument

    def _slip(self, direction: str) -> float:
        return self.instrument.slippage_ticks * self.instrument.tick_size

    def entry_fill(self, raw: float, direction: str) -> float:
        s = self._slip(direction)
        return raw + s if direction == LONG else raw - s  # 진입은 불리하게

    def exit_fill(self, raw: float, direction: str) -> float:
        s = self._slip(direction)
        return raw - s if direction == LONG else raw + s  # 청산도 불리하게

    def size(self, equity, entry_fill, stop, direction, cfg: SizingConfig) -> float:
        if cfg.mode == "fixed":
            return float(max(0, int(cfg.contracts)))
        # risk 기반: 계약수 = floor( 리스크$ / (손절폭 × 포인트가치) ), 상한 캡
        stop_dist = abs(entry_fill - stop)
        if stop_dist <= 0:
            return 0.0
        risk_per_contract = stop_dist * self.instrument.point_value
        n = math.floor(cfg.risk_dollars / risk_per_contract)
        return float(max(0, min(n, cfg.max_contracts)))

    def entry_cash_delta(self, entry_fill, qty, direction) -> float:
        return -(self.instrument.commission_per_side * qty)   # 진입 수수료만

    def mark(self, price, entry_fill, qty, direction) -> float:
        sign = 1.0 if direction == LONG else -1.0
        return sign * (price - entry_fill) * self.instrument.point_value * qty

    def exit_cash_delta(self, entry_fill, exit_fill, qty, direction) -> float:
        sign = 1.0 if direction == LONG else -1.0
        realized = sign * (exit_fill - entry_fill) * self.instrument.point_value * qty
        return realized - self.instrument.commission_per_side * qty

    def risk_basis(self, entry_fill, stop, qty, direction) -> float:
        return abs(entry_fill - stop) * self.instrument.point_value * qty   # 1R($)
