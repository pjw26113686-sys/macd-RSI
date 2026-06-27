"""선물 계약 명세 (CME 마이크로/미니) + 프리셋.

백테스트 손익은 % 수익률이 아니라 **틱 × 계약수 × 달러**로 계산된다.
핵심 값:
  - tick_size   : 최소 가격 변동폭 (예: ES 0.25)
  - tick_value  : 1틱 변동당 1계약 달러가치 (예: ES $12.50)
  - point_value : 1.0 포인트 변동당 달러가치 = tick_value / tick_size (ES $50)
  - commission_per_side : 편도 1계약 수수료(달러, 거래소+청산+중개 합산 가정)
  - slippage_ticks      : 편도 슬리피지(틱). 진입/청산 체결가에 가감.

> 주의: 아래 수수료/슬리피지는 **보수적 출발값**이다. 실제 프랍사·중개 조건과
> 대조해 조정할 것. 계약명세(tick/value)는 CME 표준이지만 변경될 수 있으니
> 실거래 전 확인 필요.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: float
    tick_value: float           # 1틱당 1계약 달러가치
    commission_per_side: float  # 편도 1계약 수수료($)
    slippage_ticks: float = 1.0 # 편도 슬리피지(틱)
    currency: str = "USD"
    description: str = ""

    @property
    def point_value(self) -> float:
        """1.0 포인트 변동당 1계약 달러가치."""
        return self.tick_value / self.tick_size

    def round_to_tick(self, price: float) -> float:
        """가격을 틱 그리드에 정렬(합성 데이터 생성/체결가 보정용)."""
        return round(price / self.tick_size) * self.tick_size


# --------------------------------------------------------------------------- #
# 프리셋 (CME 표준 명세 · 보수적 비용 출발값)
# --------------------------------------------------------------------------- #
INSTRUMENTS: dict[str, Instrument] = {
    # 미니 (E-mini)
    "ES": Instrument("ES", 0.25, 12.50, 4.0, 1.0, description="E-mini S&P 500"),
    "NQ": Instrument("NQ", 0.25, 5.00, 4.0, 1.0, description="E-mini Nasdaq-100"),
    "YM": Instrument("YM", 1.0, 5.00, 4.0, 1.0, description="E-mini Dow"),
    "RTY": Instrument("RTY", 0.10, 5.00, 4.0, 1.0, description="E-mini Russell 2000"),
    "CL": Instrument("CL", 0.01, 10.00, 4.5, 1.0, description="Crude Oil"),
    "GC": Instrument("GC", 0.10, 10.00, 4.5, 1.0, description="Gold"),
    # 마이크로 (수수료는 미니와 유사, 계약가치 1/10)
    "MES": Instrument("MES", 0.25, 1.25, 1.0, 1.0, description="Micro E-mini S&P 500"),
    "MNQ": Instrument("MNQ", 0.25, 0.50, 1.0, 1.0, description="Micro E-mini Nasdaq-100"),
    "MYM": Instrument("MYM", 1.0, 0.50, 1.0, 1.0, description="Micro E-mini Dow"),
    "M2K": Instrument("M2K", 0.10, 0.50, 1.0, 1.0, description="Micro E-mini Russell 2000"),
    "MCL": Instrument("MCL", 0.01, 1.00, 1.2, 1.0, description="Micro Crude Oil"),
    "MGC": Instrument("MGC", 0.10, 1.00, 1.2, 1.0, description="Micro Gold"),
}


def get_instrument(symbol: str) -> Instrument:
    key = symbol.upper()
    if key not in INSTRUMENTS:
        raise KeyError(
            f"알 수 없는 선물 심볼: {symbol}. 사용 가능: {sorted(INSTRUMENTS)}"
        )
    return INSTRUMENTS[key]
