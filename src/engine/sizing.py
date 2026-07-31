"""포지션 사이징 — "얼마나 살까"를 리스크 기준으로 결정 (기관급의 없는 절반).

기존 엔진은 항상 자본의 position_pct(기본 100%)를 올인했다. 기관은 **손실 위험을
일정하게** 유지하도록 크기를 조절한다. 이 모듈은 진입가·손절가·자본을 받아 수량(qty)을
돌려주는 사이저들을 제공한다. 엔진은 사이저 하나를 받아 호출할 뿐이라 규율은 불변.

계약:  sizer(cash, entry, stop, costs, position_pct, params) -> qty(float >= 0)
  - cash        현재 현금.
  - entry        슬리피지 반영 진입 체결가.
  - stop         손절가 (entry > stop 가정 — 엔진이 유효성 선검사).
  - costs        Costs (수수료율 참조).
  - position_pct 명목 상한 비율(자본 대비). 어떤 사이저도 이 명목을 넘지 않는다.
  - params       전략/설정 dict (risk_pct 등 사이저 고유 knob).

모든 사이저는 명목(entry*qty*(1+fee))이 cash*position_pct를 넘지 않도록 상한을 건다
→ 현금 음수 방지.
"""
from __future__ import annotations


def _notional_cap_qty(cash: float, entry: float, costs, position_pct: float) -> float:
    """명목 상한(자본의 position_pct)에 해당하는 최대 수량. 기존 전액 사이징과 동일식."""
    denom = entry * (1.0 + costs.fee)
    return (cash * position_pct) / denom if denom > 0 else 0.0


def fixed_fraction(cash, entry, stop, costs, position_pct, params) -> float:
    """고정 비율: 자본의 position_pct를 명목으로 진입 (엔진 기존 동작과 동일)."""
    return _notional_cap_qty(cash, entry, costs, position_pct)


def fixed_risk(cash, entry, stop, costs, position_pct, params) -> float:
    """고정 리스크: 손절까지 닿았을 때의 손실이 자본의 risk_pct가 되도록 사이징.

    qty = (cash * risk_pct) / (entry - stop).  단위당 리스크(entry-stop)가 작을수록
    크게, 클수록 작게 잡는다. 명목이 상한(position_pct)을 넘으면 상한으로 자른다.
    params["risk_pct"] (기본 0.01 = 1%).
    """
    risk_pct = float(params.get("risk_pct", 0.01))
    per_unit_risk = entry - stop
    if per_unit_risk <= 0 or risk_pct <= 0:
        return 0.0
    qty = (cash * risk_pct) / per_unit_risk
    return min(qty, _notional_cap_qty(cash, entry, costs, position_pct))


_REGISTRY = {
    "fixed_fraction": fixed_fraction,
    "fixed_risk": fixed_risk,
}


def make_sizer(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"미등록 사이저: {name!r}. 가능: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def from_params(params: dict):
    """params["sizing"]로 사이저를 고른다. 미지정이면 fixed_fraction(기존 동작)."""
    return make_sizer(params.get("sizing", "fixed_fraction"))
