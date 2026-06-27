"""선물 손익 모델 — 틱·달러 산수 + 사이징 검증."""
import pytest

from src.engine.pnl import LONG, SHORT, FuturesPnL, SizingConfig
from src.instruments import get_instrument

ES = get_instrument("ES")          # tick 0.25, $12.5/tick → point_value $50, comm $4
PNL = FuturesPnL(ES)


def test_point_value():
    assert ES.point_value == pytest.approx(50.0)


def test_long_pnl_tick_math():
    # 진입 5000, 청산 5001 (=+1.00pt=+4틱), 2계약
    entry, exit_ = 5000.0, 5001.0
    realized = PNL.exit_cash_delta(entry, exit_, qty=2, direction=LONG)
    # (5001-5000)*50*2 - 수수료(편도 4*2=8) = 100 - 8 = 92
    assert realized == pytest.approx(100.0 - 8.0)
    entry_cost = PNL.entry_cash_delta(entry, 2, LONG)
    assert entry_cost == pytest.approx(-8.0)   # 진입 수수료만
    assert realized + entry_cost == pytest.approx(100.0 - 16.0)  # 왕복 순손익


def test_short_pnl_tick_math():
    # 숏 진입 5000 → 청산 4998 (=-2pt 하락=숏 이익), 1계약
    realized = PNL.exit_cash_delta(5000.0, 4998.0, qty=1, direction=SHORT)
    assert realized == pytest.approx(2.0 * 50.0 - 4.0)   # +$100 - 수수료 $4


def test_slippage_direction():
    slip = ES.slippage_ticks * ES.tick_size
    assert PNL.entry_fill(5000.0, LONG) == pytest.approx(5000.0 + slip)
    assert PNL.exit_fill(5000.0, LONG) == pytest.approx(5000.0 - slip)
    assert PNL.entry_fill(5000.0, SHORT) == pytest.approx(5000.0 - slip)
    assert PNL.exit_fill(5000.0, SHORT) == pytest.approx(5000.0 + slip)


def test_mark_unrealized():
    # 롱 2계약, +3pt 평가익 = 3*50*2 = $300
    assert PNL.mark(5003.0, 5000.0, 2, LONG) == pytest.approx(300.0)
    # 숏은 가격 하락이 이익
    assert PNL.mark(4997.0, 5000.0, 2, SHORT) == pytest.approx(300.0)


def test_fixed_sizing():
    cfg = SizingConfig(mode="fixed", contracts=3)
    assert PNL.size(50000, 5000, 4995, LONG, cfg) == 3.0


def test_risk_sizing_caps_at_max():
    # 손절폭 5pt × $50 = $250/계약. 리스크 $600 → 2계약, max 10 이하
    cfg = SizingConfig(mode="risk", risk_dollars=600, max_contracts=10)
    assert PNL.size(50000, 5000.0, 4995.0, LONG, cfg) == 2.0
    # 작은 리스크 → 0계약(진입 스킵)
    cfg2 = SizingConfig(mode="risk", risk_dollars=100, max_contracts=10)
    assert PNL.size(50000, 5000.0, 4995.0, LONG, cfg2) == 0.0
