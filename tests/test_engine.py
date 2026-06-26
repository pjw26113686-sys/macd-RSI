"""엔진 실행의미 검증: 체결 타이밍, 손절 우선, 분할익절+본전스탑.

신호 컬럼을 직접 주입(run_on_signals)해 지표 계산과 분리한다.
"""
import pytest

from src.engine import backtest
from src.engine.position import Costs
from tests.helpers import empty_signal_frame


SMALL = {
    "macd_fast": 3, "macd_slow": 3, "macd_signal": 2,
    "rsi_period": 2, "rsi_entry_low": 50, "rsi_entry_high": 70,
    "rsi_support_band": 5, "hma_period": 4,
    "vol_ma_period": 2, "vol_ma_mult": 1.0,
    "delay_window": 5, "min_body_ratio": 0.5,
    "swing_lookback_M": 3, "stop_buffer": 0.001, "reward_ratio": 2.0,
}
SLIP = 0.0005
COSTS = Costs(fee=0.0, slippage=SLIP, sell_tax=0.0)


def _set_bar(df, i, o, h, l, c):
    df.loc[i, ["open", "high", "low", "close"]] = [o, h, l, c]


def test_entry_fills_next_bar_open():
    """종가 t에서 entry_base → 진입은 t+1 봉 시가에 체결되어야 한다."""
    df = empty_signal_frame(20)
    df["low"] = 90.0          # swing_low 유효
    df["high"] = 101.0
    df.loc[8, "enter_long"] = True   # 신호 봉 = 8
    _set_bar(df, 9, o=100.0, h=101.0, l=99.0, c=100.0)  # 체결 봉 = 9

    res = backtest.run_on_signals(df, SMALL, COSTS, initial_capital=10000.0)
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.entry_index == 9, "진입이 신호 다음 봉(t+1)에 체결되어야 함"
    assert tr.entry_price == pytest.approx(100.0 * (1 + SLIP))


def test_stop_has_priority_over_target_same_bar():
    """한 봉에서 손절·목표가 동시 도달 시 손절 우선(§0-4)."""
    df = empty_signal_frame(20)
    df["low"] = 95.0
    df["high"] = 101.0
    df.loc[8, "enter_long"] = True
    _set_bar(df, 9, 100.0, 101.0, 99.0, 100.0)   # 진입 체결
    # 12봉: 손절(저가 80)과 목표가(고가 130) 동시 도달
    _set_bar(df, 12, 100.0, 130.0, 80.0, 100.0)

    res = backtest.run_on_signals(df, SMALL, COSTS, initial_capital=10000.0)
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.last_reason == "STOP"
    assert len(tr.exits) == 1  # 분할익절 없이 손절 전량


def test_partial_tp_then_breakeven_stop():
    """목표가 → 50% 익절 + 본전스탑 이동 → 잔량 본전스탑 청산."""
    df = empty_signal_frame(20)
    df["low"] = 98.0
    df["high"] = 101.0
    df.loc[8, "enter_long"] = True
    _set_bar(df, 9, 100.0, 101.0, 99.0, 100.0)   # 진입 (fill≈100.05)
    # entry_fill≈100.05, stop≈swing_low(98)*0.999, R≈0.0205, target≈100.05*(1+0.041)≈104.2
    _set_bar(df, 11, 102.0, 130.0, 101.0, 105.0)  # 목표가 도달 → 50% 익절
    _set_bar(df, 12, 101.0, 102.0, 101.0, 101.5)  # 본전 위 유지(잔량 보유)
    _set_bar(df, 13, 100.0, 100.5, 95.0, 96.0)    # 본전(≈100.05) 하향 → 잔량 청산

    res = backtest.run_on_signals(df, SMALL, COSTS, initial_capital=10000.0)
    assert len(res.trades) == 1
    tr = res.trades[0]
    reasons = [f.reason for f in tr.exits]
    assert reasons[0] == "TP1_50%"
    assert reasons[-1] == "STOP"         # 본전스탑 = STOP 사유
    assert tr.exit_index == 13
    assert tr.qty == pytest.approx(sum(f.qty for f in tr.exits))  # 전량 청산됨


def test_no_entry_when_already_in_position():
    """보유 중에는 신규 진입하지 않는다(§3.4)."""
    df = empty_signal_frame(25)
    df["low"] = 90.0
    df["high"] = 101.0
    df.loc[8, "enter_long"] = True
    df.loc[12, "enter_long"] = True   # 보유 중 신호 → 무시되어야
    _set_bar(df, 9, 100.0, 101.0, 99.0, 100.0)

    res = backtest.run_on_signals(df, SMALL, COSTS, initial_capital=10000.0)
    # 12봉 신호는 보유중이라 무시 → EOD까지 단일 포지션
    assert len(res.trades) == 1
