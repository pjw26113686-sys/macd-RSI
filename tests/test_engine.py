"""엔진 실행의미 검증(선물): 체결 타이밍, 숏, 손절 우선, 시간/목표 청산."""
import pytest

from src.engine import backtest as bt
from src.engine.pnl import LONG, SHORT, FuturesPnL, SizingConfig
from src.instruments import get_instrument
from src.strategies.base import ExitModel
from tests.helpers import set_bar, signal_frame

ES = get_instrument("ES")          # tick 0.25, $12.5/tick, slip 1 tick
PNL = FuturesPnL(ES)
SIZE = SizingConfig(mode="fixed", contracts=2)
PARAMS = {"swing_lookback_M": 3, "stop_buffer": 0.001, "reward_ratio": 2.0}
SLIP = ES.slippage_ticks * ES.tick_size   # 0.25


def _run(df, exit_model, sizing=SIZE):
    return bt.run_on_signals(df, PARAMS, exit_model, PNL, sizing,
                             initial_capital=50000.0, instrument=ES, warmup=1)


def test_long_entry_fills_next_bar_open():
    df = signal_frame(20)
    df["low"] = 90.0
    df["high"] = 110.0
    df.loc[8, "enter_long"] = True       # 신호 봉 8
    set_bar(df, 9, o=100.0, h=110.0, l=99.0, c=100.0)  # 체결 봉 9

    res = _run(df, ExitModel(stop_mode="swing", target_mode="rr"))
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.entry_index == 9 and tr.direction == LONG
    assert tr.entry_price == pytest.approx(100.0 + SLIP)   # 롱 진입 불리하게 위로


def test_short_entry_and_stop_priority():
    df = signal_frame(20)
    df["low"] = 90.0
    df["high"] = 110.0
    df.loc[8, "enter_short"] = True
    set_bar(df, 9, o=100.0, h=101.0, l=99.0, c=100.0)     # 숏 진입
    # 12봉: 손절(고가 130, 위)과 목표(저가 70, 아래) 동시 도달 → 손절 우선
    set_bar(df, 12, o=100.0, h=130.0, l=70.0, c=100.0)

    res = _run(df, ExitModel(stop_mode="swing", target_mode="rr"))
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.direction == SHORT
    assert tr.last_reason == "STOP"
    assert len(tr.exits) == 1           # 손절 전량(분할 없음)


def test_time_exit():
    df = signal_frame(20)
    df["low"] = 95.0
    df["high"] = 105.0
    df.loc[8, "enter_long"] = True
    set_bar(df, 9, 100.0, 101.0, 99.5, 100.0)   # 진입
    res = _run(df, ExitModel(stop_mode="swing", target_mode="none", max_hold_bars=3))
    tr = res.trades[0]
    # 진입 9봉 + 3봉 보유 → 12봉 종가확정, 13봉 시가 청산
    assert tr.last_reason == "TIME"
    assert tr.exit_index == 13


def test_target_mid_exit():
    df = signal_frame(20)
    df["low"] = 95.0
    df["high"] = 102.0                   # 평소엔 목표(103) 미도달
    df["exit_target"] = 103.0           # 중심선 목표
    df.loc[8, "enter_long"] = True
    set_bar(df, 9, 100.0, 101.0, 99.5, 100.0)   # 진입(fill≈100.25)
    set_bar(df, 11, 101.0, 104.0, 101.0, 103.5) # 목표 103 도달
    res = _run(df, ExitModel(stop_mode="swing", target_mode="mid"))
    tr = res.trades[0]
    assert tr.last_reason == "TARGET"
    assert tr.exit_index == 11
    assert tr.pnl > 0                    # 100.25 → 103 롱 이익


def test_no_entry_during_warmup_or_last_bar():
    df = signal_frame(20)
    df.loc[0, "enter_long"] = True       # 워밍업(warmup=1) 경계
    df.loc[19, "enter_long"] = True      # 마지막 봉(다음봉 없음)
    res = _run(df, ExitModel(stop_mode="swing", target_mode="rr"))
    assert len(res.trades) == 0
