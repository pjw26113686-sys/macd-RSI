"""전략 작성 계약(contract) — 새 전략은 이 인터페이스만 구현하면 된다.

전략은 **신호와 청산정책만** 책임진다. 체결 타이밍·포지션 사이징·손익계산·
프랍룰 평가·비교는 전부 프레임워크(engine/prop/compare)가 처리한다.

새 전략 만들기:
  1. `TEMPLATE.py`를 복사한다.
  2. `Strategy`를 상속해 `name`, `add_signals`, `signal_columns`,
     `exit_model`, `default_params`를 채운다.
  3. `registry.py`의 STRATEGIES dict에 한 줄 등록한다.
  → 그러면 자동으로 백테스트·프랍평가·비교 표에 들어온다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class ExitModel:
    """전략의 청산 "정책" 선언. 실제 실행은 엔진(_manage_open_bar)이 한다.

    stop_mode:
      "swing" : 직전 swing_lookback_M봉 극단 ± stop_buffer (롱=저점/숏=고점)
      "atr"   : 진입가 ± atr_mult × ATR
      "fixed" : 진입가 ± stop_ticks (instrument tick 기준; 선물 전용)
    target_mode:
      "rr"    : 손익비 reward_ratio 배 (entry ± reward_ratio × 손절폭)
      "mid"   : 신호 df의 'exit_target' 컬럼값(예: 볼린저 중심선)으로 청산
      "none"  : 가격목표 없음(신호/시간 청산만)
    """
    stop_mode: str = "swing"
    target_mode: str = "rr"
    partial_tp_pct: float | None = None     # 1차 익절 비율(예: 0.5). None=전량청산
    breakeven_after_tp: bool = False         # 1차 익절 후 손절을 본전으로 이동
    max_hold_bars: int | None = None         # 시간청산(None=무제한)
    signal_exit: bool = False                # exit_long/exit_short 신호로 청산


class Strategy:
    """전략 베이스. 하위 클래스가 아래를 구현/오버라이드한다."""

    #: 전략 식별자(리포트·CLI에서 사용)
    name: str = "base"

    #: 미래참조 테스트(test_lookahead)가 t절단 동일성을 검사할 컬럼들.
    #: 최소 ['enter_long', 'enter_short'] + 전략이 만든 지표/신호 컬럼.
    signal_columns: list[str] = ["enter_long", "enter_short"]

    #: config에 값이 없을 때 쓰는 기본 파라미터.
    default_params: dict = {}

    #: 청산 정책.
    exit_model: ExitModel = ExitModel()

    def add_signals(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """OHLCV df에 지표 + 신호 컬럼을 추가해 반환(원본 비변경).

        필수 산출 컬럼:
          enter_long, enter_short  (bool) — 무포지션일 때 t+1 시가 진입 후보
        선택 컬럼:
          exit_long, exit_short    (bool) — signal_exit=True일 때 보유청산 신호
          exit_target              (float)— target_mode='mid'일 때 청산 목표가

        규율: 모든 컬럼은 t시점까지 데이터만 사용(.shift(+k)만, 미래참조 금지).
        지표는 src.signals_core의 단일 구현을 사용한다.
        """
        raise NotImplementedError

    def params(self, overrides: dict | None = None) -> dict:
        p = dict(self.default_params)
        if overrides:
            p.update({k: v for k, v in overrides.items() if v is not None})
        return p


def empty_signals(df: pd.DataFrame) -> pd.DataFrame:
    """신호 컬럼을 False/NaN으로 초기화한 복사본(전략 add_signals 시작점)."""
    out = df.copy()
    out["enter_long"] = False
    out["enter_short"] = False
    out["exit_long"] = False
    out["exit_short"] = False
    return out
