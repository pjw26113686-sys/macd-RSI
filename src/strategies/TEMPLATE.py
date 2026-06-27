"""새 전략 작성 템플릿 — 이 파일을 복사해 시작하세요.

  1) cp src/strategies/TEMPLATE.py src/strategies/my_strategy.py
  2) 클래스 이름·name·진입조건·ExitModel·default_params를 채운다.
  3) src/strategies/registry.py STRATEGIES에 한 줄 등록한다.
  → pytest(미래참조 자동검증) 통과 후 python -m src.compare 표에 자동 등장.

규율(중요): 모든 신호는 t시점까지 데이터만 사용한다(.shift(+k)만). 미래참조
(.shift(-k)) 절대 금지 — tests/test_lookahead.py가 전략별로 강제한다.
지표는 반드시 src.signals_core의 단일 구현을 쓴다(전략끼리 드리프트 방지).
"""
from __future__ import annotations

import pandas as pd

from src import signals_core as ind
from src.strategies.base import ExitModel, Strategy, empty_signals


class TemplateStrategy(Strategy):
    name = "template"

    # config 미지정 시 기본값
    default_params = {
        "some_period": 20,
        "swing_lookback_M": 10,
        "stop_buffer": 0.001,
        "reward_ratio": 2.0,
    }

    # 청산 정책(자세한 옵션은 base.ExitModel 참고)
    exit_model = ExitModel(
        stop_mode="swing",       # 직전 M봉 극단 기준 손절
        target_mode="rr",        # 손익비 reward_ratio
        signal_exit=False,
        max_hold_bars=None,
    )

    # 미래참조 테스트가 검사할 컬럼(만든 지표/신호를 모두 넣을수록 안전)
    signal_columns = ["enter_long", "enter_short"]

    def add_signals(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        ind.validate_ohlcv(df)
        out = empty_signals(df)

        # --- 여기에 지표 계산 (signals_core 사용) ---
        # 예: out["ma"] = ind.sma(out["close"], params["some_period"])

        # --- 여기에 진입조건 (bool) ---
        # out["enter_long"]  = ...   # 무포지션일 때 t+1 시가 롱 진입 후보
        # out["enter_short"] = ...   # 숏 진입 후보 (없으면 False 유지)

        return out
