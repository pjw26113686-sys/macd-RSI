"""레퍼런스 예시 2 — 볼린저 밴드터치 평균회귀(롱/숏).

진입: 봉 저가가 하단밴드 터치 → 롱(중심선 회귀 기대) / 봉 고가가 상단밴드 터치 → 숏.
청산: 중심선(mid) 목표 도달 / 손절(swing) / 시간청산. 평균회귀 베이스라인.

(docs/kim_bollinger_spec_v1.md의 이격도 다이버전스 결합은 더 정교한 변형이며,
 여기서는 프레임워크 시연을 위해 단순 밴드터치 버전을 제공한다. 모든 신호는
 t시점까지 데이터만 사용 — 미래참조 없음.)
"""
from __future__ import annotations

import pandas as pd

from src import signals_core as ind
from src.strategies.base import ExitModel, Strategy, empty_signals


class BollingerReversionStrategy(Strategy):
    name = "bollinger"

    default_params = {
        "bb_period": 20,
        "bb_std": 2.0,
        "swing_lookback_M": 10,
        "stop_buffer": 0.001,
    }

    exit_model = ExitModel(
        stop_mode="swing",
        target_mode="mid",       # 중심선 회귀 목표(exit_target 컬럼)
        signal_exit=False,
        max_hold_bars=24,        # 시간청산
    )

    signal_columns = ["bb_mid", "bb_upper", "bb_lower", "exit_target",
                      "enter_long", "enter_short"]

    def add_signals(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        ind.validate_ohlcv(df)
        out = empty_signals(df)
        mid, upper, lower = ind.bollinger(out["close"], params["bb_period"],
                                          params["bb_std"])
        out["bb_mid"] = mid
        out["bb_upper"] = upper
        out["bb_lower"] = lower
        out["exit_target"] = mid   # 중심선 회귀 목표

        out["enter_long"] = (out["low"] <= lower).fillna(False)
        out["enter_short"] = (out["high"] >= upper).fillna(False)
        return out
