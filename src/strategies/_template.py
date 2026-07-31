"""새 전략 템플릿 — 이 파일을 복사해 시작하세요.

절차:
  1) 이 파일을 src/strategies/<your_name>.py 로 복사(또는 아무 곳에나 두고 --module로 검증).
  2) generate_signals에서 진입/청산 불리언 컬럼을 계산 — **미래참조 0** 을 지키세요:
       - t시점 컬럼은 t까지의 데이터만 사용. 오직 .shift(+k)(과거참조)만 허용.
       - 롤링 지표로 "현재봉 제외 과거"를 보려면 rolling(...).shift(1).
       - 진입은 종가 확정 기준(엔진이 다음 봉 시가에 체결).
  3) 검증:  python -m src.strategy_check --module src/strategies/<your_name>.py --synthetic
     합격(무결성 통과)이 나오면 __init__.py의 import 목록에 추가해 정식 등록.

손절/목표는 엔진 공통(swing-low stop + reward_ratio)이 관리하므로 default_params에
swing_lookback_M / stop_buffer / reward_ratio 를 포함하세요.
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base import StrategySpec  # register는 정식 등록 시에만

DEFAULT_PARAMS = {
    # 전략 고유 파라미터 (예시)
    "fast": 10,
    "slow": 30,
    # 엔진 공통 손절/목표 (필수)
    "swing_lookback_M": 10,
    "stop_buffer": 0.001,
    "reward_ratio": 2.0,
}

# 스윕(오버피팅 검증)용 그리드. CSCV 순위매김을 위해 총 설정 4개 이상 권장.
PARAM_GRID = {
    "fast": [5, 10, 20],
    "slow": [30, 50],
}


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """OHLCV(time 인덱스) → enter_long/exit_all/exit_half 불리언 컬럼 추가해 반환."""
    out = df.copy()
    fast = out["close"].rolling(params["fast"]).mean()
    slow = out["close"].rolling(params["slow"]).mean()

    # 예시: 골든/데드 크로스 (미래참조 0 — shift(1)은 과거참조).
    out["enter_long"] = (fast.shift(1) <= slow.shift(1)) & (fast > slow)
    out["exit_all"] = (fast.shift(1) >= slow.shift(1)) & (fast < slow)
    out["exit_half"] = out["exit_all"]  # 잔량 청산을 따로 두려면 여기 정의

    for col in ["enter_long", "exit_all", "exit_half"]:
        out[col] = out[col].fillna(False)
    return out


def warmup_bars(params: dict) -> int:
    """지표 안정화 전 진입 금지 봉수."""
    return max(params["fast"], params["slow"]) + 1


# 하네스가 찾는 진입점. 검증만 할 땐 register 없이 SPEC만 있어도 된다.
SPEC = StrategySpec(
    name="template_ma_cross",
    category="template",
    description="이동평균 크로스(예시). 복사해서 자기 전략으로 교체하세요.",
    default_params=DEFAULT_PARAMS,
    param_grid=PARAM_GRID,
    generate_signals=generate_signals,
    warmup_bars=warmup_bars,
    exit_col="exit_all",
    exit_half_col="exit_half",
)
