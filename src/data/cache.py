"""데이터 파이프라인 공통: parquet 캐시 + 정합성 검증 + UTC 통일.

raw 다운로드 → 로컬 캐시(parquet) → 정합성 검증 → signals_core 입력
(implementation_architecture §3)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def cache_path(market: str, symbol: str, timeframe: str = "1h") -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return DATA_ROOT / market / f"{safe}_{timeframe}.parquet"


def save(df: pd.DataFrame, market: str, symbol: str, timeframe: str = "1h") -> Path:
    path = cache_path(market, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def load(market: str, symbol: str, timeframe: str = "1h") -> pd.DataFrame | None:
    path = cache_path(market, symbol, timeframe)
    if path.exists():
        return pd.read_parquet(path)
    return None


def validate_ohlcv(df: pd.DataFrame, market: str = "futures") -> pd.DataFrame:
    """중복 타임스탬프 제거, UTC 통일, 정렬, 0거래량/결측 처리.

    - 인덱스는 UTC tz-aware DatetimeIndex (name='time')로 통일.
    - 주식: 장 마감~개장 갭은 봉을 채우지 않고 그대로 둔다(세션 경계). 1h 연속봉으로
      오인하지 않도록 reindex(연속 1h)는 적용하지 않는다.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("인덱스가 DatetimeIndex가 아닙니다.")

    # UTC 통일
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "time"

    # 중복 제거 + 정렬
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # 필수 컬럼 정리
    cols = ["open", "high", "low", "close", "volume"]
    out = out[[c for c in cols if c in out.columns]].astype(float)

    # 가격 결측 행 제거 (거래정지/휴장 등)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0.0)
    return out
