"""사용자 실데이터 인제스트 — CSV/Parquet/Feather를 표준 OHLCV로.

이 환경은 외부 시세가 차단돼 자동 다운로드가 안 된다. 대신 사용자가 어디서든 받은
파일(거래소 CSV, yfinance 내보내기, parquet 캐시)을 떨어뜨리면 스키마를 표준화·검증해
엔진에 바로 물린다. 다양한 관례를 흡수한다:
  - 시간열 이름: time / date / datetime / timestamp / open_time (대소문자 무관)
  - 에폭 정수(ms 또는 s)도 자동 판별해 UTC 변환
  - 컬럼명 대소문자 무관(Open/OPEN/open), volume 없으면 0으로 채움
검증은 기존 cache.validate_ohlcv(중복제거·UTC통일·정렬·결측처리)로 통일한다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.cache import validate_ohlcv

_TIME_CANDIDATES = ["time", "date", "datetime", "timestamp", "open_time"]
_OHLCV = ["open", "high", "low", "close", "volume"]


def _to_datetime_index(series: pd.Series) -> pd.DatetimeIndex:
    """시간열을 UTC tz-aware DatetimeIndex로. 에폭 정수(ms/s)도 판별."""
    # pandas 확장 dtype(StringDtype 등)도 안전하게 처리(np.issubdtype 회피).
    if pd.api.types.is_numeric_dtype(series):
        # 큰 값이면 밀리초, 아니면 초 단위 에폭으로 판별.
        unit = "ms" if float(series.max()) > 1e12 else "s"
        return pd.DatetimeIndex(pd.to_datetime(series, unit=unit, utc=True))
    return pd.DatetimeIndex(pd.to_datetime(series, utc=True))


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """임의 스키마의 표를 time 인덱스 + open/high/low/close/volume로 표준화."""
    df = df.copy()
    lower = {str(c).lower(): c for c in df.columns}

    # 1) 시간 인덱스 확정.
    time_col = next((lower[c] for c in _TIME_CANDIDATES if c in lower), None)
    if time_col is not None:
        df.index = _to_datetime_index(df[time_col])
        df = df.drop(columns=[time_col])
    elif isinstance(df.index, pd.DatetimeIndex):
        pass  # 이미 시간 인덱스(예: parquet 캐시)
    else:
        raise ValueError(
            f"시간열을 찾을 수 없습니다. {_TIME_CANDIDATES} 중 하나가 필요합니다."
        )

    # 2) OHLCV 컬럼명 표준화(대소문자 무관). {원본컬럼명: 표준소문자}.
    lower = {str(c).lower(): c for c in df.columns}
    rename = {lower[k]: k for k in _OHLCV if k in lower}
    df = df.rename(columns=rename)

    missing = [c for c in ["open", "high", "low", "close"] if c not in df.columns]
    if missing:
        raise ValueError(f"필수 OHLC 컬럼이 없습니다: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df.index.name = "time"
    return df[_OHLCV]


def load_ohlcv_file(path: str | Path, market: str = "crypto") -> pd.DataFrame:
    """단일 파일(.csv/.tsv/.parquet/.feather)을 표준·검증된 OHLCV로 로드."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"데이터 파일 없음: {p}")
    ext = p.suffix.lower()
    if ext in (".parquet", ".pq"):
        raw = pd.read_parquet(p)
    elif ext in (".feather", ".ft"):
        raw = pd.read_feather(p)
    elif ext == ".tsv":
        raw = pd.read_csv(p, sep="\t")
    elif ext in (".csv", ".txt"):
        raw = pd.read_csv(p)
    else:
        raise ValueError(f"지원하지 않는 형식: {ext} (csv/tsv/parquet/feather)")
    return validate_ohlcv(normalize_ohlcv(raw), market)


def load_universe(paths, market: str = "crypto") -> dict[str, pd.DataFrame]:
    """여러 파일 → {심볼(파일명): OHLCV}. paths는 파일 목록 또는 디렉터리."""
    p0 = Path(paths[0]) if isinstance(paths, (list, tuple)) and paths else Path(paths)
    if not isinstance(paths, (list, tuple)) and p0.is_dir():
        files = sorted([q for q in p0.iterdir()
                        if q.suffix.lower() in (".csv", ".tsv", ".parquet", ".pq", ".feather", ".ft")])
    else:
        files = [Path(x) for x in (paths if isinstance(paths, (list, tuple)) else [paths])]
    if not files:
        raise ValueError(f"로드할 데이터 파일이 없습니다: {paths}")
    return {f.stem: load_ohlcv_file(f, market) for f in files}
