"""선물 OHLCV: 사용자 파일 로더 + 합성(random-walk) 생성기.

실데이터(CME)는 무료 소스가 드물고 샌드박스 네트워크가 막혀 있으므로:
  - load_futures(path): Databento/Sierra Chart/IQFeed 등에서 받은 CSV/parquet 주입.
  - synthetic_futures(...): 실데이터 확보 전 프레임워크·프랍평가기 엔드투엔드 검증용
    random-walk 데이터(틱 그리드 정렬). **성과 판정용이 아님**(sanity 전용).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import cache
from src.instruments import Instrument, get_instrument

# 합성 봉수 환산용 (분)
_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "1d": 1440}


def load_futures(path: str | Path, timeframe: str = "5m") -> pd.DataFrame:
    """CSV/parquet(OHLCV) 로드 → UTC 통일·정합성 검증. time 컬럼 또는 인덱스 허용."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = next((c for c in df.columns if c.lower() in ("time", "date", "datetime", "timestamp")), None)
        if time_col is None:
            raise ValueError("시간 컬럼(time/date/datetime/timestamp)을 찾을 수 없습니다.")
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.set_index(time_col)
    df.columns = [c.lower() for c in df.columns]
    return cache.validate_ohlcv(df, "futures")


def synthetic_futures(
    symbol: str = "NQ",
    n: int = 5000,
    timeframe: str = "5m",
    seed: int = 0,
    drift: float = 0.0,
    vol_ticks: float = 8.0,
    start_price: float | None = None,
    instrument: Instrument | None = None,
) -> pd.DataFrame:
    """random-walk 선물 OHLCV(틱 그리드 정렬). 검증 전용 합성 데이터.

    vol_ticks: 봉당 변동 표준편차(틱). drift: 봉당 평균 드리프트(틱).
    """
    inst = instrument or get_instrument(symbol)
    rng = np.random.default_rng(seed)
    tick = inst.tick_size
    if start_price is None:
        start_price = {"NQ": 18000.0, "ES": 5000.0, "CL": 75.0, "GC": 2000.0}.get(
            symbol.upper().lstrip("M"), 1000.0)

    steps = rng.normal(drift, vol_ticks, n) * tick
    close = start_price + np.cumsum(steps)
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0, vol_ticks * 0.6, n)) * tick
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    vol = rng.integers(500, 5000, n).astype(float)

    freq_min = _TF_MINUTES.get(timeframe, 5)
    idx = pd.date_range("2023-01-02 00:00", periods=n, freq=f"{freq_min}min",
                        tz="UTC", name="time")
    df = pd.DataFrame({
        "open": np.round(open_ / tick) * tick,
        "high": np.round(high / tick) * tick,
        "low": np.round(low / tick) * tick,
        "close": np.round(close / tick) * tick,
        "volume": vol,
    }, index=idx)
    return df


def get_futures_data(symbol: str, cfg: dict, use_cache: bool = True) -> pd.DataFrame:
    """config의 data.futures 설정에 따라 합성 또는 파일/캐시에서 로드."""
    fcfg = cfg.get("data", {}).get("futures", {})
    timeframe = fcfg.get("timeframe", "5m")
    if fcfg.get("synthetic", True):
        return synthetic_futures(
            symbol=symbol, n=fcfg.get("n_bars", 5000), timeframe=timeframe,
            seed=fcfg.get("seed", 0), drift=fcfg.get("drift", 0.0),
            vol_ticks=fcfg.get("vol_ticks", 8.0),
        )
    if use_cache:
        cached = cache.load("futures", symbol, timeframe)
        if cached is not None:
            return cached
    path = fcfg.get("path_template", "data/futures/{symbol}_{tf}.csv").format(
        symbol=symbol.replace("/", "_"), tf=timeframe)
    df = load_futures(path, timeframe)
    if use_cache:
        cache.save(df, "futures", symbol, timeframe)
    return df
