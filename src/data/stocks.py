"""미국주식 1h OHLCV 수집 — yfinance.

무료·호출제한 없음, 60분봉 최대 730일 제공. auto_adjust로 분할/배당 조정.
"""
from __future__ import annotations

import pandas as pd

from src.data import cache


def fetch_ohlcv(
    symbol: str = "AAPL",
    interval: str = "60m",
    period: str = "730d",
    auto_adjust: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    if use_cache:
        cached = cache.load("stock", symbol)
        if cached is not None:
            return cached

    import yfinance as yf  # 지연 import

    raw = yf.download(
        symbol, interval=interval, period=period,
        auto_adjust=auto_adjust, prepost=False, progress=False,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance에서 {symbol} 데이터를 받지 못했습니다.")

    # yfinance는 멀티인덱스 컬럼을 줄 수 있음 → 평탄화
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    raw.index.name = "time"

    df = cache.validate_ohlcv(raw, "stock")
    if use_cache:
        cache.save(df, "stock", symbol)
    return df
