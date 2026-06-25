"""암호화폐 1h OHLCV 수집 — ccxt (거래소 직접).

freqtrade는 암호화폐 라이브 단계로 연기. 1차 백테스트 데이터는 ccxt로 직접 수집한다.
"""
from __future__ import annotations

import time

import pandas as pd

from src.data import cache


def fetch_ohlcv(
    exchange: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since: str = "2020-01-01T00:00:00Z",
    use_cache: bool = True,
) -> pd.DataFrame:
    if use_cache:
        cached = cache.load("crypto", symbol)
        if cached is not None:
            return cached

    import ccxt  # 지연 import (테스트가 ccxt 없이도 동작하도록)

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    since_ms = ex.parse8601(since)
    limit = 1000
    all_rows = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        since_ms = batch[-1][0] + 1
        if len(batch) < limit:
            break
        time.sleep(ex.rateLimit / 1000.0)

    df = pd.DataFrame(all_rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("time")
    df = cache.validate_ohlcv(df, "crypto")
    if use_cache:
        cache.save(df, "crypto", symbol)
    return df
