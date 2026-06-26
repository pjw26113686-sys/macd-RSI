"""자체 엔진의 ccxt parquet 캐시 → freqtrade feather 포맷 변환.

freqtrade `download-data`는 거래소를 직접 호출하지만(제한 네트워크에선 차단),
이미 받아둔 parquet을 freqtrade 포맷으로 변환하면 **동일 봉**으로 백테스트할 수
있어 자체 엔진과 진정한 apples-to-apples 비교가 된다.

freqtrade feather 스키마: columns = [date(datetime, UTC), open, high, low, close, volume]
파일명 규칙: user_data/data/<exchange>/<BASE>_<QUOTE>-<timeframe>.feather

사용:
    python scripts/to_freqtrade_data.py            # 기본: BTC/USDT, binance, 1h
    python scripts/to_freqtrade_data.py --symbol BTC/USDT --exchange binance
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import cache

REPO_ROOT = Path(__file__).resolve().parents[1]


def convert(symbol: str, exchange: str, timeframe: str = "1h") -> Path:
    df = cache.load("crypto", symbol)
    if df is None:
        raise SystemExit(
            f"캐시 없음: {cache.cache_path('crypto', symbol)}\n"
            f"먼저 외부망 환경에서 `python -m src.run --market crypto`로 데이터를 받으세요."
        )
    # cache는 UTC tz-aware DatetimeIndex(name='time'). freqtrade는 'date' 컬럼 요구.
    out = df.reset_index().rename(columns={"time": "date"})
    out = out[["date", "open", "high", "low", "close", "volume"]]

    pair_name = symbol.replace("/", "_")
    dest = REPO_ROOT / "freqtrade" / "user_data" / "data" / exchange / f"{pair_name}-{timeframe}.feather"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_feather(dest)
    print(f"변환 완료: {len(out)}봉 → {dest}")
    return dest


def main():
    ap = argparse.ArgumentParser(description="parquet 캐시 → freqtrade feather 변환")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--timeframe", default="1h")
    args = ap.parse_args()
    convert(args.symbol, args.exchange, args.timeframe)


if __name__ == "__main__":
    main()
