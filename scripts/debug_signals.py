"""BollingerOpenBreakout 진입 조건 깔때기(funnel) 분석.

0 트레이드의 원인을 찾기 위해, 전략의 populate_indicators/entry 로직을 그대로
재현해 각 조건이 몇 개 캔들을 통과하는지 단계별로 집계한다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import merge_informative_pair

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "freqtrade" / "user_data" / "data" / "binance" / "futures"

BB_PERIOD, BB_STD = 20, 2.0
VOL_LOOKBACK, VOL_MULT = 20, 1.5
INF_TF = "1d"


def load(pair: str, tf: str) -> pd.DataFrame:
    safe = pair.replace("/", "_").replace(":", "_")
    df = pd.read_feather(DATA / f"{safe}-{tf}-futures.feather")
    return df


def analyze(pair: str):
    df = load(pair, "5m")
    inf = load(pair, "1d")

    bb = ta.BBANDS(inf, timeperiod=BB_PERIOD, nbdevup=BB_STD, nbdevdn=BB_STD)
    inf["bb_upper"], inf["bb_mid"], inf["bb_lower"] = (
        bb["upperband"], bb["middleband"], bb["lowerband"]
    )
    prev_high, prev_low, prev_close = inf["high"].shift(1), inf["low"].shift(1), inf["close"].shift(1)
    prev_upper, prev_mid, prev_lower = (
        inf["bb_upper"].shift(1), inf["bb_mid"].shift(1), inf["bb_lower"].shift(1)
    )
    inf["dir_short"] = ((prev_high >= prev_upper) & (prev_close < prev_mid)).astype(int)
    inf["dir_long"] = ((prev_low <= prev_lower) & (prev_close > prev_mid)).astype(int)
    inf["day_open"] = inf["open"]
    inf["day_id"] = inf["date"].astype("int64")

    print(f"\n=== {pair} ===")
    print(f"일봉 {len(inf)}개 | dir_long={int(inf['dir_long'].sum())} "
          f"dir_short={int(inf['dir_short'].sum())}")

    keep = ["date", "dir_short", "dir_long", "day_open", "day_id",
            "bb_upper", "bb_mid", "bb_lower"]
    d = merge_informative_pair(df, inf[keep], "5m", INF_TF, ffill=True)
    d["vol_sma"] = d["volume"].rolling(VOL_LOOKBACK).mean().shift(1)
    suf = f"_{INF_TF}"
    day_id_col = d[f"day_id{suf}"]
    typical = (d["high"] + d["low"] + d["close"]) / 3
    d["vwap"] = (typical * d["volume"]).groupby(day_id_col).cumsum() / d["volume"].groupby(day_id_col).cumsum()

    day_open = d[f"day_open{suf}"]
    day_id = d[f"day_id{suf}"]
    prev_close = d["close"].shift(1)
    same_day = day_id == day_id.shift(1)
    vol_ok = d["volume"] >= (d["vol_sma"] * VOL_MULT)
    above_vwap = d["close"] > d["vwap"]
    below_vwap = d["close"] < d["vwap"]

    cross_up = (d["close"] > day_open) & (prev_close <= day_open)
    cross_dn = (d["close"] < day_open) & (prev_close >= day_open)

    def funnel(label, *conds):
        m = conds[0]
        print(f"  {label}")
        names = ["dir", "cross", "same_day", "vol_ok", "vwap"]
        for n, c in zip(names, conds):
            m = m & c
            print(f"    +{n:9s}: {int(m.sum())}")

    funnel("LONG",
           d[f"dir_long{suf}"] == 1, cross_up, same_day, vol_ok, above_vwap)
    funnel("SHORT",
           d[f"dir_short{suf}"] == 1, cross_dn, same_day, vol_ok, below_vwap)

    # 추가 통계
    print(f"  [참고] cross_up 총={int(cross_up.sum())} cross_dn 총={int(cross_dn.sum())} "
          f"vol_ok 총={int(vol_ok.sum())} (전체 {len(d)})")
    # dir_long 이 5m 으로 ffill 됐을 때 몇 개 행이 1인지
    print(f"  [참고] dir_long_1d==1 행={int((d[f'dir_long{suf}']==1).sum())} "
          f"dir_short_1d==1 행={int((d[f'dir_short{suf}']==1).sum())}")


if __name__ == "__main__":
    for p in ["BTC/USDT:USDT", "ETH/USDT:USDT"]:
        analyze(p)
