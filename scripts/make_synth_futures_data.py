"""합성(synthetic) 선물 OHLCV 생성기 — 네트워크 차단 환경용 백테스트 데이터.

이 레포가 도는 샌드박스는 binance/bybit 등 모든 시세 호스트를 403 으로 막는다.
실데이터를 받을 수 없으므로, **전략이 에러 없이 돌고 lookahead/recursive-analysis
무결성 검사를 통과하는지** 검증하기 위한 현실적 합성 데이터를 만든다.

설계 포인트
-----------
- 5m 캔들을 기하 브라운 운동(GBM)으로 만들고, **1d 는 5m 를 리샘플**해서 생성한다.
  → 두 타임프레임이 항상 정합(일봉 시가 = 그날 첫 5m 시가)하므로, "당일 시가 돌파"
    로직이 의미를 가진다.
- freqtrade 선물 백테스트가 요구하는 파일을 모두 생성:
    <PAIR>-<tf>-futures.feather   (거래 캔들)
    <PAIR>-<tf>-mark.feather      (마크가격 = 거래 캔들 복제)
    <PAIR>-8h-funding_rate.feather(펀딩비 = 0)
  경로: user_data/data/<exchange>/futures/

주의: 합성 데이터의 성과(수익률)는 **무의미**하다. 목적은 무결성·동작 검증뿐.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def gen_5m(start: str, days: int, start_price: float, seed: int) -> pd.DataFrame:
    """5m OHLCV 생성 — 크립토스러운 통계 특성을 반영.

    랜덤워크에 ①팻테일(Student-t) ②변동성 군집(AR(1) on log-vol)
    ③청산·스탑헌트성 큰 꼬리(가끔 5m 봉에 거대한 상/하 윅) 를 더한다.
    → 일봉 고가가 밴드를 찌르되 종가는 되돌아오는 '밴드터치 후 반전' 패턴이
      현실적 빈도로 나타나, 평균회귀 신호를 실제로 검증할 수 있다.
    (수익률 자체는 여전히 무의미 — 무결성/동작 검증 전용.)
    """
    rng = np.random.default_rng(seed)
    n = days * 288  # 288 = 5m 봉/일
    idx = pd.date_range(start=start, periods=n, freq="5min", tz="UTC")

    bars_per_year = 288 * 365
    base_sigma = 0.70 / np.sqrt(bars_per_year)  # 연 변동성 ~70%
    mu = 0.05 / bars_per_year

    # 변동성 군집: log-vol 을 AR(1) 로 흔든다 (조용한 구간/폭발 구간 교대).
    log_vol = np.zeros(n)
    for i in range(1, n):
        log_vol[i] = 0.995 * log_vol[i - 1] + rng.normal(0, 0.05)
    sigma_t = base_sigma * np.exp(log_vol)

    # 팻테일 수익률: Student-t(df=4) 표준화 후 sigma_t 스케일.
    tdf = 4
    t = rng.standard_t(tdf, n) / np.sqrt(tdf / (tdf - 2))
    rets = mu - 0.5 * sigma_t**2 + sigma_t * t
    close = start_price * np.exp(np.cumsum(rets))

    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    # 일반 윅: 변동성 비례. 큰 꼬리: 드물게 거대한 상/하 스파이크(청산 윅).
    up_wick = np.abs(rng.normal(0, sigma_t)) * close
    dn_wick = np.abs(rng.normal(0, sigma_t)) * close
    spike_mask = rng.random(n) < 0.003          # ~0.3% 봉에서 스파이크
    spike_up = spike_mask & (rng.random(n) < 0.5)
    spike_dn = spike_mask & ~spike_up
    up_wick[spike_up] += rng.uniform(0.01, 0.05, spike_up.sum()) * close[spike_up]
    dn_wick[spike_dn] += rng.uniform(0.01, 0.05, spike_dn.sum()) * close[spike_dn]

    high = np.maximum(open_, close) + up_wick
    low = np.minimum(open_, close) - dn_wick
    low = np.maximum(low, close * 0.001)        # 가격 양수 보장

    # 거래량: 큰 가격이동/스파이크 봉에 강하게 연동(실제 돌파엔 거래량이 실린다).
    move = np.abs(rets) / (sigma_t + 1e-12)          # 변동성 대비 이동 크기(z)
    vol = rng.lognormal(mean=3.0, sigma=0.4, size=n) * (1 + 0.8 * move)
    vol[spike_mask] *= rng.uniform(2.0, 5.0, spike_mask.sum())

    df = pd.DataFrame(
        {"date": idx, "open": open_, "high": high, "low": low,
         "close": close, "volume": vol}
    )
    return df


def resample_1d(df5: pd.DataFrame) -> pd.DataFrame:
    """5m → 1d 리샘플 (정합성 보장)."""
    s = df5.set_index("date")
    agg = s.resample("1d").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna()
    return agg.reset_index()


def write_feather(df: pd.DataFrame, exchange: str, pair: str, tf: str, ctype: str):
    safe = pair.replace("/", "_").replace(":", "_")
    dest = REPO_ROOT / "freqtrade" / "user_data" / "data" / exchange / "futures"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{safe}-{tf}-{ctype}.feather"
    df.reset_index(drop=True).to_feather(path)
    print(f"  {path.name}: {len(df)}행")
    return path


def funding_df(df5: pd.DataFrame) -> pd.DataFrame:
    """8h 펀딩비 = 0 (합성이므로 펀딩비 영향 제거)."""
    s = df5.set_index("date")
    idx = pd.date_range(s.index[0].normalize(), s.index[-1], freq="8h", tz="UTC")
    return pd.DataFrame({"date": idx, "open": 0.0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--days", type=int, default=1400)
    args = ap.parse_args()

    pairs = {
        "BTC/USDT:USDT": (42000.0, 1),
        "ETH/USDT:USDT": (2300.0, 2),
        "SOL/USDT:USDT": (95.0, 3),
        "BNB/USDT:USDT": (320.0, 4),
        "XRP/USDT:USDT": (0.52, 5),
        "ADA/USDT:USDT": (0.45, 6),
        "DOGE/USDT:USDT": (0.082, 7),
        "AVAX/USDT:USDT": (28.0, 8),
    }

    for pair, (price, seed) in pairs.items():
        print(f"[{pair}]")
        df5 = gen_5m(args.start, args.days, price, seed)
        df1d = resample_1d(df5)
        # 거래 캔들
        write_feather(df5, args.exchange, pair, "5m", "futures")
        write_feather(df1d, args.exchange, pair, "1d", "futures")
        # 마크가격 = 거래 캔들 복제 (청산가 계산용)
        write_feather(df5, args.exchange, pair, "5m", "mark")
        write_feather(df1d, args.exchange, pair, "1d", "mark")
        # 펀딩비 0
        write_feather(funding_df(df5), args.exchange, pair, "8h", "funding_rate")

    print("\n합성 데이터 생성 완료. (성과는 무의미 — 무결성/동작 검증 전용)")


if __name__ == "__main__":
    main()
