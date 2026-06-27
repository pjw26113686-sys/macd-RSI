"""순수 지표 라이브러리 — 모든 전략이 공유하는 단일 구현.

전략(`src/strategies/*`)이 import해서 신호를 만든다. 같은 지표를 전략마다 다르게
구현하면 엔진 간/전략 간 드리프트가 생기므로, **지표는 여기 한 곳에서만** 정의한다.

무결성: 모든 함수는 t시점까지 데이터(현재/과거)만 사용한다(.shift(+k)만).
미래참조(.shift(-k)) 금지 — `tests/test_lookahead.py`가 전략별로 강제한다.

RSI는 Wilder(RMA) 방식 — TradingView `ta.rsi`/Pine 레퍼런스와 일치.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------- #
# 이동평균 빌딩 블록
# --------------------------------------------------------------------------- #
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder의 RMA. alpha = 1/period."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """가중이동평균. 가중치 1..period (최신봉 최대)."""
    if period <= 1:
        return series.astype(float).copy()
    weights = np.arange(1, period + 1, dtype=float)
    wsum = weights.sum()
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / wsum, raw=True
    )


def hma(close: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average = WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half = max(1, int(round(period / 2)))
    sqrt_n = max(1, int(round(np.sqrt(period))))
    raw = 2.0 * wma(close, half) - wma(close, period)
    return wma(raw, sqrt_n)


# --------------------------------------------------------------------------- #
# 모멘텀 / 변동성 지표
# --------------------------------------------------------------------------- #
def rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI. 첫 period 구간은 NaN(워밍업)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.where(avg_loss != 0.0, 100.0)


def macd(close: pd.Series, fast: int, slow: int, signal: int):
    """(macd_line, signal_line, hist) 반환."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def bollinger(close: pd.Series, period: int, std_mult: float):
    """(mid, upper, lower) 반환. mid=SMA, 밴드=mid ± std_mult*std(ddof=0)."""
    mid = close.rolling(window=period).mean()
    sd = close.rolling(window=period).std(ddof=0)
    return mid, mid + std_mult * sd, mid - std_mult * sd


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range (Wilder RMA)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return rma(tr, period)


def disparity(close: pd.Series, period: int) -> pd.Series:
    """이격도 = close / SMA(period) * 100."""
    return close / close.rolling(window=period).mean() * 100.0


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """a가 b를 상향 돌파한 봉(bool). 인과적(.shift(+1)만 사용)."""
    return (a.shift(1) <= b.shift(1)) & (a > b)


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """a가 b를 하향 돌파한 봉(bool)."""
    return (a.shift(1) >= b.shift(1)) & (a < b)


def validate_ohlcv(df: pd.DataFrame) -> None:
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV 컬럼 누락: {missing}")
