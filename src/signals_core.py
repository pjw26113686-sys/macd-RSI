"""순수 신호 모듈 — 양 엔진(현재는 단일 엔진)이 공유하는 "두뇌".

이 모듈은 **결정론적 불리언 신호 컬럼까지만** 책임진다.
포지션 상태 전이(지연진입 대기/폐기, 분할익절→본전이동→잔량청산)는
상태머신이므로 엔진 레이어(`src/engine`)가 가진다.

무결성 원칙 (strategy_spec_v2 §0):
- 모든 컬럼은 t시점까지의 데이터(현재/과거)만 사용한다. 미래참조 0.
  (오직 `.shift(+k)` 즉 과거 참조만 사용. `.shift(-k)` 금지.)
- 지표는 라이브러리 네이티브가 아니라 여기서 단일 구현한다(엔진 간 드리프트 방지).
- RSI는 Wilder(RMA) 방식 — TradingView `ta.rsi` 및 Pine 레퍼런스와 일치.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------- #
# 이동평균 / 지표 빌딩 블록
# --------------------------------------------------------------------------- #
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder의 RMA(=RMA smoothing). alpha = 1/period."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _wma(series: pd.Series, period: int) -> pd.Series:
    """가중이동평균. 가중치 1..period (최신봉이 가장 큼)."""
    if period <= 1:
        return series.astype(float).copy()
    weights = np.arange(1, period + 1, dtype=float)
    wsum = weights.sum()
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / wsum, raw=True
    )


def rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI. 첫 `period`개 구간은 NaN(워밍업)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss==0 (전부 상승) → RSI=100
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi


def hma(close: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average.
    HMA(n) = WMA( 2*WMA(p, n/2) - WMA(p, n), sqrt(n) )
    """
    half = max(1, int(round(period / 2)))
    sqrt_n = max(1, int(round(np.sqrt(period))))
    raw = 2.0 * _wma(close, half) - _wma(close, period)
    return _wma(raw, sqrt_n)


# --------------------------------------------------------------------------- #
# 1단계: 지표 컬럼
# --------------------------------------------------------------------------- #
def compute_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """OHLCV DataFrame에 지표 컬럼을 추가해 반환(원본 비변경, copy)."""
    _validate_ohlcv(df)
    out = df.copy()

    ema_fast = _ema(out["close"], params["macd_fast"])
    ema_slow = _ema(out["close"], params["macd_slow"])
    out["macd_line"] = ema_fast - ema_slow
    out["signal_line"] = _ema(out["macd_line"], params["macd_signal"])
    out["hist"] = out["macd_line"] - out["signal_line"]

    out["rsi"] = rsi_wilder(out["close"], params["rsi_period"])
    out["hma"] = hma(out["close"], params["hma_period"])
    out["vol_ma"] = out["volume"].rolling(window=params["vol_ma_period"]).mean()
    return out


# --------------------------------------------------------------------------- #
# 2단계: 불리언 신호 컬럼 (캔들 단위, 상태 비의존)
# --------------------------------------------------------------------------- #
def compute_signal_columns(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """`compute_indicators` 결과에 불리언 신호 컬럼을 추가해 반환."""
    out = df.copy()
    macd = out["macd_line"]
    sig = out["signal_line"]
    hist = out["hist"]
    rsi = out["rsi"]
    low_th = params["rsi_entry_low"]
    high_th = params["rsi_entry_high"]
    band = params["rsi_support_band"]

    # MACD 크로스
    out["macd_golden_cross"] = (macd.shift(1) <= sig.shift(1)) & (macd > sig)
    out["macd_dead_cross"] = (macd.shift(1) >= sig.shift(1)) & (macd < sig)
    out["macd_above_signal"] = macd > sig

    # 추세 / 거래량 필터
    out["hma_uptrend"] = out["hma"] > out["hma"].shift(1)
    out["volume_ok"] = out["volume"] >= params["vol_ma_mult"] * out["vol_ma"]

    # 양봉 + 몸통비율 (high==low 0분모 방어)
    rng = (out["high"] - out["low"]).replace(0.0, np.nan)
    body_ratio = (out["close"] - out["open"]) / rng
    out["is_bull_candle"] = (out["close"] > out["open"]) & (
        body_ratio.fillna(0.0) >= params["min_body_ratio"]
    )

    # RSI 50 상향 돌파
    out["rsi_cross_up_low"] = (rsi.shift(1) < low_th) & (rsi >= low_th)

    # RSI 진입밴드 / 과열 / 지지밴드 / 50하향이탈
    out["rsi_in_entry_band"] = (rsi >= low_th) & (rsi < high_th)
    out["rsi_overheated"] = rsi >= high_th
    out["rsi_in_support_band"] = (rsi >= low_th - band) & (rsi <= low_th + band)
    out["rsi_exit_below_low"] = rsi < low_th

    # 히스토그램 조기신호 (로깅 전용 — 1차 진입엔 미사용, 2차 A/B)
    out["hist_turn_up"] = (hist.shift(1) < hist) & (hist.shift(2) > hist.shift(1))

    # 기본 진입 (ENTRY_BASE) — 상태 비의존 부분만 (§3.1)
    out["entry_base"] = (
        out["macd_golden_cross"]
        & out["rsi_in_entry_band"]
        & out["hma_uptrend"]
        & out["volume_ok"]
    )
    return out


def compute_entry_candidates(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """진입 후보 컬럼(`enter_long`, `entry_kind`)을 추가해 반환.

    "무포지션이면 진입" 후보를 셋업 상태머신 1패스로 산출한다(포지션 상태 무관).
    자체 엔진과 freqtrade 전략이 **둘 다 이 컬럼을 소비**해 진입을 일치시킨다.

    상태머신 (§3.2/3.3/3.4):
      - ENTRY_BASE(최우선) → 진입, 셋업 해제.
      - 이전 봉에서 무장된 셋업이 delay_window 내에 발동 → 진입, 셋업 해제.
      - base 불발한 골든크로스에서 RSI<50 → low 셋업, RSI>=70 → high 셋업 무장.
      - 우선순위: base > delay_high > delay_low (한 봉에 동시 충족 시).

    인과성: 각 봉의 후보는 과거/현재 봉만으로 결정된다(셋업은 과거 골든크로스에서
    무장). 따라서 t시점 절단 후 계산해도 t시점 값이 불변 → 미래참조 0.
    """
    out = df.copy()
    low_th = params["rsi_entry_low"]
    dwin = params["delay_window"]
    n = len(out)

    base = out["entry_base"].to_numpy()
    gc = out["macd_golden_cross"].to_numpy()
    overheated = out["rsi_overheated"].to_numpy()
    rsi = out["rsi"].to_numpy()
    above = out["macd_above_signal"].to_numpy()
    rsi_cross = out["rsi_cross_up_low"].to_numpy()
    support = out["rsi_in_support_band"].to_numpy()
    bull = out["is_bull_candle"].to_numpy()
    hup = out["hma_uptrend"].to_numpy()
    vok = out["volume_ok"].to_numpy()

    enter = np.zeros(n, dtype=bool)
    kind = np.array([""] * n, dtype=object)
    setup = None  # {"kind": "low"|"high", "expires": int}

    for t in range(n):
        # 1) 기본 진입 (최우선)
        if base[t]:
            enter[t] = True
            kind[t] = "base"
            setup = None
            continue
        # 2) 무장된 셋업(이전 봉 무장)의 만료/발동
        if setup is not None:
            if t > setup["expires"]:
                setup = None
            elif setup["kind"] == "high" and above[t] and support[t] and bull[t] and hup[t] and vok[t]:
                enter[t] = True
                kind[t] = "delay_high"
                setup = None
                continue
            elif setup["kind"] == "low" and above[t] and rsi_cross[t] and bull[t] and hup[t] and vok[t]:
                enter[t] = True
                kind[t] = "delay_low"
                setup = None
                continue
        # 3) base 불발 골든크로스 → 셋업 무장
        if gc[t]:
            if overheated[t]:
                setup = {"kind": "high", "expires": t + dwin}
            elif rsi[t] < low_th:
                setup = {"kind": "low", "expires": t + dwin}

    out["enter_long"] = enter
    out["entry_kind"] = kind
    return out


def add_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """편의 함수: 지표 + 신호 + 진입후보 컬럼을 한 번에."""
    out = compute_signal_columns(compute_indicators(df, params), params)
    return compute_entry_candidates(out, params)


# 엔진/테스트가 t시점 결정성을 검증할 때 비교 대상으로 쓰는 신호 컬럼들.
SIGNAL_COLUMNS = [
    "macd_golden_cross",
    "macd_dead_cross",
    "macd_above_signal",
    "hma_uptrend",
    "volume_ok",
    "is_bull_candle",
    "rsi_cross_up_low",
    "rsi_in_entry_band",
    "rsi_overheated",
    "rsi_in_support_band",
    "rsi_exit_below_low",
    "hist_turn_up",
    "entry_base",
    "enter_long",
]


def _validate_ohlcv(df: pd.DataFrame) -> None:
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV 컬럼 누락: {missing}")
