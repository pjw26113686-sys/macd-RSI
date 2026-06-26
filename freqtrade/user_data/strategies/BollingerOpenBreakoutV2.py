# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
"""
BollingerOpenBreakoutV2
=======================
BollingerOpenBreakout(MVP) 의 **고도화 버전** — 자동매매 운용을 염두에 둔 개선판.

V1 대비 변경점 (자세한 근거는 docs/bollinger_backtest_report.md)
---------------------------------------------------------------
[버그수정]
  B1. day_open 이 "전일 시가"였던 문제 → "당일 시가"로 교정.
      V1 은 inf["day_open"]=inf["open"] 을 merge_informative_pair 로 넘겼는데,
      merge 가 일봉을 한 칸 미뤄 5m 행에는 '직전 완성 일봉(=어제)'의 시가가 들어왔다.
      당일 시가는 그날 00:00 에 이미 확정돼 사용해도 룩어헤드가 아니다 → 5m 에서
      그날 첫 봉의 open 을 직접 계산해 쓴다.
  B2. 방향성 신호의 off-by-one → 교정.
      V1 은 일봉 내부에서 shift(1)(=전일) 로 dir 을 만든 뒤 merge 가 다시 한 칸
      밀어, 실제로는 '그제(D-2)' 기준이 됐다. V2 는 일봉 캔들 자체에서 터치 플래그를
      계산하고 merge 의 한 칸 시프트만 적용해 '전일(D-1)' 기준으로 맞춘다.

[고도화]
  A1. 하이퍼옵트 파라미터화 — bb_std, 방향성 lookback(N일), 거래량/ VWAP 필터
      on/off 및 임계값을 IntParameter/DecimalParameter/BooleanParameter 로 노출.
      V1 의 과도하게 쌓인 AND 필터(거래 0 유발)를 데이터로 튜닝/AB 할 수 있다.
  A2. 방향성 lookback — '전일'만이 아니라 '최근 N일 내 밴드터치-반전'을 허용(N 튜닝).
  A3. 실청산 추가 — 돌파 실패(종가가 VWAP 되돌림) 신호청산 + 트레일링스톱 + ROI.
  A4. 룩어헤드 안전 VWAP/day_open 을 5m 네이티브 일자 기준으로 앵커.

룩어헤드/재귀편향
----------------
- 일봉 BB/터치플래그는 완성된 일봉에서 계산 후 merge 의 1-틱 시프트로 '전일'화.
- day_open/vwap 은 당일 5m 누적(시가는 당일 확정값) → 미래참조 아님.
- recursive-analysis: 일봉 지표는 get_pair_dataframe 가 전체 일봉을 로드하므로
  startup 에 무관하게 안정. vol_sma 만 5m rolling(작은 startup 으로 충분).
- 반드시 freqtrade lookahead-analysis / recursive-analysis 로 재확인할 것.
"""

from datetime import datetime
from functools import reduce

from pandas import DataFrame

import talib.abstract as ta
from freqtrade.strategy import (
    IStrategy,
    merge_informative_pair,
    IntParameter,
    DecimalParameter,
    BooleanParameter,
)


class BollingerOpenBreakoutV2(IStrategy):

    INTERFACE_VERSION = 3

    timeframe = "5m"
    informative_tf = "1d"
    can_short = True

    # ----- 청산 스택 (모두 하이퍼옵트 가능: --spaces roi stoploss trailing) -----
    minimal_roi = {
        "0": 0.06,
        "240": 0.03,
        "720": 0.015,
        "1200": 0.0,   # ~20시간 경과 시 본전 이상이면 정리(일봉 보유형)
    }
    stoploss = -0.03

    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    process_only_new_candles = True

    # 5m 단위. 일봉 지표는 get_pair_dataframe 가 전체 로드 → startup 작아도 안정.
    startup_candle_count = 60

    # 고정 일봉 BB 기간(표준 20). 표준편차는 튜닝.
    bb_period = 20

    # ------------------------------------------------------------------ #
    #  하이퍼옵트 파라미터
    # ------------------------------------------------------------------ #
    bb_std = DecimalParameter(1.5, 2.5, default=2.0, decimals=1, space="buy")
    # 최근 N 완성일봉 내 '밴드터치-반전'이 있었으면 방향성 성립.
    dir_lookback = IntParameter(1, 5, default=2, space="buy")

    use_vol_filter = BooleanParameter(default=True, space="buy")
    vol_lookback = IntParameter(10, 50, default=20, space="buy")
    vol_mult = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="buy")

    use_vwap_filter = BooleanParameter(default=True, space="buy")

    # 돌파 실패(종가가 VWAP 반대편으로 되돌림) 신호청산 on/off.
    exit_on_vwap = BooleanParameter(default=True, space="sell")

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, self.informative_tf) for pair in pairs]

    # ------------------------------------------------------------------ #
    #  지표
    # ------------------------------------------------------------------ #
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === 일봉(informative) ===
        inf = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.informative_tf
        )
        bb = ta.BBANDS(
            inf, timeperiod=self.bb_period,
            nbdevup=self.bb_std.value, nbdevdn=self.bb_std.value,
        )
        inf["bb_upper"] = bb["upperband"]
        inf["bb_mid"] = bb["middleband"]
        inf["bb_lower"] = bb["lowerband"]

        # 완성된 일봉에서 '밴드터치 후 반전' 플래그 (shift 없음 — merge 가 1틱 시프트).
        touch_upper = (inf["high"] >= inf["bb_upper"]) & (inf["close"] < inf["bb_mid"])
        touch_lower = (inf["low"] <= inf["bb_lower"]) & (inf["close"] > inf["bb_mid"])

        # 최근 N일 내 터치 → 방향성 (N 은 하이퍼옵트).
        N = self.dir_lookback.value
        inf["dir_short"] = touch_upper.rolling(N, min_periods=1).max().fillna(0).astype(int)
        inf["dir_long"] = touch_lower.rolling(N, min_periods=1).max().fillna(0).astype(int)

        keep = ["date", "dir_short", "dir_long", "bb_upper", "bb_mid", "bb_lower"]
        dataframe = merge_informative_pair(
            dataframe, inf[keep], self.timeframe, self.informative_tf, ffill=True
        )

        # === 5m 네이티브 당일 기준선 (룩어헤드 안전) ===
        # 당일 시가: 그날 첫 5m 의 open (00:00 에 확정). 미래참조 아님.
        day_start = dataframe["date"].dt.floor("1d")
        dataframe["day_start"] = day_start
        dataframe["day_open"] = dataframe.groupby(day_start)["open"].transform("first")

        # 앵커드 VWAP: 당일 00:00 앵커, 당일 내 누적.
        typical = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        tp_vol = typical * dataframe["volume"]
        dataframe["vwap"] = (
            tp_vol.groupby(day_start).cumsum()
            / dataframe["volume"].groupby(day_start).cumsum()
        )

        # 거래량 SMA(직전까지) — 자기참조 방지로 shift(1).
        dataframe["vol_sma"] = (
            dataframe["volume"].rolling(self.vol_lookback.value).mean().shift(1)
        )
        return dataframe

    # ------------------------------------------------------------------ #
    #  진입
    # ------------------------------------------------------------------ #
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        suf = f"_{self.informative_tf}"
        day_open = dataframe["day_open"]
        day_start = dataframe["day_start"]
        prev_close = dataframe["close"].shift(1)
        same_day = day_start == day_start.shift(1)

        if self.use_vol_filter.value:
            vol_ok = dataframe["volume"] >= (dataframe["vol_sma"] * self.vol_mult.value)
        else:
            vol_ok = dataframe["volume"] > 0

        if self.use_vwap_filter.value:
            above_vwap = dataframe["close"] > dataframe["vwap"]
            below_vwap = dataframe["close"] < dataframe["vwap"]
        else:
            above_vwap = below_vwap = dataframe["close"] > 0  # 항상 True

        long_break = (
            (dataframe[f"dir_long{suf}"] == 1)
            & (dataframe["close"] > day_open)
            & (prev_close <= day_open)
            & same_day
            & (dataframe["volume"] > 0)
            & vol_ok
            & above_vwap
        )
        short_break = (
            (dataframe[f"dir_short{suf}"] == 1)
            & (dataframe["close"] < day_open)
            & (prev_close >= day_open)
            & same_day
            & (dataframe["volume"] > 0)
            & vol_ok
            & below_vwap
        )

        # 상호배제: 한 캔들에서 롱/숏 동시 → 둘 다 폐기.
        collide = long_break & short_break
        long_break &= ~collide
        short_break &= ~collide

        # 하루 첫 돌파만(같은 day_start 내 누적 1회).
        long_first = long_break & (long_break.groupby(day_start).cumsum() == 1)
        short_first = short_break & (short_break.groupby(day_start).cumsum() == 1)

        dataframe.loc[long_first, ["enter_long", "enter_tag"]] = (1, "open_breakout_long")
        dataframe.loc[short_first, ["enter_short", "enter_tag"]] = (1, "open_breakout_short")
        return dataframe

    # ------------------------------------------------------------------ #
    #  청산 — 돌파 실패(VWAP 되돌림) 신호청산. 나머지는 ROI/트레일링/스톱.
    # ------------------------------------------------------------------ #
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.exit_on_vwap.value:
            # 롱: 종가가 VWAP 아래로 되돌아오면 추세 실패로 보고 청산.
            lx = dataframe["close"] < dataframe["vwap"]
            sx = dataframe["close"] > dataframe["vwap"]
            dataframe.loc[lx, ["exit_long", "exit_tag"]] = (1, "vwap_revert")
            dataframe.loc[sx, ["exit_short", "exit_tag"]] = (1, "vwap_revert")
        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return 1.0
