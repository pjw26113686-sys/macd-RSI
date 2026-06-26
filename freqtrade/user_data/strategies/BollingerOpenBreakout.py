# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
"""
BollingerOpenBreakout
=====================
일봉 캔들 하나를 제대로 익절하기 위한 멀티 타임프레임 전략 (1~2단계 MVP).

설계 의도
---------
1) 방향성 (일봉 볼린저밴드, 20기간 / 2STD):
   - 숏 편향: 전일 일봉 고가가 상단밴드에 도달 + 전일 종가가 중심선(20MA) 아래 마감
   - 롱 편향: 전일 일봉 저가가 하단밴드에 도달 + 전일 종가가 중심선(20MA) 위 마감
   (밴드 끝에서 중심선으로 되돌아오는 움직임을 포착)

2) 타점 (5m, 시가 돌파 진입):
   - 롱: 일봉 롱 편향 + 5m 종가가 "당일 일봉 시가" 위로 돌파
   - 숏: 일봉 숏 편향 + 5m 종가가 "당일 일봉 시가" 아래로 돌파

3) 청산: 고정 손절 + ROI (쌍바닥 동적 손절·매물대 필터는 3단계에서 추가 예정)

룩어헤드 방지
------------
- 일봉 방향성은 모두 "전일(완성된 캔들)" 기준 → merge 후 shift 로 처리.
- merge_informative_pair 가 informative 캔들을 다음 타임프레임으로 한 칸 밀어주므로
  (ffill), 5m 행에는 "직전에 완성된 일봉" 값이 들어온다. 추가로 방향성 신호는
  일봉 dataframe 안에서 shift(1) 하여 '전일' 기준으로 계산한다.

주의
----
- 이 전략은 그대로 수익을 보장하지 않는다. 반드시 dry-run 과
  lookahead-analysis / recursive-analysis 로 검증할 것.
- 5m 실행 + 1d informative 구조이므로, 데이터는 5m 와 1d 둘 다 받아야 한다.
    freqtrade download-data --exchange binance --trading-mode futures \
        --timeframes 5m 1d --pairs BTC/USDT:USDT ETH/USDT:USDT --days 365
"""

from datetime import datetime

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy,
    merge_informative_pair,
)


class BollingerOpenBreakout(IStrategy):

    INTERFACE_VERSION = 3

    # ----- 기본 모드 -----
    timeframe = "5m"           # 실행(타점) 타임프레임
    informative_tf = "1d"      # 방향성 타임프레임

    can_short = True           # 선물 양방향

    # ----- 청산 파라미터 (MVP: 고정값) -----
    # 손익비 우선 원칙: 손절 -2% 대비 익절을 더 크게.
    # "일봉 하나를 제대로 익절" 목표 → 하루 단위 보유형 ROI (분 단위 키).
    minimal_roi = {
        "0": 0.06,     # 진입 직후 +6% 도달 시 익절
        "480": 0.03,   # 8시간 경과 후 +3%
        "1200": 0.015, # 20시간(거의 하루) 경과 후 +1.5%로 완화
    }
    stoploss = -0.02   # 고정 -2% 손절 (3단계에서 쌍바닥 동적 손절로 교체 예정)

    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False

    # 일봉 BB(20) 안정값 확보를 위해 충분한 startup 캔들 확보.
    # 1d 20기간 → 5m 환산 시 여유있게. recursive-analysis 로 0% 나오는지 확인 권장.
    startup_candle_count = 200

    # ----- 볼린저밴드 설정 -----
    bb_period = 20
    bb_std = 2.0

    # ----- 거래량 필터 설정 -----
    vol_lookback = 20    # 거래량 평균 산출 봉 수 (직전 20개 5m)
    vol_mult = 1.5       # 돌파 캔들 거래량 ≥ 평균 × 1.5 (크립토 5m 기준 시작값)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, self.informative_tf) for pair in pairs]

    # ------------------------------------------------------------------ #
    #  지표 계산
    # ------------------------------------------------------------------ #
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === 1) 일봉(informative) 지표 ===
        inf = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.informative_tf
        )

        bb = ta.BBANDS(
            inf, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
        )
        inf["bb_upper"] = bb["upperband"]
        inf["bb_mid"] = bb["middleband"]
        inf["bb_lower"] = bb["lowerband"]

        # 방향성 신호를 일봉 dataframe 안에서 "전일 기준"으로 계산 (shift(1) = 전일)
        prev_high = inf["high"].shift(1)
        prev_low = inf["low"].shift(1)
        prev_close = inf["close"].shift(1)
        prev_upper = inf["bb_upper"].shift(1)
        prev_mid = inf["bb_mid"].shift(1)
        prev_lower = inf["bb_lower"].shift(1)

        # 숏 편향: 전일 고가가 상단밴드 도달 & 전일 종가가 중심선 아래
        inf["dir_short"] = (
            (prev_high >= prev_upper) & (prev_close < prev_mid)
        ).astype(int)

        # 롱 편향: 전일 저가가 하단밴드 도달 & 전일 종가가 중심선 위
        inf["dir_long"] = (
            (prev_low <= prev_lower) & (prev_close > prev_mid)
        ).astype(int)

        # 당일 일봉 시가(돌파 기준선). merge 시 5m 행으로 ffill 된다.
        inf["day_open"] = inf["open"]
        # 일봉 식별자(경계 감지용). merge 후 5m 행에서 이 값이 바뀌면 새 일봉 시작.
        inf["day_id"] = inf["date"].astype("int64")

        # === 2) 5m 와 일봉 병합 ===
        # merge_informative_pair 가 일봉 캔들을 한 칸 밀어 5m 에 정렬해 준다
        # (각 5m 행에는 '직전에 완성된 일봉' 값이 들어옴 → 룩어헤드 방지).
        keep = ["date", "dir_short", "dir_long", "day_open", "day_id",
                "bb_upper", "bb_mid", "bb_lower"]
        dataframe = merge_informative_pair(
            dataframe, inf[keep], self.timeframe, self.informative_tf, ffill=True
        )
        # 병합된 컬럼명: 원본명 + "_" + informative_tf  (예: dir_short_1d)

        # === 3) 5m 거래량 필터 지표 ===
        # 돌파 캔들 거래량을 직전 N봉 평균과 비교하기 위한 SMA.
        # shift(1) 으로 '직전까지'의 평균을 써서 현재 캔들 자신을 평균에 포함하지 않음
        # (자기참조/룩어헤드 방지).
        dataframe["vol_sma"] = (
            dataframe["volume"].rolling(self.vol_lookback).mean().shift(1)
        )

        # === 4) 일봉 앵커드 VWAP (5m 누적, day_id 로 리셋) ===
        # 크립토 선물은 명확한 세션 오픈이 없으므로, 일봉 시작(UTC 자정)을 앵커로
        # 잡아 그 일봉 안에서만 누적한다. ORB 의 '당일 시가 돌파' 로직과 정렬됨.
        suf = f"_{self.informative_tf}"
        day_id_col = dataframe[f"day_id{suf}"]
        typical = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        tp_vol = typical * dataframe["volume"]
        # groupby 누적: tp_vol 과 volume 을 각각 일봉 그룹 내 누적합
        dataframe["_cum_tpvol"] = tp_vol.groupby(day_id_col).cumsum()
        dataframe["_cum_vol"] = dataframe["volume"].groupby(day_id_col).cumsum()
        dataframe["vwap"] = dataframe["_cum_tpvol"] / dataframe["_cum_vol"]
        dataframe.drop(columns=["_cum_tpvol", "_cum_vol"], inplace=True)

        return dataframe

    # ------------------------------------------------------------------ #
    #  진입
    # ------------------------------------------------------------------ #
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        suf = f"_{self.informative_tf}"
        day_open = dataframe[f"day_open{suf}"]
        day_id = dataframe[f"day_id{suf}"]
        prev_close = dataframe["close"].shift(1)

        # --- 경계 가드(버그 1 수정) ---
        # 직전 5m 와 현재 5m 가 같은 일봉에 속할 때만 '돌파'를 인정한다.
        # 일봉이 바뀐 직후 첫 캔들은 day_open 이 점프하므로 prev_close 비교가
        # 무의미해 잘못된 진입을 유발한다 → 같은 날일 때만 통과.
        same_day = day_id == day_id.shift(1)

        # --- 거래량 필터 + VWAP 정렬 ---
        # 거래량: 돌파 캔들이 직전 평균의 vol_mult 배 이상 (기관 참여 확인).
        vol_ok = dataframe["volume"] >= (dataframe["vol_sma"] * self.vol_mult)
        # VWAP: 롱은 VWAP 위에서 돌파(매수자 함정 배제), 숏은 VWAP 아래.
        above_vwap = dataframe["close"] > dataframe["vwap"]
        below_vwap = dataframe["close"] < dataframe["vwap"]

        # --- 하루 1타점 제한(버그 3 수정) ---
        # 같은 일봉(day_id) 안에서 돌파 후보가 처음 발생한 캔들에만 진입 허용.
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

        # 상호배제(발전 포인트): 한 캔들에서 롱/숏이 동시에 서면 둘 다 버린다.
        collide = long_break & short_break
        long_break = long_break & ~collide
        short_break = short_break & ~collide

        # 하루 중 '첫' 돌파만 남기기: day_id 그룹 내 누적 발생 횟수가 1일 때만.
        long_first = long_break & (
            long_break.groupby(day_id).cumsum() == 1
        )
        short_first = short_break & (
            short_break.groupby(day_id).cumsum() == 1
        )

        dataframe.loc[long_first, ["enter_long", "enter_tag"]] = (1, "open_breakout_long")
        dataframe.loc[short_first, ["enter_short", "enter_tag"]] = (1, "open_breakout_short")
        return dataframe

    # ------------------------------------------------------------------ #
    #  청산 (MVP 는 ROI/stoploss 에 의존, 신호 청산은 비워둠)
    # ------------------------------------------------------------------ #
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    # 레버리지: MVP 는 1배 고정 (리스크 관리는 손절폭으로). 필요 시 조정.
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
