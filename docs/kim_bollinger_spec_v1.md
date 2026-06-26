# 김직선 볼린저 변곡 전략 명세 v1 — 로컬 구현 인계용 (BTC 5m, 평균회귀 롱/숏)

> 목적: 유튜브 '김직선' 볼린저 변곡 전략을, 기존 레포 인프라를 재활용해 **결정론적
> 코드로 옮길 수 있게** 한 구현 명세. 로컬 CMD Claude Code가 이 문서 + 푸시된 레포로
> 바로 구현하도록 작성됨.
>
> 범위(확정): 5분봉 / BTC/USDT(ccxt) / 평균회귀 롱·숏 / 손절=직전 10봉 고·저 /
> v1 = 더블비(σ4) 밴드터치 + 이격도 다이버전스. RSI 삼산은 2차 옵션 토글.
> 제거: 뉴스 블랙아웃(요청). 폐기: 구 MACD+RSI 모멘텀 전략.

---

## 0. 백테스트 무결성 원칙 (위반 시 결과 무의미 — 기존 레포와 동일)

1. **신호는 캔들 종가 확정에서만.** 진행 중 봉으로 판단 금지.
2. **진입 체결가 = 신호 다음 봉(t+1)의 시가.**
3. **모든 지표·신호는 t시점까지 데이터만 사용**(인과적). 특히 다이버전스 pivot은
   미래봉으로 확정되므로 **확정 시점(=형성 k봉 뒤)에만** 신호화한다.
4. **손절·목표 동시 도달 시 손절 우선**(고저 순서 불명 → 최악 가정).
5. 지표는 라이브러리 네이티브가 아니라 `signals_core`에 단일 구현(엔진 간 드리프트
   방지). RSI는 기존 `signals_core.rsi_wilder` 재활용.

---

## 1. 파라미터 (시작값 — grid search 출발점)

| 파라미터 | 기호 | 시작값 | 비고 |
|---|---|---|---|
| 볼린저 기간 | `bb_period` | 20 | 중심선 SMA, 표준편차 창 |
| 기본 밴드 σ | `bb_std_main` | 2.0 | 참조/청산 보조 |
| 더블비 σ | `bb_std_wb` | 4.0 | **진입 트리거 밴드** |
| 이격도 기간 | `disp_period` | 20 | `close/SMA*100` |
| pivot 반폭 | `pivot_k` | 3 | 좌우 k봉 비교(확정 지연 k봉) |
| pivot 최대간격 | `max_pivot_gap` | 40 | 두 pivot 비교 허용 거리(봉) |
| 밴드터치 유효창 | `touch_window` | 3 | 터치 후 N봉 내 다이버전스 결합 |
| 손절 탐색 봉수 | `swing_lookback_M` | 10 | **직전 10봉 고/저점** |
| 손절 버퍼 | `stop_buffer` | 0.001 | 노이즈 여유 |
| 목표 방식 | `target_mode` | `mid` | `mid`=중심선 회귀 / `rr`=R배수 |
| 손익비(rr 모드) | `reward_ratio` | 1.5 | target_mode=rr일 때만 |
| 최대 보유봉 | `max_hold_bars` | 12 | 시간청산(5m×12=1시간) |
| RSI 기간 | `rsi_period` | 14 | 2차 |
| RSI 과매수/과매도 | `rsi_ob`/`rsi_os` | 70 / 30 | 2차 삼산 존 |
| 삼산 탐색창 | `triple_lookback` | 30 | 2차 |
| 삼산 사용 | `use_rsi_triple` | false | 2차 토글 |
| 롱 허용 | `enable_long` | true | |
| 숏 허용 | `enable_short` | true | |

---

## 2. 지표 (`src/signals_core.py` — `compute_indicators` 재작성)

재활용 헬퍼(기존): `_ema`, `_wma`, `_rma`, `rsi_wilder`. 신규 추가:

```
bb_mid     = close.rolling(bb_period).mean()
bb_std     = close.rolling(bb_period).std(ddof=0)
bb2_upper  = bb_mid + bb_std_main * bb_std      # 참조/청산
bb2_lower  = bb_mid - bb_std_main * bb_std
bb4_upper  = bb_mid + bb_std_wb   * bb_std      # 더블비 진입존
bb4_lower  = bb_mid - bb_std_wb   * bb_std
disparity  = close / close.rolling(disp_period).mean() * 100.0
rsi        = rsi_wilder(close, rsi_period)       # 2차
# ATR(선택, 손절 대안용 — v1은 미사용)
```

> 구 MACD/시그널/히스토그램/모멘텀 신호 컬럼은 **제거**.

---

## 3. 인과적 pivot + 이격도 다이버전스 (가장 주의 — 미래참조 핫스폿)

### 3.1 인과적 pivot 정의
가격 고점 pivot at index `p`: `high[p]`가 `high[p-k .. p+k]`의 유일 최대.
→ `p+k` 봉이 있어야 확정되므로 **확정봉 t = p+k**. (저점 pivot은 low로 대칭.)

결정론적 규칙(중복 방지): `high[p] > high[p-k..p-1]` **그리고** `high[p] >= high[p+1..p+k]`.

### 3.2 다이버전스 (가격 pivot에서 이격도 값을 비교 — regular divergence)
확정된 가격 고점 pivot들을 순서대로 보관(각각 `(p, high[p], disparity[p])`).
확정봉 `t=p+k`에서 직전 확정 고점 pivot `prev`와 비교:

```
bear_div[t] = True  IF  (p - prev <= max_pivot_gap)
                    AND  high[p]      >  high[prev]       # 가격 신고가(쌍봉 우상)
                    AND  disparity[p] <  disparity[prev]  # 이격도 고점 하락(어깨)
```
저점 pivot/`bull_div`는 대칭:
```
bull_div[t] = True  IF  (p - prev <= max_pivot_gap)
                    AND  low[p]       <  low[prev]        # 가격 신저가
                    AND  disparity[p] >  disparity[prev]  # 이격도 저점 상승(엉덩이)
```

**인과성:** `bear_div[t]`는 `t`까지 데이터만 사용(pivot at `p=t-k`는 `high[p..p+k=t]`로
확정). → `tests/test_lookahead.py`가 자동 강제. **`.shift(-n)`(미래참조) 절대 금지.**

### 3.3 RSI 삼산 (2차, `use_rsi_triple`)
과매수존(`rsi>=rsi_ob`)에서 인과적 RSI 고점 pivot 3개가 `triple_lookback` 내 연속
형성되면 `rsi_triple_top[t]=True`(셋째 pivot 확정봉). 과매도/`rsi_triple_bottom` 대칭.

---

## 4. 진입 규칙 (`compute_entry_candidates` 재작성 — 양방향)

밴드터치 유효: 최근 `touch_window`봉 내에 터치가 있었는가.
```
touch_upper4_recent[t] = any(high[t-touch_window+1 .. t] >= bb4_upper[그 봉])
touch_lower4_recent[t] = any(low[t-touch_window+1 .. t]  <= bb4_lower[그 봉])
```
진입 후보(종가 t 확정 → t+1 시가 체결):
```
enter_short[t] = enable_short AND touch_upper4_recent[t] AND bear_div[t]
                 [AND (not use_rsi_triple OR rsi_triple_top[t])]
enter_long[t]  = enable_long  AND touch_lower4_recent[t] AND bull_div[t]
                 [AND (not use_rsi_triple OR rsi_triple_bottom[t])]
```
- 동시 충족(같은 봉 long&short)은 정상적으로 거의 없음. 발생 시 진입 보류(skip).
- `entry_kind` 문자열('short_div'/'long_div' 등) 로깅 유지.
- **무포지션일 때만 신규 진입**(상태는 엔진이 관리).

`SIGNAL_COLUMNS`(미래참조 테스트 대상)에 추가:
`touch_upper4, touch_lower4, bear_div, bull_div, enter_long, enter_short`
(+ 2차 `rsi_triple_top/bottom`).

---

## 5. 청산 규칙 (`src/engine` — 양방향)

진입 시 확정:
```
# 손절 = 직전 swing_lookback_M(10)봉의 극단 (체결봉 t 기준 직전 M봉)
SHORT: stop   = max(high[t-M : t]) * (1 + stop_buffer)      # 위쪽
       target = bb_mid[t]   (target_mode=mid)               # 중심선 회귀
              또는 entry - reward_ratio*(stop-entry)        (rr 모드)
LONG : stop   = min(low[t-M : t])  * (1 - stop_buffer)      # 아래쪽
       target = bb_mid[t]   또는 entry + reward_ratio*(entry-stop)
```
봉내 점검 순서 = **손절 → 목표 → 시간청산** (손절 우선, §0-4):
```
SHORT: 손절 if high>=stop (체결 max(open,stop));  목표 if low<=target (체결 min(open,target))
LONG : 손절 if low<=stop  (체결 min(open,stop));  목표 if high>=target(체결 max(open,target))
시간청산: bars_held >= max_hold_bars → 다음봉 시가 청산
```
- v1은 **분할익절 없음**(전량 청산). 구 전략의 50%/본전스탑 로직은 사용 안 함.
- 신호기반 청산(데드크로스 등) 없음 — 평균회귀는 손절/목표/시간으로만 종료.

---

## 6. 엔진 숏 지원 (`src/engine/position.py`, `backtest.py` 일반화)

가장 큰 변경. 현재 롱 전용 → 방향 일반화.

**position.py:**
- `Position`에 `direction ∈ {"LONG","SHORT"}` 추가.
- 체결가 헬퍼:
  - LONG 진입 `open*(1+slip)`, 청산 `*(1-slip)`.
  - SHORT 진입 `open*(1-slip)`, 청산(커버) `*(1+slip)`.
- 손익(수수료 양변 차감, 명목수량 `qty=capital*position_pct/entry_fill`):
  - `LONG  pnl = (exit_fill - entry_fill)*qty - fees`
  - `SHORT pnl = (entry_fill - exit_fill)*qty - fees`
- 기존 `compute_stop_and_target`는 LONG 전용이므로 **방향 인자**를 받도록 일반화
  하거나 엔진에서 직접 산정.

**backtest.py:**
- `run_on_signals` 루프: `enter_long`/`enter_short` 둘 다 소비. 무포지션일 때
  하나라도 참이면 해당 방향 `pending_entry`(방향 기록).
- `_manage_open_bar`를 방향 분기로 재작성(위 §5 순서/조건).
- 다음 봉 시가 체결, equity 마크(`cash + 미실현`)는 기존 구조 재활용.
- 기존 `_evaluate_entry`/지연진입/분할익절 경로는 **제거 또는 미사용**.

---

## 7. 데이터 (5분봉)

- `src/data/cache.py`: `cache_path(market, symbol, timeframe)`로 **타임프레임 인자화**,
  파일명 `{symbol}_{tf}.parquet`(예: `BTC_USDT_5m.parquet`). `save/load`도 tf 전달.
- `src/data/crypto.py`: `fetch_ohlcv(..., timeframe="5m")` (이미 인자 존재, config만).
- `config/params.yaml`: `data.crypto.timeframe: "5m"`, `since`는 과거 1~2년 권장.
- 정합성 검증 `validate_ohlcv`는 그대로(UTC·중복제거·결측처리).

> **네트워크 주의:** binance가 막힌 환경에선 외부망에서 한 번 받아 parquet 캐시
> 후 주입(레포 README 참고).

---

## 8. 비용 (암호화폐 — 재활용)

| 시장 | 수수료(편도) | 슬리피지(편도) |
|---|---|---|
| 암호화폐 | 0.10%(테이커) | 0.10% |

진입가/청산가에 슬리피지 적용, 양변 수수료 차감(§6). 숏 펀딩비는 v1 미반영(주석).

---

## 9. 평가지표 (`src/metrics.py` 확장)

- 추가 상수: `BARS_PER_YEAR_CRYPTO_5M = 12*24*365  # 105120`.
- 기존(총수익률/CAGR/MDD/Sharpe/승률/실현손익비/거래수)에 **꼬리위험 지표 추가**:
  - `max_consecutive_losses`(최대 연속손실)
  - `worst_trade_ret`(단일 최대손실)
  - `tail_cvar5`(하위 5% 거래수익률 평균 ≈ CVaR)
- 리포트에서 **승률보다 MDD·연속손실·단일최대손실·CVaR를 강조**(평균회귀 꼬리위험).

---

## 10. 재활용 맵 (기존 파일 → 무엇을 바꾸나)

| 파일 | 조치 |
|---|---|
| `src/signals_core.py` | 지표/신호/진입후보 **재작성**. `_ema/_wma/_rma/rsi_wilder` 유지. `SIGNAL_COLUMNS` 교체. |
| `src/engine/position.py` | `direction` 추가, 체결/손익 양방향 일반화. |
| `src/engine/backtest.py` | `enter_long/enter_short` 소비, `_manage_open_bar` 방향분기·새 청산, 구 진입/분할익절 제거. |
| `src/data/cache.py` | `timeframe` 인자화(파일명 포함). |
| `src/data/crypto.py` | 5m 사용(인자만). |
| `src/metrics.py` | 5m 연환산 + 꼬리위험 지표. |
| `config/params.yaml` | strategy 블록을 §1 표로 교체, `timeframe: 5m`. |
| `src/run.py` | 새 전략/방향 토글 노출. |
| `tests/test_signals.py` | 다이버전스 인과·밴드터치 단위테스트로 교체. |
| `tests/test_engine.py` | **숏** 진입(다음봉 시가)·손절우선·시간청산 테스트 추가. |
| `tests/test_lookahead.py` | `SIGNAL_COLUMNS` 갱신만(하네스 그대로). |
| `tests/helpers.py` | `empty_signal_frame` 컬럼 갱신, 5m 랜덤워크. |
| `freqtrade/.../*` | (2차) `KimBollingerStrategy`로 교체, `can_short=True`, futures. |
| `scripts/*` | (2차) 비교 스크립트 방향 컬럼 반영. |

---

## 11. 미래참조 검증 (필수 게이트)

- `tests/test_lookahead.py`: 무작위 t에 대해 `add_signals(df[:t+1])`와 전체의 t시점
  `SIGNAL_COLUMNS` 값이 동일해야 함. **`bear_div/bull_div/enter_*` 포함 필수** —
  pivot이 `.shift(-)` 없이 인과적으로 확정되는지 자동 검증.
- 다이버전스 단위테스트: 합성 시계열에 의도적으로 (가격 신고가 + 이격도 저고점)
  패턴을 심고, `bear_div`가 **정확히 `pivot 형성 + k`봉에서만** True인지 확인.
- 숏 엔진 테스트: 신호 t에서 t+1 시가 숏 진입, `high>=stop` 손절 우선,
  `max_hold_bars` 시간청산 동작.

검증 절차:
```
pytest -q
python -m src.run --market crypto         # 5m, 외부망 환경(또는 parquet 주입)
# 리포트에서 MDD/최대연속손실/단일최대손실/CVaR를 승률과 함께 확인
```

---

## 12. 비범위 / 주의

- "승률 99%"는 검증 목표가 아님 — **꼬리위험(MDD/CVaR) 평가가 목적.**
- 다이버전스·삼산 정의는 정답이 아니라 **튜닝 출발점**(과최적화 경계).
- 나스닥 선물 데이터, RSI 삼산 결합 A/B, freqtrade 숏 검증, hyperopt/walk-forward,
  라이브 트레이딩은 2차.
