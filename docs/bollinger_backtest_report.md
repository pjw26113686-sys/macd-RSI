# BollingerOpenBreakout — 백테스트·무결성 검증 & 고도화 리포트

작성: 자동매매화(고도화) 1차. 대상 전략 `BollingerOpenBreakout`(MVP, 5m 실행 +
1d 방향성) 을 freqtrade 에서 백테스트하고, 발견한 버그를 고친 `BollingerOpenBreakoutV2`
를 만들어 무결성(룩어헤드/재귀편향)까지 검증한 기록.

> ⚠️ **성과 수치는 무의미하다.** 이 샌드박스는 binance 등 모든 시세 호스트를 403 으로
> 막아 실데이터를 못 받는다. 그래서 **합성(synthetic) 데이터**로 돌렸고, 목적은
> 오로지 ①전략이 에러 없이 도는가 ②미래참조/재귀편향이 없는가 ③거래빈도·청산구조가
> 의도대로인가 의 **메커니즘·무결성 검증**이다. 수익률·승률·샤프는 참고조차 하지 말 것.

---

## 1. 실행 환경 (오프라인 우회)

| 항목 | 내용 |
|---|---|
| freqtrade | 2026.5.1 (`.venv` 가상환경, ta-lib manylinux 휠) |
| 데이터 | `scripts/make_synth_futures_data.py` — 5m GBM(팻테일 Student-t + 변동성군집 + 청산윅), 1d 는 5m 리샘플(정합). 8개 페어 × 1400일. |
| 마켓 로딩 | `scripts/offline_freqtrade.py` — binance 마켓 API 가 막혀 `Exchange.reload_markets` 를 합성 마켓 dict 으로 몽키패치(레버리지 티어는 freqtrade 번들 json 사용). |
| 설정 | `freqtrade/config_bollinger.json` — futures/isolated, 5m, fee 0.0005, 8 페어. |

실행 예:

```bash
python3 -m venv .venv && .venv/bin/pip install freqtrade ta-lib
.venv/bin/python scripts/make_synth_futures_data.py
.venv/bin/python scripts/offline_freqtrade.py backtesting        -c freqtrade/config_bollinger.json -s BollingerOpenBreakoutV2 --timerange 20240120-
.venv/bin/python scripts/offline_freqtrade.py lookahead-analysis -c freqtrade/config_bollinger.json -s BollingerOpenBreakoutV2 --timerange 20240120-20241231 --minimum-trade-amount 10
.venv/bin/python scripts/offline_freqtrade.py recursive-analysis -c freqtrade/config_bollinger.json -s BollingerOpenBreakoutV2 --timerange 20240201-20240401
```

> 실데이터 환경(외부망 허용)에서는 오프라인 셔임 없이 표준 절차로 돌리면 된다:
> `freqtrade download-data --trading-mode futures --timeframes 5m 1d -p BTC/USDT:USDT ... --days 365`
> 후 `freqtrade backtesting -c ... -s BollingerOpenBreakoutV2`.

---

## 2. 무결성 검증 결과 (핵심)

| 전략 | lookahead-analysis | recursive-analysis |
|---|---|---|
| BollingerOpenBreakout (V1) | **has_bias = No** (signals 11, biased 0/0/0) | 안정 (vol_sma 0.000%) |
| BollingerOpenBreakoutV2 | **has_bias = No** (signals 20, biased 0/0/0) | 안정 (bb_upper_1d 0.000%, startup 60 충분) |

- 두 버전 모두 **미래참조 없음**. V2 가 도입한 "당일 시가(5m 네이티브)" 와 "앵커드
  VWAP(당일 누적)" 도 룩어헤드를 만들지 않음을 freqtrade 가 독립 확인.
- 일봉 지표는 `get_pair_dataframe` 가 전체 일봉을 로드하므로 startup_candle_count 에
  무관하게 재귀편향이 없다(그래서 V2 는 startup 을 60 으로 낮춰도 안전).

---

## 3. V1 에서 발견한 문제

### B1. `day_open` 이 "당일"이 아니라 "전일" 시가였다 (의도 불일치)
V1 은 `inf["day_open"]=inf["open"]` 을 `merge_informative_pair` 로 5m 에 붙인다.
merge 는 룩어헤드 방지를 위해 일봉을 **한 칸 미루므로**, 당일(D) 5m 행에는 직전 완성
일봉(=어제 D-1)의 시가가 들어온다. 즉 "**전일 시가 돌파**"가 되어 설계 의도("당일
일봉 시가 돌파")와 다르다. 당일 시가는 그날 00:00 에 이미 확정돼 **사용해도
룩어헤드가 아니다.**

### B2. 방향성 신호 off-by-one (실제로는 '그제' 기준)
V1 은 일봉 내부에서 `shift(1)`(전일) 로 방향성을 만든 뒤, merge 가 다시 한 칸
미뤄 최종적으로 **D-2(그제)** 기준이 된다. "전일 반전을 본다"는 의도와 어긋난다.

### B3. 과도하게 쌓인 AND 필터 → 사실상 거래 0
`방향성(매우 희소) AND 정확한 시가돌파 AND 거래량 1.5× AND VWAP` 가 곱연산으로
붙어, 8페어 × 1400일에서도 트레이드가 거의 안 나온다(아래 표). 거래량 1.5× 가
돌파 캔들에서 특히 자주 탈락시킨다. 자동매매로 쓰기엔 표본이 너무 적어 검증·운용
모두 불가.

| | V1 | V2 |
|---|---|---|
| 거래수 (8페어, 2024-01-20~) | **수 건(~3~19)** | **132** |
| 거래빈도 | 검증 불가 수준 | ~0.1 trade/일 (합성 기준; 실데이터에선 더 많을 것) |

---

## 4. V2 변경점 (`BollingerOpenBreakoutV2.py`)

- **B1 수정**: 당일 시가 = 그날 첫 5m 의 `open`(5m 네이티브 groupby). 룩어헤드 아님.
- **B2 수정**: 일봉 캔들 자체에서 터치 플래그 계산 → merge 의 1틱 시프트만 적용해
  '전일(D-1)' 기준으로 정렬.
- **방향성 lookback(N일)**: '전일만'이 아니라 '최근 N일 내 밴드터치-반전' 허용
  (`dir_lookback`, 하이퍼옵트).
- **하이퍼옵트 파라미터화**: `bb_std`, `dir_lookback`, `use_vol_filter`/`vol_mult`/
  `vol_lookback`, `use_vwap_filter`, `exit_on_vwap`. 과한 필터를 데이터로 켜고/끄고
  튜닝 가능.
- **실청산 추가**: 돌파 실패(종가 VWAP 되돌림) 신호청산 + 트레일링스톱 + ROI + 고정손절.
- **VWAP 앵커**: 당일 00:00 앵커, 5m 네이티브 일자 그룹(전보다 견고).

---

## 5. V2 백테스트 관찰 (합성 — 성과 무의미, 구조만 본다)

청산 사유별 분해(전체 132 트레이드):

| 청산 사유 | 건수 | 승률 | 해석 |
|---|---|---|---|
| trailing_stop_loss | 16 | 100% | 추세 탄 승자 |
| roi | 11 | 100% | 목표 익절 |
| stop_loss | 6 | 0% | 고정손절 |
| **vwap_revert** | **99** | **12%** | **노이즈 휩쏘 — 순손실 주범** |

**발견**: 5m 에서 가격이 VWAP 주변을 진동해 `vwap_revert` 신호청산이 너무 자주
터지고(99/132), 작은 손실로 추세를 미리 끊는다. A/B:

| 설정 | Profit factor | Total profit % |
|---|---|---|
| `exit_on_vwap=True`(기본) | 1.02 | 0.16% |
| `exit_on_vwap=False` | **1.28** | **4.11%** |

→ 합성/엣지없는 데이터에서도 휩쏘 비용이 확인된다. **하이퍼옵트 1순위 타깃**:
`exit_on_vwap` 끄거나(또는 VWAP 이탈에 버퍼/지속봉수 조건 추가), 트레일링·ROI 중심
청산으로 전환. (기본값은 보수적으로 True 유지 — 실데이터에선 추세실패 방어가 될 수
있어 hyperopt 로 판단 권장.)

---

## 6. 자동매매(라이브)까지 로드맵

1. **실데이터 확보** — 외부망 환경에서 `download-data`(5m+1d, 1~3년, 다수 페어).
   합성은 여기까지의 *배관(plumbing)* 검증용일 뿐.
2. **하이퍼옵트** — `--spaces buy sell roi stoploss trailing`, 손익비/거래빈도 균형을
   목적함수로(예: `SharpeHyperOptLoss` 또는 커스텀). 우선 `exit_on_vwap`, `vol_mult`,
   `dir_lookback`, `bb_std` 탐색.
3. **워크포워드 / OOS** — `--timerange` 분할로 인샘플 튜닝 → 아웃오브샘플 검증.
   과최적화 방지(파라미터 안정성·플래토 확인).
4. **비용·펀딩 현실화** — 실제 fee/슬리피지/펀딩비 반영(합성은 펀딩 0). 선물 펀딩이
   보유시간 대비 손익에 미치는 영향 점검.
5. **리스크·사이징** — 포지션 사이징, `max_open_trades`, 페어 상관/동시노출 한도,
   레버리지(현재 1배 고정 → 손절폭 기반 사이징과 함께 설계).
6. **dry-run(페이퍼)** — 실거래소 웹소켓으로 수일~수주 무현금 운용, 백테스트와 체결·
   슬리피지 괴리 점검.
7. **소액 라이브 → 점진 확대** — 모니터링/알림, 비상정지, API 키 권한 최소화.

> 매 단계에서 lookahead-analysis / recursive-analysis 를 재실행해 무결성을 회귀
> 검증한다(파라미터·로직 변경이 미래참조를 새로 만들 수 있음).
