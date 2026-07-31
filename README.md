# MACD + RSI 모멘텀 전략 — 1차 백테스트 (전략 A + 모멘텀스코프)

Ross Cameron 스타일로 포장된 MACD+RSI 1시간봉 모멘텀 전략의 **신호 작동 검증용**
백테스트. 유니버스를 넓히기 전에 단일 대장주(BTC/USDT, AAPL)로 전략 신호 자체가
작동하는지부터 검증한다. 상세 규칙은 `strategy_spec_v2.md`, 아키텍처는
`implementation_architecture.md` 참조.

> 📐 **전체 설계 구조도(레이어·전략계약·검증 파이프라인 mermaid): [`docs/architecture.md`](docs/architecture.md)**

## 설계 핵심

- **두뇌는 하나(`src/signals_core.py`)** — 지표(MACD/Wilder RSI/HMA/거래량MA)와
  캔들단위 불리언 신호를 순수·벡터화로 계산. 미래참조 0(`test_lookahead`로 강제).
- **상태 전이는 엔진(`src/engine/`)** — 지연진입 대기·폐기, 분할익절→본전스탑→
  잔량청산은 상태머신이라 엔진이 보유.
- **무결성 규율(spec §0)** — 신호는 종가 확정에서만, 진입/신호청산은 다음 봉 시가
  체결, 손절·목표 동시도달 시 손절 우선, 비용(수수료·슬리피지) 반영.

## 구조

```
config/params.yaml      파라미터(§1) + 시장별 비용(§5) + 데이터 설정
src/signals_core.py     [순수] 지표 + 신호 + 진입후보(enter_long) — 단일 두뇌
src/engine/position.py  포지션 상태머신 + 체결/비용 헬퍼
src/engine/backtest.py  이벤트 루프 (run_backtest / run_on_signals)
src/data/{crypto,stocks,cache}.py   ccxt / yfinance 수집 + parquet 캐시
src/metrics.py          CAGR/MDD/Sharpe/승률/실현손익비 등
src/strategies/         [카테고리별 전략 레지스트리] momentum/mean_reversion/breakout + _template.py
src/validation/         [오버피팅 방어] 워크포워드·purged K-fold·CSCV PBO·Deflated Sharpe
src/run.py              단일 백테스트 CLI 진입점
src/validate.py         오버피팅 검증 CLI (전략/카테고리별 스윕 → PBO/DSR/워크포워드 판정카드)
src/strategy_check.py   [전략 합격 판정 하네스] 무결성 게이트 + 성과 판정 (임의 .py 로드)
tests/                  lookahead / signals / engine / validation / strategies / harness 검증
freqtrade/              독립 검증용 전략·설정 (signals_core 공유 → freqtrade/README.md)
scripts/                parquet→freqtrade 변환, 자체엔진 vs freqtrade 교차비교
```

## 카테고리별 전략 (플러그인)

전략은 카테고리(momentum/mean_reversion/breakout/…)에 속하는 자족 모듈이며, 동일한
계약(lookahead-safe `enter_long` + 청산 컬럼 산출)을 따르므로 **어떤 전략이든 같은
엔진·검증 레이어로 돌아간다**. 새 전략은 `src/strategies/`에 모듈 하나 추가하고
`register()`하면 즉시 스윕·검증 대상이 된다.

```bash
python -m src.validate --list                       # 등록된 카테고리·전략 목록
python -m src.validate --strategy breakout_donchian --synthetic
python -m src.validate --category mean_reversion --synthetic   # 카테고리 전체 검증
```

- **momentum** — `momentum_macd_rsi`: MACD 골든크로스 + Wilder RSI (기존 두뇌 이식).
- **mean_reversion** — `mean_reversion_bollinger`: 볼린저 하단 이탈 후 복귀, 중심선 회귀 청산.
- **breakout** — `breakout_donchian`: 직전 N봉 고점 돌파, M봉 저점 이탈 청산(터틀 계열).

각 전략은 자체 `default_params`·`param_grid`(스윕 범위)를 갖고, 엔진 공통 손절/목표
(swing-low stop + reward_ratio)를 재사용한다. `tests/test_strategies.py`가 모든
전략에 대해 미래참조 0·엔진연동·PBO 산출을 자동으로 강제한다.

### 새 전략 가져와 검증하기 (합격 판정 하네스)

전략을 하나 만들면 `strategy_check`가 **한 번의 명령으로** 무결성 게이트(신호 계약 →
미래참조 0 → 결정성 → 엔진 연동 → 스윕 가능)를 통과하는지 판정하고, 이어서 오버피팅
성과 판정카드를 낸다. 레지스트리 등록 없이 **임의의 .py 파일도** 바로 검증된다.

```bash
# 1) 템플릿 복사 후 generate_signals 구현
cp src/strategies/_template.py src/strategies/my_strategy.py

# 2) 가져온 전략을 그 자리에서 검증 (등록 불필요)
python -m src.strategy_check --module src/strategies/my_strategy.py --synthetic
python -m src.strategy_check --strategy breakout_donchian --synthetic   # 등록된 전략도 동일
python -m src.strategy_check --category momentum --no-perf              # 무결성 게이트만

# 3) 무결성 통과하면 src/strategies/__init__.py에 import 추가 → 정식 등록
```

무결성 게이트가 하나라도 FAIL이면 exit code 1(성과 판정은 생략). 특히 **미래참조 0**
게이트는 t시점 절단 재계산으로 미래 데이터 누수를 잡는다 —
`tests/test_strategy_check.py`가 일부러 누수를 넣은 전략이 FAIL로 잡히는지 검증한다.

## 일반인용 대시보드 (Streamlit)

클릭만으로 전략을 고르거나 `.py`를 업로드해 무결성 게이트·오버피팅 판정·수익곡선을
본다. 계산은 전부 `src/analysis.py`(테스트되는 순수 API)에 위임하고 `app.py`는
렌더링만 한다.

```bash
pip install -e ".[ui]"
streamlit run app.py
```

- 사이드바: 카테고리→전략 선택(또는 `.py` 업로드), 시장·사이징·리스크·CSCV/폴드 조절.
- 본문: ① 무결성 게이트(신호등) → ② 성과 판정(PBO/DSR/감쇠) → ③ **최우수 설정 vs
  Buy & Hold** 수익곡선 + 지표. 벤치마크를 못 이기면 바로 보인다.

## 포지션 사이징 (리스크 기반)

전액 올인(`position_pct=1.0`) 대신 **거래당 리스크를 일정하게** 유지하는 기관식
사이징을 지원한다. 엔진은 사이저 하나를 받아 수량만 다르게 잡으며, 미지정 시 기존
전액 사이징과 **완전 동일**(회귀 안전망 `tests/test_sizing.py`).

- `fixed_fraction` (기본) — 자본의 `position_pct`를 명목으로.
- `fixed_risk` — 손절까지 닿았을 때 손실이 자본의 `risk_pct`(기본 1%)가 되도록.
  단위리스크(진입가−손절가)가 작을수록 크게, 클수록 작게. 명목 상한은 `position_pct`.

```bash
python -m src.validate --strategy breakout_donchian --synthetic \
    --sizing fixed_risk --risk-pct 0.01           # 거래당 1% 리스크
```

config `backtest.sizing`/`backtest.risk_pct`로 기본값을, CLI `--sizing/--risk-pct`로
덮어쓴다. 새 사이저는 `src/engine/sizing.py`의 레지스트리에 함수 하나 추가하면 된다.

## CI

`.github/workflows/tests.yml` — push/PR마다 3개 파이썬 버전에서 전체 테스트(미래참조
게이트 포함) + pyflakes + 검증 CLI 스모크를 자동 실행한다.

## 실데이터 넣기 (파일 인제스트)

이 환경은 외부 시세가 차단돼 자동 다운로드가 안 된다. 대신 **어디서든 받은 파일을
떨어뜨리면** 표준 OHLCV로 흡수해 그대로 검증한다(`src/data/ingest.py`). 거래소 CSV
(밀리초 에폭·대문자 컬럼), yfinance 내보내기(Date 문자열), parquet 캐시 등 다양한
관례를 자동 판별한다(시간열 이름·에폭 단위·컬럼 대소문자, volume 없으면 0).

```bash
python -m src.validate --strategy breakout_donchian --csv path/to/BTCUSDT.csv
python -m src.strategy_check --module my_strategy.py --csv path/to/AAPL.csv
python -m src.portfolio --strategy breakout_donchian --data-dir path/to/universe/
python -m src.portfolio --strategy breakout_donchian --data a.csv b.csv c.parquet
```

지원 형식: `.csv/.tsv/.parquet/.feather`. 스키마 검증은 기존 `validate_ohlcv`(UTC통일·
중복제거·정렬·결측처리)로 통일된다. `data/<market>/<symbol>_1h.parquet` 캐시에 넣으면
`--market`만으로도 잡힌다.

## 다자산 포트폴리오

한 전략을 여러 종목에 배분해 합산 성과·**분산효과**를 본다. 종목 간 상관이 낮을수록
변동이 상쇄돼 위험대비수익(Sharpe)이 개선된다. 각 슬리브는 독립 엔진으로 돌고,
자산곡선을 합쳐 포트폴리오 지표·상관행렬·분산효과를 낸다.

```bash
python -m src.portfolio --strategy breakout_donchian --symbols 4 --synthetic \
    --sizing fixed_risk --risk-pct 0.01
```

산출: 포트폴리오 총수익/CAGR/MDD/Sharpe, 종목별 성과, 평균 상관계수, 포트폴리오
Sharpe vs 평균 종목 Sharpe(분산효과). `run_portfolio(spec, {symbol: df}, ...)`로 코드에서도 호출.

## 신뢰구간 (블록 부트스트랩)

단일 백테스트는 점추정 하나(Sharpe 1.8 등)만 준다. 그게 우연인지 견고한지는 알 수
없다. 봉수익률을 **블록 단위로 재표집**(자기상관 보존)해 지표 분포를 만들고 신뢰구간과
손익확률을 낸다. **구간이 0을 가로지르면 그 성과는 견고하지 않다**(예: Sharpe 3.5인데
95% 구간이 [-1.4, 7.3]이면 신뢰 불가). `validate`/`strategy_check` 판정카드와 UI에
자동 표시되고, `--no-ci`로 끄고 `--resamples`로 횟수를 조절한다.

## 오버피팅 방어 (기관급 검증)

단일 백테스트가 "좋아 보이는 것"이 우연/과최적화인지 판정한다. 여러 파라미터 설정을
스윕한 뒤, 다중검정 편향까지 반영해 세 축으로 신호등(신뢰/주의/기각)을 낸다. 위의
어떤 카테고리 전략에도 `--strategy`/`--category`로 동일하게 적용된다.

```bash
python -m src.validate --synthetic --bars 3000     # 기본 전략(momentum) 데모(네트워크 불필요)
python -m src.validate --market crypto              # 캐시 있으면 실데이터, 없으면 합성 폴백
python -m src.validate --market stock --blocks 12 --folds 6 --wf-mode anchored
```

- **PBO (`src/validation/pbo.py`)** — CSCV(조합대칭교차검증). IS 최우수 설정이 OOS
  중앙값 아래로 떨어지는 확률. 0.5면 무작위(순전한 과최적화), 낮을수록 견고.
- **Deflated Sharpe** — N번 시도의 우연 문턱을 관측 Sharpe가 넘는지(왜도·첨도 보정).
  0.95↑ 유의. `expected_max_sharpe_ratio`가 다중검정 문턱 SR₀를 산출.
- **워크포워드 (`src/validation/{splits,sweep}.py`)** — anchored·rolling 분할로 학습→
  검증 재최적화, IS→OOS 성능감쇠 측정. purged K-fold(embargo·horizon)도 제공.

검증 레이어는 기존 lookahead-safe 엔진을 그대로 소비하며(`run_sweep`의 `_run_one`이
유일한 백엔드 교체 지점 — vectorbt 등 고속 스윕 백엔드를 여기에 꽂을 수 있다),
`tests/test_validation.py`가 분할 인과성·PBO 수렴·DSR 다중검정 성질을 회귀 고정한다.

## 설치 & 실행

```bash
pip install -e .            # 또는: pip install pandas numpy pyyaml pyarrow ccxt yfinance
pytest -q                   # 13개 테스트 (lookahead 통과 = 신호 신뢰성 게이트)

python -m src.run --market crypto   # BTC/USDT 1h (ccxt)
python -m src.run --market stock    # AAPL 60m (yfinance)
```

## freqtrade 독립 검증

자체 엔진과 **같은 두뇌(`signals_core`)**를 쓰는 freqtrade 전략으로 미래참조를
독립 검증한다. `signals_core.compute_entry_candidates`의 `enter_long`을 자체 엔진과
freqtrade가 함께 소비하므로 진입이 일치한다. 절차·판정기준은 `freqtrade/README.md`.

```bash
python scripts/to_freqtrade_data.py                         # parquet → feather
freqtrade backtesting       -c freqtrade/config.json -s RossMacdRsiStrategy
freqtrade lookahead-analysis -c freqtrade/config.json -s RossMacdRsiStrategy  # biased 0 확인
python scripts/compare_engines.py                           # 진입 Jaccard ≈ 1.0
```

## 데이터 수집 / 네트워크 주의

`--market` 실행 시 `data/{market}/{symbol}_1h.parquet` 캐시가 있으면 그것을 우선
사용하고, 없으면 다운로드 후 캐시한다.

> **샌드박스/제한 네트워크 주의:** 일부 실행환경은 네트워크 정책으로 `api.binance.com`,
> Yahoo Finance 등 외부 시세 호스트를 차단(403)한다. 이 경우 자동 다운로드가 실패한다.
> 해결책:
> 1. 외부망이 열린 로컬/환경에서 한 번 받아 `data/<market>/<symbol>_1h.parquet`로
>    캐시하면, 이후 제한 환경에서도 그 캐시로 백테스트가 돌아간다.
> 2. 또는 동일 스키마(UTC tz-aware `time` 인덱스 + open/high/low/close/volume)의
>    parquet을 직접 그 경로에 떨어뜨리면 된다. `src.data.cache.validate_ohlcv`가
>    정합성(중복제거·UTC통일·결측처리)을 검증한다.

## 검증된 것 / 남은 것

- ✅ `signals_core` 미래참조 0, 지표/필터 정확성, 진입후보 추출 회귀안전망 (`pytest`).
- ✅ 엔진 체결 규율: 다음 봉 시가 진입, 손절 우선, 분할익절+본전스탑.
- ✅ 합성(random-walk) 데이터 엔드투엔드: 비용 차감 후 손실(랜덤워크 sanity).
- ✅ freqtrade 독립 검증 산출물(전략·설정·변환기·교차비교) 완성 — 실행은 외부망 환경.
- ⏳ 실데이터(BTC/AAPL) 성과 판정 + freqtrade lookahead-analysis — 외부망 환경에서 수행.

## 비범위 (2차 백로그)

`hist_turn_up` 진입결합 A/B, 유니버스 확장(생존편향·호출제한), hyperopt,
freqtrade 암호화폐 라이브, 포지션 사이징(변동성 타겟팅·켈리), 한국주식,
vectorbt 고속 스윕 백엔드, 일반인용 Streamlit UI.
(✅ 완료: 카테고리별 전략 레지스트리 — `src/strategies/` · grid search 스윕 +
 walk-forward/OOS + 오버피팅 방어 — `src/validation/`)
