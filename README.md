# MACD + RSI 모멘텀 전략 — 1차 백테스트 (전략 A + 모멘텀스코프)

Ross Cameron 스타일로 포장된 MACD+RSI 1시간봉 모멘텀 전략의 **신호 작동 검증용**
백테스트. 유니버스를 넓히기 전에 단일 대장주(BTC/USDT, AAPL)로 전략 신호 자체가
작동하는지부터 검증한다. 상세 규칙은 `strategy_spec_v2.md`, 아키텍처는
`implementation_architecture.md` 참조.

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
src/signals_core.py     [순수] 지표 + 신호 컬럼
src/engine/position.py  포지션 상태머신 + 체결/비용 헬퍼
src/engine/backtest.py  이벤트 루프 (run_backtest / run_on_signals)
src/data/{crypto,stocks,cache}.py   ccxt / yfinance 수집 + parquet 캐시
src/metrics.py          CAGR/MDD/Sharpe/승률/실현손익비 등
src/run.py              CLI 진입점
tests/                  lookahead / signals / engine 검증
```

## 설치 & 실행

```bash
pip install -e .            # 또는: pip install pandas numpy pyyaml pyarrow ccxt yfinance
pytest -q                   # 12개 테스트 (lookahead 통과 = 신호 신뢰성 게이트)

python -m src.run --market crypto   # BTC/USDT 1h (ccxt)
python -m src.run --market stock    # AAPL 60m (yfinance)
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

- ✅ `signals_core` 미래참조 0, 지표/필터 정확성 (`pytest`).
- ✅ 엔진 체결 규율: 다음 봉 시가 진입, 손절 우선, 분할익절+본전스탑.
- ✅ 합성(random-walk) 데이터 엔드투엔드: 비용 차감 후 손실(랜덤워크 sanity).
- ⏳ 실데이터(BTC/AAPL) 성과 판정 — 위 네트워크 제약으로 외부망 환경에서 수행 필요.

## 비범위 (2차 백로그)

전략 B(RSI 다이버전스), `hist_turn_up` 진입결합 A/B, 유니버스 확장(생존편향·호출제한),
grid search/hyperopt, walk-forward/OOS, freqtrade 암호화폐 라이브, 포지션 사이징,
한국주식.
