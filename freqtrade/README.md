# freqtrade 독립 검증

자체 백테스트 엔진(`src/engine`)과 **같은 두뇌(`src/signals_core.py`)**를 공유하는
freqtrade 전략으로, 미래참조·재귀편향을 독립적으로 검증한다.

## 왜 freqtrade인가

자체 엔진이 미래참조 버그를 갖고 있으면 자체 테스트만으로는 못 잡을 수 있다.
freqtrade는 성숙한 독립 엔진이고 **내장 무결성 검사**(`lookahead-analysis`,
`recursive-analysis`)를 제공한다. 진입신호(`enter_long`)를 signals_core에서 그대로
가져오므로, 두 엔진의 진입이 일치하는지도 교차 비교할 수 있다.

> **무결성 우선 스코프:** 청산(손절/목표 분할익절/데드크로스/RSI)은 합리적 수준으로
> 재현하지만, 두 엔진의 체결모델 차이(슬리피지·봉내 체결)로 **정확한 P&L 일치는
> 목표가 아니다.** 핵심 점검은 ①미래참조 0 ②진입 타임스탬프 일치다.

## 구성

```
freqtrade/
├── config.json                              # binance / BTC/USDT / 1h / fee 0.001
└── user_data/
    ├── strategies/RossMacdRsiStrategy.py     # signals_core 공유 IStrategy v3
    └── data/binance/BTC_USDT-1h.feather      # 변환기로 생성(아래)
```

전략 매핑:
- **진입**: `populate_indicators`가 `signals_core.add_signals` 호출 → `enter_long`/
  `entry_kind`를 그대로 freqtrade 진입신호로 사용.
- **손절**: `custom_stoploss` — 진입 봉 직전 M봉 스윙로우 기반(자체 엔진과 동일 산식),
  1차 익절 후 본전으로 상향.
- **목표/분할익절**: `adjust_trade_position` — `current_rate >= target`에서 50% 축소.
- **데드크로스 전량청산**: `populate_exit_trend`의 `exit_long`.
- **HALF 상태 RSI<50**: `custom_exit`.

## 실행 절차

freqtrade 설치 + BTC/USDT 1h 데이터가 있는 환경에서:

```bash
pip install freqtrade

# 1) 데이터 준비 — 둘 중 하나
#   (a) 자체 캐시 재사용(권장, 자체엔진과 동일 봉):
python -m src.run --market crypto          # 외부망에서 parquet 생성
python scripts/to_freqtrade_data.py        # parquet → feather 변환
#   (b) 또는 freqtrade로 직접 다운로드:
# freqtrade download-data -c freqtrade/config.json -t 1h --timerange 20200101-

# 2) 백테스트
freqtrade backtesting -c freqtrade/config.json -s RossMacdRsiStrategy --timeframe 1h

# 3) 독립 무결성 검사 (가장 중요)
freqtrade lookahead-analysis  -c freqtrade/config.json -s RossMacdRsiStrategy
freqtrade recursive-analysis  -c freqtrade/config.json -s RossMacdRsiStrategy

# 4) 자체 엔진과 진입 교차 비교
python scripts/compare_engines.py
```

## 판정 기준

- `lookahead-analysis` → **biased indicators 0** (미래참조 없음).
- `recursive-analysis` → 지표 startup 안정(재귀편향 없음).
- `compare_engines.py` → 진입 **Jaccard 일치율 ≈ 1.0** (enter_long 공유).
- 총수익률/승률은 방향성만 비교(정확 일치는 비목표).

## 환경 제약

이 레포가 만들어진 샌드박스는 네트워크 정책으로 `api.binance.com`을 차단(403)하고
freqtrade 설치도 무겁다. 따라서 위 명령들은 시세 접근이 가능한 환경에서 실행한다.
전략·설정·변환기·비교 스크립트는 그 환경에서 바로 동작하도록 완성돼 있다.
