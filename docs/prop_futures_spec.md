# 프랍 선물 백테스트 명세 (Apex/CME)

이 프레임워크가 일반 백테스트와 다른 점, 그리고 핵심 구현 결정을 정리한다.

## 0. 무결성 원칙 (위반 시 결과 무의미)

1. 신호는 캔들 종가확정에서만. 진입 = 신호 다음 봉(t+1) 시가 체결.
2. 모든 지표·신호는 t시점까지 데이터만 사용(인과적, `.shift(-)` 금지).
   → `tests/test_lookahead.py`가 전략별 `signal_columns`로 자동 강제.
3. 손절·목표 동시 도달 시 손절 우선(고저 순서 불명 → 최악 가정).
4. 신호청산·시간청산은 종가확정 → 다음 봉 시가 체결. 손절·목표는 지정가(닿는 봉 즉시).
5. 지표는 `src/signals_core.py` 단일 구현(전략 간 드리프트 방지).

## 1. 선물 손익(P&L)

- 손익은 **방향 × (청산가−진입가) × 포인트가치 × 계약수 − 왕복수수료**.
- `point_value = tick_value / tick_size` (예: ES 0.25틱=$12.5 → $50/pt).
- 슬리피지는 틱 단위(편도), 진입·청산 모두 불리하게 적용.
- equity = 시작잔고 + 누적실현손익 + 미실현(마진은 미반영). `src/engine/pnl.FuturesPnL`.

## 2. 포지션 사이징 (config `sizing`)

- `mode: fixed` — 항상 N계약.
- `mode: risk` — 계약수 = floor(`risk_dollars` / (손절폭 × 포인트가치)), `max_contracts` 캡.
  손절폭이 너무 크면 0계약 → 진입 스킵.

## 3. 프랍 룰 평가 (핵심 차별점)

`src/prop/evaluator.evaluate_prop`이 봉별 equity를 걸어 통과/실격을 판정한다.

### 트레일링 드로다운 (Apex)
- 임계 = (계좌 고점) − `trailing_drawdown`. 고점이 오르면 임계도 따라 오른다.
- `trailing_type: intraday` → **봉 고/저 기준 미실현 포함 equity**로 추적.
  Apex 인트라데이 트레일링DD는 wick에서 터질 수 있으므로 종가만 보면 안 된다.
  → 엔진이 `equity_high`/`equity_low`(보유 미실현 최대/최소)를 봉마다 산출한다.
- `trail_lock_at` 도달 시 임계 고정(None=계속 추적).

### 기타
- **수익목표**: 누적이익 ≥ `profit_target` → 통과(소요 거래일 기록).
- **일일손실 한도**(Topstep): 세션 손익 ≤ −`daily_loss_limit` → 실격.
  세션 경계는 `reset_tz`/`reset_time`(예: 17:00 CT) 기준.
- **일관성**(Apex 30% 출금조건): 단일일 이익 / 총이익 ≤ `consistency_pct`.

## 4. 프리셋 (출발값 — 실제 약관과 대조 필요)

`src/prop/rules.py`: `apex_25k/50k/100k/150k`(인트라데이 트레일링·일관성 30%),
`topstep_50k`(EOD 트레일링·일일손실). 수치는 자주 바뀌므로 사용 전 검증할 것.

## 5. 데이터

- 기본 **합성(random-walk) 선물**(`src/data/futures.synthetic_futures`) — 프레임워크·
  프랍평가기 동작 검증 전용(성과 판정용 아님).
- 실데이터: `config/params.yaml`의 `data.futures.synthetic: false` +
  `path_template`로 CSV/parquet(OHLCV, UTC) 주입 → `load_futures`가 정합성 검증.

## 6. 검증

```bash
pytest -q          # 지표 / lookahead(전략별) / 선물P&L(틱산수) / 숏엔진 / 프랍평가기
python -m src.compare
```
- 미래참조 게이트: 새 전략도 `test_lookahead`를 통과해야 안전.
- 프랍 sanity: 합성 equity로 트레일링DD/일일손실/수익목표/일관성 케이스를
  의도적으로 만들어 사유·시점 정확성을 단위테스트로 확인(`tests/test_prop.py`).

## 7. 다음 차 (비범위)

실데이터 성과판정, Topstep 정밀화, hyperopt/walk-forward, 라이브 자동매매
(브로커 API·주문집행), 롤오버/연속물 스티칭, 세션(RTH/ETH) 정밀 모델.
