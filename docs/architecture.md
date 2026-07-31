# 설계 구조도

퀀트 전략 실험 플랫폼의 아키텍처. 원칙은 하나: **두뇌(신호)·근육(엔진)·심판(검증)을
분리**해, 일반인은 얇은 툴 층만 만지고 기관급 엄밀성은 안쪽에 묻어둔다.

핵심 규율(전 레이어 공통): **미래참조 0**. 모든 신호는 t시점까지의 데이터만 쓰며,
진입/신호청산은 다음 봉 시가에 체결된다.

---

## 1. 레이어 구조 (전체 조감)

```mermaid
flowchart TB
  subgraph DATA["① 데이터 레이어 · src/data"]
    D1["ccxt / yfinance 수집"] --> D2["parquet 캐시<br/>validate_ohlcv (UTC·중복제거)"]
    D3["합성 데이터<br/>synthetic_ohlcv (네트워크 불필요)"]
  end

  subgraph STRAT["② 전략 레이어 (플러그인) · src/strategies"]
    B["base.py<br/>StrategySpec + 카테고리 레지스트리"]
    S1["momentum_macd_rsi<br/>← signals_core 단일두뇌"]
    S2["mean_reversion_bollinger"]
    S3["breakout_donchian"]
    T["_template.py (복붙 시작점)"]
  end

  subgraph ENGINE["③ 엔진 레이어 (lookahead-safe) · src/engine"]
    E1["backtest.py<br/>이벤트 루프 · 워밍업·체결규율"]
    E2["position.py<br/>상태머신 FULL/HALF · 비용·슬리피지"]
  end

  subgraph METRIC["④ 성과지표 · src/metrics.py"]
    M["CAGR · MDD · Sharpe · 승률 · 실현손익비"]
  end

  subgraph VALID["⑤ 오버피팅 방어 레이어 · src/validation"]
    V1["splits.py<br/>워크포워드 · purged K-fold"]
    V2["sweep.py<br/>파라미터 스윕 · 워크포워드 분석"]
    V3["pbo.py<br/>CSCV PBO · Deflated Sharpe"]
    V4["report.py<br/>신뢰/주의/기각 판정카드"]
  end

  subgraph TOOLS["⑥ 툴 / CLI · src"]
    C1["run.py<br/>단일 백테스트"]
    C2["validate.py<br/>검증 CLI (전략/카테고리별)"]
    C3["strategy_check.py<br/>합격 판정 하네스"]
  end

  subgraph VERIFY["⑦ 독립 검증 · freqtrade/ · scripts/"]
    F["freqtrade 전략<br/>같은 signals_core 공유 → 교차대조"]
  end

  DATA --> ENGINE
  STRAT --> ENGINE
  ENGINE --> METRIC
  METRIC --> VALID
  STRAT --> VALID
  VALID --> TOOLS
  ENGINE --> TOOLS
  STRAT -. 신호 공유 .-> VERIFY
```

레이어 간 의존은 **한 방향**(위→아래로만 호출)이다. 전략은 엔진을 모르고, 엔진은
검증을 모른다. 검증·툴 층이 전략과 엔진을 조립해 돌린다.

---

## 2. 전략 플러그인 계약

모든 전략은 카테고리에 속하는 자족 모듈이며, 동일 계약을 따르므로 **무엇이든 같은
엔진·검증 레이어로 돌아간다**. 새 전략 = 모듈 1개 + `register()`.

```mermaid
flowchart LR
  IN["OHLCV df<br/>(time 인덱스) + params"] --> GS["generate_signals(df, params)<br/>순수·미래참조0"]
  GS --> C1["enter_long<br/>진입 후보"]
  GS --> C2["exit_col<br/>전량 신호청산"]
  GS --> C3["exit_half_col<br/>잔량 신호청산"]

  subgraph SPEC["StrategySpec (메타)"]
    m1["name · category"]
    m2["default_params"]
    m3["param_grid — 스윕 범위"]
    m4["warmup_bars(params)"]
    m5["exit_col · exit_half_col"]
  end

  C1 --> ENG["엔진<br/>다음 봉 시가 진입/청산"]
  C2 --> ENG
  C3 --> ENG
  SPEC -. 엔진이 참조 .-> ENG
  ENG -. 공통 손절/목표 .-> STOP["swing-low stop<br/>+ reward_ratio (엔진 관리)"]
```

손절/목표는 전략이 아니라 **엔진이 공통 관리**한다(전략은 진입·청산 신호에만 집중).

---

## 3. 합격 판정 하네스 파이프라인 (핵심 흐름)

새 전략을 가져오면(레지스트리 등록 무관, 임의 `.py` 포함) 한 번에 돌아가는 판정 흐름.

```mermaid
flowchart TB
  A["새 전략<br/>--module *.py · --strategy · --category"] --> B["스펙 로드<br/>load_spec_from_module"]
  B --> GATE["무결성 게이트 (하드 통과조건)"]

  GATE --> G1["1 신호 계약<br/>컬럼·bool·인덱스 보존"]
  GATE --> G2["2 미래참조 0<br/>t시점 절단 재계산 일치"]
  GATE --> G3["3 결정성<br/>동일입력 동일출력"]
  GATE --> G4["4 엔진 연동<br/>자산 유한·양수·거래발생"]
  GATE --> G5["5 스윕 가능<br/>param_grid 충분"]

  G1 --> D{"모두 PASS?"}
  G2 --> D
  G3 --> D
  G4 --> D
  G5 --> D

  D -->|"하나라도 FAIL"| X["⛔ 불합격 · exit 1<br/>(성과 판정 생략)"]
  D -->|"모두 PASS"| P["성과 판정"]

  P --> P0["run_sweep<br/>N개 설정 백테스트 → T×N 수익률행렬"]
  P0 --> P1["cscv_pbo<br/>과최적화 확률 PBO"]
  P0 --> P2["deflated_sharpe<br/>다중검정·비정규성 보정 DSR"]
  P0 --> P3["walk_forward_analysis<br/>IS→OOS 성능감쇠"]

  P1 --> R["판정카드<br/>✅신뢰 / ⚠️주의 / ⛔기각"]
  P2 --> R
  P3 --> R
```

**무결성(규칙 위반 없음)** 과 **성과(돈이 되는가)** 를 분리한다. 무결성 통과 + 성과
기각이면 "정직하지만 안 좋은 전략"으로, 무결성 실패면 "신뢰 불가"로 판정된다.

---

## 4. 엔진 체결 규율 (무결성 §0)

```mermaid
flowchart LR
  T0["봉 t 종가 확정<br/>신호 평가"] --> T1["봉 t+1 시가<br/>진입/신호청산 체결"]
  subgraph INTRA["보유 중 봉내 관리"]
    K1["손절 도달?"] -->|"예(최우선)"| KX["STOP 청산"]
    K1 -->|"아니오"| K2["목표 도달?"]
    K2 -->|"예"| K3["50% 분할익절<br/>+ 본전스탑 이동"]
    K2 -->|"아니오"| K4["청산신호?<br/>exit_col / exit_half_col"]
  end
  T1 --> INTRA
```

- 신호는 종가 확정에서만, 체결은 다음 봉 시가 → **미래참조 방지**.
- 손절·목표 동시 도달 시 **손절 우선**(최악 가정).
- 수수료·슬리피지(·매도세) 반영. 분할익절 후 잔량은 본전스탑으로 보존.

---

## 5. 디렉터리 → 레이어 매핑

| 레이어 | 경로 | 역할 |
|---|---|---|
| ① 데이터 | `src/data/`, `synthetic_ohlcv` | 수집·캐시·정합성, 합성 폴백 |
| ② 전략 | `src/strategies/`, `src/signals_core.py` | 카테고리 플러그인 + 단일 두뇌 |
| ③ 엔진 | `src/engine/` | lookahead-safe 이벤트 백테스트 |
| ④ 지표 | `src/metrics.py` | 성과지표 산출 |
| ⑤ 검증 | `src/validation/` | 워크포워드·PBO·DSR·판정카드 |
| ⑥ 툴 | `src/run.py`, `src/validate.py`, `src/strategy_check.py` | 사용자 진입점 |
| ⑦ 독립검증 | `freqtrade/`, `scripts/` | 같은 신호 교차대조 |
| 테스트 | `tests/` | 미래참조·신호·엔진·검증·하네스 회귀 고정 |

---

## 6. 확장 지점 (설계상 열어둔 곳)

- **새 전략** → `src/strategies/`에 모듈 추가 + `register()`. 검증·툴은 자동 적용.
- **고속 스윕 백엔드(vectorbt 등)** → `src/validation/sweep.py`의 `_run_one` 한 함수만 교체.
- **일반인 UI(Streamlit 등)** → `validate.py`/`strategy_check.py`의 판정카드를 그대로 소비.
- **라이브 트레이딩** → 엔진과 같은 신호(`signals_core`/전략)를 freqtrade·nautilus로 연결.
