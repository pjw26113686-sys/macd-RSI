# 해외선물 프랍 다전략 비교 백테스트 프레임워크

새 매매 전략을 **한 파일로 추가**하면 자동으로 **선물 백테스트 · Apex 프랍룰 평가 ·
다전략 비교**까지 되는 구조. 프랍(Apex/CME) 기준의 자동매매 봇으로 가기 위한
전 단계 — "전략이 프랍 룰을 통과하는가"를 같은 기준으로 비교한다.

## 설계 핵심

- **전략은 신호만 책임진다.** 체결 타이밍·포지션 사이징·달러손익·프랍룰 평가·비교는
  전부 프레임워크가 처리. 전략 코드에 엔진/프랍 디테일이 새지 않는다.
- **지표는 단일 구현(`src/signals_core.py`).** 모든 전략이 import → 전략 간 드리프트 방지.
- **미래참조 0.** 각 전략이 `signal_columns`를 선언하고 `tests/test_lookahead.py`가
  전략별로 t절단 동일성을 자동 강제.
- **선물 손익은 계약·틱·달러.** 롱/숏 양방향. 프랍 트레일링DD는 봉 고저(wick)까지 본다.

## 구조

```
src/
  signals_core.py        지표 라이브러리(EMA/RMA/WMA/HMA/RSI/MACD/Bollinger/ATR…)
  instruments.py         선물 계약명세(tick/포인트가치/수수료) + 프리셋(ES/NQ/MES…)
  strategies/
    base.py              Strategy 프로토콜 + ExitModel(청산 정책)
    TEMPLATE.py          새 전략 작성 템플릿(복사해서 시작)
    example_ma_cross.py  예시: MA 교차(롱/숏 추세추종)
    example_bollinger.py 예시: 볼린저 밴드터치(롱/숏 평균회귀)
    registry.py          전략 등록 지점(여기에 한 줄 추가)
  engine/
    pnl.py               PnLModel: NotionalPnL(%) / FuturesPnL(계약·달러)
    position.py          포지션/거래 + 손절·목표 레벨(방향 일반화)
    backtest.py          전략-불가지론 이벤트 엔진(다음봉 시가 체결, 손절 우선)
  prop/
    rules.py             PropRuleSet + Apex/Topstep 프리셋
    evaluator.py         봉별 MTM(고저)로 트레일링DD·일일손실·수익목표·일관성 판정
  data/futures.py        합성(random-walk) 생성기 + CSV/parquet 로더
  metrics.py             달러기반 성과 + 꼬리위험(연속손실/단일최대손실/CVaR)
  compare.py             다전략×상품 비교 러너(표/CSV)
  run.py                 단일 전략 실행
config/params.yaml       전략·사이징·프랍·데이터·비교 설정
tests/                   지표 / lookahead(전략별) / 선물P&L / 숏엔진 / 프랍평가기
docs/prop_futures_spec.md  프랍룰 정의·트레일링DD 처리·실데이터 전환
```

## 설치 & 실행

```bash
pip install -e .            # 또는: pip install pandas numpy pyyaml pyarrow pytest
pytest -q                   # 26 테스트(lookahead = 신호 신뢰성 게이트)

python -m src.compare                                   # 전략×상품 비교표(+ Apex 판정)
python -m src.run --strategy bollinger --instrument NQ --prop apex_50k
python -m src.compare --csv out.csv                     # 결과 CSV 저장
```

> 기본 데이터는 **합성(random-walk) 선물**이다. 프레임워크·프랍평가기 동작 검증용이며
> **성과 판정용이 아니다**(랜덤워크는 비용 차감 후 손실이 정상 = sanity). 실데이터는
> 아래 참조.

## 새 전략 추가 (이 구조의 핵심)

```bash
cp src/strategies/TEMPLATE.py src/strategies/my_strategy.py
```
1. `add_signals`에 진입조건(`enter_long`/`enter_short`)을 채운다. 지표는 `signals_core`만 사용.
2. `exit_model`(손절/목표/시간/신호청산 정책)과 `signal_columns`를 선언한다.
3. `registry.py` `STRATEGIES`에 한 줄 등록한다.

→ `pytest -q`(미래참조 자동검증 통과) → `python -m src.compare`(표에 자동 등장).

## 프랍 룰 (Apex 위주)

`config/params.yaml`의 `prop.ruleset`으로 선택: `apex_25k/50k/100k/150k`,
`topstep_50k`. 평가기는 봉별 미실현 포함 equity(고/저)로:
- **트레일링DD** — 고점 대비 하락이 한도 초과 시 실격(인트라데이 wick 반영).
- **수익목표** — 누적이익 도달 시 통과(소요일 기록).
- **일일손실 한도**(Topstep) / **일관성 30%**(Apex 출금조건) 검사.

> 룰 수치는 **출발값**이다. 실제 약관과 대조 후 사용할 것(`src/prop/rules.py` 주석).

## 실데이터(CME) 전환

```yaml
# config/params.yaml
data:
  futures:
    synthetic: false
    path_template: "data/futures/{symbol}_{tf}.csv"   # OHLCV, UTC
```
Databento/Sierra Chart/IQFeed 등에서 받은 OHLCV(CSV/parquet, UTC)를 위 경로에
두면 `src/data/futures.load_futures`가 정합성 검증 후 사용한다.

## 비범위 (다음 차)

실데이터 성과판정, Topstep 프리셋 정밀화, hyperopt/grid-search, walk-forward/OOS,
라이브 자동매매(브로커 API·주문집행), 롤오버/연속물 스티칭, 세션(RTH/ETH) 정밀 모델.
