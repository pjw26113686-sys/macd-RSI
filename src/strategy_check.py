"""전략 합격 판정 하네스 — 새 전략을 가져와 한 번에 검증하는 툴.

새 전략(레지스트리 등록 여부 무관, 임의의 .py 파일도 가능)을 받아 다음을 순서대로
돌리고 합격/주의/불합격을 낸다:

  무결성 게이트(하드 통과 조건 — 하나라도 FAIL이면 이 전략은 신뢰 불가):
    1) 신호 계약   enter_long + 청산 컬럼이 있고 bool이며 시각 인덱스를 보존하는가
    2) 미래참조 0  전체계산과 t시점 절단계산이 t에서 일치하는가 (제일 중요한 게이트)
    3) 결정성     동일 입력에 동일 출력인가
    4) 엔진 연동   자산이 유한·양수이고 실제 거래가 발생하는가
    5) 스윕 가능   param_grid가 CSCV 순위매김에 충분한 설정을 만드는가
  성과 판정(정보성 — 무결성은 통과해도 과최적화일 수 있다):
    스윕 → CSCV PBO + Deflated Sharpe + 워크포워드 → 판정카드.

사용:
    python -m src.strategy_check --strategy breakout_donchian --synthetic
    python -m src.strategy_check --module path/to/my_strategy.py --synthetic
    python -m src.strategy_check --category mean_reversion --synthetic --no-perf

exit code: 무결성 게이트가 하나라도 FAIL이면 1, 아니면 0.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from src import metrics as _metrics
from src import strategies as strat
from src.engine import backtest as _bt
from src.engine.position import Costs
from src.strategies.base import StrategySpec
from src.validate import load_config, synthetic_ohlcv, _load_data
from src.validation import cscv_pbo, deflated_sharpe_ratio, expand_grid, run_sweep
from src.validation.report import format_validation_report
from src.validation.sweep import walk_forward_analysis

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_ICON = {PASS: "✅", WARN: "⚠️", FAIL: "⛔"}


@dataclass
class CheckResult:
    name: str
    status: str        # PASS | WARN | FAIL
    detail: str

    @property
    def is_gate_failure(self) -> bool:
        return self.status == FAIL


# --------------------------------------------------------------------------- #
# 전략 로딩 (레지스트리 or 임의 파일)
# --------------------------------------------------------------------------- #
def load_spec_from_module(path: str) -> StrategySpec:
    """임의의 .py 파일에서 StrategySpec을 찾아 반환(레지스트리 등록 불필요).

    모듈 최상위에 정의된 StrategySpec 인스턴스를 수집한다. 여러 개면 `SPEC`을 우선.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"전략 파일 없음: {p}")
    mod_name = f"_user_strategy_{p.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    specs = [v for v in vars(module).values() if isinstance(v, StrategySpec)]
    if not specs:
        raise ValueError(
            f"{p.name}에서 StrategySpec을 못 찾음. 모듈 최상위에 "
            f"`SPEC = StrategySpec(...)`를 정의하라 (src/strategies/_template.py 참고)."
        )
    if len(specs) > 1:
        named = getattr(module, "SPEC", None)
        if isinstance(named, StrategySpec):
            return named
        raise ValueError(f"{p.name}에 StrategySpec이 여러 개 — 하나를 `SPEC`으로 지정하라.")
    return specs[0]


# --------------------------------------------------------------------------- #
# 무결성 게이트
# --------------------------------------------------------------------------- #
def check_contract(spec: StrategySpec, df: pd.DataFrame) -> CheckResult:
    try:
        sig = spec.generate_signals(df, spec.default_params)
    except Exception as e:  # noqa: BLE001
        return CheckResult("신호 계약", FAIL, f"generate_signals 예외: {e!r}")

    required = ["enter_long", spec.exit_col, spec.exit_half_col]
    missing = [c for c in required if c not in sig.columns]
    if missing:
        return CheckResult("신호 계약", FAIL, f"필수 컬럼 누락: {missing}")
    if len(sig) != len(df):
        return CheckResult("신호 계약", FAIL,
                           f"행 수 불일치: 입력 {len(df)} → 출력 {len(sig)}")
    if not sig.index.equals(df.index):
        return CheckResult("신호 계약", FAIL, "시각 인덱스를 보존하지 않음")

    non_bool = [c for c in required if not is_bool_dtype(sig[c])]
    if non_bool:
        return CheckResult("신호 계약", WARN,
                           f"bool 아님(런타임 truthy로 동작하나 권장X): {non_bool}")
    return CheckResult("신호 계약", PASS,
                       f"{', '.join(required)} · bool · 인덱스 보존")


def check_lookahead(spec: StrategySpec, df: pd.DataFrame,
                    n_points: int = 25, seed: int = 123) -> CheckResult:
    """전체계산 vs t시점 절단계산이 t에서 일치해야 한다(미래참조 0)."""
    params = spec.default_params
    cols = ["enter_long", spec.exit_col, spec.exit_half_col]
    try:
        full = spec.generate_signals(df, params)
    except Exception as e:  # noqa: BLE001
        return CheckResult("미래참조 0", FAIL, f"generate_signals 예외: {e!r}")

    warm = spec.warmup_bars(params)
    lo = min(warm + 5, len(df) - 2)
    rng = np.random.default_rng(seed)
    pts = sorted({int(x) for x in rng.integers(lo, len(df) - 1, size=n_points)})
    for t in pts:
        trunc = spec.generate_signals(df.iloc[: t + 1], params)
        for col in cols:
            if bool(full.iloc[t][col]) != bool(trunc.iloc[t][col]):
                return CheckResult(
                    "미래참조 0", FAIL,
                    f"누수 감지: {col} @ t={t} "
                    f"(전체={bool(full.iloc[t][col])}, 절단={bool(trunc.iloc[t][col])})",
                )
    return CheckResult("미래참조 0", PASS, f"{len(pts)}개 시점 절단검사 통과")


def check_determinism(spec: StrategySpec, df: pd.DataFrame) -> CheckResult:
    a = spec.generate_signals(df, spec.default_params)
    b = spec.generate_signals(df, spec.default_params)
    cols = ["enter_long", spec.exit_col, spec.exit_half_col]
    for col in cols:
        if not a[col].equals(b[col]):
            return CheckResult("결정성", FAIL, f"동일 입력에 다른 출력: {col}")
    return CheckResult("결정성", PASS, "동일 입력 → 동일 출력")


def check_engine(spec: StrategySpec, df: pd.DataFrame, costs: Costs,
                 bt_cfg: dict) -> CheckResult:
    try:
        res = _bt.run_backtest(
            df, spec.default_params, costs,
            initial_capital=bt_cfg["initial_capital"],
            position_pct=bt_cfg["position_pct"], strategy=spec,
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult("엔진 연동", FAIL, f"백테스트 예외: {e!r}")

    eq = res.equity.to_numpy()
    if not np.isfinite(eq).all():
        return CheckResult("엔진 연동", FAIL, "자산 곡선에 NaN/inf")
    if (eq <= 0).any():
        return CheckResult("엔진 연동", FAIL, "자산이 0 이하로 감(파산)")
    n_tr = len(res.trades)
    if n_tr == 0:
        return CheckResult("엔진 연동", WARN, "거래 0건 — 진입 조건이 과도하게 좁을 수 있음")
    return CheckResult("엔진 연동", PASS, f"자산 유한·양수 · 거래 {n_tr}건")


def check_sweepable(spec: StrategySpec, min_configs: int = 4) -> CheckResult:
    n = len(expand_grid(spec.default_params, spec.param_grid))
    if n < min_configs:
        return CheckResult("스윕 가능", WARN,
                           f"설정 {n}개(<{min_configs}) — PBO 순위매김이 빈약함")
    return CheckResult("스윕 가능", PASS, f"param_grid → 설정 {n}개")


def run_integrity_gauntlet(spec: StrategySpec, df: pd.DataFrame, costs: Costs,
                           bt_cfg: dict) -> list[CheckResult]:
    return [
        check_contract(spec, df),
        check_lookahead(spec, df),
        check_determinism(spec, df),
        check_engine(spec, df, costs, bt_cfg),
        check_sweepable(spec),
    ]


# --------------------------------------------------------------------------- #
# 성과 판정 (기존 검증 레이어 재사용)
# --------------------------------------------------------------------------- #
def run_performance(spec: StrategySpec, df: pd.DataFrame, symbol: str, bpy: int,
                    market: str, costs: Costs, bt_cfg: dict, args) -> str:
    params_list = expand_grid(spec.default_params, spec.param_grid)
    sweep = run_sweep(
        df, params_list, costs, bars_per_year=bpy,
        initial_capital=bt_cfg["initial_capital"],
        position_pct=bt_cfg["position_pct"], strategy=spec,
    )
    pbo_res = cscv_pbo(sweep["returns"].to_numpy(), n_blocks=args.blocks)
    best_i = int(np.argmax(sweep["sr_trials"]))
    dsr_res = deflated_sharpe_ratio(
        sweep["returns"].iloc[:, best_i].to_numpy(), sr_trials=sweep["sr_trials"])
    wfa_res = None
    if not args.no_wfa:
        wfa_res = walk_forward_analysis(
            df, params_list, costs, bars_per_year=bpy,
            n_splits=args.folds, mode=args.wf_mode, embargo=args.embargo,
            initial_capital=bt_cfg["initial_capital"],
            position_pct=bt_cfg["position_pct"], strategy=spec)
    title = f"{spec.category}/{spec.name}·{market.upper()}·{symbol}"
    return format_validation_report(title, pbo_res, dsr_res, wfa_res)


# --------------------------------------------------------------------------- #
# 리포트
# --------------------------------------------------------------------------- #
def format_gauntlet(spec: StrategySpec, checks: list[CheckResult],
                    perf_card: str | None) -> tuple[str, bool]:
    n_fail = sum(c.is_gate_failure for c in checks)
    n_pass = sum(c.status == PASS for c in checks)
    gate_ok = n_fail == 0

    lines = [
        f"┏━━━━━ 전략 합격 판정 · {spec.name} ({spec.category}) ━━━━━",
        f"┃ {spec.description}",
        "┃ ── 무결성 게이트 ──────────────────────",
    ]
    for c in checks:
        lines.append(f"┃  [{_ICON[c.status]} {c.status}] {c.name:10s} {c.detail}")
    verdict = "합격(무결성 통과)" if gate_ok else "불합격(무결성 실패)"
    lines.append(f"┃  → 무결성 {n_pass}/{len(checks)} PASS · {_ICON[PASS if gate_ok else FAIL]} {verdict}")
    if perf_card is not None:
        lines.append("┃ ── 성과 판정 ──────────────────────────")
        for pl in perf_card.splitlines():
            lines.append("┃  " + pl)
    lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines), gate_ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _resolve_specs(args) -> list[StrategySpec]:
    if args.module:
        return [load_spec_from_module(args.module)]
    if args.category:
        specs = strat.by_category(args.category)
        if not specs:
            raise SystemExit(f"미등록 카테고리: {args.category!r}. 가능: {strat.categories()}")
        return specs
    return [strat.get(args.strategy)]


def run(args) -> bool:
    cfg = load_config(Path(args.config))
    df, symbol, bpy, market = _load_data(args, cfg)
    cost_cfg = cfg["costs"][market]
    costs = Costs(fee=cost_cfg["fee"], slippage=cost_cfg["slippage"],
                  sell_tax=cost_cfg.get("sell_tax", 0.0))
    bt_cfg = cfg["backtest"]

    all_gates_ok = True
    for spec in _resolve_specs(args):
        checks = run_integrity_gauntlet(spec, df, costs, bt_cfg)
        gate_ok = all(not c.is_gate_failure for c in checks)
        # 무결성 실패 시 성과 판정은 무의미하므로 생략.
        perf_card = None
        if gate_ok and not args.no_perf:
            perf_card = run_performance(spec, df, symbol, bpy, market, costs, bt_cfg, args)
        report, ok = format_gauntlet(spec, checks, perf_card)
        print(report)
        all_gates_ok = all_gates_ok and ok
    return all_gates_ok


def main():
    ap = argparse.ArgumentParser(description="전략 합격 판정 하네스(무결성 게이트 + 성과 판정)")
    src = ap.add_argument_group("전략 지정 (택1)")
    src.add_argument("--strategy", default="momentum_macd_rsi", help="레지스트리 전략 이름")
    src.add_argument("--category", default=None, help="카테고리 전체")
    src.add_argument("--module", default=None, help="임의의 .py 파일에서 전략 로드")

    ap.add_argument("--market", choices=["crypto", "stock"], default="crypto")
    ap.add_argument("--synthetic", action="store_true", help="합성 데이터 강제 사용")
    ap.add_argument("--bars", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--blocks", type=int, default=16, help="CSCV 블록 수(짝수)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--wf-mode", choices=["anchored", "rolling"], default="rolling")
    ap.add_argument("--embargo", type=int, default=0)
    ap.add_argument("--no-wfa", action="store_true", help="워크포워드 생략")
    ap.add_argument("--no-perf", action="store_true", help="무결성 게이트만(성과 판정 생략)")
    ap.add_argument("--config", default=str(load_config.__defaults__[0]))
    args = ap.parse_args()

    ok = run(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
