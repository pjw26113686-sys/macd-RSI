"""전략 평가 프로그래밍 API — UI/CLI가 공유하는 계산 코어.

CLI는 결과를 '출력'하지만, UI는 결과를 '데이터'로 받아 그려야 한다. 이 모듈은 전략을
평가해 (무결성 게이트 + PBO/DSR/워크포워드 + 최우수 설정 수익곡선 + 벤치마크)를
구조화된 dict로 돌려준다. Streamlit 앱과 테스트가 이걸 소비한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import metrics as _metrics
from src.engine import backtest as _bt
from src.engine.position import Costs
from src.strategy_check import run_integrity_gauntlet
from src.validate import apply_sizing
from src.validation import cscv_pbo, deflated_sharpe_ratio, expand_grid, run_sweep
from src.validation.report import _verdict_level
from src.validation.sweep import walk_forward_analysis


def buy_hold_equity(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """벤치마크: 첫 봉 종가에 전액 매수 후 보유(Buy & Hold) 자산곡선."""
    close = df["close"].astype(float)
    base = close.iloc[0]
    eq = initial_capital * (close / base)
    eq.name = "buy_hold"
    return eq


def evaluate_strategy(
    spec,
    df: pd.DataFrame,
    costs: Costs,
    bt_cfg: dict,
    bars_per_year: int,
    *,
    blocks: int = 16,
    folds: int = 5,
    wf_mode: str = "rolling",
    embargo: int = 0,
    do_wfa: bool = True,
    sizing: str | None = None,
    risk_pct: float | None = None,
) -> dict:
    """전략을 평가해 UI/리포트용 구조화 결과를 반환.

    무결성 게이트가 FAIL이면 성과 계산은 생략하고 integrity/gate_ok만 채운다.
    """
    class _A:  # apply_sizing이 기대하는 얇은 args 객체
        pass
    a = _A()
    a.sizing, a.risk_pct = sizing, risk_pct

    integrity = run_integrity_gauntlet(spec, df, costs, bt_cfg)
    gate_ok = all(not c.is_gate_failure for c in integrity)
    out: dict = {
        "name": spec.name, "category": spec.category, "description": spec.description,
        "integrity": integrity, "gate_ok": gate_ok,
    }
    if not gate_ok:
        return out

    params_list = apply_sizing(expand_grid(spec.default_params, spec.param_grid), bt_cfg, a)
    sweep = run_sweep(
        df, params_list, costs, bars_per_year=bars_per_year,
        initial_capital=bt_cfg["initial_capital"], position_pct=bt_cfg["position_pct"],
        strategy=spec,
    )
    pbo = cscv_pbo(sweep["returns"].to_numpy(), n_blocks=blocks)
    best_i = int(np.argmax(sweep["sr_trials"]))
    dsr = deflated_sharpe_ratio(
        sweep["returns"].iloc[:, best_i].to_numpy(), sr_trials=sweep["sr_trials"])

    wfa = None
    if do_wfa:
        wfa = walk_forward_analysis(
            df, params_list, costs, bars_per_year=bars_per_year,
            n_splits=folds, mode=wf_mode, embargo=embargo,
            initial_capital=bt_cfg["initial_capital"], position_pct=bt_cfg["position_pct"],
            strategy=spec)

    # 최우수 설정의 자산곡선 + 벤치마크(Buy & Hold).
    best_params = params_list[best_i]
    best_res = _bt.run_backtest(
        df, best_params, costs, initial_capital=bt_cfg["initial_capital"],
        position_pct=bt_cfg["position_pct"], strategy=spec)
    best_metrics = _metrics.compute_metrics(best_res, bars_per_year=bars_per_year)

    degradation = wfa["degradation"] if wfa else float("nan")
    level, reason = _verdict_level(pbo["pbo"], dsr["dsr"], degradation)

    out.update({
        "pbo": pbo, "dsr": dsr, "wfa": wfa,
        "best_index": best_i, "best_params": best_params, "best_metrics": best_metrics,
        "n_configs": len(params_list),
        "equity": best_res.equity, "benchmark": buy_hold_equity(df, bt_cfg["initial_capital"]),
        "verdict_level": level, "verdict_reason": reason,
    })
    return out
