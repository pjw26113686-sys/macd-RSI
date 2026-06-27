"""다전략 비교 러너 — 등록된 전략 × 선물상품 × 프랍룰을 한 표로 비교.

각 (전략, 상품) 조합: 데이터 → 신호 → 선물 백테스트 → 성과지표 + 프랍 평가.
새 전략을 registry에 등록하면 자동으로 이 표에 등장한다.

    python -m src.compare
    python -m src.compare --config config/params.yaml --csv out.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src import metrics
from src.data import futures
from src.engine import backtest
from src.engine.pnl import FuturesPnL, SizingConfig
from src.instruments import get_instrument
from src.prop import evaluator
from src.prop.rules import get_ruleset
from src.strategies.registry import get_strategy

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "params.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sizing(cfg: dict) -> SizingConfig:
    s = cfg.get("sizing", {})
    return SizingConfig(
        mode=s.get("mode", "fixed"),
        contracts=s.get("contracts", 1),
        risk_dollars=s.get("risk_dollars", 250.0),
        max_contracts=s.get("max_contracts", 10),
    )


def run_compare(cfg: dict) -> pd.DataFrame:
    comp = cfg.get("compare", {})
    strat_names = comp.get("strategies") or list(
        cfg.get("strategies", {}).keys()) or ["ma_cross", "bollinger"]
    instruments = comp.get("instruments", ["NQ"])
    ruleset = get_ruleset(cfg.get("prop", {}).get("ruleset", "apex_50k"))
    sizing = _sizing(cfg)
    tf = cfg.get("data", {}).get("futures", {}).get("timeframe", "5m")
    bpy = metrics.bars_per_year(tf)
    strat_params = cfg.get("strategies", {})

    rows = []
    data_cache = {}
    for sym in instruments:
        inst = get_instrument(sym)
        if sym not in data_cache:
            data_cache[sym] = futures.get_futures_data(sym, cfg)
        df = data_cache[sym]
        pnl = FuturesPnL(inst)
        for name in strat_names:
            strat = get_strategy(name)
            params = strat.params(strat_params.get(name))
            res = backtest.run_backtest(
                df, strat, params, pnl, sizing,
                initial_capital=ruleset.start_balance, instrument=inst)
            m = metrics.compute_metrics(res, bars_per_year=bpy)
            pr = evaluator.evaluate_prop(res.equity, res.equity_high,
                                         res.equity_low, ruleset)
            rows.append({
                "strategy": name, "instrument": sym, "trades": m["n_trades"],
                "net_pnl": m["net_pnl"], "MDD$": m["MDD_dollar"],
                "Sharpe": m["Sharpe"], "consec_loss": m["max_consecutive_losses"],
                "prop": "PASS" if pr.passed else ("FAIL" if pr.failed else "—"),
                "days": pr.days_to_pass, "fail_reason": pr.fail_reason,
                "max_trail$": pr.max_trailing_used,
            })
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame, ruleset_name: str) -> str:
    if df.empty:
        return "(결과 없음)"
    out = df.copy()
    out["net_pnl"] = out["net_pnl"].map(lambda x: f"${x:,.0f}")
    out["MDD$"] = out["MDD$"].map(lambda x: f"${x:,.0f}")
    out["max_trail$"] = out["max_trail$"].map(lambda x: f"${x:,.0f}")
    out["Sharpe"] = out["Sharpe"].map(lambda x: f"{x:.2f}" if x == x else "n/a")
    out["days"] = out["days"].map(lambda x: "" if x is None or x != x else int(x))
    out["fail_reason"] = out["fail_reason"].fillna("")
    header = f"=== 전략 비교 (프랍: {ruleset_name}) ===\n"
    return header + out.to_string(index=False)


def main():
    ap = argparse.ArgumentParser(description="다전략 프랍 선물 비교 백테스트")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--csv", default=None, help="결과 CSV 저장 경로")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    df = run_compare(cfg)
    ruleset_name = cfg.get("prop", {}).get("ruleset", "apex_50k")
    print(format_table(df, ruleset_name))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n저장: {args.csv}")


if __name__ == "__main__":
    main()
