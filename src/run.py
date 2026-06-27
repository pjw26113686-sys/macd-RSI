"""CLI 단일 실행: 데이터 → 전략 신호 → 선물 백테스트 → 성과 + 프랍 리포트.

    python -m src.run --strategy bollinger --instrument NQ --prop apex_50k
    python -m src.run --strategy ma_cross --instrument ES
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src import metrics
from src.data import futures
from src.engine import backtest
from src.engine.pnl import FuturesPnL
from src.instruments import get_instrument
from src.prop import evaluator
from src.prop.rules import get_ruleset
from src.strategies.registry import STRATEGIES, get_strategy

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "params.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single(strategy_name, symbol, prop_name, cfg):
    from src.compare import _sizing
    inst = get_instrument(symbol)
    strat = get_strategy(strategy_name)
    params = strat.params(cfg.get("strategies", {}).get(strategy_name))
    df = futures.get_futures_data(symbol, cfg)
    pnl = FuturesPnL(inst)
    ruleset = get_ruleset(prop_name)

    res = backtest.run_backtest(df, strat, params, pnl, _sizing(cfg),
                                initial_capital=ruleset.start_balance, instrument=inst)
    tf = cfg.get("data", {}).get("futures", {}).get("timeframe", "5m")
    m = metrics.compute_metrics(res, bars_per_year=metrics.bars_per_year(tf))
    pr = evaluator.evaluate_prop(res.equity, res.equity_high, res.equity_low, ruleset)

    title = f"{strategy_name} · {symbol} · {len(df)}봉"
    print(metrics.format_report(title, m))
    print(f"  ── 프랍 ──")
    print(f"  {pr.summary()}")
    return res, m, pr


def main():
    ap = argparse.ArgumentParser(description="단일 전략 프랍 선물 백테스트")
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="ma_cross")
    ap.add_argument("--instrument", default="NQ")
    ap.add_argument("--prop", default="apex_50k")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    run_single(args.strategy, args.instrument, args.prop, cfg)


if __name__ == "__main__":
    main()
