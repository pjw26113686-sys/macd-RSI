"""CLI 진입점: 데이터 → 신호 → 백테스트 → 리포트.

사용:
    python -m src.run --market crypto
    python -m src.run --market stock
    python -m src.run --market crypto --no-cache
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src import metrics
from src.engine import backtest
from src.engine.position import Costs

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "params.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_data(market: str, cfg: dict, use_cache: bool):
    d = cfg["data"][market]
    if market == "crypto":
        from src.data import crypto
        return crypto.fetch_ohlcv(
            exchange=d["exchange"], symbol=d["symbol"], timeframe=d["timeframe"],
            since=d["since"], use_cache=use_cache,
        ), d["symbol"]
    elif market == "stock":
        from src.data import stocks
        return stocks.fetch_ohlcv(
            symbol=d["symbol"], interval=d["interval"], period=d["period"],
            auto_adjust=d["auto_adjust"], use_cache=use_cache,
        ), d["symbol"]
    raise ValueError(f"알 수 없는 market: {market}")


def run_market(market: str, cfg: dict, use_cache: bool = True):
    df, symbol = _load_data(market, cfg, use_cache)
    params = cfg["strategy"]
    cost_cfg = cfg["costs"][market]
    costs = Costs(fee=cost_cfg["fee"], slippage=cost_cfg["slippage"],
                  sell_tax=cost_cfg.get("sell_tax", 0.0))
    bt = cfg["backtest"]

    result = backtest.run_backtest(
        df, params, costs,
        initial_capital=bt["initial_capital"], position_pct=bt["position_pct"],
    )
    bpy = (metrics.BARS_PER_YEAR_CRYPTO if market == "crypto"
           else metrics.BARS_PER_YEAR_STOCK)
    m = metrics.compute_metrics(result, bars_per_year=bpy)
    title = f"{market.upper()} · {symbol} · {len(df)}봉"
    print(metrics.format_report(title, m))
    return result, m


def main():
    ap = argparse.ArgumentParser(description="MACD+RSI 전략 1차 백테스트")
    ap.add_argument("--market", choices=["crypto", "stock"], required=True)
    ap.add_argument("--no-cache", action="store_true", help="캐시 무시하고 재다운로드")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    run_market(args.market, cfg, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
