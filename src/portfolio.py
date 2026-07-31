"""다자산 포트폴리오 — 한 전략을 여러 종목에 배분해 합산 성과·분산효과를 본다.

단일 종목 성과는 그 종목 운에 좌우된다. 같은 전략을 N개 종목에 나눠 담으면 상관이
낮을수록 변동이 상쇄돼 위험대비수익이 좋아진다(분산효과). 이 모듈은:
  - 종목별로 자본을 배분(기본 균등)해 각각 백테스트(각 슬리브는 독립 엔진),
  - 슬리브 자산곡선을 합쳐 포트폴리오 자산곡선·지표 산출,
  - 종목 간 수익률 상관행렬과 분산효과(포트폴리오 Sharpe vs 평균 종목 Sharpe)를 계산.

CLI:
    python -m src.portfolio --strategy breakout_donchian --symbols 4 --synthetic --bars 3000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import metrics as _metrics
from src import strategies as strat
from src.engine import backtest as _bt
from src.engine.position import Costs
from src.validate import (CONFIG_PATH, _bars_per_year, apply_sizing, load_config,
                          synthetic_ohlcv)


def run_portfolio(
    spec,
    data: dict[str, pd.DataFrame],
    costs: Costs,
    bt_cfg: dict,
    bars_per_year: int,
    weights: dict[str, float] | None = None,
    params: dict | None = None,
) -> dict:
    """전략 하나를 여러 종목에 배분해 포트폴리오 성과를 계산.

    Parameters
    ----------
    spec : StrategySpec.
    data : {symbol: OHLCV df}.
    weights : {symbol: 비중}. None이면 균등. 합이 1이 아니면 정규화.
    params : 전략 파라미터. None이면 spec.default_params.

    Returns
    -------
    dict: portfolio(지표), per_asset({sym: 지표}), correlation(DataFrame),
          diversification(dict), equity(Series), initial_capital.
    """
    symbols = list(data)
    if not symbols:
        raise ValueError("data가 비었다")
    if weights is None:
        weights = {s: 1.0 / len(symbols) for s in symbols}
    wsum = sum(weights.values())
    weights = {s: weights[s] / wsum for s in symbols}  # 정규화
    params = spec.default_params if params is None else params

    total_cap = float(bt_cfg["initial_capital"])
    equities: dict[str, pd.Series] = {}
    per_asset: dict[str, dict] = {}
    all_trades: list = []

    for sym in symbols:
        cap = total_cap * weights[sym]
        res = _bt.run_backtest(
            data[sym], params, costs, initial_capital=cap,
            position_pct=bt_cfg["position_pct"], strategy=spec)
        equities[sym] = res.equity
        per_asset[sym] = {
            **_metrics.equity_metrics(res.equity, bars_per_year, cap),
            **_metrics.trade_metrics(res.trades),
            "weight": weights[sym], "capital": cap,
        }
        all_trades.extend(res.trades)

    # 슬리브 자산곡선 정렬(합집합 인덱스, 시작 전은 각 슬리브 초기자본으로 채움).
    union = None
    for eq in equities.values():
        union = eq.index if union is None else union.union(eq.index)
    aligned = {}
    for sym, eq in equities.items():
        cap = total_cap * weights[sym]
        aligned[sym] = eq.reindex(union).ffill().fillna(cap)
    eq_df = pd.DataFrame(aligned)
    port_equity = eq_df.sum(axis=1)
    port_equity.name = "portfolio"

    portfolio = {
        **_metrics.equity_metrics(port_equity, bars_per_year, total_cap),
        **_metrics.trade_metrics(all_trades),
    }

    # 종목 간 봉수익률 상관 + 분산효과.
    rets_df = eq_df.pct_change().dropna()
    corr = rets_df.corr()
    asset_sharpes = [m["Sharpe"] for m in per_asset.values() if m["Sharpe"] == m["Sharpe"]]
    mean_asset_sharpe = float(np.mean(asset_sharpes)) if asset_sharpes else float("nan")
    off_diag = corr.where(~np.eye(len(corr), dtype=bool))
    avg_corr = float(np.nanmean(off_diag.to_numpy())) if len(corr) > 1 else float("nan")
    diversification = {
        "portfolio_sharpe": portfolio["Sharpe"],
        "mean_asset_sharpe": mean_asset_sharpe,
        "sharpe_gain": (portfolio["Sharpe"] - mean_asset_sharpe
                        if portfolio["Sharpe"] == portfolio["Sharpe"] and mean_asset_sharpe == mean_asset_sharpe
                        else float("nan")),
        "avg_pairwise_corr": avg_corr,
        "n_assets": len(symbols),
    }

    return {
        "portfolio": portfolio,
        "per_asset": per_asset,
        "correlation": corr,
        "diversification": diversification,
        "equity": port_equity,
        "initial_capital": total_cap,
    }


def _synthetic_universe(n_symbols: int, bars: int, seed: int) -> dict[str, pd.DataFrame]:
    """종목마다 다른 시드의 합성 시계열(서로 다른 레짐)로 유니버스 구성."""
    return {f"SYN{i+1}": synthetic_ohlcv(bars, seed=seed + i * 101)
            for i in range(n_symbols)}


def _fmt(result: dict, spec, market: str) -> str:
    def pct(x, d=1):
        return "n/a" if x != x else f"{x*100:.{d}f}%"

    def num(x, d=2):
        return "n/a" if x != x else f"{x:.{d}f}"

    p = result["portfolio"]
    d = result["diversification"]
    lines = [
        f"╔══════ 다자산 포트폴리오 · {spec.category}/{spec.name} · {market.upper()} ══════",
        f"║  종목수/총자본     : {d['n_assets']}개 / {result['initial_capital']:.0f}",
        f"║  포트폴리오 총수익 : {pct(p['total_return'])}   CAGR {pct(p['CAGR'])}",
        f"║  포트폴리오 MDD    : {pct(p['MDD'])}   Sharpe {num(p['Sharpe'])}   거래 {p['n_trades']}",
        "║  ── 분산효과 ─────────────────────────",
        f"║  평균 종목 Sharpe  : {num(d['mean_asset_sharpe'])}",
        f"║  포트폴리오 Sharpe : {num(d['portfolio_sharpe'])}  (개선 {num(d['sharpe_gain'])})",
        f"║  평균 상관계수     : {num(d['avg_pairwise_corr'])}  (낮을수록 분산효과 큼)",
        "║  ── 종목별 ───────────────────────────",
    ]
    for sym, m in result["per_asset"].items():
        lines.append(f"║  {sym:8s} w{m['weight']*100:4.0f}% · 수익 {pct(m['total_return'])} · "
                     f"Sharpe {num(m['Sharpe'])} · 거래 {m['n_trades']}")
    lines.append("╚═══════════════════════════════════════")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="다자산 포트폴리오 백테스트")
    ap.add_argument("--strategy", default="breakout_donchian")
    ap.add_argument("--symbols", type=int, default=4, help="합성 종목 수")
    ap.add_argument("--market", choices=["crypto", "stock"], default="crypto")
    ap.add_argument("--synthetic", action="store_true", help="합성 유니버스 사용")
    ap.add_argument("--data", nargs="+", default=None,
                    help="실데이터 파일 목록(csv/parquet/…). 주면 합성 대신 사용")
    ap.add_argument("--data-dir", default=None,
                    help="실데이터 디렉터리(안의 모든 표 파일을 유니버스로)")
    ap.add_argument("--bars", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sizing", choices=["fixed_fraction", "fixed_risk"], default=None)
    ap.add_argument("--risk-pct", type=float, default=None, dest="risk_pct")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    spec = strat.get(args.strategy)
    cost_cfg = cfg["costs"][args.market]
    costs = Costs(fee=cost_cfg["fee"], slippage=cost_cfg["slippage"],
                  sell_tax=cost_cfg.get("sell_tax", 0.0))
    bt_cfg = cfg["backtest"]
    bpy = _bars_per_year(args.market)

    if args.data or args.data_dir:
        from src.data.ingest import load_universe
        data = load_universe(args.data or args.data_dir, args.market)
        print(f"[i] 실데이터 유니버스 {len(data)}종목: {', '.join(data)}")
    else:
        if not args.synthetic:
            print("[!] 실데이터(--data/--data-dir) 미지정 → 합성 유니버스로 진행.")
        data = _synthetic_universe(args.symbols, args.bars, args.seed)

    params = apply_sizing([spec.default_params], bt_cfg, args)[0]
    result = run_portfolio(spec, data, costs, bt_cfg, bpy, params=params)
    print(_fmt(result, spec, args.market))


if __name__ == "__main__":
    main()
