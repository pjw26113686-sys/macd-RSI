"""백테스트 성과 지표 — 달러기반(선물) + 비율 + 꼬리위험.

프랍 평가는 prop/evaluator.py가 별도로 담당한다. 여기서는 순수 성과지표만.
연환산 계수(bars_per_year)는 timeframe별로 주입한다(시장·봉마다 다름).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# timeframe → 연환산 봉수 (24시간 시장=선물/암호화폐 근사)
BARS_PER_YEAR = {
    "1m": 60 * 24 * 365,
    "5m": 12 * 24 * 365,
    "15m": 4 * 24 * 365,
    "1h": 24 * 365,
    "1d": 365,
}
BARS_PER_YEAR_STOCK = 252 * 7   # 미국 정규장 1h봉 근사


def bars_per_year(timeframe: str, default: int = 24 * 365) -> int:
    return BARS_PER_YEAR.get(timeframe, default)


def _max_consecutive_losses(trades) -> int:
    streak = best = 0
    for t in trades:
        if not t.is_win:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def compute_metrics(result, bars_per_year: int = 24 * 365) -> dict:
    equity = result.equity
    trades = result.trades
    init = result.initial_capital

    final = float(equity.iloc[-1]) if len(equity) else init
    net_pnl = final - init
    total_return = final / init - 1.0 if init else float("nan")

    rets = equity.pct_change().dropna()
    n_bars = len(equity)
    if n_bars > 1 and final > 0 and init > 0:
        years = n_bars / bars_per_year
        cagr = (final / init) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    else:
        cagr = float("nan")

    sharpe = ((rets.mean() / rets.std(ddof=0)) * np.sqrt(bars_per_year)
              if rets.std(ddof=0) > 0 else float("nan"))

    running_max = equity.cummax()
    mdd = float((equity / running_max - 1.0).min()) if len(equity) else 0.0
    mdd_dollar = float((equity - running_max).min()) if len(equity) else 0.0

    n_trades = len(trades)
    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    win_rate = len(wins) / n_trades if n_trades else float("nan")
    pnls = [t.pnl for t in trades]
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
    realized_rr = (avg_win / abs(avg_loss)) if avg_loss < 0 else float("nan")
    expectancy = float(np.mean(pnls)) if pnls else 0.0
    avg_bars_held = float(np.mean([t.bars_held for t in trades])) if n_trades else float("nan")

    # 꼬리위험
    worst_trade = float(min(pnls)) if pnls else 0.0
    max_consec_loss = _max_consecutive_losses(trades)
    if pnls:
        srt = np.sort(pnls)
        k = max(1, int(len(srt) * 0.05))
        tail_cvar5 = float(np.mean(srt[:k]))
    else:
        tail_cvar5 = 0.0

    return {
        "net_pnl": net_pnl,
        "total_return": total_return,
        "CAGR": cagr,
        "MDD": mdd,
        "MDD_dollar": mdd_dollar,
        "Sharpe": sharpe,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "realized_RR": realized_rr,
        "max_consecutive_losses": max_consec_loss,
        "worst_trade": worst_trade,
        "tail_cvar5": tail_cvar5,
        "avg_bars_held": avg_bars_held,
        "n_trades": n_trades,
        "final_equity": final,
    }


def format_report(name: str, m: dict) -> str:
    def pct(x):
        return "n/a" if x != x else f"{x*100:6.2f}%"

    def num(x, d=2):
        return "n/a" if x != x else f"{x:.{d}f}"

    def usd(x):
        return "n/a" if x != x else f"${x:,.0f}"

    lines = [
        f"================ {name} ================",
        f"  거래횟수        : {m['n_trades']}",
        f"  순손익          : {usd(m['net_pnl'])}",
        f"  총수익률        : {pct(m['total_return'])}",
        f"  CAGR           : {pct(m['CAGR'])}",
        f"  MDD            : {pct(m['MDD'])}  ({usd(m['MDD_dollar'])})",
        f"  Sharpe         : {num(m['Sharpe'])}",
        f"  승률           : {pct(m['win_rate'])}",
        f"  기대값/거래     : {usd(m['expectancy'])}",
        f"  실현손익비      : {num(m['realized_RR'])}",
        f"  최대연속손실    : {m['max_consecutive_losses']}",
        f"  단일최대손실    : {usd(m['worst_trade'])}",
        f"  CVaR5%         : {usd(m['tail_cvar5'])}",
        f"  평균보유봉수    : {num(m['avg_bars_held'], 1)}",
        f"  최종자산        : {usd(m['final_equity'])}",
    ]
    return "\n".join(lines)
