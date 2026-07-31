"""백테스트 성과 지표 (strategy_spec_v2 §6).

산출: 총수익률, CAGR, MDD, 샤프, 승률, 실현손익비, 평균보유봉, 거래횟수.
"""
from __future__ import annotations

import numpy as np

# 1시간봉 기준 연환산 계수 (24h*365 ≈ 8760). 주식 정규장(~6.5h*252)도 근사로 동일계수
# 사용 시 왜곡되므로, bars_per_year를 시장별로 주입한다.
BARS_PER_YEAR_CRYPTO = 24 * 365
BARS_PER_YEAR_STOCK = 252 * 7  # 미국 정규장 1h봉 ≈ 하루 7봉(09:30~16:00)


def equity_metrics(equity, bars_per_year: int, initial_capital: float) -> dict:
    """자산곡선(equity)만으로 계산되는 지표: 총수익률·CAGR·MDD·Sharpe·최종자산.

    포트폴리오처럼 단일 trades 리스트가 없는 경우에도 쓰도록 분리했다.
    """
    init = float(initial_capital)
    final = float(equity.iloc[-1]) if len(equity) else init
    total_return = final / init - 1.0 if init else float("nan")

    rets = equity.pct_change().dropna()
    n_bars = len(equity)
    if n_bars > 1:
        years = n_bars / bars_per_year
        cagr = (final / init) ** (1.0 / years) - 1.0 if years > 0 and final > 0 and init > 0 else float("nan")
    else:
        cagr = float("nan")

    sharpe = ((rets.mean() / rets.std(ddof=0)) * np.sqrt(bars_per_year)
              if rets.std(ddof=0) > 0 else float("nan"))

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    mdd = float(drawdown.min()) if len(drawdown) else 0.0

    return {
        "total_return": total_return,
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "final_equity": final,
    }


def trade_metrics(trades) -> dict:
    """거래 리스트로 계산되는 지표: 승률·평균손익·실현손익비·평균보유봉·거래횟수."""
    n_trades = len(trades)
    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    win_rate = len(wins) / n_trades if n_trades else float("nan")
    avg_win = np.mean([t.ret for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.ret for t in losses]) if losses else 0.0
    realized_rr = (avg_win / abs(avg_loss)) if avg_loss < 0 else float("nan")
    avg_bars_held = np.mean([t.bars_held for t in trades]) if n_trades else float("nan")
    return {
        "win_rate": win_rate,
        "avg_win_ret": float(avg_win),
        "avg_loss_ret": float(avg_loss),
        "realized_RR": realized_rr,
        "avg_bars_held": float(avg_bars_held) if n_trades else float("nan"),
        "n_trades": n_trades,
    }


def compute_metrics(result, bars_per_year: int) -> dict:
    """단일 백테스트 결과의 전체 지표(자산곡선 + 거래통계)."""
    em = equity_metrics(result.equity, bars_per_year, result.initial_capital)
    tm = trade_metrics(result.trades)
    # 기존 키 순서 유지(리포트 호환).
    return {
        "total_return": em["total_return"],
        "CAGR": em["CAGR"],
        "MDD": em["MDD"],
        "Sharpe": em["Sharpe"],
        "win_rate": tm["win_rate"],
        "avg_win_ret": tm["avg_win_ret"],
        "avg_loss_ret": tm["avg_loss_ret"],
        "realized_RR": tm["realized_RR"],
        "avg_bars_held": tm["avg_bars_held"],
        "n_trades": tm["n_trades"],
        "final_equity": em["final_equity"],
    }


def format_report(name: str, m: dict) -> str:
    def pct(x):
        return "n/a" if x != x else f"{x*100:6.2f}%"

    def num(x, d=2):
        return "n/a" if x != x else f"{x:.{d}f}"

    lines = [
        f"================ {name} ================",
        f"  거래횟수        : {m['n_trades']}",
        f"  총수익률        : {pct(m['total_return'])}",
        f"  CAGR           : {pct(m['CAGR'])}",
        f"  MDD            : {pct(m['MDD'])}",
        f"  Sharpe         : {num(m['Sharpe'])}",
        f"  승률           : {pct(m['win_rate'])}",
        f"  평균수익(승)    : {pct(m['avg_win_ret'])}",
        f"  평균손실(패)    : {pct(m['avg_loss_ret'])}",
        f"  실현손익비      : {num(m['realized_RR'])}",
        f"  평균보유봉수    : {num(m['avg_bars_held'], 1)}",
        f"  최종자산        : {num(m['final_equity'])}",
    ]
    return "\n".join(lines)
