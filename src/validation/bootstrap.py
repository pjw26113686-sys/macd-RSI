"""블록 부트스트랩 신뢰구간 — "이 성과가 얼마나 운인가"를 구간으로.

단일 백테스트는 점추정 하나(예: Sharpe 1.2)만 준다. 그 값이 표본 하나에서 나온
우연인지, 견고한지는 알 수 없다. 여기서는 봉수익률을 **블록 단위로 재표집**(자기상관
보존)해 지표 분포를 만들고, 신뢰구간과 손익확률을 낸다.

무빙블록 부트스트랩: 길이 `block`의 연속 조각을 무작위 위치에서 뽑아 이어붙여 원래
길이의 대체표본을 만든다. 봉 간 자기상관(추세·군집변동성)을 어느 정도 보존한다.
"""
from __future__ import annotations

import numpy as np


def _moving_block_resample(returns: np.ndarray, block: int, rng) -> np.ndarray:
    """무빙블록 부트스트랩으로 원래 길이의 대체표본 1개를 만든다."""
    n = len(returns)
    if n == 0:
        return returns.copy()
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    max_start = n - block
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    pieces = [returns[s:s + block] for s in starts]
    return np.concatenate(pieces)[:n]


def _metrics_from_returns(r: np.ndarray, bars_per_year: int) -> dict:
    """재표집된 봉수익률에서 총수익·CAGR·Sharpe·MDD 산출."""
    if len(r) == 0:
        nan = float("nan")
        return {"total_return": nan, "CAGR": nan, "Sharpe": nan, "MDD": nan}
    equity = np.cumprod(1.0 + r)
    total_return = float(equity[-1] - 1.0)
    n = len(r)
    years = n / bars_per_year
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else float("nan")
    sd = r.std(ddof=0)
    sharpe = float((r.mean() / sd) * np.sqrt(bars_per_year)) if sd > 0 else float("nan")
    running_max = np.maximum.accumulate(equity)
    mdd = float((equity / running_max - 1.0).min())
    return {"total_return": total_return, "CAGR": cagr, "Sharpe": sharpe, "MDD": mdd}


def bootstrap_confidence_intervals(
    returns,
    bars_per_year: int,
    n_resamples: int = 1000,
    block: int | None = None,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """봉수익률 블록 부트스트랩으로 지표별 신뢰구간 + 손익확률.

    Parameters
    ----------
    returns : 봉별 수익률(equity.pct_change) 1차원.
    bars_per_year : 연환산 계수.
    n_resamples : 재표집 횟수.
    block : 블록 길이. None이면 sqrt(T) 근방.
    ci : 신뢰수준(0.95 → 2.5/97.5 백분위).
    seed : 재현용.

    Returns
    -------
    dict:
      metrics: {지표: {point, lo, hi, mean}} — total_return/CAGR/Sharpe/MDD.
      prob_positive: 재표집 총수익 > 0 비율.
      n_resamples, block, ci.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 8:
        keys = ["total_return", "CAGR", "Sharpe", "MDD"]
        nan = float("nan")
        return {
            "metrics": {k: {"point": nan, "lo": nan, "hi": nan, "mean": nan} for k in keys},
            "prob_positive": float("nan"), "n_resamples": 0,
            "block": 0, "ci": ci,
        }
    if block is None:
        block = max(2, int(round(np.sqrt(T))))
    rng = np.random.default_rng(seed)

    point = _metrics_from_returns(r, bars_per_year)
    dists: dict[str, list] = {k: [] for k in point}
    n_pos = 0
    for _ in range(n_resamples):
        rs = _moving_block_resample(r, block, rng)
        m = _metrics_from_returns(rs, bars_per_year)
        for k, v in m.items():
            dists[k].append(v)
        if m["total_return"] > 0:
            n_pos += 1

    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 + ci) / 2.0 * 100.0
    metrics = {}
    for k, vals in dists.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            metrics[k] = {"point": point[k], "lo": float("nan"),
                          "hi": float("nan"), "mean": float("nan")}
        else:
            metrics[k] = {
                "point": point[k],
                "lo": float(np.percentile(arr, lo_q)),
                "hi": float(np.percentile(arr, hi_q)),
                "mean": float(arr.mean()),
            }
    return {
        "metrics": metrics,
        "prob_positive": n_pos / n_resamples,
        "n_resamples": n_resamples,
        "block": block,
        "ci": ci,
    }
