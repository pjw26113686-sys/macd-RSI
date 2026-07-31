"""파라미터 스윕 + 워크포워드 재최적화 분석 (엔진 비의존).

두 산출물:
  run_sweep             N개 설정을 전기간 백테스트 → (T x N) 시점별 수익률 행렬 +
                        설정별 지표. 이 행렬이 pbo.cscv_pbo / deflated_sharpe의 입력.
  walk_forward_analysis 각 워크포워드 폴드에서 IS 최우수 설정을 뽑아 OOS로만 평가 →
                        IS→OOS 성능감쇠를 실거래에 가깝게 측정.

백엔드: 현재는 자체 엔진(src.engine.backtest, lookahead-safe). vectorbt 등 고속
백엔드는 동일 시그니처의 `_run_one`을 갈아끼우면 되도록 좁게 격리했다.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from src import metrics as _metrics
from src.engine import backtest as _bt
from src.engine.position import Costs
from src.validation.splits import walk_forward_splits


# --------------------------------------------------------------------------- #
# 그리드 전개
# --------------------------------------------------------------------------- #
def expand_grid(base_params: dict, grid: dict) -> list[dict]:
    """base_params 위에 grid의 데카르트 곱을 얹은 설정 리스트를 만든다.

    grid = {"macd_fast": [8, 12], "rsi_period": [9, 14]} → 4개 설정.
    grid가 비면 base_params 하나만 담긴 리스트.
    """
    if not grid:
        return [dict(base_params)]
    keys = list(grid.keys())
    combos = product(*[grid[k] for k in keys])
    out = []
    for values in combos:
        p = dict(base_params)
        p.update(dict(zip(keys, values)))
        out.append(p)
    return out


# --------------------------------------------------------------------------- #
# 단일 백테스트 (백엔드 격리 지점)
# --------------------------------------------------------------------------- #
def _run_one(
    df: pd.DataFrame, params: dict, costs: Costs,
    initial_capital: float, position_pct: float,
):
    """한 설정을 백테스트하고 BacktestResult를 반환. 백엔드 교체 시 이 함수만 바꾼다."""
    return _bt.run_backtest(
        df, params, costs,
        initial_capital=initial_capital, position_pct=position_pct,
    )


def _bar_returns(result) -> pd.Series:
    """봉별 수익률(=equity.pct_change). 워밍업/무포지션 구간은 0."""
    return result.equity.pct_change().fillna(0.0)


# --------------------------------------------------------------------------- #
# 전기간 스윕 → (T x N) 수익률 행렬
# --------------------------------------------------------------------------- #
def run_sweep(
    df: pd.DataFrame,
    params_list: list[dict],
    costs: Costs,
    bars_per_year: int,
    initial_capital: float = 10000.0,
    position_pct: float = 1.0,
) -> dict:
    """설정 리스트를 전기간 백테스트해 검증 입력을 만든다.

    Returns
    -------
    dict:
      returns   : (T x N) DataFrame — 열=설정 인덱스, 행=시각. cscv_pbo 입력.
      metrics   : 설정별 compute_metrics dict 리스트.
      sr_trials : 설정별 관측단위 Sharpe(무연환산) ndarray — deflated_sharpe 입력.
      params    : 입력 params_list 그대로(추적용).
    """
    if not params_list:
        raise ValueError("params_list가 비었다")

    ret_cols: dict[int, pd.Series] = {}
    metric_rows: list[dict] = []
    sr_trials: list[float] = []

    for i, params in enumerate(params_list):
        res = _run_one(df, params, costs, initial_capital, position_pct)
        rets = _bar_returns(res)
        ret_cols[i] = rets.reset_index(drop=True)
        metric_rows.append(_metrics.compute_metrics(res, bars_per_year=bars_per_year))
        # 관측단위(무연환산) Sharpe — DSR/CSCV 정합용.
        std = rets.std(ddof=0)
        sr_trials.append(float(rets.mean() / std) if std > 0 else 0.0)

    returns_df = pd.DataFrame(ret_cols)
    return {
        "returns": returns_df,
        "metrics": metric_rows,
        "sr_trials": np.asarray(sr_trials, dtype=float),
        "params": params_list,
    }


# --------------------------------------------------------------------------- #
# 워크포워드 재최적화 분석
# --------------------------------------------------------------------------- #
def _select_metric_value(m: dict, key: str) -> float:
    v = m.get(key, float("nan"))
    return v if v == v else float("-inf")  # NaN은 최악으로 취급(선택되지 않게)


def walk_forward_analysis(
    df: pd.DataFrame,
    params_list: list[dict],
    costs: Costs,
    bars_per_year: int,
    n_splits: int = 5,
    train_min_frac: float = 0.5,
    mode: str = "rolling",
    embargo: int = 0,
    select_by: str = "Sharpe",
    initial_capital: float = 10000.0,
    position_pct: float = 1.0,
) -> dict:
    """워크포워드: 각 폴드에서 IS 최우수 설정을 뽑아 OOS로만 평가.

    실거래 재현: 학습구간에서 `select_by`(기본 Sharpe)로 최우수를 고르고, 그 설정을
    검증구간에서만 돌린다. IS 평균성능과 OOS 평균성능의 차이가 과최적화 감쇠다.

    Returns
    -------
    dict:
      folds        : 폴드별 {is_metric, oos_metric, chosen_params, n_test} 리스트.
      is_mean      : 폴드 IS select_by 평균.
      oos_mean     : 폴드 OOS select_by 평균.
      degradation  : is_mean - oos_mean (양수 = OOS에서 성능 하락).
      oos_metrics  : OOS 지표들의 폴드 평균(요약).
    """
    n = len(df)
    splits = walk_forward_splits(
        n, n_splits=n_splits, train_min_frac=train_min_frac,
        mode=mode, embargo=embargo,
    )
    if not splits:
        raise ValueError("생성된 워크포워드 폴드가 없다 — 파라미터를 확인하라")

    folds = []
    is_vals, oos_vals = [], []
    oos_metric_acc: dict[str, list[float]] = {}

    for train_idx, test_idx in splits:
        train_df = df.iloc[train_idx[0]: train_idx[-1] + 1]
        test_df = df.iloc[test_idx[0]: test_idx[-1] + 1]

        # 학습구간에서 최우수 설정 선택.
        best_i, best_val, best_is_m = None, float("-inf"), None
        for i, params in enumerate(params_list):
            res = _run_one(train_df, params, costs, initial_capital, position_pct)
            m = _metrics.compute_metrics(res, bars_per_year=bars_per_year)
            val = _select_metric_value(m, select_by)
            if val > best_val:
                best_i, best_val, best_is_m = i, val, m

        # 선택 설정을 검증구간에서만 평가.
        oos_res = _run_one(
            test_df, params_list[best_i], costs, initial_capital, position_pct
        )
        oos_m = _metrics.compute_metrics(oos_res, bars_per_year=bars_per_year)

        is_v = _select_metric_value(best_is_m, select_by)
        oos_v = _select_metric_value(oos_m, select_by)
        is_vals.append(is_v)
        oos_vals.append(oos_v)
        for k, v in oos_m.items():
            if isinstance(v, (int, float)) and v == v:
                oos_metric_acc.setdefault(k, []).append(float(v))

        folds.append({
            "chosen_index": best_i,
            "chosen_params": params_list[best_i],
            "is_metric": best_is_m,
            "oos_metric": oos_m,
            "n_test": int(len(test_df)),
        })

    finite_is = [v for v in is_vals if np.isfinite(v)]
    finite_oos = [v for v in oos_vals if np.isfinite(v)]
    is_mean = float(np.mean(finite_is)) if finite_is else float("nan")
    oos_mean = float(np.mean(finite_oos)) if finite_oos else float("nan")

    return {
        "folds": folds,
        "select_by": select_by,
        "is_mean": is_mean,
        "oos_mean": oos_mean,
        "degradation": (is_mean - oos_mean) if (finite_is and finite_oos) else float("nan"),
        "oos_metrics": {k: float(np.mean(v)) for k, v in oos_metric_acc.items()},
        "n_folds": len(folds),
    }
