"""오버피팅 확률(PBO) + Deflated Sharpe Ratio.

참고: Bailey, Borwein, Lopez de Prado, Zhu — "The Probability of Backtest
Overfitting"(2014), 그리고 Bailey & Lopez de Prado — "The Deflated Sharpe
Ratio"(2014).

핵심 질문 두 개:
  1) 여러 설정 중 IS(학습구간)에서 제일 좋아 보인 설정이, OOS(검증구간)에서도
     좋은가? — 아니라면 그 "좋아 보임"은 과최적화다. → CSCV로 PBO 산출.
  2) N개 설정을 시도하면 순전히 운으로도 높은 Sharpe가 나온다. 관측 Sharpe가
     그 우연 기대치를 넘는가? → Deflated Sharpe Ratio.

CSCV(Combinatorially Symmetric Cross-Validation):
  (T x N) 수익률 행렬을 S개의 연속 블록으로 나누고, S/2개를 IS로 뽑는 모든 조합에
  대해 IS 최우수 설정을 OOS에서 순위매김한다. IS 최우수가 OOS 중앙값 아래로
  떨어지는(=logit<=0) 조합의 비율이 PBO다. 0.5면 무작위(순전히 과최적화),
  0에 가까울수록 견고.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy import stats

_EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------- #
# CSCV → PBO
# --------------------------------------------------------------------------- #
def _block_sharpe_stats(returns: np.ndarray, n_blocks: int):
    """블록별 (count, sum, sumsq)를 미리 계산 — 조합마다 O(N)으로 IS/OOS Sharpe 산출.

    returns: (T, N). T를 n_blocks로 나누어 떨어지게 앞에서 자른다.
    반환: cnt(S,), s1(S, N), s2(S, N).
    """
    T, N = returns.shape
    block_len = T // n_blocks
    if block_len < 1:
        raise ValueError(f"관측 수 T={T}가 블록 수 S={n_blocks}보다 작다")
    usable = block_len * n_blocks
    r = returns[:usable]
    r_blocks = r.reshape(n_blocks, block_len, N)
    cnt = np.full(n_blocks, block_len, dtype=float)
    s1 = r_blocks.sum(axis=1)                    # (S, N)
    s2 = (r_blocks ** 2).sum(axis=1)             # (S, N)
    return cnt, s1, s2


def _sharpe_from_agg(cnt: float, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """집계된 (count, sum, sumsq)에서 설정별 Sharpe(관측단위, 무연환산) 계산."""
    mean = s1 / cnt
    var = np.maximum(s2 / cnt - mean ** 2, 0.0)
    std = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(std > 0, mean / std, 0.0)
    return sr


def cscv_pbo(returns: np.ndarray, n_blocks: int = 16) -> dict:
    """CSCV로 오버피팅 확률(PBO)을 계산.

    Parameters
    ----------
    returns : (T, N) ndarray — T개 시점 x N개 설정의 시점별 수익률.
    n_blocks : 짝수 블록 수 S. C(S, S/2) 조합을 평가한다(S=16 → 12870).

    Returns
    -------
    dict:
      pbo            : IS 최우수가 OOS 중앙값 아래인 조합 비율 (0~1). 낮을수록 견고.
      logits         : 조합별 relative-rank logit 배열.
      prob_oos_loss  : IS 최우수 설정이 OOS에서 손실(Sharpe<0)일 확률.
      perf_degradation: IS 최우수의 (IS Sharpe, OOS Sharpe) 회귀 기울기.
                        1이면 감쇠 없음, <0이면 IS 성능이 OOS 손실로 뒤집힘.
      n_combos       : 평가한 조합 수.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2:
        raise ValueError("returns는 (T, N) 2차원이어야 한다")
    T, N = returns.shape
    if N < 2:
        raise ValueError("설정(N)이 최소 2개 필요하다 — 순위매김 대상")
    if n_blocks % 2 != 0 or n_blocks < 2:
        raise ValueError("n_blocks는 2 이상 짝수여야 한다")

    cnt, s1, s2 = _block_sharpe_stats(returns, n_blocks)
    total_cnt = cnt[0] * n_blocks  # 블록 등길이

    block_ids = range(n_blocks)
    logits: list[float] = []
    is_best_sr: list[float] = []
    oos_of_best: list[float] = []

    for is_blocks in combinations(block_ids, n_blocks // 2):
        is_mask = np.zeros(n_blocks, dtype=bool)
        is_mask[list(is_blocks)] = True

        is_cnt = cnt[is_mask].sum()
        oos_cnt = total_cnt - is_cnt
        is_sr = _sharpe_from_agg(is_cnt, s1[is_mask].sum(0), s2[is_mask].sum(0))
        oos_sr = _sharpe_from_agg(oos_cnt, s1[~is_mask].sum(0), s2[~is_mask].sum(0))

        n_star = int(np.argmax(is_sr))
        # OOS에서 n_star의 상대순위 w ∈ (0,1). 평균순위로 동점 처리.
        rank = float(stats.rankdata(oos_sr)[n_star])  # 1..N
        w = rank / (N + 1.0)
        w = min(max(w, 1e-6), 1.0 - 1e-6)
        logits.append(math.log(w / (1.0 - w)))
        is_best_sr.append(float(is_sr[n_star]))
        oos_of_best.append(float(oos_sr[n_star]))

    logits_arr = np.asarray(logits)
    oos_best_arr = np.asarray(oos_of_best)
    is_best_arr = np.asarray(is_best_sr)

    pbo = float(np.mean(logits_arr <= 0.0))
    prob_oos_loss = float(np.mean(oos_best_arr < 0.0))

    # IS 최우수 성능 → OOS 성능 회귀 기울기 (감쇠 진단).
    if np.std(is_best_arr) > 0:
        slope = float(np.polyfit(is_best_arr, oos_best_arr, 1)[0])
    else:
        slope = float("nan")

    return {
        "pbo": pbo,
        "logits": logits_arr,
        "prob_oos_loss": prob_oos_loss,
        "perf_degradation": slope,
        "n_combos": len(logits),
    }


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio
# --------------------------------------------------------------------------- #
def expected_max_sharpe_ratio(sr_variance: float, n_trials: int) -> float:
    """N번 시도했을 때 참 Sharpe=0인데도 순전히 운으로 기대되는 최대 Sharpe.

    E[max SR] ≈ sqrt(Var(SR)) * [ (1-γ)·Z⁻¹(1-1/N) + γ·Z⁻¹(1-1/(N·e)) ]
    (γ = 오일러-마스케로니 상수, Z⁻¹ = 표준정규 역CDF). 관측단위 Sharpe 기준.
    """
    if n_trials < 1:
        raise ValueError("n_trials >= 1")
    if sr_variance < 0:
        raise ValueError("sr_variance >= 0")
    if n_trials == 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_variance) * (
        (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
    )


def deflated_sharpe_ratio(
    returns: np.ndarray,
    sr_trials: np.ndarray | None = None,
    n_trials: int | None = None,
    sr_benchmark: float | None = None,
) -> dict:
    """Deflated Sharpe Ratio — 다중검정 + 비정규성 보정 후 참 Sharpe>기준일 확률.

    DSR = Z[ (SR̂ - SR₀)·sqrt(T-1) / sqrt(1 - γ₃·SR̂ + (γ₄-1)/4·SR̂²) ]
    SR̂ = 선택 전략의 관측단위 Sharpe, SR₀ = 다중검정 기대 최대치(우연 문턱),
    γ₃/γ₄ = 수익률의 왜도/첨도(정규=3), T = 관측 수.

    Parameters
    ----------
    returns : 선택된(최우수) 전략의 시점별 수익률 1차원 배열.
    sr_trials : 스윕에서 나온 설정별 관측단위 Sharpe 배열. 주면 이로부터 SR₀ 추정.
    n_trials : sr_trials 없이 시도 수만 알 때. 이때 SR₀는 SR̂의 분산 대용으로 근사.
    sr_benchmark : 비교 기준 Sharpe(관측단위). 기본은 SR₀(우연 문턱).

    Returns
    -------
    dict: dsr, sr_observed, sr0(우연 문턱), n_trials, skew, kurtosis.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = r.size
    if T < 4:
        return {
            "dsr": float("nan"), "sr_observed": float("nan"),
            "sr0": float("nan"), "n_trials": n_trials or 1,
            "skew": float("nan"), "kurtosis": float("nan"),
        }

    std = r.std(ddof=1)
    sr_hat = float(r.mean() / std) if std > 0 else 0.0
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # 정규=3

    # 우연 문턱 SR₀ 추정.
    if sr_trials is not None:
        srt = np.asarray(sr_trials, dtype=float)
        srt = srt[np.isfinite(srt)]
        N = max(int(srt.size), 1)
        sr_var = float(np.var(srt, ddof=1)) if srt.size > 1 else 0.0
    elif n_trials is not None:
        N = max(int(n_trials), 1)
        sr_var = sr_hat ** 2  # 시행 분산 미지 시 보수적 근사
    else:
        N = 1
        sr_var = 0.0
    sr0 = expected_max_sharpe_ratio(sr_var, N)

    threshold = sr0 if sr_benchmark is None else float(sr_benchmark)

    denom = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat ** 2)
    if denom <= 0:
        dsr = float("nan")
    else:
        z = (sr_hat - threshold) * math.sqrt(T - 1) / math.sqrt(denom)
        dsr = float(stats.norm.cdf(z))

    return {
        "dsr": dsr,
        "sr_observed": sr_hat,
        "sr0": float(sr0),
        "n_trials": N,
        "skew": skew,
        "kurtosis": kurt,
    }
