"""시계열 train/test 분할 — 워크포워드 & purged K-fold.

일반 K-fold를 시계열에 쓰면 미래로 학습해 과거를 예측하는 누수가 생긴다.
여기서는 두 가지 시계열 전용 분할을 제공한다:

  walk_forward_splits  과거로 학습 → 미래로 검증 (anchored: 학습창이 누적,
                       rolling: 학습창 고정 슬라이딩). 실거래 재현에 가장 가깝다.
  purged_kfold_splits  Lopez de Prado식 purged K-fold. 각 폴드를 OOS로 쓰되,
                       테스트 구간과 겹치는(라벨 지평 horizon / embargo) 학습표본을
                       잘라내(purge) 인접 누수를 제거한다.

모든 분할은 (train_idx, test_idx) 정수 인덱스 배열 튜플의 리스트를 반환한다.
분할 자체는 인과성(train은 test보다 과거 또는 purge된 원거리)을 보장하지만,
지표 워밍업 편향 방지는 엔진(_warmup_bars)이 각 슬라이스 내부에서 책임진다.
"""
from __future__ import annotations

import numpy as np


def walk_forward_splits(
    n: int,
    n_splits: int = 5,
    train_min_frac: float = 0.5,
    mode: str = "rolling",
    embargo: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """워크포워드 분할.

    앞쪽 `train_min_frac` 비율을 최초 학습 블록으로 확보하고, 나머지 구간을
    `n_splits`개의 연속 OOS 폴드로 나눈다. 각 폴드마다:
      - anchored: 학습 = [0, test_start), 검증 = 그 다음 폴드. 학습창이 누적 성장.
      - rolling : 학습 = [test_start - L, test_start) (고정길이 L), 검증 = 다음 폴드.
    `embargo`>0이면 학습 끝과 검증 시작 사이에 그만큼의 봉을 비워 인접누수를 막는다.

    Parameters
    ----------
    n : 전체 봉 수.
    n_splits : OOS 폴드 수.
    train_min_frac : 최초 학습 블록이 차지하는 비율 (0<f<1).
    mode : "anchored" | "rolling".
    embargo : 학습-검증 사이 비우는 봉 수.
    """
    if mode not in ("anchored", "rolling"):
        raise ValueError(f"mode는 'anchored'|'rolling'만 허용: {mode!r}")
    if not (0.0 < train_min_frac < 1.0):
        raise ValueError("train_min_frac는 (0,1) 범위여야 한다")
    if n_splits < 1:
        raise ValueError("n_splits >= 1")

    train_min = int(round(n * train_min_frac))
    if train_min < 1 or train_min >= n:
        raise ValueError("train_min_frac로 확보되는 학습 블록이 비었거나 전체를 덮는다")

    remaining = n - train_min
    if remaining < n_splits:
        raise ValueError(
            f"검증 구간({remaining}봉)이 n_splits({n_splits})보다 작다 — 폴드가 빈다"
        )

    # 남은 구간을 n_splits개의 (가능한 균등) 연속 폴드 경계로 나눈다.
    edges = np.linspace(train_min, n, n_splits + 1, dtype=int)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        test_start, test_end = int(edges[i]), int(edges[i + 1])
        if test_end <= test_start:
            continue  # 반올림으로 빈 폴드가 생기면 건너뜀
        train_end = max(0, test_start - embargo)
        if mode == "anchored":
            train_start = 0
        else:  # rolling
            train_start = max(0, train_end - train_min)
        if train_end - train_start < 1:
            continue
        train_idx = np.arange(train_start, train_end, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)
        splits.append((train_idx, test_idx))
    return splits


def purged_kfold_splits(
    n: int,
    n_splits: int = 5,
    embargo: int = 0,
    horizon: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged K-fold (Lopez de Prado, AFML §7).

    인덱스를 `n_splits`개의 연속 폴드로 나눈다. 각 폴드를 test로 두고, 나머지 전부를
    train으로 쓰되 test와 시간적으로 인접해 정보가 새는 학습표본을 잘라낸다(purge):
      - test 시작 이전 `horizon`봉: 그 표본의 결과(라벨/보유지평)가 test 구간과
        겹치므로 학습에서 제거.
      - test 종료 이후 `embargo`봉: 자기상관 누수 차단용 완충.

    Parameters
    ----------
    n : 전체 봉 수.
    n_splits : 폴드 수(>=2).
    embargo : test 종료 후 학습에서 비우는 봉 수.
    horizon : test 시작 전 학습에서 비우는 봉 수(전략의 최대 보유지평 근사).
    """
    if n_splits < 2:
        raise ValueError("purged K-fold는 n_splits >= 2")
    if n_splits > n:
        raise ValueError("n_splits가 표본 수보다 크다")

    edges = np.linspace(0, n, n_splits + 1, dtype=int)
    all_idx = np.arange(n, dtype=int)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        test_start, test_end = int(edges[i]), int(edges[i + 1])
        if test_end <= test_start:
            continue
        test_idx = np.arange(test_start, test_end, dtype=int)

        # purge 구간: [test_start - horizon, test_end + embargo)
        purge_start = max(0, test_start - horizon)
        purge_end = min(n, test_end + embargo)
        purged = np.zeros(n, dtype=bool)
        purged[purge_start:purge_end] = True

        train_idx = all_idx[~purged]
        if train_idx.size < 1:
            continue
        splits.append((train_idx, test_idx))
    return splits
