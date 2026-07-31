"""오버피팅 방어(검증) 레이어 — 기관급 성능 판정의 핵심.

일반인용 실험 플랫폼이 가장 쉽게 거짓말하는 지점이 "백테스트가 좋아 보인다"이다.
이 패키지는 그 좋아 보임이 **우연/과최적화인지**를 숫자로 판정한다.

구성:
  splits.py  시계열 train/test 분할 (워크포워드 anchored·rolling, purged K-fold).
  sweep.py   파라미터 그리드 스윕 → (T x N) 수익률 행렬 + 워크포워드 재최적화 분석.
  pbo.py     CSCV 기반 오버피팅 확률(PBO), Deflated Sharpe, IS→OOS 성능감쇠.
  report.py  PBO/DSR/감쇠 → 신뢰/주의/기각 판정카드 (일반인용).

모든 함수는 기존 엔진(src.engine)의 lookahead-safe 규율을 그대로 소비한다.
"""
from src.validation.bootstrap import bootstrap_confidence_intervals
from src.validation.pbo import (
    cscv_pbo,
    deflated_sharpe_ratio,
    expected_max_sharpe_ratio,
)
from src.validation.splits import (
    purged_kfold_splits,
    walk_forward_splits,
)
from src.validation.sweep import (
    expand_grid,
    run_sweep,
    walk_forward_analysis,
)

__all__ = [
    "walk_forward_splits",
    "purged_kfold_splits",
    "expand_grid",
    "run_sweep",
    "walk_forward_analysis",
    "cscv_pbo",
    "deflated_sharpe_ratio",
    "expected_max_sharpe_ratio",
    "bootstrap_confidence_intervals",
]
