"""자체 엔진 vs freqtrade 교차 비교 (무결성 검증).

같은 BTC/USDT 1h 데이터로:
  1) 자체 엔진(src.engine.backtest) 실행 → 진입 타임스탬프·요약지표
  2) freqtrade 백테스트 결과 json 파싱 → 진입 타임스탬프·요약지표
  3) 진입 타임스탬프 일치율(공유 enter_long이므로 ≈100% 기대) + 요약 비교 출력.

청산·P&L은 체결모델 차이로 정확히 같지 않다(설계상). 핵심 점검은 **진입 일치**다.

사전 준비:
    python -m src.run --market crypto            # BTC parquet 생성(외부망 환경)
    python scripts/to_freqtrade_data.py
    freqtrade backtesting -c freqtrade/config.json -s RossMacdRsiStrategy
사용:
    python scripts/compare_engines.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src import metrics
from src.data import cache
from src.engine import backtest
from src.engine.position import Costs

REPO_ROOT = Path(__file__).resolve().parents[1]
BT_RESULTS = REPO_ROOT / "freqtrade" / "user_data" / "backtest_results"


def run_self_engine(symbol: str = "BTC/USDT"):
    df = cache.load("crypto", symbol)
    if df is None:
        raise SystemExit("BTC parquet 캐시가 없습니다. 먼저 `python -m src.run --market crypto`.")
    cfg = yaml.safe_load(open(REPO_ROOT / "config" / "params.yaml", encoding="utf-8"))
    c = cfg["costs"]["crypto"]
    costs = Costs(fee=c["fee"], slippage=c["slippage"], sell_tax=c.get("sell_tax", 0.0))
    res = backtest.run_backtest(df, cfg["strategy"], costs,
                                initial_capital=cfg["backtest"]["initial_capital"])
    m = metrics.compute_metrics(res, bars_per_year=metrics.BARS_PER_YEAR_CRYPTO)
    entries = pd.to_datetime([t.entry_time for t in res.trades], utc=True).floor("h")
    return m, set(entries)


def load_freqtrade_result():
    last = BT_RESULTS / ".last_result.json"
    if not last.exists():
        raise SystemExit(
            f"freqtrade 결과 없음: {last}\n"
            f"먼저 `freqtrade backtesting -c freqtrade/config.json -s RossMacdRsiStrategy` 실행."
        )
    latest_name = json.loads(last.read_text())["latest_backtest"]
    data = json.loads((BT_RESULTS / latest_name).read_text())
    strat = next(iter(data["strategy"].values()))
    trades = strat.get("trades", [])
    entries = pd.to_datetime([t["open_date"] for t in trades], utc=True).floor("h")
    summary = {
        "n_trades": strat.get("total_trades", len(trades)),
        "total_return": strat.get("profit_total", float("nan")),
        "win_rate": (strat.get("wins", 0) / max(1, strat.get("total_trades", 1))),
    }
    return summary, set(entries)


def main():
    self_m, self_entries = run_self_engine()
    ft_m, ft_entries = load_freqtrade_result()

    inter = self_entries & ft_entries
    union = self_entries | ft_entries
    jaccard = len(inter) / len(union) if union else float("nan")

    print("================ 진입 타임스탬프 일치 ================")
    print(f"  자체 엔진 진입수 : {len(self_entries)}")
    print(f"  freqtrade 진입수 : {len(ft_entries)}")
    print(f"  공통(교집합)     : {len(inter)}")
    print(f"  자체 only        : {len(self_entries - ft_entries)}")
    print(f"  freqtrade only   : {len(ft_entries - self_entries)}")
    print(f"  Jaccard 일치율   : {jaccard:.3f}  (1.0 = 완전일치, enter_long 공유 → ≈1 기대)")
    print("\n================ 요약지표 (방향성 비교) ================")
    print(f"  {'항목':<14}{'자체엔진':>14}{'freqtrade':>14}")
    print(f"  {'거래수':<14}{self_m['n_trades']:>14}{ft_m['n_trades']:>14}")
    print(f"  {'총수익률':<12}{self_m['total_return']*100:>13.2f}%{ft_m['total_return']*100:>13.2f}%")
    print(f"  {'승률':<14}{self_m['win_rate']*100:>13.2f}%{ft_m['win_rate']*100:>13.2f}%")
    print("\n  ※ 청산·P&L 차이는 체결모델 차이(슬리피지/봉내체결)로 설계상 허용.")
    print("    핵심 점검은 진입 Jaccard 일치율이 1.0에 가까운지다.")


if __name__ == "__main__":
    main()
