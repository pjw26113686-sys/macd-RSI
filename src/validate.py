"""오버피팅 검증 CLI — 파라미터 스윕 → CSCV PBO + Deflated Sharpe + 워크포워드.

전략이 "좋아 보이는 것"이 우연/과최적화인지 판정한다. 단일 백테스트 리포트(src.run)
와 달리, 여러 설정을 시도했을 때의 다중검정 편향까지 반영한다.

사용:
    python -m src.validate --market crypto            # 캐시 있으면 실데이터, 없으면 합성
    python -m src.validate --synthetic --bars 3000    # 합성 데이터로 데모
    python -m src.validate --market stock --blocks 12 --folds 6

판정카드의 세 축:
    PBO            IS 최우수 설정이 OOS 중앙값 아래로 떨어지는 확률(과최적화 확률).
    Deflated SR    N번 시도의 우연 문턱을 관측 Sharpe가 넘는지(비정규성 보정 포함).
    워크포워드     실거래 재현식 IS→OOS 성능감쇠.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import metrics as _metrics
from src import strategies as strat
from src.engine.position import Costs
from src.validation import cscv_pbo, deflated_sharpe_ratio, expand_grid, run_sweep
from src.validation.report import format_validation_report
from src.validation.sweep import walk_forward_analysis

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "params.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def synthetic_ohlcv(n: int, seed: int = 7) -> pd.DataFrame:
    """레짐 전환 합성 OHLCV — 상승/하락/횡보가 번갈아 나오는 GBM.

    순수 랜덤워크보다 추세 구조가 있어 진입이 발생하고, 그럼에도 전략이 우연을
    넘지 못하도록 드리프트는 약하게. (네트워크 차단 환경 데모/회귀용)
    """
    rng = np.random.default_rng(seed)
    regime_len = max(50, n // 12)
    drift = np.zeros(n)
    i = 0
    while i < n:
        mu = rng.choice([0.0006, -0.0005, 0.0]) # 상승/하락/횡보
        j = min(n, i + regime_len)
        drift[i:j] = mu
        i = j
    vol = 0.01
    rets = drift + rng.normal(0.0, vol, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC", name="time")
    intrabar = np.abs(rng.normal(0.0, vol, n)) * close
    high = close + intrabar
    low = close - intrabar
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    low = np.minimum.reduce([low, open_, close])
    high = np.maximum.reduce([high, open_, close])
    volume = rng.integers(800, 2400, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _bars_per_year(market: str) -> int:
    return (_metrics.BARS_PER_YEAR_CRYPTO if market == "crypto"
            else _metrics.BARS_PER_YEAR_STOCK)


def _load_data(args, cfg):
    """실데이터 캐시 → 없으면 합성으로 폴백. (df, symbol, bars_per_year, market) 반환.

    합성 데이터라도 --market의 비용·연환산은 존중한다(예: stock이면 stock 비용).
    """
    if args.synthetic:
        return (synthetic_ohlcv(args.bars, seed=args.seed), "SYNTHETIC",
                _bars_per_year(args.market), args.market)

    from src.data import cache
    d = cfg["data"][args.market]
    symbol = d["symbol"]
    df = cache.load(args.market, symbol)
    if df is None:
        print(f"[!] {args.market}/{symbol} 캐시 없음 → 합성 데이터로 폴백 "
              f"(실데이터 검증은 외부망 환경에서 캐시 후 재실행).")
        return (synthetic_ohlcv(args.bars, seed=args.seed), "SYNTHETIC",
                _bars_per_year(args.market), args.market)
    df = cache.validate_ohlcv(df, args.market)
    return df, symbol, _bars_per_year(args.market), args.market


def apply_sizing(params_list: list[dict], bt_cfg: dict, args) -> list[dict]:
    """사이징 설정(config backtest + CLI 오버라이드)을 각 params에 주입.

    엔진 사이저는 params["sizing"]/["risk_pct"]를 읽으므로, 포트폴리오 레벨 설정을
    전략 params로 흘려보낸다. CLI(--sizing/--risk-pct)가 config를 덮어쓴다.
    """
    sizing = getattr(args, "sizing", None) or bt_cfg.get("sizing", "fixed_fraction")
    risk_pct = getattr(args, "risk_pct", None)
    if risk_pct is None:
        risk_pct = bt_cfg.get("risk_pct", 0.01)
    return [{**p, "sizing": sizing, "risk_pct": risk_pct} for p in params_list]


def print_catalog():
    """등록된 카테고리·전략 목록 출력."""
    print("등록된 전략 (카테고리별):")
    for cat in strat.categories():
        print(f"\n  [{cat}]")
        for s in strat.by_category(cat):
            print(f"    - {s.name:26s} {s.description}")
    print("\n사용: python -m src.validate --strategy <name> [--synthetic]")
    print("      python -m src.validate --category <category> [--synthetic]")


def validate_spec(spec, df, symbol, bpy, market, costs, bt, args):
    """한 전략 스펙을 스윕→PBO/DSR/워크포워드로 검증하고 판정카드를 출력."""
    params_list = apply_sizing(expand_grid(spec.default_params, spec.param_grid), bt, args)
    print(f"\n[i] {spec.name} ({spec.category}) · {symbol} · {len(df)}봉 · "
          f"설정 {len(params_list)}개 · 사이징 {params_list[0]['sizing']} 스윕 …")
    sweep = run_sweep(
        df, params_list, costs, bars_per_year=bpy,
        initial_capital=bt["initial_capital"], position_pct=bt["position_pct"],
        strategy=spec,
    )
    pbo_res = cscv_pbo(sweep["returns"].to_numpy(), n_blocks=args.blocks)

    best_i = int(np.argmax(sweep["sr_trials"]))
    dsr_res = deflated_sharpe_ratio(
        sweep["returns"].iloc[:, best_i].to_numpy(), sr_trials=sweep["sr_trials"],
    )

    wfa_res = None
    if not args.no_wfa:
        wfa_res = walk_forward_analysis(
            df, params_list, costs, bars_per_year=bpy,
            n_splits=args.folds, mode=args.wf_mode, embargo=args.embargo,
            initial_capital=bt["initial_capital"], position_pct=bt["position_pct"],
            strategy=spec,
        )

    title = f"{spec.category}/{spec.name}·{market.upper()}·{symbol}"
    print(format_validation_report(title, pbo_res, dsr_res, wfa_res))
    return pbo_res, dsr_res, wfa_res


def run_validation(args):
    cfg = load_config(Path(args.config))
    df, symbol, bpy, market = _load_data(args, cfg)

    cost_cfg = cfg["costs"][market]
    costs = Costs(fee=cost_cfg["fee"], slippage=cost_cfg["slippage"],
                  sell_tax=cost_cfg.get("sell_tax", 0.0))
    bt = cfg["backtest"]

    if args.category:
        specs = strat.by_category(args.category)
        if not specs:
            raise SystemExit(f"미등록 카테고리: {args.category!r}. "
                             f"가능: {strat.categories()}")
    else:
        specs = [strat.get(args.strategy)]

    results = {}
    for spec in specs:
        results[spec.name] = validate_spec(
            spec, df, symbol, bpy, market, costs, bt, args
        )
    return results


def main():
    ap = argparse.ArgumentParser(description="오버피팅 검증(PBO/DSR/워크포워드)")
    ap.add_argument("--strategy", default="momentum_macd_rsi",
                    help="검증할 전략 이름 (기본: momentum_macd_rsi)")
    ap.add_argument("--category", default=None,
                    help="카테고리 전체 검증 (momentum/mean_reversion/breakout)")
    ap.add_argument("--list", action="store_true", help="등록된 전략 목록 출력 후 종료")
    ap.add_argument("--market", choices=["crypto", "stock"], default="crypto")
    ap.add_argument("--synthetic", action="store_true", help="합성 데이터 강제 사용")
    ap.add_argument("--bars", type=int, default=3000, help="합성 데이터 봉 수")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--blocks", type=int, default=16, help="CSCV 블록 수(짝수)")
    ap.add_argument("--folds", type=int, default=5, help="워크포워드 폴드 수")
    ap.add_argument("--wf-mode", choices=["anchored", "rolling"], default="rolling")
    ap.add_argument("--embargo", type=int, default=0, help="학습-검증 완충 봉수")
    ap.add_argument("--no-wfa", action="store_true", help="워크포워드 생략(PBO/DSR만)")
    ap.add_argument("--sizing", choices=["fixed_fraction", "fixed_risk"], default=None,
                    help="포지션 사이징 (기본: config backtest.sizing)")
    ap.add_argument("--risk-pct", type=float, default=None, dest="risk_pct",
                    help="fixed_risk의 거래당 리스크 비율 (예: 0.01)")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args()

    if args.list:
        print_catalog()
        return
    run_validation(args)


if __name__ == "__main__":
    main()
