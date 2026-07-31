"""일반인용 Streamlit 대시보드 — 클릭만으로 전략 실험·검증.

실행:
    pip install -e ".[ui]"
    streamlit run app.py

계산은 전부 src.analysis.evaluate_strategy에 위임한다(이 파일은 렌더링만). 카테고리·
전략을 고르거나 .py 파일을 업로드해 무결성 게이트 + 오버피팅 판정 + 수익곡선을 본다.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src import strategies as strat
from src.analysis import evaluate_strategy
from src.engine.position import Costs
from src.strategy_check import load_spec_from_module
from src.validate import _bars_per_year, load_config, synthetic_ohlcv

st.set_page_config(page_title="퀀트 전략 검증", page_icon="📈", layout="wide")

_LEVEL_COLOR = {"신뢰": "🟢", "주의": "🟡", "기각": "🔴"}
_CHECK_ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "⛔"}


@st.cache_data
def _load_config():
    return load_config()


def _pick_spec(sidebar) -> object | None:
    """사이드바에서 전략 선택(등록 전략 또는 업로드 .py)."""
    source = sidebar.radio("전략 소스", ["등록된 전략", ".py 업로드"])
    if source == "등록된 전략":
        cat = sidebar.selectbox("카테고리", strat.categories())
        specs = {s.name: s for s in strat.by_category(cat)}
        name = sidebar.selectbox("전략", list(specs))
        return specs[name]
    up = sidebar.file_uploader("전략 .py 파일", type=["py"])
    if up is None:
        return None
    tmp = Path(tempfile.gettempdir()) / up.name
    tmp.write_bytes(up.getvalue())
    try:
        return load_spec_from_module(str(tmp))
    except Exception as e:  # noqa: BLE001
        sidebar.error(f"로드 실패: {e}")
        return None


def _render_integrity(result: dict):
    st.subheader("① 무결성 게이트")
    cols = st.columns(len(result["integrity"]))
    for col, c in zip(cols, result["integrity"]):
        col.metric(c.name, _CHECK_ICON[c.status], help=c.detail)
    if result["gate_ok"]:
        st.success("무결성 통과 — 성과 판정으로 진행")
    else:
        st.error("무결성 실패 — 이 전략은 신뢰할 수 없습니다(성과 판정 생략)")


def _render_verdict(result: dict):
    lvl = result["verdict_level"]
    st.subheader(f"② 성과 판정: {_LEVEL_COLOR.get(lvl, '')} {lvl}")
    st.caption(result["verdict_reason"])
    pbo, dsr = result["pbo"], result["dsr"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PBO (과최적화확률)", f"{pbo['pbo']*100:.1f}%", help="낮을수록 견고 · 20%↑ 주의")
    c2.metric("Deflated Sharpe", f"{dsr['dsr']*100:.1f}%", help="참Sharpe>우연 문턱 확률 · 95%↑ 유의")
    c3.metric("OOS 손실확률", f"{pbo['prob_oos_loss']*100:.1f}%")
    if result["wfa"]:
        c4.metric("IS→OOS 감쇠", f"{result['wfa']['degradation']:.2f}", help="양수=OOS 하락")


def _render_equity(result: dict):
    st.subheader("③ 최우수 설정 vs Buy & Hold")
    eq = result["equity"].rename("전략")
    bh = result["benchmark"].reindex(eq.index).rename("Buy & Hold")
    chart_df = pd.concat([eq, bh], axis=1)
    st.line_chart(chart_df)

    m = result["best_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총수익률", f"{m['total_return']*100:.1f}%")
    c2.metric("MDD", f"{m['MDD']*100:.1f}%")
    c3.metric("Sharpe", f"{m['Sharpe']:.2f}" if m["Sharpe"] == m["Sharpe"] else "n/a")
    c4.metric("거래횟수", f"{m['n_trades']}")
    final_bh = float(bh.iloc[-1])
    st.caption(f"전략 최종자산 {m['final_equity']:.0f} vs Buy&Hold {final_bh:.0f} · "
               f"설정 {result['n_configs']}개 스윕 중 최우수")

    ci = result.get("ci")
    if ci and ci.get("n_resamples"):
        st.subheader("④ 신뢰구간 (블록 부트스트랩)")
        st.caption(f"{ci['n_resamples']}회 재표집 · {ci['ci']:.0%} 구간 · "
                   f"구간이 0을 가로지르면 그 성과는 견고하지 않음")
        cim = ci["metrics"]

        def _band(key, is_pct=True):
            b = cim[key]
            f = (lambda x: f"{x*100:.1f}%") if is_pct else (lambda x: f"{x:.2f}")
            return f"{f(b['point'])}  [{f(b['lo'])}, {f(b['hi'])}]"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총수익률 (95% CI)", _band("total_return"))
        c2.metric("Sharpe (95% CI)", _band("Sharpe", is_pct=False))
        c3.metric("MDD (95% CI)", _band("MDD"))
        c4.metric("손익확률", f"{ci['prob_positive']*100:.1f}%")


def main():
    st.title("📈 퀀트 전략 실험·검증")
    st.caption("전략을 고르거나 가져와서, 과최적화가 아닌지 기관급 기준으로 판정합니다.")
    sb = st.sidebar
    sb.header("설정")
    spec = _pick_spec(sb)

    market = sb.selectbox("시장(비용/연환산)", ["crypto", "stock"])
    bars = sb.slider("합성 데이터 봉 수", 800, 6000, 3000, step=200)
    seed = sb.number_input("시드", value=7, step=1)
    sizing = sb.selectbox("포지션 사이징", ["fixed_fraction", "fixed_risk"])
    risk_pct = sb.slider("거래당 리스크 %(fixed_risk)", 0.1, 5.0, 1.0, step=0.1) / 100.0
    blocks = sb.slider("CSCV 블록 수(짝수)", 8, 20, 14, step=2)
    folds = sb.slider("워크포워드 폴드", 3, 8, 5)
    do_wfa = sb.checkbox("워크포워드 실행", value=True)

    sb.info("이 환경은 외부 시세가 차단되어 합성 데이터로 시연합니다. 실데이터는 "
            "data/ 캐시(parquet)를 넣으면 됩니다.")

    if spec is None:
        st.warning("전략을 선택하거나 .py를 업로드하세요.")
        return

    if not sb.button("▶ 검증 실행", type="primary"):
        st.info(f"선택됨: **{spec.name}** ({spec.category}) — {spec.description}")
        return

    cfg = _load_config()
    cost_cfg = cfg["costs"][market]
    costs = Costs(fee=cost_cfg["fee"], slippage=cost_cfg["slippage"],
                  sell_tax=cost_cfg.get("sell_tax", 0.0))
    bt_cfg = cfg["backtest"]
    df = synthetic_ohlcv(int(bars), seed=int(seed))

    with st.spinner("스윕·검증 중 …"):
        result = evaluate_strategy(
            spec, df, costs, bt_cfg, _bars_per_year(market),
            blocks=int(blocks), folds=int(folds), do_wfa=do_wfa,
            sizing=sizing, risk_pct=float(risk_pct))

    _render_integrity(result)
    if result["gate_ok"]:
        _render_verdict(result)
        _render_equity(result)


if __name__ == "__main__":
    main()
