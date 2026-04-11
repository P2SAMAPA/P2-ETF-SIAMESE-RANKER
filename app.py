"""
app.py
P2 ETF Siamese Ranker — Streamlit Dashboard
Replicates SAMBA design: hero card, metrics panels, conviction charts, signal history.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P2 ETF Siamese Ranker",
    page_icon="🔁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — replicates SAMBA light/lavender design ──────────────────────
st.markdown("""
<style>
  /* Global */
  html, body, [data-testid="stAppViewContainer"] {
    background: #f9f9fb;
    font-family: 'Inter', sans-serif;
  }
  [data-testid="stMainBlockContainer"] { padding-top: 1.5rem; }

  /* Title */
  .p2-title { font-size: 1.85rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
  .p2-subtitle { font-size: 0.88rem; color: #888; margin-top: 0.15rem; margin-bottom: 1.2rem; }

  /* Hero card */
  .hero-card {
    background: #f0eeff;
    border-radius: 14px;
    padding: 2rem 2.5rem 1.6rem;
    margin-bottom: 1.5rem;
  }
  .hero-ticker { font-size: 3.2rem; font-weight: 800; color: #1a1a2e; line-height: 1; }
  .hero-conviction { font-size: 1.45rem; font-weight: 600; color: #6c63ff; margin-top: 0.4rem; }
  .hero-meta { font-size: 0.8rem; color: #888; margin-top: 0.5rem; }
  .hero-badge {
    display: inline-block; background: #e8e3ff; color: #5a52cc;
    border-radius: 20px; padding: 3px 14px; font-size: 0.75rem;
    font-weight: 600; margin-top: 0.6rem; letter-spacing: 0.03em;
  }
  .hero-runners { font-size: 0.88rem; color: #555; margin-top: 0.9rem; }
  .hero-runners span { font-weight: 600; color: #1a1a2e; }

  /* Section labels */
  .section-badge {
    display: inline-block; border: 1.5px solid #bbb; border-radius: 6px;
    padding: 3px 12px; font-size: 0.75rem; font-weight: 600;
    color: #555; letter-spacing: 0.06em; margin-bottom: 0.7rem;
  }
  .section-badge.sw { border-color: #6c63ff; color: #6c63ff; background: #f0eeff; }

  /* Metric cards */
  .metric-row { display: flex; gap: 12px; margin-bottom: 1.2rem; flex-wrap: wrap; }
  .metric-card {
    flex: 1; min-width: 100px;
    background: #fff; border: 1px solid #e8e8f0; border-radius: 10px;
    padding: 0.8rem 1rem; text-align: center;
  }
  .metric-label { font-size: 0.7rem; font-weight: 600; color: #999; letter-spacing: 0.07em; text-transform: uppercase; }
  .metric-value { font-size: 1.3rem; font-weight: 700; margin-top: 4px; }
  .metric-green { color: #16a34a; }
  .metric-red   { color: #dc2626; }
  .metric-black { color: #1a1a2e; }

  /* Window label */
  .window-pill {
    display: inline-block; background: #f0eeff; color: #6c63ff;
    border-radius: 20px; padding: 3px 12px; font-size: 0.75rem;
    font-weight: 500; margin-bottom: 0.7rem;
  }

  /* Chart footer */
  .chart-footer { font-size: 0.72rem; color: #aaa; margin-top: 0.2rem; }

  /* Signal history */
  .sig-hit-rate { font-size: 0.82rem; color: #888; margin-bottom: 0.4rem; }

  /* Divider */
  hr.p2-div { border: none; border-top: 1px solid #e8e8f0; margin: 1.4rem 0; }

  /* Streamlit tab overrides */
  [data-testid="stTabs"] button[data-baseweb="tab"] {
    font-weight: 600; font-size: 0.9rem; color: #888;
  }
  [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    color: #6c63ff; border-bottom: 2px solid #6c63ff;
  }
</style>
""", unsafe_allow_html=True)


# ── Data loading helpers ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_output(module: str) -> Optional[Dict]:
    """Load latest combined output from local file or HF."""
    local = f"outputs/{module}_output.json"
    if os.path.exists(local):
        with open(local) as f:
            return json.load(f)
    try:
        from hf_storage import pull_latest_ranking
        return pull_latest_ranking(module)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_backtest(module: str, mode: str) -> Optional[Dict]:
    local = f"outputs/backtest/{module}_{mode}.json"
    if os.path.exists(local):
        with open(local) as f:
            return json.load(f)
    try:
        from hf_storage import pull_backtest_results
        return pull_backtest_results(module, mode)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_signal_history(module: str) -> Optional[pd.DataFrame]:
    local = f"outputs/signal_history_{module}.csv"
    if os.path.exists(local):
        return pd.read_csv(local)
    try:
        from hf_storage import pull_signal_history
        return pull_signal_history(module)
    except Exception:
        return None


def parse_cum_returns(data: Dict) -> pd.DataFrame:
    """Convert cumulative return dicts to DataFrame."""
    strat = pd.Series(data.get("strategy", {}))
    bench = pd.Series(data.get("benchmark", {}))
    strat.index = pd.to_datetime(strat.index)
    bench.index = pd.to_datetime(bench.index)
    df = pd.DataFrame({"Strategy": strat, "Benchmark": bench}).sort_index()
    return df


# ── UI Components ─────────────────────────────────────────────────────────────

def render_hero_card(output_data: Dict, module_label: str):
    """Render the top ETF hero card (replicates SAMBA style)."""
    if not output_data:
        st.info("No signal available. Run predict to generate a ranking.")
        return

    # Prefer shrinking window output for hero
    sw = output_data.get("shrinking_window", output_data.get("fixed_split", {}))
    fs = output_data.get("fixed_split", sw)

    top_etf = sw.get("top_pick", "—")
    conviction = sw.get("top_conviction", 0)
    signal_date = sw.get("signal_date", "—")
    generated = sw.get("generated_utc", "—")
    source = sw.get("source", "shrinking_window").replace("_", " ").title()
    horizon = sw.get("best_horizon_days", "—")
    backend = sw.get("model_backend", "siamese").upper()

    second = sw.get("second")
    third = sw.get("third")

    runners_html = ""
    if second:
        runners_html += f"2nd: <span>{second['etf']}</span> {second['score']*100:.1f}%&nbsp;&nbsp;"
    if third:
        runners_html += f"3rd: <span>{third['etf']}</span> {third['score']*100:.1f}%"

    st.markdown(f"""
    <div class="hero-card">
      <div class="hero-ticker">{top_etf}</div>
      <div class="hero-conviction">{conviction*100:.1f}% conviction</div>
      <div class="hero-meta">
        Signal for {signal_date} &nbsp;·&nbsp; Generated {generated}
        &nbsp;·&nbsp; H={horizon}d &nbsp;·&nbsp; {backend}
      </div>
      <span class="hero-badge">Source: {source}</span>
      <div class="hero-runners">{runners_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value, color: str = "black"):
    color_cls = f"metric-{color}"
    if isinstance(value, float):
        fmt_val = f"{value*100:.1f}%" if abs(value) < 10 else f"{value:.2f}"
    else:
        fmt_val = str(value) if value is not None else "—"
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {color_cls}">{fmt_val}</div>
    </div>"""


def render_metrics_row(metrics: Dict):
    ann_ret = metrics.get("ann_return")
    ann_vol = metrics.get("ann_vol")
    sharpe = metrics.get("sharpe")
    max_dd = metrics.get("max_dd")
    hit_rate = metrics.get("hit_rate")

    ann_color = "green" if ann_ret and ann_ret > 0 else "red"
    sharpe_color = "green" if sharpe and sharpe > 0 else "red"
    dd_color = "red"

    cards = (
        render_metric_card("ANN RETURN", ann_ret, ann_color)
        + render_metric_card("ANN VOL", ann_vol, "black")
        + render_metric_card("SHARPE", sharpe, sharpe_color)
        + render_metric_card("MAX DD (PEAK→TROUGH)", max_dd, dd_color)
        + render_metric_card("HIT RATE", hit_rate, "black")
    )
    st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)


def render_cum_chart(cum_df: pd.DataFrame, strategy_label: str, bench_label: str):
    """Render cumulative return line chart — purple solid + grey dashed."""
    import plotly.graph_objects as go

    fig = go.Figure()

    if "Strategy" in cum_df.columns:
        fig.add_trace(go.Scatter(
            x=cum_df.index, y=cum_df["Strategy"],
            name=f"RANKER ({strategy_label})",
            line=dict(color="#6c63ff", width=2),
            hovertemplate="%{y:.3f}<extra></extra>",
        ))

    if "Benchmark" in cum_df.columns:
        fig.add_trace(go.Scatter(
            x=cum_df.index, y=cum_df["Benchmark"],
            name=bench_label,
            line=dict(color="#aaa", width=1.5, dash="dot"),
            hovertemplate="%{y:.3f}<extra></extra>",
        ))

    fig.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=11)),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_stacked_conviction_chart(ranking: List[Dict], module: str):
    """Render stacked conviction scores bar chart for all ETFs."""
    if not ranking:
        st.info("No conviction data available.")
        return

    import plotly.graph_objects as go

    etfs = [r["etf"] for r in ranking]
    scores = [r["score"] * 100 for r in ranking]

    colors = ["#6c63ff" if i == 0 else f"rgba(108,99,255,{max(0.15, 0.85 - i*0.07)})" for i in range(len(etfs))]

    fig = go.Figure(go.Bar(
        x=etfs,
        y=scores,
        marker_color=colors,
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Conviction %"),
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_signal_history(module: str):
    """Render signal history table."""
    df = load_signal_history(module)

    if df is None or len(df) == 0:
        # Show placeholder row if no history
        df = pd.DataFrame([{
            "date": datetime.now().strftime("%Y-%m-%d"),
            "pick": "—",
            "conviction": "—",
            "actual_return": "—",
            "hit": "—",
        }])
    else:
        df = df.copy()
        if "actual_return" in df.columns:
            df["hit"] = df.apply(
                lambda r: "✅" if (r.get("actual_return") is not None and r["actual_return"] > 0)
                else ("❌" if (r.get("actual_return") is not None) else "—"),
                axis=1
            )
        # Format
        for col in ["conviction"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{float(x)*100:.1f}%" if x not in [None, "—", ""] else "—")
        if "actual_return" in df.columns:
            df["actual_return"] = df["actual_return"].apply(
                lambda x: f"{float(x)*100:.2f}%" if x not in [None, "—", ""] else "—"
            )

    n_hits = len(df[df.get("hit", pd.Series()) == "✅"]) if "hit" in df.columns else 0
    n_total = len(df[df.get("actual_return", pd.Series()) != "—"]) if "actual_return" in df.columns else 0
    hit_pct = f"{n_hits/n_total*100:.1f}%" if n_total > 0 else "0.0%"

    st.markdown(f'<div class="sig-hit-rate">Hit rate: <b>{hit_pct}</b> ({n_hits}/{n_total} signals)</div>', unsafe_allow_html=True)

    display_cols = [c for c in ["date", "pick", "conviction", "actual_return", "hit"] if c in df.columns]
    st.dataframe(
        df[display_cols].sort_values("date", ascending=False) if "date" in df.columns else df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": "Date",
            "pick": "Pick",
            "conviction": "Conviction",
            "actual_return": "Actual Return",
            "hit": "Hit",
        }
    )


def render_backtest_panels(
    module: str,
    benchmark_label: str,
    output_data: Optional[Dict],
):
    """Render Fixed Split + Shrinking Window backtest panels side by side."""
    fs_bt = load_backtest(module, "fixed_split")
    sw_bt = load_backtest(module, "shrinking_window")

    col_left, col_right = st.columns(2)

    # ── Fixed Split ────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-badge">FIXED SPLIT (80/10/10)</div>', unsafe_allow_html=True)

        if fs_bt:
            test_start = fs_bt.get("test_start", "—")
            test_end = fs_bt.get("test_end", "—")
            st.markdown(f'<div class="window-pill">Test: {test_start} → {test_end}</div>', unsafe_allow_html=True)
            render_metrics_row(fs_bt.get("metrics", {}))

            cum = fs_bt.get("cumulative_returns", {})
            if cum:
                cum_df = parse_cum_returns(cum)
                fs_top = output_data.get("fixed_split", {}).get("top_pick", "Strategy") if output_data else "Strategy"
                render_cum_chart(cum_df, fs_top, benchmark_label)

            st.markdown(
                f'<div class="chart-footer">Test Return: {fs_bt.get("metrics", {}).get("ann_return", 0)*100:.2f}% · '
                f'Sharpe: {fs_bt.get("metrics", {}).get("sharpe", 0):.3f}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("Fixed split backtest not yet available. Run predict.py to generate.")

    # ── Shrinking Window ───────────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-badge sw">SHRINKING WINDOW</div>', unsafe_allow_html=True)

        if sw_bt and sw_bt.get("window_results"):
            # Show the most recent window
            latest_window = sw_bt["window_results"][-1]
            window_label = latest_window.get("window", "—")
            oos_start = latest_window.get("oos_start", "—")
            oos_end = latest_window.get("oos_end", "—")
            st.markdown(
                f'<div class="window-pill">Window: {window_label} · OOS: {oos_start} → {oos_end}</div>',
                unsafe_allow_html=True
            )
            render_metrics_row(latest_window.get("metrics", {}))

            cum = latest_window.get("cumulative_returns", {})
            if cum:
                cum_df = parse_cum_returns(cum)
                sw_top = output_data.get("shrinking_window", {}).get("top_pick", "Strategy") if output_data else "Strategy"
                render_cum_chart(cum_df, sw_top, benchmark_label)

            consensus_etf = sw_bt.get("consensus_etf", "—")
            st.markdown(
                f'<div class="chart-footer">Consensus ETF: <b>{consensus_etf}</b> · '
                f'Windows: {sw_bt.get("n_windows", 0)}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("Shrinking window backtest not yet available. Run predict.py to generate.")


def render_module_tab(module: str, benchmark_label: str, module_label: str):
    """Render full tab content for one module."""
    output_data = load_output(module)

    # Hero card
    render_hero_card(output_data, module_label)

    st.markdown('<hr class="p2-div">', unsafe_allow_html=True)

    # Conviction scores — stacked bar
    st.markdown("**Conviction Scores — Current Ranking**")
    if output_data:
        sw = output_data.get("shrinking_window", output_data.get("fixed_split", {}))
        ranking = sw.get("ranking", [])
        render_stacked_conviction_chart(ranking, module)
    else:
        st.info("No ranking data available.")

    st.markdown('<hr class="p2-div">', unsafe_allow_html=True)

    # Backtest panels
    render_backtest_panels(module, benchmark_label, output_data)

    st.markdown('<hr class="p2-div">', unsafe_allow_html=True)

    # Signal history
    st.markdown("**Signal History**")
    render_signal_history(module)


# ── Sidebar — manual run controls ────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        st.caption("Run pipeline manually or configure settings.")

        module_choice = st.selectbox("Module", ["Both", "FI / Alts", "Equity Sectors"])
        backend_choice = st.selectbox("Backend", ["Auto (Siamese → LGBM)", "Force LightGBM"])
        run_backtest = st.checkbox("Run full backtest", value=True)

        force_lgbm = backend_choice == "Force LightGBM"

        if st.button("▶ Run Now", type="primary", use_container_width=True):
            with st.spinner("Running pipeline..."):
                try:
                    import sys
                    sys.path.insert(0, ".")

                    if module_choice in ["Both", "FI / Alts"]:
                        from fi_predict import main as fi_main
                        fi_main(run_backtest=run_backtest, force_lgbm=force_lgbm)

                    if module_choice in ["Both", "Equity Sectors"]:
                        from equity_predict import main as eq_main
                        eq_main(run_backtest=run_backtest, force_lgbm=force_lgbm)

                    st.cache_data.clear()
                    st.success("Pipeline complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Pipeline error: {e}")

        st.markdown("---")
        st.caption("**Data source**")
        st.caption("P2SAMAPA/fi-etf-macro-signal-master-data")
        st.caption("**Results**")
        st.caption("P2SAMAPA/p2-etf-siamese-ranker-results")

        st.markdown("---")
        last_run = "—"
        for module in ["fi", "equity"]:
            p = f"outputs/{module}_output.json"
            if os.path.exists(p):
                with open(p) as f:
                    d = json.load(f)
                generated = d.get("fixed_split", {}).get("generated_utc", "—")
                if generated != "—":
                    last_run = generated
                    break
        st.caption(f"Last run: {last_run}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    # Header
    st.markdown('<div class="p2-title">P2 ETF Siamese Ranker</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="p2-subtitle">Cross-Sectional Siamese Ranking &nbsp;·&nbsp; '
        'Relative Alpha Engine &nbsp;·&nbsp; FI + Equity Modules</div>',
        unsafe_allow_html=True
    )

    # Module tabs
    tab_fi, tab_eq = st.tabs(["🔵 Option A — Fixed Income / Alts", "🔵 Option B — Equity Sectors"])

    with tab_fi:
        render_module_tab("fi", "AGG", "FI / Alts")

    with tab_eq:
        render_module_tab("equity", "SPY", "Equity Sectors")


if __name__ == "__main__":
    main()
