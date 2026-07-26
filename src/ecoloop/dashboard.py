"""Streamlit dashboard for Eco-Loop: live replay, comparison, and audit trail."""
from __future__ import annotations

import json
import math
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st

from .config import RunConfig


def _num(val, default=0):
    """Coerce None or missing values to a number for safe formatting."""
    return default if val is None else float(val)


_NUMERIC_KEYS = (
    "hvac_power_kw",
    "latency_ms",
    "total_steps",
    "estimated_energy_kwh",
    "peak_power_kw",
    "comfort_compliance_pct",
    "safety_clamped",
)


def _sanitize_event(event: dict) -> dict:
    """Replace None numeric values with 0 so format strings don't crash."""
    for key in _NUMERIC_KEYS:
        if key in event and event[key] is None:
            event[key] = 0
    return event


def _load_audit(run_dir: Path) -> list[dict]:
    """Load audit.jsonl events."""
    events = []
    audit_path = run_dir / "audit.jsonl"
    if not audit_path.exists():
        return events
    with open(audit_path, encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
                events.append(_sanitize_event(event))
            except json.JSONDecodeError:
                continue
    return events


def _load_metrics(run_dir: Path) -> dict | None:
    """Load metrics.json if it exists."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    if isinstance(metrics, dict):
        for key in _NUMERIC_KEYS:
            if key in metrics and metrics[key] is None:
                metrics[key] = 0
    return metrics


def _find_run_dirs(config: RunConfig) -> list[Path]:
    """Find all run output directories that contain an audit.jsonl."""
    if not config.paths.data_output.exists():
        return []
    return sorted(
        [d for d in config.paths.data_output.iterdir() if (d / "audit.jsonl").exists()],
        reverse=True,
    )


def render_dashboard(config: RunConfig) -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="Eco-Loop Dashboard",
        page_icon="",
        layout="wide",
    )

    # Custom CSS for a polished look
    st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    .stMetric { background: linear-gradient(135deg, #1e293b, #334155); padding: 12px; border-radius: 10px; }
    .stMetric label { color: #94a3b8 !important; font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a, #1e293b); }
    div[data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }
    h1 { color: #38bdf8 !important; }
    h2 { color: #7dd3fc !important; border-bottom: 1px solid #334155; padding-bottom: 0.3rem; }
    h3 { color: #bae6fd !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("Eco-Loop: Autonomous Building Controller")
    st.caption("Safety-governed LLM-driven HVAC optimization with EnergyPlus digital twin")

    run_dirs = _find_run_dirs(config)
    if not run_dirs:
        st.warning("No run output directories found. Run a simulation first with:")
        st.code("uv run ecoloop demo --mode compare", language="powershell")
        return

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Page",
        ["Control Loop", "Energy Comparison", "Comfort Analysis", "Audit Trail", "Architecture"],
    )

    # Run selector
    sim_runs = [d for d in run_dirs if d.name.startswith("sim-") or d.name.startswith("baseline-") or d.name.startswith("ai-")]
    if not sim_runs:
        sim_runs = run_dirs

    selected_dir = st.sidebar.selectbox(
        "Run Output",
        sim_runs,
        format_func=lambda p: p.name,
    )

    events = _load_audit(selected_dir)
    metrics = _load_metrics(selected_dir)

    # Sidebar summary
    if metrics:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Run Summary**")
        st.sidebar.metric("Steps", metrics.get("total_steps", 0))
        st.sidebar.metric("Energy", f"{metrics.get('estimated_energy_kwh', 0):.2f} kWh")
        st.sidebar.metric("Comfort", f"{metrics.get('comfort_compliance_pct', 0):.1f}%")
        mode = metrics.get("mode", "unknown")
        st.sidebar.metric("Mode", mode.upper())

    if not events:
        st.warning(f"No events in {selected_dir.name}")
        return

    # Pages
    if page == "Control Loop":
        _render_replay(events, metrics)
    elif page == "Energy Comparison":
        _render_energy_comparison(config, run_dirs)
    elif page == "Comfort Analysis":
        _render_comfort(events)
    elif page == "Audit Trail":
        _render_audit(events)
    else:
        _render_architecture()


def _render_replay(events: list[dict], metrics: dict | None) -> None:
    """Render the live/replay loop page with time-series charts."""
    st.header("Control Loop Replay")
    st.caption(f"{len(events)} control steps recorded")

    # Step slider
    selected_idx = st.slider("Control step", 0, len(events) - 1, len(events) - 1)
    event = events[selected_idx]

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Step", event["step"])
    col2.metric("Timestamp", event.get("timestamp", "")[:19])
    col3.metric("HVAC Power", f"{(event.get('hvac_power_kw') or 0):.1f} kW")
    col4.metric("Safety", event.get("safety_status", "N/A"))
    col5.metric("Outdoor", f"{event.get('outdoor_temp_c', 0):.1f} C")

    col1a, col2a, col3a, col4a = st.columns(4)
    col1a.metric("LLM Source", event.get("llm_source", "N/A"))
    col2a.metric("Model", event.get("llm_model", "N/A"))
    col3a.metric("Latency", f"{event.get('latency_ms', 0):.0f} ms")
    col4a.metric("Mode", event.get("mode", "N/A"))

    # Time-series charts
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
    except ImportError:
        st.warning("Plotly not installed. Install with: uv pip install plotly pandas")
        return

    # Zone temperatures over time
    st.subheader("Zone Temperatures Over Time")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Zone Temperatures (C)", "HVAC Power (kW)"),
                        vertical_spacing=0.1)

    steps_range = list(range(selected_idx + 1))
    zone_names = sorted(events[0].get("zone_temps", {}).keys()) if events else []

    colors = ["#38bdf8", "#34d399", "#fbbf24", "#f87171", "#a78bfa"]
    for i, zone in enumerate(zone_names):
        temps = [events[s].get("zone_temps", {}).get(zone, 0) for s in steps_range]
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(x=steps_range, y=temps, mode="lines", name=zone,
                                 line=dict(color=color, width=2), opacity=0.85), row=1, col=1)

    # Outdoor temp
    outdoor = [events[s].get("outdoor_temp_c", 0) for s in steps_range]
    fig.add_trace(go.Scatter(x=steps_range, y=outdoor, mode="lines", name="Outdoor",
                             line=dict(color="#94a3b8", width=2, dash="dot")), row=1, col=1)

    # HVAC power
    powers = [events[s].get("hvac_power_kw", 0) or 0 for s in steps_range]
    fig.add_trace(go.Scatter(x=steps_range, y=powers, mode="lines+markers", name="HVAC Power",
                             line=dict(color="#f472b6", width=2),
                             marker=dict(size=3)), row=2, col=1)

    fig.update_layout(height=500, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.8)")
    fig.update_xaxes(title_text="Control Step", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # LLM source distribution
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("LLM Source Distribution")
        sources = {}
        for e in events[:selected_idx + 1]:
            src = e.get("llm_source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        if sources:
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(sources.keys()),
                values=list(sources.values()),
                hole=0.4,
                marker=dict(colors=["#38bdf8", "#34d399", "#fbbf24", "#f87171"]),
            )])
            fig_pie.update_layout(height=300, template="plotly_dark",
                                  paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Safety Status Distribution")
        safety_dist = {}
        for e in events[:selected_idx + 1]:
            s = e.get("safety_status", "unknown")
            safety_dist[s] = safety_dist.get(s, 0) + 1
        if safety_dist:
            fig_bar = go.Figure(data=[go.Bar(
                x=list(safety_dist.keys()),
                y=list(safety_dist.values()),
                marker=dict(color=["#34d399" if k == "accepted" else "#fbbf24" if k == "clamped" else "#f87171"
                                   for k in safety_dist.keys()]),
            )])
            fig_bar.update_layout(height=300, template="plotly_dark",
                                  paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)

    # Action applied
    st.subheader("Action Applied")
    action = event.get("action")
    if action:
        zone_sps = action.get("zone_setpoints", {})
        if zone_sps:
            sp_data = []
            for zone, sp in sorted(zone_sps.items()):
                sp_data.append({
                    "Zone": zone,
                    "Heating (C)": sp.get("heating_c", "N/A"),
                    "Cooling (C)": sp.get("cooling_c", "N/A"),
                    "Deadband (C)": round(sp.get("cooling_c", 0) - sp.get("heating_c", 0), 1),
                })
            st.table(pd.DataFrame(sp_data))
        st.write(f"**Mode:** {action.get('mode', 'N/A')} | **Rationale:** {action.get('rationale', 'N/A')}")

    # Safety details
    reasons = event.get("safety_reasons", [])
    if reasons:
        st.write("**Safety reasons:**")
        for r in reasons:
            st.write(f"- {r}")

    # Run summary
    if metrics:
        st.subheader("Run Summary")
        cols = st.columns(6)
        cols[0].metric("Total Steps", metrics.get("total_steps", 0))
        cols[1].metric("Energy (kWh)", f"{metrics.get('estimated_energy_kwh', 0):.2f}")
        cols[2].metric("Comfort %", f"{metrics.get('comfort_compliance_pct', 0):.1f}%")
        cols[3].metric("LLM Primary", metrics.get("llm_calls_primary", 0))
        cols[4].metric("LLM Fallback", metrics.get("llm_calls_fallback", 0))
        cols[5].metric("Safety Clamped", metrics.get("safety_clamped", 0))


def _render_energy_comparison(config: RunConfig, run_dirs: list[Path]) -> None:
    """Render baseline vs AI energy comparison with Plotly."""
    st.header("Energy Comparison: Baseline vs AI")

    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
    except ImportError:
        st.warning("Plotly not installed. Run: uv pip install plotly pandas")
        _render_energy_basic(config, run_dirs)
        return

    # Find baseline and AI runs
    baseline_runs = sorted([d for d in run_dirs if d.name.startswith("baseline-")], reverse=True)
    ai_runs = sorted([d for d in run_dirs if d.name.startswith("ai-")], reverse=True)

    if not baseline_runs and not ai_runs:
        st.warning("No baseline or AI runs found. Run: uv run ecoloop demo --mode compare")
        return

    # Comparison metrics cards
    if baseline_runs and ai_runs:
        b_metrics = _load_metrics(baseline_runs[0])
        a_metrics = _load_metrics(ai_runs[0])

        if b_metrics and a_metrics:
            b_kwh = b_metrics.get("estimated_energy_kwh", 1)
            a_kwh = a_metrics.get("estimated_energy_kwh", 1)
            savings_pct = (b_kwh - a_kwh) / b_kwh * 100 if b_kwh > 0 else 0

            # Top-level savings banner
            if savings_pct > 0:
                st.success(f"AI controller achieved **{savings_pct:.1f}% energy savings** vs baseline")
            else:
                st.info(f"AI controller energy: {a_kwh:.2f} kWh vs baseline: {b_kwh:.2f} kWh")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Baseline Energy", f"{b_kwh:.2f} kWh")
            with col2:
                st.metric("AI Energy", f"{a_kwh:.2f} kWh", delta=f"{savings_pct:.1f}% saved" if savings_pct > 0 else None)
            with col3:
                b_comfort = b_metrics.get("comfort_compliance_pct", 0)
                a_comfort = a_metrics.get("comfort_compliance_pct", 0)
                st.metric("Baseline Comfort", f"{b_comfort:.1f}%")
            with col4:
                st.metric("AI Comfort", f"{a_comfort:.1f}%")

    # Time-series comparison chart
    st.subheader("HVAC Power Over Time")

    chart_data = []
    for run_dir, label, color in [
        (baseline_runs[0] if baseline_runs else None, "Baseline", "#3b82f6"),
        (ai_runs[0] if ai_runs else None, "AI Controller", "#22c55e"),
    ]:
        if run_dir is None:
            continue
        events = _load_audit(run_dir)
        if not events:
            continue

        times = list(range(len(events)))
        powers = [e.get("hvac_power_kw", 0) or 0 for e in events]
        chart_data.append((times, powers, label, color))

    if chart_data:
        fig = go.Figure()
        for times, powers, label, color in chart_data:
            fig.add_trace(go.Scatter(
                x=times, y=powers, mode="lines", name=label,
                line=dict(color=color, width=2), opacity=0.85,
                fill="tozeroy",
            ))
        fig.update_layout(
            title="HVAC Power Consumption Over Time",
            xaxis_title="Control Step",
            yaxis_title="Power (kW)",
            height=400,
            hovermode="x unified",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.8)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cumulative energy chart
    st.subheader("Cumulative Energy Consumption")
    if chart_data:
        fig2 = go.Figure()
        for times, powers, label, color in chart_data:
            cumulative = []
            total = 0.0
            for p in powers:
                total += p * 0.25  # 15-min = 0.25h
                cumulative.append(total)
            fig2.add_trace(go.Scatter(
                x=times, y=cumulative, mode="lines", name=label,
                line=dict(color=color, width=3),
            ))
        fig2.update_layout(
            title="Cumulative Energy (kWh)",
            xaxis_title="Control Step",
            yaxis_title="Cumulative Energy (kWh)",
            height=350,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.8)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Bar chart comparison
    if baseline_runs and ai_runs:
        b_metrics = _load_metrics(baseline_runs[0])
        a_metrics = _load_metrics(ai_runs[0])
        if b_metrics and a_metrics:
            st.subheader("Summary Comparison")
            col_l, col_r = st.columns(2)
            with col_l:
                df_energy = pd.DataFrame({
                    "Controller": ["Baseline", "AI"],
                    "kWh": [b_metrics.get("estimated_energy_kwh", 0), a_metrics.get("estimated_energy_kwh", 0)],
                })
                fig3 = px.bar(df_energy, x="Controller", y="kWh", color="Controller",
                              color_discrete_map={"Baseline": "#3b82f6", "AI": "#22c55e"},
                              title="Total Energy Consumption")
                fig3.update_layout(height=300, showlegend=False, template="plotly_dark",
                                   paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig3, use_container_width=True)

            with col_r:
                df_comfort = pd.DataFrame({
                    "Controller": ["Baseline", "AI"],
                    "Compliance %": [b_metrics.get("comfort_compliance_pct", 0), a_metrics.get("comfort_compliance_pct", 0)],
                })
                fig4 = px.bar(df_comfort, x="Controller", y="Compliance %", color="Controller",
                              color_discrete_map={"Baseline": "#3b82f6", "AI": "#22c55e"},
                              title="Comfort Compliance")
                fig4.update_layout(height=300, showlegend=False, template="plotly_dark",
                                   paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig4, use_container_width=True)


def _render_energy_basic(config: RunConfig, run_dirs: list[Path]) -> None:
    """Basic energy comparison without plotly."""
    baseline_runs = sorted([d for d in run_dirs if d.name.startswith("baseline-")], reverse=True)
    ai_runs = sorted([d for d in run_dirs if d.name.startswith("ai-")], reverse=True)

    if baseline_runs and ai_runs:
        b_metrics = _load_metrics(baseline_runs[0])
        a_metrics = _load_metrics(ai_runs[0])
        if b_metrics and a_metrics:
            import pandas as pd
            df = pd.DataFrame({
                "kWh": [b_metrics.get("estimated_energy_kwh", 0), a_metrics.get("estimated_energy_kwh", 0)]
            }, index=["Baseline", "AI Controller"])
            st.bar_chart(df)


def _render_comfort(events: list[dict]) -> None:
    """Render comfort analysis over time."""
    st.header("Comfort Analysis")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
    except ImportError:
        st.warning("Plotly not installed. Install with: uv pip install plotly")
        return

    times = []
    comfort_pcts = []
    pmv_data = {zone: [] for zone in events[0].get("zone_temps", {}).keys()} if events else {}

    for e in events:
        ts = datetime.fromisoformat(e["timestamp"])
        times.append(ts)
        zt = e.get("zone_temps", {})
        action = e.get("action", {})
        zone_sps = action.get("zone_setpoints", {}) if action else {}
        violations = 0
        total = len(zt)
        for zone, temp in zt.items():
            sp = zone_sps.get(zone, {})
            h_sp = sp.get("heating_c", 18)
            c_sp = sp.get("cooling_c", 28)
            if temp < h_sp or temp > c_sp:
                violations += 1
            # Use real PMV from log, fallback to estimate if missing
            pmv_est = e.get("zone_pmv", {}).get(zone, _estimate_pmv(temp, h_sp, c_sp))
            pmv_data.setdefault(zone, []).append(pmv_est)
        compliance = (total - violations) / total * 100 if total > 0 else 100
        comfort_pcts.append(compliance)

    # Comfort compliance chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Thermal Comfort Compliance (%)", "Estimated PMV by Zone"),
                        vertical_spacing=0.12)

    fig.add_trace(go.Scatter(
        x=times, y=comfort_pcts, mode="lines+markers", name="Compliance %",
        line=dict(color="#22c55e", width=3),
        marker=dict(size=3),
    ), row=1, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="#ef4444",
                  annotation_text="80% threshold", row=1, col=1)

    # Per-zone PMV
    colors = ["#38bdf8", "#34d399", "#fbbf24", "#f87171", "#a78bfa"]
    for i, (zone, vals) in enumerate(sorted(pmv_data.items())):
        if vals:
            fig.add_trace(go.Scatter(
                x=times, y=vals, mode="lines", name=zone,
                line=dict(color=colors[i % len(colors)], width=1.5),
                opacity=0.7,
            ), row=2, col=1)

    # Comfort band shading
    fig.add_hrect(y0=-0.5, y1=0.5, line_width=0, fillcolor="#22c55e", opacity=0.1,
                  annotation_text="Comfort Band", row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="#ef4444", row=2, col=1)
    fig.add_hline(y=-0.5, line_dash="dash", line_color="#ef4444", row=2, col=1)

    fig.update_layout(
        height=600,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.8)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(range=[0, 105], row=1, col=1)
    fig.update_yaxes(range=[-2, 2], row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Comfort summary table
    st.subheader("Comfort Summary by Zone")
    if events:
        last = events[-1]
        zt = last.get("zone_temps", {})
        action = last.get("action", {})
        zone_sps = action.get("zone_setpoints", {}) if action else {}
        rows = []
        for zone in sorted(zt.keys()):
            temp = zt[zone]
            sp = zone_sps.get(zone, {})
            h = sp.get("heating_c", 18)
            c = sp.get("cooling_c", 28)
            pmv = last.get("zone_pmv", {}).get(zone, _estimate_pmv(temp, h, c))
            status = "Comfortable" if -0.5 <= pmv <= 0.5 else "Uncomfortable"
            occ = last.get("occupancy", {}).get(zone, 0)
            rows.append({
                "Zone": zone,
                "Temp (C)": f"{temp:.1f}",
                "Heating SP": f"{h:.1f}",
                "Cooling SP": f"{c:.1f}",
                "Est. PMV": f"{pmv:.2f}",
                "Status": status,
                "Occupied": "Yes" if occ > 0 else "No",
            })
        st.table(pd.DataFrame(rows))


def _render_audit(events: list[dict]) -> None:
    """Render searchable audit trail."""
    st.header("Audit Trail")

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events", len(events))
    primary = sum(1 for e in events if e.get("llm_source") == "primary")
    fallback = sum(1 for e in events if e.get("llm_source") == "fallback")
    det = sum(1 for e in events if e.get("llm_source") == "deterministic")
    col2.metric("LLM Primary", primary)
    col3.metric("LLM Fallback", fallback)
    col4.metric("Deterministic", det)

    # Search/filter
    search = st.text_input("Filter events (contains text):", "")

    filtered = events
    if search:
        filtered = [e for e in events if search.lower() in json.dumps(e).lower()]

    st.caption(f"Showing {len(filtered)} of {len(events)} events")

    # CSV download
    try:
        import pandas as pd
        flat_events = []
        for e in filtered:
            flat = {
                "step": e.get("step"),
                "timestamp": e.get("timestamp", "")[:19],
                "mode": e.get("mode"),
                "llm_source": e.get("llm_source"),
                "llm_model": e.get("llm_model"),
                "latency_ms": e.get("latency_ms", 0),
                "safety_status": e.get("safety_status"),
                "hvac_power_kw": e.get("hvac_power_kw", 0),
                "outdoor_temp_c": e.get("outdoor_temp_c", 0),
            }
            zt = e.get("zone_temps", {})
            for zone, temp in zt.items():
                flat[f"temp_{zone}"] = temp
            flat_events.append(flat)
        if flat_events:
            df = pd.DataFrame(flat_events)
            csv_data = df.to_csv(index=False)
            st.download_button("Download as CSV", csv_data, "audit_log.csv", "text/csv")
    except ImportError:
        pass

    for event in filtered:
        ts = event.get("timestamp", "")[:19]
        source = event.get("llm_source", "N/A")
        safety = event.get("safety_status", "N/A")
        action = event.get("action", {})
        zone_sps = action.get("zone_setpoints", {}) if action else {}
        rationale = action.get("rationale", "") if action else ""

        # Color-coded status
        status_color = {"accepted": "green", "clamped": "orange", "fallback": "red"}.get(safety, "gray")

        with st.expander(f"Step {event['step']} -- {ts} -- :{status_color}[{safety}] -- {source}"):
            cols = st.columns(4)
            cols[0].write(f"**Mode:** {event.get('mode')}")
            cols[1].write(f"**LLM:** {event.get('llm_model')}")
            cols[2].write(f"**Latency:** {event.get('latency_ms', 0):.0f} ms")
            cols[3].write(f"**Power:** {(event.get('hvac_power_kw') or 0):.1f} kW")

            if zone_sps:
                st.write("**Zone Setpoints:**")
                for zone, sp in sorted(zone_sps.items()):
                    h = sp.get("heating_c", "N/A")
                    c = sp.get("cooling_c", "N/A")
                    st.write(f"  - {zone}: heating={h} C, cooling={c} C")

            if rationale:
                st.write(f"**Rationale:** {rationale}")

            reasons = event.get("safety_reasons", [])
            if reasons:
                st.write("**Safety Reasons:**")
                for r in reasons:
                    st.write(f"- {r}")


def _render_architecture() -> None:
    """Render system architecture document."""
    st.header("System Architecture")

    st.markdown("""
### Multi-Agent LLM Architecture

The system uses a **hierarchical multi-agent architecture** powered by LangGraph:

```
                    Building State (from EnergyPlus/Synthetic)
                                    |
                            StateReader + Evaluators
                    (comfort, energy, forecast, MPC, oscillation)
                                    |
                    +---------------+---------------+
                    |               |               |
               EnergyAgent    ComfortAgent    ForecastAgent
               (minimize kWh) (maintain PMV)  (pre-condition)
                    |               |               |
                    +-------+-------+-------+-------+
                            |
                    SupervisorAgent
                    (merge with priority:
                     Safety > Comfort > Energy)
                            |
                    SafetyEngine
                    (clamp, deadband, rate-limit)
                            |
                    Approved Setpoints
                            |
                    Audit Log (JSONL)
                            |
                    Dashboard + Metrics
```

### LLM Provider Chain

```
Azure OpenAI (gpt-5.4-mini)  -->  Ollama (Llama 3.1 8B)  -->  Deterministic Fallback
      (primary)                       (fallback)                    (hold setpoints)
```

### Safety Rules (Non-Negotiable)

| Rule | Limit |
|------|-------|
| Heating setpoint | 18-24 C |
| Cooling setpoint | 22-28 C |
| Minimum deadband | 2 C |
| Max delta per 15-min step | 0.5 C |
| Max delta per hour | 2.0 C |
| PMV comfort band | [-0.5, 0.5] (ASHRAE-55 Cat II) |

### MPC Optimizer

A simplified **3R-2C thermal network model** predicts zone temperatures over a 4-hour
horizon. The MPC optimizer enumerates setpoint candidates and scores each on:
- **Energy cost**: estimated HVAC power consumption
- **Comfort penalty**: deviation from comfort band (weighted 3x for occupied zones)

The MPC recommendation is included in the evaluator summary provided to the LLM agents.

### MCP Tools (Model Context Protocol)

The system exposes 8 tools via FastMCP for any MCP-compatible client:

| Tool | Description |
|------|-------------|
| `read_zone_telemetry` | Current zone temps, humidity, occupancy |
| `read_zone_history` | Recent control step history |
| `read_safety_limits` | Active safety engine constraints |
| `check_action_safety` | Validate a proposed action |
| `propose_setpoints` | Propose and validate setpoint changes |
| `run_simulation_comparison` | Full baseline vs AI comparison |
| `read_audit_log` | Recent audit log entries |
| `list_output_directories` | Available simulation outputs |
    """)


# Helpers

def _build_zone_history(events: list[dict], current_idx: int, key: str) -> "pd.DataFrame":
    """Build a DataFrame of zone values up to current_idx."""
    import pandas as pd

    zones = set()
    for e in events[: current_idx + 1]:
        if key == "zone_temps":
            zones.update(e.get(key, {}).keys())
        elif key == "action":
            action = e.get(key, {})
            zones.update(action.get("zone_setpoints", {}).keys())

    data = {}
    for zone in sorted(zones):
        values = []
        for e in events[: current_idx + 1]:
            if key == "zone_temps":
                values.append(e.get(key, {}).get(zone, None))
            elif key == "action":
                action = e.get("action", {})
                sp = action.get("zone_setpoints", {}).get(zone, {})
                values.append(sp.get("cooling_c", None))
        data[zone] = values

    return pd.DataFrame(data)


def _estimate_pmv(temp_c: float, heating_sp: float, cooling_sp: float) -> float:
    """Simplified PMV estimate from zone temperature vs setpoints.

    Returns a rough PMV estimate:
    - 0 when temp is at comfort band center
    - +/- 1 when temp is at extreme ends
    """
    mid = (heating_sp + cooling_sp) / 2.0
    bandwidth = max(cooling_sp - heating_sp, 1.0)
    deviation = (temp_c - mid) / (bandwidth / 2.0)
    return max(-2.0, min(2.0, deviation))
