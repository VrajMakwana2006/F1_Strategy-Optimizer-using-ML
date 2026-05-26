"""
F1 Pit Wall Pro — Master Dashboard
===================================
Run with:   streamlit run dashboard.py
"""

import os, time, math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

os.makedirs("f1_api_cache", exist_ok=True)

import fastf1
fastf1.Cache.enable_cache("f1_api_cache")

from data_ingestion   import fetch_global_clean_dataset
from tire_model       import AdvancedRacePaceModel
from realtime_engine  import UniversalStrategyEngine

# ─────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Pit Wall Pro",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* ---- Fonts ---- */
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

  html, body, [class*=\"css\"] {
    font-family: 'Rajdhani', sans-serif;
    background: #070a0f !important;
    color: #e8ecf0;
  }

  /* ---- Sidebar ---- */
  section[data-testid="stSidebar"] {
    background: #0b0f17 !important;
    border-right: 1px solid #1c2535;
  }
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] p {
    color: #8fa3bb !important;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  section[data-testid="stSidebar"] .stTextInput input,
  section[data-testid="stSidebar"] .stNumberInput input {
    background: #111827 !important;
    border: 1px solid #1f2d3d !important;
    color: #e2f0fb !important;
    font-family: 'Share Tech Mono', monospace;
    border-radius: 4px;
  }

  /* ---- Main background ---- */
  .main .block-container { padding: 1.2rem 2rem; max-width: 1600px; }

  /* ---- KPI metric cards ---- */
  [data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1520 0%, #111c2b 100%);
    border: 1px solid #1c2d40;
    border-radius: 8px;
    padding: 1rem 1.4rem;
    position: relative;
    overflow: hidden;
  }
  [data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #e10600, #ff4d4d, transparent);
  }
  [data-testid="stMetricLabel"] { color: #5d7a94 !important; font-size: 0.72rem !important; letter-spacing: 0.12em; text-transform: uppercase; }
  [data-testid="stMetricValue"] { color: #f0f6fc !important; font-family: 'Share Tech Mono', monospace !important; font-size: 2rem !important; }
  [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

  /* ---- Divider ---- */
  hr { border-color: #1a2535 !important; margin: 0.8rem 0; }

  /* ---- Tables ---- */
  .stTable table { background: #0a0f18; border: 1px solid #1c2535; border-radius: 6px; width: 100%; }
  .stTable thead tr th { background: #0e1723 !important; color: #5d7a94 !important; font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 0.6rem 1rem; border-bottom: 1px solid #1c2535; }
  .stTable tbody tr td { color: #c8d8e8 !important; font-family: 'Share Tech Mono', monospace; font-size: 0.88rem; padding: 0.5rem 1rem; border-bottom: 1px solid #131e2c; }
  .stTable tbody tr:hover td { background: #101825 !important; }

  /* ---- Alert/info boxes ---- */
  .stAlert { border-radius: 6px; border-left-width: 3px; }

  /* ---- Plotly chart container ---- */
  .js-plotly-plot { border-radius: 8px; }

  /* ---- Section header style ---- */
  .sec-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3d5a73;
    margin-bottom: 0.4rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #1a2840;
  }

  /* ---- Tab strip ---- */
  .stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1c2535; gap: 0; }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d5a73 !important;
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 1.2rem;
  }
  .stTabs [aria-selected="true"] {
    color: #e10600 !important;
    border-bottom: 2px solid #e10600 !important;
  }

  /* ---- Spinner ---- */
  .stSpinner > div { border-top-color: #e10600 !important; }

  /* ---- Scrollbar ---- */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: #070a0f; }
  ::-webkit-scrollbar-thumb { background: #1c2d40; border-radius: 3px; }

  /* ---- BOX BOX banner ---- */
  .boxbox {
    background: linear-gradient(90deg, #e10600 0%, #b30500 100%);
    color: #fff;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 900;
    font-size: 1.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    text-align: center;
    padding: 0.7rem 1rem;
    border-radius: 6px;
    animation: pulse_red 1.2s ease-in-out infinite;
  }
  @keyframes pulse_red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(225,6,0,0.5); }
    50%       { box-shadow: 0 0 20px 6px rgba(225,6,0,0.3); }
  }

  /* ---- Compound badges ---- */
  .badge { display:inline-block; padding:2px 10px; border-radius:3px; font-family:'Share Tech Mono',monospace; font-size:0.78rem; font-weight:700; letter-spacing:0.06em; }
  .badge-soft   { background:#e10600; color:#fff; }
  .badge-medium { background:#ffd700; color:#111; }
  .badge-hard   { background:#e8ecf0; color:#111; }
  .badge-inter  { background:#39b54a; color:#fff; }
  .badge-wet    { background:#0070ff; color:#fff; }
  
  /* ---- Historic & Dictionary styling blocks ---- */
  .history-card {
    background: #0d1420;
    border: 1px solid #1c2c40;
    border-radius: 6px;
    padding: 1.2rem;
    margin-bottom: 1rem;
  }
  .history-era {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    color: #e10600;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
  }
  .term-desc {
    background: #090f16;
    border-left: 3px solid #e10600;
    padding: 1.2rem;
    border-radius: 0 6px 6px 0;
    font-size: 1.05rem;
    line-height: 1.6;
    color: #d1dfed;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
TRACKS = {
    "Bahrain":      {"laps": 57, "country": "BH", "lap_record": 90.558},
    "Saudi Arabia": {"laps": 50, "country": "SA", "lap_record": 89.293},
    "Australia":    {"laps": 58, "country": "AU", "lap_record": 87.236},
    "Monaco":       {"laps": 78, "country": "MC", "lap_record": 74.260},
    "Silverstone":  {"laps": 52, "country": "GB", "lap_record": 87.097},
    "Monza":        {"laps": 53, "country": "IT", "lap_record": 80.827},
    "Spa":          {"laps": 44, "country": "BE", "lap_record": 105.452},
    "Suzuka":       {"laps": 53, "country": "JP", "lap_record": 91.538},
    "Singapore":    {"laps": 62, "country": "SG", "lap_record": 103.303},
    "Abu Dhabi":    {"laps": 58, "country": "AE", "lap_record": 87.789},
}
TRACKS_LIST  = list(TRACKS.keys())
MAX_LAPS_MAP = {t: TRACKS[t]["laps"] for t in TRACKS}

COMPOUND_COLORS = {
    "SOFT":         "#e10600",
    "MEDIUM":       "#ffd700",
    "HARD":         "#e8ecf0",
    "INTERMEDIATE": "#39b54a",
    "WET":          "#0070ff",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Rajdhani, sans-serif", color="#8fa3bb", size=11),
    xaxis=dict(gridcolor="#111e2e", showgrid=True, zeroline=False, linecolor="#1c2535"),
    yaxis=dict(gridcolor="#111e2e", showgrid=True, zeroline=False, linecolor="#1c2535"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1c2535", borderwidth=1),
)


# ─────────────────────────────────────────────
#  BRAIN COMPILATION (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def compile_master_brain():
    model = AdvancedRacePaceModel(model_path="f1_brain.pkl")

    # Try loading a pre-saved model first
    if not model.load():
        with st.spinner("⚙  No saved model found — downloading race data and training…"):
            df = fetch_global_clean_dataset(
                years=[2022, 2023, 2024],
                tracks=TRACKS_LIST,
            )
        with st.spinner("🧠  Training GBM on multi-year dataset…"):
            model.train_global(df)

    engine = UniversalStrategyEngine(
        global_pace_model=model,
        tracks_list=TRACKS_LIST,
        max_laps_map=MAX_LAPS_MAP,
        default_track_temp=35.0,
    )
    return model, engine


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def fmt_laptime(seconds: float) -> str:
    if seconds <= 0 or math.isinf(seconds):
        return "–"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}:{s:06.3f}"


def compound_badge(c: str) -> str:
    css = {"SOFT": "soft", "MEDIUM": "medium", "HARD": "hard",
           "INTERMEDIATE": "inter", "WET": "wet"}.get(c.upper(), "medium")
    return f'<span class="badge badge-{css}">{c}</span>'


def build_laptime_chart(projection: list) -> go.Figure:
    df = pd.DataFrame(projection)
    fig = go.Figure()
    for comp, grp in df.groupby("compound"):
        fig.add_trace(go.Scatter(
            x=grp["lap"], y=grp["time"],
            name=comp,
            mode="lines+markers",
            marker=dict(size=4, color=COMPOUND_COLORS.get(comp, "#aaa")),
            line=dict(width=2, color=COMPOUND_COLORS.get(comp, "#aaa")),
            hovertemplate="Lap %{x}<br>%{y:.3f}s<extra>" + comp + "</extra>",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="PREDICTED LAP TIMES", font=dict(
            family="Barlow Condensed, sans-serif", size=13, color="#3d5a73"), x=0),
        xaxis_title="Lap", yaxis_title="Lap Time (s)",
        height=300,
    )
    return fig


def build_tyre_deg_chart(track: str, compounds: list,
                         model: AdvancedRacePaceModel,
                         track_temp: float = 35.0) -> go.Figure:
    fig = go.Figure()
    max_age = 40
    for comp in compounds:
        times = []
        for age in range(1, max_age + 1):
            t = model.predict_lap_time(track=track, lap_number=20, tire_age=age,
                                       compound=comp, track_temp=track_temp)
            times.append(t)
        deg_per_lap = [(times[i + 1] - times[i]) for i in range(len(times) - 1)]
        fig.add_trace(go.Bar(
            x=list(range(2, max_age + 1)),
            y=deg_per_lap,
            name=comp,
            marker_color=COMPOUND_COLORS.get(comp, "#aaa"),
            opacity=0.85,
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        title=dict(text="TYRE DEGRADATION RATE (s/lap)", font=dict(
            family="Barlow Condensed, sans-serif", size=13, color="#3d5a73"), x=0),
        xaxis_title="Tyre Age", yaxis_title="Δ s/lap",
        height=280,
    )
    return fig


def build_strategy_gantt(actions: list, current_lap: int,
                         total_laps: int, current_compound: str,
                         current_tire_age: int) -> go.Figure:
    """Horizontal stint bar chart."""
    stints = []
    # Current stint
    stints.append({
        "compound": current_compound,
        "start": current_lap - current_tire_age + 1,
        "end": current_lap,
    })
    prev_end = current_lap
    for a in actions:
        stints.append({
            "compound": a["Compound"],
            "start": a["Lap"],
            "end": a.get("end_lap", total_laps),
        })
        stints[-2]["end"] = a["Lap"] - 1
        prev_end = a["Lap"]
    if stints:
        stints[-1]["end"] = total_laps

    fig = go.Figure()
    for i, s in enumerate(stints):
        fig.add_trace(go.Bar(
            y=["Strategy"],
            x=[s["end"] - s["start"] + 1],
            base=[s["start"]],
            orientation="h",
            name=s["compound"],
            marker_color=COMPOUND_COLORS.get(s["compound"], "#555"),
            marker_line=dict(width=1, color="#070a0f"),
            text=s["compound"],
            textposition="inside",
            insidetextfont=dict(family="Barlow Condensed, sans-serif",
                                color="#070a0f" if s["compound"] in ("MEDIUM", "HARD") else "#fff",
                                size=11),
            hovertemplate=f"{s['compound']}: Lap {s['start']}–{s['end']}<extra></extra>",
            showlegend=False,
        ))
    # Current lap marker
    fig.add_vline(x=current_lap, line_width=1.5, line_dash="dot", line_color="#e10600",
                  annotation_text="NOW", annotation_font_color="#e10600",
                  annotation_font_size=10)
    fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",

    font=dict(
        family="Rajdhani, sans-serif",
        color="#8fa3bb",
        size=11
    ),

    margin=dict(
        l=40,
        r=20,
        t=40,
        b=40
    ),

    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="#1c2535",
        borderwidth=1
    ),

    barmode="stack",

    title=dict(
        text="STINT MAP",
        font=dict(
            family="Barlow Condensed, sans-serif",
            size=13,
            color="#3d5a73"
        ),
        x=0
    ),

    xaxis=dict(
        range=[1, total_laps + 1],
        title="Lap",
        gridcolor="#111e2e",
        showgrid=True,
        zeroline=False,
        linecolor="#1c2535"
    ),

    yaxis=dict(
        visible=False,
        gridcolor="#111e2e",
        showgrid=True,
        zeroline=False,
        linecolor="#1c2535"
    ),

    height=120
)
    return fig


# ─────────────────────────────────────────────
#  SIDEBAR NAVIGATION & INPUT COUPLING
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Barlow Condensed,sans-serif;font-weight:900;font-size:1.5rem;letter-spacing:0.12em;color:#e10600;margin-bottom:0.2rem;">🏁 PIT WALL PRO</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem;color:#3d5a73;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;">AI Strategy Engine</div>', unsafe_allow_html=True)

    # NEW: Application Main Navigation Selector
    st.markdown('<div class="sec-header">CONSOLE CONTROL</div>', unsafe_allow_html=True)
    app_view = st.selectbox("Select Display Deck", ["📡 Live Pit Wall", "📚 F1 Glossary & History"], index=0)

    # Wrap the existing live data selectors so they only load when the strategy engine view is operational
    if app_view == "📡 Live Pit Wall":
        st.markdown('<div class="sec-header" style="margin-top:0.8rem;">Session</div>', unsafe_allow_html=True)
        selected_track    = st.selectbox("Grand Prix", TRACKS_LIST)
        selected_driver   = st.text_input("Driver Code", value="VER").upper()
        total_race_laps   = MAX_LAPS_MAP[selected_track]

        st.markdown('<div class="sec-header" style="margin-top:0.8rem;">Live Telemetry</div>', unsafe_allow_html=True)
        current_lap       = st.slider("Current Lap", 1, total_race_laps, min(15, total_race_laps))
        current_compound  = st.selectbox("Tyre Compound", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"], index=1)
        tire_age          = st.number_input("Tyre Age (laps)", 1, 50, 12)

        st.markdown('<div class="sec-header" style="margin-top:0.8rem;">Track Conditions</div>', unsafe_allow_html=True)
        flag_status       = st.radio("Flag", ["GREEN 🟢", "VSC 🟡", "SC 🟡", "RED 🔴"],
                                      index=0, horizontal=True)
        flag_key          = flag_status.split()[0]
        weather           = st.radio("Weather", ["DRY", "WET"], index=0, horizontal=True)
        track_temp        = st.slider("Track Temp (°C)", 20, 65, 35)

        st.markdown('<div class="sec-header" style="margin-top:0.8rem;">Car Settings</div>', unsafe_allow_html=True)
        base_pit_loss     = st.number_input("Pit Lane Loss (s)", 18.0, 35.0, 23.0, step=0.5)
        engine_mode       = st.select_slider("Engine Mode", ["SAVE", "BALANCED", "PUSH"], value="BALANCED")
        stops_remaining   = st.radio("Max Stops", [1, 2, 3], index=1, horizontal=True)

        st.markdown('<div class="sec-header" style="margin-top:0.8rem;">Retrain Model</div>', unsafe_allow_html=True)
        if st.button("🔄 Force Retrain", use_container_width=True):
            if os.path.exists("f1_brain.pkl"):
                os.remove("f1_brain.pkl")
            st.cache_resource.clear()
            st.rerun()


# ─────────────────────────────────────────────
#  MAIN EXECUTIVE SYSTEM NAVIGATION BRANCH
# ─────────────────────────────────────────────
if app_view == "📡 Live Pit Wall":

    #  HEADER BAR
    laps_left = total_race_laps - current_lap
    pct_done  = current_lap / total_race_laps * 100

    c_head = st.columns([3, 1, 1, 1])
    with c_head[0]:
        st.markdown(
            f'<div style="font-family:Barlow Condensed,sans-serif;font-weight:900;'
            f'font-size:2rem;letter-spacing:0.08em;color:#f0f6fc;">'
            f'{selected_track.upper()} GRAND PRIX &nbsp;'
            f'<span style="color:#3d5a73;font-size:1rem;">{selected_driver}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Race progress bar
        st.markdown(
            f'<div style="background:#0d1520;border-radius:3px;height:6px;margin-top:4px;">'
            f'<div style="background:linear-gradient(90deg,#e10600,#ff6b6b);'
            f'width:{pct_done:.1f}%;height:6px;border-radius:3px;transition:width 0.3s;"></div>'
            f'</div>'
            f'<div style="font-size:0.68rem;color:#3d5a73;margin-top:2px;">'
            f'LAP {current_lap} / {total_race_laps} — {laps_left} LAPS REMAINING</div>',
            unsafe_allow_html=True,
        )

    with c_head[1]:
        flag_colors = {"GREEN": "#39b54a", "VSC": "#ffd700", "SC": "#ffd700", "RED": "#e10600"}
        st.markdown(
            f'<div style="text-align:center;background:#0d1520;border:1px solid #1c2535;'
            f'border-radius:6px;padding:0.5rem;">'
            f'<div style="font-size:0.65rem;color:#3d5a73;letter-spacing:0.1em;">FLAG</div>'
            f'<div style="font-size:1.2rem;font-weight:700;color:{flag_colors.get(flag_key,"#aaa")};">'
            f'{flag_key}</div></div>', unsafe_allow_html=True
        )
    with c_head[2]:
        st.markdown(
            f'<div style="text-align:center;background:#0d1520;border:1px solid #1c2535;'
            f'border-radius:6px;padding:0.5rem;">'
            f'<div style="font-size:0.65rem;color:#3d5a73;letter-spacing:0.1em;">TYRE</div>'
            f'<div style="margin-top:2px;">{compound_badge(current_compound)}</div></div>',
            unsafe_allow_html=True,
        )
    with c_head[3]:
        st.markdown(
            f'<div style="text-align:center;background:#0d1520;border:1px solid #1c2535;'
            f'border-radius:6px;padding:0.5rem;">'
            f'<div style="font-size:0.65rem;color:#3d5a73;letter-spacing:0.1em;">TRACK TEMP</div>'
            f'<div style="font-size:1.2rem;font-weight:700;color:#f0f6fc;">{track_temp}°C</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    #  BOOT BRAIN
    with st.spinner("🚀  Loading AI brain…"):
        ml_model, master_engine = compile_master_brain()

    #  RUN OPTIMISER
    vsc_loss = base_pit_loss - 10.0
    memo     = {}

    with st.spinner("⚡  Computing optimal strategy…"):
        time_to_flag, optimal_actions = master_engine.calculate_optimal_path(
            track=selected_track,
            current_lap=current_lap,
            tire_age=int(tire_age),
            compound=current_compound,
            flag_status=flag_key,
            track_temp=float(track_temp),
            weather=weather,
            engine_mode=engine_mode,
            base_pit_loss=float(base_pit_loss),
            vsc_pit_loss=float(vsc_loss),
            stops_remaining=int(stops_remaining),
            memo=memo,
        )

    # Lap-time projection for chart
    projection = master_engine.project_lap_times(
        track=selected_track,
        strategy_actions=optimal_actions,
        current_lap=current_lap,
        current_compound=current_compound,
        current_tire_age=int(tire_age),
        total_laps=total_race_laps,
        track_temp=float(track_temp),
        weather=weather,
        engine_mode=engine_mode,
    )

    #  TABS
    tab_strategy, tab_pace, tab_tyres, tab_stint, tab_compare = st.tabs([
        "⚑  STRATEGY DECK",
        "📈  PACE PROJECTION",
        "🔴  TYRE ANALYSIS",
        "🟦  STINT MAP",
        "⚖  SCENARIO COMPARE",
    ])

    #  TAB 1 — STRATEGY DECK
    with tab_strategy:
        if optimal_actions and optimal_actions[0]["Lap"] == current_lap:
            st.markdown(
                f'<div class="boxbox">🚨 &nbsp; BOX, BOX, BOX! &nbsp; '
                f'PIT {selected_driver} THIS LAP → {optimal_actions[0]["Compound"]} &nbsp; 🚨</div>',
                unsafe_allow_html=True,
            )
            st.write("")
        elif optimal_actions and optimal_actions[0]["Lap"] - current_lap <= 3:
            laps_until = optimal_actions[0]["Lap"] - current_lap
            st.warning(
                f"⚠️  **PREPARE TO PIT** — {selected_driver} boxes in "
                f"**{laps_until} lap{'s' if laps_until > 1 else ''}** on Lap "
                f"{optimal_actions[0]['Lap']} → {optimal_actions[0]['Compound']}"
            )

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("TIME TO FLAG", fmt_laptime(time_to_flag))
        with k2:
            st.metric("PLANNED STOPS", len(optimal_actions))
        with k3:
            next_pit = optimal_actions[0]["Lap"] if optimal_actions else "–"
            st.metric("NEXT PIT LAP", next_pit)
        with k4:
            next_comp = optimal_actions[0]["Compound"] if optimal_actions else "–"
            st.metric("NEXT COMPOUND", next_comp)

        st.divider()

        if not optimal_actions:
            st.success(
                f"✅ **STAY OUT** — Optimal strategy is to maintain current stint to Lap {total_race_laps}. "
                f"No pit stops required from Lap {current_lap}."
            )
        else:
            df_act = pd.DataFrame(optimal_actions)
            laps_list = [current_lap] + df_act["Lap"].tolist()
            df_act["Stint Length"] = df_act["Lap"].diff().fillna(df_act["Lap"].iloc[0] - current_lap).astype(int).astype(str) + " laps"
            df_act["Laps Left After"] = (total_race_laps - df_act["Lap"]).astype(str) + " laps"
            st.dataframe(
                df_act.rename(columns={"Lap": "PIT LAP", "Action": "ACTION",
                                       "Pit Loss (s)": "PIT LOSS (s)", "Compound": "COMPOUND"}),
                use_container_width=True,
                hide_index=True,
            )

        cur_pred = ml_model.predict_lap_time(
            selected_track, current_lap, int(tire_age), current_compound, float(track_temp)
        )
        lap_record = TRACKS[selected_track]["lap_record"]
        delta_record = cur_pred - lap_record

        st.divider()
        cr1, cr2, cr3 = st.columns(3)
        with cr1:
            st.metric("CURRENT PRED LAP TIME", fmt_laptime(cur_pred))
        with cr2:
            st.metric("CIRCUIT LAP RECORD", fmt_laptime(lap_record))
        with cr3:
            st.metric("DELTA TO RECORD", f"+{fmt_laptime(delta_record)}")

    #  TAB 2 — PACE PROJECTION
    with tab_pace:
        if projection:
            fig_pace = build_laptime_chart(projection)
            for a in optimal_actions:
                fig_pace.add_vline(
                    x=a["Lap"], line_width=1.2, line_dash="dash", line_color="#ffd700",
                    annotation_text=f"PIT→{a['Compound']}",
                    annotation_font_color="#ffd700", annotation_font_size=9,
                    annotation_textangle=-90,
                )
            st.plotly_chart(fig_pace, use_container_width=True)

            if projection:
                proj_df = pd.DataFrame(projection)
                st.markdown('<div class="sec-header">STINT PACE SUMMARY</div>', unsafe_allow_html=True)
                summary = proj_df.groupby("compound")["time"].agg(
                    Mean=lambda x: round(x.mean(), 3),
                    Best=lambda x: round(x.min(), 3),
                    Worst=lambda x: round(x.max(), 3),
                    Laps="count",
                ).reset_index().rename(columns={"compound": "Compound"})
                st.dataframe(summary, use_container_width=True, hide_index=True)
        else:
            st.info("No projection data available.")

    #  TAB 3 — TYRE ANALYSIS
    with tab_tyres:
        t1, t2 = st.columns([2, 1])
        with t1:
            selected_compounds = st.multiselect(
                "Compare Compounds", ["SOFT", "MEDIUM", "HARD"],
                default=["SOFT", "MEDIUM", "HARD"]
            )
            if selected_compounds:
                fig_deg = build_tyre_deg_chart(selected_track, selected_compounds, ml_model, float(track_temp))
                st.plotly_chart(fig_deg, use_container_width=True)

        with t2:
            st.markdown('<div class="sec-header">CLIFF LAPS ESTIMATE</div>', unsafe_allow_html=True)
            for comp in selected_compounds if selected_compounds else ["SOFT", "MEDIUM", "HARD"]:
                times = [ml_model.predict_lap_time(selected_track, 20, a, comp, float(track_temp))
                         for a in range(1, 50)]
                deg = np.diff(times)
                cliff = int(np.argmax(deg > np.percentile(deg, 85)) + 1) if len(deg) else 30
                st.markdown(
                    f'<div style="margin-bottom:0.5rem;">{compound_badge(comp)} '
                    f'<span style="color:#8fa3bb;font-size:0.85rem;">cliff ~lap </span>'
                    f'<span style="color:#f0f6fc;font-family:Share Tech Mono,monospace;">{cliff}</span></div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="sec-header" style="margin-top:0.8rem;">PACE AT KEY TYRE AGES</div>', unsafe_allow_html=True)
        ages_to_show = [1, 5, 10, 15, 20, 25, 30]
        rows = []
        for comp in ["SOFT", "MEDIUM", "HARD"]:
            row = {"Compound": comp}
            for age in ages_to_show:
                row[f"Age {age}"] = fmt_laptime(
                    ml_model.predict_lap_time(selected_track, 20, age, comp, float(track_temp))
                )
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    #  TAB 4 — STINT MAP
    with tab_stint:
        if optimal_actions:
            fig_gantt = build_strategy_gantt(
                optimal_actions, current_lap, total_race_laps,
                current_compound, int(tire_age)
            )
            st.plotly_chart(fig_gantt, use_container_width=True)
        else:
            fig_gantt = build_strategy_gantt(
                [], current_lap, total_race_laps, current_compound, int(tire_age)
            )
            st.plotly_chart(fig_gantt, use_container_width=True)

        st.markdown('<div class="sec-header">STINT DETAILS</div>', unsafe_allow_html=True)
        stints_info = [{
            "Stint": 1,
            "Compound": current_compound,
            "Started": current_lap - int(tire_age) + 1,
            "Laps So Far": int(tire_age),
            "Status": "ACTIVE 🟢",
        }]
        for i, a in enumerate(optimal_actions):
            stints_info.append({
                "Stint": i + 2,
                "Compound": a["Compound"],
                "Started": a["Lap"],
                "Laps So Far": "–",
                "Status": "PLANNED",
            })
        st.dataframe(pd.DataFrame(stints_info), use_container_width=True, hide_index=True)

    #  TAB 5 — SCENARIO COMPARE
    with tab_compare:
        st.markdown('<div class="sec-header">HEAD-TO-HEAD STRATEGY COMPARISON</div>', unsafe_allow_html=True)
        st.caption("Define two strategies manually and compare their projected total time from current lap.")

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**Strategy A**")
            a_pit1_lap  = st.number_input("A — Pit 1 Lap", 1, total_race_laps, min(20, total_race_laps - 5), key="a1l")
            a_pit1_comp = st.selectbox("A — Pit 1 Compound", ["SOFT", "MEDIUM", "HARD"], index=1, key="a1c")
            a_use_pit2  = st.checkbox("Add 2nd stop (A)", key="a2_cb")
            a_pit2_lap, a_pit2_comp = total_race_laps, "MEDIUM"
            if a_use_pit2:
                a_pit2_lap  = st.number_input("A — Pit 2 Lap", a_pit1_lap + 1, total_race_laps, min(a_pit1_lap + 15, total_race_laps - 3), key="a2l")
                a_pit2_comp = st.selectbox("A — Pit 2 Compound", ["SOFT", "MEDIUM", "HARD"], key="a2c")

        with sc2:
            st.markdown("**Strategy B**")
            b_pit1_lap  = st.number_input("B — Pit 1 Lap", 1, total_race_laps, min(30, total_race_laps - 5), key="b1l")
            b_pit1_comp = st.selectbox("B — Pit 1 Compound", ["SOFT", "MEDIUM", "HARD"], index=2, key="b1c")
            b_use_pit2  = st.checkbox("Add 2nd stop (B)", key="b2_cb")
            b_pit2_lap, b_pit2_comp = total_race_laps, "HARD"
            if b_use_pit2:
                b_pit2_lap  = st.number_input("B — Pit 2 Lap", b_pit1_lap + 1, total_race_laps, min(b_pit1_lap + 15, total_race_laps - 3), key="b2l")
                b_pit2_comp = st.selectbox("B — Pit 2 Compound", ["SOFT", "MEDIUM", "HARD"], key="b2c")

        if st.button("⚖ Run Comparison", use_container_width=True):
            from simulation_engine import RaceSimulator
            sim = RaceSimulator(
                total_laps=total_race_laps - current_lap + 1,
                pit_penalty=float(base_pit_loss),
                pace_model=ml_model,
                track=selected_track,
                track_temp=float(track_temp),
            )

            pit_a = {max(1, a_pit1_lap - current_lap + 1): a_pit1_comp}
            if a_use_pit2:
                pit_a[max(2, a_pit2_lap - current_lap + 1)] = a_pit2_comp

            pit_b = {max(1, b_pit1_lap - current_lap + 1): b_pit1_comp}
            if b_use_pit2:
                pit_b[max(2, b_pit2_lap - current_lap + 1)] = b_pit2_comp

            total_a, hist_a = sim.run_strategy(current_compound, pit_a)
            total_b, hist_b = sim.run_strategy(current_compound, pit_b)
            delta = abs(total_a - total_b)
            winner = "A" if total_a < total_b else "B"

            st.divider()
            res1, res2, res3 = st.columns(3)
            with res1:
                st.metric("Strategy A Total", fmt_laptime(total_a))
            with res2:
                st.metric("Strategy B Total", fmt_laptime(total_b))
            with res3:
                st.metric(f"Strategy {winner} wins by", fmt_laptime(delta))

            fig_cmp = go.Figure()
            for label, hist, color in [("Strategy A", hist_a, "#e10600"), ("Strategy B", hist_b, "#0070ff")]:
                fig_cmp.add_trace(go.Scatter(
                    x=hist["LapNumber"] + current_lap - 1,
                    y=hist["CumulativeTime"],
                    name=label, mode="lines",
                    line=dict(width=2, color=color),
                    hovertemplate="Lap %{x}<br>%{y:.1f}s<extra>" + label + "</extra>",
                ))
            fig_cmp.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="CUMULATIVE RACE TIME", font=dict(
                    family="Barlow Condensed, sans-serif", size=13, color="#3d5a73"), x=0),
                xaxis_title="Lap", yaxis_title="Cumulative Time (s)", height=300,
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            if len(hist_a) == len(hist_b):
                delta_times = hist_b["LapTime"].values - hist_a["LapTime"].values
                fig_delta = go.Figure(go.Bar(
                    x=hist_a["LapNumber"] + current_lap - 1,
                    y=delta_times,
                    marker_color=["#e10600" if d < 0 else "#0070ff" for d in delta_times],
                    hovertemplate="Lap %{x}<br>B–A: %{y:.3f}s<extra></extra>",
                ))
                fig_delta.update_layout(
                    **PLOTLY_LAYOUT,
                    title=dict(text="LAP-BY-LAP DELTA (B minus A)", font=dict(
                        family="Barlow Condensed, sans-serif", size=13, color="#3d5a73"), x=0),
                    xaxis_title="Lap", yaxis_title="Delta (s)", height=250,
                )
                st.plotly_chart(fig_delta, use_container_width=True)


# ─────────────────────────────────────────────
#  NEW: MISSION CONTROL PAGE — GLOSSARY & HISTORY
# ─────────────────────────────────────────────
else:
    st.markdown('<div style="font-family:Barlow Condensed,sans-serif;font-weight:900;font-size:2.2rem;letter-spacing:0.05em;color:#f0f6fc;">FORMULA 1 ARCHIVES & COMPENDIUM</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.85rem;color:#1c2d40;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:1.5rem;">Historical Chronology and Mechanical Reference Guide</div>', unsafe_allow_html=True)

    st.divider()

    # --- BRIEF HISTORY SECTION ---
    st.markdown('<div class="sec-header">A BRIEF HISTORY OF FORMULA ONE</div>', unsafe_allow_html=True)
    
    col_hist1, col_hist2, col_hist3 = st.columns(3)
    with col_hist1:
        st.markdown("""
        <div class="history-card">
            <div class="history-era">1950s – 1960s: The Genesis & Garagistes</div>
            The FIA Formula One World Championship officially commenced in 1950 at Silverstone. 
            Early eras were dominated by front-engined titans from Alfa Romeo and Ferrari driven by 
            legends like Juan Manuel Fangio. By the 1960s, British "garagiste" constructors like Cooper 
            and Lotus revolutionized the sport forever by moving the engine behind the driver.
        </div>
        """, unsafe_allow_html=True)

    with col_hist2:
        st.markdown("""
        <div class="history-card">
            <div class="history-era">1970s – 1990s: Aerodynamics & Turbo Monsters</div>
            Colin Chapman pioneered massive "ground effect" venturi tunnels, turning cars into inverted wings. 
            The 1980s saw power figures surge past 1,000 HP from qualifying turbo engines alongside fierce 
            rivalries like Senna vs. Prost. High-tech electronic aids like active suspension characterized 
            the early 1990s before safety mandates re-stabilized parameters.
        </div>
        """, unsafe_allow_html=True)

    with col_hist3:
        st.markdown("""
        <div class="history-card">
            <div class="history-era">2000s – Present: V10 Screamers to Hybrid Efficiency</div>
            The early 2000s introduced ultra-high revving V10 engine architectures and structural dominance 
            from Michael Schumacher. In 2014, F1 made its most monumental architectural shift, introducing 1.6-liter 
            V6 turbo-charged Hybrid Power Units with cutting-edge energy recovery systems (ERS), setting new bounds for 
            thermal efficiency in international motorsport.
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # --- TABS FOR CORE F1 TERMS ---
    st.markdown('<div class="sec-header">INTERACTIVE MECHANICAL DICTIONARY</div>', unsafe_allow_html=True)
    st.caption("Click through the tabs below to explore underlying structural regulations, racing strategies, and behavioral dynamics.")

    # Instantiate terms list
    tab_drs, tab_undercut, tab_overcut, tab_ers, tab_parc, tab_apex, tab_grain, tab_blister = st.tabs([
        "DRS", "UNDERCUT", "OVERCUT", "ERS", "PARC FERMÉ", "APEX", "GRAINING", "BLISTERING"
    ])

    with tab_drs:
        st.markdown("""
        <div class="term-desc">
            <strong>DRS (Drag Reduction System):</strong> Introduced in 2011, DRS allows drivers to hydraulically 
            open a flap in their rear wing to shed aerodynamic drag when tracking within 1.0 second of a leading vehicle 
            at pre-mapped detection lines. This dramatically spikes straight-line velocity by roughly 10–12 km/h, assisting overtaking maneuvers.
        </div>
        """, unsafe_allow_html=True)

    with tab_undercut:
        st.markdown("""
        <div class="term-desc">
            <strong>Undercut Strategy:</strong> A strategic track-position play where a trailing car pits earlier than 
            the car directly ahead. By mounting clean, high-grip fresh tires a few laps earlier, the driver extracts immediate 
            pace lines that easily jump ahead of the rival once that rival finally pits and returns to the track.
        </div>
        """, unsafe_allow_html=True)

    with tab_overcut:
        st.markdown("""
        <div class="term-desc">
            <strong>Overcut Strategy:</strong> The mechanical inversion of an undercut. A driver elects to stay out on older tires 
            while their rival boxes early. If the rival returns into dense racing traffic or struggles to clear tire-warming thresholds, 
            the overcutting car can run fast, uninhibited pace metrics in clean air and pit later while remaining ahead.
        </div>
        """, unsafe_allow_html=True)

    with tab_ers:
        st.markdown("""
        <div class="term-desc">
            <strong>ERS (Energy Recovery System):</strong> An advanced dual-motor layout composed of the <strong>MGU-K</strong> 
            (Motor Generator Unit - Kinetic, which harvests energy ordinarily lost under heavy braking friction) and the 
            <strong>MGU-H</strong> (Motor Generator Unit - Heat, harvesting energy from exhaust turbocharger gases). Together, 
            they feed power to a shared battery pack to unleash an auxiliary ~160 brake horsepower during a lap.
        </div>
        """, unsafe_allow_html=True)

    with tab_parc:
        st.markdown("""
        <div class="term-desc">
            <strong>Parc Fermé:</strong> Literally translating from French to "closed park," this denotes secure containment rules. 
            The moment a vehicle drives out of the garage during Qualifying, it enters Parc Fermé conditions. Teams are strictly 
            forbidden from making major structural or aerodynamic setup changes (suspension adjustments, weight distributions) 
            prior to Sunday's Grand Prix finish.
        </div>
        """, unsafe_allow_html=True)

    with tab_apex:
        st.markdown("""
        <div class="term-desc">
            <strong>Apex (Clipping Point):</strong> The innermost geometric vertex of a cornering trajectory. By positioning 
            the car perfectly flush against the interior curb at the apex, the driver straightens the incoming corner angle, 
            allowing them to get back onto full engine throttle inputs as early as possible on exit paths.
        </div>
        """, unsafe_allow_html=True)

    with tab_grain:
        st.markdown("""
        <div class="term-desc">
            <strong>Graining:</strong> Occurs when tires slide laterally across the track surface before they reach optimal thermal 
            operating limits. The cold, brittle rubber tread splits apart, forming tiny loose corrugated rolls that stick to the tire 
            contact patch. This significantly degrades tire traction lines until the surface layer eventually shears away clean.
        </div>
        """, unsafe_allow_html=True)

    with tab_blister:
        st.markdown("""
        <div class="term-desc">
            <strong>Blistering:</strong> The exact thermal opposite of graining. When a tire core gets excessively hot while 
            the external surface remains relatively cold, the internal rubber compound liquefies and boils. The resulting gas pockets 
            blister outwards, blowing chunks of rubber away from the tire carcass and presenting critical structural safety degradation.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;font-size:0.65rem;color:#1c2d40;letter-spacing:0.15em;">'
    'F1 PIT WALL PRO &nbsp;·&nbsp; GBM STRATEGY ENGINE &nbsp;·&nbsp; DATA: FASTF1 API &nbsp;·&nbsp; NOT AFFILIATED WITH FIA OR FORMULA ONE GROUP'
    '</div>',
    unsafe_allow_html=True,
)