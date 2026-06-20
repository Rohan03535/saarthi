"""
SAARTHI - Phase 6: Live Ops Console
Streamlit application acting as a traffic control-desk tool.
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib
import json
import math
from datetime import datetime, timedelta

# ---- CONFIG & THEME ----------------------------------------------------------
st.set_page_config(page_title="SAARTHI Live Ops", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root / Background ───────────────────────────── */
.stApp                        { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] { background-color: #0d1117; }
.block-container              { padding-top: 1rem !important; }

/* ── Hide Streamlit chrome ───────────────────────── */
#MainMenu, footer, header     { visibility: hidden; }
.stDeployButton               { display: none; }

/* ── Top status bar ─────────────────────────────── */
.status-bar {
    background: linear-gradient(90deg, #161b22 0%, #1c2333 100%);
    border-bottom: 1px solid #21262d;
    padding: 8px 20px;
    display: flex; align-items: center; gap: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: #8b949e;
    margin-bottom: 12px;
    border-radius: 6px;
}
.status-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px; }
.status-dot.live  { background:#238636; box-shadow: 0 0 6px #238636; animation: pulse 2s infinite; }
.status-dot.alert { background:#da3633; box-shadow: 0 0 6px #da3633; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

/* ── Header ─────────────────────────────────────── */
.saarthi-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px 20px;
    margin-bottom: 16px;
    display: flex; align-items: center; justify-content: space-between;
}
.saarthi-title  { font-size: 22px; font-weight: 700; color: #58a6ff; letter-spacing: 1px; }
.saarthi-sub    { font-size: 12px; color: #8b949e; margin-top: 2px; }
.saarthi-badge  {
    background: #1f6feb22; border: 1px solid #1f6feb;
    color: #58a6ff; font-size: 11px; font-weight: 600;
    padding: 4px 10px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Mode radio buttons ──────────────────────────── */
.stRadio > div { gap: 8px !important; }
.stRadio label {
    background: #21262d !important; border: 1px solid #30363d !important;
    border-radius: 6px !important; padding: 6px 16px !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: #8b949e !important; cursor: pointer !important;
    transition: all 0.2s !important;
}
.stRadio label:has(input:checked) {
    background: #1f6feb22 !important; border-color: #1f6feb !important;
    color: #58a6ff !important;
}

/* ── Section headers in sidebar ─────────────────── */
.section-header {
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: #8b949e;
    border-bottom: 1px solid #21262d;
    padding-bottom: 4px; margin: 14px 0 10px 0;
}

/* ── Metric boxes ────────────────────────────────── */
.metric-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.metric-box.green { border-left-color: #238636; }
.metric-box.amber { border-left-color: #d29922; }
.metric-box.red   { border-left-color: #da3633; }
.metric-value { font-size: 22px; font-weight: 700; color: #ffffff; font-family:'JetBrains Mono',monospace; }
.metric-label { font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.metric-sub   { font-size: 10px; color: #6e7681; margin-top: 3px; }

/* ── Priority Queue Cards ────────────────────────── */
.queue-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #30363d;
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 6px;
    transition: border-left-color 0.2s;
}
.queue-card.crit  { border-left-color: #da3633; }
.queue-card.high  { border-left-color: #d29922; }
.queue-card.med   { border-left-color: #58a6ff; }
.queue-card.low   { border-left-color: #30363d; }

.card-rank  { font-size: 10px; color: #6e7681; font-family:'JetBrains Mono',monospace; }
.card-score { font-size: 16px; font-weight: 700; color: #ffffff; font-family:'JetBrains Mono',monospace; }
.card-cause { font-size: 13px; font-weight: 600; color: #e6edf3; }
.card-meta  { font-size: 11px; color: #8b949e; margin-top: 3px; line-height: 1.5; }

/* ── Tags ────────────────────────────────────────── */
.tag {
    padding: 2px 7px; border-radius: 4px;
    font-size: 10px; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 5px; vertical-align: middle;
}
.tag-mdl  { background: #1f6feb33; color: #58a6ff; border: 1px solid #1f6feb55; }
.tag-fbk  { background: #da363333; color: #ff7b72; border: 1px solid #da363355; }
.tag-grow { background: #da363322; color: #ff7b72; }
.tag-shrk { background: #d2992222; color: #f0b429; }
.tag-clrd { background: #23863622; color: #3fb950; }

/* ── Phase indicator pills ───────────────────────── */
.phase-pill {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 10px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.phase-pill.growing  { background:#da363333; color:#ff7b72; }
.phase-pill.shrinking{ background:#d2992222; color:#f0b429; }
.phase-pill.cleared  { background:#23863622; color:#3fb950; }

/* ── Map legend strip ────────────────────────────── */
.map-legend {
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: 8px 14px; margin-top: 8px;
    display: flex; align-items: center; gap: 18px; font-size: 11px; color: #8b949e;
}
.legend-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:4px; }

/* ── Inject result card ──────────────────────────── */
.inject-result {
    background: #161b22; border: 1px solid #238636;
    border-radius: 8px; padding: 14px 16px; margin-top: 12px;
}
.inject-title { font-size: 13px; font-weight: 700; color: #3fb950; margin-bottom: 8px; }
.inject-row   { font-size: 12px; color: #c9d1d9; margin-bottom: 4px; }
.inject-unit  { font-size: 14px; font-weight: 700; color: #ffffff; margin-top: 8px; }

/* ── Slider ──────────────────────────────────────── */
.stSlider > div > div > div { background: #1f6feb !important; }

/* ── Scrollable queue ────────────────────────────── */
.queue-scroll { max-height: 480px; overflow-y: auto; padding-right: 4px; }
.queue-scroll::-webkit-scrollbar { width: 4px; }
.queue-scroll::-webkit-scrollbar-track { background: #0d1117; }
.queue-scroll::-webkit-scrollbar-thumb { background: #30363d; border-radius: 2px; }

/* ── Disclaimer footer ───────────────────────────── */
.disclaimer {
    font-size: 10px; color: #6e7681; text-align: center;
    border-top: 1px solid #21262d; padding-top: 6px; margin-top: 6px;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# ---- FILE PATHS --------------------------------------------------------------
TRIAGE_PATH  = r"C:\Users\Rohan\Pictures\saarthi\triage_queue.parquet"
ASSIGN_PATH  = r"C:\Users\Rohan\Pictures\saarthi\phase4_assignments.parquet"
CASCADE_PATH = r"C:\Users\Rohan\Pictures\saarthi\phase5_cascade_snapshots.parquet"
METRICS_PATH = r"C:\Users\Rohan\Pictures\saarthi\phase4_metrics.json"
MODEL_PATH   = r"C:\Users\Rohan\Pictures\saarthi\rsf_model.pkl"

# ---- CONSTANTS ---------------------------------------------------------------
UNIT_DEPOTS = {
    1: (12.9716, 77.5946), 2: (13.0358, 77.5970), 3: (12.9200, 77.6200),
    4: (12.9716, 77.5100), 5: (12.9900, 77.7200), 6: (13.0100, 77.5500),
    7: (12.9500, 77.5700), 8: (13.0600, 77.6800),
}
LOW_CONF_TYPES = {"tree_fall", "public_event", "road_conditions", "others"}

PHASE_COLORS = {
    "growing":   "#ff4444", # Red
    "shrinking": "#ffaa00", # Amber
    "cleared":   "#44ff88", # Green
}

EVENT_CAUSE_BASE = {
    "accident": 1.00, "water_logging": 0.90, "tree_fall": 0.85, "congestion": 0.80,
    "procession": 0.75, "vip_movement": 0.75, "protest": 0.70, "vehicle_breakdown": 0.60,
    "construction": 0.65, "public_event": 0.55, "pot_holes": 0.40, "road_conditions": 0.45,
    "debris": 0.50, "others": 0.40, "unknown": 0.40, "other_cause": 0.40,
}
VEH_TYPE_MULT = {
    "bmtc_bus": 1.00, "ksrtc_bus": 1.00, "private_bus": 0.95, "heavy_vehicle": 0.90,
    "truck": 0.88, "lcv": 0.55, "private_car": 0.45, "taxi": 0.45, "auto": 0.30,
    "others": 0.50, "unknown": 0.50,
}
CORRIDOR_TIER = {
    "mysore road": 1.00, "tumkur road": 1.00, "bellary road 1": 1.00, "bellary road 2": 0.95,
    "hosur road": 0.95, "old madras road": 0.90, "magadi road": 0.85, "bannerghatta road": 0.85,
    "bannerghata road": 0.85, "orr north 1": 0.90, "orr north 2": 0.88, "orr east 1": 0.88,
    "orr east 2": 0.85, "orr west 1": 0.85, "orr west 2": 0.83, "orr south 1": 0.83,
    "orr south 2": 0.80, "west of chord road": 0.75, "intermediate ring road": 0.70,
    "sarjapur road": 0.70, "non-corridor": 0.35, "other_corridor": 0.50,
}

# ---- DATA LOADING ------------------------------------------------------------
@st.cache_data
def load_data():
    triage  = pd.read_parquet(TRIAGE_PATH)
    assigns = pd.read_parquet(ASSIGN_PATH)
    cascade = pd.read_parquet(CASCADE_PATH)
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    
    # Pre-process triage to calculate active windows (using same logic as Phase 4)
    triage["start_dt"] = pd.to_datetime(triage["start_datetime"], utc=False)
    # The active window end is estimated in Phase 4, but for the UI we can just use 
    # the timestep ranges available in cascade for each incident.
    cascade["timestep"] = pd.to_datetime(cascade["timestep"], utc=False)
    assigns["timestep"] = pd.to_datetime(assigns["timestep"], utc=False)
    
    return triage, assigns, cascade, metrics

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

triage_df, assign_df, cascade_df, metrics = load_data()
bundle = load_model()

# Extract unique timesteps for the slider
all_timesteps = sorted(cascade_df["timestep"].unique())

# Default to March 7 12:30 (peak timestep)
try:
    default_step_idx = all_timesteps.index(pd.to_datetime("2024-03-07 12:30:00+05:30"))
except ValueError:
    default_step_idx = int(len(all_timesteps)*0.65)

# ---- HELPER FUNCTIONS FOR SCORING ---------------------------------------------
def predict_clearance(row_dict):
    rsf = bundle["rsf"]
    prep = bundle["preprocessor"]
    fnames = bundle["feature_names"]
    
    cause_grp = row_dict["event_cause"] if row_dict["event_cause"] in bundle["top_causes"] else "other_cause"
    corr_grp  = row_dict["corridor"] if row_dict["corridor"] in bundle["top_corridors"] else "other_corridor"
    
    row = pd.DataFrame([{
        "event_cause_grouped": cause_grp,
        "corridor_grouped": corr_grp,
        "veh_type": row_dict["veh_type"],
        "is_planned": 0,
        "is_high_prio": 1 if row_dict["priority"] == "high" else 0,
        "requires_road_closure": 1 if row_dict["requires_road_closure"] else 0,
        "hour_of_day_ist": row_dict["hour_of_day_ist"],
    }])[bundle["all_features"]]
    
    X_enc = pd.DataFrame(prep.transform(row), columns=fnames)
    surv_fn = rsf.predict_survival_function(X_enc)[0]
    
    # Extract risk score
    risk_raw = rsf.predict(X_enc)[0]
    # Rough normalization using the training min/max bounds (assumed 0 to ~10)
    risk_norm = min(1.0, max(0.0, risk_raw / 6.0)) # heuristic normalization for inject
    
    t_vals = surv_fn.x
    s_vals = surv_fn.y
    idx = np.searchsorted(-s_vals, -0.50, side="left")
    med = float(t_vals[-1]) if idx >= len(t_vals) else float(t_vals[idx])
    return med, risk_norm

def compute_triage_score(row_dict, risk_norm):
    ci = EVENT_CAUSE_BASE.get(row_dict["event_cause"], 0.40) * VEH_TYPE_MULT.get(row_dict["veh_type"], 0.50)
    if row_dict["requires_road_closure"]:
        ci = min(1.0, ci + 0.20)
    cc = CORRIDOR_TIER.get(row_dict["corridor"], 0.55)
    
    is_fbk = row_dict["event_cause"] in LOW_CONF_TYPES
    if is_fbk:
        pscale = 0.50 if row_dict["priority"] == "high" else 0.25
        score = ci * cc * pscale * 10
        src = "[FBK]"
    else:
        score = ci * risk_norm * cc * 10
        src = "[MDL]"
    
    return min(10.0, score), ci, cc, src

def travel_time_min(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    km = R * 2 * math.asin(math.sqrt(a))
    return (km / 20.0) * 60.0

# ---- LAYOUT ------------------------------------------------------------------
# Header
st.markdown("""
<div class="saarthi-header">
  <div>
    <div class="saarthi-title">SAARTHI</div>
    <div class="saarthi-sub">Live Traffic-Incident Command System &nbsp;|&nbsp; Bengaluru Traffic Police</div>
  </div>
  <div class="saarthi-badge">PROTOTYPE v1.0</div>
</div>
""", unsafe_allow_html=True)

# Status bar
st.markdown("""
<div class="status-bar">
  <span><span class="status-dot live"></span>SYSTEM ONLINE</span>
  <span>|</span>
  <span>Source: BTP Incident Log (8,173 records)</span>
  <span>|</span>
  <span>Model: Random Survival Forest &nbsp; C-index 0.60</span>
  <span>|</span>
  <span>Dispatch: Priority-Weighted Greedy &nbsp; 37% PW-delay reduction vs baseline</span>
</div>
""", unsafe_allow_html=True)

mode = st.radio("Operating Mode", ["Replay Log", "Live Inject"], horizontal=True, label_visibility="collapsed")

col_main, col_side = st.columns([7, 3])

# ---- MODE: REPLAY ------------------------------------------------------------
if mode == "Replay Log":
    with col_side:
        st.markdown("### ⏱️ Time Control")
        step_idx = st.slider("Timestep", 0, len(all_timesteps)-1, default_step_idx, label_visibility="collapsed")
        current_t = all_timesteps[step_idx]
        st.write(f"**Current Time:** {pd.to_datetime(current_t).strftime('%Y-%m-%d %H:%M IST')}")
        
        # Cumulative metrics up to current_t
        past_assigns = assign_df[assign_df["timestep"] <= current_t]
        opt_past = past_assigns[past_assigns["strategy"] == "OPT"]
        bsl_past = past_assigns[past_assigns["strategy"] == "BSL"]
        
        opt_delay_saved_pct = metrics["pct_improvement"]
        
        st.markdown("<div class='section-header'>Impact (Cumulative)</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">PW-Delay Reduction vs Naive Baseline</div>
            <div class="metric-value">{opt_delay_saved_pct}%</div>
            <div class="metric-sub">SAARTHI OPT vs nearest-unit dispatch</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-box green">
            <div class="metric-label">Incidents Intercepted by OPT</div>
            <div class="metric-value">{len(opt_past):,}</div>
            <div class="metric-sub">Dispatched units from this session</div>
        </div>
        """, unsafe_allow_html=True)
        n_active_now = len(cascade_df[(cascade_df["timestep"] == current_t) & (cascade_df["phase"] != "cleared")])
        st.markdown(f"""
        <div class="metric-box amber">
            <div class="metric-label">Active Incidents Now</div>
            <div class="metric-value">{n_active_now}</div>
            <div class="metric-sub">Growing or shrinking at selected timestep</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Live Priority Queue</div>", unsafe_allow_html=True)
        
        # Get active incidents at current_t from cascade (exclude cleared)
        active_snaps_queue = cascade_df[
            (cascade_df["timestep"] == current_t) & (cascade_df["phase"] != "cleared")
        ].sort_values("priority_score", ascending=False)
        
        if len(active_snaps_queue) == 0:
            st.info("No active incidents at this timestep.")
        else:
            st.markdown("<div class='queue-scroll'>", unsafe_allow_html=True)
            for rank, (_, snap) in enumerate(active_snaps_queue.head(20).iterrows(), 1):
                tag_class = "tag-mdl" if snap["event_cause"] not in LOW_CONF_TYPES else "tag-fbk"
                tag_label = "[MDL]" if snap["event_cause"] not in LOW_CONF_TYPES else "[FBK]"
                phase_class = snap["phase"]
                score = snap["priority_score"]
                card_class = "crit" if score >= 3.0 else ("high" if score >= 2.0 else ("med" if score >= 1.0 else "low"))
                cause_clean = str(snap["event_cause"]).replace("_", " ").title()
                corridor_clean = str(snap["corridor"]).title()
                st.markdown(f"""
                <div class="queue-card {card_class}">
                    <div class="card-rank">#{rank:02d} &nbsp; <span class="tag {tag_class}">{tag_label}</span>
                        <span class="phase-pill {phase_class}" style="margin-left:4px">{phase_class.upper()}</span>
                    </div>
                    <div style="display:flex;align-items:baseline;gap:8px;margin:3px 0;">
                        <span class="card-score">{score:.2f}</span>
                        <span class="card-cause">{cause_clean}</span>
                    </div>
                    <div class="card-meta">
                        {corridor_clean} &nbsp;&bull;&nbsp; Impact {snap['congestion_impact']:.2f} &nbsp;&bull;&nbsp; Corr {snap['corridor_crit']:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            if len(active_snaps_queue) > 20:
                st.markdown(f"<div class='disclaimer'>+{len(active_snaps_queue)-20} more active incidents not shown</div>", unsafe_allow_html=True)

    with col_main:
        active_snaps = cascade_df[cascade_df["timestep"] == current_t]
        
        # Build Folium Map
        # Center map dynamically or default to Bangalore CBD
        center_lat = active_snaps["latitude"].mean() if len(active_snaps) > 0 else 12.9716
        center_lon = active_snaps["longitude"].mean() if len(active_snaps) > 0 else 77.5946
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB dark_matter")
        
        # Draw cascade circles
        for _, snap in active_snaps.iterrows():
            color = PHASE_COLORS.get(snap["phase"], "#888888")
            # Cap radius for visualization to avoid overwhelming the map
            r_meters = min(snap["radius_km"] * 1000, 3000) 
            
            # Opacity based on intensity
            fill_opacity = max(0.1, snap["intensity"] * 0.6)
            
            popup_html = f"""
            <b>{str(snap['event_cause']).replace('_', ' ').title()}</b><br>
            Score: {snap['priority_score']:.2f}<br>
            Phase: {snap['phase'].capitalize()}<br>
            Corridor: {str(snap['corridor']).title()}
            """
            
            folium.Circle(
                location=[snap["latitude"], snap["longitude"]],
                radius=max(r_meters, 150),  # Minimum visual radius
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=fill_opacity,
                popup=folium.Popup(popup_html, max_width=200)
            ).add_to(m)
            
            # Add a small dot at the epicenter
            folium.CircleMarker(
                location=[snap["latitude"], snap["longitude"]],
                radius=3, color="white", fill=True, fill_color="white", fill_opacity=0.8
            ).add_to(m)
            
        # Draw Response Units (based on last known assignments up to current_t)
        # In Replay, we approximate unit position. If a unit was assigned recently (within 2 hours), 
        # we show it at the incident. Else at depot.
        active_assigns = opt_past[opt_past["timestep"] >= current_t - timedelta(hours=2)]
        unit_locs = {}
        for uid in range(1, 9):
            unit_locs[uid] = {"lat": UNIT_DEPOTS[uid][0], "lon": UNIT_DEPOTS[uid][1], "status": "Free", "color": "blue"}
            
        # Update with active assignments
        for _, a in active_assigns.iterrows():
            uid = int(a["unit_id"])
            inc = triage_df[triage_df["id"] == a["incident_id"]].iloc[0]
            unit_locs[uid] = {"lat": inc["latitude"], "lon": inc["longitude"], "status": "Dispatched", "color": "#1f6feb"}
            
        for uid, loc in unit_locs.items():
            folium.Marker(
                location=[loc["lat"], loc["lon"]],
                icon=folium.Icon(color="darkblue" if loc["status"]=="Free" else "blue", icon="car", prefix="fa"),
                tooltip=f"Unit {uid} - {loc['status']}"
            ).add_to(m)

        # Render Map
        st_folium(m, width=900, height=570, returned_objects=[])
        st.markdown("""
        <div class="map-legend">
            <span><span class="legend-dot" style="background:#ff4444"></span>Growing (unattended)</span>
            <span><span class="legend-dot" style="background:#ffaa00"></span>Shrinking (unit en route)</span>
            <span><span class="legend-dot" style="background:#44ff88"></span>Cleared</span>
            <span style="margin-left:auto;font-style:italic;">Circle radius = modelled congestion footprint. ILLUSTRATIVE — not a measured vehicle count.</span>
        </div>
        """, unsafe_allow_html=True)


# ---- MODE: INJECT ------------------------------------------------------------
elif mode == "Live Inject":
    with col_side:
        st.markdown("### 🚨 Inject Incident")
        st.write("Click the map to set location, or manually inject below.")
        
        with st.form("inject_form"):
            cause = st.selectbox("Event Cause", ["vehicle_breakdown", "accident", "water_logging", "tree_fall", "construction", "pot_holes", "others"])
            veh = st.selectbox("Vehicle Type", ["private_car", "heavy_vehicle", "bmtc_bus", "two_wheeler", "unknown"])
            corridor = st.selectbox("Corridor (Approx)", ["tumkur road", "mysore road", "bellary road 1", "orr east 1", "non-corridor", "other_corridor"])
            priority = st.selectbox("Reported Priority", ["high", "low"])
            road_closed = st.checkbox("Requires Road Closure", value=False)
            
            submitted = st.form_submit_button("Inject & Triage", type="primary")

        if submitted:
            # Generate a mock incident at map center if clicked, else default
            lat, lon = 12.9716, 77.5946
            if 'last_clicked' in st.session_state and st.session_state.last_clicked:
                lat = st.session_state.last_clicked['lat']
                lon = st.session_state.last_clicked['lng']
            
            row_dict = {
                "event_cause": cause, "veh_type": veh, "corridor": corridor,
                "priority": priority, "requires_road_closure": road_closed,
                "hour_of_day_ist": datetime.now().hour
            }
            
            med_min, risk_norm = predict_clearance(row_dict)
            score, ci, cc, src_tag = compute_triage_score(row_dict, risk_norm)
            
            # Find best unit (Greedy SAARTHI policy)
            best_unit = None
            best_composite = -1.0
            best_tt = 0
            for uid, loc in UNIT_DEPOTS.items():
                tt = travel_time_min(loc[0], loc[1], lat, lon)
                comp = score / (tt + 1.0)
                if comp > best_composite:
                    best_composite = comp
                    best_unit = uid
                    best_tt = tt
            
            st.session_state.injected_incident = {
                "cause": cause, "lat": lat, "lon": lon, "score": score, 
                "tag": src_tag, "med_min": med_min, "unit": best_unit, "eta": best_tt, "ci": ci
            }

        if 'injected_incident' in st.session_state:
            inc = st.session_state.injected_incident
            tag_cls = 'tag-mdl' if inc['tag'] == '[MDL]' else 'tag-fbk'
            clr_str = f"{inc['med_min']:.0f} min" if inc['tag'] == '[MDL]' and inc['med_min'] < 1440 else "Duration uncertain — triaged by severity"
            st.markdown(f"""
            <div class="inject-result">
                <div class="inject-title">TRIAGE COMPLETE</div>
                <div class="inject-row"><span class="tag {tag_cls}">{inc['tag']}</span> Priority Score: <strong>{inc['score']:.2f}</strong></div>
                <div class="inject-row">Congestion Impact: {inc['ci']:.2f}</div>
                <div class="inject-row">Est. Clearance: {clr_str}</div>
                <div class="inject-unit">DISPATCH: Unit {inc['unit']} &rarr; ETA {inc['eta']:.0f} min</div>
                <div class="disclaimer" style="margin-top:8px;">Haversine / 20 km/h travel estimate. No routing API.</div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_main:
        # Default Map
        m2 = folium.Map(location=[12.9716, 77.5946], zoom_start=11, tiles="CartoDB dark_matter")
        m2.add_child(folium.LatLngPopup()) # Allow clicking
        
        # Draw Units at Depots
        for uid, loc in UNIT_DEPOTS.items():
            folium.Marker(
                location=[loc[0], loc[1]],
                icon=folium.Icon(color="darkblue", icon="car", prefix="fa"),
                tooltip=f"Unit {uid} (Depot)"
            ).add_to(m2)
            
        # Draw injected incident if exists
        if 'injected_incident' in st.session_state:
            inc = st.session_state.injected_incident
            folium.Circle(
                location=[inc["lat"], inc["lon"]],
                radius=400,
                color="#ff4444",
                weight=1, fill=True, fill_color="#ff4444", fill_opacity=0.6,
                popup="Injected Incident"
            ).add_to(m2)
            
            folium.PolyLine(
                locations=[(UNIT_DEPOTS[inc["unit"]][0], UNIT_DEPOTS[inc["unit"]][1]), (inc["lat"], inc["lon"])],
                color="#58a6ff", weight=2, dash_array="10"
            ).add_to(m2)
        
        out = st_folium(m2, width=900, height=600)
        
        # Capture clicks for the inject mode
        if out.get("last_clicked"):
            st.session_state.last_clicked = out["last_clicked"]

