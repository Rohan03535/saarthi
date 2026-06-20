"""
SAARTHI City OS - The Unified Mobility Operating System
========================================================
Three-tab command center:
  Tab 1: Tactical Triage   (Proven dispatch engine + 241 hook)
  Tab 2: Strategic Command  (Planned events, barricading, diversions, ecosystem)
  Tab 3: Autonomous Edge-AI (CV-triggered gridlock detection)
"""

import streamlit as st
import folium
from folium import plugins as folium_plugins
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import cv2
import json
import time
import os
import math
import tempfile

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAARTHI City OS",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Root ──────────────────────────────────────────── */
.stApp { background: #060a13; color: #c9d1d9; font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] { background: #060a13; }
.block-container { padding-top: 0.5rem !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Master Header ─────────────────────────────────── */
.os-header {
    background: linear-gradient(135deg, #0a1628 0%, #111d35 50%, #0d1a2d 100%);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 12px 24px;
    margin-bottom: 8px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 4px 20px rgba(0,100,255,0.08);
}
.os-logo {
    font-size: 28px; font-weight: 800; letter-spacing: 2px;
    background: linear-gradient(135deg, #60a5fa, #3b82f6, #2563eb);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.os-subtitle { font-size: 11px; color: #64748b; letter-spacing: 1px; margin-top: 2px; }
.os-badge {
    background: rgba(59,130,246,0.12); border: 1px solid #3b82f6;
    color: #60a5fa; font-size: 10px; font-weight: 700;
    padding: 4px 12px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    animation: pulse-badge 2s infinite;
}
@keyframes pulse-badge { 0%,100%{opacity:1} 50%{opacity:0.6} }

/* ── Status Bar ────────────────────────────────────── */
.status-strip {
    background: linear-gradient(90deg, #0f1729 0%, #131d33 100%);
    border: 1px solid #1a2744; border-radius: 6px;
    padding: 6px 16px; margin-bottom: 10px;
    display: flex; align-items: center; gap: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #64748b;
}
.sd { width:7px; height:7px; border-radius:50%; display:inline-block; margin-right:4px; }
.sd-g { background:#22c55e; box-shadow: 0 0 8px #22c55e; }
.sd-r { background:#ef4444; box-shadow: 0 0 8px #ef4444; }

/* ── Tab Styling ───────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #0a1224; border-radius: 8px;
    padding: 4px; border: 1px solid #1a2744;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 6px !important;
    color: #64748b !important; font-weight: 600 !important;
    font-size: 13px !important; padding: 8px 20px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a5f, #1a365d) !important;
    color: #60a5fa !important;
    box-shadow: 0 2px 10px rgba(59,130,246,0.2) !important;
}

/* ── Section Headers ───────────────────────────────── */
.sec-h {
    font-size: 10px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #475569;
    border-bottom: 1px solid #1a2744;
    padding-bottom: 4px; margin: 16px 0 10px 0;
}

/* ── Metric Cards ──────────────────────────────────── */
.m-card {
    background: linear-gradient(135deg, #0f172a, #111d35);
    border: 1px solid #1e3a5f; border-radius: 8px;
    padding: 14px 16px; margin-bottom: 8px;
    transition: border-color 0.3s;
}
.m-card:hover { border-color: #3b82f6; }
.m-card.red   { border-left: 3px solid #ef4444; }
.m-card.green { border-left: 3px solid #22c55e; }
.m-card.blue  { border-left: 3px solid #3b82f6; }
.m-card.amber { border-left: 3px solid #f59e0b; }
.m-val { font-size: 26px; font-weight: 800; color: #f1f5f9; font-family:'JetBrains Mono',monospace; }
.m-lbl { font-size: 9px; color: #64748b; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px; }
.m-sub { font-size: 10px; color: #475569; margin-top: 4px; }

/* ── Hook Card (Hero) ──────────────────────────────── */
.hook-card {
    background: linear-gradient(135deg, #0f172a 0%, #1a1a2e 50%, #16213e 100%);
    border: 1px solid #22c55e; border-radius: 10px;
    padding: 18px 22px; margin-bottom: 12px;
    box-shadow: 0 0 20px rgba(34,197,94,0.08);
}
.hook-title { font-size: 13px; font-weight: 700; color: #22c55e; margin-bottom: 8px; letter-spacing: 1px; }
.hook-number { font-size: 42px; font-weight: 800; color: #ffffff; font-family:'JetBrains Mono',monospace; line-height: 1; }
.hook-desc { font-size: 12px; color: #94a3b8; margin-top: 6px; line-height: 1.5; }

/* ── Queue Cards ───────────────────────────────────── */
.q-card {
    background: #0f172a; border: 1px solid #1e3a5f;
    border-left: 3px solid #334155; border-radius: 6px;
    padding: 8px 12px; margin-bottom: 5px;
}
.q-card.crit { border-left-color: #ef4444; }
.q-card.high { border-left-color: #f59e0b; }
.q-card.med  { border-left-color: #3b82f6; }
.q-rank  { font-size: 9px; color: #475569; font-family:'JetBrains Mono',monospace; }
.q-score { font-size: 15px; font-weight: 700; color: #f1f5f9; font-family:'JetBrains Mono',monospace; }
.q-cause { font-size: 12px; font-weight: 600; color: #e2e8f0; }
.q-meta  { font-size: 10px; color: #64748b; margin-top: 2px; }

/* ── Tags ──────────────────────────────────────────── */
.tag { padding:2px 7px; border-radius:4px; font-size:9px; font-weight:700; font-family:'JetBrains Mono',monospace; margin-right:4px; }
.tag-mdl { background:rgba(59,130,246,0.2); color:#60a5fa; border:1px solid rgba(59,130,246,0.3); }
.tag-fbk { background:rgba(239,68,68,0.2); color:#f87171; border:1px solid rgba(239,68,68,0.3); }

/* ── Scenario Cards ────────────────────────────────── */
.scenario-card {
    background: linear-gradient(135deg, #0f172a, #1a1a2e);
    border: 1px solid #1e3a5f; border-radius: 8px;
    padding: 14px; cursor: pointer;
    transition: all 0.3s;
}
.scenario-card:hover { border-color: #3b82f6; transform: translateY(-2px); }
.scenario-card.active { border-color: #ef4444; box-shadow: 0 0 15px rgba(239,68,68,0.15); }
.sc-title { font-size: 14px; font-weight: 700; color: #f1f5f9; }
.sc-meta  { font-size: 11px; color: #64748b; margin-top: 4px; }

/* ── Ecosystem Toggles ─────────────────────────────── */
.eco-toggle {
    background: #0f172a; border: 1px solid #1e3a5f; border-radius: 8px;
    padding: 12px; margin-bottom: 8px;
    display: flex; align-items: center; gap: 10px;
}
.eco-icon { font-size: 20px; }
.eco-label { font-size: 12px; font-weight: 600; color: #e2e8f0; }
.eco-sub { font-size: 10px; color: #64748b; }
.eco-active { border-color: #22c55e; box-shadow: 0 0 10px rgba(34,197,94,0.1); }

/* ── CV Feed ───────────────────────────────────────── */
.cv-frame {
    border: 2px solid #1e3a5f; border-radius: 8px;
    overflow: hidden;
}
.cv-frame.alert { border-color: #ef4444; box-shadow: 0 0 20px rgba(239,68,68,0.3); animation: flash-border 0.5s infinite; }
@keyframes flash-border { 0%,100%{border-color:#ef4444} 50%{border-color:#991b1b} }

.density-gauge {
    background: #0f172a; border: 1px solid #1e3a5f; border-radius: 8px;
    padding: 14px; text-align: center;
}
.gauge-value { font-size: 48px; font-weight: 800; font-family:'JetBrains Mono',monospace; }
.gauge-green  { color: #22c55e; }
.gauge-amber  { color: #f59e0b; }
.gauge-red    { color: #ef4444; }

/* ── Alert Banner ──────────────────────────────────── */
.alert-banner {
    background: linear-gradient(90deg, #7f1d1d, #991b1b, #7f1d1d);
    border: 1px solid #ef4444; border-radius: 8px;
    padding: 16px; text-align: center;
    animation: flash-bg 1s infinite;
}
@keyframes flash-bg { 0%,100%{opacity:1} 50%{opacity:0.85} }
.alert-text { font-size: 18px; font-weight: 800; color: #ffffff; letter-spacing: 2px; }
.alert-sub { font-size: 12px; color: #fca5a5; margin-top: 4px; }

/* ── Map Legend ────────────────────────────────────── */
.map-legend {
    background: #0f172a; border: 1px solid #1e3a5f; border-radius: 6px;
    padding: 8px 14px; margin-top: 6px;
    display: flex; align-items: center; gap: 16px;
    font-size: 10px; color: #64748b;
}
.ld { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:4px; }

/* ── Barricade Deploy List ─────────────────────────── */
.deploy-card {
    background: #0f172a; border: 1px solid #f59e0b;
    border-radius: 6px; padding: 10px 14px; margin-bottom: 6px;
}
.deploy-loc { font-size: 13px; font-weight: 700; color: #f59e0b; }
.deploy-meta { font-size: 11px; color: #94a3b8; margin-top: 2px; }

/* ── Divider ───────────────────────────────────────── */
.divider { height: 1px; background: #1e3a5f; margin: 12px 0; }

/* ── Disclaimer ────────────────────────────────────── */
.disclaimer { font-size: 9px; color: #475569; text-align: center; font-style: italic; padding-top: 4px; }

/* ── Scrollbar ─────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #060a13; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── File Paths ────────────────────────────────────────────────────────────────
BASE_DIR     = r"C:\Users\Rohan\Pictures\saarthi"
TRIAGE_PATH  = os.path.join(BASE_DIR, "triage_queue.parquet")
ASSIGN_PATH  = os.path.join(BASE_DIR, "phase4_assignments.parquet")
CASCADE_PATH = os.path.join(BASE_DIR, "phase5_cascade_snapshots.parquet")
METRICS_PATH = os.path.join(BASE_DIR, "phase4_metrics.json")
VIDEO_PATH   = os.path.join(BASE_DIR, "demo_assets", "traffic_cam.mp4")
HOOK_IMG     = os.path.join(BASE_DIR, "saarthi_impact_hook.png")

LOW_CONF_TYPES = {"tree_fall", "public_event"}

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    triage  = pd.read_parquet(TRIAGE_PATH)
    assigns = pd.read_parquet(ASSIGN_PATH)
    cascade = pd.read_parquet(CASCADE_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return triage, assigns, cascade, metrics

triage_df, assign_df, cascade_df, metrics = load_data()

# Compute hook numbers
@st.cache_data
def compute_hook():
    opt_ids = assign_df[assign_df["strategy"] == "OPT"]["incident_id"].unique()
    bsl_ids = assign_df[assign_df["strategy"] == "BSL"]["incident_id"].unique()
    t = triage_df.copy()
    t["d_opt"] = t["id"].isin(opt_ids)
    t["d_bsl"] = t["id"].isin(bsl_ids)
    hp = t[t["priority_score"] >= 2.0]
    rescued = hp[hp["d_opt"] & ~hp["d_bsl"]]
    lp = t[t["priority_score"] < 1.0]
    dropped = lp[lp["d_bsl"] & ~lp["d_opt"]]
    return len(rescued), len(dropped), rescued["priority_score"].mean(), dropped["priority_score"].mean()

n_rescued, n_dropped, mean_rescued, mean_dropped = compute_hook()

# ── Scenario Data ─────────────────────────────────────────────────────────────
SCENARIOS = {
    "Political Rally at Freedom Park": {
        "center": [12.9716, 77.5756],
        "radius_km": 1.5,
        "crowd": "50,000+",
        "duration": "4 hours",
        "severity": "CRITICAL",
        "corridors": ["Seshadri Road", "Kasturba Road", "Race Course Road", "Majestic"],
        "barricades": [
            {"name": "Anand Rao Circle",      "lat": 12.9790, "lon": 77.5713, "units": 3, "type": "Full Block"},
            {"name": "Mysore Bank Circle",     "lat": 12.9650, "lon": 77.5740, "units": 2, "type": "Partial"},
            {"name": "KR Circle",              "lat": 12.9752, "lon": 77.5828, "units": 2, "type": "Full Block"},
            {"name": "Cubbon Rd / MG Rd Jn",   "lat": 12.9748, "lon": 77.5870, "units": 2, "type": "Partial"},
            {"name": "Seshadri Rd Gate",       "lat": 12.9780, "lon": 77.5690, "units": 2, "type": "Full Block"},
            {"name": "Race Course Rd South",   "lat": 12.9660, "lon": 77.5780, "units": 1, "type": "Partial"},
        ],
        "diversions": [
            {"name": "Northern Bypass via Palace Road",
             "color": "#22c55e",
             "points": [[12.9830, 77.5650], [12.9850, 77.5730], [12.9860, 77.5850], [12.9840, 77.5950]]},
            {"name": "Southern Bypass via Bull Temple Rd",
             "color": "#3b82f6",
             "points": [[12.9600, 77.5650], [12.9580, 77.5750], [12.9590, 77.5880], [12.9620, 77.5960]]},
            {"name": "Eastern Bypass via Richmond Rd",
             "color": "#a855f7",
             "points": [[12.9700, 77.5900], [12.9680, 77.5950], [12.9650, 77.6010], [12.9680, 77.6080]]},
        ],
    },
    "IPL Match at Chinnaswamy Stadium": {
        "center": [12.9788, 77.5996],
        "radius_km": 2.0,
        "crowd": "40,000",
        "duration": "5 hours",
        "severity": "HIGH",
        "corridors": ["MG Road", "Cubbon Road", "Queens Road", "St Marks Road"],
        "barricades": [
            {"name": "MG Road Metro Station",  "lat": 12.9757, "lon": 77.6069, "units": 3, "type": "Full Block"},
            {"name": "Cubbon Park Main Gate",   "lat": 12.9763, "lon": 77.5920, "units": 2, "type": "Partial"},
            {"name": "Queens Rd / Palace Jn",   "lat": 12.9840, "lon": 77.5940, "units": 2, "type": "Full Block"},
            {"name": "Richmond Circle",         "lat": 12.9670, "lon": 77.5990, "units": 2, "type": "Partial"},
            {"name": "Trinity Circle",          "lat": 12.9720, "lon": 77.6095, "units": 2, "type": "Full Block"},
            {"name": "Minsk Square",            "lat": 12.9820, "lon": 77.5870, "units": 1, "type": "Partial"},
        ],
        "diversions": [
            {"name": "Northern Bypass via Bellary Rd",
             "color": "#22c55e",
             "points": [[12.9900, 77.5850], [12.9920, 77.5950], [12.9910, 77.6100], [12.9880, 77.6200]]},
            {"name": "Southern Bypass via Hosur Rd",
             "color": "#3b82f6",
             "points": [[12.9600, 77.5900], [12.9580, 77.6000], [12.9570, 77.6100], [12.9590, 77.6200]]},
        ],
    },
    "VIP Motorcade: Airport to Vidhana Soudha": {
        "center": [12.9791, 77.5913],
        "radius_km": 1.8,
        "crowd": "N/A (VIP corridor lock)",
        "duration": "2 hours",
        "severity": "EXTREME",
        "corridors": ["Airport Road", "Old Airport Road", "Bellary Road", "Raj Bhavan Road"],
        "barricades": [
            {"name": "Mekhri Circle",           "lat": 12.9950, "lon": 77.5780, "units": 3, "type": "Full Block"},
            {"name": "Windsor Manor Junction",  "lat": 12.9830, "lon": 77.5900, "units": 2, "type": "Full Block"},
            {"name": "Raj Bhavan Gate",          "lat": 12.9810, "lon": 77.5850, "units": 3, "type": "Full Block"},
            {"name": "Cubbon Park North Gate",   "lat": 12.9810, "lon": 77.5930, "units": 2, "type": "Partial"},
            {"name": "Vidhana Soudha Circle",    "lat": 12.9795, "lon": 77.5908, "units": 3, "type": "Full Block"},
        ],
        "diversions": [
            {"name": "Eastern Bypass via Old Airport Rd",
             "color": "#22c55e",
             "points": [[13.0050, 77.5900], [12.9950, 77.6050], [12.9850, 77.6100], [12.9750, 77.6050]]},
            {"name": "Western Bypass via Tumkur Rd",
             "color": "#f59e0b",
             "points": [[13.0050, 77.5650], [12.9950, 77.5600], [12.9800, 77.5620], [12.9700, 77.5700]]},
        ],
    },
}

# ── Helper: Haversine ─────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="os-header">
  <div>
    <div class="os-logo">SAARTHI</div>
    <div class="os-subtitle">UNIFIED MOBILITY OPERATING SYSTEM &nbsp;|&nbsp; BENGALURU TRAFFIC POLICE</div>
  </div>
  <div class="os-badge"><span class="sd sd-g"></span> CITY OS v2.0</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="status-strip">
  <span><span class="sd sd-g"></span>SYSTEM ONLINE</span>
  <span>|</span>
  <span>Source: BTP ASTraM Log &middot; 8,173 records</span>
  <span>|</span>
  <span>Model: Random Survival Forest &middot; C-index 0.60</span>
  <span>|</span>
  <span>Dispatch Engine: 37% PW-delay reduction &middot; 47% held-out</span>
  <span>|</span>
  <span>Hook: {n_rescued} critical incidents rescued</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "⚡  TACTICAL TRIAGE",
    "🎯  STRATEGIC COMMAND",
    "👁️  AUTONOMOUS EDGE-AI",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: TACTICAL TRIAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_map, col_side = st.columns([7, 3])

    with col_side:
        # ── Hero Hook Card ────────────────────────────────────────────────
        st.markdown(f"""
        <div class="hook-card">
            <div class="hook-title">THE SATURATION HOOK</div>
            <div class="hook-number">{n_rescued}</div>
            <div class="hook-desc">
                Critical chokepoints <strong>rescued</strong> by SAARTHI that naive
                dispatch abandoned. Traded {n_dropped} low-priority calls
                (avg score {mean_dropped:.2f}) for {n_rescued} high-impact incidents
                (avg score {mean_rescued:.2f}). That's a <strong>{mean_rescued/mean_dropped:.0f}x priority ratio</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Key Metrics ───────────────────────────────────────────────────
        st.markdown('<div class="sec-h">PERFORMANCE METRICS</div>', unsafe_allow_html=True)

        pct = metrics["pct_improvement"]
        st.markdown(f"""
        <div class="m-card blue">
            <div class="m-lbl">Priority-Weighted Delay Reduction</div>
            <div class="m-val">{pct}%</div>
            <div class="m-sub">vs nearest-unit baseline &middot; 8,057 incidents &middot; 5 months</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="m-card green">
            <div class="m-lbl">Held-Out Validation (March 7)</div>
            <div class="m-val">47.3%</div>
            <div class="m-sub">45 test-only incidents &middot; Zero leakage &middot; Same 8 units</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="m-card amber">
            <div class="m-lbl">Total Incidents in Dataset</div>
            <div class="m-val">8,173</div>
            <div class="m-sub">BTP ASTraM &middot; Nov 2023 - Apr 2024 &middot; Real anonymised data</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Priority Queue ────────────────────────────────────────────────
        st.markdown('<div class="sec-h">LIVE PRIORITY QUEUE (PEAK WINDOW)</div>', unsafe_allow_html=True)

        # Show a representative timestep
        timesteps = sorted(cascade_df["timestep"].unique())
        # Find the peak active-incident timestep
        active_counts = cascade_df[cascade_df["phase"] != "cleared"].groupby("timestep").size()
        if len(active_counts) > 0:
            peak_t = active_counts.idxmax()
        else:
            peak_t = timesteps[len(timesteps)//2]

        active_q = cascade_df[
            (cascade_df["timestep"] == peak_t) & (cascade_df["phase"] != "cleared")
        ].sort_values("priority_score", ascending=False)

        for rank, (_, row) in enumerate(active_q.head(12).iterrows(), 1):
            score = row["priority_score"]
            card_cls = "crit" if score >= 3.0 else ("high" if score >= 2.0 else "med")
            tag_cls = "tag-mdl" if row["event_cause"] not in LOW_CONF_TYPES else "tag-fbk"
            tag_lbl = "MDL" if row["event_cause"] not in LOW_CONF_TYPES else "FBK"
            cause = str(row["event_cause"]).replace("_", " ").title()
            corridor = str(row["corridor"]).title()
            st.markdown(f"""
            <div class="q-card {card_cls}">
                <div class="q-rank">#{rank:02d} <span class="tag {tag_cls}">{tag_lbl}</span></div>
                <div style="display:flex;align-items:baseline;gap:8px;margin:2px 0;">
                    <span class="q-score">{score:.2f}</span>
                    <span class="q-cause">{cause}</span>
                </div>
                <div class="q-meta">{corridor} &middot; Impact {row['congestion_impact']:.2f} &middot; Phase: {row['phase'].upper()}</div>
            </div>
            """, unsafe_allow_html=True)

        if len(active_q) > 12:
            st.markdown(f'<div class="disclaimer">+{len(active_q)-12} more active incidents</div>', unsafe_allow_html=True)

    with col_map:
        # ── Cascade Map ───────────────────────────────────────────────────
        snaps = cascade_df[cascade_df["timestep"] == peak_t]
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB dark_matter")

        for _, s in snaps.iterrows():
            if pd.isna(s.get("latitude")) or pd.isna(s.get("longitude")):
                continue
            phase = s["phase"]
            color = "#ef4444" if phase == "growing" else ("#f59e0b" if phase == "shrinking" else "#22c55e")
            r_km = s.get("radius_km", 0.1)
            folium.Circle(
                location=[s["latitude"], s["longitude"]],
                radius=max(r_km * 1000, 50),
                color=color, fill=True, fill_color=color, fill_opacity=0.25, weight=1,
                tooltip=f"{s['event_cause']} | Score: {s['priority_score']:.2f} | {phase}",
            ).add_to(m)

        st_folium(m, width=900, height=620, returned_objects=[])

        st.markdown("""
        <div class="map-legend">
            <span><span class="ld" style="background:#ef4444"></span>Growing</span>
            <span><span class="ld" style="background:#f59e0b"></span>Shrinking</span>
            <span><span class="ld" style="background:#22c55e"></span>Cleared</span>
            <span style="margin-left:auto;font-style:italic;">Illustrative congestion footprint. Not a measured vehicle count.</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: STRATEGIC COMMAND
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_ctrl, col_smap = st.columns([3, 7])

    with col_ctrl:
        st.markdown('<div class="sec-h">PLANNED EVENT SCENARIOS</div>', unsafe_allow_html=True)

        scenario_name = st.selectbox(
            "Select Scenario",
            list(SCENARIOS.keys()),
            label_visibility="collapsed",
        )
        sc = SCENARIOS[scenario_name]

        # Scenario info card
        sev_color = "#ef4444" if sc["severity"] == "EXTREME" else ("#f59e0b" if sc["severity"] == "CRITICAL" else "#3b82f6")
        st.markdown(f"""
        <div class="scenario-card active">
            <div class="sc-title">{scenario_name}</div>
            <div class="sc-meta">
                <span style="color:{sev_color};font-weight:700;">{sc['severity']}</span> &middot;
                Crowd: {sc['crowd']} &middot; Duration: {sc['duration']}
            </div>
            <div class="sc-meta" style="margin-top:6px;">
                Affected: {', '.join(sc['corridors'])}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Blast Radius Control ──────────────────────────────────────────
        st.markdown('<div class="sec-h">CONGESTION BLAST RADIUS</div>', unsafe_allow_html=True)
        blast_radius = st.slider("Predicted Radius (km)", 0.5, 4.0, sc["radius_km"], 0.1)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Barricade Deployment List ─────────────────────────────────────
        st.markdown('<div class="sec-h">BARRICADE DEPLOYMENT</div>', unsafe_allow_html=True)
        total_units = 0
        for bp in sc["barricades"]:
            total_units += bp["units"]
            st.markdown(f"""
            <div class="deploy-card">
                <div class="deploy-loc">🚧 {bp['name']}</div>
                <div class="deploy-meta">{bp['type']} &middot; Deploy {bp['units']} units</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="m-card red">
            <div class="m-lbl">Total Units Required</div>
            <div class="m-val">{total_units}</div>
            <div class="m-sub">For barricade deployment</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Ecosystem Toggles ─────────────────────────────────────────────
        st.markdown('<div class="sec-h">ECOSYSTEM INTEGRATION</div>', unsafe_allow_html=True)

        eco_sms = st.toggle("📱 Citizen Geo-Fenced SMS Alert", value=True)
        if eco_sms:
            st.markdown(f"""
            <div class="eco-toggle eco-active">
                <div class="eco-icon">📱</div>
                <div>
                    <div class="eco-label">ACTIVE: Citizen Alert</div>
                    <div class="eco-sub">"BTP Alert: Avoid {sc['corridors'][0]} due to {scenario_name.split(' at ')[0].lower()}. Use marked diversions. Expected delay: 45+ mins."</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        eco_flipkart = st.toggle("📦 Flipkart Logistics API Push", value=True)
        if eco_flipkart:
            st.markdown("""
            <div class="eco-toggle eco-active">
                <div class="eco-icon">📦</div>
                <div>
                    <div class="eco-label">ACTIVE: Logistics Geo-fence</div>
                    <div class="eco-sub">Pushing congestion zone to e-commerce delivery networks (Flipkart e-kart, Swiggy, Dunzo). Rerouting 340+ active deliveries.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        eco_signal = st.toggle("🚦 Adaptive Signal Green Wave", value=False)
        if eco_signal:
            st.markdown("""
            <div class="eco-toggle eco-active">
                <div class="eco-icon">🚦</div>
                <div>
                    <div class="eco-label">ACTIVE: Signal Override</div>
                    <div class="eco-sub">Forcing 90-second green phase on diversion corridors to flush civilian traffic out of danger zone.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_smap:
        # ── Strategic Map ─────────────────────────────────────────────────
        sm = folium.Map(location=sc["center"], zoom_start=14, tiles="CartoDB dark_matter")

        # Blast Radius
        folium.Circle(
            location=sc["center"],
            radius=blast_radius * 1000,
            color="#ef4444", fill=True, fill_color="#ef4444",
            fill_opacity=0.12, weight=2,
            tooltip=f"Predicted Congestion Zone: {blast_radius}km radius",
        ).add_to(sm)

        # Inner high-impact zone
        folium.Circle(
            location=sc["center"],
            radius=blast_radius * 500,
            color="#ff6b6b", fill=True, fill_color="#ff6b6b",
            fill_opacity=0.18, weight=1,
            tooltip="High-Impact Core Zone",
        ).add_to(sm)

        # Event marker
        folium.Marker(
            location=sc["center"],
            icon=folium.DivIcon(html=f"""
                <div style="background:#ef4444;color:white;padding:4px 10px;border-radius:20px;
                font-size:11px;font-weight:700;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,0.4);
                font-family:'Inter',sans-serif;">
                ⚠️ {scenario_name.split(' at ')[-1].upper()}
                </div>"""),
            tooltip=scenario_name,
        ).add_to(sm)

        # Barricade markers
        for bp in sc["barricades"]:
            bc = "#f59e0b" if bp["type"] == "Full Block" else "#fb923c"
            folium.Marker(
                location=[bp["lat"], bp["lon"]],
                icon=folium.DivIcon(html=f"""
                    <div style="background:{bc};color:#000;padding:3px 8px;border-radius:4px;
                    font-size:10px;font-weight:700;white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,0.3);
                    font-family:'JetBrains Mono',monospace;">
                    🚧 {bp['name']}
                    </div>"""),
                tooltip=f"{bp['name']} | {bp['type']} | {bp['units']} units",
            ).add_to(sm)

        # Diversion routes
        for dv in sc["diversions"]:
            folium.PolyLine(
                locations=dv["points"],
                color=dv["color"], weight=4, opacity=0.85,
                tooltip=f"DIVERSION: {dv['name']}",
                dash_array="10 6",
            ).add_to(sm)
            # Label at midpoint
            mid_idx = len(dv["points"]) // 2
            folium.Marker(
                location=dv["points"][mid_idx],
                icon=folium.DivIcon(html=f"""
                    <div style="background:{dv['color']};color:#fff;padding:2px 8px;border-radius:10px;
                    font-size:9px;font-weight:700;white-space:nowrap;box-shadow:0 2px 4px rgba(0,0,0,0.3);
                    font-family:'JetBrains Mono',monospace;">
                    ↗ {dv['name'].split(' via ')[-1]}
                    </div>"""),
            ).add_to(sm)

        st_folium(sm, width=950, height=620, returned_objects=[])

        st.markdown("""
        <div class="map-legend">
            <span><span class="ld" style="background:#ef4444"></span>Blast Radius</span>
            <span><span class="ld" style="background:#f59e0b"></span>Barricade Point</span>
            <span><span class="ld" style="background:#22c55e"></span>Diversion Route</span>
            <span style="margin-left:auto;font-style:italic;">Simulated planned-event response. No live ASTraM connection.</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: AUTONOMOUS EDGE-AI
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_feed, col_ai = st.columns([6, 4])

    with col_ai:
        st.markdown('<div class="sec-h">EDGE-AI STATUS</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="m-card blue">
            <div class="m-lbl">Detection Engine</div>
            <div class="m-val" style="font-size:16px;">OpenCV Density Tracker</div>
            <div class="m-sub">Background subtraction + contour analysis &middot; Real-time</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="m-card amber">
            <div class="m-lbl">Camera Feed</div>
            <div class="m-val" style="font-size:16px;">CAM-07 Hosur Road Jn</div>
            <div class="m-sub">Synthetic CCTV simulation &middot; 800x600 &middot; 20 FPS</div>
        </div>
        """, unsafe_allow_html=True)

        density_placeholder = st.empty()
        status_placeholder  = st.empty()
        trigger_placeholder = st.empty()
        action_placeholder  = st.empty()

    with col_feed:
        st.markdown('<div class="sec-h">LIVE CCTV FEED</div>', unsafe_allow_html=True)
        frame_placeholder = st.empty()

    # ── CV Processing ─────────────────────────────────────────────────────
    if os.path.exists(VIDEO_PATH):
        start_btn = col_ai.button("▶ START AUTONOMOUS MONITORING", type="primary", use_container_width=True)

        if start_btn:
            cap = cv2.VideoCapture(VIDEO_PATH)
            if not cap.isOpened():
                st.error("Could not open video file.")
            else:
                triggered = False
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % 2 != 0:  # Process every other frame for speed
                        continue

                    # ── Brightness thresholding to detect vehicles ─────────
                    # Cars are bright colored objects on a dark gray road
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Road is ~55 brightness, cars are 150+
                    _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

                    # Clean up — remove noise, merge nearby blobs
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

                    # Remove lane markings and overlay text (small bright objects)
                    # by filtering contours on area
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    vehicle_count = 0
                    display = frame.copy()

                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if 300 < area < 12000:  # Vehicle-sized blobs only
                            x, y, w, h = cv2.boundingRect(cnt)
                            aspect = w / max(h, 1)
                            # Filter out long thin lane markings (aspect > 5 or < 0.2)
                            if 0.2 < aspect < 5.0:
                                # Draw bounding box
                                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(display, "VEH", (x, y-5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                                vehicle_count += 1

                    # Calculate density (calibrated: 40 vehicles = 100%)
                    density = min(vehicle_count / 40.0, 1.0)

                    # ── HUD Overlay on frame ──────────────────────────────
                    cv2.putText(display, f"Vehicles: {vehicle_count}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, f"Density: {density*100:.0f}%", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                               (0, 255, 0) if density < 0.6 else ((0, 180, 255) if density < 0.85 else (0, 0, 255)), 2)

                    if density >= 0.85:
                        cv2.putText(display, "GRIDLOCK DETECTED", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Display frame
                    frame_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    # Update density gauge
                    gauge_cls = "gauge-green" if density < 0.6 else ("gauge-amber" if density < 0.85 else "gauge-red")
                    density_placeholder.markdown(f"""
                    <div class="density-gauge">
                        <div class="m-lbl">TRAFFIC DENSITY</div>
                        <div class="gauge-value {gauge_cls}">{density*100:.0f}%</div>
                        <div class="m-sub">{vehicle_count} vehicles detected in frame</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Status
                    if density < 0.6:
                        status_placeholder.markdown("""
                        <div class="m-card green"><div class="m-lbl">STATUS</div><div class="m-val" style="font-size:16px;color:#22c55e;">NORMAL FLOW</div></div>
                        """, unsafe_allow_html=True)
                    elif density < 0.85:
                        status_placeholder.markdown("""
                        <div class="m-card amber"><div class="m-lbl">STATUS</div><div class="m-val" style="font-size:16px;color:#f59e0b;">CONGESTION BUILDING</div></div>
                        """, unsafe_allow_html=True)
                    else:
                        status_placeholder.markdown("""
                        <div class="m-card red"><div class="m-lbl">STATUS</div><div class="m-val" style="font-size:16px;color:#ef4444;">GRIDLOCK DETECTED</div></div>
                        """, unsafe_allow_html=True)

                    # ── AUTO-TRIGGER ──────────────────────────────────────
                    if density >= 0.85 and not triggered:
                        triggered = True
                        trigger_placeholder.markdown("""
                        <div class="alert-banner">
                            <div class="alert-text">⚠️ AUTONOMOUS TRIGGER FIRED</div>
                            <div class="alert-sub">CV density exceeded 85% — SAARTHI activated automatically</div>
                        </div>
                        """, unsafe_allow_html=True)

                        action_placeholder.markdown("""
                        <div class="m-card" style="border-color:#22c55e; margin-top:10px;">
                            <div class="m-lbl" style="color:#22c55e;">AUTONOMOUS ACTIONS EXECUTED</div>
                            <div style="font-size:12px;color:#e2e8f0;line-height:2;margin-top:6px;">
                                ✅ Incident auto-created at CAM-07 location<br>
                                ✅ Priority score computed: <strong>3.42</strong> (Tier-1 corridor)<br>
                                ✅ Unit 3 dispatched — ETA 7 min<br>
                                ✅ Barricades placed at 2 upstream junctions<br>
                                ✅ Diversion route activated via Bannerghatta Rd<br>
                                ✅ Citizen SMS alert triggered (2 km geo-fence)<br>
                                ✅ Flipkart logistics API notified — 23 deliveries rerouted
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    time.sleep(0.05)  # ~20 FPS display rate

                cap.release()
    else:
        with col_feed:
            st.warning(f"Video file not found at `{VIDEO_PATH}`. Run `demo_assets/generate_traffic_video.py` first.")

    with col_ai:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="disclaimer">
            Synthetic CCTV feed for prototype demonstration.
            Production system would connect to BTP's live camera network via ASTraM.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding: 10px 0;">
    <span style="font-size:11px; color:#475569;">
        SAARTHI City OS &middot; Built for Flipkart Gridlock 2.0 &middot;
        Bengaluru Traffic Police + Flipkart &middot;
        <span style="color:#64748b;">Prototype — no live ASTraM/MapmyIndia API connection</span>
    </span>
</div>
""", unsafe_allow_html=True)
