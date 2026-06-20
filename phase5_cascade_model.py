"""
SAARTHI - Phase 5: Cascade Congestion Model
============================================

PURPOSE:
  Illustrative spatial model showing how an unattended incident's congestion
  footprint grows along its corridor over time, and shrinks once a response
  unit is dispatched or the incident clears.

  THIS IS A VISUALIZATION AID, NOT A DEFENSIBLE MEASUREMENT.
  It makes dispatch leverage visible. It does not produce vehicle counts,
  absolute delays, or any number we claim is real. All parameters are
  labelled assumptions.

SPREAD MODEL (simple, two-phase):

  Phase A - GROWING (incident active, no unit dispatched yet):
    radius_km(t) = min(R_max, base_radius + growth_rate_km_per_min * t_elapsed_min)

    growth_rate_km_per_min = corridor_criticality * congestion_impact * K_GROW
    K_GROW = 0.015 km/min  (ASSUMPTION: ~0.9 km/hr spread on a fully-blocked
                            tier-1 corridor. A blocked National Highway lane
                            creates a queue growing at ~1 km/hr in BTP estimates;
                            we use 0.9 to stay conservative.)

    R_max = corridor_criticality * R_MAX_TIER1_KM
    R_MAX_TIER1_KM = 2.5 km  (ASSUMPTION: max observable congestion radius
                              from a single incident on a tier-1 corridor,
                              based on BTP operational experience cited in
                              media reports. Not from this dataset.)

    base_radius = 0.1 km (incident footprint at t=0)

  Phase B - SHRINKING (unit dispatched or incident cleared):
    radius_km(t) = R_at_dispatch * max(0, 1 - t_since_dispatch / clearance_time_min)
    Shrinks linearly to 0 by the time the unit is predicted to clear it.

  Intensity (0 to 1, dimensionless):
    intensity = radius_km / R_max
    Displayed as circle opacity/size on map. NOT a vehicle count.

ALL OUTPUTS LABELLED ILLUSTRATIVE.
No output from this phase should be presented as a measured number.

OUTPUTS:
  phase5_cascade_snapshots.parquet  - per-incident per-step (radius, intensity, phase)
  phase5_sample_day_plot.png        - static matplotlib visualization of one day
"""

import pandas as pd
import numpy as np
import math
import warnings
import matplotlib
matplotlib.use("Agg")   # headless - no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")

# ---- CONFIG ------------------------------------------------------------------
TRIAGE_PATH    = r"C:\Users\Rohan\Pictures\saarthi\triage_queue.parquet"
CLEAN_PATH     = r"C:\Users\Rohan\Pictures\saarthi\saarthi_clean.parquet"
ASSIGN_PATH    = r"C:\Users\Rohan\Pictures\saarthi\phase4_assignments.parquet"
CASCADE_OUT    = r"C:\Users\Rohan\Pictures\saarthi\phase5_cascade_snapshots.parquet"
PLOT_OUT       = r"C:\Users\Rohan\Pictures\saarthi\phase5_sample_day_plot.png"

STEP_MINUTES        = 30      # must match Phase 4 replay step
MEDIAN_CLEARANCE    = 48.5    # Phase 1b fallback

# ---- SPREAD MODEL PARAMETERS (all assumptions) -------------------------------
K_GROW            = 0.015   # km/min growth rate at corridor_crit=1, cong_impact=1
R_MAX_TIER1_KM    = 2.5     # km, max radius on a tier-1 corridor
BASE_RADIUS_KM    = 0.1     # km, footprint at incident start

ASSUMPTIONS = [
    f"K_GROW = {K_GROW} km/min: spread rate on a fully-blocked tier-1 corridor.",
    f"  Rationale: ~0.9 km/hr queue growth (conservative vs BTP ~1 km/hr estimate).",
    f"R_MAX_TIER1 = {R_MAX_TIER1_KM} km: maximum observable radius on tier-1 corridor.",
    f"  Rationale: illustrative cap; not measured from this dataset.",
    f"Base radius = {BASE_RADIUS_KM} km: incident footprint at t=0.",
    f"Shrink model: linear decay from R_at_dispatch to 0 over clearance_time_min.",
    f"Intensity = radius_km / R_max (dimensionless, for visualization only).",
    f"THIS MODEL IS ILLUSTRATIVE. Output is not a vehicle count or absolute delay.",
]

print("=" * 72)
print("SAARTHI - Phase 5: Cascade Congestion Model")
print("=" * 72)
print()
print("[MODEL ASSUMPTIONS - printed verbatim for judges]")
for a in ASSUMPTIONS:
    print(f"  {a}")

# ---- LOAD DATA ---------------------------------------------------------------
triage  = pd.read_parquet(TRIAGE_PATH)
clean   = pd.read_parquet(CLEAN_PATH)
assigns = pd.read_parquet(ASSIGN_PATH)

# Merge clearance / resolution info
merge_cols = ["id", "clearance_time_minutes", "censored",
              "start_datetime", "closed_datetime", "end_datetime", "resolved_datetime"]
merge_cols = [c for c in merge_cols if c in clean.columns]
triage = triage.merge(clean[merge_cols], on="id", how="left", suffixes=("", "_clean"))

triage["start_dt"] = pd.to_datetime(triage["start_datetime"], utc=False)

def pick_resolution(row):
    for col in ["closed_datetime", "end_datetime", "resolved_datetime"]:
        val = row.get(col)
        if pd.notna(val):
            return pd.to_datetime(val)
    return pd.NaT

res_df = clean.set_index("id")[["closed_datetime", "end_datetime", "resolved_datetime"]]
res_df["resolution_dt"] = res_df.apply(pick_resolution, axis=1)
triage = triage.merge(res_df[["resolution_dt"]], on="id", how="left")

def est_resolution(row):
    if pd.notna(row.get("resolution_dt")):
        return row["resolution_dt"]
    ctm = row.get("clearance_time_minutes")
    if pd.notna(ctm) and float(ctm) > 0:
        return row["start_dt"] + pd.Timedelta(minutes=float(ctm))
    return row["start_dt"] + pd.Timedelta(minutes=MEDIAN_CLEARANCE)

def sim_clearance(row):
    ctm = row.get("clearance_time_minutes")
    if pd.notna(ctm) and 0 < float(ctm) <= 1440:
        return float(ctm)
    return MEDIAN_CLEARANCE

triage["est_resolution_dt"] = triage.apply(est_resolution, axis=1)
triage["sim_clearance_min"] = triage.apply(sim_clearance, axis=1)

# Filter valid lat/lon
triage["latitude"]  = pd.to_numeric(triage["latitude"],  errors="coerce")
triage["longitude"] = pd.to_numeric(triage["longitude"], errors="coerce")
triage = triage.dropna(subset=["start_dt", "latitude", "longitude"])
triage = triage[triage["latitude"].between(11.0, 14.0) & triage["longitude"].between(77.0, 78.0)]
triage = triage.reset_index(drop=True)

print(f"\n[LOAD] {len(triage):,} incidents, {len(assigns):,} assignment records")

# ---- BUILD DISPATCH LOOKUP ---------------------------------------------------
# For each incident, find earliest OPT dispatch timestep from Phase 4
opt_assigns = assigns[assigns["strategy"] == "OPT"].copy()
opt_assigns["timestep"] = pd.to_datetime(opt_assigns["timestep"], utc=False)
dispatch_map = (
    opt_assigns.groupby("incident_id")["timestep"].min().to_dict()
)
print(f"[LOAD] {len(dispatch_map):,} incidents have an OPT dispatch record")

# ---- SPREAD MODEL FUNCTIONS --------------------------------------------------
def r_max_for(corridor_crit, cong_impact):
    return max(BASE_RADIUS_KM, corridor_crit * cong_impact * R_MAX_TIER1_KM)

def growth_rate(corridor_crit, cong_impact):
    return corridor_crit * cong_impact * K_GROW

def radius_at(t_elapsed_min, r_max, g_rate,
              dispatched, t_since_dispatch_min, r_at_dispatch, clearance_min):
    """
    Returns (radius_km, phase_label).
    phase_label: 'growing' | 'shrinking' | 'cleared'
    """
    if dispatched:
        if clearance_min <= 0:
            return 0.0, "cleared"
        frac_remaining = max(0.0, 1.0 - t_since_dispatch_min / clearance_min)
        r = r_at_dispatch * frac_remaining
        if r < 0.01:
            return 0.0, "cleared"
        return r, "shrinking"
    else:
        r = min(r_max, BASE_RADIUS_KM + g_rate * t_elapsed_min)
        return r, "growing"

# ---- SIMULATION: COMPUTE CASCADE SNAPSHOTS -----------------------------------
sim_start = triage["start_dt"].min().floor("H")
sim_end   = triage["start_dt"].max().ceil("H")
step_td   = pd.Timedelta(minutes=STEP_MINUTES)

print(f"\n[SIM] Computing cascade snapshots ...")
print(f"      Window: {sim_start} to {sim_end}")

snapshot_rows = []
t = sim_start

while t <= sim_end:
    # Active incidents at this timestamp
    active_mask = (triage["start_dt"] <= t) & (triage["est_resolution_dt"] > t)
    active = triage[active_mask]

    for _, inc in active.iterrows():
        inc_id  = inc["id"]
        t_elapsed = (t - inc["start_dt"]).total_seconds() / 60.0

        crit   = float(inc.get("corridor_criticality", 0.55))
        ci     = float(inc.get("congestion_impact",    0.40))
        clr    = float(inc["sim_clearance_min"])

        r_max  = r_max_for(crit, ci)
        g_rate = growth_rate(crit, ci)

        # Check if dispatched (Phase 4 OPT)
        dispatch_t = dispatch_map.get(inc_id)
        if dispatch_t is not None and t >= dispatch_t:
            t_since_d = (t - dispatch_t).total_seconds() / 60.0
            # r_at_dispatch: radius when unit arrived
            t_at_d = max(0.0, (dispatch_t - inc["start_dt"]).total_seconds() / 60.0)
            r_at_d = min(r_max, BASE_RADIUS_KM + g_rate * t_at_d)
            r, phase = radius_at(t_elapsed, r_max, g_rate,
                                 True, t_since_d, r_at_d, clr)
        else:
            r, phase = radius_at(t_elapsed, r_max, g_rate,
                                 False, 0.0, 0.0, clr)

        intensity = r / r_max if r_max > 0 else 0.0

        snapshot_rows.append({
            "timestep":         t,
            "incident_id":      inc_id,
            "latitude":         inc["latitude"],
            "longitude":        inc["longitude"],
            "corridor":         inc["corridor"],
            "event_cause":      inc["event_cause"],
            "priority_score":   inc["priority_score"],
            "corridor_crit":    round(crit, 3),
            "congestion_impact":round(ci,   3),
            "radius_km":        round(r,    4),
            "r_max_km":         round(r_max,4),
            "intensity":        round(intensity, 4),
            "phase":            phase,
            "clearance_min":    round(clr, 1),
        })

    t += step_td

cascade = pd.DataFrame(snapshot_rows)
print(f"[SIM] Done. {len(cascade):,} snapshot rows across {cascade['timestep'].nunique():,} timesteps")

# ---- PRINT MODEL SUMMARY -----------------------------------------------------
print()
print("=" * 72)
print("CASCADE MODEL SUMMARY")
print("=" * 72)
print(f"  Unique incidents in cascade : {cascade['incident_id'].nunique():,}")
print()
print("  Phase distribution (all snapshots):")
print(cascade["phase"].value_counts().to_string())
print()
print("  Radius stats (growing phase, km):")
growing = cascade[cascade["phase"] == "growing"]["radius_km"]
print(f"    median : {growing.median():.3f} km")
print(f"    p75    : {growing.quantile(0.75):.3f} km")
print(f"    p95    : {growing.quantile(0.95):.3f} km")
print(f"    max    : {growing.max():.3f} km")
print()
print("  Radius stats (shrinking phase, km):")
shrink = cascade[cascade["phase"] == "shrinking"]["radius_km"]
if len(shrink) > 0:
    print(f"    median : {shrink.median():.3f} km")
    print(f"    p75    : {shrink.quantile(0.75):.3f} km")
else:
    print("    (no shrinking snapshots)")

# ---- STATIC VISUALIZATION: ONE SAMPLE DAY ------------------------------------
# Pick the day with most active incidents for a compelling still frame
print()
print("[PLOT] Generating static sample-day visualization ...")

cascade["date"] = cascade["timestep"].dt.date
daily_counts = cascade.groupby("date")["incident_id"].nunique()
sample_date  = daily_counts.idxmax()
print(f"  Sample day chosen : {sample_date}  ({daily_counts[sample_date]} unique incidents)")

# Pick a peak-hour snapshot from that day
day_snaps = cascade[cascade["date"] == sample_date]
peak_step  = day_snaps.groupby("timestep")["incident_id"].count().idxmax()
snap       = day_snaps[day_snaps["timestep"] == peak_step].copy()
print(f"  Peak timestep     : {peak_step}  ({len(snap)} active incidents)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor="#0d1117")
fig.suptitle(
    f"SAARTHI - Cascade Congestion Model\n"
    f"Sample: {sample_date} at {str(peak_step)[:16]} IST\n"
    f"ILLUSTRATIVE ONLY - radius is a model parameter, not a measured distance",
    color="white", fontsize=13, y=0.98
)

PHASE_COLORS = {
    "growing":  "#ff4444",
    "shrinking":"#ffaa00",
    "cleared":  "#44ff88",
}

# Bengaluru bounding box
LAT_MIN, LAT_MAX = 12.82, 13.15
LON_MIN, LON_MAX = 77.45, 77.80

for ax_idx, (ax, show_dispatched) in enumerate(zip(axes, [False, True])):
    ax.set_facecolor("#0d1117")
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("equal")
    ax.tick_params(colors="gray", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    if ax_idx == 0:
        ax.set_title("WITHOUT SAARTHI (all growing)", color="white", fontsize=11, pad=8)
        # All incidents shown as growing (no dispatch)
        plot_snap = snap.copy()
        plot_snap["phase"]     = "growing"
        plot_snap["radius_km"] = plot_snap.apply(
            lambda r: min(r_max_for(r["corridor_crit"], r["congestion_impact"]),
                          BASE_RADIUS_KM + growth_rate(r["corridor_crit"], r["congestion_impact"])
                          * max(0.0, (peak_step - triage.set_index("id")
                                      .loc[r["incident_id"], "start_dt"]
                                      if r["incident_id"] in triage.set_index("id").index
                                      else pd.Timedelta(0)).total_seconds() / 60.0)
                         ), axis=1
        )
    else:
        ax.set_title("WITH SAARTHI (dispatched = shrinking)", color="white", fontsize=11, pad=8)
        plot_snap = snap.copy()

    for _, row in plot_snap.iterrows():
        phase   = row["phase"]
        color   = PHASE_COLORS.get(phase, "#888888")
        r_deg   = row["radius_km"] / 111.0  # rough km-to-degrees conversion
        # Draw filled circle (congestion zone)
        circle = plt.Circle(
            (row["longitude"], row["latitude"]),
            max(r_deg, 0.002),
            color=color, alpha=0.25, linewidth=0
        )
        ax.add_patch(circle)
        # Draw incident dot
        ax.scatter(row["longitude"], row["latitude"],
                   c=color, s=30, zorder=5, alpha=0.9)

    # Legend
    for phase, col in PHASE_COLORS.items():
        ax.scatter([], [], c=col, s=40, label=phase.capitalize(), alpha=0.9)
    ax.legend(loc="lower right", fontsize=8, facecolor="#1a1a2e", labelcolor="white",
              edgecolor="#333")
    ax.set_xlabel("Longitude", color="gray", fontsize=8)
    ax.set_ylabel("Latitude",  color="gray", fontsize=8)

# Assumption footnote
fig.text(
    0.5, 0.01,
    "ASSUMPTIONS: growth_rate = corridor_crit x cong_impact x 0.015 km/min  |  "
    "R_max = corridor_crit x 2.5 km  |  shrink: linear to 0 over clearance_time  |  "
    "Radius is NOT a measured congestion distance. Visualization purpose only.",
    ha="center", va="bottom", color="#888888", fontsize=7, style="italic"
)

plt.tight_layout(rect=[0, 0.04, 1, 0.95])
plt.savefig(PLOT_OUT, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"[PLOT] Saved to: {PLOT_OUT}")

# ---- SAVE SNAPSHOTS ----------------------------------------------------------
# Drop the date helper column before saving
cascade_save = cascade.drop(columns=["date"])
cascade_save.to_parquet(CASCADE_OUT, index=False)
print(f"[SAVE] Cascade snapshots saved to: {CASCADE_OUT}")
print(f"       {len(cascade_save):,} rows  |  {cascade_save['timestep'].nunique():,} timesteps  |  "
      f"{cascade_save['incident_id'].nunique():,} unique incidents")

# ---- WHAT PHASE 6 GETS -------------------------------------------------------
print()
print("=" * 72)
print("PHASE 6 INTERFACE CONTRACT")
print("=" * 72)
print("  Columns available per timestep slice:")
for col in cascade_save.columns:
    print(f"    {col}")
print()
print("  Phase 6 draws each incident as a circle on the map:")
print("    - radius = radius_km (illustrative, not measured)")
print("    - color  = red (growing) / amber (shrinking) / green (cleared)")
print("    - opacity = intensity (0-1)")
print("    - Annotation: event_cause, corridor, priority_score, phase")
print("    - All circles labeled ILLUSTRATIVE in the UI footer")

print()
print("=" * 72)
print("Phase 5 COMPLETE. Cascade data ready for Phase 6.")
print("=" * 72)
