"""
SAARTHI - Phase 7: Digital-Twin Proof
======================================

GOAL: Honest comparison of SAARTHI vs naive dispatch on held-out incidents.

SPLIT HONESTY NOTE:
  Phase 2 used a random 80/20 row split (random_state=42, stratified on event flag).
  The split is NOT temporal — both train and test span the full date range.
  For Phase 7 we therefore replay ONLY the 20% held-out test-set incidents
  (incidents the RSF model never saw during training) from the chosen day.
  We do NOT replay the full day's incidents, because ~80% of those trained the model.

CHOSEN DAY: March 7, 2024
  Reason: highest test-set incident count of any single day (n=45).
  This gives the largest held-out window for a meaningful replay.
  The 45 incidents span morning through evening, including peak-hour.

STRATEGIES (IDENTICAL stream — only dispatch policy differs):
  BSL: nearest available unit -> newest incident (realistic status-quo)
  OPT: SAARTHI priority-weighted dispatch (priority_score / (travel_time+1))

METRIC: priority-weighted delay (score x wait-time in minutes)
  - The only metric we validated in Phase 4
  - We also directly compute: median minutes sooner OPT dispatches high-priority
    incidents vs BSL (raw arithmetic from the run, no assumptions)
  - NO conversion to vehicle-hours, rupees, or any physical unit

LEAKAGE CHECK:
  - Survival model trained only on the 80% training split (random_state=42)
  - Replayed incidents are exclusively from the 20% test split
  - Clearance durations used in simulation = actual data (not model output)
  - Triage scores use the RSF risk ranking on held-out incidents
"""

import pandas as pd
import numpy as np
import math
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# ---- CONFIG ------------------------------------------------------------------
CLEAN_PATH   = r"C:\Users\Rohan\Pictures\saarthi\saarthi_clean.parquet"
TRIAGE_PATH  = r"C:\Users\Rohan\Pictures\saarthi\triage_queue.parquet"
MODEL_PATH   = r"C:\Users\Rohan\Pictures\saarthi\rsf_model.pkl"
OUT_JSON     = r"C:\Users\Rohan\Pictures\saarthi\phase7_twin_results.json"

REPLAY_DATE      = "2024-03-07"
N_UNITS          = 8
STEP_MINUTES     = 15        # finer steps for a single-day replay
AVG_SPEED_KMPH   = 20.0
MEDIAN_CLEARANCE = 48.5
RANDOM_STATE     = 42
TEST_SIZE        = 0.20

UNIT_DEPOTS = {
    1: (12.9716, 77.5946), 2: (13.0358, 77.5970), 3: (12.9200, 77.6200),
    4: (12.9716, 77.5100), 5: (12.9900, 77.7200), 6: (13.0100, 77.5500),
    7: (12.9500, 77.5700), 8: (13.0600, 77.6800),
}

LOW_CONF_TYPES = {"tree_fall", "public_event", "road_conditions", "others"}
EVENT_CAUSE_BASE = {
    "accident":1.00,"water_logging":0.90,"tree_fall":0.85,"congestion":0.80,
    "procession":0.75,"vip_movement":0.75,"protest":0.70,"vehicle_breakdown":0.60,
    "construction":0.65,"public_event":0.55,"pot_holes":0.40,"road_conditions":0.45,
    "debris":0.50,"others":0.40,"unknown":0.40,"other_cause":0.40,
}
VEH_TYPE_MULT = {
    "bmtc_bus":1.00,"ksrtc_bus":1.00,"private_bus":0.95,"heavy_vehicle":0.90,
    "truck":0.88,"lcv":0.55,"private_car":0.45,"taxi":0.45,"auto":0.30,
    "others":0.50,"unknown":0.50,
}
CORRIDOR_TIER = {
    "mysore road":1.00,"tumkur road":1.00,"bellary road 1":1.00,"bellary road 2":0.95,
    "hosur road":0.95,"old madras road":0.90,"magadi road":0.85,"bannerghatta road":0.85,
    "bannerghata road":0.85,"orr north 1":0.90,"orr north 2":0.88,"orr east 1":0.88,
    "orr east 2":0.85,"orr west 1":0.85,"orr west 2":0.83,"orr south 1":0.83,
    "orr south 2":0.80,"west of chord road":0.75,"intermediate ring road":0.70,
    "sarjapur road":0.70,"non-corridor":0.35,"other_corridor":0.50,
}

print("=" * 72)
print("SAARTHI - Phase 7: Digital-Twin Proof")
print("=" * 72)

# ---- STEP 1: RECONSTRUCT TEST SET -------------------------------------------
print("\n[SPLIT] Reconstructing Phase 2 train/test split ...")
df = pd.read_parquet(CLEAN_PATH)
df = df.dropna(subset=["start_datetime"]).reset_index(drop=True)
df["start_dt"] = pd.to_datetime(df["start_datetime"], utc=False)

event_flag = (df["censored"] == 0).values.astype(bool)
idx = np.arange(len(df))
idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=event_flag)
df_te = df.iloc[idx_te].reset_index(drop=True)
df_tr = df.iloc[idx_tr].reset_index(drop=True)

train_ids = set(df_tr["id"].values)
test_ids  = set(df_te["id"].values)

print(f"  Training set   : {len(df_tr):,} incidents  (random_state={RANDOM_STATE})")
print(f"  Test set       : {len(df_te):,} incidents")
print(f"  Train date range: {df_tr['start_dt'].min().date()} to {df_tr['start_dt'].max().date()}")
print(f"  Test  date range: {df_te['start_dt'].min().date()} to {df_te['start_dt'].max().date()}")
print(f"  NOTE: Split is RANDOM (not temporal) — both sets span all dates.")

# ---- STEP 2: EXTRACT REPLAY DAY INCIDENTS (TEST SET ONLY) -------------------
print(f"\n[DAY] Chosen replay day: {REPLAY_DATE}")
day_mask = df_te["start_dt"].dt.date == pd.to_datetime(REPLAY_DATE).date()
replay_df = df_te[day_mask].copy().reset_index(drop=True)
print(f"  Test-set incidents on {REPLAY_DATE}: {len(replay_df)}")
print(f"  (These {len(replay_df)} incidents were NEVER seen by the RSF during training)")

# Sanity check — none of these IDs should be in train_ids
overlap = set(replay_df["id"].values) & train_ids
assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} replay incidents found in training set!"
print(f"  Leakage check PASSED: 0 of {len(replay_df)} replay incidents are in training set.")

# ---- STEP 3: LOAD MODEL AND SCORE REPLAY INCIDENTS --------------------------
print("\n[MODEL] Loading RSF bundle and scoring replay incidents ...")
bundle       = joblib.load(MODEL_PATH)
rsf          = bundle["rsf"]
preprocessor = bundle["preprocessor"]
feature_names= bundle["feature_names"]
all_features = bundle["all_features"]
top_causes   = bundle["top_causes"]
top_corridors= bundle["top_corridors"]

# Feature engineering (same as Phase 2)
replay_df["corridor_grouped"]    = replay_df["corridor"].apply(lambda x: x if x in top_corridors else "other_corridor")
replay_df["event_cause_grouped"] = replay_df["event_cause"].apply(lambda x: x if x in top_causes else "other_cause")
replay_df["is_planned"]          = (replay_df["event_type"] == "planned").astype(int)
replay_df["is_high_prio"]        = (replay_df["priority"] == "high").astype(int)
replay_df["hour_of_day_ist"]     = replay_df["hour_of_day_ist"].fillna(12).astype(int)

X_replay = pd.DataFrame(
    preprocessor.transform(replay_df[all_features]),
    columns=feature_names,
    index=replay_df.index,
)
risk_raw  = rsf.predict(X_replay)
r_min, r_max = risk_raw.min(), risk_raw.max()
risk_norm = (risk_raw - r_min) / (r_max - r_min + 1e-9)
replay_df["rsf_risk_norm"] = risk_norm

# Compute triage components
def get_ci(row):
    base = EVENT_CAUSE_BASE.get(row["event_cause"], 0.40)
    mult = VEH_TYPE_MULT.get(row["veh_type"], 0.50)
    bonus = 0.20 if row.get("requires_road_closure") else 0.0
    return min(1.0, base * mult + bonus)

def get_cc(row):
    return CORRIDOR_TIER.get(row["corridor"], 0.55)

def get_score(row):
    ci = get_ci(row)
    cc = get_cc(row)
    if row["event_cause"] in LOW_CONF_TYPES:
        pscale = 0.50 if str(row["priority"]).lower() == "high" else 0.25
        return ci * cc * pscale * 10
    return ci * row["rsf_risk_norm"] * cc * 10

replay_df["congestion_impact"]    = replay_df.apply(get_ci, axis=1)
replay_df["corridor_criticality"] = replay_df.apply(get_cc, axis=1)
replay_df["priority_score"]       = replay_df.apply(get_score, axis=1)

print(f"  Priority scores computed. Range: {replay_df['priority_score'].min():.2f} - {replay_df['priority_score'].max():.2f}")

# ---- STEP 4: BUILD SIMULATION WINDOWS ----------------------------------------
# Clearance duration for simulation
def sim_clearance(row):
    ctm = row.get("clearance_time_minutes")
    if pd.notna(ctm) and 0 < float(ctm) <= 1440:
        return float(ctm)
    return MEDIAN_CLEARANCE

replay_df["sim_clearance_min"] = replay_df.apply(sim_clearance, axis=1)

# Resolution time for active-window calculation
def est_resolution(row):
    for col in ["closed_datetime", "end_datetime", "resolved_datetime"]:
        val = row.get(col)
        if pd.notna(val):
            return pd.to_datetime(val)
    return row["start_dt"] + pd.Timedelta(minutes=float(row["sim_clearance_min"]))

replay_df["est_resolution_dt"] = replay_df.apply(est_resolution, axis=1)

# Validate lat/lon
replay_df["latitude"]  = pd.to_numeric(replay_df["latitude"],  errors="coerce")
replay_df["longitude"] = pd.to_numeric(replay_df["longitude"], errors="coerce")
replay_df = replay_df.dropna(subset=["latitude", "longitude"])
replay_df = replay_df[replay_df["latitude"].between(11.0, 14.0) & replay_df["longitude"].between(77.0, 78.0)]
replay_df = replay_df.reset_index(drop=True)

print(f"\n[REPLAY] {len(replay_df)} incidents with valid lat/lon for simulation.")
print(f"  Incident window: {replay_df['start_dt'].min().strftime('%H:%M')} to {replay_df['start_dt'].max().strftime('%H:%M')} IST")
print(f"  Event type breakdown:")
print("  " + replay_df["event_cause"].value_counts().to_string().replace("\n", "\n  "))

# ---- STEP 5: RUN DUAL SIMULATION ---------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def travel_time_min(lat1, lon1, lat2, lon2):
    return (haversine_km(lat1, lon1, lat2, lon2) / AVG_SPEED_KMPH) * 60.0

class Unit:
    def __init__(self, uid, lat, lon):
        self.uid = uid; self.lat = lat; self.lon = lon; self.busy_until = None
    def is_free(self, t): return self.busy_until is None or t >= self.busy_until
    def assign(self, dest_lat, dest_lon, eta_min, clearance_min, t):
        self.lat = dest_lat; self.lon = dest_lon
        self.busy_until = t + pd.Timedelta(minutes=eta_min + clearance_min)

units_opt = {uid: Unit(uid, lat, lon) for uid, (lat, lon) in UNIT_DEPOTS.items()}
units_bsl = {uid: Unit(uid, lat, lon) for uid, (lat, lon) in UNIT_DEPOTS.items()}

sim_start = replay_df["start_dt"].min().floor("30min")
sim_end   = (replay_df["est_resolution_dt"].max() + pd.Timedelta(hours=1)).ceil("H")
step_td   = pd.Timedelta(minutes=STEP_MINUTES)

opt_total_pw = 0.0; bsl_total_pw = 0.0
assigned_opt = {}  # incident_id -> dispatch_timestep
assigned_bsl = {}  # incident_id -> dispatch_timestep
t = sim_start

print(f"\n[SIM] Replaying {REPLAY_DATE} ...")
print(f"  Step size: {STEP_MINUTES} min  |  {N_UNITS} units  |  Speed: {AVG_SPEED_KMPH} km/h Haversine")

while t <= sim_end:
    active_mask = (replay_df["start_dt"] <= t) & (replay_df["est_resolution_dt"] > t)
    active = replay_df[active_mask].copy()

    if len(active) > 0:
        # -- OPT: priority x proximity
        free_opt = [u for u in units_opt.values() if u.is_free(t)]
        pending  = active[~active["id"].isin(assigned_opt)]
        for unit in free_opt:
            if pending.empty: break
            best_c, best_inc, best_tt = -1.0, None, None
            for _, inc in pending.iterrows():
                tt = travel_time_min(unit.lat, unit.lon, inc["latitude"], inc["longitude"])
                c  = inc["priority_score"] / (tt + 1.0)
                if c > best_c: best_c = c; best_inc = inc; best_tt = tt
            unit.assign(best_inc["latitude"], best_inc["longitude"], best_tt, best_inc["sim_clearance_min"], t)
            assigned_opt[best_inc["id"]] = {"dispatch_t": t, "tt": best_tt, "score": best_inc["priority_score"]}
            pending = pending[pending["id"] != best_inc["id"]]

        # -- BSL: nearest to newest
        free_bsl = [u for u in units_bsl.values() if u.is_free(t)]
        pending_b = active[~active["id"].isin(assigned_bsl)].sort_values("start_dt", ascending=False)
        for unit in free_bsl:
            if pending_b.empty: break
            nearest_inc, nearest_tt = None, float("inf")
            for _, inc in pending_b.iterrows():
                tt = travel_time_min(unit.lat, unit.lon, inc["latitude"], inc["longitude"])
                if tt < nearest_tt: nearest_tt = tt; nearest_inc = inc
            unit.assign(nearest_inc["latitude"], nearest_inc["longitude"], nearest_tt, nearest_inc["sim_clearance_min"], t)
            assigned_bsl[nearest_inc["id"]] = {"dispatch_t": t, "tt": nearest_tt, "score": nearest_inc["priority_score"]}
            pending_b = pending_b[pending_b["id"] != nearest_inc["id"]]

        # Accumulate priority-weighted wait
        unhandled_opt = active[~active["id"].isin(assigned_opt)]
        unhandled_bsl = active[~active["id"].isin(assigned_bsl)]
        opt_total_pw += (unhandled_opt["priority_score"] * STEP_MINUTES).sum()
        bsl_total_pw += (unhandled_bsl["priority_score"] * STEP_MINUTES).sum()

    t += step_td

# ---- STEP 6: PER-INCIDENT WAIT TIME -----------------------------------------
# Wait time = minutes from incident start to dispatch
def wait_minutes(inc_id, assignment_dict, replay_df):
    if inc_id not in assignment_dict:
        return None  # never dispatched
    start_t = replay_df.loc[replay_df["id"] == inc_id, "start_dt"].iloc[0]
    disp_t  = assignment_dict[inc_id]["dispatch_t"]
    return (disp_t - start_t).total_seconds() / 60.0

replay_df["wait_opt"] = replay_df["id"].apply(lambda x: wait_minutes(x, assigned_opt, replay_df))
replay_df["wait_bsl"] = replay_df["id"].apply(lambda x: wait_minutes(x, assigned_bsl, replay_df))

# ---- STEP 7: HEADLINE METRICS -----------------------------------------------
pw_saved = bsl_total_pw - opt_total_pw
pct_impr = (pw_saved / bsl_total_pw * 100) if bsl_total_pw > 0 else 0.0

# High-priority incidents dispatched by both — minutes sooner
both_dispatched = replay_df.dropna(subset=["wait_opt", "wait_bsl"])
high_prio_both  = both_dispatched[both_dispatched["priority_score"] >= 1.5]
if len(high_prio_both) > 0:
    high_prio_both = high_prio_both.copy()
    high_prio_both["minutes_sooner"] = high_prio_both["wait_bsl"] - high_prio_both["wait_opt"]
    med_sooner = high_prio_both["minutes_sooner"].median()
    mean_sooner = high_prio_both["minutes_sooner"].mean()
else:
    med_sooner = mean_sooner = None

print("\n" + "=" * 72)
print("PHASE 7 HEADLINE RESULTS")
print("=" * 72)
print()
print(f"  Day replayed            : {REPLAY_DATE}")
print(f"  Incidents replayed      : {len(replay_df)} (test-set only, never seen by RSF)")
print(f"  Incidents in train set  : {len(df_tr)} (not replayed)")
print(f"  Units                   : {N_UNITS} (same depots, same clearance durations)")
print(f"  Only dispatch policy differs between OPT and BSL.")
print()
print(f"  BSL accumulated PW-delay  : {bsl_total_pw:,.1f} score-min")
print(f"  OPT accumulated PW-delay  : {opt_total_pw:,.1f} score-min")
print(f"  Reduction (OPT vs BSL)    : {pw_saved:,.1f} score-min  ({pct_impr:.1f}%)")
print()
print(f"  Incidents dispatched by OPT : {len(assigned_opt)}")
print(f"  Incidents dispatched by BSL : {len(assigned_bsl)}")
if med_sooner is not None:
    print()
    print(f"  HIGH-PRIORITY incidents dispatched by both (score >= 1.5): n={len(high_prio_both)}")
    print(f"    Median minutes sooner (OPT vs BSL)  : {med_sooner:.1f} min")
    print(f"    Mean   minutes sooner (OPT vs BSL)  : {mean_sooner:.1f} min")
    print(f"    (Directly computed: dispatch_t(BSL) - dispatch_t(OPT) per incident)")
    print(f"    Positive = OPT dispatched first. Negative = BSL dispatched first.")
    print()
    neg = (high_prio_both["minutes_sooner"] < 0).sum()
    pos = (high_prio_both["minutes_sooner"] > 0).sum()
    zer = (high_prio_both["minutes_sooner"] == 0).sum()
    print(f"    OPT earlier: {pos}, BSL earlier: {neg}, Same time: {zer}")

# ---- STEP 8: DEBUG OUTPUT ---------------------------------------------------
print()
print("=" * 72)
print("DEBUG - FULL PER-INCIDENT WAIT DISTRIBUTION")
print("=" * 72)

print("\n  BSL wait-time distribution (minutes to dispatch):")
bsl_waits = replay_df["wait_bsl"].dropna()
if len(bsl_waits) > 0:
    print(f"    n dispatched : {len(bsl_waits)}")
    print(f"    median       : {bsl_waits.median():.1f} min")
    print(f"    mean         : {bsl_waits.mean():.1f} min")
    print(f"    p25          : {bsl_waits.quantile(0.25):.1f} min")
    print(f"    p75          : {bsl_waits.quantile(0.75):.1f} min")
    print(f"    p90          : {bsl_waits.quantile(0.90):.1f} min")
    print(f"    max          : {bsl_waits.max():.1f} min")
    print(f"    never dispatched: {replay_df['wait_bsl'].isna().sum()}")

print("\n  OPT wait-time distribution (minutes to dispatch):")
opt_waits = replay_df["wait_opt"].dropna()
if len(opt_waits) > 0:
    print(f"    n dispatched : {len(opt_waits)}")
    print(f"    median       : {opt_waits.median():.1f} min")
    print(f"    mean         : {opt_waits.mean():.1f} min")
    print(f"    p25          : {opt_waits.quantile(0.25):.1f} min")
    print(f"    p75          : {opt_waits.quantile(0.75):.1f} min")
    print(f"    p90          : {opt_waits.quantile(0.90):.1f} min")
    print(f"    max          : {opt_waits.max():.1f} min")
    print(f"    never dispatched: {replay_df['wait_opt'].isna().sum()}")

print("\n  Priority score distribution of replayed incidents:")
ps = replay_df["priority_score"]
print(f"    median : {ps.median():.2f}  |  p75: {ps.quantile(0.75):.2f}  |  max: {ps.max():.2f}")
print(f"    high-priority (score>=1.5): {(ps>=1.5).sum()}  |  lower: {(ps<1.5).sum()}")

print("\n  Clearance duration used in sim (actual or 48.5 fallback):")
cl = replay_df["sim_clearance_min"]
print(f"    Using actual clearance: {(replay_df['clearance_time_minutes'].notna() & (replay_df['clearance_time_minutes']>0)).sum()}")
print(f"    Using 48.5 fallback   : {(replay_df['clearance_time_minutes'].isna() | (replay_df['clearance_time_minutes']<=0)).sum()}")
print(f"    Median clearance used : {cl.median():.1f} min")

print()
print("=" * 72)
print("SELF-AUDIT")
print("=" * 72)
print(f"  Replay incidents are exclusively from 20% test split (random_state={RANDOM_STATE})")
print(f"  Leakage check: 0 replay incidents overlap with training set (assert passed)")
print(f"  Split is random (not temporal): honestly disclosed above")
print(f"  Clearance durations: actual data where available, {MEDIAN_CLEARANCE} min fallback")
print(f"  Travel time: Haversine / {AVG_SPEED_KMPH} km/h — no routing API")
print(f"  PW-delay metric: same formula used in Phase 4 validation (consistent)")
print(f"  'Minutes sooner' = dispatch_t(BSL) - dispatch_t(OPT), computed directly")
print(f"  No conversion to vehicle-hours, rupees, or physical units (per audit)")

# ---- SAVE RESULTS ------------------------------------------------------------
results = {
    "replay_date":             REPLAY_DATE,
    "n_incidents_replayed":    len(replay_df),
    "n_units":                 N_UNITS,
    "step_minutes":            STEP_MINUTES,
    "bsl_total_pw_wait":       round(bsl_total_pw, 1),
    "opt_total_pw_wait":       round(opt_total_pw, 1),
    "pw_saved":                round(pw_saved, 1),
    "pct_improvement":         round(pct_impr, 2),
    "n_dispatched_opt":        len(assigned_opt),
    "n_dispatched_bsl":        len(assigned_bsl),
    "high_prio_n":             int(len(high_prio_both)) if med_sooner is not None else 0,
    "median_minutes_sooner":   round(float(med_sooner), 1) if med_sooner is not None else None,
    "mean_minutes_sooner":     round(float(mean_sooner), 1) if mean_sooner is not None else None,
    "opt_wait_median_min":     round(opt_waits.median(), 1) if len(opt_waits) > 0 else None,
    "bsl_wait_median_min":     round(bsl_waits.median(), 1) if len(bsl_waits) > 0 else None,
    "leakage_check":           "PASSED - 0 replay incidents in training set",
    "split_note":              "Random 80/20 row split (not temporal) - honestly disclosed",
    "metric_note":             "PW-delay only. No vehicle-hours or rupee conversion.",
}
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[SAVE] Results saved to: {OUT_JSON}")
print("\n" + "=" * 72)
print("Phase 7 COMPLETE.")
print("=" * 72)
