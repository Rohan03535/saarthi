"""
SAARTHI — Phase 1: Data Foundation
Load, profile, and build the core clearance_time_minutes label.
"""

import pandas as pd
import numpy as np
import pytz
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\Rohan\Downloads\Astram event data_anonymized - Astram event data_anonymizedb40ac87 (1).csv"
IST = pytz.timezone("Asia/Kolkata")
OUT_PARQUET = r"C:\Users\Rohan\Pictures\saarthi\saarthi_clean.parquet"

# ── 1. LOAD ────────────────────────────────────────────────────────────────────
print("=" * 70)
print("SAARTHI — Phase 1: Data Foundation")
print("=" * 70)

raw = pd.read_csv(CSV_PATH, low_memory=False)
print(f"\n[LOAD] Raw shape: {raw.shape[0]:,} rows × {raw.shape[1]} columns")
print(f"[LOAD] Columns: {list(raw.columns)}\n")

# ── 2. PARSE + CONVERT TIMESTAMPS TO IST ──────────────────────────────────────
ts_cols = ["start_datetime", "end_datetime", "closed_datetime",
           "resolved_datetime", "created_date", "modified_datetime"]

for col in ts_cols:
    if col in raw.columns:
        raw[col] = pd.to_datetime(raw[col], utc=True, errors="coerce")
        raw[col] = raw[col].dt.tz_convert(IST)

print("[TIMESTAMPS] All timestamp columns converted to IST (Asia/Kolkata).")

# ── 3. BUILD CLEARANCE TIME ────────────────────────────────────────────────────
# Resolution hierarchy: end_datetime → closed_datetime → resolved_datetime
# We pick whichever comes first (some records have multiples).

def pick_resolution_time(row):
    """Return the earliest non-null resolution timestamp."""
    candidates = []
    for col in ["end_datetime", "closed_datetime", "resolved_datetime"]:
        val = row.get(col)
        if pd.notna(val):
            candidates.append(val)
    return min(candidates) if candidates else pd.NaT

raw["resolution_datetime"] = raw.apply(pick_resolution_time, axis=1)
raw["resolution_source"] = raw.apply(
    lambda r: (
        "end_datetime"      if pd.notna(r.get("end_datetime"))
        else "closed_datetime"  if pd.notna(r.get("closed_datetime"))
        else "resolved_datetime" if pd.notna(r.get("resolved_datetime"))
        else "none"
    ),
    axis=1
)

# Clearance time in minutes
raw["clearance_time_minutes"] = (
    (raw["resolution_datetime"] - raw["start_datetime"])
    .dt.total_seconds() / 60
)

# Sanity: negative or zero clearance times are invalid — flag and nullify
neg_mask = raw["clearance_time_minutes"] <= 0
print(f"[QA] Negative/zero clearance times (will be treated as censored): {neg_mask.sum()}")
raw.loc[neg_mask, "clearance_time_minutes"] = np.nan
raw.loc[neg_mask, "resolution_datetime"] = pd.NaT
raw.loc[neg_mask, "resolution_source"] = "none"

# ── 4. CENSORING FLAG ─────────────────────────────────────────────────────────
# censored = 1  →  incident still active, no valid resolution time (survival model uses this)
# censored = 0  →  observed full clearance
raw["censored"] = raw["clearance_time_minutes"].isna().astype(int)

# ── 5. EXTRACT IST TIME FEATURES ──────────────────────────────────────────────
raw["hour_of_day_ist"]  = raw["start_datetime"].dt.hour
raw["day_of_week_ist"]  = raw["start_datetime"].dt.day_name()
raw["date_ist"]         = raw["start_datetime"].dt.date

# ── 6. CLEAN KEY CATEGORICAL COLUMNS ──────────────────────────────────────────
cat_fill = {
    "event_cause":   "unknown",
    "event_type":    "unplanned",
    "priority":      "Low",
    "veh_type":      "unknown",
    "corridor":      "Non-corridor",
    "zone":          "unknown",
    "junction":      "unknown",
    "police_station":"unknown",
}
for col, default in cat_fill.items():
    if col in raw.columns:
        raw[col] = raw[col].fillna(default).str.strip().str.lower()

# requires_road_closure → bool
if "requires_road_closure" in raw.columns:
    raw["requires_road_closure"] = (
        raw["requires_road_closure"]
        .astype(str).str.strip().str.upper()
        .map({"TRUE": 1, "FALSE": 0, "YES": 1, "NO": 0, "1": 1, "0": 0})
        .fillna(0).astype(int)
    )

# status normalise
if "status" in raw.columns:
    raw["status"] = raw["status"].fillna("unknown").str.strip().str.lower()

# ── 7. PROFILE ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DATASET PROFILE")
print("=" * 70)
print(f"Total rows            : {len(raw):,}")
print(f"Total columns         : {raw.shape[1]}")

# Clearance time breakdown
obs   = (raw["censored"] == 0).sum()
cens  = (raw["censored"] == 1).sum()
print(f"\nClearance time label")
print(f"  Observed (full clearance)  : {obs:,}  ({obs/len(raw)*100:.1f}%)")
print(f"  Censored (still active)    : {cens:,}  ({cens/len(raw)*100:.1f}%)")

print(f"\nResolution source breakdown")
print(raw["resolution_source"].value_counts().to_string())

print(f"\nClearance time stats (observed only, minutes)")
ct = raw.loc[raw["censored"] == 0, "clearance_time_minutes"]
print(f"  Count  : {ct.count():,}")
print(f"  Mean   : {ct.mean():.1f} min")
print(f"  Median : {ct.median():.1f} min")
print(f"  Std    : {ct.std():.1f} min")
print(f"  p5     : {ct.quantile(0.05):.1f} min")
print(f"  p95    : {ct.quantile(0.95):.1f} min")
print(f"  Max    : {ct.max():.1f} min")

print(f"\nNull counts (key columns)")
key_cols = [
    "start_datetime","end_datetime","closed_datetime","resolved_datetime",
    "event_cause","event_type","priority","requires_road_closure",
    "veh_type","corridor","zone","junction","latitude","longitude","status"
]
for c in key_cols:
    if c in raw.columns:
        n = raw[c].isna().sum()
        print(f"  {c:<30}: {n:>5} null  ({n/len(raw)*100:.1f}%)")

print(f"\nValue distributions (key categoricals)")
for col in ["event_cause","event_type","priority","status","corridor","veh_type"]:
    if col in raw.columns:
        print(f"\n  {col}")
        vc = raw[col].value_counts().head(15)
        for v, cnt in vc.items():
            print(f"    {str(v):<35} {cnt:>5}  ({cnt/len(raw)*100:.1f}%)")

print(f"\nDate range (IST)")
valid_dates = raw["date_ist"].dropna()
print(f"  Earliest start : {valid_dates.min()}")
print(f"  Latest start   : {valid_dates.max()}")

print(f"\nStatus breakdown")
print(raw["status"].value_counts().to_string())

# ── 8. SAVE CLEAN DATAFRAME ───────────────────────────────────────────────────
# Drop columns we won't use in modelling to keep parquet lean
drop_cols = [
    "map_file", "description", "cargo_material", "reason_breakdown",
    "age_of_truck", "route_path", "comment", "meta_data", "kgid",
    "resolved_at_address", "citizen_accident_id", "gba_identifier",
    "veh_no", "address", "end_address", "resolved_at_address",
    "resolved_at_latitude", "resolved_at_longitude",
    "client_id", "created_by_id", "last_modified_by_id",
    "assigned_to_police_id", "closed_by_id", "resolved_by_id",
]
drop_cols = [c for c in drop_cols if c in raw.columns]
clean = raw.drop(columns=drop_cols)

clean.to_parquet(OUT_PARQUET, index=False)
print(f"\n[SAVE] Clean dataframe saved to: {OUT_PARQUET}")
print(f"[SAVE] Shape: {clean.shape[0]:,} rows × {clean.shape[1]} columns")

print("\n" + "=" * 70)
print("Phase 1 COMPLETE — awaiting approval to proceed to Phase 2.")
print("=" * 70)
