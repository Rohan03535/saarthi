"""
SAARTHI — Phase 3: Incident Priority / Triage Score

TWO-PATH FORMULA (transparent, all weights printed):

  [MDL] model-driven path (event types with Phase 2 C-index >= 0.55):
        score = congestion_impact × rsf_risk_norm × corridor_criticality × 10
        Max possible: 10.0

  [FBK] fallback path (C-index < 0.55 — duration model not trusted):
        score = congestion_impact × corridor_criticality × priority_scale × 10
        priority_scale: High=0.50, Low=0.25
        Max possible: 5.0  ← hard cap; duration uncertainty costs half the ceiling
        Duration term is REMOVED for fallback — the score reflects only real,
        observable factors (what's blocking, where, operator-stated urgency).

Rationale for two-path design:
  Giving fallback incidents a fixed duration value inflates their scores and
  defeats the purpose of survival modelling. A tree fall on Bellary Road 1
  should rank high because of road closure + tier-1 corridor, NOT because we
  assigned it an arbitrary 0.70 duration risk. With this design, a [FBK]
  incident beats a [MDL] incident only if its congestion footprint and corridor
  genuinely warrant it — not because the model is uncertain about its duration.

Scale: 0–10. Higher = dispatch first.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
PARQUET_PATH = r"C:\Users\Rohan\Pictures\saarthi\saarthi_clean.parquet"
MODEL_PATH   = r"C:\Users\Rohan\Pictures\saarthi\rsf_model.pkl"
SCORE_OUT    = r"C:\Users\Rohan\Pictures\saarthi\triage_queue.parquet"

# ── LOW-CONFIDENCE TYPES (Phase 2 C-index < 0.55 on test set) ────────────────
LOW_CONF_TYPES = {"tree_fall", "public_event", "road_conditions", "others"}
# These get duration_risk = 0.5 (neutral) and stated priority drives the modifier.
# Reason printed on screen for full transparency.

# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT TABLES — printed verbatim so judges can audit
# ══════════════════════════════════════════════════════════════════════════════

EVENT_CAUSE_BASE = {
    # Higher = incident causes more congestion per unit time
    "accident":         1.00,  # potential secondary crashes, full-stop blockage
    "water_logging":    0.90,  # spreads across lanes, weather-driven
    "tree_fall":        0.85,  # full lane block, debris hazard
    "congestion":       0.80,  # systemic, spreads upstream
    "procession":       0.75,  # planned but wide footprint
    "vip_movement":     0.75,  # corridor-wide closure
    "protest":          0.70,  # variable, can escalate
    "vehicle_breakdown":0.60,  # point block, high volume
    "construction":     0.65,  # planned, partial block
    "public_event":     0.55,  # managed but high footfall
    "pot_holes":        0.40,  # reduces speed, rarely full stop
    "road_conditions":  0.45,  # catch-all slow-down
    "debris":           0.50,
    "others":           0.40,
    "unknown":          0.40,
    "other_cause":      0.40,
}

VEH_TYPE_MULT = {
    # Physical footprint + passenger/cargo disruption
    "bmtc_bus":     1.00,   # max: large, high-occupancy, fixed route disruption
    "ksrtc_bus":    1.00,
    "private_bus":  0.95,
    "heavy_vehicle":0.90,   # large, slow to clear
    "truck":        0.88,
    "lcv":          0.55,
    "private_car":  0.45,
    "taxi":         0.45,
    "auto":         0.30,
    "others":       0.50,
    "unknown":      0.50,   # no veh_type logged
}

CORRIDOR_TIER = {
    # Based on Bengaluru arterial hierarchy and incident volume in dataset
    # Tier 1 — National-Highway-grade arterials
    "mysore road":       1.00,
    "tumkur road":       1.00,
    "bellary road 1":    1.00,
    "bellary road 2":    0.95,
    "hosur road":        0.95,
    "old madras road":   0.90,
    "magadi road":       0.85,
    "bannerghatta road": 0.85,
    "bannerghata road":  0.85,
    # Tier 2 — Outer Ring Road segments (high-volume, connects zones)
    "orr north 1":       0.90,
    "orr north 2":       0.88,
    "orr east 1":        0.88,
    "orr east 2":        0.85,
    "orr west 1":        0.85,
    "orr west 2":        0.83,
    "orr south 1":       0.83,
    "orr south 2":       0.80,
    # Tier 3 — Secondary corridors
    "west of chord road":0.75,
    "intermediate ring road": 0.70,
    "sarjapur road":     0.70,
    # Non-corridor — off-arterial
    "non-corridor":      0.35,
    "other_corridor":    0.50,
}
CORRIDOR_DEFAULT = 0.55  # fallback for corridors not in the table

ROAD_CLOSURE_BONUS = 0.20   # flat additive to congestion_impact if road is closed

# [FBK] priority scale — replaces the entire duration term for fallback incidents.
# Caps fallback at 50% of the model-path ceiling (max 5.0 vs max 10.0).
FBK_PRIORITY_SCALE = {
    "high": 0.50,
    "low":  0.25,
}

print("=" * 70)
print("SAARTHI — Phase 3: Incident Triage Score")
print("=" * 70)

# ── PRINT WEIGHT TABLES ───────────────────────────────────────────────────────
print("\n[WEIGHTS] event_cause base impact scores:")
for k, v in sorted(EVENT_CAUSE_BASE.items(), key=lambda x: -x[1]):
    print(f"  {k:<25} {v:.2f}")

print("\n[WEIGHTS] vehicle type multipliers:")
for k, v in sorted(VEH_TYPE_MULT.items(), key=lambda x: -x[1]):
    print(f"  {k:<25} {v:.2f}")

print("\n[WEIGHTS] corridor criticality tiers:")
for k, v in sorted(CORRIDOR_TIER.items(), key=lambda x: -x[1]):
    print(f"  {k:<30} {v:.2f}")

print(f"\n[WEIGHTS] Road closure bonus      : +{ROAD_CLOSURE_BONUS:.2f} to congestion_impact")
print(f"[WEIGHTS] Low-confidence types    : {sorted(LOW_CONF_TYPES)}")
print(f"\n[WEIGHTS] TWO-PATH FORMULA:")
print(f"  [MDL] score = congestion_impact x rsf_risk_norm x corridor_criticality x 10  (max 10.0)")
print(f"  [FBK] score = congestion_impact x corridor_criticality x priority_scale x 10  (max 5.0)")
print(f"  [FBK] priority_scale : High={FBK_PRIORITY_SCALE['high']}, Low={FBK_PRIORITY_SCALE['low']}")
print(f"  Rationale: duration uncertainty costs the full ceiling; fallback incidents rank on"
      f" observable footprint only, not a synthetic duration estimate.")

# ── LOAD DATA + MODEL ─────────────────────────────────────────────────────────
df    = pd.read_parquet(PARQUET_PATH)
bundle = joblib.load(MODEL_PATH)
rsf          = bundle["rsf"]
preprocessor = bundle["preprocessor"]
feature_names= bundle["feature_names"]
all_features = bundle["all_features"]
top_causes   = bundle["top_causes"]
top_corridors= bundle["top_corridors"]

print(f"\n[LOAD] {len(df):,} incidents loaded.")

# ── COMPONENT FUNCTIONS ───────────────────────────────────────────────────────

def congestion_impact(cause, veh, road_closure):
    base  = EVENT_CAUSE_BASE.get(cause, 0.40)
    mult  = VEH_TYPE_MULT.get(veh, 0.50)
    bonus = ROAD_CLOSURE_BONUS if road_closure else 0.0
    return min(1.0, base * mult + bonus)

def corridor_criticality(corridor):
    return CORRIDOR_TIER.get(corridor, CORRIDOR_DEFAULT)

def build_features_for_rsf(row_df):
    """Encode a slice of the dataframe for RSF prediction."""
    row_df = row_df.copy()

    row_df["corridor_grouped"] = row_df["corridor"].apply(
        lambda x: x if x in top_corridors else "other_corridor"
    )
    row_df["event_cause_grouped"] = row_df["event_cause"].apply(
        lambda x: x if x in top_causes else "other_cause"
    )
    row_df["is_planned"]   = (row_df["event_type"] == "planned").astype(int)
    row_df["is_high_prio"] = (row_df["priority"]   == "high").astype(int)
    row_df["hour_of_day_ist"] = row_df["hour_of_day_ist"].fillna(12).astype(int)

    X = pd.DataFrame(
        preprocessor.transform(row_df[all_features]),
        columns=feature_names,
        index=row_df.index,
    )
    return X

def plain_english_reason(cause, veh, corridor, road_closure,
                          ci_score, ci_raw, dur_uncertain, stated_prio):
    parts = []

    # What kind of incident
    parts.append(f"{cause.replace('_',' ').title()}")
    if veh not in ("unknown", "others"):
        parts.append(f"({veh.replace('_',' ')})")

    # Where
    if corridor != "non-corridor":
        parts.append(f"on {corridor.title()}")
    else:
        parts.append("off-corridor")

    if road_closure:
        parts.append("— ROAD CLOSED")

    # Scoring logic
    parts.append(f"| congestion={ci_score:.2f}")

    if dur_uncertain:
        parts.append(f"| duration UNCERTAIN (model C-index low for this type)")
        parts.append(f"| triaged by stated priority ({stated_prio.upper()})")
    else:
        parts.append(f"| duration-risk={ci_raw:.2f}")

    return " ".join(parts)

# ── SCORE ALL INCIDENTS ───────────────────────────────────────────────────────
# Operate on the full dataset — in Phase 6 (live ops) this filters to status==active.
# Here we score all statuses so judges can see the full distribution.
# Mark which are "live" (active).

df["is_active"] = df["status"] == "active"

print(f"\n[DEBUG] Status breakdown:")
print(df["status"].value_counts().to_string())
print(f"\n[DEBUG] Active incidents (live queue): {df['is_active'].sum():,}")

# Build RSF risk scores for all rows
print("\n[RSF] Computing risk scores for all incidents ...")
# Drop rows with null start_datetime (same as Phase 2)
score_df = df.dropna(subset=["start_datetime"]).copy()
X_all = build_features_for_rsf(score_df)
risk_scores_raw = rsf.predict(X_all)   # cumulative hazard — higher = longer clearance

# Normalise risk scores to [0, 1] across the whole dataset
r_min, r_max = risk_scores_raw.min(), risk_scores_raw.max()
risk_scores_norm = (risk_scores_raw - r_min) / (r_max - r_min + 1e-9)
score_df["rsf_risk_raw"]  = risk_scores_raw
score_df["rsf_risk_norm"] = risk_scores_norm
print(f"[RSF] Risk scores computed for {len(score_df):,} incidents.")

# ── COMPUTE TRIAGE COMPONENTS ─────────────────────────────────────────────────
records = []
for _, row in score_df.iterrows():
    cause      = row["event_cause"]
    veh        = row["veh_type"]
    corridor   = row["corridor"]
    road_cl    = bool(row["requires_road_closure"])
    stated_p   = row["priority"]
    rsf_norm   = row["rsf_risk_norm"]
    is_active  = row["is_active"]

    # Component 1: congestion impact
    ci = congestion_impact(cause, veh, road_cl)

    # Component 2: corridor criticality
    cc = corridor_criticality(corridor)

    # Two-path scoring
    dur_uncertain = cause in LOW_CONF_TYPES
    if dur_uncertain:
        # [FBK]: duration term removed — score from observable factors only
        pscale    = FBK_PRIORITY_SCALE.get(stated_p, 0.25)
        score     = ci * cc * pscale * 10          # max = 1×1×0.50×10 = 5.0
        dur_risk  = pscale                          # store for display
        dur_label = "fallback"
    else:
        # [MDL]: full product formula with RSF risk score
        score     = ci * rsf_norm * cc * 10        # max = 1×1×1×10 = 10.0
        dur_risk  = rsf_norm
        dur_label = "model"

    reason = plain_english_reason(cause, veh, corridor, road_cl,
                                  ci, rsf_norm, dur_uncertain, stated_p)

    records.append({
        "id":                    row["id"],
        "status":                row["status"],
        "is_active":             is_active,
        "event_cause":           cause,
        "veh_type":              veh,
        "corridor":              corridor,
        "requires_road_closure": road_cl,
        "priority_stated":       stated_p,
        "hour_of_day_ist":       row["hour_of_day_ist"],
        "latitude":              row["latitude"],
        "longitude":             row["longitude"],
        "start_datetime":        row["start_datetime"],
        "congestion_impact":     round(ci,  3),
        "corridor_criticality":  round(cc,  3),
        "duration_risk":         round(dur_risk, 3),
        "duration_source":       dur_label,
        "priority_score":        round(score, 2),
        "plain_reason":          reason,
    })

scored = pd.DataFrame(records).sort_values("priority_score", ascending=False)
scored = scored.reset_index(drop=True)
scored.index += 1  # 1-based rank

# ── PRINT PRIORITY QUEUE (active incidents) ───────────────────────────────────
active_q = scored[scored["is_active"]].copy()
active_q = active_q.reset_index(drop=True)
active_q.index += 1

print(f"\n{'='*70}")
print(f"LIVE PRIORITY QUEUE — Active Incidents Only (n={len(active_q):,})")
print(f"{'='*70}")
print(f"  Rank | Score | C.Imp | D.Risk | Corr | Src  | Incident")
print(f"  {'-'*80}")
for rank, row in active_q.head(30).iterrows():
    src_tag = "[MDL]" if row["duration_source"] == "model" else "[FBK]"
    print(
        f"  {rank:>4} | {row['priority_score']:>5.2f} | "
        f"{row['congestion_impact']:>5.2f} | {row['duration_risk']:>6.2f} | "
        f"{row['corridor_criticality']:>4.2f} | {src_tag} | "
        f"{row['plain_reason'][:70]}"
    )

if len(active_q) > 30:
    print(f"  ... ({len(active_q)-30} more active incidents not shown)")

# ── SCORE DISTRIBUTION DEBUG ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print("DEBUG — SCORE DISTRIBUTION ACROSS ALL INCIDENTS")
print(f"{'='*70}")

print(f"\nAll incidents (n={len(scored):,}):")
print(scored["priority_score"].describe().round(3).to_string())

print(f"\nActive incidents (n={len(active_q):,}):")
print(active_q["priority_score"].describe().round(3).to_string())

print(f"\nDuration source breakdown (all incidents):")
print(scored["duration_source"].value_counts().to_string())

print(f"\nLow-confidence fallback incidents breakdown:")
fb = scored[scored["duration_source"] == "fallback"]
print(fb["event_cause"].value_counts().to_string())

print(f"\nTop-5 corridors by mean priority score (all incidents):")
print(
    scored.groupby("corridor")["priority_score"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
    .head(5)
    .round(3)
    .to_string()
)

print(f"\nHigh-road-closure incidents: {scored['requires_road_closure'].sum():,}")
print(f"  Mean score with closure    : "
      f"{scored[scored['requires_road_closure']]['priority_score'].mean():.2f}")
print(f"  Mean score without closure : "
      f"{scored[~scored['requires_road_closure']]['priority_score'].mean():.2f}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
scored.to_parquet(SCORE_OUT, index=True)
print(f"\n[SAVE] Scored queue saved to: {SCORE_OUT}")
print(f"[SAVE] {len(scored):,} incidents scored, {len(active_q):,} currently active.")

print(f"\n{'='*70}")
print("Phase 3 COMPLETE. Awaiting approval for Phase 4.")
print(f"{'='*70}")
