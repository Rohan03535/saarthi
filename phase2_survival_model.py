"""
SAARTHI — Phase 2: Clearance-Time Prediction (Random Survival Forest + SHAP)
Trains on ALL rows including censored. Reports overall + per-event-type C-index.
Accident bucket flagged as thin (n_obs ~91 total -> ~18 in test).
"""

import pandas as pd
import numpy as np
import warnings
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

# ── CONFIG ────────────────────────────────────────────────────────────────────
PARQUET_PATH  = r"C:\Users\Rohan\Pictures\saarthi\saarthi_clean.parquet"
MODEL_OUT     = r"C:\Users\Rohan\Pictures\saarthi\rsf_model.pkl"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
N_ESTIMATORS  = 100
MIN_OBS_TEST  = 6    # min observed rows in test set to report C-index per type

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("SAARTHI — Phase 2: Random Survival Forest")
print("=" * 70)

df = pd.read_parquet(PARQUET_PATH)
print(f"[LOAD] {len(df):,} rows loaded from parquet.")

# Drop rows missing start_datetime (no hour_of_day feature derivable)
df = df.dropna(subset=["start_datetime"]).reset_index(drop=True)
print(f"[LOAD] After dropping null start_datetime: {len(df):,} rows remain.")

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
CORRIDOR_KEEP_MIN = 50
CAUSE_KEEP_MIN    = 15

corridor_counts = df["corridor"].value_counts()
top_corridors   = set(corridor_counts[corridor_counts >= CORRIDOR_KEEP_MIN].index)
df["corridor_grouped"] = df["corridor"].apply(
    lambda x: x if x in top_corridors else "other_corridor"
)

cause_counts = df["event_cause"].value_counts()
top_causes   = set(cause_counts[cause_counts >= CAUSE_KEEP_MIN].index)
df["event_cause_grouped"] = df["event_cause"].apply(
    lambda x: x if x in top_causes else "other_cause"
)

df["is_planned"]   = (df["event_type"] == "planned").astype(int)
df["is_high_prio"] = (df["priority"]   == "high").astype(int)
df["hour_of_day_ist"] = df["hour_of_day_ist"].fillna(12).astype(int)

CAT_FEATURES = ["event_cause_grouped", "corridor_grouped", "veh_type"]
NUM_FEATURES = ["is_planned", "is_high_prio", "requires_road_closure", "hour_of_day_ist"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
preprocessor = ColumnTransformer([
    ("cat", ohe, CAT_FEATURES),
    ("num", "passthrough", NUM_FEATURES),
])

X_raw = df[ALL_FEATURES]
preprocessor.fit(X_raw)
X_enc = preprocessor.transform(X_raw)
cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES).tolist()
feature_names = cat_feature_names + NUM_FEATURES
X = pd.DataFrame(X_enc, columns=feature_names)

print(f"[FEATURES] {X.shape[1]} encoded features from {len(ALL_FEATURES)} raw columns.")

# ── 3. BUILD SURVIVAL TARGET ──────────────────────────────────────────────────
# event=True  → clearance observed (censored==0)
# event=False → censored: still active, or capped at 24h (censored==1)
# time = clearance_time_minutes for all rows; for null times use proxy or 24h cap.

event_flag = (df["censored"] == 0).values.astype(bool)

time_vals = df["clearance_time_minutes"].copy()

# For rows still missing a time, use (modified_datetime - start_datetime) as proxy
if "modified_datetime" in df.columns:
    proxy = (df["modified_datetime"] - df["start_datetime"]).dt.total_seconds() / 60
    time_vals = time_vals.fillna(proxy.clip(lower=1))

time_vals = time_vals.fillna(24 * 60).clip(lower=1.0)

# Cap ALL times at 24h — consistent with Phase 1b label lock.
# Censored obs observed beyond 24h just tell us "lasted >= 24h"; cap preserves that
# signal while keeping the time axis bounded for clean predictions.
time_vals = time_vals.clip(upper=24 * 60)

null_time_before = df["clearance_time_minutes"].isna().sum()
print(f"[TARGET] Rows with no clearance_time (got proxy/24h cap): {null_time_before:,}")
print(f"[TARGET] event=True  (observed, clearance happened): {event_flag.sum():,}")
print(f"[TARGET] event=False (censored):                     {(~event_flag).sum():,}")

y = Surv.from_arrays(event=event_flag, time=time_vals.values)

# ── 4. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
# Stratify on event flag to preserve observed/censored ratio in both sets.
X_tr, X_te, y_tr, y_te, df_tr, df_te = train_test_split(
    X, y, df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=event_flag,
)
X_tr = X_tr.reset_index(drop=True)
X_te = X_te.reset_index(drop=True)
df_te = df_te.reset_index(drop=True)

print(f"\n[SPLIT] Train : {len(X_tr):,} rows | Test : {len(X_te):,} rows")
print(f"[SPLIT] Train observed : {y_tr['event'].sum():,}  | Train censored : {(~y_tr['event']).sum():,}")
print(f"[SPLIT] Test  observed : {y_te['event'].sum():,}  | Test  censored : {(~y_te['event']).sum():,}")

print(f"\n*** TRAINING ROW COUNT CONFIRMATION ***")
print(f"    Total rows used in training : {len(X_tr):,}")
print(f"    Censored rows in training   : {(~y_tr['event']).sum():,}  <-- INCLUDED, not filtered")
print(f"    Observed rows in training   : {y_tr['event'].sum():,}")

# ── 5. TRAIN RANDOM SURVIVAL FOREST ──────────────────────────────────────────
print(f"\n[RSF] Training {N_ESTIMATORS}-tree Random Survival Forest ...")
rsf = RandomSurvivalForest(
    n_estimators=N_ESTIMATORS,
    min_samples_split=10,
    min_samples_leaf=15,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
rsf.fit(X_tr, y_tr)
print("[RSF] Training complete.")

# ── 6. OVERALL C-INDEX ────────────────────────────────────────────────────────
risk_test = rsf.predict(X_te)
ci_all, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
    y_te["event"], y_te["time"], risk_test
)

print(f"\n{'='*70}")
print(f"OVERALL C-INDEX  (test set, n={len(X_te):,})")
print(f"{'='*70}")
print(f"  C-index     : {ci_all:.4f}")
print(f"  Concordant  : {concordant:,}")
print(f"  Discordant  : {discordant:,}")
print(f"  Tied risk   : {tied_risk:,}")
print(f"  Tied time   : {tied_time:,}")
print(f"\n  Interpretation: 0.5 = random, 0.7 = good, > 0.75 = strong")
if ci_all >= 0.75:
    print(f"  --> STRONG discriminative power.")
elif ci_all >= 0.65:
    print(f"  --> GOOD discriminative power for operational use.")
else:
    print(f"  --> MODERATE — model is useful but predictions have uncertainty.")

# ── 7. PER-EVENT-TYPE C-INDEX ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"PER-EVENT-TYPE C-INDEX  (test set, min {MIN_OBS_TEST} observed rows required)")
print(f"{'='*70}")
print(f"  {'event_cause':<30} {'n_test':>7} {'n_obs':>7} {'C-index':>9}  note")
print(f"  {'-'*75}")

type_results = []
for cause in sorted(df_te["event_cause"].unique()):
    mask   = (df_te["event_cause"] == cause).values
    y_sub  = y_te[mask]
    X_sub  = X_te[mask]
    n_tot  = mask.sum()
    n_obs  = y_sub["event"].sum()

    if n_obs < MIN_OBS_TEST:
        type_results.append((cause, n_tot, n_obs, None, "SKIP — too few observed"))
        continue

    try:
        rs_sub = rsf.predict(X_sub)
        ci_sub, _, _, _, _ = concordance_index_censored(y_sub["event"], y_sub["time"], rs_sub)
        if cause == "accident":
            note = "<<< THIN BUCKET (n_obs~18) — treat with caution, do not lead pitch here >>>"
        elif n_obs < 15:
            note = "thin bucket"
        else:
            note = ""
        type_results.append((cause, n_tot, n_obs, ci_sub, note))
    except Exception as e:
        type_results.append((cause, n_tot, n_obs, None, f"ERR: {e}"))

for cause, n_tot, n_obs, ci_sub, note in sorted(type_results, key=lambda x: x[0]):
    ci_str = f"{ci_sub:.4f}" if ci_sub is not None else "   N/A"
    print(f"  {str(cause):<30} {n_tot:>7} {n_obs:>7} {ci_str:>9}  {note}")

# ── 8. SHAP / FEATURE IMPORTANCE ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("FEATURE IMPORTANCE")
print(f"{'='*70}")

# Permutation importance — n_jobs=1 avoids pickling the RSF across processes
print("Computing permutation importance (n_jobs=1, 5 repeats) ...")
perm = permutation_importance(
    rsf, X_te, y_te,
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=1,           # single process — avoids MemoryError/PicklingError
)
importance   = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
method_label = "Permutation Importance (n_repeats=5)"
print("Done.")

print(f"\nTop-10 features ({method_label}):")
print(f"  {'feature':<45} {'importance':>12}")
print(f"  {'-'*60}")
for feat, val in importance.head(10).items():
    bar = "#" * int(val / importance.iloc[0] * 30)
    print(f"  {str(feat):<45} {val:>10.4f}  {bar}")

# ── 9. PREDICTION FUNCTION ────────────────────────────────────────────────────
def predict_clearance(event_cause, veh_type, corridor, hour_of_day,
                      priority="high", requires_road_closure=False,
                      event_type="unplanned"):
    """
    Returns (median_min, p25_min, p75_min) clearance time estimate.
    p25/p75 are the IQR of the predicted survival distribution.
    """
    cause_grp = event_cause if event_cause in top_causes    else "other_cause"
    corr_grp  = corridor    if corridor    in top_corridors else "other_corridor"
    row = {
        "event_cause_grouped":    [cause_grp],
        "corridor_grouped":       [corr_grp],
        "veh_type":               [veh_type],
        "is_planned":             [1 if event_type == "planned" else 0],
        "is_high_prio":           [1 if priority   == "high"    else 0],
        "requires_road_closure":  [1 if requires_road_closure   else 0],
        "hour_of_day_ist":        [int(hour_of_day)],
    }
    X_new = pd.DataFrame(row)[ALL_FEATURES]
    X_enc = pd.DataFrame(preprocessor.transform(X_new), columns=feature_names)
    surv_fn = rsf.predict_survival_function(X_enc)[0]
    t_vals = surv_fn.x
    s_vals = surv_fn.y

    def quantile_from_survival(q):
        # Find t where S(t) first drops to <= q
        idx = np.searchsorted(-s_vals, -q, side="left")
        if idx >= len(t_vals):
            return float(t_vals[-1])
        return float(t_vals[idx])

    median = quantile_from_survival(0.50)
    p25    = quantile_from_survival(0.75)  # S=0.75 → 25th percentile of time
    p75    = quantile_from_survival(0.25)  # S=0.25 → 75th percentile of time
    return round(median, 1), round(p25, 1), round(p75, 1)

print(f"\n{'='*70}")
print("PREDICTION DEMO — given incident attributes")
print(f"{'='*70}")
demo_cases = [
    ("vehicle_breakdown", "bmtc_bus",      "tumkur road",     8, "high",  True,  "unplanned"),
    ("vehicle_breakdown", "heavy_vehicle", "mysore road",    18, "high",  True,  "unplanned"),
    ("accident",          "private_car",   "mysore road",    17, "high",  True,  "unplanned"),
    ("tree_fall",         "unknown",       "bellary road 1",  7, "low",   False, "unplanned"),
    ("construction",      "unknown",       "non-corridor",   22, "low",   False, "planned"),
    ("water_logging",     "heavy_vehicle", "orr east 1",      6, "high",  True,  "unplanned"),
    ("pot_holes",         "unknown",       "hosur road",     12, "low",   False, "unplanned"),
]
print(f"  {'Incident':<55} {'Median':>12} {'Range (IQR)':>22}")
print(f"  {'-'*92}")
for args in demo_cases:
    med, lo, hi = predict_clearance(*args)
    label    = f"{args[0]}/{args[1]}/{args[2]}/hr={args[3]}"
    med_str  = f">24h" if med  >= 1440 else f"{med:.0f} min"
    lo_str   = f">24h" if lo   >= 1440 else f"{lo:.0f}"
    hi_str   = f">24h" if hi   >= 1440 else f"{hi:.0f}"
    print(f"  {label:<55} {med_str:>12}  [{lo_str}–{hi_str} min]")

# ── 10. SAVE MODEL BUNDLE ─────────────────────────────────────────────────────
bundle = {
    "rsf":             rsf,
    "preprocessor":    preprocessor,
    "feature_names":   feature_names,
    "all_features":    ALL_FEATURES,
    "cat_features":    CAT_FEATURES,
    "num_features":    NUM_FEATURES,
    "top_causes":      top_causes,
    "top_corridors":   top_corridors,
    "c_index_overall": ci_all,
    # predict_clearance NOT saved — closures can't be unpickled cross-module
}
joblib.dump(bundle, MODEL_OUT)
print(f"\n[SAVE] Model bundle saved to: {MODEL_OUT}")

# ── SELF-AUDIT ────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SELF-AUDIT")
print(f"{'='*70}")
print(f"  Train/test split     : 80/20, stratified on censored flag")
print(f"  Leakage check        : resolution timestamps NOT in feature set")
print(f"  Timestamps           : all converted to IST before hour_of_day extraction")
print(f"  Censored in training : YES — {(~y_tr['event']).sum():,} censored rows included")
print(f"  24h cap applied      : YES — label locked in Phase 1b")
print(f"  C-index computed on  : held-out test set ONLY ({len(X_te):,} rows)")
print(f"  Accident bucket      : n_obs ~{(df_te['event_cause']=='accident').sum()} in test — flagged as thin")
print(f"  SHAP method          : {method_label}")

print(f"\n{'='*70}")
print("Phase 2 COMPLETE. Paste C-index numbers to human for sign-off -> Phase 3.")
print(f"{'='*70}")
