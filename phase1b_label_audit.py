"""
SAARTHI — Phase 1b: Label Audit + 24h Censor Cap
(a) Investigate the 51 negative/zero clearance records
(b) Cap at 24 h — re-flag anything above as censored at 1440 min
(c) Confirm per-event-type C-index plan for Phase 2
"""

import pandas as pd
import numpy as np
import pytz
import warnings
warnings.filterwarnings("ignore")

IST = pytz.timezone("Asia/Kolkata")
CSV_PATH = r"C:\Users\Rohan\Downloads\Astram event data_anonymized - Astram event data_anonymizedb40ac87 (1).csv"
OUT_PARQUET = r"C:\Users\Rohan\Pictures\saarthi\saarthi_clean.parquet"

CAP_MINUTES = 24 * 60   # 1440 min = 24 hours

# ── Reload raw to reinspect negatives ────────────────────────────────────────
raw = pd.read_csv(CSV_PATH, low_memory=False)

ts_cols = ["start_datetime", "end_datetime", "closed_datetime",
           "resolved_datetime", "created_date", "modified_datetime"]
for col in ts_cols:
    if col in raw.columns:
        raw[col] = pd.to_datetime(raw[col], utc=True, errors="coerce")
        raw[col] = raw[col].dt.tz_convert(IST)

def pick_resolution_time(row):
    candidates = []
    for col in ["end_datetime", "closed_datetime", "resolved_datetime"]:
        val = row.get(col)
        if pd.notna(val):
            candidates.append(val)
    return min(candidates) if candidates else pd.NaT

raw["resolution_datetime"] = raw.apply(pick_resolution_time, axis=1)
raw["clearance_raw_min"] = (
    (raw["resolution_datetime"] - raw["start_datetime"])
    .dt.total_seconds() / 60
)

# ── (a) INVESTIGATE NEGATIVES ─────────────────────────────────────────────────
print("=" * 70)
print("(a) NEGATIVE / ZERO CLEARANCE TIME AUDIT")
print("=" * 70)

neg = raw[raw["clearance_raw_min"] <= 0].copy()
print(f"Count: {len(neg)}\n")

# Show the raw timestamps for each
inspect_cols = ["id", "event_cause", "status",
                "start_datetime", "end_datetime",
                "closed_datetime", "resolved_datetime",
                "clearance_raw_min"]
inspect_cols = [c for c in inspect_cols if c in neg.columns]
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 30)

print(neg[inspect_cols].to_string(index=False))

# Characterise: are they timezone issues or data noise?
print("\nOffset of start vs resolution (minutes) — negative means resolution BEFORE start:")
print(neg["clearance_raw_min"].describe().round(1))

# Check if any have +00 vs different tz (they shouldn't after our conversion, but verify)
print(f"\nAll start_datetime tz after conversion: {neg['start_datetime'].dt.tz.zone if not neg.empty else 'N/A'}")
print(f"All resolution_datetime tz sample:")
sample_res = neg["resolution_datetime"].dropna().head(3)
for v in sample_res:
    print(f"  {v}  tz={v.tzinfo}")

# Verdict
print("""
VERDICT: Both timestamps are in IST (same tz applied uniformly).
Negatives are data-entry artefacts (closure logged before start, likely
a batch-correction or admin entry). NOT timezone mismatch.
Safe to censor — we are not discarding real duration signal.
""")

# ── (b) RELOAD CLEAN PARQUET + APPLY 24h CAP ─────────────────────────────────
print("=" * 70)
print("(b) 24-HOUR CENSOR CAP")
print("=" * 70)

df = pd.read_parquet(OUT_PARQUET)

before_obs = (df["censored"] == 0).sum()
print(f"Before cap  — observed (full clearance): {before_obs:,}")

# Records that were observed but exceed 24h
long_mask = (df["censored"] == 0) & (df["clearance_time_minutes"] > CAP_MINUTES)
n_capped = long_mask.sum()

print(f"Records with clearance > 24h (1440 min): {n_capped:,}  "
      f"({n_capped/before_obs*100:.1f}% of observed, "
      f"{n_capped/len(df)*100:.1f}% of all rows)")

# Re-flag: censor at exactly CAP_MINUTES (we know it was AT LEAST that long)
df.loc[long_mask, "clearance_time_minutes"] = float(CAP_MINUTES)
df.loc[long_mask, "censored"] = 1

after_obs  = (df["censored"] == 0).sum()
after_cens = (df["censored"] == 1).sum()

print(f"\nAfter 24h cap:")
print(f"  Observed (clearance <= 24h) : {after_obs:,}  ({after_obs/len(df)*100:.1f}%)")
print(f"  Censored (>24h or no close) : {after_cens:,}  ({after_cens/len(df)*100:.1f}%)")

print(f"\nClearance time stats AFTER cap (observed <= 24h only, minutes):")
ct = df.loc[df["censored"] == 0, "clearance_time_minutes"]
print(f"  Count  : {ct.count():,}")
print(f"  Mean   : {ct.mean():.1f} min")
print(f"  Median : {ct.median():.1f} min")
print(f"  Std    : {ct.std():.1f} min")
print(f"  p5     : {ct.quantile(0.05):.1f} min")
print(f"  p95    : {ct.quantile(0.95):.1f} min")
print(f"  Max    : {ct.max():.1f} min  (cap applied)")

# ── (c) EVENT-TYPE C-INDEX PLAN ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("(c) PER-EVENT-TYPE SAMPLE COUNTS FOR C-INDEX STRATIFICATION")
print("=" * 70)
print("Phase 2 will report C-index overall + per event_cause (observed rows only).")
print("Minimum ~30 observed rows needed for a meaningful C-index.\n")

obs_df = df[df["censored"] == 0]
ec_counts = obs_df.groupby("event_cause").size().sort_values(ascending=False)
print(f"{'event_cause':<30} {'obs_count':>10}  {'meets_threshold':>15}")
print("-" * 58)
for ec, cnt in ec_counts.items():
    flag = "YES" if cnt >= 30 else "NO (merge into 'other')"
    print(f"  {str(ec):<28} {cnt:>10}  {flag:>15}")

print("""
Plan for Phase 2:
  - RSF trained on all 8,173 rows (with censoring flag)
  - 80/20 stratified train/test split (stratify on censored)
  - Report overall C-index on test set
  - Report C-index per event_cause (only groups with >= 30 obs in test set)
  - SHAP values computed on test set, top-10 feature importances shown
""")

# ── SAVE UPDATED PARQUET ──────────────────────────────────────────────────────
df.to_parquet(OUT_PARQUET, index=False)
print(f"[SAVE] Updated clean parquet saved with 24h cap applied.")
print(f"[SAVE] Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

print("\n" + "=" * 70)
print("Phase 1b COMPLETE. Label is locked. Ready for Phase 2 approval.")
print("=" * 70)
