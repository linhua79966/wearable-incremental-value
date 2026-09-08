from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "data" / "dataset_B_mmh_si" / "Survival"
DERIVED = ROOT / "derived" / "dataset_B_phase1"
AUDIT = ROOT / "audit"
DERIVED.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)

LONG = REPO / "consv_sds.csv"
SHORT = REPO / "time_fixed_conservative.csv"

TABLE_OUT = DERIVED / "dataset_B_lagged_onset_modeling_table.csv"
DICT_OUT = DERIVED / "dataset_B_lagged_onset_feature_dictionary.csv"
TRIAL_OUT = DERIVED / "dataset_B_task_trial_qc.csv"
AUDIT_OUT = AUDIT / "dataset_B_lagged_onset_build_audit.txt"

META_LONG = {
    "ID", "Index", "Time", "Time1", "Time2",
    "State", "Fatigue", "Task", "Resting_hr",
}

PHYS_DYNAMIC = {"HRRMean", "HRRCV", "HRRSD"}

DEMOGRAPHICS = ["Sex", "Age", "Height_cm", "Weight_kg"]

# -------------------------------------------------------------------------
# Load
# -------------------------------------------------------------------------
long_df = pd.read_csv(LONG)
short_df = pd.read_csv(SHORT)

for c in ["ID","Index","Time","Time1","Time2","State","Fatigue","Resting_hr"]:
    if c in long_df.columns:
        long_df[c] = pd.to_numeric(long_df[c], errors="coerce")

for c in ["ID","Index","Time","Fatigue","Fatigue_rate","Age","Height_cm","Weight_kg","Resting_hr"]:
    if c in short_df.columns:
        short_df[c] = pd.to_numeric(short_df[c], errors="coerce")

# Dynamic wearable columns = all long-form non-metadata sensor descriptors.
dynamic_sensor_cols = [
    c for c in long_df.columns
    if c not in META_LONG
]

if not dynamic_sensor_cols:
    raise SystemExit("No dynamic sensor features detected.")

phys_cols = [c for c in dynamic_sensor_cols if c in PHYS_DYNAMIC]
motion_cols = [c for c in dynamic_sensor_cols if c not in PHYS_DYNAMIC]

location_groups = {}
for loc in ["Wrist", "Hip", "Chest", "Ankle"]:
    location_groups[loc] = [c for c in dynamic_sensor_cols if c.startswith(loc + "_")]

# One short-form record per task-specific ID.
short_one = short_df.drop_duplicates(subset=["ID"]).copy()

# -------------------------------------------------------------------------
# Task-trial QC and quarantine
# -------------------------------------------------------------------------
trial_rows = []
quarantine_ids = set()

for idv, g in long_df.groupby("ID"):
    g = g.sort_values(["Time1","Time2"]).copy()

    if len(g) == 0:
        continue

    idx = int(g["Index"].iloc[0])
    task = str(g["Task"].iloc[0])
    event_count = int((g["State"] == 1).sum())
    long_event_any = int(event_count > 0)
    long_last_time2 = float(g["Time2"].max())

    s = short_one[short_one["ID"] == idv]
    if len(s) != 1:
        short_time = np.nan
        short_fatigue = np.nan
        short_index = np.nan
        short_task = None
        short_match = False
    else:
        sr = s.iloc[0]
        short_time = float(sr["Time"]) if pd.notna(sr["Time"]) else np.nan
        short_fatigue = int(sr["Fatigue"]) if pd.notna(sr["Fatigue"]) else np.nan
        short_index = int(sr["Index"]) if pd.notna(sr["Index"]) else np.nan
        short_task = str(sr["Task"])
        short_match = True

    index_match = bool(short_match and short_index == idx)
    task_match = bool(short_match and short_task == task)
    time_match = bool(
        short_match
        and pd.notna(short_time)
        and np.isclose(short_time, long_last_time2, rtol=0, atol=1e-9)
    )
    event_match = bool(
        short_match
        and pd.notna(short_fatigue)
        and int(short_fatigue) == int(long_event_any)
    )

    intervals_connected = True
    if len(g) > 1:
        intervals_connected = bool(
            np.allclose(
                g["Time2"].to_numpy()[:-1],
                g["Time1"].to_numpy()[1:],
                rtol=0,
                atol=1e-9,
            )
        )

    event_on_last = True
    if event_count:
        event_positions = np.where(g["State"].to_numpy() == 1)[0]
        event_on_last = bool(
            event_count == 1 and event_positions[0] == len(g) - 1
        )

    reasons = []
    if not short_match:
        reasons.append("missing_short_record")
    if not index_match:
        reasons.append("index_mismatch")
    if not task_match:
        reasons.append("task_mismatch")
    if not time_match:
        reasons.append("long_short_time_mismatch")
    if not event_match:
        reasons.append("long_short_event_mismatch")
    if not intervals_connected:
        reasons.append("disconnected_intervals")
    if event_count > 1:
        reasons.append("multiple_state_events")
    if event_count and not event_on_last:
        reasons.append("event_not_last_interval")

    quarantine = len(reasons) > 0
    if quarantine:
        quarantine_ids.add(int(idv))

    trial_rows.append({
        "ID": int(idv),
        "Index": idx,
        "Task": task,
        "n_long_rows": len(g),
        "event_count": event_count,
        "long_event_any": long_event_any,
        "long_last_time2": long_last_time2,
        "short_time": short_time,
        "short_fatigue": short_fatigue,
        "index_match": index_match,
        "task_match": task_match,
        "time_match": time_match,
        "event_match": event_match,
        "intervals_connected": intervals_connected,
        "event_on_last_interval": event_on_last,
        "quarantine": quarantine,
        "quarantine_reason": ";".join(reasons),
    })

trial_qc = pd.DataFrame(trial_rows)
trial_qc.to_csv(TRIAL_OUT, index=False)

# -------------------------------------------------------------------------
# Build one-step-ahead table
# -------------------------------------------------------------------------
rows = []

for idv, g in long_df.groupby("ID"):
    idv = int(idv)

    if idv in quarantine_ids:
        continue

    g = g.sort_values(["Time1","Time2"]).reset_index(drop=True).copy()

    # Since first interval has no previous sensor window, prediction starts
    # from the second available interval.
    if len(g) < 2:
        continue

    s = short_one[short_one["ID"] == idv]
    if len(s) != 1:
        continue
    sr = s.iloc[0]

    for k in range(1, len(g)):
        current = g.iloc[k]
        prev = g.iloc[k - 1]

        row = {
            # Identifiers / grouping
            "worker_index": int(current["Index"]),
            "task_specific_id": idv,
            "task": str(current["Task"]),
            "prediction_interval_index": k + 1,
            "prediction_time_min": float(current["Time1"]),
            "target_interval_end_min": float(current["Time2"]),

            # Primary target: onset in NEXT/current predicted interval
            "fatigue_onset_next_interval": int(current["State"]),

            # Context available at prediction time
            "elapsed_exposure_min": float(current["Time1"]),

            # Baseline physiology handled separately
            "resting_hr": float(current["Resting_hr"]) if pd.notna(current["Resting_hr"]) else np.nan,

            # Demographics from short form
            "sex": str(sr["Sex"]) if "Sex" in sr.index and pd.notna(sr["Sex"]) else np.nan,
            "age": float(sr["Age"]) if "Age" in sr.index and pd.notna(sr["Age"]) else np.nan,
            "height_cm": float(sr["Height_cm"]) if "Height_cm" in sr.index and pd.notna(sr["Height_cm"]) else np.nan,
            "weight_kg": float(sr["Weight_kg"]) if "Weight_kg" in sr.index and pd.notna(sr["Weight_kg"]) else np.nan,

            # QC provenance
            "lag_source_time1_min": float(prev["Time1"]),
            "lag_source_time2_min": float(prev["Time2"]),
        }

        # Lag every dynamic wearable feature by one 10-min interval.
        for c in dynamic_sensor_cols:
            val = pd.to_numeric(pd.Series([prev[c]]), errors="coerce").iloc[0]
            row[f"wearable_lag1__{c}"] = float(val) if pd.notna(val) else np.nan

        rows.append(row)

model = pd.DataFrame(rows)

if model.empty:
    raise SystemExit("No modeling rows were created.")

# -------------------------------------------------------------------------
# Role dictionary
# -------------------------------------------------------------------------
core_context = ["task", "elapsed_exposure_min"]
extended_context = ["sex", "age", "height_cm", "weight_kg"]
baseline_phys = ["resting_hr"]

phys_lag = [f"wearable_lag1__{c}" for c in phys_cols]
motion_lag = [f"wearable_lag1__{c}" for c in motion_cols]
all_lag = [f"wearable_lag1__{c}" for c in dynamic_sensor_cols]

dict_rows = []

for c in model.columns:
    if c in ["worker_index", "task_specific_id"]:
        role = "identifier"
        family = "identifier"
    elif c == "fatigue_onset_next_interval":
        role = "primary_outcome"
        family = "outcome"
    elif c in core_context:
        role = "context_core"
        family = "task_exposure_context"
    elif c in extended_context:
        role = "context_extended"
        family = "demographic_context"
    elif c in baseline_phys:
        role = "baseline_physiology"
        family = "baseline_physiology"
    elif c in phys_lag:
        role = "wearable_feature"
        family = "dynamic_physiology"
    elif c in motion_lag:
        role = "wearable_feature"
        original = c.replace("wearable_lag1__", "", 1)
        loc = next((x for x in ["Wrist","Hip","Chest","Ankle"] if original.startswith(x + "_")), None)
        family = f"{loc.lower()}_motion" if loc else "motion"
    elif c in ["prediction_interval_index", "prediction_time_min", "target_interval_end_min",
               "lag_source_time1_min", "lag_source_time2_min"]:
        role = "quality_control_only"
        family = "time_qc"
    else:
        role = "other"
        family = "other"

    dict_rows.append({
        "column": c,
        "role": role,
        "family": family,
        "primary_predictor_eligible": int(
            role in {
                "context_core",
                "context_extended",
                "baseline_physiology",
                "wearable_feature",
            }
        ),
    })

dictionary = pd.DataFrame(dict_rows)

model.to_csv(TABLE_OUT, index=False)
dictionary.to_csv(DICT_OUT, index=False)

# -------------------------------------------------------------------------
# Audit report
# -------------------------------------------------------------------------
lines = []
def add(s=""):
    lines.append(str(s))

add("DATASET B — ONE-STEP-AHEAD FATIGUE-ONSET TABLE BUILD AUDIT")
add("=" * 118)

add("1. TASK-TRIAL QC")
add("-" * 118)
add(f"Original task-specific trials: {len(trial_qc)}")
add(f"Quarantined task-specific trials: {int(trial_qc['quarantine'].sum())}")
if trial_qc["quarantine"].any():
    add(trial_qc.loc[
        trial_qc["quarantine"],
        ["ID","Index","Task","n_long_rows","event_count","long_last_time2",
         "short_time","short_fatigue","quarantine_reason"]
    ].to_string(index=False))
add()
add(f"Retained task-specific trials: {int((~trial_qc['quarantine']).sum())}")
add()

add("2. MODELING TABLE")
add("-" * 118)
add(f"Rows: {len(model)}")
add(f"Columns: {len(model.columns)}")
add(f"Underlying workers (Index): {model['worker_index'].nunique()}")
add(f"Task-specific IDs: {model['task_specific_id'].nunique()}")
add(f"MMH rows: {int((model['task']=='MMH').sum())}")
add(f"SI rows: {int((model['task']=='SI').sum())}")
add()

add("3. TARGET DISTRIBUTION")
add("-" * 118)
y = model["fatigue_onset_next_interval"]
add(f"Positive onset intervals: {int(y.sum())}")
add(f"Negative at-risk intervals: {int((y==0).sum())}")
add(f"Positive prevalence: {float(y.mean()):.6f}")
add()

event_by_worker = model.groupby("worker_index")["fatigue_onset_next_interval"].max()
add(f"Workers with at least one retained onset event: {int(event_by_worker.sum())}/{len(event_by_worker)}")
add()

add("4. PREDICTION TIMING")
add("-" * 118)
add("For target interval [Time1_t, Time2_t]:")
add("  predictors use wearable summaries ONLY from previous interval [Time1_{t-1}, Time2_{t-1}].")
add("  elapsed_exposure_min = Time1_t is known at prediction time.")
add("  therefore event-interval sensor summaries are NOT used to predict that event.")
add()

timing_ok = np.isclose(
    model["lag_source_time2_min"].to_numpy(),
    model["prediction_time_min"].to_numpy(),
    rtol=0,
    atol=1e-9,
)
add(f"Previous sensor interval ends exactly at prediction time: {int(timing_ok.sum())}/{len(timing_ok)}")
add()

add("5. FEATURE ROLES")
add("-" * 118)
add("Core context:")
for c in core_context:
    add(f"  {c}")
add("Extended context:")
for c in extended_context:
    add(f"  {c}")
add("Baseline physiology:")
for c in baseline_phys:
    add(f"  {c}")
add(f"Dynamic physiology lagged features: {len(phys_lag)}")
add(f"Dynamic motion lagged features: {len(motion_lag)}")
add(f"All dynamic wearable lagged features: {len(all_lag)}")
for loc, cols in location_groups.items():
    add(f"  {loc}: {len(cols)} source features")
add()

add("6. MISSINGNESS")
add("-" * 118)
eligible = core_context + extended_context + baseline_phys + all_lag
miss = model[eligible].isna().mean().sort_values(ascending=False)
add(f"Predictor columns with >20% missingness: {int((miss > 0.20).sum())}/{len(miss)}")
add(f"Predictor columns with 100% missingness: {int((miss >= 1.0).sum())}/{len(miss)}")
add("Top 20 missingness rates:")
for c, r in miss.head(20).items():
    add(f"  {c}: {r:.4f}")
add()

add("7. LEAKAGE EXCLUSIONS")
add("-" * 118)
add("Never use the following as predictors:")
add("  long-form Fatigue (task-trial future/event flag)")
add("  current State (target)")
add("  long/short Time (final event/censoring time)")
add("  Fatigue_rate")
add("  task_specific_id / worker_index")
add("  current target-interval sensor summaries")
add()

add("8. NEXT BENCHMARK DESIGN")
add("-" * 118)
add("Participant group = worker_index (Index), NOT task-specific ID.")
add("Primary comparison:")
add("  B0a: Task + elapsed exposure")
add("  B0b: B0a + demographics")
add("  B0c: B0b + resting HR baseline physiology")
add("  B1 : lagged dynamic wearable only")
add("  B2a: B0a + lagged dynamic wearable")
add("  B2b: B0b + lagged dynamic wearable")
add("Primary outcome: onset in the next 10-min interval.")
add("Recommended uncertainty: participant-cluster bootstrap on repeated grouped OOF predictions.")
add("Recommended discrimination metric: PR-AUC; also report ROC-AUC.")
add("Recommended probabilistic metrics: participant-balanced Brier score and log loss.")
add()
add("=" * 118)
add("END OF AUDIT")

AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("Dataset B lagged-onset table build complete:")
print(f"  {TABLE_OUT}")
print(f"  {DICT_OUT}")
print(f"  {TRIAL_OUT}")
print(f"  {AUDIT_OUT}")
print("")
print("Upload dataset_B_lagged_onset_build_audit.txt to ChatGPT.")
