from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "dataset_A_northwestern_mxd" / "processed_full" / "MxD_Data_User_Study"
OUTDIR = ROOT / "derived" / "dataset_A_phase1"
AUDIT = ROOT / "audit"
OUTDIR.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)

TABLE_OUT = OUTDIR / "dataset_A_phase1_modeling_table.csv"
DICT_OUT = OUTDIR / "dataset_A_phase1_feature_dictionary.csv"
AUDIT_OUT = AUDIT / "dataset_A_phase1_feature_build_audit.txt"

# ---------------------------------------------------------------------
# Scientific design
# ---------------------------------------------------------------------
CONTEXT_CORE = [
    "physical_fatigue_initial",
    "mental_fatigue_initial",
    "task",
    "rep",
    "weights_added",
]

CONTEXT_EXTENDED = CONTEXT_CORE + [
    "age",
    "weight",
    "height",
    "gender",
]

OUTCOME_COLS = [
    "physical_fatigue_final",
    "mental_fatigue_final",
    "performance_rating",
]

# Gyroscope deliberately excluded.
PHYS_CHANNELS = [
    "HR_Processed",
    "HR",
    "HRV",
    "RR",
    "ECG",
    "Temperature",
]

CHEST_ACC = ["acc_X", "acc_Y", "acc_Z"]

IMU_ACC = []
for i in range(1, 6):
    IMU_ACC += [f"IMU_{i}_ax_g_", f"IMU_{i}_ay_g_", f"IMU_{i}_az_g_"]

META_MAP = {
    "Physical Fatigue - Initial": "physical_fatigue_initial",
    "Physical Fatigue - Final": "physical_fatigue_final",
    "Mental Fatigue - Initial": "mental_fatigue_initial",
    "Mental Fatigue - Final": "mental_fatigue_final",
    "Performance rating": "performance_rating",
    "Age": "age",
    "Weight": "weight",
    "Height": "height",
    "Weights added": "weights_added",
    "Gender": "gender",
}

def parse_identity(path):
    s = str(path)
    task = "Composite" if "\\composite\\" in s.lower() else ("WireHarness" if "\\ziptie\\" in s.lower() else None)

    # Standard P### first; otherwise nonstandard Pxxx such as PA00.
    m = re.search(r"(P\d{3})[CZ]\d{3}S\d{3}", s, flags=re.I)
    if m:
        pid = m.group(1).upper()
        standard = True
    else:
        m = re.search(r"(P[A-Z0-9]{3})[CZ]\d{3}S\d{3}", s, flags=re.I)
        pid = m.group(1).upper() if m else None
        standard = False

    mrep = re.search(r"Rep(\d+)_", Path(path).name, flags=re.I)
    rep = int(mrep.group(1)) if mrep else None

    return task, pid, standard, rep

def first_numeric(df, col):
    if col not in df.columns:
        return np.nan
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.iloc[0]) if len(s) else np.nan

def clean_numeric(series):
    """
    Conservative cleaning:
    - coerce nonnumeric values to NaN
    - keep all finite numeric values
    - DO NOT treat -2 as missing because the current documentation has not
      established a universal sentinel interpretation and -2 can be physically
      valid for acceleration.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s[~np.isfinite(s)] = np.nan
    return s

def summarize_1d(series):
    s = clean_numeric(series).dropna()
    if len(s) == 0:
        return {
            "mean": np.nan, "std": np.nan, "median": np.nan,
            "q25": np.nan, "q75": np.nan, "p05": np.nan, "p95": np.nan,
            "rms": np.nan, "range": np.nan, "slope": np.nan,
            "valid_fraction": 0.0,
        }

    x = s.to_numpy(dtype=float)
    q05, q25, q50, q75, q95 = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])

    if len(x) >= 2:
        t = np.linspace(-0.5, 0.5, len(x))
        denom = np.sum((t - t.mean()) ** 2)
        slope = np.sum((t - t.mean()) * (x - x.mean())) / denom if denom > 0 else np.nan
    else:
        slope = np.nan

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "p05": float(q05),
        "p95": float(q95),
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "range": float(np.max(x) - np.min(x)),
        "slope": float(slope) if np.isfinite(slope) else np.nan,
        "valid_fraction": float(len(x) / len(series)) if len(series) else np.nan,
    }

def add_channel_features(row, prefix, series):
    stats = summarize_1d(series)
    for k, v in stats.items():
        row[f"sensor__{prefix}__{k}"] = v

def add_vector_features(row, prefix, df, xcol, ycol, zcol):
    if not all(c in df.columns for c in [xcol, ycol, zcol]):
        return

    x = clean_numeric(df[xcol])
    y = clean_numeric(df[ycol])
    z = clean_numeric(df[zcol])

    valid = x.notna() & y.notna() & z.notna()
    if not valid.any():
        mag = pd.Series(np.nan, index=df.index)
    else:
        mag = pd.Series(np.nan, index=df.index, dtype=float)
        mag.loc[valid] = np.sqrt(
            x.loc[valid].to_numpy() ** 2 +
            y.loc[valid].to_numpy() ** 2 +
            z.loc[valid].to_numpy() ** 2
        )

    add_channel_features(row, f"{prefix}_magnitude", mag)

    if valid.any():
        xa = np.abs(x.loc[valid].to_numpy())
        ya = np.abs(y.loc[valid].to_numpy())
        za = np.abs(z.loc[valid].to_numpy())
        row[f"sensor__{prefix}__sma"] = float(np.mean(xa + ya + za))
    else:
        row[f"sensor__{prefix}__sma"] = np.nan

def time_quality(df):
    if "Time" not in df.columns:
        return np.nan, np.nan, np.nan
    t = clean_numeric(df["Time"]).dropna()
    if len(t) < 2:
        return np.nan, np.nan, np.nan
    dt = t.diff().dropna()
    dt = dt[(dt > 0) & np.isfinite(dt)]
    return (
        float(t.min()),
        float(t.max()),
        float(dt.median()) if len(dt) else np.nan,
    )

rows = []
failures = []

rep_files = sorted([p for p in BASE.rglob("Rep*.csv") if re.match(r"Rep\d+_", p.name, flags=re.I)])

for idx, path in enumerate(rep_files, start=1):
    task, pid, pid_standard, rep = parse_identity(path)

    try:
        header = list(pd.read_csv(path, nrows=0).columns)
        needed = set(["Time"] + list(META_MAP.keys()) + PHYS_CHANNELS + CHEST_ACC + IMU_ACC)
        usecols = [c for c in header if c in needed]
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
    except Exception as e:
        failures.append((str(path), f"{type(e).__name__}: {e}"))
        continue

    row = {
        "participant_id": pid,
        "participant_id_standard": int(bool(pid_standard)),
        "task": task,
        "rep": rep,
        "source_file": str(path.relative_to(BASE)),
        "n_sensor_rows": len(df),
    }

    # Context and outcome metadata
    for src, dst in META_MAP.items():
        row[dst] = first_numeric(df, src)

    # Primary target and change score
    if pd.notna(row.get("physical_fatigue_initial")) and pd.notna(row.get("physical_fatigue_final")):
        row["physical_fatigue_change"] = (
            row["physical_fatigue_final"] - row["physical_fatigue_initial"]
        )
    else:
        row["physical_fatigue_change"] = np.nan

    # Time fields are QC only, not primary predictors
    tmin, tmax, dtmed = time_quality(df)
    row["qc__time_min"] = tmin
    row["qc__time_max"] = tmax
    row["qc__median_positive_dt"] = dtmed

    # Scalar physiological channels
    for c in PHYS_CHANNELS:
        if c in df.columns:
            prefix = re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_").lower()
            add_channel_features(row, prefix, df[c])

    # Chest acceleration axes
    for c in CHEST_ACC:
        if c in df.columns:
            prefix = re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_").lower()
            add_channel_features(row, prefix, df[c])
    add_vector_features(row, "chest_acc", df, "acc_X", "acc_Y", "acc_Z")

    # Five IMU acceleration axes + magnitudes
    for i in range(1, 6):
        ax, ay, az = f"IMU_{i}_ax_g_", f"IMU_{i}_ay_g_", f"IMU_{i}_az_g_"

        for c, axis in [(ax, "x"), (ay, "y"), (az, "z")]:
            if c in df.columns:
                add_channel_features(row, f"imu{i}_acc_{axis}", df[c])

        add_vector_features(row, f"imu{i}_acc", df, ax, ay, az)

    rows.append(row)

    if idx % 50 == 0 or idx == len(rep_files):
        print(f"Processed {idx}/{len(rep_files)} repetition files")

model = pd.DataFrame(rows)

# Stable ordering
# IMPORTANT:
# task and rep are scientific context variables, not identifiers.
# Keeping them out of id_cols prevents duplicate column names.
id_cols = [
    "participant_id", "participant_id_standard", "source_file",
]
context_cols = [c for c in CONTEXT_EXTENDED if c in model.columns]
out_cols = [c for c in [
    "physical_fatigue_final",
    "physical_fatigue_change",
    "mental_fatigue_final",
    "performance_rating",
] if c in model.columns]
qc_cols = sorted([c for c in model.columns if c.startswith("qc__") or c == "n_sensor_rows"])
sensor_cols = sorted([c for c in model.columns if c.startswith("sensor__")])

ordered = id_cols + context_cols + out_cols + qc_cols + sensor_cols
ordered += [c for c in model.columns if c not in ordered]

# Defensive de-duplication: preserve first occurrence only.
ordered = list(dict.fromkeys(ordered))
model = model.loc[:, ordered]

# Hard-stop if any duplicate column names remain.
if model.columns.duplicated().any():
    dupes = model.columns[model.columns.duplicated()].tolist()
    raise RuntimeError(f"Duplicate modeling-table columns remain: {dupes}")

model.to_csv(TABLE_OUT, index=False)

# ---------------------------------------------------------------------
# Feature dictionary
# ---------------------------------------------------------------------
dictionary = []

def role_for(c):
    if c in id_cols:
        return "identifier"
    if c in CONTEXT_CORE:
        return "context_core"
    if c in CONTEXT_EXTENDED:
        return "context_extended"
    if c == "physical_fatigue_final":
        return "primary_outcome"
    if c == "physical_fatigue_change":
        return "secondary_outcome"
    if c in ["mental_fatigue_final", "performance_rating"]:
        return "excluded_posttask"
    if c.startswith("sensor__"):
        return "wearable_feature"
    if c.startswith("qc__") or c == "n_sensor_rows":
        return "quality_control_only"
    return "other"

def family_for(c):
    if c.startswith("sensor__hr") or c.startswith("sensor__rr"):
        return "physiology_hr"
    if c.startswith("sensor__ecg"):
        return "physiology_ecg"
    if c.startswith("sensor__temperature"):
        return "physiology_temperature"
    if c.startswith("sensor__chest_acc"):
        return "chest_acceleration"
    if re.match(r"sensor__imu\d+_acc", c):
        return "imu_acceleration"
    if c in CONTEXT_CORE:
        return "context_core"
    if c in CONTEXT_EXTENDED:
        return "context_extended"
    if c in ["physical_fatigue_final", "physical_fatigue_change"]:
        return "outcome"
    if c in ["mental_fatigue_final", "performance_rating"]:
        return "posttask_excluded"
    if c.startswith("qc__") or c == "n_sensor_rows":
        return "quality_control"
    return "identifier_or_other"

for c in model.columns:
    dictionary.append({
        "column": c,
        "role": role_for(c),
        "family": family_for(c),
        "included_in_primary_predictor_set": int(
            role_for(c) in {"context_core", "context_extended", "wearable_feature"}
        ),
        "notes": (
            "Gyroscope features are intentionally absent."
            if c.startswith("sensor__") else ""
        ),
    })

pd.DataFrame(dictionary).to_csv(DICT_OUT, index=False)

# ---------------------------------------------------------------------
# Audit report
# ---------------------------------------------------------------------
lines = []
def add(s=""):
    lines.append(str(s))

add("DATASET A — PHASE I REPETITION-LEVEL FEATURE BUILD AUDIT")
add("=" * 118)
add(f"Source repetition files discovered: {len(rep_files)}")
add(f"Successfully processed: {len(model)}")
add(f"Failures: {len(failures)}")
for fp, err in failures[:20]:
    add(f"  {fp} | {err}")
add()

add("1. MODELING TABLE SIZE")
add("-" * 118)
add(f"Rows: {len(model)}")
add(f"Columns: {len(model.columns)}")
add(f"Duplicate column names: {int(model.columns.duplicated().sum())}")
add(f"Wearable feature columns: {len(sensor_cols)}")
add(f"QC-only columns: {len(qc_cols)}")
add()

add("2. PARTICIPANT / TASK COVERAGE")
add("-" * 118)
task_series = model.loc[:, "task"]
if isinstance(task_series, pd.DataFrame):
    raise RuntimeError("Internal error: duplicate 'task' columns detected after de-duplication.")
for task in sorted(task_series.dropna().astype(str).unique()):
    g = model[task_series.astype(str) == task]
    add(
        f"{task}: repetitions={len(g)}, unique participant labels={g['participant_id'].nunique()}, "
        f"standard participant IDs={g.loc[g.participant_id_standard==1, 'participant_id'].nunique()}"
    )
add(
    f"Overall unique participant labels: {model['participant_id'].nunique()} "
    f"(standard={model.loc[model.participant_id_standard==1, 'participant_id'].nunique()}, "
    f"nonstandard={model.loc[model.participant_id_standard==0, 'participant_id'].nunique()})"
)
add()

add("3. OUTCOME DISTRIBUTION")
add("-" * 118)
for c in ["physical_fatigue_initial", "physical_fatigue_final", "physical_fatigue_change"]:
    s = pd.to_numeric(model[c], errors="coerce")
    add(
        f"{c}: n={s.notna().sum()}, missing={s.isna().sum()}, unique={s.nunique(dropna=True)}, "
        f"min={s.min()}, median={s.median()}, mean={s.mean():.4f}, max={s.max()}"
    )
add()

add("4. FEATURE FAMILIES")
add("-" * 118)
dict_df = pd.DataFrame(dictionary)
fam = (
    dict_df[dict_df["role"] == "wearable_feature"]
    .groupby("family")
    .size()
    .sort_values(ascending=False)
)
add(fam.to_string())
add()

add("5. MISSINGNESS")
add("-" * 118)
sensor_missing = model[sensor_cols].isna().mean().sort_values(ascending=False)
add("Top 20 wearable features by missing rate:")
for c, r in sensor_missing.head(20).items():
    add(f"  {c}: {r:.4f}")
add()
add(f"Wearable features with >20% missingness: {int((sensor_missing > 0.20).sum())}/{len(sensor_missing)}")
add(f"Wearable features with 100% missingness: {int((sensor_missing >= 1.0).sum())}/{len(sensor_missing)}")
add()

add("6. CONSTANT / NEAR-CONSTANT FEATURES")
add("-" * 118)
constant = []
for c in sensor_cols:
    s = pd.to_numeric(model[c], errors="coerce").dropna()
    if len(s) and s.nunique() <= 1:
        constant.append(c)
add(f"Constant wearable features: {len(constant)}")
for c in constant[:50]:
    add(f"  {c}")
add()

add("7. SCIENTIFIC ROLE DEFINITIONS")
add("-" * 118)
add("Primary outcome:")
add("  physical_fatigue_final")
add()
add("Secondary outcome:")
add("  physical_fatigue_change = final - initial")
add()
add("Core context:")
for c in CONTEXT_CORE:
    add(f"  {c}")
add()
add("Extended context:")
for c in CONTEXT_EXTENDED:
    if c not in CONTEXT_CORE:
        add(f"  {c}")
add()
add("Explicitly excluded from primary predictors:")
add("  mental_fatigue_final")
add("  performance_rating")
add("  all gyroscope channels/features")
add("  source_file, n_sensor_rows and qc__* fields")
add()

add("8. PA00 HANDLING")
add("-" * 118)
pa = model[model["participant_id"] == "PA00"]
add(f"PA00 repetitions: {len(pa)}")
add("Primary recommendation:")
add("  Keep PA00 as an independent participant-group label in the all-data analysis,")
add("  but run a prespecified sensitivity analysis excluding PA00 because its identifier is nonstandard.")
add("  Do NOT remap PA00 to another P### identity without external provenance.")
add()

add("9. NEXT MODELING STEP")
add("-" * 118)
add("Use participant-grouped validation only.")
add("No random row-level or repetition-level split may place the same participant in train and test.")
add("Recommended first benchmark:")
add("  M0a: core context")
add("  M0b: extended context")
add("  M1 : wearable only")
add("  M2a: core context + wearable")
add("  M2b: extended context + wearable")
add("Compare incremental value using paired outer-fold differences in MAE, RMSE and R^2.")
add("All imputation, scaling and feature filtering must be fit inside the training fold.")
add()
add("=" * 118)
add("END OF AUDIT")

AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("")
print("Phase I feature build complete:")
print(f"  {TABLE_OUT}")
print(f"  {DICT_OUT}")
print(f"  {AUDIT_OUT}")
print("")
print("Upload dataset_A_phase1_feature_build_audit.txt to ChatGPT.")
