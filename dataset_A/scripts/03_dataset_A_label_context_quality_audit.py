from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "data" / "dataset_A_northwestern_mxd" / "official_code"
PM = REPO / "Predictive_Models"
AUDIT = ROOT / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)

TXT_OUT = AUDIT / "dataset_A_label_context_quality_audit.txt"
CSV_OUT = AUDIT / "dataset_A_segment_manifest.csv"

FOLDERS = {
    "composite_train": ("Composite", "train"),
    "composite_test": ("Composite", "test"),
    "ziptie_train": ("WireHarness", "train"),
    "ziptie_test": ("WireHarness", "test"),
}

META_COLS = [
    "Physical Fatigue - Initial",
    "Physical Fatigue - Final",
    "Mental Fatigue - Initial",
    "Mental Fatigue - Final",
    "Performance rating",
    "Age",
    "Weight",
    "Height",
    "Weights added",
    "Gender",
]

PHYS_COLS = ["HR_Processed", "HR", "HRV", "RR", "RRSQI", "ECGSQI", "ECG", "Temperature"]
CHEST_ACC = ["acc_X", "acc_Y", "acc_Z"]

GYRO_COLS = []
ACC_COLS = []
for i in range(1, 6):
    ACC_COLS += [f"IMU_{i}_ax_g_", f"IMU_{i}_ay_g_", f"IMU_{i}_az_g_"]
    GYRO_COLS += [f"IMU_{i}_gx_dps_", f"IMU_{i}_gy_dps_", f"IMU_{i}_gz_dps_"]

KEEP_COLS = set(["Time"] + META_COLS + PHYS_COLS + CHEST_ACC + ACC_COLS + GYRO_COLS)

def parse_segment(filename):
    stem = Path(filename).stem
    if stem.startswith("Initial_Rest_"):
        return "Initial_Rest", np.nan
    if stem.startswith("Final_Rest_"):
        return "Final_Rest", np.nan
    m = re.match(r"Rep(\d+)_", stem, flags=re.I)
    if m:
        return f"Rep{int(m.group(1))}", int(m.group(1))
    return stem, np.nan

def pid_from_folder(folder_name):
    m = re.match(r"(P\d+)", folder_name)
    return m.group(1) if m else folder_name

def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def first_nonnull(s):
    x = s.dropna()
    return x.iloc[0] if len(x) else np.nan

def eq_fraction(a, b):
    a = safe_numeric(a)
    b = safe_numeric(b)
    mask = a.notna() & b.notna()
    if not mask.any():
        return np.nan
    # Exact duplication is what we want to detect; allow microscopic float serialization noise.
    return float(np.isclose(a[mask].to_numpy(), b[mask].to_numpy(),
                            rtol=0.0, atol=1e-12).mean())

records = []
gyro_detail = []
errors = []

for folder_name, (task, split) in FOLDERS.items():
    base = PM / folder_name
    if not base.exists():
        errors.append(f"Missing folder: {base}")
        continue

    for pdir in sorted([x for x in base.iterdir() if x.is_dir()]):
        participant = pid_from_folder(pdir.name)

        for csv_path in sorted(pdir.glob("*.csv")):
            segment, rep = parse_segment(csv_path.name)

            try:
                # Header first so usecols is robust to slight schema differences.
                header = list(pd.read_csv(csv_path, nrows=0).columns)
                usecols = [c for c in header if c in KEEP_COLS]
                df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
            except Exception as e:
                errors.append(f"{csv_path}: {type(e).__name__}: {e}")
                continue

            row = {
                "task": task,
                "split": split,
                "participant": participant,
                "session_folder": pdir.name,
                "segment": segment,
                "rep": rep,
                "file": str(csv_path.relative_to(REPO)),
                "n_rows": len(df),
            }

            if "Time" in df.columns:
                t = safe_numeric(df["Time"])
                row["time_min"] = t.min()
                row["time_max"] = t.max()
                dt = t.diff().dropna()
                dt = dt[np.isfinite(dt)]
                row["time_step_median"] = dt.median() if len(dt) else np.nan
            else:
                row["time_min"] = np.nan
                row["time_max"] = np.nan
                row["time_step_median"] = np.nan

            # Metadata / outcomes: verify whether they are constant inside each segment.
            for c in META_COLS:
                key = c.replace(" ", "_").replace("-", "").lower()
                if c in df.columns:
                    s = safe_numeric(df[c])
                    row[f"{key}__first"] = first_nonnull(s)
                    row[f"{key}__mean"] = s.mean()
                    row[f"{key}__min"] = s.min()
                    row[f"{key}__max"] = s.max()
                    row[f"{key}__nunique"] = s.dropna().nunique()
                    row[f"{key}__missing_rate"] = s.isna().mean()
                else:
                    row[f"{key}__first"] = np.nan
                    row[f"{key}__mean"] = np.nan
                    row[f"{key}__min"] = np.nan
                    row[f"{key}__max"] = np.nan
                    row[f"{key}__nunique"] = np.nan
                    row[f"{key}__missing_rate"] = 1.0

            # Sensor missing / sentinel checks.
            sensor_present = [c for c in (PHYS_COLS + CHEST_ACC + ACC_COLS + GYRO_COLS) if c in df.columns]
            sensor_vals = df[sensor_present].apply(pd.to_numeric, errors="coerce") if sensor_present else pd.DataFrame()
            if not sensor_vals.empty:
                row["sensor_missing_rate"] = float(sensor_vals.isna().to_numpy().mean())
                arr = sensor_vals.to_numpy(dtype=float)
                finite = np.isfinite(arr)
                row["sensor_minus2_rate"] = float(((arr == -2) & finite).sum() / max(finite.sum(), 1))
            else:
                row["sensor_missing_rate"] = np.nan
                row["sensor_minus2_rate"] = np.nan

            # Critical check: are gx, gy, gz copies of each other?
            for i in range(1, 6):
                gx, gy, gz = [f"IMU_{i}_g{axis}_dps_" for axis in ("x", "y", "z")]
                if all(c in df.columns for c in (gx, gy, gz)):
                    f_xy = eq_fraction(df[gx], df[gy])
                    f_xz = eq_fraction(df[gx], df[gz])
                    f_yz = eq_fraction(df[gy], df[gz])
                    row[f"imu{i}_gyro_xy_equal_frac"] = f_xy
                    row[f"imu{i}_gyro_xz_equal_frac"] = f_xz
                    row[f"imu{i}_gyro_yz_equal_frac"] = f_yz
                    gyro_detail.append({
                        "task": task,
                        "split": split,
                        "participant": participant,
                        "segment": segment,
                        "imu": i,
                        "xy_equal_fraction": f_xy,
                        "xz_equal_fraction": f_xz,
                        "yz_equal_fraction": f_yz,
                    })

            records.append(row)

manifest = pd.DataFrame(records)
manifest.to_csv(CSV_OUT, index=False)

lines = []
def add(s=""):
    lines.append(str(s))

add("DATASET A — LABEL / CONTEXT / SENSOR QUALITY AUDIT")
add("=" * 110)
add(f"Repository: {REPO}")
add(f"CSV segments successfully audited: {len(manifest)}")
add(f"Errors: {len(errors)}")
if errors:
    for e in errors[:20]:
        add(f"  ERROR: {e}")
add()

if manifest.empty:
    add("FATAL: no CSV data could be audited.")
    TXT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Audit failed. See: {TXT_OUT}")
    raise SystemExit(1)

# ------------------------------------------------------------------
add("1. PARTICIPANT / TASK / SPLIT STRUCTURE")
add("-" * 110)
pivot = (
    manifest.groupby(["task", "split"])["participant"]
    .nunique()
    .rename("n_participants")
    .reset_index()
)
add(pivot.to_string(index=False))
add()

all_pids = sorted(manifest["participant"].dropna().unique())
add(f"Unique participant IDs across current GitHub subset: {len(all_pids)}")
add("IDs: " + ", ".join(all_pids))
add()

for task in sorted(manifest["task"].unique()):
    tr = set(manifest[(manifest.task == task) & (manifest.split == "train")]["participant"])
    te = set(manifest[(manifest.task == task) & (manifest.split == "test")]["participant"])
    add(f"{task}: train/test participant overlap = {sorted(tr & te)}")

task_sets = {
    task: set(manifest[manifest.task == task]["participant"])
    for task in sorted(manifest["task"].unique())
}
if len(task_sets) >= 2:
    names = list(task_sets)
    add(f"Cross-task participant overlap ({names[0]} vs {names[1]}): {sorted(task_sets[names[0]] & task_sets[names[1]])}")
add()

# ------------------------------------------------------------------
add("2. SEGMENT COMPLETENESS")
add("-" * 110)
seg_table = (
    manifest.groupby(["task", "split", "segment"])
    .size()
    .rename("n_files")
    .reset_index()
)
add(seg_table.to_string(index=False))
add()

reps = manifest[manifest["rep"].notna()].copy()
rep_counts = (
    reps.groupby(["task", "participant"])["rep"]
    .agg(["count", "nunique", "min", "max"])
    .reset_index()
)
add("Repetition completeness by participant/task:")
add(rep_counts.to_string(index=False))
add()

# ------------------------------------------------------------------
add("3. FATIGUE LABEL STRUCTURE")
add("-" * 110)

phys_i = "physical_fatigue__initial__first"
phys_f = "physical_fatigue__final__first"
ment_i = "mental_fatigue__initial__first"
ment_f = "mental_fatigue__final__first"
perf = "performance_rating__first"

for col, label in [
    (phys_i, "Physical fatigue initial"),
    (phys_f, "Physical fatigue final"),
    (ment_i, "Mental fatigue initial"),
    (ment_f, "Mental fatigue final"),
    (perf, "Performance rating"),
]:
    if col in manifest.columns:
        s = pd.to_numeric(manifest[col], errors="coerce")
        add(f"{label}: n={s.notna().sum()}, unique={s.nunique(dropna=True)}, "
            f"min={s.min()}, median={s.median()}, max={s.max()}")
add()

# Check whether metadata labels are constant within each raw segment.
add("Within-segment non-constant metadata/outcome fields:")
nonconstant_summary = []
for original in META_COLS:
    key = original.replace(" ", "_").replace("-", "").lower()
    ncol = f"{key}__nunique"
    if ncol in manifest.columns:
        n_bad = int((pd.to_numeric(manifest[ncol], errors="coerce") > 1).sum())
        nonconstant_summary.append((original, n_bad))
        add(f"  {original}: {n_bad} segment files with >1 unique value")
add()

# Per-repetition fatigue progression.
if not reps.empty and phys_i in reps.columns and phys_f in reps.columns:
    rr = reps[["task", "split", "participant", "rep", phys_i, phys_f, ment_i, ment_f, perf]].copy()
    rr = rr.sort_values(["task", "participant", "rep"])
    add("Per-repetition fatigue values:")
    add(rr.to_string(index=False))
    add()

    grp = rr.groupby(["task", "participant"])
    final_var = grp[phys_f].nunique(dropna=True)
    initial_var = grp[phys_i].nunique(dropna=True)

    add(f"Participant/task sequences with Physical Fatigue - Final varying across repetitions: "
        f"{int((final_var > 1).sum())}/{len(final_var)}")
    add(f"Participant/task sequences with Physical Fatigue - Initial varying across repetitions: "
        f"{int((initial_var > 1).sum())}/{len(initial_var)}")

    # Check transition logic: is Rep(k) initial approximately previous Rep(k-1) final?
    continuity = []
    for (task, pid), g in rr.groupby(["task", "participant"]):
        g = g.sort_values("rep")
        prev_f = None
        for _, r in g.iterrows():
            if prev_f is not None and pd.notna(prev_f) and pd.notna(r[phys_i]):
                continuity.append(abs(float(r[phys_i]) - float(prev_f)))
            prev_f = r[phys_f]
    if continuity:
        continuity = np.asarray(continuity, dtype=float)
        add(f"Rep-to-rep fatigue continuity | next initial - previous final |: "
            f"median={np.median(continuity):.4g}, max={np.max(continuity):.4g}, "
            f"exact_zero_fraction={(continuity == 0).mean():.3f}")
add()

# ------------------------------------------------------------------
add("4. PRIMARY CONTEXT ELIGIBILITY")
add("-" * 110)
add("Pre-task / low-cost candidates that can be evaluated as primary context:")
add("  - Physical Fatigue - Initial")
add("  - Mental Fatigue - Initial")
add("  - Task identity (derived from Composite / WireHarness folder)")
add("  - Repetition number (derived from Rep1...Rep5; treat as protocol/order context)")
add("  - Weights added (if prespecified before the repetition)")
add("  - Age / Weight / Height / Gender (worker baseline descriptors)")
add()
add("Variables that MUST NOT enter the primary context model when predicting Physical Fatigue - Final:")
add("  - Physical Fatigue - Final  [TARGET]")
add("  - Mental Fatigue - Final    [post-task / outcome-adjacent]")
add("  - Performance rating        [post-task / outcome-adjacent unless timing is proven otherwise]")
add()

# ------------------------------------------------------------------
add("5. SENSOR MODALITIES PRESENT")
add("-" * 110)
present_cols = set()
for path in manifest["file"]:
    pass
# The schema was already constrained by files actually read.
sample_csv = next((p for f in FOLDERS for p in (PM / f).rglob("*.csv")), None)
if sample_csv:
    h = list(pd.read_csv(sample_csv, nrows=0).columns)
    for c in PHYS_COLS + CHEST_ACC + ACC_COLS + GYRO_COLS:
        if c in h:
            present_cols.add(c)

add("Physiology / quality channels:")
add("  " + ", ".join([c for c in PHYS_COLS if c in present_cols]))
add("Chest acceleration:")
add("  " + ", ".join([c for c in CHEST_ACC if c in present_cols]))
add("Five-IMU acceleration channels:")
add("  " + ", ".join([c for c in ACC_COLS if c in present_cols]))
add("Five-IMU gyroscope channels:")
add("  " + ", ".join([c for c in GYRO_COLS if c in present_cols]))
add()

# ------------------------------------------------------------------
add("6. CRITICAL GYROSCOPE-DUPLICATION CHECK")
add("-" * 110)
gyro_df = pd.DataFrame(gyro_detail)
if gyro_df.empty:
    add("No gyroscope triplets available.")
else:
    summary = (
        gyro_df.groupby("imu")[["xy_equal_fraction", "xz_equal_fraction", "yz_equal_fraction"]]
        .agg(["mean", "median", "min", "max"])
    )
    add(summary.to_string())
    add()

    for pair in ["xy_equal_fraction", "xz_equal_fraction", "yz_equal_fraction"]:
        s = pd.to_numeric(gyro_df[pair], errors="coerce")
        add(f"{pair}: segments >=99% identical = {int((s >= 0.99).sum())}/{s.notna().sum()}")
    add()

    all_three = (
        (gyro_df["xy_equal_fraction"] >= 0.99) &
        (gyro_df["xz_equal_fraction"] >= 0.99) &
        (gyro_df["yz_equal_fraction"] >= 0.99)
    )
    add(f"IMU/segment triplets with gx, gy, gz all >=99% identical: "
        f"{int(all_three.sum())}/{len(gyro_df)}")
    add()
    add("Interpretation rule:")
    add("  If this proportion is high, do NOT treat the GitHub CSV gyroscope channels as valid independent axes.")
    add("  The original raw Zenodo files (or another verified raw source) must then be checked before any")
    add("  raw-signal Foundation Model training that uses gyroscope information.")
add()

# ------------------------------------------------------------------
add("7. SENSOR MISSING / SENTINEL CHECK")
add("-" * 110)
for c in ["sensor_missing_rate", "sensor_minus2_rate"]:
    if c in manifest.columns:
        s = pd.to_numeric(manifest[c], errors="coerce")
        add(f"{c}: mean={s.mean():.6f}, median={s.median():.6f}, max={s.max():.6f}")
add()

# ------------------------------------------------------------------
add("8. PUBLICATION-READINESS DECISION RULES")
add("-" * 110)
add("Current GitHub subset:")
add("  - Suitable for schema validation, feature engineering tests, leakage audits, and pilot pipeline runs.")
add("  - NOT sufficient by itself for the planned publication-level multi-dataset evidence if it contains only")
add("    the small public train/test participant subset.")
add()
add("Full Dataset A requirement:")
add("  - Full participant-level processed data are sufficient for classical/handcrafted incremental-value experiments.")
add("  - Full raw time-series data are required for the planned wearable foundation-model / raw-signal experiments.")
add("  - Participant identity must remain grouped in every split; no row/window-level random split is acceptable.")
add()
add("Primary modeling order once full data are available:")
add("  M0: context only")
add("  M1: handcrafted wearable only")
add("  M2: context + handcrafted wearable")
add("  M3: pretrained/foundation wearable representation")
add("  M4: context + foundation representation")
add("  M5: context-residualized domain-generalized wearable model")
add()
add("=" * 110)
add("END OF AUDIT")

TXT_OUT.write_text("\n".join(lines), encoding="utf-8")

print(f"Audit complete:")
print(f"  {TXT_OUT}")
print(f"  {CSV_OUT}")
print("")
print("Upload dataset_A_label_context_quality_audit.txt to ChatGPT.")
