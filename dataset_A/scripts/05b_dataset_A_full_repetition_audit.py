from pathlib import Path
from collections import Counter, defaultdict
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "dataset_A_northwestern_mxd" / "processed_full" / "MxD_Data_User_Study"
AUDIT = ROOT / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)

TXT_OUT = AUDIT / "dataset_A_full_repetition_audit.txt"
REP_OUT = AUDIT / "dataset_A_full_repetition_manifest.csv"
SESSION_OUT = AUDIT / "dataset_A_full_session_manifest.csv"
ANOM_OUT = AUDIT / "dataset_A_full_path_anomalies.csv"

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

SENSOR_COLS = [
    "HR_Processed", "HR", "HRV", "RR", "RRSQI", "ECGSQI", "ECG",
    "acc_X", "acc_Y", "acc_Z", "Temperature",
]
for i in range(1, 6):
    SENSOR_COLS += [
        f"IMU_{i}_ax_g_", f"IMU_{i}_ay_g_", f"IMU_{i}_az_g_",
        f"IMU_{i}_gx_dps_", f"IMU_{i}_gy_dps_", f"IMU_{i}_gz_dps_",
    ]

KEEP = set(["Time"] + META_COLS + SENSOR_COLS)

def extract_identity(text):
    """
    Accept P001C..., P001Z..., or malformed PA00Z...
    Returns participant_id, task_code, numeric_task_id, session_id.
    """
    s = str(text)

    # Standard
    m = re.search(r"(P\d{3})([CZ])(\d{3})S(\d{3})", s, flags=re.I)
    if m:
        return {
            "participant_id": m.group(1).upper(),
            "task_code": m.group(2).upper(),
            "task_numeric_id": m.group(3),
            "session_id": m.group(4),
            "identity_valid": True,
        }

    # Malformed/nonstandard, e.g. PA00Z001S003
    m = re.search(r"(P[A-Z0-9]{3})([CZ])(\d{3})S(\d{3})", s, flags=re.I)
    if m:
        return {
            "participant_id": m.group(1).upper(),
            "task_code": m.group(2).upper(),
            "task_numeric_id": m.group(3),
            "session_id": m.group(4),
            "identity_valid": False,
        }

    return {
        "participant_id": None,
        "task_code": None,
        "task_numeric_id": None,
        "session_id": None,
        "identity_valid": False,
    }

def task_from_directory(path):
    parts = [x.lower() for x in Path(path).parts]
    if "composite" in parts:
        return "Composite"
    if "ziptie" in parts:
        return "WireHarness"
    return None

def task_code_expected(task):
    return "C" if task == "Composite" else ("Z" if task == "WireHarness" else None)

def segment_from_name(name):
    stem = Path(name).stem
    if stem.startswith("Initial_Rest_"):
        return "Initial_Rest", np.nan
    if stem.startswith("Final_Rest_"):
        return "Final_Rest", np.nan
    m = re.match(r"Rep(\d+)_", stem, flags=re.I)
    if m:
        return f"Rep{int(m.group(1))}", int(m.group(1))
    return stem, np.nan

def first_num(df, col):
    if col not in df.columns:
        return np.nan
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s.iloc[0] if len(s) else np.nan

def nunique_num(df, col):
    if col not in df.columns:
        return np.nan
    return pd.to_numeric(df[col], errors="coerce").dropna().nunique()

def equal_fraction(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = a.notna() & b.notna()
    if not mask.any():
        return np.nan
    aa = a[mask].to_numpy()
    bb = b[mask].to_numpy()
    return float(np.isclose(aa, bb, rtol=0.0, atol=1e-12).mean())

def finite_std(s):
    s = pd.to_numeric(s, errors="coerce")
    x = s[np.isfinite(s)]
    return float(x.std(ddof=0)) if len(x) else np.nan

def duration_summary(df):
    if "Time" not in df.columns:
        return np.nan, np.nan, np.nan
    t = pd.to_numeric(df["Time"], errors="coerce").dropna()
    if len(t) < 2:
        return np.nan, np.nan, np.nan
    dt = t.diff().dropna()
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(t.min()), float(t.max()), float(dt.median()) if len(dt) else np.nan

all_csv = sorted(BASE.rglob("*.csv"))

rows = []
anomalies = []

for path in all_csv:
    rel = path.relative_to(BASE)
    task_dir = task_from_directory(rel)
    seg, rep = segment_from_name(path.name)

    folder_id = extract_identity(path.parent.name)
    file_id = extract_identity(path.name)

    # Prefer filename identity if standard; otherwise folder identity.
    chosen = file_id if file_id["participant_id"] else folder_id

    expected_code = task_code_expected(task_dir)
    if folder_id["task_code"] and expected_code and folder_id["task_code"] != expected_code:
        anomalies.append({
            "relative_path": str(rel),
            "type": "folder_task_code_mismatch",
            "detail": f"directory task={task_dir}, folder code={folder_id['task_code']}",
        })
    if file_id["task_code"] and expected_code and file_id["task_code"] != expected_code:
        anomalies.append({
            "relative_path": str(rel),
            "type": "filename_task_code_mismatch",
            "detail": f"directory task={task_dir}, filename code={file_id['task_code']}",
        })
    if folder_id["participant_id"] and file_id["participant_id"] and folder_id["participant_id"] != file_id["participant_id"]:
        anomalies.append({
            "relative_path": str(rel),
            "type": "participant_folder_filename_mismatch",
            "detail": f"folder={folder_id['participant_id']} filename={file_id['participant_id']}",
        })
    if folder_id["session_id"] and file_id["session_id"] and folder_id["session_id"] != file_id["session_id"]:
        anomalies.append({
            "relative_path": str(rel),
            "type": "session_folder_filename_mismatch",
            "detail": f"folder S{folder_id['session_id']} filename S{file_id['session_id']}",
        })
    if chosen["participant_id"] is None:
        anomalies.append({
            "relative_path": str(rel),
            "type": "participant_unparsed",
            "detail": "No participant identifier could be parsed",
        })
    elif not chosen["identity_valid"]:
        anomalies.append({
            "relative_path": str(rel),
            "type": "nonstandard_participant_id",
            "detail": chosen["participant_id"],
        })

    try:
        header = list(pd.read_csv(path, nrows=0).columns)
        usecols = [c for c in header if c in KEEP]
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
    except Exception as e:
        anomalies.append({
            "relative_path": str(rel),
            "type": "read_error",
            "detail": f"{type(e).__name__}: {e}",
        })
        continue

    tmin, tmax, dtmed = duration_summary(df)

    r = {
        "task": task_dir,
        "participant_id": chosen["participant_id"],
        "participant_id_standard": chosen["identity_valid"],
        "folder_name": path.parent.name,
        "filename": path.name,
        "relative_path": str(rel),
        "segment": seg,
        "rep": rep,
        "n_rows": len(df),
        "time_min": tmin,
        "time_max": tmax,
        "median_time_step": dtmed,
    }

    for c in META_COLS:
        key = re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_").lower()
        r[key] = first_num(df, c)
        r[key + "_nunique_within_file"] = nunique_num(df, c)

    # Basic sensor quality
    present_sensor = [c for c in SENSOR_COLS if c in df.columns]
    if present_sensor:
        num = df[present_sensor].apply(pd.to_numeric, errors="coerce")
        arr = num.to_numpy(dtype=float)
        finite = np.isfinite(arr)
        r["sensor_missing_rate"] = float((~finite).sum() / arr.size) if arr.size else np.nan
        r["sensor_minus2_rate"] = float(((arr == -2) & finite).sum() / max(finite.sum(), 1))
    else:
        r["sensor_missing_rate"] = np.nan
        r["sensor_minus2_rate"] = np.nan

    # Full gyro duplication check
    for i in range(1, 6):
        gx = f"IMU_{i}_gx_dps_"
        gy = f"IMU_{i}_gy_dps_"
        gz = f"IMU_{i}_gz_dps_"
        if all(c in df.columns for c in [gx, gy, gz]):
            r[f"imu{i}_gyro_xy_equal"] = equal_fraction(df[gx], df[gy])
            r[f"imu{i}_gyro_xz_equal"] = equal_fraction(df[gx], df[gz])
            r[f"imu{i}_gyro_yz_equal"] = equal_fraction(df[gy], df[gz])

    # Acceleration-axis duplication check
    for i in range(1, 6):
        ax = f"IMU_{i}_ax_g_"
        ay = f"IMU_{i}_ay_g_"
        az = f"IMU_{i}_az_g_"
        if all(c in df.columns for c in [ax, ay, az]):
            r[f"imu{i}_acc_xy_equal"] = equal_fraction(df[ax], df[ay])
            r[f"imu{i}_acc_xz_equal"] = equal_fraction(df[ax], df[az])
            r[f"imu{i}_acc_yz_equal"] = equal_fraction(df[ay], df[az])
            r[f"imu{i}_acc_x_std"] = finite_std(df[ax])
            r[f"imu{i}_acc_y_std"] = finite_std(df[ay])
            r[f"imu{i}_acc_z_std"] = finite_std(df[az])

    rows.append(r)

df_all = pd.DataFrame(rows)
pd.DataFrame(anomalies).to_csv(ANOM_OUT, index=False)

# Session summary
session_rows = []
for (task, pid, folder), g in df_all.groupby(["task", "participant_id", "folder_name"], dropna=False):
    reps = sorted(pd.to_numeric(g["rep"], errors="coerce").dropna().astype(int).unique().tolist())
    session_rows.append({
        "task": task,
        "participant_id": pid,
        "folder_name": folder,
        "participant_id_standard": bool(g["participant_id_standard"].all()),
        "n_csv_files": len(g),
        "n_repetition_files": int(g["rep"].notna().sum()),
        "reps_present": ",".join(map(str, reps)),
        "has_initial_rest": bool((g["segment"] == "Initial_Rest").any()),
        "has_final_rest": bool((g["segment"] == "Final_Rest").any()),
    })
session_df = pd.DataFrame(session_rows)
session_df.to_csv(SESSION_OUT, index=False)

# Main repetition manifest only
rep_df = df_all[df_all["rep"].notna()].copy()
rep_df = rep_df.sort_values(["task", "participant_id", "rep"])
rep_df.to_csv(REP_OUT, index=False)

# ------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------
lines = []
def add(s=""):
    lines.append(str(s))

add("DATASET A — CORRECTED FULL REPETITION-LEVEL AUDIT")
add("=" * 118)
add(f"Root: {BASE}")
add(f"CSV files audited: {len(df_all)}")
add(f"Repetition files: {len(rep_df)}")
add(f"Path anomalies logged: {len(anomalies)}")
add()

add("1. PARTICIPANT / TASK STRUCTURE")
add("-" * 118)

for task in sorted(session_df["task"].dropna().unique()):
    x = session_df[session_df["task"] == task]
    pids = sorted(x["participant_id"].dropna().unique())
    standard = sorted(x.loc[x["participant_id_standard"], "participant_id"].dropna().unique())
    nonstandard = sorted(x.loc[~x["participant_id_standard"], "participant_id"].dropna().unique())
    add(f"{task}: participant/session folders={len(x)}, unique IDs={len(pids)}, standard IDs={len(standard)}")
    add("  Standard IDs: " + ", ".join(standard))
    if nonstandard:
        add("  Nonstandard IDs: " + ", ".join(nonstandard))

comp = set(session_df.loc[(session_df.task=="Composite") & session_df.participant_id_standard, "participant_id"])
wire = set(session_df.loc[(session_df.task=="WireHarness") & session_df.participant_id_standard, "participant_id"])

add()
add(f"Standard participant IDs in union: {len(comp | wire)}")
add(f"Standard IDs completing BOTH tasks: {len(comp & wire)}")
add("  " + ", ".join(sorted(comp & wire)))
add(f"Composite only: {len(comp-wire)}")
add("  " + ", ".join(sorted(comp-wire)))
add(f"WireHarness only: {len(wire-comp)}")
add("  " + ", ".join(sorted(wire-comp)))
nonstd_all = sorted(session_df.loc[~session_df.participant_id_standard, "participant_id"].dropna().unique())
add(f"Nonstandard participant IDs requiring quarantine/provenance check: {len(nonstd_all)}")
add("  " + ", ".join(nonstd_all))
add()

add("2. REPETITION COMPLETENESS")
add("-" * 118)
rep_counts = (
    rep_df.groupby(["task","participant_id"])
    .agg(n_rep_files=("rep","count"),
         n_unique_reps=("rep","nunique"),
         min_rep=("rep","min"),
         max_rep=("rep","max"))
    .reset_index()
)
add(rep_counts.to_string(index=False))
add()
complete = rep_counts[
    (rep_counts.n_unique_reps == 5) &
    (rep_counts.min_rep == 1) &
    (rep_counts.max_rep == 5)
]
add(f"Complete 5-repetition participant-task sequences: {len(complete)}/{len(rep_counts)}")
incomplete = rep_counts.drop(complete.index)
if len(incomplete):
    add("Incomplete sequences:")
    add(incomplete.to_string(index=False))
add()

add("3. FATIGUE LABEL INTEGRITY")
add("-" * 118)
for c in [
    "physical_fatigue_initial",
    "physical_fatigue_final",
    "mental_fatigue_initial",
    "mental_fatigue_final",
]:
    s = pd.to_numeric(rep_df[c], errors="coerce")
    add(f"{c}: n={s.notna().sum()}, unique={s.nunique(dropna=True)}, "
        f"min={s.min()}, median={s.median()}, max={s.max()}, missing={s.isna().sum()}")

# Within-file constancy
for c in [
    "physical_fatigue_initial_nunique_within_file",
    "physical_fatigue_final_nunique_within_file",
    "mental_fatigue_initial_nunique_within_file",
    "mental_fatigue_final_nunique_within_file",
]:
    s = pd.to_numeric(rep_df[c], errors="coerce")
    add(f"{c}: files with >1 unique value = {int((s > 1).sum())}")
add()

# Sequence continuity
continuity = []
vary_final = 0
seq_count = 0
for (task,pid), g in rep_df.groupby(["task","participant_id"]):
    g = g.sort_values("rep")
    seq_count += 1
    if g["physical_fatigue_final"].nunique(dropna=True) > 1:
        vary_final += 1
    prev = None
    for _, rr in g.iterrows():
        if prev is not None and pd.notna(prev) and pd.notna(rr["physical_fatigue_initial"]):
            continuity.append(abs(float(rr["physical_fatigue_initial"]) - float(prev)))
        prev = rr["physical_fatigue_final"]

add(f"Sequences with varying Physical Fatigue - Final: {vary_final}/{seq_count}")
if continuity:
    a = np.asarray(continuity)
    add(f"Next initial vs previous final absolute difference: "
        f"median={np.median(a):.4g}, max={np.max(a):.4g}, exact-zero fraction={(a==0).mean():.4f}")
add()

add("4. DEMOGRAPHIC CONSISTENCY ACROSS TASKS")
add("-" * 118)
demo_cols = ["age","weight","height","gender"]
for pid in sorted(comp & wire):
    gc = rep_df[(rep_df.task=="Composite") & (rep_df.participant_id==pid)]
    gw = rep_df[(rep_df.task=="WireHarness") & (rep_df.participant_id==pid)]
    diffs = []
    for c in demo_cols:
        vc = pd.to_numeric(gc[c], errors="coerce").dropna()
        vw = pd.to_numeric(gw[c], errors="coerce").dropna()
        mc = vc.median() if len(vc) else np.nan
        mw = vw.median() if len(vw) else np.nan
        if pd.notna(mc) and pd.notna(mw) and not np.isclose(mc, mw, atol=1e-9, rtol=0):
            diffs.append(f"{c}:{mc}->{mw}")
    if diffs:
        add(f"{pid}: MISMATCH | " + "; ".join(diffs))
add("If no participant lines appear above, paired demographic fields are exactly consistent.")
add()

add("5. FULL-ARCHIVE GYROSCOPE DUPLICATION")
add("-" * 118)
gyro_fields = [c for c in rep_df.columns if re.match(r"imu\d+_gyro_[xyz][yz]?_equal", c)]
# Use specific columns to avoid regex ambiguity
for i in range(1,6):
    cols = [f"imu{i}_gyro_xy_equal", f"imu{i}_gyro_xz_equal", f"imu{i}_gyro_yz_equal"]
    if all(c in rep_df.columns for c in cols):
        for c in cols:
            s = pd.to_numeric(rep_df[c], errors="coerce")
            add(f"{c}: mean={s.mean():.6f}, median={s.median():.6f}, "
                f">=0.99={int((s>=0.99).sum())}/{s.notna().sum()}")

triplet_all = []
for _, r in rep_df.iterrows():
    for i in range(1,6):
        cols = [f"imu{i}_gyro_xy_equal", f"imu{i}_gyro_xz_equal", f"imu{i}_gyro_yz_equal"]
        vals = [r.get(c, np.nan) for c in cols]
        if all(pd.notna(v) for v in vals):
            triplet_all.append(all(float(v) >= 0.99 for v in vals))
if triplet_all:
    add(f"Gyro IMU/repetition triplets with gx,gy,gz all >=99% identical: "
        f"{sum(triplet_all)}/{len(triplet_all)} ({np.mean(triplet_all):.4f})")
add()

add("6. ACCELEROMETER AXIS SANITY CHECK")
add("-" * 118)
acc_bad = 0
acc_total = 0
for _, r in rep_df.iterrows():
    for i in range(1,6):
        cols = [f"imu{i}_acc_xy_equal", f"imu{i}_acc_xz_equal", f"imu{i}_acc_yz_equal"]
        vals = [r.get(c, np.nan) for c in cols]
        for v in vals:
            if pd.notna(v):
                acc_total += 1
                acc_bad += int(float(v) >= 0.99)
add(f"IMU acceleration axis-pairs >=99% identical: {acc_bad}/{acc_total}")
add("A near-zero count supports retaining acceleration channels for modeling.")
add()

add("7. PATH / SESSION ANOMALIES")
add("-" * 118)
anom_df = pd.DataFrame(anomalies)
if len(anom_df):
    add(anom_df["type"].value_counts().to_string())
    add()
    for typ, g in anom_df.groupby("type"):
        add(f"[{typ}] examples:")
        for _, rr in g.head(10).iterrows():
            add(f"  {rr['relative_path']} | {rr['detail']}")
else:
    add("No anomalies detected.")
add()

add("8. DATASET-A MODELING DECISION")
add("-" * 118)
add("Primary analysis unit: one repetition (Rep1-Rep5), NOT sensor rows.")
add("Primary outcome: Physical Fatigue - Final.")
add("Primary context candidates:")
add("  Physical Fatigue - Initial; Mental Fatigue - Initial; Task; Repetition number;")
add("  Weights added; Age; Weight; Height; Gender.")
add("Excluded from primary predictors:")
add("  Physical Fatigue - Final [target]; Mental Fatigue - Final [post-task];")
add("  Performance rating [post-task/outcome-adjacent unless timing is proven].")
add("Wearable streams retained provisionally:")
add("  HR/HRV/ECG/Temperature; chest acceleration; 5-IMU acceleration.")
add("Gyroscope:")
add("  QUARANTINE if the full-archive duplication result confirms gx=gy=gz.")
add()
add("Important interpretation:")
add("  These CSVs are segmented/resampled time-series, not precomputed engineered feature tables.")
add("  Therefore they support BOTH:")
add("    Phase I  - leakage-controlled handcrafted feature extraction + incremental-value benchmarking")
add("    Phase II - GPU time-series representation / foundation-model experiments")
add("  They should not be described as untouched/raw sensor files unless provenance to the original raw archive is verified.")
add()
add("=" * 118)
add("END OF AUDIT")

TXT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("Corrected full repetition-level audit complete:")
print(f"  {TXT_OUT}")
print(f"  {REP_OUT}")
print(f"  {SESSION_OUT}")
print(f"  {ANOM_OUT}")
print()
print("Upload dataset_A_full_repetition_audit.txt to ChatGPT.")
