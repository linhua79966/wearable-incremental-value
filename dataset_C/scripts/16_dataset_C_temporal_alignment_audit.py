from pathlib import Path
from collections import Counter
import math
import re
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DSROOT = ROOT / "data" / "dataset_C_shoulder_rotation" / "raw" / "WSD4FEDSRM"
SENSOR_ROOT = DSROOT / "EMG, IMU, and PPG data"
BORG = DSROOT / "Borg data" / "borg_data.csv"
AUDIT = ROOT / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)

TXT_OUT = AUDIT / "dataset_C_temporal_alignment_audit.txt"
TRIAL_OUT = AUDIT / "dataset_C_temporal_alignment_trials.csv"
BORG_LONG_OUT = AUDIT / "dataset_C_borg_long_corrected.csv"

TASK_MAP = {
    "task1_35i": "30-40_ internal rotation",
    "task2_45i": "40-50_ internal rotation",
    "task3_55i": "50-60_ internal rotation",
    "task4_35e": "30-40_ external rotation",
    "task5_45e": "40-50_ external rotation",
    "task6_55e": "50-60_ external rotation",
}

def add(lines, s=""):
    lines.append(str(s))

def norm_col(c):
    return re.sub(r"\s+", "", str(c).strip().lower())

def count_csv_rows_fast(path):
    """
    Count data rows without parsing the whole CSV into pandas.
    Assumes one header line, which is true for these sensor files.
    """
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            n += block.count(b"\n")
    # If final line has no newline, count it.
    try:
        with open(path, "rb") as f:
            if path.stat().st_size > 0:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    n += 1
    except OSError:
        pass
    return max(n - 1, 0)

def parse_subject_num(s):
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else None

# -------------------------------------------------------------------------
# Borg: correct blank continuation rows by forward-fill within original order
# -------------------------------------------------------------------------
borg = pd.read_csv(BORG)

lookup = {norm_col(c): c for c in borg.columns}
subject_col = lookup["subject"]
task_col = lookup["task_order"]
before_col = lookup["before_task"]
end_col = lookup["end_of_trial"]
duration_col = lookup["length_of_trial_(sec)"]

raw_missing_subject = int(borg[subject_col].isna().sum())
borg[subject_col] = borg[subject_col].ffill()

# Identify regular 10-s Borg columns despite whitespace in headers.
time_col_map = {}
for c in borg.columns:
    z = norm_col(c)
    m = re.fullmatch(r"(\d+)_sec", z)
    if m:
        time_col_map[int(m.group(1))] = c

regular_times = sorted(time_col_map)

# Validate corrected subject-task pairs.
pairs = borg[[subject_col, task_col]].copy()
pair_dupes = int(pairs.duplicated().sum())
unique_subjects = borg[subject_col].nunique(dropna=True)
unique_tasks = borg[task_col].nunique(dropna=True)
unique_pairs = len(pairs.drop_duplicates())

# -------------------------------------------------------------------------
# Borg longitudinal audit + exact-grid candidate schedule
# -------------------------------------------------------------------------
borg_long_rows = []
borg_trial_qc = {}

for ridx, br in borg.iterrows():
    subject = str(br[subject_col]).strip()
    subject_num = parse_subject_num(subject)
    task = str(br[task_col]).strip()
    duration = pd.to_numeric(pd.Series([br[duration_col]]), errors="coerce").iloc[0]
    before = pd.to_numeric(pd.Series([br[before_col]]), errors="coerce").iloc[0]
    end = pd.to_numeric(pd.Series([br[end_col]]), errors="coerce").iloc[0]

    obs_regular = []
    for t in regular_times:
        v = pd.to_numeric(
            pd.Series([br[time_col_map[t]]]),
            errors="coerce"
        ).iloc[0]
        if pd.notna(v):
            obs_regular.append((t, float(v)))

    obs_times = [t for t, _ in obs_regular]
    obs_vals = [v for _, v in obs_regular]

    # Expected scheduled regular Borg times are multiples of 10 strictly
    # before end-of-trial duration. Example: duration 60 -> 10..50.
    if pd.notna(duration) and duration > 0:
        expected_regular = [
            t for t in regular_times
            if t < float(duration) - 1e-9
        ]
    else:
        expected_regular = []

    expected_present = all(t in obs_times for t in expected_regular)
    unexpected_after_end = [t for t in obs_times if pd.notna(duration) and t >= float(duration) - 1e-9]

    internal_gaps = []
    if obs_times:
        for t in range(10, max(obs_times) + 1, 10):
            if t not in obs_times:
                internal_gaps.append(t)

    seq = []
    if pd.notna(before):
        seq.append((0.0, float(before), "before_task"))
    seq += [(float(t), float(v), "regular") for t, v in obs_regular]
    if pd.notna(duration) and pd.notna(end):
        seq.append((float(duration), float(end), "end_of_trial"))

    seq = sorted(seq, key=lambda x: x[0])

    monotonic_violations = 0
    for a, b in zip(seq[:-1], seq[1:]):
        if b[1] < a[1] - 1e-12:
            monotonic_violations += 1

    exact10_transition_targets = []
    # Main candidate uses current Borg at t, prior wearable window [t-10,t],
    # to predict Borg at t+10. Thus t must be >=10.
    values_by_time = {float(t): float(v) for t, v, _ in seq}
    for t in sorted(values_by_time):
        if t < 10:
            continue
        target_t = t + 10.0
        if target_t in values_by_time:
            exact10_transition_targets.append((t, target_t))

    borg_trial_qc[(subject_num, task)] = {
        "duration_sec": float(duration) if pd.notna(duration) else np.nan,
        "before_borg": float(before) if pd.notna(before) else np.nan,
        "end_borg": float(end) if pd.notna(end) else np.nan,
        "n_regular_borg": len(obs_regular),
        "max_regular_time": max(obs_times) if obs_times else np.nan,
        "expected_regular_count": len(expected_regular),
        "expected_regular_complete": expected_present,
        "unexpected_regular_at_or_after_end": len(unexpected_after_end),
        "internal_regular_gaps": len(internal_gaps),
        "borg_monotonic_violations": monotonic_violations,
        "n_exact10_forecast_targets": len(exact10_transition_targets),
    }

    for t, v, source in seq:
        borg_long_rows.append({
            "subject": subject,
            "subject_num": subject_num,
            "task_label": task,
            "time_sec": t,
            "borg_rpe": v,
            "source": source,
            "trial_duration_sec": duration,
        })

borg_long = pd.DataFrame(borg_long_rows)
borg_long.to_csv(BORG_LONG_OUT, index=False)

# -------------------------------------------------------------------------
# Sensor row-count audit for all 204 trials
# -------------------------------------------------------------------------
trial_rows = []
t0 = time.time()

for i, br in borg.iterrows():
    subject = str(br[subject_col]).strip()
    subject_num = parse_subject_num(subject)
    task = str(br[task_col]).strip()
    folder = TASK_MAP.get(task)
    duration = float(
        pd.to_numeric(pd.Series([br[duration_col]]), errors="coerce").iloc[0]
    )

    trial_dir = SENSOR_ROOT / folder / f"Subject {subject_num}"

    emg_dir = trial_dir / "EMG data"
    imu_dir = trial_dir / "IMU data"
    ppg_file = trial_dir / "PPG data" / "ppg.csv"

    emg_files = sorted(emg_dir.glob("*.csv")) if emg_dir.exists() else []
    imu_files = sorted(imu_dir.rglob("*.csv")) if imu_dir.exists() else []

    emg_counts = {}
    for fp in emg_files:
        emg_counts[fp.name] = count_csv_rows_fast(fp)

    imu_counts = {}
    for fp in imu_files:
        imu_counts[str(fp.relative_to(imu_dir))] = count_csv_rows_fast(fp)

    ppg_count = count_csv_rows_fast(ppg_file) if ppg_file.exists() else np.nan

    # Known published corruption: exact Shoulder IMU files for subject 3, task1_35i.
    known_corrupt = (subject_num == 3 and task == "task1_35i")
    corrupt_rel = {
        r"Shoulder\acc_shoulder.csv",
        r"Shoulder\gyr_shoulder.csv",
        r"Shoulder\mag_shoulder.csv",
    }

    imu_counts_clean = {
        k: v
        for k, v in imu_counts.items()
        if not (known_corrupt and k in corrupt_rel)
    }

    def minmax(vals):
        vals = list(vals)
        if not vals:
            return np.nan, np.nan
        return float(min(vals)), float(max(vals))

    emg_min, emg_max = minmax(emg_counts.values())
    imu_min, imu_max = minmax(imu_counts_clean.values())

    q = borg_trial_qc[(subject_num, task)]

    row = {
        "subject": subject,
        "subject_num": subject_num,
        "task_label": task,
        "task_folder": folder,
        "trial_dir_exists": trial_dir.exists(),
        "duration_sec": duration,

        "n_emg_files": len(emg_files),
        "emg_rows_min": emg_min,
        "emg_rows_max": emg_max,
        "emg_all_equal_rows": bool(
            len(emg_counts) > 0 and len(set(emg_counts.values())) == 1
        ),

        "n_imu_files_total": len(imu_files),
        "n_imu_files_after_known_corrupt_quarantine": len(imu_counts_clean),
        "imu_rows_min": imu_min,
        "imu_rows_max": imu_max,
        "imu_all_equal_rows_after_quarantine": bool(
            len(imu_counts_clean) > 0 and len(set(imu_counts_clean.values())) == 1
        ),

        "ppg_file_exists": ppg_file.exists(),
        "ppg_rows": ppg_count,

        "known_corrupt_shoulder_imu_trial": known_corrupt,

        **q,
    }

    trial_rows.append(row)

    if (i + 1) % 25 == 0 or i + 1 == len(borg):
        print(f"Audited sensor lengths {i+1}/{len(borg)} trials")

trial_df = pd.DataFrame(trial_rows)

# -------------------------------------------------------------------------
# Infer effective sampling rates from row-count / Borg-duration ratios.
# Use minima/maxima consistency first; robust median across trials.
# -------------------------------------------------------------------------
for prefix, rows_col in [
    ("emg", "emg_rows_min"),
    ("imu", "imu_rows_min"),
    ("ppg", "ppg_rows"),
]:
    ratio = pd.to_numeric(trial_df[rows_col], errors="coerce") / pd.to_numeric(
        trial_df["duration_sec"], errors="coerce"
    )
    trial_df[f"{prefix}_rows_per_sec"] = ratio

def robust_rate(col):
    s = pd.to_numeric(trial_df[col], errors="coerce")
    s = s[np.isfinite(s) & (s > 0)]
    if not len(s):
        return np.nan
    return float(np.median(s))

emg_rate_est = robust_rate("emg_rows_per_sec")
imu_rate_est = robust_rate("imu_rows_per_sec")
ppg_rate_est = robust_rate("ppg_rows_per_sec")

# nearest integer is useful if data are exactly at standard rates.
emg_rate_int = int(round(emg_rate_est)) if np.isfinite(emg_rate_est) else None
imu_rate_int = int(round(imu_rate_est)) if np.isfinite(imu_rate_est) else None
ppg_rate_int = int(round(ppg_rate_est)) if np.isfinite(ppg_rate_est) else None

for prefix, rows_col, rate in [
    ("emg", "emg_rows_min", emg_rate_int),
    ("imu", "imu_rows_min", imu_rate_int),
    ("ppg", "ppg_rows", ppg_rate_int),
]:
    if rate:
        inferred_duration = pd.to_numeric(
            trial_df[rows_col], errors="coerce"
        ) / rate
        trial_df[f"{prefix}_duration_est_sec"] = inferred_duration
        trial_df[f"{prefix}_duration_error_sec"] = (
            inferred_duration - trial_df["duration_sec"]
        )

trial_df.to_csv(TRIAL_OUT, index=False)

# -------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------
lines = []
add(lines, "DATASET C — SUBJECT-FILLED TEMPORAL ALIGNMENT AUDIT")
add(lines, "=" * 118)
add(lines, f"Borg rows: {len(borg)}")
add(lines, f"Originally blank subject cells: {raw_missing_subject}")
add(lines, f"Unique subjects after forward-fill: {unique_subjects}")
add(lines, f"Unique tasks: {unique_tasks}")
add(lines, f"Unique subject-task pairs after forward-fill: {unique_pairs}")
add(lines, f"Duplicate subject-task pairs after forward-fill: {pair_dupes}")
add(lines)

add(lines, "1. BORG LONGITUDINAL STRUCTURE")
add(lines, "-" * 118)
add(lines, f"Regular Borg grid columns: {len(regular_times)}")
add(lines, "  " + ", ".join(map(str, regular_times)))
add(lines, f"Corrected Borg longitudinal rows (before + regular + end): {len(borg_long)}")
add(lines)

for c in [
    "expected_regular_complete",
    "unexpected_regular_at_or_after_end",
    "internal_regular_gaps",
    "borg_monotonic_violations",
]:
    if c == "expected_regular_complete":
        add(lines, f"Trials with all expected regular Borg points present: "
                   f"{int(trial_df[c].sum())}/{len(trial_df)}")
    else:
        add(lines, f"Trials with {c} > 0: {int((trial_df[c] > 0).sum())}/{len(trial_df)}")
add(lines)

add(lines, f"Total exact +10 s forecast targets available from Borg schedule: "
           f"{int(trial_df['n_exact10_forecast_targets'].sum())}")
add(lines, f"Per-trial exact +10 s targets: min={trial_df['n_exact10_forecast_targets'].min()}, "
           f"median={trial_df['n_exact10_forecast_targets'].median()}, "
           f"max={trial_df['n_exact10_forecast_targets'].max()}")
add(lines)

add(lines, "2. TRIAL DIRECTORY / MODALITY COMPLETENESS")
add(lines, "-" * 118)
add(lines, f"Trial directories found: {int(trial_df['trial_dir_exists'].sum())}/{len(trial_df)}")
add(lines, f"Trials with 6 EMG files: {int((trial_df['n_emg_files']==6).sum())}/{len(trial_df)}")
add(lines, f"Trials with 18 IMU files before corruption quarantine: "
           f"{int((trial_df['n_imu_files_total']==18).sum())}/{len(trial_df)}")
add(lines, f"Trials with PPG file: {int(trial_df['ppg_file_exists'].sum())}/{len(trial_df)}")
add(lines, f"Known corrupted Shoulder-IMU trial flags: "
           f"{int(trial_df['known_corrupt_shoulder_imu_trial'].sum())}")
add(lines)

add(lines, "3. WITHIN-TRIAL SENSOR LENGTH CONSISTENCY")
add(lines, "-" * 118)
add(lines, f"EMG: trials where all 6 channels have identical rows: "
           f"{int(trial_df['emg_all_equal_rows'].sum())}/{len(trial_df)}")
add(lines, f"IMU: trials where all retained IMU files have identical rows: "
           f"{int(trial_df['imu_all_equal_rows_after_quarantine'].sum())}/{len(trial_df)}")
add(lines)

add(lines, "4. EFFECTIVE SAMPLING-RATE ESTIMATES")
add(lines, "-" * 118)
for label, col, med, rounded in [
    ("EMG", "emg_rows_per_sec", emg_rate_est, emg_rate_int),
    ("IMU", "imu_rows_per_sec", imu_rate_est, imu_rate_int),
    ("PPG", "ppg_rows_per_sec", ppg_rate_est, ppg_rate_int),
]:
    s = pd.to_numeric(trial_df[col], errors="coerce")
    add(lines, f"{label}: row-count / Borg-duration ratio "
               f"min={s.min():.4f}, median={med:.4f}, max={s.max():.4f}; "
               f"nearest integer rate={rounded} Hz")
add(lines)

add(lines, "5. SENSOR-DURATION AGREEMENT USING INFERRED INTEGER RATES")
add(lines, "-" * 118)
for prefix, rate in [
    ("emg", emg_rate_int),
    ("imu", imu_rate_int),
    ("ppg", ppg_rate_int),
]:
    c = f"{prefix}_duration_error_sec"
    if c not in trial_df.columns:
        continue
    s = pd.to_numeric(trial_df[c], errors="coerce").dropna()
    add(lines, f"{prefix.upper()} @ {rate} Hz: duration error (sensor-derived - Borg)")
    add(lines, f"  min={s.min():.4f}s, median={s.median():.4f}s, max={s.max():.4f}s")
    add(lines, f"  |error| <=0.5 s: {int((s.abs() <= 0.5).sum())}/{len(s)}")
    add(lines, f"  |error| <=1.0 s: {int((s.abs() <= 1.0).sum())}/{len(s)}")
    add(lines, f"  |error| >2.0 s: {int((s.abs() > 2.0).sum())}/{len(s)}")
add(lines)

add(lines, "6. LARGEST ALIGNMENT DEVIATIONS")
add(lines, "-" * 118)
for prefix in ["emg","imu","ppg"]:
    c = f"{prefix}_duration_error_sec"
    if c not in trial_df.columns:
        continue
    tmp = trial_df[
        ["subject","task_label","duration_sec",c]
    ].copy()
    tmp["abs_error"] = tmp[c].abs()
    tmp = tmp.sort_values("abs_error", ascending=False).head(15)
    add(lines, f"[{prefix.upper()}]")
    add(lines, tmp.to_string(index=False))
add(lines)

add(lines, "7. PROPOSED MAIN FORECASTING UNIT — NOT YET TRAINED")
add(lines, "-" * 118)
add(lines, "Main scientific target if alignment is acceptable:")
add(lines, "  Borg(t+10 s) predicted at time t.")
add(lines)
add(lines, "Predictors available at time t:")
add(lines, "  Context: Borg(t), elapsed time t, task direction/load, demographics.")
add(lines, "  Wearable: sensor window [t-10 s, t] only.")
add(lines)
add(lines, "Therefore:")
add(lines, "  sensor data from (t, t+10] are NEVER used to predict Borg(t+10).")
add(lines, "  before_task at t=0 cannot form a main sample because no pre-task wearable window is available.")
add(lines, "  exact +10-s Borg transitions are preferred for the primary analysis.")
add(lines, "  end_of_trial may be included only when it lies exactly on the +10-s grid; otherwise it is secondary.")
add(lines)
add(lines, "Known corruption handling:")
add(lines, "  For Subject 3 / task1_35i, quarantine Shoulder IMU acc/gyr/mag only.")
add(lines, "  Do not discard that entire subject, trial, EMG, PPG, or non-Shoulder IMU modalities.")
add(lines)

add(lines, "8. NEXT DECISION RULE")
add(lines, "-" * 118)
add(lines, "If most EMG/IMU durations match Borg duration closely, build the handcrafted")
add(lines, "10-s-window benchmark first using participant-grouped validation.")
add(lines)
add(lines, "PPG should enter the primary multimodal model only if its duration/alignment audit")
add(lines, "is comparably stable. Otherwise retain PPG as a separate sensitivity/modality analysis.")
add(lines)
add(lines, "GPU/raw-signal representation learning begins only after the leakage-controlled")
add(lines, "handcrafted benchmark is established.")
add(lines)

add(lines, "=" * 118)
add(lines, "END OF AUDIT")

AUDIT_OUT = TXT_OUT
AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("")
print("Dataset C temporal alignment audit complete.")
print(f"  {TXT_OUT}")
print(f"  {TRIAL_OUT}")
print(f"  {BORG_LONG_OUT}")
print(f"Elapsed minutes: {(time.time()-t0)/60:.2f}")
print("")
print("Upload dataset_C_temporal_alignment_audit.txt to ChatGPT.")
