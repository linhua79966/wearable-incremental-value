from pathlib import Path
from collections import Counter, defaultdict
import re
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "dataset_C_shoulder_rotation" / "raw"
AUDIT = ROOT / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)

TXT_OUT = AUDIT / "dataset_C_corrected_structure_audit.txt"
TRIAL_OUT = AUDIT / "dataset_C_trial_inventory.csv"
SENSOR_OUT = AUDIT / "dataset_C_sensor_schema_inventory.csv"

def add(lines, s=""):
    lines.append(str(s))

def natural_key(s):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", str(s))
    ]

# ---------------------------------------------------------------------
# Locate dataset root from borg_data.csv rather than assuming nesting.
# ---------------------------------------------------------------------
borg_hits = list(RAW.rglob("borg_data.csv"))
if len(borg_hits) != 1:
    raise SystemExit(f"Expected exactly one borg_data.csv; found {len(borg_hits)}")

BORG = borg_hits[0]
DSROOT = BORG.parent.parent

sensor_root_hits = [
    p for p in DSROOT.iterdir()
    if p.is_dir() and p.name.lower() == "emg, imu, and ppg data"
]
if len(sensor_root_hits) != 1:
    raise SystemExit(
        f"Expected one exact 'EMG, IMU, and PPG data' folder; found {len(sensor_root_hits)}"
    )
SENSOR_ROOT = sensor_root_hits[0]

borg = pd.read_csv(BORG)

# Clean column names only for lookup, preserve originals.
col_lookup = {
    re.sub(r"\s+", "", c.strip().lower()): c
    for c in borg.columns
}
subject_col = col_lookup.get("subject")
task_col = col_lookup.get("task_order")

if subject_col is None or task_col is None:
    raise SystemExit("Could not find subject/task_order in Borg table.")

# ---------------------------------------------------------------------
# Exact task mapping.
# ---------------------------------------------------------------------
task_folder_map = {
    "task1_35i": "30-40_ internal rotation",
    "task2_45i": "40-50_ internal rotation",
    "task3_55i": "50-60_ internal rotation",
    "task4_35e": "30-40_ external rotation",
    "task5_45e": "40-50_ external rotation",
    "task6_55e": "50-60_ external rotation",
}

# actual task folders
task_dirs = {
    p.name: p
    for p in SENSOR_ROOT.iterdir()
    if p.is_dir()
}

# ---------------------------------------------------------------------
# Metadata inventory outside sensor root.
# ---------------------------------------------------------------------
metadata_files = [
    p for p in DSROOT.rglob("*")
    if p.is_file() and SENSOR_ROOT not in p.parents
]

# ---------------------------------------------------------------------
# Trial / modality inventory.
# ---------------------------------------------------------------------
trial_rows = []
sensor_schema_rows = []
all_emg_names = Counter()
all_imu_locations = Counter()
all_imu_names = Counter()
all_ppg_names = Counter()

for _, br in borg.iterrows():
    subject_label = str(br[subject_col]).strip()
    m = re.search(r"(\d+)", subject_label)
    subject_num = int(m.group(1)) if m else None

    task_label = str(br[task_col]).strip()
    expected_task_folder = task_folder_map.get(task_label)

    task_dir = task_dirs.get(expected_task_folder)
    subj_dir = task_dir / f"Subject {subject_num}" if task_dir and subject_num else None

    exists = bool(subj_dir and subj_dir.exists())

    emg_dir = subj_dir / "EMG data" if exists else None
    imu_dir = subj_dir / "IMU data" if exists else None
    ppg_dir = subj_dir / "PPG data" if exists else None

    emg_files = sorted(
        [p for p in emg_dir.iterdir() if p.is_file()],
        key=lambda p: natural_key(p.name)
    ) if emg_dir and emg_dir.exists() else []

    # IMU: files nested one level under body-location folders.
    imu_files = sorted(
        [p for p in imu_dir.rglob("*") if p.is_file()],
        key=lambda p: natural_key(str(p.relative_to(imu_dir)))
    ) if imu_dir and imu_dir.exists() else []

    imu_locations = sorted(
        [p.name for p in imu_dir.iterdir() if p.is_dir()],
        key=natural_key
    ) if imu_dir and imu_dir.exists() else []

    ppg_files = sorted(
        [p for p in ppg_dir.rglob("*") if p.is_file()],
        key=lambda p: natural_key(str(p.relative_to(ppg_dir)))
    ) if ppg_dir and ppg_dir.exists() else []

    for p in emg_files:
        all_emg_names[p.name] += 1

    for loc in imu_locations:
        all_imu_locations[loc] += 1

    for p in imu_files:
        rel = p.relative_to(imu_dir)
        all_imu_names[str(rel)] += 1

    for p in ppg_files:
        all_ppg_names[str(p.relative_to(ppg_dir))] += 1

    trial_rows.append({
        "subject": subject_label,
        "subject_num": subject_num,
        "task_label": task_label,
        "expected_task_folder": expected_task_folder,
        "trial_folder_exists": exists,
        "emg_dir_exists": bool(emg_dir and emg_dir.exists()),
        "imu_dir_exists": bool(imu_dir and imu_dir.exists()),
        "ppg_dir_exists": bool(ppg_dir and ppg_dir.exists()),
        "n_emg_files": len(emg_files),
        "n_imu_locations": len(imu_locations),
        "n_imu_files": len(imu_files),
        "n_ppg_files": len(ppg_files),
        "imu_locations": ";".join(imu_locations),
        "emg_names": ";".join(p.name for p in emg_files),
        "ppg_names": ";".join(str(p.relative_to(ppg_dir)) for p in ppg_files) if ppg_dir else "",
    })

trial_df = pd.DataFrame(trial_rows)
trial_df.to_csv(TRIAL_OUT, index=False)

# ---------------------------------------------------------------------
# Exact sensor schemas from unique relative file types.
# ---------------------------------------------------------------------
sample_groups = []

# choose first existing trial for each task
for task_label, folder_name in task_folder_map.items():
    td = task_dirs.get(folder_name)
    if not td:
        continue
    subjects = sorted(
        [p for p in td.iterdir() if p.is_dir() and re.match(r"(?i)subject\s+\d+", p.name)],
        key=lambda p: natural_key(p.name)
    )
    if subjects:
        sample_groups.append((task_label, subjects[0]))

for task_label, subj_dir in sample_groups:
    for modality, base in [
        ("EMG", subj_dir / "EMG data"),
        ("IMU", subj_dir / "IMU data"),
        ("PPG", subj_dir / "PPG data"),
    ]:
        if not base.exists():
            continue
        for fp in sorted([p for p in base.rglob("*") if p.is_file()], key=lambda p: natural_key(str(p))):
            try:
                x = pd.read_csv(fp, nrows=3)
                sensor_schema_rows.append({
                    "task_label": task_label,
                    "subject_folder": subj_dir.name,
                    "modality": modality,
                    "relative_sensor_file": str(fp.relative_to(base)),
                    "n_columns": len(x.columns),
                    "columns": " | ".join(map(str, x.columns)),
                })
            except Exception as e:
                sensor_schema_rows.append({
                    "task_label": task_label,
                    "subject_folder": subj_dir.name,
                    "modality": modality,
                    "relative_sensor_file": str(fp.relative_to(base)),
                    "n_columns": np.nan,
                    "columns": f"READ_ERROR: {type(e).__name__}: {e}",
                })

schema_df = pd.DataFrame(sensor_schema_rows)
schema_df.to_csv(SENSOR_OUT, index=False)

# ---------------------------------------------------------------------
# Borg uniqueness / completeness
# ---------------------------------------------------------------------
borg_pairs = borg[[subject_col, task_col]].astype(str)
pair_dupes = int(borg_pairs.duplicated().sum())

expected_pairs = {
    (f"subject_{s}", t)
    for s in range(1, 35)
    for t in task_folder_map
}
observed_pairs = {
    (str(r[subject_col]).strip(), str(r[task_col]).strip())
    for _, r in borg.iterrows()
}
missing_borg_pairs = sorted(expected_pairs - observed_pairs, key=lambda z: (natural_key(z[0]), z[1]))
extra_borg_pairs = sorted(observed_pairs - expected_pairs, key=lambda z: (natural_key(z[0]), z[1]))

# ---------------------------------------------------------------------
# Exact corrupted location
# Use path relative to SENSOR_ROOT; do not inspect absolute project path.
# ---------------------------------------------------------------------
corrupt_dir = (
    SENSOR_ROOT
    / "30-40_ internal rotation"
    / "Subject 3"
    / "IMU data"
    / "Shoulder"
)

corrupt_files = sorted(
    [p for p in corrupt_dir.iterdir() if p.is_file()],
    key=lambda p: natural_key(p.name)
) if corrupt_dir.exists() else []

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
lines = []
add(lines, "DATASET C — CORRECTED STRUCTURE / MODALITY AUDIT")
add(lines, "=" * 118)
add(lines, f"Dataset root: {DSROOT}")
add(lines, f"Sensor root: {SENSOR_ROOT}")
add(lines, f"Borg rows: {len(borg)}")
add(lines, f"Borg subject-task duplicate rows: {pair_dupes}")
add(lines, f"Expected 34 x 6 subject-task pairs: {34*6}")
add(lines, f"Observed unique subject-task pairs: {len(observed_pairs)}")
add(lines, f"Missing expected Borg pairs: {len(missing_borg_pairs)}")
add(lines, f"Unexpected Borg pairs: {len(extra_borg_pairs)}")
if missing_borg_pairs:
    add(lines, "Missing pairs:")
    for z in missing_borg_pairs:
        add(lines, f"  {z}")
if extra_borg_pairs:
    add(lines, "Unexpected pairs:")
    for z in extra_borg_pairs:
        add(lines, f"  {z}")
add(lines)

add(lines, "1. ACTUAL TASK FOLDERS")
add(lines, "-" * 118)
for name in sorted(task_dirs, key=natural_key):
    add(lines, f"  {name}")
add(lines)

add(lines, "2. BORG -> SENSOR TRIAL MAPPING")
add(lines, "-" * 118)
add(lines, f"Trial folders found: {int(trial_df['trial_folder_exists'].sum())}/{len(trial_df)}")
add(lines, f"EMG directories found: {int(trial_df['emg_dir_exists'].sum())}/{len(trial_df)}")
add(lines, f"IMU directories found: {int(trial_df['imu_dir_exists'].sum())}/{len(trial_df)}")
add(lines, f"PPG directories found: {int(trial_df['ppg_dir_exists'].sum())}/{len(trial_df)}")
add(lines)

add(lines, "Per-trial file-count distributions:")
for c in ["n_emg_files","n_imu_locations","n_imu_files","n_ppg_files"]:
    add(lines, f"{c}:")
    add(lines, trial_df[c].value_counts().sort_index().to_string())
add(lines)

add(lines, "3. UNIQUE EMG FILE NAMES")
add(lines, "-" * 118)
for name, count in sorted(all_emg_names.items(), key=lambda x: natural_key(x[0])):
    add(lines, f"{name}: present in {count}/204 trials")
add(lines)

add(lines, "4. UNIQUE IMU BODY LOCATIONS")
add(lines, "-" * 118)
for name, count in sorted(all_imu_locations.items(), key=lambda x: natural_key(x[0])):
    add(lines, f"{name}: present in {count}/204 trials")
add(lines)

add(lines, "5. UNIQUE IMU RELATIVE FILES")
add(lines, "-" * 118)
for name, count in sorted(all_imu_names.items(), key=lambda x: natural_key(x[0])):
    add(lines, f"{name}: present in {count}/204 trials")
add(lines)

add(lines, "6. UNIQUE PPG FILES")
add(lines, "-" * 118)
for name, count in sorted(all_ppg_names.items(), key=lambda x: natural_key(x[0])):
    add(lines, f"{name}: present in {count}/204 trials")
add(lines)

add(lines, "7. NON-SENSOR METADATA FILES")
add(lines, "-" * 118)
for fp in sorted(metadata_files, key=lambda p: natural_key(str(p.relative_to(DSROOT)))):
    add(lines, f"{fp.relative_to(DSROOT)} | {fp.stat().st_size} bytes")
add(lines)

add(lines, "8. EXACT KNOWN CORRUPTED LOCATION")
add(lines, "-" * 118)
add(lines, f"Directory exists: {corrupt_dir.exists()}")
add(lines, f"Exact directory: {corrupt_dir.relative_to(DSROOT) if corrupt_dir.exists() else corrupt_dir}")
add(lines, f"Files in exact corrupted Shoulder directory: {len(corrupt_files)}")
for fp in corrupt_files:
    add(lines, f"  {fp.name} | {fp.stat().st_size} bytes")
add(lines)
add(lines, "Only this exact trial/location should be quarantined from Shoulder-IMU analyses unless further corruption is found.")
add(lines)

add(lines, "9. TRIALS WITH STRUCTURAL IRREGULARITIES")
add(lines, "-" * 118)

mode_cols = {}
for c in ["n_emg_files","n_imu_locations","n_imu_files","n_ppg_files"]:
    vc = trial_df[c].value_counts()
    mode_cols[c] = int(vc.index[0]) if len(vc) else None

irreg_mask = ~trial_df["trial_folder_exists"]
for c, mode in mode_cols.items():
    if mode is not None:
        irreg_mask = irreg_mask | (trial_df[c] != mode)

irreg = trial_df[irreg_mask]
add(lines, f"Trials differing from modal structure: {len(irreg)}")
if len(irreg):
    add(lines, irreg[
        ["subject","task_label","expected_task_folder","trial_folder_exists",
         "n_emg_files","n_imu_locations","n_imu_files","n_ppg_files","imu_locations"]
    ].to_string(index=False))
add(lines)

add(lines, "10. SENSOR SCHEMA EXAMPLES")
add(lines, "-" * 118)
if len(schema_df):
    # one unique schema per modality/file basename
    tmp = schema_df.copy()
    tmp["basename"] = tmp["relative_sensor_file"].apply(lambda x: Path(x).name)
    tmp = tmp.drop_duplicates(subset=["modality","basename","columns"])
    for _, r in tmp.iterrows():
        add(lines, f"[{r['modality']}] {r['relative_sensor_file']}")
        add(lines, f"  columns={r['columns']}")
else:
    add(lines, "No sensor schemas were read.")
add(lines)

add(lines, "11. NEXT REQUIRED CHECK BEFORE MODELING")
add(lines, "-" * 118)
add(lines, "Do NOT train yet.")
add(lines, "Next audit must verify temporal alignment between:")
add(lines, "  Borg time points (10, 20, ... seconds),")
add(lines, "  EMG sample counts,")
add(lines, "  IMU sample counts,")
add(lines, "  PPG sample counts.")
add(lines)
add(lines, "We must determine whether each modality starts at the same trial origin and")
add(lines, "whether recorded lengths match Borg trial duration closely enough to define")
add(lines, "strict prior-window -> next-Borg prediction samples without future leakage.")
add(lines)
add(lines, "=" * 118)
add(lines, "END OF AUDIT")

TXT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("Corrected Dataset C structure audit complete:")
print(f"  {TXT_OUT}")
print(f"  {TRIAL_OUT}")
print(f"  {SENSOR_OUT}")
print("")
print("Upload dataset_C_corrected_structure_audit.txt to ChatGPT.")
