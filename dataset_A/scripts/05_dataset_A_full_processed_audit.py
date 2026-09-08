from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import math
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "dataset_A_northwestern_mxd"
FULL = DATA_ROOT / "processed_full"
AUDIT = ROOT / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)

TXT_OUT = AUDIT / "dataset_A_full_processed_audit.txt"
FILE_OUT = AUDIT / "dataset_A_full_processed_file_inventory.csv"
TABLE_OUT = AUDIT / "dataset_A_full_processed_table_inventory.csv"
SCHEMA_OUT = AUDIT / "dataset_A_full_processed_schema_summary.csv"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def add(lines, text=""):
    lines.append(str(text))

def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def extract_pid(text):
    m = re.search(r"(?i)\bP0*(\d{1,3})\b", str(text))
    if not m:
        m = re.search(r"(?i)(?:participant|subject|user)[_\-\s]*0*(\d{1,3})", str(text))
    return f"P{int(m.group(1)):03d}" if m else None

def infer_task(text):
    t = str(text).lower()
    if "composite" in t or re.search(r"p\d+c\d+s\d+", t):
        return "Composite"
    if "ziptie" in t or "wire" in t or "harness" in t or re.search(r"p\d+z\d+s\d+", t):
        return "WireHarness"
    return None

def read_header(path):
    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            return list(pd.read_csv(path, nrows=0).columns)
        if ext in {".tsv", ".txt"}:
            try:
                return list(pd.read_csv(path, nrows=0, sep=None, engine="python").columns)
            except Exception:
                return []
        if ext in {".xlsx", ".xls"}:
            xl = pd.ExcelFile(path)
            if not xl.sheet_names:
                return []
            return list(pd.read_excel(path, sheet_name=xl.sheet_names[0], nrows=0).columns)
        if ext == ".parquet":
            return list(pd.read_parquet(path).columns)
        if ext in {".pkl", ".pickle"}:
            obj = pd.read_pickle(path)
            return list(obj.columns) if isinstance(obj, pd.DataFrame) else []
    except Exception:
        return []
    return []

def read_small(path, nrows=20):
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(path, nrows=nrows, sep=None, engine="python")
    if ext in {".xlsx", ".xls"}:
        xl = pd.ExcelFile(path)
        return pd.read_excel(path, sheet_name=xl.sheet_names[0], nrows=nrows)
    if ext == ".parquet":
        return pd.read_parquet(path).head(nrows)
    if ext in {".pkl", ".pickle"}:
        obj = pd.read_pickle(path)
        return obj.head(nrows) if isinstance(obj, pd.DataFrame) else None
    return None

def count_csv_rows(path):
    # Efficient enough for ~300 MB archive; avoids loading full file solely for row count.
    try:
        with path.open("rb") as f:
            n = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b""))
        return max(n - 1, 0)
    except Exception:
        return np.nan

def data_frame_full(path):
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python")
    if ext in {".xlsx", ".xls"}:
        xl = pd.ExcelFile(path)
        return pd.read_excel(path, sheet_name=xl.sheet_names[0])
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in {".pkl", ".pickle"}:
        obj = pd.read_pickle(path)
        return obj if isinstance(obj, pd.DataFrame) else None
    return None

def find_col(columns, aliases):
    nc = {norm(c): c for c in columns}
    for a in aliases:
        if norm(a) in nc:
            return nc[norm(a)]
    # contains match as fallback
    for c in columns:
        z = norm(c)
        if any(norm(a) in z for a in aliases):
            return c
    return None

def numeric_summary(series):
    s = pd.to_numeric(series, errors="coerce")
    return {
        "n": int(s.notna().sum()),
        "missing_rate": float(s.isna().mean()) if len(s) else np.nan,
        "nunique": int(s.nunique(dropna=True)),
        "min": float(s.min()) if s.notna().any() else np.nan,
        "median": float(s.median()) if s.notna().any() else np.nan,
        "max": float(s.max()) if s.notna().any() else np.nan,
    }

# ---------------------------------------------------------------------
# 1. File inventory
# ---------------------------------------------------------------------
lines = []
add(lines, "DATASET A — FULL PROCESSED DATA SCIENTIFIC AUDIT")
add(lines, "=" * 118)
add(lines, f"Processed-full root: {FULL}")
add(lines, f"Exists: {FULL.exists()}")
add(lines)

if not FULL.exists():
    add(lines, "FATAL: processed_full folder does not exist.")
    TXT_OUT.write_text("\n".join(lines), encoding="utf-8")
    raise SystemExit(f"Folder not found: {FULL}")

all_files = [p for p in FULL.rglob("*") if p.is_file()]
all_dirs = [p for p in FULL.rglob("*") if p.is_dir()]

file_records = []
for pth in all_files:
    rel = pth.relative_to(FULL)
    file_records.append({
        "relative_path": str(rel),
        "name": pth.name,
        "extension": pth.suffix.lower() or "<no_ext>",
        "size_bytes": pth.stat().st_size,
        "participant_from_path": extract_pid(str(rel)),
        "task_from_path": infer_task(str(rel)),
    })
file_df = pd.DataFrame(file_records)
file_df.to_csv(FILE_OUT, index=False)

ext_counts = Counter(r["extension"] for r in file_records)
add(lines, "1. FILE INVENTORY")
add(lines, "-" * 118)
add(lines, f"Directories: {len(all_dirs)}")
add(lines, f"Files: {len(all_files)}")
add(lines, f"Total extracted size: {sum(r['size_bytes'] for r in file_records):,} bytes")
add(lines, "Extensions:")
for ext, n in sorted(ext_counts.items(), key=lambda x: (-x[1], x[0])):
    add(lines, f"  {ext:15s} {n}")
add(lines)

# Top-level tree
add(lines, "Top-level entries:")
for x in sorted(FULL.iterdir(), key=lambda p: p.name.lower()):
    typ = "DIR" if x.is_dir() else "FILE"
    add(lines, f"  [{typ}] {x.name}")
add(lines)

# ---------------------------------------------------------------------
# 2. Tabular file schema inventory
# ---------------------------------------------------------------------
tab_exts = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".pkl", ".pickle"}
table_records = []
schema_counter = Counter()
schema_examples = defaultdict(list)

for pth in all_files:
    if pth.suffix.lower() not in tab_exts:
        continue

    header = read_header(pth)
    if not header:
        continue

    schema = tuple(map(str, header))
    schema_key = " | ".join(schema)
    schema_counter[schema_key] += 1
    if len(schema_examples[schema_key]) < 5:
        schema_examples[schema_key].append(str(pth.relative_to(FULL)))

    n_rows = count_csv_rows(pth) if pth.suffix.lower() == ".csv" else np.nan
    table_records.append({
        "relative_path": str(pth.relative_to(FULL)),
        "extension": pth.suffix.lower(),
        "size_bytes": pth.stat().st_size,
        "n_rows": n_rows,
        "n_columns": len(header),
        "participant_from_path": extract_pid(str(pth.relative_to(FULL))),
        "task_from_path": infer_task(str(pth.relative_to(FULL))),
        "columns": " | ".join(map(str, header)),
    })

table_df = pd.DataFrame(table_records)
table_df.to_csv(TABLE_OUT, index=False)

schema_rows = []
for idx, (schema_key, count) in enumerate(schema_counter.most_common(), start=1):
    cols = schema_key.split(" | ") if schema_key else []
    schema_rows.append({
        "schema_id": idx,
        "file_count": count,
        "n_columns": len(cols),
        "example_files": " ; ".join(schema_examples[schema_key]),
        "columns": schema_key,
    })
schema_df = pd.DataFrame(schema_rows)
schema_df.to_csv(SCHEMA_OUT, index=False)

add(lines, "2. TABULAR SCHEMA INVENTORY")
add(lines, "-" * 118)
add(lines, f"Readable tabular files: {len(table_df)}")
add(lines, f"Distinct schemas: {len(schema_df)}")
if not schema_df.empty:
    add(lines, "Most common schemas:")
    for _, r in schema_df.head(12).iterrows():
        add(lines, f"  Schema {int(r['schema_id'])}: files={int(r['file_count'])}, columns={int(r['n_columns'])}")
        add(lines, f"    Examples: {r['example_files']}")
        cols = str(r["columns"]).split(" | ")
        add(lines, "    Columns: " + " | ".join(cols[:80]))
add(lines)

# ---------------------------------------------------------------------
# 3. Candidate "processed feature" tables
# ---------------------------------------------------------------------
outcome_aliases = [
    "final_fatigue", "physical fatigue final", "physical_fatigue_final",
    "fatigue_final", "final fatigue", "borg_rating", "rpe"
]
initial_aliases = [
    "init_fatigue", "initial_fatigue", "physical fatigue initial",
    "physical_fatigue_initial", "fatigue_initial", "initial fatigue"
]
participant_aliases = [
    "participant", "participant_id", "subject", "subject_id", "user", "user_id",
    "pid", "worker", "worker_id"
]
task_aliases = ["task", "task_type", "condition", "activity"]
rep_aliases = ["rep", "repetition", "repetition_number", "rep_num", "cycle", "cycle_number"]

feature_candidates = []
for rec in table_records:
    cols = rec["columns"].split(" | ") if rec["columns"] else []
    final_col = find_col(cols, outcome_aliases)
    init_col = find_col(cols, initial_aliases)
    if final_col or init_col:
        rr = dict(rec)
        rr["final_col"] = final_col
        rr["initial_col"] = init_col
        rr["participant_col"] = find_col(cols, participant_aliases)
        rr["task_col"] = find_col(cols, task_aliases)
        rr["rep_col"] = find_col(cols, rep_aliases)
        feature_candidates.append(rr)

add(lines, "3. CANDIDATE PROCESSED FEATURE TABLES")
add(lines, "-" * 118)
add(lines, f"Files whose schema contains an initial/final fatigue-like field: {len(feature_candidates)}")
for rec in feature_candidates[:40]:
    add(lines, f"  {rec['relative_path']}")
    add(lines, f"    rows≈{rec['n_rows']} cols={rec['n_columns']} "
               f"final={rec['final_col']} initial={rec['initial_col']} "
               f"participant={rec['participant_col']} task={rec['task_col']} rep={rec['rep_col']}")
add(lines)

# ---------------------------------------------------------------------
# 4. Deep audit of candidate processed tables
#    Process manageable feature tables in full; avoid loading huge raw-like files.
# ---------------------------------------------------------------------
deep_records = []
combined_small_tables = []

for rec in feature_candidates:
    pth = FULL / rec["relative_path"]
    # Process full table unless very large; archive is only ~300 MB total,
    # but this protects against accidental raw mega-tables.
    if pth.stat().st_size > 120_000_000:
        continue
    try:
        df = data_frame_full(pth)
    except Exception:
        continue
    if df is None or df.empty:
        continue

    cols = list(df.columns)
    final_col = find_col(cols, outcome_aliases)
    init_col = find_col(cols, initial_aliases)
    pid_col = find_col(cols, participant_aliases)
    task_col = find_col(cols, task_aliases)
    rep_col = find_col(cols, rep_aliases)

    # Basic identity
    pid_path = extract_pid(rec["relative_path"])
    task_path = infer_task(rec["relative_path"])

    # Detect known engineered features from original code
    normalized_cols = {norm(c): c for c in cols}
    engineered_groups = {
        "hr": [c for c in cols if any(k in norm(c) for k in ["avg_hr", "del_hr", "median_hr", "std_hr", "skew_hr", "kurt_hr"])],
        "hrv": [c for c in cols if "hrv" in norm(c)],
        "temperature": [c for c in cols if "temp" in norm(c)],
        "imu_acc": [c for c in cols if any(k in norm(c) for k in ["rms_acc", "rms_imu"]) and "gyro" not in norm(c)],
        "imu_vel": [c for c in cols if "vel" in norm(c)],
        "imu_gyro": [c for c in cols if "gyro" in norm(c)],
        "jerk": [c for c in cols if "jerk" in norm(c) or "ldlj" in norm(c)],
        "kinetic": [c for c in cols if "kinetic" in norm(c)],
        "demographic": [c for c in cols if norm(c) in {"age", "height", "weight", "gender", "sex"}],
    }

    drec = {
        "relative_path": rec["relative_path"],
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "participant_from_path": pid_path,
        "task_from_path": task_path,
        "participant_col": pid_col,
        "task_col": task_col,
        "rep_col": rep_col,
        "final_col": final_col,
        "initial_col": init_col,
        "duplicate_full_rows": int(df.duplicated().sum()),
    }

    if pid_col:
        drec["participant_unique_in_col"] = int(df[pid_col].astype(str).nunique(dropna=True))
    if task_col:
        drec["task_unique_in_col"] = int(df[task_col].astype(str).nunique(dropna=True))
    if rep_col:
        drec["rep_unique_in_col"] = int(pd.to_numeric(df[rep_col], errors="coerce").nunique(dropna=True))

    if final_col:
        s = numeric_summary(df[final_col])
        for k, v in s.items():
            drec[f"final_{k}"] = v
    if init_col:
        s = numeric_summary(df[init_col])
        for k, v in s.items():
            drec[f"initial_{k}"] = v

    for g, cc in engineered_groups.items():
        drec[f"{g}_feature_count"] = len(cc)

    deep_records.append(drec)

    # Keep compact feature tables for cross-file consolidated diagnostics.
    if len(df) <= 2_000_000:
        tmp = df.copy()
        tmp["__source_file"] = rec["relative_path"]
        tmp["__pid_path"] = pid_path
        tmp["__task_path"] = task_path
        combined_small_tables.append(tmp)

deep_df = pd.DataFrame(deep_records)

add(lines, "4. DEEP FEATURE-TABLE AUDIT")
add(lines, "-" * 118)
if deep_df.empty:
    add(lines, "No candidate processed feature tables could be fully read.")
else:
    show_cols = [c for c in [
        "relative_path","n_rows","n_columns","participant_from_path","task_from_path",
        "participant_col","task_col","rep_col","final_col","initial_col",
        "duplicate_full_rows","final_n","final_nunique","final_min","final_median","final_max",
        "initial_n","initial_nunique","hr_feature_count","hrv_feature_count",
        "temperature_feature_count","imu_acc_feature_count","imu_vel_feature_count",
        "imu_gyro_feature_count","jerk_feature_count","kinetic_feature_count","demographic_feature_count"
    ] if c in deep_df.columns]
    add(lines, deep_df[show_cols].to_string(index=False))
add(lines)

# ---------------------------------------------------------------------
# 5. Consolidated participant/task structure from paths and data columns
# ---------------------------------------------------------------------
all_pids_from_paths = sorted({x for x in file_df["participant_from_path"].dropna().unique()})
add(lines, "5. PARTICIPANT / TASK COVERAGE IN FULL ARCHIVE")
add(lines, "-" * 118)
add(lines, f"Unique participant IDs recoverable from paths: {len(all_pids_from_paths)}")
add(lines, "IDs: " + ", ".join(all_pids_from_paths))
add(lines)

path_cross = (
    file_df.dropna(subset=["participant_from_path"])
    .groupby(["task_from_path", "participant_from_path"])
    .size()
    .rename("file_count")
    .reset_index()
)
if not path_cross.empty:
    add(lines, "Files by inferred task and participant:")
    add(lines, path_cross.to_string(index=False))
add(lines)

task_to_pids = {}
for task in ["Composite", "WireHarness"]:
    pids = set(
        file_df.loc[file_df["task_from_path"] == task, "participant_from_path"]
        .dropna().astype(str)
    )
    task_to_pids[task] = pids
    add(lines, f"{task}: {len(pids)} participants")
    add(lines, "  " + ", ".join(sorted(pids)))

if task_to_pids["Composite"] or task_to_pids["WireHarness"]:
    both = task_to_pids["Composite"] & task_to_pids["WireHarness"]
    only_c = task_to_pids["Composite"] - task_to_pids["WireHarness"]
    only_w = task_to_pids["WireHarness"] - task_to_pids["Composite"]
    add(lines, f"Paired participants in BOTH tasks: {len(both)}")
    add(lines, "  " + ", ".join(sorted(both)))
    add(lines, f"Composite-only participants: {len(only_c)}")
    add(lines, "  " + ", ".join(sorted(only_c)))
    add(lines, f"WireHarness-only participants: {len(only_w)}")
    add(lines, "  " + ", ".join(sorted(only_w)))
add(lines)

# ---------------------------------------------------------------------
# 6. Gyroscope propagation audit in engineered feature names
# ---------------------------------------------------------------------
add(lines, "6. GYROSCOPE-PROPAGATION AUDIT")
add(lines, "-" * 118)
gyro_schema_hits = []
for rec in feature_candidates:
    cols = rec["columns"].split(" | ") if rec["columns"] else []
    hits = [c for c in cols if "gyro" in norm(c)]
    if hits:
        gyro_schema_hits.append((rec["relative_path"], hits))

if gyro_schema_hits:
    add(lines, f"Processed fatigue-related tables containing gyro-derived feature names: {len(gyro_schema_hits)}")
    for path, hits in gyro_schema_hits[:30]:
        add(lines, f"  {path}")
        add(lines, "    " + " | ".join(hits))
    add(lines)
    add(lines, "IMPORTANT:")
    add(lines, "  The public GitHub segmented CSVs had gx=gy=gz for all 690 audited IMU/segment triplets.")
    add(lines, "  Any engineered gyro feature derived from those duplicated channels must be treated as potentially")
    add(lines, "  contaminated until provenance is verified against the original raw source.")
else:
    add(lines, "No gyro-derived feature names were found in candidate processed fatigue tables.")
add(lines)

# ---------------------------------------------------------------------
# 7. Context / wearable / outcome eligibility from observed schemas
# ---------------------------------------------------------------------
all_observed_cols = []
seen = set()
for rec in table_records:
    for c in rec["columns"].split(" | "):
        if c not in seen:
            seen.add(c)
            all_observed_cols.append(c)

def cols_matching(keys):
    out = []
    for c in all_observed_cols:
        z = norm(c)
        if any(norm(k) in z for k in keys):
            out.append(c)
    return out

context_matches = cols_matching([
    "init_fatigue","initial_fatigue","physical fatigue initial","mental fatigue initial",
    "age","height","weight","gender","task","rep","repetition","weights_added"
])
wearable_matches = cols_matching([
    "avg_hr","del_hr","median_hr","std_hr","skew_hr","kurt_hr","hrv","temp",
    "rms_acc","rms_imu","vel","gyro","jerk","ldlj","kinetic","ecg"
])
outcome_matches = cols_matching([
    "final_fatigue","physical fatigue final","mental fatigue final","performance","rpe","borg"
])

add(lines, "7. OBSERVED VARIABLE-ROLE CANDIDATES")
add(lines, "-" * 118)
add(lines, "Potential pre-task/context variables observed:")
for c in context_matches:
    add(lines, f"  - {c}")
add(lines)
add(lines, "Potential wearable/engineered sensor variables observed:")
for c in wearable_matches:
    add(lines, f"  - {c}")
add(lines)
add(lines, "Potential outcome / post-task variables observed:")
for c in outcome_matches:
    add(lines, f"  - {c}")
add(lines)

# ---------------------------------------------------------------------
# 8. Publication-level readiness summary
# ---------------------------------------------------------------------
add(lines, "8. PUBLICATION-LEVEL READINESS CHECK")
add(lines, "-" * 118)

n_path_pids = len(all_pids_from_paths)
if n_path_pids >= 30:
    add(lines, f"PASS: Full archive exposes >=30 participant IDs from paths ({n_path_pids}).")
elif n_path_pids > 10:
    add(lines, f"PARTIAL PASS: Archive exposes more than the 10-person GitHub subset ({n_path_pids}).")
else:
    add(lines, f"WARNING: Only {n_path_pids} participant IDs are recoverable from paths.")

if feature_candidates:
    add(lines, f"PASS: {len(feature_candidates)} fatigue-related processed table files were identified.")
else:
    add(lines, "WARNING: No fatigue-related processed table was identified automatically.")

if gyro_schema_hits:
    add(lines, "CAUTION: Gyro-derived engineered features exist and require provenance verification.")
else:
    add(lines, "PASS/NEUTRAL: No gyro-derived processed feature names were detected.")

add(lines)
add(lines, "Decision logic for the next step:")
add(lines, "  A. If full participant-level feature tables are present -> build leakage-controlled Phase-I benchmark.")
add(lines, "  B. Exclude or quarantine gyro-derived features unless their source is independently verified.")
add(lines, "  C. Preserve participant grouping in every train/validation/test split.")
add(lines, "  D. Treat Physical Fatigue - Final as the primary outcome; initial fatigue is eligible context.")
add(lines, "  E. Keep Mental Fatigue - Final and Performance rating out of the primary predictor set.")
add(lines)
add(lines, "=" * 118)
add(lines, "END OF AUDIT")

TXT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("Full processed-data audit complete:")
print(f"  {TXT_OUT}")
print(f"  {FILE_OUT}")
print(f"  {TABLE_OUT}")
print(f"  {SCHEMA_OUT}")
print("")
print("Upload dataset_A_full_processed_audit.txt to ChatGPT.")
