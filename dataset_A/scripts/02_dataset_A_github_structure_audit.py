from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "data" / "dataset_A_northwestern_mxd" / "official_code"
PM = REPO / "Predictive_Models"
AUDIT = ROOT / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)
OUT = AUDIT / "dataset_A_github_structure_audit.txt"

lines = []
def add(x=""):
    lines.append(str(x))

add("DATASET A — OFFICIAL GITHUB STRUCTURE AUDIT")
add("=" * 100)
add(f"Repository path: {REPO}")
add(f"Repository exists: {REPO.exists()}")
add()

if not REPO.exists():
    add("ERROR: official_code folder does not exist.")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    raise SystemExit(f"Repository not found: {REPO}")

# Overall extension inventory
files = [p for p in REPO.rglob("*") if p.is_file()]
ext_counts = Counter((p.suffix.lower() or "<no_ext>") for p in files)
add("1. OVERALL FILE INVENTORY")
add("-" * 100)
add(f"Total files: {len(files)}")
for ext, n in sorted(ext_counts.items(), key=lambda x: (-x[1], x[0])):
    add(f"{ext:15s} {n}")
add()

# Key folders
folders = [
    "composite_train",
    "composite_test",
    "ziptie_train",
    "ziptie_test",
]
add("2. PREDICTIVE MODEL DATA FOLDERS")
add("-" * 100)

all_headers = defaultdict(list)

for folder_name in folders:
    d = PM / folder_name
    add(f"[{folder_name}]")
    add(f"Exists: {d.exists()}")
    if not d.exists():
        add()
        continue

    participant_dirs = sorted([p for p in d.iterdir() if p.is_dir()])
    add(f"Participant/session folders: {len(participant_dirs)}")
    add("Names: " + ", ".join(p.name for p in participant_dirs))

    folder_files = [p for p in d.rglob("*") if p.is_file()]
    add(f"Files: {len(folder_files)}")
    fext = Counter((p.suffix.lower() or "<no_ext>") for p in folder_files)
    add("Extensions: " + ", ".join(f"{k}={v}" for k, v in sorted(fext.items())))

    # inspect a few text/csv-like files
    inspected = 0
    for p in folder_files:
        if inspected >= 8:
            break
        if p.suffix.lower() not in {".csv", ".txt"}:
            continue

        rel = p.relative_to(REPO)
        add(f"  SAMPLE FILE: {rel}")
        try:
            with p.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
                reader = csv.reader(f)
                rows = []
                for i, row in enumerate(reader):
                    rows.append(row)
                    if i >= 2:
                        break
            if rows:
                header = rows[0]
                add(f"    Header columns ({len(header)}):")
                add("    " + " | ".join(header[:120]))
                all_headers[folder_name].append(header)
                if len(rows) > 1:
                    add("    First data row preview:")
                    add("    " + " | ".join(rows[1][:40]))
            else:
                add("    Empty file")
        except Exception as e:
            add(f"    Could not parse as delimited text: {type(e).__name__}: {e}")
        inspected += 1
    add()

# RAR / LFS check
add("3. PROCESSED FEATURE ARCHIVE CHECK")
add("-" * 100)
rar = PM / "MxD_Data_User_Study.rar"
add(f"Archive exists: {rar.exists()}")
if rar.exists():
    add(f"Local file size: {rar.stat().st_size} bytes")
    try:
        head = rar.read_bytes()[:300]
        text = head.decode("utf-8", errors="ignore")
        if "git-lfs.github.com/spec" in text:
            add("Status: Git LFS pointer only (the 101 MB payload was not included in the repository ZIP).")
        else:
            add("Status: appears to contain binary payload rather than an LFS pointer.")
    except Exception as e:
        add(f"Archive header check failed: {e}")
add()

# Source-code lexical audit for candidate context / wearable / outcome variables
add("4. SOURCE-CODE VARIABLE-SEMANTICS AUDIT")
add("-" * 100)

source_files = []
for name in [
    "helper_functions_user_study_2_0.py",
    "Data_Analysis_Regression_fixed_Train_Test.ipynb",
]:
    p = PM / name
    if p.exists():
        source_files.append(p)

terms = [
    "fatigue", "physical", "mental", "borg", "rpe",
    "age", "weight", "height", "gender",
    "vest", "task", "rep", "repetition",
    "heart", "hr", "hrv", "ecg", "temperature",
    "acc", "accelerometer", "gyro", "imu",
    "participant", "subject", "user",
]

for p in source_files:
    add(f"[{p.name}]")
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        add(f"Could not read: {e}")
        continue

    lower = text.lower()
    counts = {t: lower.count(t) for t in terms}
    for t, n in counts.items():
        if n:
            add(f"  {t:15s}: {n}")

    # Show limited snippets around especially important concepts.
    for key in ["fatigue", "physical", "mental", "age", "weight", "height", "heart", "gyro"]:
        matches = list(re.finditer(re.escape(key), lower))
        if not matches:
            continue
        add(f"  Snippets for '{key}':")
        for m in matches[:4]:
            lo = max(0, m.start() - 100)
            hi = min(len(text), m.end() + 180)
            snippet = re.sub(r"\s+", " ", text[lo:hi])
            add("    " + snippet)
    add()

# Aggregate header candidates
add("5. UNIQUE COLUMN-NAME CANDIDATES FROM SAMPLE FILES")
add("-" * 100)
unique_cols = []
seen = set()
for hs in all_headers.values():
    for h in hs:
        for c in h:
            c2 = c.strip()
            if c2 and c2 not in seen:
                seen.add(c2)
                unique_cols.append(c2)

add(f"Unique observed columns: {len(unique_cols)}")
for c in unique_cols:
    add(c)
add()

# Automatic grouping hints—not final scientific classification.
context_keys = [
    "age", "weight", "height", "gender", "sex", "mental",
    "initial", "task", "rep", "repetition", "vest", "performance"
]
wearable_keys = [
    "ecg", "heart", "hr", "hrv", "temp", "acc", "gyro",
    "imu", "sensor", "chest", "arm", "wrist"
]
outcome_keys = ["fatigue", "borg", "rpe", "target", "label"]

def candidates(keys):
    result = []
    for c in unique_cols:
        lc = c.lower()
        if any(k in lc for k in keys):
            result.append(c)
    return result

add("6. AUTOMATIC CANDIDATE GROUPING (FOR REVIEW ONLY)")
add("-" * 100)
add("Potential CONTEXT columns:")
for c in candidates(context_keys):
    add(f"  - {c}")
add()
add("Potential WEARABLE columns:")
for c in candidates(wearable_keys):
    add(f"  - {c}")
add()
add("Potential OUTCOME columns:")
for c in candidates(outcome_keys):
    add(f"  - {c}")
add()

add("=" * 100)
add("END OF AUDIT")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Audit complete: {OUT}")
print("Upload dataset_A_github_structure_audit.txt to ChatGPT.")
