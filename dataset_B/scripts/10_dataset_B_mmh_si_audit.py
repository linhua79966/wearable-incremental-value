from pathlib import Path
import re
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "data" / "dataset_B_mmh_si" / "Survival"
AUDIT = ROOT / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)

TXT_OUT = AUDIT / "dataset_B_mmh_si_audit.txt"
LONG_OUT = AUDIT / "dataset_B_mmh_si_long_preview.csv"
SHORT_OUT = AUDIT / "dataset_B_mmh_si_short_preview.csv"

LONG = REPO / "consv_sds.csv"
SHORT = REPO / "time_fixed_conservative.csv"

def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def find_candidates(columns, keys):
    out = []
    for c in columns:
        z = norm(c)
        if any(k in z for k in keys):
            out.append(c)
    return out

def add(lines, s=""):
    lines.append(str(s))

lines = []
add(lines, "DATASET B — MMH / SUPPLY INSERTION SCIENTIFIC AUDIT")
add(lines, "=" * 118)
add(lines, f"Repository: {REPO}")
add(lines, f"Long-form file exists: {LONG.exists()}")
add(lines, f"Short-form file exists: {SHORT.exists()}")
add(lines)

if not LONG.exists() or not SHORT.exists():
    add(lines, "FATAL: expected Case Study 1 data files are missing.")
    TXT_OUT.write_text("\n".join(lines), encoding="utf-8")
    raise SystemExit("Dataset B files missing.")

long_df = pd.read_csv(LONG)
short_df = pd.read_csv(SHORT)

long_df.head(50).to_csv(LONG_OUT, index=False)
short_df.head(50).to_csv(SHORT_OUT, index=False)

add(lines, "1. TABLE DIMENSIONS")
add(lines, "-" * 118)
add(lines, f"consv_sds.csv: rows={len(long_df)}, columns={len(long_df.columns)}")
add(lines, f"time_fixed_conservative.csv: rows={len(short_df)}, columns={len(short_df.columns)}")
add(lines)

add(lines, "2. LONG-FORM COLUMNS")
add(lines, "-" * 118)
for c in long_df.columns:
    add(lines, f"  - {c}")
add(lines)

add(lines, "3. SHORT-FORM COLUMNS")
add(lines, "-" * 118)
for c in short_df.columns:
    add(lines, f"  - {c}")
add(lines)

candidate_groups = {
    "participant": ["subject", "participant", "user", "worker", "id"],
    "task": ["task", "condition", "mmh", "si"],
    "time": ["time", "minute", "visit", "interval"],
    "fatigue_rpe": ["rpe", "fatigue", "exertion", "borg"],
    "event_ttf": ["event", "status", "ttf", "survival", "censor"],
    "demographic": ["age", "sex", "gender", "height", "weight", "bmi"],
    "heart": ["heart", "hr", "rr"],
    "imu_acc": ["acc", "acceler", "ax", "ay", "az"],
    "imu_gyro": ["gyro", "gx", "gy", "gz", "angular"],
    "sensor_feature": [
        "mean", "median", "std", "sd", "var", "rms", "range",
        "skew", "kurt", "jerk", "entropy", "energy"
    ],
}

for table_name, df in [("LONG", long_df), ("SHORT", short_df)]:
    add(lines, f"4. {table_name} — AUTOMATIC SEMANTIC CANDIDATES")
    add(lines, "-" * 118)
    for group, keys in candidate_groups.items():
        hits = find_candidates(df.columns, keys)
        add(lines, f"{group}: {len(hits)}")
        for c in hits[:80]:
            add(lines, f"  - {c}")
    add(lines)

pid_candidates = find_candidates(long_df.columns, ["subject", "participant", "worker", "user", "id"])
task_candidates = find_candidates(long_df.columns, ["task", "condition"])

def summarize_candidate(df, c):
    s = df[c]
    return (
        f"{c}: dtype={s.dtype}, missing={int(s.isna().sum())}, "
        f"unique={s.nunique(dropna=True)}, examples={list(s.dropna().astype(str).unique()[:20])}"
    )

add(lines, "5. KEY IDENTIFIER / TASK DISTRIBUTIONS")
add(lines, "-" * 118)
for c in pid_candidates[:10]:
    add(lines, summarize_candidate(long_df, c))
for c in task_candidates[:10]:
    add(lines, summarize_candidate(long_df, c))
add(lines)

add(lines, "6. LOW-CARDINALITY LONG-FORM COLUMNS")
add(lines, "-" * 118)
for c in long_df.columns:
    nun = long_df[c].nunique(dropna=True)
    if 1 <= nun <= 30:
        vals = list(long_df[c].dropna().astype(str).unique()[:30])
        add(lines, f"{c}: unique={nun} | {vals}")
add(lines)

add(lines, "7. LOW-CARDINALITY SHORT-FORM COLUMNS")
add(lines, "-" * 118)
for c in short_df.columns:
    nun = short_df[c].nunique(dropna=True)
    if 1 <= nun <= 30:
        vals = list(short_df[c].dropna().astype(str).unique()[:30])
        add(lines, f"{c}: unique={nun} | {vals}")
add(lines)

add(lines, "8. MISSINGNESS — LONG FORM")
add(lines, "-" * 118)
miss = long_df.isna().mean().sort_values(ascending=False)
for c, r in miss.items():
    if r > 0:
        add(lines, f"{c}: {r:.4f}")
if not (miss > 0).any():
    add(lines, "No missing values.")
add(lines)

pid_col = None
for preference in ["subject", "subject_id", "participant", "participant_id", "id"]:
    for c in long_df.columns:
        if norm(c) == preference:
            pid_col = c
            break
    if pid_col:
        break
if pid_col is None and pid_candidates:
    pid_col = pid_candidates[0]

add(lines, "9. REPEATED-MEASUREMENT STRUCTURE")
add(lines, "-" * 118)
add(lines, f"Selected participant candidate for diagnostics: {pid_col}")
if pid_col:
    counts = long_df.groupby(pid_col).size().sort_values()
    add(lines, f"Unique participants in long-form: {counts.index.nunique()}")
    add(lines, f"Rows per participant: min={counts.min()}, median={counts.median()}, max={counts.max()}")
    add(lines, counts.to_string())
add(lines)

task_col = None
for preference in ["task", "condition", "task_type"]:
    for c in long_df.columns:
        if norm(c) == preference:
            task_col = c
            break
    if task_col:
        break
if task_col is None and task_candidates:
    task_col = task_candidates[0]

add(lines, "10. TASK × PARTICIPANT STRUCTURE")
add(lines, "-" * 118)
add(lines, f"Selected task candidate: {task_col}")
if pid_col and task_col:
    cross = (
        long_df.groupby([task_col, pid_col])
        .size()
        .rename("n_long_rows")
        .reset_index()
    )
    add(lines, cross.to_string(index=False))
    add(lines)
    add(lines, "Participants by task:")
    for task, g in cross.groupby(task_col):
        add(lines, f"  {task}: {g[pid_col].nunique()}")
add(lines)

add(lines, "11. OFFICIAL R-CODE LEXICAL AUDIT")
add(lines, "-" * 118)
terms = [
    "RPE", "fatigue", "Subject", "Task", "MMH", "SI",
    "time", "status", "event", "censor", "sensor", "feature",
]
for name in ["0_data_preprocessing.Rmd", "1_baseline_modeling_evaluation.Rmd", "2_joint_modeling_evaluation.Rmd"]:
    p = REPO / name
    if not p.exists():
        continue
    txt = p.read_text(encoding="utf-8", errors="replace")
    low = txt.lower()
    add(lines, f"[{name}]")
    for term in terms:
        n = low.count(term.lower())
        if n:
            add(lines, f"  {term}: {n}")
    for key in ["rpe", "task", "subject", "fatigue", "event", "status"]:
        pos = [m.start() for m in re.finditer(key, low)]
        if pos:
            add(lines, f"  snippets '{key}':")
            for x in pos[:4]:
                lo = max(0, x - 120)
                hi = min(len(txt), x + 260)
                sn = re.sub(r"\s+", " ", txt[lo:hi])
                add(lines, "    " + sn)
    add(lines)

add(lines, "12. SCIENTIFIC ELIGIBILITY QUESTIONS FOR OUR PAPER")
add(lines, "-" * 118)
add(lines, "We must establish before modeling:")
add(lines, "  A. Whether an observed fatigue/RPE measure is available BEFORE or at each sensor interval.")
add(lines, "  B. Whether task identity and elapsed/protocol time can form a low-cost Context baseline.")
add(lines, "  C. Which wearable descriptors are genuinely time-varying and available without outcome leakage.")
add(lines, "  D. Whether the compiled 2026 repository preserves participant-independent structure for MMH and SI.")
add(lines, "  E. Whether our target should be next/future RPE, final fatigue, fatigue transition, or time-to-fatigue.")
add(lines, "  F. Whether the dataset is sufficiently comparable with Dataset A for a common incremental-value estimand.")
add(lines)
add(lines, "No modeling decision is made by this audit script itself.")
add(lines)
add(lines, "=" * 118)
add(lines, "END OF AUDIT")

TXT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("Dataset B audit complete:")
print(f"  {TXT_OUT}")
print(f"  {LONG_OUT}")
print(f"  {SHORT_OUT}")
print("")
print("Upload dataset_B_mmh_si_audit.txt to ChatGPT.")
