from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "data" / "dataset_B_mmh_si" / "Survival"
AUDIT = ROOT / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)

LONG = REPO / "consv_sds.csv"
SHORT = REPO / "time_fixed_conservative.csv"

TXT_OUT = AUDIT / "dataset_B_temporal_semantics_audit.txt"
MAP_OUT = AUDIT / "dataset_B_id_index_task_map.csv"
SEQ_OUT = AUDIT / "dataset_B_event_sequence_summary.csv"
EVENT_ROWS_OUT = AUDIT / "dataset_B_event_rows.csv"

long_df = pd.read_csv(LONG)
short_df = pd.read_csv(SHORT)

lines = []
def add(s=""):
    lines.append(str(s))

# -------------------------------------------------------------------------
# Basic coercion
# -------------------------------------------------------------------------
for c in ["ID", "Index", "Time", "Time1", "Time2", "State", "Fatigue"]:
    if c in long_df.columns:
        long_df[c] = pd.to_numeric(long_df[c], errors="coerce")

for c in ["ID", "Index", "Time", "Fatigue", "Fatigue_rate", "Age", "Height_cm", "Weight_kg"]:
    if c in short_df.columns:
        short_df[c] = pd.to_numeric(short_df[c], errors="coerce")

# -------------------------------------------------------------------------
# ID -> Index -> Task mapping
# -------------------------------------------------------------------------
long_map = (
    long_df.groupby(["ID", "Index", "Task"], as_index=False)
    .agg(
        n_long_rows=("State", "size"),
        event_rows=("State", "sum"),
        fatigue_flag_min=("Fatigue", "min"),
        fatigue_flag_max=("Fatigue", "max"),
        first_time1=("Time1", "min"),
        last_time2=("Time2", "max"),
        short_time_from_long=("Time", "first"),
    )
)

short_cols = [c for c in [
    "ID", "Index", "Subject", "Task", "Time", "Fatigue", "Fatigue_rate",
    "Resting_hr", "Sex", "Age", "Height_cm", "Weight_kg"
] if c in short_df.columns]

short_one = short_df[short_cols].drop_duplicates(subset=["ID"]).copy()

mapping = long_map.merge(
    short_one,
    on="ID",
    how="left",
    suffixes=("_long", "_short"),
    validate="one_to_one",
)
mapping.to_csv(MAP_OUT, index=False)

# -------------------------------------------------------------------------
# Sequence-level event semantics
# -------------------------------------------------------------------------
seq_rows = []

for (idv, idx, task), g in long_df.groupby(["ID", "Index", "Task"]):
    g = g.sort_values(["Time1", "Time2"]).copy()

    state = pd.to_numeric(g["State"], errors="coerce").fillna(0)
    fatigue = pd.to_numeric(g["Fatigue"], errors="coerce")

    event_pos = np.where(state.to_numpy() == 1)[0]
    event_count = len(event_pos)

    if event_count:
        first_event_row = g.iloc[event_pos[0]]
        event_time1 = first_event_row["Time1"]
        event_time2 = first_event_row["Time2"]
        event_is_last_row = bool(event_pos[0] == len(g) - 1)
    else:
        event_time1 = np.nan
        event_time2 = np.nan
        event_is_last_row = np.nan

    intervals_connected = True
    if len(g) > 1:
        prev_stop = g["Time2"].to_numpy()[:-1]
        next_start = g["Time1"].to_numpy()[1:]
        intervals_connected = bool(np.allclose(prev_stop, next_start, rtol=0, atol=1e-9))

    seq_rows.append({
        "ID": idv,
        "Index": idx,
        "Task": task,
        "n_rows": len(g),
        "event_count": event_count,
        "event_time1": event_time1,
        "event_time2": event_time2,
        "event_is_last_row": event_is_last_row,
        "fatigue_min": fatigue.min(),
        "fatigue_max": fatigue.max(),
        "fatigue_unique": fatigue.nunique(dropna=True),
        "subject_level_fatigue_flag": int(fatigue.max()) if fatigue.notna().any() else np.nan,
        "state_max": int(state.max()) if len(state) else np.nan,
        "intervals_connected": intervals_connected,
        "first_time1": g["Time1"].min(),
        "last_time2": g["Time2"].max(),
        "short_time_like": g["Time"].iloc[0] if len(g) else np.nan,
    })

seq = pd.DataFrame(seq_rows)
seq.to_csv(SEQ_OUT, index=False)

event_rows = long_df[long_df["State"] == 1].copy()
event_rows.to_csv(EVENT_ROWS_OUT, index=False)

# -------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------
add("DATASET B — TEMPORAL / PARTICIPANT SEMANTICS AUDIT")
add("=" * 118)
add(f"Long rows: {len(long_df)}")
add(f"Short rows: {len(short_df)}")
add(f"Task-specific IDs: {long_df['ID'].nunique()}")
add(f"Underlying participant Index values: {long_df['Index'].nunique()}")
add()

add("1. ID -> INDEX -> TASK MAPPING")
add("-" * 118)
show_cols = [c for c in [
    "ID","Index_long","Task_long","Subject","Index_short","Task_short",
    "n_long_rows","event_rows","fatigue_flag_min","fatigue_flag_max",
    "first_time1","last_time2","short_time_from_long","Time","Fatigue","Fatigue_rate"
] if c in mapping.columns]
add(mapping[show_cols].to_string(index=False))
add()

# Check mapping uniqueness
id_to_index = long_df.groupby("ID")["Index"].nunique()
id_to_task = long_df.groupby("ID")["Task"].nunique()

add(f"IDs mapping to >1 Index: {int((id_to_index > 1).sum())}")
add(f"IDs mapping to >1 Task: {int((id_to_task > 1).sum())}")
add()

add("2. UNDERLYING PARTICIPANT TASK COVERAGE")
add("-" * 118)
coverage = (
    long_df[["Index","Task","ID"]]
    .drop_duplicates()
    .groupby("Index")
    .agg(
        n_tasks=("Task","nunique"),
        tasks=("Task", lambda x: ",".join(sorted(set(map(str,x))))),
        ids=("ID", lambda x: ",".join(map(str, sorted(set(x))))),
    )
    .reset_index()
)
add(coverage.to_string(index=False))
add()
add(f"Participants completing both MMH and SI: {int((coverage['n_tasks'] == 2).sum())}/{len(coverage)}")
add(f"Participants completing only one task: {int((coverage['n_tasks'] == 1).sum())}/{len(coverage)}")
add()

add("3. STATE / FATIGUE SEMANTICS")
add("-" * 118)
add(seq[[
    "ID","Index","Task","n_rows","event_count","event_time1","event_time2",
    "event_is_last_row","fatigue_unique","subject_level_fatigue_flag","state_max",
    "intervals_connected","first_time1","last_time2","short_time_like"
]].to_string(index=False))
add()

add(f"Task-trials with >=1 State event: {int((seq['event_count'] > 0).sum())}/{len(seq)}")
add(f"Task-trials with exactly 1 State event: {int((seq['event_count'] == 1).sum())}/{len(seq)}")
add(f"Task-trials with >1 State event: {int((seq['event_count'] > 1).sum())}/{len(seq)}")
add(f"State events that occur on last available interval: "
    f"{int((seq.loc[seq.event_count>0,'event_is_last_row'] == True).sum())}/"
    f"{int((seq.event_count>0).sum())}")
add(f"Sequences with disconnected 10-min intervals: {int((seq['intervals_connected'] == False).sum())}")
add()

# Compare Fatigue vs whether an event occurred.
seq["event_any"] = (seq["event_count"] > 0).astype(int)
fatigue_event_match = (
    seq["subject_level_fatigue_flag"].astype(float) == seq["event_any"].astype(float)
)
add(f"Subject/task-level Fatigue flag equals presence of any State event: "
    f"{int(fatigue_event_match.sum())}/{len(seq)}")
add()

add("4. EVENT TIME DISTRIBUTION")
add("-" * 118)
events = seq[seq["event_count"] > 0].copy()
if len(events):
    add(f"Event task-trials: {len(events)}")
    add(f"Event Time2: min={events['event_time2'].min()}, "
        f"median={events['event_time2'].median()}, max={events['event_time2'].max()}")
    add("Event Time2 counts:")
    add(events["event_time2"].value_counts().sort_index().to_string())
else:
    add("No events detected.")
add()

add("5. LONG-vs-SHORT CONSISTENCY")
add("-" * 118)

# Construct comparable columns from sequence summary and short table.
cmp = seq.merge(
    short_df[[c for c in ["ID","Index","Task","Time","Fatigue"] if c in short_df.columns]],
    on="ID",
    how="left",
    suffixes=("_seq","_short"),
)

checks = {}

if "Index_short" in cmp.columns:
    checks["Index"] = np.isclose(
        pd.to_numeric(cmp["Index_seq"], errors="coerce"),
        pd.to_numeric(cmp["Index_short"], errors="coerce"),
        equal_nan=False
    )
if "Task_short" in cmp.columns:
    checks["Task"] = cmp["Task_seq"].astype(str) == cmp["Task_short"].astype(str)
if "Time" in cmp.columns:
    checks["Time"] = np.isclose(
        pd.to_numeric(cmp["short_time_like"], errors="coerce"),
        pd.to_numeric(cmp["Time"], errors="coerce"),
        equal_nan=False
    )
if "Fatigue_short" in cmp.columns:
    checks["Fatigue"] = np.isclose(
        pd.to_numeric(cmp["subject_level_fatigue_flag"], errors="coerce"),
        pd.to_numeric(cmp["Fatigue_short"], errors="coerce"),
        equal_nan=False
    )

for name, mask in checks.items():
    add(f"{name} agreement: {int(np.sum(mask))}/{len(mask)}")
add()

add("6. PREDICTION-LEAKAGE IMPLICATIONS")
add("-" * 118)
add("Participant grouping:")
add("  Use Index, NOT ID. ID is task-specific; Index represents the underlying worker used by the official split code.")
add()
add("Do NOT use as predictors:")
add("  - Fatigue: task-trial-level future/event flag if the audit confirms it equals whether fatigue eventually occurred.")
add("  - State: interval event label / target.")
add("  - Time: final task-trial event/censoring time if it is constant within ID and therefore contains future information.")
add()
add("Potential low-cost context:")
add("  - Task")
add("  - current interval start/end time (Time1 / Time2), interpreted as elapsed protocol exposure")
add("  - demographics merged from short form (Sex, Age, Height_cm, Weight_kg)")
add("  - Resting_hr should be analyzed separately because it is a physiological wearable-derived baseline, not pure task context.")
add()
add("Wearable features:")
add("  - HRRMean, HRRCV, HRRSD")
add("  - wrist/hip/chest/ankle jerk, acceleration, posture summaries")
add()
add("Primary candidate target if semantics pass:")
add("  State = 1 for fatigue onset in the current 10-min interval among at-risk observations.")
add("  This yields a discrete-time fatigue-onset prediction problem.")
add()

add("7. DATASET-A COMPARABILITY")
add("-" * 118)
add("Dataset B does NOT preserve a continuous per-interval RPE score in the released long-form table.")
add("Therefore it cannot reproduce Dataset A's continuous final-fatigue regression estimand exactly.")
add("The common estimand should instead be framed at the higher level:")
add("  incremental predictive value of wearable information beyond low-cost worker/task/exposure context")
add("under participant-independent validation.")
add("Dataset-specific outcomes/metrics should be retained rather than forcing incompatible labels into one scale.")
add()

add("=" * 118)
add("END OF AUDIT")

TXT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("Dataset B temporal semantics audit complete:")
print(f"  {TXT_OUT}")
print(f"  {MAP_OUT}")
print(f"  {SEQ_OUT}")
print(f"  {EVENT_ROWS_OUT}")
print("")
print("Upload dataset_B_temporal_semantics_audit.txt to ChatGPT.")
