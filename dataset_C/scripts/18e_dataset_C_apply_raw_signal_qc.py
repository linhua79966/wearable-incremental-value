
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived" / "dataset_C_phase1"
AUDIT = ROOT / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)

START_IN = DERIVED / "dataset_C_handcrafted_start_anchor.csv"
END_IN = DERIVED / "dataset_C_handcrafted_end_anchor.csv"

START_OUT = DERIVED / "dataset_C_handcrafted_start_anchor_qc.csv"
END_OUT = DERIVED / "dataset_C_handcrafted_end_anchor_qc.csv"
MANIFEST_OUT = AUDIT / "dataset_C_raw_signal_qc_manifest.csv"
AUDIT_OUT = AUDIT / "dataset_C_raw_signal_qc_application_audit.txt"

# =============================================================================
# Outcome-independent QC manifest
#
# These rows were identified only after raw-signal inspection:
#
# 1) Subject 3 / task2_45i:
#    EMG anterior-deltoid and infraspinatus contain catastrophic raw corruption
#    at the beginning of the file. Under START anchoring this contaminates
#    the [0,10] s wearable window used by forecast row t=10.
#
# 2) Subject 27 / task1_35i:
#    Sternum accelerometer contains a two-sample spike at raw time
#    50.90-50.91 s. This contaminates START t=60 and END t=50 windows.
#
# To preserve exact paired START/END samples, use the UNION of all affected
# forecast rows and remove those row_ids from BOTH tables.
#
# No outcome values or model results are used in this rule.
# =============================================================================

qc_manifest = pd.DataFrame([
    {
        "row_id": "S03__task2_45i__t10",
        "subject_num": 3,
        "task_label": "task2_45i",
        "affected_anchor": "START",
        "raw_modality": "EMG",
        "raw_sensor_or_channel": "anterior_deltoid + infraspinatus",
        "raw_corrupt_segment": "file start; gross corruption concentrated in initial segment",
        "qc_action": "remove forecast row from BOTH anchors to preserve paired cohort",
        "qc_basis": "raw-signal integrity only; independent of target/model performance",
    },
    {
        "row_id": "S27__task1_35i__t50",
        "subject_num": 27,
        "task_label": "task1_35i",
        "affected_anchor": "END",
        "raw_modality": "IMU acceleration",
        "raw_sensor_or_channel": "Sternum acc_x/acc_y/acc_z",
        "raw_corrupt_segment": "50.90-50.91 s; two-sample multiaxis spike",
        "qc_action": "remove forecast row from BOTH anchors to preserve paired cohort",
        "qc_basis": "raw-signal integrity only; independent of target/model performance",
    },
    {
        "row_id": "S27__task1_35i__t60",
        "subject_num": 27,
        "task_label": "task1_35i",
        "affected_anchor": "START",
        "raw_modality": "IMU acceleration",
        "raw_sensor_or_channel": "Sternum acc_x/acc_y/acc_z",
        "raw_corrupt_segment": "50.90-50.91 s; two-sample multiaxis spike",
        "qc_action": "remove forecast row from BOTH anchors to preserve paired cohort",
        "qc_basis": "raw-signal integrity only; independent of target/model performance",
    },
])

qc_manifest.to_csv(MANIFEST_OUT, index=False)

# =============================================================================
# Load and verify paired input
# =============================================================================
start = pd.read_csv(START_IN)
end = pd.read_csv(END_IN)

required = {"row_id", "subject_num", "task_label", "current_time_sec", "target_borg"}
for label, df in [("START", start), ("END", end)]:
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{label}: missing required columns {sorted(missing)}")
    if df["row_id"].duplicated().any():
        raise SystemExit(f"{label}: duplicate row_id values.")

start = start.sort_values("row_id").reset_index(drop=True)
end = end.sort_values("row_id").reset_index(drop=True)

if list(start["row_id"]) != list(end["row_id"]):
    raise SystemExit("START and END input row_id sets/order differ.")

if not np.allclose(
    start["target_borg"].to_numpy(dtype=float),
    end["target_borg"].to_numpy(dtype=float),
    equal_nan=True,
):
    raise SystemExit("START and END targets differ before QC.")

qc_ids = set(qc_manifest["row_id"])

# Every documented QC row must exist.
missing_qc = sorted(qc_ids - set(start["row_id"]))
if missing_qc:
    raise SystemExit(f"QC row_ids not found in source tables: {missing_qc}")

# Capture pre-QC rows for audit.
pre_rows = start[start["row_id"].isin(qc_ids)][
    ["row_id","subject_num","task_label","current_time_sec","target_time_sec",
     "current_borg","target_borg"]
].copy()

# =============================================================================
# Apply exact union-row quarantine to BOTH anchors
# =============================================================================
start_qc = start[~start["row_id"].isin(qc_ids)].copy()
end_qc = end[~end["row_id"].isin(qc_ids)].copy()

start_qc = start_qc.sort_values("row_id").reset_index(drop=True)
end_qc = end_qc.sort_values("row_id").reset_index(drop=True)

if list(start_qc["row_id"]) != list(end_qc["row_id"]):
    raise RuntimeError("START/END row sets differ after QC.")

if not np.allclose(
    start_qc["target_borg"].to_numpy(dtype=float),
    end_qc["target_borg"].to_numpy(dtype=float),
    equal_nan=True,
):
    raise RuntimeError("START/END targets differ after QC.")

if set(start_qc["row_id"]) & qc_ids:
    raise RuntimeError("QC rows remain in START.")
if set(end_qc["row_id"]) & qc_ids:
    raise RuntimeError("QC rows remain in END.")

start_qc.to_csv(START_OUT, index=False)
end_qc.to_csv(END_OUT, index=False)

# =============================================================================
# Audit
# =============================================================================
lines = []
A = lines.append

A("DATASET C — RAW-SIGNAL QC APPLICATION AUDIT")
A("=" * 118)
A(f"START rows before QC: {len(start)}")
A(f"END rows before QC: {len(end)}")
A(f"Union forecast rows quarantined: {len(qc_ids)}")
A(f"START rows after QC: {len(start_qc)}")
A(f"END rows after QC: {len(end_qc)}")
A(f"Participants retained: {start_qc['subject_num'].nunique()}")
A(
    "Subject-task trials represented after QC: "
    f"{start_qc[['subject_num','task_label']].drop_duplicates().shape[0]}"
)
A("")

A("1. SCIENTIFIC QC PRINCIPLE")
A("-" * 118)
A("QC is based exclusively on raw-signal integrity.")
A("No target value, prediction error, model coefficient, or model-performance change is used to define the quarantine.")
A("No signal value is clipped, winsorized, interpolated, replaced, or repaired.")
A("No participant is removed.")
A("Only forecast rows whose required raw wearable window overlaps a verified corrupted raw segment are quarantined.")
A("The UNION of affected START/END forecast rows is removed from BOTH anchor tables to preserve an exactly paired cohort.")
A("")

A("2. VERIFIED RAW CORRUPTION")
A("-" * 118)
A("Case A — Subject 3 / task2_45i / EMG:")
A("  Catastrophic corruption is concentrated at the beginning of anterior_deltoid.csv and infraspinatus.csv.")
A("  Under START anchoring, forecast row t=10 uses [0,10] s and therefore overlaps the corrupted segment.")
A("")
A("Case B — Subject 27 / task1_35i / Sternum acceleration:")
A("  A two-sample multiaxis spike occurs at raw time 50.90-50.91 s.")
A("  It contaminates START forecast row t=60 and END forecast row t=50.")
A("")

A("3. QUARANTINED FORECAST ROWS")
A("-" * 118)
A(pre_rows.to_string(index=False))
A("")

A("4. POST-QC PAIRED-DESIGN CHECKS")
A("-" * 118)
A(f"Identical START/END row_id sequence: {list(start_qc['row_id']) == list(end_qc['row_id'])}")
A(
    "Identical START/END targets: "
    f"{np.allclose(start_qc['target_borg'], end_qc['target_borg'], equal_nan=True)}"
)
A(f"QC rows absent from START: {len(set(start_qc['row_id']) & qc_ids) == 0}")
A(f"QC rows absent from END: {len(set(end_qc['row_id']) & qc_ids) == 0}")
A("")

A("5. POST-QC OUTCOME DISTRIBUTION")
A("-" * 118)
for c in ["current_borg", "target_borg", "borg_change_10s"]:
    if c in start_qc.columns:
        s = pd.to_numeric(start_qc[c], errors="coerce")
        A(
            f"{c}: n={s.notna().sum()}, unique={s.nunique(dropna=True)}, "
            f"min={s.min()}, median={s.median()}, mean={s.mean():.6f}, max={s.max()}"
        )
A("")

A("6. NEXT STEP")
A("-" * 118)
A("Rerun the complete dual-anchor repeated nested benchmark on these QC-clean tables.")
A("Use the same feature sets, models, participant-grouped validation, hyperparameter grids, metrics, and bootstrap design.")
A("Do not compare the cleaned model results to decide whether the QC was justified; QC justification is already established by raw data integrity.")
A("HGB must be rerun as well so the final Dataset C result table comes from one common QC-clean cohort.")
A("")

A("=" * 118)
A("END OF AUDIT")

AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("Dataset C raw-signal QC applied.")
print(f"  {START_OUT}")
print(f"  {END_OUT}")
print(f"  {MANIFEST_OUT}")
print(f"  {AUDIT_OUT}")
print("")
print("Upload dataset_C_raw_signal_qc_application_audit.txt to ChatGPT.")
