from pathlib import Path
import re
import time
import hashlib
import numpy as np
import pandas as pd

# =============================================================================
# Dataset C — Foundation-model raw-window input audit
#
# Purpose
#   Audit the LOCKED 18f QC-clean Dataset C cohort before any GPU/FM training.
#   This script:
#     - does NOT train a model;
#     - does NOT alter QC;
#     - does NOT delete rows based on outcomes or prediction performance;
#     - does NOT clip, winsorize, interpolate, repair, or normalize raw signals;
#     - reconstructs the exact prior [t-10,t] START/END raw windows;
#     - verifies EMG/IMU sample counts, finite values, temporal boundaries,
#       known quarantine handling, and START/END paired-cohort integrity.
#
# Locked scientific design
#   Outcome estimand: Borg(t+10)
#   Available state/context at t: current Borg + task/load/time context
#   Wearable interval: [t-10,t] only
#   EMG nominal rate: 1000 Hz
#   IMU nominal rate: 100 Hz
#   Window length: 10 s
#
# Expected locked cohort:
#   1141 paired rows, 34 participants
#
# Outputs
#   audit/dataset_C_foundation_raw_window_audit.txt
#   derived/dataset_C_foundation_raw_windows/
#       dataset_C_foundation_raw_window_manifest.csv
#       dataset_C_foundation_raw_window_issues.csv
#       dataset_C_foundation_raw_window_channel_summary.csv
#
# IMPORTANT
#   Subject 3 / task1_35i / Shoulder ACC+GYR is the previously documented
#   whole-location quarantine. It is recorded as unavailable/masked here and
#   must NOT be silently restored.
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "dataset_C_shoulder_rotation" / "raw" / "WSD4FEDSRM"
SR = DS / "EMG, IMU, and PPG data"
DERIVED = ROOT / "derived" / "dataset_C_phase1"

START_LOCKED = DERIVED / "dataset_C_handcrafted_start_anchor_qc.csv"
END_LOCKED = DERIVED / "dataset_C_handcrafted_end_anchor_qc.csv"

OUTDIR = ROOT / "derived" / "dataset_C_foundation_raw_windows"
AUDITDIR = ROOT / "audit"
OUTDIR.mkdir(parents=True, exist_ok=True)
AUDITDIR.mkdir(parents=True, exist_ok=True)

MANIFEST_OUT = OUTDIR / "dataset_C_foundation_raw_window_manifest.csv"
ISSUES_OUT = OUTDIR / "dataset_C_foundation_raw_window_issues.csv"
CHANNEL_SUMMARY_OUT = OUTDIR / "dataset_C_foundation_raw_window_channel_summary.csv"
AUDIT_OUT = AUDITDIR / "dataset_C_foundation_raw_window_audit.txt"

EMG_HZ = 1000
IMU_HZ = 100
WIN_SEC = 10.0

EXPECTED_ROWS = 1141
EXPECTED_PARTICIPANTS = 34
EXPECTED_EMG_SAMPLES = int(EMG_HZ * WIN_SEC)
EXPECTED_IMU_SAMPLES = int(IMU_HZ * WIN_SEC)

TASK = {
    "task1_35i": ("30-40_ internal rotation", "internal", 35),
    "task2_45i": ("40-50_ internal rotation", "internal", 45),
    "task3_55i": ("50-60_ internal rotation", "internal", 55),
    "task4_35e": ("30-40_ external rotation", "external", 35),
    "task5_45e": ("40-50_ external rotation", "external", 45),
    "task6_55e": ("50-60_ external rotation", "external", 55),
}

EMGS = [
    "anterior_deltoid.csv",
    "infraspinatus.csv",
    "latissimus_dorsi.csv",
    "pectoralis_major.csv",
    "posterior_deltoid.csv",
    "upper_trapezius.csv",
]

LOCS = [
    "Forearm",
    "Hand",
    "Pelvis",
    "Shoulder",
    "Sternum",
    "Upper arm",
]

KINDS = ["acc", "gyr"]

# Verified raw-signal QC rows removed by 18e/18f.
LOCKED_REMOVED_QC_IDS = {
    "S03__task2_45i__t10",
    "S27__task1_35i__t50",
    "S27__task1_35i__t60",
}

# Whole-location quarantine inherited from the primary handcrafted build.
def is_known_whole_location_quarantine(subject_num, task_label, location, kind):
    return (
        int(subject_num) == 3
        and str(task_label) == "task1_35i"
        and str(location) == "Shoulder"
        and str(kind) in {"acc", "gyr"}
    )


def safe(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def b_start(t, rate, n):
    a = int(round((float(t) - WIN_SEC) * rate))
    b = int(round(float(t) * rate))
    if a >= 0 and b <= n and b > a:
        return a, b
    return None


def b_end(t, dur, rate, n):
    a = int(round(n - (float(dur) - (float(t) - WIN_SEC)) * rate))
    b = int(round(n - (float(dur) - float(t)) * rate))
    if a >= 0 and b <= n and b > a:
        return a, b
    return None


def read_numeric_csv(fp, ncols=None):
    df = pd.read_csv(fp)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    arr = df.to_numpy(dtype=float)
    if ncols is not None:
        if arr.ndim != 2 or arr.shape[1] < ncols:
            raise ValueError(f"{fp}: expected >= {ncols} numeric columns, got shape={arr.shape}")
        arr = arr[:, :ncols]
    return arr


def finite_stats(x):
    a = np.asarray(x, dtype=float)
    finite = np.isfinite(a)
    vals = a[finite]
    out = {
        "n_values": int(a.size),
        "n_finite": int(finite.sum()),
        "n_nonfinite": int((~finite).sum()),
        "finite_fraction": float(finite.mean()) if a.size else np.nan,
        "finite_abs_max": float(np.max(np.abs(vals))) if vals.size else np.nan,
        "finite_abs_q999": float(np.quantile(np.abs(vals), 0.999)) if vals.size else np.nan,
        "finite_min": float(np.min(vals)) if vals.size else np.nan,
        "finite_max": float(np.max(vals)) if vals.size else np.nan,
    }
    return out


def hash_row_ids(values):
    s = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


t0 = time.time()

# =============================================================================
# 1. Load and validate the locked 18f paired cohort
# =============================================================================
for fp in [START_LOCKED, END_LOCKED]:
    if not fp.exists():
        raise SystemExit(f"Missing locked 18f input table: {fp}")

start = pd.read_csv(START_LOCKED)
end = pd.read_csv(END_LOCKED)

required = {
    "row_id",
    "subject_num",
    "task_label",
    "current_time_sec",
    "target_time_sec",
    "current_borg",
    "target_borg",
    "trial_duration_sec_qc",
}

for label, df in [("START", start), ("END", end)]:
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{label}: missing required locked columns: {sorted(missing)}")
    if df["row_id"].duplicated().any():
        raise SystemExit(f"{label}: duplicate row_id values found.")

start = start.sort_values("row_id").reset_index(drop=True)
end = end.sort_values("row_id").reset_index(drop=True)

if list(start["row_id"]) != list(end["row_id"]):
    raise SystemExit("Locked START/END row_id sets/order differ.")

if not np.array_equal(
    start["subject_num"].to_numpy(),
    end["subject_num"].to_numpy(),
):
    raise SystemExit("Locked START/END participant vectors differ.")

if not np.allclose(
    start["target_borg"].to_numpy(dtype=float),
    end["target_borg"].to_numpy(dtype=float),
    equal_nan=True,
):
    raise SystemExit("Locked START/END targets differ.")

if not np.allclose(
    start["current_time_sec"].to_numpy(dtype=float),
    end["current_time_sec"].to_numpy(dtype=float),
    equal_nan=True,
):
    raise SystemExit("Locked START/END current_time_sec differs.")

if not np.allclose(
    start["trial_duration_sec_qc"].to_numpy(dtype=float),
    end["trial_duration_sec_qc"].to_numpy(dtype=float),
    equal_nan=True,
):
    raise SystemExit("Locked START/END trial duration differs.")

locked_ids = set(start["row_id"].astype(str))
removed_still_present = sorted(LOCKED_REMOVED_QC_IDS & locked_ids)
if removed_still_present:
    raise SystemExit(
        "18e/18f quarantined row IDs unexpectedly remain in locked cohort: "
        + ", ".join(removed_still_present)
    )

# =============================================================================
# 2. Trial-level raw cache
# =============================================================================
trial_cache = {}
file_issues = []

def load_trial(subject_num, task_label):
    key = (int(subject_num), str(task_label))
    if key in trial_cache:
        return trial_cache[key]

    if task_label not in TASK:
        raise KeyError(f"Unknown task label: {task_label}")

    folder, _, _ = TASK[task_label]
    td = SR / folder / f"Subject {int(subject_num)}"

    trial = {"emg": {}, "imu": {}, "root": td}

    for fn in EMGS:
        fp = td / "EMG data" / fn
        if not fp.exists():
            file_issues.append({
                "scope": "file",
                "subject_num": int(subject_num),
                "task_label": str(task_label),
                "anchor": "",
                "row_id": "",
                "modality": "EMG",
                "channel": Path(fn).stem,
                "issue": "missing_file",
                "detail": str(fp),
            })
            trial["emg"][fn] = None
        else:
            try:
                trial["emg"][fn] = read_numeric_csv(fp, ncols=1)[:, 0]
            except Exception as e:
                file_issues.append({
                    "scope": "file",
                    "subject_num": int(subject_num),
                    "task_label": str(task_label),
                    "anchor": "",
                    "row_id": "",
                    "modality": "EMG",
                    "channel": Path(fn).stem,
                    "issue": "read_error",
                    "detail": repr(e),
                })
                trial["emg"][fn] = None

    for loc in LOCS:
        lf = safe(loc)
        for kind in KINDS:
            k = (loc, kind)

            if is_known_whole_location_quarantine(subject_num, task_label, loc, kind):
                trial["imu"][k] = {
                    "status": "known_quarantine",
                    "array": None,
                    "path": str(td / "IMU data" / loc / f"{kind}_{lf}.csv"),
                }
                continue

            fp = td / "IMU data" / loc / f"{kind}_{lf}.csv"
            if not fp.exists():
                file_issues.append({
                    "scope": "file",
                    "subject_num": int(subject_num),
                    "task_label": str(task_label),
                    "anchor": "",
                    "row_id": "",
                    "modality": f"IMU_{kind.upper()}",
                    "channel": loc,
                    "issue": "missing_file",
                    "detail": str(fp),
                })
                trial["imu"][k] = {
                    "status": "missing_file",
                    "array": None,
                    "path": str(fp),
                }
            else:
                try:
                    arr = read_numeric_csv(fp, ncols=3)
                    trial["imu"][k] = {
                        "status": "available",
                        "array": arr,
                        "path": str(fp),
                    }
                except Exception as e:
                    file_issues.append({
                        "scope": "file",
                        "subject_num": int(subject_num),
                        "task_label": str(task_label),
                        "anchor": "",
                        "row_id": "",
                        "modality": f"IMU_{kind.upper()}",
                        "channel": loc,
                        "issue": "read_error",
                        "detail": repr(e),
                    })
                    trial["imu"][k] = {
                        "status": "read_error",
                        "array": None,
                        "path": str(fp),
                    }

    trial_cache[key] = trial
    return trial


# =============================================================================
# 3. Window-level audit on the exact locked 1141 rows
# =============================================================================
manifest_rows = []
issue_rows = list(file_issues)
channel_rows = []

def add_issue(row, anchor, modality, channel, issue, detail):
    issue_rows.append({
        "scope": "window",
        "subject_num": int(row.subject_num),
        "task_label": str(row.task_label),
        "anchor": anchor,
        "row_id": str(row.row_id),
        "modality": modality,
        "channel": channel,
        "issue": issue,
        "detail": str(detail),
    })


for idx, row in start.iterrows():
    sn = int(row.subject_num)
    task = str(row.task_label)
    t = float(row.current_time_sec)
    dur = float(row.trial_duration_sec_qc)
    rid = str(row.row_id)

    if not np.isclose(float(row.target_time_sec), t + 10.0, atol=1e-9):
        add_issue(
            row, "BOTH", "TARGET", "",
            "target_time_not_exact_plus_10s",
            f"current={t}, target={row.target_time_sec}",
        )

    if t < WIN_SEC:
        add_issue(
            row, "BOTH", "TIME", "",
            "current_time_before_full_prior_window",
            f"t={t}",
        )

    trial = load_trial(sn, task)

    for anchor in ["START", "END"]:
        anchor_key = anchor.lower()

        n_available_channels = 0
        n_known_masked_channels = 0
        n_missing_or_error_channels = 0
        n_window_nonfinite_values = 0
        all_window_shapes_ok = True
        all_available_windows_finite = True

        # -------------------------
        # EMG: six 1-D channels
        # -------------------------
        for fn in EMGS:
            ch = Path(fn).stem
            arr = trial["emg"].get(fn)

            if arr is None:
                n_missing_or_error_channels += 1
                all_window_shapes_ok = False
                all_available_windows_finite = False
                add_issue(row, anchor, "EMG", ch, "channel_unavailable", "raw file unavailable")
                continue

            bounds = (
                b_start(t, EMG_HZ, len(arr))
                if anchor_key == "start"
                else b_end(t, dur, EMG_HZ, len(arr))
            )

            if bounds is None:
                n_missing_or_error_channels += 1
                all_window_shapes_ok = False
                all_available_windows_finite = False
                add_issue(
                    row, anchor, "EMG", ch,
                    "window_out_of_bounds",
                    f"n={len(arr)}, t={t}, dur={dur}",
                )
                continue

            a, b = bounds
            w = arr[a:b]
            st = finite_stats(w)

            shape_ok = len(w) == EXPECTED_EMG_SAMPLES
            finite_ok = st["n_nonfinite"] == 0

            n_available_channels += 1
            n_window_nonfinite_values += st["n_nonfinite"]
            all_window_shapes_ok &= shape_ok
            all_available_windows_finite &= finite_ok

            if not shape_ok:
                add_issue(
                    row, anchor, "EMG", ch,
                    "unexpected_sample_count",
                    f"got={len(w)}, expected={EXPECTED_EMG_SAMPLES}, bounds=({a},{b})",
                )
            if not finite_ok:
                add_issue(
                    row, anchor, "EMG", ch,
                    "nonfinite_raw_values",
                    f"count={st['n_nonfinite']}",
                )

            channel_rows.append({
                "row_id": rid,
                "subject_num": sn,
                "task_label": task,
                "anchor": anchor,
                "modality": "EMG",
                "channel": ch,
                "sampling_rate_hz": EMG_HZ,
                "window_start_index": a,
                "window_end_index_exclusive": b,
                "n_samples": len(w),
                "expected_n_samples": EXPECTED_EMG_SAMPLES,
                "shape_ok": int(shape_ok),
                "status": "available",
                **st,
            })

        # -----------------------------------------------
        # IMU: 6 locations x ACC/GYR; each has 3 raw axes
        # -----------------------------------------------
        for loc in LOCS:
            for kind in KINDS:
                obj = trial["imu"].get((loc, kind), {})
                status = obj.get("status", "missing")
                arr = obj.get("array")

                if status == "known_quarantine":
                    n_known_masked_channels += 1
                    channel_rows.append({
                        "row_id": rid,
                        "subject_num": sn,
                        "task_label": task,
                        "anchor": anchor,
                        "modality": f"IMU_{kind.upper()}",
                        "channel": loc,
                        "sampling_rate_hz": IMU_HZ,
                        "window_start_index": np.nan,
                        "window_end_index_exclusive": np.nan,
                        "n_samples": np.nan,
                        "expected_n_samples": EXPECTED_IMU_SAMPLES,
                        "shape_ok": np.nan,
                        "status": "known_quarantine_mask",
                        "n_values": np.nan,
                        "n_finite": np.nan,
                        "n_nonfinite": np.nan,
                        "finite_fraction": np.nan,
                        "finite_abs_max": np.nan,
                        "finite_abs_q999": np.nan,
                        "finite_min": np.nan,
                        "finite_max": np.nan,
                    })
                    continue

                if arr is None:
                    n_missing_or_error_channels += 1
                    all_window_shapes_ok = False
                    all_available_windows_finite = False
                    add_issue(
                        row, anchor, f"IMU_{kind.upper()}", loc,
                        "channel_unavailable",
                        f"status={status}",
                    )
                    continue

                bounds = (
                    b_start(t, IMU_HZ, len(arr))
                    if anchor_key == "start"
                    else b_end(t, dur, IMU_HZ, len(arr))
                )

                if bounds is None:
                    n_missing_or_error_channels += 1
                    all_window_shapes_ok = False
                    all_available_windows_finite = False
                    add_issue(
                        row, anchor, f"IMU_{kind.upper()}", loc,
                        "window_out_of_bounds",
                        f"n={len(arr)}, t={t}, dur={dur}",
                    )
                    continue

                a, b = bounds
                w = arr[a:b, :3]
                st = finite_stats(w)

                shape_ok = w.shape == (EXPECTED_IMU_SAMPLES, 3)
                finite_ok = st["n_nonfinite"] == 0

                n_available_channels += 1
                n_window_nonfinite_values += st["n_nonfinite"]
                all_window_shapes_ok &= shape_ok
                all_available_windows_finite &= finite_ok

                if not shape_ok:
                    add_issue(
                        row, anchor, f"IMU_{kind.upper()}", loc,
                        "unexpected_sample_count",
                        f"got_shape={w.shape}, expected=({EXPECTED_IMU_SAMPLES},3), bounds=({a},{b})",
                    )
                if not finite_ok:
                    add_issue(
                        row, anchor, f"IMU_{kind.upper()}", loc,
                        "nonfinite_raw_values",
                        f"count={st['n_nonfinite']}",
                    )

                channel_rows.append({
                    "row_id": rid,
                    "subject_num": sn,
                    "task_label": task,
                    "anchor": anchor,
                    "modality": f"IMU_{kind.upper()}",
                    "channel": loc,
                    "sampling_rate_hz": IMU_HZ,
                    "window_start_index": a,
                    "window_end_index_exclusive": b,
                    "n_samples": w.shape[0],
                    "expected_n_samples": EXPECTED_IMU_SAMPLES,
                    "shape_ok": int(shape_ok),
                    "status": "available",
                    **st,
                })

        manifest_rows.append({
            "row_id": rid,
            "subject_num": sn,
            "task_label": task,
            "anchor": anchor,
            "current_time_sec": t,
            "target_time_sec": float(row.target_time_sec),
            "trial_duration_sec_qc": dur,
            "current_borg": float(row.current_borg),
            "target_borg": float(row.target_borg),
            "emg_expected_shape": f"6x{EXPECTED_EMG_SAMPLES}",
            "imu_expected_shape": f"12_streams_x{EXPECTED_IMU_SAMPLES}x3",
            "n_available_streams": n_available_channels,
            "n_known_masked_streams": n_known_masked_channels,
            "n_missing_or_error_streams": n_missing_or_error_channels,
            "window_nonfinite_value_count": n_window_nonfinite_values,
            "all_available_window_shapes_ok": int(all_window_shapes_ok),
            "all_available_windows_finite": int(all_available_windows_finite),
            "foundation_input_preflight_pass": int(
                all_window_shapes_ok
                and all_available_windows_finite
                and n_missing_or_error_channels == 0
            ),
        })

    if (idx + 1) % 100 == 0 or (idx + 1) == len(start):
        print(f"Audited locked forecast rows {idx+1}/{len(start)}")

manifest = pd.DataFrame(manifest_rows)
channels = pd.DataFrame(channel_rows)
issues = pd.DataFrame(issue_rows)

# =============================================================================
# 4. Cross-anchor consistency checks
# =============================================================================
if len(manifest) != 2 * len(start):
    raise RuntimeError(
        f"Expected {2 * len(start)} manifest rows (START+END), found {len(manifest)}."
    )

start_manifest = (
    manifest[manifest["anchor"] == "START"]
    .sort_values("row_id")
    .reset_index(drop=True)
)
end_manifest = (
    manifest[manifest["anchor"] == "END"]
    .sort_values("row_id")
    .reset_index(drop=True)
)

if list(start_manifest["row_id"]) != list(end_manifest["row_id"]):
    raise RuntimeError("Raw-window audit START/END row IDs differ.")

# Known whole-location quarantine should affect:
# Subject 3 / task1_35i rows only, 2 streams per anchor (Shoulder ACC + GYR).
known_mask = channels["status"].eq("known_quarantine_mask") if len(channels) else pd.Series(dtype=bool)
known_mask_df = channels.loc[known_mask].copy() if len(channels) else pd.DataFrame()

unexpected_mask_rows = pd.DataFrame()
if len(known_mask_df):
    unexpected_mask_rows = known_mask_df[
        ~(
            (known_mask_df["subject_num"] == 3)
            & (known_mask_df["task_label"] == "task1_35i")
            & (known_mask_df["channel"] == "Shoulder")
            & (known_mask_df["modality"].isin(["IMU_ACC", "IMU_GYR"]))
        )
    ].copy()

if len(unexpected_mask_rows):
    raise RuntimeError("Unexpected known-quarantine mask rows were generated.")

# =============================================================================
# 5. Compact channel summary
# =============================================================================
if len(channels):
    channel_summary = (
        channels.groupby(
            ["anchor", "modality", "channel", "status"],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_windows=("row_id", "count"),
            n_shape_fail=("shape_ok", lambda s: int(pd.to_numeric(s, errors="coerce").eq(0).sum())),
            total_nonfinite_values=("n_nonfinite", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            max_abs_value=("finite_abs_max", "max"),
            q999_abs_value_max=("finite_abs_q999", "max"),
        )
    )
else:
    channel_summary = pd.DataFrame()

# =============================================================================
# 6. Save outputs
# =============================================================================
manifest.to_csv(MANIFEST_OUT, index=False)
channels.to_csv(CHANNEL_SUMMARY_OUT, index=False)

if len(issues):
    issues.to_csv(ISSUES_OUT, index=False)
else:
    pd.DataFrame(
        columns=[
            "scope", "subject_num", "task_label", "anchor", "row_id",
            "modality", "channel", "issue", "detail",
        ]
    ).to_csv(ISSUES_OUT, index=False)

# =============================================================================
# 7. Audit report
# =============================================================================
lines = []
A = lines.append

A("DATASET C — FOUNDATION-MODEL RAW-WINDOW INPUT AUDIT")
A("=" * 118)
A(f"Locked START rows: {len(start)}")
A(f"Locked END rows: {len(end)}")
A(f"Participants: {start['subject_num'].nunique()}")
A(f"Subject-task trials represented: {start[['subject_num','task_label']].drop_duplicates().shape[0]}")
A(f"START/END identical row IDs: {list(start['row_id']) == list(end['row_id'])}")
A(f"START/END identical targets: {np.allclose(start['target_borg'], end['target_borg'], equal_nan=True)}")
A(f"Locked row-id SHA256: {hash_row_ids(start['row_id'])}")
A("")

A("1. FIXED FOUNDATION INPUT DESIGN")
A("-" * 118)
A(f"EMG nominal sampling rate: {EMG_HZ} Hz")
A(f"IMU nominal sampling rate: {IMU_HZ} Hz")
A(f"Prior raw-signal window: {WIN_SEC:.0f} s")
A(f"Expected EMG samples per muscle/window: {EXPECTED_EMG_SAMPLES}")
A(f"Expected IMU samples per ACC/GYR stream/window: {EXPECTED_IMU_SAMPLES}")
A("Raw window uses [t-10,t] only; target remains Borg(t+10).")
A("PPG and magnetometer remain outside this primary preflight.")
A("No training, normalization, clipping, winsorization, interpolation, repair, or target-driven QC is performed.")
A("")

A("2. LOCKED-COHORT GUARDRAILS")
A("-" * 118)
A(f"Expected QC-clean rows: {EXPECTED_ROWS}; observed: {len(start)}")
A(f"Expected participants: {EXPECTED_PARTICIPANTS}; observed: {start['subject_num'].nunique()}")
A(f"18e/18f quarantined row IDs still present: {len(removed_still_present)}")
for rid in sorted(LOCKED_REMOVED_QC_IDS):
    A(f"  excluded as expected: {rid} -> {rid not in locked_ids}")
A("")

A("3. START/END FOUNDATION INPUT PREFLIGHT")
A("-" * 118)
for anchor in ["START", "END"]:
    m = manifest[manifest["anchor"] == anchor]
    A(f"[{anchor}] windows audited: {len(m)}")
    A(f"[{anchor}] all available shapes OK: {int(m['all_available_window_shapes_ok'].sum())}/{len(m)}")
    A(f"[{anchor}] all available windows finite: {int(m['all_available_windows_finite'].sum())}/{len(m)}")
    A(f"[{anchor}] zero unexpected missing/error streams: {int((m['n_missing_or_error_streams'] == 0).sum())}/{len(m)}")
    A(f"[{anchor}] foundation-input preflight PASS rows: {int(m['foundation_input_preflight_pass'].sum())}/{len(m)}")
    A(f"[{anchor}] rows with known masked streams: {int((m['n_known_masked_streams'] > 0).sum())}")
    A(f"[{anchor}] total known masked stream-windows: {int(m['n_known_masked_streams'].sum())}")
    A(f"[{anchor}] total raw non-finite values in available windows: {int(m['window_nonfinite_value_count'].sum())}")
A("")

A("4. KNOWN WHOLE-LOCATION QUARANTINE")
A("-" * 118)
A("Inherited rule: Subject 3 / task1_35i / Shoulder ACC+GYR is unavailable by design.")
A("Those streams are recorded as known_quarantine_mask and are NOT silently restored.")
A(f"Known masked channel-windows recorded: {len(known_mask_df)}")
if len(known_mask_df):
    A(
        "Rows affected (unique row_id): "
        f"{known_mask_df['row_id'].nunique()}"
    )
    A(
        "Affected anchors: "
        + ", ".join(sorted(known_mask_df["anchor"].unique().astype(str)))
    )
A("")

A("5. ISSUE SUMMARY")
A("-" * 118)
if len(issues):
    A(f"Total issue records: {len(issues)}")
    for (scope, issue), g in issues.groupby(["scope", "issue"], dropna=False):
        A(f"  {scope} | {issue}: {len(g)}")
else:
    A("No unexpected file/window issues detected.")
A("")

A("6. RAW MAGNITUDE SCREEN — DESCRIPTIVE ONLY")
A("-" * 118)
A("These extrema are descriptive integrity checks only; they do not trigger outcome/model-driven row deletion.")
if len(channel_summary):
    for _, r in channel_summary.sort_values(
        ["anchor", "modality", "channel", "status"]
    ).iterrows():
        A(
            f"{r['anchor']} | {r['modality']} | {r['channel']} | {r['status']} | "
            f"n={int(r['n_windows'])} | shape_fail={int(r['n_shape_fail'])} | "
            f"nonfinite={int(r['total_nonfinite_values'])} | "
            f"max_abs={r['max_abs_value']}"
        )
A("")

A("7. DECISION GATE")
A("-" * 118)

unexpected_issue_count = 0
if len(issues):
    # Known quarantine is represented in channel table, not issue table.
    unexpected_issue_count = len(issues)

pass_rows_start = int(
    manifest.loc[manifest["anchor"] == "START", "foundation_input_preflight_pass"].sum()
)
pass_rows_end = int(
    manifest.loc[manifest["anchor"] == "END", "foundation_input_preflight_pass"].sum()
)

global_pass = (
    len(start) == EXPECTED_ROWS
    and start["subject_num"].nunique() == EXPECTED_PARTICIPANTS
    and len(removed_still_present) == 0
    and pass_rows_start == len(start)
    and pass_rows_end == len(end)
    and unexpected_issue_count == 0
)

A(f"GLOBAL FOUNDATION RAW-INPUT AUDIT PASS: {global_pass}")

if global_pass:
    A("Interpretation: locked 1141-row START/END raw windows are structurally ready for frozen-FM representation extraction.")
    A("Known Subject 3 / task1_35i Shoulder ACC/GYR masks remain an explicit modeling issue to handle with masking/modality policy.")
else:
    A("Interpretation: DO NOT start GPU/FM training yet.")
    A("Inspect dataset_C_foundation_raw_window_issues.csv and channel summary before proceeding.")
A("")

A("8. OUTPUTS")
A("-" * 118)
A(str(MANIFEST_OUT))
A(str(ISSUES_OUT))
A(str(CHANNEL_SUMMARY_OUT))
A(str(AUDIT_OUT))
A("")
A(f"Elapsed minutes: {(time.time() - t0)/60:.2f}")
A("=" * 118)
A("END OF FOUNDATION RAW-WINDOW INPUT AUDIT")

AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("Created:")
for fp in [MANIFEST_OUT, ISSUES_OUT, CHANNEL_SUMMARY_OUT, AUDIT_OUT]:
    print(" ", fp)
print("")
print(f"GLOBAL FOUNDATION RAW-INPUT AUDIT PASS: {global_pass}")
print(f"Elapsed minutes: {(time.time() - t0)/60:.2f}")
print("")
print("Next: upload dataset_C_foundation_raw_window_audit.txt to ChatGPT.")
