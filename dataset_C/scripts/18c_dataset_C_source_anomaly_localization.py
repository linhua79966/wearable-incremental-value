
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived" / "dataset_C_phase1"
AUDIT = ROOT / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)

START = DERIVED / "dataset_C_handcrafted_start_anchor.csv"
END = DERIVED / "dataset_C_handcrafted_end_anchor.csv"
DICT = DERIVED / "dataset_C_handcrafted_feature_dictionary.csv"

TXT_OUT = AUDIT / "dataset_C_source_anomaly_localization.txt"
ROW_OUT = AUDIT / "dataset_C_source_anomaly_rows.csv"
FEATURE_OUT = AUDIT / "dataset_C_source_anomaly_features.csv"

start = pd.read_csv(START)
end = pd.read_csv(END)
fd = pd.read_csv(DICT)

wear = fd.loc[fd["role"] == "wearable_feature", ["column", "family"]].copy()
wear = wear[wear["column"].isin(start.columns)].copy()

META = [
    "row_id", "subject_num", "task_label",
    "current_time_sec", "target_time_sec",
    "current_borg", "target_borg"
]
META = [c for c in META if c in start.columns]

# -------------------------------------------------------------------------
# Feature-level raw extrema and cross-anchor contrast
# -------------------------------------------------------------------------
feature_rows = []

for _, rr in wear.iterrows():
    c = rr["column"]
    fam = rr["family"]

    for anchor, df in [("START", start), ("END", end)]:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x)

        if finite.any():
            ax = np.abs(x[finite])
            q50 = float(np.quantile(ax, 0.50))
            q95 = float(np.quantile(ax, 0.95))
            q99 = float(np.quantile(ax, 0.99))
            mx = float(np.max(ax))
        else:
            q50 = q95 = q99 = mx = np.nan

        feature_rows.append({
            "anchor": anchor,
            "family": fam,
            "feature": c,
            "abs_q50": q50,
            "abs_q95": q95,
            "abs_q99": q99,
            "abs_max": mx,
            "nonfinite_n": int((~finite).sum()),
        })

feat = pd.DataFrame(feature_rows)

s = feat[feat["anchor"] == "START"].drop(columns="anchor").rename(
    columns={
        "abs_q50": "start_abs_q50",
        "abs_q95": "start_abs_q95",
        "abs_q99": "start_abs_q99",
        "abs_max": "start_abs_max",
        "nonfinite_n": "start_nonfinite_n",
    }
)
e = feat[feat["anchor"] == "END"].drop(columns=["anchor", "family"]).rename(
    columns={
        "abs_q50": "end_abs_q50",
        "abs_q95": "end_abs_q95",
        "abs_q99": "end_abs_q99",
        "abs_max": "end_abs_max",
        "nonfinite_n": "end_nonfinite_n",
    }
)

fm = s.merge(e, on="feature", how="left", validate="one_to_one")
fm["start_over_end_max_ratio"] = (
    fm["start_abs_max"] / fm["end_abs_max"].replace(0, np.nan)
)
fm["start_max_over_q99"] = (
    fm["start_abs_max"] / fm["start_abs_q99"].replace(0, np.nan)
)
fm["end_max_over_q99"] = (
    fm["end_abs_max"] / fm["end_abs_q99"].replace(0, np.nan)
)

fm = fm.sort_values(
    ["start_over_end_max_ratio", "start_abs_max"],
    ascending=[False, False]
)
fm.to_csv(FEATURE_OUT, index=False)

# -------------------------------------------------------------------------
# Row-level localization.
#
# These are DIAGNOSTIC gross-anomaly flags only; they are NOT exclusion
# thresholds and are not used to alter any outcome/model result.
# -------------------------------------------------------------------------
def feature_groups(df):
    groups = {
        "emg": [],
        "imu_acc": [],
        "imu_gyr": [],
    }

    for c in wear["column"]:
        if c not in df.columns:
            continue
        lc = c.lower()
        if "__emg_" in lc:
            groups["emg"].append(c)
        elif "__imu_" in lc and "_acc_" in lc:
            groups["imu_acc"].append(c)
        elif "__imu_" in lc and "_gyr_" in lc:
            groups["imu_gyr"].append(c)

    return groups

groups = feature_groups(start)

# Very loose thresholds chosen only to expose gross physical/data corruption.
# They must NOT be interpreted as formal physiological QC cut-points.
DIAG_THRESHOLDS = {
    "emg": [1e3, 1e6, 1e12],
    "imu_acc": [1e3, 1e4, 1e5],
    "imu_gyr": [1e2, 1e3, 1e4],
}

row_records = []

for anchor, df in [("START", start), ("END", end)]:
    for i, r in df.iterrows():
        rec = {c: r[c] for c in META}
        rec["anchor"] = anchor

        any_flag = False

        for family, cols in groups.items():
            arr = df.loc[i, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(arr)

            if finite.any():
                absarr = np.abs(arr[finite])
                mx = float(np.max(absarr))
            else:
                mx = np.nan

            rec[f"{family}_feature_abs_max"] = mx

            for th in DIAG_THRESHOLDS[family]:
                key = f"{family}_gt_{th:g}"
                flag = bool(np.isfinite(mx) and mx > th)
                rec[key] = int(flag)
                any_flag = any_flag or flag

        if any_flag:
            # Identify top 10 feature/value pairs in this row.
            vals = []
            for c in wear["column"]:
                v = pd.to_numeric(pd.Series([r[c]]), errors="coerce").iloc[0]
                if pd.notna(v) and np.isfinite(v):
                    vals.append((abs(float(v)), c, float(v)))

            vals.sort(reverse=True, key=lambda z: z[0])
            for rank, (_, c, v) in enumerate(vals[:10], start=1):
                rec[f"top{rank}_feature"] = c
                rec[f"top{rank}_value"] = v

            row_records.append(rec)

rows = pd.DataFrame(row_records)
if len(rows):
    rows = rows.sort_values(
        ["anchor", "emg_feature_abs_max", "imu_acc_feature_abs_max"],
        ascending=[True, False, False]
    )
rows.to_csv(ROW_OUT, index=False)

# -------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------
lines = []
A = lines.append

A("DATASET C — SOURCE-ANOMALY LOCALIZATION")
A("=" * 118)
A(f"Rows per anchor: {len(start)}")
A(f"Wearable features: {len(wear)}")
A("This audit does NOT delete, clip, winsorize, scale, tune, or refit anything.")
A("Diagnostic thresholds below are only for locating gross anomalies.")
A("")

A("1. FEATURES WITH LARGEST START-vs-END RAW-MAX CONTRAST")
A("-" * 118)
show = fm[
    [
        "family", "feature",
        "start_abs_q99", "start_abs_max",
        "end_abs_q99", "end_abs_max",
        "start_over_end_max_ratio",
        "start_max_over_q99", "end_max_over_q99",
    ]
].head(40)
A(show.to_string(index=False))
A("")

A("2. GROSS-ANOMALY ROW COUNTS")
A("-" * 118)

for anchor in ["START", "END"]:
    z = rows[rows["anchor"] == anchor] if len(rows) else pd.DataFrame()
    A(f"[{anchor}] rows with any gross diagnostic flag: {len(z)}")

    if len(z):
        A(f"  participants involved: {z['subject_num'].nunique()}")
        A(f"  subject-task trials involved: {z[['subject_num','task_label']].drop_duplicates().shape[0]}")

        for family, ths in DIAG_THRESHOLDS.items():
            for th in ths:
                c = f"{family}_gt_{th:g}"
                if c in z.columns:
                    A(f"  {c}: {int(z[c].sum())}")
    A("")

A("3. START GROSS-ANOMALY ROWS")
A("-" * 118)
if len(rows):
    z = rows[rows["anchor"] == "START"].copy()
    cols = META + [
        "emg_feature_abs_max",
        "imu_acc_feature_abs_max",
        "imu_gyr_feature_abs_max",
        "top1_feature", "top1_value",
        "top2_feature", "top2_value",
        "top3_feature", "top3_value",
    ]
    cols = [c for c in cols if c in z.columns]
    A(z[cols].head(100).to_string(index=False))
else:
    A("No gross-anomaly rows found.")
A("")

A("4. END GROSS-ANOMALY ROWS")
A("-" * 118)
if len(rows):
    z = rows[rows["anchor"] == "END"].copy()
    cols = META + [
        "emg_feature_abs_max",
        "imu_acc_feature_abs_max",
        "imu_gyr_feature_abs_max",
        "top1_feature", "top1_value",
        "top2_feature", "top2_value",
        "top3_feature", "top3_value",
    ]
    cols = [c for c in cols if c in z.columns]
    A(z[cols].head(100).to_string(index=False))
else:
    A("No gross-anomaly rows found.")
A("")

A("5. CONCENTRATION BY SUBJECT / TASK")
A("-" * 118)
if len(rows):
    for anchor in ["START", "END"]:
        z = rows[rows["anchor"] == anchor]
        A(f"[{anchor}]")
        if len(z):
            tab = (
                z.groupby(["subject_num", "task_label"])
                .size()
                .reset_index(name="flagged_rows")
                .sort_values("flagged_rows", ascending=False)
            )
            A(tab.head(40).to_string(index=False))
        else:
            A("None")
        A("")

A("6. DECISION RULE FOR NEXT STEP")
A("-" * 118)
A("If catastrophic START values are concentrated in a small number of subject-task/time windows,")
A("inspect the corresponding raw sensor files and define outcome-independent sensor-quality quarantine.")
A("")
A("If anomalies are widespread, do NOT delete rows post hoc. Instead redesign the handcrafted")
A("representation/preprocessing using a prespecified robust transformation and rerun BOTH anchors identically.")
A("")
A("Any final QC rule must be based on sensor-data validity, never on whether model performance improves.")
A("")
A("=" * 118)
A("END OF AUDIT")

TXT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("Source-anomaly localization complete.")
print(f"  {TXT_OUT}")
print(f"  {ROW_OUT}")
print(f"  {FEATURE_OUT}")
print("")
print("Upload dataset_C_source_anomaly_localization.txt to ChatGPT.")
