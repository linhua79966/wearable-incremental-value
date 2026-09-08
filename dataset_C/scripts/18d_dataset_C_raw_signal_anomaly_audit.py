
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "dataset_C_shoulder_rotation" / "raw" / "WSD4FEDSRM"
SR = DS / "EMG, IMU, and PPG data"
AUD = ROOT / "audit"
AUD.mkdir(parents=True, exist_ok=True)

TXT_OUT = AUD / "dataset_C_raw_signal_anomaly_audit.txt"
CSV_OUT = AUD / "dataset_C_raw_signal_anomaly_points.csv"

# Exact raw files implicated by the outcome-independent localization audit.
FILES = [
    {
        "case": "S03_task2_45i_EMG_anterior_deltoid",
        "subject": 3,
        "task": "task2_45i",
        "path": SR / "40-50_ internal rotation" / "Subject 3" / "EMG data" / "anterior_deltoid.csv",
        "rate_hz": 1000,
        "modality": "EMG",
    },
    {
        "case": "S03_task2_45i_EMG_infraspinatus",
        "subject": 3,
        "task": "task2_45i",
        "path": SR / "40-50_ internal rotation" / "Subject 3" / "EMG data" / "infraspinatus.csv",
        "rate_hz": 1000,
        "modality": "EMG",
    },
    {
        "case": "S27_task1_35i_IMU_sternum_acc",
        "subject": 27,
        "task": "task1_35i",
        "path": SR / "30-40_ internal rotation" / "Subject 27" / "IMU data" / "Sternum" / "acc_sternum.csv",
        "rate_hz": 100,
        "modality": "IMU_ACC",
    },
]

def contiguous_runs(indices):
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return []
    runs = []
    s = p = int(indices[0])
    for x in indices[1:]:
        x = int(x)
        if x == p + 1:
            p = x
        else:
            runs.append((s, p))
            s = p = x
    runs.append((s, p))
    return runs

def robust_summary(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    ax = np.abs(x)
    return {
        "n": len(x),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "abs_q50": float(np.quantile(ax, 0.50)),
        "abs_q95": float(np.quantile(ax, 0.95)),
        "abs_q99": float(np.quantile(ax, 0.99)),
        "abs_q999": float(np.quantile(ax, 0.999)),
        "abs_max": float(np.max(ax)),
    }

rows = []
lines = []
A = lines.append

A("DATASET C — RAW-SIGNAL ANOMALY AUDIT")
A("=" * 118)
A("Purpose: inspect exact raw sensor samples underlying the gross handcrafted anomalies.")
A("This audit does NOT delete, clip, winsorize, interpolate, scale, or refit any model.")
A("")

for spec in FILES:
    fp = spec["path"]
    A(f"CASE: {spec['case']}")
    A("-" * 118)
    A(f"Path: {fp}")
    A(f"Exists: {fp.exists()}")

    if not fp.exists():
        A("")
        continue

    df = pd.read_csv(fp)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    A(f"Rows: {len(df)}")
    A(f"Columns: {list(df.columns)}")
    A(f"Nominal rate: {spec['rate_hz']} Hz")
    A("")

    # Channel summaries and exact largest points.
    for c in df.columns:
        x = df[c].to_numpy(dtype=float)
        s = robust_summary(x)

        A(f"[{c}]")
        for k, v in s.items():
            A(f"  {k}: {v}")

        finite = np.isfinite(x)
        idx = np.where(finite)[0]
        vals = x[finite]
        order = np.argsort(np.abs(vals))[::-1][:20]

        A("  top absolute raw samples:")
        for rank, oi in enumerate(order, start=1):
            ii = int(idx[oi])
            vv = float(vals[oi])
            A(
                f"    {rank:02d}. index={ii}, time={ii/spec['rate_hz']:.6f}s, value={vv:.12g}"
            )
            rows.append({
                "case": spec["case"],
                "modality": spec["modality"],
                "file": str(fp),
                "channel": c,
                "rank_abs": rank,
                "sample_index": ii,
                "time_sec_from_file_start": ii / spec["rate_hz"],
                "value": vv,
            })

        # Data-relative tail flags: compare to each channel's own q99.9.
        if len(vals):
            ax = np.abs(vals)
            q999 = float(np.quantile(ax, 0.999))
            med = float(np.median(ax))

            for mult in [10, 100, 1000, 1e6]:
                base = q999 if q999 > 0 else max(med, 1e-12)
                threshold = base * mult
                hit = np.where(np.isfinite(x) & (np.abs(x) > threshold))[0]
                runs = contiguous_runs(hit)
                A(
                    f"  > {mult:g} x own |x| q99.9 ({threshold:.12g}): "
                    f"{len(hit)} samples; {len(runs)} contiguous runs"
                )
                for rs, re in runs[:20]:
                    A(
                        f"    run {rs}:{re} | "
                        f"{rs/spec['rate_hz']:.6f}-{re/spec['rate_hz']:.6f}s | "
                        f"length={re-rs+1}"
                    )
        A("")

    # Row-level vector maxima for multi-axis files.
    if len(df.columns) > 1:
        arr = df.to_numpy(dtype=float)
        finite_rows = np.all(np.isfinite(arr), axis=1)
        mag = np.full(len(arr), np.nan)
        mag[finite_rows] = np.sqrt(np.sum(arr[finite_rows] ** 2, axis=1))

        finite_idx = np.where(np.isfinite(mag))[0]
        if len(finite_idx):
            order = finite_idx[np.argsort(mag[finite_idx])[::-1][:30]]
            A("[VECTOR MAGNITUDE TOP SAMPLES]")
            for rank, ii in enumerate(order, start=1):
                vals = ", ".join(
                    f"{c}={df.iloc[ii][c]:.12g}" for c in df.columns
                )
                A(
                    f"  {rank:02d}. index={ii}, time={ii/spec['rate_hz']:.6f}s, "
                    f"magnitude={mag[ii]:.12g} | {vals}"
                )
            A("")

    # Specific window summaries implicated by 18c.
    if spec["case"].startswith("S03_"):
        windows = [(0, 10), (10, 20), (20, 30)]
    else:
        # S27 anomaly mapped around these windows by START/END.
        windows = [(40, 50), (50, 60), (60, 70), (70, 80)]

    A("[WINDOW SUMMARIES]")
    for a_sec, b_sec in windows:
        a = int(round(a_sec * spec["rate_hz"]))
        b = int(round(b_sec * spec["rate_hz"]))
        if a >= len(df):
            continue
        b = min(b, len(df))
        A(f"  window [{a_sec},{b_sec}) sec -> samples [{a},{b})")
        for c in df.columns:
            x = df[c].iloc[a:b].to_numpy(dtype=float)
            s = robust_summary(x)
            if s:
                A(
                    f"    {c}: n={s['n']}, abs_q99={s['abs_q99']:.12g}, "
                    f"abs_q999={s['abs_q999']:.12g}, abs_max={s['abs_max']:.12g}"
                )
    A("")

pd.DataFrame(rows).to_csv(CSV_OUT, index=False)

# -------------------------------------------------------------------------
# Add raw text inspection around the top offending sample in each file.
# This can reveal malformed exponent strings or parser-valid corruption.
# -------------------------------------------------------------------------
A("RAW TEXT NEIGHBORHOODS AROUND MAXIMUM ABSOLUTE SAMPLE")
A("=" * 118)

for spec in FILES:
    fp = spec["path"]
    if not fp.exists():
        continue

    df = pd.read_csv(fp)
    for c in df.columns:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x)
        if not finite.any():
            continue

        candidate = np.where(finite)[0]
        imax = int(candidate[np.argmax(np.abs(x[finite]))])

        A(f"{spec['case']} | {c} | max index={imax}")

        # Read only a small text neighborhood by line number.
        # CSV has one header line, so data sample index i is file line i+2.
        lo = max(0, imax - 3)
        hi = min(len(df) - 1, imax + 3)

        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            for line_no, text in enumerate(f):
                sample_idx = line_no - 1
                if sample_idx < lo:
                    continue
                if sample_idx > hi:
                    break
                A(f"  sample_index={sample_idx}: {text.rstrip()}")
        A("")

A("DECISION RULE")
A("=" * 118)
A("A final quarantine is justified only if raw data show localized, grossly implausible")
A("sensor/file corruption independent of any prediction outcome.")
A("")
A("If corruption is confined to a contiguous raw segment:")
A("  quarantine only forecast windows overlapping that segment for the affected modality/trial.")
A("")
A("If an entire raw channel/file is corrupted:")
A("  quarantine that channel/file for that trial and preserve unaffected modalities.")
A("")
A("Do not remove an entire participant merely because one local sensor segment is invalid.")
A("")
A("=" * 118)
A("END OF AUDIT")

TXT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("Raw-signal anomaly audit complete.")
print(f"  {TXT_OUT}")
print(f"  {CSV_OUT}")
print("")
print("Upload dataset_C_raw_signal_anomaly_audit.txt to ChatGPT.")
