from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from momentfm import MOMENTPipeline


# =============================================================================
# Script 31 — Dataset C real locked-row MOMENT-1-base smoke
#
# PURPOSE
#   Validate the REAL Dataset C raw-window -> MOMENT frozen embedding chain
#   before any full 1141-row extraction.
#
# LOCKED DESIGN
#   - exact 18f QC-clean START/END cohort;
#   - 1141 paired rows, 34 participants;
#   - exact +10 s Borg forecasting geometry already locked upstream;
#   - IMU only, 100 Hz, exact 10 s = 1000 samples;
#   - no resampling, normalization, clipping, winsorization, interpolation,
#     repair, or target-driven transformation;
#   - each physical IMU location separately as 6 channels:
#       ACC_x, ACC_y, ACC_z, GYR_x, GYR_y, GYR_z;
#   - MOMENT-1-base locked revision/weights from Script 30;
#   - frozen embedding mode only;
#   - 1000 -> [0:512] + [512:1000] with 24-sample right pad;
#   - padded tail excluded through input_mask;
#   - chunk embeddings pooled by valid-patch weights 64/125 and 61/125;
#   - available physical locations mean-pooled to one 768-D row embedding;
#   - Subject 3 / task1_35i Shoulder remains omitted/unavailable, never
#     zero-filled as observed signal.
#
# SMOKE ROWS
#   ordinary:            S01__task1_35i__t10
#   known Shoulder mask: S03__task1_35i__t10
#   Both START and END anchors are processed.
#
# NO predictive-value result is produced here.
# =============================================================================


ROOT = Path(__file__).resolve().parents[2]

DS = ROOT / "data" / "dataset_C_shoulder_rotation" / "raw" / "WSD4FEDSRM"
SR = DS / "EMG, IMU, and PPG data"

DERIVED = ROOT / "derived" / "dataset_C_phase1"
START_LOCKED = DERIVED / "dataset_C_handcrafted_start_anchor_qc.csv"
END_LOCKED = DERIVED / "dataset_C_handcrafted_end_anchor_qc.csv"

MODEL_DIR = ROOT / "third_party" / "MOMENT-1-base_locked_5e44b0e"
CONFIG_FP = MODEL_DIR / "config.json"
WEIGHTS_FP = MODEL_DIR / "model.safetensors"

OUTDIR = ROOT / "derived" / "dataset_C_moment_imu"
AUDITDIR = ROOT / "audit"
OUTDIR.mkdir(parents=True, exist_ok=True)
AUDITDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUTDIR / "dataset_C_moment1base_locked_row_smoke_summary.csv"
EMBED_OUT = OUTDIR / "dataset_C_moment1base_locked_row_smoke_embeddings.npz"
AUDIT_OUT = AUDITDIR / "dataset_C_moment1base_locked_row_smoke_audit_31.txt"

IMU_HZ = 100
WIN_SEC = 10.0
EXPECTED_SAMPLES = 1000
SEQ_LEN = 512
PATCH_LEN = 8
D_MODEL = 768

CHUNK_A_VALID = 512
CHUNK_B_VALID = 488
CHUNK_B_PAD = 24
PATCHES_A = 64
PATCHES_B = 61
PATCHES_TOTAL = 125

LOCKED_REVISION = "5e44b0ea26376a176360f87831124e018f876d96"
EXPECTED_CONFIG_SHA256 = "F1C66C2BB845229C0ED27A1600DBCC956B85AB21F9E5FD8A1663E6641BED7755"
EXPECTED_WEIGHTS_SHA256 = "1A436826FFE618273EC62B9656DC4CAB8EDC470364F104E90542A4EBC14FB825"

EXPECTED_ORDINARY_ROW_ID = "S01__task1_35i__t10"
EXPECTED_MASKED_ROW_ID = "S03__task1_35i__t10"

TASK = {
    "task1_35i": ("30-40_ internal rotation", "internal", 35),
    "task2_45i": ("40-50_ internal rotation", "internal", 45),
    "task3_55i": ("50-60_ internal rotation", "internal", 55),
    "task4_35e": ("30-40_ external rotation", "external", 35),
    "task5_45e": ("40-50_ external rotation", "external", 45),
    "task6_55e": ("50-60_ external rotation", "external", 55),
}

LOCS = ["Forearm", "Hand", "Pelvis", "Shoulder", "Sternum", "Upper arm"]
CHANNEL_ORDER = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

LOCKED_REMOVED_QC_IDS = {
    "S03__task2_45i__t10",
    "S27__task1_35i__t50",
    "S27__task1_35i__t60",
}


def safe(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def sha256_file(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def readnum(fp: Path) -> np.ndarray:
    x = pd.read_csv(fp)
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    a = x.to_numpy(dtype=np.float32)
    if a.ndim != 2 or a.shape[1] < 3:
        raise RuntimeError(f"{fp}: expected >=3 numeric columns, got {a.shape}")
    return a[:, :3]


def b_start(t: float, rate: int, n: int):
    a = int(round((float(t) - WIN_SEC) * rate))
    b = int(round(float(t) * rate))
    return (a, b) if a >= 0 and b <= n and b > a else None


def b_end(t: float, dur: float, rate: int, n: int):
    a = int(round(n - (float(dur) - (float(t) - WIN_SEC)) * rate))
    b = int(round(n - (float(dur) - float(t)) * rate))
    return (a, b) if a >= 0 and b <= n and b > a else None


def known_location_quarantine(subject_num: int, task_label: str, location: str) -> bool:
    return (
        int(subject_num) == 3
        and str(task_label) == "task1_35i"
        and str(location) == "Shoulder"
    )


# =============================================================================
# 1. LOCKED COHORT + MODEL PROVENANCE GUARDS
# =============================================================================

for fp in [START_LOCKED, END_LOCKED, CONFIG_FP, WEIGHTS_FP]:
    if not fp.exists():
        raise SystemExit(f"Missing required locked input: {fp}")

config_sha = sha256_file(CONFIG_FP)
weights_sha = sha256_file(WEIGHTS_FP)

if config_sha != EXPECTED_CONFIG_SHA256:
    raise SystemExit(f"config.json SHA256 mismatch: {config_sha}")
if weights_sha != EXPECTED_WEIGHTS_SHA256:
    raise SystemExit(f"model.safetensors SHA256 mismatch: {weights_sha}")

cfg = json.loads(CONFIG_FP.read_text(encoding="utf-8"))
if int(cfg["seq_len"]) != SEQ_LEN:
    raise SystemExit(f"Unexpected MOMENT seq_len: {cfg['seq_len']}")
if int(cfg["patch_len"]) != PATCH_LEN:
    raise SystemExit(f"Unexpected MOMENT patch_len: {cfg['patch_len']}")
if int(cfg["t5_config"]["d_model"]) != D_MODEL:
    raise SystemExit(f"Unexpected MOMENT d_model: {cfg['t5_config']['d_model']}")

S = pd.read_csv(START_LOCKED).sort_values("row_id").reset_index(drop=True)
E = pd.read_csv(END_LOCKED).sort_values("row_id").reset_index(drop=True)

required = {
    "row_id",
    "subject_num",
    "task_label",
    "current_time_sec",
    "target_time_sec",
    "trial_duration_sec_qc",
}

for label, df in [("START", S), ("END", E)]:
    miss = required - set(df.columns)
    if miss:
        raise SystemExit(f"{label}: missing columns {sorted(miss)}")
    if df["row_id"].duplicated().any():
        raise SystemExit(f"{label}: duplicate row_id")

if len(S) != 1141 or len(E) != 1141:
    raise SystemExit(f"Expected 1141 locked rows, got START={len(S)} END={len(E)}")

if S["subject_num"].nunique() != 34:
    raise SystemExit(f"Expected 34 participants, got {S['subject_num'].nunique()}")

if list(S["row_id"].astype(str)) != list(E["row_id"].astype(str)):
    raise SystemExit("START/END row IDs differ.")

if set(S["row_id"].astype(str)) & LOCKED_REMOVED_QC_IDS:
    raise SystemExit("Removed QC row unexpectedly remains in locked cohort.")

for rid in [EXPECTED_ORDINARY_ROW_ID, EXPECTED_MASKED_ROW_ID]:
    if rid not in set(S["row_id"].astype(str)):
        raise SystemExit(f"Required deterministic smoke row missing: {rid}")

SMOKE_CASES = [
    ("ordinary", EXPECTED_ORDINARY_ROW_ID),
    ("known_shoulder_mask", EXPECTED_MASKED_ROW_ID),
]


# =============================================================================
# 2. LOCKED MOMENT-1-BASE LOAD / FREEZE
# =============================================================================

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable.")

device = torch.device("cuda:0")

model = MOMENTPipeline.from_pretrained(
    str(MODEL_DIR),
    model_kwargs={"task_name": "embedding"},
)
model.init()
model.eval()

for p in model.parameters():
    p.requires_grad_(False)

if any(p.requires_grad for p in model.parameters()):
    raise RuntimeError("MOMENT is not fully frozen.")

model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 3. RAW TRIAL CACHE
# =============================================================================

trial_cache: dict[tuple[int, str], dict] = {}


def load_trial(sn: int, task: str):
    key = (int(sn), str(task))
    if key in trial_cache:
        return trial_cache[key]

    if task not in TASK:
        raise RuntimeError(f"Unknown task label: {task}")

    folder = TASK[task][0]
    td = SR / folder / f"Subject {int(sn)}"

    out = {}

    for loc in LOCS:
        if known_location_quarantine(sn, task, loc):
            out[loc] = None
            continue

        lf = safe(loc)
        acc_fp = td / "IMU data" / loc / f"acc_{lf}.csv"
        gyr_fp = td / "IMU data" / loc / f"gyr_{lf}.csv"

        if not acc_fp.exists() or not gyr_fp.exists():
            raise RuntimeError(
                f"Missing IMU file(s): subject={sn}, task={task}, loc={loc}"
            )

        acc = readnum(acc_fp)
        gyr = readnum(gyr_fp)

        if len(acc) != len(gyr):
            raise RuntimeError(
                f"ACC/GYR length mismatch: subject={sn}, task={task}, loc={loc}, "
                f"acc={len(acc)}, gyr={len(gyr)}"
            )

        out[loc] = {
            "acc": acc,
            "gyr": gyr,
            "n": len(acc),
            "acc_fp": acc_fp,
            "gyr_fp": gyr_fp,
        }

    trial_cache[key] = out
    return out


# =============================================================================
# 4. MOMENT LOCATION ENCODER — EXACT LOCKED 1000-SAMPLE POLICY
# =============================================================================

mask_a_cpu = torch.ones((1, SEQ_LEN), dtype=torch.long)
mask_b_cpu = torch.zeros((1, SEQ_LEN), dtype=torch.long)
mask_b_cpu[:, :CHUNK_B_VALID] = 1


def encode_location(raw_6x1000: np.ndarray):
    if raw_6x1000.shape != (6, EXPECTED_SAMPLES):
        raise RuntimeError(f"Expected (6,1000), got {raw_6x1000.shape}")

    if not np.isfinite(raw_6x1000).all():
        raise RuntimeError("Non-finite raw IMU input.")

    x1000 = torch.from_numpy(raw_6x1000.astype(np.float32, copy=False)).unsqueeze(0)

    chunk_a = x1000[:, :, :CHUNK_A_VALID].contiguous()
    chunk_b = torch.zeros((1, 6, SEQ_LEN), dtype=torch.float32)
    chunk_b[:, :, :CHUNK_B_VALID] = x1000[:, :, CHUNK_A_VALID:EXPECTED_SAMPLES]

    reconstructed = torch.cat([chunk_a, chunk_b[:, :, :CHUNK_B_VALID]], dim=-1)
    sample_conservation_exact = bool(torch.equal(reconstructed, x1000))

    if not sample_conservation_exact:
        raise RuntimeError("1000-sample conservation failed.")

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    with torch.inference_mode():
        oa = model(
            x_enc=chunk_a.to(device),
            input_mask=mask_a_cpu.to(device),
            reduction="mean",
        )
        ob = model(
            x_enc=chunk_b.to(device),
            input_mask=mask_b_cpu.to(device),
            reduction="mean",
        )

        ea = oa.embeddings
        eb = ob.embeddings

        pooled = (PATCHES_A * ea + PATCHES_B * eb) / PATCHES_TOTAL

    torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    if tuple(ea.shape) != (1, D_MODEL):
        raise RuntimeError(f"Unexpected chunk A embedding shape: {tuple(ea.shape)}")
    if tuple(eb.shape) != (1, D_MODEL):
        raise RuntimeError(f"Unexpected chunk B embedding shape: {tuple(eb.shape)}")
    if tuple(pooled.shape) != (1, D_MODEL):
        raise RuntimeError(f"Unexpected pooled location embedding shape: {tuple(pooled.shape)}")

    if not torch.isfinite(ea).all():
        raise RuntimeError("Non-finite chunk A embedding.")
    if not torch.isfinite(eb).all():
        raise RuntimeError("Non-finite chunk B embedding.")
    if not torch.isfinite(pooled).all():
        raise RuntimeError("Non-finite pooled location embedding.")

    return {
        "embedding": pooled.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "chunk_a_shape": tuple(ea.shape),
        "chunk_b_shape": tuple(eb.shape),
        "sample_conservation_exact": sample_conservation_exact,
        "elapsed_seconds": elapsed,
        "peak_mb": peak_mb,
    }


# =============================================================================
# 5. REAL LOCKED-ROW SMOKE
# =============================================================================

summary_rows = []
npz_payload = {}

t_global = time.time()

for case_name, row_id in SMOKE_CASES:
    srow = S.loc[S["row_id"].astype(str) == row_id].iloc[0]
    erow = E.loc[E["row_id"].astype(str) == row_id].iloc[0]

    # START/END metadata lock.
    for c in ["subject_num", "task_label", "current_time_sec", "trial_duration_sec_qc"]:
        sv = srow[c]
        ev = erow[c]
        try:
            same = bool(np.isclose(float(sv), float(ev), atol=1e-9))
        except Exception:
            same = str(sv) == str(ev)
        if not same:
            raise RuntimeError(f"{row_id}: START/END metadata differs for {c}")

    sn = int(srow["subject_num"])
    task = str(srow["task_label"])
    t = float(srow["current_time_sec"])
    dur = float(srow["trial_duration_sec_qc"])

    trial = load_trial(sn, task)

    for anchor in ["START", "END"]:
        location_embeddings = []
        available_locations = []
        masked_locations = []
        location_seconds = []
        location_peak_mb = []
        sample_conservation = []
        raw_finite_flags = []

        for loc in LOCS:
            obj = trial[loc]

            if obj is None:
                masked_locations.append(loc)
                continue

            n = int(obj["n"])

            bounds = (
                b_start(t, IMU_HZ, n)
                if anchor == "START"
                else b_end(t, dur, IMU_HZ, n)
            )

            if bounds is None:
                raise RuntimeError(
                    f"{row_id} {anchor} {loc}: 10-s window out of bounds."
                )

            a, b = bounds

            acc = obj["acc"][a:b, :3]
            gyr = obj["gyr"][a:b, :3]

            if acc.shape != (EXPECTED_SAMPLES, 3):
                raise RuntimeError(
                    f"{row_id} {anchor} {loc}: bad ACC shape {acc.shape}"
                )

            if gyr.shape != (EXPECTED_SAMPLES, 3):
                raise RuntimeError(
                    f"{row_id} {anchor} {loc}: bad GYR shape {gyr.shape}"
                )

            raw = np.concatenate([acc.T, gyr.T], axis=0).astype(np.float32)

            if raw.shape != (6, EXPECTED_SAMPLES):
                raise RuntimeError(
                    f"{row_id} {anchor} {loc}: bad 6-channel shape {raw.shape}"
                )

            raw_finite = bool(np.isfinite(raw).all())
            raw_finite_flags.append(raw_finite)

            if not raw_finite:
                raise RuntimeError(
                    f"{row_id} {anchor} {loc}: non-finite raw IMU values."
                )

            enc = encode_location(raw)

            location_embeddings.append(enc["embedding"])
            available_locations.append(loc)
            location_seconds.append(float(enc["elapsed_seconds"]))
            location_peak_mb.append(float(enc["peak_mb"]))
            sample_conservation.append(bool(enc["sample_conservation_exact"]))

            torch.cuda.empty_cache()

        if not location_embeddings:
            raise RuntimeError(f"{row_id} {anchor}: no available IMU locations.")

        global_embedding = np.mean(
            np.stack(location_embeddings, axis=0),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)

        if global_embedding.shape != (D_MODEL,):
            raise RuntimeError(
                f"{row_id} {anchor}: global embedding shape {global_embedding.shape}"
            )

        if not np.isfinite(global_embedding).all():
            raise RuntimeError(
                f"{row_id} {anchor}: non-finite global embedding."
            )

        # Mask policy checks are explicit, not inferred.
        if case_name == "ordinary":
            if masked_locations:
                raise RuntimeError(
                    f"Ordinary smoke row unexpectedly masked locations: {masked_locations}"
                )
            if len(available_locations) != 6:
                raise RuntimeError(
                    f"Ordinary smoke row expected 6 locations, got {len(available_locations)}"
                )

        if case_name == "known_shoulder_mask":
            if masked_locations != ["Shoulder"]:
                raise RuntimeError(
                    f"Known-mask row expected only Shoulder masked, got {masked_locations}"
                )
            if len(available_locations) != 5:
                raise RuntimeError(
                    f"Known-mask row expected 5 locations, got {len(available_locations)}"
                )
            if "Shoulder" in available_locations:
                raise RuntimeError("Known masked Shoulder was incorrectly encoded.")

        npz_payload[f"{case_name}__{anchor}__moment768"] = global_embedding

        summary_rows.append({
            "case": case_name,
            "row_id": row_id,
            "subject_num": sn,
            "task_label": task,
            "anchor": anchor,
            "current_time_sec": t,
            "trial_duration_sec_qc": dur,
            "input_sampling_rate_hz": IMU_HZ,
            "resampling_applied": False,
            "window_seconds": WIN_SEC,
            "samples_per_axis": EXPECTED_SAMPLES,
            "channels_per_location": 6,
            "channel_order": "|".join(CHANNEL_ORDER),
            "chunk_policy": "512_valid|488_valid_plus_24_masked",
            "chunk_valid_patch_weights": "64/125|61/125",
            "available_location_count": len(available_locations),
            "available_locations": "|".join(available_locations),
            "masked_location_count": len(masked_locations),
            "masked_locations": "|".join(masked_locations),
            "raw_all_finite": bool(all(raw_finite_flags)),
            "sample_conservation_all_locations": bool(all(sample_conservation)),
            "global_embedding_dim": int(global_embedding.shape[0]),
            "global_embedding_finite": bool(np.isfinite(global_embedding).all()),
            "mean_seconds_per_available_location": float(np.mean(location_seconds)),
            "max_seconds_per_available_location": float(np.max(location_seconds)),
            "max_location_peak_allocated_mb": float(np.max(location_peak_mb)),
        })


summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_OUT, index=False)
np.savez_compressed(EMBED_OUT, **npz_payload)


# =============================================================================
# 6. AUDIT / DECISION GATE
# =============================================================================

lines = []
A = lines.append

A("DATASET C — MOMENT-1-BASE REAL LOCKED-ROW START/END SMOKE AUDIT (SCRIPT 31)")
A("=" * 118)
A(f"Locked cohort rows: {len(S)}")
A(f"Locked participants: {S['subject_num'].nunique()}")
A(f"START/END identical row IDs: {list(S['row_id'].astype(str)) == list(E['row_id'].astype(str))}")
A(f"Ordinary smoke row: {EXPECTED_ORDINARY_ROW_ID}")
A(f"Known-mask smoke row: {EXPECTED_MASKED_ROW_ID}")
A("")

A("1. LOCKED MODEL PROVENANCE")
A("-" * 118)
A(f"Model directory: {MODEL_DIR}")
A(f"Locked revision: {LOCKED_REVISION}")
A(f"config.json SHA256: {config_sha}")
A(f"model.safetensors SHA256: {weights_sha}")
A(f"config seq_len: {cfg['seq_len']}")
A(f"config patch_len: {cfg['patch_len']}")
A(f"config d_model: {cfg['t5_config']['d_model']}")
A(f"Model class: {model.__class__.__name__}")
A(f"Task name: {model.task_name}")
A(f"Total parameters: {total_params}")
A(f"Trainable parameters: {trainable_params}")
A(f"All parameters frozen: {all(not p.requires_grad for p in model.parameters())}")
A(f"GPU: {torch.cuda.get_device_name(0)}")
A("")

A("2. LOCKED REAL-INPUT POLICY")
A("-" * 118)
A("Primary modality: IMU only.")
A("Dataset C nominal IMU sampling rate: 100 Hz.")
A("Exact 10-s input window: 1000 samples per axis.")
A("Resampling: NO.")
A("Raw waveform normalization/clipping/winsorization/interpolation/repair: NONE.")
A("Per-location channel order:")
A("  ACC_x, ACC_y, ACC_z, GYR_x, GYR_y, GYR_z")
A("Each physical location is encoded separately.")
A("MOMENT sequence handling:")
A("  chunk A = first 512 original samples; 512 valid.")
A("  chunk B = final 488 original samples + 24 right-pad values.")
A("  padded 24 samples are invalid in input_mask.")
A("  valid-patch chunk weights = 64/125 and 61/125.")
A("  all 1000 original samples are used exactly once.")
A("Row representation = mean over AVAILABLE physical-location 768-D embeddings.")
A("Subject 3/task1_35i Shoulder is omitted from location pooling; never zero-filled as observed signal.")
A("No Borg target is used in extraction.")
A("")

A("3. REAL LOCKED-ROW SMOKE RESULTS")
A("-" * 118)

for _, r in summary.iterrows():
    A(
        f"{r['case']} | {r['row_id']} | {r['anchor']} | "
        f"available_locations={r['available_location_count']} | "
        f"masked={r['masked_locations'] if str(r['masked_locations']) else '<none>'} | "
        f"raw_finite={r['raw_all_finite']} | "
        f"sample_conservation={r['sample_conservation_all_locations']} | "
        f"embedding_dim={r['global_embedding_dim']} | "
        f"embedding_finite={r['global_embedding_finite']} | "
        f"max_peak_MB={r['max_location_peak_allocated_mb']:.2f} | "
        f"mean_sec/location={r['mean_seconds_per_available_location']:.3f}"
    )

A("")

A("4. SCIENTIFIC / TECHNICAL GUARDRAILS")
A("-" * 118)
A("PASS: exact locked 18f QC-clean rows are used.")
A("PASS: START/END paired cohort is unchanged.")
A("PASS: raw 100-Hz IMU is used without resampling.")
A("PASS: exact START/END 10-s window geometry is preserved.")
A("PASS: no supervised target enters extraction.")
A("PASS: MOMENT weights/revision are locked before scientific results.")
A("PASS: model is fully frozen and evaluation-only.")
A("PASS: 1000 samples are conserved exactly through chunking.")
A("PASS: the final 24 pad values are excluded through input_mask.")
A("PASS: known Shoulder quarantine remains a missing physical location, not a zero-filled observed signal.")
A("")

expected_rows = 4

all_ok = (
    len(summary) == expected_rows
    and summary["raw_all_finite"].all()
    and summary["sample_conservation_all_locations"].all()
    and summary["global_embedding_finite"].all()
    and set(summary["global_embedding_dim"].astype(int)) == {D_MODEL}
    and set(summary.loc[summary["case"] == "ordinary", "available_location_count"].astype(int)) == {6}
    and set(summary.loc[summary["case"] == "ordinary", "masked_location_count"].astype(int)) == {0}
    and set(summary.loc[summary["case"] == "known_shoulder_mask", "available_location_count"].astype(int)) == {5}
    and set(summary.loc[summary["case"] == "known_shoulder_mask", "masked_location_count"].astype(int)) == {1}
    and set(summary.loc[summary["case"] == "known_shoulder_mask", "masked_locations"].astype(str)) == {"Shoulder"}
    and trainable_params == 0
    and config_sha == EXPECTED_CONFIG_SHA256
    and weights_sha == EXPECTED_WEIGHTS_SHA256
)

A("5. DECISION GATE")
A("-" * 118)
A(f"DATASET C MOMENT-1-BASE LOCKED-ROW SMOKE PASS: {all_ok}")

if all_ok:
    A("Interpretation: real QC-clean START/END Dataset C IMU windows can be encoded with the locked frozen")
    A("MOMENT-1-base comparator under the prespecified 512/488+mask and location-aware pooling policy.")
    A("No predictive-value claim is made at this stage.")
    A("Ready for full 1141-row START/END frozen MOMENT extraction.")
else:
    A("Interpretation: DO NOT run full 1141-row MOMENT extraction; inspect the smoke summary first.")

A("")
A("6. OUTPUTS")
A("-" * 118)
A(str(SUMMARY_OUT))
A(str(EMBED_OUT))
A(str(AUDIT_OUT))
A("")
A(f"Elapsed minutes: {(time.time() - t_global) / 60:.2f}")
A("=" * 118)
A("END OF MOMENT-1-BASE REAL LOCKED-ROW SMOKE AUDIT")

AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("Created:")
print(" ", SUMMARY_OUT)
print(" ", EMBED_OUT)
print(" ", AUDIT_OUT)
print()
print(f"DATASET C MOMENT-1-BASE LOCKED-ROW SMOKE PASS: {all_ok}")
print("Upload dataset_C_moment1base_locked_row_smoke_audit_31.txt to ChatGPT.")
