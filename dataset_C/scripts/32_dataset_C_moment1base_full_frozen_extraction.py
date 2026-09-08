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
# Script 32 — FULL frozen MOMENT-1-base IMU extraction for locked Dataset C
#
# SCIENTIFIC DESIGN IS FIXED BEFORE DOWNSTREAM RESULTS
#   - locked QC-clean cohort: 1141 rows, 34 participants
#   - paired START + END anchors
#   - raw IMU only
#   - 100 Hz, exact 10 s = 1000 samples/axis
#   - NO resampling
#   - NO waveform normalization/clipping/winsorization/interpolation/repair
#   - six channels per physical location:
#       ACC_x, ACC_y, ACC_z, GYR_x, GYR_y, GYR_z
#   - each physical location encoded separately
#   - official frozen AutonLab/MOMENT-1-base
#   - locked HF revision:
#       5e44b0ea26376a176360f87831124e018f876d96
#   - locked config / safetensors SHA256 from Script 30
#   - native MOMENT seq_len=512, patch_len=8, d_model=768
#
# 1000-SAMPLE INPUT POLICY
#   chunk A = samples [0:512]       -> 512 valid samples = 64 valid patches
#   chunk B = samples [512:1000]    -> 488 valid samples + 24 right-pad
#   last 24 pad samples are invalid in input_mask = 61 valid patches
#   chunk pooling weights = 64/125 and 61/125
#
# ROW REPRESENTATION
#   mean-reduced 768-D embedding per physical location
#   -> valid-patch-weighted mean across the two chunks
#   -> mean across AVAILABLE physical locations
#   -> one frozen 768-D row representation
#
# KNOWN MASK POLICY
#   Subject 3 / task1_35i / Shoulder is unavailable and omitted from the
#   across-location mean; it is NEVER zero-filled as observed signal.
#
# THIS SCRIPT DOES NOT
#   - fit any supervised model
#   - use Borg outcome during extraction
#   - inspect downstream prediction errors
#   - fine-tune or adapt MOMENT
#   - change Dataset C QC
#   - choose a representation based on downstream result direction
#
# RESUME
#   Progress is checkpointed every SAVE_EVERY row-pairs.
#   Re-running resumes only if row IDs, model hashes, revision, and locked
#   input-policy metadata all match.
#
# FINAL OUTPUTS
#   derived/dataset_C_moment_imu/
#     dataset_C_moment1base_imu_locked_metadata.csv
#     dataset_C_moment1base_imu_start_mean768.npz
#     dataset_C_moment1base_imu_end_mean768.npz
#     dataset_C_moment1base_imu_extraction_manifest.csv
#
#   audit/
#     dataset_C_moment1base_full_frozen_extraction_audit_32.txt
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

META_OUT = OUTDIR / "dataset_C_moment1base_imu_locked_metadata.csv"
START_OUT = OUTDIR / "dataset_C_moment1base_imu_start_mean768.npz"
END_OUT = OUTDIR / "dataset_C_moment1base_imu_end_mean768.npz"
MANIFEST_OUT = OUTDIR / "dataset_C_moment1base_imu_extraction_manifest.csv"

PARTIAL_OUT = OUTDIR / "_dataset_C_moment1base_imu_partial_resume.npz"
PARTIAL_META = OUTDIR / "_dataset_C_moment1base_imu_partial_resume.json"

AUDIT_OUT = AUDITDIR / "dataset_C_moment1base_full_frozen_extraction_audit_32.txt"

IMU_HZ = 100
WIN_SEC = 10.0
EXPECTED_SAMPLES = 1000

SEQ_LEN = 512
PATCH_LEN = 8
EMBED_DIM = 768

CHUNK_A_VALID = 512
CHUNK_B_VALID = 488
CHUNK_B_PAD = 24

PATCHES_A = 64
PATCHES_B = 61
PATCHES_TOTAL = 125

SAVE_EVERY = 10
BASE_SEED = 20260905

LOCKED_REVISION = "5e44b0ea26376a176360f87831124e018f876d96"
EXPECTED_CONFIG_SHA256 = "F1C66C2BB845229C0ED27A1600DBCC956B85AB21F9E5FD8A1663E6641BED7755"
EXPECTED_WEIGHTS_SHA256 = "1A436826FFE618273EC62B9656DC4CAB8EDC470364F104E90542A4EBC14FB825"

REPRESENTATION_NAME = (
    "moment1base_embedding_mean__"
    "two_chunk_valid_patch_weight_64_61__"
    "available_location_mean"
)

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


# =============================================================================
# Utility
# =============================================================================

def safe(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def sha256_file(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def row_ids_sha256(row_ids) -> str:
    s = "\n".join(map(str, row_ids)).encode("utf-8")
    return hashlib.sha256(s).hexdigest().upper()


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


def save_embedding_npz(
    fp: Path,
    row_ids,
    arr: np.ndarray,
    anchor: str,
):
    np.savez_compressed(
        fp,
        row_id=np.asarray(row_ids, dtype=str),
        embedding=np.asarray(arr, dtype=np.float32),
        anchor=np.asarray([anchor]),
        representation=np.asarray([REPRESENTATION_NAME]),
        model_name=np.asarray(["AutonLab/MOMENT-1-base"]),
        model_revision=np.asarray([LOCKED_REVISION]),
        model_safetensors_sha256=np.asarray([EXPECTED_WEIGHTS_SHA256]),
        config_sha256=np.asarray([EXPECTED_CONFIG_SHA256]),
        sampling_rate_hz=np.asarray([IMU_HZ], dtype=np.int32),
        window_seconds=np.asarray([WIN_SEC], dtype=np.float32),
        input_samples=np.asarray([EXPECTED_SAMPLES], dtype=np.int32),
        seq_len=np.asarray([SEQ_LEN], dtype=np.int32),
        patch_len=np.asarray([PATCH_LEN], dtype=np.int32),
        chunk_a_valid_samples=np.asarray([CHUNK_A_VALID], dtype=np.int32),
        chunk_b_valid_samples=np.asarray([CHUNK_B_VALID], dtype=np.int32),
        chunk_b_pad_samples=np.asarray([CHUNK_B_PAD], dtype=np.int32),
        chunk_patch_weights=np.asarray([PATCHES_A, PATCHES_B], dtype=np.int32),
    )


# =============================================================================
# 1. Locked cohort / model guards
# =============================================================================

for fp in [START_LOCKED, END_LOCKED, CONFIG_FP, WEIGHTS_FP]:
    if not fp.exists():
        raise SystemExit(f"Missing required locked input: {fp}")

config_sha = sha256_file(CONFIG_FP)
weights_sha = sha256_file(WEIGHTS_FP)

if config_sha != EXPECTED_CONFIG_SHA256:
    raise SystemExit(
        f"config.json hash mismatch: expected {EXPECTED_CONFIG_SHA256}, got {config_sha}"
    )

if weights_sha != EXPECTED_WEIGHTS_SHA256:
    raise SystemExit(
        f"model.safetensors hash mismatch: expected {EXPECTED_WEIGHTS_SHA256}, got {weights_sha}"
    )

cfg = json.loads(CONFIG_FP.read_text(encoding="utf-8"))

if int(cfg["seq_len"]) != SEQ_LEN:
    raise SystemExit(f"Unexpected config seq_len={cfg['seq_len']}")

if int(cfg["patch_len"]) != PATCH_LEN:
    raise SystemExit(f"Unexpected config patch_len={cfg['patch_len']}")

if int(cfg["t5_config"]["d_model"]) != EMBED_DIM:
    raise SystemExit(f"Unexpected config d_model={cfg['t5_config']['d_model']}")

S = pd.read_csv(START_LOCKED).sort_values("row_id").reset_index(drop=True)
E = pd.read_csv(END_LOCKED).sort_values("row_id").reset_index(drop=True)

required = {
    "row_id",
    "subject_num",
    "task_label",
    "current_time_sec",
    "target_time_sec",
    "trial_duration_sec_qc",
    "current_borg",
    "target_borg",
}

for label, df in [("START", S), ("END", E)]:
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{label}: missing required columns {sorted(missing)}")
    if df["row_id"].duplicated().any():
        raise SystemExit(f"{label}: duplicate row_id")

if len(S) != 1141 or len(E) != 1141:
    raise SystemExit(
        f"Expected 1141 paired rows; got START={len(S)}, END={len(E)}"
    )

if S["subject_num"].nunique() != 34:
    raise SystemExit(
        f"Expected 34 participants; got {S['subject_num'].nunique()}"
    )

if list(S["row_id"].astype(str)) != list(E["row_id"].astype(str)):
    raise SystemExit("START/END row IDs differ.")

if not np.allclose(
    S["target_borg"].to_numpy(float),
    E["target_borg"].to_numpy(float),
    equal_nan=True,
):
    raise SystemExit("START/END targets differ.")

if set(S["row_id"].astype(str)) & LOCKED_REMOVED_QC_IDS:
    raise SystemExit("18e/18f removed QC rows unexpectedly remain.")

ROW_IDS = S["row_id"].astype(str).tolist()
ROW_HASH = row_ids_sha256(ROW_IDS)

meta_cols = [
    "row_id",
    "subject_num",
    "task_label",
    "rotation_direction",
    "load_band_mid_pct_mvic",
    "current_time_sec",
    "current_borg",
    "target_time_sec",
    "target_borg",
    "borg_change_10s",
    "trial_duration_sec_qc",
]
meta_cols = [c for c in meta_cols if c in S.columns]
S[meta_cols].to_csv(META_OUT, index=False)


# =============================================================================
# 2. Frozen locked MOMENT model
# =============================================================================

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable.")

device = torch.device("cuda:0")
torch.manual_seed(BASE_SEED)

model = MOMENTPipeline.from_pretrained(
    str(MODEL_DIR),
    model_kwargs={"task_name": "embedding"},
)
model.init()
model.eval()

for p in model.parameters():
    p.requires_grad_(False)

if any(p.requires_grad for p in model.parameters()):
    raise RuntimeError("MOMENT model is not fully frozen.")

model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

if trainable_params != 0:
    raise RuntimeError("Frozen-model guard failed.")


# =============================================================================
# 3. Resume state
# =============================================================================

N = len(S)

start_emb = np.full((N, EMBED_DIM), np.nan, dtype=np.float32)
end_emb = np.full((N, EMBED_DIM), np.nan, dtype=np.float32)

completed = np.zeros(N, dtype=bool)

n_locations_start = np.zeros(N, dtype=np.int16)
n_locations_end = np.zeros(N, dtype=np.int16)

masked_start = np.zeros(N, dtype=np.int16)
masked_end = np.zeros(N, dtype=np.int16)

seconds_row = np.full(N, np.nan, dtype=np.float32)
peak_mb_row = np.full(N, np.nan, dtype=np.float32)

resume_used = False

LOCKED_POLICY_META = {
    "row_hash": ROW_HASH,
    "n_rows": N,
    "embed_dim": EMBED_DIM,
    "model_revision": LOCKED_REVISION,
    "config_sha256": EXPECTED_CONFIG_SHA256,
    "weights_sha256": EXPECTED_WEIGHTS_SHA256,
    "sampling_rate_hz": IMU_HZ,
    "window_seconds": WIN_SEC,
    "input_samples": EXPECTED_SAMPLES,
    "seq_len": SEQ_LEN,
    "patch_len": PATCH_LEN,
    "chunk_a_valid": CHUNK_A_VALID,
    "chunk_b_valid": CHUNK_B_VALID,
    "chunk_b_pad": CHUNK_B_PAD,
    "patches_a": PATCHES_A,
    "patches_b": PATCHES_B,
    "representation": REPRESENTATION_NAME,
}

if PARTIAL_OUT.exists() and PARTIAL_META.exists():
    try:
        m = json.loads(PARTIAL_META.read_text(encoding="utf-8"))

        compatible = all(
            m.get(k) == v for k, v in LOCKED_POLICY_META.items()
        )

        if compatible:
            z = np.load(PARTIAL_OUT, allow_pickle=False)

            start_emb[:] = z["start_emb"]
            end_emb[:] = z["end_emb"]
            completed[:] = z["completed"].astype(bool)
            n_locations_start[:] = z["n_locations_start"]
            n_locations_end[:] = z["n_locations_end"]
            masked_start[:] = z["masked_start"]
            masked_end[:] = z["masked_end"]
            seconds_row[:] = z["seconds_row"]
            peak_mb_row[:] = z["peak_mb_row"]

            # Check that every resumed completed row is actually finite.
            resumed_idx = np.where(completed)[0]
            if len(resumed_idx):
                if not np.isfinite(start_emb[resumed_idx]).all():
                    raise RuntimeError("Resumed START embeddings contain non-finite values.")
                if not np.isfinite(end_emb[resumed_idx]).all():
                    raise RuntimeError("Resumed END embeddings contain non-finite values.")

            resume_used = True
            print(
                f"Resume checkpoint accepted: "
                f"{int(completed.sum())}/{N} rows already complete."
            )
        else:
            print(
                "Existing MOMENT partial checkpoint is incompatible with the "
                "locked row/model/input policy; starting clean."
            )
    except Exception as e:
        print(f"Could not use partial checkpoint ({e}); starting clean.")


def save_partial():
    np.savez_compressed(
        PARTIAL_OUT,
        start_emb=start_emb,
        end_emb=end_emb,
        completed=completed.astype(np.uint8),
        n_locations_start=n_locations_start,
        n_locations_end=n_locations_end,
        masked_start=masked_start,
        masked_end=masked_end,
        seconds_row=seconds_row,
        peak_mb_row=peak_mb_row,
    )

    m = dict(LOCKED_POLICY_META)
    m["completed"] = int(completed.sum())
    m["updated_unix"] = time.time()

    PARTIAL_META.write_text(
        json.dumps(m, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# 4. Raw trial loader — one active trial only
# =============================================================================

_active_trial_key = None
_active_trial_data = None


def load_trial(sn: int, task: str):
    global _active_trial_key, _active_trial_data

    key = (int(sn), str(task))

    if key == _active_trial_key:
        return _active_trial_data

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
                f"Missing IMU file(s): subject={sn}, task={task}, location={loc}"
            )

        acc = readnum(acc_fp)
        gyr = readnum(gyr_fp)

        if len(acc) != len(gyr):
            raise RuntimeError(
                f"ACC/GYR length mismatch: subject={sn}, task={task}, "
                f"location={loc}, acc={len(acc)}, gyr={len(gyr)}"
            )

        out[loc] = {
            "acc": acc,
            "gyr": gyr,
            "n": len(acc),
        }

    _active_trial_key = key
    _active_trial_data = out
    return out


# =============================================================================
# 5. MOMENT exact per-location encoder
# =============================================================================

mask_a_cpu = torch.ones((1, SEQ_LEN), dtype=torch.long)

mask_b_cpu = torch.zeros((1, SEQ_LEN), dtype=torch.long)
mask_b_cpu[:, :CHUNK_B_VALID] = 1


def encode_location(raw_6x1000: np.ndarray):
    if raw_6x1000.shape != (6, EXPECTED_SAMPLES):
        raise RuntimeError(
            f"Expected per-location raw shape (6,{EXPECTED_SAMPLES}); "
            f"got {raw_6x1000.shape}"
        )

    if not np.isfinite(raw_6x1000).all():
        raise RuntimeError("Non-finite raw IMU input.")

    x1000 = torch.from_numpy(
        raw_6x1000.astype(np.float32, copy=False)
    ).unsqueeze(0)

    chunk_a = x1000[:, :, :CHUNK_A_VALID].contiguous()

    chunk_b = torch.zeros(
        (1, 6, SEQ_LEN),
        dtype=torch.float32,
    )
    chunk_b[:, :, :CHUNK_B_VALID] = x1000[:, :, CHUNK_A_VALID:EXPECTED_SAMPLES]

    # Exact conservation check: all original 1000 samples used once.
    reconstructed = torch.cat(
        [chunk_a, chunk_b[:, :, :CHUNK_B_VALID]],
        dim=-1,
    )

    if not torch.equal(reconstructed, x1000):
        raise RuntimeError("1000-sample chunk conservation failed.")

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    with torch.inference_mode():
        out_a = model(
            x_enc=chunk_a.to(device),
            input_mask=mask_a_cpu.to(device),
            reduction="mean",
        )

        out_b = model(
            x_enc=chunk_b.to(device),
            input_mask=mask_b_cpu.to(device),
            reduction="mean",
        )

        emb_a = out_a.embeddings
        emb_b = out_b.embeddings

        pooled = (
            PATCHES_A * emb_a + PATCHES_B * emb_b
        ) / PATCHES_TOTAL

    torch.cuda.synchronize(device)

    elapsed = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    if tuple(emb_a.shape) != (1, EMBED_DIM):
        raise RuntimeError(
            f"Unexpected chunk-A embedding shape: {tuple(emb_a.shape)}"
        )

    if tuple(emb_b.shape) != (1, EMBED_DIM):
        raise RuntimeError(
            f"Unexpected chunk-B embedding shape: {tuple(emb_b.shape)}"
        )

    if tuple(pooled.shape) != (1, EMBED_DIM):
        raise RuntimeError(
            f"Unexpected pooled location embedding shape: {tuple(pooled.shape)}"
        )

    if not torch.isfinite(emb_a).all():
        raise RuntimeError("Non-finite chunk-A MOMENT embedding.")

    if not torch.isfinite(emb_b).all():
        raise RuntimeError("Non-finite chunk-B MOMENT embedding.")

    if not torch.isfinite(pooled).all():
        raise RuntimeError("Non-finite pooled MOMENT location embedding.")

    result = pooled.squeeze(0).detach().cpu().numpy().astype(np.float32)

    del out_a, out_b, emb_a, emb_b, pooled
    torch.cuda.empty_cache()

    return result, elapsed, peak_mb


# =============================================================================
# 6. One-anchor row extraction
# =============================================================================

def extract_anchor(row: pd.Series, trial: dict, anchor: str):
    sn = int(row["subject_num"])
    task = str(row["task_label"])
    t = float(row["current_time_sec"])
    dur = float(row["trial_duration_sec_qc"])
    rid = str(row["row_id"])

    location_embeddings = []
    masked_locations = []

    max_peak_mb = 0.0
    total_location_seconds = 0.0

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
                f"{rid} {anchor} {loc}: 10-s window out of bounds."
            )

        a, b = bounds

        acc = obj["acc"][a:b, :3]
        gyr = obj["gyr"][a:b, :3]

        if acc.shape != (EXPECTED_SAMPLES, 3):
            raise RuntimeError(
                f"{rid} {anchor} {loc}: ACC shape {acc.shape}"
            )

        if gyr.shape != (EXPECTED_SAMPLES, 3):
            raise RuntimeError(
                f"{rid} {anchor} {loc}: GYR shape {gyr.shape}"
            )

        raw = np.concatenate(
            [acc.T, gyr.T],
            axis=0,
        ).astype(np.float32)

        if raw.shape != (6, EXPECTED_SAMPLES):
            raise RuntimeError(
                f"{rid} {anchor} {loc}: raw shape {raw.shape}"
            )

        if not np.isfinite(raw).all():
            raise RuntimeError(
                f"{rid} {anchor} {loc}: non-finite raw IMU values."
            )

        loc_embedding, sec, peak_mb = encode_location(raw)

        location_embeddings.append(loc_embedding)
        total_location_seconds += float(sec)
        max_peak_mb = max(max_peak_mb, float(peak_mb))

    if not location_embeddings:
        raise RuntimeError(
            f"{rid} {anchor}: no available IMU locations."
        )

    global_embedding = np.mean(
        np.stack(location_embeddings, axis=0),
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    if global_embedding.shape != (EMBED_DIM,):
        raise RuntimeError(
            f"{rid} {anchor}: global embedding shape {global_embedding.shape}"
        )

    if not np.isfinite(global_embedding).all():
        raise RuntimeError(
            f"{rid} {anchor}: global embedding non-finite."
        )

    return (
        global_embedding,
        len(location_embeddings),
        len(masked_locations),
        total_location_seconds,
        max_peak_mb,
    )


# =============================================================================
# 7. Full extraction
# =============================================================================

t_run = time.time()
fresh_done = 0

for i in range(N):
    if completed[i]:
        continue

    row_s = S.iloc[i]
    row_e = E.iloc[i]

    if str(row_s["row_id"]) != str(row_e["row_id"]):
        raise RuntimeError(f"START/END row mismatch at index {i}")

    # Metadata equality across anchors.
    for c in [
        "subject_num",
        "task_label",
        "current_time_sec",
        "trial_duration_sec_qc",
    ]:
        sv = row_s[c]
        ev = row_e[c]

        try:
            same = bool(np.isclose(float(sv), float(ev), atol=1e-9))
        except Exception:
            same = str(sv) == str(ev)

        if not same:
            raise RuntimeError(
                f"{row_s['row_id']}: START/END metadata differs for {c}"
            )

    sn = int(row_s["subject_num"])
    task = str(row_s["task_label"])

    trial = load_trial(sn, task)

    row_t0 = time.time()

    (
        s_emb,
        nls,
        mls,
        s_forward_sec,
        pks,
    ) = extract_anchor(
        row_s,
        trial,
        "START",
    )

    (
        e_emb,
        nle,
        mle,
        e_forward_sec,
        pke,
    ) = extract_anchor(
        row_e,
        trial,
        "END",
    )

    start_emb[i] = s_emb
    end_emb[i] = e_emb

    n_locations_start[i] = nls
    n_locations_end[i] = nle

    masked_start[i] = mls
    masked_end[i] = mle

    seconds_row[i] = float(time.time() - row_t0)
    peak_mb_row[i] = float(max(pks, pke))

    completed[i] = True
    fresh_done += 1

    if (
        fresh_done % SAVE_EVERY == 0
        or int(completed.sum()) == N
    ):
        save_partial()

        elapsed = time.time() - t_run
        fresh_rate = elapsed / max(fresh_done, 1)

        remaining = N - int(completed.sum())
        eta_min = remaining * fresh_rate / 60.0

        print(
            f"Completed {int(completed.sum())}/{N} locked row-pairs | "
            f"fresh this run={fresh_done} | "
            f"mean {fresh_rate:.2f} sec/row-pair | "
            f"ETA {eta_min:.1f} min"
        )

if not completed.all():
    raise RuntimeError(
        f"Extraction ended incomplete: {int(completed.sum())}/{N}"
    )


# =============================================================================
# 8. Final validation
# =============================================================================

for name, arr in [
    ("START MOMENT", start_emb),
    ("END MOMENT", end_emb),
]:
    if arr.shape != (N, EMBED_DIM):
        raise RuntimeError(
            f"{name}: unexpected shape {arr.shape}"
        )

    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr).all(axis=1))[0]
        raise RuntimeError(
            f"{name}: non-finite rows {bad[:20].tolist()}"
        )

expected_mask = (
    (S["subject_num"] == 3)
    & (S["task_label"] == "task1_35i")
).to_numpy()

if int(expected_mask.sum()) != 17:
    raise RuntimeError(
        f"Locked known-mask row count changed: expected 17, got {int(expected_mask.sum())}"
    )

if not np.array_equal(masked_start > 0, expected_mask):
    raise RuntimeError(
        "START masked-row structure differs from locked policy."
    )

if not np.array_equal(masked_end > 0, expected_mask):
    raise RuntimeError(
        "END masked-row structure differs from locked policy."
    )

if not np.all(n_locations_start[~expected_mask] == 6):
    raise RuntimeError(
        "Unexpected START location count outside known mask."
    )

if not np.all(n_locations_end[~expected_mask] == 6):
    raise RuntimeError(
        "Unexpected END location count outside known mask."
    )

if not np.all(n_locations_start[expected_mask] == 5):
    raise RuntimeError(
        "Known START mask rows do not have exactly 5 locations."
    )

if not np.all(n_locations_end[expected_mask] == 5):
    raise RuntimeError(
        "Known END mask rows do not have exactly 5 locations."
    )


# =============================================================================
# 9. Final outputs
# =============================================================================

save_embedding_npz(
    START_OUT,
    ROW_IDS,
    start_emb,
    "START",
)

save_embedding_npz(
    END_OUT,
    ROW_IDS,
    end_emb,
    "END",
)

manifest = pd.DataFrame({
    "row_id": ROW_IDS,
    "subject_num": S["subject_num"].to_numpy(),
    "task_label": S["task_label"].astype(str).to_numpy(),
    "current_time_sec": S["current_time_sec"].to_numpy(float),
    "target_time_sec": S["target_time_sec"].to_numpy(float),

    "start_available_location_count": n_locations_start,
    "end_available_location_count": n_locations_end,

    "start_masked_location_count": masked_start,
    "end_masked_location_count": masked_end,

    "seconds_for_start_plus_end_row_pair": seconds_row,
    "max_peak_allocated_mb": peak_mb_row,
})

manifest.to_csv(MANIFEST_OUT, index=False)

# Only remove resume files after all final outputs exist.
if PARTIAL_OUT.exists():
    PARTIAL_OUT.unlink()

if PARTIAL_META.exists():
    PARTIAL_META.unlink()


# =============================================================================
# 10. Audit
# =============================================================================

elapsed = time.time() - t_run


def cosine_rows(A: np.ndarray, B: np.ndarray):
    num = np.sum(A * B, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)

    return np.divide(
        num,
        den,
        out=np.full_like(num, np.nan, dtype=np.float32),
        where=den > 0,
    )


cos = cosine_rows(start_emb, end_emb)

lines = []
A = lines.append

A("DATASET C — FULL FROZEN MOMENT-1-BASE IMU REPRESENTATION EXTRACTION AUDIT (SCRIPT 32)")
A("=" * 118)

A(f"Locked rows: {N}")
A(f"Participants: {S['subject_num'].nunique()}")
A(
    "Subject-task trials: "
    f"{S[['subject_num','task_label']].drop_duplicates().shape[0]}"
)
A(
    "START/END identical row IDs: "
    f"{list(S['row_id'].astype(str)) == list(E['row_id'].astype(str))}"
)
A(f"Locked row-id SHA256: {ROW_HASH}")
A(f"Resume checkpoint used at start: {resume_used}")
A("")

A("1. MODEL PROVENANCE")
A("-" * 118)

A("Model: AutonLab/MOMENT-1-base")
A(f"Locked revision: {LOCKED_REVISION}")
A(f"Local model directory: {MODEL_DIR}")
A(f"config.json SHA256: {config_sha}")
A(f"model.safetensors SHA256: {weights_sha}")
A(f"Config seq_len: {cfg['seq_len']}")
A(f"Config patch_len: {cfg['patch_len']}")
A(f"Config d_model: {cfg['t5_config']['d_model']}")
A(f"Model class: {model.__class__.__name__}")
A(f"Task name: {model.task_name}")
A(f"Total parameters: {total_params}")
A(f"Trainable parameters: {trainable_params}")
A(
    "All model parameters frozen: "
    f"{all(not p.requires_grad for p in model.parameters())}"
)
A(f"GPU: {torch.cuda.get_device_name(0)}")
A("")

A("2. FIXED REPRESENTATION / INPUT POLICY")
A("-" * 118)

A("Primary modality: IMU only.")
A(f"Sampling rate: {IMU_HZ} Hz.")
A("Resampling: NO.")
A(f"Exact prior window: {WIN_SEC:.0f} s = {EXPECTED_SAMPLES} samples/axis.")
A(
    "Per location channels: "
    "ACC_x, ACC_y, ACC_z, GYR_x, GYR_y, GYR_z."
)
A("Each physical location is encoded separately.")
A("No waveform normalization/clipping/winsorization/interpolation/repair.")
A("")
A("MOMENT sequence policy:")
A("  chunk A = samples [0:512], 512 valid samples = 64 valid patches.")
A("  chunk B = samples [512:1000], 488 valid samples + 24 right-pad.")
A("  final 24 padded samples are invalid in input_mask.")
A("  chunk valid-patch pooling weights = 64/125 and 61/125.")
A("  all 1000 original samples are used exactly once.")
A("")
A(
    "Representation: MOMENT embedding reduction='mean' per chunk "
    "-> valid-patch-weighted two-chunk mean "
    "-> available-location mean -> 768-D."
)
A(
    "Known Subject 3/task1_35i Shoulder location is omitted from "
    "location pooling; never zero-filled."
)
A("No supervised model fitting or downstream result inspection occurs in this script.")
A("")

A("3. EXTRACTION COMPLETENESS")
A("-" * 118)

A(f"Completed rows: {int(completed.sum())}/{N}")
A(f"START embedding shape: {start_emb.shape}")
A(f"END embedding shape: {end_emb.shape}")
A(f"START all finite: {np.isfinite(start_emb).all()}")
A(f"END all finite: {np.isfinite(end_emb).all()}")

A(
    "Rows with known masked location START: "
    f"{int((masked_start > 0).sum())}"
)
A(
    "Rows with known masked location END: "
    f"{int((masked_end > 0).sum())}"
)

A(
    "Ordinary rows with 6 locations START: "
    f"{int((n_locations_start == 6).sum())}"
)
A(
    "Ordinary rows with 6 locations END: "
    f"{int((n_locations_end == 6).sum())}"
)

A(
    "Known-mask rows with 5 locations START: "
    f"{int((n_locations_start == 5).sum())}"
)
A(
    "Known-mask rows with 5 locations END: "
    f"{int((n_locations_end == 5).sum())}"
)
A("")

A("4. RUNTIME / GPU")
A("-" * 118)

A(f"Elapsed minutes this run: {elapsed / 60:.2f}")
A(
    "Median seconds per START+END row-pair: "
    f"{float(np.nanmedian(seconds_row)):.4f}"
)
A(
    "Mean seconds per START+END row-pair: "
    f"{float(np.nanmean(seconds_row)):.4f}"
)
A(
    "Max recorded peak allocated MB: "
    f"{float(np.nanmax(peak_mb_row)):.2f}"
)
A("")

A("5. START/END REPRESENTATION SIMILARITY — DESCRIPTIVE QC ONLY")
A("-" * 118)

A(
    "MOMENT mean768 cosine: "
    f"median={float(np.nanmedian(cos)):.8f}, "
    f"mean={float(np.nanmean(cos)):.8f}, "
    f"min={float(np.nanmin(cos)):.8f}, "
    f"max={float(np.nanmax(cos)):.8f}"
)
A(
    "These cosine statistics are representation/infrastructure diagnostics only."
)
A(
    "They are NOT predictive-value evidence and do not alter the locked analysis plan."
)
A("")

A("6. OUTPUT FILE HASHES")
A("-" * 118)

for fp in [
    META_OUT,
    START_OUT,
    END_OUT,
    MANIFEST_OUT,
]:
    A(
        f"{fp.name} | bytes={fp.stat().st_size} | "
        f"SHA256={sha256_file(fp)}"
    )

A("")

A("7. DECISION GATE")
A("-" * 118)

global_pass = (
    completed.all()
    and start_emb.shape == (N, EMBED_DIM)
    and end_emb.shape == (N, EMBED_DIM)
    and np.isfinite(start_emb).all()
    and np.isfinite(end_emb).all()
    and int((masked_start > 0).sum()) == 17
    and int((masked_end > 0).sum()) == 17
    and np.all(n_locations_start[expected_mask] == 5)
    and np.all(n_locations_end[expected_mask] == 5)
    and np.all(n_locations_start[~expected_mask] == 6)
    and np.all(n_locations_end[~expected_mask] == 6)
    and trainable_params == 0
    and config_sha == EXPECTED_CONFIG_SHA256
    and weights_sha == EXPECTED_WEIGHTS_SHA256
)

A(
    "DATASET C FULL FROZEN MOMENT-1-BASE IMU EXTRACTION PASS: "
    f"{global_pass}"
)

if global_pass:
    A(
        "Interpretation: the complete locked 1141-row START/END cohort "
        "has frozen MOMENT-1-base IMU representations under the "
        "prespecified chunk/mask/location-aware policy."
    )
    A(
        "No incremental predictive-value claim has yet been evaluated."
    )
else:
    A(
        "Interpretation: DO NOT run the downstream MOMENT incremental benchmark."
    )

A("")
A("8. NEXT STEP")
A("-" * 118)
A("Upload this audit to ChatGPT.")
A(
    "Only after review: evaluate MOMENT as the single generic frozen "
    "time-series comparator against the literal locked 18f C0a/C0b baselines "
    "and exact locked outer splits."
)

A("=" * 118)
A("END OF FULL FROZEN MOMENT-1-BASE EXTRACTION AUDIT")

AUDIT_OUT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print()
print("Created:")

for fp in [
    META_OUT,
    START_OUT,
    END_OUT,
    MANIFEST_OUT,
    AUDIT_OUT,
]:
    print(" ", fp)

print()
print(
    "DATASET C FULL FROZEN MOMENT-1-BASE IMU EXTRACTION PASS: "
    f"{global_pass}"
)
print(f"Elapsed minutes this run: {elapsed / 60:.2f}")
print(
    "Upload dataset_C_moment1base_full_frozen_extraction_audit_32.txt "
    "to ChatGPT."
)
