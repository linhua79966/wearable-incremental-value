from pathlib import Path
import re
import time
import json
import hashlib
import numpy as np
import pandas as pd
import torch

torch.uint64 = torch.int64
torch.uint32 = torch.int32
torch.uint16 = torch.int16

from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel

# =============================================================================
# 24_dataset_C_normwear_imu_full_frozen_extraction.py
#
# FULL frozen NormWear-IMU representation extraction for the LOCKED Dataset C
# QC-clean cohort.
#
# Scientific design is fixed BEFORE downstream results:
#   - 1141 rows, 34 participants
#   - START + END dual anchor
#   - raw IMU only
#   - 100 Hz, no resampling
#   - 10-s [t-10,t] window
#   - six channels/location: ACC xyz + GYR xyz
#   - physical locations encoded separately
#   - known S03/task1_35i Shoulder location omitted, never zero-filled
#
# Two representations are extracted:
#
# PRIMARY:
#   patch mean -> 6-channel mean -> available-location mean = 768-D
#
# PRESPECIFIED SENSITIVITY:
#   first encoder patch [CLS] -> 6-channel mean -> available-location mean
#   = 768-D
#
# This script DOES NOT:
#   - fit any supervised model;
#   - read prediction errors/incremental-value results;
#   - fine-tune NormWear;
#   - modify QC;
#   - normalize/clip/winsorize/interpolate/repair raw signals;
#   - choose pooling based on downstream performance.
#
# Resume:
#   Progress is checkpointed every SAVE_EVERY rows.
#   Re-running resumes from the partial .npz if compatible.
#
# Final outputs:
#   derived/dataset_C_normwear_imu/
#     dataset_C_normwear_imu_locked_metadata.csv
#     dataset_C_normwear_imu_start_mean768.npz
#     dataset_C_normwear_imu_end_mean768.npz
#     dataset_C_normwear_imu_start_cls768.npz
#     dataset_C_normwear_imu_end_cls768.npz
#     dataset_C_normwear_imu_extraction_manifest.csv
#
#   audit/
#     dataset_C_normwear_imu_full_frozen_extraction_audit.txt
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "dataset_C_shoulder_rotation" / "raw" / "WSD4FEDSRM"
SR = DS / "EMG, IMU, and PPG data"

DERIVED = ROOT / "derived" / "dataset_C_phase1"
START_LOCKED = DERIVED / "dataset_C_handcrafted_start_anchor_qc.csv"
END_LOCKED = DERIVED / "dataset_C_handcrafted_end_anchor_qc.csv"

HF_ROOT = ROOT / "third_party" / "NormWear_HF_official"
HF_WEIGHTS = HF_ROOT / "model.safetensors"
HF_REVISION = HF_ROOT / "LOCKED_HF_REVISION.txt"

OUTDIR = ROOT / "derived" / "dataset_C_normwear_imu"
AUDITDIR = ROOT / "audit"
OUTDIR.mkdir(parents=True, exist_ok=True)
AUDITDIR.mkdir(parents=True, exist_ok=True)

META_OUT = OUTDIR / "dataset_C_normwear_imu_locked_metadata.csv"
START_MEAN_OUT = OUTDIR / "dataset_C_normwear_imu_start_mean768.npz"
END_MEAN_OUT = OUTDIR / "dataset_C_normwear_imu_end_mean768.npz"
START_CLS_OUT = OUTDIR / "dataset_C_normwear_imu_start_cls768.npz"
END_CLS_OUT = OUTDIR / "dataset_C_normwear_imu_end_cls768.npz"
MANIFEST_OUT = OUTDIR / "dataset_C_normwear_imu_extraction_manifest.csv"

PARTIAL_OUT = OUTDIR / "_dataset_C_normwear_imu_partial_resume.npz"
PARTIAL_META = OUTDIR / "_dataset_C_normwear_imu_partial_resume.json"

AUDIT_OUT = AUDITDIR / "dataset_C_normwear_imu_full_frozen_extraction_audit.txt"

IMU_HZ = 100
WIN_SEC = 10.0
EXPECTED_SAMPLES = int(IMU_HZ * WIN_SEC)
EMBED_DIM = 768

SAVE_EVERY = 25
BASE_SEED = 20260903

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


def safe(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def b_start(t, rate, n):
    a = int(round((float(t) - WIN_SEC) * rate))
    b = int(round(float(t) * rate))
    return (a, b) if a >= 0 and b <= n and b > a else None


def b_end(t, dur, rate, n):
    a = int(round(n - (float(dur) - (float(t) - WIN_SEC)) * rate))
    b = int(round(n - (float(dur) - float(t)) * rate))
    return (a, b) if a >= 0 and b <= n and b > a else None


def readnum(fp):
    x = pd.read_csv(fp)
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    a = x.to_numpy(dtype=np.float32)
    if a.ndim != 2 or a.shape[1] < 3:
        raise RuntimeError(f"{fp}: expected >=3 numeric columns, got {a.shape}")
    return a[:, :3]


def known_location_quarantine(subject_num, task_label, location):
    return (
        int(subject_num) == 3
        and str(task_label) == "task1_35i"
        and str(location) == "Shoulder"
    )


def sha256_file(fp):
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def row_ids_sha256(row_ids):
    s = "\n".join(map(str, row_ids)).encode("utf-8")
    return hashlib.sha256(s).hexdigest().upper()


def save_array_npz(fp, row_ids, arr, representation, anchor):
    np.savez_compressed(
        fp,
        row_id=np.asarray(row_ids, dtype=str),
        embedding=np.asarray(arr, dtype=np.float32),
        anchor=np.asarray([anchor]),
        representation=np.asarray([representation]),
        sampling_rate_hz=np.asarray([IMU_HZ], dtype=np.int32),
        window_seconds=np.asarray([WIN_SEC], dtype=np.float32),
    )


# =============================================================================
# 1. Locked cohort
# =============================================================================
for fp in [START_LOCKED, END_LOCKED, HF_WEIGHTS]:
    if not fp.exists():
        raise SystemExit(f"Missing required input: {fp}")

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
    raise SystemExit(f"Expected 1141 rows, got START={len(S)} END={len(E)}")

if S["subject_num"].nunique() != 34:
    raise SystemExit(f"Expected 34 participants, got {S['subject_num'].nunique()}")

if list(S["row_id"]) != list(E["row_id"]):
    raise SystemExit("START/END row IDs differ.")

if not np.allclose(
    S["target_borg"].to_numpy(float),
    E["target_borg"].to_numpy(float),
    equal_nan=True,
):
    raise SystemExit("START/END targets differ.")

if set(S["row_id"]) & LOCKED_REMOVED_QC_IDS:
    raise SystemExit("18e/18f removed QC rows unexpectedly remain.")

ROW_IDS = S["row_id"].astype(str).tolist()
ROW_HASH = row_ids_sha256(ROW_IDS)

# Metadata is saved once and must remain identical across embeddings.
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
# 2. Frozen official HF model, validated 22d load path
# =============================================================================
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable.")

device = torch.device("cuda")
torch.manual_seed(BASE_SEED)

config = AutoConfig.from_pretrained(
    str(HF_ROOT),
    trust_remote_code=True,
    local_files_only=True,
)

model = AutoModel.from_config(
    config,
    trust_remote_code=True,
)

if not hasattr(model, "normwear"):
    raise RuntimeError("Official HF wrapper has no model.normwear.")

state = load_file(str(HF_WEIGHTS), device="cpu")

base_keys = set(model.normwear.state_dict().keys())
state_keys = set(state.keys())

if base_keys != state_keys:
    raise RuntimeError(
        f"Checkpoint/base-model mismatch: "
        f"missing={len(base_keys-state_keys)}, "
        f"unexpected={len(state_keys-base_keys)}"
    )

result = model.normwear.load_state_dict(state, strict=True)
if result.missing_keys or result.unexpected_keys:
    raise RuntimeError("strict=True load unexpectedly reported mismatches.")

del state

if any(getattr(p, "is_meta", False) for p in model.parameters()):
    raise RuntimeError("Meta parameters remain after checkpoint load.")

model.eval()
for p in model.parameters():
    p.requires_grad_(False)

if any(p.requires_grad for p in model.parameters()):
    raise RuntimeError("Model is not fully frozen.")

model = model.to(device)

# =============================================================================
# 3. Resume state
# =============================================================================
N = len(S)

start_mean = np.full((N, EMBED_DIM), np.nan, dtype=np.float32)
end_mean = np.full((N, EMBED_DIM), np.nan, dtype=np.float32)
start_cls = np.full((N, EMBED_DIM), np.nan, dtype=np.float32)
end_cls = np.full((N, EMBED_DIM), np.nan, dtype=np.float32)

completed = np.zeros(N, dtype=bool)
n_locations_start = np.zeros(N, dtype=np.int16)
n_locations_end = np.zeros(N, dtype=np.int16)
masked_start = np.zeros(N, dtype=np.int16)
masked_end = np.zeros(N, dtype=np.int16)
seconds_row = np.full(N, np.nan, dtype=np.float32)
peak_mb_row = np.full(N, np.nan, dtype=np.float32)

resume_used = False

if PARTIAL_OUT.exists() and PARTIAL_META.exists():
    try:
        m = json.loads(PARTIAL_META.read_text(encoding="utf-8"))
        if (
            m.get("row_hash") == ROW_HASH
            and int(m.get("n_rows", -1)) == N
            and int(m.get("embed_dim", -1)) == EMBED_DIM
        ):
            z = np.load(PARTIAL_OUT, allow_pickle=False)
            start_mean[:] = z["start_mean"]
            end_mean[:] = z["end_mean"]
            start_cls[:] = z["start_cls"]
            end_cls[:] = z["end_cls"]
            completed[:] = z["completed"].astype(bool)
            n_locations_start[:] = z["n_locations_start"]
            n_locations_end[:] = z["n_locations_end"]
            masked_start[:] = z["masked_start"]
            masked_end[:] = z["masked_end"]
            seconds_row[:] = z["seconds_row"]
            peak_mb_row[:] = z["peak_mb_row"]
            resume_used = True
            print(
                f"Resume checkpoint accepted: "
                f"{int(completed.sum())}/{N} rows already complete."
            )
        else:
            print("Existing partial checkpoint incompatible; starting clean.")
    except Exception as e:
        print(f"Could not use partial checkpoint ({e}); starting clean.")


def save_partial():
    np.savez_compressed(
        PARTIAL_OUT,
        start_mean=start_mean,
        end_mean=end_mean,
        start_cls=start_cls,
        end_cls=end_cls,
        completed=completed.astype(np.uint8),
        n_locations_start=n_locations_start,
        n_locations_end=n_locations_end,
        masked_start=masked_start,
        masked_end=masked_end,
        seconds_row=seconds_row,
        peak_mb_row=peak_mb_row,
    )

    PARTIAL_META.write_text(
        json.dumps(
            {
                "row_hash": ROW_HASH,
                "n_rows": N,
                "embed_dim": EMBED_DIM,
                "completed": int(completed.sum()),
                "hf_revision": (
                    HF_REVISION.read_text(encoding="utf-8").strip()
                    if HF_REVISION.exists()
                    else None
                ),
                "checkpoint_sha256": sha256_file(HF_WEIGHTS),
                "updated_unix": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# 4. Raw trial cache
# =============================================================================
trial_cache = {}


def load_trial(sn, task):
    key = (int(sn), str(task))
    if key in trial_cache:
        return trial_cache[key]

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
                f"ACC/GYR length mismatch: subject={sn}, task={task}, "
                f"loc={loc}, acc={len(acc)}, gyr={len(gyr)}"
            )

        out[loc] = {"acc": acc, "gyr": gyr, "n": len(acc)}

    trial_cache[key] = out
    return out


# =============================================================================
# 5. One-anchor row extraction
# =============================================================================
def extract_anchor(row, trial, anchor):
    sn = int(row["subject_num"])
    task = str(row["task_label"])
    t = float(row["current_time_sec"])
    dur = float(row["trial_duration_sec_qc"])
    rid = str(row["row_id"])

    loc_mean_embeddings = []
    loc_cls_embeddings = []
    masked_locations = []
    max_peak_mb = 0.0

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
            raise RuntimeError(f"{rid} {anchor} {loc}: window out of bounds.")

        a, b = bounds
        acc = obj["acc"][a:b, :3]
        gyr = obj["gyr"][a:b, :3]

        if acc.shape != (EXPECTED_SAMPLES, 3):
            raise RuntimeError(f"{rid} {anchor} {loc}: ACC shape {acc.shape}")
        if gyr.shape != (EXPECTED_SAMPLES, 3):
            raise RuntimeError(f"{rid} {anchor} {loc}: GYR shape {gyr.shape}")

        raw = np.concatenate([acc.T, gyr.T], axis=0).astype(np.float32)

        if raw.shape != (6, EXPECTED_SAMPLES):
            raise RuntimeError(f"{rid} {anchor} {loc}: raw shape {raw.shape}")
        if not np.isfinite(raw).all():
            raise RuntimeError(f"{rid} {anchor} {loc}: non-finite raw values")

        x = torch.from_numpy(raw).unsqueeze(0).to(device)

        torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            outpack = model(
                x,
                return_spec=False,
                return_enc_out=True,
                return_dec_out=False,
                zero_shot_input_pack=None,
            )

        torch.cuda.synchronize()

        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        max_peak_mb = max(max_peak_mb, peak_mb)

        z = outpack.get("enc_out")
        if z is None:
            raise RuntimeError(f"{rid} {anchor} {loc}: enc_out missing")

        if (
            z.ndim != 4
            or z.shape[0] != 1
            or z.shape[1] != 6
            or z.shape[-1] != EMBED_DIM
        ):
            raise RuntimeError(
                f"{rid} {anchor} {loc}: embedding shape {tuple(z.shape)}"
            )

        if not torch.isfinite(z).all():
            raise RuntimeError(
                f"{rid} {anchor} {loc}: non-finite embedding"
            )

        patch_mean = z.mean(dim=2)          # [1,6,768]
        loc_mean = patch_mean.mean(dim=1)  # [1,768]

        cls_ch = z[:, :, 0, :]             # [1,6,768]
        loc_cls = cls_ch.mean(dim=1)       # [1,768]

        loc_mean_embeddings.append(loc_mean.squeeze(0).cpu())
        loc_cls_embeddings.append(loc_cls.squeeze(0).cpu())

        del x, outpack, z, patch_mean, loc_mean, cls_ch, loc_cls
        torch.cuda.empty_cache()

    if not loc_mean_embeddings:
        raise RuntimeError(f"{rid} {anchor}: no usable IMU locations")

    global_mean = torch.stack(loc_mean_embeddings).mean(dim=0).numpy().astype(np.float32)
    global_cls = torch.stack(loc_cls_embeddings).mean(dim=0).numpy().astype(np.float32)

    if global_mean.shape != (EMBED_DIM,) or global_cls.shape != (EMBED_DIM,):
        raise RuntimeError(f"{rid} {anchor}: global embedding shape failure")

    if not np.isfinite(global_mean).all() or not np.isfinite(global_cls).all():
        raise RuntimeError(f"{rid} {anchor}: non-finite global embedding")

    return (
        global_mean,
        global_cls,
        len(loc_mean_embeddings),
        len(masked_locations),
        max_peak_mb,
    )


# =============================================================================
# 6. Full extraction
# =============================================================================
t0 = time.time()
fresh_done = 0

for i in range(N):
    if completed[i]:
        continue

    row_s = S.iloc[i]
    row_e = E.iloc[i]

    if str(row_s["row_id"]) != str(row_e["row_id"]):
        raise RuntimeError(f"Row mismatch at index {i}")

    sn = int(row_s["subject_num"])
    task = str(row_s["task_label"])
    trial = load_trial(sn, task)

    tr = time.time()

    (
        sm,
        sc,
        nls,
        mls,
        pks,
    ) = extract_anchor(row_s, trial, "START")

    (
        em,
        ec,
        nle,
        mle,
        pke,
    ) = extract_anchor(row_e, trial, "END")

    start_mean[i] = sm
    start_cls[i] = sc
    end_mean[i] = em
    end_cls[i] = ec

    n_locations_start[i] = nls
    n_locations_end[i] = nle
    masked_start[i] = mls
    masked_end[i] = mle

    seconds_row[i] = float(time.time() - tr)
    peak_mb_row[i] = float(max(pks, pke))

    completed[i] = True
    fresh_done += 1

    if (
        fresh_done % SAVE_EVERY == 0
        or completed.sum() == N
    ):
        save_partial()

        elapsed = time.time() - t0
        newly_rate = elapsed / max(fresh_done, 1)
        remaining = N - int(completed.sum())
        eta_min = remaining * newly_rate / 60.0

        print(
            f"Completed {int(completed.sum())}/{N} locked rows | "
            f"fresh this run={fresh_done} | "
            f"mean {newly_rate:.2f} sec/row | "
            f"ETA {eta_min:.1f} min"
        )

if not completed.all():
    raise RuntimeError(
        f"Extraction ended incomplete: {int(completed.sum())}/{N}"
    )

# =============================================================================
# 7. Final validation
# =============================================================================
for name, arr in [
    ("START mean", start_mean),
    ("END mean", end_mean),
    ("START cls", start_cls),
    ("END cls", end_cls),
]:
    if arr.shape != (N, EMBED_DIM):
        raise RuntimeError(f"{name}: unexpected shape {arr.shape}")
    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr).all(axis=1))[0]
        raise RuntimeError(f"{name}: non-finite rows {bad[:20].tolist()}")

# Expected mask structure: 17 rows for S03/task1_35i; one physical location.
expected_mask = (
    (S["subject_num"] == 3)
    & (S["task_label"] == "task1_35i")
).to_numpy()

if not np.array_equal(masked_start > 0, expected_mask):
    raise RuntimeError("START masked-row structure differs from locked policy.")

if not np.array_equal(masked_end > 0, expected_mask):
    raise RuntimeError("END masked-row structure differs from locked policy.")

if not np.all(n_locations_start[~expected_mask] == 6):
    raise RuntimeError("Unexpected START location count outside known mask.")

if not np.all(n_locations_end[~expected_mask] == 6):
    raise RuntimeError("Unexpected END location count outside known mask.")

if not np.all(n_locations_start[expected_mask] == 5):
    raise RuntimeError("Known START mask rows do not have exactly 5 locations.")

if not np.all(n_locations_end[expected_mask] == 5):
    raise RuntimeError("Known END mask rows do not have exactly 5 locations.")

# =============================================================================
# 8. Final outputs
# =============================================================================
save_array_npz(
    START_MEAN_OUT,
    ROW_IDS,
    start_mean,
    "patch_mean__channel_mean__available_location_mean",
    "START",
)

save_array_npz(
    END_MEAN_OUT,
    ROW_IDS,
    end_mean,
    "patch_mean__channel_mean__available_location_mean",
    "END",
)

save_array_npz(
    START_CLS_OUT,
    ROW_IDS,
    start_cls,
    "first_encoder_patch_cls__channel_mean__available_location_mean",
    "START",
)

save_array_npz(
    END_CLS_OUT,
    ROW_IDS,
    end_cls,
    "first_encoder_patch_cls__channel_mean__available_location_mean",
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
    "seconds_for_start_plus_end": seconds_row,
    "max_peak_allocated_mb": peak_mb_row,
})

manifest.to_csv(MANIFEST_OUT, index=False)

# Remove resume files only after all final outputs exist.
if PARTIAL_OUT.exists():
    PARTIAL_OUT.unlink()
if PARTIAL_META.exists():
    PARTIAL_META.unlink()

# =============================================================================
# 9. Audit
# =============================================================================
elapsed = time.time() - t0

def cosine_rows(A, B):
    num = np.sum(A * B, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return np.divide(
        num,
        den,
        out=np.full_like(num, np.nan, dtype=np.float32),
        where=den > 0,
    )

cos_mean = cosine_rows(start_mean, end_mean)
cos_cls = cosine_rows(start_cls, end_cls)

lines = []
A = lines.append

A("DATASET C — FULL FROZEN NORMWEAR-IMU REPRESENTATION EXTRACTION AUDIT")
A("=" * 118)
A(f"Locked rows: {N}")
A(f"Participants: {S['subject_num'].nunique()}")
A(f"Subject-task trials: {S[['subject_num','task_label']].drop_duplicates().shape[0]}")
A(f"START/END identical row IDs: {list(S['row_id']) == list(E['row_id'])}")
A(f"Locked row-id SHA256: {ROW_HASH}")
A(f"Resume checkpoint used at start: {resume_used}")
A("")
A("1. MODEL PROVENANCE")
A("-" * 118)
A(f"Official HF directory: {HF_ROOT}")
A(
    "Locked HF revision: "
    + (
        HF_REVISION.read_text(encoding='utf-8').strip()
        if HF_REVISION.exists()
        else "UNAVAILABLE"
    )
)
A(f"model.safetensors SHA256: {sha256_file(HF_WEIGHTS)}")
A(f"Wrapper class: {type(model)}")
A(f"Base-model prefix: {getattr(model, 'base_model_prefix', None)}")
A(f"All model parameters frozen: {all(not p.requires_grad for p in model.parameters())}")
A("")
A("2. FIXED REPRESENTATION POLICY")
A("-" * 118)
A("Primary modality: IMU only.")
A(f"Sampling rate: {IMU_HZ} Hz.")
A("Resampling: NO.")
A(f"Prior window: {WIN_SEC:.0f} s = {EXPECTED_SAMPLES} samples/axis.")
A("Per location channels: ACC_x, ACC_y, ACC_z, GYR_x, GYR_y, GYR_z.")
A("Each physical location is encoded separately.")
A("PRIMARY representation: patch mean -> six-channel mean -> available-location mean -> 768-D.")
A("SENSITIVITY representation: first encoder patch [CLS] -> six-channel mean -> available-location mean -> 768-D.")
A("Known Subject 3/task1_35i Shoulder location is omitted from location pooling; never zero-filled.")
A("No normalization/clipping/winsorization/interpolation/repair/target-driven transformation.")
A("No supervised model fitting or downstream result inspection occurs in this script.")
A("")
A("3. EXTRACTION COMPLETENESS")
A("-" * 118)
A(f"Completed rows: {int(completed.sum())}/{N}")
A(f"START primary embedding shape: {start_mean.shape}")
A(f"END primary embedding shape: {end_mean.shape}")
A(f"START CLS sensitivity shape: {start_cls.shape}")
A(f"END CLS sensitivity shape: {end_cls.shape}")
A(f"START primary all finite: {np.isfinite(start_mean).all()}")
A(f"END primary all finite: {np.isfinite(end_mean).all()}")
A(f"START CLS all finite: {np.isfinite(start_cls).all()}")
A(f"END CLS all finite: {np.isfinite(end_cls).all()}")
A(f"Rows with known masked location START: {int((masked_start>0).sum())}")
A(f"Rows with known masked location END: {int((masked_end>0).sum())}")
A(f"Ordinary rows with 6 locations START: {int((n_locations_start==6).sum())}")
A(f"Ordinary rows with 6 locations END: {int((n_locations_end==6).sum())}")
A(f"Known-mask rows with 5 locations START: {int((n_locations_start==5).sum())}")
A(f"Known-mask rows with 5 locations END: {int((n_locations_end==5).sum())}")
A("")
A("4. RUNTIME / GPU")
A("-" * 118)
A(f"Elapsed minutes this run: {elapsed/60:.2f}")
A(f"Median seconds per row-pair: {float(np.nanmedian(seconds_row)):.4f}")
A(f"Mean seconds per row-pair: {float(np.nanmean(seconds_row)):.4f}")
A(f"Max recorded peak allocated MB: {float(np.nanmax(peak_mb_row)):.2f}")
A("")
A("5. START/END REPRESENTATION SIMILARITY — DESCRIPTIVE QC ONLY")
A("-" * 118)
A(
    "Primary mean-pool cosine: "
    f"median={float(np.nanmedian(cos_mean)):.8f}, "
    f"mean={float(np.nanmean(cos_mean)):.8f}, "
    f"min={float(np.nanmin(cos_mean)):.8f}, "
    f"max={float(np.nanmax(cos_mean)):.8f}"
)
A(
    "CLS sensitivity cosine: "
    f"median={float(np.nanmedian(cos_cls)):.8f}, "
    f"mean={float(np.nanmean(cos_cls)):.8f}, "
    f"min={float(np.nanmin(cos_cls)):.8f}, "
    f"max={float(np.nanmax(cos_cls)):.8f}"
)
A("These cosine statistics are representation/infrastructure diagnostics only.")
A("They are NOT predictive-value results and do not alter the locked analysis plan.")
A("")
A("6. OUTPUT FILE HASHES")
A("-" * 118)
for fp in [
    META_OUT,
    START_MEAN_OUT,
    END_MEAN_OUT,
    START_CLS_OUT,
    END_CLS_OUT,
    MANIFEST_OUT,
]:
    A(f"{fp.name} | bytes={fp.stat().st_size} | SHA256={sha256_file(fp)}")
A("")
A("7. DECISION GATE")
A("-" * 118)

global_pass = (
    completed.all()
    and np.isfinite(start_mean).all()
    and np.isfinite(end_mean).all()
    and np.isfinite(start_cls).all()
    and np.isfinite(end_cls).all()
    and int((masked_start > 0).sum()) == 17
    and int((masked_end > 0).sum()) == 17
    and np.all(n_locations_start[expected_mask] == 5)
    and np.all(n_locations_end[expected_mask] == 5)
    and np.all(n_locations_start[~expected_mask] == 6)
    and np.all(n_locations_end[~expected_mask] == 6)
)

A(f"DATASET C FULL FROZEN NORMWEAR-IMU EXTRACTION PASS: {global_pass}")

if global_pass:
    A("Interpretation: the complete locked 1141-row START/END cohort has reproducible frozen NormWear-IMU")
    A("representations under the prespecified pooling and mask-aware policy.")
    A("No incremental predictive-value claim has yet been evaluated.")
else:
    A("Interpretation: DO NOT run the downstream benchmark.")

A("")
A("8. NEXT STEP")
A("-" * 118)
A("Upload this audit to ChatGPT.")
A("Only after review: build the participant-grouped frozen-representation downstream benchmark,")
A("keeping the handcrafted 18f C0a/C0b context baselines unchanged.")
A("=" * 118)
A("END OF FULL FROZEN EXTRACTION AUDIT")

AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("")
print("Created:")
for fp in [
    META_OUT,
    START_MEAN_OUT,
    END_MEAN_OUT,
    START_CLS_OUT,
    END_CLS_OUT,
    MANIFEST_OUT,
    AUDIT_OUT,
]:
    print(" ", fp)

print("")
print(f"DATASET C FULL FROZEN NORMWEAR-IMU EXTRACTION PASS: {global_pass}")
print(f"Elapsed minutes this run: {elapsed/60:.2f}")
print("Upload dataset_C_normwear_imu_full_frozen_extraction_audit.txt to ChatGPT.")
