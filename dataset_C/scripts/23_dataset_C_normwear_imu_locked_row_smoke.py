from pathlib import Path
import re
import time
import hashlib
import numpy as np
import pandas as pd
import torch

# Compatibility aliases used by official NormWear HF model card/code.
torch.uint64 = torch.int64
torch.uint32 = torch.int32
torch.uint16 = torch.int16

from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel

# =============================================================================
# 23_dataset_C_normwear_imu_locked_row_smoke.py
#
# Purpose
#   First REAL Dataset C pretrained NormWear smoke test.
#
# It uses:
#   - the LOCKED 18f QC-clean 1141-row cohort;
#   - exact START/END 10-s raw windows from the locked handcrafted design;
#   - IMU only (primary NormWear-supported modality);
#   - raw 100 Hz samples WITHOUT resampling;
#   - six channels per physical IMU location:
#         ACC_x, ACC_y, ACC_z, GYR_x, GYR_y, GYR_z
#   - one physical location at a time;
#   - patch mean -> channel mean -> location embedding (768-D);
#   - mean over AVAILABLE locations -> global 768-D representation;
#   - CLS pooling is retained only as a prespecified sensitivity representation.
#
# Smoke cases:
#   A) one deterministic ordinary locked row;
#   B) one deterministic Subject 3 / task1_35i row with the locked
#      Shoulder ACC/GYR whole-location quarantine.
#
# For both cases, START and END are processed.
#
# This script does NOT:
#   - fit a downstream predictive model;
#   - inspect incremental-value results;
#   - fine-tune NormWear;
#   - alter QC;
#   - impute or zero-fill a quarantined Shoulder location;
#   - resample the 100-Hz Dataset C IMU.
#
# Output:
#   audit/dataset_C_normwear_imu_locked_row_smoke_audit.txt
#   derived/dataset_C_foundation_raw_windows/
#       dataset_C_normwear_imu_locked_row_smoke_summary.csv
#       dataset_C_normwear_imu_locked_row_smoke_embeddings.npz
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

OUTDIR = ROOT / "derived" / "dataset_C_foundation_raw_windows"
AUDITDIR = ROOT / "audit"
OUTDIR.mkdir(parents=True, exist_ok=True)
AUDITDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUTDIR / "dataset_C_normwear_imu_locked_row_smoke_summary.csv"
EMBED_OUT = OUTDIR / "dataset_C_normwear_imu_locked_row_smoke_embeddings.npz"
AUDIT_OUT = AUDITDIR / "dataset_C_normwear_imu_locked_row_smoke_audit.txt"

IMU_HZ = 100
WIN_SEC = 10.0
EXPECTED_SAMPLES = int(IMU_HZ * WIN_SEC)

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

BASE_SEED = 20260903


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


# =============================================================================
# 1. Locked cohort guardrails
# =============================================================================
for fp in [START_LOCKED, END_LOCKED, HF_WEIGHTS]:
    if not fp.exists():
        raise SystemExit(f"Missing required locked input: {fp}")

S = pd.read_csv(START_LOCKED).sort_values("row_id").reset_index(drop=True)
E = pd.read_csv(END_LOCKED).sort_values("row_id").reset_index(drop=True)

required = {
    "row_id", "subject_num", "task_label", "current_time_sec",
    "target_time_sec", "trial_duration_sec_qc",
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

if list(S["row_id"]) != list(E["row_id"]):
    raise SystemExit("START/END row IDs differ.")

if set(S["row_id"]) & LOCKED_REMOVED_QC_IDS:
    raise SystemExit("18e/18f removed QC rows unexpectedly remain.")

# Deterministic smoke cases.
normal_candidates = S[
    ~((S["subject_num"] == 3) & (S["task_label"] == "task1_35i"))
].copy()
if len(normal_candidates) == 0:
    raise SystemExit("No ordinary smoke row available.")
normal_row_id = str(normal_candidates.iloc[0]["row_id"])

mask_candidates = S[
    (S["subject_num"] == 3)
    & (S["task_label"] == "task1_35i")
].copy()
if len(mask_candidates) == 0:
    raise SystemExit("No locked Subject 3 / task1_35i row available.")
masked_row_id = str(mask_candidates.iloc[0]["row_id"])

SMOKE_CASES = [
    ("ordinary", normal_row_id),
    ("known_shoulder_mask", masked_row_id),
]

# =============================================================================
# 2. Official HF model: reproduce the validated 22d strict base-model load.
# =============================================================================
device = torch.device("cuda")
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable.")

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
    raise RuntimeError("Official HF wrapper has no model.normwear base model.")

state = load_file(str(HF_WEIGHTS), device="cpu")

base_keys = set(model.normwear.state_dict().keys())
state_keys = set(state.keys())
if base_keys != state_keys:
    raise RuntimeError(
        f"HF checkpoint/base-model keys differ: "
        f"missing={len(base_keys-state_keys)}, unexpected={len(state_keys-base_keys)}"
    )

result = model.normwear.load_state_dict(state, strict=True)
if result.missing_keys or result.unexpected_keys:
    raise RuntimeError("strict=True load unexpectedly reported mismatches.")

del state

if any(getattr(p, "is_meta", False) for p in model.parameters()):
    raise RuntimeError("Meta parameters remain after strict load.")

model.eval()
for p in model.parameters():
    p.requires_grad_(False)

if any(p.requires_grad for p in model.parameters()):
    raise RuntimeError("Model is not fully frozen.")

model = model.to(device)

# =============================================================================
# 3. Raw trial cache
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

        out[loc] = {
            "acc": acc,
            "gyr": gyr,
            "n": len(acc),
        }

    trial_cache[key] = out
    return out


# =============================================================================
# 4. Smoke extraction
# =============================================================================
summary_rows = []
npz_payload = {}

t_global = time.time()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

for case_name, row_id in SMOKE_CASES:
    srow = S.loc[S["row_id"] == row_id].iloc[0]
    erow = E.loc[E["row_id"] == row_id].iloc[0]

    # Metadata must be identical across locked anchors.
    for c in ["subject_num", "task_label", "current_time_sec", "trial_duration_sec_qc"]:
        if str(srow[c]) != str(erow[c]):
            # numeric text formatting could differ; check numeric where possible.
            try:
                if not np.isclose(float(srow[c]), float(erow[c]), atol=1e-9):
                    raise RuntimeError(f"{row_id}: START/END metadata differs for {c}")
            except Exception:
                if srow[c] != erow[c]:
                    raise RuntimeError(f"{row_id}: START/END metadata differs for {c}")

    sn = int(srow["subject_num"])
    task = str(srow["task_label"])
    t = float(srow["current_time_sec"])
    dur = float(srow["trial_duration_sec_qc"])

    trial = load_trial(sn, task)

    anchor_globals_mean = {}
    anchor_globals_cls = {}

    for anchor in ["START", "END"]:
        location_mean_embeddings = []
        location_cls_embeddings = []
        available_locations = []
        masked_locations = []
        per_location_shapes = []
        per_location_seconds = []
        per_location_peak_mb = []
        all_raw_finite = True

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

            finite = bool(np.isfinite(raw).all())
            all_raw_finite &= finite
            if not finite:
                raise RuntimeError(
                    f"{row_id} {anchor} {loc}: non-finite raw IMU values."
                )

            x = torch.from_numpy(raw).unsqueeze(0).to(device)

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()

            with torch.inference_mode():
                outpack = model(
                    x,
                    return_spec=False,
                    return_enc_out=True,
                    return_dec_out=False,
                    zero_shot_input_pack=None,
                )

            torch.cuda.synchronize()
            sec = time.time() - t0
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2

            if "enc_out" not in outpack:
                raise RuntimeError(
                    f"{row_id} {anchor} {loc}: enc_out missing."
                )

            z = outpack["enc_out"]

            if z.ndim != 4 or z.shape[0] != 1 or z.shape[1] != 6 or z.shape[-1] != 768:
                raise RuntimeError(
                    f"{row_id} {anchor} {loc}: unexpected embedding shape {tuple(z.shape)}"
                )
            if not torch.isfinite(z).all():
                raise RuntimeError(
                    f"{row_id} {anchor} {loc}: non-finite embedding."
                )

            # Primary: mean across all encoder patches, then mean across six IMU axes.
            patch_mean = z.mean(dim=2)          # [1,6,768]
            loc_mean = patch_mean.mean(dim=1)  # [1,768]

            # Prespecified sensitivity: official first-patch [CLS], mean across axes.
            cls_ch = z[:, :, 0, :]             # [1,6,768]
            loc_cls = cls_ch.mean(dim=1)       # [1,768]

            location_mean_embeddings.append(loc_mean.squeeze(0).detach().cpu())
            location_cls_embeddings.append(loc_cls.squeeze(0).detach().cpu())
            available_locations.append(loc)
            per_location_shapes.append(str(tuple(z.shape)))
            per_location_seconds.append(sec)
            per_location_peak_mb.append(peak_mb)

            del x, outpack, z, patch_mean, loc_mean, cls_ch, loc_cls
            torch.cuda.empty_cache()

        if not location_mean_embeddings:
            raise RuntimeError(f"{row_id} {anchor}: no available IMU locations.")

        global_mean = torch.stack(location_mean_embeddings, dim=0).mean(dim=0)
        global_cls = torch.stack(location_cls_embeddings, dim=0).mean(dim=0)

        if global_mean.shape != (768,) or global_cls.shape != (768,):
            raise RuntimeError(
                f"{row_id} {anchor}: global embedding shape failure."
            )
        if not torch.isfinite(global_mean).all() or not torch.isfinite(global_cls).all():
            raise RuntimeError(
                f"{row_id} {anchor}: global embedding non-finite."
            )

        anchor_globals_mean[anchor] = global_mean.numpy()
        anchor_globals_cls[anchor] = global_cls.numpy()

        npz_payload[f"{case_name}__{anchor}__mean_pool"] = global_mean.numpy()
        npz_payload[f"{case_name}__{anchor}__cls_pool"] = global_cls.numpy()

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
            "available_location_count": len(available_locations),
            "available_locations": "|".join(available_locations),
            "masked_location_count": len(masked_locations),
            "masked_locations": "|".join(masked_locations),
            "raw_all_finite": all_raw_finite,
            "embedding_shapes": "|".join(per_location_shapes),
            "global_mean_dim": int(global_mean.numel()),
            "global_cls_dim": int(global_cls.numel()),
            "mean_pool_finite": bool(torch.isfinite(global_mean).all().item()),
            "cls_pool_finite": bool(torch.isfinite(global_cls).all().item()),
            "mean_location_forward_seconds": float(np.mean(per_location_seconds)),
            "max_location_forward_seconds": float(np.max(per_location_seconds)),
            "max_location_peak_allocated_mb": float(np.max(per_location_peak_mb)),
        })

    # Descriptive START/END representation similarity only; not a scientific result.
    for pool_name, d in [
        ("mean_pool", anchor_globals_mean),
        ("cls_pool", anchor_globals_cls),
    ]:
        a = np.asarray(d["START"], dtype=np.float64)
        b = np.asarray(d["END"], dtype=np.float64)
        den = np.linalg.norm(a) * np.linalg.norm(b)
        cos = float(np.dot(a, b) / den) if den > 0 else np.nan
        npz_payload[f"{case_name}__start_end_cosine__{pool_name}"] = np.array([cos], dtype=np.float32)

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_OUT, index=False)
np.savez_compressed(EMBED_OUT, **npz_payload)

# =============================================================================
# 5. Audit
# =============================================================================
lines = []
A = lines.append

A("DATASET C — NORMWEAR IMU REAL LOCKED-ROW START/END SMOKE AUDIT")
A("=" * 118)
A(f"Locked cohort rows: {len(S)}")
A(f"Locked participants: {S['subject_num'].nunique()}")
A(f"START/END identical row IDs: {list(S['row_id']) == list(E['row_id'])}")
A(f"Ordinary smoke row: {normal_row_id}")
A(f"Known-mask smoke row: {masked_row_id}")
A("")
A("1. FROZEN MODEL PROVENANCE")
A("-" * 118)
A(f"Official HF directory: {HF_ROOT}")
A(f"Locked HF revision: {HF_REVISION.read_text(encoding='utf-8').strip() if HF_REVISION.exists() else 'UNAVAILABLE'}")
A(f"model.safetensors SHA256: {sha256_file(HF_WEIGHTS)}")
A(f"HF wrapper class: {type(model)}")
A(f"Base-model prefix: {getattr(model, 'base_model_prefix', None)}")
A(f"All parameters frozen: {all(not p.requires_grad for p in model.parameters())}")
A("")
A("2. LOCKED IMU INPUT POLICY")
A("-" * 118)
A("Primary modality in this step: IMU only.")
A("Dataset C IMU nominal sampling rate: 100 Hz.")
A("Resampling applied: NO.")
A("Reason: preserve official NormWear behavior for <=256-Hz inputs; do not invent a 65-Hz conversion for this dataset.")
A("Each physical location is encoded separately as six channels in fixed order:")
A("  ACC_x, ACC_y, ACC_z, GYR_x, GYR_y, GYR_z")
A("Each channel contributes the full locked 10-s [t-10,t] window = 1000 samples.")
A("No normalization, clipping, winsorization, interpolation, repair, or target-driven transformation is applied.")
A("Primary pooling: mean across encoder patches -> mean across six axes -> mean across available locations.")
A("Prespecified sensitivity pooling: first encoder patch [CLS] -> mean across six axes -> mean across available locations.")
A("The known Subject 3/task1_35i Shoulder ACC+GYR quarantine is handled by OMITTING that physical location")
A("from the across-location mean; it is never zero-filled and the forecast row is not deleted.")
A("")
A("3. SMOKE RESULTS")
A("-" * 118)

for _, r in summary.iterrows():
    A(
        f"{r['case']} | {r['row_id']} | {r['anchor']} | "
        f"available_locations={r['available_location_count']} | "
        f"masked={r['masked_locations'] if r['masked_locations'] else '<none>'} | "
        f"shapes={r['embedding_shapes']} | "
        f"mean_pool_finite={r['mean_pool_finite']} | "
        f"cls_pool_finite={r['cls_pool_finite']} | "
        f"max_peak_MB={r['max_location_peak_allocated_mb']:.2f} | "
        f"mean_sec/location={r['mean_location_forward_seconds']:.3f}"
    )

A("")
A("4. START/END DESCRIPTIVE COSINE SIMILARITY — SMOKE ONLY")
A("-" * 118)
for case_name, _ in SMOKE_CASES:
    for pool in ["mean_pool", "cls_pool"]:
        val = float(npz_payload[f"{case_name}__start_end_cosine__{pool}"][0])
        A(f"{case_name} | {pool} | cosine={val:.8f}")
A("These cosine values are infrastructure diagnostics only and are NOT a scientific result.")
A("")
A("5. DECISION GATE")
A("-" * 118)

expected_rows = 4
all_ok = (
    len(summary) == expected_rows
    and summary["raw_all_finite"].all()
    and summary["mean_pool_finite"].all()
    and summary["cls_pool_finite"].all()
    and set(summary.loc[summary["case"] == "ordinary", "available_location_count"]) == {6}
    and set(summary.loc[summary["case"] == "ordinary", "masked_location_count"]) == {0}
    and set(summary.loc[summary["case"] == "known_shoulder_mask", "available_location_count"]) == {5}
    and set(summary.loc[summary["case"] == "known_shoulder_mask", "masked_location_count"]) == {1}
)

A(f"DATASET C NORMWEAR IMU LOCKED-ROW SMOKE PASS: {all_ok}")
if all_ok:
    A("Interpretation: real QC-clean START/END Dataset C IMU windows can be encoded with the locked frozen")
    A("official-HF NormWear model under the fixed per-location/mask-aware pooling policy.")
    A("No predictive-value claim is made at this stage.")
else:
    A("Interpretation: DO NOT run full 1141-row extraction; inspect the smoke summary first.")
A("")
A("6. OUTPUTS")
A("-" * 118)
A(str(SUMMARY_OUT))
A(str(EMBED_OUT))
A(str(AUDIT_OUT))
A("")
A(f"Elapsed minutes: {(time.time() - t_global)/60:.2f}")
A("=" * 118)
A("END OF REAL LOCKED-ROW NORMWEAR IMU SMOKE AUDIT")

AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

print("Created:")
print(" ", SUMMARY_OUT)
print(" ", EMBED_OUT)
print(" ", AUDIT_OUT)
print("")
print(f"DATASET C NORMWEAR IMU LOCKED-ROW SMOKE PASS: {all_ok}")
print("Upload dataset_C_normwear_imu_locked_row_smoke_audit.txt to ChatGPT.")
