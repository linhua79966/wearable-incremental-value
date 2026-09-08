from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from momentfm import MOMENTPipeline


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "audit"
MODEL_DIR = ROOT / "third_party" / "MOMENT-1-base_locked_5e44b0e"
AUDIT_OUT = AUDIT_DIR / "moment1base_locked_model_load_gpu_smoke_audit_30.txt"

REPO_ID = "AutonLab/MOMENT-1-base"
REVISION = "5e44b0ea26376a176360f87831124e018f876d96"

EXPECTED_SEQ_LEN = 512
EXPECTED_PATCH_LEN = 8
EXPECTED_D_MODEL = 768

AUDIT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def add(lines: list[str], s: str = "") -> None:
    lines.append(str(s))


lines: list[str] = []

add(lines, "MOMENT-1-BASE LOCKED-REVISION MODEL-LOAD + GPU SMOKE AUDIT (SCRIPT 30)")
add(lines, "=" * 118)
add(lines, f"Python executable: {sys.executable}")
add(lines, f"Python version: {sys.version}")
add(lines, f"Platform: {platform.platform()}")
add(lines, f"torch: {torch.__version__}")
add(lines, f"torch CUDA build: {torch.version.cuda}")
add(lines, f"CUDA available: {torch.cuda.is_available()}")
add(lines, f"Repository: {REPO_ID}")
add(lines, f"Locked revision: {REVISION}")
add(lines, "Purpose: download only config.json + model.safetensors at the locked revision, load embedding-mode MOMENT-1-base,")
add(lines, "freeze all parameters, move to GPU, and test the exact 512/488+24-mask chunk policy without using Borg outcomes.")
add(lines, "")

global_pass = False

try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)

    add(lines, "1. LOCKED SNAPSHOT DOWNLOAD")
    add(lines, "-" * 118)
    add(lines, f"GPU: {gpu_name}")

    snapshot_path = snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        allow_patterns=["config.json", "model.safetensors"],
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    snapshot_path = Path(snapshot_path)

    config_path = snapshot_path / "config.json"
    weights_path = snapshot_path / "model.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing locked config: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing locked safetensors: {weights_path}")

    config_sha = sha256_file(config_path)
    weights_sha = sha256_file(weights_path)

    add(lines, f"Local locked snapshot: {snapshot_path}")
    add(lines, f"config.json bytes: {config_path.stat().st_size}")
    add(lines, f"config.json SHA256: {config_sha}")
    add(lines, f"model.safetensors bytes: {weights_path.stat().st_size}")
    add(lines, f"model.safetensors SHA256: {weights_sha}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    seq_len = int(config["seq_len"])
    patch_len = int(config["patch_len"])
    d_model = int(config["t5_config"]["d_model"])

    add(lines, f"config seq_len: {seq_len}")
    add(lines, f"config patch_len: {patch_len}")
    add(lines, f"config d_model: {d_model}")

    if seq_len != EXPECTED_SEQ_LEN:
        raise RuntimeError(f"Expected seq_len={EXPECTED_SEQ_LEN}, found {seq_len}.")
    if patch_len != EXPECTED_PATCH_LEN:
        raise RuntimeError(f"Expected patch_len={EXPECTED_PATCH_LEN}, found {patch_len}.")
    if d_model != EXPECTED_D_MODEL:
        raise RuntimeError(f"Expected d_model={EXPECTED_D_MODEL}, found {d_model}.")

    add(lines, "")
    add(lines, "2. MODEL LOAD / FREEZE")
    add(lines, "-" * 118)

    model = MOMENTPipeline.from_pretrained(
        str(snapshot_path),
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    add(lines, f"Model class: {model.__class__.__name__}")
    add(lines, f"Task name after init: {model.task_name}")
    add(lines, f"Total parameters: {total_params}")
    add(lines, f"Trainable parameters after freeze: {trainable_params}")

    if trainable_params != 0:
        raise RuntimeError("Frozen-model policy failed: trainable parameters remain.")

    model = model.to(device)

    add(lines, "")
    add(lines, "3. EXACT INPUT-POLICY GPU SMOKE")
    add(lines, "-" * 118)
    add(lines, "Policy lock:")
    add(lines, "  original physical-location window = 6 channels x 1000 samples at 100 Hz")
    add(lines, "  chunk A = samples 0:512, all 512 valid")
    add(lines, "  chunk B = samples 512:1000, 488 valid + 24 right-pad invalid")
    add(lines, "  chunk embeddings weighted by 64/125 and 61/125 valid patches")
    add(lines, "  reduction='mean' inside MOMENT averages across channels and valid patches")
    add(lines, "")

    # Deterministic synthetic data; no scientific result is inferred from this.
    g = torch.Generator(device="cpu")
    g.manual_seed(20260905)
    x1000 = torch.randn((1, 6, 1000), generator=g, dtype=torch.float32)

    chunk_a = x1000[:, :, :512].contiguous()
    chunk_b_valid = x1000[:, :, 512:1000].contiguous()
    chunk_b = torch.zeros((1, 6, 512), dtype=torch.float32)
    chunk_b[:, :, :488] = chunk_b_valid

    mask_a = torch.ones((1, 512), dtype=torch.long)
    mask_b = torch.zeros((1, 512), dtype=torch.long)
    mask_b[:, :488] = 1

    # Check exact conservation of the 1000 original samples.
    reconstructed = torch.cat([chunk_a, chunk_b[:, :, :488]], dim=-1)
    sample_conservation_exact = torch.equal(reconstructed, x1000)

    if not sample_conservation_exact:
        raise RuntimeError("1000-sample chunk policy failed exact sample conservation.")

    torch.cuda.reset_peak_memory_stats(device)

    def embed_once():
        with torch.inference_mode():
            oa = model(
                x_enc=chunk_a.to(device),
                input_mask=mask_a.to(device),
                reduction="mean",
            )
            ob = model(
                x_enc=chunk_b.to(device),
                input_mask=mask_b.to(device),
                reduction="mean",
            )

            ea = oa.embeddings
            eb = ob.embeddings

            pooled = (64.0 * ea + 61.0 * eb) / 125.0
            return ea.detach().cpu(), eb.detach().cpu(), pooled.detach().cpu()

    ea1, eb1, ep1 = embed_once()
    ea2, eb2, ep2 = embed_once()

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    add(lines, f"chunk A embedding shape: {tuple(ea1.shape)}")
    add(lines, f"chunk B embedding shape: {tuple(eb1.shape)}")
    add(lines, f"pooled embedding shape: {tuple(ep1.shape)}")
    add(lines, f"chunk A finite: {bool(torch.isfinite(ea1).all())}")
    add(lines, f"chunk B finite: {bool(torch.isfinite(eb1).all())}")
    add(lines, f"pooled finite: {bool(torch.isfinite(ep1).all())}")
    add(lines, f"sample conservation exact: {sample_conservation_exact}")
    add(lines, f"repeat max abs diff chunk A: {(ea1 - ea2).abs().max().item():.12g}")
    add(lines, f"repeat max abs diff chunk B: {(eb1 - eb2).abs().max().item():.12g}")
    add(lines, f"repeat max abs diff pooled: {(ep1 - ep2).abs().max().item():.12g}")
    add(lines, f"GPU peak allocated MB: {peak_mb:.2f}")

    if tuple(ea1.shape) != (1, EXPECTED_D_MODEL):
        raise RuntimeError(f"Unexpected chunk A embedding shape {tuple(ea1.shape)}.")
    if tuple(eb1.shape) != (1, EXPECTED_D_MODEL):
        raise RuntimeError(f"Unexpected chunk B embedding shape {tuple(eb1.shape)}.")
    if tuple(ep1.shape) != (1, EXPECTED_D_MODEL):
        raise RuntimeError(f"Unexpected pooled embedding shape {tuple(ep1.shape)}.")
    if not torch.isfinite(ea1).all():
        raise RuntimeError("Non-finite chunk A embedding.")
    if not torch.isfinite(eb1).all():
        raise RuntimeError("Non-finite chunk B embedding.")
    if not torch.isfinite(ep1).all():
        raise RuntimeError("Non-finite pooled embedding.")

    # eval + inference_mode should be deterministic on repeated identical input.
    if not torch.equal(ea1, ea2):
        raise RuntimeError("Repeated chunk A embedding is not bitwise identical.")
    if not torch.equal(eb1, eb2):
        raise RuntimeError("Repeated chunk B embedding is not bitwise identical.")
    if not torch.equal(ep1, ep2):
        raise RuntimeError("Repeated pooled embedding is not bitwise identical.")

    add(lines, "")
    add(lines, "4. SCIENTIFIC / TECHNICAL GUARDRAILS")
    add(lines, "-" * 118)
    add(lines, "PASS: official MOMENT-1-base revision is pinned before any Dataset C scientific result.")
    add(lines, "PASS: only safetensors weights are placed in the locked local model directory.")
    add(lines, "PASS: model is frozen and evaluation-only.")
    add(lines, "PASS: no Borg target or supervised downstream model is used in this script.")
    add(lines, "PASS: no resampling or truncation is used.")
    add(lines, "PASS: 1000 original samples are used exactly once across the two chunks.")
    add(lines, "PASS: padded tail is excluded by input_mask.")
    add(lines, "PASS: chunk pooling weights are fixed before downstream results: 64/125 and 61/125.")
    add(lines, "")
    add(lines, "NOTE: This script is a model-load/synthetic-policy smoke only.")
    add(lines, "A separate locked-row raw-window smoke must PASS before full 1141-row START/END extraction.")

    global_pass = True

except Exception as exc:
    add(lines, "")
    add(lines, "ERROR")
    add(lines, "-" * 118)
    add(lines, f"{type(exc).__name__}: {exc}")
    global_pass = False

add(lines, "")
add(lines, "5. DECISION GATE")
add(lines, "-" * 118)
add(lines, f"MOMENT-1-BASE LOCKED MODEL-LOAD GPU SMOKE PASS: {global_pass}")

if global_pass:
    add(lines, "Ready for a locked Dataset C raw-window row smoke test.")
else:
    add(lines, "Do NOT proceed to Dataset C MOMENT extraction.")

AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")

print()
print("Created:")
print(f"  {AUDIT_OUT}")
print()
print(f"MOMENT-1-BASE LOCKED MODEL-LOAD GPU SMOKE PASS: {global_pass}")
print("Upload moment1base_locked_model_load_gpu_smoke_audit_30.txt to ChatGPT.")
