#!/usr/bin/env python3
"""
complete.py

Minimal text generation script that loads the latest distilled checkpoint.
"""

from __future__ import annotations

import argparse
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from distill import swap_hf_decoders, STUDENT_DECODER_KIND, STUDENT_DECODER_KWARGS

HF_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
HF_MODEL_DIR = "models/smolllm2_135m"
HF_TRUST_REMOTE_CODE = False


def _hf_load_path() -> str:
    if os.path.isdir(HF_MODEL_DIR):
        return HF_MODEL_DIR
    return HF_MODEL_ID


def _pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _find_latest_run_dir(runs_dir: str) -> str:
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"Runs dir not found: {runs_dir}")
    candidates = []
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name)
        if os.path.isdir(path):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No run directories in: {runs_dir}")
    return max(candidates, key=lambda p: os.path.getmtime(p))


def _find_latest_student_ckpt(run_dir: str) -> str:
    pattern = re.compile(r"^student_(\d+)\.pt$")
    best_step = None
    best_path = None
    for name in os.listdir(run_dir):
        m = pattern.match(name)
        if not m:
            continue
        step = int(m.group(1))
        if best_step is None or step > best_step:
            best_step = step
            best_path = os.path.join(run_dir, name)
    if best_path is None:
        raise FileNotFoundError(f"No student_*.pt checkpoints in: {run_dir}")
    return best_path


def _load_state_dict_strict(model: torch.nn.Module, state_dict: dict) -> None:
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing or unexpected:
        msg = []
        if missing:
            msg.append(f"missing={len(missing)}")
        if unexpected:
            msg.append(f"unexpected={len(unexpected)}")
        raise RuntimeError("Checkpoint/model mismatch (" + ", ".join(msg) + ")")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="here is my prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--run-dir", type=str, default=None, help="Specific run directory to load from.")
    parser.add_argument("--ckpt", type=str, default=None, help="Specific checkpoint file to load.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _pick_dtype(device)

    hf_path = _hf_load_path()
    tok = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=HF_TRUST_REMOTE_CODE)
    model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=dtype,
        trust_remote_code=HF_TRUST_REMOTE_CODE,
    ).to(device)
    swap_hf_decoders(model, decoder_kind=STUDENT_DECODER_KIND, decoder_kwargs=STUDENT_DECODER_KWARGS)

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        run_dir = args.run_dir or _find_latest_run_dir(args.runs_dir)
        ckpt_path = _find_latest_student_ckpt(run_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    _load_state_dict_strict(model, ckpt.get("student", {}))
    model.eval()

    inputs = tok(args.prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )

    print(tok.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
