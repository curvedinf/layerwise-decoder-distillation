#!/usr/bin/env python3
"""
train.py

Train (next-token LM / cross-entropy) starting from a checkpoint, typically produced by `distill.py`.

Notes:
- This script intentionally uses Hugging Face `AutoModelForCausalLM` (no custom GPT-2 model code).
- The student decoder/FFN swap must match what was used when the checkpoint was created.
  Configure that via the constants near the top of this file (edit in-code).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import uuid
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# Constants

SEED = 1337

# HF base checkpoint (used when checkpoint metadata does not override it)
HF_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
HF_MODEL_DIR = "models/smolllm2_135m"
HF_TRUST_REMOTE_CODE = False

# Decoder swap (must match distill checkpoint)
STUDENT_DECODER_KIND = "prototype"  # "prototype" | "custom"
STUDENT_DECODER_KWARGS = dict(hidden_mult=4, activation="relu2", gated=False, zero_init_out=True)

# Data (NanoGPT-style .bin shards)
TRAIN_GLOB = "data/fineweb_edu/train_*.bin"
VAL_GLOB = "data/fineweb_edu/val_*.bin"
DATA_MAGIC = 20240520
DATA_VERSION = 1

# Training
DEVICE_BATCH_SIZE = 16
SEQ_LEN = 512
GRAD_ACCUM_STEPS = 16

NUM_TRAIN_STEPS = 1000
WARMUP_STEPS = 20
WARMDOWN_STEPS = 980

LR = 3e-4
WEIGHT_DECAY = 0.0
BETAS = (0.9, 0.95)
TRY_FUSED_ADAMW = True

# Validation/checkpointing
VAL_TOKENS = 1048576
VAL_EVERY = 0  # 0 disables periodic val; always run at end
SAVE_EVERY = 0  # 0 disables periodic saves; always save at end

USE_COMPILE = True
COMPILE_WITH_DDP = False


# -----------------------------------------------------------------------------
# Minimal decoder swap implementation (duplicated from distill.py)

class DecoderPrototype(torch.nn.Module):
    def __init__(
        self,
        n_embd: int,
        *,
        hidden_mult: int = 4,
        activation: str = "relu2",
        gated: bool = False,
        zero_init_out: bool = True,
    ) -> None:
        super().__init__()
        self.activation = str(activation)
        self.gated = bool(gated)
        hidden = int(hidden_mult) * int(n_embd)
        if self.gated:
            self.fc = torch.nn.Linear(int(n_embd), 2 * hidden, bias=False)
            self.proj = torch.nn.Linear(hidden, int(n_embd), bias=False)
        else:
            self.fc = torch.nn.Linear(int(n_embd), hidden, bias=False)
            self.proj = torch.nn.Linear(hidden, int(n_embd), bias=False)
        if bool(zero_init_out):
            torch.nn.init.zeros_(self.proj.weight)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu2":
            return F.relu(x).square()
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "silu":
            return F.silu(x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        if self.gated:
            a, b = x.chunk(2, dim=-1)
            x = F.silu(a) * b
        else:
            x = self._act(x)
        return self.proj(x)


class CustomDecoderPrototype(torch.nn.Module):
    def __init__(self, n_embd: int, **kwargs) -> None:
        super().__init__()
        # Replace this with your custom block.
        self.impl = DecoderPrototype(n_embd, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)


def build_decoder(kind: str, n_embd: int, kwargs: dict) -> torch.nn.Module:
    kind = str(kind)
    if kind == "prototype":
        return DecoderPrototype(n_embd, **kwargs)
    if kind == "custom":
        return CustomDecoderPrototype(n_embd, **kwargs)
    raise ValueError(f"Unknown decoder kind: {kind}")


class HFDecoderAdapter(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.inner(hidden_states)


def resolve_hf_layers(hf_model: torch.nn.Module) -> list[torch.nn.Module]:
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return list(hf_model.model.layers)
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return list(hf_model.transformer.h)
    if hasattr(hf_model, "gpt_neox") and hasattr(hf_model.gpt_neox, "layers"):
        return list(hf_model.gpt_neox.layers)
    raise RuntimeError("Unsupported HF model architecture: could not locate transformer layers list.")


def resolve_hf_decoder_submodule(layer: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    for name in ("mlp", "ffn", "feed_forward", "feedforward", "moe", "dense_h_to_4h"):
        if hasattr(layer, name):
            mod = getattr(layer, name)
            if isinstance(mod, torch.nn.Module):
                return name, mod
    raise RuntimeError(f"Unsupported layer type: cannot find decoder/FFN submodule on {type(layer).__name__}")


def swap_hf_decoders(hf_model: torch.nn.Module, *, decoder_kind: str, decoder_kwargs: dict) -> None:
    layers = resolve_hf_layers(hf_model)
    hidden_size = getattr(hf_model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(hf_model.config, "n_embd", None)
    if hidden_size is None:
        raise RuntimeError("Could not resolve hidden size from HF config (expected hidden_size or n_embd).")

    for layer in layers:
        attr, _old = resolve_hf_decoder_submodule(layer)
        new = build_decoder(decoder_kind, int(hidden_size), dict(decoder_kwargs))
        setattr(layer, attr, HFDecoderAdapter(new))


# -----------------------------------------------------------------------------
# Data loader (binary shards)

def _peek_data_shard(filename: str) -> int:
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if int(header[0]) != DATA_MAGIC:
        raise RuntimeError(f"Bad magic in {filename}: got {int(header[0])}, expected {DATA_MAGIC}")
    if int(header[1]) != DATA_VERSION:
        raise RuntimeError(f"Bad version in {filename}: got {int(header[1])}, expected {DATA_VERSION}")
    return int(header[2])


def _load_data_shard(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        if int(header[0]) != DATA_MAGIC:
            raise RuntimeError(f"Bad magic in {filename}: got {int(header[0])}, expected {DATA_MAGIC}")
        if int(header[1]) != DATA_VERSION:
            raise RuntimeError(f"Bad version in {filename}: got {int(header[1])}, expected {DATA_VERSION}")
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    if int(tokens.shape[0]) != ntok:
        raise RuntimeError(f"Token count mismatch in {filename}: got {tokens.shape[0]}, expected {ntok}")
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern: str, B: int, T: int, rank: int, world_size: int) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.B = int(B)
        self.T = int(T)

        self.files = sorted(glob.glob(filename_pattern))
        if not self.files:
            raise RuntimeError(f"No files match pattern: {filename_pattern}")

        self.ntok_total = 0
        min_needed = self.world_size * self.B * self.T + 1
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            if shard_ntok < min_needed:
                raise RuntimeError(f"Shard too small: {fname} has {shard_ntok}, need >= {min_needed}")
            self.ntok_total += shard_ntok

        self._shard_idx = 0
        self._tokens: np.ndarray | None = None
        self._pos = 0
        self.reset()

    def reset(self) -> None:
        self._shard_idx = 0
        self._tokens = _load_data_shard(self.files[self._shard_idx])
        self._pos = self.rank * (self.B * self.T)

    def _advance_shard(self) -> None:
        self._shard_idx = (self._shard_idx + 1) % len(self.files)
        self._tokens = _load_data_shard(self.files[self._shard_idx])
        self._pos = self.rank * (self.B * self.T)

    def next_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._tokens is not None
        buf = self._tokens[self._pos : self._pos + self.B * self.T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long, device=device)
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)

        self._pos += self.B * self.T * self.world_size
        if self._pos + (self.B * self.T * self.world_size + 1) > int(self._tokens.shape[0]):
            self._advance_shard()

        return x, y


# -----------------------------------------------------------------------------
# DDP helpers

def ddp_setup() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(device)
    return rank, local_rank, world_size, device


def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_master(rank: int) -> bool:
    return rank == 0


# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ResumeInfo:
    global_step: int = 0
    run_id: str = ""


def _hf_load_path() -> str:
    if os.path.isdir(HF_MODEL_DIR):
        return HF_MODEL_DIR
    return HF_MODEL_ID


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, required=True, help="Checkpoint path to resume from (distill/train).")
    parser.add_argument("--out-dir", type=str, default="runs", help="Output directory for logs/checkpoints.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    rank, local_rank, world_size, device = ddp_setup()
    master = is_master(rank)

    run_id = str(uuid.uuid4())
    out_dir = os.path.join(args.out_dir, run_id)
    if master:
        os.makedirs(out_dir, exist_ok=True)

    hf_path = _hf_load_path()
    if master:
        print(f"[hf] base checkpoint: {hf_path}")

    tok = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=HF_TRUST_REMOTE_CODE)
    if tok.eos_token_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id; expected a standard causal LM tokenizer.")

    student_raw = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=HF_TRUST_REMOTE_CODE,
    ).to(device)
    swap_hf_decoders(student_raw, decoder_kind=STUDENT_DECODER_KIND, decoder_kwargs=STUDENT_DECODER_KWARGS)

    # Resume checkpoint.
    ckpt = torch.load(args.resume, map_location="cpu")
    resume = ResumeInfo(
        global_step=int(ckpt.get("global_step", ckpt.get("step", 0))),
        run_id=str(ckpt.get("run_id", "")),
    )
    state = ckpt.get("student", ckpt.get("model", None))
    if state is None or not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a student/model state dict.")
    missing, unexpected = student_raw.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch: missing={len(missing)} unexpected={len(unexpected)}")

    student: torch.nn.Module = student_raw
    if dist.is_initialized():
        student = DDP(student, device_ids=[local_rank], find_unused_parameters=True)

    if (not args.no_compile) and USE_COMPILE:
        if dist.is_initialized() and not COMPILE_WITH_DDP:
            if master:
                print("[compile] DDP detected; skipping torch.compile (COMPILE_WITH_DDP=False).")
        else:
            student = torch.compile(student)

    train_loader = DistributedDataLoader(TRAIN_GLOB, DEVICE_BATCH_SIZE, SEQ_LEN, rank, world_size)
    val_loader = DistributedDataLoader(VAL_GLOB, DEVICE_BATCH_SIZE, SEQ_LEN, rank, world_size)

    tokens_per_step = DEVICE_BATCH_SIZE * SEQ_LEN * world_size * GRAD_ACCUM_STEPS
    tokens_per_val_step = DEVICE_BATCH_SIZE * SEQ_LEN * world_size
    if VAL_TOKENS % tokens_per_val_step != 0:
        raise RuntimeError(f"VAL_TOKENS must be divisible by B*T*world_size ({tokens_per_val_step}).")
    val_steps = VAL_TOKENS // tokens_per_val_step

    def _make_optimizer() -> torch.optim.Optimizer:
        kwargs = dict(lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
        if TRY_FUSED_ADAMW:
            try:
                return torch.optim.AdamW(student_raw.parameters(), **kwargs, fused=True)
            except TypeError:
                pass
        return torch.optim.AdamW(student_raw.parameters(), **kwargs)

    opt = _make_optimizer()
    if isinstance(ckpt, dict) and "optimizer" in ckpt and isinstance(ckpt["optimizer"], dict):
        try:
            opt.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            if master:
                print(f"[warn] failed to load optimizer state: {exc}")

    def lr_scale(step: int) -> float:
        if step < WARMUP_STEPS:
            return float(step + 1) / float(WARMUP_STEPS)
        plateau_end = max(0, NUM_TRAIN_STEPS - WARMDOWN_STEPS)
        if step < plateau_end:
            return 1.0
        if WARMDOWN_STEPS <= 0:
            return 1.0
        decay_ratio = float(NUM_TRAIN_STEPS - step) / float(WARMDOWN_STEPS)
        return max(0.0, min(1.0, decay_ratio))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_scale)
    # Move scheduler forward to the resumed step (best-effort; doesn't exactly match if you changed constants).
    for _ in range(max(0, resume.global_step)):
        sched.step()

    if master:
        meta = dict(
            run_id=run_id,
            resume_path=args.resume,
            resume_run_id=resume.run_id,
            resume_global_step=resume.global_step,
            hf_model_path=hf_path,
            hf_model_id=HF_MODEL_ID,
            hf_model_dir=HF_MODEL_DIR,
            hf_config=student_raw.config.to_dict() if hasattr(student_raw, "config") else {},
            decoder_kind=STUDENT_DECODER_KIND,
            decoder_kwargs=STUDENT_DECODER_KWARGS,
            train_glob=TRAIN_GLOB,
            val_glob=VAL_GLOB,
            device_batch_size=DEVICE_BATCH_SIZE,
            seq_len=SEQ_LEN,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            num_train_steps=NUM_TRAIN_STEPS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            world_size=world_size,
        )
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        print(f"[run] out_dir={out_dir}")
        print(f"[run] resume_step={resume.global_step} tokens/step={tokens_per_step}")

    global_step = int(resume.global_step)
    t0 = time.time()

    try:
        train_loader.reset()
        for step in range(global_step, NUM_TRAIN_STEPS + 1):
            last_step = step == NUM_TRAIN_STEPS

            if last_step or (VAL_EVERY > 0 and step % VAL_EVERY == 0):
                student.eval()
                val_loader.reset()
                val_loss = 0.0
                with torch.no_grad():
                    for _ in range(val_steps):
                        x_val, y_val = val_loader.next_batch(device)
                        logits = student(input_ids=x_val).logits
                        val_loss += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y_val.view(-1))
                val_loss = val_loss / float(val_steps)
                if dist.is_initialized():
                    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                    val_loss = val_loss / float(world_size)
                if master:
                    print(f"[val] step={step} loss={float(val_loss.item()):.4f}")

            if last_step:
                break

            student.train()
            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0

            for micro in range(GRAD_ACCUM_STEPS):
                x, y = train_loader.next_batch(device)
                logits = student(input_ids=x).logits
                loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
                (loss / GRAD_ACCUM_STEPS).backward()
                loss_acc += float(loss.detach().item())

            opt.step()
            sched.step()

            if master and ((step + 1) % 10 == 0 or (step + 1) == NUM_TRAIN_STEPS):
                elapsed = time.time() - t0
                tok_s = ((step + 1 - global_step) * tokens_per_step) / max(1e-9, elapsed)
                print(f"[train] step={step+1}/{NUM_TRAIN_STEPS} loss={(loss_acc/GRAD_ACCUM_STEPS):.4f} tok/s={tok_s:.0f}")

            if master and (SAVE_EVERY > 0 and (step + 1) % SAVE_EVERY == 0):
                torch.save(
                    dict(
                        run_id=run_id,
                        global_step=step + 1,
                        hf_model_path=hf_path,
                        student=student_raw.state_dict(),
                        optimizer=opt.state_dict(),
                    ),
                    os.path.join(out_dir, f"state_step{step+1:06d}.pt"),
                )

        if master:
            torch.save(
                dict(
                    run_id=run_id,
                    global_step=NUM_TRAIN_STEPS,
                    hf_model_path=hf_path,
                    student=student_raw.state_dict(),
                    optimizer=opt.state_dict(),
                ),
                os.path.join(out_dir, "state_final.pt"),
            )

    finally:
        ddp_cleanup()


if __name__ == "__main__":
    main()

