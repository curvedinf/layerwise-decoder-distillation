#!/usr/bin/env python3
"""
distill.py

Decoder LM distillation with a layer-wise rotation schedule.

How To Specify The Student Decoder

This project requires teacher and student attention to be identical. The intended axis of change
is the per-layer decoder/FFN module (a `(B,T,C)->(B,T,C)` block).

Teacher Model Requirement

The teacher is always a Hugging Face `AutoModelForCausalLM` loaded from `HF_MODEL_DIR` (or
`HF_MODEL_ID` if the directory does not exist). The student is initialized from the same HF
checkpoint to guarantee identical attention weights and tokenization compatibility, and then the
per-layer decoder/FFN submodule can be swapped to a custom implementation.

Edit the constants near the top of this file:

- `STUDENT_DECODER_KIND`
  - `"prototype"`: use `DecoderPrototype` (simple MLP-like reference).
  - `"custom"`: use `CustomDecoderPrototype` (you edit/replace its implementation).
- `STUDENT_DECODER_KWARGS`
  - Passed as `**kwargs` to the decoder constructor.
  - Keep the signature stable across runs for reproducibility.

If you want to plug in your own block:

1. Replace the body of `CustomDecoderPrototype` (search for the class below).
2. Set `STUDENT_DECODER_KIND = "custom"`.
3. Set `STUDENT_DECODER_KWARGS = {...}` to whatever your block needs.

Configuration philosophy:
- Most options are constants in this file (edit at the top).
- CLI args are intentionally minimal (paths + a small number of toggles).
- No environment variables are required for configuration.
  - If launched under torchrun, standard DDP env vars may be present, but are not treated as config.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
import uuid
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# Constants (project defaults)

SEED = 1337

# Data
TRAIN_GLOB = "data/fineweb_edu/train_*.bin"
VAL_GLOB = "data/fineweb_edu/val_*.bin"
DATA_MAGIC = 20240520
DATA_VERSION = 1

# HF teacher/student base checkpoint
HF_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
HF_MODEL_DIR = "models/smolllm2_135m"
HF_TRUST_REMOTE_CODE = False

# Model
# For distillation, teacher and student are Hugging Face decoder LMs.
# The student is initialized from the same checkpoint, then its decoder/FFN can be swapped.

# Layer specs (architecture choices)
#
# These constants define what *type* of layer implementations the teacher and student are built from.
# They are intended to be edited in-code (not via CLI) so runs are maximally reproducible.
#
# Requirement: attention must be identical between teacher and student.
# Only the decoder/FFN submodule is intended to differ (when desired).
#
# Attention options:
# - "sdpa": PyTorch scaled_dot_product_attention with rotary embeddings.
#
# Decoder/FFN options:
# - "prototype": A generic, easy-to-swap `(B,T,C)->(B,T,C)` module defined in this file.
# - "custom": Use `CustomDecoderPrototype` (edit/replace its implementation to your needs).
#
# Decoder/FFN contract:
# - Must be an `nn.Module` with `forward(x: Tensor[B,T,C]) -> Tensor[B,T,C]`.
# - Must not change sequence length or embedding dim.
#
TEACHER_ATTN_TYPE = "sdpa"
TEACHER_ATTN_QK_NORM = True
TEACHER_POSENC = "rotary"  # currently only "rotary"

ZERO_INIT_OUT_PROJ = True
LOGIT_CAP = 30.0  # set <=0 to disable

STUDENT_DECODER_KIND = "prototype"
#
# By default, the student decoder is made ~50% smaller than the teacher's decoder by inferring
# the teacher's per-layer intermediate width and scaling it.
STUDENT_DECODER_WIDTH_SCALE = 0.5  # set to 1.0 for parity; set to None to disable inference
STUDENT_DECODER_KWARGS = dict(
    # If `hidden_dim` is not provided, and `STUDENT_DECODER_WIDTH_SCALE` is not None,
    # `swap_hf_decoders()` will set `hidden_dim = round_down_8(int(teacher_hidden_dim * scale))`.
    hidden_dim=None,
    hidden_mult=4,
    activation="relu2",
    gated=False,
    zero_init_out=ZERO_INIT_OUT_PROJ,
)

# Training
DEVICE_BATCH_SIZE = 16
SEQ_LEN = 512
GRAD_ACCUM_STEPS = 16

ROTATIONS = 1
STEPS_PER_LAYER = 100
TARGET_TRAINABLE = "decoder"  # "decoder" | "attn" | "block"

USE_COMPILE = True
COMPILE_WITH_DDP = False

# Distillation loss (decoder output matching)
DECODER_LOSS = "mse"  # currently only "mse"

# Optimizer
LR = 3e-4
WEIGHT_DECAY = 0.0
BETAS = (0.9, 0.95)
TRY_FUSED_ADAMW = True

# Validation/logging/checkpointing
VAL_TOKENS = 1048576
VAL_EVERY_LAYER = True
LOG_EVERY = 10  # steps (within the per-layer inner loop)


# -----------------------------------------------------------------------------
# DDP helpers (optional; single-process works without torchrun)

def _ddp_setup() -> tuple[int, int, int, torch.device]:
    # If torchrun is used, these will exist. Otherwise default to single process.
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(device)

    return rank, local_rank, world_size, device


def _ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_master(rank: int) -> bool:
    return rank == 0


# -----------------------------------------------------------------------------
# Data loading (binary shard format)

def _peek_data_shard(filename: str) -> int:
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if int(header[0]) != DATA_MAGIC:
        raise RuntimeError(f"Bad magic in {filename}: got {int(header[0])}, expected {DATA_MAGIC}")
    if int(header[1]) != DATA_VERSION:
        raise RuntimeError(f"Bad version in {filename}: got {int(header[1])}, expected {DATA_VERSION}")
    ntok = int(header[2])
    return ntok


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
        self.rank = rank
        self.world_size = world_size
        self.B = int(B)
        self.T = int(T)

        self.files = sorted(glob.glob(filename_pattern))
        if not self.files:
            raise RuntimeError(f"No files match pattern: {filename_pattern}")

        self.ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            min_needed = self.world_size * self.B * self.T + 1
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
        B, T = self.B, self.T
        buf = self._tokens[self._pos : self._pos + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long, device=device)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self._pos += B * T * self.world_size
        if self._pos + (B * T * self.world_size + 1) > int(self._tokens.shape[0]):
            self._advance_shard()

        return x, y


# -----------------------------------------------------------------------------
# Decoder/FFN implementations (used for swapping into HF model layers)


class DecoderPrototype(nn.Module):
    """
    Generic `(B,T,C)->(B,T,C)` decoder/FFN prototype.

    If you have your own decoder module, prefer implementing it as `CustomDecoderPrototype`
    and set `STUDENT_DECODER_KIND="custom"` at the top of this file.
    """

    def __init__(
        self,
        n_embd: int,
        *,
        hidden_dim: int | None = None,
        hidden_mult: int = 4,
        activation: str = "relu2",
        gated: bool = False,
        zero_init_out: bool = True,
    ) -> None:
        super().__init__()
        self.activation = str(activation)
        self.gated = bool(gated)
        if hidden_dim is None:
            hidden = int(hidden_mult) * int(n_embd)
        else:
            hidden = int(hidden_dim)
        if self.gated:
            self.fc = nn.Linear(int(n_embd), 2 * hidden, bias=False)
            self.proj = nn.Linear(hidden, int(n_embd), bias=False)
        else:
            self.fc = nn.Linear(int(n_embd), hidden, bias=False)
            self.proj = nn.Linear(hidden, int(n_embd), bias=False)
        if bool(zero_init_out):
            nn.init.zeros_(self.proj.weight)

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


class CustomDecoderPrototype(nn.Module):
    """
    Edit this module to your needs, then set:

    - `STUDENT_DECODER_KIND = "custom"`
    - `STUDENT_DECODER_KWARGS = {...}` (any kwargs you need)
    """

    def __init__(self, n_embd: int, **kwargs) -> None:
        super().__init__()
        # Default implementation is the same as the built-in prototype.
        self.impl = DecoderPrototype(n_embd, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)


def build_decoder(kind: str, n_embd: int, kwargs: dict) -> nn.Module:
    kind = str(kind)
    if kind == "prototype":
        return DecoderPrototype(n_embd, **kwargs)
    if kind == "custom":
        return CustomDecoderPrototype(n_embd, **kwargs)
    raise ValueError(f"Unknown decoder kind: {kind}")


# -----------------------------------------------------------------------------
# Distillation + layer rotation

def _hf_load_path() -> str:
    if os.path.isdir(HF_MODEL_DIR):
        return HF_MODEL_DIR
    return HF_MODEL_ID


def resolve_hf_layers(hf_model: nn.Module) -> list[nn.Module]:
    """
    Return the list of transformer layers for common decoder-only architectures.
    Extend this function as needed when adding new model families.
    """
    # LLaMA/Mistral/Qwen2-style: model.model.layers
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return list(hf_model.model.layers)
    # GPT-2-style: model.transformer.h
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return list(hf_model.transformer.h)
    # GPT-NeoX-style: model.gpt_neox.layers
    if hasattr(hf_model, "gpt_neox") and hasattr(hf_model.gpt_neox, "layers"):
        return list(hf_model.gpt_neox.layers)
    raise RuntimeError("Unsupported HF model architecture: could not locate transformer layers list.")


def resolve_hf_decoder_submodule(layer: nn.Module) -> tuple[str, nn.Module]:
    """
    Return (attr_name, module) for the layer's FFN/decoder submodule.
    """
    for name in ("mlp", "ffn", "feed_forward", "feedforward", "moe", "dense_h_to_4h"):
        if hasattr(layer, name):
            mod = getattr(layer, name)
            if isinstance(mod, nn.Module):
                return name, mod
    raise RuntimeError(f"Unsupported layer type: cannot find decoder/FFN submodule on {type(layer).__name__}")


class HFDecoderAdapter(nn.Module):
    """
    Adapter that wraps our generic `(B,T,C)->(B,T,C)` decoder into a HF-compatible
    `(hidden_states, *args, **kwargs)->hidden_states` signature.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.inner(hidden_states)


def round_down_8(x: int) -> int:
    return max(8, (int(x) // 8) * 8)


def infer_decoder_intermediate_dim(decoder: nn.Module) -> int:
    # Heuristic: maximum out_features among Linear submodules.
    # Works for common decoders (LLaMA-style MLP, GPT-2 MLP, NeoX MLP).
    best = None
    for m in decoder.modules():
        if isinstance(m, nn.Linear):
            of = int(m.out_features)
            if best is None or of > best:
                best = of
    if best is None:
        raise RuntimeError(f"Could not infer intermediate dim from decoder module: {type(decoder).__name__}")
    return int(best)


def swap_hf_decoders(hf_model: nn.Module, *, decoder_kind: str, decoder_kwargs: dict) -> None:
    layers = resolve_hf_layers(hf_model)
    hidden_size = getattr(hf_model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(hf_model.config, "n_embd", None)
    if hidden_size is None:
        raise RuntimeError("Could not resolve hidden size from HF config (expected hidden_size or n_embd).")

    for layer in layers:
        attr, _old = resolve_hf_decoder_submodule(layer)
        new_kwargs = dict(decoder_kwargs)
        if decoder_kind == "prototype":
            scale = STUDENT_DECODER_WIDTH_SCALE
            if new_kwargs.get("hidden_dim") in (None, 0) and scale is not None:
                old_inter = infer_decoder_intermediate_dim(_old)
                new_kwargs["hidden_dim"] = round_down_8(int(old_inter * float(scale)))
        new = build_decoder(decoder_kind, int(hidden_size), new_kwargs)
        setattr(layer, attr, HFDecoderAdapter(new))


class StudentDecoderRunner(nn.Module):
    """
    Lightweight wrapper that exposes just the target layer's decoder forward.
    Used so DDP can synchronize gradients while avoiding full-model forward passes.
    """

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model
        self.layers = resolve_hf_layers(hf_model)
        self.decoder_attrs = [resolve_hf_decoder_submodule(layer)[0] for layer in self.layers]
        self.layer_idx = 0

    def set_layer(self, layer_idx: int) -> None:
        self.layer_idx = int(layer_idx)

    def forward(self, decoder_input: torch.Tensor) -> torch.Tensor:
        attr = self.decoder_attrs[self.layer_idx]
        decoder = getattr(self.layers[self.layer_idx], attr)
        out = decoder(decoder_input)
        if isinstance(out, tuple):
            out = out[0]
        return out


def set_trainable_for_layer_hf(hf_model: nn.Module, layer_idx: int, target: str) -> None:
    for p in hf_model.parameters():
        p.requires_grad = False

    layers = resolve_hf_layers(hf_model)
    if not (0 <= layer_idx < len(layers)):
        raise IndexError(f"layer_idx out of range: {layer_idx} not in [0,{len(layers)})")
    layer = layers[layer_idx]

    if target == "block":
        params = layer.parameters()
    elif target in ("decoder", "mlp"):
        attr, dec = resolve_hf_decoder_submodule(layer)
        _ = attr
        params = dec.parameters()
    elif target == "attn":
        if hasattr(layer, "self_attn"):
            params = layer.self_attn.parameters()
        elif hasattr(layer, "attn"):
            params = layer.attn.parameters()
        else:
            raise RuntimeError(f"Could not find attention module on layer type: {type(layer).__name__}")
    else:
        raise ValueError(f"Unknown TARGET_TRAINABLE: {target}")

    for p in params:
        p.requires_grad = True

class _StopForward(Exception):
    pass


@torch.no_grad()
def capture_decoder_input(hf_model: nn.Module, decoder: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Run the HF model forward until the target decoder is about to execute.
    Cache and return the decoder input (latent) without running the decoder or later layers.
    """
    captured: dict[str, torch.Tensor] = {}

    def _pre_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if "x" not in captured:
            captured["x"] = inputs[0].detach()
        raise _StopForward()

    handle = decoder.register_forward_pre_hook(_pre_hook)
    try:
        hf_model(input_ids=input_ids, use_cache=False)
    except _StopForward:
        pass
    finally:
        handle.remove()

    if "x" not in captured:
        raise RuntimeError("Failed to capture decoder input: decoder was not invoked.")
    return captured["x"]


def decoder_loss(student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
    if DECODER_LOSS == "mse":
        return F.mse_loss(student_out.float(), teacher_out.float())
    raise ValueError(f"Unknown DECODER_LOSS: {DECODER_LOSS}")


def load_state_dict_strict(model: nn.Module, state_dict: dict) -> None:
    # Tolerate common wrappers (DDP/module prefixes).
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
        # Keep it actionable; don't spam huge lists.
        msg = []
        if missing:
            msg.append(f"missing={len(missing)}")
        if unexpected:
            msg.append(f"unexpected={len(unexpected)}")
        raise RuntimeError("Checkpoint/model mismatch (" + ", ".join(msg) + ")")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="runs", help="Output directory for logs/checkpoints.")
    parser.add_argument("--steps-per-layer", type=int, default=STEPS_PER_LAYER, help="Steps per layer per rotation.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    rank, local_rank, world_size, device = _ddp_setup()
    master = _is_master(rank)

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

    teacher = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=HF_TRUST_REMOTE_CODE,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student is initialized from the same checkpoint to keep attention identical.
    student_raw = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=HF_TRUST_REMOTE_CODE,
    ).to(device)
    swap_hf_decoders(student_raw, decoder_kind=STUDENT_DECODER_KIND, decoder_kwargs=STUDENT_DECODER_KWARGS)

    student_runner = StudentDecoderRunner(student_raw)
    student: nn.Module = student_runner
    if dist.is_initialized():
        # Layer-wise freezing can produce unused params in the backward graph.
        student = DDP(student, device_ids=[local_rank], find_unused_parameters=True)

    if (not args.no_compile) and USE_COMPILE:
        if dist.is_initialized() and not COMPILE_WITH_DDP:
            if master:
                print("[compile] DDP detected; skipping torch.compile (COMPILE_WITH_DDP=False).")
        else:
            student = torch.compile(student)

    train_loader = DistributedDataLoader(TRAIN_GLOB, DEVICE_BATCH_SIZE, SEQ_LEN, rank, world_size)
    val_loader = DistributedDataLoader(VAL_GLOB, DEVICE_BATCH_SIZE, SEQ_LEN, rank, world_size)

    # Validation steps must divide evenly.
    tokens_per_val_step = DEVICE_BATCH_SIZE * SEQ_LEN * world_size
    if VAL_TOKENS % tokens_per_val_step != 0:
        raise RuntimeError(
            f"VAL_TOKENS ({VAL_TOKENS}) must be divisible by B*T*world_size ({tokens_per_val_step})."
        )
    val_steps = VAL_TOKENS // tokens_per_val_step

    # Optimizer (single optimizer; params are toggled trainable per-layer).
    # This keeps the code simple; per-parameter-group optimizers can be added later.
    def _make_optimizer() -> torch.optim.Optimizer:
        kwargs = dict(lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
        if TRY_FUSED_ADAMW:
            try:
                return torch.optim.AdamW(student_raw.parameters(), **kwargs, fused=True)
            except TypeError:
                pass
        return torch.optim.AdamW(student_raw.parameters(), **kwargs)

    opt = _make_optimizer()

    tokens_per_step = DEVICE_BATCH_SIZE * SEQ_LEN * world_size * GRAD_ACCUM_STEPS
    teacher_layers = resolve_hf_layers(teacher)
    num_layers = len(teacher_layers)
    total_steps = num_layers * int(args.steps_per_layer) * ROTATIONS

    if master:
        meta = dict(
            run_id=run_id,
            hf_model_path=hf_path,
            hf_model_id=HF_MODEL_ID,
            hf_model_dir=HF_MODEL_DIR,
            hf_config=teacher.config.to_dict() if hasattr(teacher, "config") else {},
            tokenizer={
                "name_or_path": getattr(tok, "name_or_path", hf_path),
                "vocab_size": int(getattr(tok, "vocab_size", -1)),
                "eos_token_id": tok.eos_token_id,
            },
            train_glob=TRAIN_GLOB,
            val_glob=VAL_GLOB,
            device_batch_size=DEVICE_BATCH_SIZE,
            seq_len=SEQ_LEN,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            rotations=ROTATIONS,
            steps_per_layer=int(args.steps_per_layer),
            target_trainable=TARGET_TRAINABLE,
            student_decoder_kind=STUDENT_DECODER_KIND,
            student_decoder_kwargs=STUDENT_DECODER_KWARGS,
            student_decoder_width_scale=STUDENT_DECODER_WIDTH_SCALE,
            decoder_loss=DECODER_LOSS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            world_size=world_size,
        )
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            import json

            json.dump(meta, f, indent=2, sort_keys=True)

        print(f"[run] out_dir={out_dir}")
        print(f"[run] world_size={world_size} total_steps={total_steps} tokens/step={tokens_per_step}")

    global_step = 0
    t0 = time.time()

    try:
        for rotation in range(ROTATIONS):
            for layer_idx in range(num_layers):
                set_trainable_for_layer_hf(student_raw, layer_idx, TARGET_TRAINABLE)
                student_runner.set_layer(layer_idx)
                student.train()
                teacher.eval()
                train_loader.reset()

                t_attr, t_dec = resolve_hf_decoder_submodule(teacher_layers[layer_idx])
                _ = t_attr

                if master:
                    print(f"[layer] rotation={rotation} layer={layer_idx} steps={int(args.steps_per_layer)}")

                for step_in_layer in range(int(args.steps_per_layer)):
                    # Gradient accumulation loop.
                    opt.zero_grad(set_to_none=True)
                    loss_acc = 0.0

                    for micro in range(GRAD_ACCUM_STEPS):
                        x, _ = train_loader.next_batch(device)
                        decoder_input = capture_decoder_input(teacher, t_dec, x)
                        with torch.no_grad():
                            t_out = t_dec(decoder_input)
                            if isinstance(t_out, tuple):
                                t_out = t_out[0]
                        s_out = student(decoder_input)
                        loss = decoder_loss(s_out, t_out)

                        (loss / GRAD_ACCUM_STEPS).backward()
                        loss_acc += float(loss.detach().item())

                    opt.step()
                    global_step += 1

                    if master and (step_in_layer % LOG_EVERY == 0 or step_in_layer + 1 == int(args.steps_per_layer)):
                        elapsed = time.time() - t0
                        tok_s = (global_step * tokens_per_step) / max(1e-9, elapsed)
                        avg_loss = loss_acc / GRAD_ACCUM_STEPS
                        print(
                            f"[train] step={global_step}/{total_steps} "
                            f"rot={rotation} layer={layer_idx} s={step_in_layer+1}/{int(args.steps_per_layer)} "
                            f"loss={avg_loss:.4f} tok/s={tok_s:.0f}"
                        )

                # Optional validation at layer boundaries.
                if VAL_EVERY_LAYER:
                    student.eval()
                    teacher.eval()
                    val_loader.reset()
                    val_loss = 0.0
                    with torch.no_grad():
                        for _ in range(val_steps):
                            x_val, _ = val_loader.next_batch(device)
                            dec_in = capture_decoder_input(teacher, t_dec, x_val)
                            t_out = t_dec(dec_in)
                            if isinstance(t_out, tuple):
                                t_out = t_out[0]
                            s_out = student(dec_in)
                            val_loss += decoder_loss(s_out, t_out)
                    val_loss = val_loss / float(val_steps)
                    if dist.is_initialized():
                        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                        val_loss = val_loss / float(world_size)
                    if master:
                        print(f"[val] rotation={rotation} layer={layer_idx} decoder_loss={float(val_loss.item()):.4f}")

                # Save intermediate checkpoint (per layer).
                if master:
                    ckpt = dict(
                        run_id=run_id,
                        hf_model_path=hf_path,
                        hf_config=teacher.config.to_dict() if hasattr(teacher, "config") else {},
                        global_step=global_step,
                        rotation=rotation,
                        layer_idx=layer_idx,
                        target_trainable=TARGET_TRAINABLE,
                        student=student_raw.state_dict(),
                        optimizer=opt.state_dict(),
                    )
                    torch.save(ckpt, os.path.join(out_dir, f"student_layer{layer_idx:03d}.pt"))

        if master:
            ckpt = dict(
                run_id=run_id,
                hf_model_path=hf_path,
                hf_config=teacher.config.to_dict() if hasattr(teacher, "config") else {},
                global_step=global_step,
                rotation=ROTATIONS,
                layer_idx=None,
                target_trainable=TARGET_TRAINABLE,
                student=student_raw.state_dict(),
                optimizer=opt.state_dict(),
            )
            torch.save(ckpt, os.path.join(out_dir, "student_final.pt"))

    finally:
        _ddp_cleanup()


if __name__ == "__main__":
    main()
