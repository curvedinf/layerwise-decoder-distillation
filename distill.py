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
import re
import inspect
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
LR = 0.01
WEIGHT_DECAY = 0.0
BETAS = (0.9, 0.95)
TRY_FUSED_ADAMW = True

# Validation/logging/checkpointing
VAL_TOKENS = 1048576
VAL_EVERY_LAYER = True
LOG_EVERY = 10  # steps (within the per-layer inner loop)
SAVE_EVERY = 100  # global steps; set <=0 to disable periodic student_XXXX.pt saves
LATENT_CACHE_DEVICE = "cpu"  # store cached layer outputs on this device


# -----------------------------------------------------------------------------
# DDP helpers (optional; single-process works without torchrun)

def _ddp_setup(
    rank: int,
    local_rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> tuple[int, int, int, torch.device]:
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if world_size > 1 and not dist.is_initialized():
        init_method = f"tcp://{master_addr}:{master_port}"
        dist.init_process_group(backend="nccl", init_method=init_method, rank=rank, world_size=world_size)
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
        old_param = next(_old.parameters(), None)
        if old_param is not None:
            new = new.to(device=old_param.device, dtype=old_param.dtype)
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


@torch.no_grad()
def build_initial_hidden_states(
    hf_model: nn.Module, input_ids: torch.Tensor, position_ids: torch.Tensor
) -> torch.Tensor:
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "embed_tokens"):
        return hf_model.model.embed_tokens(input_ids)
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "wte"):
        hidden = hf_model.transformer.wte(input_ids)
        if hasattr(hf_model.transformer, "wpe"):
            hidden = hidden + hf_model.transformer.wpe(position_ids)
        if hasattr(hf_model.transformer, "drop"):
            hidden = hf_model.transformer.drop(hidden)
        return hidden
    if hasattr(hf_model, "gpt_neox") and hasattr(hf_model.gpt_neox, "embed_in"):
        hidden = hf_model.gpt_neox.embed_in(input_ids)
        if hasattr(hf_model.gpt_neox, "emb_dropout"):
            hidden = hf_model.gpt_neox.emb_dropout(hidden)
        return hidden
    return hf_model.get_input_embeddings()(input_ids)


def run_teacher_layer(
    layer: nn.Module,
    decoder: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    layer_sig_params: set[str],
    rotary_emb: nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    captured_in: dict[str, torch.Tensor] = {}
    captured_out: dict[str, torch.Tensor] = {}

    def _pre_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if "x" not in captured_in:
            captured_in["x"] = inputs[0].detach()

    def _post_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], outputs) -> None:
        if "y" not in captured_out:
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            captured_out["y"] = out.detach()

    h1 = decoder.register_forward_pre_hook(_pre_hook)
    h2 = decoder.register_forward_hook(_post_hook)
    try:
        kwargs = {}
        if "attention_mask" in layer_sig_params:
            kwargs["attention_mask"] = None
        if "position_ids" in layer_sig_params:
            kwargs["position_ids"] = position_ids
        if "position_embeddings" in layer_sig_params and rotary_emb is not None:
            kwargs["position_embeddings"] = rotary_emb(hidden_states, position_ids)
        if "cache_position" in layer_sig_params:
            kwargs["cache_position"] = position_ids[0]
        if "use_cache" in layer_sig_params:
            kwargs["use_cache"] = False
        out = layer(hidden_states, **kwargs)
    finally:
        h1.remove()
        h2.remove()

    if isinstance(out, tuple):
        out = out[0]
    if "x" not in captured_in or "y" not in captured_out:
        raise RuntimeError("Failed to capture decoder input/output for layer forward.")
    return captured_in["x"], captured_out["y"], out


def build_hidden_cache(
    train_loader: "DistributedDataLoader",
    hf_model: nn.Module,
    total_micro_steps: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    cache_device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    train_loader.reset()
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    cache: list[torch.Tensor] = []
    for _ in range(total_micro_steps):
        x, _ = train_loader.next_batch(device)
        hidden = build_initial_hidden_states(hf_model, x, pos_ids)
        cache.append(hidden.detach().to(cache_device))
    return cache, pos_ids


def decoder_loss(student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
    if DECODER_LOSS == "mse":
        return F.mse_loss(student_out.float(), teacher_out.float())
    raise ValueError(f"Unknown DECODER_LOSS: {DECODER_LOSS}")


@dataclass
class CachedCheckpoint:
    name: str
    kind: str
    payload: dict


def move_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_cpu(v) for v in obj)
    return obj


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


def _student_ckpt_name(step: int) -> str:
    return f"student_{int(step):06d}.pt"


def _find_latest_student_ckpt(run_dir: str) -> tuple[str | None, int | None]:
    pattern = re.compile(r"^student_(\d+)\.pt$")
    best_step = None
    best_path = None
    if not os.path.isdir(run_dir):
        return None, None
    for name in os.listdir(run_dir):
        m = pattern.match(name)
        if not m:
            continue
        step = int(m.group(1))
        if best_step is None or step > best_step:
            best_step = step
            best_path = os.path.join(run_dir, name)
    return best_path, best_step


def _resolve_run_dir(out_dir: str, run_dir: str | None, new_run: bool) -> tuple[str, str]:
    if run_dir:
        return run_dir, os.path.basename(os.path.normpath(run_dir))
    os.makedirs(out_dir, exist_ok=True)
    if not new_run:
        candidates = []
        for name in os.listdir(out_dir):
            path = os.path.join(out_dir, name)
            if os.path.isdir(path):
                candidates.append(path)
        if candidates:
            latest = max(candidates, key=lambda p: os.path.getmtime(p))
            return latest, os.path.basename(latest)
    run_id = str(uuid.uuid4())
    return os.path.join(out_dir, run_id), run_id


def _cache_checkpoint(
    memory_cache: list[CachedCheckpoint],
    name: str,
    kind: str,
    ckpt: dict,
) -> None:
    memory_cache.append(CachedCheckpoint(name=name, kind=kind, payload=move_to_cpu(ckpt)))
    print(f"[ckpt] cached in memory: {name} ({kind})")


def _evict_kinds(memory_cache: list[CachedCheckpoint], kinds: set[str]) -> None:
    if not memory_cache:
        return
    memory_cache[:] = [entry for entry in memory_cache if entry.kind not in kinds]


def _consolidate_to_intermediate(
    memory_cache: list[CachedCheckpoint],
    ckpt: dict,
    name: str,
) -> None:
    # Keep only a single intermediate checkpoint in memory.
    _evict_kinds(memory_cache, {"step", "layer", "intermediate"})
    _cache_checkpoint(memory_cache, name, "intermediate", ckpt)


def _evict_intermediate(memory_cache: list[CachedCheckpoint]) -> None:
    _evict_kinds(memory_cache, {"intermediate"})


def _save_checkpoint(
    ckpt: dict,
    name: str,
    out_dir: str,
    memory_cache: list[CachedCheckpoint],
    use_memory_cache: bool,
    force_disk: bool = False,
    kind: str = "generic",
) -> None:
    if (not force_disk) and use_memory_cache:
        _cache_checkpoint(memory_cache, name, kind, ckpt)
        return
    torch.save(ckpt, os.path.join(out_dir, name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="runs", help="Output directory for logs/checkpoints.")
    parser.add_argument("--steps-per-layer", type=int, default=STEPS_PER_LAYER, help="Steps per layer per rotation.")
    parser.add_argument("--batch-size", type=int, default=DEVICE_BATCH_SIZE, help="Per-device batch size.")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Sequence length.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, help="Save student checkpoint every N steps.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory to resume/save into.")
    parser.add_argument("--new-run", action="store_true", help="Force a new run directory under --out-dir.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--memory-cache", action="store_true", help="Keep checkpoints in memory instead of writing to disk.")
    parser.add_argument("--rank", type=int, default=0, help="Process rank (for multi-process runs).")
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=0, help="Local rank.")
    parser.add_argument("--world-size", type=int, default=1, help="World size (number of processes).")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1", help="Master address for DDP.")
    parser.add_argument("--master-port", type=int, default=29500, help="Master port for DDP.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    rank, local_rank, world_size, device = _ddp_setup(
        args.rank,
        args.local_rank,
        args.world_size,
        args.master_addr,
        args.master_port,
    )
    master = _is_master(rank)

    out_dir, run_id = _resolve_run_dir(args.out_dir, args.run_dir, args.new_run)
    if master:
        os.makedirs(out_dir, exist_ok=True)
    memory_cache: list[CachedCheckpoint] = []

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
        # torch.compile is brittle with the decoder-only runner (dynamic layer selection).
        # Skip by default to avoid fake-tensor device errors.
        if isinstance(student, StudentDecoderRunner):
            if master:
                print("[compile] Skipping torch.compile for decoder-only runner.")
        elif dist.is_initialized() and not COMPILE_WITH_DDP:
            if master:
                print("[compile] DDP detected; skipping torch.compile (COMPILE_WITH_DDP=False).")
        else:
            student = torch.compile(student)

    train_loader = DistributedDataLoader(TRAIN_GLOB, args.batch_size, args.seq_len, rank, world_size)
    val_loader = DistributedDataLoader(VAL_GLOB, args.batch_size, args.seq_len, rank, world_size)

    # Validation steps must divide evenly.
    tokens_per_val_step = args.batch_size * args.seq_len * world_size
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

    tokens_per_step = args.batch_size * args.seq_len * world_size * GRAD_ACCUM_STEPS
    teacher_layers = resolve_hf_layers(teacher)
    teacher_decoders = [resolve_hf_decoder_submodule(layer)[1] for layer in teacher_layers]
    layer_sig_params = [set(inspect.signature(layer.forward).parameters.keys()) for layer in teacher_layers]
    rotary_emb = getattr(getattr(teacher, "model", None), "rotary_emb", None)
    num_layers = len(teacher_layers)
    total_steps = num_layers * int(args.steps_per_layer) * ROTATIONS

    steps_per_layer = int(args.steps_per_layer)
    total_micro_steps = steps_per_layer * GRAD_ACCUM_STEPS
    cache_device = torch.device(LATENT_CACHE_DEVICE)

    resume_path, resume_step = _find_latest_student_ckpt(out_dir)
    global_step = 0
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        load_state_dict_strict(student_raw, ckpt.get("student", {}))
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        global_step = int(ckpt.get("global_step", resume_step or 0))
        if master:
            print(f"[resume] {resume_path} (global_step={global_step})")

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
            device_batch_size=int(args.batch_size),
            seq_len=int(args.seq_len),
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
            save_every=int(args.save_every),
            resume_from=resume_path,
        )
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            import json

            json.dump(meta, f, indent=2, sort_keys=True)

        print(f"[run] out_dir={out_dir}")
        print(f"[run] world_size={world_size} total_steps={total_steps} tokens/step={tokens_per_step}")

    steps_per_rotation = num_layers * steps_per_layer
    start_rotation = min(ROTATIONS, global_step // steps_per_rotation) if steps_per_rotation > 0 else 0
    rem = global_step - (start_rotation * steps_per_rotation)
    start_layer = min(num_layers, rem // steps_per_layer) if steps_per_layer > 0 else 0
    start_step_in_layer = rem - (start_layer * steps_per_layer)

    t0 = time.time()

    try:
        for rotation in range(start_rotation, ROTATIONS):
            hidden_cache, pos_ids = build_hidden_cache(
                train_loader,
                teacher,
                total_micro_steps,
                args.batch_size,
                args.seq_len,
                device,
                cache_device,
            )

            # Resume: advance cache to the correct layer/step within this rotation.
            if rotation == start_rotation and (start_layer > 0 or start_step_in_layer > 0):
                for li in range(start_layer):
                    for micro_idx in range(total_micro_steps):
                        hidden_states = hidden_cache[micro_idx].to(device, non_blocking=True)
                        _dec_in, _dec_out, new_hidden = run_teacher_layer(
                            teacher_layers[li],
                            teacher_decoders[li],
                            hidden_states,
                            pos_ids,
                            layer_sig_params[li],
                            rotary_emb,
                        )
                        hidden_cache[micro_idx] = new_hidden.detach().to(cache_device)

                if start_step_in_layer > 0:
                    li = start_layer
                    for micro_idx in range(start_step_in_layer * GRAD_ACCUM_STEPS):
                        hidden_states = hidden_cache[micro_idx].to(device, non_blocking=True)
                        _dec_in, _dec_out, new_hidden = run_teacher_layer(
                            teacher_layers[li],
                            teacher_decoders[li],
                            hidden_states,
                            pos_ids,
                            layer_sig_params[li],
                            rotary_emb,
                        )
                        hidden_cache[micro_idx] = new_hidden.detach().to(cache_device)

            layer_start = start_layer if rotation == start_rotation else 0
            for layer_idx in range(layer_start, num_layers):
                step_start = start_step_in_layer if (rotation == start_rotation and layer_idx == layer_start) else 0
                set_trainable_for_layer_hf(student_raw, layer_idx, TARGET_TRAINABLE)
                student_runner.set_layer(layer_idx)
                student.train()
                teacher.eval()

                if master:
                    print(f"[layer] rotation={rotation} layer={layer_idx} steps={steps_per_layer}")

                for step_in_layer in range(step_start, steps_per_layer):
                    # Gradient accumulation loop.
                    opt.zero_grad(set_to_none=True)
                    loss_acc = 0.0

                    for micro in range(GRAD_ACCUM_STEPS):
                        micro_idx = step_in_layer * GRAD_ACCUM_STEPS + micro
                        hidden_states = hidden_cache[micro_idx].to(device, non_blocking=True)
                        dec_in, t_out, new_hidden = run_teacher_layer(
                            teacher_layers[layer_idx],
                            teacher_decoders[layer_idx],
                            hidden_states,
                            pos_ids,
                            layer_sig_params[layer_idx],
                            rotary_emb,
                        )
                        s_out = student(dec_in)
                        loss = decoder_loss(s_out, t_out)
                        hidden_cache[micro_idx] = new_hidden.detach().to(cache_device)

                        (loss / GRAD_ACCUM_STEPS).backward()
                        loss_acc += float(loss.detach().item())

                    opt.step()
                    global_step += 1

                    if args.save_every and args.save_every > 0 and (global_step % int(args.save_every) == 0):
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
                            _save_checkpoint(
                                ckpt,
                                _student_ckpt_name(global_step),
                                out_dir,
                                memory_cache,
                                args.memory_cache,
                                kind="step",
                            )

                    if master and (step_in_layer % LOG_EVERY == 0 or step_in_layer + 1 == steps_per_layer):
                        elapsed = time.time() - t0
                        tok_s = (global_step * tokens_per_step) / max(1e-9, elapsed)
                        avg_loss = loss_acc / GRAD_ACCUM_STEPS
                        print(
                            f"[train] step={global_step}/{total_steps} "
                            f"rot={rotation} layer={layer_idx} s={step_in_layer+1}/{steps_per_layer} "
                            f"loss={avg_loss:.4f} tok/s={tok_s:.0f}"
                        )

                # Optional validation at layer boundaries.
                if VAL_EVERY_LAYER:
                    t_dec = teacher_decoders[layer_idx]
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
                    name = f"student_layer{layer_idx:03d}.pt"
                    _save_checkpoint(ckpt, name, out_dir, memory_cache, args.memory_cache, kind="layer")
                    if args.memory_cache:
                        intermediate_name = f"student_intermediate_layer{layer_idx:03d}.pt"
                        _consolidate_to_intermediate(memory_cache, ckpt, intermediate_name)

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
            _save_checkpoint(
                ckpt,
                _student_ckpt_name(global_step),
                out_dir,
                memory_cache,
                args.memory_cache,
                force_disk=True,
                kind="final",
            )
            if args.memory_cache:
                _evict_intermediate(memory_cache)

    finally:
        _ddp_cleanup()


if __name__ == "__main__":
    main()
