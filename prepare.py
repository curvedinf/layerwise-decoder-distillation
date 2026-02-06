#!/usr/bin/env python3
"""
prepare_smolllm2_135m_4gb.py

Downloads SmolLM2 135M from Hugging Face, then streams and tokenizes ~4 GiB of
FineWeb-Edu into NanoGPT-style .bin shards (uint16 tokens + int32 header).

Design constraints:
- Most configuration is constants at the top.
- Minimal CLI (output dir + smoke test).
- No environment variables required.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
from tqdm import tqdm

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Constants

# Model
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
# Keep this intentionally short. If you want multiple models cached, edit this constant.
MODEL_LOCAL_DIR = "models/smolllm2_135m"

# Dataset (streaming)
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_SPLIT = "train"
TEXT_FIELD = "text"

# Amount of raw text to collect (in UTF-8 bytes of TEXT_FIELD)
TARGET_TRAIN_BYTES = 4 * 1024**3  # 4 GiB
TARGET_VAL_BYTES = 256 * 1024**2  # 256 MiB (separate from train)

# Token stream conventions
APPEND_EOS = True

# Output (NanoGPT .bin shards)
# Keep this intentionally short. If you want multiple prepared variants,
# set `--out-dir ...` at runtime or edit this constant.
OUT_DIR = "data/fineweb_edu"
TRAIN_PREFIX = "train"
VAL_PREFIX = "val"

DATA_MAGIC = 20240520
DATA_VERSION = 1
HEADER_INTS = 256

# Sharding
TOKENS_PER_SHARD = 64 * 1024 * 1024  # 64M tokens ~= 128 MiB of uint16 payload

# Tokenization batching
# Fast tokenizers run `encode_batch` in Rust and can parallelize internally.
TOKENIZE_BATCH_DOCS = 1024

# Shuffle (streaming)
SHUFFLE_BUFFER = 50_000
TRAIN_SHUFFLE_SEED = 1337
VAL_SHUFFLE_SEED = 1338


# -----------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_bin(path: str, tokens: np.ndarray) -> dict:
    if tokens.dtype != np.uint16:
        raise ValueError(f"tokens must be uint16, got {tokens.dtype}")
    if tokens.ndim != 1:
        raise ValueError(f"tokens must be 1D, got shape {tokens.shape}")

    header = np.zeros((HEADER_INTS,), dtype=np.int32)
    header[0] = DATA_MAGIC
    header[1] = DATA_VERSION
    header[2] = int(tokens.shape[0])

    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header.tobytes(order="C"))
        f.write(tokens.tobytes(order="C"))
    os.replace(tmp, path)

    # Verify size and compute sha256 for accounting.
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    size_bytes = os.path.getsize(path)
    expected_size = HEADER_INTS * 4 + int(tokens.shape[0]) * 2
    if size_bytes != expected_size:
        raise RuntimeError(f"Size mismatch for {path}: got {size_bytes}, expected {expected_size}")

    return {
        "path": path,
        "sha256": h.hexdigest(),
        "ntok": int(tokens.shape[0]),
        "bytes": int(size_bytes),
    }


def _tokenize_iterable(
    ds_iter,
    tokenizer,
    *,
    target_bytes: int,
    out_dir: str,
    prefix: str,
    record_hashes: set[int] | None,
    exclude_hashes: set[int] | None,
    pbar_desc: str,
) -> dict:
    eos_id = tokenizer.eos_token_id
    if APPEND_EOS and eos_id is None:
        raise RuntimeError("APPEND_EOS=True but tokenizer has no eos_token_id.")

    total_bytes = 0
    total_docs = 0
    total_tokens = 0
    shard_idx = 0
    shard_tokens: list[int] = []
    shards: list[dict] = []

    # In streaming mode, progress by bytes.
    pbar = tqdm(total=target_bytes, desc=pbar_desc, unit="B", unit_scale=True)

    def flush() -> None:
        nonlocal shard_idx, shard_tokens, total_tokens, shards
        if not shard_tokens:
            return
        arr = np.asarray(shard_tokens, dtype=np.uint16)
        path = os.path.join(out_dir, f"{prefix}_{shard_idx:06d}.bin")
        meta = _write_bin(path, arr)
        shards.append(meta)
        total_tokens += int(arr.shape[0])
        shard_tokens = []
        shard_idx += 1

    batch_texts: list[str] = []
    batch_raw_lens: list[int] = []
    batch_keys: list[int | None] = []

    def flush_batch() -> None:
        nonlocal batch_texts, batch_raw_lens, batch_keys, shard_tokens, total_docs, total_bytes
        if not batch_texts:
            return
        enc = tokenizer(
            batch_texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = enc["input_ids"]
        assert len(input_ids) == len(batch_texts)
        for ids, raw_len, key in zip(input_ids, batch_raw_lens, batch_keys):
            if APPEND_EOS:
                ids = list(ids) + [int(eos_id)]
            # Enforce uint16 compatibility.
            if ids and max(ids) > 65535:
                raise RuntimeError(
                    f"Tokenizer produced id > 65535 (max={max(ids)}). "
                    "This output format requires vocab <= 65535 or a remapping scheme."
                )
            shard_tokens.extend(ids)
            total_docs += 1
            total_bytes += raw_len
            pbar.update(min(raw_len, max(0, target_bytes - pbar.n)))
            if key is not None and record_hashes is not None:
                record_hashes.add(key)
            if len(shard_tokens) >= TOKENS_PER_SHARD:
                flush()

        batch_texts = []
        batch_raw_lens = []
        batch_keys = []

    for ex in ds_iter:
        text = ex.get(TEXT_FIELD, None)
        if text is None or not isinstance(text, str) or not text:
            continue

        raw = text.encode("utf-8", errors="ignore")
        raw_len = len(raw)
        if raw_len == 0:
            continue

        # Enforce shard-disjointness between val and train (best-effort).
        key = None
        if exclude_hashes is not None or record_hashes is not None:
            h = hashlib.blake2b(raw, digest_size=8).digest()
            key = int.from_bytes(h, "little", signed=False)
            if exclude_hashes is not None and key in exclude_hashes:
                continue

        batch_texts.append(text)
        batch_raw_lens.append(raw_len)
        batch_keys.append(key)

        if len(batch_texts) >= TOKENIZE_BATCH_DOCS:
            flush_batch()
            if total_bytes >= target_bytes:
                break

        if total_bytes >= target_bytes:
            break

    flush_batch()
    if total_bytes < target_bytes:
        # allow shortfall checks to trigger in main
        pass

    flush()
    pbar.close()

    return {
        "prefix": prefix,
        "target_bytes": int(target_bytes),
        "total_bytes": int(total_bytes),
        "total_docs": int(total_docs),
        "total_tokens": int(total_tokens),
        "num_shards": int(len(shards)),
        "shards": shards,
    }


@dataclass(frozen=True)
class PrepareConfig:
    model_id: str = MODEL_ID
    dataset_name: str = DATASET_NAME
    dataset_config: str = DATASET_CONFIG
    dataset_split: str = DATASET_SPLIT
    text_field: str = TEXT_FIELD
    target_train_bytes: int = TARGET_TRAIN_BYTES
    target_val_bytes: int = TARGET_VAL_BYTES
    append_eos: bool = APPEND_EOS
    tokens_per_shard: int = TOKENS_PER_SHARD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument("--smoke-test", action="store_true", help="Tokenize a tiny amount for validation.")
    args = parser.parse_args()

    cfg = PrepareConfig()
    out_dir = args.out_dir
    _ensure_dir(out_dir)
    _ensure_dir(MODEL_LOCAL_DIR)

    train_bytes = 32 * 1024**2 if args.smoke_test else cfg.target_train_bytes
    val_bytes = 8 * 1024**2 if args.smoke_test else cfg.target_val_bytes

    t0 = time.time()

    # 1) Download model snapshot (weights + tokenizer files).
    snapshot_download(
        repo_id=cfg.model_id,
        local_dir=MODEL_LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    # 2) Load tokenizer from local dir to guarantee we are using downloaded assets.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR, use_fast=True)
    # This project packs a flat token stream; we do not want warnings for long documents.
    tokenizer.model_max_length = int(1e30)
    vocab_size = int(tokenizer.vocab_size)
    if vocab_size > 65535:
        raise RuntimeError(f"Tokenizer vocab_size={vocab_size} > 65535; cannot emit uint16 tokens.")

    # 3) Stream dataset, shuffled, and build val + train with explicit disjointness.
    base = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split, streaming=True)

    # Separate iterables with different shuffles to minimize overlap even before hashing.
    val_iter = base.shuffle(seed=VAL_SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)
    train_iter = base.shuffle(seed=TRAIN_SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)

    seen_val_hashes: set[int] = set()
    val_meta = _tokenize_iterable(
        val_iter,
        tokenizer,
        target_bytes=val_bytes,
        out_dir=out_dir,
        prefix=VAL_PREFIX,
        record_hashes=seen_val_hashes,
        exclude_hashes=None,
        pbar_desc="val bytes",
    )

    # For train, avoid val hashes to ensure shard-disjointness.
    train_meta = _tokenize_iterable(
        train_iter,
        tokenizer,
        target_bytes=train_bytes,
        out_dir=out_dir,
        prefix=TRAIN_PREFIX,
        record_hashes=None,
        exclude_hashes=seen_val_hashes,
        pbar_desc="train bytes",
    )

    # 4) Accounting checks.
    def _glob_count(prefix: str) -> int:
        return len(sorted(glob.glob(os.path.join(out_dir, f"{prefix}_*.bin"))))

    train_shards_on_disk = _glob_count(TRAIN_PREFIX)
    val_shards_on_disk = _glob_count(VAL_PREFIX)
    if train_shards_on_disk != train_meta["num_shards"]:
        raise RuntimeError("Train shard count mismatch between manifest and disk.")
    if val_shards_on_disk != val_meta["num_shards"]:
        raise RuntimeError("Val shard count mismatch between manifest and disk.")

    if val_meta["total_bytes"] < val_bytes:
        raise RuntimeError(f"Val bytes short: got {val_meta['total_bytes']}, expected >= {val_bytes}")
    if train_meta["total_bytes"] < train_bytes:
        raise RuntimeError(f"Train bytes short: got {train_meta['total_bytes']}, expected >= {train_bytes}")

    manifest = {
        "config": asdict(cfg),
        "smoke_test": bool(args.smoke_test),
        "model_local_dir": MODEL_LOCAL_DIR,
        "tokenizer": {
            "vocab_size": int(vocab_size),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
        },
        "val": val_meta,
        "train": train_meta,
        "elapsed_s": float(time.time() - t0),
    }

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"[ok] wrote manifest: {os.path.join(out_dir, 'manifest.json')}")
    print(f"[ok] val: bytes={val_meta['total_bytes']} docs={val_meta['total_docs']} tok={val_meta['total_tokens']} shards={val_meta['num_shards']}")
    print(
        f"[ok] train: bytes={train_meta['total_bytes']} docs={train_meta['total_docs']} tok={train_meta['total_tokens']} shards={train_meta['num_shards']}"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    # Work around rare interpreter-shutdown crashes in the streaming stack by exiting without
    # finalization after all outputs are durably written.
    os._exit(0)


if __name__ == "__main__":
    main()
