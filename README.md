# layerwise-decoder-distillation

Train and distill decoder-only language models using a simple token-shard format and a layer-wise rotation schedule.

## Quickstart (AMD/ROCm)

1. Create a venv and install minimal ROCm PyTorch deps:

```bash
./setup_amd.sh
```

Optional (rare) ROCm architecture override:

```bash
./setup_amd.sh --gfx942
```

2. Download SmolLM2 135M and build ~4 GiB of tokenized shards:

```bash
./.venv/bin/python prepare.py
```

Outputs:

- Model snapshot: `models/smolllm2_135m/`
- Token shards + manifest: `data/fineweb_edu/`

3. Train (next-token LM training):

```bash
torchrun --standalone --nproc_per_node 1 train.py
```

4. Distill (layer-wise rotation schedule):

```bash
torchrun --standalone --nproc_per_node 1 distill.py --out-dir runs
```

## Scripts

### `prepare.py`

- Downloads `HuggingFaceTB/SmolLM2-135M` (snapshot) and tokenizer.
- Streams `HuggingFaceFW/fineweb-edu` (`sample-10BT`) and tokenizes approximately:
  - Train: 4 GiB of UTF-8 bytes from the `text` field
  - Val: 256 MiB of UTF-8 bytes from the `text` field
- Writes NanoGPT-style `.bin` shards (header + `uint16` tokens) and a `manifest.json` with sha256/size/token counts.

CLI:

```bash
./.venv/bin/python prepare.py --out-dir data/my_dataset
./.venv/bin/python prepare.py --smoke-test
```

### `train.py`

Decoder LM training (pretraining-style next-token cross entropy) on `.bin` shards.

### `distill.py`

Decoder LM distillation with a **layer-wise rotation schedule**:

- For each layer index `i`, run `steps_per_layer` optimization steps while only that layer (or a chosen submodule within it) is trainable.
- Save a checkpoint after each layer.
- After visiting all layers, save a final checkpoint.

This is controlled mostly by constants at the top of `distill.py`, with a minimal CLI.

## Data Format (`.bin` shards)

Each shard is:

- Header: 256 `int32` values
  - `header[0] = 20240520` (magic)
  - `header[1] = 1` (version)
  - `header[2] = ntok` (token count)
- Body: `ntok` tokens as `uint16`

Batching is a flat token stream:

- Slice a contiguous window of `B*T + 1` tokens
- `x = buf[:-1].view(B, T)`
- `y = buf[1:].view(B, T)`

Sequences may cross document boundaries.

## Notes

- This repo ignores `models/` and `data/` in git; they are expected to be large.
- AMD/ROCm compatibility is a soft requirement; prefer portable defaults. Per-arch flags are intended for rare cases.
