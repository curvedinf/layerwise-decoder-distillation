# layerwise-decoder-distillation

Layer-wise decoder distillation for Hugging Face decoder-only language models. Includes a token-shard data pipeline, teacher/student decoder swapping, and a rotation schedule with per-layer checkpoints.

Distillation works by capturing the teacher’s decoder-input latent at a chosen layer, running both teacher and student decoders on that same latent, and updating only the student decoder to match the teacher output while the rest of the model stays aligned to the teacher; this repeats layer-by-layer for a fixed number of steps with optional validation, and checkpoints keep the teacher backbone but replace all decoders with the distilled student versions.

This project is conceptually related to RADLADS (Rapid Attention Distillation to Linear Attention Decoders at Scale), which motivates the follow-on full-model logits distillation phase implemented here.

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

Edit the single command line in `distill.sh`, then run:

```bash
./distill.sh
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

## Reference

- Daniel Goldstein, Eric Alcaide, Janna Lu, Eugene Cheah. “RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale.” COLM 2025. arXiv:2505.03005v4.
