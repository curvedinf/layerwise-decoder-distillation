#!/usr/bin/env bash
./.venv/bin/python distill.py --out-dir runs --steps-per-layer 100 --batch-size 64 --seq-len 8192 --no-compile --memory-cache --save-every 10
