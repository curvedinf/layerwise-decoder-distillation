#!/usr/bin/env bash
set -euo pipefail

# Minimal ROCm/AMD setup for this repo.
# Creates a local venv and installs PyTorch ROCm wheels plus core Python deps.
#
# Architecture selection is best-effort and only needed in rare cases.
# Examples:
#   ./setup_amd.sh                 # auto / toolchain defaults
#   ./setup_amd.sh --gfx1100
#   ./setup_amd.sh --gfx908
#   ./setup_amd.sh --gfx942
#   ./setup_amd.sh --rocm-arch gfx942

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

ROCM_ARCH=""

usage() {
  cat <<'EOF'
Usage: ./setup_amd.sh [--rocm-arch gfxXXXX] [--gfxXXXX]

Notes:
- `--gfxXXXX` is generic. The script parses the `gfx...` token and exports it to
  `PYTORCH_ROCM_ARCH` and `HIP_ARCHITECTURE` so builds target that architecture.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rocm-arch)
      ROCM_ARCH="${2:-}"
      shift 2
      ;;
    --gfx*)
      ROCM_ARCH="${1#--}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Prefer the currently active venv if one is activated. Otherwise use repo-local .venv.
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  VENV_DIR="${VIRTUAL_ENV}"
  echo "Using active venv: ${VENV_DIR}"
else
  VENV_DIR="$SCRIPT_DIR/.venv"
  echo "Using repo venv: ${VENV_DIR}"
fi

if [[ -n "${ROCM_ARCH:-}" ]]; then
  if [[ ! "$ROCM_ARCH" =~ ^gfx[0-9a-z]+$ ]]; then
    echo "Invalid ROCm arch: '$ROCM_ARCH' (expected e.g. gfx1100, gfx942)" >&2
    exit 2
  fi
  echo "Using ROCm arch override: ${ROCM_ARCH}"
  export PYTORCH_ROCM_ARCH="${ROCM_ARCH}"
  export HIP_ARCHITECTURE="${ROCM_ARCH}"
else
  echo "No ROCm arch override provided (portable defaults)."
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  # Only create a venv if we are using the repo venv path (i.e. no active venv).
  if [[ "$VENV_DIR" != "$SCRIPT_DIR/.venv" ]]; then
    echo "Active venv selected but ${VENV_DIR}/bin/python is missing or not executable." >&2
    exit 1
  fi
  echo "Creating venv at $VENV_DIR using $PYTHON_BIN..."
  if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    echo "Failed to create venv. Install the python venv package for your Python version." >&2
    echo "Example (Ubuntu): sudo apt install python3-venv -y" >&2
    exit 1
  fi
fi

VENV_PY="$VENV_DIR/bin/python"

"$VENV_PY" -m pip install --upgrade pip wheel

# NOTE: This intentionally installs only the minimal deps required to run training code.
# Optional performance kernels (xformers/flash-attn/etc.) are not installed here.
"$VENV_PY" -m pip install --upgrade \
  --index-url https://download.pytorch.org/whl/rocm6.4 \
  torch==2.8.0 \
  torchvision \
  torchaudio \
  pytorch-triton-rocm==3.4.0 \
  triton==3.4.0

"$VENV_PY" -m pip install --upgrade --index-url https://pypi.org/simple \
  numpy \
  tqdm

"$VENV_PY" - <<'PY'
import importlib.metadata as md

for name in ("torch", "triton", "pytorch-triton-rocm", "numpy", "tqdm"):
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name} not installed")
PY
