#!/bin/bash
#SBATCH --job-name=xai_layer_activation
#SBATCH --partition=NvidiaAll
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-layer-activation-%j.out

set -euo pipefail

echo "==== Job info ===="
echo "JobID: ${SLURM_JOB_ID:-NA}"
echo "Node:  $(hostname)"
echo "User:  $USER"
echo "CWD:   $(pwd)"
echo "=================="

# -----------------------------
# Paths / venv
# -----------------------------
cd ~/vision_lab/vision_lab
source venv/bin/activate
cd ~/vision_lab/vision_lab/main/scripts

# -----------------------------
# Node-local TEMP + caches
# -----------------------------
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

export XDG_CACHE_HOME="$TMPDIR/xdg_cache"
export TORCH_HOME="$TMPDIR/torch"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"

# Optional (harmless if unused)
export HF_HOME="$TMPDIR/hf"
mkdir -p "$HF_HOME"

# -----------------------------
# Less noisy logs + stable matplotlib
# -----------------------------
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

# -----------------------------
# Environment checks
# -----------------------------
echo "==== Environment checks ===="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
nvidia-smi || true

python - <<'EOF'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF
echo "============================"

# -----------------------------
# Run Layer Activation XAI
# -----------------------------
python xai_layer_activation.py

echo "==== Layer Activation job finished ===="