#!/bin/bash
#SBATCH --job-name=coatnet_tiny
#SBATCH --partition=NvidiaAll
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out
#SBATCH --exclusive

set -euo pipefail

echo "==== Job info ===="
echo "JobID: ${SLURM_JOB_ID:-NA}"
echo "Node:  $(hostname)"
echo "CWD:   $(pwd)"
echo "User:  $USER"
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
export HF_HOME="$TMPDIR/hf"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$HF_HOME"

# -----------------------------
# Logs / CUDA / AMP
# -----------------------------
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# -----------------------------
# Environment check
# -----------------------------
nvidia-smi || true

python - <<'EOF'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

# -----------------------------
# Run training
# -----------------------------
python train_coatnet-tiny0-tl.py