#!/bin/bash
#SBATCH --job-name=emonext_base_SA
#SBATCH --partition=NvidiaAll
#SBATCH --time=90:00:00
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
# Node-local TEMP + caches (avoid NFS .nfs* issues)
# -----------------------------
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

export XDG_CACHE_HOME="$TMPDIR/xdg_cache"
export TORCH_HOME="$TMPDIR/torch"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"

# Optional: if you use HF/datasets anywhere
export HF_HOME="$TMPDIR/hf"
mkdir -p "$HF_HOME"

# -----------------------------
# Less noisy logs + stable matplotlib in batch
# -----------------------------
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

# -----------------------------
# JAX/XLA (safe even if you don't use JAX; won't hurt)
# -----------------------------
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# -----------------------------
# Environment checks
# -----------------------------
echo "==== Environment checks ===="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
nvidia-smi || true

python - <<'EOF'
import sys
try:
    import torch
    print("Python:", sys.version.split()[0])
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch check failed:", e)
EOF
echo "============================"

# -----------------------------
# Run training
# -----------------------------
python train.py model=emonext_base epochs=450 bs=512 optimizer=adamw lr=1e-4 weight_decay=5e-2 \
  scheduler=warmup_cosine warmup_epochs=10 min_lr=1e-6 amp=true grad_clip=1.0 early_stop=60 \
  class_weight=true label_smoothing=0.05 \
  loss=emonext emonext_lambda=0.2 \
  mix_prob=0.0 mixup_alpha=0.0 cutmix_alpha=0.0 \
  ema=true ema_decay=0.9999 eval_with_ema=true
