#!/bin/bash
#SBATCH --job-name=emocatnets_v2_one_head_base
#SBATCH --export=ALL,TMPDIR=/home/h/hosseinis/.tmp
#SBATCH --partition=NvidiaAll
#SBATCH --time=55:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out

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
# Node-local TEMP + caches (avoid NFS .nfs* issues, avoid /tmp/user/<uid>)
# -----------------------------
if [[ -n "${SLURM_TMPDIR:-}" && -w "${SLURM_TMPDIR:-}" ]]; then
  TMPBASE="$SLURM_TMPDIR"
else
  TMPBASE="/tmp"
fi

export TMPDIR="$TMPBASE/$USER/slurm_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"

export TMP="$TMPDIR"
export TEMP="$TMPDIR"

export XDG_CACHE_HOME="$TMPDIR/xdg_cache"
export TORCH_HOME="$TMPDIR/torch"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export HF_HOME="$TMPDIR/hf"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$MPLCONFIGDIR" "$HF_HOME"

export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# -----------------------------

# Optional: if you use HF/datasets anywhere

# -----------------------------
# Less noisy logs + stable matplotlib in batch
# -----------------------------
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

# -----------------------------
# JAX/XLA (safe even if you don't use JAX; won't hurt)

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

python train.py dataloader=grey model=emocatnetsv2onehead_base epochs=300 bs=24 grad_accum=1 optimizer=adamw lr=2e-4 weight_decay=9e-3 \
    scheduler=warmup_cosine warmup_epochs=5 min_lr=3e-7 amp=true grad_clip=1.0 early_stop=300 \
    class_weight=true label_smoothing=0.07 \
    mix_prob=0.2 mixup_alpha=0.3 cutmix_alpha=0.4 \
    ema=true ema_decay=0.9996 eval_with_ema=true
