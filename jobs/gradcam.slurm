#!/bin/bash
#SBATCH --job-name=xai_gradcam
#SBATCH --exclusive
#SBATCH --partition=NvidiaAll
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-gradcam-%j.out

set -euo pipefail

echo "==== Job info ===="
echo "JobID: ${SLURM_JOB_ID:-NA}"
echo "Node:  $(hostname)"
echo "User:  $USER"
echo "=================="

# -----------------------------
# Paths / venv
# -----------------------------
cd ~/vision_lab/vision_lab
source venv/bin/activate
cd ~/vision_lab/vision_lab/main/scripts

# -----------------------------
# Node-local TEMP
# -----------------------------
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

export XDG_CACHE_HOME="$TMPDIR/xdg_cache"
export TORCH_HOME="$TMPDIR/torch"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"

# -----------------------------
# Environment checks
# -----------------------------
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
nvidia-smi || true

python - <<'EOF'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

# -----------------------------
# Run Grad-CAM
# -----------------------------
python xai_gradcam.py

echo "==== Grad-CAM job finished ===="