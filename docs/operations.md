# Operations, Jobs, and Evaluation

## 1. Cluster Jobs (`jobs/`)

`jobs/` contains ready-to-run SLURM scripts for many training variants.

Typical usage:

```bash
sbatch jobs/train_resnet18.slurm
```

Most training jobs include:

- venv activation
- node-local cache/temp setup (`TMPDIR`, `XDG_CACHE_HOME`, `HF_HOME`, `TORCH_HOME`)
- environment checks (`nvidia-smi`, torch CUDA check)
- one `python train.py ...` command

## 2. XAI Scripts

Main scripts in `main/scripts`:

- `xai_gradcam.py`
- `xai_occlusion.py`
- `xai_layer_activation.py`

Corresponding SLURM launcher examples in `jobs/`:

- `gradcam.slurm`
- `occlusion.slurm`
- `layer_activation.slurm`

Outputs are written under `main/xai_results/` (or script-specific output roots).

## 3. Testing Utilities (`testing/`)

- `testing/download_jaffe.py`: download and preprocess JAFFE test data.
- `testing/eval_models.py`: evaluate multiple models and export metrics.

Note:

- Current testing scripts include hardcoded absolute paths; adjust them before using on another machine.

## 4. Standalone Deliverables

- `classification_model/run_classification.py`: image-folder classification pipeline.
- `demo/run_video_demo.py`: video FER demo.
- `submission/`: handoff-ready copies of both components.

## 5. Artifact Locations

- Training runs: `training_output/runs/*`
- Run index: `training_output/runs_index.csv`
- SLURM logs: `slurm-<jobid>.out` (in working directory unless overridden)
