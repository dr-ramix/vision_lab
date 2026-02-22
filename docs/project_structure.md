# Project Structure

## Top-Level Layout

```text
vision_lab/
├── README.md
├── requirements.txt
├── docs/                      # Project documentation
├── main/                      # Python package and core scripts
│   ├── pyproject.toml
│   ├── src/fer/
│   │   ├── config/            # Train settings and CLI override parsing
│   │   ├── dataset/           # Dataloaders and dataset pipeline code
│   │   ├── engine/            # Trainer/evaluator/checkpoint helpers
│   │   ├── inference/         # Pretrained-model loading and HF hub integration
│   │   ├── metrics/           # Classification metrics and confusion matrix logic
│   │   ├── models/            # Model implementations and model registry
│   │   ├── pre_processing/    # Face detection + preprocessing utilities
│   │   ├── training/          # Training runner, losses, schedules, artifacts
│   │   ├── utils/             # Shared helper utilities
│   │   └── xai/               # Grad-CAM, occlusion, layer activation methods
│   ├── scripts/               # Executable scripts for data/train/inference/XAI
│   ├── weights/               # Local weight modules/assets
│   └── xai_results/           # XAI outputs (generated)
├── jobs/                      # SLURM training/XAI job scripts
├── training_output/           # Generated run artifacts + index
│   ├── runs/
│   └── runs_index.csv
├── testing/                   # Extra evaluation utilities (JAFFE, model eval)
├── classification_model/      # Standalone image classification entrypoint
├── demo/                      # Standalone video demo entrypoint
├── submission/                # Submission-ready copies of demo/classification
├── reports/                   # Reports and analysis outputs
├── download_sources.slurm     # SLURM wrapper for dataset download
└── prepare_images_raw.slurm   # SLURM wrapper for raw split preparation
```

## Core Code vs Runtime Artifacts

- Source code lives in `main/src/fer` and `main/scripts`.
- Cluster jobs live in `jobs` and top-level `*.slurm` files.
- Training artifacts are written to `training_output/runs/...`.
- Experiment index rows are appended to `training_output/runs_index.csv`.
