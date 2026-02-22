# Training Guide

## Entry Point

Main training script:

- `main/scripts/train.py`

Run from `main/scripts`:

```bash
python train.py key=value key=value ...
```

## Examples

```bash
# Baseline
python train.py dataloader=grey model=resnet18 epochs=30 bs=64 lr=3e-4

# With warmup cosine and EMA
python train.py dataloader=grey model=emocatnetsv2_tiny scheduler=warmup_cosine warmup_epochs=5 ema=true ema_decay=0.9999

# Mixed loader
python train.py dataloader=mixed model=convnext_tiny epochs=100 bs=32
```

## How CLI Overrides Work

- Arguments are parsed as `key=value`.
- Only keys present in `TrainSettings` (`main/src/fer/config/defaults.py`) are allowed.
- Unknown keys fail fast.

Parsing supports booleans, numbers, `none/null`, quoted strings, and Python literals.

## Important Settings (From `TrainSettings`)

- `model`: model key used by `fer.models.registry.make_model`.
- `dataloader`: loader alias used by `fer.dataset.dataloaders.build`.
- `images_root`: defaults to `main/src/fer/dataset/standardized`.
- `epochs`, `bs`, `lr`, `optimizer`, `scheduler`.
- `amp`, `grad_accum`, `grad_clip`, `early_stop`.
- `class_weight`, `label_smoothing`.
- `mix_prob`, `mixup_alpha`, `cutmix_alpha`.
- `ema`, `ema_decay`, `eval_with_ema`.
- `select_metric` (best-checkpoint metric).

## Dataloader Aliases

Configured in `main/src/fer/dataset/dataloaders/build.py`.

Common aliases:

- `grey` / `gray` (default in settings)
- `mixed` / `main` / `default`
- `hist_eq`
- `fer2013`
- `fer2013_no_int`
- `int_norm`

Each alias expects a specific sublayout under `images_root`.

## Outputs

Runs are written to:

- `training_output/runs/<timestamp>__<model>__user-...__/`

Each run contains:

- `config.json`, `meta.json`
- `checkpoints/`
- `logs/`
- `metrics/`
- `mappings/`
- `previews/`
- `exports/`
- `notes/`

Global index file:

- `training_output/runs_index.csv`

## SLURM Training Jobs

Predefined jobs are in `jobs/`.

Typical pattern:

```bash
sbatch jobs/train_resnet18.slurm
```

These jobs usually:

- activate `venv`
- switch to `main/scripts`
- set node-local caches (`TMPDIR`, `HF_HOME`, `TORCH_HOME`, etc.)
- run `python train.py ...`
