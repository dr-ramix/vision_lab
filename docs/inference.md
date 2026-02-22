# Inference and Pretrained Weights
## You dont need this for final submission 
##Only if you want to test models for other task
## Core Modules

- Inference API: `main/src/fer/inference/models.py`
- Weight resolution: `main/src/fer/inference/hub.py`
- Training model registry used for reconstruction: `main/src/fer/models/registry.py`

## Weight Sources

`InferenceModelBase.load()` supports:

- `source="local"`: only local files in `main/src/fer/inference/weights/<weights_id>`
- `source="project"`: use local, otherwise download into project weights folder
- `source="cache"`: download to Hugging Face cache only

## Required Files Per Model Folder

Inside `main/src/fer/inference/weights/<weights_id>/`:

- `config.json`
- `model.safetensors` or `model_state_dict.pt`

Implementation detail:

- If `model.safetensors` exists but is empty, loader falls back to non-empty `.pt`.

## Download Weights

```bash
cd main/scripts
python download_weights.py --model resnet50
python download_weights.py --all
```

Useful flags:

- `--repo`
- `--revision`
- `--force`
- `--dry-run`

## Verify Local Weights

```bash
cd main/scripts
python verify_weights.py
python verify_weights.py --model ResNet50
python verify_weights.py --weights-id resnet50
python verify_weights.py --non-strict
```

## Minimal Python Usage

```python
from fer.inference.models import ResNet50

model = ResNet50().load(source="project", device="cpu")
```

The loader will:

1. Resolve weights (`local` / `project` / `cache`)
2. Read `config.json`
3. Build architecture through `make_model(...)`
4. Load checkpoint
5. Move model to requested device and set eval mode

## Extra Sanity Script

```bash
cd main/scripts
python test_import_model.py
```

This performs import/load/forward checks for a small model set.
