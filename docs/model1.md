# Models Reference

Model creation for training and inference is centralized in:

- `main/src/fer/models/registry.py`

Training entrypoint (`main/scripts/train.py`) uses `make_model(model_name, ...)` with the `model` setting.

Inference wrappers in `main/src/fer/inference/models.py` map importable class names to registry model keys via `InferenceSpec(arch_name=...)`.

To inspect currently registered model names, check the `register_model(...)` entries in `registry.py`.
