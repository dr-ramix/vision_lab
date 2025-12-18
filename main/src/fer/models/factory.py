from __future__ import annotations
from typing import Dict, Type
import torch.nn as nn

_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str):
    """Decorator to register a model class under a string name."""
    def wrapper(cls: Type[nn.Module]):
        _MODEL_REGISTRY[name] = cls
        return cls
    return wrapper

def available_models() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())

def build_model(cfg: dict) -> nn.Module:
    """
    cfg example:
    {
      "model": {"name": "simple_cnn", "num_classes": 7, "in_ch": 3}
    }
    """
    name = cfg["model"]["name"]
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {available_models()}")
    kwargs = dict(cfg["model"])
    kwargs.pop("name", None)
    return _MODEL_REGISTRY[name](**kwargs)
