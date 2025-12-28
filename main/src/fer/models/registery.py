from __future__ import annotations
from typing import Callable, Dict
import torch.nn as nn

from fer.models.cnn_resnet18 import ResNet18FER
from fer.models.cnn_vanilla import CNNVanilla

_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "resnet18": lambda num_classes, in_channels, transfer: ResNet18FER(
        num_classes=num_classes,
        in_channels=in_channels,
        # if your ResNet18FER supports pretrained/transfer, add it here
        # pretrained=transfer,
    ),
    "cnn_vanilla": lambda num_classes, in_channels, transfer: CNNVanilla(),
}

def available_models() -> list[str]:
    return sorted(_REGISTRY.keys())

def make_model(name: str, *, num_classes: int, in_channels: int = 3, transfer: bool = False) -> nn.Module:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {available_models()}")
    return _REGISTRY[name](num_classes=num_classes, in_channels=in_channels, transfer=transfer)
