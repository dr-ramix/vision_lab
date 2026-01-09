from __future__ import annotations

from typing import Callable, Dict, List
import torch.nn as nn

from fer.models.cnn_resnet18 import ResNet18FER
from fer.models.cnn_vanilla import CNNVanilla
from fer.models.coatnet import CoAtNet
from fer.models.cnn_resnet50 import ResNet50FER
from fer.models.convnext import convnextfer
from fer.models.emonext import emonext_fer
from fer.models.emocatnets import emocatnets_fer
from fer.models.mobilenetv2 import mobilenetv2_fer
from fer.models.mobilenetv3 import (
    mobilenetv3_tiny_fer,
    mobilenetv3_small_fer,
    mobilenetv3_base_fer,
    mobilenetv3_large_fer,
    mobilenetv3_xlarge_fer,
)
from fer.models.emocatnets_v2 import emocatnets_v2_fer


# ------------------------------------------------------------
# Model registry
# ------------------------------------------------------------
# Each entry is a function that returns nn.Module.
# All functions must accept the SAME arguments.
#
# Signature:
#   factory(num_classes: int, in_channels: int, transfer: bool) -> nn.Module
# ------------------------------------------------------------

_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str, factory: Callable[..., nn.Module]) -> None:
    """
    Register a model factory.
    Keep this explicit (no decorators, no magic).
    """
    key = name.strip().lower()
    if not key:
        raise ValueError("Model name cannot be empty.")
    if key in _REGISTRY:
        raise ValueError(f"Model '{key}' already registered.")
    _REGISTRY[key] = factory


def available_models() -> List[str]:
    return sorted(_REGISTRY.keys())


def make_model(
    name: str,
    *,
    num_classes: int,
    in_channels: int = 3,
    transfer: bool = False,
) -> nn.Module:
    """
    Create a model by name.

    Used directly by train.py / runner.py.
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {available_models()}")
    return _REGISTRY[key](
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    )


# ------------------------------------------------------------
# Register models here (explicit & readable)
# ------------------------------------------------------------

register_model(
    "resnet18",
    lambda num_classes, in_channels=3, transfer=False, **_: ResNet18FER(
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "resnet50",
    lambda num_classes, in_channels=3, transfer=False, **_: ResNet50FER(
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "cnn_vanilla",
    lambda num_classes, **_: CNNVanilla(num_classes=num_classes),
)

register_model(
    "coatnet_tiny",
    lambda num_classes, in_channels=3, **_: CoAtNet(
        inp_h=64,
        inp_w=64,
        config="coatnet-0",
        num_classes=num_classes,
        in_channels=in_channels,
        ),
)

register_model(
    "coatnet_small",
    lambda num_classes, in_channels=3, **_: CoAtNet(
        inp_h=64,
        inp_w=64,
        config="coatnet-1",
        num_classes=num_classes,
        in_channels=in_channels,
        ),
)

register_model(
    "coatnet_base",
    lambda num_classes, in_channels=3, **_: CoAtNet(
        inp_h=64,
        inp_w=64,
        config="coatnet-2",
        num_classes=num_classes,
        in_channels=in_channels,
        ),
)

register_model(
    "coatnet_large",
    lambda num_classes, in_channels=3, **_: CoAtNet(
        inp_h=64,
        inp_w=64,
        config="coatnet-3",
        num_classes=num_classes,
        in_channels=in_channels,
        ),
)

register_model(
    "coatnet_xlarge",
    lambda num_classes, in_channels=3, **_: CoAtNet(
        inp_h=64,
        inp_w=64,
        config="coatnet-4",
        num_classes=num_classes,
        in_channels=in_channels,
        ),
)

register_model(
    "coatnet_huge",
    lambda num_classes, in_channels=3, **_: CoAtNet(
        inp_h=64,
        inp_w=64,
        config="coatnet-5",
        num_classes=num_classes,
        in_channels=in_channels,
        ),
)

register_model(
    "convnext_tiny",
    lambda num_classes, in_channels=3, transfer=False, **_: convnextfer(
        size="tiny",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "convnext_small",
    lambda num_classes, in_channels=3, transfer=False, **_: convnextfer(
        size="small",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "convnext_base",
    lambda num_classes, in_channels=3, transfer=False, **_: convnextfer(
        size="base",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "convnext_large",
    lambda num_classes, in_channels=3, transfer=False, **_: convnextfer(
        size="large",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "convnext_xlarge",
    lambda num_classes, in_channels=3, transfer=False, **_: convnextfer(
        size="xlarge",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)


register_model(
    "emonext_tiny",
    lambda num_classes, in_channels=3, transfer=False, **_: emonext_fer(
        size="tiny",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emonext_small",
    lambda num_classes, in_channels=3, transfer=False, **_: emonext_fer(
        size="small",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emonext_base",
    lambda num_classes, in_channels=3, transfer=False, **_: emonext_fer(
        size="base",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emonext_large",
    lambda num_classes, in_channels=3, transfer=False, **_: emonext_fer(
        size="large",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emonext_xlarge",
    lambda num_classes, in_channels=3, transfer=False, **_: emonext_fer(
        size="xlarge",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnets_tiny",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_fer(
        size="tiny",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnets_small",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_fer(
        size="small",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnets_base",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_fer(
        size="base",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnets_large",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_fer(
        size="large",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnet_xlarge",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_fer(
        size="xlarge",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "mobilenetv2",
    lambda num_classes, in_channels=3, transfer=False, **_: mobilenetv2_fer(
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)



register_model("mobilenetv3_tiny",   lambda num_classes, in_channels=3, transfer=False, **_: mobilenetv3_tiny_fer(num_classes=num_classes, in_channels=in_channels, transfer=transfer))
register_model("mobilenetv3_small",  lambda num_classes, in_channels=3, transfer=False, **_: mobilenetv3_small_fer(num_classes=num_classes, in_channels=in_channels, transfer=transfer))
register_model("mobilenetv3_base",   lambda num_classes, in_channels=3, transfer=False, **_: mobilenetv3_base_fer(num_classes=num_classes, in_channels=in_channels, transfer=transfer))
register_model("mobilenetv3_large",  lambda num_classes, in_channels=3, transfer=False, **_: mobilenetv3_large_fer(num_classes=num_classes, in_channels=in_channels, transfer=transfer))
register_model("mobilenetv3_xlarge", lambda num_classes, in_channels=3, transfer=False, **_: mobilenetv3_xlarge_fer(num_classes=num_classes, in_channels=in_channels, transfer=transfer))


# ------------------------------------------------------------
# EmoCatNets-v2 (Residual STN + 64->64 stem + CBAM + T@8x8 + multi-scale head)
# ------------------------------------------------------------

register_model(
    "emocatnetsv2_tiny",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_v2_fer(
        size="tiny",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv2_small",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_v2_fer(
        size="small",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv2_base",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_v2_fer(
        size="base",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv2_large",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_v2_fer(
        size="large",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)
