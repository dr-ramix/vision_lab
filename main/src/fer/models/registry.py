from __future__ import annotations

from typing import Callable, Dict, List
import torch.nn as nn

from fer.models.cnn_resnet18 import ResNet18FER
from fer.models.cnn_vanilla import CNNVanilla
from fer.models.coatnet import CoAtNet
from fer.models.coatnetv2 import CoAtNetV2
from fer.models.cnn_resnet50 import ResNet50FER
from fer.models.convnext import convnextfer
from fer.models.emonext import emonext_fer
from fer.models.emocatnets import emocatnets_fer
from fer.models.emocatnets_v0 import emocatnets_v0_fer
from fer.models.mobilenetv2 import mobilenetv2_fer
from fer.models.mobilenetv3 import (
    mobilenetv3_tiny_fer,
    mobilenetv3_small_fer,
    mobilenetv3_base_fer,
    mobilenetv3_large_fer,
    mobilenetv3_xlarge_fer,
)
from fer.models.emocatnets_v2 import emocatnetsv2_fer
from fer.models.cnn_resnet101 import ResNet101FER
from fer.models.convnext_fer import convnextfer_v2
from fer.models.efficientnetv2 import EfficientNetV2
from fer.models.emocatnets_v3 import emocatnetsv3_fer
from fer.models.emocatnets_v2_fine import emocatnetsv2_fine_fer
from fer.models.emocatnets_v2_one_head import emocatnetv2onehead_fer
from fer.models.emocatnets_v3_fine import emocatnetsv3fine_fer
from fer.models.emocatnets_fine import emocatnetsfine_fer
from fer.models.emocatnets_v2_k5 import emocatnets_v2_k5_fer
from fer.models.emcoatnets_v2_residuals import emocatnetsv2residual_fer

from fer.models.cnn_resnet50_fine import resnet50fine_fer
from fer.models.cnn_resenet101_fine import resnet101fine_fer
from fer.models.convnext_fer_fine import convnextfer_v2_fine

from fer.models.coatnext import coatnext_fer





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
    "coatnetv2_tiny",
    lambda num_classes, in_channels=3, **_: CoAtNetV2(
        inp_h=64,
        inp_w=64,
        config="coatnetv2-0",
        num_classes=num_classes,
        in_channels=in_channels,
        ),
)

register_model(
    "coatnetv2_small",
    lambda num_classes, in_channels=3, **_: CoAtNetV2(
        inp_h=64,
        inp_w=64,
        config="coatnetv2-1",
        num_classes=num_classes,
        in_channels=in_channels,
        ),
)

register_model(
    "coatnetv2_base",
    lambda num_classes, in_channels=3, **_: CoAtNetV2(
        inp_h=64,
        inp_w=64,
        config="coatnetv2-2",
        num_classes=num_classes,
        in_channels=in_channels,
        ),
)

register_model(
    "coatnetv2_large",
    lambda num_classes, in_channels=3, **_: CoAtNetV2(
        inp_h=64,
        inp_w=64,
        config="coatnetv2-3",
        num_classes=num_classes,
        in_channels=in_channels,
        ),
)

register_model(
    "coatnetv2_xlarge",
    lambda num_classes, in_channels=3, **_: CoAtNetV2(
        inp_h=64,
        inp_w=64,
        config="coatnetv2-4",
        num_classes=num_classes,
        in_channels=in_channels,
        ),
)

register_model(
    "coatnetv2_huge",
    lambda num_classes, in_channels=3, **_: CoAtNetV2(
        inp_h=64,
        inp_w=64,
        config="coatnetv2-5",
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
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv2_fer(
        size="tiny",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv2_small",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv2_fer(
        size="small",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv2_base",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv2_fer(
        size="base",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv2_large",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv2_fer(
        size="large",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)




register_model(
    "resnet101",
    lambda num_classes, in_channels=3, transfer=False, **_: ResNet101FER(
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)



# ------------------------------------------------------------
# ConvNeXt-v2 (FER)
# - Residual stem (64 -> 64)
# - Delayed downsampling: 64 -> 32 -> 16 -> 8
# ------------------------------------------------------------

register_model(
    "convnextv2_tiny",
    lambda num_classes, in_channels=3, transfer=False, **_: convnextfer_v2(
        size="tiny",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "convnextv2_small",
    lambda num_classes, in_channels=3, transfer=False, **_: convnextfer_v2(
        size="small",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "convnextv2_base",
    lambda num_classes, in_channels=3, transfer=False, **_: convnextfer_v2(
        size="base",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "convnextv2_large",
    lambda num_classes, in_channels=3, transfer=False, **_: convnextfer_v2(
        size="large",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "convnextv2_xlarge",
    lambda num_classes, in_channels=3, transfer=False, **_: convnextfer_v2(
        size="xlarge",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "efficientnetv2-s",
    lambda num_classes, in_channels=3, transfer=False, **_: EfficientNetV2(
        model_name="efficientnetv2-s",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "efficientnetv2-b",
    lambda num_classes, in_channels=3, transfer=False, **_: EfficientNetV2(
        model_name="efficientnetv2-m",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "efficientnetv2-l",
    lambda num_classes, in_channels=3, transfer=False, **_: EfficientNetV2(
        model_name="efficientnetv2-l",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "efficientnetv2-xl",
    lambda num_classes, in_channels=3, transfer=False, **_: EfficientNetV2(
        model_name="efficientnetv2-xl",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)



# ------------------------------------------------------------
# EmoCatNets-v3 (SLOWER downsampling: 64 -> 64 -> 32 -> 16, T@16x16, proj_16)
# ------------------------------------------------------------

register_model(
    "emocatnetsv3_tiny",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv3_fer(
        size="tiny",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv3_small",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv3_fer(
        size="small",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv3_base",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv3_fer(
        size="base",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv3_large",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv3_fer(
        size="large",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)


# ------------------------------------------------------------
# EmoCatNets-v1 Fine (ConvNeXt-initialized backbone)
# ------------------------------------------------------------

register_model(
    "emocatnetsfine_tiny",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsfine_fer(
        size="tiny",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,   # True -> load ConvNeXt weights into stage1-3 + downsample_layer_1-3
    ),
)

register_model(
    "emocatnetsfine_small",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsfine_fer(
        size="small",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)

register_model(
    "emocatnetsfine_base",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsfine_fer(
        size="base",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)

register_model(
    "emocatnetsfine_large",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsfine_fer(
        size="large",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)

register_model(
    "emocatnetsfine_xlarge",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsfine_fer(
        size="xlarge",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)

# ------------------------------------------------------------
# EmoCatNets-v2 Fine (ConvNeXt-initialized backbone)
# ------------------------------------------------------------
register_model(
    "emocatnetsv2fine_tiny",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv2_fine_fer(
        size="tiny",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,   # True -> load timm convnext pretrained into stage1-3 + down1-3
    ),
)

register_model(
    "emocatnetsv2fine_small",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv2_fine_fer(
        size="small",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)

register_model(
    "emocatnetsv2fine_base",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv2_fine_fer(
        size="base",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)

register_model(
    "emocatnetsv2fine_large",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv2_fine_fer(
        size="large",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)


# ------------------------------------------------------------
# EmoCatNets-v3 Fine (ConvNeXt-initialized backbone)
# ------------------------------------------------------------

register_model(
    "emocatnetsv3fine_tiny",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv3fine_fer(
        size="tiny",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,   # True -> load convnext weights into stage1-3 + down1-2
    ),
)

register_model(
    "emocatnetsv3fine_small",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv3fine_fer(
        size="small",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)

register_model(
    "emocatnetsv3fine_base",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv3fine_fer(
        size="base",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)

register_model(
    "emocatnetsv3fine_large",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv3fine_fer(
        size="large",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)


# ------------------------------------------------------------
# EmoCatNets-v2-K5 (5x5 DWConv variant, NO transfer)
# ------------------------------------------------------------

register_model(
    "emocatnetsv2k5_tiny",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_v2_k5_fer(
        size="tiny",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv2k5_small",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_v2_k5_fer(
        size="small",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv2k5_base",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_v2_k5_fer(
        size="base",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv2k5_large",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_v2_k5_fer(
        size="large",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)





register_model(
    "coatnext_tiny",
    lambda num_classes, in_channels=3, transfer=False, **_: coatnext_fer(
        size="tiny",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)

register_model(
    "coatnext_small",
    lambda num_classes, in_channels=3, transfer=False, **_: coatnext_fer(
        size="small",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)

register_model(
    "coatnext_base",
    lambda num_classes, in_channels=3, transfer=False, **_: coatnext_fer(
        size="base",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)

register_model(
    "coatnext_large",
    lambda num_classes, in_channels=3, transfer=False, **_: coatnext_fer(
        size="large",
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)



register_model(
    "emocatnetsv0_tiny",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_v0_fer(
        size="tiny",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv0_small",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_v0_fer(
        size="small",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv0_base",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_v0_fer(
        size="base",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv0_large",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnets_v0_fer(
        size="large",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)


register_model(
    "emocatnetsv2onehead_tiny",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetv2onehead_fer(
        size="tiny",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv2onehead_small",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetv2onehead_fer(
        size="small",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv2onehead_base",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetv2onehead_fer(
        size="base",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv2onehead_large",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetv2onehead_fer(
        size="large",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)


register_model(
    "emocatnetsv2residual_tiny",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv2residual_fer(
        size="tiny",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv2residual_small",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv2residual_fer(
        size="small",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv2residual_base",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv2residual_fer(
        size="base",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)

register_model(
    "emocatnetsv2residual_large",
    lambda num_classes, in_channels=3, transfer=False, **_: emocatnetsv2residual_fer(
        size="large",
        num_classes=num_classes,
        in_channels=in_channels,
    ),
)


register_model(
    "resnet50_fine",
    lambda num_classes, in_channels=3, transfer=False, **_:resnet50fine_fer(
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)

register_model(
    "resnet101_fine",
    lambda num_classes, in_channels=3, transfer=False, **_: resnet101fine_fer(
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)

register_model(
    "convnext_fer_fine",
    lambda num_classes, in_channels=3, transfer=False, **_: convnextfer_v2_fine(
        num_classes=num_classes,
        in_channels=in_channels,
        transfer=transfer,
    ),
)
