# fer/models/resnet50_fine.py
# ============================================================
# Fine-tuning for ResNet50FER
#
# Transfers timm ResNet-50 pretrained weights into:
#   - conv1, bn1
#   - layer1, layer2, layer3, layer4
# ============================================================

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import timm

from fer.models.resnet50 import ResNet50FER


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if sd and next(iter(sd.keys())).startswith("module."):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


@torch.no_grad()
def transfer_resnet50_into_resnet50fer(
    model: nn.Module,
    resnet_name: str = "resnet50",
    pretrained: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    timm ResNet-50 -> your ResNet50FER

    Maps:
      conv1, bn1
      layer1..layer4 (all bottleneck blocks)

    Skips:
      fc (classifier head)
    """

    dst_sd = model.state_dict()

    required_prefixes = [
        "conv1.", "bn1.",
        "layer1.", "layer2.", "layer3.", "layer4."
    ]
    missing = [p for p in required_prefixes if not any(k.startswith(p) for k in dst_sd)]
    if missing:
        raise RuntimeError(
            f"Destination model is missing expected prefixes: {missing}"
        )

    src = timm.create_model(resnet_name, pretrained=pretrained).eval()
    src_sd = _strip_module_prefix(src.state_dict())

    updates: Dict[str, torch.Tensor] = {}
    copied = 0
    skipped_missing = 0
    skipped_shape = 0

    for k_src, v_src in src_sd.items():
        # skip classifier
        if k_src.startswith("fc."):
            continue

        # direct name match for stem + layers
        if (
            k_src.startswith("conv1.") or
            k_src.startswith("bn1.") or
            k_src.startswith("layer1.") or
            k_src.startswith("layer2.") or
            k_src.startswith("layer3.") or
            k_src.startswith("layer4.")
        ):
            k_dst = k_src
        else:
            continue

        if k_dst not in dst_sd:
            skipped_missing += 1
            continue
        if dst_sd[k_dst].shape != v_src.shape:
            skipped_shape += 1
            continue

        updates[k_dst] = v_src
        copied += 1

    if updates:
        dst_sd.update(updates)
        model.load_state_dict(dst_sd, strict=False)

    if verbose:
        print(
            f"[resnet50 transfer] pretrained={pretrained} | "
            f"copied={copied} skipped_missing={skipped_missing} skipped_shape={skipped_shape}"
        )
        for s in list(updates.keys())[:10]:
            print("  loaded:", s)

    return dict(
        copied=copied,
        skipped_missing=skipped_missing,
        skipped_shape=skipped_shape,
    )


# ------------------------------------------------------------
# Factory (registry-friendly)
# ------------------------------------------------------------

def resnet50fine_fer(
    *,
    num_classes: int = 6,
    in_channels: int = 3,
    transfer: bool = False,
    resnet_pretrained: bool = True,
    verbose: bool = True,
    **kwargs,
) -> ResNet50FER:
    """
    Same registry-style signature:
      factory(num_classes, in_channels, transfer) -> nn.Module

    transfer=True loads timm ResNet-50 weights into conv + layers
    """

    model = ResNet50FER(
        num_classes=num_classes,
        in_channels=in_channels,
        **kwargs,
    )

    if transfer:
        transfer_resnet50_into_resnet50fer(
            model=model,
            resnet_name="resnet50",
            pretrained=resnet_pretrained,
            verbose=verbose,
        )

    return model


# ------------------------------------------------------------
# Optional: param groups helper (recommended for transfer=True)
# ------------------------------------------------------------

def make_resnet50_param_groups(
    model: nn.Module,
    base_lr: float,
    backbone_lr_mult: float = 0.1,
) -> list[dict]:
    """
    backbone (loaded):
      conv1, bn1, layer1..layer4

    new modules:
      fc
    """
    backbone, rest = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")):
            backbone.append(p)
        else:
            rest.append(p)

    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": base_lr * backbone_lr_mult})
    if rest:
        groups.append({"params": rest, "lr": base_lr})

    return groups