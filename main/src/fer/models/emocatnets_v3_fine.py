# fer/models/emocatnets_v3_fine.py
from __future__ import annotations

"""
EmoCatNets-v3 (FER 64×64) — fine-tuning helper

This file provides:
1) Weight transfer from timm ConvNeXt/ConvNeXtV2 into EmoCatNetsV3 (compatible parts only).
2) A registry-friendly factory `emocatnetsv3fine_fer(...)`.
3) Optional optimizer param-groups helper for using a smaller LR on the transferred backbone.

What gets transferred (when shapes match):
- stage1/stage2/stage3  <- timm stages.0 / stages.1 / stages.2
- down1/down2/down3     <- timm downsample_layers.1 / .2 / .3

What is NOT transferred:
- stem (because your stem is stride=1 and differs from timm)
- stage4 transformer tail
- cbam modules
- head / final_ln

Notes:
- Your ConvNextBlockV2 uses NHWC + Linear like timm ConvNeXt blocks, but your stage modules are
  `nn.Sequential(block0, block1, ...)` whereas timm uses `stages.i.blocks.j.*`.
  Therefore we MUST map `stages.i.blocks.j.*` -> `stage{i+1}.{j}.*`.
- This implementation handles both:
  - convnext_*    (no GRN): loads what exists; GRN weights will be missing (fine).
  - convnextv2_*  (has GRN): loads GRN too (best match for your blocks).
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import timm

from fer.models.emocatnets_v3 import EmoCatNetsV3, EMOCATNETS_V3_SIZES


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if sd and next(iter(sd.keys())).startswith("module."):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _remap_block_names(k: str) -> str:
    """
    timm ConvNeXt(/V2) block names -> your ConvNextBlockV2 names
      dwconv  -> depthwise_conv
      norm    -> layer_norm
      pwconv1 -> pointwise_conv1
      pwconv2 -> pointwise_conv2
    """
    k = k.replace(".dwconv.", ".depthwise_conv.")
    k = k.replace(".norm.", ".layer_norm.")
    k = k.replace(".pwconv1.", ".pointwise_conv1.")
    k = k.replace(".pwconv2.", ".pointwise_conv2.")
    return k


# ------------------------------------------------------------
# ConvNeXt -> EmoCatNetsV3 weight transfer
# ------------------------------------------------------------

@torch.no_grad()
def transfer_convnext_into_emocatnetsv3(
    model: nn.Module,
    convnext_name: str = "convnextv2_tiny",
    pretrained: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Load timm ConvNeXt/ConvNeXtV2 pretrained weights into compatible parts of EmoCatNetsV3.

    Mapping (timm -> EmoCat):
      downsample_layers.1.*  -> down1.*
      downsample_layers.2.*  -> down2.*
      downsample_layers.3.*  -> down3.*
      stages.0.blocks.j.*    -> stage1.j.*
      stages.1.blocks.j.*    -> stage2.j.*
      stages.2.blocks.j.*    -> stage3.j.*

    Skips:
      - downsample_layers.0.* (timm stem)
      - stages.3.* (timm last stage)
      - norm.*, head.*
      - anything that doesn't exist or doesn't match shape
    """
    src = timm.create_model(convnext_name, pretrained=pretrained).eval()
    src_sd = _strip_module_prefix(src.state_dict())
    dst_sd = model.state_dict()

    # Validate expected prefixes in destination
    required = ["stage1.", "stage2.", "stage3.", "down1.", "down2.", "down3."]
    missing_prefixes = [p for p in required if not any(k.startswith(p) for k in dst_sd.keys())]
    if missing_prefixes:
        raise RuntimeError(
            f"Destination model state_dict missing expected prefixes: {missing_prefixes}. "
            f"Check EmoCatNetsV3 attribute names."
        )

    copied = 0
    skipped_missing = 0
    skipped_shape = 0

    updates: Dict[str, torch.Tensor] = {}

    for k_src, v_src in src_sd.items():
        # skip timm stem + last stage + head/norm
        if k_src.startswith("downsample_layers.0."):
            continue
        if k_src.startswith("stages.3."):
            continue
        if k_src.startswith("head.") or k_src.startswith("norm."):
            continue

        k_dst: Optional[str] = None

        # Downsample layers
        if k_src.startswith("downsample_layers.1."):
            k_dst = "down1." + k_src[len("downsample_layers.1."):]
        elif k_src.startswith("downsample_layers.2."):
            k_dst = "down2." + k_src[len("downsample_layers.2."):]
        elif k_src.startswith("downsample_layers.3."):
            k_dst = "down3." + k_src[len("downsample_layers.3."):]
        # Stages (map stages.i.blocks.j.* -> stage{i+1}.{j}.*)
        elif k_src.startswith("stages.0.blocks."):
            tail = k_src[len("stages.0.blocks."):]   # e.g. "0.dwconv.weight"
            k_dst = _remap_block_names("stage1." + tail)
        elif k_src.startswith("stages.1.blocks."):
            tail = k_src[len("stages.1.blocks."):]
            k_dst = _remap_block_names("stage2." + tail)
        elif k_src.startswith("stages.2.blocks."):
            tail = k_src[len("stages.2.blocks."):]
            k_dst = _remap_block_names("stage3." + tail)
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
        model.load_state_dict({**dst_sd, **updates}, strict=False)

    if verbose:
        print(
            f"[emocatnetsv3 transfer] convnext='{convnext_name}' pretrained={pretrained} | "
            f"copied={copied} skipped_missing={skipped_missing} skipped_shape={skipped_shape}"
        )
        for s in list(updates.keys())[:10]:
            print("  loaded:", s)

    return {"copied": copied, "skipped_missing": skipped_missing, "skipped_shape": skipped_shape}


# ------------------------------------------------------------
# Factory (registry-friendly)
# ------------------------------------------------------------

# Best default match (ConvNeXtV2 has GRN).
_SIZE_TO_CONVNEXT: Dict[str, str] = {
    "nano":  "convnextv2_nano",   # if timm doesn't have nano, override via convnext_name
    "tiny":  "convnextv2_tiny",
    "small": "convnextv2_small",
    "base":  "convnextv2_base",
    "large": "convnextv2_large",
    "xlarge": "convnextv2_xlarge",
}


def emocatnetsv3fine_fer(
    *,
    size: str = "tiny",
    in_channels: int = 3,
    num_classes: int = 6,
    transfer: bool = False,
    convnext_name: Optional[str] = None,
    convnext_pretrained: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> EmoCatNetsV3:
    """
    Fine-tuning factory.

    If transfer=True, loads ConvNeXt/ConvNeXtV2 pretrained weights into:
      stage1-3 + down1-3 (when matching shapes exist).
    """
    size = size.lower()
    if size not in EMOCATNETS_V3_SIZES:
        raise ValueError(f"Unknown size='{size}'. Valid: {list(EMOCATNETS_V3_SIZES.keys())}")

    cfg = EMOCATNETS_V3_SIZES[size]
    model = EmoCatNetsV3(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=cfg.depths,
        dims=cfg.dims,
        drop_path_rate=cfg.drop_path_rate,
        layer_scale_init_value=cfg.layer_scale_init_value,
        head_init_scale=cfg.head_init_scale,
        num_heads=cfg.num_heads,
        attn_dropout=cfg.attn_dropout,
        proj_dropout=cfg.proj_dropout,
        cbam_reduction=cfg.cbam_reduction,
        **kwargs,
    )

    if transfer:
        cn = convnext_name or _SIZE_TO_CONVNEXT.get(size, "convnextv2_tiny")
        transfer_convnext_into_emocatnetsv3(
            model=model,
            convnext_name=cn,
            pretrained=convnext_pretrained,
            verbose=verbose,
        )

    return model


# ------------------------------------------------------------
# Optional: optimizer param groups
# ------------------------------------------------------------

def make_emocatnetsv3_param_groups(
    model: nn.Module,
    base_lr: float,
    backbone_lr_mult: float = 0.1,
) -> list[dict]:
    """
    Two LR groups:
      - backbone (transferred): stage1-3 + down1-3
      - rest (new): stem, cbam*, stage4, final_ln, head, etc.
    """
    backbone, rest = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(("stage1", "stage2", "stage3", "down1", "down2", "down3")):
            backbone.append(p)
        else:
            rest.append(p)

    groups: list[dict] = []
    if backbone:
        groups.append({"params": backbone, "lr": base_lr * backbone_lr_mult})
    if rest:
        groups.append({"params": rest, "lr": base_lr})
    return groups
