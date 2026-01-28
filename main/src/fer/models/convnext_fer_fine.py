# fer/models/convnextfer_v2_fine.py
# ============================================================
# Fine-tuning for ConvNeXtFERv2
#
# Transfers timm ConvNeXt pretrained weights into:
#   - stage1, stage2, stage3        (ConvNeXt blocks)
#   - downsample_layer_1/2/3        (LN + Conv stride2)
# ============================================================

from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn
import timm

from fer.models.convnextfer_v2 import ConvNeXtFERv2, CONVNEXTFER_SIZES


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if sd and next(iter(sd.keys())).startswith("module."):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _remap_block_names(k: str) -> str:
    """
    timm ConvNeXt block -> your ConvNextBlock
      dwconv   -> depthwise_conv
      norm     -> layer_norm
      pwconv1  -> pointwise_conv1
      pwconv2  -> pointwise_conv2
    """
    k = k.replace(".dwconv.", ".depthwise_conv.")
    k = k.replace(".norm.", ".layer_norm.")
    k = k.replace(".pwconv1.", ".pointwise_conv1.")
    k = k.replace(".pwconv2.", ".pointwise_conv2.")
    return k


# ------------------------------------------------------------
# Transfer
# ------------------------------------------------------------

@torch.no_grad()
def transfer_convnext_into_convnextfer_v2(
    model: nn.Module,
    convnext_name: str = "convnext_tiny",
    pretrained: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    timm ConvNeXt -> ConvNeXtFERv2 (conv backbone only)

    Maps:
      timm downsample_layers.1 -> downsample_layer_1
      timm downsample_layers.2 -> downsample_layer_2
      timm downsample_layers.3 -> downsample_layer_3

      timm stages.0 -> stage1
      timm stages.1 -> stage2
      timm stages.2 -> stage3

    Skips:
      downsample_layers.0 (stem)
      stages.3 (your stage4 is task-specific)
      norm / head
    """

    dst_sd = model.state_dict()

    required_prefixes = [
        "stage1.", "stage2.", "stage3.",
        "downsample_layer_1.", "downsample_layer_2.", "downsample_layer_3.",
    ]
    missing = [p for p in required_prefixes if not any(k.startswith(p) for k in dst_sd)]
    if missing:
        raise RuntimeError(
            f"Destination model missing expected prefixes: {missing}"
        )

    src = timm.create_model(convnext_name, pretrained=pretrained).eval()
    src_sd = _strip_module_prefix(src.state_dict())

    updates: Dict[str, torch.Tensor] = {}
    copied = 0
    skipped_missing = 0
    skipped_shape = 0

    for k_src, v_src in src_sd.items():

        # ---- explicit skips ----
        if k_src.startswith("downsample_layers.0."):
            continue
        if k_src.startswith("stages.3."):
            continue
        if k_src.startswith(("norm.", "head.")):
            continue

        k_dst = None

        # ---- downsampling ----
        if k_src.startswith("downsample_layers.1."):
            k_dst = "downsample_layer_1." + k_src[len("downsample_layers.1."):]
        elif k_src.startswith("downsample_layers.2."):
            k_dst = "downsample_layer_2." + k_src[len("downsample_layers.2."):]
        elif k_src.startswith("downsample_layers.3."):
            k_dst = "downsample_layer_3." + k_src[len("downsample_layers.3."):]
        # ---- stages ----
        elif k_src.startswith("stages.0."):
            tail = k_src[len("stages.0."):].replace("blocks.", "")
            k_dst = _remap_block_names("stage1." + tail)
        elif k_src.startswith("stages.1."):
            tail = k_src[len("stages.1."):].replace("blocks.", "")
            k_dst = _remap_block_names("stage2." + tail)
        elif k_src.startswith("stages.2."):
            tail = k_src[len("stages.2."):].replace("blocks.", "")
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
        dst_sd.update(updates)
        model.load_state_dict(dst_sd, strict=False)

    if verbose:
        print(
            f"[convnextfer-v2 transfer] convnext='{convnext_name}' pretrained={pretrained} | "
            f"copied={copied} skipped_missing={skipped_missing} skipped_shape={skipped_shape}"
        )
        for k in list(updates.keys())[:10]:
            print("  loaded:", k)

    return dict(
        copied=copied,
        skipped_missing=skipped_missing,
        skipped_shape=skipped_shape,
    )


# ------------------------------------------------------------
# Factory (registry-friendly)
# ------------------------------------------------------------

_SIZE_TO_CONVNEXT = {
    "tiny":   "convnext_tiny",
    "small":  "convnext_small",
    "base":   "convnext_base",
    "large":  "convnext_large",
    "xlarge": "convnext_xlarge",
}


def convnextfer_v2_fine(
    *,
    size: str = "tiny",
    in_channels: int = 3,
    num_classes: int = 6,
    transfer: bool = False,
    convnext_name: Optional[str] = None,
    convnext_pretrained: bool = True,
    verbose: bool = True,
    **kwargs,
) -> ConvNeXtFERv2:
    """
    Registry-style factory:
      factory(num_classes, in_channels, transfer) -> nn.Module
    """

    size = size.lower()
    if size not in CONVNEXTFER_SIZES:
        raise ValueError(f"Unknown size='{size}'. Valid: {list(CONVNEXTFER_SIZES.keys())}")

    cfg = CONVNEXTFER_SIZES[size]
    model = ConvNeXtFERv2(
        in_channels=in_channels,
        num_classes=num_classes,
        depths=cfg.depths,
        dims=cfg.dims,
        drop_path_rate=cfg.drop_path_rate,
        layer_scale_init_value=cfg.layer_scale_init_value,
        head_init_scale=cfg.head_init_scale,
        **kwargs,
    )

    if transfer:
        cn = convnext_name or _SIZE_TO_CONVNEXT.get(size, "convnext_tiny")
        transfer_convnext_into_convnextfer_v2(
            model=model,
            convnext_name=cn,
            pretrained=convnext_pretrained,
            verbose=verbose,
        )

    return model


# ------------------------------------------------------------
# Optional: param groups helper
# ------------------------------------------------------------

def make_convnextfer_v2_param_groups(
    model: nn.Module,
    base_lr: float,
    backbone_lr_mult: float = 0.1,
) -> list[dict]:
    """
    backbone (loaded):
      stage1-3 + downsample_layer_1-3

    new modules:
      stem, stage4, final_ln, head
    """

    backbone, rest = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith((
            "stage1", "stage2", "stage3",
            "downsample_layer_1", "downsample_layer_2", "downsample_layer_3",
        )):
            backbone.append(p)
        else:
            rest.append(p)

    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": base_lr * backbone_lr_mult})
    if rest:
        groups.append({"params": rest, "lr": base_lr})

    return groups
