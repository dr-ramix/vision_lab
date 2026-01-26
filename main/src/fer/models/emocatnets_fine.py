# fer/models/emocatnets_fine.py
# ============================================================
# Fine-tuning/transfer wrapper for YOUR EmoCatNets (C-C-C-T @ 32/16/8/4)
#
# Transfers timm ConvNeXt pretrained weights into:
#   - stage1, stage2, stage3  (ConvNextBlock stacks)
#   - downsample_layer_1/2/3  (LN + conv2 stride2)
#
# Skips (no equivalents in ConvNeXt):
#   - stn, stem (k2,s2 != k4,s4), se1..se4, stage4(transformer), final_ln, head
#
# Works with your exact attribute names:
#   stage1/stage2/stage3, downsample_layer_1/2/3
# ============================================================

from __future__ import annotations

from typing import Optional, Dict
import torch
import torch.nn as nn
import timm

# Import your existing implementation + sizes
from fer.models.emocatnets import EmoCatNets, EMOCATNETS_SIZES


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if sd and next(iter(sd.keys())).startswith("module."):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _remap_block_names(k: str) -> str:
    """
    timm ConvNeXt block names -> your ConvNextBlock names
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


@torch.no_grad()
def transfer_convnext_into_emocatnets(
    model: nn.Module,
    convnext_name: str = "convnext_tiny",
    pretrained: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    timm ConvNeXt -> your EmoCatNets (conv parts only)

    Maps:
      timm downsample_layers.1 -> downsample_layer_1
      timm downsample_layers.2 -> downsample_layer_2
      timm downsample_layers.3 -> downsample_layer_3

      timm stages.0 -> stage1
      timm stages.1 -> stage2
      timm stages.2 -> stage3

    Skips:
      downsample_layers.0 (stem), stages.3, norm/head
    """
    # sanity: required prefixes in destination
    dst_sd = model.state_dict()
    required_prefixes = [
        "stage1.", "stage2.", "stage3.",
        "downsample_layer_1.", "downsample_layer_2.", "downsample_layer_3."
    ]
    missing = [p for p in required_prefixes if not any(k.startswith(p) for k in dst_sd.keys())]
    if missing:
        raise RuntimeError(
            f"Destination model is missing expected prefixes: {missing}. "
            f"Check attribute names in EmoCatNets."
        )

    src = timm.create_model(convnext_name, pretrained=pretrained).eval()
    src_sd = _strip_module_prefix(src.state_dict())

    updates: Dict[str, torch.Tensor] = {}
    copied = 0
    skipped_missing = 0
    skipped_shape = 0

    for k_src, v_src in src_sd.items():
        # skip ConvNeXt stem + last stage + classifier bits
        if k_src.startswith("downsample_layers.0."):
            continue
        if k_src.startswith("stages.3."):
            continue
        if k_src.startswith("head.") or k_src.startswith("norm."):
            continue

        k_dst = None

        # downsample layers mapping
        if k_src.startswith("downsample_layers.1."):
            k_dst = "downsample_layer_1." + k_src[len("downsample_layers.1."):]
        elif k_src.startswith("downsample_layers.2."):
            k_dst = "downsample_layer_2." + k_src[len("downsample_layers.2."):]
        elif k_src.startswith("downsample_layers.3."):
            k_dst = "downsample_layer_3." + k_src[len("downsample_layers.3."):]
        # stages mapping (ConvNeXt conv stages only)
        elif k_src.startswith("stages.0."):
            tail = k_src[len("stages.0."):]
            tail = tail.replace("blocks.", "")  # timm: stages.i.blocks.j.*
            k_dst = _remap_block_names("stage1." + tail)
        elif k_src.startswith("stages.1."):
            tail = k_src[len("stages.1."):]
            tail = tail.replace("blocks.", "")
            k_dst = _remap_block_names("stage2." + tail)
        elif k_src.startswith("stages.2."):
            tail = k_src[len("stages.2."):]
            tail = tail.replace("blocks.", "")
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
            f"[emocatnets transfer] convnext='{convnext_name}' pretrained={pretrained} | "
            f"copied={copied} skipped_missing={skipped_missing} skipped_shape={skipped_shape}"
        )
        for s in list(updates.keys())[:10]:
            print("  loaded:", s)

    return dict(copied=copied, skipped_missing=skipped_missing, skipped_shape=skipped_shape)


# ------------------------------------------------------------
# Factory (registry-friendly)
# ------------------------------------------------------------

_SIZE_TO_CONVNEXT = {
    "tiny":  "convnext_tiny",
    "small": "convnext_small",
    "base":  "convnext_base",
    "large": "convnext_large",
    "xlarge": "convnext_xlarge",
}


def emocatnetsfine_fer(
    *,
    size: str = "tiny",
    in_channels: int = 3,
    num_classes: int = 6,
    transfer: bool = False,
    convnext_name: Optional[str] = None,
    convnext_pretrained: bool = True,
    verbose: bool = True,
    **kwargs,
) -> EmoCatNets:
    """
    Same signature style as your registry expects:
      factory(num_classes, in_channels, transfer) -> nn.Module

    transfer=True loads timm ConvNeXt weights into stage1-3 + downsample_layer_1-3
    """
    size = size.lower()
    if size not in EMOCATNETS_SIZES:
        raise ValueError(f"Unknown size='{size}'. Valid: {list(EMOCATNETS_SIZES.keys())}")

    cfg = EMOCATNETS_SIZES[size]
    model = EmoCatNets(
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
        se_reduction=cfg.se_reduction,
        **kwargs,
    )

    if transfer:
        cn = convnext_name or _SIZE_TO_CONVNEXT.get(size, "convnext_tiny")
        transfer_convnext_into_emocatnets(
            model=model,
            convnext_name=cn,
            pretrained=convnext_pretrained,
            verbose=verbose,
        )

    return model


# ------------------------------------------------------------
# Optional: param groups helper (recommended for transfer=True)
# ------------------------------------------------------------

def make_emocatnets_param_groups(
    model: nn.Module,
    base_lr: float,
    backbone_lr_mult: float = 0.1,
) -> list[dict]:
    """
    backbone (loaded): stage1-3 + downsample_layer_1-3
    new modules: stn, stem, se*, stage4, final_ln, head
    """
    backbone, rest = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(("stage1", "stage2", "stage3", "downsample_layer_1", "downsample_layer_2", "downsample_layer_3")):
            backbone.append(p)
        else:
            rest.append(p)

    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": base_lr * backbone_lr_mult})
    if rest:
        groups.append({"params": rest, "lr": base_lr})
    return groups
