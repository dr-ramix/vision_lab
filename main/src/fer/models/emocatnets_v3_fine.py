# fer/models/emocatnets_v3_fine.py
from __future__ import annotations

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import timm

# Import your v3 model + sizes
from fer.models.emocatnets_v3 import EmoCatNetsV3, EMOCATNETS_V3_SIZES


# ------------------------------------------------------------
# ConvNeXt -> EmoCatNetsV3 weight transfer
# ------------------------------------------------------------

def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if sd and next(iter(sd.keys())).startswith("module."):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _remap_block_names(dst_key: str) -> str:
    """
    timm ConvNeXt(/V2) block names -> your ConvNextBlockV2 names
      dwconv  -> depthwise_conv
      norm    -> layer_norm
      pwconv1 -> pointwise_conv1
      pwconv2 -> pointwise_conv2

    Notes:
      - GRN keys (".grn.gamma", ".grn.beta") already match your GRN module naming.
      - LayerScale param is usually called ".gamma" in both.
    """
    dst_key = dst_key.replace(".dwconv.", ".depthwise_conv.")
    dst_key = dst_key.replace(".norm.", ".layer_norm.")
    dst_key = dst_key.replace(".pwconv1.", ".pointwise_conv1.")
    dst_key = dst_key.replace(".pwconv2.", ".pointwise_conv2.")
    return dst_key


@torch.no_grad()
def transfer_convnext_into_emocatnetsv3(
    model: nn.Module,
    convnext_name: str = "convnextv2_tiny",
    pretrained: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Load timm ConvNeXt(/ConvNeXtV2) pretrained weights into compatible pieces of EmoCatNetsV3:
      - stage1 <- timm stages.0
      - stage2 <- timm stages.1
      - stage3 <- timm stages.2
      - down1  <- timm downsample_layers.1
      - down2  <- timm downsample_layers.2

    Skips:
      - stem / stage4 / head / final norm
      - all non-ConvNeXt modules (stn, cbam*, transformer, etc.)

    Works with both:
      - convnext_*     (v1, no GRN)  -> loads DW/LN/PW + gamma where possible
      - convnextv2_*   (v2, GRN)     -> loads DW/LN/PW + GRN + gamma where possible
    """
    src = timm.create_model(convnext_name, pretrained=pretrained).eval()
    src_sd = _strip_module_prefix(src.state_dict())
    dst_sd = model.state_dict()

    # Ensure expected prefixes exist in the destination model
    required = ["stage1.", "stage2.", "stage3.", "down1.", "down2."]
    missing_prefixes = [p for p in required if not any(k.startswith(p) for k in dst_sd.keys())]
    if missing_prefixes:
        raise RuntimeError(
            f"Destination model state_dict missing expected prefixes: {missing_prefixes}. "
            f"Check your EmoCatNetsV3 attribute names."
        )

    copied = 0
    skipped_missing = 0
    skipped_shape = 0

    updates: Dict[str, torch.Tensor] = {}

    for k_src, v_src in src_sd.items():
        # Skip convnext stem (downsample_layers.0), stage4, head/norm
        if k_src.startswith("downsample_layers.0."):
            continue
        if k_src.startswith("stages.3."):
            continue
        if k_src.startswith("head.") or k_src.startswith("norm."):
            continue

        k_dst: Optional[str] = None

        # downsample_layers.1 -> down1
        if k_src.startswith("downsample_layers.1."):
            k_dst = "down1." + k_src[len("downsample_layers.1."):]
        # downsample_layers.2 -> down2
        elif k_src.startswith("downsample_layers.2."):
            k_dst = "down2." + k_src[len("downsample_layers.2."):]
        # stages.0 -> stage1
        elif k_src.startswith("stages.0."):
            tail = k_src[len("stages.0."):]
            tail = tail.replace("blocks.", "")  # timm uses stages.i.blocks.j.*
            k_dst = _remap_block_names("stage1." + tail)
        # stages.1 -> stage2
        elif k_src.startswith("stages.1."):
            tail = k_src[len("stages.1."):]
            tail = tail.replace("blocks.", "")
            k_dst = _remap_block_names("stage2." + tail)
        # stages.2 -> stage3
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
            f"[emocatnetsv3 transfer] convnext='{convnext_name}' pretrained={pretrained} | "
            f"copied={copied} skipped_missing={skipped_missing} skipped_shape={skipped_shape}"
        )
        for s in list(updates.keys())[:10]:
            print("  loaded:", s)

    return dict(copied=copied, skipped_missing=skipped_missing, skipped_shape=skipped_shape)


# ------------------------------------------------------------
# Factory (registry-friendly)
# ------------------------------------------------------------

# IMPORTANT:
# Your blocks are ConvNeXtV2-style (GRN), so convnextv2_* is the best default match.
_SIZE_TO_CONVNEXT: Dict[str, str] = {
    "tiny":  "convnextv2_tiny",
    "small": "convnextv2_small",
    "base":  "convnextv2_base",
    "large": "convnextv2_large",
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
    Factory for fine-tuning.

    transfer=True:
      loads ConvNeXt(/V2) pretrained weights into:
        stage1-3 + down1-2
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
        cn = convnext_name or _SIZE_TO_CONVNEXT[size]
        transfer_convnext_into_emocatnetsv3(
            model=model,
            convnext_name=cn,
            pretrained=convnext_pretrained,
            verbose=verbose,
        )

    return model


# ------------------------------------------------------------
# Optional: param groups helper (best practice)
# ------------------------------------------------------------

def make_emocatnetsv3_param_groups(
    model: nn.Module,
    base_lr: float,
    backbone_lr_mult: float = 0.1,
) -> list[dict]:
    """
    backbone (loaded): stage1-3 + down1-2
    rest (new or not loaded): stn, stem, cbam*, down3, stage4, head, etc.
    """
    backbone, rest = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(("stage1", "stage2", "stage3", "down1", "down2")):
            backbone.append(p)
        else:
            rest.append(p)

    groups: list[dict] = []
    if backbone:
        groups.append({"params": backbone, "lr": base_lr * backbone_lr_mult})
    if rest:
        groups.append({"params": rest, "lr": base_lr})
    return groups
