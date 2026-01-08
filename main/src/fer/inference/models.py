# main/src/fer/inference/models.py
from __future__ import annotations

"""
Importable pretrained inference models.

Goal:
  from fer.inference.models import ResNet50, CoAtNetCCCTTiny
  model = ResNet50().load(device="cuda")  # default: uses local/manual weights only

Terminology used everywhere:

- class_name:
    Python import symbol (e.g. "ResNet50")

- weights_id:
    Folder name containing weights files:
      HF:    drRamix/EMO_NETS_LMU/<weights_id>/*
      Local: main/src/fer/inference/weights/<weights_id>/*
    Contains:
      config.json (can be empty)
      model.safetensors (preferred) OR model_state_dict.pt
    NOTE: Files may exist but be 0 bytes (placeholders). We handle that.

- arch_name:
    Architecture key in your TRAINING registry:
      from fer.models.registry import make_model
      model = make_model(arch_name, num_classes=..., in_channels=..., transfer=...)

Config and overrides:

- config.json may contain: num_classes, in_channels, transfer
- If config.json is empty/missing fields -> defaults from InferenceSpec
- load(...) can override num_classes/in_channels/transfer explicitly
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type

import torch

from fer.models.registry import make_model
from fer.inference.hub import ensure_weights, ResolvedWeights

try:
    from safetensors.torch import load_file as safetensors_load_file
except Exception:
    safetensors_load_file = None


# -----------------------------------------------------------------------------
# Spec
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class InferenceSpec:
    class_name: str   # import name
    weights_id: str   # HF/local folder containing weights+config
    arch_name: str    # key for fer.models.registry.make_model(...)

    # defaults (config.json can override)
    default_num_classes: int = 7
    default_in_channels: int = 3
    default_transfer: bool = False


# -----------------------------------------------------------------------------
# Base class
# -----------------------------------------------------------------------------

class InferenceModelBase:
    """
    Resolves weights, builds architecture via training registry, loads checkpoint.
    """

    spec: InferenceSpec

    def load(
        self,
        *,
        # ---- weights resolution ----
        source: str = "local",  # "local" | "project" | "cache"
        repo_id: str = "drRamix/EMO_NETS_LMU",
        revision: str = "main",
        force_download: bool = False,

        # ---- model construction args (to make_model) ----
        num_classes: Optional[int] = None,
        in_channels: Optional[int] = None,
        transfer: Optional[bool] = None,

        # ---- checkpoint loading ----
        strict: bool = True,

        # ---- device / mode ----
        device: str | torch.device = "cpu",
        eval_mode: bool = True,
    ) -> torch.nn.Module:
        """
        Returns a torch.nn.Module loaded with pretrained weights.

        source:
          - "local":   ONLY load from fer/inference/weights/<weights_id> (no downloads)
          - "project": if missing locally, download into fer/inference/weights/<weights_id>
          - "cache":   use Hugging Face cache only (no project files)
        """
        resolved = ensure_weights(
            self.spec.weights_id,
            repo_id=repo_id,
            revision=revision,
            source=source,
            force=force_download,
        )

        cfg = _read_json_safe(resolved.config_path)

        # config -> defaults
        nc = int(cfg.get("num_classes", self.spec.default_num_classes))
        ic = int(cfg.get("in_channels", self.spec.default_in_channels))
        tr = bool(cfg.get("transfer", self.spec.default_transfer))

        # user overrides -> final
        if num_classes is not None:
            nc = int(num_classes)
        if in_channels is not None:
            ic = int(in_channels)
        if transfer is not None:
            tr = bool(transfer)

        # build model via training registry
        model = make_model(
            self.spec.arch_name,
            num_classes=nc,
            in_channels=ic,
            transfer=tr,
        )

        # load state dict (robust to empty placeholders)
        sd = _load_state_dict_robust(resolved)
        sd = _strip_prefix(sd, "model.")
        sd = _strip_prefix(sd, "module.")

        try:
            missing, unexpected = model.load_state_dict(sd, strict=strict)
        except RuntimeError as e:
            raise RuntimeError(
                f"[{self.spec.class_name}] weights mismatch\n"
                f"  weights_id: {self.spec.weights_id}\n"
                f"  arch_name:  {self.spec.arch_name}\n"
                f"  args: num_classes={nc}, in_channels={ic}, transfer={tr}\n"
                f"  config: {resolved.config_path}\n"
                f"  weights:{resolved.weights_path}\n"
                f"Original error: {e}"
            ) from e

        if strict and (missing or unexpected):
            raise RuntimeError(
                f"[{self.spec.class_name}] state_dict mismatch (strict=True)\n"
                f"Missing keys: {missing[:25]}{' ...' if len(missing) > 25 else ''}\n"
                f"Unexpected keys: {unexpected[:25]}{' ...' if len(unexpected) > 25 else ''}"
            )

        model.to(device)
        if eval_mode:
            model.eval()
        return model


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _read_json_safe(p: Path) -> Dict[str, Any]:
    """
    Allows empty config.json (0 bytes) and returns {}.
    """
    try:
        txt = p.read_text(encoding="utf-8").strip()
        if not txt:
            return {}
        return json.loads(txt)
    except Exception:
        return {}

def _is_nonempty_file(p: Path) -> bool:
    try:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False

def _load_state_dict_pt(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {path}")
    return obj

def _load_state_dict_safetensors(path: Path) -> Dict[str, torch.Tensor]:
    if safetensors_load_file is None:
        raise RuntimeError(
            f"safetensors not installed but found {path}. Install with: pip install safetensors"
        )
    return safetensors_load_file(str(path))

def _load_state_dict_robust(r: ResolvedWeights) -> Dict[str, torch.Tensor]:
    """
    Robust to the common situation:
      - model.safetensors exists but is 0 bytes (placeholder)
      - model_state_dict.pt is real

    We try:
      1) If r.format == "safetensors" and file is non-empty -> safetensors load
         otherwise fallback to pt if present.
      2) If r.format == "pt" -> pt load
      3) As a last resort, look for the other file in the same folder.
    """
    folder = r.folder
    st = folder / "model.safetensors"
    pt = folder / "model_state_dict.pt"

    if r.format == "safetensors":
        if _is_nonempty_file(r.weights_path):
            return _load_state_dict_safetensors(r.weights_path)
        # fallback
        if _is_nonempty_file(pt):
            return _load_state_dict_pt(pt)
        raise RuntimeError(
            f"Selected safetensors file is empty/invalid: {r.weights_path} "
            f"and no valid {pt.name} found."
        )

    # r.format == "pt"
    if _is_nonempty_file(r.weights_path):
        return _load_state_dict_pt(r.weights_path)

    # last resort: try the other format if it exists
    if _is_nonempty_file(st):
        return _load_state_dict_safetensors(st)
    if _is_nonempty_file(pt):
        return _load_state_dict_pt(pt)

    raise RuntimeError(
        f"No valid (non-empty) checkpoint found in {folder}. "
        f"Checked: {st.name}, {pt.name}"
    )

def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix):]: v for k, v in sd.items()}


# -----------------------------------------------------------------------------
# Registration table
# -----------------------------------------------------------------------------
# weights_id -> folder on HF/local
# arch_name  -> key in fer.models.registry (can be different)
# -----------------------------------------------------------------------------

SPECS = [
    # ResNets
    InferenceSpec(class_name="ResNet18", weights_id="resnet18", arch_name="resnet18"),
    InferenceSpec(class_name="ResNet50", weights_id="resnet50", arch_name="resnet50"),

    # MobileNets
    InferenceSpec(class_name="MobileNetV2",      weights_id="mobilenetsv2",       arch_name="mobilenetv2"),
    InferenceSpec(class_name="MobileNetV3Tiny",  weights_id="mobilenetsv3_tiny",  arch_name="mobilenetv3_tiny"),
    InferenceSpec(class_name="MobileNetV3Small", weights_id="mobilenetsv3_small", arch_name="mobilenetv3_small"),
    InferenceSpec(class_name="MobileNetV3Base",  weights_id="mobilenetsv3_base",  arch_name="mobilenetv3_base"),
    InferenceSpec(class_name="MobileNetV3Large", weights_id="mobilenetsv3_large", arch_name="mobilenetv3_large"),

    # ConvNeXt
    InferenceSpec(class_name="ConvNeXtTiny",  weights_id="convnext_tiny",  arch_name="convnext_tiny"),
    InferenceSpec(class_name="ConvNeXtSmall", weights_id="convnext_small", arch_name="convnext_small"),
    InferenceSpec(class_name="ConvNeXtBase",  weights_id="convnext_base",  arch_name="convnext_base"),
    InferenceSpec(class_name="ConvNeXtLarge", weights_id="convnext_large", arch_name="convnext_large"),

    # EmoNeXt
    InferenceSpec(class_name="EmoNeXtTiny",  weights_id="emonext_tiny",  arch_name="emonext_tiny"),
    InferenceSpec(class_name="EmoNeXtSmall", weights_id="emonext_small", arch_name="emonext_small"),
    InferenceSpec(class_name="EmoNeXtBase",  weights_id="emonext_base",  arch_name="emonext_base"),
    InferenceSpec(class_name="EmoNeXtLarge", weights_id="emonext_large", arch_name="emonext_large"),

    # EmoCatNets
    # FIXED: your training registry has "emocatnets_tiny" (plural). Use that.
    InferenceSpec(class_name="EmoCatNetsTiny",  weights_id="emocatnets_tiny",  arch_name="emocatnets_tiny"),
    InferenceSpec(class_name="EmoCatNetsSmall", weights_id="emocatnets_small", arch_name="emocatnets_small"),
    InferenceSpec(class_name="EmoCatNetsBase",  weights_id="emocatnets_base",  arch_name="emocatnets_base"),
    InferenceSpec(class_name="EmoCatNetsLarge", weights_id="emocatnets_large", arch_name="emocatnets_large"),

    # CoAtNet (HF folders exist on HF as coatnet_ccct_*)
    InferenceSpec(class_name="CoAtNetCCCTTiny",  weights_id="coatnet_ccct_tiny",  arch_name="coatnet_tiny"),
    InferenceSpec(class_name="CoAtNetCCCTSmall", weights_id="coatnet_ccct_small", arch_name="coatnet_tiny"),
    InferenceSpec(class_name="CoAtNetCCCTBase",  weights_id="coatnet_ccct_base",  arch_name="coatnet_tiny"),
    InferenceSpec(class_name="CoAtNetCCCTLarge", weights_id="coatnet_ccct_large", arch_name="coatnet_tiny"),
]


# -----------------------------------------------------------------------------
# Generate importable classes automatically from SPECS
# -----------------------------------------------------------------------------

def _make_class(spec: InferenceSpec) -> Type[InferenceModelBase]:
    return type(
        spec.class_name,
        (InferenceModelBase,),
        {
            "spec": spec,
            "__doc__": (
                f"{spec.class_name}\n"
                f"  weights_id: {spec.weights_id}\n"
                f"  arch_name:  {spec.arch_name}\n\n"
                f"Example:\n"
                f"  from fer.inference.models import {spec.class_name}\n"
                f"  model = {spec.class_name}().load(device='cuda', source='local')\n"
            ),
        },
    )

__all__ = ["InferenceModelBase", "InferenceSpec", "SPECS"]

for _spec in SPECS:
    _cls = _make_class(_spec)
    globals()[_spec.class_name] = _cls
    __all__.append(_spec.class_name)
