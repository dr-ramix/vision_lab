# main/scripts/verify_weights.py
from __future__ import annotations

"""
Verify that your pretrained weights are:
  1) present locally under fer/inference/weights/<weights_id>/
  2) have required files (config.json + weights)
  3) can instantiate the architecture via fer.models.registry.make_model(arch_name, ...)
  4) can load the checkpoint (strict or non-strict)

This catches:
  - missing folders / missing files
  - wrong weights_id mapping
  - wrong arch_name mapping (architecture mismatch)
  - wrong num_classes / in_channels / transfer in config.json
  - checkpoint prefix issues ("model.", "module.")
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from fer.inference.hub import resolve_local
from fer.inference.models import SPECS, InferenceSpec
from fer.models.registry import make_model

try:
    from safetensors.torch import load_file as safetensors_load_file
except Exception:
    safetensors_load_file = None


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    class_name: str
    weights_id: str
    arch_name: str
    message: str


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def read_config(path: Path) -> Dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix):]: v for k, v in sd.items()}

def load_state_dict(weights_path: Path) -> Dict[str, torch.Tensor]:
    """
    Supports:
      - model.safetensors
      - model_state_dict.pt (or other .pt name)
    """
    if weights_path.suffix == ".safetensors":
        if safetensors_load_file is None:
            raise RuntimeError(
                "safetensors not installed but found a .safetensors file. "
                "Install with: pip install safetensors"
            )
        return safetensors_load_file(str(weights_path))

    obj = torch.load(weights_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {weights_path}")
    return obj


def final_args_from_config(spec: InferenceSpec, cfg: Dict) -> Tuple[int, int, bool]:
    """
    Determine (num_classes, in_channels, transfer) based on:
      - spec defaults
      - config.json overrides
    (No CLI overrides here; verifier checks what's on disk.)
    """
    num_classes = int(cfg.get("num_classes", spec.default_num_classes))
    in_channels = int(cfg.get("in_channels", spec.default_in_channels))
    transfer = bool(cfg.get("transfer", spec.default_transfer))
    return num_classes, in_channels, transfer


# -----------------------------------------------------------------------------
# Core verify
# -----------------------------------------------------------------------------

def verify_one(spec: InferenceSpec, *, strict: bool) -> VerifyResult:
    try:
        resolved = resolve_local(spec.weights_id)  # local/manual only
        cfg = read_config(resolved.config_path)

        num_classes, in_channels, transfer = final_args_from_config(spec, cfg)

        model = make_model(
            spec.arch_name,
            num_classes=num_classes,
            in_channels=in_channels,
            transfer=transfer,
        )

        sd = load_state_dict(resolved.weights_path)
        sd = strip_prefix(sd, "model.")
        sd = strip_prefix(sd, "module.")

        missing, unexpected = model.load_state_dict(sd, strict=strict)

        if strict and (missing or unexpected):
            return VerifyResult(
                ok=False,
                class_name=spec.class_name,
                weights_id=spec.weights_id,
                arch_name=spec.arch_name,
                message=(
                    f"strict mismatch: missing={len(missing)} unexpected={len(unexpected)} "
                    f"(try --non-strict to inspect further)"
                ),
            )

        return VerifyResult(
            ok=True,
            class_name=spec.class_name,
            weights_id=spec.weights_id,
            arch_name=spec.arch_name,
            message=(
                f"OK ({resolved.format}) | "
                f"num_classes={num_classes}, in_channels={in_channels}, transfer={transfer} | strict={strict}"
            ),
        )

    except Exception as e:
        return VerifyResult(
            ok=False,
            class_name=spec.class_name,
            weights_id=spec.weights_id,
            arch_name=spec.arch_name,
            message=str(e),
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Verify local fer/inference/weights/* against model registry")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--model", help="Verify only one import name, e.g. ResNet50, ConvNeXtBase")
    group.add_argument("--weights-id", help="Verify only one weights folder, e.g. resnet50, convnext_base")

    ap.add_argument(
        "--non-strict",
        action="store_true",
        help="Use strict=False when loading state_dict (useful for debugging mismatches)",
    )
    args = ap.parse_args()

    strict = not args.non_strict

    specs = SPECS
    if args.model:
        specs = [s for s in SPECS if s.class_name.lower() == args.model.lower()]
        if not specs:
            print(f"Unknown model class '{args.model}'. Known: {[s.class_name for s in SPECS]}", file=sys.stderr)
            return 2
    if args.weights_id:
        specs = [s for s in specs if s.weights_id.lower() == args.weights_id.lower()]
        if not specs:
            print(f"No spec uses weights_id='{args.weights_id}'.", file=sys.stderr)
            return 2

    ok = 0
    fail = 0

    for spec in specs:
        r = verify_one(spec, strict=strict)
        if r.ok:
            print(f"[OK]   {r.class_name:20s}  weights_id={r.weights_id:18s}  arch={r.arch_name:16s}  {r.message}")
            ok += 1
        else:
            print(f"[FAIL] {r.class_name:20s}  weights_id={r.weights_id:18s}  arch={r.arch_name:16s}  {r.message}", file=sys.stderr)
            fail += 1

    print(f"\nSummary: OK={ok}, FAIL={fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
