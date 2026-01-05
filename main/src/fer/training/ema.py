from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class EMASwap:
    """
    Returned by EMA.apply_to(model). Call .restore(model) after evaluation.
    """
    backup_state: Dict[str, torch.Tensor]

    def restore(self, model: nn.Module) -> None:
        model.load_state_dict(self.backup_state, strict=True)


class EMA:
    """
    Exponential Moving Average of model parameters (and buffers).

    Features:
      - Keeps a frozen shadow model (same architecture).
      - Updates shadow with EMA of state_dict (params + buffers).
      - Can swap EMA weights into a live model for evaluation, then restore.
      - Supports saving/loading EMA state.

    Notes:
      - If you use AMP / DDP, call ema.update(model) AFTER optimizer.step().
      - Works with BatchNorm buffers too because we update the full state_dict.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
        update_buffers: bool = True,
    ):
        self.decay = float(decay)
        self.update_buffers = bool(update_buffers)

        # Deepcopy -> same module structure
        self.shadow = copy.deepcopy(model).eval()

        # Optionally move EMA weights to CPU to save VRAM
        if device is not None:
            self.shadow.to(device)

        for p in self.shadow.parameters():
            p.requires_grad_(False)

        # Track if we ever updated (useful for debugging)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update EMA weights from `model`.

        Uses:
          shadow = decay*shadow + (1-decay)*model
        """
        d = self.decay
        msd = model.state_dict()
        ssd = self.shadow.state_dict()

        # Update params and (optionally) buffers
        for k, sv in ssd.items():
            mv = msd.get(k, None)
            if mv is None:
                continue

            # Some buffers are integer (e.g. BN num_batches_tracked)
            if (not self.update_buffers) and (k not in dict(model.named_parameters())):
                # skip buffers if requested
                continue

            if torch.is_floating_point(sv) and torch.is_floating_point(mv):
                sv.copy_(sv * d + mv.detach() * (1.0 - d))
            else:
                # For non-float tensors: copy directly (e.g. int buffers)
                sv.copy_(mv)

        self.num_updates += 1

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Save EMA state (shadow model weights + meta).
        """
        return {
            "decay": torch.tensor(self.decay),
            "num_updates": torch.tensor(self.num_updates),
            **{f"shadow.{k}": v.clone().detach().cpu() for k, v in self.shadow.state_dict().items()},
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor], strict: bool = True) -> None:
        """
        Load EMA state saved by state_dict().
        """
        if "decay" in state:
            self.decay = float(state["decay"].item())
        if "num_updates" in state:
            self.num_updates = int(state["num_updates"].item())

        shadow_state = {}
        for k, v in state.items():
            if k.startswith("shadow."):
                shadow_state[k[len("shadow."):]] = v
        self.shadow.load_state_dict(shadow_state, strict=strict)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> EMASwap:
        """
        Replace `model` weights with EMA weights. Returns an EMASwap you can use to restore.

        Example:
            swap = ema.apply_to(model)
            val = evaluate(model)
            swap.restore(model)
        """
        backup = {k: v.clone().detach() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow.state_dict(), strict=True)
        return EMASwap(backup_state=backup)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """
        Copy EMA weights into model (no restore backup).
        Useful when you want to keep EMA permanently at the end.
        """
        model.load_state_dict(self.shadow.state_dict(), strict=True)

    def to(self, device: torch.device) -> "EMA":
        """
        Move EMA shadow model to device.
        """
        self.shadow.to(device)
        return self
