from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class WarmupThenCosine:
    """
    Epoch-based schedule:
      - Linear warmup from 0 -> base_lr over `warmup_epochs`
      - Cosine decay from base_lr -> min_lr over remaining epochs

    Notes:
      - Call .step() once per epoch (typically at end of epoch).
      - Supports checkpoint resume via state_dict/load_state_dict.
      - Optionally preserves param-group LR ratios if you pass base_lrs.
    """

    def __init__(
        self,
        optimizer,
        base_lr: float,
        min_lr: float,
        warmup_epochs: int,
        total_epochs: int,
        *,
        base_lrs: Optional[List[float]] = None,
        last_epoch: int = 0,
    ):
        self.opt = optimizer
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup = max(int(warmup_epochs), 0)
        self.total = max(int(total_epochs), 1)

        # Epoch counter (1-based in step, but we store as int >=0)
        self.epoch = int(last_epoch)

        # If base_lrs not given, assume all groups share base_lr
        if base_lrs is None:
            self.base_lrs = [self.base_lr for _ in optimizer.param_groups]
        else:
            if len(base_lrs) != len(optimizer.param_groups):
                raise ValueError("base_lrs length must match optimizer.param_groups")
            self.base_lrs = [float(x) for x in base_lrs]

        # Apply current lr immediately if resuming
        if self.epoch > 0:
            self._apply_lr(self._lr_at_epoch(self.epoch))

    def _lr_at_epoch(self, e: int) -> float:
        """
        Returns the scalar LR multiplier schedule for epoch e (1..total).
        """
        e = max(int(e), 1)

        if self.warmup > 0 and e <= self.warmup:
            # linear warmup: 0 -> base
            return self.base_lr * (e / self.warmup)

        # cosine phase
        denom = max(1, (self.total - self.warmup))
        t = (e - self.warmup) / denom
        # clamp to [0,1]
        t = max(0.0, min(1.0, float(t)))

        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * t))

    def _apply_lr(self, lr_scalar: float) -> None:
        """
        Apply lr_scalar to all param groups while preserving ratios via base_lrs.
        If base_lrs are all equal, this is equivalent to setting the same lr.
        """
        # lr_scalar here is "absolute lr" when base_lrs == base_lr.
        # For ratio-preserving groups, scale each group relative to base_lr.
        for pg, blr in zip(self.opt.param_groups, self.base_lrs):
            # preserve group ratio: blr/base_lr
            ratio = (blr / self.base_lr) if self.base_lr != 0 else 1.0
            pg["lr"] = float(lr_scalar * ratio)

    def step(self) -> float:
        """
        Advance by 1 epoch and set optimizer LR.
        Returns the new LR of group 0.
        """
        self.epoch += 1
        lr = self._lr_at_epoch(self.epoch)
        self._apply_lr(lr)
        return self.lr

    def set_epoch(self, epoch: int) -> None:
        """
        Set epoch counter (useful for resume) and apply LR accordingly.
        """
        self.epoch = max(int(epoch), 0)
        if self.epoch > 0:
            lr = self._lr_at_epoch(self.epoch)
            self._apply_lr(lr)

    @property
    def lr(self) -> float:
        return float(self.opt.param_groups[0]["lr"])

    def state_dict(self) -> Dict[str, Any]:
        return {
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup": self.warmup,
            "total": self.total,
            "epoch": self.epoch,
            "base_lrs": list(self.base_lrs),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.base_lr = float(state["base_lr"])
        self.min_lr = float(state["min_lr"])
        self.warmup = int(state["warmup"])
        self.total = int(state["total"])
        self.epoch = int(state["epoch"])
        self.base_lrs = [float(x) for x in state.get("base_lrs", self.base_lrs)]
        
        # Apply LR at loaded epoch
        if self.epoch > 0:
            lr = self._lr_at_epoch(self.epoch)
            self._apply_lr(lr)
