import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerActivationXAI:

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        aggregation: str = "mean",
    ):
        self.model = model.eval()
        self.aggregation = aggregation

        if target_layer is None:
            target_layer = self._find_last_conv_layer(model)

        self.target_layer = target_layer
        self.activations = None

        self._register_hook()

    # ---------------------------------------------------------
    # Hook handling
    # ---------------------------------------------------------
    def _register_hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        self.hook = self.target_layer.register_forward_hook(forward_hook)

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    # ---------------------------------------------------------
    # Automatic layer discovery (model-agnostic)
    # ---------------------------------------------------------
    @staticmethod
    def _find_last_conv_layer(model: nn.Module) -> nn.Module:
        """
        Finds the last Conv2d layer in a model.
        Works for ConvNeXt, MobileNet, EfficientNet, CoAtNet (CNN stages).
        """
        for layer in reversed(list(model.modules())):
            if isinstance(layer, nn.Conv2d):
                return layer
        raise ValueError("No Conv2d layer found in the model.")

    # ---------------------------------------------------------
    # Forward + activation processing
    # ---------------------------------------------------------
    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        
        _ = self.model(x)

        if self.activations is None:
            raise RuntimeError("Forward hook did not capture activations.")

        act = self.activations  # (B, C, H', W')

        if self.aggregation == "mean":
            act = act.mean(dim=1)
        elif self.aggregation == "sum":
            act = act.sum(dim=1)
        elif self.aggregation == "max":
            act = act.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        act = act - act.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        act = act / (act.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)

        act = F.interpolate(
            act.unsqueeze(1),
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return act