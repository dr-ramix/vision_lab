from __future__ import annotations
from pathlib import Path
import json
import torch


class _BaseFERTrained:
    """
    Base class for trained FER models (inference only).
    """

    RUN_DIR: Path  # must be overridden

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        run_dir = Path(self.RUN_DIR)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Load TorchScript model
        self.model = torch.jit.load(
            run_dir / "exports" / "model_frozen.ts",
            map_location=self.device,
        )
        self.model.eval()

        # Load class labels
        with open(run_dir / "mappings" / "class_order.json", "r") as f:
            self.class_order = json.load(f)["class_order"]

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """
        x: Tensor [B, 3, H, W], normalized
        """
        logits = self.model(x.to(self.device))
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        return {
            "pred_idx": preds.cpu().tolist(),
            "pred_label": [self.class_order[i] for i in preds.cpu().tolist()],
            "probs": probs.cpu().tolist(),
        }
        
    @property
    def accuracy(self) -> float:
        """
        Test accuracy of this trained model (from training artifacts).
        """
        with open(self.RUN_DIR / "metrics" / "final_summary.json", "r") as f:
            return float(json.load(f)["test_acc"])


class ResNet18FER_Trained(_BaseFERTrained):
    """
    ResNet18 FER model trained on <date / dataset>.
    """

    RUN_DIR = (
        Path(__file__).resolve().parents[3]
        / "training_output"
        / "runs"
        / "2025-01-10_14-32-01__resnet18__user-alice__a3f9c2"
    )


class CNNVanillaFER_Trained(_BaseFERTrained):
    RUN_DIR = (
        Path(__file__).resolve().parents[3]
        / "training_output"
        / "runs"
        / "2025-01-11_09-18-44__cnn_vanilla__user-bob__91ac02"
    )