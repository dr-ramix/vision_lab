import torch
import torch.nn.functional as F
import numpy as np


class OcclusionSaliency:
    def __init__(
        self,
        model: torch.nn.Module,
        window_size =(16, 16),
        stride = (8, 8),
        occlusion_value: float = 0.0,
        batch_size: int = 32,
        device = None,
    ):
        self.model = model.eval()
        self.window_h, self.window_w = window_size
        self.stride_h, self.stride_w = stride
        self.occlusion_value = occlusion_value
        self.batch_size = batch_size
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class = None,
        normalize: bool = True,
    ):
        input_tensor = input_tensor.to(self.device)
        _, C, H, W = input_tensor.shape

        logits = self.model(input_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        baseline_score = logits[0, target_class].item()

        saliency = torch.zeros((H, W), device=self.device)
        count_map = torch.zeros((H, W), device=self.device)

        occluded_batch = []
        coords = []

        for y in range(0, H - self.window_h + 1, self.stride_h):
            for x in range(0, W - self.window_w + 1, self.stride_w):

                occluded = input_tensor.clone()
                occluded[:, :, y:y + self.window_h, x:x + self.window_w] = self.occlusion_value

                occluded_batch.append(occluded)
                coords.append((y, x))

                if len(occluded_batch) == self.batch_size:
                    self._process_batch(
                        occluded_batch,
                        coords,
                        saliency,
                        count_map,
                        baseline_score,
                        target_class,
                    )
                    occluded_batch, coords = [], []

        if occluded_batch:
            self._process_batch(
                occluded_batch,
                coords,
                saliency,
                count_map,
                baseline_score,
                target_class,
            )

        saliency /= count_map.clamp(min=1)

        saliency = saliency.clamp(min=0)

        if normalize:
            saliency -= saliency.min()
            saliency /= saliency.max() + 1e-8

        return saliency.detach().cpu().numpy()

    def _process_batch(
        self,
        batch,
        coords,
        saliency,
        count_map,
        baseline_score,
        target_class,
    ):
        batch_tensor = torch.cat(batch, dim=0)
        scores = self.model(batch_tensor)[:, target_class]

        deltas = baseline_score - scores

        for delta, (y, x) in zip(deltas, coords):
            saliency[y:y + self.window_h, x:x + self.window_w] += delta
            count_map[y:y + self.window_h, x:x + self.window_w] += 1