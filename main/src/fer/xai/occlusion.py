import torch
import numpy as np

def occlusion_saliency(
    model,
    input_tensor,
    target_class,
    sliding_window_shapes=(16, 16),
    stride=(8, 8),
    occlusion_value=0.0,
):
    model.eval()

    _, C, H, W = input_tensor.shape
    window_h, window_w = sliding_window_shapes
    stride_h, stride_w = stride

    #prediction
    baseline_score = model(input_tensor)[0, target_class].item()

    saliency = torch.zeros((H, W), device=input_tensor.device)

    for y in range(0, H - window_h + 1, stride_h):
        for x in range(0, W - window_w + 1, stride_w):

            occluded = input_tensor.clone()

            occluded[
                :, :, y:y + window_h, x:x + window_w
            ] = occlusion_value

            score = model(occluded)[0, target_class].item()
            delta = baseline_score - score

            saliency[y:y + window_h, x:x + window_w] += delta

    saliency = saliency.clamp(min=0)
    saliency = saliency.cpu().numpy()

    saliency -= saliency.min()
    saliency /= saliency.max() + 1e-8

    return saliency