import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms

from fer.models.cnn_resnet18 import ResNet18FER
from fer.xai.occlusion import OcclusionSaliency

DATASET_ROOT = "../src/fer/dataset/standardized/images_mtcnn_cropped_norm/test"
WEIGHTS_PATH = "../weights/resnet18fer/model_state_dict.pt"
OUTPUT_PATH = "../xai_results/occlusion/occlusion.png"

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = ImageFolder(DATASET_ROOT, transform=transform)

target_class = "anger"
target_idx = dataset.class_to_idx[target_class]

x = None
for img, lbl in dataset:
    if lbl == target_idx:
        x = img.unsqueeze(0)
        break

if x is None:
    raise RuntimeError(f"No image found for class '{target_class}'")

x = x.to(device)

model = ResNet18FER(num_classes=6)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.to(device)
model.eval()

occlusion = OcclusionSaliency(
    model=model,
    window_size=(8, 8),     
    stride=(4, 4),
    occlusion_value=0.0,    
    batch_size=32,
)

heatmap = occlusion(
    input_tensor=x,
    target_class=target_idx
)

heatmap_resized = cv2.resize(heatmap, (64, 64))

importance = heatmap_resized

threshold = np.percentile(importance, 80)
mask = importance >= threshold

img = x[0].permute(1, 2, 0).detach().cpu().numpy()
img = (img - img.min()) / (img.max() - img.min() + 1e-8)

masked_img = img.copy()
masked_img[mask] = img.mean()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img)
axes[0].set_title("Input Image")
axes[0].axis("off")

axes[1].imshow(importance, cmap="RdBu")
axes[1].set_title("Occlusion Importance")
axes[1].axis("off")

axes[2].imshow(masked_img)
axes[2].set_title("Image w/ Important Regions Removed")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close()

print(f"Occlusion result saved to: {OUTPUT_PATH}")