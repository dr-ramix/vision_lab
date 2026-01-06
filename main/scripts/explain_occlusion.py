import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms

from fer.models.cnn_resnet18 import ResNet18FER
from fer.xai.occlusion import occlusion_saliency

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

heatmap = occlusion_saliency(
    model=model,
    input_tensor=x,
    target_class=target_idx,
)

heatmap_resized = cv2.resize(heatmap, (64, 64))

img = x[0].permute(1, 2, 0).detach().cpu().numpy()
img = (img - img.min()) / (img.max() - img.min() + 1e-8)

heatmap_color = cv2.applyColorMap(
    np.uint8(255 * heatmap_resized),
    cv2.COLORMAP_JET
)
heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
heatmap_color = heatmap_color / 255.0

overlay = 0.6 * img + 0.4 * heatmap_color
overlay = np.clip(overlay, 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img)
axes[0].set_title("Input Image")
axes[0].axis("off")

axes[1].imshow(heatmap_resized, cmap="jet")
axes[1].set_title("Occlusion Heatmap")
axes[1].axis("off")

axes[2].imshow(overlay)
axes[2].set_title("Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close()

print(f"Occlusion result saved to: {OUTPUT_PATH}")