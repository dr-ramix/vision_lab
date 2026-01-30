import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms

from fer.models.cnn_resnet18 import ResNet18FER
from fer.xai.occlusion import OcclusionSaliency


# -----------------------------
# Paths
# -----------------------------
DATASET_ROOT = "../src/fer/dataset/standardized/images_mtcnn_cropped_norm/test"
WEIGHTS_PATH = "../weights/resnet18fer/model_state_dict.pt"
OUTPUT_PATH = "../xai_results/occlusion/occlusion.png"

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Transforms & dataset
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = ImageFolder(DATASET_ROOT, transform=transform)

target_class = "anger"
target_idx = dataset.class_to_idx[target_class]


# -----------------------------
# Select one image of target class
# -----------------------------
x = None
for img, lbl in dataset:
    if lbl == target_idx:
        x = img.unsqueeze(0)
        break

if x is None:
    raise RuntimeError(f"No image found for class '{target_class}'")

x = x.to(device)


# -----------------------------
# Load model
# -----------------------------
model = ResNet18FER(num_classes=6)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.to(device)
model.eval()


# -----------------------------
# Occlusion Saliency
# -----------------------------
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

heatmap = cv2.resize(heatmap, (64, 64))


# -----------------------------
# Prepare images
# -----------------------------
img = x[0].permute(1, 2, 0).detach().cpu().numpy()
img = (img - img.min()) / (img.max() - img.min() + 1e-8)

# Threshold top 20% most important regions
threshold = np.percentile(heatmap, 80)
mask = heatmap >= threshold

# Occluded image (important regions removed)
occluded_img = img.copy()
occluded_img[mask] = img.mean()

# Convert occluded image back to tensor
occluded_tensor = torch.from_numpy(occluded_img) \
    .permute(2, 0, 1) \
    .unsqueeze(0) \
    .float() \
    .to(device)


# -----------------------------
# Model predictions
# -----------------------------
with torch.no_grad():
    orig_probs = torch.softmax(model(x), dim=1)
    occ_probs = torch.softmax(model(occluded_tensor), dim=1)

orig_conf = orig_probs[0, target_idx].item()
occ_conf = occ_probs[0, target_idx].item()


# -----------------------------
# Visualization
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img)
axes[0].set_title(f"Original Image\nConf: {orig_conf:.2f}")
axes[0].axis("off")

axes[1].imshow(heatmap, cmap="RdBu")
axes[1].set_title("Occlusion Importance")
axes[1].axis("off")

axes[2].imshow(occluded_img)
axes[2].set_title(f"Occluded Image\nConf: {occ_conf:.2f}")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close()

print(f"Occlusion result saved to: {OUTPUT_PATH}")