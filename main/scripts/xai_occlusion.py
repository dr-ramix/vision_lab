import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms

from fer.models.emocatnets_v2 import emocatnetsv2_fer
from fer.xai.occlusion import OcclusionSaliency


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


DATASET_ROOT = "../src/fer/dataset/standardized/only_mtcnn_cropped/grey/png/test"
WEIGHTS_PATH = "../weights/emocatnetsv2/model_state_dict_emocat_v2.pt"
OUTPUT_DIR = Path("../xai_results/occlusion").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Saving Occlusion maps to:", OUTPUT_DIR)

TOP_PERCENT = 20
MAX_IMAGES = 200

WINDOW_SIZE = (8, 8)
STRIDE = (4, 4)
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(" Occlusion Faithfulness Evaluation ")
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Top percent removed: {TOP_PERCENT}%")


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

dataset = ImageFolder(DATASET_ROOT, transform=transform)
num_classes = len(dataset.classes)

print("Classes:", dataset.classes)


per_class = MAX_IMAGES // num_classes

class_indices = defaultdict(list)
for idx, (_, label) in enumerate(dataset):
    class_indices[label].append(idx)

selected_indices = []
for label, indices in class_indices.items():
    k = min(per_class, len(indices))
    selected_indices.extend(random.sample(indices, k))

random.shuffle(selected_indices)

print(f"Selected {len(selected_indices)} images "
      f"({per_class} per class)")



model = emocatnetsv2_fer(size="tiny", in_channels=3,num_classes=num_classes)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("Model loaded.")


occlusion = OcclusionSaliency(
    model=model,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    occlusion_value=0.0,
    batch_size=BATCH_SIZE,
)


confidence_drops = []
per_class_drops = {c: [] for c in range(num_classes)}

processed = 0

for idx in tqdm(selected_indices, desc="Processing images"):
    if MAX_IMAGES and processed >= MAX_IMAGES:
        break

    x, y = dataset[idx]

    x = x.unsqueeze(0).to(DEVICE)
    y = torch.tensor([y], device=DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)
        orig_conf = probs[0, y].item()

    if orig_conf < 0.5:
        continue

    heatmap = occlusion(
        input_tensor=x,
        target_class=y.item(),
        normalize=True
    )

    if torch.is_tensor(heatmap):
        heatmap = heatmap.detach().cpu().numpy()

    if heatmap.shape[0] != 64:
        heatmap = cv2.resize(heatmap, (64, 64))

    img = x[0].detach().cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    overlay = np.clip(0.6 * img + 0.4 * heatmap_color, 0, 1)

    cls_name = dataset.classes[y.item()].replace(" ", "_")

    class_dir = OUTPUT_DIR / cls_name
    class_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].imshow(img)
    axes[0].set_title("Input")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Occlusion")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        class_dir / f"occlusion_{processed:04d}.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)

    threshold = np.percentile(heatmap, 100 - TOP_PERCENT)
    mask = heatmap >= threshold

    img = x[0].permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 0.5) + 0.5
    mean_val = img.mean()

    occluded_img = img.copy()
    occluded_img[mask] = mean_val
    occluded_img = (occluded_img - 0.5) / 0.5

    occluded_tensor = torch.from_numpy(occluded_img) \
        .permute(2, 0, 1) \
        .unsqueeze(0) \
        .float() \
        .to(DEVICE)

    with torch.no_grad():
        occ_probs = F.softmax(model(occluded_tensor), dim=1)
        occ_conf = occ_probs[0, y].item()

    drop = orig_conf - occ_conf

    confidence_drops.append(drop)
    per_class_drops[y.item()].append(drop)
    processed += 1

confidence_drops = np.array(confidence_drops)

print("\nRESULTS:")
print(f"Images evaluated: {len(confidence_drops)}")
print(f"Mean confidence drop: {confidence_drops.mean():.4f}")
print(f"Std confidence drop:  {confidence_drops.std():.4f}")

print("\nPer-class confidence drops:")
for cls_idx, drops in per_class_drops.items():
    if len(drops) == 0:
        continue
    print(
        f"{dataset.classes[cls_idx]:>10s}: "
        f"{np.mean(drops):.4f} ± {np.std(drops):.4f}"
    )