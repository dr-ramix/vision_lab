# ----------------------------
# MobileNetV3 Large Modellsetup für Input mit 64x64
# ----------------------------

import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.transforms import Resize

# ----------------------------
# Konfiguration
# ----------------------------
image_size = 64
num_classes = 6

# Sicherheitscheck
if num_classes != 6:
    raise ValueError("Your dataloader enforces exactly 6 classes (CLASS_ORDER).")

# ----------------------------
# Laden von MobileNetV3 Large
# ----------------------------
weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1  # vortrainierte ImageNet-Gewichte
model = mobilenet_v3_large(weights=weights)

# ----------------------------
# Anpassen der Classifier
# ----------------------------
in_features = model.classifier[3].in_features  # bei MobileNetV3 Large ist der Linear Layer an Index 3
model.classifier[3] = nn.Linear(in_features, cfg.num_classes)

# ----------------------------
# Backbone einfrieren
# ----------------------------
for p in model.features.parameters():
    p.requires_grad = False

# ----------------------------
# Transform für 64x64 Input
# ----------------------------
# Für Bilder, die nicht automatisch auf 64x64 skaliert werden:
resize_transform = Resize((cfg.image_size, cfg.image_size))
