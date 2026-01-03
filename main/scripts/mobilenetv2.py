# ----------------------------
# MobileNetV2 Modellsetup für Input mit 64x64
# ----------------------------

import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.transforms import Resize

# ----------------------------
# Konfiguration
# ----------------------------
cfg.image_size = 64 # Hier wird die Inputgröße auf 64x64 gesetzt

# Sicherheitscheck für Klassen
if cfg.num_classes != 6:
    raise ValueError("Your dataloader enforces exactly 6 classes (CLASS_ORDER).")

# ----------------------------
# Laden von MobileNetV2
# ----------------------------
weights = MobileNet_V2_Weights.IMAGENET1K_V1  # vortrainierte ImageNet-Gewichte
model = mobilenet_v2(weights=weights)

# ----------------------------
# Anpassen der Classifier
# ----------------------------
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, cfg.num_classes)

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
