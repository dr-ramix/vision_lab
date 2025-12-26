# - Dieses Skript erzeugt DataLoader, die Batches liefern:
#     xb = Bilder als Tensor
#     yb = Labels als Integer-Klassenindex
#
#
# Unsere Bilder liegen auf der Festplatte als PNG/JPG vor und sind dort
# normalerweise "uint8" Pixelwerte im Bereich 0..255.
#
# Wenn wir im Dataset die Transformation "transforms.ToTensor()" benutzen:
#   - wird das Bild in einen float32 Tensor umgewandelt
#   - und automatisch durch 255 geteilt
#
# Beispiel:
#   Pixelwert 0   -> 0.0
#   Pixelwert 255 -> 1.0
#   Pixelwert 128 -> 0.502...
#
# Wichtig:
# - Dein Modell bekommt also keine uint8 (0..255), sondern float32 (0..1).
#
# ------------------------------------------------------------
#  Was genau liefern train_loader / val_loader / test_loader?
# ------------------------------------------------------------
# Jeder DataLoader liefert in einer Schleife Batches:
#
#   for xb, yb in train_loader:
#       ...
#
# Dabei gilt:
#   xb: Tensor der Bilder
#       Shape: (B, C, H, W)
#         B = batch_size (z.B. 64)
#         C = channels (bei uns 3!)
#         H,W = Höhe/Breite (bei uns 64x64)
#
#       Beispiel:
#         xb.shape == (64, 3, 64, 64)
#
#       Datentyp:
#         xb.dtype == torch.float32
#
#       Wertebereich:
#         xb.min() >= 0.0 und xb.max() <= 1.0   (typischerweise)
#
#   yb: Tensor der Labels
#       Shape: (B,)
#         Beispiel: yb.shape == (64,)
#
#       Datentyp:
#         yb.dtype == torch.int64
#
#       Inhalt:
#         yb enthält KLASSENINDIZES (keine One-Hot Vektoren!)
#         Beispiel: [0, 3, 5, 1, 1, 2, ...]
#
# ------------------------------------------------------------
# Feste Label-Zuordnung (extrem wichtig!)
# ------------------------------------------------------------
# Wir erzwingen explizit diese Zuordnung:
#
#   anger     -> 0
#   disgust   -> 1
#   fear      -> 2
#   happiness -> 3
#   sadness   -> 4
#   surprise  -> 5
#
# Das bedeutet:
# - Wenn dein Modell "class 0" predicted, heißt das "anger".
# - Confusion Matrix / Accuracy etc. muss genau diese Reihenfolge benutzen.
#
# ------------------------------------------------------------
# Augmentations (NUR train, NICHT val/test)
# ------------------------------------------------------------
# Im train_loader werden zufällig (pro Bild, pro Epoch) Augmentations angewandt:
#
# - Horizontal Flip:
#     p = 0.5
#     -> 50% Wahrscheinlichkeit pro Sample im Train-Set
#
# - Gaussian Blur (leichter Blur):
#     p = 0.15
#     kernel_size = 3
#     sigma in [0.1, 1.0]
#
# - Contrast Veränderung:
#     p = 0.30
#     ColorJitter(contrast=0.25) bedeutet:
#         contrast factor wird zufällig in [0.75, 1.25] gewählt
#     -> manchmal wird Kontrast erhöht, manchmal verringert
#
# WICHTIG:
# - Diese Augmentations werden NICHT gespeichert.
# - Sie passieren "on-the-fly" bei jedem Zugriff in __getitem__.
# - val_loader und test_loader haben KEINE Random Augmentations,
#   damit Evaluation reproduzierbar bleibt.
#
# ------------------------------------------------------------
# Was muss das Modell erwarten?
# ------------------------------------------------------------
# INPUT:
# - 3 Kanäle (C=3), weil wir normiertes Graustufenbild auf 3 Kanäle gestackt speichern.
# - Größe: 64x64
# - Wertebereich: [0,1]
#
# Also: model input shape ist (B, 3, 64, 64)
#
# OUTPUT:
# - 6 Klassen (weil 6 Emotionen)
# - logits shape: (B, 6)
#
# ------------------------------------------------------------
# Welche Loss-Funktion?
# ------------------------------------------------------------
# Standard: torch.nn.CrossEntropyLoss()
#
# CrossEntropyLoss erwartet:
# - logits: (B, 6) float
# - target: (B,) int64 mit Klassenindex (0..5)
#
# NICHT one-hot encoden!
#
# ------------------------------------------------------------
# GPU/CPU und .to(device)
# ------------------------------------------------------------
# Wenn das Modell auf GPU ist, müssen auch die Daten auf GPU:
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)
#
# for xb, yb in train_loader:
#     xb = xb.to(device)
#     yb = yb.to(device)
#     logits = model(xb)
#
# Sonst gibt's einen Fehler "tensors not on same device".


from pathlib import Path
import torch

from main.src.fer.dataset.dataloaders.dataloader import build_dataloaders

project_root = Path(__file__).resolve().parents[0]  # wo train_simple_model.py liegt
images_root = project_root / "src" / "fer" / "dataset" / "standardized" / "images_mtcnn_cropped_norm"

# num_workers:
# - wie viele Prozesse parallel die Daten laden & Augmentations anwenden
# - 4 ist oft ein guter Startwert
dls = build_dataloaders(images_root, batch_size=64, num_workers=4)

train_loader = dls.train
val_loader = dls.val
test_loader = dls.test

# Fixes Mapping:
# {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'sadness':4, 'surprise':5}
class_to_idx = dls.class_to_idx

# Ab hier Modell erstellen und trainieren

# zum Beispiel
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)

# for xb, yb in train_loader:
#     xb = xb.to(device)  # (B,C,64,64)
#     yb = yb.to(device)  # (B,)
#     logits = model(xb)
#     # loss = ...
