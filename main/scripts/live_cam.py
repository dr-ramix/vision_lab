# live_fer.py
# Start: python live_fer.py
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# =========================
# IMPORTS (anpassen an deine Repo-Struktur)
# =========================
# Diese Imports sollen GENAU deine existierenden Klassen nutzen:
# - MTCNNFaceCropper (dein mtcnn code)
# - BasicImageProcessor (dein preprocessing code)
#
# Beispiel, wenn deine Dateien so heißen:
#   main/src/fer/face/mtcnn_face_cropper.py
#   main/src/fer/preprocessing/basic_image_processor.py
#
# Passe die Pfade entsprechend an.
from fer.pre_processing.face_detection.mtcnn import MTCNNFaceCropper
from fer.pre_processing.basic_img_norms import BasicImageProcessor

# Dein Modell (so wie du es gepostet hast)
from main.src.fer.models.cnn_vanilla import CNNVanilla

# Optional: gleiche Klassenreihenfolge wie Training
# In eurem Training gibt es CLASS_ORDER / CLASS_TO_IDX (falls verfügbar).
# Wenn der Import bei dir nicht existiert, kommentiere ihn aus und setze CLASS_ORDER manuell unten.
from main.src.fer.dataset.dataloaders.dataloader import CLASS_ORDER


# =========================
# KONFIG
# =========================
CAMERA_INDEX = 0
FACE_PROB_THRESHOLD = 0.90

# Sobald dein Teampartner pusht, einfach hier eintragen:
# - best.ckpt.pt (enthält "model_state") ODER
# - model_state_dict.pt (reines state_dict)
WEIGHTS_PATH = ""  # z.B. "results/fer_cnnvanilla_.../best.ckpt.pt"

# Wenn du CLASS_ORDER importierst, nutzt du automatisch die Trainings-Reihenfolge.
# Fallback (falls du CLASS_ORDER nicht importieren willst/kannst):
# CLASS_ORDER = ["angry", "disgust", "fear", "happy", "sad", "surprise"]

WINDOW_NAME = "Live FER"



def load_class_order_from_runfolder(weights_path: Path):
    """
    Wenn du eine results/run-Ordnerstruktur hast:
      - class_order.json liegt neben best.ckpt.pt
    Falls nicht vorhanden: None zurück.
    """
    try:
        run_dir = weights_path.parent
        p = run_dir / "class_order.json"
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return None


def load_weights_into_model(model: torch.nn.Module, weights_path: Path):
    """
    Unterstützt:
      1) best.ckpt.pt mit key "model_state" (euer save_best_checkpoint) :contentReference[oaicite:3]{index=3}
      2) reines state_dict (z.B. model_state_dict.pt) :contentReference[oaicite:4]{index=4}
    """
    obj = torch.load(str(weights_path), map_location="cpu")
    if isinstance(obj, dict) and "model_state" in obj:
        sd = obj["model_state"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    else:
        sd = obj
    model.load_state_dict(sd, strict=True)


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor):
    """
    x: (1,3,64,64) float32
    returns: (class_idx, confidence)
    """
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    return int(idx), float(conf)


def draw_box_and_text(frame_bgr: np.ndarray, box_xyxy, text: str):
    x1, y1, x2, y2 = map(int, box_xyxy)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)

    y_top = max(0, y1 - th - baseline - 6)
    cv2.rectangle(frame_bgr, (x1, y_top), (x1 + tw + 8, y1), (0, 255, 0), -1)
    cv2.putText(frame_bgr, text, (x1 + 4, y1 - 6), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # 1) Deine vorhandenen Klassen initialisieren
    cropper = MTCNNFaceCropper(
        keep_all=True,
        min_prob=0.0,      # wir filtern später mit FACE_PROB_THRESHOLD
        width_half=1.3,
        device=str(device)
    )  # nutzt intern facenet_pytorch.MTCNN :contentReference[oaicite:5]{index=5}

    processor = BasicImageProcessor(target_size=(64, 64))  # deine Pipeline :contentReference[oaicite:6]{index=6}

    # 2) Modell
    num_classes = len(CLASS_ORDER)
    model = CNNVanilla(num_classes=num_classes).to(device).eval()

    # Labels: bevorzugt aus class_order.json neben den weights (wie im Training gespeichert) :contentReference[oaicite:7]{index=7}
    labels = list(CLASS_ORDER)

    # 3) Optional: Weights laden (wenn vorhanden)
    if WEIGHTS_PATH.strip():
        wp = Path(WEIGHTS_PATH)
        if not wp.exists():
            raise FileNotFoundError(f"Weights nicht gefunden: {wp}")
        co = load_class_order_from_runfolder(wp)
        if co is not None:
            labels = list(co)
            if len(labels) != num_classes:
                # falls CLASS_ORDER Import abweicht, bauen wir das Model neu passend zur class_order.json
                num_classes = len(labels)
                model = CNNVanilla(num_classes=num_classes).to(device).eval()

        load_weights_into_model(model, wp)
        model.to(device).eval()
        print(f"[INFO] Loaded weights: {wp}")
        print(f"[INFO] Labels: {labels}")
    else:
        print("[WARN] WEIGHTS_PATH ist leer. Script läuft, aber Predictions sind zufällig, bis du Weights einträgst.")

    # 4) Kamera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {CAMERA_INDEX} konnte nicht geöffnet werden.")
    print("[INFO] Press 'q' to quit.")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # OpenCV BGR -> PIL RGB (MTCNN erwartet PIL RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)

        # 5) Bounding Boxes direkt aus MTCNN: boxes+probs = "Gesicht erkannt?" + Box-Koordinaten :contentReference[oaicite:8]{index=8}
        boxes, probs, _lms = cropper.mtcnn.detect(pil, landmarks=True)

        # 6) Deine Crop/Alignment-Logik (liefert FaceCropResult Liste; face_index korrespondiert zur detect-Reihenfolge) :contentReference[oaicite:9]{index=9}
        crops = cropper.process_pil(pil)
        crop_by_idx = {r.face_index: r.crop for r in crops}

        if boxes is not None and probs is not None:
            for i, (box, p) in enumerate(zip(boxes, probs)):
                if p is None or float(p) < FACE_PROB_THRESHOLD:
                    continue
                if i not in crop_by_idx:
                    continue

                face_pil = crop_by_idx[i]  # PIL RGB crop

                # PIL(RGB) -> numpy(BGR) für BasicImageProcessor
                face_rgb = np.array(face_pil)
                face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

                # 7) Dein Preprocessing: liefert u.a. normalized_rgb_vis (64,64,3) uint8 :contentReference[oaicite:10]{index=10}
                proc = processor.process_bgr(face_bgr)
                img_u8 = proc.normalized_rgb_vis  # (64,64,3)

                # 8) Tensor fürs Modell (Training-typisch: float in [0,1])
                x = img_u8.astype(np.float32) / 255.0
                x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
                x = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,3,64,64)

                # 9) Prediction + Text
                cls_idx, conf = predict(model, x)
                label = labels[cls_idx] if 0 <= cls_idx < len(labels) else str(cls_idx)
                text = f"{label} ({conf:.2f})"
                draw_box_and_text(frame_bgr, box, text)

        cv2.imshow(WINDOW_NAME, frame_bgr)

        # Taste 'q' beendet sauber
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # Fenster wurde über x geschlossen
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
