import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List

from fer.pre_processing.basic_img_norms import BasicImageProcessor
from fer.pre_processing.face_detection.mtcnn import MTCNNFaceCropper, FaceCropResult
from fer.models.cnn_resnet18 import ResNet18FER

# =========================================================
# CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "weights" / "resnet18fer" / "model_state_dict.pt"

IMG_SIZE = 64
DETECT_EVERY = 5
DETECT_MAX_SIZE = 480
MIN_PROB = 0.5
SMOOTH_ALPHA = 0.65

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]

EMOTION_COLORS = {
    "Angry":    (0, 0, 255),     
    "Disgust":  (0, 180, 0),     
    "Fear":     (180, 0, 180),   
    "Happy":    (0, 220, 220),   
    "Sad":      (255, 120, 0),   
    "Surprise": (0, 200, 255),   
}


torch.set_grad_enabled(False)

# =========================================================
# MODEL
# =========================================================

model = ResNet18FER(num_classes=6, in_channels=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

processor = BasicImageProcessor(target_size=(64, 64))

# =========================================================
# FACE CROPPER
# =========================================================

cropper = MTCNNFaceCropper(
    keep_all=True,
    min_prob=0.0,
    crop_scale=1.15,
    device=DEVICE,
)

# =========================================================
# HELPERS
# =========================================================

def resize_for_detection(frame_rgb):
    h, w = frame_rgb.shape[:2]
    scale = min(1.0, DETECT_MAX_SIZE / max(h, w))
    if scale < 1.0:
        frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
    return frame_rgb, scale


def preprocess_faces(faces: List[FaceCropResult]):
    tensors = []
    for f in faces:
        face_rgb = np.array(f.crop)
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        proc = processor.process_bgr(face_bgr)
        img = proc.normalized_rgb_vis.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        tensors.append(torch.from_numpy(img))
    return torch.stack(tensors).to(DEVICE)


def should_exit(name):
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        return True
    if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
        return True
    return False

def draw_box_layout(frame, x1, y1, x2, y2, emotion, conf):
    color = EMOTION_COLORS.get(emotion, (0, 255, 0))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{emotion.upper()}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)

    pad = 6
    y_label = max(th + pad + 2, y1)

    cv2.rectangle(
        frame,
        (x1, y_label - th - pad * 2),
        (x1 + tw + pad * 2, y_label),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x1 + pad, y_label - pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
    )

# =========================================================
# STATE
# =========================================================

stable_boxes = None
cached_faces: List[FaceCropResult] = []
cached_scale = 1.0
emotion_ema = {}
EMA_ALPHA = 0.7  

# =========================================================
# WEBCAM LOOP
# =========================================================

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

frame_idx = 0

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_idx += 1
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ---------------- FACE DETECTION ----------------
    if frame_idx % DETECT_EVERY == 0:
        small_rgb, scale = resize_for_detection(frame_rgb)
        pil_small = Image.fromarray(small_rgb)

        boxes, probs = cropper.mtcnn.detect(pil_small, landmarks=False)
        faces = cropper.process_pil(pil_small)

        if boxes is not None and faces:
            boxes = boxes.astype(np.float32)

            if stable_boxes is None or len(stable_boxes) != len(boxes):
                stable_boxes = boxes.copy()
            else:
                stable_boxes = (
                    SMOOTH_ALPHA * stable_boxes
                    + (1.0 - SMOOTH_ALPHA) * boxes
                )

            cached_faces = faces
            cached_scale = scale

    if stable_boxes is not None and len(emotion_ema) != len(stable_boxes):
        emotion_ema.clear()    

    faces = cached_faces

    # ---------------- MODEL ----------------
    if faces and stable_boxes is not None:
        batch = preprocess_faces(faces)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)

        inv_scale = 1.0 / cached_scale

        for i, box in enumerate(stable_boxes):
            
            probs_np = probs[i].cpu().numpy()

            if i not in emotion_ema:
                emotion_ema[i] = probs_np
            else:
                emotion_ema[i] = (
                    EMA_ALPHA * emotion_ema[i]
                    + (1 - EMA_ALPHA) * probs_np
                )
            emotion_id = emotion_ema[i].argmax()
            emotion = EMOTIONS[emotion_id]
            confidence = emotion_ema[i][emotion_id]
            if confidence < MIN_PROB:
                continue
            x1, y1, x2, y2 = (box * inv_scale).astype(int)

            draw_box_layout(
            frame_bgr,
            x1, y1, x2, y2,
            emotion,
            confidence
        )
        

    cv2.imshow("FER Webcam Demo", frame_bgr)
    if should_exit("FER Webcam Demo"):
        break

cap.release()
cv2.destroyAllWindows()