import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from fer.pre_processing.basic_img_norms import BasicImageProcessor
from fer.pre_processing.face_detection.mtcnn import MTCNNFaceCropper
from fer.models.cnn_resnet18 import ResNet18FER
from fer.xai.occlusion import OcclusionSaliency

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "weights" / "resnet18fer" / "model_state_dict.pt"

IMG_SIZE = 64
DETECT_EVERY = 5
DETECT_MAX_SIZE = 480
SMOOTH_ALPHA = 0.6

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]


EMOTION_COLORS = {
    "Angry": (60, 60, 255),
    "Disgust": (80, 180, 80),
    "Fear": (180, 80, 180),
    "Happy": (80, 220, 220),
    "Sad": (200, 120, 60),
    "Surprise": (80, 200, 255),
}


model = ResNet18FER(num_classes=6, in_channels=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

occlusion = OcclusionSaliency(
    model=model,
    window_size=(8, 8),
    stride=(8, 8),      
    occlusion_value=0.0,
    batch_size=32,
)
processor = BasicImageProcessor(target_size=(IMG_SIZE, IMG_SIZE))

cropper = MTCNNFaceCropper(
    keep_all=True,
    min_prob=0.0,
    crop_scale=1.15,
    device=DEVICE,
)
OCCLUSION_EVERY = 15 
cached_occlusion = {}

def resize_for_detection(frame_rgb):
    h, w = frame_rgb.shape[:2]
    scale = min(1.0, DETECT_MAX_SIZE / max(h, w))
    if scale < 1.0:
        frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
    return frame_rgb, scale


def preprocess_face(face_bgr):
    face = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    proc = processor.process_bgr(face)
    img = proc.normalized_rgb_vis.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).to(DEVICE)


def draw_emotion_box(frame, x1, y1, x2, y2, emotion, conf):
    color = EMOTION_COLORS[emotion]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{emotion} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def apply_neutral_xai_background(frame, cam_mask, face_boxes):
    h, w = cam_mask.shape

    cam_mask = np.clip(cam_mask, 0, 1)
    cam_mask = cv2.GaussianBlur(cam_mask, (251, 251), 0)

    inv_cam = 1.0 - cam_mask
    inv_cam = np.clip(inv_cam, 0, 1)

    # Convert to heatmap (neutral CAM look)
    heatmap = cv2.applyColorMap(
    ((1.0 - inv_cam) * 255).astype(np.uint8),
    cv2.COLORMAP_JET
)

    heatmap = heatmap.astype(np.float32) / 255.0
    frame_f = frame.astype(np.float32) / 255.0

    blended = frame_f * 0.5 + heatmap * 0.5
    blended = (blended * 255).astype(np.uint8)

    mask_outside_faces = np.ones((h, w), dtype=np.uint8)

    for (x1, y1, x2, y2) in face_boxes:
        mask_outside_faces[y1:y2, x1:x2] = 0

    mask_outside_faces = mask_outside_faces[..., None]

    frame[:] = np.where(mask_outside_faces, blended, frame)


def should_exit(name):
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        return True
    if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
        return True
    return False


stable_boxes = None
cached_scale = 1.0
frame_idx = 0
last_emotion = None

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_idx += 1
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if frame_idx % DETECT_EVERY == 0:
        small, scale = resize_for_detection(frame_rgb)
        boxes, _ = cropper.mtcnn.detect(Image.fromarray(small), landmarks=False)

        if boxes is not None:
            boxes = boxes.astype(np.float32)
            stable_boxes = boxes if stable_boxes is None else (
                SMOOTH_ALPHA * stable_boxes + (1 - SMOOTH_ALPHA) * boxes
            )
            cached_scale = scale

    if stable_boxes is None:
        cv2.imshow("FER XAI Demo", frame_bgr)
        if should_exit("FER XAI Demo"):
            break
        continue

    inv_scale = 1.0 / cached_scale
    h, w = frame_bgr.shape[:2]

    global_cam = np.zeros((h, w), dtype=np.float32)
    face_boxes = []

    for box in stable_boxes:
        x1, y1, x2, y2 = (box * inv_scale).astype(int)
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)

        if x2 <= x1 or y2 <= y1:
            continue

        face_boxes.append((x1, y1, x2, y2))

        face_crop = frame_bgr[y1:y2, x1:x2]
        face_tensor = preprocess_face(face_crop)

        with torch.no_grad():
            probs = torch.softmax(model(face_tensor), dim=1)

        emotion_id = probs.argmax().item()
        emotion = EMOTIONS[emotion_id]
        conf = probs[0, emotion_id].item()
        cache_key = (emotion_id, x2 - x1, y2 - y1)

        if cache_key not in cached_occlusion:
            heatmap = occlusion(face_tensor, emotion_id)
            cached_occlusion[cache_key] = heatmap
        else:
            heatmap = cached_occlusion[cache_key]

        cam_face = cv2.resize(heatmap, (x2 - x1, y2 - y1))
        global_cam[y1:y2, x1:x2] = np.maximum(global_cam[y1:y2, x1:x2], cam_face)

        overlay_face = show_cam_on_image(
            cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
            cam_face,
            use_rgb=True
        )

        frame_bgr[y1:y2, x1:x2] = cv2.cvtColor(overlay_face, cv2.COLOR_RGB2BGR)
        draw_emotion_box(frame_bgr, x1, y1, x2, y2, emotion, conf)

    apply_neutral_xai_background(frame_bgr, global_cam, face_boxes)

    cv2.imshow("FER XAI Demo", frame_bgr)
    if should_exit("FER XAI Demo"):
        break

cap.release()
cv2.destroyAllWindows()