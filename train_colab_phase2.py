# ============================================================
# CrackScan Phase 2 — Bounding Box Detection Training
# Run cell-by-cell in Google Colab (free T4 GPU)
#
# Two paths:
#   Path A (Fast):   Download a pre-labelled detection dataset from Roboflow
#   Path B (Advanced): Auto-label SDNET2018 with Grounding DINO + SAM
#
# Output: crack_detector_det.onnx  →  place in api/models/
# ============================================================


# ── CELL 1: GPU check ────────────────────────────────────────
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# ── CELL 2: Install dependencies ─────────────────────────────
# !pip install ultralytics roboflow supervision -q
# For Path B auto-labeling:
# !pip install groundingdino-py segment-anything -q


# ═══════════════════════════════════════════════════════════════
# PATH A — Pre-labelled dataset from Roboflow (RECOMMENDED)
# ═══════════════════════════════════════════════════════════════

# ── CELL 3A: Download pre-labelled crack detection dataset ────
#
# Option 1: Use the Roboflow API (free account required)
#
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")  # sign up at roboflow.com

# Search roboflow.com for "concrete crack detection" datasets.
# Good public datasets:
#   - "Crack Detection" by Roboflow (concrete-crack-detection)
#   - "Bridge Crack" dataset
#   - "SDNET Detection" (pre-annotated version)
#
project = rf.workspace("YOUR_WORKSPACE").project("concrete-crack-detection")
dataset = project.version(1).download("yolov8")
DATASET_YAML = dataset.location + "/data.yaml"
print("Dataset downloaded to:", dataset.location)

#
# Option 2: Manual download — go to https://universe.roboflow.com
# Search "concrete crack" → Export → YOLOv8 format → Download ZIP
# Unzip to /content/crack_dataset/ and set:
# DATASET_YAML = "/content/crack_dataset/data.yaml"


# ═══════════════════════════════════════════════════════════════
# PATH B — Auto-label SDNET2018 with Grounding DINO + SAM
# (Keeps your existing dataset; produces YOLO-format labels)
# ═══════════════════════════════════════════════════════════════

# ── CELL 3B: Setup Grounding DINO ────────────────────────────
#
# Grounding DINO is a zero-shot open-set object detector.
# We prompt it with the text "crack" to find crack regions.
#

import os, sys
from pathlib import Path
import cv2
import numpy as np
import json

# Clone and install Grounding DINO
# !git clone https://github.com/IDEA-Research/GroundingDINO.git /content/GroundingDINO -q
# %cd /content/GroundingDINO
# !pip install -e . -q
# %cd /content

# Download weights
# !mkdir -p /content/weights
# !wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
#       -O /content/weights/groundingdino_swint_ogc.pth

SDNET_DIR      = Path("/content/data/surface-crack-detection/Positive")  # cracked images only
LABELS_OUT_DIR = Path("/content/auto_labels/cracked")
LABELS_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── CELL 4B: Run zero-shot auto-labeling ─────────────────────
#
# NOTE: This is the "AI-Engineering Path" — it uses Grounding DINO
# to generate bounding box annotations automatically.
# Expect ~80–85% IoU quality vs human labels. Sufficient for training.
#

from groundingdino.util.inference import load_model, load_image, predict, annotate

MODEL_CONFIG = "/content/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_WEIGHTS = "/content/weights/groundingdino_swint_ogc.pth"

model = load_model(MODEL_CONFIG, MODEL_WEIGHTS)

TEXT_PROMPT    = "crack . concrete crack . surface crack"
BOX_THRESHOLD  = 0.25
TEXT_THRESHOLD = 0.25

images = list(SDNET_DIR.glob("*.jpg"))[:5000]  # use first 5k for speed
print(f"Auto-labeling {len(images)} cracked images...")

skipped = 0
for img_path in images:
    image_source, image = load_image(str(img_path))
    h, w = image_source.shape[:2]

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    if len(boxes) == 0:
        skipped += 1
        continue

    # Save YOLO-format label file  (class cx cy w h  — all normalised)
    label_path = LABELS_OUT_DIR / (img_path.stem + ".txt")
    with open(label_path, "w") as f:
        for box in boxes:
            cx, cy, bw, bh = box.tolist()   # already normalised by DINO
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

print(f"Done. Labels written: {len(images) - skipped}  Skipped (no detections): {skipped}")


# ── CELL 5B: Build YOLOv8 detection dataset structure ────────
#
# Expected layout for YOLOv8-det:
#   dataset/
#     images/
#       train/  val/  test/
#     labels/
#       train/  val/  test/
#

import shutil
from sklearn.model_selection import train_test_split

YOLO_DET_DIR = Path("/content/yolo_det_dataset")
labelled_images = [
    p for p in images
    if (LABELS_OUT_DIR / (p.stem + ".txt")).exists()
]

# For uncracked images — label file is empty (no crack boxes)
NEG_DIR = Path("/content/data/surface-crack-detection/Negative")
neg_images = list(NEG_DIR.glob("*.jpg"))[:len(labelled_images)]

all_images = [(p, True) for p in labelled_images] + [(p, False) for p in neg_images]

train, temp = train_test_split(all_images, test_size=0.25, random_state=42)
val,   test = train_test_split(temp,       test_size=0.40, random_state=42)

def copy_split(items, split_name):
    img_dir = YOLO_DET_DIR / "images" / split_name
    lbl_dir = YOLO_DET_DIR / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path, is_cracked in items:
        shutil.copy(img_path, img_dir / img_path.name)
        if is_cracked:
            src_lbl = LABELS_OUT_DIR / (img_path.stem + ".txt")
            if src_lbl.exists():
                shutil.copy(src_lbl, lbl_dir / (img_path.stem + ".txt"))
            else:
                # Auto-labeling produced no boxes → empty label = uncracked
                (lbl_dir / (img_path.stem + ".txt")).write_text("")
        else:
            # No crack → empty label file
            (lbl_dir / (img_path.stem + ".txt")).write_text("")

for name, items in [("train", train), ("val", val), ("test", test)]:
    copy_split(items, name)
    print(f"{name}: {len(items)} images")

# Write data.yaml
yaml_content = f"""
path: {YOLO_DET_DIR}
train: images/train
val:   images/val
test:  images/test

nc: 1
names: ['cracked']
"""
with open(YOLO_DET_DIR / "data.yaml", "w") as f:
    f.write(yaml_content.strip())

DATASET_YAML = str(YOLO_DET_DIR / "data.yaml")
print("Dataset YAML written to:", DATASET_YAML)


# ═══════════════════════════════════════════════════════════════
# TRAINING — same for both Path A and Path B
# ═══════════════════════════════════════════════════════════════

# ── CELL 6: Train YOLOv8n-det ────────────────────────────────
from ultralytics import YOLO

# Detection model — NOT the classifier
# yolov8n.pt = nano (fastest, use for demo)
# yolov8s.pt = small (~2× slower, ~3% better mAP)
model = YOLO("yolov8n.pt")

results = model.train(
    data=DATASET_YAML,
    epochs=50,
    imgsz=640,           # detection uses 640 (not 224 like classification)
    batch=32,
    patience=10,
    optimizer="AdamW",
    lr0=1e-3,
    lrf=0.01,
    cos_lr=True,
    augment=True,
    mosaic=1.0,          # mosaic augmentation — very effective for crack detection
    mixup=0.1,
    degrees=10,          # rotation augmentation — cracks appear at all angles
    flipud=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.2,
    project="/content/runs_det",
    name="crack_det",
    pretrained=True,
    device=0,
    save=True,
    verbose=True,
)

print("Best model saved to:", results.save_dir)


# ── CELL 7: Evaluate on test set ─────────────────────────────
from ultralytics import YOLO

best = YOLO("/content/runs_det/crack_det/weights/best.pt")
metrics = best.val(data=DATASET_YAML, split="test")

print(f"\nmAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision:{metrics.box.p.mean():.4f}")
print(f"Recall:   {metrics.box.r.mean():.4f}")

# Target benchmarks for SDNET2018:
# mAP50 > 0.70 is good for hairline cracks (hard to detect)
# mAP50 > 0.80 is excellent


# ── CELL 8: Visualise predictions ────────────────────────────
from ultralytics import YOLO
import cv2, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

best = YOLO("/content/runs_det/crack_det/weights/best.pt")

test_dir = Path(DATASET_YAML).parent / "images" / "test"
samples  = list(test_dir.glob("*.jpg"))[:8]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for idx, img_path in enumerate(samples):
    result = best.predict(str(img_path), verbose=False, conf=0.25)[0]
    annotated = result.plot()   # returns BGR numpy array with boxes drawn
    ax = axes[idx // 4][idx % 4]
    ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    n_boxes = len(result.boxes)
    ax.set_title(f"{n_boxes} crack(s) found", fontsize=9,
                 color="tomato" if n_boxes else "limegreen")
    ax.axis("off")

plt.tight_layout()
plt.savefig("/content/detection_predictions.png", dpi=130)
plt.show()
print("Saved to /content/detection_predictions.png")


# ── CELL 9: Export to ONNX ───────────────────────────────────
best = YOLO("/content/runs_det/crack_det/weights/best.pt")

onnx_path = best.export(
    format="onnx",
    imgsz=640,       # must match training imgsz
    simplify=True,
    opset=17,
    dynamic=False,   # fixed batch=1 for API use
)
print("ONNX model exported to:", onnx_path)


# ── CELL 10: Verify ONNX output shape ────────────────────────
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(str(onnx_path))
print("Input  name :", sess.get_inputs()[0].name)
print("Input  shape:", sess.get_inputs()[0].shape)
print("Output name :", sess.get_outputs()[0].name)
print("Output shape:", sess.get_outputs()[0].shape)
# Expected output: [1, 5+num_classes, num_anchors]
# For 1-class (cracked only): [1, 6, 8400]


# ── CELL 11: Quick ONNX inference test ───────────────────────
import onnxruntime as ort
import numpy as np, cv2
from pathlib import Path

CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45

sess       = ort.InferenceSession(str(onnx_path))
input_name = sess.get_inputs()[0].name

def letterbox(img, size=640):
    h0, w0 = img.shape[:2]
    scale   = size / max(h0, w0)
    nh, nw  = int(h0 * scale), int(w0 * scale)
    img     = cv2.resize(img, (nw, nh))
    canvas  = np.full((size, size, 3), 114, np.uint8)
    py      = (size - nh) // 2
    px      = (size - nw) // 2
    canvas[py:py+nh, px:px+nw] = img
    return canvas, scale, px, py

test_img_path = str(samples[0])
img_bgr = cv2.imread(test_img_path)
letterboxed, scale, px, py = letterbox(img_bgr)
tensor = (letterboxed[:,:,::-1].astype(np.float32) / 255.0)
tensor = tensor.transpose(2,0,1)[np.newaxis,:]

raw = sess.run(None, {input_name: tensor})[0]      # (1, 6, 8400)
preds = raw[0].T                                   # (8400, 6)

score = preds[:,4] * preds[:,5]
mask  = score > CONF_THRESHOLD
if mask.any():
    p  = preds[mask]
    s  = score[mask]
    cx, cy, bw, bh = p[:,0], p[:,1], p[:,2], p[:,3]
    x1 = ((cx - bw/2 - px) / scale).astype(int)
    y1 = ((cy - bh/2 - py) / scale).astype(int)
    x2 = ((cx + bw/2 - px) / scale).astype(int)
    y2 = ((cy + bh/2 - py) / scale).astype(int)
    print(f"Found {mask.sum()} boxes before NMS")
    for i in range(min(3, len(x1))):
        print(f"  Box {i}: ({x1[i]},{y1[i]}) → ({x2[i]},{y2[i]})  conf={s[i]:.2f}")
else:
    print("No detections above threshold on this sample.")


# ── CELL 12: Download weights ─────────────────────────────────
from google.colab import files

files.download("/content/runs_det/crack_det/weights/best.pt")
files.download(str(onnx_path))

print("\nPlace the downloaded files in your project:")
print("  best.pt               → api/models/crack_det_best.pt   (backup)")
print("  crack_detector_det.onnx → api/models/crack_detector_det.onnx  ← API uses this")
print("\nRename the ONNX file to: crack_detector_det.onnx")
print("The API will automatically switch from MockDetector to Phase 2 detection.")


# ── CELL 13: Save class names ────────────────────────────────
import json
class_names = {0: "cracked"}   # single class for detection
with open("/content/class_names_det.json", "w") as f:
    json.dump(class_names, f, indent=2)
files.download("/content/class_names_det.json")
print("Place as: api/models/class_names.json")


# ══ NOTES ════════════════════════════════════════════════════
#
# SEVERITY mapping (api/detector.py _box_severity):
#   Hairline : box area < 0.5% of image
#   Moderate : box area 0.5–4%
#   Severe   : box area > 4%
#
# You can tune these thresholds after reviewing predictions on
# your real field images. The bounding box area is a strong proxy
# for crack width when images are taken at a consistent distance.
#
# PHASE 3 IDEAS:
#   - Add a segmentation head (YOLOv8-seg) for pixel-level masks
#   - Estimate crack width in mm using known image scale (GSD)
#   - Track crack growth over time by comparing inspections at same location
#   - Add a REST endpoint to export inspection reports as PDF
# ═════════════════════════════════════════════════════════════
