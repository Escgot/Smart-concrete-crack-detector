# ============================================================
# Concrete Crack Detector — Training Notebook
# Run cell-by-cell in Google Colab (free T4 GPU)
#
# Dataset: SDNET2018 — 56,000 labelled concrete crack images
# Model:   YOLOv8 classification (ultralytics)
# Output:  best.pt  →  crack_detector.onnx
# ============================================================

# ── CELL 1: Check GPU ────────────────────────────────────────
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
# Expected: Tesla T4 on Colab free tier

# ── CELL 2: Install dependencies ─────────────────────────────
# !pip install ultralytics kagglehub opencv-python-headless albumentations -q

# ── CELL 3: Download SDNET2018 from Kaggle ───────────────────
# Option A — Kaggle API (recommended, fastest)
# Upload your kaggle.json first: Files → Upload
# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d arunrk7/surface-crack-detection -p /content/data --unzip

# Option B — Direct from UCI (no account needed, slower)
import os, urllib.request, zipfile

DATA_ROOT = "/content/data"
os.makedirs(DATA_ROOT, exist_ok=True)

# SDNET2018 is hosted on Mendeley Data — download the zip
# If the URL changes, search "SDNET2018 dataset download"
UCI_URL = "https://digitalcommons.usu.edu/cgi/viewcontent.cgi?filename=0&article=1047&context=all_datasets&type=additional"
# NOTE: For Colab, use the Kaggle route (Option A) — it is more reliable.
# The dataset has two folders: Positive (cracked) and Negative (uncracked)
print("Download dataset via Kaggle (Option A) or manually place images in:")
print(f"  {DATA_ROOT}/Positive/  ← cracked images")
print(f"  {DATA_ROOT}/Negative/  ← uncracked images")

# ── CELL 4: Explore the dataset ──────────────────────────────
import os
from pathlib import Path

# After download, SDNET2018 has structure:
#   surface-crack-detection/
#     Positive/  (cracked)   — 20,000 images
#     Negative/  (uncracked) — 20,000 images
# We'll also add a SEVERITY sub-task after binary classification.

data_dir = Path("/content/data/surface-crack-detection")
pos = list((data_dir / "Positive").glob("*.jpg"))
neg = list((data_dir / "Negative").glob("*.jpg"))
print(f"Cracked images:   {len(pos)}")
print(f"Uncracked images: {len(neg)}")

import random
import matplotlib.pyplot as plt
import cv2

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, img_path in enumerate(random.sample(pos, 4)):
    img = cv2.imread(str(img_path))
    axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title("Cracked", color="tomato")
    axes[0, i].axis("off")
for i, img_path in enumerate(random.sample(neg, 4)):
    img = cv2.imread(str(img_path))
    axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title("Uncracked", color="limegreen")
    axes[1, i].axis("off")
plt.tight_layout()
plt.savefig("/content/dataset_samples.png", dpi=120)
plt.show()

# ── CELL 5: Organise into YOLOv8 classification structure ────
#
# YOLOv8-cls expects:
#   dataset/
#     train/
#       cracked/    ← images
#       uncracked/  ← images
#     val/
#       cracked/
#       uncracked/
#     test/
#       cracked/
#       uncracked/

import shutil
from sklearn.model_selection import train_test_split

YOLO_DIR = Path("/content/yolo_dataset")
SPLITS = {"train": 0.75, "val": 0.15, "test": 0.10}

def build_split(images, label, split_dir):
    dest = split_dir / label
    dest.mkdir(parents=True, exist_ok=True)
    for src in images:
        shutil.copy(src, dest / src.name)

# Split indices
train_pos, temp_pos = train_test_split(pos, test_size=0.25, random_state=42)
val_pos, test_pos   = train_test_split(temp_pos, test_size=0.40, random_state=42)

train_neg, temp_neg = train_test_split(neg, test_size=0.25, random_state=42)
val_neg, test_neg   = train_test_split(temp_neg, test_size=0.40, random_state=42)

for split_name, (p_imgs, n_imgs) in [
    ("train", (train_pos, train_neg)),
    ("val",   (val_pos,   val_neg)),
    ("test",  (test_pos,  test_neg)),
]:
    split_dir = YOLO_DIR / split_name
    build_split(p_imgs, "cracked",   split_dir)
    build_split(n_imgs, "uncracked", split_dir)
    print(f"{split_name}: {len(p_imgs)} cracked, {len(n_imgs)} uncracked")

# ── CELL 6: Train YOLOv8-cls ─────────────────────────────────
from ultralytics import YOLO

# Load pretrained YOLOv8 nano classifier (smallest — fast to fine-tune)
# For better accuracy use yolov8s-cls.pt (small) at cost of ~2x time
model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=str(YOLO_DIR),
    epochs=30,           # 30 epochs ≈ 25 min on T4 — increase to 50 for production
    imgsz=224,           # standard for classification
    batch=64,
    patience=10,         # early stopping
    optimizer="AdamW",
    lr0=1e-3,
    lrf=0.01,
    cos_lr=True,
    augment=True,        # built-in augmentation (flip, rotate, colour jitter)
    dropout=0.3,         # regularisation
    project="/content/runs",
    name="crack_cls",
    pretrained=True,
    device=0,            # GPU 0; use 'cpu' if no GPU
    save=True,
    verbose=True,
)

print("Best model saved to:", results.save_dir)

# ── CELL 7: Evaluate on test set ─────────────────────────────
from ultralytics import YOLO
import json

best_model = YOLO("/content/runs/crack_cls/weights/best.pt")

# Validate on test split
metrics = best_model.val(data=str(YOLO_DIR), split="test")
print(f"\nTest accuracy (top1): {metrics.top1:.4f}")
print(f"Test accuracy (top5): {metrics.top5:.4f}")

# ── CELL 8: Confusion matrix + sample predictions ─────────────
from ultralytics import YOLO
import cv2, numpy as np
import matplotlib.pyplot as plt

best_model = YOLO("/content/runs/crack_cls/weights/best.pt")

# Sample 8 test images and show predictions
test_cracked   = list((YOLO_DIR / "test" / "cracked").glob("*.jpg"))[:4]
test_uncracked = list((YOLO_DIR / "test" / "uncracked").glob("*.jpg"))[:4]
samples = test_cracked + test_uncracked

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for idx, path in enumerate(samples):
    r = best_model.predict(str(path), verbose=False)[0]
    top_cls = r.probs.top1
    conf    = r.probs.top1conf.item()
    label   = r.names[top_cls]
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    ax = axes[idx // 4][idx % 4]
    ax.imshow(img)
    colour = "tomato" if label == "cracked" else "limegreen"
    ax.set_title(f"{label}\n{conf:.2%}", color=colour, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("/content/predictions.png", dpi=120)
plt.show()

# ── CELL 9: Export to ONNX ───────────────────────────────────
# ONNX runs on CPU without PyTorch — perfect for production server
best_model = YOLO("/content/runs/crack_cls/weights/best.pt")
onnx_path = best_model.export(format="onnx", imgsz=224, simplify=True, opset=17)
print("ONNX model exported to:", onnx_path)

# ── CELL 10: Download model weights ──────────────────────────
from google.colab import files

# Download both .pt and .onnx — keep both
files.download("/content/runs/crack_cls/weights/best.pt")
files.download(str(onnx_path))
print("Download both files and place them in: api/models/")

# ── CELL 11: Quick inference test ────────────────────────────
import onnxruntime as ort
import numpy as np
import cv2

CLASS_NAMES = ["cracked", "uncracked"]  # order from training — verify with r.names

def preprocess(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    # ImageNet normalisation
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img - mean) / std
    img  = img.transpose(2, 0, 1)           # HWC → CHW
    return img[np.newaxis, :].astype(np.float32)  # add batch dim

sess = ort.InferenceSession(str(onnx_path))
input_name = sess.get_inputs()[0].name

# Test on a sample image
test_img = str(test_cracked[0])
logits = sess.run(None, {input_name: preprocess(test_img)})[0][0]
probs  = np.exp(logits) / np.exp(logits).sum()  # softmax
pred   = CLASS_NAMES[np.argmax(probs)]
conf   = float(np.max(probs))

print(f"Image: {test_img}")
print(f"Prediction: {pred}  ({conf:.2%} confidence)")

# ── CELL 12: Save class names for the API ────────────────────
import json

# IMPORTANT: Run this after training to capture the exact class order
best_model_check = YOLO("/content/runs/crack_cls/weights/best.pt")
dummy = best_model_check.predict(str(test_cracked[0]), verbose=False)[0]
class_names = dummy.names  # {0: 'cracked', 1: 'uncracked'} — actual order

with open("/content/class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

print("Class names:", class_names)
files.download("/content/class_names.json")
# Place this file in api/models/class_names.json

# ── NOTES FOR SEVERITY CLASSIFICATION (Phase 2) ──────────────
#
# After binary classification works, add severity:
#
# Strategy: Use crack WIDTH as a proxy for severity.
# SDNET2018 images are 227×227px at known physical scale.
# You can estimate crack width by:
#   1. Thresholding the crack region (Otsu's method)
#   2. Measuring the width of the thinnest connected component
#   3. Mapping pixel width → physical width using image scale
#
# Severity labels:
#   - Hairline: < 0.2 mm  (structural monitoring, low urgency)
#   - Moderate:  0.2–0.5 mm (schedule inspection)
#   - Severe:   > 0.5 mm  (immediate attention)
#
# Alternatively, train a 3-class classifier on manually labelled subsets.
# This is left as Phase 2 — binary classification is already publishable.
