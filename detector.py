"""
Crack Detection Inference Engine
Runs the exported ONNX model on CPU — no GPU needed in production.

Model input:  224×224 RGB image, ImageNet-normalised, NCHW float32
Model output: logits [batch, num_classes]

Place trained weights at:
  api/models/crack_detector.onnx
  api/models/class_names.json
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR  = Path(__file__).parent / "models"
ONNX_PATH   = MODELS_DIR / "crack_detector.onnx"
NAMES_PATH  = MODELS_DIR / "class_names.json"

# ---------------------------------------------------------------------------
# Severity heuristic (computer vision, no ML needed)
# ---------------------------------------------------------------------------

def estimate_severity(img_bgr: np.ndarray, is_cracked: bool) -> str:
    """
    Estimate crack severity from image texture using classical CV.
    Only called when the classifier says 'cracked'.

    Returns: "hairline" | "moderate" | "severe" | "none"

    Method:
      1. Convert to greyscale, apply Gaussian blur
      2. Canny edge detection
      3. Measure the ratio of edge pixels (proxy for crack density/width)
    """
    if not is_cracked:
        return "none"

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold to isolate dark crack regions
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Canny edges
    edges = cv2.Canny(blurred, 50, 150)

    # Edge pixel ratio
    h, w = edges.shape
    edge_ratio = np.count_nonzero(edges) / (h * w)

    # Morphological: find connected crack components
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Largest crack area as fraction of image
    max_area = max((cv2.contourArea(c) for c in contours), default=0)
    area_ratio = max_area / (h * w)

    # Classification heuristic (tuned on SDNET2018 visual inspection)
    if edge_ratio < 0.03 and area_ratio < 0.01:
        return "hairline"
    elif edge_ratio < 0.08 and area_ratio < 0.05:
        return "moderate"
    else:
        return "severe"


def draw_severity_overlay(img_bgr: np.ndarray) -> np.ndarray:
    """
    Draw crack region overlay on the image using thresholding.
    Returns a copy of the image with coloured contours drawn.
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel  = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = img_bgr.copy()
    # Draw large contours only (filter noise)
    h, w = img_bgr.shape[:2]
    min_area = h * w * 0.001
    large = [c for c in contours if cv2.contourArea(c) > min_area]
    cv2.drawContours(overlay, large, -1, (0, 255, 128), 2)
    return overlay


# ---------------------------------------------------------------------------
# Pre/Post processing
# ---------------------------------------------------------------------------

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    BGR image (any size) → NCHW float32 tensor ready for ONNX inference.
    Matches the preprocessing used during YOLOv8-cls training.
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)          # HWC → CHW
    return img[np.newaxis, :]             # add batch dim → NCHW


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

SEVERITY_URGENCY = {
    "none":     0,
    "hairline": 1,
    "moderate": 2,
    "severe":   3,
}

SEVERITY_COLOR = {
    "none":     "#00e5b0",
    "hairline": "#ffe066",
    "moderate": "#ffb347",
    "severe":   "#ff5f6d",
}

SEVERITY_ACTION = {
    "none":     "No action required. Surface is in good condition.",
    "hairline": "Monitor periodically. Document and re-inspect in 6–12 months.",
    "moderate": "Schedule professional inspection within 3 months.",
    "severe":   "Immediate structural assessment recommended.",
}


@dataclass
class DetectionResult:
    is_cracked: bool
    confidence: float          # 0.0 – 1.0
    severity: str              # none / hairline / moderate / severe
    severity_color: str
    action: str
    class_probabilities: dict  # {"cracked": 0.97, "uncracked": 0.03}
    inference_time_ms: float
    model_version: str = "yolov8n-cls-sdnet2018"


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class CrackDetector:
    """
    Loads the ONNX model once and runs inference on demand.
    Thread-safe for use with FastAPI.
    """

    def __init__(self, onnx_path: Path = ONNX_PATH, names_path: Path = NAMES_PATH):
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "onnxruntime not installed. Run: pip install onnxruntime"
            )

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}.\n"
                "Run the Colab training notebook and copy best.pt/.onnx to api/models/"
            )

        # Load class names from training
        if names_path.exists():
            with open(names_path) as f:
                raw = json.load(f)
            # YOLOv8 exports as {"0": "cracked", "1": "uncracked"}
            self.class_names = {int(k): v for k, v in raw.items()}
        else:
            # Fallback — verify this matches your actual training order
            self.class_names = {0: "cracked", 1: "uncracked"}

        # ONNX session — CPU provider is sufficient
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 2
        sess_opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, img_bgr: np.ndarray) -> DetectionResult:
        """
        Run crack detection on a BGR image (as returned by cv2.imread).
        """
        t0 = time.perf_counter()

        tensor  = preprocess(img_bgr)
        logits  = self.session.run([self.output_name], {self.input_name: tensor})[0][0]
        probs   = softmax(logits)

        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        top_idx    = int(np.argmax(probs))
        top_conf   = float(probs[top_idx])
        top_label  = self.class_names[top_idx]
        is_cracked = top_label == "cracked"

        severity = estimate_severity(img_bgr, is_cracked)

        class_probs = {
            self.class_names[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        }

        return DetectionResult(
            is_cracked=is_cracked,
            confidence=round(top_conf, 4),
            severity=severity,
            severity_color=SEVERITY_COLOR[severity],
            action=SEVERITY_ACTION[severity],
            class_probabilities=class_probs,
            inference_time_ms=round(inference_ms, 2),
        )


# ---------------------------------------------------------------------------
# Demo / mock detector (used when model weights are not yet available)
# ---------------------------------------------------------------------------

class MockDetector:
    """
    Returns plausible fake results so the API and frontend can be
    developed and tested before training is complete.
    Replace with CrackDetector once you have the ONNX weights.
    """

    def predict(self, img_bgr: np.ndarray) -> DetectionResult:
        import random
        # Simulate based on image darkness (darker = more likely cracked)
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(gray.mean())

        # Darker images tend to contain cracks in SDNET2018
        crack_prob = max(0.05, min(0.98, 1.0 - (mean_brightness / 255)))
        is_cracked = crack_prob > 0.5
        conf = crack_prob if is_cracked else (1.0 - crack_prob)

        if is_cracked:
            severity = random.choice(["hairline", "moderate", "severe"])
        else:
            severity = "none"

        return DetectionResult(
            is_cracked=is_cracked,
            confidence=round(conf, 4),
            severity=severity,
            severity_color=SEVERITY_COLOR[severity],
            action=SEVERITY_ACTION[severity],
            class_probabilities={
                "cracked":   round(crack_prob, 4),
                "uncracked": round(1 - crack_prob, 4),
            },
            inference_time_ms=round(random.uniform(8, 25), 2),
            model_version="mock-demo",
        )


# ---------------------------------------------------------------------------
# Factory — auto-selects real vs mock
# ---------------------------------------------------------------------------

def load_detector() -> "CrackDetector | MockDetector":
    """
    Returns a real CrackDetector if the ONNX weights exist,
    otherwise a MockDetector for development.
    """
    if ONNX_PATH.exists():
        print(f"[detector] Loading ONNX model from {ONNX_PATH}")
        return CrackDetector()
    else:
        print("[detector] WARNING: No ONNX model found — using MockDetector.")
        print(f"[detector] Expected model at: {ONNX_PATH}")
        return MockDetector()
