"""
Crack Detection Inference Engine — Phase 2 (Bounding Box Detection)
====================================================================
Supports two model modes, auto-selected at startup:

  Phase 2 (YOLOv8-det ONNX) — returns bounding boxes + class scores
    Place weights at: api/models/crack_detector_det.onnx

  Phase 1 (YOLOv8-cls ONNX) — falls back to classification-only
    Place weights at: api/models/crack_detector.onnx

  Mock — deterministic fake results for development (no weights needed)

Detection model input:  640×640 RGB image, normalised [0,1], NCHW float32
Detection model output: [batch, 5+num_classes, num_anchors]  (YOLO format)
                         columns: cx, cy, w, h, obj_conf, class_conf…

Classification model input:  224×224 RGB, ImageNet-normalised, NCHW float32
Classification model output: logits [batch, num_classes]
"""

import json
import time
from dataclasses import dataclass, field
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

MODELS_DIR   = Path(__file__).parent / "models"
ONNX_DET     = MODELS_DIR / "crack_detector_det.onnx"   # Phase 2 detection
ONNX_CLS     = MODELS_DIR / "crack_detector.onnx"       # Phase 1 classification
NAMES_PATH   = MODELS_DIR / "class_names.json"

# ---------------------------------------------------------------------------
# Severity constants
# ---------------------------------------------------------------------------

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

# Colour used to draw bounding boxes per severity (BGR for OpenCV)
SEVERITY_BGR = {
    "none":     (0, 229, 176),
    "hairline": (66, 228, 255),
    "moderate": (51, 152, 255),
    "severe":   (61,  63, 255),
}

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """A single detected crack bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    severity: str           # hairline / moderate / severe (derived from box area)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area_fraction(self) -> float:
        """Fraction of image area (caller must set image_area first)."""
        return (self.width * self.height)

    def to_dict(self) -> dict:
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "width": self.width, "height": self.height,
            "confidence": round(self.confidence, 4),
            "severity": self.severity,
        }


@dataclass
class DetectionResult:
    is_cracked: bool
    confidence: float               # 0.0 – 1.0  (highest box conf, or cls conf)
    severity: str                   # none / hairline / moderate / severe
    severity_color: str
    action: str
    class_probabilities: dict       # {"cracked": 0.97, "uncracked": 0.03}
    inference_time_ms: float
    model_version: str = "yolov8n-det-sdnet2018"
    bounding_boxes: list = field(default_factory=list)   # list[BoundingBox]
    detection_mode: str = "detection"                    # "detection" | "classification" | "mock"

    def boxes_as_dicts(self) -> list[dict]:
        return [b.to_dict() for b in self.bounding_boxes]


# ---------------------------------------------------------------------------
# Shared image pre/post processing
# ---------------------------------------------------------------------------

CLS_SIZE = 224
DET_SIZE = 640

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_cls(img_bgr: np.ndarray) -> np.ndarray:
    """BGR image → NCHW float32 for classification model (224×224, ImageNet-norm)."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CLS_SIZE, CLS_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)
    return img[np.newaxis, :]


def preprocess_det(img_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """
    BGR image → NCHW float32 for detection model (640×640, [0,1]-norm).
    Returns: (tensor, scale, pad_x, pad_y)
    scale and pad values are needed to map predicted boxes back to original coords.
    """
    h0, w0 = img_bgr.shape[:2]
    scale = DET_SIZE / max(h0, w0)
    nh, nw = int(h0 * scale), int(w0 * scale)

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Letterbox: pad to 640×640 with grey (114)
    canvas = np.full((DET_SIZE, DET_SIZE, 3), 114, dtype=np.uint8)
    pad_y = (DET_SIZE - nh) // 2
    pad_x = (DET_SIZE - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img

    tensor = canvas.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)      # HWC → CHW
    return tensor[np.newaxis, :], scale, pad_x, pad_y


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# Phase 2: bounding box post-processing
# ---------------------------------------------------------------------------

IOU_THRESHOLD  = 0.45
CONF_THRESHOLD = 0.25   # minimum objectness × class_conf to keep a box


def decode_detections(
    raw: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    orig_h: int,
    orig_w: int,
    conf_thresh: float = CONF_THRESHOLD,
    iou_thresh:  float = IOU_THRESHOLD,
) -> list[BoundingBox]:
    """
    Decode raw ONNX detection output → list of BoundingBox in original image coords.

    YOLO detection head output shape: [1, 5+num_cls, num_anchors]
      row layout: cx, cy, w, h, obj_conf, cls0_conf, cls1_conf, …

    We assume class 0 = cracked (the only class we care about).
    """
    # raw shape: (1, num_outputs, num_anchors) — transpose to (num_anchors, num_outputs)
    preds = raw[0].T          # (num_anchors, 5 + num_classes)

    boxes_out: list[BoundingBox] = []

    # Filter by objectness × class confidence
    obj_conf  = preds[:, 4]
    cls_conf  = preds[:, 5] if preds.shape[1] > 5 else obj_conf
    score     = obj_conf * cls_conf
    mask      = score > conf_thresh

    if not mask.any():
        return boxes_out

    preds  = preds[mask]
    scores = score[mask]

    # cx, cy, w, h → x1, y1, x2, y2 (in letterboxed 640×640 space)
    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # NMS
    boxes_np = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    indices  = cv2.dnn.NMSBoxes(
        bboxes=boxes_np.tolist(),
        scores=scores.tolist(),
        score_threshold=conf_thresh,
        nms_threshold=iou_thresh,
    )
    if len(indices) == 0:
        return boxes_out

    indices = indices.flatten()
    img_area = orig_h * orig_w

    for idx in indices:
        # Map back: remove padding, undo scale
        rx1 = int((boxes_np[idx, 0] - pad_x) / scale)
        ry1 = int((boxes_np[idx, 1] - pad_y) / scale)
        rx2 = int((boxes_np[idx, 2] - pad_x) / scale)
        ry2 = int((boxes_np[idx, 3] - pad_y) / scale)

        # Clip to image bounds
        rx1 = max(0, min(rx1, orig_w - 1))
        ry1 = max(0, min(ry1, orig_h - 1))
        rx2 = max(rx1 + 1, min(rx2, orig_w))
        ry2 = max(ry1 + 1, min(ry2, orig_h))

        box_area_frac = ((rx2 - rx1) * (ry2 - ry1)) / img_area
        severity = _box_severity(box_area_frac)

        boxes_out.append(BoundingBox(
            x1=rx1, y1=ry1, x2=rx2, y2=ry2,
            confidence=float(scores[idx]),
            severity=severity,
        ))

    return boxes_out


def _box_severity(area_fraction: float) -> str:
    """Map detected crack bounding-box area (fraction of image) → severity label."""
    if area_fraction < 0.005:
        return "hairline"
    elif area_fraction < 0.04:
        return "moderate"
    else:
        return "severe"


def _aggregate_severity(boxes: list[BoundingBox]) -> str:
    """Worst-case severity across all detected boxes."""
    order = {"hairline": 1, "moderate": 2, "severe": 3}
    if not boxes:
        return "none"
    return max(boxes, key=lambda b: order.get(b.severity, 0)).severity


# ---------------------------------------------------------------------------
# Overlay rendering (Phase 2 — annotated bounding boxes)
# ---------------------------------------------------------------------------

def draw_boxes_overlay(img_bgr: np.ndarray, boxes: list[BoundingBox]) -> np.ndarray:
    """
    Draw colour-coded bounding boxes + confidence labels on the image.
    Returns a copy; original is NOT modified.
    """
    overlay = img_bgr.copy()
    h, w = overlay.shape[:2]

    for box in boxes:
        color = SEVERITY_BGR.get(box.severity, (61, 63, 255))
        # Main rectangle
        cv2.rectangle(overlay, (box.x1, box.y1), (box.x2, box.y2), color, 2)

        # Label background
        label = f"{box.severity.upper()}  {box.confidence:.0%}"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = max(box.y1 - 6, lh + baseline)
        cv2.rectangle(
            overlay,
            (box.x1, label_y - lh - baseline),
            (box.x1 + lw + 6, label_y + baseline),
            color, -1,
        )
        cv2.putText(
            overlay, label,
            (box.x1 + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

    return overlay


# ---------------------------------------------------------------------------
# Phase 1 severity heuristic (kept for classification-mode fallback)
# ---------------------------------------------------------------------------

def estimate_severity_cls(img_bgr: np.ndarray, is_cracked: bool) -> str:
    if not is_cracked:
        return "none"

    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges   = cv2.Canny(blurred, 50, 150)

    h, w = edges.shape
    edge_ratio = np.count_nonzero(edges) / (h * w)

    kernel  = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area    = max((cv2.contourArea(c) for c in contours), default=0)
    area_ratio  = max_area / (h * w)

    if edge_ratio < 0.03 and area_ratio < 0.01:
        return "hairline"
    elif edge_ratio < 0.08 and area_ratio < 0.05:
        return "moderate"
    else:
        return "severe"


def draw_severity_overlay(img_bgr: np.ndarray) -> np.ndarray:
    """Phase 1 contour overlay — used when only the cls model is available."""
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel  = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    min_area = h * w * 0.001
    large = [c for c in contours if cv2.contourArea(c) > min_area]
    cv2.drawContours(overlay, large, -1, (0, 255, 128), 2)
    return overlay


# ---------------------------------------------------------------------------
# Phase 2 — YOLOv8 Detection Detector
# ---------------------------------------------------------------------------

class CrackDetectorDet:
    """
    Phase 2: YOLOv8-det ONNX model.
    Returns bounding boxes for each crack region.

    Expected model export:
        yolo export model=crack_det.pt format=onnx imgsz=640 simplify=True opset=17
    """

    def __init__(self, onnx_path: Path = ONNX_DET, names_path: Path = NAMES_PATH):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime not installed.")
        if not onnx_path.exists():
            raise FileNotFoundError(f"Detection ONNX not found at {onnx_path}")

        if names_path.exists():
            with open(names_path) as f:
                raw = json.load(f)
            self.class_names = {int(k): v for k, v in raw.items()}
        else:
            self.class_names = {0: "cracked"}

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
        h0, w0 = img_bgr.shape[:2]
        t0 = time.perf_counter()

        tensor, scale, pad_x, pad_y = preprocess_det(img_bgr)
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0]

        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        boxes = decode_detections(raw, scale, pad_x, pad_y, h0, w0)
        is_cracked = len(boxes) > 0
        severity   = _aggregate_severity(boxes)
        confidence = max((b.confidence for b in boxes), default=0.0) if boxes else 0.0

        # Pseudo class probabilities derived from top confidence
        crack_prob = float(confidence) if is_cracked else 1.0 - float(confidence)
        class_probs = {
            "cracked":   round(crack_prob, 4),
            "uncracked": round(1.0 - crack_prob, 4),
        }

        return DetectionResult(
            is_cracked=is_cracked,
            confidence=round(confidence, 4),
            severity=severity,
            severity_color=SEVERITY_COLOR[severity],
            action=SEVERITY_ACTION[severity],
            class_probabilities=class_probs,
            inference_time_ms=round(inference_ms, 2),
            model_version="yolov8n-det-sdnet2018",
            bounding_boxes=boxes,
            detection_mode="detection",
        )


# ---------------------------------------------------------------------------
# Phase 1 — YOLOv8 Classification Detector (unchanged, kept as fallback)
# ---------------------------------------------------------------------------

class CrackDetectorCls:
    """Phase 1 classification-only detector. Used when det model is absent."""

    def __init__(self, onnx_path: Path = ONNX_CLS, names_path: Path = NAMES_PATH):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime not installed.")
        if not onnx_path.exists():
            raise FileNotFoundError(f"Classification ONNX not found at {onnx_path}")

        if names_path.exists():
            with open(names_path) as f:
                raw = json.load(f)
            self.class_names = {int(k): v for k, v in raw.items()}
        else:
            self.class_names = {0: "cracked", 1: "uncracked"}

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
        t0 = time.perf_counter()

        tensor = preprocess_cls(img_bgr)
        logits = self.session.run([self.output_name], {self.input_name: tensor})[0][0]
        probs  = softmax(logits)

        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        top_idx    = int(np.argmax(probs))
        top_conf   = float(probs[top_idx])
        top_label  = self.class_names[top_idx]
        is_cracked = top_label == "cracked"

        severity = estimate_severity_cls(img_bgr, is_cracked)

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
            model_version="yolov8n-cls-sdnet2018",
            bounding_boxes=[],
            detection_mode="classification",
        )


# ---------------------------------------------------------------------------
# Mock Detector (development — no weights required)
# ---------------------------------------------------------------------------

class MockDetector:
    """
    Returns plausible fake results including synthetic bounding boxes.
    Automatically used when no ONNX weights are present.
    """

    def predict(self, img_bgr: np.ndarray) -> DetectionResult:
        import random
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(gray.mean())

        crack_prob = max(0.05, min(0.98, 1.0 - (mean_brightness / 255)))
        is_cracked = crack_prob > 0.5
        conf = crack_prob if is_cracked else (1.0 - crack_prob)

        boxes: list[BoundingBox] = []

        if is_cracked:
            h, w = img_bgr.shape[:2]
            num_boxes = random.randint(1, 3)
            for _ in range(num_boxes):
                x1 = random.randint(0, w // 2)
                y1 = random.randint(0, h // 2)
                x2 = random.randint(x1 + 20, min(x1 + w // 3, w))
                y2 = random.randint(y1 + 10, min(y1 + h // 3, h))
                box_conf = random.uniform(0.55, 0.97)
                area_frac = ((x2 - x1) * (y2 - y1)) / (h * w)
                boxes.append(BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=round(box_conf, 4),
                    severity=_box_severity(area_frac),
                ))

        severity = _aggregate_severity(boxes) if boxes else "none"

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
            inference_time_ms=round(__import__("random").uniform(8, 40), 2),
            model_version="mock-det-demo",
            bounding_boxes=boxes,
            detection_mode="mock",
        )


# ---------------------------------------------------------------------------
# Factory — auto-selects det → cls → mock
# ---------------------------------------------------------------------------

def load_detector():
    """
    Priority:
      1. Phase 2 detection model  (crack_detector_det.onnx)
      2. Phase 1 classification   (crack_detector.onnx)
      3. MockDetector             (development fallback)
    """
    if ONNX_DET.exists():
        print(f"[detector] Phase 2 — Loading detection ONNX from {ONNX_DET}")
        return CrackDetectorDet()
    if ONNX_CLS.exists():
        print(f"[detector] Phase 1 — Loading classification ONNX from {ONNX_CLS}")
        return CrackDetectorCls()
    print("[detector] WARNING: No ONNX model found — using MockDetector (Phase 2 mode).")
    print(f"[detector] For Phase 2: place model at {ONNX_DET}")
    print(f"[detector] For Phase 1: place model at {ONNX_CLS}")
    return MockDetector()
