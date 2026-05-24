"""
CrackScan — Inference Engine  (Phase 1 + 2 + 3)
================================================
Model priority at startup:

  Phase 3 → crack_detector_seg.onnx   YOLOv8n-seg  (pixel mask + EN 206 measurement)
  Phase 2 → crack_detector_det.onnx   YOLOv8n-det  (bounding box severity)
  Phase 1 → crack_detector.onnx       YOLOv8n-cls  (binary crack / no-crack)
  Fallback → MockDetector             deterministic, no weights required

Phase 3 pipeline
----------------
  image  →  letterbox 640×640  →  ONNX seg inference
         →  decode predictions  →  NMS
         →  reconstruct binary masks  →  calibrate (ArUco / LiDAR / EXIF / default)
         →  measure each mask (skeletonize + EDT)
         →  aggregate → EN 206 severity → DetectionResult
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
    _ONNX = True
except ImportError:                          # pragma: no cover
    _ONNX = False

# ── Paths ────────────────────────────────────────────────────

_MODELS = Path(__file__).parent / "models"

SEG_ONNX   = _MODELS / "crack_detector_seg.onnx"   # Phase 3
DET_ONNX   = _MODELS / "crack_detector_det.onnx"   # Phase 2
CLS_ONNX   = _MODELS / "crack_detector.onnx"       # Phase 1
NAMES_JSON = _MODELS / "class_names.json"

# ── Severity maps ────────────────────────────────────────────

_SEV_COLOR = {
    "none":     "#4caf50",
    "hairline": "#2196f3",
    "moderate": "#ff9800",
    "severe":   "#f44336",
}
_SEV_ACTION = {
    "none":     "No action required.",
    "hairline": "Monitor. Re-inspect in 6–12 months.",
    "moderate": "Schedule professional inspection within 3 months.",
    "severe":   "Immediate structural assessment required.",
}
_SEV_BGR = {
    "none":     (100, 200, 100),
    "hairline": (200, 200,   0),
    "moderate": (  0, 160, 255),
    "severe":   (  0,   0, 220),
}


def _conf_to_severity(conf: float) -> str:
    """Phase 1 heuristic: raw confidence → severity bucket."""
    if conf < 0.50: return "none"
    if conf < 0.70: return "hairline"
    if conf < 0.90: return "moderate"
    return "severe"


def _area_to_severity(area_frac: float) -> str:
    """Phase 2 heuristic: bounding-box area fraction → severity."""
    if area_frac < 0.005: return "hairline"
    if area_frac < 0.040: return "moderate"
    return "severe"


def _width_to_severity(max_width_mm: float) -> str:
    """Phase 3: EN 206 classification from measured crack width."""
    if max_width_mm <= 0.00: return "none"
    if max_width_mm <  0.20: return "hairline"
    if max_width_mm <= 0.50: return "moderate"
    return "severe"


# ── Data classes ─────────────────────────────────────────────

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    severity: str
    mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    measurement: Optional[dict] = None      # CrackMeasurement.to_dict()

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def area_fraction(self, img_w: int, img_h: int) -> float:
        return (self.width * self.height) / max(img_w * img_h, 1)

    def to_dict(self) -> dict:
        d = {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "width":  self.width,
            "height": self.height,
            "confidence": round(self.confidence, 4),
            "severity":   self.severity,
        }
        if self.measurement:
            d["measurement"] = self.measurement
        return d


@dataclass
class DetectionResult:
    is_cracked:          bool
    confidence:          float
    severity:            str
    action:              str
    class_probabilities: dict
    inference_time_ms:   float
    model_version:       str
    detection_mode:      str   # "segmentation" | "detection" | "classification" | "mock"
    bounding_boxes:      list[BoundingBox] = field(default_factory=list)
    # Phase 3 measurement fields
    measurement_aggregate:   dict  = field(default_factory=dict)
    calibration_method:      str   = "none"
    px_per_mm:               float = 0.0
    calibration_uncertainty: float = 0.0

    @property
    def severity_color(self) -> str:
        return _SEV_COLOR.get(self.severity, "#9e9e9e")

    def boxes_as_dicts(self) -> list[dict]:
        return [b.to_dict() for b in self.bounding_boxes]


# ── Image preprocessing ───────────────────────────────────────

def _letterbox(
    img: np.ndarray, size: int = 640
) -> tuple[np.ndarray, float, int, int]:
    """Pad image to square while preserving aspect ratio."""
    h0, w0   = img.shape[:2]
    scale    = size / max(h0, w0)
    nh, nw   = int(h0 * scale), int(w0 * scale)
    resized  = cv2.resize(img, (nw, nh))
    canvas   = np.full((size, size, 3), 114, np.uint8)
    py, px   = (size - nh) // 2, (size - nw) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, scale, px, py


def _to_tensor(img_bgr: np.ndarray, size: int = 640):
    """Letterbox + BGR→RGB + normalise + NCHW."""
    lb, scale, pad_x, pad_y = _letterbox(img_bgr, size)
    tensor = lb[:, :, ::-1].astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[np.newaxis, :])
    return tensor, scale, pad_x, pad_y


# ── NMS ──────────────────────────────────────────────────────

def _nms(boxes_cxywh: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45):
    """Pure-numpy Non-Maximum Suppression."""
    if len(boxes_cxywh) == 0:
        return []
    x1 = boxes_cxywh[:, 0] - boxes_cxywh[:, 2] / 2
    y1 = boxes_cxywh[:, 1] - boxes_cxywh[:, 3] / 2
    x2 = boxes_cxywh[:, 0] + boxes_cxywh[:, 2] / 2
    y2 = boxes_cxywh[:, 1] + boxes_cxywh[:, 3] / 2
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep  = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thr]
    return keep


# ── Segmentation mask reconstruction ─────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def _reconstruct_mask(
    coeffs:          np.ndarray,   # [32]
    protos:          np.ndarray,   # [32, 160, 160]
    box_xyxy_input:  tuple,        # (x1,y1,x2,y2) in input-space pixels
    orig_shape:      tuple,        # (H, W) of the original image
    input_size:      int = 640,
    proto_size:      int = 160,
) -> np.ndarray:
    """
    Reconstruct a binary segmentation mask from YOLOv8-seg outputs.

    The coefficient vector for each detection is combined with the shared
    prototype masks (a linear combination), then sigmoid-activated, cropped
    to the bounding box, and upsampled to original image resolution.
    """
    mask_logits = (coeffs @ protos.reshape(32, -1)).reshape(proto_size, proto_size)
    mask        = _sigmoid(mask_logits)

    ps = proto_size / input_size
    x1p = max(0,          int(box_xyxy_input[0] * ps))
    y1p = max(0,          int(box_xyxy_input[1] * ps))
    x2p = min(proto_size, int(box_xyxy_input[2] * ps))
    y2p = min(proto_size, int(box_xyxy_input[3] * ps))

    cropped = np.zeros((proto_size, proto_size), dtype=np.float32)
    if x2p > x1p and y2p > y1p:
        cropped[y1p:y2p, x1p:x2p] = mask[y1p:y2p, x1p:x2p]

    H, W = orig_shape
    full = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)
    return (full > 0.5).astype(np.uint8) * 255


# ─────────────────────────────────────────────────────────────
# Detectors
# ─────────────────────────────────────────────────────────────

class SegmentationDetector:
    """
    Phase 3: YOLOv8n-seg ONNX.
    Decodes segmentation masks, calibrates, runs measurement pipeline.

    ONNX output shapes:
      output0: [1, 4+nc+32, 8400]  — box + class + mask coefficients
      output1: [1, 32, 160, 160]   — prototype masks
    """

    MODEL_VERSION = "yolov8n-seg-sdnet2018-v3"
    INPUT_SIZE    = 640
    CONF_THR      = 0.25
    IOU_THR       = 0.45

    def __init__(self, model_path: Path):
        self._sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name
        out0_shape = self._sess.get_outputs()[0].shape
        # dim1 = 4 (box) + nc (classes) + 32 (mask coeffs)
        self._nc = max(1, out0_shape[1] - 36)

    def predict(
        self,
        img_bgr: np.ndarray,
        exif_bytes:        Optional[bytes] = None,
        lidar_distance_m:  Optional[float] = None,
        manual_px_per_mm:  Optional[float] = None,
    ) -> DetectionResult:
        from api.calibration import calibrate
        from api.measurement import measure_mask, aggregate_measurements

        t0 = time.perf_counter()
        H, W = img_bgr.shape[:2]
        tensor, scale, pad_x, pad_y = _to_tensor(img_bgr, self.INPUT_SIZE)

        out0, out1 = self._sess.run(None, {self._input_name: tensor})
        preds  = out0[0].T          # [8400, 4+nc+32]
        protos = out1[0]            # [32, 160, 160]

        nc     = self._nc
        scores = preds[:, 4:4 + nc].max(axis=1)
        boxes  = preds[:, :4]
        coeffs = preds[:, 4 + nc:]

        keep_mask = scores > self.CONF_THR
        if not keep_mask.any():
            infer_ms = (time.perf_counter() - t0) * 1000
            return self._empty_result(float(scores.max()), infer_ms)

        f_scores = scores[keep_mask]
        f_boxes  = boxes[keep_mask]
        f_coeffs = coeffs[keep_mask]

        keep = _nms(f_boxes, f_scores, self.IOU_THR)

        # Calibrate once per image
        calib = calibrate(img_bgr, exif_bytes, lidar_distance_m, manual_px_per_mm)

        bboxes:       list[BoundingBox] = []
        measurements: list              = []

        for idx in keep:
            cx, cy, bw, bh = f_boxes[idx]
            x1 = max(0, int((cx - bw / 2 - pad_x) / scale))
            y1 = max(0, int((cy - bh / 2 - pad_y) / scale))
            x2 = min(W, int((cx + bw / 2 - pad_x) / scale))
            y2 = min(H, int((cy + bh / 2 - pad_y) / scale))

            box_in = (cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2)
            seg_mask = _reconstruct_mask(f_coeffs[idx], protos, box_in, (H, W))

            m = measure_mask(seg_mask, calib.px_per_mm, calib.uncertainty)
            measurements.append(m)

            sev = (
                _width_to_severity(m.max_width_mm) if m.max_width_mm > 0
                else _area_to_severity(((x2 - x1) * (y2 - y1)) / (W * H))
            )

            bboxes.append(BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(f_scores[idx]),
                severity=sev,
                mask=seg_mask,
                measurement=m.to_dict(),
            ))

        agg  = aggregate_measurements(measurements)
        sev  = agg.get("en206_class", "none") if bboxes else "none"
        conf = max(b.confidence for b in bboxes) if bboxes else 0.0

        return DetectionResult(
            is_cracked=bool(bboxes),
            confidence=round(float(conf), 4),
            severity=sev,
            action=_SEV_ACTION[sev],
            class_probabilities={"cracked": round(float(conf), 4),
                                  "uncracked": round(1 - float(conf), 4)},
            inference_time_ms=round((time.perf_counter() - t0) * 1000, 2),
            model_version=self.MODEL_VERSION,
            detection_mode="segmentation",
            bounding_boxes=bboxes,
            measurement_aggregate=agg,
            calibration_method=calib.method,
            px_per_mm=round(calib.px_per_mm, 3),
            calibration_uncertainty=round(calib.uncertainty, 3),
        )

    def _empty_result(self, max_score: float, infer_ms: float) -> DetectionResult:
        return DetectionResult(
            is_cracked=False,
            confidence=round(float(max_score), 4),
            severity="none",
            action=_SEV_ACTION["none"],
            class_probabilities={"cracked": 0.0, "uncracked": 1.0},
            inference_time_ms=round(infer_ms, 2),
            model_version=self.MODEL_VERSION,
            detection_mode="segmentation",
            measurement_aggregate={
                "max_width_mm": 0.0, "total_length_mm": 0.0,
                "total_area_mm2": 0.0, "crack_count": 0,
                "en206_class": "none", "width_uncertainty_mm": 0.0,
            },
        )


class DetectionDetector:
    """
    Phase 2: YOLOv8n-det ONNX.
    Bounding box detection with area-based severity heuristic.

    ONNX output shape: [1, 4+nc, 8400]
    """

    MODEL_VERSION = "yolov8n-det-sdnet2018-v2"
    INPUT_SIZE    = 640
    CONF_THR      = 0.25
    IOU_THR       = 0.45

    def __init__(self, model_path: Path):
        self._sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name
        self._nc = max(1, self._sess.get_outputs()[0].shape[1] - 4)

    def predict(self, img_bgr: np.ndarray, **_) -> DetectionResult:
        t0 = time.perf_counter()
        H, W = img_bgr.shape[:2]
        tensor, scale, pad_x, pad_y = _to_tensor(img_bgr, self.INPUT_SIZE)

        raw = self._sess.run(None, {self._input_name: tensor})[0]
        preds  = raw[0].T                       # [8400, 4+nc]
        boxes  = preds[:, :4]
        scores = preds[:, 4:4 + self._nc].max(axis=1)

        keep_mask = scores > self.CONF_THR
        if not keep_mask.any():
            infer_ms = (time.perf_counter() - t0) * 1000
            return DetectionResult(
                is_cracked=False,
                confidence=float(scores.max()),
                severity="none",
                action=_SEV_ACTION["none"],
                class_probabilities={"cracked": 0.0, "uncracked": 1.0},
                inference_time_ms=round(infer_ms, 2),
                model_version=self.MODEL_VERSION,
                detection_mode="detection",
            )

        keep = _nms(boxes[keep_mask], scores[keep_mask], self.IOU_THR)
        f_boxes  = boxes[keep_mask]
        f_scores = scores[keep_mask]

        bboxes: list[BoundingBox] = []
        for idx in keep:
            cx, cy, bw, bh = f_boxes[idx]
            x1 = max(0, int((cx - bw / 2 - pad_x) / scale))
            y1 = max(0, int((cy - bh / 2 - pad_y) / scale))
            x2 = min(W, int((cx + bw / 2 - pad_x) / scale))
            y2 = min(H, int((cy + bh / 2 - pad_y) / scale))
            sev = _area_to_severity(((x2 - x1) * (y2 - y1)) / (W * H))
            bboxes.append(BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(f_scores[idx]),
                severity=sev,
            ))

        sev  = max(bboxes, key=lambda b: list(_SEV_ACTION).index(b.severity) if b.severity in _SEV_ACTION else 0).severity if bboxes else "none"
        conf = max(b.confidence for b in bboxes) if bboxes else 0.0

        return DetectionResult(
            is_cracked=bool(bboxes),
            confidence=round(conf, 4),
            severity=sev,
            action=_SEV_ACTION[sev],
            class_probabilities={"cracked": round(conf, 4),
                                  "uncracked": round(1 - conf, 4)},
            inference_time_ms=round((time.perf_counter() - t0) * 1000, 2),
            model_version=self.MODEL_VERSION,
            detection_mode="detection",
            bounding_boxes=bboxes,
        )


class ClassificationDetector:
    """
    Phase 1: YOLOv8n-cls ONNX. Binary crack / no-crack.

    ONNX output shape: [1, num_classes]
    """

    MODEL_VERSION = "yolov8n-cls-sdnet2018-v1"
    INPUT_SIZE    = 224

    def __init__(self, model_path: Path):
        self._sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name
        # Load class names
        if NAMES_JSON.exists():
            with open(NAMES_JSON) as f:
                raw = json.load(f)
            # class_names.json can be {0: "cracked", 1: "uncracked"} or list
            if isinstance(raw, dict):
                self._classes = [raw[str(i)] for i in range(len(raw))]
            else:
                self._classes = raw
        else:
            self._classes = ["cracked", "uncracked"]

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def predict(self, img_bgr: np.ndarray, **_) -> DetectionResult:
        t0 = time.perf_counter()

        img = cv2.resize(img_bgr, (self.INPUT_SIZE, self.INPUT_SIZE))
        tensor = img[:, :, ::-1].astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[np.newaxis, :])

        raw = self._sess.run(None, {self._input_name: tensor})[0][0]
        probs = self._softmax(raw)

        class_probs = {name: float(probs[i]) for i, name in enumerate(self._classes)}
        crack_prob  = class_probs.get("cracked", float(probs[0]))
        is_cracked  = crack_prob >= 0.5
        sev         = _conf_to_severity(crack_prob) if is_cracked else "none"

        return DetectionResult(
            is_cracked=is_cracked,
            confidence=round(crack_prob, 4),
            severity=sev,
            action=_SEV_ACTION[sev],
            class_probabilities=class_probs,
            inference_time_ms=round((time.perf_counter() - t0) * 1000, 2),
            model_version=self.MODEL_VERSION,
            detection_mode="classification",
        )


class MockDetector:
    """
    Deterministic fallback — no weights required.
    Results cycle through all severity levels for easy UI testing.
    """

    MODEL_VERSION = "mock-v1"

    _CYCLE = [
        (False, 0.04, "none"),
        (True,  0.62, "hairline"),
        (True,  0.81, "moderate"),
        (True,  0.96, "severe"),
    ]
    _counter = 0

    def predict(self, img_bgr: np.ndarray, **_) -> DetectionResult:
        t0 = time.perf_counter()
        is_cracked, conf, sev = self._CYCLE[MockDetector._counter % 4]
        MockDetector._counter += 1

        boxes: list[BoundingBox] = []
        if is_cracked:
            H, W = img_bgr.shape[:2]
            x1, y1 = int(W * 0.2), int(H * 0.2)
            x2, y2 = int(W * 0.8), int(H * 0.8)
            boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2,
                                     confidence=conf, severity=sev))

        return DetectionResult(
            is_cracked=is_cracked,
            confidence=conf,
            severity=sev,
            action=_SEV_ACTION[sev],
            class_probabilities={"cracked": conf, "uncracked": round(1 - conf, 4)},
            inference_time_ms=round((time.perf_counter() - t0) * 1000, 2),
            model_version=self.MODEL_VERSION,
            detection_mode="mock",
            bounding_boxes=boxes,
        )


# ── Loader ────────────────────────────────────────────────────

def load_detector():
    """
    Load the highest-capability available detector.
    Prints a startup banner so the user knows which mode is active.
    """
    if _ONNX:
        for path, cls, label in [
            (SEG_ONNX, SegmentationDetector,   "Phase 3 — Segmentation + Measurement"),
            (DET_ONNX, DetectionDetector,       "Phase 2 — Bounding Box Detection"),
            (CLS_ONNX, ClassificationDetector,  "Phase 1 — Classification"),
        ]:
            if path.exists():
                try:
                    det = cls(path)
                    print(f"[CrackScan] ✓ {label}  ({path.name})")
                    return det
                except Exception as e:
                    print(f"[CrackScan] ✗ Failed to load {path.name}: {e}")

    print("[CrackScan] ⚠  MockDetector active — place ONNX weights in api/models/ "
          "to enable real inference.")
    return MockDetector()


# ── Overlay helpers ───────────────────────────────────────────

def draw_segmentation_overlay(img_bgr: np.ndarray, boxes: list[BoundingBox]) -> np.ndarray:
    """Phase 3: tinted mask fill + bounding rect + measurement annotation."""
    from api.measurement import draw_segmentation_overlay as _draw
    return _draw(img_bgr, boxes)


def draw_boxes_overlay(img_bgr: np.ndarray, boxes: list[BoundingBox]) -> np.ndarray:
    """Phase 2: coloured bounding rectangles with severity label."""
    out = img_bgr.copy()
    for b in boxes:
        color = _SEV_BGR.get(b.severity, (200, 200, 200))
        cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), color, 2)
        label = f"{b.severity.upper()} {b.confidence:.0%}"
        lx, ly = b.x1, max(b.y1 - 6, 14)
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_severity_overlay(img_bgr: np.ndarray) -> np.ndarray:
    """Phase 1 fallback: edge contour highlight for classified-as-cracked images."""
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges   = cv2.Canny(gray, 50, 150)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    out     = img_bgr.copy()
    out[dilated > 0] = (0, 80, 255)
    return out
