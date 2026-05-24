# 🏗️ CrackScan — Concrete Crack Detector (Phase 2)

> **AI-powered structural surface inspection and asset management.** > Upload a field photo to receive automated bounding-box localization, precise severity classification, and a GPS-tagged inspection log designed for civil infrastructure monitoring.

**Tech Stack:** YOLOv8-det · ONNX Runtime · FastAPI · Vanilla JS  
**Training Pipeline:** Grounding DINO · SAM (Zero-shot Auto-labeling)  
**Compliance Standard:** Informed by EN 206 crack classification principles  

---

## 📂 Project Structure

```text
crack-detector/
├── api/
│   ├── main.py         ← FastAPI routes (/detect, /inspections, /stats)
│   ├── detector.py     ← YOLOv8 NMS + Bounding Box decode + Severity heuristics
│   ├── database.py     ← SQLite inspection log (Phase 2 schema)
│   └── models/
│       ├── crack_detector_det.onnx   ← Phase 2: Detection weights (add after training)
│       └── crack_detector.onnx       ← Phase 1: Classification fallback
├── frontend/
│   └── index.html      ← Full inspector UI with Canvas Box Drawing
├── notebooks/
│   └── train_colab_phase2.py ← Detection training script (Path A & B)
├── tests/
│   └── test_detector.py ← Unit tests (preprocessing, NMS, DB logic)
├── Dockerfile
└── requirements.txt
```
---

## 🚀 Quick Start (Mock Model — No Training Needed)

You can demo the full API, UI, and synthetic bounding boxes without training the model first. The engine auto-selects the best available model (Phase 2 Det → Phase 1 Cls → Mock).

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the API server
uvicorn api.main:app --reload --port 8001

# 3. Open the frontend
open frontend/index.html
# Or serve it via HTTP:
python -m http.server 3000 --directory frontend
```

**API Documentation**: Available at http://localhost:8001/docs

---

## 🧠 Training the Production Model

### Step 1 — Open the Colab Notebook

Upload `notebooks/train_colab.py` to Google Colab.  
Enable GPU acceleration: Runtime → Change runtime type → T4 GPU (free).

### Step 2 — Fetch the Dataset

Path A: The Fast Path (Roboflow)
Use a pre-labeled detection dataset directly from Roboflow via the API.
This bypasses the need for manual labeling and moves straight into YOLOv8 training.

Path B: The AI-Engineering Path (Grounding DINO)
Keep the original SDNET2018 dataset and use a zero-shot vision model (Grounding DINO) to automatically scan all 20,000 positive images, identify the cracks, and generate YOLO-format `.txt` bounding box labels without human intervention.

### Step 3 — Export and Deploy Weights

Once training completes, the notebook exports an ONNX graph. Download the weights and place them in the `api/models/` directory:

Place both production files in the `api/models/` directory:
```
api/models/crack_detector.onnx
api/models/class_names.json
```

*Restart the API. The engine will detect `crack_detector_det.onnx` and automatically upgrade the UI and endpoints to Phase 2 mode.*

---

## 📡 API Reference

### `POST /detect`
Submit an image to the inference engine and receive localized bounding boxes.

```bash
curl -X POST http://localhost:8001/detect \
  -F "file=@/path/to/concrete.jpg" \
  -F "latitude=36.8065" \
  -F "longitude=10.1815"
```

Standard Response:
```json
{
  "inspection_id": 43,
  "is_cracked": true,
  "confidence": 0.9421,
  "severity": "moderate",
  "severity_color": "#ff9833",
  "action": "Schedule professional inspection within 3 months.",
  "class_probabilities": {"cracked": 0.9421, "uncracked": 0.0579},
  "inference_time_ms": 24.1,
  "overlay_url": "/uploads/xyz789_overlay.jpg",
  "model_version": "yolov8n-det-sdnet2018",
  "detection_mode": "detection",
  "box_count": 2,
  "bounding_boxes": [
    {
      "x1": 120, "y1": 45, "x2": 180, "y2": 310,
      "width": 60, "height": 265,
      "confidence": 0.9421,
      "severity": "moderate"
    },
    {
      "x1": 400, "y1": 200, "x2": 450, "y2": 280,
      "width": 50, "height": 80,
      "confidence": 0.7812,
      "severity": "hairline"
    }
  ]
}
```

### `GET /inspections`
Retrieve the GPS-tagged inspection log, now supporting bounding box arrays.
```
?limit=50&offset=0&mode=detection&severity=severe
```

### `GET /stats`
Fetch aggregated dashboard statistics (total logs, crack rates, severity distributions).

---

## 📐 Severity Classification Heuristic

| Severity | Visual indicator | Recommended action |
|----------|-----------------|-------------------|
| None     | 0 boxes detected | No action required |
| Hairline | < 0.5% of image area | Monitor, re-inspect in 6–12 months |
| Moderate | 0.5% – 4.0% of image area | Schedule inspection within 3 months |
| Severe   | > 4.0% of image area | Immediate structural assessment |

⚠️ Disclaimer: This automated heuristic is indicative only. It is designed to assist asset managers in triaging field data and should always be confirmed with a professional structural assessment.

---

## 🌐 Deployment Architecture

### Backend (Dockerized)
The backend is fully containerized and runs inference strictly on the CPU via ONNX Runtime.
```bash
docker build -t crackscan-api .
docker run -p 8001:8001 crackscan-api
```

### Cloud Deployment (e.g., Render Free Tier)
Connect your GitHub repository and set the start command to:  
`uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### Frontend (Static)
The `frontend/index.html` file is entirely self-contained. and renders the bounding box overlays seamlessly. Deploy to Netlify by dragging the `frontend/` folder into the dashboard.

---

## 🧪 Testing

The robust test suite validates image preprocessing, mathematical stability, NMS (Non-Maximum Suppression) box decoding, severity logic, the Mock fallback, and the Phase 2 SQLite database schema.
```bash
pytest tests/ -v
# Output: All tests passed (Model weights not required)
```

Tests cover: preprocessing, softmax stability, severity heuristic, MockDetector, all database operations.

---

## 📄 License

Distributed under the MIT License.
