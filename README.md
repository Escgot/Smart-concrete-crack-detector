# 🏗️ CrackScan — Concrete Crack Detector

> **AI-powered structural surface inspection and asset management.** > Upload a field photo to receive automated crack detection, severity classification, and a GPS-tagged inspection log designed for civil infrastructure monitoring.

**Tech Stack:** YOLOv8 · ONNX Runtime · FastAPI · Vanilla JS · Capacitor.js  
**Dataset:** SDNET2018 (56,000 labeled concrete images)  
**Compliance Standard:** Informed by EN 206 crack classification principles (Width-based)

---

## 📂 Project Structure

```text
crack-detector/
├── api/
│   ├── main.py         ← FastAPI routes (/detect, /inspections, /stats)
│   ├── detector.py     ← ONNX inference engine + width measurement algorithm
│   ├── database.py     ← SQLite inspection log (stores crack widths)
│   └── models/
│       ├── crack_detector_det.onnx   ← (add after training)
│       └── class_names.json          ← (add after training)
├── frontend/
│   ├── index.html      ← Full inspector UI (PWA + Capacitor bridge)
│   └── manifest.json   ← PWA manifest
├── notebooks/
│   └── train_colab_phase2.py  ← Training script (run in Google Colab)
├── tests/
│   └── test_detector.py ← Unit tests (preprocessing, width calc, severity mapping)
├── Dockerfile
├── requirements.txt
├── package.json        ← Capacitor JS dependencies
└── capacitor.config.ts ← Capacitor configuration for iOS/Android
```
---

## 🚀 Quick Start (Mock Model — No Training Needed)

You can demo the full API and frontend without training the model first. If no ONNX weights are found, the system automatically falls back to a deterministic `MockDetector`.

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

API docs at: http://localhost:8001/docs

---

## 🧠 Training the Production Model

### Step 1 — Open the Colab Notebook

Upload `notebooks/train_colab.py` to Google Colab.  
Enable GPU acceleration: Runtime → Change runtime type → T4 GPU (free).

### Step 2 — Fetch the Dataset

Option A (Recommended): Use Kaggle via API.  
```
# Upload kaggle.json to Colab, then:
kaggle datasets download -d arunrk7/surface-crack-detection -p /content/data --unzip
```

Option B: Search for the "SDNET2018 dataset" on UCI/Mendeley and download manually.

### Step 3 — Execute Training

Training takes ~25 minutes on T4 GPU.  
Run all cells in the notebook. Training takes approximately 25 minutes on a T4 GPU.

*Expected validation accuracy: >92% using YOLOv8n-cls.*

### Step 4 — Export and Deploy Weights

The notebook will generate and download the following:
- `best.pt` — PyTorch weights (keep as backup)
- `crack_detector.onnx` — Highly optimized weights for production deployment.
- `class_names.json` — class order index from training

Place both production files in the `api/models/`directory:
```
api/models/crack_detector.onnx
api/models/class_names.json
```

*Restart the API. It will automatically detect the weights and switch from the `MockDetector` to the live `CrackDetector` engine.*

---

## 📡 API Reference

### `POST /detect`
Submit an image to the inference engine and receive a localized assessment.

```bash
curl -X POST http://localhost:8001/detect \
  -F "file=@/path/to/concrete.jpg" \
  -F "latitude=36.8065" \
  -F "longitude=10.1815" \
  -F "location_name=Tunis Ring Road Pier 12" \
  -F "user_note=North face, moisture visible"
```

Standard Response:
```json
{
  "inspection_id": 42,
  "is_cracked": true,
  "confidence": 0.9732,
  "severity": "moderate",
  "severity_color": "#ff9833",
  "action": "Schedule professional inspection within 3 months.",
  "class_probabilities": {"cracked": 0.9732, "uncracked": 0.0268},
  "inference_time_ms": 18.4,
  "overlay_url": "/uploads/abc123_overlay.jpg",
  "model_version": "yolov8n-det-sdnet2018",
  "detection_mode": "detection",
  "box_count": 1,
  "scale_mm_per_px": 0.78,
  "bounding_boxes": [
    {
      "x1": 150, "y1": 100, "x2": 250, "y2": 300,
      "width": 100, "height": 200,
      "confidence": 0.9732,
      "severity": "moderate",
      "crack_width_px": 3.8,
      "crack_width_mm": 2.96
    }
  ]
}
```

### `GET /inspections`
Retrieve the GPS-tagged inspection log.
```
?limit=50&offset=0&severity=severe
```

### `GET /stats`
Fetch aggregated dashboard statistics (total logs, crack rates, severity distributions).

---

## 📐 Phase 3: Width-Based Severity Measurement

Phase 3 introduces a physically-grounded severity classification. Instead of mapping bounding-box area fractions, the system:
1. Skeletonizes the crack pixels inside each bounding box
2. Measures the perpendicular width at sample points along the skeleton
3. Takes the 95th percentile width in pixels
4. Multiplies by the `scale_mm_per_px` (Ground Sampling Distance) to find the width in mm
5. Maps to the EN 206 severity thresholds:

| Crack Width (mm) | Severity | Visual indicator | Recommended action |
|------------------|----------|-----------------|-------------------|
| 0                | None     | No crack detected | No action required |
| < 0.2 mm         | Hairline | Very fine crack   | Monitor, re-inspect in 6–12 months |
| 0.2–0.5 mm       | Moderate | Visible crack     | Schedule inspection within 3 months |
| > 0.5 mm         | Severe   | Wide crack        | Immediate structural assessment |

⚠️ *Disclaimer: Scale calibration depends on camera distance. The default 0.78 mm/px assumes SDNET2018 parameters (camera ~30 cm from surface).*

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

### Frontend (Browser / PWA)
The `frontend/index.html` file is entirely self-contained.  
It includes a PWA manifest (`manifest.json`) for installation on mobile browsers.
To deploy via Netlify, drag the `frontend/` folder into the Netlify dashboard.
Update the `API` constant in the JavaScript to point to your live backend URL.

### Phase 4: Mobile App (Capacitor.js)
The frontend is wrapped with Capacitor.js to compile as a native iOS/Android application.
To build the mobile app:
```bash
# 1. Install Node.js dependencies
npm install

# 2. Sync the frontend code into the native projects
npx cap sync

# 3. Open in Android Studio / Xcode to build
npx cap open android
# or
npx cap open ios
```
The Capacitor bridge automatically replaces the browser camera and geolocation APIs with native device plugins when running inside the app shell.

---

## 🧪 Testing

The test suite covers image preprocessing, mathematical stability (softmax validation), the severity heuristic, the MockDetector fallback, and all database operations. Model weights are not required to run tests.

```bash
pytest tests/ -v
# 30 passed — no model weights required
```

Tests cover: preprocessing, softmax stability, severity heuristic, MockDetector, all database operations.

---

## 📄 License

Distributed under the MIT License.
