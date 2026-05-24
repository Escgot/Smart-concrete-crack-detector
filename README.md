# CrackScan — Concrete Crack Detector

AI-powered concrete surface inspection. Upload a photo → get crack detection, severity classification, and a GPS-tagged inspection log.

**Stack:** YOLOv8 · ONNX Runtime · FastAPI · Vanilla JS  
**Dataset:** SDNET2018 (56,000 labelled concrete images)  
**Standard:** Informed by EN 206 crack classification principles

---

## Project Structure

```
crack-detector/
├── api/
│   ├── main.py         ← FastAPI routes (/detect, /inspections, /stats)
│   ├── detector.py     ← ONNX inference engine + severity heuristic
│   ├── database.py     ← SQLite inspection log
│   └── models/
│       ├── crack_detector.onnx   ← (add after training)
│       └── class_names.json      ← (add after training)
├── frontend/
│   └── index.html      ← Full inspector UI (no build step needed)
├── notebooks/
│   └── train_colab.py  ← Training script (run in Google Colab)
├── tests/
│   └── test_detector.py ← 30 tests, 100% pass
├── Dockerfile
└── requirements.txt
```

---

## Quick Start (with mock model — no training needed)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run API (uses MockDetector automatically if no ONNX file found)
uvicorn api.main:app --reload --port 8001

# 3. Open frontend
open frontend/index.html
# or serve it:
python -m http.server 3000 --directory frontend
```

API docs at: http://localhost:8001/docs

---

## Training the Real Model

### Step 1 — Open the Colab notebook

Upload `notebooks/train_colab.py` to Google Colab.  
Enable GPU: Runtime → Change runtime type → T4 GPU (free).

### Step 2 — Get the dataset

Option A (recommended): Kaggle  
```
# Upload kaggle.json to Colab, then:
kaggle datasets download -d arunrk7/surface-crack-detection -p /content/data --unzip
```

Option B: UCI/Mendeley — search "SDNET2018 dataset" and download manually.

### Step 3 — Run all cells

Training takes ~25 minutes on T4 GPU.  
Expected validation accuracy: **>92%** with YOLOv8n-cls.

### Step 4 — Download weights

The notebook will download:
- `best.pt` — PyTorch weights (keep as backup)
- `crack_detector.onnx` — for production
- `class_names.json` — class order from training

Place both in `api/models/`:
```
api/models/crack_detector.onnx
api/models/class_names.json
```

Restart the API — it will automatically switch from MockDetector to CrackDetector.

---

## API Reference

### `POST /detect`
Upload an image, get crack detection result.

```bash
curl -X POST http://localhost:8001/detect \
  -F "file=@/path/to/concrete.jpg" \
  -F "latitude=36.8065" \
  -F "longitude=10.1815" \
  -F "location_name=Tunis Ring Road Pier 12" \
  -F "user_note=North face, moisture visible"
```

Response:
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
  "model_version": "yolov8n-cls-sdnet2018"
}
```

### `GET /inspections`
```
?limit=50&offset=0&severity=severe
```

### `GET /stats`
Dashboard aggregates — total, cracked, uncracked, crack rate, by severity.

---

## Severity Classification

| Severity | Visual indicator | Recommended action |
|----------|-----------------|-------------------|
| None     | No crack detected | No action |
| Hairline | Very fine crack, < ~0.2 mm | Monitor, re-inspect in 6–12 months |
| Moderate | Visible crack, 0.2–0.5 mm | Schedule inspection within 3 months |
| Severe   | Wide crack, > ~0.5 mm | Immediate structural assessment |

Severity is estimated from the uploaded image using classical CV (edge density + connected component analysis). It is indicative only — always confirm with a professional structural assessment.

---

## Deployment

### Docker (backend)
```bash
docker build -t crackscan-api .
docker run -p 8001:8001 crackscan-api
```

### Render (free tier)
Connect your GitHub repo. Set start command:  
`uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### Frontend
The `frontend/index.html` is a single self-contained file.  
Deploy to Netlify by dragging the `frontend/` folder to the Netlify dashboard.  
Update the `API` constant in the JS to point to your deployed backend URL.

---

## Tests

```bash
pytest tests/ -v
# 30 passed — no model weights required
```

Tests cover: preprocessing, softmax stability, severity heuristic, MockDetector, all database operations.

---

## Extending the Project

**Phase 2 — Bounding box detection (more impressive CV):**  
Use Grounding DINO or SAM to auto-label crack regions → train YOLOv8-det → return bounding boxes drawn on the image.

**Phase 3 — Severity from crack width:**  
SDNET2018 images are at known scale. Measure connected component width in pixels → convert to mm → map to severity label. More physically grounded than the current heuristic.

**Phase 4 — Mobile app:**  
The frontend uses the standard browser camera API — it already works on mobile browsers. For an app store version, wrap in Capacitor.js (2 hours of work).

---

## CV / Portfolio Notes

- 30 tests, 100% passing — mention this
- ONNX export means **no GPU needed in production** — mention this
- MockDetector means the API and frontend are fully demoable without training — useful for live demos
- The inspection log + dashboard turns it from a classifier into an **asset management tool** — frame it this way on your CV

## License

MIT
