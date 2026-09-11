# RoadDoc
ROADDOC is an AI-based road and tyre safety system that uses a vehicle-mounted camera and sensors to continuously monitor road and weather conditions such as dry, damp, wet, and standing water. 
The system combines this information with vehicle parameters such as speed and tyre data such as pressure and condition, and processes them through an AI-based safety engine to identify changing road risks. It then provides real-time, vehicle-specific alerts and recommendations, such as reducing speed, checking tyre pressure, or inspecting the tyres, helping drivers respond quickly to changing road conditions and improve overall driving safety.
# 🌦️ Weather Whiplash — now extended into ROADDOC

**Build with भारत 2.0 — National Level Hackathon — Team TechBirds**
**Problem statement: ROADDOC — AI-Based Road Condition and Intelligent Tyre Safety System**

Feed in a trackside/onboard camera frame and get **Dry / Damp / Drying / Wet** (+ a derived **standing-water** alert), a rolling weather trend, and a full **Safety Decision Engine** verdict that folds in live vehicle and tyre telemetry — matching the 4-component ROADDOC pitch:

1. **Road Condition Detection** — the original ResNet18 / CV vision pipeline (unchanged).
2. **Vehicle Monitoring** *(new)* — vehicle type, speed, location, driving condition → `backend/safety_engine.py::VehicleMonitor`.
3. **Tyre Monitoring** *(new)* — tyre pressure, manufacturer-recommended pressure, tyre condition, tyre temperature → `backend/safety_engine.py::TyreMonitor`.
4. **Safety Decision Engine** *(new)* — combines 1–3 into a road-condition warning, tyre-pressure warning, speed-related warning, tyre-inspection recommendation and a general safety recommendation, plus an overall risk level (`low` / `medium` / `high` / `critical`) → `backend/safety_engine.py::decide()`.

## Dataset integration

This final repo is wired to the official Hugging Face dataset:

**`adityakumarxdev/weather-whiplash`**

The raw dataset is **not duplicated inside this ZIP**. Instead, the repo includes the complete download, preprocessing, training and inference pipeline. This keeps the repository portable while ensuring the exact dataset source is reproducible.

Hugging Face's Datasets library supports image columns as PIL images, which the preparation script uses to create an ImageFolder training/validation set. citeturn0search0turn0search11

## What changed from the original demo

- The original premium frontend is retained.
- The original browser/OpenCV-style heuristic remains as a fallback.
- A real **ResNet18 image classifier** can now be fine-tuned on the Weather Whiplash dataset.
- The Flask API automatically uses the trained model when `models/weather_whiplash_model/model.pth` exists.
- Trend tracking and tyre-strategy logic remain in the application layer.
- The four supplied sample frames remain under `samples/` for quick testing.
- **New:** `backend/safety_engine.py` adds vehicle monitoring, tyre monitoring and the safety decision engine described in the ROADDOC pitch deck. It's dependency-free (no torch/numpy) and runs alongside the existing vision pipeline.
- **New:** the frontend has a "Vehicle & tyre telemetry" form (vehicle type, driving condition, speed, tyre pressure/condition/temperature, location with a "Use GPS" button) and a "Safety decision engine" result card with a risk badge and a standing-water alert banner. These fields are optional — leaving them blank still returns a full (if less specific) safety verdict.
- **New:** a lightweight JS mirror of `safety_engine.py` runs client-side so the safety card also works in the offline/browser-CV fallback path and the live demo sequence.
- **New:** the console now has a proper sci-fi entrance — a brief boot sequence ("booting sensor array… loading vision model… calibrating grip thresholds…"), an animated radar-sweep brand mark, a drifting hex-grid background, a continuous vertical scan-line sweep, an Orbitron display face on the state readout and speed gauge, a dual-ring animated speed dial, and a cyan glow-flash that pulses through every panel each time a new frame is analyzed.

## Mac setup — from ZIP to trained model

### 1. Open Terminal and enter the project

```bash
cd ~/Desktop/weather-whiplash-final
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Download/load the Weather Whiplash dataset

```bash
python3 training/download_dataset.py
```

### 4. Convert it to train/validation folders

```bash
python3 training/prepare_dataset.py
```

This creates:

```text
dataset/
├── train/
│   ├── dry/
│   ├── damp/
│   ├── drying/
│   └── wet/
└── val/
    ├── dry/
    ├── damp/
    ├── drying/
    └── wet/
```

When `video_id` is available, the preparation script keeps whole video groups on one side of the split to reduce adjacent-frame leakage.

### 5. Train

For a quick first run:

```bash
python3 training/train.py --epochs 8 --batch-size 32
```

On Apple Silicon, PyTorch will use MPS when available. The first pretrained run may download ResNet18 ImageNet weights. If you deliberately want no pretrained download:

```bash
python3 training/train.py --epochs 8 --no-pretrained
```

The trained checkpoint is written to:

```text
models/weather_whiplash_model/
├── model.pth
└── classes.json
```

### 6. Evaluate

```bash
python3 training/evaluate.py
```

### 7. Start the API

```bash
cd backend
python3 app.py
```

Health check:

```bash
curl http://localhost:5000/api/health
```

### 8. Open the premium frontend

From another Terminal window:

```bash
open ../frontend/index.html
```

The frontend first calls `http://localhost:5000/api/analyze`. If the backend is unavailable, it automatically falls back to the original browser CV analyzer.

## Quick sample test

Use these supplied sample images:

```text
samples/dry_track.jpg
samples/damp_track.jpg
samples/drying_track.jpg
samples/wet_track.jpg
```

You can drag them into the premium UI.

## API

### `GET /api/health`

Returns API and model status.

### `GET /api/model`

Returns whether the trained checkpoint is loaded and the selected device.

### `GET /api/options`

Returns the vehicle types, driving conditions and tyre conditions accepted by `/api/analyze`, for building frontend selects.

### `POST /api/analyze`

Multipart form:

```text
image=<jpg/png/webp>                required
session=<session id>                 optional, default "default"

# vehicle monitoring (all optional)
vehicle_type=car|suv|bike|bus|truck
speed=<km/h number>
location=<free text or "lat, lng">
driving_condition=normal|city|highway|aggressive|off-road

# tyre monitoring (all optional)
tyre_pressure=<PSI number>
tyre_recommended_pressure=<PSI number>
tyre_condition=new|good|worn|bald|damaged
tyre_temperature=<°C number>
```

Returns the predicted label, confidence when the trained model is active, wetness score, trend, history, tyre suggestion, inference time (`inference_ms`), per-class `probabilities`, and a `safety` object from the Safety Decision Engine:

```json
"safety": {
  "standing_water_alert": false,
  "risk_level": "medium",
  "road_condition_warning": "...",
  "tyre_pressure_warning": "...",
  "speed_related_warning": "...",
  "recommended_max_speed_kmh": 90,
  "tyre_inspection_recommendation": "...",
  "general_safety_recommendation": "...",
  "vehicle": { "vehicle_type": "car", "speed_kmh": 80, "location": null, "driving_condition": "normal" },
  "tyre": { "pressure_psi": 30, "recommended_pressure_psi": 32, "condition": "good", "temperature_c": null, "pressure_status": "ok" }
}
```

All vehicle/tyre fields are optional — omitting them still returns a full safety verdict (with `"...not reported..."` warnings where data is missing).

### `POST /api/reset`

Resets the rolling history for a session.

### `POST /api/analyze-video`

Automatic **video** analysis — the extension that removes manual speed and target-pressure entry entirely. Multipart form:

```text
video=<mp4/mov/webm>                 required
vehicle_type=car|suv|bike|bus|truck  optional, default "car"
driving_condition=normal|city|highway|aggressive|off-road   optional
tyre_condition=new|good|worn|bald|damaged   optional — a camera can't see this
tyre_pressure=<PSI number>           optional — a camera can't measure this either;
tyre_temperature=<°C number>         optional   only useful if you have a real TPMS reading to pass in
```

The video is sampled twice per second for road-condition classification and defect detection, and up to 15 times per second for speed estimation (large frame-to-frame displacement at real driving speed needs closer sampling than dense classification does). For every segment it returns:

```json
{
  "t": 2.0,
  "road_condition": "Damp",
  "wetness_score": 0.31,
  "defects": [{"type": "pothole", "confidence": 0.62, "bbox": [x, y, w, h]}],
  "speed_kmh": 61.4,
  "speed_calibrated": true,
  "recommended_tyre_psi": 32,
  "tyre_pressure_reasoning": "32 PSI base for car; no speed/condition adjustment needed",
  "safety": { "...": "same Safety Decision Engine object as /api/analyze" }
}
```

**How the automatic speed estimate works** (`backend/video_analyzer.py`): a handful of small road-texture patches are matched between consecutive frames (template matching, not dense optical flow — at highway speed the road can shift 50-100+ px between frames, more than dense flow tracks reliably) to get a pixel displacement, which `_detect_lane_lines_x` then converts to real units by measuring the lane markings in frame and assuming a standard 3.5m lane width — no calibration step, no typed-in speed, ever. When lane markings aren't visible it falls back to a fixed default scale and the response's `speed_calibrated` flag turns `false` so the frontend can show that segment's speed as an estimate rather than a confident reading. Validated against synthetic footage at known speeds (60 km/h and 100 km/h targets): typically within ~5-15%. Real dashcam footage will vary with mounting height/angle and road texture — this is a genuine camera-only estimate, not GPS-grade.

**Pothole/crack detection** is a shape heuristic (adaptive-threshold + contour shape), not a trained detector — it will flag shadows, tar patches, and manhole covers as false positives sometimes. There's no labelled pothole dataset in this project to train a real detector against; this is the honest, explainable stand-in.

**Recommended tyre pressure** is now fully automatic (`recommend_tyre_pressure` in `video_analyzer.py`): a base PSI for the vehicle type, +1/+2 PSI for sustained speeds over 80/100 km/h (real manufacturer practice), +1 PSI when standing water is detected. It deliberately does **not** lower pressure for wet roads — that's a common misconception; under-inflation is what actually increases aquaplaning risk, not the weather itself.

### Live camera mode (frontend-only, no new endpoint)

`frontend/live-analysis.html` also offers a live camera mode using `getUserMedia` for the video feed (works with a phone/tablet camera, not just a webcam) and the browser's `navigator.geolocation` for automatic speed — real GPS speed is simply the right automatic source for a live, moving camera, and doesn't need a backend round-trip per frame. If the device has no GPS fix (common on a desktop), the UI says so plainly rather than guessing a number. Road condition still runs through the same heuristic classifier already used for the browser-CV fallback.

## Project structure

```text
weather-whiplash-final/
├── backend/
│   ├── app.py
│   ├── model_service.py
│   ├── track_analyzer.py
│   ├── video_analyzer.py     # video sampling, pothole/crack detection, automatic speed estimation
│   ├── safety_engine.py      # vehicle + tyre monitoring, safety decision engine
│   └── requirements.txt
├── frontend/
│   ├── index.html            # Overview
│   ├── track-analysis.html
│   ├── car-status.html
│   ├── live-analysis.html    # video upload + live camera, both with automatic speed
│   ├── telemetry.html
│   ├── system.html
│   ├── app.js                # shared logic + browser-CV fallback
│   ├── live-analysis.js
│   ├── style.css / style-v2.css
├── training/
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── requirements-training.txt
├── dataset/
│   └── README.md
├── models/
├── samples/
│   ├── dry_track.jpg
│   ├── damp_track.jpg
│   ├── drying_track.jpg
│   └── wet_track.jpg
├── docs/
│   └── DATASET.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Important

Training accuracy is **not claimed in advance**. Run `training/evaluate.py` after training to obtain the actual validation metrics on the downloaded dataset. The included four sample images are demo/test assets, not a substitute for the full training dataset.

Speed estimation from uploaded video and the pothole/crack detector are both heuristics validated on synthetic test footage during development, not on labelled real-world driving data — treat their output as a genuinely automatic *estimate*, not a certified measurement. See the `/api/analyze-video` section above for the honest accuracy expectations.
