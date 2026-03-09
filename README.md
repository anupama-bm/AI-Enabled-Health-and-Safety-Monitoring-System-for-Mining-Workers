### *The AI Safety Monitor That Runs on What You Already Have*


---

## What Is This?

A real-time fatigue detector. A PPE compliance checker. A contactless vital-sign estimator.

Built for one webcam and a laptop. Runs locally. Works offline. Costs nothing to deploy beyond the hardware you already own.

This is not a proof of concept. This is not a demo. This is a system designed to run in a field environment — in the same conditions where people get hurt.

---

## The Problem It Solves

Every year, mining workers die or suffer serious injury from **preventable causes**:

- A worker falls asleep on shift, milliseconds before operating heavy machinery
- A helmet left in a truck instead of on a head
- Breathing anomalies that go unnoticed for hours in poorly ventilated environments

The technology to detect all of these has existed for years. It's just always required an enterprise contract.

Sentinel doesn't.

---

## What It Actually Does

###  Fatigue Detection — Because Microsleep Kills

The system tracks every blink. Literally every blink.

MediaPipe's **468-point face mesh** runs every frame, computing an **Eye Aspect Ratio (EAR)** — a geometric ratio that collapses when eyes close. This isn't motion detection. This is precise landmark geometry.

- **Individual blinks** are detected and logged with a self-calibrating threshold per person — because not every face is the same
- **Microsleep events** are flagged when eyes stay closed for more than ~0.6 seconds
- **Yawns** are tracked through Mouth Aspect Ratio (MAR), with duration and slope filtering to distinguish a yawn from someone speaking
- **PERCLOS** — the gold-standard drowsiness metric used in clinical fatigue research — is computed over a rolling 60-second window
- A **weighted fatigue score (0–100)** uses an intentionally slow exponential moving average, so the score builds with genuine fatigue rather than spiking on a single tired moment

###  PPE Compliance — No Helmet? You'll Know Immediately

A custom-trained YOLOv8 model runs on a background thread, checking for:

**Helmet · Goggles · Vest · Gloves · Boots**

Detection uses asymmetric ON/OFF thresholds with temporal smoothing.



<img width="1280" height="731" alt="image" src="https://github.com/user-attachments/assets/94a100c2-c62b-4121-b00e-040fa15ee6a0" />


Worker in frame, helmet nowhere in sight


<img width="1280" height="718" alt="image" src="https://github.com/user-attachments/assets/c50c0a32-173c-4b72-bb0b-21971cef7f08" />



There's a specific behavioral fix that matters: when a worker steps away from the camera, **all PPE states clear instantly**. The system behaves correctly in the situation where it matters most.

###  Respiratory Rate 

**Farneback dense optical flow** is computed on a chest region-of-interest placed just below the detected face. The vertical motion signal oscillates with the rise and fall of breathing.

A **30-second rolling FFT analysis** extracts breaths per minute from that oscillation signal.

Dense optical flow was chosen over sparse point tracking deliberately — it produces a more stable signal when the chest isn't perfectly still.

###  Heart Rate & SpO₂ — From a Camera

The green channel mean from a forehead ROI is sampled every frame. Over a 15-second window, **FFT analysis** extracts the dominant frequency corresponding to heart rate.

SpO₂ is approximated from the red-to-green channel variance ratio — an rPPG-inspired approach that works reasonably under stable lighting.

*The Known Limitations section of the code is completely upfront about where these estimates fall short. There's no overclaiming here.*

###  Live Dashboard

Flask serves a web dashboard over the local network. Video streams via **MJPEG**. Stats poll every **100ms**.

Multiple workers can be registered and monitored in separate sessions. Any device on the same network can watch.


<img width="1280" height="727" alt="image" src="https://github.com/user-attachments/assets/260a8b6b-21c5-4acf-9811-aebc40bc5bbe" />

Registering a new worker.


<img width="1280" height="728" alt="image" src="https://github.com/user-attachments/assets/322a8aeb-8345-4768-886d-77ce556fe80a" />

Active session: fatigue score, heart rate, respiration, SpO₂, PERCLOS, EAR, blink count, microsleeps, and yawns — all live, all in one view. Alerts log on the left as they fire.


<img width="1280" height="731" alt="image" src="https://github.com/user-attachments/assets/2c240f34-c760-4a2e-bf39-9e0c30232c2a" />

---

## How It Works

1. Register the worker — Name, ID, shift, gender. Done in under 10 seconds.

2. Webcam starts streaming — Every frame enters the pipeline locally. Nothing is uploaded. Nothing leaves the machine.

3. Face is mapped — MediaPipe places 468 landmarks on the face, isolating the eyes, mouth, and forehead. Every measurement that follows comes from this map.

4. Fatigue is tracked — Eye Aspect Ratio is computed every frame. Sustained closure = microsleep. Mouth shape and duration = yawn. PERCLOS accumulates over 60 seconds. One fatigue score. Rises with real tiredness, not noise.

5. Vital signs are estimated from the image — Chest movement via optical flow gives respiration rate. Green channel fluctuation on the forehead gives heart rate. Red-to-green variance ratio gives SpO₂. No wearable. No contact.

6. PPE runs on a separate thread — YOLOv8 checks for helmet, goggles, vest, gloves, and boots in parallel. Neither thread waits on the other.

7. Alerts fire instantly — Microsleep, missing helmet, low SpO₂ — logged the moment it happens, with a timestamp. Worker leaves the frame? All PPE states clear immediately.

8. Dashboard updates every 100ms — Any browser on the local network. Live video, all vitals, all PPE status, all alerts — one view, always current.

---

## The Model

`best.pt` was trained on Google Colab using YOLOv8.

Training data was assembled by merging multiple PPE datasets sourced from Roboflow Universe — covering different environments, different lighting conditions, and different worker demographics. The merged dataset was cleaned and relabelled where necessary.

**Six classes:** `person` · `helmet` · `goggles` · `vest` · `gloves` · `boots`

---

## Why These Specific Tools ##

MediaPipe Face Mesh — 468 facial landmarks, fast on CPU, accurate enough to catch a microsleep.
YOLOv8 — real-time PPE detection with no GPU needed. Trained on custom data without fighting the framework.
OpenCV Farneback — dense optical flow for breathing detection. More stable than tracking individual points.
SciPy rfft — extracts heart rate and breath rate from raw signal. Twice as fast as full FFT for rolling windows.
Flask — streams live video, serves the dashboard, fits in one file. Clone it and it runs.
Chart.js — live stats at 100ms refresh, no build step, no lag.

**On the choice of Flask over FastAPI:**

FastAPI can stream MJPEG. Flask can also stream MJPEG. The difference is that Flask does it with less ceremony around async generators.

More importantly: this entire application lives in one file. Routes, HTML templates, and application logic all coexist in `app.py`. For a system that needs to be cloned and running on-site in under five minutes, that's not a stylistic choice. That's a design requirement.

Flask-CORS handles cross-origin requests from the browser in a single line.

---

## Getting Started

**Requirements: Python 3.9+. A webcam.**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Open **http://localhost:5003**

To monitor from another device on the same network, replace `localhost` with your machine's local IP.

---

## Project Structure

```
miners/
├── app.py           # Entire backend: pipeline, API, and HTML templates
├── best.pt          # Custom-trained YOLOv8 PPE model
├── yolov8n.pt       # Base weights used during training
└── requirements.txt
```

No template folders. No build steps. No configuration files.

**Clone and run.**


A standard webcam. A laptop. Python.


*Built with MediaPipe · YOLOv8 · OpenCV · Flask · SciPy · Chart.js*
