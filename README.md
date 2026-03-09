# AI-Enabled-Health-and-Safety-Monitoring-System-for-Mining Workers


Mining kills people. Not because the technology to prevent it doesn't exist — but because that technology has historically required expensive hardware, wearables, and cloud infrastructure that most mining operations can't justify.
This project does real-time fatigue monitoring, PPE compliance checking, and contactless vital sign estimation using nothing but a standard webcam and a laptop. It runs locally, works offline, and costs nothing to deploy beyond the hardware you already have.


What It Does


Fatigue Detection
The system tracks eye closure using MediaPipe's 468-point face mesh and computes Eye Aspect Ratio (EAR) every frame. It detects individual blinks (with a self-calibrating threshold per person), flags microsleep events when eyes stay closed for more than ~0.6 seconds, tracks yawns through Mouth Aspect Ratio (MAR) with duration and slope filtering to avoid false positives from talking, and maintains a PERCLOS score over a 60-second rolling window. These feed into a weighted fatigue score (0–100) that uses an intentionally slow EMA so the score rises gradually with genuine fatigue rather than spiking on a single yawn.


PPE Compliance
A custom YOLOv8 model runs on a background thread and checks for helmets, goggles, vests, gloves, and boots in real time. Detection uses temporal smoothing with asymmetric ON/OFF thresholds to prevent flickering. When the worker steps away from the camera, all PPE states clear instantly rather than waiting for the inference buffer to drain — a specific fix that makes the system behave correctly in practice.


Respiratory Rate
Farneback dense optical flow is computed on a chest ROI placed just below the detected face. The vertical motion signal oscillates with breathing and is FFT-analysed over a 30-second rolling buffer to extract breaths per minute.


Heart Rate and SpO₂ (rPPG-inspired)
The green channel mean from a forehead ROI is sampled and FFT-analysed over 15 seconds to estimate heart rate. SpO₂ is approximated from the red-to-green channel variance ratio. These are estimates that work reasonably under stable lighting — the Known Limitations section is upfront about where they fall short.


Live Dashboard
Flask serves a web dashboard over the local network. Video streams via MJPEG. Stats poll every 100ms. Multiple workers can be registered and monitored in separate sessions.

The best.pt model was trained on Google Colab using YOLOv8. The training data was assembled by merging multiple PPE datasets sourced from Roboflow Universe — covering different environments, lighting conditions, and worker demographics — to improve generalisation across real-world scenarios. The merged dataset was cleaned, relabelled where necessary.The model handles six classes: person, helmet, goggles, vest, gloves, and boots, with label matching.


Face analysis — MediaPipe Face Mesh: 468 landmarks vs dlib's 68, faster on CPU, better partial occlusion handling.
PPE detection — YOLOv8 (Ultralytics): Runs on CPU at real-time speeds, mature training pipeline for custom datasets.
Optical flow — OpenCV Farneback: Dense flow gives a more stable respiration signal than sparse point tracking.
Signal processing — SciPy rfft: Real-input FFT is twice as fast as full FFT for this use case.
Web server — Flask: MJPEG streaming via generator functions works cleanly; single-file deployment.
Charts — Chart.js: No build step, handles 100ms polling without animation lag with animation: false.

Flask was chosen because the main requirement was to stream live video ,Flask handles MJPEG streaming. FastAPI can do the same thing but requires more ceremony around async generators. The second reason is that this entire application lives in one file. Flask is designed for that. The routes, the HTML templates, and the application logic all coexist in app.py, for a system meant to be cloned and run immediately on-site, that matters.
Flask-CORS was added as a single line to handle cross-origin requests from the browser polling the API endpoints. The whole setup is minimal, predictable, and easy for anyone else to read and deploy — which is exactly what a monitoring system running in a field environment needs to be.


Getting Started

Requirements: Python 3.9+, a webcam.

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt<br>
python app.py<br>
Open http://localhost:5003. <br>
To access from another device on the same network, replace localhost with your machine's local IP.


miners/<br>
|--app.py          # Entire backend: pipeline, API, and HTML templates<br>
|--best.pt         # Custom-trained YOLOv8 PPE model<br>
|--yolov8n.pt      # Base weights used during training<br>
|--requirements.txt<br>
<br>
Everything runs from a single file. No template folders, no build steps, no configuration files. Clone and run.

