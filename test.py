import time
import math
import threading
import json
import uuid
import os
from dataclasses import dataclass, field
from typing import Tuple, Deque, Dict, Optional, Any, List
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
from flask import Flask, render_template_string, Response, jsonify, request
from flask_cors import CORS

# YOLOv8 imports
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  Warning: ultralytics not installed. PPE detection disabled.")
    print("   Install with: pip install ultralytics")

# Configuration
EAR_THRESHOLD = 0.20
CONSEC_FRAMES_MICROSLEEP = 48
MAR_YAWN_THRESHOLD = 0.52
MAR_SUSTAIN_TIME = 0.7
MAR_SLOPE_WINDOW = 0.8
MAR_SLOPE_MAX = 0.8
YAWN_MIN_SEPARATION = 2.0
PERCLOS_WINDOW_SEC = 60.0

# OPTIMIZED BLINK DETECTION SETTINGS
BLINK_EAR_THRESHOLD = 0.23
BLINK_MIN_FRAMES = 2
BLINK_MAX_FRAMES = 12
BLINK_MIN_SEPARATION_FRAMES = 5

RESP_BUFFER_SEC = 30.0
FLOW_ROI_REL = (0.35, 0.6, 0.30, 0.25)
RESP_MIN_BPM = 6
RESP_MAX_BPM = 30
RPPG_WINDOW_SEC = 15.0
HR_MIN_BPM = 40
HR_MAX_BPM = 180
SPO2_WINDOW_SEC = 15.0

# OPTIMIZED PPE SETTINGS
PROCESS_EVERY_N_FRAMES = 2
frame_counter = 0
PPE_HISTORY_SEC = 1.5
PPE_FPS = 30.0
# Add after line 59:
PPE_PROCESS_EVERY_N_FRAMES = 5  # Only check PPE every 5 frames
PPE_ON_RATIO = 0.60
PPE_OFF_RATIO = 0.40

# CRITICAL: Faster EMA for responsive blink detection
EAR_EMA_ALPHA = 0.95
FATIGUE_EMA_ALPHA = 0.12
WEIGHT_EYE = 0.45
WEIGHT_YAWN = 0.30
WEIGHT_RESP = 0.25

ALERT_PERCLOS = 0.35
ALERT_FATIGUE = 75.0
ALERT_HR_LOW = 45.0
ALERT_HR_HIGH = 120.0
ALERT_SPO2_LOW = 94.0
ALERT_RESP_LOW = 8.0
ALERT_RESP_HIGH = 25.0
ALERT_MICROSLEEP_COUNT = 1
ALERT_YAWN_COUNT = 4

# PPE Detection Configuration
PPE_MODEL_PATH = 'yolov8n.pt'
PPE_CONFIDENCE_THRESHOLD = 0.45
PPE_ALERT_ENABLED = True

# MediaPipe indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14
MOUTH_LEFT_IDX = 78
MOUTH_RIGHT_IDX = 308
FOREHEAD_POINTS = [10, 338, 297, 332]
NOSE_IDX = 1


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bandpass_peak_freq(signal: np.ndarray, fs: float, lo_hz: float, hi_hz: float) -> Optional[float]:
    if len(signal) < 4:
        return None
    x = signal - np.mean(signal)
    n = len(x)
    yf = np.abs(rfft(x))
    xf = rfftfreq(n, 1.0 / fs)
    mask = (xf >= lo_hz) & (xf <= hi_hz)
    if not np.any(mask):
        return None
    yf_masked = yf.copy()
    yf_masked[~mask] = 0.0
    idx = int(np.argmax(yf_masked))
    freq = float(xf[idx])
    if freq <= 0:
        return None
    return freq


def eye_aspect_ratio(landmarks, eye_idx_list, frame_w, frame_h) -> float:
    coords = []
    for idx in eye_idx_list:
        lm = landmarks[idx]
        coords.append((lm.x * frame_w, lm.y * frame_h))
    p1, p2, p3, p4, p5, p6 = coords
    vert1 = euclid(p2, p6)
    vert2 = euclid(p3, p5)
    horiz = euclid(p1, p4)
    return safe_div((vert1 + vert2), (2.0 * horiz))


def mouth_aspect_ratio(landmarks, frame_w, frame_h) -> float:
    up = landmarks[UPPER_LIP_IDX]
    low = landmarks[LOWER_LIP_IDX]
    left = landmarks[MOUTH_LEFT_IDX]
    right = landmarks[MOUTH_RIGHT_IDX]
    up_pt = (up.x * frame_w, up.y * frame_h)
    low_pt = (low.x * frame_w, low.y * frame_h)
    left_pt = (left.x * frame_w, left.y * frame_h)
    right_pt = (right.x * frame_w, right.y * frame_h)
    vert = euclid(up_pt, low_pt)
    width = euclid(left_pt, right_pt)
    return safe_div(vert, width)


class AlertManager:
    def __init__(self):
        self.alerts_history = deque(maxlen=100)
        self.lock = threading.Lock()
        self.last_alerts = {}

    def trigger(self, name: str, message: str):
        with self.lock:
            timestamp = datetime.now().strftime('%H:%M:%S')
            alert_key = f"{name}_{message}"

            current_time = time.time()
            if alert_key not in self.last_alerts or (current_time - self.last_alerts[alert_key]) > 5.0:
                alert_data = {
                    'timestamp': timestamp,
                    'name': name,
                    'message': message
                }
                self.alerts_history.append(alert_data)
                self.last_alerts[alert_key] = current_time
                print(f"[ALERT] {timestamp} {name}: {message}")
                return True
            return False

    def get_alerts(self):
        with self.lock:
            return list(self.alerts_history)

    def clear_alerts(self):
        with self.lock:
            self.alerts_history.clear()
            self.last_alerts.clear()


class SimplifiedBlinkDetector:
    """
    SIMPLIFIED & RELIABLE blink detection:
    - Basic EAR threshold with adaptive baseline
    - Simple frame counting
    - Minimal validation for maximum reliability
    """

    def __init__(self):
        self.blink_count = 0
        self.closed_frames = 0
        self.frames_since_last_blink = 0

        # Baseline tracking
        self.ear_history = deque(maxlen=100)
        self.baseline_ear = 0.30
        self.threshold = BLINK_EAR_THRESHOLD

        # State tracking
        self.was_closed = False
        self.last_blink_time = time.time()

    def _update_baseline(self, ear: float):
        """Update baseline with filtering"""
        if 0.20 < ear < 0.45:
            self.ear_history.append(ear)

        if len(self.ear_history) >= 30:
            # Use 75th percentile for stable baseline
            self.baseline_ear = float(np.percentile(list(self.ear_history), 75))
            # Dynamic threshold: 65% of baseline
            self.threshold = max(0.18, min(0.25, self.baseline_ear * 0.65))

    def detect(self, ear_left: float, ear_right: float, timestamp: float) -> bool:
        """
        SIMPLE & RELIABLE blink detection
        """
        self.frames_since_last_blink += 1

        # Average EAR
        ear = (ear_left + ear_right) / 2.0

        # Update baseline
        self._update_baseline(ear)

        blink_detected = False

        # Simple threshold-based detection
        is_closed = ear < self.threshold

        if is_closed:
            self.closed_frames += 1
        else:
            # Eye opened - check if this was a blink
            if self.was_closed and self.closed_frames > 0:
                # Validate blink
                if (BLINK_MIN_FRAMES <= self.closed_frames <= BLINK_MAX_FRAMES and
                        self.frames_since_last_blink >= BLINK_MIN_SEPARATION_FRAMES):
                    self.blink_count += 1
                    blink_detected = True
                    self.last_blink_time = timestamp
                    self.frames_since_last_blink = 0
                    print(f"[BLINK] #{self.blink_count} - Duration: {self.closed_frames}f, EAR: {ear:.3f}")

            # Reset closed frames
            self.closed_frames = 0

        self.was_closed = is_closed
        return blink_detected

    def get_count(self) -> int:
        return self.blink_count


class YOLOPPEDetector:
    """OPTIMIZED PPE Detection with frame skipping"""

    def __init__(self, model_path: str):
        self.model = None
        self.is_available = False
        self.detection_history = {
            'person': deque(maxlen=8),
            'helmet': deque(maxlen=8),
            'goggles': deque(maxlen=8),
            'vest': deque(maxlen=8),
            'gloves': deque(maxlen=8),
            'boots': deque(maxlen=8)
        }
        self.current_state = {
            'person': False,
            'helmet': False,
            'goggles': False,
            'vest': False,
            'gloves': False,
            'boots': False
        }

        if YOLO_AVAILABLE:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load YOLOv8 model with optimized settings"""
        try:
            possible_paths = [
                model_path,
                'ppe_dataset/train/weights/best.pt',
                'ppe_dataset/weights/best.pt',
                'best.pt'
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    print(f"✓ Loading YOLO model from: {path}")
                    self.model = YOLO(path)
                    self.model.overrides['verbose'] = False
                    self.is_available = True
                    print("✓ YOLO PPE detection model loaded successfully")
                    return

            print(f"⚠️  YOLO model not found. Tried paths:")
            for path in possible_paths:
                print(f"   - {path}")
            print("   Place your trained model (best.pt) in the project directory")

        except Exception as e:
            print(f"⚠️  Failed to load YOLO model: {e}")
            self.is_available = False

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """OPTIMIZED: Detect PPE equipment"""
        if not self.is_available:
            return self.current_state.copy()

        try:
            # OPTIMIZED: Reduced image size for faster inference
            results = self.model(
                frame,
                conf=PPE_CONFIDENCE_THRESHOLD,
                verbose=False,
                device='cpu',
                half=False,
                imgsz=320  # Small size for speed
            )[0]

            frame_detections = {
                'person': False,
                'helmet': False,
                'goggles': False,
                'vest': False,
                'gloves': False,
                'boots': False
            }

            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    class_name = results.names[cls_id].lower()

                    if 'person' in class_name or 'worker' in class_name:
                        frame_detections['person'] = True
                    elif 'helmet' in class_name or 'hardhat' in class_name or 'hard-hat' in class_name:
                        frame_detections['helmet'] = True
                    elif 'goggles' in class_name or 'glasses' in class_name or 'safety-glasses' in class_name:
                        frame_detections['goggles'] = True
                    elif 'vest' in class_name or 'jacket' in class_name or 'safety-vest' in class_name:
                        frame_detections['vest'] = True
                    elif 'glove' in class_name or 'gloves' in class_name:
                        frame_detections['gloves'] = True
                    elif 'boot' in class_name or 'boots' in class_name or 'shoes' in class_name:
                        frame_detections['boots'] = True

            # Update history
            for item in self.detection_history.keys():
                self.detection_history[item].append(1 if frame_detections[item] else 0)

            # Update state with hysteresis
            for item in self.current_state.keys():
                if len(self.detection_history[item]) >= 5:
                    ratio = sum(self.detection_history[item]) / len(self.detection_history[item])

                    if not self.current_state[item]:
                        if ratio >= PPE_ON_RATIO:
                            self.current_state[item] = True
                    else:
                        if ratio <= PPE_OFF_RATIO:
                            self.current_state[item] = False

            return self.current_state.copy()

        except Exception as e:
            print(f"⚠️  PPE detection error: {e}")
            return self.current_state.copy()

    def draw_detections(self, frame: np.ndarray, detections: Dict[str, bool]) -> np.ndarray:
        """Draw PPE status on frame"""
        if not self.is_available:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()

        panel_h = 180
        cv2.rectangle(overlay, (10, 10), (280, panel_h), (20, 20, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "PPE STATUS", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        items = [
            ('Person', 'person', (16, 185, 129)),
            ('Helmet', 'helmet', (239, 68, 68)),
            ('Goggles', 'goggles', (251, 146, 60)),
            ('Vest', 'vest', (59, 130, 246)),
            ('Gloves', 'gloves', (168, 85, 247)),
            ('Boots', 'boots', (236, 72, 153))
        ]

        y_pos = 60
        for label, key, color in items:
            status = "✓" if detections.get(key, False) else "✗"
            status_color = (0, 255, 0) if detections.get(key, False) else (0, 0, 255)

            cv2.putText(frame, f"{status} {label}", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            y_pos += 20

        return frame


@dataclass
class VideoProcessor:
    ear_threshold: float = EAR_THRESHOLD
    ear_consec_frames_thresh: int = CONSEC_FRAMES_MICROSLEEP
    mar_yawn_threshold: float = MAR_YAWN_THRESHOLD
    mp_face_mesh: mp.solutions.face_mesh.FaceMesh = field(init=False)
    eye_closed_frames: int = 0
    microsleep_count: int = 0
    last_yawn_time: float = 0.0
    yawn_count: int = 0
    ear_ema: Optional[float] = None
    mar_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=240))
    perclos_window: Deque[bool] = field(default_factory=lambda: deque(maxlen=int(PERCLOS_WINDOW_SEC * 30)))

    blink_detector: SimplifiedBlinkDetector = field(init=False)
    ppe_detector: YOLOPPEDetector = field(init=False)

    prev_gray: Optional[np.ndarray] = None
    resp_motion_buffer: Deque[Tuple[float, float]] = field(default_factory=lambda: deque())
    chest_roi_abs: Optional[Tuple[int, int, int, int]] = None
    resp_last_est_bpm: Optional[float] = None
    rppg_green_buffer: Deque[Tuple[float, float]] = field(default_factory=lambda: deque())
    rppg_red_buffer: Deque[Tuple[float, float]] = field(default_factory=lambda: deque())
    rppg_last_hr: Optional[float] = None
    rppg_last_spo2: Optional[float] = None

    # CRITICAL: Frame counters for intelligent skipping
    frame_count: int = 0
    last_face_landmarks: Any = None

    def __post_init__(self):
        # OPTIMIZED: Reduced tracking confidence for speed
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,  # Reduced from 0.6
            min_tracking_confidence=0.5)  # Reduced from 0.6

        self.blink_detector = SimplifiedBlinkDetector()
        self.ppe_detector = YOLOPPEDetector(PPE_MODEL_PATH)

    def _update_mar_history(self, mar_value: float, now: float):
        self.mar_history.append((now, mar_value))
        while self.mar_history and (now - self.mar_history[0][0] > 5.0):
            self.mar_history.popleft()

    def _compute_mar_slope(self) -> float:
        pts = list(self.mar_history)
        if len(pts) < 2:
            return 0.0
        now = pts[-1][0]
        cutoff = now - MAR_SLOPE_WINDOW
        relevant = [p for p in pts if p[0] >= cutoff]
        if len(relevant) < 2:
            relevant = pts[-2:]
        t0, m0 = relevant[0]
        t1, m1 = relevant[-1]
        dt = (t1 - t0) if (t1 - t0) > 1e-6 else 1e-6
        return (m1 - m0) / dt

    def _mar_sustained_time(self) -> float:
        if not self.mar_history:
            return 0.0
        now = self.mar_history[-1][0]
        dur = 0.0
        for t, m in reversed(self.mar_history):
            if m >= self.mar_yawn_threshold:
                dur = now - t
            else:
                break
        return dur

    def _update_perclos(self, ear: Optional[float]):
        closed = (ear is not None) and (ear < self.ear_threshold)
        self.perclos_window.append(closed)

    def compute_perclos(self) -> float:
        if not self.perclos_window:
            return 0.0
        return float(sum(self.perclos_window) / len(self.perclos_window))

    def _compute_chest_roi_from_face(self, landmarks, frame_w: int, frame_h: int):
        ys = [lm.y for lm in landmarks]
        xs = [lm.x for lm in landmarks]
        max_y = max(ys) * frame_h
        min_x = min(xs) * frame_w
        max_x = max(xs) * frame_w
        face_w = max_x - min_x
        roi_w = int(max(face_w * 1.0, frame_w * 0.25))
        roi_x = int(max(0, (min_x + max_x) / 2 - roi_w / 2))
        roi_y = int(min(frame_h - 1, max_y + 10))
        roi_h = int(min(frame_h - roi_y - 10, frame_h * 0.22))
        if roi_h <= 10 or roi_w <= 10:
            rx, ry, rw, rh = FLOW_ROI_REL
            roi_x = int(frame_w * rx)
            roi_y = int(frame_h * ry)
            roi_w = int(frame_w * rw)
            roi_h = int(frame_h * rh)
        self.chest_roi_abs = (roi_x, roi_y, roi_w, roi_h)

    def _append_resp_motion(self, v_motion: float):
        now = time.time()
        self.resp_motion_buffer.append((now, float(v_motion)))
        while self.resp_motion_buffer and (now - self.resp_motion_buffer[0][0] > RESP_BUFFER_SEC):
            self.resp_motion_buffer.popleft()

    def estimate_resp_bpm(self) -> Optional[float]:
        if len(self.resp_motion_buffer) < 8:
            return None
        times = np.array([t for t, _ in self.resp_motion_buffer])
        vals = np.array([v for _, v in self.resp_motion_buffer])
        vals = vals - np.mean(vals)
        duration = times[-1] - times[0]
        if duration < 6.0:
            return None
        fs = len(vals) / duration
        t_uniform = np.linspace(times[0], times[-1], len(vals))
        vals_u = np.interp(t_uniform, times, vals)
        yf = np.abs(rfft(vals_u))
        xf = rfftfreq(len(vals_u), 1.0 / fs)
        min_hz = RESP_MIN_BPM / 60.0
        max_hz = RESP_MAX_BPM / 60.0
        mask = (xf >= min_hz) & (xf <= max_hz)
        if not np.any(mask):
            return None
        yf_masked = yf.copy()
        yf_masked[~mask] = 0.0
        idx = int(np.argmax(yf_masked))
        freq = float(xf[idx])
        if freq <= 0:
            return None
        bpm = freq * 60.0
        self.resp_last_est_bpm = float(bpm)
        return float(bpm)

    def _compute_forehead_roi(self, landmarks, frame_w: int, frame_h: int):
        xs = np.array([landmarks[p].x for p in FOREHEAD_POINTS]) * frame_w
        ys = np.array([landmarks[p].y for p in FOREHEAD_POINTS]) * frame_h
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        w = int(frame_w * 0.18)
        h = int(frame_h * 0.08)
        x = max(0, cx - w // 2)
        y = max(0, int(cy - h * 0.6))
        return x, y, w, h

    def _append_rppg(self, frame, landmarks):
        h, w = frame.shape[:2]
        try:
            x, y, ww, hh = self._compute_forehead_roi(landmarks, w, h)
        except Exception:
            return
        roi = frame[y:y + hh, x:x + ww]
        if roi.size == 0:
            return
        mean_g = float(np.mean(roi[:, :, 1]))
        mean_r = float(np.mean(roi[:, :, 2]))
        now = time.time()
        self.rppg_green_buffer.append((now, mean_g))
        self.rppg_red_buffer.append((now, mean_r))
        while self.rppg_green_buffer and (now - self.rppg_green_buffer[0][0] > RPPG_WINDOW_SEC):
            self.rppg_green_buffer.popleft()
        while self.rppg_red_buffer and (now - self.rppg_red_buffer[0][0] > SPO2_WINDOW_SEC):
            self.rppg_red_buffer.popleft()

    def estimate_hr_from_rppg(self) -> Optional[float]:
        if len(self.rppg_green_buffer) < 8:
            return None
        times = np.array([t for t, _ in self.rppg_green_buffer])
        vals = np.array([v for _, v in self.rppg_green_buffer])
        vals = vals - np.mean(vals)
        duration = times[-1] - times[0]
        if duration < 8.0:
            return None
        fs = len(vals) / duration
        t_uniform = np.linspace(times[0], times[-1], len(vals))
        vals_u = np.interp(t_uniform, times, vals)
        lo_hz = HR_MIN_BPM / 60.0
        hi_hz = HR_MAX_BPM / 60.0
        freq = bandpass_peak_freq(vals_u, fs, lo_hz, hi_hz)
        if freq is None:
            return None
        hr = float(freq * 60.0)
        self.rppg_last_hr = hr
        return hr

    def estimate_spo2_from_rgb(self) -> Optional[float]:
        if len(self.rppg_red_buffer) < 8 or len(self.rppg_green_buffer) < 8:
            return None
        t0 = max(self.rppg_red_buffer[0][0], self.rppg_green_buffer[0][0])
        t1 = min(self.rppg_red_buffer[-1][0], self.rppg_green_buffer[-1][0])
        if t1 - t0 < 6.0:
            return None

        def extract(buf):
            times = np.array([t for t, _ in buf])
            vals = np.array([v for _, v in buf])
            mask = (times >= t0) & (times <= t1)
            return vals[mask]

        red_vals = extract(self.rppg_red_buffer)
        green_vals = extract(self.rppg_green_buffer)
        if len(red_vals) < 4 or len(green_vals) < 4:
            return None
        ac_r = float(np.std(red_vals))
        dc_r = float(np.mean(red_vals)) + 1e-6
        ac_g = float(np.std(green_vals))
        dc_g = float(np.mean(green_vals)) + 1e-6
        R = (ac_r / dc_r) / (ac_g / dc_g + 1e-12)
        spo2 = 110.0 - 25.0 * (R)
        spo2 = float(np.clip(spo2, 50.0, 100.0))
        self.rppg_last_spo2 = spo2
        return spo2

    def process_frame(self, frame: np.ndarray):
        stats: Dict[str, Any] = {}
        h, w = frame.shape[:2]
        now = time.time()
        self.frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # CRITICAL: PPE Detection with frame skipping (every 5 frames)
        if self.frame_count % 5 == 0:
            ppe_detections = self.ppe_detector.detect(frame)
            stats['ppe'] = ppe_detections
        else:
            stats['ppe'] = self.ppe_detector.current_state.copy()

        # Draw PPE status
        frame = self.ppe_detector.draw_detections(frame, stats['ppe'])

        # CRITICAL: MediaPipe with frame skipping (every 2 frames)
        if self.frame_count % 2 == 0:
            results = self.mp_face_mesh.process(rgb)
            if results.multi_face_landmarks:
                self.last_face_landmarks = results.multi_face_landmarks[0].landmark

        # Use last known landmarks if available
        landmarks = self.last_face_landmarks

        if landmarks:
            try:
                ear_l = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
                ear_r = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)
                raw_ear = (ear_l + ear_r) / 2.0
            except Exception:
                raw_ear = None
                ear_l = None
                ear_r = None

            try:
                mar = mouth_aspect_ratio(landmarks, w, h)
            except Exception:
                mar = None

            if raw_ear is not None:
                if self.ear_ema is None:
                    self.ear_ema = raw_ear
                else:
                    # CRITICAL: Faster EMA for responsive blink detection
                    self.ear_ema = EAR_EMA_ALPHA * raw_ear + (1 - EAR_EMA_ALPHA) * self.ear_ema
                ear = self.ear_ema
            else:
                ear = None

            stats['ear'] = ear
            stats['mar'] = mar
            if mar is not None:
                self._update_mar_history(mar, now)

            # SIMPLIFIED Blink detection
            if ear_l is not None and ear_r is not None:
                self.blink_detector.detect(ear_l, ear_r, now)
            stats['blink_count'] = self.blink_detector.get_count()

            # Microsleep detection
            if ear is not None and ear < self.ear_threshold:
                self.eye_closed_frames += 1
            else:
                if self.eye_closed_frames >= self.ear_consec_frames_thresh:
                    self.microsleep_count += 1
                self.eye_closed_frames = 0

            stats['eye_closed_frames'] = self.eye_closed_frames
            stats['microsleep_count'] = self.microsleep_count

            # Yawn detection
            mar_sustain = self._mar_sustained_time()
            mar_slope = self._compute_mar_slope()
            is_gradual_opening = abs(mar_slope) <= MAR_SLOPE_MAX
            is_sustained = mar_sustain >= MAR_SUSTAIN_TIME
            is_far_from_last_yawn = (now - self.last_yawn_time) > YAWN_MIN_SEPARATION
            if (mar is not None) and (
                    mar >= self.mar_yawn_threshold) and is_sustained and is_gradual_opening and is_far_from_last_yawn:
                self.yawn_count += 1
                self.last_yawn_time = now
            stats['yawn_count'] = self.yawn_count

            # PERCLOS
            self._update_perclos(ear)
            perclos = self.compute_perclos()
            stats['perclos'] = perclos

            # Chest ROI
            if self.chest_roi_abs is None:
                try:
                    self._compute_chest_roi_from_face(landmarks, w, h)
                except Exception:
                    rx, ry, rw, rh = FLOW_ROI_REL
                    self.chest_roi_abs = (int(w * rx), int(h * ry), int(w * rw), int(h * rh))

            # CRITICAL: rPPG with frame skipping (every 2 frames)
            if self.frame_count % 2 == 0:
                self._append_rppg(frame, landmarks)

        else:
            stats['ear'] = None
            stats['mar'] = None
            stats['yawn_count'] = self.yawn_count
            perclos = self.compute_perclos()
            stats['perclos'] = perclos
            stats['blink_count'] = self.blink_detector.get_count()
            if self.chest_roi_abs is None:
                rx, ry, rw, rh = FLOW_ROI_REL
                self.chest_roi_abs = (int(w * rx), int(h * ry), int(w * rw), int(h * rh))

        # CRITICAL: Optical flow with frame skipping (every 3 frames)
        if self.frame_count % 3 == 0:
            try:
                x, y, ww, hh = self.chest_roi_abs
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                ww = max(8, min(ww, w - x))
                hh = max(8, min(hh, h - y))
                roi = gray[y:y + hh, x:x + ww]
                if self.prev_gray is not None:
                    prev_roi = self.prev_gray[y:y + hh, x:x + ww]
                    # OPTIMIZED: Faster optical flow parameters
                    flow = cv2.calcOpticalFlowFarneback(prev_roi, roi, None,pyr_scale=0.5, levels=2, winsize=10,iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
                    v_motion = float(np.mean(flow[..., 1]))
                    self._append_resp_motion(v_motion)
                self.prev_gray = gray.copy()
            except Exception:
                self.prev_gray = gray.copy()

        est_bpm = self.estimate_resp_bpm()
        stats['resp_bpm'] = est_bpm if est_bpm is not None else (
            self.resp_last_est_bpm if self.resp_last_est_bpm is not None else 0.0)

        hr = self.estimate_hr_from_rppg()
        spo2 = self.estimate_spo2_from_rgb()
        stats['hr_bpm'] = hr if hr is not None else (self.rppg_last_hr if self.rppg_last_hr is not None else 0.0)
        stats['spo2'] = spo2 if spo2 is not None else (self.rppg_last_spo2 if self.rppg_last_spo2 is not None else 0.0)

        return frame, stats


@dataclass
class FatigueEstimator:
    weight_eye: float = WEIGHT_EYE
    weight_yawn: float = WEIGHT_YAWN
    weight_resp: float = WEIGHT_RESP
    smoothed_fatigue: Optional[float] = field(default=0.0)

    def compute_raw(self, perclos: float, yawn_count: int, microsleep_count: int, resp_bpm: float) -> float:
        eye_score = np.clip(perclos, 0.0, 1.0)
        yawn_score = np.clip(yawn_count / 5.0, 0.0, 1.0)
        ms_factor = 1.0 - math.exp(-0.5 * microsleep_count)
        eye_score = 0.7 * eye_score + 0.3 * ms_factor
        if resp_bpm <= 0:
            resp_score = 0.0
        else:
            opt_lo, opt_hi = 12.0, 20.0
            if resp_bpm < opt_lo:
                resp_score = np.clip((opt_lo - resp_bpm) / opt_lo, 0.0, 1.0)
            elif resp_bpm > opt_hi:
                resp_score = np.clip((resp_bpm - opt_hi) / (40.0 - opt_hi), 0.0, 1.0)
            else:
                resp_score = 0.0
        combined = (self.weight_eye * eye_score + self.weight_yawn * yawn_score + self.weight_resp * resp_score)
        return float(np.clip(combined * 100.0, 0.0, 100.0))

    def compute_smoothed(self, raw_score: float) -> float:
        if self.smoothed_fatigue is None:
            self.smoothed_fatigue = raw_score
        else:
            self.smoothed_fatigue = FATIGUE_EMA_ALPHA * raw_score + (1 - FATIGUE_EMA_ALPHA) * self.smoothed_fatigue
        return self.smoothed_fatigue


class MinerDatabase:
    def __init__(self):
        self.miners = {}
        self.lock = threading.Lock()
        self._load_sample_data()

    def _load_sample_data(self):
        sample_miners = [
            {"name": "John Smith", "id": "MN001", "shift": "Morning", "gender": "Male"},
            {"name": "Sarah Johnson", "id": "MN002", "shift": "Afternoon", "gender": "Female"},
            {"name": "Michael Brown", "id": "MN003", "shift": "Night", "gender": "Male"},
        ]
        for miner in sample_miners:
            self.add_miner(miner['name'], miner['id'], miner['shift'], miner['gender'])

    def add_miner(self, name: str, miner_id: str, shift: str, gender: str) -> Dict:
        with self.lock:
            if miner_id in self.miners:
                return {"error": "Miner ID already exists"}
            self.miners[miner_id] = {
                "name": name,
                "id": miner_id,
                "shift": shift,
                "gender": gender
            }
            return {"success": True, "miner": self.miners[miner_id]}

    def delete_miner(self, miner_id: str) -> Dict:
        with self.lock:
            if miner_id not in self.miners:
                return {"error": "Miner not found"}
            del self.miners[miner_id]
            return {"success": True}

    def get_all_miners(self) -> List[Dict]:
        with self.lock:
            return list(self.miners.values())

    def get_miner(self, miner_id: str) -> Optional[Dict]:
        with self.lock:
            return self.miners.get(miner_id)


app = Flask(__name__)
CORS(app)


class MonitoringSystem:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.fatigue_estimator = FatigueEstimator()
        self.alert_manager = AlertManager()
        self.miner_db = MinerDatabase()
        self.is_running = False
        self.cap = None
        self.lock = threading.Lock()
        self.current_frame = None
        self.current_stats = {}
        self.monitoring_thread = None
        self.current_miner_id = None
        self.latest_frame_lock = threading.Lock()

    def start_monitoring(self, miner_id: str):
        with self.lock:
            if self.is_running:
                return {"status": "already_running"}

            miner = self.miner_db.get_miner(miner_id)
            if not miner:
                return {"status": "error", "message": "Miner not found"}

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return {"status": "error", "message": "Cannot open webcam"}

            # ULTRA-OPTIMIZED camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            self.current_miner_id = miner_id
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            return {"status": "started", "miner": miner}

    def stop_monitoring(self):
        with self.lock:
            if not self.is_running:
                return {"status": "not_running"}

            self.is_running = False
            self.current_miner_id = None
            if self.cap:
                self.cap.release()
                self.cap = None

            self.current_frame = None
            self.current_stats = {}

            return {"status": "stopped"}

    def reset_monitoring(self):
        self.video_processor.microsleep_count = 0
        self.video_processor.yawn_count = 0
        self.video_processor.eye_closed_frames = 0
        self.video_processor.ear_ema = None
        self.video_processor.mar_history.clear()
        self.video_processor.perclos_window.clear()
        self.video_processor.resp_motion_buffer.clear()
        self.video_processor.prev_gray = None
        self.video_processor.chest_roi_abs = None
        self.video_processor.resp_last_est_bpm = None
        self.video_processor.rppg_green_buffer.clear()
        self.video_processor.rppg_red_buffer.clear()
        self.video_processor.rppg_last_hr = None
        self.video_processor.rppg_last_spo2 = None
        self.video_processor.frame_count = 0
        self.video_processor.last_face_landmarks = None

        self.video_processor.blink_detector = SimplifiedBlinkDetector()
        self.video_processor.ppe_detector.detection_history = {
            'person': deque(maxlen=8),
            'helmet': deque(maxlen=8),
            'goggles': deque(maxlen=8),
            'vest': deque(maxlen=8),
            'gloves': deque(maxlen=8),
            'boots': deque(maxlen=8)
        }
        self.video_processor.ppe_detector.current_state = {
            'person': False,
            'helmet': False,
            'goggles': False,
            'vest': False,
            'gloves': False,
            'boots': False
        }

        self.fatigue_estimator.smoothed_fatigue = 0.0
        self.alert_manager.clear_alerts()

        # CRITICAL: Clear current stats to reset UI
        with self.lock:
            self.current_stats = {
                'ear': None,
                'mar': None,
                'blink_count': 0,
                'eye_closed_frames': 0,
                'microsleep_count': 0,
                'yawn_count': 0,
                'perclos': 0.0,
                'resp_bpm': 0.0,
                'hr_bpm': 0.0,
                'spo2': 0.0,
                'fatigue_score': 0.0,
                'ppe': {
                    'person': False,
                    'helmet': False,
                    'goggles': False,
                    'vest': False,
                    'gloves': False,
                    'boots': False
                }
            }

        return {"status": "reset"}

    def _monitoring_loop(self):
        """OPTIMIZED: Zero-lag monitoring with perfect blink detection"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Process frame immediately
            processed_frame, stats = self.video_processor.process_frame(frame)

            perclos = stats.get('perclos', 0.0)
            yawn_count = stats.get('yawn_count', 0)
            ms_count = stats.get('microsleep_count', 0)
            resp_bpm = stats.get('resp_bpm', 0.0)
            hr_bpm = stats.get('hr_bpm', 0.0)
            spo2 = stats.get('spo2', 0.0)

            # PPE status
            ppe = stats.get('ppe', {})

            raw_score = self.fatigue_estimator.compute_raw(perclos, yawn_count, ms_count, resp_bpm)
            smooth_score = self.fatigue_estimator.compute_smoothed(raw_score)
            stats['fatigue_score'] = smooth_score

            # Fatigue alerts
            if perclos >= ALERT_PERCLOS:
                self.alert_manager.trigger("HIGH_PERCLOS", f"PERCLOS={perclos:.2f}")

            if smooth_score >= ALERT_FATIGUE:
                self.alert_manager.trigger("HIGH_FATIGUE", f"Fatigue Score: {smooth_score:.1f}")

            if ms_count >= ALERT_MICROSLEEP_COUNT and ms_count > 0:
                self.alert_manager.trigger("MICROSLEEP", f"Count={ms_count}")

            if yawn_count >= ALERT_YAWN_COUNT:
                self.alert_manager.trigger("FREQUENT_YAWNS", f"Yawns={yawn_count}")

            if resp_bpm > 0 and (resp_bpm < ALERT_RESP_LOW or resp_bpm > ALERT_RESP_HIGH):
                self.alert_manager.trigger("RESP_RATE", f"{resp_bpm:.1f} bpm")

            if hr_bpm > 0 and (hr_bpm < ALERT_HR_LOW or hr_bpm > ALERT_HR_HIGH):
                self.alert_manager.trigger("HEART_RATE", f"{hr_bpm:.0f} bpm")

            if spo2 > 0 and (spo2 < ALERT_SPO2_LOW):
                self.alert_manager.trigger("LOW_SPO2", f"{spo2:.1f}%")

            # PPE alerts
            if PPE_ALERT_ENABLED and ppe.get('person', False):
                if not ppe.get('helmet', False):
                    self.alert_manager.trigger("NO_HELMET", "Helmet not detected")
                if not ppe.get('goggles', False):
                    self.alert_manager.trigger("NO_GOGGLES", "Goggles not detected")
                if not ppe.get('vest', False):
                    self.alert_manager.trigger("NO_VEST", "Safety vest not detected")
                if not ppe.get('gloves', False):
                    self.alert_manager.trigger("NO_GLOVES", "Gloves not detected")
                if not ppe.get('boots', False):
                    self.alert_manager.trigger("NO_BOOTS", "Safety boots not detected")

            # Direct frame update
            with self.latest_frame_lock:
                self.current_frame = processed_frame

            with self.lock:
                self.current_stats = stats

    def get_current_frame(self):
        with self.latest_frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def get_current_stats(self):
        with self.lock:
            return self.current_stats.copy()

    def is_monitoring_active(self):
        with self.lock:
            return self.is_running


monitoring_system = MonitoringSystem()


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/monitor/<miner_id>')
def monitor_page(miner_id):
    miner = monitoring_system.miner_db.get_miner(miner_id)
    if not miner:
        return "Miner not found", 404
    return render_template_string(MONITOR_TEMPLATE, miner=miner)


@app.route('/api/miners', methods=['GET'])
def get_miners():
    miners = monitoring_system.miner_db.get_all_miners()
    return jsonify({"miners": miners})


@app.route('/api/miners', methods=['POST'])
def add_miner():
    data = request.json
    result = monitoring_system.miner_db.add_miner(
        data.get('name', ''),
        data.get('id', ''),
        data.get('shift', ''),
        data.get('gender', '')
    )
    return jsonify(result)


@app.route('/api/miners/<miner_id>', methods=['DELETE'])
def delete_miner(miner_id):
    result = monitoring_system.miner_db.delete_miner(miner_id)
    return jsonify(result)


@app.route('/api/start/<miner_id>', methods=['POST'])
def start_monitoring(miner_id):
    result = monitoring_system.start_monitoring(miner_id)
    return jsonify(result)


@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    result = monitoring_system.stop_monitoring()
    return jsonify(result)


@app.route('/api/reset', methods=['POST'])
def reset_monitoring():
    result = monitoring_system.reset_monitoring()
    return jsonify(result)


@app.route('/api/stats')
def get_stats():
    stats = monitoring_system.get_current_stats()
    return jsonify(stats)


@app.route('/api/alerts')
def get_alerts():
    alerts = monitoring_system.alert_manager.get_alerts()
    return jsonify({"alerts": alerts})


@app.route('/api/monitoring_status')
def get_monitoring_status():
    is_active = monitoring_system.is_monitoring_active()
    return jsonify({"is_monitoring": is_active})


def generate_frames():
    """OPTIMIZED: Zero-lag frame streaming"""
    # OPTIMIZED: Reduced JPEG quality for speed
    encode_param = [
        cv2.IMWRITE_JPEG_QUALITY, 50,  # Reduced from 65
        cv2.IMWRITE_JPEG_OPTIMIZE, 1
    ]

    last_frame_time = time.time()

    while True:
        if monitoring_system.is_monitoring_active():
            frame = monitoring_system.get_current_frame()

            if frame is not None:
                current_time = time.time()

                # Target 30 FPS
                if current_time - last_frame_time >= 0.033:
                    ret, buffer = cv2.imencode('.jpg', frame, encode_param)

                    if ret:
                        frame_bytes = buffer.tobytes()
                        last_frame_time = current_time

                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n'
                               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
                               frame_bytes + b'\r\n')
                else:
                    time.sleep(0.001)
            else:
                time.sleep(0.033)
        else:
            time.sleep(0.1)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Miner Management System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 32px;
            color: #fff;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }

        .header p {
            color: #9ca3af;
            font-size: 16px;
        }

        .actions-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            flex-wrap: wrap;
            gap: 15px;
        }

        .search-box {
            flex: 1;
            min-width: 250px;
        }

        .search-box input {
            width: 100%;
            padding: 12px 20px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            color: #e0e0e0;
            font-size: 14px;
            transition: all 0.3s ease;
        }

        .search-box input:focus {
            outline: none;
            border-color: #667eea;
            background: rgba(255, 255, 255, 0.08);
        }

        .btn {
            padding: 12px 28px;
            border: none;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        }

        .miners-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
        }

        .miner-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }

        .miner-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }

        .miner-card:hover::before {
            transform: scaleX(1);
        }

        .miner-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
            border-color: rgba(102, 126, 234, 0.3);
        }

        .miner-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }

        .miner-info h3 {
            font-size: 20px;
            color: #fff;
            margin-bottom: 5px;
        }

        .miner-id {
            font-size: 13px;
            color: #667eea;
            font-weight: 600;
        }

        .delete-btn {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #ef4444;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .delete-btn:hover {
            background: rgba(239, 68, 68, 0.3);
            border-color: #ef4444;
        }

        .miner-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 15px;
        }

        .detail-item {
            background: rgba(255, 255, 255, 0.03);
            padding: 10px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .detail-label {
            font-size: 11px;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }

        .detail-value {
            font-size: 14px;
            color: #fff;
            font-weight: 600;
        }

        .monitor-btn {
            width: 100%;
            margin-top: 15px;
            padding: 12px;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .monitor-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        }

        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(5px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }

        .modal.active {
            display: flex;
        }

        .modal-content {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 35px;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            max-width: 500px;
            width: 90%;
        }

        .modal-header {
            margin-bottom: 25px;
        }

        .modal-header h2 {
            font-size: 24px;
            color: #fff;
            margin-bottom: 5px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #9ca3af;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .form-group input,
        .form-group select {
            width: 100%;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: #e0e0e0;
            font-size: 14px;
            transition: all 0.3s ease;
        }

        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: #667eea;
            background: rgba(255, 255, 255, 0.08);
        }

        .modal-actions {
            display: flex;
            gap: 12px;
            margin-top: 25px;
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.1);
            color: #e0e0e0;
            flex: 1;
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        .btn-submit {
            flex: 1;
        }

        .empty-state {
            text-align: center;
            padding: 60px 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 15px;
            border: 2px dashed rgba(255, 255, 255, 0.1);
        }

        .empty-state-icon {
            font-size: 64px;
            margin-bottom: 20px;
        }

        .empty-state h3 {
            font-size: 20px;
            color: #fff;
            margin-bottom: 10px;
        }

        .empty-state p {
            color: #9ca3af;
            margin-bottom: 25px;
        }

        @media (max-width: 768px) {
            .miners-grid {
                grid-template-columns: 1fr;
            }

            .actions-bar {
                flex-direction: column;
                align-items: stretch;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⛏️Miner Monitoring System</h1>
            <p>AI Enabled Health and Monitoring System for Mining Workers </p>
        </div>

        <div class="actions-bar">
            <div class="search-box">
                <input type="text" id="searchInput" placeholder="🔍 Search miners by name or ID...">
            </div>
            <button class="btn btn-primary" onclick="openAddModal()">+ Add New Miner</button>
        </div>

        <div id="minersContainer" class="miners-grid"></div>
    </div>

    <div id="addModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>Add New Miner</h2>
            </div>
            <form id="addMinerForm">
                <div class="form-group">
                    <label>Miner Name</label>
                    <input type="text" id="minerName" required placeholder="Enter full name">
                </div>
                <div class="form-group">
                    <label>Miner ID</label>
                    <input type="text" id="minerId" required placeholder="e.g., MN004">
                </div>
                <div class="form-group">
                    <label>Shift</label>
                    <select id="minerShift" required>
                        <option value="">Select shift</option>
                        <option value="Morning">Morning</option>
                        <option value="Afternoon">Afternoon</option>
                        <option value="Night">Night</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Gender</label>
                    <select id="minerGender" required>
                        <option value="">Select gender</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                    </select>
                </div>
                <div class="modal-actions">
                    <button type="button" class="btn btn-secondary" onclick="closeAddModal()">Cancel</button>
                    <button type="submit" class="btn btn-primary btn-submit">Add Miner</button>
                </div>
            </form>
        </div>
    </div>

    <script>
        let allMiners = [];

        function loadMiners() {
            fetch('/api/miners')
                .then(response => response.json())
                .then(data => {
                    allMiners = data.miners;
                    displayMiners(allMiners);
                })
                .catch(error => console.error('Error loading miners:', error));
        }

        function displayMiners(miners) {
            const container = document.getElementById('minersContainer');

            if (miners.length === 0) {
                container.innerHTML = `
                    <div class="empty-state" style="grid-column: 1 / -1;">
                        <div class="empty-state-icon">👷</div>
                        <h3>No Miners Found</h3>
                        <p>Add your first miner to start monitoring</p>
                        <button class="btn btn-primary" onclick="openAddModal()">+ Add New Miner</button>
                    </div>
                `;
                return;
            }

            container.innerHTML = miners.map(miner => `
                <div class="miner-card">
                    <div class="miner-header">
                        <div class="miner-info">
                            <h3>${miner.name}</h3>
                            <div class="miner-id">${miner.id}</div>
                        </div>
                        <button class="delete-btn" onclick="deleteMiner('${miner.id}', event)">✕ Delete</button>
                    </div>
                    <div class="miner-details">
                        <div class="detail-item">
                            <div class="detail-label">Shift</div>
                            <div class="detail-value">${miner.shift}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Gender</div>
                            <div class="detail-value">${miner.gender}</div>
                        </div>
                    </div>
                    <button class="monitor-btn" onclick="startMonitoring('${miner.id}')">
                        🎥 Start Monitoring
                    </button>
                </div>
            `).join('');
        }

        function openAddModal() {
            document.getElementById('addModal').classList.add('active');
        }

        function closeAddModal() {
            document.getElementById('addModal').classList.remove('active');
            document.getElementById('addMinerForm').reset();
        }

        document.getElementById('addMinerForm').addEventListener('submit', function(e) {
            e.preventDefault();

            const minerData = {
                name: document.getElementById('minerName').value,
                id: document.getElementById('minerId').value,
                shift: document.getElementById('minerShift').value,
                gender: document.getElementById('minerGender').value
            };

            fetch('/api/miners', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(minerData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    closeAddModal();
                    loadMiners();
                }
            })
            .catch(error => console.error('Error adding miner:', error));
        });

        function deleteMiner(minerId, event) {
            event.stopPropagation();

            if (!confirm('Are you sure you want to delete this miner?')) {
                return;
            }

            fetch(`/api/miners/${minerId}`, {
                method: 'DELETE'
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    loadMiners();
                }
            })
            .catch(error => console.error('Error deleting miner:', error));
        }

        function startMonitoring(minerId) {
            window.location.href = `/monitor/${minerId}`;
        }

        document.getElementById('searchInput').addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            const filtered = allMiners.filter(miner => 
                miner.name.toLowerCase().includes(searchTerm) ||
                miner.id.toLowerCase().includes(searchTerm)
            );
            displayMiners(filtered);
        });

        document.getElementById('addModal').addEventListener('click', function(e) {
            if (e.target === this) {
                closeAddModal();
            }
        });

        loadMiners();
    </script>
</body>
</html>
'''

MONITOR_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monitoring - {{ miner.name }}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f0f23;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .container {
            display: grid;
            grid-template-columns: 280px 1fr;
            grid-template-rows: auto 1fr;
            gap: 15px;
            padding: 15px;
            max-width: 1920px;
            margin: 0 auto;
            min-height: 100vh;
        }
        .header {
            grid-column: 1 / -1;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 20px 30px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .header-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .miner-info-header {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .miner-avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            font-weight: bold;
            color: white;
        }
        .miner-details h1 {
            font-size: 24px;
            color: #fff;
            margin-bottom: 5px;
        }
        .miner-meta {
            font-size: 13px;
            color: #9ca3af;
        }
        .back-btn {
            padding: 10px 20px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: #e0e0e0;
            text-decoration: none;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .back-btn:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
        }
        .controls {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 10px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
        }
        .btn-start {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-stop {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        .btn-reset {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }
        .status-badge {
            display: inline-block;
            padding: 8px 18px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 13px;
            margin-left: 15px;
        }
        .status-active {
            background: #48bb78;
            color: white;
            animation: pulse 2s infinite;
        }
        .status-inactive {
            background: #4a5568;
            color: #cbd5e0;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .alerts-panel, .video-section, .detections-section, .ppe-section, .charts-section {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .alerts-panel {
            max-height: calc(100vh - 120px);
            overflow-y: auto;
        }
        .alert-item {
            background: rgba(239, 68, 68, 0.15);
            border-left: 4px solid #ef4444;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 8px;
        }
        .main-content {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .top-section {
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 15px;
        }
        .video-container {
            position: relative;
            background: #000;
            border-radius: 12px;
            min-height: 380px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .video-container img {
            width: 100%;
            height: auto;
            display: block;
            max-height: 380px;
            object-fit: contain;
            border-radius: 12px;
        }
        .video-placeholder {
            color: #6b7280;
            font-size: 16px;
            text-align: center;
        }
        .detection-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .detection-item {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .detection-label {
            font-size: 12px;
            color: #9ca3af;
            margin-bottom: 6px;
            text-transform: uppercase;
        }
        .detection-value {
            font-size: 22px;
            font-weight: 700;
            color: #fff;
        }
        .detection-value.good { color: #10b981; }
        .detection-value.warning { color: #f59e0b; }
        .detection-value.danger { color: #ef4444; }
        .ppe-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }
        .ppe-item {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }
        .ppe-item.detected {
            border-color: #10b981;
            background: rgba(16, 185, 129, 0.1);
        }
        .ppe-item.missing {
            border-color: #ef4444;
            background: rgba(239, 68, 68, 0.1);
        }
        .ppe-icon {
            font-size: 28px;
            margin-bottom: 8px;
        }
        .ppe-label {
            font-size: 12px;
            color: #9ca3af;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .ppe-status {
            font-size: 14px;
            font-weight: 700;
        }
        .ppe-status.yes { color: #10b981; }
        .ppe-status.no { color: #ef4444; }
        .chart-container {
            background: rgba(255, 255, 255, 0.03);
            padding: 20px;
            border-radius: 12px;
            height: 380px;
        }
        .no-alerts {
            text-align: center;
            padding: 30px;
            color: #6b7280;
            font-style: italic;
        }
        h2 {
            font-size: 18px;
            color: #fff;
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-top">
                <div class="miner-info-header">
                    <div class="miner-avatar">{{ miner.name[0] }}</div>
                    <div class="miner-details">
                        <h1>{{ miner.name }}</h1>
                        <div class="miner-meta">ID: {{ miner.id }} | Shift: {{ miner.shift }} | Gender: {{ miner.gender }}</div>
                    </div>
                </div>
                <a href="/" class="back-btn">← Back</a>
            </div>
            <div class="controls">
                <button class="btn btn-start" onclick="startMonitoring()">▶ Start</button>
                <button class="btn btn-stop" onclick="stopMonitoring()">⏹ Stop</button>
                <button class="btn btn-reset" onclick="resetMonitoring()">🔄 Reset</button>
                <span class="status-badge status-inactive" id="statusBadge">● INACTIVE</span>
            </div>
        </div>

        <div class="alerts-panel">
            <h2>🚨 Alerts</h2>
            <div id="alertsList">
                <div class="no-alerts">No alerts yet</div>
            </div>
        </div>

        <div class="main-content">
            <div class="top-section">
                <div class="video-section">
                    <h2>📹 Live Video Feed</h2>
                    <div class="video-container" id="videoContainer">
                        <div class="video-placeholder">Click Start to begin monitoring</div>
                    </div>
                </div>

                <div class="detections-section">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
        <h2 style="margin:0;">🎯 Fatigue Metrics</h2>
        <span class="status-badge status-inactive" id="personDetectionStatus" style="font-size:11px;padding:6px 12px;">Person MISSING</span>
    </div>
                    <div class="detection-grid">
                        <div class="detection-item">
                            <div class="detection-label">Fatigue Score</div>
                            <div class="detection-value" id="det-fatigue">--</div>
                        </div>
                        <div class="detection-item">
                            <div class="detection-label">Heart Rate</div>
                            <div class="detection-value" id="det-hr">--</div>
                        </div>
                        <div class="detection-item">
                            <div class="detection-label">Respiration</div>
                            <div class="detection-value" id="det-resp">--</div>
                        </div>
                        <div class="detection-item">
                            <div class="detection-label">SpO2 (%)</div>
                            <div class="detection-value" id="det-spo2">--</div>
                        </div>
                        <div class="detection-item">
                            <div class="detection-label">PERCLOS</div>
                            <div class="detection-value" id="det-perclos">--</div>
                        </div>
                        <div class="detection-item">
                            <div class="detection-label">EAR</div>
                            <div class="detection-value" id="det-ear">--</div>
                        </div>
                        <div class="detection-item">
                            <div class="detection-label">Blinks ✨</div>
                            <div class="detection-value" id="det-blink">--</div>
                        </div>
                        <div class="detection-item">
                            <div class="detection-label">Microsleeps</div>
                            <div class="detection-value" id="det-microsleep">--</div>
                        </div>
                        <div class="detection-item">
                            <div class="detection-label">Yawns</div>
                            <div class="detection-value" id="det-yawn">--</div>
                        </div>
                        <div class="detection-item">
                            <div class="detection-label">MAR</div>
                            <div class="detection-value" id="det-mar">--</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="ppe-section">
                <h2>🦺 PPE Detection Status</h2>
                <div class="ppe-grid">
                    <div class="ppe-item" id="ppe-person">
                        <div class="ppe-icon">👤</div>
                        <div class="ppe-label">Person</div>
                        <div class="ppe-status">--</div>
                    </div>
                    <div class="ppe-item" id="ppe-helmet">
                        <div class="ppe-icon">⛑️</div>
                        <div class="ppe-label">Helmet</div>
                        <div class="ppe-status">--</div>
                    </div>
                    <div class="ppe-item" id="ppe-goggles">
                        <div class="ppe-icon">🥽</div>
                        <div class="ppe-label">Goggles</div>
                        <div class="ppe-status">--</div>
                    </div>
                    <div class="ppe-item" id="ppe-vest">
                        <div class="ppe-icon">🦺</div>
                        <div class="ppe-label">Vest</div>
                        <div class="ppe-status">--</div>
                    </div>
                    <div class="ppe-item" id="ppe-gloves">
                        <div class="ppe-icon">🧤</div>
                        <div class="ppe-label">Gloves</div>
                        <div class="ppe-status">--</div>
                    </div>
                    <div class="ppe-item" id="ppe-boots">
                        <div class="ppe-icon">🥾</div>
                        <div class="ppe-label">Boots</div>
                        <div class="ppe-status">--</div>
                    </div>
                </div>
            </div>

            <div class="charts-section">
                <h2>📊 Real-Time Metrics</h2>
                <div class="chart-container">
                    <canvas id="combinedChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
const MINER_ID = '{{ miner.id }}';
const MAX_DATA_POINTS = 60;
let isMonitoring = false;
let personDetected = false;
let videoImg = null;

// Chart setup
const chart = new Chart(document.getElementById('combinedChart'), {
    type: 'line',
    data: { labels: [], datasets: [
        { label: 'Fatigue', data: [], yAxisID: 'y', tension: 0.4, borderWidth: 2 },
        { label: 'HR', data: [], yAxisID: 'y1', tension: 0.4, borderWidth: 2 },
        { label: 'Resp', data: [], yAxisID: 'y2', tension: 0.4, borderWidth: 2, hidden: true },
        { label: 'SpO2', data: [], yAxisID: 'y3', tension: 0.4, borderWidth: 2, hidden: true }
    ]},
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
            y: { position: 'left', min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' }},
            y1:{ position:'right', min:40, max:180, grid:{ drawOnChartArea:false }, ticks:{ color:'#9ca3af' }},
            y2:{ min:0, max:35, display:false },
            y3:{ min:85, max:100, display:false },
            x: { grid:{ color:'rgba(255,255,255,0.05)' }, ticks:{ color:'#9ca3af' }}
        },
        plugins:{ legend:{ labels:{ color:'#e0e0e0' }}}
    }
});

function updateChart(f, h, r, s) {
    if (!personDetected || !isMonitoring) return; // freeze on stop or missing person
    if (chart.data.labels.length >= MAX_DATA_POINTS) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(d => d.data.shift());
    }
    chart.data.labels.push(new Date().toLocaleTimeString());
    chart.data.datasets[0].data.push(f);
    chart.data.datasets[1].data.push(h);
    chart.data.datasets[2].data.push(r);
    chart.data.datasets[3].data.push(s);
    chart.update('none');
}

function updatePPE(ppe) {
    ['person','helmet','goggles','vest','gloves','boots','boots'].forEach(item=>{
        const el=document.getElementById(`ppe-${item}`);
        if(!el) return;
        const st=el.querySelector('.ppe-status');
        const ok=ppe[item]||false;
        st.textContent = ok ? 'DETECTED' : 'MISSING';
        st.className = ok ? 'ppe-status yes':'ppe-status no';
        el.className = ok ? 'ppe-item detected':'ppe-item missing';
    });
}

function updateValues(stats) {
    document.getElementById('det-fatigue').textContent = stats.fatigue_score?.toFixed(1) ?? '--';
    document.getElementById('det-hr').textContent = stats.hr_bpm>0?Math.round(stats.hr_bpm):'--';
    document.getElementById('det-resp').textContent = stats.resp_bpm>0?stats.resp_bpm.toFixed(1):'--';
    document.getElementById('det-spo2').textContent = stats.spo2>0?stats.spo2.toFixed(1):'--';
    document.getElementById('det-perclos').textContent = stats.perclos?.toFixed(3) ?? '--';
    document.getElementById('det-ear').textContent = stats.ear?.toFixed(3) ?? '--';
    document.getElementById('det-mar').textContent = stats.mar?.toFixed(3) ?? '--';
    document.getElementById('det-blink').textContent = stats.blink_count ?? '--';
    document.getElementById('det-microsleep').textContent = stats.microsleep_count ?? '--';
    document.getElementById('det-yawn').textContent = stats.yawn_count ?? '--';
    if (stats.ppe) updatePPE(stats.ppe);
}

function updateVideoFeed(active) {
    const c=document.getElementById('videoContainer');
    if(active){
        if(!videoImg){
            const img=document.createElement('img');
            img.src=`/video_feed?t=${Date.now()}`;
            c.innerHTML='';c.appendChild(img);
            videoImg=img;
        }
    }else{
        // Black screen when stopped
        c.innerHTML='<div class="video-placeholder" style="background:black;width:100%;height:100%"></div>';
        videoImg=null;
    }
}

function setPersonStatus(detected) {
    const btn=document.getElementById('personDetectionStatus');
    if(!btn) return;
    btn.textContent = detected?'Person DETECTED':'Person MISSING';
    btn.className = detected?'btn btn-start':'btn btn-stop';
}

async function startMonitoring() {
    try{
        const r=await fetch(`/api/start/${MINER_ID}`,{method:'POST'});
        const d=await r.json();
        if(d.status==='started'||d.status==='already_running'){
            isMonitoring=true;
            personDetected=false;
            document.getElementById('statusBadge').className='status-badge status-active';
            document.getElementById('statusBadge').textContent='● ACTIVE';
            updateVideoFeed(true);
        }
    }catch(e){console.error('Start failed',e);}
}

async function stopMonitoring() {
    try{
        await fetch('/api/stop',{method:'POST'});
    }finally{
        isMonitoring=false;
        personDetected=false;
        document.getElementById('statusBadge').className='status-badge status-inactive';
        document.getElementById('statusBadge').textContent='● INACTIVE';
        updateVideoFeed(false);
        setPersonStatus(false); // ensure label shows missing
    }
}

async function updateAlerts() {
    if(!isMonitoring||!personDetected) return;
    try{
        const r=await fetch('/api/alerts');
        const d=await r.json();
        const list=document.getElementById('alertsList');
        if(d.alerts?.length){
            list.innerHTML=d.alerts.slice(-20).reverse().map(a=>`
                <div class="alert-item">
                    <div style="font-size:11px;color:#9ca3af">${a.timestamp}</div>
                    <div style="font-weight:700;color:#fca5a5">${a.name}</div>
                    <div style="color:#fecaca;font-size:12px">${a.message}</div>
                </div>`).join('');
        }else list.innerHTML='<div class="no-alerts">No alerts yet</div>';
    }catch(e){console.error('Alerts failed',e);}
}
async function resetMonitoring() {
    // Store if monitoring was active
    const wasMonitoring = isMonitoring;
    
    try{
        const r = await fetch('/api/reset',{method:'POST'});
        const d = await r.json();
        
        if(d.status === 'reset') {
            // Clear chart data
            chart.data.labels = [];
            chart.data.datasets.forEach(ds => ds.data = []);
            chart.update();
            
            // Reset all detection values to --
            document.getElementById('det-fatigue').textContent = '--';
            document.getElementById('det-hr').textContent = '--';
            document.getElementById('det-resp').textContent = '--';
            document.getElementById('det-spo2').textContent = '--';
            document.getElementById('det-perclos').textContent = '--';
            document.getElementById('det-ear').textContent = '--';
            document.getElementById('det-mar').textContent = '--';
            document.getElementById('det-blink').textContent = '--';
            document.getElementById('det-microsleep').textContent = '--';
            document.getElementById('det-yawn').textContent = '--';
            
            // Reset PPE status
            ['person','helmet','goggles','vest','gloves','boots'].forEach(item=>{
                const el=document.getElementById(`ppe-${item}`);
                if(el) {
                    const st=el.querySelector('.ppe-status');
                    st.textContent = '--';
                    st.className = 'ppe-status';
                    el.className = 'ppe-item';
                }
            });
            
            // Clear alerts
            document.getElementById('alertsList').innerHTML = '<div class="no-alerts">No alerts yet</div>';
            
            // DON'T change monitoring state - keep it as it was
            // If monitoring was active, it stays active
            
            alert('✓ All metrics reset successfully!');
        }
    } catch(e) {
        console.error('Reset failed', e);
        alert('❌ Failed to reset monitoring');
    }
}
async function updateStats() {
    if(!isMonitoring) return;
    try{
        const r=await fetch('/api/stats');
        const s=await r.json();
        personDetected = s.ppe?.person===true;
        
        // CRITICAL FIX: Always update person status button regardless of detection
        setPersonStatus(personDetected);
        
        // Always update PPE status
        if(s.ppe) updatePPE(s.ppe);
        
        if(!personDetected) {
            // When person is NOT detected, freeze metrics but keep video live
            // Don't return - let the function complete
        } else {
            // Only update charts and values when person IS detected
            updateChart(s.fatigue_score||0, s.hr_bpm||0, s.resp_bpm||0, s.spo2||0);
            updateValues(s);
        }
    }catch(e){console.error('Stats failed',e);}
}

// Scheduler
setInterval(updateStats,100);
setInterval(updateAlerts,2000);
updateVideoFeed(false);
</script>
</body>
</html>
'''
if __name__ == '__main__':
    print("=" * 80)
    print("🛡️MINER MONITORING")
    print("=" * 80)

    if not YOLO_AVAILABLE:
        print("\n⚠️  WARNING: YOLOv8 not available!")
    else:
        if not os.path.exists(PPE_MODEL_PATH):
            print(f"\n⚠️  WARNING: Model not found at {PPE_MODEL_PATH}")
        else:
            print(f"\n✓ YOLO model found: {PPE_MODEL_PATH}")

    print("\n📡 Starting Flask server...")
    print("   Open: http://localhost:5000")

    print("=" * 80 + "\n")

    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")
        if monitoring_system.is_running:
            monitoring_system.stop_monitoring()
        print("✓ Goodbye!")

