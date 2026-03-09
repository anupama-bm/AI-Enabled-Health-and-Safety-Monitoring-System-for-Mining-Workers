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
from flask import Flask, render_template_string, Response, jsonify, request
from flask_cors import CORS

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  Warning: ultralytics not installed. PPE detection disabled.")

# ── Thresholds ────────────────────────────────────────────────────────────────
EAR_THRESHOLD           = 0.21
CONSEC_FRAMES_MICROSLEEP= 18
MAR_YAWN_THRESHOLD      = 0.50
MAR_SUSTAIN_TIME        = 0.8
MAR_SLOPE_WINDOW        = 0.8
MAR_SLOPE_MAX           = 0.9
YAWN_MIN_SEPARATION     = 2.5
PERCLOS_WINDOW_SEC      = 60.0


BLINK_EAR_DROP_RATIO    = 0.72
BLINK_MIN_FRAMES        = 2
BLINK_MAX_FRAMES        = 12
BLINK_MIN_SEPARATION_MS = 120


RESP_BUFFER_SEC  = 30.0
FLOW_ROI_REL     = (0.35, 0.6, 0.30, 0.25)
RESP_MIN_BPM     = 6
RESP_MAX_BPM     = 30
RPPG_WINDOW_SEC  = 15.0
HR_MIN_BPM       = 40
HR_MAX_BPM       = 180
SPO2_WINDOW_SEC  = 15.0

# PPE
PPE_MODEL_PATH              = 'best.pt'
PPE_CONFIDENCE_THRESHOLD    = 0.40
PPE_ALERT_ENABLED           = True
PPE_ON_RATIO                = 0.40
PPE_OFF_RATIO               = 0.25

# EMA / weights
EAR_EMA_ALPHA    = 0.60
FATIGUE_EMA_ALPHA= 0.12
WEIGHT_EYE       = 0.45
WEIGHT_YAWN      = 0.30
WEIGHT_RESP      = 0.25

# Alerts
ALERT_PERCLOS        = 0.35
ALERT_FATIGUE        = 75.0
ALERT_HR_LOW         = 45.0
ALERT_HR_HIGH        = 120.0
ALERT_SPO2_LOW       = 94.0
ALERT_RESP_LOW       = 8.0
ALERT_RESP_HIGH      = 25.0
ALERT_MICROSLEEP_COUNT = 1
ALERT_YAWN_COUNT     = 4

# MediaPipe landmark indices
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
UPPER_LIP_IDX   = 13
LOWER_LIP_IDX   = 14
MOUTH_LEFT_IDX  = 78
MOUTH_RIGHT_IDX = 308
FOREHEAD_POINTS = [10, 338, 297, 332]

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def bandpass_peak_freq(signal, fs, lo_hz, hi_hz):
    if len(signal) < 4: return None
    x = signal - np.mean(signal)
    yf = np.abs(rfft(x))
    xf = rfftfreq(len(x), 1.0/fs)
    mask = (xf >= lo_hz) & (xf <= hi_hz)
    if not np.any(mask): return None
    yf2 = np.where(mask, yf, 0.0)
    freq = float(xf[int(np.argmax(yf2))])
    return freq if freq > 0 else None

def eye_aspect_ratio(landmarks, idx_list, w, h):
    pts = [(landmarks[i].x*w, landmarks[i].y*h) for i in idx_list]
    p1,p2,p3,p4,p5,p6 = pts
    return safe_div(euclid(p2,p6)+euclid(p3,p5), 2.0*euclid(p1,p4))

def mouth_aspect_ratio(landmarks, w, h):
    up  = (landmarks[UPPER_LIP_IDX].x*w,  landmarks[UPPER_LIP_IDX].y*h)
    low = (landmarks[LOWER_LIP_IDX].x*w,  landmarks[LOWER_LIP_IDX].y*h)
    lft = (landmarks[MOUTH_LEFT_IDX].x*w, landmarks[MOUTH_LEFT_IDX].y*h)
    rgt = (landmarks[MOUTH_RIGHT_IDX].x*w,landmarks[MOUTH_RIGHT_IDX].y*h)
    return safe_div(euclid(up,low), euclid(lft,rgt))


class AlertManager:
    def __init__(self):
        self.alerts_history = deque(maxlen=100)
        self.lock = threading.Lock()
        self.last_alerts: Dict[str, float] = {}

    def trigger(self, name, message):
        with self.lock:
            key = f"{name}_{message}"
            now = time.time()
            if key not in self.last_alerts or now - self.last_alerts[key] > 5.0:
                self.alerts_history.append({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'name': name, 'message': message})
                self.last_alerts[key] = now
                return True
        return False

    def get_alerts(self):
        with self.lock: return list(self.alerts_history)

    def clear_alerts(self):
        with self.lock:
            self.alerts_history.clear()
            self.last_alerts.clear()


class BlinkDetector:

    def __init__(self):
        self.count         = 0
        self.closed_frames = 0
        self.open_frames   = 0        # frames since eye last opened
        self._in_blink     = False
        self._last_blink_ts= 0.0


        self._baseline_buf = deque(maxlen=150)
        self.baseline      = 0.28
        self.threshold     = 0.20
    def _update_baseline(self, ear_avg):
        """Feed open-eye samples to baseline."""
        if ear_avg > self.threshold:          # only open-eye samples
            self._baseline_buf.append(ear_avg)
        if len(self._baseline_buf) >= 30:
            self.baseline  = float(np.percentile(list(self._baseline_buf), 80))
            self.threshold = max(0.15, self.baseline * BLINK_EAR_DROP_RATIO)

    # ── public ────────────────────────────────────────────────────────────────
    def update(self, ear_l: float, ear_r: float, ts: float) -> bool:
        ear = (ear_l + ear_r) / 2.0
        self._update_baseline(ear)

        is_closed = ear < self.threshold
        blink_detected = False

        if is_closed:
            self.closed_frames += 1
            self._in_blink = True
            self.open_frames = 0
        else:
            if self._in_blink:
                # eye just re-opened — validate
                gap_ms  = (ts - self._last_blink_ts) * 1000
                gap_ok  = gap_ms >= BLINK_MIN_SEPARATION_MS
                dur_ok  = BLINK_MIN_FRAMES <= self.closed_frames <= BLINK_MAX_FRAMES
                if dur_ok and gap_ok:
                    self.count += 1
                    self._last_blink_ts = ts
                    blink_detected = True
                    print(f"[BLINK] #{self.count}  dur={self.closed_frames}f  gap={gap_ms:.0f}ms  "
                          f"ear={ear:.3f}  thr={self.threshold:.3f}  base={self.baseline:.3f}")
                elif dur_ok and not gap_ok:
                    self.count += 1
                    self._last_blink_ts = ts
                    blink_detected = True
                    print(f"[BLINK-CONSEC] #{self.count}  dur={self.closed_frames}f  gap={gap_ms:.0f}ms  "
                          f"ear={ear:.3f}  thr={self.threshold:.3f}")
                self.closed_frames = 0
                self._in_blink = False
            self.open_frames += 1

        return blink_detected

    def get_count(self): return self.count


# ── YOLO PPE Detector ─────────────────────────────────────────────────────────
class YOLOPPEDetector:
    _ITEMS = ('person','helmet','goggles','vest','gloves','boots')

    def __init__(self, model_path):
        self.model        = None
        self.is_available = False
        self._lock        = threading.Lock()
        self._pending     = None          # latest frame waiting for inference
        self._stop        = threading.Event()

        # Temporal smoothing
        self._history = {k: deque(maxlen=8) for k in self._ITEMS}
        self.current_state = {k: False for k in self._ITEMS}

        # FIX: track last time a person was seen to enable instant PPE clear
        self._person_present = False

        if YOLO_AVAILABLE:
            self._load_model(model_path)
        if self.is_available:
            t = threading.Thread(target=self._loop, daemon=True)
            t.start()

    def _load_model(self, path):
        for p in [path, 'ppe_dataset/train/weights/best.pt',
                  'ppe_dataset/weights/best.pt', 'best.pt']:
            if os.path.exists(p):
                self.model = YOLO(p)
                self.model.overrides['verbose'] = False
                self.is_available = True
                print(f"✓ YOLO loaded: {p}")
                return
        print("⚠️  YOLO model not found — PPE disabled")

    def submit(self, frame: np.ndarray):
        """Non-blocking; overwrites stale frame."""
        small = cv2.resize(frame, (320, 240))
        with self._lock:
            self._pending = small

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                frame = self._pending
                self._pending = None
            if frame is None:
                time.sleep(0.025)
                continue
            try:
                res  = self.model(frame, conf=PPE_CONFIDENCE_THRESHOLD,
                                  verbose=False, device='cpu', imgsz=320)[0]
                seen = {k: False for k in self._ITEMS}
                if res.boxes is not None:
                    for box in res.boxes:
                        n = res.names[int(box.cls[0])].lower()
                        if   'person'  in n or 'worker'  in n: seen['person']  = True
                        elif 'helmet'  in n or 'hardhat' in n: seen['helmet']  = True
                        elif 'goggle'  in n or 'glasses' in n: seen['goggles'] = True
                        elif 'vest'    in n or 'jacket'  in n: seen['vest']    = True
                        elif 'glove'   in n:                   seen['gloves']  = True
                        elif 'boot'    in n or 'shoe'    in n: seen['boots']   = True
                with self._lock:
                    for k in self._ITEMS:
                        self._history[k].append(1 if seen[k] else 0)
                        h   = self._history[k]
                        win = 2 if k == 'person' else 5
                        on  = 0.50 if k == 'person' else PPE_ON_RATIO
                        off = 0.25 if k == 'person' else PPE_OFF_RATIO
                        if len(h) >= win:
                            r = sum(h) / len(h)
                            if not self.current_state[k]:
                                if r >= on:  self.current_state[k] = True
                            else:
                                if r <= off: self.current_state[k] = False
            except Exception as e:
                print(f"⚠️  PPE: {e}")
            time.sleep(0.10)   # ~10 inferences/sec

    def get_state(self):
        with self._lock: return self.current_state.copy()

    def force_person(self, val: bool):
        """
        Called by face-mesh path so person detection is instant.
        FIX: When person leaves (val=False), immediately clear ALL PPE states
        and flush their histories — no need to wait for the smoothing buffer
        to drain (~5 s). This gives instant feedback when the worker steps away.
        """
        with self._lock:
            self._history['person'].append(1 if val else 0)
            h = self._history['person']
            if len(h) >= 2:
                r = sum(h) / len(h)
                if val and r >= 0.50:
                    self.current_state['person'] = True
                if not val and r <= 0.25:
                    self.current_state['person'] = False

            # ── FIX: instant clear of all PPE when no person present ──────
            was_present = self._person_present
            self._person_present = val

            if was_present and not val:
                # Person just left — immediately reset all PPE items
                for k in self._ITEMS:
                    if k != 'person':
                        self._history[k].clear()
                        self.current_state[k] = False

    def reset(self):
        with self._lock:
            for k in self._ITEMS:
                self._history[k].clear()
                self.current_state[k] = False
            self._person_present = False

    def draw(self, frame, state):
        if not self.is_available: return frame
        ov = frame.copy()
        cv2.rectangle(ov, (10,10), (240,160), (20,20,40), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, "PPE", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y = 58
        for lbl, k in [('Person','person'),('Helmet','helmet'),('Goggles','goggles'),
                        ('Vest','vest'),('Gloves','gloves'),('Boots','boots')]:
            col = (0,220,80) if state.get(k) else (0,60,220)
            cv2.putText(frame, f"{'OK' if state.get(k) else 'XX'} {lbl}", (20,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            y += 17
        return frame


# ── Video Processor ───────────────────────────────────────────────────────────
@dataclass
class VideoProcessor:
    ear_threshold:            float = EAR_THRESHOLD
    ear_consec_frames_thresh: int   = CONSEC_FRAMES_MICROSLEEP
    mar_yawn_threshold:       float = MAR_YAWN_THRESHOLD

    # internal state
    mp_face_mesh: Any = field(init=False)
    eye_closed_frames:  int   = 0
    microsleep_count:   int   = 0
    last_yawn_time:     float = 0.0
    yawn_count:         int   = 0
    ear_ema: Optional[float]  = None

    mar_history:    Deque = field(default_factory=lambda: deque(maxlen=300))
    perclos_window: Deque = field(default_factory=lambda: deque(
                                    maxlen=int(PERCLOS_WINDOW_SEC*30)))

    blink_detector: BlinkDetector  = field(init=False)
    ppe_detector:   YOLOPPEDetector= field(init=False)

    prev_gray:          Optional[np.ndarray] = None
    resp_motion_buffer: Deque = field(default_factory=deque)
    chest_roi_abs:      Optional[Tuple] = None
    resp_last_est_bpm:  Optional[float] = None

    rppg_green_buffer: Deque = field(default_factory=deque)
    rppg_red_buffer:   Deque = field(default_factory=deque)
    rppg_last_hr:      Optional[float] = None
    rppg_last_spo2:    Optional[float] = None

    frame_count:         int = 0
    last_face_landmarks: Any = None
    _lm_w: int = 640
    _lm_h: int = 480

    # ── consecutive-closed tracking for microsleep (separate from blink) ──────
    _closed_streak: int = 0   # frames EAR has been below microsleep threshold

    def __post_init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.blink_detector = BlinkDetector()
        self.ppe_detector   = YOLOPPEDetector(PPE_MODEL_PATH)

    # ── MAR helpers ───────────────────────────────────────────────────────────
    def _upd_mar(self, mar, now):
        self.mar_history.append((now, mar))
        while self.mar_history and now - self.mar_history[0][0] > 5.0:
            self.mar_history.popleft()

    def _mar_slope(self):
        pts = list(self.mar_history)
        if len(pts) < 2: return 0.0
        now = pts[-1][0]
        rel = [p for p in pts if p[0] >= now - MAR_SLOPE_WINDOW] or pts[-2:]
        t0,m0 = rel[0]; t1,m1 = rel[-1]
        return (m1-m0) / max(t1-t0, 1e-6)

    def _mar_sustained(self):
        if not self.mar_history: return 0.0
        now = self.mar_history[-1][0]
        dur = 0.0
        for t,m in reversed(self.mar_history):
            if m >= self.mar_yawn_threshold: dur = now - t
            else: break
        return dur

    # ── PERCLOS ───────────────────────────────────────────────────────────────
    def _upd_perclos(self, ear):
        self.perclos_window.append(ear is not None and ear < self.ear_threshold)

    def compute_perclos(self):
        if not self.perclos_window: return 0.0
        return float(sum(self.perclos_window) / len(self.perclos_window))

    # ── Chest ROI ─────────────────────────────────────────────────────────────
    def _set_chest_roi(self, landmarks, w, h):
        ys = [lm.y for lm in landmarks]; xs = [lm.x for lm in landmarks]
        max_y = max(ys)*h; min_x = min(xs)*w; max_x = max(xs)*w
        fw  = max_x - min_x
        rw  = int(max(fw*1.0, w*0.25))
        rx  = int(max(0, (min_x+max_x)/2 - rw/2))
        ry  = int(min(h-1, max_y+10))
        rh  = int(min(h-ry-10, h*0.22))
        if rh <= 10 or rw <= 10:
            ax,ay,aw,ah = FLOW_ROI_REL
            rx,ry,rw,rh = int(w*ax),int(h*ay),int(w*aw),int(h*ah)
        self.chest_roi_abs = (rx,ry,rw,rh)

    # ── Respiration ───────────────────────────────────────────────────────────
    def _append_resp(self, v):
        now = time.time()
        self.resp_motion_buffer.append((now, float(v)))
        while self.resp_motion_buffer and now-self.resp_motion_buffer[0][0]>RESP_BUFFER_SEC:
            self.resp_motion_buffer.popleft()

    def estimate_resp_bpm(self):
        if len(self.resp_motion_buffer) < 8: return None
        times = np.array([t for t,_ in self.resp_motion_buffer])
        vals  = np.array([v for _,v in self.resp_motion_buffer])
        dur = times[-1]-times[0]
        if dur < 6.0: return None
        fs   = len(vals)/dur
        vals_u = np.interp(np.linspace(times[0],times[-1],len(vals)), times, vals-np.mean(vals))
        yf = np.abs(rfft(vals_u)); xf = rfftfreq(len(vals_u),1.0/fs)
        mask = (xf>=RESP_MIN_BPM/60.0)&(xf<=RESP_MAX_BPM/60.0)
        if not np.any(mask): return None
        yf2 = np.where(mask,yf,0.0)
        freq = float(xf[int(np.argmax(yf2))])
        if freq <= 0: return None
        bpm = freq*60.0; self.resp_last_est_bpm = bpm; return bpm

    # ── rPPG ──────────────────────────────────────────────────────────────────
    def _forehead_roi(self, landmarks, w, h):
        xs = np.array([landmarks[p].x for p in FOREHEAD_POINTS])*w
        ys = np.array([landmarks[p].y for p in FOREHEAD_POINTS])*h
        cx,cy = int(np.mean(xs)), int(np.mean(ys))
        ww,hh = int(w*0.18), int(h*0.08)
        return max(0,cx-ww//2), max(0,int(cy-hh*0.6)), ww, hh

    def _append_rppg(self, frame, landmarks):
        fh,fw = frame.shape[:2]
        try: x,y,ww,hh = self._forehead_roi(landmarks,fw,fh)
        except: return
        roi = frame[y:y+hh, x:x+ww]
        if roi.size==0: return
        now = time.time()
        self.rppg_green_buffer.append((now, float(np.mean(roi[:,:,1]))))
        self.rppg_red_buffer.append((now,   float(np.mean(roi[:,:,2]))))
        while self.rppg_green_buffer and now-self.rppg_green_buffer[0][0]>RPPG_WINDOW_SEC:
            self.rppg_green_buffer.popleft()
        while self.rppg_red_buffer   and now-self.rppg_red_buffer[0][0]  >SPO2_WINDOW_SEC:
            self.rppg_red_buffer.popleft()

    def estimate_hr(self):
        if len(self.rppg_green_buffer)<8: return None
        times = np.array([t for t,_ in self.rppg_green_buffer])
        vals  = np.array([v for _,v in self.rppg_green_buffer])
        dur = times[-1]-times[0]
        if dur<8.0: return None
        fs   = len(vals)/dur
        vals_u = np.interp(np.linspace(times[0],times[-1],len(vals)),times,vals-np.mean(vals))
        freq = bandpass_peak_freq(vals_u,fs,HR_MIN_BPM/60.0,HR_MAX_BPM/60.0)
        if freq is None: return None
        hr = freq*60.0; self.rppg_last_hr=hr; return hr

    def estimate_spo2(self):
        if len(self.rppg_red_buffer)<8 or len(self.rppg_green_buffer)<8: return None
        t0 = max(self.rppg_red_buffer[0][0], self.rppg_green_buffer[0][0])
        t1 = min(self.rppg_red_buffer[-1][0],self.rppg_green_buffer[-1][0])
        if t1-t0<6.0: return None
        def _ex(buf):
            ts = np.array([t for t,_ in buf]); vs = np.array([v for _,v in buf])
            return vs[(ts>=t0)&(ts<=t1)]
        rv=_ex(self.rppg_red_buffer); gv=_ex(self.rppg_green_buffer)
        if len(rv)<4 or len(gv)<4: return None
        R = (np.std(rv)/(np.mean(rv)+1e-6)) / (np.std(gv)/(np.mean(gv)+1e-6)+1e-12)
        spo2 = float(np.clip(110.0-25.0*R, 50.0, 100.0))
        self.rppg_last_spo2=spo2; return spo2

    # ── Main per-frame call ───────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray):
        stats: Dict[str,Any] = {}
        h, w = frame.shape[:2]
        now  = time.time()
        self.frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── PPE: non-blocking submit (background thread) ───────────────────
        self.ppe_detector.submit(frame)

        # ── MediaPipe on EVERY frame at 75 % scale ─────────────────────────
        scale   = 0.75
        small   = cv2.resize(frame, (int(w*scale), int(h*scale)))
        sh, sw  = small.shape[:2]
        results = self.mp_face_mesh.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        face_present = False
        if results.multi_face_landmarks:
            self.last_face_landmarks = results.multi_face_landmarks[0].landmark
            self._lm_w = sw; self._lm_h = sh
            face_present = True

        # ── Person detection: face-mesh wins over YOLO (instant feedback) ──
        self.ppe_detector.force_person(face_present)
        stats['ppe'] = self.ppe_detector.get_state()

        lm = self.last_face_landmarks

        if lm:
            lw, lh = self._lm_w, self._lm_h

            # ── EAR ────────────────────────────────────────────────────────
            try:
                ear_l = eye_aspect_ratio(lm, LEFT_EYE_IDX,  lw, lh)
                ear_r = eye_aspect_ratio(lm, RIGHT_EYE_IDX, lw, lh)
                raw_ear = (ear_l + ear_r) / 2.0
            except:
                ear_l = ear_r = raw_ear = None

            # ── MAR ────────────────────────────────────────────────────────
            try:    mar = mouth_aspect_ratio(lm, lw, lh)
            except: mar = None

            # ── EMA (slower alpha = smoother, fewer noise-blinks) ──────────
            if raw_ear is not None:
                self.ear_ema = (raw_ear if self.ear_ema is None
                                else EAR_EMA_ALPHA*raw_ear + (1-EAR_EMA_ALPHA)*self.ear_ema)
            ear = self.ear_ema
            stats['ear'] = ear
            stats['mar'] = mar

            if mar is not None: self._upd_mar(mar, now)

            # ── Blink detection ────────────────────────────────────────────
            if ear_l is not None and ear_r is not None:
                self.blink_detector.update(ear_l, ear_r, now)
            stats['blink_count'] = self.blink_detector.get_count()

            # ── Microsleep (long eye-closure) ──────────────────────────────
            if raw_ear is not None and raw_ear < self.ear_threshold:
                self._closed_streak += 1
            else:
                if self._closed_streak >= self.ear_consec_frames_thresh:
                    self.microsleep_count += 1
                    print(f"[MICROSLEEP] #{self.microsleep_count}  streak={self._closed_streak}f")
                self._closed_streak = 0

            stats['eye_closed_frames'] = self._closed_streak
            stats['microsleep_count']  = self.microsleep_count

            # ── Yawn detection ─────────────────────────────────────────────
            if (mar is not None
                    and mar >= self.mar_yawn_threshold
                    and self._mar_sustained() >= MAR_SUSTAIN_TIME
                    and abs(self._mar_slope()) <= MAR_SLOPE_MAX
                    and now - self.last_yawn_time > YAWN_MIN_SEPARATION):
                self.yawn_count  += 1
                self.last_yawn_time = now
                print(f"[YAWN] #{self.yawn_count}  MAR={mar:.3f}")
            stats['yawn_count'] = self.yawn_count

            # ── PERCLOS ────────────────────────────────────────────────────
            self._upd_perclos(ear)
            stats['perclos'] = self.compute_perclos()

            # ── Chest ROI init ─────────────────────────────────────────────
            if self.chest_roi_abs is None:
                try:    self._set_chest_roi(lm, w, h)
                except:
                    ax,ay,aw,ah = FLOW_ROI_REL
                    self.chest_roi_abs=(int(w*ax),int(h*ay),int(w*aw),int(h*ah))

            # ── rPPG every 3rd frame ───────────────────────────────────────
            if self.frame_count % 3 == 0:
                self._append_rppg(frame, lm)

        else:
            # No face this frame — maintain last values
            stats['ear']              = None
            stats['mar']              = None
            stats['eye_closed_frames']= self._closed_streak
            stats['microsleep_count'] = self.microsleep_count
            stats['yawn_count']       = self.yawn_count
            stats['perclos']          = self.compute_perclos()
            stats['blink_count']      = self.blink_detector.get_count()

            if self.chest_roi_abs is None:
                ax,ay,aw,ah = FLOW_ROI_REL
                self.chest_roi_abs=(int(w*ax),int(h*ay),int(w*aw),int(h*ah))

        # ── Optical flow every 3rd frame ──────────────────────────────────
        if self.frame_count % 3 == 0:
            try:
                x,y,rw,rh = self.chest_roi_abs
                x  = max(0,min(x,w-1)); y  = max(0,min(y,h-1))
                rw = max(8,min(rw,w-x)); rh = max(8,min(rh,h-y))
                roi = gray[y:y+rh, x:x+rw]
                if self.prev_gray is not None:
                    prev_roi = self.prev_gray[y:y+rh, x:x+rw]
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_roi, roi, None,
                        pyr_scale=0.5, levels=2, winsize=10,
                        iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
                    self._append_resp(float(np.mean(flow[...,1])))
                self.prev_gray = gray.copy()
            except:
                self.prev_gray = gray.copy()

        # ── Vitals ────────────────────────────────────────────────────────
        rb  = self.estimate_resp_bpm()
        hr  = self.estimate_hr()
        spo = self.estimate_spo2()
        stats['resp_bpm'] = rb  if rb  is not None else (self.resp_last_est_bpm or 0.0)
        stats['hr_bpm']   = hr  if hr  is not None else (self.rppg_last_hr      or 0.0)
        stats['spo2']     = spo if spo is not None else (self.rppg_last_spo2    or 0.0)

        # ── Overlay ───────────────────────────────────────────────────────
        frame = self.ppe_detector.draw(frame, stats['ppe'])
        return frame, stats


# ── Fatigue Estimator ─────────────────────────────────────────────────────────
@dataclass
class FatigueEstimator:
    weight_eye:  float = WEIGHT_EYE
    weight_yawn: float = WEIGHT_YAWN
    weight_resp: float = WEIGHT_RESP
    smoothed:    float = 0.0

    def compute_raw(self, perclos, yawn_count, ms_count, resp_bpm):
        eye   = np.clip(perclos, 0.0, 1.0)
        yawn  = np.clip(yawn_count/5.0, 0.0, 1.0)
        ms_f  = 1.0 - math.exp(-0.5*ms_count)
        eye   = 0.7*eye + 0.3*ms_f
        if resp_bpm <= 0:
            resp = 0.0
        elif resp_bpm < 12:
            resp = np.clip((12-resp_bpm)/12, 0.0, 1.0)
        elif resp_bpm > 20:
            resp = np.clip((resp_bpm-20)/20, 0.0, 1.0)
        else:
            resp = 0.0
        return float(np.clip((self.weight_eye*eye + self.weight_yawn*yawn + self.weight_resp*resp)*100, 0.0, 100.0))

    def smooth(self, raw):
        self.smoothed = FATIGUE_EMA_ALPHA*raw + (1-FATIGUE_EMA_ALPHA)*self.smoothed
        return self.smoothed


# ── Miner Database ────────────────────────────────────────────────────────────
class MinerDatabase:
    def __init__(self):
        self.miners: Dict[str,Dict] = {}
        self.lock = threading.Lock()
        for m in [{"name":"John Smith","id":"MN001","shift":"Morning","gender":"Male"},
                  {"name":"Sarah Johnson","id":"MN002","shift":"Afternoon","gender":"Female"},
                  {"name":"Michael Brown","id":"MN003","shift":"Night","gender":"Male"}]:
            self.add_miner(m['name'],m['id'],m['shift'],m['gender'])

    def add_miner(self,name,mid,shift,gender):
        with self.lock:
            if mid in self.miners: return {"error":"ID exists"}
            self.miners[mid]={"name":name,"id":mid,"shift":shift,"gender":gender}
            return {"success":True,"miner":self.miners[mid]}

    def delete_miner(self,mid):
        with self.lock:
            if mid not in self.miners: return {"error":"Not found"}
            del self.miners[mid]; return {"success":True}

    def get_all(self):
        with self.lock: return list(self.miners.values())

    def get(self,mid):
        with self.lock: return self.miners.get(mid)


# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


class MonitoringSystem:
    def __init__(self):
        self.vp    = VideoProcessor()
        self.fe    = FatigueEstimator()
        self.am    = AlertManager()
        self.db    = MinerDatabase()
        self.lock  = threading.Lock()
        self._frame_lock = threading.Lock()
        self.is_running  = False
        self.cap         = None
        self.cur_frame   = None
        self.cur_stats   = {}
        self.cur_miner   = None
        self._thread     = None

    # ── Start / Stop / Reset ──────────────────────────────────────────────────
    def start(self, miner_id):
        with self.lock:
            if self.is_running: return {"status":"already_running"}
            m = self.db.get(miner_id)
            if not m: return {"status":"error","message":"Miner not found"}
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened(): return {"status":"error","message":"Cannot open webcam"}
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS,          30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cur_miner  = miner_id
            self.is_running = True
            self._thread    = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return {"status":"started","miner":m}

    def stop(self):
        with self.lock:
            if not self.is_running: return {"status":"not_running"}
            self.is_running = False
            self.cur_miner  = None
            if self.cap: self.cap.release(); self.cap = None
            with self._frame_lock: self.cur_frame = None
            self.cur_stats = {}
            return {"status":"stopped"}

    def reset(self):
        vp = self.vp
        vp.microsleep_count=0; vp.yawn_count=0
        vp._closed_streak=0;   vp.ear_ema=None
        vp.mar_history.clear(); vp.perclos_window.clear()
        vp.resp_motion_buffer.clear(); vp.prev_gray=None
        vp.chest_roi_abs=None; vp.resp_last_est_bpm=None
        vp.rppg_green_buffer.clear(); vp.rppg_red_buffer.clear()
        vp.rppg_last_hr=None; vp.rppg_last_spo2=None
        vp.frame_count=0; vp.last_face_landmarks=None
        vp.blink_detector = BlinkDetector()
        vp.ppe_detector.reset()
        self.fe.smoothed=0.0
        self.am.clear_alerts()
        with self.lock:
            self.cur_stats={
                'ear':None,'mar':None,'blink_count':0,'eye_closed_frames':0,
                'microsleep_count':0,'yawn_count':0,'perclos':0.0,
                'resp_bpm':0.0,'hr_bpm':0.0,'spo2':0.0,'fatigue_score':0.0,
                'ppe':{k:False for k in YOLOPPEDetector._ITEMS}}
        return {"status":"reset"}

    # ── Main capture/processing loop ──────────────────────────────────────────
    def _loop(self):
        while self.is_running:
            if not self.cap.grab():
                time.sleep(0.005); continue

            ret, frame = self.cap.retrieve()
            if not ret: continue

            processed, stats = self.vp.process_frame(frame)

            # ── Fatigue ───────────────────────────────────────────────────
            raw   = self.fe.compute_raw(stats.get('perclos',0),
                                        stats.get('yawn_count',0),
                                        stats.get('microsleep_count',0),
                                        stats.get('resp_bpm',0))
            stats['fatigue_score'] = self.fe.smooth(raw)

            # ── Alerts ────────────────────────────────────────────────────
            perclos = stats.get('perclos',0)
            fs      = stats['fatigue_score']
            ms      = stats.get('microsleep_count',0)
            yn      = stats.get('yawn_count',0)
            rr      = stats.get('resp_bpm',0)
            hr      = stats.get('hr_bpm',0)
            sp      = stats.get('spo2',0)
            ppe     = stats.get('ppe',{})

            if perclos >= ALERT_PERCLOS:             self.am.trigger("HIGH_PERCLOS",f"PERCLOS={perclos:.2f}")
            if fs      >= ALERT_FATIGUE:             self.am.trigger("HIGH_FATIGUE",f"Score={fs:.1f}")
            if ms      >= ALERT_MICROSLEEP_COUNT:    self.am.trigger("MICROSLEEP",  f"Count={ms}")
            if yn      >= ALERT_YAWN_COUNT:          self.am.trigger("FREQUENT_YAWNS",f"Yawns={yn}")
            if rr>0 and (rr<ALERT_RESP_LOW or rr>ALERT_RESP_HIGH): self.am.trigger("RESP_RATE",f"{rr:.1f}bpm")
            if hr>0 and (hr<ALERT_HR_LOW   or hr>ALERT_HR_HIGH):   self.am.trigger("HEART_RATE",f"{hr:.0f}bpm")
            if sp>0 and  sp<ALERT_SPO2_LOW:          self.am.trigger("LOW_SPO2",   f"{sp:.1f}%")

            if PPE_ALERT_ENABLED and ppe.get('person'):
                for item,tag in [('helmet','NO_HELMET'),('goggles','NO_GOGGLES'),
                                  ('vest','NO_VEST'),  ('gloves','NO_GLOVES'),('boots','NO_BOOTS')]:
                    if not ppe.get(item): self.am.trigger(tag, f"{item.capitalize()} not detected")

            # ── Store ─────────────────────────────────────────────────────
            with self._frame_lock: self.cur_frame = processed
            with self.lock:        self.cur_stats = stats

    # ── Accessors ─────────────────────────────────────────────────────────────
    def get_frame(self):
        with self._frame_lock: return self.cur_frame

    def get_stats(self):
        with self.lock: return self.cur_stats.copy()

    def active(self):
        with self.lock: return self.is_running


sys = MonitoringSystem()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@app.route('/monitor/<mid>')
def monitor_page(mid):
    m = sys.db.get(mid)
    if not m: return "Miner not found", 404
    return render_template_string(MONITOR_TEMPLATE, miner=m)

@app.route('/api/miners',          methods=['GET'])
def get_miners():  return jsonify({"miners": sys.db.get_all()})

@app.route('/api/miners',          methods=['POST'])
def add_miner():
    d = request.json
    return jsonify(sys.db.add_miner(d.get('name',''), d.get('id',''),
                                    d.get('shift',''), d.get('gender','')))

@app.route('/api/miners/<mid>',    methods=['DELETE'])
def del_miner(mid): return jsonify(sys.db.delete_miner(mid))

@app.route('/api/start/<mid>',     methods=['POST'])
def start(mid): return jsonify(sys.start(mid))

@app.route('/api/stop',            methods=['POST'])
def stop():  return jsonify(sys.stop())

@app.route('/api/reset',           methods=['POST'])
def reset(): return jsonify(sys.reset())

@app.route('/api/stats')
def stats(): return jsonify(sys.get_stats())

@app.route('/api/alerts')
def alerts(): return jsonify({"alerts": sys.am.get_alerts()})

@app.route('/api/monitoring_status')
def mon_status(): return jsonify({"is_monitoring": sys.active()})


# ── Streaming ─────────────────────────────────────────────────────────────────
def gen_frames():
    enc_params = [cv2.IMWRITE_JPEG_QUALITY, 55]
    last_id    = None
    while True:
        if sys.active():
            with sys._frame_lock:
                frame = sys.cur_frame
            if frame is None:
                time.sleep(0.01); continue
            fid = id(frame)
            if fid == last_id:
                time.sleep(0.008); continue
            last_id = fid
            ret, buf = cv2.imencode('.jpg', frame, enc_params)
            if ret:
                data = buf.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(data)).encode() + b'\r\n\r\n'
                       + data + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ── HTML Templates ────────────────────────────────────────────────────────────
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Miner Management System</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:'Segoe UI',sans-serif; background:linear-gradient(135deg,#0f0f23,#1a1a2e); color:#e0e0e0; min-height:100vh; padding:20px; }
        .container { max-width:1400px; margin:0 auto; }
        .header { background:linear-gradient(135deg,#1a1a2e,#16213e); padding:30px; border-radius:15px; box-shadow:0 8px 32px rgba(0,0,0,.4); border:1px solid rgba(255,255,255,.1); margin-bottom:30px; }
        .header h1 { font-size:32px; background:linear-gradient(135deg,#667eea,#764ba2); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:10px; }
        .header p { color:#9ca3af; font-size:16px; }
        .actions-bar { display:flex; justify-content:space-between; align-items:center; margin-bottom:25px; flex-wrap:wrap; gap:15px; }
        .search-box { flex:1; min-width:250px; }
        .search-box input { width:100%; padding:12px 20px; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.1); border-radius:10px; color:#e0e0e0; font-size:14px; }
        .search-box input:focus { outline:none; border-color:#667eea; }
        .btn { padding:12px 28px; border:none; border-radius:10px; font-size:14px; font-weight:600; cursor:pointer; transition:all .3s; text-transform:uppercase; }
        .btn-primary { background:linear-gradient(135deg,#667eea,#764ba2); color:white; }
        .btn-primary:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(102,126,234,.5); }
        .miners-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:20px; }
        .miner-card { background:linear-gradient(135deg,#1a1a2e,#16213e); border-radius:15px; padding:25px; box-shadow:0 8px 32px rgba(0,0,0,.4); border:1px solid rgba(255,255,255,.1); transition:all .3s; position:relative; overflow:hidden; }
        .miner-card::before { content:''; position:absolute; top:0; left:0; right:0; height:4px; background:linear-gradient(90deg,#667eea,#764ba2); transform:scaleX(0); transition:transform .3s; }
        .miner-card:hover::before { transform:scaleX(1); }
        .miner-card:hover { transform:translateY(-5px); box-shadow:0 12px 40px rgba(102,126,234,.3); }
        .miner-header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:15px; }
        .miner-info h3 { font-size:20px; color:#fff; margin-bottom:5px; }
        .miner-id { font-size:13px; color:#667eea; font-weight:600; }
        .delete-btn { background:rgba(239,68,68,.2); border:1px solid rgba(239,68,68,.3); color:#ef4444; padding:6px 12px; border-radius:6px; font-size:12px; cursor:pointer; transition:all .3s; }
        .delete-btn:hover { background:rgba(239,68,68,.3); }
        .miner-details { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:15px; }
        .detail-item { background:rgba(255,255,255,.03); padding:10px; border-radius:8px; border:1px solid rgba(255,255,255,.05); }
        .detail-label { font-size:11px; color:#9ca3af; text-transform:uppercase; margin-bottom:4px; }
        .detail-value { font-size:14px; color:#fff; font-weight:600; }
        .monitor-btn { width:100%; margin-top:15px; padding:12px; background:linear-gradient(135deg,#10b981,#059669); color:white; border:none; border-radius:8px; font-weight:600; cursor:pointer; transition:all .3s; text-transform:uppercase; }
        .monitor-btn:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(16,185,129,.4); }
        .modal { display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,.8); backdrop-filter:blur(5px); z-index:1000; align-items:center; justify-content:center; }
        .modal.active { display:flex; }
        .modal-content { background:linear-gradient(135deg,#1a1a2e,#16213e); padding:35px; border-radius:15px; box-shadow:0 20px 60px rgba(0,0,0,.6); border:1px solid rgba(255,255,255,.1); max-width:500px; width:90%; }
        .modal-header { margin-bottom:25px; }
        .modal-header h2 { font-size:24px; color:#fff; }
        .form-group { margin-bottom:20px; }
        .form-group label { display:block; margin-bottom:8px; color:#9ca3af; font-size:13px; font-weight:600; text-transform:uppercase; }
        .form-group input,.form-group select { width:100%; padding:12px 16px; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.1); border-radius:8px; color:#e0e0e0; font-size:14px; }
        .form-group input:focus,.form-group select:focus { outline:none; border-color:#667eea; }
        .modal-actions { display:flex; gap:12px; margin-top:25px; }
        .btn-secondary { background:rgba(255,255,255,.1); color:#e0e0e0; flex:1; }
        .btn-submit { flex:1; }
        .empty-state { text-align:center; padding:60px 20px; background:linear-gradient(135deg,#1a1a2e,#16213e); border-radius:15px; border:2px dashed rgba(255,255,255,.1); }
        .empty-state-icon { font-size:64px; margin-bottom:20px; }
        .empty-state h3 { font-size:20px; color:#fff; margin-bottom:10px; }
        .empty-state p { color:#9ca3af; margin-bottom:25px; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>⛏️ Miner Monitoring System</h1>
        <p>AI-Enabled Health & Safety Monitoring for Mining Workers</p>
    </div>
    <div class="actions-bar">
        <div class="search-box"><input type="text" id="searchInput" placeholder="Search miners by name or ID..."></div>
        <button class="btn btn-primary" onclick="openAdd()">+ Add New Miner</button>
    </div>
    <div id="minersContainer" class="miners-grid"></div>
</div>

<div id="addModal" class="modal">
    <div class="modal-content">
        <div class="modal-header"><h2>Add New Miner</h2></div>
        <form id="addForm">
            <div class="form-group"><label>Name</label><input type="text" id="fName" required placeholder="Full name"></div>
            <div class="form-group"><label>Miner ID</label><input type="text" id="fId" required placeholder="e.g. MN004"></div>
            <div class="form-group"><label>Shift</label>
                <select id="fShift" required><option value="">Select</option><option>Morning</option><option>Afternoon</option><option>Night</option></select></div>
            <div class="form-group"><label>Gender</label>
                <select id="fGender" required><option value="">Select</option><option>Male</option><option>Female</option><option>Other</option></select></div>
            <div class="modal-actions">
                <button type="button" class="btn btn-secondary" onclick="closeAdd()">Cancel</button>
                <button type="submit" class="btn btn-primary btn-submit">Add Miner</button>
            </div>
        </form>
    </div>
</div>

<script>
let allMiners=[];
function load(){
    fetch('/api/miners').then(r=>r.json()).then(d=>{allMiners=d.miners;render(allMiners);});
}
function render(miners){
    const c=document.getElementById('minersContainer');
    if(!miners.length){
        c.innerHTML=`<div class="empty-state" style="grid-column:1/-1"><div class="empty-state-icon"></div><h3>No Miners Found</h3><p>Add your first miner to start monitoring</p><button class="btn btn-primary" onclick="openAdd()">+ Add</button></div>`;
        return;
    }
    c.innerHTML=miners.map(m=>`
        <div class="miner-card">
            <div class="miner-header">
                <div class="miner-info"><h3>${m.name}</h3><div class="miner-id">${m.id}</div></div>
                <button class="delete-btn" onclick="del('${m.id}',event)">✕</button>
            </div>
            <div class="miner-details">
                <div class="detail-item"><div class="detail-label">Shift</div><div class="detail-value">${m.shift}</div></div>
                <div class="detail-item"><div class="detail-label">Gender</div><div class="detail-value">${m.gender}</div></div>
            </div>
            <button class="monitor-btn" onclick="location.href='/monitor/${m.id}'"> Start Monitoring</button>
        </div>`).join('');
}
function openAdd(){document.getElementById('addModal').classList.add('active');}
function closeAdd(){document.getElementById('addModal').classList.remove('active');document.getElementById('addForm').reset();}
document.getElementById('addForm').addEventListener('submit',e=>{
    e.preventDefault();
    fetch('/api/miners',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({name:fName.value,id:fId.value,shift:fShift.value,gender:fGender.value})})
    .then(r=>r.json()).then(d=>{if(d.error)alert(d.error);else{closeAdd();load();}});
});
function del(id,e){
    e.stopPropagation();
    if(!confirm('Delete this miner?')) return;
    fetch(`/api/miners/${id}`,{method:'DELETE'}).then(r=>r.json()).then(d=>{if(d.error)alert(d.error);else load();});
}
document.getElementById('searchInput').addEventListener('input',e=>{
    const t=e.target.value.toLowerCase();
    render(allMiners.filter(m=>m.name.toLowerCase().includes(t)||m.id.toLowerCase().includes(t)));
});
document.getElementById('addModal').addEventListener('click',e=>{if(e.target===e.currentTarget)closeAdd();});
load();
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
    <title>Monitoring – {{ miner.name }}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        *{margin:0;padding:0;box-sizing:border-box;}
        body{font-family:'Segoe UI',sans-serif;background:#0f0f23;color:#e0e0e0;min-height:100vh;}
        .container{display:grid;grid-template-columns:260px 1fr;grid-template-rows:auto 1fr;gap:14px;padding:14px;max-width:1920px;margin:0 auto;min-height:100vh;}
        .header{grid-column:1/-1;background:linear-gradient(135deg,#1a1a2e,#16213e);padding:18px 28px;border-radius:14px;border:1px solid rgba(255,255,255,.1);}
        .header-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;}
        .miner-info-hdr{display:flex;align-items:center;gap:14px;}
        .avatar{width:48px;height:48px;border-radius:50%;background:linear-gradient(135deg,#667eea,#764ba2);display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:bold;color:#fff;}
        .miner-info-hdr h1{font-size:22px;color:#fff;margin-bottom:4px;}
        .meta{font-size:12px;color:#9ca3af;}
        .back-btn{padding:9px 18px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);border-radius:8px;color:#e0e0e0;text-decoration:none;font-size:13px;font-weight:600;transition:all .3s;}
        .back-btn:hover{background:rgba(255,255,255,.18);}
        .controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;}
        .btn{padding:9px 22px;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:all .3s;text-transform:uppercase;}
        .btn-start{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;}
        .btn-stop{background:linear-gradient(135deg,#f093fb,#f5576c);color:#fff;}
        .btn-reset{background:linear-gradient(135deg,#4facfe,#00f2fe);color:#fff;}
        .badge{display:inline-block;padding:7px 16px;border-radius:20px;font-weight:600;font-size:12px;}
        .badge-active{background:#48bb78;color:#fff;animation:pulse 2s infinite;}
        .badge-inactive{background:#4a5568;color:#cbd5e0;}
        @keyframes pulse{0%,100%{opacity:1;}50%{opacity:.7;}}
        /* sidebar alerts */
        .alerts-panel{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:14px;padding:18px;border:1px solid rgba(255,255,255,.1);max-height:calc(100vh - 110px);overflow-y:auto;}
        .alert-item{background:rgba(239,68,68,.15);border-left:4px solid #ef4444;padding:10px;margin-bottom:9px;border-radius:7px;}
        .main{display:flex;flex-direction:column;gap:14px;}
        .top-row{display:grid;grid-template-columns:1.15fr 1fr;gap:14px;}
        .panel{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:14px;padding:18px;border:1px solid rgba(255,255,255,.1);}
        .video-box{position:relative;background:#000;border-radius:10px;min-height:360px;display:flex;align-items:center;justify-content:center;}
        .video-box img{width:100%;height:auto;max-height:360px;object-fit:contain;border-radius:10px;display:block;}
        .placeholder{color:#6b7280;font-size:15px;text-align:center;}
        .det-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
        .det-item{background:rgba(255,255,255,.04);border-radius:9px;padding:13px;border:1px solid rgba(255,255,255,.08);}
        .det-label{font-size:11px;color:#9ca3af;margin-bottom:5px;text-transform:uppercase;}
        .det-val{font-size:21px;font-weight:700;color:#fff;}
        .det-val.ok{color:#10b981;} .det-val.warn{color:#f59e0b;} .det-val.bad{color:#ef4444;}

        /* ── PPE Grid: fixed layout so cards stay compact and horizontal ── */
        .ppe-grid{
            display:grid;
            grid-template-columns:repeat(3,1fr);
            grid-auto-rows:90px;        /* fixed row height — prevents tall cards */
            gap:10px;
            align-items:stretch;
        }
        .ppe-item{
            background:rgba(255,255,255,.04);
            border-radius:10px;
            padding:10px 8px;
            border:1px solid rgba(255,255,255,.08);
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            gap:4px;
            text-align:center;
            min-height:0;               /* override any implicit stretch */
            overflow:hidden;
        }
        .ppe-item.yes{border-color:#10b981;background:rgba(16,185,129,.12);}
        .ppe-item.no {border-color:#ef4444;background:rgba(239,68,68,.12);}
        .ppe-icon{font-size:22px;line-height:1;flex-shrink:0;}
        .ppe-lbl{font-size:10px;color:#9ca3af;text-transform:uppercase;letter-spacing:.5px;line-height:1;}
        .ppe-st{font-size:12px;font-weight:700;line-height:1;}
        .ppe-st.yes{color:#10b981;}
        .ppe-st.no {color:#ef4444;}

        .chart-box{background:rgba(255,255,255,.03);padding:18px;border-radius:10px;height:340px;}
        .no-alerts{text-align:center;padding:25px;color:#6b7280;font-style:italic;}
        h2{font-size:17px;color:#fff;margin-bottom:13px;}
        .person-badge{display:inline-block;padding:6px 14px;border-radius:16px;font-size:11px;font-weight:700;}
        .person-badge.yes{background:#10b981;color:#fff;}
        .person-badge.no{background:#ef4444;color:#fff;}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <div class="header-top">
            <div class="miner-info-hdr">
                <div class="avatar">{{ miner.name[0] }}</div>
                <div>
                    <h1>{{ miner.name }}</h1>
                    <div class="meta">ID: {{ miner.id }} | Shift: {{ miner.shift }} | {{ miner.gender }}</div>
                </div>
            </div>
            <a href="/" class="back-btn">← Back</a>
        </div>
        <div class="controls">
            <button class="btn btn-start" onclick="startMon()">Start</button>
            <button class="btn btn-stop"  onclick="stopMon()">Stop</button>
            <button class="btn btn-reset" onclick="resetMon()"> Reset</button>
            <span class="badge badge-inactive" id="badge">INACTIVE</span>
        </div>
    </div>

    <!-- Sidebar -->
    <div class="alerts-panel">
        <h2> Alerts</h2>
        <div id="alertsList"><div class="no-alerts">No alerts yet</div></div>
    </div>

    <!-- Main -->
    <div class="main">
        <div class="top-row">
            <!-- Video -->
            <div class="panel">
                <h2>Live Feed</h2>
                <div class="video-box" id="vbox"><div class="placeholder">Click Start to begin</div></div>
            </div>
            <!-- Metrics -->
            <div class="panel">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:13px;">
                    <h2 style="margin:0;"> Fatigue Metrics</h2>
                    <span class="person-badge no" id="personBadge">No Person</span>
                </div>
                <div class="det-grid">
                    <div class="det-item"><div class="det-label">Fatigue Score</div><div class="det-val" id="v-fatigue">--</div></div>
                    <div class="det-item"><div class="det-label">Heart Rate</div><div class="det-val" id="v-hr">--</div></div>
                    <div class="det-item"><div class="det-label">Respiration</div><div class="det-val" id="v-resp">--</div></div>
                    <div class="det-item"><div class="det-label">SpO₂ (%)</div><div class="det-val" id="v-spo2">--</div></div>
                    <div class="det-item"><div class="det-label">PERCLOS</div><div class="det-val" id="v-perclos">--</div></div>
                    <div class="det-item"><div class="det-label">EAR</div><div class="det-val" id="v-ear">--</div></div>
                    <div class="det-item"><div class="det-label">Blinks </div><div class="det-val" id="v-blink">--</div></div>
                    <div class="det-item"><div class="det-label">Microsleeps </div><div class="det-val" id="v-ms">--</div></div>
                    <div class="det-item"><div class="det-label">Yawns </div><div class="det-val" id="v-yawn">--</div></div>
                    <div class="det-item"><div class="det-label">MAR</div><div class="det-val" id="v-mar">--</div></div>
                </div>
            </div>
        </div>

        <!-- PPE -->
        <div class="panel">
            <h2> PPE Status</h2>
            <div class="ppe-grid">
                <div class="ppe-item" id="p-person"> <div class="ppe-icon"></div><div class="ppe-lbl">Person</div> <div class="ppe-st">--</div></div>
                <div class="ppe-item" id="p-helmet"> <div class="ppe-icon">️</div><div class="ppe-lbl">Helmet</div> <div class="ppe-st">--</div></div>
                <div class="ppe-item" id="p-goggles"><div class="ppe-icon"></div><div class="ppe-lbl">Goggles</div><div class="ppe-st">--</div></div>
                <div class="ppe-item" id="p-vest">   <div class="ppe-icon"></div><div class="ppe-lbl">Vest</div>   <div class="ppe-st">--</div></div>
                <div class="ppe-item" id="p-gloves"> <div class="ppe-icon"></div><div class="ppe-lbl">Gloves</div> <div class="ppe-st">--</div></div>
                <div class="ppe-item" id="p-boots">  <div class="ppe-icon"></div><div class="ppe-lbl">Boots</div>  <div class="ppe-st">--</div></div>
            </div>
        </div>

        <!-- Chart -->
        <div class="panel">
            <h2>Real-Time Metrics</h2>
            <div class="chart-box"><canvas id="chart"></canvas></div>
        </div>
    </div>
</div>

<script>
const MID='{{ miner.id }}';
const MAX_PTS=60;
let monitoring=false, personOk=false, videoEl=null;

const chart=new Chart(document.getElementById('chart'),{
    type:'line',
    data:{labels:[],datasets:[
        {label:'Fatigue',data:[],yAxisID:'y', borderColor:'#f5576c',backgroundColor:'rgba(245,87,108,.15)',tension:.4,borderWidth:2,fill:true},
        {label:'HR',     data:[],yAxisID:'y1',borderColor:'#4facfe',backgroundColor:'rgba(79,172,254,.1)', tension:.4,borderWidth:2},
        {label:'Resp',   data:[],yAxisID:'y2',borderColor:'#43e97b',backgroundColor:'transparent',        tension:.4,borderWidth:1,hidden:true},
        {label:'SpO2',   data:[],yAxisID:'y3',borderColor:'#fa709a',backgroundColor:'transparent',        tension:.4,borderWidth:1,hidden:true}
    ]},
    options:{
        responsive:true,maintainAspectRatio:false,animation:false,
        scales:{
            x:{grid:{color:'rgba(255,255,255,.05)'},ticks:{color:'#9ca3af',maxTicksLimit:10}},
            y:{position:'left',min:0,max:100,grid:{color:'rgba(255,255,255,.05)'},ticks:{color:'#9ca3af'},title:{display:true,text:'Fatigue',color:'#9ca3af'}},
            y1:{position:'right',min:40,max:180,grid:{drawOnChartArea:false},ticks:{color:'#9ca3af'}},
            y2:{min:0,max:35,display:false},
            y3:{min:85,max:100,display:false}
        },
        plugins:{legend:{labels:{color:'#e0e0e0'}}}
    }
});

function pushChart(f,h,r,s){
    if(!monitoring||!personOk) return;
    if(chart.data.labels.length>=MAX_PTS){chart.data.labels.shift();chart.data.datasets.forEach(d=>d.data.shift());}
    chart.data.labels.push(new Date().toLocaleTimeString());
    [f,h,r,s].forEach((v,i)=>chart.data.datasets[i].data.push(v));
    chart.update('none');
}

function updPPE(ppe){
    ['person','helmet','goggles','vest','gloves','boots'].forEach(k=>{
        const el=document.getElementById(`p-${k}`);if(!el)return;
        const ok=!!ppe[k];
        el.className=`ppe-item ${ok?'yes':'no'}`;
        el.querySelector('.ppe-st').textContent=ok?'✓ ON':'✗ OFF';
        el.querySelector('.ppe-st').className=`ppe-st ${ok?'yes':'no'}`;
    });
}

function updMetrics(s){
    const set=(id,v,fmt)=>{ const el=document.getElementById(id); el.textContent=(v!==null&&v!==undefined&&v!==0&&v!=='')?(fmt?fmt(v):v):'--'; };
    set('v-fatigue',s.fatigue_score,v=>v.toFixed(1));
    set('v-hr',     s.hr_bpm>0?s.hr_bpm:null, v=>Math.round(v));
    set('v-resp',   s.resp_bpm>0?s.resp_bpm:null, v=>v.toFixed(1));
    set('v-spo2',   s.spo2>0?s.spo2:null, v=>v.toFixed(1));
    set('v-perclos',s.perclos,v=>v.toFixed(3));
    set('v-ear',    s.ear,v=>v.toFixed(3));
    set('v-mar',    s.mar,v=>v.toFixed(3));
    document.getElementById('v-blink').textContent=s.blink_count??'--';
    document.getElementById('v-ms').textContent   =s.microsleep_count??'--';
    document.getElementById('v-yawn').textContent =s.yawn_count??'--';
    if(s.ppe) updPPE(s.ppe);
}

function setVideo(on){
    const b=document.getElementById('vbox');
    if(on){
        if(!videoEl){
            const img=document.createElement('img');
            img.src='/video_feed?t='+Date.now();
            b.innerHTML='';b.appendChild(img);videoEl=img;
        }
    } else {
        b.innerHTML='<div class="placeholder" style="background:#000;width:100%;height:100%;display:flex;align-items:center;justify-content:center;">Monitoring stopped</div>';
        videoEl=null;
    }
}

function setBadge(active){
    const el=document.getElementById('badge');
    el.textContent=active?'● ACTIVE':'● INACTIVE';
    el.className=`badge ${active?'badge-active':'badge-inactive'}`;
}

function setPersonBadge(ok){
    const el=document.getElementById('personBadge');
    el.textContent=ok?'Person Detected':'No Person';
    el.className=`person-badge ${ok?'yes':'no'}`;
}

async function startMon(){
    const r=await fetch(`/api/start/${MID}`,{method:'POST'}).then(r=>r.json());
    if(r.status==='started'||r.status==='already_running'){
        monitoring=true; setBadge(true); setVideo(true);
    }
}
async function stopMon(){
    await fetch('/api/stop',{method:'POST'});
    monitoring=false; personOk=false;
    setBadge(false); setVideo(false); setPersonBadge(false);
}
async function resetMon(){
    const r=await fetch('/api/reset',{method:'POST'}).then(r=>r.json());
    if(r.status==='reset'){
        chart.data.labels=[];chart.data.datasets.forEach(d=>d.data=[]);chart.update();
        ['v-fatigue','v-hr','v-resp','v-spo2','v-perclos','v-ear','v-mar','v-blink','v-ms','v-yawn']
            .forEach(id=>document.getElementById(id).textContent='--');
        ['person','helmet','goggles','vest','gloves','boots'].forEach(k=>{
            const el=document.getElementById(`p-${k}`);if(!el)return;
            el.className='ppe-item';el.querySelector('.ppe-st').textContent='--';el.querySelector('.ppe-st').className='ppe-st';
        });
        document.getElementById('alertsList').innerHTML='<div class="no-alerts">No alerts yet</div>';
    }
}

// Stats poll — 100 ms
setInterval(async()=>{
    if(!monitoring) return;
    try{
        const s=await fetch('/api/stats').then(r=>r.json());
        personOk=!!(s.ppe&&s.ppe.person);
        setPersonBadge(personOk);
        if(s.ppe) updPPE(s.ppe);
        if(personOk){ updMetrics(s); pushChart(s.fatigue_score||0,s.hr_bpm||0,s.resp_bpm||0,s.spo2||0); }
    }catch(e){}
},100);

// Alerts poll — 2 s
setInterval(async()=>{
    if(!monitoring||!personOk) return;
    try{
        const d=await fetch('/api/alerts').then(r=>r.json());
        const list=document.getElementById('alertsList');
        if(d.alerts?.length){
            list.innerHTML=d.alerts.slice(-20).reverse().map(a=>`
                <div class="alert-item">
                    <div style="font-size:10px;color:#9ca3af">${a.timestamp}</div>
                    <div style="font-weight:700;color:#fca5a5">${a.name}</div>
                    <div style="color:#fecaca;font-size:12px">${a.message}</div>
                </div>`).join('');
        } else list.innerHTML='<div class="no-alerts">No alerts yet</div>';
    }catch(e){}
},2000);

setVideo(false);
</script>
</body>
</html>
'''

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*70)
    print("⛏️  MINER MONITORING SYSTEM")
    print("="*70)
    if not YOLO_AVAILABLE:
        print(" YOLOv8 not available (pip install ultralytics)")
    elif not os.path.exists(PPE_MODEL_PATH):
        print(f" YOLO model not found at {PPE_MODEL_PATH}")
    else:
        print(f"✓  YOLO model: {PPE_MODEL_PATH}")
    print("\n http://localhost:5003")
    print("="*70+"\n")
    try:
        app.run(host='0.0.0.0', port=5003, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n Stopped")
        if sys.active(): sys.stop()