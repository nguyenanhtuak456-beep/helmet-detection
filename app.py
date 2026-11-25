# -*- coding: utf-8 -*-
"""
AI TRAFFIC VISION - Intelligent Traffic Monitoring System
YOLOv8s Detection + ByteTrack Tracking (Dark Cyan Command Center Dashboard).
"""

import html as html_lib
import hashlib
import threading
import time
from datetime import datetime
import uuid
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
import subprocess
import torch

from core.history_db import (
    init_db,
    add_history,
    get_history,
    get_history_by_id,
    delete_history,
)
from collections import defaultdict, deque
from ultralytics import YOLO

try:
    import altair as alt
except Exception:
    alt = None

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    WEBRTC_OK = True
except Exception:
    WEBRTC_OK = False

st.set_page_config(
    page_title="AI Traffic Vision",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_db()

# =====================================================================
# HẰNG SỐ LOGIC & BẢNG MÀU BOUNDING BOX
# =====================================================================
from core.model_config import get_model_path

try:
    DEFAULT_MODEL_PATH = get_model_path()
except FileNotFoundError as e:
    DEFAULT_MODEL_PATH = None
    MODEL_PATH_ERROR = str(e)

# Ánh xạ ID lớp từ model.names {0: '0', 1: '1', 2: '2'}
CLASS_ID_MAP = {
    0: "helmet",
    1: "motorbike",
    2: "no-helmet",
}

LABEL_COLORS = {
    "helmet": (136, 148, 13),     # Teal #0D9488 (BGR)
    "motobike": (235, 99, 37),    # Royal Blue #2563EB (BGR)
    "motorbike": (235, 99, 37),
    "no-helmet": (68, 68, 239),   # Coral Red #EF4444 (BGR)
    "no_helmet": (68, 68, 239),
}

BBOX_EMA_ALPHA = 0.12


def normalize_detection_class(name, cls_id=None):
    """Chuẩn hoá cả nhãn dạng số ('0','1','2') lẫn nhãn chữ từ model.names."""
    raw = str(name).strip().lower().replace("_", " ").replace("-", " ")
    raw = " ".join(raw.split())

    if raw in ("0", "1", "2"):
        return CLASS_ID_MAP[int(raw)]

    if any(k in raw for k in ("no helmet", "nohelmet", "without helmet", "khong doi mu", "khong mu")):
        return "no-helmet"
    if any(k in raw for k in ("helmet", "doi mu")):
        return "helmet"
    if any(k in raw for k in ("motobike", "motorbike", "motorcycle", "bike", "xe may")):
        return "motorbike"

    if cls_id is not None and int(cls_id) in CLASS_ID_MAP:
        return CLASS_ID_MAP[int(cls_id)]
    return raw


DISPLAY_CLASS_NAMES = {
    "helmet": "Đội mũ",
    "no-helmet": "Không đội mũ",
    "motorbike": "Xe máy",
    "motobike": "Xe máy",
}

DISPLAY_OVERLAY_NAMES = {
    "helmet": "DOI MU",
    "no-helmet": "KHONG DOI MU",
    "motorbike": "XE MAY",
    "motobike": "XE MAY",
}


def display_class_name(name):
    return DISPLAY_CLASS_NAMES.get(normalize_detection_class(name), str(name))


def display_overlay_name(name):
    return DISPLAY_OVERLAY_NAMES.get(normalize_detection_class(name), str(name))


class TrackBBoxSmoother:
    """Làm mượt bbox sau ByteTrack cho TẤT CẢ lớp, gồm xe, helmet và no-helmet."""
    def __init__(self, alpha=BBOX_EMA_ALPHA):
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self.state = {}

    def update(self, track_id, xyxy):
        box = np.asarray(xyxy, dtype=np.float32)
        if track_id is None:
            return tuple(float(v) for v in box)

        x1, y1, x2, y2 = box
        current = np.array([
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            max(x2 - x1, 1.0),
            max(y2 - y1, 1.0),
        ], dtype=np.float32)

        previous = self.state.get(int(track_id))
        if previous is None:
            smooth = current
        else:
            px, py, pw, ph = previous
            cx, cy, cw, ch = current
            diag = max(float(np.hypot(pw, ph)), 1.0)
            center_jump = float(np.hypot(cx - px, cy - py)) / diag
            size_ratio = max(
                cw / max(pw, 1.0), pw / max(cw, 1.0),
                ch / max(ph, 1.0), ph / max(ch, 1.0),
            )
            if center_jump > 2.5 or size_ratio > 2.8:
                smooth = current
            else:
                motion = float(np.clip(center_jump / 1.5, 0.0, 1.0))
                alpha = self.alpha + (0.30 - self.alpha) * motion
                smooth = alpha * current + (1.0 - alpha) * previous

        self.state[int(track_id)] = smooth
        cx, cy, bw, bh = smooth
        return (
            float(cx - bw / 2.0), float(cy - bh / 2.0),
            float(cx + bw / 2.0), float(cy + bh / 2.0),
        )


class TrackLabelStabilizer:
    """Ổn định nhãn theo Track ID bằng khởi tạo ngay + hysteresis khi đổi lớp."""
    def __init__(self, window=9, switch_streak=4, conf_alpha=0.30):
        self.history = defaultdict(lambda: deque(maxlen=int(window)))
        self.stable = {}
        self.pending_class = {}
        self.pending_count = defaultdict(int)
        self.class_conf = defaultdict(dict)
        self.switch_streak = int(switch_streak)
        self.conf_alpha = float(conf_alpha)

    def update(self, track_id, raw_class, raw_conf, accept_vote=True):
        tid = int(track_id)
        raw_class = normalize_detection_class(raw_class)
        raw_conf = float(raw_conf)
        if not accept_vote:
            current = self.stable.get(tid)
            if current is None:
                return None, raw_conf
            return current, float(self.class_conf[tid].get(current, raw_conf))

        old_conf = self.class_conf[tid].get(raw_class, raw_conf)
        self.class_conf[tid][raw_class] = (
            self.conf_alpha * raw_conf + (1.0 - self.conf_alpha) * old_conf
        )
        hist = self.history[tid]
        hist.append(raw_class)

        current = self.stable.get(tid)
        if current is None:
            current = raw_class
            self.stable[tid] = current
        elif raw_class == current:
            self.pending_class.pop(tid, None)
            self.pending_count[tid] = 0
        else:
            if self.pending_class.get(tid) == raw_class:
                self.pending_count[tid] += 1
            else:
                self.pending_class[tid] = raw_class
                self.pending_count[tid] = 1
            if self.pending_count[tid] >= self.switch_streak:
                current = raw_class
                self.stable[tid] = current
                self.pending_class.pop(tid, None)
                self.pending_count[tid] = 0

        stable_conf = self.class_conf[tid].get(current, raw_conf)
        return current, float(stable_conf)


def suppress_duplicate_head_boxes(head_boxes, iou_threshold=0.55):
    """Loại bỏ khung trùng trên cùng một đầu (ví dụ vừa bắt mũ vừa bắt mặt)."""
    kept = []
    for item in sorted(head_boxes, key=lambda x: x[5], reverse=True):
        x1, y1, x2, y2 = item[:4]
        area = max(x2 - x1, 1.0) * max(y2 - y1, 1.0)
        duplicate = False
        for prev in kept:
            px1, py1, px2, py2 = prev[:4]
            ix1, iy1 = max(x1, px1), max(y1, py1)
            ix2, iy2 = min(x2, px2), min(y2, py2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            prev_area = max(px2 - px1, 1.0) * max(py2 - py1, 1.0)
            union = area + prev_area - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def suppress_duplicate_bike_boxes(bike_boxes, iou_threshold=0.60):
    """Loại bỏ khung xe máy vẽ chồng lên cùng một chiếc xe."""
    kept = []
    for item in sorted(bike_boxes, key=lambda x: x[5], reverse=True):
        x1, y1, x2, y2 = item[:4]
        area = max(x2 - x1, 1.0) * max(y2 - y1, 1.0)
        duplicate = False
        for prev in kept:
            px1, py1, px2, py2 = prev[:4]
            ix1, iy1 = max(x1, px1), max(y1, py1)
            ix2, iy2 = min(x2, px2), min(y2, py2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            prev_area = max(px2 - px1, 1.0) * max(py2 - py1, 1.0)
            union = area + prev_area - inter
            if (inter / union if union > 0 else 0.0) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def is_head_on_motorbike(head_box, bike_boxes):
    """Kiểm tra đầu/mũ có thực sự thuộc người ngồi trên một xe máy hay không."""
    if not bike_boxes:
        return False
    hx1, hy1, hx2, hy2 = head_box[:4]
    hw = max(hx2 - hx1, 1.0)
    hh = max(hy2 - hy1, 1.0)
    cx = (hx1 + hx2) / 2.0
    cy = (hy1 + hy2) / 2.0

    for bx1, by1, bx2, by2, _, _, _ in bike_boxes:
        bw = max(bx2 - bx1, 1.0)
        bh = max(by2 - by1, 1.0)
        in_x = (bx1 <= cx <= bx2)
        in_y = (by1 - 0.25 * bh <= cy <= by1 + 0.45 * bh)
        valid_scale = (0.11 * bw <= hw <= 0.55 * bw) and (0.10 * bh <= hh <= 0.42 * bh)
        if in_x and in_y and valid_scale:
            return True
    return False


NO_HELMET_MIN_VOTE_RATIO = 0.65
HEAD_MIN_VOTE_RATIO = 0.60


def classify_head_track(votes, min_track_frames, no_helmet_min_frames=3):
    """Ổn định phân loại: no-helmet cần tối thiểu 3 phiếu và tỷ lệ 65%."""
    if len(votes) < int(min_track_frames):
        return None
    weighted = []
    for v in votes:
        if isinstance(v, (tuple, list)) and len(v) >= 2:
            label, confidence = v[0], float(v[1])
        else:
            label, confidence = v, 1.0
        label = normalize_detection_class(label)
        if label in ("helmet", "no-helmet"):
            weighted.append((label, max(0.05, min(confidence, 1.0))))
    total = sum(w for _, w in weighted)
    if total <= 0:
        return None
    helmet_votes = sum(w for label, w in weighted if label == "helmet")
    no_helmet_votes = sum(w for label, w in weighted if label == "no-helmet")
    raw_nohelmet = sum(
        1 for v in votes
        if normalize_detection_class(v[0] if isinstance(v, (tuple, list)) else v) == "no-helmet"
    )
    if raw_nohelmet >= int(no_helmet_min_frames) and no_helmet_votes / total >= NO_HELMET_MIN_VOTE_RATIO:
        return "no-helmet"
    if helmet_votes / total >= HEAD_MIN_VOTE_RATIO:
        return "helmet"
    return None


class HeadObjectTracker:
    """Tracker riêng cho đầu/mũ, không phụ thuộc Track ID ByteTrack."""
    def __init__(self, max_age=12, bbox_alpha=BBOX_EMA_ALPHA, label_alpha=0.30):
        self.max_age = int(max_age)
        self.tracks = {}
        self.next_id = 1
        self.frame_number = 0
        self.bbox_smoother = TrackBBoxSmoother(bbox_alpha)
        self.label_stabilizer = TrackLabelStabilizer(conf_alpha=label_alpha)

    @staticmethod
    def _xyxy_to_state(box):
        x1, y1, x2, y2 = [float(v) for v in box]
        return np.array([
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            max(x2 - x1, 1.0),
            max(y2 - y1, 1.0),
        ], dtype=np.float32)

    @staticmethod
    def _state_to_xyxy(state):
        cx, cy, bw, bh = state
        return (float(cx - bw / 2), float(cy - bh / 2),
                float(cx + bw / 2), float(cy + bh / 2))

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        aa = max(1.0, ax2 - ax1) * max(1.0, ay2 - ay1)
        ba = max(1.0, bx2 - bx1) * max(1.0, by2 - by1)
        union = aa + ba - inter
        return inter / union if union > 0 else 0.0

    def _predicted_box(self, track, frame_number):
        dt = max(1, frame_number - track["last_frame"])
        state = track["state"].copy()
        state[0] += track["velocity"][0] * dt
        state[1] += track["velocity"][1] * dt
        return self._state_to_xyxy(state)

    def update(self, detections, conf_helmet, conf_no_helmet):
        self.frame_number += 1
        frame_number = self.frame_number
        active_ids = [
            tid for tid, track in self.tracks.items()
            if frame_number - track["last_frame"] <= self.max_age
        ]

        pair_costs = []
        for di, det in enumerate(detections):
            dbox = tuple(float(v) for v in det[:4])
            ds = self._xyxy_to_state(dbox)
            for tid in active_ids:
                track = self.tracks[tid]
                dt = max(1, frame_number - track["last_frame"])
                pbox = self._predicted_box(track, frame_number)
                ps = self._xyxy_to_state(pbox)
                dist = float(np.hypot(ds[0] - ps[0], ds[1] - ps[1]))
                diag = max(float(np.hypot(max(ds[2], ps[2]), max(ds[3], ps[3]))), 20.0)
                size_ratio = max(
                    ds[2] / max(ps[2], 1.0), ps[2] / max(ds[2], 1.0),
                    ds[3] / max(ps[3], 1.0), ps[3] / max(ds[3], 1.0),
                )
                speed = float(np.hypot(track["velocity"][0], track["velocity"][1]))
                gate = max(24.0, diag * 1.55 + min(speed * dt, 80.0) * 0.35)
                if size_ratio > 2.8 or dist > gate:
                    continue
                iou = self._iou(pbox, dbox)
                if iou < 0.01 and dist > diag * 0.85:
                    continue
                size_penalty = min(abs(float(np.log(max(size_ratio, 1e-6)))), 1.0)
                cost = 0.60 * (1.0 - iou) + 0.32 * min(dist / gate, 1.0) + 0.08 * size_penalty
                if cost > 0.78:
                    continue
                pair_costs.append((cost, tid, di, ds))

        pair_costs.sort(key=lambda x: x[0])
        assigned_tracks, assigned_dets = set(), set()
        assignments = {}
        for cost, tid, di, ds in pair_costs:
            if tid in assigned_tracks or di in assigned_dets:
                continue
            old = self.tracks[tid]
            dt = max(1, frame_number - old["last_frame"])
            measured_v = (ds[:2] - old["state"][:2]) / float(dt)
            velocity = 0.65 * old["velocity"] + 0.35 * measured_v
            old["state"] = ds
            old["velocity"] = velocity.astype(np.float32)
            old["last_frame"] = frame_number
            assignments[di] = tid
            assigned_tracks.add(tid)
            assigned_dets.add(di)

        for di, det in enumerate(detections):
            if di in assignments:
                continue
            tid = self.next_id
            self.next_id += 1
            state = self._xyxy_to_state(det[:4])
            self.tracks[tid] = {
                "state": state,
                "velocity": np.zeros(2, dtype=np.float32),
                "last_frame": frame_number,
            }
            assignments[di] = tid

        stale_ids = [tid for tid, tr in self.tracks.items()
                     if frame_number - tr["last_frame"] > self.max_age]
        for tid in stale_ids:
            self.tracks.pop(tid, None)
            self.bbox_smoother.state.pop(tid, None)
            self.label_stabilizer.history.pop(tid, None)
            self.label_stabilizer.stable.pop(tid, None)
            self.label_stabilizer.pending_class.pop(tid, None)
            self.label_stabilizer.pending_count.pop(tid, None)
            self.label_stabilizer.class_conf.pop(tid, None)

        output = []
        for di, det in enumerate(detections):
            x1, y1, x2, y2, raw_cls, raw_conf = det
            tid = assignments[di]
            raw_cls = normalize_detection_class(raw_cls)
            raw_conf = float(raw_conf)
            threshold = conf_helmet if raw_cls == "helmet" else conf_no_helmet
            vote_valid = raw_conf >= threshold
            stable_cls, stable_conf = self.label_stabilizer.update(
                tid, raw_cls, raw_conf, accept_vote=vote_valid
            )
            smoothed = self.bbox_smoother.update(tid, (x1, y1, x2, y2))
            if stable_cls is None:
                continue
            stable_threshold = conf_helmet if stable_cls == "helmet" else conf_no_helmet
            if stable_conf < stable_threshold:
                continue
            output.append((*smoothed, stable_cls, stable_conf, tid,
                           raw_cls if vote_valid else None))
        return output


# =====================================================================
# BẢNG MÀU GIAO DIỆN DARK CYAN
# =====================================================================
C_CYAN = "#22d3ee"
C_GREEN = "#34d399"
C_RED = "#fb7185"
C_PURPLE = "#60a5fa"
C_TEXT = "#e5f3ff"
C_MUTED = "#8ba3b8"
C_GRID = "rgba(148, 183, 207, 0.14)"


# =====================================================================
# HÀM LOGIC XỬ LÝ VIDEO & TRACKING
# =====================================================================
from core.model_loader import load_yolo_model as load_yolo_model_uncached


@st.cache_resource
def _load_yolo_model_cached(model_path, mtime):
    return load_yolo_model_uncached(model_path)


def load_yolo_model(model_path):
    if not model_path:
        raise ValueError("Đường dẫn model chưa được cấu hình.")

    mtime = os.path.getmtime(model_path)
    return _load_yolo_model_cached(model_path, mtime)


def get_model_classes_text(model_path):
    try:
        m = load_yolo_model(model_path)
        names = getattr(m, "names", {})

        if isinstance(names, dict):
            return ", ".join(
                f"{k}:{normalize_detection_class(v, k)}"
                for k, v in names.items()
            )

        if isinstance(names, (list, tuple)):
            return ", ".join(
                f"{i}:{normalize_detection_class(v, i)}"
                for i, v in enumerate(names)
            )

        return f"Định dạng tên lớp không hợp lệ: {type(names).__name__}"

    except Exception as e:
        return f"Lỗi đọc model: {type(e).__name__}: {e}"


from core.tracker_config import get_custom_tracker_yaml


from core.video_utils import convert_to_browser_mp4


def run_detection(
    input_video_path,
    model_path,
    imgsz_option,
    conf_helmet,
    conf_no_helmet,
    conf_bike,
    ema_alpha,
    min_track_frames,
    filter_on_bike,
    show_track_id,
    show_helmet=True,
    show_no_helmet=True,
    show_motorbike=True,
    progress_state=None,
    pause_event=None,
    model_override=None,
    only_violation_logs=False,
):
    """Pipeline YOLOv8 + ByteTrack; hỗ trợ cập nhật tiến trình và tạm dừng an toàn giữa các frame."""
    model = model_override if model_override is not None else load_yolo_model(model_path)
    if hasattr(model, "predictor") and model.predictor is not None:
        model.predictor = None

    custom_tracker_path = get_custom_tracker_yaml()
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Không thể mở video đầu vào. Hãy kiểm tra định dạng hoặc tệp video.")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_dim = 1280
    scale = min(max_dim / max(orig_w, orig_h), 1.0)
    out_w = int(orig_w * scale) // 2 * 2
    out_h = int(orig_h * scale) // 2 * 2

    frame_step = 2 if orig_fps > 35 else 1
    out_fps = orig_fps / frame_step

    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_video_path = output_temp.name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, out_fps, (out_w, out_h))

    progress_bar = st.progress(0) if progress_state is None else None
    status_text = st.empty() if progress_state is None else None

    bike_track_frames = defaultdict(int)
    head_track_votes = defaultdict(list)

    conf_list = []
    sec_violator_ids = defaultdict(set)
    sec_helmet_ids = defaultdict(set)
    sec_untracked_max = defaultdict(int)
    recent_events = []

    def remember_crop(image, box, label, confidence, track_id, second):
        try:
            if only_violation_logs and normalize_detection_class(label) != "no-helmet":
                return
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, x2 = max(0, min(x1, w - 1)), max(1, min(x2, w))
            y1, y2 = max(0, min(y1, h - 1)), max(1, min(y2, h))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                return
            crop = cv2.resize(crop, (240, 150), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            if ok:
                recent_events.append({
                    "label": str(label), "confidence": float(confidence),
                    "track_id": track_id, "second": int(second), "image": encoded.tobytes()
                })
                del recent_events[:-40]
        except Exception:
            return

    max_sec = 0
    frame_idx = 0

    label_stabilizer = TrackLabelStabilizer(conf_alpha=ema_alpha)
    bbox_smoother = TrackBBoxSmoother(BBOX_EMA_ALPHA)
    head_tracker = HeadObjectTracker(max_age=12, bbox_alpha=BBOX_EMA_ALPHA, label_alpha=ema_alpha)
    use_tracking = True

    min_conf = min(conf_helmet, conf_no_helmet, conf_bike)

    while cap.isOpened():
        if pause_event is not None:
            while pause_event.is_set():
                if progress_state is not None:
                    progress_state["status"] = "paused"
                time.sleep(0.12)
            if progress_state is not None:
                progress_state["status"] = "running"

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        if use_tracking:
            try:
                results = model.track(
                    frame,
                    imgsz=imgsz_option,
                    conf=min_conf,
                    iou=0.45,
                    agnostic_nms=False,
                    tracker=custom_tracker_path,
                    persist=True,
                    verbose=False
                )[0]
            except Exception:
                use_tracking = False
                results = model(
                    frame,
                    imgsz=imgsz_option,
                    conf=min_conf,
                    iou=0.45,
                    agnostic_nms=False,
                    verbose=False
                )[0]
        else:
            results = model(
                frame,
                imgsz=imgsz_option,
                conf=min_conf,
                iou=0.45,
                agnostic_nms=False,
                verbose=False
            )[0]

        frame_vis = cv2.resize(frame, (out_w, out_h))
        scale_x = out_w / orig_w
        scale_y = out_h / orig_h
        current_sec = int(frame_idx // orig_fps)
        max_sec = max(max_sec, current_sec)
        untracked_no_helmet = 0

        bike_boxes = []
        raw_head_detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            raw_conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            raw_cls = normalize_detection_class(model.names[cls_id], cls_id=cls_id)

            if raw_cls in ("helmet", "no-helmet"):
                if raw_cls == "no-helmet" and min(x2 - x1, y2 - y1) < max(12.0, orig_w * 0.010):
                    continue
                raw_head_detections.append((x1, y1, x2, y2, raw_cls, raw_conf))
                continue

            if box.id is not None:
                tid = int(box.id[0].item())
                stable_cls, smooth_conf = label_stabilizer.update(tid, raw_cls, raw_conf)
                x1, y1, x2, y2 = bbox_smoother.update(tid, (x1, y1, x2, y2))
                if stable_cls is None:
                    continue
            else:
                tid, stable_cls, smooth_conf = None, raw_cls, raw_conf

            if stable_cls in ("motobike", "motorbike") and smooth_conf >= conf_bike:
                bike_boxes.append((x1, y1, x2, y2, "motorbike", smooth_conf, tid))

        bike_boxes = suppress_duplicate_bike_boxes(bike_boxes, iou_threshold=0.60)
        raw_head_detections = suppress_duplicate_head_boxes(raw_head_detections, iou_threshold=0.55)
        head_boxes = head_tracker.update(raw_head_detections, conf_helmet, conf_no_helmet)

        for bx1, by1, bx2, by2, label_name, conf, tid in bike_boxes:
            if tid is not None:
                bike_track_frames[tid] += 1
            conf_list.append(conf)
            if show_motorbike:
                remember_crop(frame, (bx1, by1, bx2, by2), "motorbike", conf, tid, current_sec)
            if not show_motorbike:
                continue

            x1, y1 = int(bx1 * scale_x), int(by1 * scale_y)
            x2, y2 = int(bx2 * scale_x), int(by2 * scale_y)
            color = LABEL_COLORS.get(label_name, (235, 99, 37))

            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
            id_prefix = f"#{tid} " if (show_track_id and tid is not None) else ""
            text = f"{id_prefix}{display_overlay_name(label_name)} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            y_bot = min(y2 + th + 4, out_h - 2)
            cv2.rectangle(frame_vis, (x1, y_bot - th - 4), (x1 + tw, y_bot + 2), color, -1)
            cv2.putText(
                frame_vis, text, (x1, y_bot - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )

        filtered_heads = suppress_duplicate_head_boxes(head_boxes, iou_threshold=0.55)

        for hx1, hy1, hx2, hy2, label_name, conf, tid, raw_label_name in filtered_heads:
            if filter_on_bike and not is_head_on_motorbike((hx1, hy1, hx2, hy2), bike_boxes):
                continue

            conf_list.append(conf)
            if tid is not None:
                if raw_label_name is not None:
                    head_track_votes[tid].append((raw_label_name, float(conf)))
                    if raw_label_name == "no-helmet":
                        sec_violator_ids[current_sec].add(tid)
                    elif raw_label_name == "helmet":
                        sec_helmet_ids[current_sec].add(tid)
            elif label_name == "no-helmet" and raw_label_name == "no-helmet":
                untracked_no_helmet += 1

            class_visible = (
                (label_name == "helmet" and show_helmet)
                or (label_name == "no-helmet" and show_no_helmet)
            )
            if class_visible:
                remember_crop(frame, (hx1, hy1, hx2, hy2), label_name, conf, tid, current_sec)
            if not class_visible:
                continue

            x1, y1 = int(hx1 * scale_x), int(hy1 * scale_y)
            x2, y2 = int(hx2 * scale_x), int(hy2 * scale_y)
            color = LABEL_COLORS.get(label_name, (255, 255, 255))

            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
            id_prefix = f"H{tid} " if (show_track_id and tid is not None) else ""
            text = f"{id_prefix}{display_overlay_name(label_name)} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
            y_text = max(y1 - 4, th + 4)
            cv2.rectangle(frame_vis, (x1, y_text - th - 4), (x1 + tw, y_text + 2), color, -1)
            cv2.putText(
                frame_vis, text, (x1, y_text - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2
            )

        sec_untracked_max[current_sec] = max(
            sec_untracked_max[current_sec], untracked_no_helmet
        )

        out.write(frame_vis)
        frame_idx += 1
        if total_frames > 0:
            fraction = min(frame_idx / total_frames, 1.0)
            if progress_state is not None:
                progress_state["fraction"] = fraction
                progress_state["frame"] = min(frame_idx, total_frames)
                progress_state["total"] = total_frames
                progress_state["status"] = "running"
            else:
                progress_bar.progress(fraction)
                percent = int(round(fraction * 100))
                status_text.text(f"Đang xử lý: {percent}% ({min(frame_idx, total_frames)}/{total_frames} frames)")

    cap.release()
    out.release()

    unique_motorbike = sum(
        1 for frames in bike_track_frames.values() if frames >= 2
    )
    unique_helmet = 0
    valid_violator_ids = set()
    uncertain_head_tracks = 0
    for tid, votes in head_track_votes.items():
        final_cls = classify_head_track(votes, 2, no_helmet_min_frames=3)
        if final_cls == "helmet":
            unique_helmet += 1
        elif final_cls == "no-helmet":
            valid_violator_ids.add(tid)
        elif len(votes) >= 2:
            uncertain_head_tracks += 1

    unique_no_helmet = len(valid_violator_ids)

    valid_helmet_ids = {
        tid for tid, votes in head_track_votes.items()
        if classify_head_track(votes, 2, no_helmet_min_frames=3) == "helmet"
    }
    helmet_per_sec = {}
    violations_per_sec = {}
    for sec in range(0, max_sec + 1):
        helmet_per_sec[sec] = len(sec_helmet_ids[sec] & valid_helmet_ids)
        valid_in_sec = len(sec_violator_ids[sec] & valid_violator_ids)
        violations_per_sec[sec] = max(valid_in_sec, sec_untracked_max[sec])

    final_output_path = output_video_path.replace(".mp4", "_h264.mp4")
    is_h264_ok = convert_to_browser_mp4(output_video_path, final_output_path)
    processed_path = final_output_path if is_h264_ok else output_video_path

    saved_drive_path = None
    drive_save_dir = "/content/drive/MyDrive/ML/ketqua"
    if os.path.exists("/content/drive/MyDrive/ML"):
        os.makedirs(drive_save_dir, exist_ok=True)
        drive_video_path = os.path.join(drive_save_dir, "helmet_detected_output.mp4")
        shutil.copy2(processed_path, drive_video_path)
        saved_drive_path = "Drive của tôi > ML > ketqua > helmet_detected_output.mp4"

    stats_result = {
        "motorbike": unique_motorbike,
        "helmet": unique_helmet,
        "no_helmet": unique_no_helmet,
        "avg_conf": (sum(conf_list) / len(conf_list) * 100) if conf_list else 0.0,
        "timeline": violations_per_sec,
        "helmet_timeline": helmet_per_sec,
        "head_track_ids": len(head_track_votes),
        "uncertain_head_tracks": uncertain_head_tracks,
        "no_helmet_vote_ratio": NO_HELMET_MIN_VOTE_RATIO,
        "bbox_alpha": BBOX_EMA_ALPHA,
    }
    result_payload = {
        "processed_video_path": processed_path,
        "stats": stats_result,
        "recent_detections": recent_events[-40:],
        "saved_drive_path": saved_drive_path,
    }

    if progress_state is not None:
        progress_state["fraction"] = 1.0
        progress_state["frame"] = total_frames or frame_idx
        progress_state["total"] = total_frames or frame_idx
        progress_state["status"] = "done"
        return result_payload

    st.session_state.processed_video_path = processed_path
    st.session_state.saved_drive_path = saved_drive_path
    st.session_state.stats = stats_result
    st.session_state.recent_detections = recent_events[-40:]
    progress_bar.progress(1.0)
    status_text.success("Xử lý video hoàn tất")
    return result_payload


# =====================================================================
# UI HELPERS - DARK CYAN COMMAND CENTER CSS & THÀNH PHẦN GIAO DIỆN
# =====================================================================
APP_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #f1f5f9;
  --sidebar: #ffffff;
  --panel: #ffffff;
  --panel-sub: #f8fafc;
  --line: #e2e8f0;
  --line-strong: #cbd5e1;
  --text: #0f172a;
  --text-sec: #334155;
  --muted: #64748b;
  --blue: #2563eb;
  --blue-soft: #eff6ff;
  --teal: #0d9488;
  --teal-soft: #f0fdfa;
  --red: #ef4444;
  --red-soft: #fef2f2;
  --green: #10b981;
  --green-soft: #ecfdf5;
  --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp, [class*="css"] {
  font-family: var(--font) !important;
  font-size: 14.5px !important;
  line-height: 1.5 !important;
  color: var(--text) !important;
  -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg) !important; }
header[data-testid="stHeader"] { display: none !important; height: 0 !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {
  display: none !important;
  visibility: hidden !important;
}
.block-container {
  width: 100% !important;
  max-width: none !important;
  padding: 1rem clamp(1rem, 1.8vw, 1.8rem) 1.6rem !important;
  margin-top: 0 !important;
}
[data-testid="stColumn"] { min-width: 0 !important; box-sizing: border-box !important; }

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
  width: 274px !important;
  min-width: 274px !important;
  max-width: 274px !important;
  background: var(--sidebar) !important;
  border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.15rem 0.95rem 1.35rem !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: 0.2rem !important; }

.sb-brand-text {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 55px;
}

.sb-brand-divider {
    border-bottom: 1px solid var(--line);
    margin: 14px 0;
}

.sb-brand-text b {
    display: block;
    font-size: 16.5px !important;
    line-height: 1.2;
    font-weight: 800;
    letter-spacing: -0.35px;
    color: #0f172a !important;
    white-space: nowrap;
}

.sb-brand-text small {
    display: block;
    color: #64748b !important;
    font-size: 11px !important;
    margin-top: 3px;
    white-space: nowrap;
}
.sb-section {
  font-size: 10.5px !important;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color: #94a3b8 !important;
  font-weight: 700;
  margin: 18px 8px 8px;
}

[data-testid="stSidebar"] .stButton > button {
  width: 100%;
  height: 44px !important;
  min-height: 44px !important;
  justify-content: flex-start !important;
  text-align: left !important;
  border: 1px solid transparent !important;
  border-radius: 10px !important;
  background: transparent !important;
  color: #475569 !important;
  padding: 0 12px !important;
  font-size: 13.5px !important;
  font-weight: 600 !important;
  box-shadow: none !important;
  transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: #f1f5f9 !important;
  border-color: #e2e8f0 !important;
  color: #0f172a !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(90deg, #eff6ff 0%, #f0f9ff 100%) !important;
  border-color: #bfdbfe !important;
  color: #1d4ed8 !important;
  font-weight: 700 !important;
  box-shadow: inset 3.5px 0 0 #2563eb !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] p {
  color: #1d4ed8 !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
  border: 1px solid var(--line) !important;
  border-radius: 11px !important;
  background: #f8fafc !important;
  overflow: hidden !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  padding: 0.6rem 0.75rem !important;
  color: #334155 !important;
  font-size: 13.5px !important;
  font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg { flex: 0 0 1rem !important; color: #64748b !important; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #334155 !important; font-size: 12.5px !important; }

.sb-system {
  margin-top: 22px;
  padding: 13px 14px;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%);
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}
.sb-system-head {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.9px;
  text-transform: uppercase;
  color: #64748b;
  margin-bottom: 5px;
}
.sb-system-status {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 12.5px;
  font-weight: 750;
  color: #059669;
  margin-bottom: 6px;
}
.sb-system .sys-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #10b981;
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.18);
}
.sb-system-meta { font-size: 11px; line-height: 1.5; color: #475569; word-break: break-word; }

.atv-breadcrumb {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 9px 15px;
  margin-bottom: 14px;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
}
.atv-bc-left { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #64748b; }
.atv-bc-left b { color: #0f172a; font-weight: 700; }
.atv-bc-right { display: flex; align-items: center; gap: 8px; font-size: 11.5px; color: #475569; }
.atv-bc-tag {
  background: #f1f5f9;
  border: 1px solid #e2e8f0;
  border-radius: 7px;
  padding: 3px 9px;
  font-weight: 600;
  color: #334155;
}

.stButton > button, .stDownloadButton > button {
  border-radius: 10px !important;
  border: 1px solid #cbd5e1 !important;
  background: #ffffff !important;
  color: #1e293b !important;
  min-height: 42px !important;
  font-weight: 650 !important;
  font-size: 13.5px !important;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
  transition: all 0.15s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: #f8fafc !important;
  border-color: #94a3b8 !important;
  color: #0f172a !important;
  transform: translateY(-1px);
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
  border: 1px solid #1d4ed8 !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.24) !important;
}
.stButton > button[kind="primary"] p,
.stButton > button[data-testid="stBaseButton-primary"] p { color: #ffffff !important; }
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
  background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
  box-shadow: 0 6px 16px rgba(37, 99, 235, 0.32) !important;
}
.stButton > button:disabled {
  background: #f1f5f9 !important;
  border-color: #e2e8f0 !important;
  color: #94a3b8 !important;
  box-shadow: none !important;
  transform: none !important;
}

[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
[data-testid="stCheckbox"] label, [data-testid="stToggle"] label, label {
  color: #1e293b !important;
  font-size: 13.5px !important;
  font-weight: 550 !important;
}
[data-testid="stCaptionContainer"], .stCaption { color: #64748b !important; font-size: 12px !important; }
input, textarea { color: #0f172a !important; background: #ffffff !important; }
div[data-baseweb="input"], div[data-baseweb="select"] > div {
  background: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 9px !important;
}
div[data-baseweb="select"] * { color: #0f172a !important; }
[data-testid="stSlider"] [role="slider"] {
  background: #2563eb !important;
  border: 2px solid #ffffff !important;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2) !important;
}
[data-testid="stProgress"] > div > div > div { background: linear-gradient(90deg, #2563eb, #0d9488) !important; }

div[class*="st-key-main_video_panel"],
div[class*="st-key-control_panel"],
div[class*="st-key-timeline_panel"],
div[class*="st-key-recent_panel"],
div[class*="st-key-panel_"],
div[class*="st-key-ov_"],
div[class*="st-key-history_"] {
  box-sizing: border-box !important;
  width: 100% !important;
  max-width: 100% !important;
  background: #ffffff !important;
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  padding: 16px 18px !important;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04) !important;
  min-height: 100%;
}
.card-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin: 0 0 12px;
  color: #0f172a !important;
  font-size: 15.5px !important;
  font-weight: 750 !important;
  letter-spacing: -0.2px;
}
.sec { display: flex; align-items: center; gap: 8px; margin: 14px 0 10px; }
.sec h2, .pn-head h3 { font-size: 16px !important; font-weight: 750 !important; color: #0f172a !important; margin: 0; }
.pn-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 10px; }
.pn-head small { font-size: 11.5px; color: #64748b; }

.live-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  padding: 5px 11px;
  font-size: 10.5px;
  font-weight: 750;
  letter-spacing: 0.45px;
  white-space: nowrap;
}
.live-pill i { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.live-pill.wait { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
.live-pill.wait i { background: #94a3b8; }
.live-pill.ready { background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }
.live-pill.ready i { background: #0284c7; }
.live-pill.active { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
.live-pill.active i { background: #ef4444; box-shadow: 0 0 8px rgba(239, 68, 68, 0.7); animation: atvblink 1s infinite; }
.live-pill.paused { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
.live-pill.paused i { background: #f59e0b; }
.live-pill.done { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
.live-pill.done i { background: #10b981; }
@keyframes atvblink { 50% { opacity: 0.35; } }

.legend { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 0 0 12px; }
.legend span, .chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid #e2e8f0;
  background: #f8fafc;
  padding: 5px 11px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  color: #334155;
}
.legend i { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.chip.ok { color: #047857; border-color: #a7f3d0; background: #ecfdf5; }
.chip.warn { color: #b45309; border-color: #fde68a; background: #fef3c7; }
.chip.bad { color: #dc2626; border-color: #fecaca; background: #fef2f2; }
.chip.tech { color: #0369a1; border-color: #bae6fd; background: #e0f2fe; }

.diag-note {
  margin-top: 8px;
  padding: 8px 12px;
  border-radius: 9px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  color: #475569;
  font-size: 12px;
}

.hud-box {
  position: relative;
  border: 1px dashed #cbd5e1;
  border-radius: 14px;
  background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
  min-height: 270px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: #64748b;
  padding: 24px;
  font-size: 13.5px;
}
.hud-box b { display: block; color: #0f172a; font-size: 17px; font-weight: 750; margin-bottom: 6px; }
.hud-corner { position: absolute; width: 22px; height: 22px; border-color: #94a3b8; border-style: solid; pointer-events: none; }
.hud-corner.tl { top: 12px; left: 12px; border-width: 2.5px 0 0 2.5px; border-top-left-radius: 5px; }
.hud-corner.tr { top: 12px; right: 12px; border-width: 2.5px 2.5px 0 0; border-top-right-radius: 5px; }
.hud-corner.bl { bottom: 12px; left: 12px; border-width: 0 0 2.5px 2.5px; border-bottom-left-radius: 5px; }
.hud-corner.br { bottom: 12px; right: 12px; border-width: 0 2.5px 2.5px 0; border-bottom-right-radius: 5px; }

.empty {
  border: 1px dashed #cbd5e1;
  border-radius: 12px;
  background: #f8fafc;
  min-height: 210px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: #64748b;
  padding: 20px;
  font-size: 13.5px;
}
.empty b { display: block; color: #0f172a; font-size: 16px; font-weight: 750; margin-bottom: 6px; }

[data-testid="stFileUploader"] {
  background: #f8fafc !important;
  border: 1.5px dashed #cbd5e1 !important;
  border-radius: 12px !important;
  padding: 6px 10px !important;
  margin: 2px 0 12px !important;
}
[data-testid="stFileUploaderDropzone"] { background: transparent !important; border: 0 !important; min-height: 54px !important; }
[data-testid="stFileUploaderDropzone"] * { color: #475569 !important; font-size: 12.5px !important; }
[data-testid="stFileUploader"] button, [data-testid="stFileUploader"] button p {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  background: #2563eb !important;
  border: 1px solid #1d4ed8 !important;
  font-weight: 700 !important;
  border-radius: 8px !important;
  opacity: 1 !important;
}
[data-testid="stFileUploader"] button:hover { background: #1d4ed8 !important; }
[data-testid="stVideo"] {
  border: 1px solid #cbd5e1 !important;
  border-radius: 13px !important;
  overflow: hidden !important;
  background: #0f172a !important;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08) !important;
}
video { border-radius: 11px !important; }

.source-file {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  padding: 9px 13px;
  border: 1px solid #e2e8f0;
  background: #f8fafc;
  border-radius: 10px;
  color: #0f172a;
  font-size: 12.5px;
  font-weight: 650;
}
.source-file span { color: #64748b; font-weight: 500; white-space: nowrap; }
.progress-caption { font-size: 12.5px; color: #334155; font-weight: 650; margin-top: 6px; }

.control-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px !important;
  font-weight: 650 !important;
  color: #1e293b !important;
  margin: 8px 0 4px !important;
}
.val-pill {
  background: #f1f5f9;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 1px 8px;
  font-size: 12px;
  font-weight: 750;
}
.control-divider { height: 1px; background: #e2e8f0; margin: 12px 0; }
.stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 9px 11px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 9px;
  color: #475569;
  font-size: 12.5px;
  margin-bottom: 10px;
}
.stat-row .v { color: #0f172a; font-weight: 750; }

.kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 14px 0 18px; }
.smart-kpi {
  position: relative;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-left: 4px solid var(--accent);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
  transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
}
.smart-kpi:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
  border-color: var(--accent);
}
.smart-kpi.danger {
  background: linear-gradient(145deg, #ffffff 0%, #fef2f2 100%);
  border-color: #fecaca;
  border-left: 4px solid #ef4444;
}
.sk-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.sk-title { font-size: 11px; font-weight: 750; letter-spacing: 0.55px; text-transform: uppercase; color: #64748b; }
.sk-ico {
  width: 34px;
  height: 34px;
  border-radius: 9px;
  background: var(--tint);
  color: var(--accent);
  display: grid;
  place-items: center;
  font-size: 17px;
  font-weight: 700;
}
.sk-val { font-size: 28px; line-height: 1.1; font-weight: 800; letter-spacing: -0.7px; color: #0f172a; }
.sk-val small { font-size: 13px; font-weight: 650; color: #64748b; margin-left: 4px; }
.sk-sub { margin-top: 6px; font-size: 11.5px; font-weight: 600; color: var(--sub-color, #64748b); }

.chart-legend { display: flex; gap: 14px; font-size: 12px; font-weight: 600; color: #475569; }
.chart-legend i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }
.donut-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 8px 6px 4px; text-align: center; }
.donut-cap { font-size: 11.5px; font-weight: 650; color: #475569; margin-top: -6px; }
.donut-pct { font-size: 18px; font-weight: 800; color: #0d9488; }

.event-tag {
  display: inline-block;
  margin-top: 6px;
  padding: 3px 8px;
  border-radius: 6px;
  font-size: 10.5px;
  font-weight: 700;
  color: #ffffff;
}
.event-meta { font-size: 12px; color: #0f172a; font-weight: 700; margin-top: 4px; }
.event-time { font-size: 11px; color: #64748b; margin-top: 1px; }

.ov-hero {
  background: linear-gradient(135deg, #ffffff 0%, #eff6ff 60%, #f0fdfa 100%);
  border: 1px solid #dbeafe;
  border-radius: 16px;
  padding: 20px 22px;
  margin-bottom: 14px;
  box-shadow: 0 2px 10px rgba(37, 99, 235, 0.05);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
}
.ov-hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: #dbeafe;
  color: #1d4ed8;
  font-size: 11px;
  font-weight: 750;
  letter-spacing: 0.6px;
  text-transform: uppercase;
  padding: 4px 10px;
  border-radius: 999px;
  margin-bottom: 8px;
}
.ov-hero h1 { font-size: 21px !important; font-weight: 800 !important; color: #0f172a !important; margin: 0 0 6px !important; letter-spacing: -0.4px; }
.ov-hero p { font-size: 13.5px !important; color: #475569 !important; margin: 0 !important; max-width: 760px; }
.ov-hero-stats { display: flex; gap: 10px; flex-wrap: wrap; }
.ov-pill { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 11px; padding: 9px 13px; min-width: 120px; }
.ov-pill small { display: block; font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.4px; }
.ov-pill b { display: block; font-size: 14px; font-weight: 800; color: #0f172a; margin-top: 2px; }

.ov-mode-head { display: flex; align-items: center; gap: 11px; margin-bottom: 8px; }
.ov-mode-ico { width: 38px; height: 38px; border-radius: 10px; display: grid; place-items: center; font-size: 18px; font-weight: 700; flex-shrink: 0; }
.ov-mode-title { font-size: 15px; font-weight: 750; color: #0f172a; line-height: 1.25; }
.ov-mode-sub { font-size: 11.5px; font-weight: 600; color: #64748b; }
.ov-mode-desc { font-size: 13px; color: #475569; margin: 6px 0 12px; min-height: 58px; }

.ov-step {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  padding: 10px 12px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 11px;
  margin-bottom: 9px;
}
.ov-step-num {
  width: 24px;
  height: 24px;
  border-radius: 7px;
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  color: #1d4ed8;
  font-size: 12px;
  font-weight: 800;
  display: grid;
  place-items: center;
  flex-shrink: 0;
  margin-top: 1px;
}
.ov-step-body b { display: block; font-size: 13px; font-weight: 750; color: #0f172a; margin-bottom: 2px; }
.ov-step-body span { display: block; font-size: 12.2px; color: #475569; line-height: 1.45; }

.ov-spec-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  padding: 9px 12px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  margin-bottom: 8px;
  font-size: 12.5px;
}
.ov-spec-row span { color: #475569; font-weight: 550; }
.ov-spec-row b { color: #0f172a; font-weight: 750; text-align: right; }

.kpi {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-top: 3px solid var(--accent);
  border-radius: 13px;
  padding: 13px 14px;
  color: #0f172a;
  box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
}
.kpi .k-top { display: flex; justify-content: space-between; align-items: flex-start; }
.kpi .k-label { font-size: 11.5px; font-weight: 750; color: #475569; }
.kpi .k-vi { font-size: 11px; color: #64748b; }
.kpi .k-value { font-size: 26px; font-weight: 800; color: var(--accent); margin: 4px 0 2px; }
.kpi .k-unit { font-size: 12.5px; font-weight: 600; color: #64748b; margin-left: 4px; }
.kpi .k-note { font-size: 11px; color: #64748b; }

div[class*="st-key-export_bar"] {
  box-sizing: border-box !important;
  width: 100% !important;
  position: sticky;
  bottom: 0;
  z-index: 20;
  background: rgba(255, 255, 255, 0.96) !important;
  backdrop-filter: blur(8px);
  border: 1px solid #e2e8f0 !important;
  border-radius: 13px !important;
  padding: 9px 14px !important;
  margin-top: 14px !important;
  box-shadow: 0 -4px 18px rgba(15, 23, 42, 0.06) !important;
}
.atv-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  padding: 12px 4px 0;
  margin-top: 10px;
  border-top: 1px solid #e2e8f0;
  color: #64748b;
  font-size: 11.5px;
}
.atv-footer b { color: #0f172a; }

[data-testid="stDataFrame"] { border: 1px solid #e2e8f0 !important; border-radius: 11px !important; overflow: hidden; }
[data-testid="stExpander"] { border: 1px solid #e2e8f0 !important; border-radius: 12px !important; background: #ffffff !important; }
[data-testid="stExpander"] summary { color: #0f172a !important; font-size: 13px !important; font-weight: 650 !important; }

@media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 760px) {
  [data-testid="stSidebar"] { min-width: 245px !important; max-width: 245px !important; }
  .block-container { padding: 0.75rem !important; }
  .kpi-grid { grid-template-columns: 1fr; }
  .atv-footer { flex-direction: column; align-items: flex-start; }
}


/* ================================================================
   DARK CYAN OVERRIDES — presentation layer only; no inference logic
   ================================================================ */
:root {
  color-scheme: dark;
  --bg: #080f1b;
  --sidebar: #0b1524;
  --panel: #101d2d;
  --panel-sub: #142438;
  --line: #20364b;
  --line-strong: #31516b;
  --text: #e6f4ff;
  --text-sec: #bfd2e2;
  --muted: #8ba3b8;
  --blue: #38bdf8;
  --blue-soft: rgba(56,189,248,.12);
  --teal: #22d3ee;
  --teal-soft: rgba(34,211,238,.12);
  --red: #fb7185;
  --red-soft: rgba(251,113,133,.12);
  --green: #34d399;
  --green-soft: rgba(52,211,153,.12);
}
html, body, .stApp, [class*="css"], [data-testid="stAppViewContainer"] {
  color: var(--text) !important;
  background: var(--bg) !important;
}
.stApp {
  background:
    radial-gradient(ellipse at 78% 0%, rgba(8,145,178,.13), transparent 34%),
    radial-gradient(ellipse at 0% 70%, rgba(37,99,235,.08), transparent 32%),
    var(--bg) !important;
}
[data-testid="stAppViewContainer"] > .main { background: transparent !important; }
.block-container { color: var(--text) !important; }

/* Sidebar / brand */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
  background: linear-gradient(180deg, #0b1727 0%, #0a1422 100%) !important;
  border-right: 1px solid #1d354b !important;
}
[data-testid="stSidebar"] * { color: var(--text-sec); }
.sb-brand { border-bottom-color: #20384e !important; }
.sb-brand b { color: #effbff !important; }
.sb-brand small { color: #8ba3b8 !important; }
.sb-light { background: #101f31 !important; border-color: #2b4c64 !important; box-shadow: 0 0 22px rgba(34,211,238,.12) !important; }
.sb-section { color: #6f91aa !important; }
[data-testid="stSidebar"] .stButton > button {
  color: #b6cadb !important; border-color: transparent !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(34,211,238,.08) !important; border-color: rgba(34,211,238,.20) !important; color: #e8fbff !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(100deg, rgba(8,145,178,.25), rgba(37,99,235,.13)) !important;
  border: 1px solid rgba(34,211,238,.38) !important;
  color: #67e8f9 !important;
  box-shadow: inset 3px 0 0 #22d3ee, 0 0 18px rgba(34,211,238,.07) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] p { color: #67e8f9 !important; }
[data-testid="stSidebar"] [data-testid="stExpander"],
[data-testid="stExpander"] {
  background: #101e2e !important; border: 1px solid #284157 !important; border-radius: 12px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stExpander"] summary { color: #d9eaf6 !important; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #b4c8d8 !important; }
.sb-system {
  background: linear-gradient(145deg, #102438, #0e1b2b) !important;
  border-color: #25465d !important; box-shadow: inset 0 1px rgba(255,255,255,.025) !important;
}
.sb-system-head { color: #7f9db4 !important; }
.sb-system-status { color: #5eead4 !important; }
.sb-system-meta { color: #a4bacb !important; }

/* Breadcrumb and surfaces */
.atv-breadcrumb {
  background: rgba(16,29,45,.88) !important; border-color: #254057 !important;
  box-shadow: 0 8px 24px rgba(0,0,0,.16) !important; backdrop-filter: blur(12px);
}
.atv-bc-left, .atv-bc-right { color: #91aabd !important; }
.atv-bc-left b { color: #e5f6ff !important; }
.atv-bc-tag { background: #15283b !important; border-color: #2b465e !important; color: #b8d1e2 !important; }
div[class*="st-key-main_video_panel"],
div[class*="st-key-control_panel"],
div[class*="st-key-timeline_panel"],
div[class*="st-key-recent_panel"],
div[class*="st-key-panel_"],
div[class*="st-key-ov_"],
div[class*="st-key-history_"] {
  background: linear-gradient(155deg, rgba(17,32,49,.98), rgba(13,25,40,.98)) !important;
  border: 1px solid #254158 !important; border-radius: 16px !important;
  box-shadow: 0 12px 32px rgba(0,0,0,.20), inset 0 1px rgba(255,255,255,.025) !important;
}
.card-title, .sec h2, .pn-head h3 { color: #e7f6ff !important; }
.pn-head small { color: #8ca8bc !important; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  background: #172a3d !important; border: 1px solid #34516a !important;
  color: #d8eaf7 !important; box-shadow: 0 3px 10px rgba(0,0,0,.14) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: #1b354b !important; border-color: #22d3ee !important; color: #f0fdff !important;
  box-shadow: 0 0 0 1px rgba(34,211,238,.12), 0 7px 18px rgba(0,0,0,.2) !important;
}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%) !important;
  border: 1px solid #22d3ee !important; color: #f0fdff !important;
  box-shadow: 0 5px 18px rgba(8,145,178,.24) !important;
}
.stButton > button[kind="primary"] p, .stButton > button[data-testid="stBaseButton-primary"] p { color: #f0fdff !important; }
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
  background: linear-gradient(135deg, #06b6d4, #0891b2) !important; box-shadow: 0 7px 22px rgba(34,211,238,.24) !important;
}
.stButton > button:disabled { background: #172536 !important; border-color: #263b4f !important; color: #61788c !important; }

/* Inputs, selectboxes, checkboxes, sliders */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
[data-testid="stCheckbox"] label, [data-testid="stToggle"] label, label,
[data-testid="stMarkdownContainer"] p { color: #c5d8e7 !important; }
[data-testid="stCaptionContainer"], .stCaption { color: #8ba3b8 !important; }
input, textarea { color: #e6f4ff !important; background: #102033 !important; caret-color: #22d3ee !important; }
input::placeholder, textarea::placeholder { color: #718ba0 !important; }
div[data-baseweb="input"], div[data-baseweb="select"] > div {
  background: #102033 !important; border: 1px solid #34516a !important; border-radius: 9px !important;
}
div[data-baseweb="select"] *, [data-baseweb="popover"] * { color: #d9eaf6 !important; }
[data-baseweb="popover"] [role="option"] { background: #12263a !important; }
[data-baseweb="popover"] [role="option"]:hover { background: #1b3b52 !important; }
[data-testid="stSlider"] [role="slider"] {
  background: #22d3ee !important; border: 2px solid #e6fbff !important;
  box-shadow: 0 0 0 3px rgba(34,211,238,.18), 0 0 14px rgba(34,211,238,.22) !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] { background: #34516a !important; }
[data-testid="stProgress"] > div { background: #1a2e42 !important; }
[data-testid="stProgress"] > div > div > div { background: linear-gradient(90deg, #0891b2, #22d3ee) !important; }
[data-testid="stCheckbox"] svg { color: #22d3ee !important; }

/* Video / uploader / HUD */
[data-testid="stFileUploader"] { background: #102033 !important; border: 1.5px dashed #3a6078 !important; }
[data-testid="stFileUploaderDropzone"] * { color: #a9c1d2 !important; }
[data-testid="stFileUploader"] button, [data-testid="stFileUploader"] button p {
  color: #eafcff !important; -webkit-text-fill-color: #eafcff !important;
  background: #087f9b !important; border-color: #22d3ee !important;
}
[data-testid="stFileUploader"] button:hover { background: #0891b2 !important; }
[data-testid="stVideo"] { border: 1px solid #2c6077 !important; background: #030912 !important; box-shadow: 0 0 0 1px rgba(34,211,238,.07), 0 10px 28px rgba(0,0,0,.32) !important; }
video { background: #030912 !important; }
.hud-box {
  background: radial-gradient(ellipse at center, rgba(8,145,178,.10), transparent 70%), #0a1625 !important;
  border: 1px dashed #2e6178 !important; color: #9bb7c9 !important;
}
.hud-box b { color: #dffaff !important; }
.hud-corner { border-color: #22d3ee !important; }
.legend { color: #b4c9d9 !important; }
.source-file, .stat-row {
  background: #14283b !important; border-color: #2b465e !important; color: #d8eaf7 !important;
}
.source-file span, .stat-row .v { color: #a8c0d1 !important; }
.progress-caption { color: #c2d8e8 !important; }
.control-label { color: #c8dce9 !important; }
.val-pill { background: rgba(34,211,238,.09) !important; border-color: rgba(34,211,238,.25) !important; }
.control-divider { background: #294258 !important; }

/* KPI cards and analytics */
.kpi-grid { gap: 14px !important; }
.smart-kpi, .kpi {
  background: linear-gradient(145deg, #14283b 0%, #101e30 100%) !important;
  border: 1px solid #2a465d !important; border-left: 4px solid var(--accent) !important;
  box-shadow: 0 8px 22px rgba(0,0,0,.16), inset 0 1px rgba(255,255,255,.025) !important;
}
.smart-kpi:hover { border-color: var(--accent) !important; box-shadow: 0 0 0 1px color-mix(in srgb, var(--accent) 20%, transparent), 0 12px 28px rgba(0,0,0,.25) !important; }
.smart-kpi.danger { background: linear-gradient(145deg, #321d2b, #1b1c2b) !important; border-color: #71394b !important; }
.sk-title, .kpi .k-label { color: #a6bdce !important; }
.sk-val, .kpi .k-value { color: #eafaff !important; }
.sk-val small, .sk-sub, .kpi .k-vi, .kpi .k-unit, .kpi .k-note { color: #8faabd !important; }
.sk-ico { background: rgba(34,211,238,.10) !important; }
.chart-legend { color: #b6ccdc !important; }
.donut-box { background: #13263a !important; border-color: #2a465d !important; }
.donut-cap { color: #a7bdcd !important; }
.donut-pct { color: #5eeaf5 !important; }
.event-meta { color: #e1f1fa !important; }
.event-time { color: #8ba3b8 !important; }
.event-tag { color: #07121e !important; }
.ov-hero { background: linear-gradient(120deg, #102b40 0%, #102236 58%, #102c35 100%) !important; border-color: #28546b !important; box-shadow: 0 8px 26px rgba(0,0,0,.18) !important; }
.ov-hero-eyebrow { background: rgba(34,211,238,.13) !important; color: #67e8f9 !important; }
.ov-hero h1, .ov-pill b, .ov-mode-title, .ov-step-body b, .ov-spec-row b { color: #e7f7ff !important; }
.ov-hero p, .ov-mode-desc, .ov-step-body span { color: #a9c0d1 !important; }
.ov-pill { background: #142b40 !important; border-color: #31536b !important; }
.ov-pill small, .ov-mode-sub, .ov-spec-row span { color: #91aabd !important; }
.ov-step, .ov-spec-row { background: #13263a !important; border-color: #2b465d !important; }
.ov-step-num { background: rgba(34,211,238,.12) !important; border-color: rgba(34,211,238,.30) !important; color: #67e8f9 !important; }

/* Tables, dataframes, alerts and footer */
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid #29465e !important; border-radius: 11px !important; overflow: hidden !important; }
[data-testid="stDataFrame"] * { color: #d5e7f3 !important; }
[data-testid="stAlert"] { background: #14283b !important; border: 1px solid #31516a !important; color: #d8eaf7 !important; }
.atv-footer { border-top-color: #294258 !important; color: #819caf !important; }
.atv-footer b { color: #cceaf7 !important; }
div[class*="st-key-export_bar"] {
  background: rgba(12,25,40,.94) !important; border-color: #2a4a61 !important;
  box-shadow: 0 -8px 24px rgba(0,0,0,.22) !important;
}
hr { border-color: #294258 !important; }

@media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 760px) {
  [data-testid="stSidebar"] { min-width: 245px !important; max-width: 245px !important; }
  .block-container { padding: .75rem !important; }
  .kpi-grid { grid-template-columns: 1fr; }
  .atv-footer { flex-direction: column; align-items: flex-start; }
}

/* ================================================================
   POLISH LAYER v2 - chỉ chỉnh giao diện, không đụng logic suy luận
   ================================================================ */
/* 1. Logo/tên app ở sidebar: chữ đang giữ màu tối trên nền tối */
.sb-brand-text b { color: #effbff !important; -webkit-text-fill-color: #effbff !important; background: none !important; }
.sb-brand-text small { color: #8ba3b8 !important; -webkit-text-fill-color: #8ba3b8 !important; }
.sb-brand-divider { background: #20384e !important; border-color: #20384e !important; }
[data-testid="stSidebar"] [data-testid="stImage"] img {
  border-radius: 12px; box-shadow: 0 0 0 1px rgba(34,211,238,.25), 0 0 22px rgba(34,211,238,.14);
}

/* 2. Khối trắng chói (trạng thái trống) -> tối, nét đứt */
.empty {
  background: linear-gradient(180deg, rgba(20,36,56,.78), rgba(16,29,45,.55)) !important;
  border: 1.5px dashed var(--line-strong) !important;
  border-radius: 14px !important;
  color: var(--muted) !important;
}
.empty b { color: var(--text) !important; }

/* 3. Việt hóa nút Upload và dòng giới hạn của Streamlit */
[data-testid="stFileUploader"] button p { font-size: 0 !important; line-height: 0 !important; }
[data-testid="stFileUploader"] button p::after {
  content: "Chọn tệp"; font-size: 13px !important; line-height: 1.2 !important; letter-spacing: .2px;
}
/* nút "Chọn tệp": bỏ viền oval kép do p bị tô nền riêng */
[data-testid="stFileUploader"] button p {
  background: transparent !important; border: 0 !important; box-shadow: none !important; padding: 0 !important;
}
/* dòng giới hạn: nhắm vào khối không chứa nút, độc lập tên thẻ */
[data-testid="stFileUploaderDropzone"] > div:not(:has(button)) * { font-size: 0 !important; }
[data-testid="stFileUploaderDropzone"] > div:not(:has(button))::after {
  content: "Tối đa 100MB mỗi tệp"; font-size: 12.5px; color: #a9c1d2; white-space: nowrap;
}
div[class*="st-key-dashboard_video_upload"] [data-testid="stFileUploaderDropzone"] > div:not(:has(button))::after {
  content: "Tối đa 100MB · MP4, AVI, MOV, M4V";
}
div[class*="st-key-image_uploader"] [data-testid="stFileUploaderDropzone"] > div:not(:has(button))::after {
  content: "Tối đa 100MB · JPG, PNG, BMP, WEBP";
}

/* 4. Icon KPI dạng SVG thay emoji, số liệu thẳng hàng */
.sk-ico svg { width: 18px; height: 18px; display: block; }
.sk-val, .kpi .k-value { font-variant-numeric: tabular-nums; letter-spacing: -.5px; }
.sk-val.is-empty { color: #5f7a90 !important; }

/* 5. Chuyển động nhẹ, focus rõ, thanh cuộn đồng bộ tông */
.smart-kpi, .kpi, .atv-footer, [data-testid="stSidebar"] .stButton > button {
  transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease, background .18s ease;
}
.smart-kpi:hover { transform: translateY(-2px); }
button:focus-visible, [data-baseweb] :focus-visible { outline: 2px solid #22d3ee !important; outline-offset: 2px; }
::selection { background: rgba(34,211,238,.28); }
* { scrollbar-width: thin; scrollbar-color: #31516b transparent; }
*::-webkit-scrollbar { width: 8px; height: 8px; }
*::-webkit-scrollbar-thumb { background: #31516b; border-radius: 8px; }
*::-webkit-scrollbar-track { background: transparent; }

/* 6. Giữ nút mở lại sidebar: CSS gốc ẩn cả header nên thu gọn xong không mở lại được */
header[data-testid="stHeader"] {
  display: block !important; height: 3rem !important; min-height: 0 !important;
  background: transparent !important; pointer-events: none !important; z-index: 999 !important;
}
header[data-testid="stHeader"] * { pointer-events: none !important; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapsedControl"] *, [data-testid="collapsedControl"] * {
  display: flex !important; visibility: visible !important; pointer-events: auto !important;
}
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
  align-items: center; background: #0b1727 !important; border: 1px solid #2b4c64 !important;
  border-radius: 10px !important; margin: 10px 0 0 10px !important; padding: 2px !important;
  color: #67e8f9 !important; box-shadow: 0 0 18px rgba(34,211,238,.18) !important;
}
[data-testid="stExpandSidebarButton"] svg, [data-testid="stSidebarCollapsedControl"] svg, [data-testid="collapsedControl"] svg {
  color: #67e8f9 !important; fill: currentColor;
}

/* 7. Chip / pill trắng -> hợp tông nền tối */
.legend span, .chip {
  background: rgba(20,36,56,.9) !important; border-color: #2a465d !important; color: #cfe3f1 !important;
}
.chip.ok { background: rgba(52,211,153,.12) !important; border-color: rgba(52,211,153,.35) !important; color: #6ee7b7 !important; }
.chip.warn { background: rgba(251,191,36,.12) !important; border-color: rgba(251,191,36,.35) !important; color: #fcd34d !important; }
.chip.bad { background: rgba(251,113,133,.12) !important; border-color: rgba(251,113,133,.38) !important; color: #fda4af !important; }
.chip.tech { background: rgba(56,189,248,.12) !important; border-color: rgba(56,189,248,.35) !important; color: #7dd3fc !important; }
.live-pill.wait { background: rgba(148,163,184,.12) !important; border-color: #31516b !important; color: #b6cadb !important; }
.live-pill.ready { background: rgba(56,189,248,.12) !important; border-color: rgba(56,189,248,.35) !important; color: #7dd3fc !important; }
.live-pill.active { background: rgba(251,113,133,.12) !important; border-color: rgba(251,113,133,.38) !important; color: #fda4af !important; }
.live-pill.paused { background: rgba(251,191,36,.12) !important; border-color: rgba(251,191,36,.35) !important; color: #fcd34d !important; }
.live-pill.done { background: rgba(52,211,153,.12) !important; border-color: rgba(52,211,153,.35) !important; color: #6ee7b7 !important; }

/* 8. Sidebar luôn mở trên máy tính: CSS gốc ép width 274px nên trạng thái thu gọn của Streamlit bị lỗi.
   Bỏ nút thu gọn và vô hiệu hóa trạng thái đã thu gọn. Trên màn hình nhỏ vẫn để Streamlit tự xử lý. */
@media (min-width: 992px) {
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stSidebarHeader"] button { display: none !important; }
  [data-testid="stSidebar"], [data-testid="stSidebar"][aria-expanded="false"] {
    transform: none !important; margin-left: 0 !important; left: 0 !important;
    visibility: visible !important; opacity: 1 !important;
  }
}
"""


def render_html(markup):
    cleaned = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


def inject_css():
    st.markdown("<style>\n" + APP_CSS + "\n</style>", unsafe_allow_html=True)


def bgr_to_hex(bgr):
    b, g, r = bgr
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def panel(key):
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


def section_title(title, subtitle=""):
    render_html(f'<div class="sec"><h2>{title}</h2></div>')


def panel_head(title, caption):
    render_html(f'<div class="pn-head"><h3>{title}</h3><small>{caption}</small></div>')


def empty_state(title, text):
    render_html(f'<div class="empty"><b>{title}</b>{text}</div>')


def hud_empty_state(title, text):
    render_html(
        f"""
        <div class="hud-box">
          <span class="hud-corner tl"></span>
          <span class="hud-corner tr"></span>
          <span class="hud-corner bl"></span>
          <span class="hud-corner br"></span>
          <b>{title}</b>
          <span>{text}</span>
        </div>
        """
    )


def get_device_label():
    try:
        if torch.cuda.is_available():
            return "GPU: " + torch.cuda.get_device_name(0), True
    except Exception:
        pass
    return "CPU", False


def format_duration(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def get_video_info(video_path):
    info = {"width": 0, "height": 0, "fps": 0.0, "frames": 0, "duration": 0.0}
    try:
        cap_info = cv2.VideoCapture(video_path)
        info["width"] = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info["fps"] = cap_info.get(cv2.CAP_PROP_FPS) or 0.0
        info["frames"] = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_info.release()
        if info["fps"] > 0 and info["frames"] > 0:
            info["duration"] = info["frames"] / info["fps"]
    except Exception:
        pass
    return info


def render_legend():
    hex_helmet = bgr_to_hex(LABEL_COLORS["helmet"])
    hex_bike = bgr_to_hex(LABEL_COLORS["motobike"])
    hex_no = bgr_to_hex(LABEL_COLORS["no-helmet"])
    render_html(
        f"""
        <div class="legend">
          <span><i style="background:{hex_bike}"></i>Xe máy</span>
          <span><i style="background:{hex_helmet}"></i>Đội mũ (Helmet)</span>
          <span><i style="background:{hex_no}"></i>Không đội mũ (Vi phạm)</span>
        </div>
        """
    )


ICONS = {
    "bike": '<circle cx="5.5" cy="17.5" r="3.5"/><circle cx="18.5" cy="17.5" r="3.5"/><path d="M15 6h2l2 5.5M5.5 17.5 9 9h5l4.5 8.5M8 6h3"/>',
    "shield": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M8.5 12l2.5 2.5L16 9.5"/>',
    "alert": '<path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18h.01"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "bolt": '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>',
}


def icon(name, size=20):
    return (
        f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>'
    )


def kpi_card(label, vi_label, value, unit, note, accent, icon_name):
    return (
        f'<div class="kpi" style="--accent:{accent}">'
        f'<div class="k-top"><div><div class="k-label">{label}</div><div class="k-vi">{vi_label}</div></div>'
        f'<div class="k-ico" style="color:{accent}">{icon(icon_name)}</div></div>'
        f'<div class="k-value">{value}<span class="k-unit">{unit}</span></div>'
        f'<div class="k-note">{note}</div>'
        f'</div>'
    )


def style_chart(chart):
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=C_MUTED, titleColor=C_TEXT, gridColor=C_GRID,
            domainColor="#34516a", tickColor="#34516a", labelFontSize=11.5, titleFontSize=12
        )
        .configure_legend(labelColor=C_TEXT, titleColor=C_MUTED, labelFontSize=12)
    )


def show_altair(chart):
    try:
        st.altair_chart(style_chart(chart), use_container_width=True, theme=None)
    except TypeError:
        st.altair_chart(style_chart(chart), use_container_width=True)


def show_table(df):
    try:
        st.dataframe(df, use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True)


def render_footer():
    render_html(
        """
        <div class="atv-footer">
          <span><b>AI TRAFFIC VISION v1.0</b> · Hệ thống giám sát giao thông thông minh</span>
          <span>Powered by YOLOv8s + ByteTrack Tracking</span>
        </div>
        """
    )


# =====================================================================
# NHẬN DIỆN HÌNH ẢNH & WEBCAM
# =====================================================================
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
MODE_VIDEO = "Video"
MODE_IMAGE = "Hình ảnh"
MODE_WEBCAM = "Webcam"

RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


class FrameAnalyzer:
    def __init__(self, model, params, use_tracking, max_dim=1280):
        self.model = model
        self.p = params
        self.use_tracking = use_tracking
        self.max_dim = max_dim
        self.tracker_path = get_custom_tracker_yaml() if use_tracking else None
        self.min_conf = min(
            params["conf_helmet"], params["conf_no_helmet"], params["conf_bike"]
        )
        self.label_stabilizer = TrackLabelStabilizer(conf_alpha=params["ema_alpha"])
        self.head_tracker = HeadObjectTracker(max_age=12, bbox_alpha=BBOX_EMA_ALPHA,
                                              label_alpha=params["ema_alpha"])
        self.bike_track_frames = defaultdict(int)
        self.head_track_votes = defaultdict(list)
        self.bbox_smoother = TrackBBoxSmoother(BBOX_EMA_ALPHA)
        self.frame_idx = 0
        self.conf_sum = 0.0
        self.conf_n = 0

    def _infer(self, frame, conf_override=None):
        infer_conf = float(conf_override) if conf_override is not None else self.min_conf
        kwargs = dict(
            imgsz=self.p["imgsz"], conf=infer_conf, iou=0.45,
            agnostic_nms=False, verbose=False,
        )
        if self.use_tracking:
            try:
                return self.model.track(
                    frame, tracker=self.tracker_path, persist=True, **kwargs
                )[0]
            except Exception:
                self.use_tracking = False
        return self.model(frame, **kwargs)[0]

    @staticmethod
    def _label(vis, text, x1, y1, y2, color, above, fs, th, h):
        (tw, tth), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        if above:
            y_t = max(y1 - 4, tth + 4)
        else:
            y_t = min(y2 + tth + 4, h - 2)
        cv2.rectangle(vis, (x1, y_t - tth - 4), (x1 + tw, y_t + 2), color, -1)
        cv2.putText(
            vis, text, (x1, y_t - 2), cv2.FONT_HERSHEY_SIMPLEX,
            fs, (255, 255, 255), th
        )

    def analyze(self, frame, probe_low_conf=False):
        h, w = frame.shape[:2]
        scale = min(self.max_dim / max(h, w), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]

        probe_conf = min(0.10, self.min_conf) if probe_low_conf else self.min_conf
        results = self._infer(frame, conf_override=probe_conf)
        vis = frame.copy()
        fs = max(0.45, min(w, h) / 900.0)
        th = 1 if fs < 0.8 else 2
        box_th = max(2, int(round(min(w, h) / 400)))

        raw_probe = []
        bike_boxes, raw_head_detections = [], []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            raw_conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            orig_name = str(self.model.names[cls_id])
            raw_cls = normalize_detection_class(orig_name, cls_id=cls_id)
            raw_probe.append((raw_cls, raw_conf))

            if raw_cls in ("helmet", "no-helmet"):
                if raw_cls == "no-helmet" and min(x2 - x1, y2 - y1) < max(12.0, w * 0.010):
                    continue
                raw_head_detections.append((x1, y1, x2, y2, raw_cls, raw_conf))
                continue

            if box.id is not None:
                tid = int(box.id[0].item())
                stable_cls, smooth_conf = self.label_stabilizer.update(tid, raw_cls, raw_conf)
                x1, y1, x2, y2 = self.bbox_smoother.update(tid, (x1, y1, x2, y2))
            else:
                tid, smooth_conf, stable_cls = None, raw_conf, raw_cls

            if stable_cls in ("motobike", "motorbike") and smooth_conf >= self.p["conf_bike"]:
                bike_boxes.append((x1, y1, x2, y2, "motorbike", smooth_conf, tid))

        bike_boxes = suppress_duplicate_bike_boxes(bike_boxes, iou_threshold=0.60)
        raw_head_detections = suppress_duplicate_head_boxes(raw_head_detections, iou_threshold=0.55)
        head_boxes = self.head_tracker.update(
            raw_head_detections, self.p["conf_helmet"], self.p["conf_no_helmet"]
        )

        frame_confs = []
        show_id = self.p["show_track_id"] and self.use_tracking

        for x1, y1, x2, y2, name, conf, tid in bike_boxes:
            if tid is not None:
                self.bike_track_frames[tid] += 1
            frame_confs.append(conf)
            color = LABEL_COLORS.get(name, (235, 99, 37))
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(vis, p1, p2, color, box_th)
            prefix = f"#{tid} " if (show_id and tid is not None) else ""
            self._label(vis, f"{prefix}{display_overlay_name(name)} {conf:.2f}", p1[0], p1[1], p2[1],
                        color, False, fs, th, h)

        filtered = suppress_duplicate_head_boxes(head_boxes, iou_threshold=0.55)

        n_helmet = n_no_helmet = 0
        for x1, y1, x2, y2, name, conf, tid, raw_name in filtered:
            if self.p["filter_on_bike"] and not is_head_on_motorbike((x1, y1, x2, y2), bike_boxes):
                continue

            frame_confs.append(conf)
            if tid is not None and raw_name is not None:
                self.head_track_votes[tid].append((raw_name, float(conf)))
            if name == "helmet":
                n_helmet += 1
            else:
                n_no_helmet += 1

            color = LABEL_COLORS.get(name, (255, 255, 255))
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(vis, p1, p2, color, box_th)
            prefix = f"H{tid} " if (show_id and tid is not None) else ""
            self._label(vis, f"{prefix}{display_overlay_name(name)} {conf:.2f}", p1[0], p1[1], p2[1],
                        color, True, fs, th, h)

        self.conf_sum += sum(frame_confs)
        self.conf_n += len(frame_confs)
        self.frame_idx += 1

        raw_summary = ""
        if raw_probe:
            top_items = sorted(raw_probe, key=lambda x: x[1], reverse=True)[:4]
            raw_summary = ", ".join(f"{display_class_name(cls_n)} ({c:.2f})" for cls_n, c in top_items)

        counts = {
            "motorbike": len(bike_boxes),
            "helmet": n_helmet,
            "no_helmet": n_no_helmet,
            "avg_conf": (sum(frame_confs) / len(frame_confs) * 100) if frame_confs else 0.0,
            "raw_count": len(raw_probe),
            "raw_summary": raw_summary,
        }
        return vis, counts

    def summary(self):
        mf = self.p["min_track_frames"]
        bikes = sum(1 for f in self.bike_track_frames.values() if f >= mf)
        helmet = viol = 0
        for votes in self.head_track_votes.values():
            cls = classify_head_track(votes, 2, no_helmet_min_frames=3)
            helmet += int(cls == "helmet")
            viol += int(cls == "no-helmet")
        return {"motorbike": bikes, "helmet": helmet, "no_helmet": viol}


def analyze_still(model_path, bgr, params):
    model = load_yolo_model(model_path)
    if getattr(model, "predictor", None) is not None:
        model.predictor = None
    return FrameAnalyzer(model, params, use_tracking=False).analyze(bgr, probe_low_conf=True)


def decode_image(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def show_image(img):
    try:
        st.image(img, use_container_width=True)
    except TypeError:
        st.image(img, use_column_width=True)


def render_kpi_row(c, container=None):
    html = (
        '<div class="kpi-grid">'
        + kpi_card("XE MÁY", "Tổng số xe được phát hiện", f"{c['motorbike']}", "xe",
                   c.get("note_bike", "Phát hiện được"), C_PURPLE, "bike")
        + kpi_card("ĐỘI MŨ", "Số người đội mũ", f"{c['helmet']}", "người",
                   c.get("note_helmet", "Tuân thủ"), C_GREEN, "shield")
        + kpi_card("KHÔNG ĐỘI MŨ", "Số người không đội mũ", f"{c['no_helmet']}", "người",
                   c.get("note_viol", "Vi phạm"), C_RED, "alert")
        + kpi_card(c.get("4_label", "ĐỘ TIN CẬY"), c.get("4_vi", "Độ tin cậy trung bình"),
                   c["4_value"], c.get("4_unit", "%"),
                   c.get("4_note", "Trung bình các detection hợp lệ"),
                   C_CYAN, c.get("4_icon", "target"))
        + '</div>'
    )
    (container or st).markdown(html, unsafe_allow_html=True)


def render_still_result(title, original_bytes, annotated_bytes, counts, key, file_stem):
    total = counts["helmet"] + counts["no_helmet"]
    rate = f"{counts['helmet'] / total * 100:.0f}%" if total else "—"
    with panel(f"panel_still_{key}"):
        panel_head(
            html_lib.escape(title),
            f"{counts['motorbike']} xe · {counts['helmet']} đội mũ · "
            f"{counts['no_helmet']} không đội mũ · tuân thủ {rate}"
        )
        if counts["no_helmet"] > 0:
            st.markdown(
                f'<span class="chip bad">⚠️ Phát hiện {counts["no_helmet"]} người không đội mũ</span>',
                unsafe_allow_html=True
            )
        elif counts["helmet"] > 0:
            st.markdown('<span class="chip ok">✓ Tất cả người được ghi nhận đều đội mũ</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="chip warn">Không phát hiện đầu/mũ hợp lệ</span>',
                        unsafe_allow_html=True)
            raw_cnt = int(counts.get("raw_count", 0))
            raw_sum = counts.get("raw_summary", "")
            if raw_cnt > 0:
                render_html(
                    f'<div class="diag-note">Chẩn đoán: Mô hình tìm thấy <b>{raw_cnt}</b> đối tượng thô '
                    f'(cao nhất: <b>{html_lib.escape(raw_sum)}</b>) nhưng bị lọc do thấp hơn ngưỡng tin cậy hoặc không nằm trên xe máy. '
                    f'Nếu ảnh chụp xe đạp điện (không có nhãn xe máy), hãy bỏ chọn mục <b>"Chỉ lấy đầu/mũ trên xe máy"</b> ở thanh bên trái.</div>'
                )

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.caption("Ảnh ban đầu")
            show_image(original_bytes)
        with c2:
            st.caption("Kết quả nhận diện")
            show_image(annotated_bytes)
        st.download_button(
            "⬇️ Tải ảnh kết quả (.jpg)", data=annotated_bytes,
            file_name=f"{file_stem}_detected.jpg", mime="image/jpeg",
            key=f"dl_still_{key}"
        )


def encode_jpg(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else b""


def render_image_mode(model_path, model_exists, params):
    section_title("Nhận diện hình ảnh")

    if "image_results" not in st.session_state:
        st.session_state.image_results = None
    if "image_sig" not in st.session_state:
        st.session_state.image_sig = None

    with panel("panel_image_input"):
        panel_head("ẢNH ĐẦU VÀO", "Tệp ảnh giao thông · JPG, PNG, BMP, WEBP")
        render_legend()
        files = st.file_uploader(
            "Chọn một hoặc nhiều ảnh:", type=IMAGE_TYPES,
            accept_multiple_files=True, key="image_uploader"
        )
        sig = tuple((f.name, f.size) for f in files) if files else ()
        if sig != st.session_state.image_sig:
            st.session_state.image_sig = sig
            st.session_state.image_results = None

        if files:
            if st.button("▶  Bắt đầu nhận diện ảnh", type="primary",
                         use_container_width=True, key="btn_image_detect"):
                if not model_exists:
                    st.error(f"❌ Không tìm thấy tệp mô hình tại `{model_path}`.")
                else:
                    progress = st.progress(0)
                    out = []
                    for i, f in enumerate(files):
                        data = f.getvalue()
                        bgr = decode_image(data)
                        if bgr is None:
                            out.append({"name": f.name, "error": True})
                        else:
                            vis, counts = analyze_still(model_path, bgr, params)
                            out.append({
                                "name": f.name, "error": False, "original": data,
                                "annotated": encode_jpg(vis), "counts": counts,
                            })
                        progress.progress((i + 1) / len(files))
                    st.session_state.image_results = out
        else:
            hud_empty_state("Chưa có ảnh đầu vào", "Tải lên một hoặc nhiều ảnh để bắt đầu nhận diện.")

    results = st.session_state.image_results
    if not results:
        return

    ok_results = [r for r in results if not r["error"]]
    for r in results:
        if r["error"]:
            st.warning(f"Không đọc được ảnh: {r['name']}")
    if not ok_results:
        return

    section_title("Phân tích hình ảnh")
    tot_bike = sum(r["counts"]["motorbike"] for r in ok_results)
    tot_h = sum(r["counts"]["helmet"] for r in ok_results)
    tot_v = sum(r["counts"]["no_helmet"] for r in ok_results)
    confs = [r["counts"]["avg_conf"] for r in ok_results if r["counts"]["avg_conf"] > 0]
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    render_kpi_row({
        "motorbike": tot_bike, "helmet": tot_h, "no_helmet": tot_v,
        "note_bike": f"Tổng trên {len(ok_results)} ảnh",
        "note_helmet": "Cộng dồn các ảnh",
        "note_viol": "Cộng dồn các ảnh",
        "4_value": f"{avg_conf:.1f}",
    })

    for i, r in enumerate(ok_results):
        stem = os.path.splitext(r["name"])[0]
        render_still_result(r["name"], r["original"], r["annotated"], r["counts"], f"img{i}", stem)

    rows = []
    for r in ok_results:
        c = r["counts"]
        t = c["helmet"] + c["no_helmet"]
        rows.append({
            "Tên ảnh": r["name"], "Xe máy": c["motorbike"], "Đội mũ": c["helmet"],
            "Không đội mũ": c["no_helmet"],
            "Tỷ lệ đội mũ (%)": round(c["helmet"] / t * 100, 1) if t else None,
            "Độ tin cậy TB (%)": round(c["avg_conf"], 1),
        })
    df = pd.DataFrame(rows)
    section_title("Xuất kết quả")
    with panel("panel_image_export"):
        show_table(df)
        st.download_button(
            "⇩  Tải Xuống Bảng Thống Kê (.csv)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="Image_Detection_Report.csv", mime="text/csv",
            key="download_image_csv"
        )


if WEBRTC_OK:
    class HelmetVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.lock = threading.Lock()
            self.analyzer = None
            self.config_key = None
            self.counts = {"motorbike": 0, "helmet": 0, "no_helmet": 0, "avg_conf": 0.0}
            self.fps = 0.0

        def configure(self, key, factory):
            with self.lock:
                if key != self.config_key:
                    self.analyzer = factory()
                    self.config_key = key

        def snapshot(self):
            with self.lock:
                summ = self.analyzer.summary() if self.analyzer else None
                return dict(self.counts), self.fps, summ

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            t0 = time.perf_counter()
            with self.lock:
                if self.analyzer is not None:
                    try:
                        img, self.counts = self.analyzer.analyze(img)
                    except Exception:
                        pass
            dt = time.perf_counter() - t0
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if not self.fps else 0.8 * self.fps + 0.2 * inst
            return av.VideoFrame.from_ndarray(img, format="bgr24")


def live_stats_html(counts, fps, summ):
    summ = summ or {"motorbike": 0, "helmet": 0, "no_helmet": 0}
    return (
        '<div class="kpi-grid">'
        + kpi_card("XE MÁY", "Số xe trong khung hiện tại", f"{counts['motorbike']}", "xe",
                   f"Tích lũy: {summ['motorbike']} xe duy nhất", C_PURPLE, "bike")
        + kpi_card("ĐỘI MŨ", "Số người đội mũ trong khung hiện tại", f"{counts['helmet']}", "người",
                   f"Tích lũy: {summ['helmet']} người duy nhất", C_GREEN, "shield")
        + kpi_card("KHÔNG ĐỘI MŨ", "Số người vi phạm trong khung hiện tại", f"{counts['no_helmet']}", "người",
                   f"Tích lũy: {summ['no_helmet']} người duy nhất", C_RED, "alert")
        + kpi_card("TỐC ĐỘ XỬ LÝ", "Số khung hình mỗi giây", f"{fps:.1f}", "FPS",
                   "Phụ thuộc GPU/CPU và imgsz", C_CYAN, "chart")
        + '</div>'
    )


def render_webcam_mode(model_path, model_exists, params):
    section_title("Nhận diện webcam")
    if not model_exists:
        st.error(f"❌ Không tìm thấy tệp mô hình tại `{model_path}`.")
        return

    tab_live, tab_snap = st.tabs(["Trực tiếp", "Chụp ảnh"])

    with tab_snap:
        with panel("panel_cam_snap"):
            panel_head("CHỤP ẢNH TỪ WEBCAM", "Chụp một ảnh để nhận diện")
            render_legend()
            photo = st.camera_input("Chụp ảnh từ webcam", key="cam_snapshot")
        if photo is not None:
            data = photo.getvalue()
            bgr = decode_image(data)
            if bgr is None:
                st.warning("Không đọc được ảnh từ webcam.")
            else:
                with st.spinner("Đang nhận diện..."):
                    vis, counts = analyze_still(model_path, bgr, params)
                render_still_result("Ảnh chụp từ webcam", data, encode_jpg(vis),
                                    counts, "snap", "webcam_snapshot")

    with tab_live:
        with panel("panel_cam_live"):
            panel_head("WEBCAM TRỰC TIẾP", "Nhận diện và theo dõi phương tiện theo thời gian thực")
            render_legend()
            if not WEBRTC_OK:
                st.info(
                    "Chế độ webcam trực tiếp cần thư viện `streamlit-webrtc`. Cài đặt bằng lệnh:\n\n"
                    "`pip install streamlit-webrtc av`\n\n"
                    "rồi chạy lại ứng dụng. Trong lúc đó bạn vẫn dùng được tab **Chụp ảnh**."
                )
            else:
                ctx = webrtc_streamer(
                    key="helmet-live",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIG,
                    media_stream_constraints={
                        "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
                        "audio": False,
                    },
                    video_processor_factory=HelmetVideoProcessor,
                    async_processing=True,
                )
                st.caption(
                    "Nhấn START để cho phép trình duyệt truy cập camera. Số liệu bên dưới cập nhật liên tục."
                )
                stats_ph = st.empty()
                cfg_key = (model_path, os.path.getmtime(model_path)) + tuple(sorted(params.items()))
                factory = lambda: FrameAnalyzer(load_yolo_model(model_path), params, use_tracking=True)

                while ctx.state.playing:
                    proc = ctx.video_processor
                    if proc is not None:
                        proc.configure(cfg_key, factory)
                        counts, fps, summ = proc.snapshot()
                        stats_ph.markdown(live_stats_html(counts, fps, summ), unsafe_allow_html=True)
                    time.sleep(0.5)


# ------------------------ BACKGROUND VIDEO TASK ------------------------
def _video_detection_worker(task, model, args):
    started = time.perf_counter()
    try:
        result = run_detection(
            *args,
            progress_state=task["progress"],
            pause_event=task["pause_event"],
            model_override=model,
            only_violation_logs=task.get("only_violation_logs", False),
        )
        task["result"] = result
        task["elapsed"] = max(time.perf_counter() - started, 0.001)
        task["status"] = "done"
    except Exception as exc:
        task["error"] = f"{type(exc).__name__}: {exc}"
        task["status"] = "error"
    finally:
        task["done"] = True


def _render_video_status():
    task = st.session_state.get("video_detection_task")
    has_source = bool(st.session_state.get("video_source_ready", False))
    processed_path = st.session_state.get("processed_video_path")
    if task and not task.get("done"):
        if task["pause_event"].is_set():
            css, label = "paused", "TẠM DỪNG"
        else:
            css, label = "active", "LIVE | ĐANG PHÂN TÍCH"
    elif processed_path and os.path.exists(processed_path):
        css, label = "done", "ĐÃ XỬ LÝ"
    elif has_source:
        css, label = "ready", "SẴN SÀNG"
    else:
        css, label = "wait", "CHỜ DỮ LIỆU"
    render_html(f'<div style="text-align:right"><span class="live-pill {css}"><i></i>{label}</span></div>')


def render_video_progress():
    """Hiển thị tiến trình xử lý video và điều khiển tạm dừng/tiếp tục."""
    task = st.session_state.get("video_detection_task")
    if not task:
        return

    progress = task.get("progress") or {}
    fraction = float(progress.get("fraction", 0.0) or 0.0)
    fraction = max(0.0, min(1.0, fraction))
    frame = int(progress.get("frame", 0) or 0)
    total = int(progress.get("total", 0) or 0)

    if task.get("error"):
        st.error(f"Xử lý video thất bại: {task['error']}")
        return

    if task.get("done"):
        st.progress(1.0 if task.get("status") == "done" else fraction)
        if task.get("status") == "done":
            if not task.get("toasted"):
                task["toasted"] = True
                st.toast("Đã xử lý xong video", icon=":material/check_circle:")
            st.success("Đã xử lý video hoàn tất.")
        return

    st.markdown("**TIẾN TRÌNH XỬ LÝ VIDEO**")
    st.progress(fraction)
    if total > 0:
        st.caption(f"Đã xử lý {frame:,} / {total:,} khung hình ({fraction * 100:.1f}%).")
    else:
        st.caption(f"Đã xử lý {frame:,} khung hình ({fraction * 100:.1f}%).")

    pause_event = task.get("pause_event")
    if pause_event is not None:
        is_paused = pause_event.is_set()
        button_label = "▶ Tiếp tục xử lý" if is_paused else "Ⅱ Tạm dừng xử lý"
        if st.button(button_label, use_container_width=True, key="video_progress_pause_toggle"):
            if is_paused:
                pause_event.clear()
            else:
                pause_event.set()
            st.rerun()


def render_history_mode(device_label):
    st.markdown("## LỊCH SỬ PHÂN TÍCH")
    st.caption("Quản lý các video đã được AI phân tích.")

    history_items = get_history(limit=100)

    if not history_items:
        st.info("Chưa có lịch sử phân tích video.")
        return

    st.write(f"Tổng số lần phân tích: {len(history_items)}")

    for item in history_items:
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"**{item['video_name']}**")
                st.caption(f"Thời gian: {item['created_at']}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Xe máy", item["motorbike"])
                c2.metric("Có mũ", item["helmet"])
                c3.metric("Không mũ", item["no_helmet"])

                st.caption(
                    f"Độ tin cậy trung bình: {item['avg_conf']:.2%} | "
                    f"Thời gian xử lý: {item['elapsed']:.2f} giây"
                )

            with col2:
                if st.button(
                    "Xem chi tiết",
                    key=f"history_view_{item['id']}"
                ):
                    st.session_state.selected_history_id = item["id"]

                if st.button(
                    "Xóa",
                    key=f"history_delete_{item['id']}"
                ):
                    delete_history(item["id"])

                    if st.session_state.get("selected_history_id") == item["id"]:
                        st.session_state.pop("selected_history_id", None)

                    st.rerun()

    selected_id = st.session_state.get("selected_history_id")

    if selected_id is not None:
        selected = get_history_by_id(selected_id)

        if selected:
            st.divider()
            st.markdown("### CHI TIẾT PHÂN TÍCH")

            st.write("Tên video:", selected["video_name"])
            st.write("Thời gian:", selected["created_at"])

            st.markdown("#### Thống kê nhận diện")
            st.json(selected.get("stats", {}))

            st.markdown("#### Cấu hình phân tích")
            st.json(selected.get("config", {}))

            video_path = selected.get("result_video_path")

            if video_path and os.path.exists(video_path):
                st.video(video_path)
            else:
                st.warning("Không tìm thấy video kết quả.")

            if st.button("Đóng chi tiết"):
                st.session_state.pop("selected_history_id", None)
                st.rerun()


# =====================================================================
# KHỞI TẠO SESSION & GIAO DIỆN CHÍNH (DARK CYAN MODE)
# =====================================================================
inject_css()

if "input_mode" not in st.session_state:
    st.session_state.input_mode = MODE_VIDEO
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "nav_analysis"
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = None
if "uploaded_video_name" not in st.session_state:
    st.session_state.uploaded_video_name = None
if "uploaded_video_digest" not in st.session_state:
    st.session_state.uploaded_video_digest = None
if "video_detection_task" not in st.session_state:
    st.session_state.video_detection_task = None
if "video_source_ready" not in st.session_state:
    st.session_state.video_source_ready = False
if "recent_detections" not in st.session_state:
    st.session_state.recent_detections = []


def set_mode(mode, nav_key=None):
    st.session_state.input_mode = mode
    if nav_key is not None:
        st.session_state.active_nav = nav_key
    elif mode == MODE_IMAGE:
        st.session_state.active_nav = "nav_image"
    else:
        st.session_state.active_nav = "nav_analysis"


# ------------------------ SIDEBAR ------------------------
with st.sidebar:
    logo_col, title_col = st.columns([1.2, 4])

    with logo_col:
        st.image("assets/logo.png", width=65)

    with title_col:
        render_html(
            '<div class="sb-brand-text">'
            '<b>AI TRAFFIC VISION</b>'
            '<small>Giám sát giao thông thông minh</small>'
            '</div>'
        )

    st.markdown(
        '<div class="sb-brand-divider"></div>',
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="sb-section">KHÔNG GIAN LÀM VIỆC</div>', unsafe_allow_html=True)
    nav_items = [
        ("Phân tích video", MODE_VIDEO, "nav_analysis", ":material/play_circle:"),
        ("Ảnh", MODE_IMAGE, "nav_image", ":material/image:"),
    ]
    for label, mode, key, icon_name in nav_items:
        is_active = st.session_state.active_nav == key
        try:
            st.button(
                label, key=key, icon=icon_name, use_container_width=True,
                type="primary" if is_active else "secondary",
                on_click=set_mode, args=(mode, key),
            )
        except TypeError:
            st.button(
                label, key=key, use_container_width=True,
                type="primary" if is_active else "secondary",
                on_click=set_mode, args=(mode, key),
            )

    st.markdown('<div class="sb-section">HỆ THỐNG</div>', unsafe_allow_html=True)
    is_hist_active = st.session_state.active_nav == "nav_history"
    try:
        st.button(
            "Lịch sử phát hiện", key="nav_history", icon=":material/history:",
            use_container_width=True,
            type="primary" if is_hist_active else "secondary",
            on_click=set_mode, args=(MODE_VIDEO, "nav_history")
        )
    except TypeError:
        st.button(
            "Lịch sử phát hiện", key="nav_history", use_container_width=True,
            type="primary" if is_hist_active else "secondary",
            on_click=set_mode, args=(MODE_VIDEO, "nav_history")
        )

    with st.expander("Cài đặt mô hình", expanded=st.session_state.show_settings):
        model_path = st.text_input(
            "Tệp trọng số (.pt)",
            value=DEFAULT_MODEL_PATH or "",
            key="model_path_v3",
        )
        # Streamlit text_input cần chuỗi; chuẩn hóa để tránh None gây TypeError.
        model_path = (model_path or "").strip()
        model_exists = bool(model_path) and os.path.isfile(model_path)
        imgsz_option = st.selectbox("Kích thước ảnh suy luận", [960, 1280, 640], index=0, key="ui_imgsz_v3")
        if st.session_state.input_mode == MODE_IMAGE:
            conf_helmet = st.slider("Ngưỡng tin cậy Đội mũ", 0.10, 1.00, 0.35, 0.05, key="ui_conf_helmet_v3")
            conf_no_helmet = st.slider("Ngưỡng tin cậy Không đội mũ", 0.10, 1.00, 0.40, 0.05, key="ui_conf_nohelmet_v3")
            conf_bike = st.slider("Ngưỡng tin cậy Xe máy", 0.10, 1.00, 0.35, 0.05, key="ui_conf_bike_v3")
        else:
            conf_helmet = float(st.session_state.get("right_conf_helmet_v3", 0.35))
            conf_no_helmet = float(st.session_state.get("right_conf_nohelmet_v3", 0.40))
            conf_bike = float(st.session_state.get("right_conf_bike_v3", 0.35))
        ema_alpha = 0.30
        min_track_frames = 3
        if st.session_state.input_mode == MODE_IMAGE:
            filter_on_bike = st.checkbox("Chỉ lấy đầu/mũ trên xe máy", value=True, key="ui_filter_bike_v3")
            show_track_id = st.checkbox("Hiển thị mã theo dõi (Track ID)", value=True, key="ui_show_track_v3")
        else:
            filter_on_bike = bool(st.session_state.get("right_filter_bike_v3", True))
            show_track_id = bool(st.session_state.get("right_track_id_v3", True))

    model_path = (model_path or "").strip()
    model_exists = bool(model_path) and os.path.isfile(model_path)
    device_label, device_is_gpu = get_device_label()
    task_now = st.session_state.get("video_detection_task")
    is_running_now = bool(task_now and not task_now.get("done"))

    if is_running_now:
        sys_title = "ĐANG HOẠT ĐỘNG"
        sys_color = "#2563eb"
    elif model_exists:
        sys_title = "SẴN SÀNG"
        sys_color = "#059669"
    else:
        sys_title = "CHƯA TÌM THẤY MODEL"
        sys_color = "#dc2626"

    short_pt = os.path.basename(os.path.dirname(os.path.dirname(model_path))) + "/" + os.path.basename(model_path) if model_exists else "Không tồn tại"
    classes_txt = get_model_classes_text(model_path) if model_exists else "—"

    render_html(
        f"""
        <div class="sb-system">
          <div class="sb-system-head">Trạng thái hệ thống</div>
          <div class="sb-system-status" style="color:{sys_color}">
            <span class="sys-dot" style="background:{sys_color}"></span>{sys_title}
          </div>
          <div class="sb-system-meta">
            Model: <b>{html_lib.escape(short_pt)}</b><br>
            Lớp: <b>{html_lib.escape(classes_txt)}</b><br>
            {html_lib.escape(device_label)} · ImgSz: {imgsz_option}
          </div>
        </div>
        """
    )

input_mode = st.session_state.input_mode

detect_params = {
    "imgsz": imgsz_option,
    "conf_helmet": conf_helmet,
    "conf_no_helmet": conf_no_helmet,
    "conf_bike": conf_bike,
    "ema_alpha": ema_alpha,
    "min_track_frames": min_track_frames,
    "filter_on_bike": filter_on_bike,
    "show_track_id": show_track_id,
}

# ------------------------ ROUTE THEO MỤC SIDEBAR ------------------------
if input_mode == MODE_IMAGE:
    render_image_mode(model_path, model_exists, detect_params)
    render_footer()
    st.stop()
elif st.session_state.active_nav == "nav_history":
    render_history_mode(device_label)
    render_footer()
    st.stop()

# ------------------------ PHÂN TÍCH VIDEO (nav_analysis) ------------------------
if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None
if "stats" not in st.session_state:
    st.session_state.stats = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None
if "saved_drive_path" not in st.session_state:
    st.session_state.saved_drive_path = None
if "run_config" not in st.session_state:
    st.session_state.run_config = None

input_video_path = None
video_info = None
source_name = None
source_size = 0

if st.session_state.uploaded_video_path and os.path.exists(st.session_state.uploaded_video_path):
    input_video_path = st.session_state.uploaded_video_path
    source_name = st.session_state.uploaded_video_name or "video.mp4"
    source_size = os.path.getsize(input_video_path)

st.session_state.video_source_ready = bool(input_video_path)
if input_video_path:
    video_info = get_video_info(input_video_path)

# ------------------------ BREADCRUMB BAR ------------------------
res_tag = f"{video_info['width']}×{video_info['height']} @ {video_info['fps']:.0f} FPS" if (video_info and video_info.get("width")) else "Chưa nạp nguồn video"
render_html(
    f"""
    <div class="atv-breadcrumb">
      <div class="atv-bc-left">
        <span>🏠 Phân tích video</span>
        <span>/</span>
        <b>Giám sát giao thông thời gian thực</b>
      </div>
      <div class="atv-bc-right">
        <span class="atv-bc-tag">{html_lib.escape(res_tag)}</span>
        <span class="atv-bc-tag">{html_lib.escape(device_label)}</span>
      </div>
    </div>
    """
)

# ------------------------ MAIN LIVE DASHBOARD ------------------------
main_col, right_col = st.columns([2.15, 0.92], gap="large")
with main_col:
    with panel("main_video_panel"):
        title_col, status_col = st.columns([1, 0.34], gap="small")
        with title_col:
            render_html('<div class="card-title" style="margin-bottom:0"><span>◉  Xem trước video thời gian thực</span></div>')
        with status_col:
            _render_video_status()
        render_legend()

        if not input_video_path:
            up_col_l, up_col, up_col_r = st.columns([0.10, 0.80, 0.10])
            with up_col:
                uploaded_file = st.file_uploader(
                    "Tải video giao thông", type=["mp4", "avi", "mov", "m4v"],
                    key="dashboard_video_upload", help="Hỗ trợ định dạng MP4, AVI, MOV, M4V.",
                )
            if uploaded_file is not None:
                upload_bytes = uploaded_file.getvalue()
                suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(upload_bytes)
                tmp.close()
                st.session_state.uploaded_video_path = tmp.name
                st.session_state.uploaded_video_name = uploaded_file.name
                st.session_state.uploaded_video_digest = hashlib.sha256(upload_bytes).hexdigest()
                st.session_state.last_uploaded_name = uploaded_file.name
                st.session_state.processed_video_path = None
                st.session_state.stats = None
                st.session_state.recent_detections = []
                st.session_state.saved_drive_path = None
                st.session_state.run_config = None
                st.session_state.video_detection_task = None
                st.rerun()
            hud_empty_state("Chưa có video", "Tải video lên để bắt đầu giám sát giao thông.")
        else:
            action_col, info_col = st.columns([0.34, 0.66], gap="small")
            with action_col:
                popover_context = st.popover("↻  Đổi video khác") if hasattr(st, "popover") else st.expander("Đổi video khác", expanded=False)
                with popover_context:
                    replacement = st.file_uploader(
                        "Chọn video thay thế", type=["mp4", "avi", "mov", "m4v"],
                        key="dashboard_video_replacement",
                    )
                    if replacement is not None:
                        replacement_bytes = replacement.getvalue()
                        replacement_digest = hashlib.sha256(replacement_bytes).hexdigest()
                        if replacement_digest != st.session_state.get("uploaded_video_digest"):
                            suffix = os.path.splitext(replacement.name)[1] or ".mp4"
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                            tmp.write(replacement_bytes)
                            tmp.close()
                            st.session_state.uploaded_video_path = tmp.name
                            st.session_state.uploaded_video_name = replacement.name
                            st.session_state.uploaded_video_digest = replacement_digest
                            st.session_state.last_uploaded_name = replacement.name
                            st.session_state.processed_video_path = None
                            st.session_state.stats = None
                            st.session_state.recent_detections = []
                            st.session_state.saved_drive_path = None
                            st.session_state.run_config = None
                            st.session_state.video_detection_task = None
                            st.rerun()
            with info_col:
                render_html(
                    f'<div class="source-file">{html_lib.escape(source_name or "Video")}'
                    f'<span>{source_size / 1024 / 1024:.1f} MB</span></div>'
                )

            if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
                st.video(st.session_state.processed_video_path)
            else:
                st.video(input_video_path)

        render_video_progress()

with right_col:
    with panel("control_panel"):
        render_html('<div class="card-title"><span>☷  Điều khiển phát hiện</span></div>')

        st.markdown(
            f'<div class="control-label"><span>Ngưỡng tin cậy Đội mũ</span>'
            f'<span class="val-pill" style="color:#0d9488">{conf_helmet:.2f}</span></div>',
            unsafe_allow_html=True
        )
        conf_helmet = st.slider("Ngưỡng tin cậy Đội mũ", 0.10, 1.00, float(conf_helmet), 0.05, key="right_conf_helmet_v3", label_visibility="collapsed")

        st.markdown(
            f'<div class="control-label"><span>Ngưỡng tin cậy Không đội mũ</span>'
            f'<span class="val-pill" style="color:#ef4444">{conf_no_helmet:.2f}</span></div>',
            unsafe_allow_html=True
        )
        conf_no_helmet = st.slider("Ngưỡng tin cậy Không đội mũ", 0.10, 1.00, float(conf_no_helmet), 0.05, key="right_conf_nohelmet_v3", label_visibility="collapsed")

        st.markdown(
            f'<div class="control-label"><span>Ngưỡng tin cậy Xe máy</span>'
            f'<span class="val-pill" style="color:#2563eb">{conf_bike:.2f}</span></div>',
            unsafe_allow_html=True
        )
        conf_bike = st.slider("Ngưỡng tin cậy Xe máy", 0.10, 1.00, float(conf_bike), 0.05, key="right_conf_bike_v3", label_visibility="collapsed")

        st.markdown('<div class="control-divider"></div>', unsafe_allow_html=True)
        show_helmet = st.checkbox("Hiển thị nhãn Đội mũ", value=True, key="show_class_helmet")
        show_no_helmet = st.checkbox("Hiển thị nhãn Không đội mũ", value=True, key="show_class_nohelmet")
        show_motorbike = st.checkbox("Hiển thị khung Xe máy", value=True, key="show_class_motorbike")
        filter_on_bike = st.checkbox("Chỉ nhận diện đầu/mũ trên xe máy", value=filter_on_bike, key="right_filter_bike_v3")
        show_track_id = st.checkbox("Hiển thị mã theo dõi (Track ID)", value=show_track_id, key="right_track_id_v3")
        st.markdown('<div class="control-divider"></div>', unsafe_allow_html=True)
        render_html(f'<div class="stat-row"><span>Kích thước suy luận</span><span class="v">{imgsz_option} × {imgsz_option}</span></div>')

        task = st.session_state.get("video_detection_task")
        task_running = bool(task and not task.get("done"))
        task_waiting_commit = bool(task and task.get("done") and task.get("result") and not task.get("applied"))
        if input_video_path and not task_running and not task_waiting_commit:
            if st.button("▶  Bắt đầu xử lý", type="primary", use_container_width=True, key="dashboard_start_detection"):
                if not model_exists:
                    st.error(f"Không tìm thấy model: {model_path}")
                else:
                    try:
                        with st.spinner("Đang nạp model YOLOv8s..."):
                            model_obj = load_yolo_model(model_path)
                        run_config = {
                            "file_name": source_name, "video": video_info,
                            "imgsz": imgsz_option, "conf_helmet": conf_helmet,
                            "conf_no_helmet": conf_no_helmet, "conf_bike": conf_bike,
                            "ema_alpha": ema_alpha, "bbox_alpha": BBOX_EMA_ALPHA,
                            "min_track_frames": min_track_frames,
                            "filter_on_bike": filter_on_bike, "show_track_id": show_track_id,
                        }
                        task = {
                            "done": False, "applied": False, "status": "running",
                            "error": None, "result": None, "elapsed": 0.0,
                            "source_video_path": input_video_path,"history_saved": False,
                            "pause_event": threading.Event(),
                            "only_violation_logs": bool(st.session_state.get("recent_only_violations", False)),
                            "progress": {"fraction": 0.0, "frame": 0, "total": int(video_info.get("frames", 0)), "status": "running"},
                            "run_config": run_config,
                        }
                        st.session_state.video_detection_task = task
                        run_args = (
                            input_video_path, model_path, imgsz_option,
                            conf_helmet, conf_no_helmet, conf_bike, ema_alpha,
                            min_track_frames, filter_on_bike, show_track_id,
                            show_helmet, show_no_helmet, show_motorbike,
                        )
                        worker = threading.Thread(
                            target=_video_detection_worker,
                            args=(task, model_obj, run_args),
                            daemon=True,
                            name="AI-Traffic-Vision-Detection",
                        )
                        worker.start()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Không thể khởi chạy xử lý: {exc}")
        elif task_running:
            st.info("Video đang được xử lý. Tiến trình và nút tạm dừng nằm bên dưới khung xem trước.")
        elif task_waiting_commit:
            st.info("Đang cập nhật kết quả xử lý...")
        elif not input_video_path:
            st.button("▶  Bắt đầu xử lý", disabled=True, use_container_width=True, key="dashboard_disabled_start")
        elif task and task.get("done") and task.get("error"):
            st.error("Lần xử lý gần nhất gặp lỗi. Hãy kiểm tra video/model rồi chạy lại.")
        else:
            st.button("✓  Xử lý hoàn tất", disabled=True, use_container_width=True, key="dashboard_done_button")

# ------------------------ FOUR SMART KPI CARDS ------------------------
stats = st.session_state.stats or {"motorbike": 0, "helmet": 0, "no_helmet": 0, "avg_conf": 0.0}
elapsed = (st.session_state.run_config or {}).get("elapsed", 0.0)
fps_value = (video_info.get("frames", 0) / elapsed) if video_info and elapsed > 0 else 0.0

total_heads = stats.get("helmet", 0) + stats.get("no_helmet", 0)
helmet_rate_pct = (stats.get("helmet", 0) / total_heads * 100.0) if total_heads > 0 else 0.0
viol_rate_pct = (stats.get("no_helmet", 0) / total_heads * 100.0) if total_heads > 0 else 0.0

bike_sub = f"Độ tin cậy TB: {stats.get('avg_conf', 0.0):.1f}%" if stats.get("avg_conf", 0.0) > 0 else "Phương tiện theo dõi"
helmet_sub = f"Tỷ lệ tuân thủ: {helmet_rate_pct:.1f}%" if total_heads > 0 else "Chưa có dữ liệu"
viol_sub = f"Tỷ lệ vi phạm: {viol_rate_pct:.1f}%" if total_heads > 0 else "Chưa ghi nhận vi phạm"
fps_sub = device_label

smart_kpis = [
    ("TỔNG XE MÁY", f"{stats['motorbike']}", "", bike_sub, icon("bike", 18), "#2563eb", "#eff6ff", "#2563eb", ""),
    ("ĐỘI MŨ BẢO HIỂM", f"{stats['helmet']}", "", helmet_sub, icon("shield", 18), "#0d9488", "#f0fdfa", "#0d9488", ""),
    ("KHÔNG ĐỘI MŨ (VI PHẠM)", f"{stats['no_helmet']}", "", viol_sub, icon("alert", 18), "#ef4444", "#fef2f2", "#dc2626", "danger"),
    ("TỐC ĐỘ XỬ LÝ", f"{fps_value:.1f}", "FPS", fps_sub, icon("bolt", 18), "#10b981", "#ecfdf5", "#64748b", ""),
]

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
if st.session_state.stats is None:
    smart_kpis = [
        (t, "—", "", s, i, a, tn, sc, ex)
        for (t, _v, _u, s, i, a, tn, sc, ex) in smart_kpis
    ]
kpi_cols = st.columns(4, gap="medium")
for col, (title_txt, val_txt, unit_txt, sub_txt, ico_txt, accent_c, tint_c, sub_c, extra_cls) in zip(kpi_cols, smart_kpis):
    unit_html = f"<small>{unit_txt}</small>" if unit_txt else ""
    with col:
        render_html(
            f"""
            <div class="smart-kpi {extra_cls}" style="--accent:{accent_c};--tint:{tint_c};--sub-color:{sub_c}">
              <div class="sk-top">
                <span class="sk-title">{title_txt}</span>
                <span class="sk-ico">{ico_txt}</span>
              </div>
              <div class="sk-val">{val_txt}{unit_html}</div>
              <div class="sk-sub">{sub_txt}</div>
            </div>
            """
        )

# ------------------------ LOWER ANALYTICS ROW ------------------------
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
stats = st.session_state.stats
with st.container():
    with panel("timeline_panel"):
        render_html(
            '<div class="card-title"><span>Thống kê theo thời gian</span>'
            '<span class="chart-legend"><span><i style="background:#2dd4bf"></i>Đội mũ</span>'
            '<span><i style="background:#fb7185"></i>Không đội mũ</span></span></div>'
        )
        if stats and stats.get("timeline"):
            times = sorted(stats["timeline"].keys())
            chart_df = pd.DataFrame({
                "Thời gian (giây)": times,
                "Đội mũ": [stats.get("helmet_timeline", {}).get(t, 0) for t in times],
                "Không đội mũ": [stats["timeline"].get(t, 0) for t in times],
            })
            chart_col, donut_col = st.columns([1.95, 1], gap="small")
            with chart_col:
                if alt is not None:
                    long_df = chart_df.melt("Thời gian (giây)", var_name="Loại", value_name="Số lượng")
                    color_scale = alt.Scale(domain=["Đội mũ", "Không đội mũ"], range=["#2dd4bf", "#fb7185"])
                    base = alt.Chart(long_df).encode(
                        x=alt.X("Thời gian (giây):Q", title="Giây", axis=alt.Axis(tickMinStep=1, format="d")),
                        y=alt.Y("Số lượng:Q", title="Số người", axis=alt.Axis(tickMinStep=1, format="d")),
                        color=alt.Color("Loại:N", scale=color_scale, legend=None),
                        tooltip=["Thời gian (giây)", "Loại", "Số lượng"],
                    )
                    area = base.mark_area(opacity=0.14)
                    line = base.mark_line(point=alt.OverlayMarkDef(size=36, filled=True), strokeWidth=2.5)
                    show_altair((area + line).properties(height=210))
                else:
                    st.line_chart(chart_df.set_index("Thời gian (giây)"))

            with donut_col:
                h_cnt = int(stats.get("helmet", 0))
                nh_cnt = int(stats.get("no_helmet", 0))
                tot_cnt = h_cnt + nh_cnt
                rate_val = (h_cnt / tot_cnt * 100.0) if tot_cnt > 0 else 0.0
                if alt is not None and tot_cnt > 0:
                    donut_df = pd.DataFrame({
                        "Nhóm": ["Đội mũ", "Không đội mũ"],
                        "Số lượng": [h_cnt, nh_cnt],
                    })
                    d_scale = alt.Scale(domain=["Đội mũ", "Không đội mũ"], range=["#2dd4bf", "#fb7185"])
                    arc = alt.Chart(donut_df).mark_arc(
                        innerRadius=44, outerRadius=68, stroke="#101d2d", strokeWidth=3
                    ).encode(
                        theta=alt.Theta("Số lượng:Q"),
                        color=alt.Color("Nhóm:N", scale=d_scale, legend=None),
                        tooltip=["Nhóm", "Số lượng"],
                    )
                    center_txt = alt.Chart(pd.DataFrame({"t": [f"{rate_val:.1f}%"]})).mark_text(
                        size=16, fontWeight="bold", color="#eafaff"
                    ).encode(text="t:N")
                    show_altair((arc + center_txt).properties(height=165))
                    render_html(
                        f'<div style="text-align:center;margin-top:-6px">'
                        f'<div class="donut-cap">Tỷ lệ tuân thủ</div>'
                        f'<div class="donut-pct">{rate_val:.1f}%</div></div>'
                    )
                else:
                    render_html(
                        '<div class="donut-box" style="min-height:200px;display:flex;flex-direction:column;justify-content:center">'
                        '<div class="donut-cap">Tỷ lệ tuân thủ</div>'
                        '<div class="donut-pct">—</div></div>'
                    )
        else:
            render_html('<div class="empty" style="min-height:210px"><b>Chưa có dữ liệu thống kê</b>Biểu đồ diễn biến và tỷ lệ tuân thủ sẽ xuất hiện sau khi xử lý video.</div>')


if stats:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    with st.expander("Xem bảng dữ liệu chi tiết", expanded=False):
        detail_col, summary_col = st.columns([1.25, 1], gap="large")
        with detail_col:
            times = sorted(stats.get("timeline", {}).keys())
            detail_df = pd.DataFrame({
                "Thời gian (giây)": times,
                "Đội mũ": [stats.get("helmet_timeline", {}).get(t, 0) for t in times],
                "Không đội mũ": [stats["timeline"].get(t, 0) for t in times],
            })
            show_table(detail_df)
        with summary_col:
            summary_df = pd.DataFrame({
                "Chỉ số": ["Xe máy", "Đội mũ", "Không đội mũ", "Mã đầu theo dõi", "Mã theo dõi chưa đủ bằng chứng", "Độ tin cậy trung bình"],
                "Giá trị": [stats["motorbike"], stats["helmet"], stats["no_helmet"], stats.get("head_track_ids", 0), stats.get("uncertain_head_tracks", 0), f'{stats["avg_conf"]:.1f}%'],
            })
            show_table(summary_df)

# ------------------------ EXPORT BAR ------------------------
with panel("export_bar"):
    exp_a, exp_b, exp_c = st.columns([1.0, 1.0, 2.2], gap="medium")
    video_ready = bool(st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path))
    with exp_a:
        if video_ready:
            with open(st.session_state.processed_video_path, "rb") as f:
                st.download_button("⬇  Xuất MP4", data=f, file_name="helmet_detected_output.mp4", mime="video/mp4", use_container_width=True, key="download_video_reference")
        else:
            st.button("⬇  Xuất MP4", disabled=True, use_container_width=True, key="download_video_disabled")
    with exp_b:
        if stats:
            times = sorted(stats.get("timeline", {}).keys())
            export_df = pd.DataFrame({
                "Thời gian (giây)": times,
                "Đội mũ": [stats.get("helmet_timeline", {}).get(t, 0) for t in times],
                "Không đội mũ": [stats["timeline"].get(t, 0) for t in times],
            })
            st.download_button("⇩  Xuất CSV", data=export_df.to_csv(index=False).encode("utf-8-sig"), file_name="Traffic_Analysis.csv", mime="text/csv", use_container_width=True, key="download_csv_reference")
        else:
            st.button("⇩  Xuất CSV", disabled=True, use_container_width=True, key="download_csv_disabled")
    with exp_c:
        render_html('<div class="atv-footer" style="border:0;margin:0;padding:10px 2px;justify-content:flex-end"><span><b>AI Traffic Vision v1.0</b>&nbsp;&nbsp;|&nbsp;&nbsp;Powered by YOLOv8s + ByteTrack</span></div>')
