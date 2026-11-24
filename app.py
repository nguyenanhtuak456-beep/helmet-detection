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
