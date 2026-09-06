from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import queue
import json
import site
import sys
import time
import threading
import ctypes
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QTouchEvent, QWheelEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

FRAME_W = 1920
FRAME_H = 1080
UYVY_FRAME_BYTES = FRAME_W * FRAME_H * 2
INPUT_MODE_QUERY_DEFAULT = "1080i59.94"
OUTPUT_MODE_QUERY_DEFAULT = "1080i59.94"
WINDOWED_PREVIEW_MAX_W = 640
WINDOWED_PREVIEW_MAX_H = 360
FULLSCREEN_PREVIEW_MAX_W = 1280
FULLSCREEN_PREVIEW_MAX_H = 720
PREVIEW_DOWNSAMPLE_LABEL_TO_FACTOR = {
    "Full (1:1)": 1.0,
    "Half (1/2)": 0.5,
    "Quarter (1/4)": 0.25,
}
PREVIEW_BT709_ACCURATE = os.environ.get("VP_PREVIEW_BT709_ACCURATE", "0") == "1"

SR_FLAVOR_LABEL_TO_NAME = {
    "Bilinear (Fast)": "bilinear",
    "Bilinear + Edge Boost (Realtime)": "bilinear_sharp",
    "Bicubic (Balanced)": "bicubic",
    "Bicubic + Sharpen (Crisp)": "bicubic_sharpen",
}
SR_FLAVOR_NAME_TO_LABEL = {value: key for key, value in SR_FLAVOR_LABEL_TO_NAME.items()}

DEINTERLACE_METHOD_LABEL_TO_NAME = {
    "Bob (Fast)": "bob",
    "Blend (Stable)": "blend",
    "Edge Adaptive (Field Aware)": "edge_adaptive",
    "Motion Adaptive (Broadcast: Deinterlace->Scale->Interlace)": "edge_adaptive",
}
DEINTERLACE_METHOD_NAME_TO_LABEL = {value: key for key, value in DEINTERLACE_METHOD_LABEL_TO_NAME.items()}

INTERLACED_DEFAULT_DEINTERLACE_METHOD = "edge_adaptive"
PROGRESSIVE_DEFAULT_DEINTERLACE_METHOD = "bob"

DENOISE_METHOD_LABEL_TO_NAME = {
    "Off": "off",
    "Luma Gaussian 3x3 (Balanced)": "luma_gaussian3x3",
    "Luma Median 3x3 (Stronger)": "luma_median3x3",
    "Luma Bilateral 3x3 (Artifact Cleaner)": "luma_bilateral3x3",
    "Luma Bilateral 5x5 (Still Image Heavy)": "luma_bilateral5x5",
    "Field Temporal Luma (Advanced)": "field_temporal_luma",
}
DENOISE_METHOD_NAME_TO_LABEL = {value: key for key, value in DENOISE_METHOD_LABEL_TO_NAME.items()}

AI_SR_POST_DENOISE_LABEL_TO_NAME = {
    "Off": "off",
    "Luma Gaussian 3x3": "luma_gaussian3x3",
    "Luma Median 3x3": "luma_median3x3",
    "Luma Bilateral 3x3": "luma_bilateral3x3",
    "Luma Bilateral 5x5": "luma_bilateral5x5",
}
AI_SR_POST_DENOISE_NAME_TO_LABEL = {value: key for key, value in AI_SR_POST_DENOISE_LABEL_TO_NAME.items()}

AI_SR_POST_ARTIFACT_REDUCTION_LABEL_TO_NAME = {
    "Off": "off",
    "Luma Bilateral 3x3": "luma_bilateral3x3",
    "Luma Bilateral 5x5": "luma_bilateral5x5",
}
AI_SR_POST_ARTIFACT_REDUCTION_NAME_TO_LABEL = {
    value: key for key, value in AI_SR_POST_ARTIFACT_REDUCTION_LABEL_TO_NAME.items()
}

RTX_POST_SCALE_METHOD_LABEL_TO_NAME = {
    "Nearest (Pixelated)": "nearest",
    "Bilinear (Fast)": "bilinear",
    "Bicubic (Balanced)": "bicubic",
    "Lanczos (Sharp)": "lanczos",
}
RTX_POST_SCALE_METHOD_NAME_TO_LABEL = {value: key for key, value in RTX_POST_SCALE_METHOD_LABEL_TO_NAME.items()}

COLOR_SPACE_LABEL_TO_NAME = {
    "Rec.709 (SDR)": "rec709",
    "Rec.2020 HLG (HDR)": "rec2020_hlg",
}
COLOR_SPACE_NAME_TO_LABEL = {value: key for key, value in COLOR_SPACE_LABEL_TO_NAME.items()}

COLOR_RANGE_LABEL_TO_NAME = {
    "Limited (Video)": "limited",
    "Full (Data)": "full",
}
COLOR_RANGE_NAME_TO_LABEL = {value: key for key, value in COLOR_RANGE_LABEL_TO_NAME.items()}

WORKER_PRIORITY_LABEL_TO_NAME = {
    "Normal": "normal",
    "Above Normal": "above_normal",
    "High": "high",
}
WORKER_PRIORITY_NAME_TO_LABEL = {value: key for key, value in WORKER_PRIORITY_LABEL_TO_NAME.items()}

INTERLACED_FIELD2_PHASE_MIN = -1.0
INTERLACED_FIELD2_PHASE_MAX = 2.0


def _clamp_interlaced_field2_phase_fraction(value: float) -> float:
    return max(INTERLACED_FIELD2_PHASE_MIN, min(INTERLACED_FIELD2_PHASE_MAX, float(value)))


def _mode_name_is_interlaced(mode_label: str) -> bool:
    mode_text = str(mode_label).strip().lower()
    if not mode_text:
        return False

    mode_name = mode_text.split("(", 1)[0].strip()
    if "progressive" in mode_name or "psf" in mode_name:
        return False
    if "interlace" in mode_name:
        return True
    return ("i" in mode_name) and any(ch.isdigit() for ch in mode_name)


def _decklink_timecode_format_name(format_code: object) -> str:
    try:
        code = int(format_code) & 0xFFFFFFFF
    except Exception:
        return ""

    if d is not None:
        format_map = {
            int(getattr(d, "TIMECODE_FORMAT_RP188_VITC1", 0)) & 0xFFFFFFFF: "RP188 VITC1",
            int(getattr(d, "TIMECODE_FORMAT_RP188_VITC2", 0)) & 0xFFFFFFFF: "RP188 VITC2",
            int(getattr(d, "TIMECODE_FORMAT_RP188_LTC", 0)) & 0xFFFFFFFF: "RP188 LTC",
            int(getattr(d, "TIMECODE_FORMAT_RP188_HIGH_FRAME_RATE", 0)) & 0xFFFFFFFF: "RP188 HFRTC",
            int(getattr(d, "TIMECODE_FORMAT_RP188_ANY", 0)) & 0xFFFFFFFF: "RP188 Any",
            int(getattr(d, "TIMECODE_FORMAT_VITC", 0)) & 0xFFFFFFFF: "VITC",
            int(getattr(d, "TIMECODE_FORMAT_VITC_FIELD2", 0)) & 0xFFFFFFFF: "VITC Field 2",
            int(getattr(d, "TIMECODE_FORMAT_SERIAL", 0)) & 0xFFFFFFFF: "Serial",
        }
        if code in format_map:
            return format_map[code]

    fallback_map = {
        0x72707631: "RP188 VITC1",
        0x72703132: "RP188 VITC2",
        0x72706C74: "RP188 LTC",
        0x72706872: "RP188 HFRTC",
        0x72703138: "RP188 Any",
        0x76697463: "VITC",
        0x76697432: "VITC Field 2",
        0x73657269: "Serial",
    }
    return fallback_map.get(code, f"0x{code:08X}")


def _extract_decklink_frame_timecode_info(frame: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "present": False,
        "text": "",
        "format_code": 0,
        "format_name": "",
    }
    if frame is None:
        return payload

    try:
        has_timecode = bool(getattr(frame, "has_timecode", False))
        timecode_text = ""
        format_code = 0
        if has_timecode:
            getter = getattr(frame, "get_timecode", None)
            raw_text = getter() if callable(getter) else getattr(frame, "timecode", "")
            timecode_text = "" if raw_text is None else str(raw_text).strip()
            format_code = int(getattr(frame, "timecode_format", 0))
        elif bool(getattr(frame, "has_atc_timecode", False)):
            getter = getattr(frame, "get_atc_timecode", None)
            raw_text = getter() if callable(getter) else getattr(frame, "atc_timecode", "")
            timecode_text = "" if raw_text is None else str(raw_text).strip()
            format_code = int(getattr(frame, "atc_timecode_format", 0))
            has_timecode = bool(timecode_text)
        if not has_timecode or not timecode_text:
            return payload

        payload["present"] = True
        payload["text"] = timecode_text
        payload["format_code"] = format_code
        payload["format_name"] = _decklink_timecode_format_name(format_code)
    except Exception:
        return payload
    return payload

try:
    import decklink_wrapper as d
except Exception:
    d = None

try:
    import cv2
except Exception:
    cv2 = None

# Running as `python gui/app.py` sets sys.path[0] to the gui folder; add project root
# so `gui.processor_worker` and sibling imports resolve consistently.
_project_root_for_imports = str(Path(__file__).resolve().parents[1])
if _project_root_for_imports not in sys.path:
    sys.path.insert(0, _project_root_for_imports)

_worker_import_error: Exception | None = None
try:
    from gui.processor_worker import run_processor_worker
except Exception as exc_gui_import:
    try:
        from processor_worker import run_processor_worker
    except Exception as exc_local_import:
        run_processor_worker = None
        _worker_import_error = exc_local_import
    else:
        _worker_import_error = None
else:
    _worker_import_error = None


_CV2_RGB_RING: list[np.ndarray] = []
_CV2_RGB_RING_INDEX = 0


def _uyvy_to_rgb_bt709_limited(yuv422: np.ndarray, dst: np.ndarray | None = None) -> np.ndarray:
    return _uyvy_to_rgb_limited(yuv422, "rec709", dst=dst)


def _normalize_color_space_name(color_space: str) -> str:
    normalized = str(color_space).strip().lower().replace(" ", "").replace("-", "_")
    if normalized in {"rec709", "rec_709", "bt709"}:
        return "rec709"
    if normalized in {"rec2020_hlg", "rec2020hlg", "bt2020_hlg", "bt2020hlg"}:
        return "rec2020_hlg"
    return "rec709"


def _normalize_color_range_name(color_range: str) -> str:
    normalized = str(color_range).strip().lower()
    if normalized in {"full", "data", "pc"}:
        return "full"
    return "limited"


def _clamp_ai_inference_fps(value: int) -> int:
    return max(1, min(60, int(value)))


def _legacy_ai_frame_interval_to_fps(interval_frames: int) -> int:
    # Legacy configs used "frame_interval" (run every N frames). New runtime
    # uses explicit target inference FPS. Map old defaults to a practical rate.
    interval = max(1, int(interval_frames))
    return _clamp_ai_inference_fps(int(round(30.0 / float(interval))))


def _normalize_worker_priority_name(priority_name: str) -> str:
    normalized = str(priority_name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"normal", "default"}:
        return "normal"
    if normalized in {"above_normal", "abovenormal", "high_normal", "highnormal"}:
        return "above_normal"
    if normalized in {"high", "high_priority", "highpriority"}:
        return "high"
    return "above_normal"


def _uyvy_to_rgb_limited(
    yuv422: np.ndarray,
    color_space: str,
    color_range: str = "limited",
    dst: np.ndarray | None = None,
) -> np.ndarray:
    if yuv422.ndim != 3 or yuv422.shape[2] != 2:
        raise ValueError(f"Expected UYVY array shape [H, W, 2], got {tuple(yuv422.shape)}")

    h, w, _ = yuv422.shape
    if (w & 1) != 0:
        raise ValueError(f"UYVY width must be even, got {w}")

    if dst is None:
        rgb = np.empty((h, w, 3), dtype=np.uint8)
    else:
        if dst.shape != (h, w, 3) or dst.dtype != np.uint8:
            raise ValueError("Destination RGB buffer must be uint8 with shape [H, W, 3]")
        rgb = dst

    packed = yuv422.reshape(h, w // 2, 4)
    u = packed[:, :, 0].astype(np.float32)
    y0 = packed[:, :, 1].astype(np.float32)
    v = packed[:, :, 2].astype(np.float32)
    y1 = packed[:, :, 3].astype(np.float32)

    cs = _normalize_color_space_name(color_space)
    cr = _normalize_color_range_name(color_range)
    d = u - 128.0
    e = v - 128.0

    if cr == "full":
        c0 = y0
        c1 = y1
        if cs == "rec2020_hlg":
            r0 = np.clip(c0 + 1.474600 * e, 0.0, 255.0).astype(np.uint8)
            g0 = np.clip(c0 - 0.164553 * d - 0.571353 * e, 0.0, 255.0).astype(np.uint8)
            b0 = np.clip(c0 + 1.881400 * d, 0.0, 255.0).astype(np.uint8)

            r1 = np.clip(c1 + 1.474600 * e, 0.0, 255.0).astype(np.uint8)
            g1 = np.clip(c1 - 0.164553 * d - 0.571353 * e, 0.0, 255.0).astype(np.uint8)
            b1 = np.clip(c1 + 1.881400 * d, 0.0, 255.0).astype(np.uint8)
        else:
            r0 = np.clip(c0 + 1.574800 * e, 0.0, 255.0).astype(np.uint8)
            g0 = np.clip(c0 - 0.187324 * d - 0.468124 * e, 0.0, 255.0).astype(np.uint8)
            b0 = np.clip(c0 + 1.855600 * d, 0.0, 255.0).astype(np.uint8)

            r1 = np.clip(c1 + 1.574800 * e, 0.0, 255.0).astype(np.uint8)
            g1 = np.clip(c1 - 0.187324 * d - 0.468124 * e, 0.0, 255.0).astype(np.uint8)
            b1 = np.clip(c1 + 1.855600 * d, 0.0, 255.0).astype(np.uint8)
    else:
        c0 = y0 - 16.0
        c1 = y1 - 16.0
        if cs == "rec2020_hlg":
            r0 = np.clip(1.164383 * c0 + 1.678674 * e, 0.0, 255.0).astype(np.uint8)
            g0 = np.clip(1.164383 * c0 - 0.187326 * d - 0.650424 * e, 0.0, 255.0).astype(np.uint8)
            b0 = np.clip(1.164383 * c0 + 2.141772 * d, 0.0, 255.0).astype(np.uint8)

            r1 = np.clip(1.164383 * c1 + 1.678674 * e, 0.0, 255.0).astype(np.uint8)
            g1 = np.clip(1.164383 * c1 - 0.187326 * d - 0.650424 * e, 0.0, 255.0).astype(np.uint8)
            b1 = np.clip(1.164383 * c1 + 2.141772 * d, 0.0, 255.0).astype(np.uint8)
        else:
            r0 = np.clip(1.164383 * c0 + 1.792741 * e, 0.0, 255.0).astype(np.uint8)
            g0 = np.clip(1.164383 * c0 - 0.213249 * d - 0.532909 * e, 0.0, 255.0).astype(np.uint8)
            b0 = np.clip(1.164383 * c0 + 2.112402 * d, 0.0, 255.0).astype(np.uint8)

            r1 = np.clip(1.164383 * c1 + 1.792741 * e, 0.0, 255.0).astype(np.uint8)
            g1 = np.clip(1.164383 * c1 - 0.213249 * d - 0.532909 * e, 0.0, 255.0).astype(np.uint8)
            b1 = np.clip(1.164383 * c1 + 2.112402 * d, 0.0, 255.0).astype(np.uint8)

    rgb[:, 0::2, 0] = r0
    rgb[:, 0::2, 1] = g0
    rgb[:, 0::2, 2] = b0
    rgb[:, 1::2, 0] = r1
    rgb[:, 1::2, 1] = g1
    rgb[:, 1::2, 2] = b1
    return rgb


class SafeRotatingFileHandler(RotatingFileHandler):
    def __init__(self, *args, rollover_retry_interval_s: float = 15.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rollover_retry_interval_s = max(1.0, float(rollover_retry_interval_s))
        self._skip_rollover_until = 0.0

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        if time.monotonic() < self._skip_rollover_until:
            return False
        return super().shouldRollover(record)

    def doRollover(self) -> None:
        try:
            super().doRollover()
            self._skip_rollover_until = 0.0
            return
        except PermissionError:
            # On Windows another process may hold app.log open. Keep logging to
            # the current file and retry rollover after a cooldown.
            self._skip_rollover_until = time.monotonic() + self._rollover_retry_interval_s

        if self.stream:
            try:
                self.stream.flush()
                self.stream.close()
            except Exception:
                pass
        self.stream = self._open()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("video_processor_gui")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = SafeRotatingFileHandler(
        log_dir / "app.log",
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
        rollover_retry_interval_s=15.0,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
    )
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(stream_handler)
    return logger


LOGGER = setup_logger()

_OUTPUT_SCHEDULE_STATE: dict[int, dict[str, object]] = {}
_RPC_E_CHANGED_MODE_HEX = "0x80010106"

_ROI_TELEMETRY_SLOT_COUNT = 16
_ROI_TM_ACTIVE = 0
_ROI_TM_FRAME_PROGRESS = 1
_ROI_TM_TOTAL_FRAMES = 2
_ROI_TM_INTERP_MODE_CODE = 3
_ROI_TM_APPLIED_X = 4
_ROI_TM_APPLIED_Y = 5
_ROI_TM_APPLIED_W = 6
_ROI_TM_APPLIED_H = 7
_ROI_TM_START_X = 8
_ROI_TM_START_Y = 9
_ROI_TM_START_W = 10
_ROI_TM_START_H = 11
_ROI_TM_TARGET_X = 12
_ROI_TM_TARGET_Y = 13
_ROI_TM_TARGET_W = 14
_ROI_TM_TARGET_H = 15


def initialize_com_for_decklink() -> None:
    if sys.platform != "win32":
        return

    try:
        # decklink_wrapper expects MTA on this machine; using STA triggers 0x80010106 changed-mode failures.
        COINIT_MULTITHREADED = 0x0
        RPC_E_CHANGED_MODE = -2147417850  # 0x80010106
        ole32 = ctypes.windll.ole32
        hr = ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
        # S_OK=0, S_FALSE=1 (already initialized on this thread with same model).
        if hr not in (0, 1):
            if hr == RPC_E_CHANGED_MODE:
                LOGGER.info("CoInitializeEx already set by Qt (hr=0x%08X)", hr & 0xFFFFFFFF)
            else:
                LOGGER.warning("CoInitializeEx returned hr=0x%08X", hr & 0xFFFFFFFF)
        else:
            LOGGER.info("COM initialized for DeckLink (hr=0x%08X)", hr & 0xFFFFFFFF)
    except Exception:
        LOGGER.exception("Failed to initialize COM for DeckLink")


def _is_changed_mode_error(exc: Exception) -> bool:
    return _RPC_E_CHANGED_MODE_HEX in str(exc)


def _call_decklink_api_in_mta_thread(api_name: str, *args: object) -> object:
    if d is None:
        raise RuntimeError("decklink_wrapper is not available")

    result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

    def _worker() -> None:
        coinitialized = False
        try:
            if sys.platform == "win32":
                hr = ctypes.windll.ole32.CoInitializeEx(None, 0x0)  # COINIT_MULTITHREADED
                # S_OK=0, S_FALSE=1.
                coinitialized = hr in (0, 1)

            result = getattr(d, api_name)(*args)
            result_queue.put(("ok", result))
        except Exception as worker_exc:
            result_queue.put(("err", worker_exc))
        finally:
            if sys.platform == "win32" and coinitialized:
                ctypes.windll.ole32.CoUninitialize()

    worker = threading.Thread(target=_worker, name=f"decklink-mta-{api_name}", daemon=True)
    worker.start()
    worker.join(timeout=10.0)

    if worker.is_alive():
        raise TimeoutError(f"DeckLink API call timed out in MTA thread: {api_name}")

    status, payload = result_queue.get()
    if status == "err":
        raise payload  # type: ignore[misc]
    return payload


def _call_decklink_api(api_name: str, *args: object) -> object:
    if d is None:
        raise RuntimeError("decklink_wrapper is not available")

    if sys.platform == "win32":
        # Qt typically initializes the GUI thread in a different COM apartment
        # than the DeckLink wrapper expects. Route catalog/mode queries through
        # a short-lived MTA thread instead of probing the GUI thread first.
        return _call_decklink_api_in_mta_thread(api_name, *args)

    api = getattr(d, api_name)
    try:
        return api(*args)
    except Exception as exc:
        if sys.platform == "win32" and _is_changed_mode_error(exc):
            LOGGER.info(
                "DeckLink API %s hit COM changed-mode on GUI thread; retrying in MTA worker thread",
                api_name,
            )
            return _call_decklink_api_in_mta_thread(api_name, *args)
        raise


@dataclass
class Roi:
    x: int
    y: int
    w: int
    h: int


@dataclass
class RoiKeyframe:
    roi: Roi
    duration_frames: int
    interpolation_mode: str


def clamp_roi(roi: Roi, width: int = FRAME_W, height: int = FRAME_H) -> Roi:
    # Keep ROI size stable while moving: clamp size first, then clamp position.
    max_w_frame = max(2, width)
    max_h_frame = max(2, height)

    # ROI is locked to 16:9 to match input/output display aspect.
    w = max(2, min(roi.w, max_w_frame))
    w &= ~1
    if w < 2:
        w = 2

    h = max(2, int(round(w * 9.0 / 16.0)))
    if h > max_h_frame:
        h = max_h_frame
        w = max(2, int(round(h * 16.0 / 9.0)))
        w = min(w, max_w_frame)
        w &= ~1
        if w < 2:
            w = 2
        h = max(2, int(round(w * 9.0 / 16.0)))
        if h > max_h_frame:
            h = max_h_frame

    max_x = max(0, width - w)
    max_y = max(0, height - h)
    x = max(0, min(roi.x, max_x))
    y = max(0, min(roi.y, max_y))

    x &= ~1
    if x > max_x:
        x = max(0, max_x & ~1)

    return Roi(x, y, w, h)


def roi_scale_from_roi(roi: Roi) -> float:
    rw = FRAME_W / max(1, roi.w)
    rh = FRAME_H / max(1, roi.h)
    return max(rw, rh)


def roi_from_scale(scale: float, center_x: float, center_y: float) -> Roi:
    if scale < 1.0:
        scale = 1.0
    w = int(FRAME_W / scale)
    h = int(FRAME_H / scale)
    w = max(2, w & ~1)
    h = max(2, h)
    x = int(round(center_x - (w / 2)))
    y = int(round(center_y - (h / 2)))
    return clamp_roi(Roi(x, y, w, h))


def _downsample_uyvy422_safe(yuv422: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    if cv2 is None:
        return yuv422

    src_h, src_w, _ = yuv422.shape
    out_w = max(2, min(int(target_w), src_w))
    out_h = max(1, min(int(target_h), src_h))
    if (out_w & 1) != 0:
        out_w -= 1
    if out_w < 2:
        out_w = 2

    if out_w == src_w and out_h == src_h:
        return yuv422

    packed = yuv422.reshape(src_h, src_w // 2, 4)
    y_plane = np.empty((src_h, src_w), dtype=np.uint8)
    y_plane[:, 0::2] = packed[:, :, 1]
    y_plane[:, 1::2] = packed[:, :, 3]
    u_plane = packed[:, :, 0]
    v_plane = packed[:, :, 2]

    out_y = cv2.resize(y_plane, (out_w, out_h), interpolation=cv2.INTER_AREA)
    out_u = cv2.resize(u_plane, (out_w // 2, out_h), interpolation=cv2.INTER_AREA)
    out_v = cv2.resize(v_plane, (out_w // 2, out_h), interpolation=cv2.INTER_AREA)

    out_packed = np.empty((out_h, out_w // 2, 4), dtype=np.uint8)
    out_packed[:, :, 0] = out_u
    out_packed[:, :, 1] = out_y[:, 0::2]
    out_packed[:, :, 2] = out_v
    out_packed[:, :, 3] = out_y[:, 1::2]
    return out_packed.reshape(out_h, out_w, 2)


def uyvy_to_qimage(
    frame_bytes: bytes,
    preview_max_w: int | None = None,
    preview_max_h: int | None = None,
    color_space: str = "rec709",
    color_range: str = "limited",
) -> tuple[QImage, np.ndarray | None]:
    if len(frame_bytes) != UYVY_FRAME_BYTES:
        raise ValueError("Invalid UYVY frame byte length.")

    global _CV2_RGB_RING_INDEX
    if not _CV2_RGB_RING:
        # Two buffers avoid input/output previews aliasing each other within one tick.
        _CV2_RGB_RING.extend(
            [
                np.empty((FRAME_H, FRAME_W, 3), dtype=np.uint8),
                np.empty((FRAME_H, FRAME_W, 3), dtype=np.uint8),
            ]
        )

    yuv422 = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 2)

    work_yuv = yuv422
    work_h = FRAME_H
    work_w = FRAME_W

    if (
        cv2 is not None
        and preview_max_w is not None
        and preview_max_h is not None
    ):
        target_w = max(1, min(int(preview_max_w), FRAME_W))
        target_h = max(1, min(int(preview_max_h), FRAME_H))
        if target_w < FRAME_W or target_h < FRAME_H:
            work_yuv = _downsample_uyvy422_safe(yuv422, target_w, target_h)
            work_h, work_w = int(work_yuv.shape[0]), int(work_yuv.shape[1])

    if work_h == FRAME_H and work_w == FRAME_W:
        rgb = _CV2_RGB_RING[_CV2_RGB_RING_INDEX]
        _CV2_RGB_RING_INDEX = (_CV2_RGB_RING_INDEX + 1) % len(_CV2_RGB_RING)
    else:
        rgb = np.empty((work_h, work_w, 3), dtype=np.uint8)

    color_space_name = _normalize_color_space_name(color_space)

    # Keep preview interaction responsive by default using OpenCV's optimized
    # conversion path for Rec.709. Use explicit matrix conversion for other
    # color spaces so preview matches processing.
    if cv2 is not None and not PREVIEW_BT709_ACCURATE and color_space_name == "rec709" and _normalize_color_range_name(color_range) == "limited":
        cv2.cvtColor(work_yuv, cv2.COLOR_YUV2RGB_UYVY, dst=rgb)
    else:
        _uyvy_to_rgb_limited(work_yuv, color_space_name, color_range=color_range, dst=rgb)

    image = QImage(rgb.data, work_w, work_h, work_w * 3, QImage.Format_RGB888)
    if preview_max_w is not None and preview_max_h is not None:
        target_w = max(1, min(int(preview_max_w), work_w))
        target_h = max(1, min(int(preview_max_h), work_h))
        if target_w < work_w or target_h < work_h:
            return image.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.FastTransformation), None
    return image, rgb


def looks_zeroed_uyvy_frame(frame_bytes: bytes) -> bool:
    if len(frame_bytes) != UYVY_FRAME_BYTES:
        return False

    # Sample sparsely so this check remains cheap in real-time preview.
    sample = np.frombuffer(frame_bytes, dtype=np.uint8)[::4096]
    return sample.size > 0 and int(np.count_nonzero(sample)) == 0


def tight_uyvy_bytes(frame: object) -> bytes:
    row_bytes = int(frame.row_bytes)
    expected_row_bytes = FRAME_W * 2
    if row_bytes < expected_row_bytes:
        raise RuntimeError(f"Captured row_bytes {row_bytes} is smaller than expected {expected_row_bytes}")

    raw = memoryview(frame)
    if row_bytes == expected_row_bytes:
        return raw.tobytes()

    raw_np = np.frombuffer(raw, dtype=np.uint8)
    expected_total = row_bytes * FRAME_H
    if raw_np.size < expected_total:
        raise RuntimeError(f"Captured frame buffer is smaller than expected ({raw_np.size} < {expected_total})")
    return raw_np[:expected_total].reshape(FRAME_H, row_bytes)[:, :expected_row_bytes].tobytes()


def write_frame_to_output(out: object, frame_bytes: bytes) -> None:
    expected_row_bytes = FRAME_W * 2
    if out.row_bytes < expected_row_bytes:
        raise RuntimeError(f"Output row_bytes {out.row_bytes} is smaller than expected {expected_row_bytes}")

    if out.row_bytes == expected_row_bytes:
        payload = frame_bytes
    else:
        src = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(FRAME_H, expected_row_bytes)
        padded = np.zeros((FRAME_H, out.row_bytes), dtype=np.uint8)
        padded[:, :expected_row_bytes] = src
        payload = padded.tobytes()

    out_id = id(out)
    state = _OUTPUT_SCHEDULE_STATE.get(out_id)
    if state is None:
        schedule_fn = getattr(out, "schedule_frame_copy", None)
        start_fn = getattr(out, "start_scheduled_playback", None)
        buffered_fn = getattr(out, "buffered_video_frame_count", None)
        state = {
            "enabled": callable(schedule_fn) and callable(start_fn),
            "can_query_buffered": callable(buffered_fn),
            "started": False,
            "display_time": 0,
            "frame_duration": int(getattr(out, "frame_duration", 0)) if hasattr(out, "frame_duration") else 0,
            "time_scale": int(getattr(out, "time_scale", 0)) if hasattr(out, "time_scale") else 0,
        }
        _OUTPUT_SCHEDULE_STATE[out_id] = state

    if state["enabled"]:
        frame_duration = int(state["frame_duration"])
        time_scale = int(state["time_scale"])
        if frame_duration > 0 and time_scale > 0:
            try:
                out.schedule_frame_copy(
                    payload,
                    int(state["display_time"]),
                    frame_duration,
                    time_scale,
                )
                state["display_time"] = int(state["display_time"]) + frame_duration

                if not bool(state["started"]):
                    should_start = False
                    if bool(state.get("can_query_buffered", False)):
                        try:
                            buffered_count = int(out.buffered_video_frame_count())
                            # Small preroll prevents startup underflow while keeping latency low.
                            should_start = buffered_count >= 2
                        except Exception:
                            # Wrapper does not reliably expose buffered count; start without preroll.
                            state["can_query_buffered"] = False
                            should_start = True
                    else:
                        # No buffered count support in wrapper; start immediately.
                        should_start = True

                    if should_start:
                        out.start_scheduled_playback(0, time_scale, 1.0)
                        state["started"] = True
                return
            except Exception:
                LOGGER.exception("Scheduled DeckLink output failed; falling back to sync output")
                state["enabled"] = False

    out.display_frame_sync(payload)


def clear_output_schedule_state(out: object | None) -> None:
    if out is None:
        return
    _OUTPUT_SCHEDULE_STATE.pop(id(out), None)


class SyntheticUyvySource:
    def __init__(self, width: int = FRAME_W, height: int = FRAME_H) -> None:
        self.width = width
        self.height = height
        self.t = 0
        self._x = np.arange(width, dtype=np.uint16)[None, :]
        self._y = np.arange(height, dtype=np.uint16)[:, None]

    def next_frame(self) -> bytes:
        phase = self.t
        self.t = (self.t + 3) % 256

        luma = ((self._x + self._y + phase) & 0xFF).astype(np.uint8)
        u = (((self._y // 4) + 64 + phase) & 0xFF).astype(np.uint8)
        v = (((self._x // 8) + 96 + phase) & 0xFF).astype(np.uint8)

        packed = np.empty((self.height, self.width // 2, 4), dtype=np.uint8)
        packed[:, :, 0] = u[:, : self.width // 2]
        packed[:, :, 1] = luma[:, 0::2]
        packed[:, :, 2] = v[:, : self.width // 2]
        packed[:, :, 3] = luma[:, 1::2]
        return packed.tobytes()


class RoiCanvas(QWidget):
    roiChanged = Signal(int, int, int, int)
    scaleChanged = Signal(float)
    fullscreenRequested = Signal(str)

    def __init__(self, view_name: str = "input") -> None:
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.setMinimumSize(160, 90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._view_name = view_name

        self._image: QImage | None = None
        self._image_backing: np.ndarray | None = None
        self._roi = Roi(0, 0, FRAME_W, FRAME_H)
        self._visual_roi_overlay_transition: tuple[float, float, float, float] | None = None
        self._visual_roi_overlay_drag: tuple[float, float, float, float] | None = None

        self._drag_mode = "none"
        self._drag_start_pos = QPointF()
        self._drag_start_roi = self._roi

        self._last_touch_emit_ts = 0.0
        self._default_touch_emit_interval_s = 1.0 / 60.0
        drag_emit_hz = max(60.0, min(120.0, float(os.environ.get("VP_MANUAL_DRAG_EMIT_HZ", "90"))))
        self._drag_move_touch_emit_interval_s = 1.0 / drag_emit_hz
        self._touch_emit_interval_s = self._default_touch_emit_interval_s
        self._touch_emit_pending = False
        self._touch_emit_pending_scale = False
        self._smoothing_percent = 4
        self._latency_smoothing_percent = 0
        self._drag_x_hysteresis_px = max(0.10, min(1.20, float(os.environ.get("VP_ROI_DRAG_X_HYSTERESIS_PX", "0.45"))))
        self._interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        self._interaction_filtered_target_roi: Roi | None = None
        self._interaction_target_roi: Roi | None = None
        self._interaction_target_emit_scale = False
        self._interaction_interp_timer = QTimer(self)
        self._interaction_interp_timer.setInterval(16)
        self._interaction_interp_timer.timeout.connect(self._on_interaction_interp_tick)
        self._interaction_emit_flush_timer = QTimer(self)
        self._interaction_emit_flush_timer.setSingleShot(True)
        self._interaction_emit_flush_timer.timeout.connect(self._flush_interaction_emit)

    def set_image(self, image: QImage, backing: np.ndarray | None = None) -> None:
        self._image = image
        self._image_backing = backing
        self.update()

    def set_roi(self, roi: Roi) -> None:
        self._cancel_interaction_interpolation()
        self._visual_roi_overlay_transition = None
        self._visual_roi_overlay_drag = None
        self._apply_roi_local(roi)
        self._interaction_target_roi = roi

    def cancel_pending_interaction_updates(self) -> None:
        # Drop queued interaction/touch emits so old manual gestures cannot
        # override a programmatic keyframe recall endpoint.
        self._cancel_interaction_interpolation()
        self._touch_emit_pending = False
        self._touch_emit_pending_scale = False

    def set_visual_roi_overlay(self, x: float, y: float, w: float, h: float) -> None:
        self._visual_roi_overlay_transition = (float(x), float(y), float(w), float(h))
        self.update()

    def clear_visual_roi_overlay(self) -> None:
        if self._visual_roi_overlay_transition is None:
            return
        self._visual_roi_overlay_transition = None
        self.update()

    def _set_drag_visual_roi_overlay(self, x: float, y: float, w: float, h: float) -> None:
        # Keep a float-domain overlay while dragging so the ROI box tracks the
        # pointer smoothly even when controller ROI is quantized to even pixels.
        ow = max(2.0, min(float(FRAME_W), float(w)))
        oh = max(2.0, min(float(FRAME_H), float(h)))
        ox = max(0.0, min(float(FRAME_W) - ow, float(x)))
        oy = max(0.0, min(float(FRAME_H) - oh, float(y)))
        self._visual_roi_overlay_drag = (ox, oy, ow, oh)
        self.update()

    def _clear_drag_visual_roi_overlay(self) -> None:
        if self._visual_roi_overlay_drag is None:
            return
        self._visual_roi_overlay_drag = None
        self.update()

    def drag_visual_roi_overlay(self) -> tuple[float, float, float, float] | None:
        return self._visual_roi_overlay_drag

    def roi(self) -> Roi:
        return self._roi

    def set_smoothing_percent(self, value: int) -> None:
        self._smoothing_percent = max(0, min(10, int(value)))

    def set_latency_smoothing_percent(self, value: int) -> None:
        self._latency_smoothing_percent = max(0, min(100, int(value)))

    def set_drag_x_hysteresis_px(self, value: float) -> None:
        self._drag_x_hysteresis_px = max(0.10, min(1.20, float(value)))

    def _resize_handle_rect(self, roi_rect: QRectF) -> QRectF:
        roi_min_edge = max(1.0, min(roi_rect.width(), roi_rect.height()))
        # Keep handle usable while preserving a move area on small ROIs.
        handle_size = max(8.0, min(48.0, roi_min_edge * 0.45))
        if roi_min_edge > 18.0:
            handle_size = min(handle_size, roi_min_edge - 10.0)
        else:
            handle_size = min(handle_size, roi_min_edge * 0.5)
        handle_size = max(6.0, min(handle_size, roi_min_edge))
        return QRectF(
            roi_rect.right() - handle_size,
            roi_rect.bottom() - handle_size,
            handle_size,
            handle_size,
        )

    def _is_roi_near_frame_edge(self, roi: Roi, margin: int = 0) -> bool:
        m = max(0, int(margin))
        max_x = max(0, FRAME_W - roi.w)
        max_y = max(0, FRAME_H - roi.h)
        return (
            roi.x <= m
            or roi.y <= m
            or roi.x >= (max_x - m)
            or roi.y >= (max_y - m)
        )

    def _quantize_drag_axis_with_hysteresis(
        self,
        target_value: float,
        current_value: int,
        quantum: int,
        hysteresis_px: float,
    ) -> int:
        q = max(1, int(quantum))
        current = int(current_value)
        snapped = int(round(float(target_value) / float(q))) * q
        if snapped == current:
            return current

        if snapped > current:
            # Require crossing most of the next quantized step before advancing.
            threshold = float(current + q) - float(hysteresis_px)
            if float(target_value) < threshold:
                return current
        else:
            threshold = float(current - q) + float(hysteresis_px)
            if float(target_value) > threshold:
                return current
        return snapped

    def paintEvent(self, event) -> None:
        del event
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)

        image_rect = self._image_rect()
        if self._image is not None:
            p.drawImage(image_rect, self._image)

        overlay = self._visual_roi_overlay_drag
        if overlay is None:
            overlay = self._visual_roi_overlay_transition

        if overlay is not None:
            overlay_x, overlay_y, overlay_w, overlay_h = overlay
            roi_rect_w = self._frame_to_widget_rect_float(overlay_x, overlay_y, overlay_w, overlay_h)
        else:
            roi_rect_w = self._frame_to_widget_rect(self._roi)
        p.setRenderHint(QPainter.Antialiasing, True)

        p.setPen(QPen(Qt.yellow, 2))
        p.drawRect(roi_rect_w)

        p.setPen(QPen(Qt.green, 1))
        scale = roi_scale_from_roi(self._roi)
        p.drawText(12, 24, f"ROI: x={self._roi.x} y={self._roi.y} w={self._roi.w} h={self._roi.h}")
        p.drawText(12, 44, f"Scale: {scale:.2f}x")

        # Keep the handle in the bottom-right corner without consuming tiny ROIs.
        p.fillRect(self._resize_handle_rect(roi_rect_w), Qt.yellow)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        step = 8
        resize_step = 16
        roi = self._roi

        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self._apply_scale(roi_scale_from_roi(roi) * 1.08, self._roi_center())
            return
        if event.key() == Qt.Key_Minus:
            self._apply_scale(roi_scale_from_roi(roi) / 1.08, self._roi_center())
            return

        if event.modifiers() & Qt.ShiftModifier:
            if event.key() in (Qt.Key_Left, Qt.Key_Up):
                new_w = roi.w + resize_step
                new_h = int(round(new_w * 9.0 / 16.0))
                roi = Roi(roi.x, roi.y, new_w, new_h)
            elif event.key() in (Qt.Key_Right, Qt.Key_Down):
                new_w = roi.w - resize_step
                new_h = int(round(new_w * 9.0 / 16.0))
                roi = Roi(roi.x, roi.y, new_w, new_h)
            else:
                super().keyPressEvent(event)
                return
        else:
            if event.key() == Qt.Key_Left:
                roi = Roi(roi.x - step, roi.y, roi.w, roi.h)
            elif event.key() == Qt.Key_Right:
                roi = Roi(roi.x + step, roi.y, roi.w, roi.h)
            elif event.key() == Qt.Key_Up:
                roi = Roi(roi.x, roi.y - step, roi.w, roi.h)
            elif event.key() == Qt.Key_Down:
                roi = Roi(roi.x, roi.y + step, roi.w, roi.h)
            else:
                super().keyPressEvent(event)
                return

        self._set_roi_and_emit(clamp_roi(roi))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return

        self._cancel_interaction_interpolation()
        self.setFocus(Qt.MouseFocusReason)
        self._drag_start_pos = event.position()
        self._drag_start_roi = self._roi

        roi_rect = self._frame_to_widget_rect(self._roi)
        handle_rect = self._resize_handle_rect(roi_rect)

        if handle_rect.contains(event.position()):
            self._drag_mode = "resize"
        elif roi_rect.contains(event.position()):
            self._drag_mode = "move"
        else:
            self._drag_mode = "none"

        if self._drag_mode != "none":
            self._set_drag_visual_roi_overlay(
                float(self._roi.x),
                float(self._roi.y),
                float(self._roi.w),
                float(self._roi.h),
            )
            if self._drag_mode == "move":
                self._touch_emit_interval_s = self._drag_move_touch_emit_interval_s
            else:
                self._touch_emit_interval_s = self._default_touch_emit_interval_s

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_mode == "none":
            return

        dx = event.position().x() - self._drag_start_pos.x()
        dy = event.position().y() - self._drag_start_pos.y()

        image_rect = self._image_rect()
        if image_rect.width() <= 0 or image_rect.height() <= 0:
            return

        sx = FRAME_W / image_rect.width()
        sy = FRAME_H / image_rect.height()

        if self._drag_mode == "move":
            target_x = float(self._drag_start_roi.x) + (dx * sx)
            target_y = float(self._drag_start_roi.y) + (dy * sy)
            self._set_drag_visual_roi_overlay(
                target_x,
                target_y,
                float(self._drag_start_roi.w),
                float(self._drag_start_roi.h),
            )

            # ROI x must remain even for UYVY 4:2:2; use hysteresis-aware
            # quantization to avoid rapid back/forth near quantization edges.
            quant_x = self._quantize_drag_axis_with_hysteresis(
                target_x,
                self._roi.x,
                quantum=2,
                hysteresis_px=self._drag_x_hysteresis_px,
            )
            quant_y = self._quantize_drag_axis_with_hysteresis(
                target_y,
                self._roi.y,
                quantum=1,
                hysteresis_px=0.45,
            )
            new_roi = Roi(
                int(quant_x),
                int(quant_y),
                self._drag_start_roi.w,
                self._drag_start_roi.h,
            )
        else:
            dw_x = dx * sx
            dw_y = dy * sy * (16.0 / 9.0)
            dw = dw_x if abs(dw_x) >= abs(dw_y) else dw_y
            # Resize around center so the whole ROI scales symmetrically.
            new_w = float(self._drag_start_roi.w) + (2.0 * dw)
            new_h = max(2.0, new_w * 9.0 / 16.0)
            center_x = float(self._drag_start_roi.x) + (float(self._drag_start_roi.w) / 2.0)
            center_y = float(self._drag_start_roi.y) + (float(self._drag_start_roi.h) / 2.0)
            new_x = center_x - (new_w / 2.0)
            new_y = center_y - (new_h / 2.0)
            self._set_drag_visual_roi_overlay(new_x, new_y, new_w, new_h)
            new_roi = Roi(
                int(round(new_x)),
                int(round(new_y)),
                int(round(new_w)),
                int(round(new_h)),
            )

        target_roi = clamp_roi(new_roi)
        emit_scale = self._drag_mode != "move"
        self._queue_interpolated_roi(
            target_roi,
            emit_scale=emit_scale,
            anchor_to_current=(self._drag_mode == "resize"),
            apply_latency_filter=(self._drag_mode != "move"),
        )

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        del event
        self._drag_mode = "none"
        self._touch_emit_interval_s = self._default_touch_emit_interval_s
        self._clear_drag_visual_roi_overlay()
        self._flush_interaction_emit()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        angle_delta = event.angleDelta().y()
        delta = angle_delta
        if delta == 0:
            return

        # Ignore touchpad/touch pinch-like wheel streams to avoid unstable zoom.
        if event.pixelDelta().y() != 0 or bool(event.modifiers() & Qt.ControlModifier):
            return

        effective_delta = float(delta)

        # Exponential scaling keeps wheel notches crisp while smoothing high-rate
        # touchpad/pinch delta bursts.
        base_step = 1.08
        sensitivity = math.log(base_step) / 120.0
        factor = math.exp(effective_delta * sensitivity)
        target_scale = roi_scale_from_roi(self._roi) * factor
        anchor_frame = self._widget_to_frame(event.position())
        self._apply_scale(target_scale, anchor_frame, touch_throttle=True)
        self._schedule_interaction_emit_flush()

    def event(self, event) -> bool:
        et = event.type()
        if et in (QEvent.Type.TouchBegin, QEvent.Type.TouchUpdate, QEvent.Type.TouchEnd):
            # Pinch touch input is intentionally disabled due to unstable tablet behavior.
            event.accept()
            return True
        return super().event(event)

    def _apply_scale(
        self,
        new_scale: float,
        anchor_frame: QPointF,
        emit_scale: bool = True,
        touch_throttle: bool = False,
    ) -> None:
        new_scale = max(1.0, min(new_scale, 16.0))
        center = anchor_frame
        new_roi = roi_from_scale(new_scale, center.x(), center.y())
        if touch_throttle:
            self._queue_interpolated_roi(clamp_roi(new_roi), emit_scale=emit_scale)
            return
        self._set_roi_and_emit(new_roi, emit_scale=emit_scale)

    def _queue_interpolated_roi(
        self,
        target_roi: Roi,
        emit_scale: bool = True,
        anchor_to_current: bool = False,
        apply_latency_filter: bool = True,
    ) -> None:
        raw_target = clamp_roi(target_roi)
        latency = 0.0
        if apply_latency_filter:
            latency = max(0.0, min(1.0, self._latency_smoothing_percent / 100.0))
        if anchor_to_current:
            # Treat each manual drag update as a fresh interpolation segment
            # from the currently displayed ROI to the latest pointer target.
            self._interaction_filtered_target_roi = self._roi
            self._interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}

        if latency > 0.0:
            beta = 1.0 - (0.82 * latency)
            if anchor_to_current:
                prev = self._roi
            else:
                prev = self._interaction_filtered_target_roi if self._interaction_filtered_target_roi is not None else raw_target
            filtered = clamp_roi(
                Roi(
                    int(round(prev.x + (raw_target.x - prev.x) * beta)),
                    int(round(prev.y + (raw_target.y - prev.y) * beta)),
                    int(round(prev.w + (raw_target.w - prev.w) * beta)),
                    int(round(prev.h + (raw_target.h - prev.h) * beta)),
                )
            )
            self._interaction_filtered_target_roi = filtered
            self._interaction_target_roi = filtered
        else:
            self._interaction_filtered_target_roi = raw_target
            self._interaction_target_roi = raw_target
        self._interaction_target_emit_scale = self._interaction_target_emit_scale or emit_scale
        if not self._interaction_interp_timer.isActive():
            self._interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        target_scale = roi_scale_from_roi(self._interaction_target_roi)
        smoothing = max(0.0, min(1.0, self._smoothing_percent / 10.0))
        if target_scale >= 6.0:
            base_interval_ms = 8
        elif target_scale >= 4.0:
            base_interval_ms = 10
        else:
            base_interval_ms = 16
        interval_scale = 1.20 - (0.60 * smoothing)
        interval_ms = int(round(base_interval_ms * interval_scale))
        self._interaction_interp_timer.setInterval(max(6, min(24, interval_ms)))
        if not self._interaction_interp_timer.isActive():
            self._interaction_interp_timer.start()
        self._schedule_interaction_emit_flush()

    def _on_interaction_interp_tick(self) -> None:
        target_roi = self._interaction_target_roi
        if target_roi is None:
            self._interaction_interp_timer.stop()
            return

        next_roi = self._interpolate_roi_step(self._roi, target_roi)
        emit_scale = self._interaction_target_emit_scale
        self._set_roi_and_emit_touch_throttled(next_roi, emit_scale=emit_scale)

        if self._is_roi_close(next_roi, target_roi):
            self._set_roi_and_emit_touch_throttled(target_roi, emit_scale=emit_scale)
            self._interaction_filtered_target_roi = None
            self._interaction_target_roi = None
            self._interaction_target_emit_scale = False
            self._interaction_interp_timer.stop()

    def _interpolate_roi_step(self, current: Roi, target: Roi) -> Roi:
        moving_only = current.w == target.w and current.h == target.h
        zoom_scale = roi_scale_from_roi(target)
        smoothing = max(0.0, min(1.0, self._smoothing_percent / 10.0))
        # Use gentler easing for translation-only motion so ROI travel feels smoother.
        if moving_only:
            if zoom_scale >= 6.0:
                alpha_pos = 0.10
            elif zoom_scale >= 4.0:
                alpha_pos = 0.13
            else:
                alpha_pos = 0.16
        else:
            alpha_pos = 0.24
        alpha_size = 0.22

        # Higher smoothing decreases per-step movement, reducing small-ROI jitter.
        alpha_scale = 1.20 - (0.60 * smoothing)
        alpha_pos = max(0.06, min(0.35, alpha_pos * alpha_scale))
        alpha_size = max(0.08, min(0.40, alpha_size * alpha_scale))

        if zoom_scale >= 6.0:
            lag_limit = 6
        elif zoom_scale >= 4.0:
            lag_limit = 10
        else:
            lag_limit = 14

        near_target_deadband = 1 + int(round(smoothing * 2.0))

        def _step(c: int, t: int, alpha: float, key: str, low_latency: bool = False) -> int:
            delta = t - c
            if delta == 0:
                self._interp_residual[key] = 0.0
                return c

            abs_delta = abs(delta)
            effective_alpha = alpha
            if low_latency:
                # Speed up catch-up on large pointer/finger moves while preserving
                # smoothing on small micro-adjustments.
                accel = (abs_delta / (abs_delta + 56.0)) * 0.40
                effective_alpha = min(0.70, alpha + accel)

                # Ignore tiny near-target noise so small hand tremor doesn't jitter ROI.
                if abs_delta <= near_target_deadband:
                    self._interp_residual[key] = 0.0
                    return t

            raw_move = (delta * effective_alpha) + float(self._interp_residual[key])
            sign = 1 if raw_move > 0 else -1
            move_abs = int(abs(raw_move))
            move = sign * move_abs if move_abs > 0 else 0
            self._interp_residual[key] = raw_move - float(move)

            if low_latency:
                overshoot = abs_delta - lag_limit
                if overshoot > 0:
                    sign = 1 if delta > 0 else -1
                    min_catch_up = int(math.ceil(overshoot * 0.65))
                    enforced = sign * max(abs(move), min_catch_up)
                    if enforced != move:
                        move = enforced
                        self._interp_residual[key] = 0.0

            if move == 0:
                if abs_delta >= max(2, near_target_deadband + 1):
                    move = 1 if delta > 0 else -1
                    self._interp_residual[key] = 0.0
                else:
                    return c
            return c + move

        x = _step(current.x, target.x, alpha_pos, "x", low_latency=moving_only)
        y = _step(current.y, target.y, alpha_pos, "y", low_latency=moving_only)
        w = _step(current.w, target.w, alpha_size, "w")
        h = _step(current.h, target.h, alpha_size, "h")
        return clamp_roi(Roi(x, y, w, h))

    def _is_roi_close(self, roi_a: Roi, roi_b: Roi) -> bool:
        return (
            abs(roi_a.x - roi_b.x) <= 1
            and abs(roi_a.y - roi_b.y) <= 1
            and abs(roi_a.w - roi_b.w) <= 2
            and abs(roi_a.h - roi_b.h) <= 2
        )

    def _apply_roi_local(self, roi: Roi) -> None:
        self._roi = clamp_roi(roi)
        self.update()

    def _cancel_interaction_interpolation(self) -> None:
        self._interaction_interp_timer.stop()
        self._interaction_emit_flush_timer.stop()
        self._interaction_filtered_target_roi = None
        self._interaction_target_roi = None
        self._interaction_target_emit_scale = False
        self._interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}

    def _set_roi_and_emit(self, roi: Roi, emit_scale: bool = True) -> None:
        self._cancel_interaction_interpolation()
        self._apply_roi_local(roi)
        self.roiChanged.emit(roi.x, roi.y, roi.w, roi.h)
        if emit_scale:
            self.scaleChanged.emit(roi_scale_from_roi(roi))

    def _set_roi_and_emit_touch_throttled(self, roi: Roi, emit_scale: bool = True) -> None:
        self._apply_roi_local(roi)
        now = time.perf_counter()
        if (now - self._last_touch_emit_ts) >= self._touch_emit_interval_s:
            self._last_touch_emit_ts = now
            self._touch_emit_pending = False
            self._touch_emit_pending_scale = False
            self.roiChanged.emit(self._roi.x, self._roi.y, self._roi.w, self._roi.h)
            if emit_scale:
                self.scaleChanged.emit(roi_scale_from_roi(self._roi))
            return
        self._touch_emit_pending = True
        self._touch_emit_pending_scale = self._touch_emit_pending_scale or emit_scale

    def _flush_pending_touch_emit(self) -> None:
        if not self._touch_emit_pending:
            return
        self._last_touch_emit_ts = time.perf_counter()
        self._touch_emit_pending = False
        emit_scale = self._touch_emit_pending_scale
        self._touch_emit_pending_scale = False
        self.roiChanged.emit(self._roi.x, self._roi.y, self._roi.w, self._roi.h)
        if emit_scale:
            self.scaleChanged.emit(roi_scale_from_roi(self._roi))

    def _flush_interaction_emit(self) -> None:
        target_roi = self._interaction_target_roi
        emit_scale = self._interaction_target_emit_scale
        self._interaction_interp_timer.stop()
        self._interaction_target_roi = None
        self._interaction_target_emit_scale = False

        if target_roi is not None and (
            target_roi.x != self._roi.x
            or target_roi.y != self._roi.y
            or target_roi.w != self._roi.w
            or target_roi.h != self._roi.h
        ):
            self._set_roi_and_emit(target_roi, emit_scale=emit_scale)

        self._flush_pending_touch_emit()

    def _schedule_interaction_emit_flush(self) -> None:
        # Trailing-edge flush ensures the final zoom state is always propagated.
        self._interaction_emit_flush_timer.start(40)

    def _roi_center(self) -> QPointF:
        return QPointF(self._roi.x + (self._roi.w / 2.0), self._roi.y + (self._roi.h / 2.0))

    def _image_rect(self) -> QRectF:
        if self.width() <= 1 or self.height() <= 1:
            return QRectF(0, 0, 1, 1)
        return QRectF(0.0, 0.0, float(self.width()), float(self.height()))

    def _widget_to_frame(self, point: QPointF) -> QPointF:
        image_rect = self._image_rect()
        if image_rect.width() <= 0 or image_rect.height() <= 0:
            return QPointF(0, 0)

        x = (point.x() - image_rect.left()) * (FRAME_W / image_rect.width())
        y = (point.y() - image_rect.top()) * (FRAME_H / image_rect.height())
        x = max(0.0, min(float(FRAME_W), x))
        y = max(0.0, min(float(FRAME_H), y))
        return QPointF(x, y)

    def _frame_to_widget_rect(self, roi: Roi) -> QRectF:
        return self._frame_to_widget_rect_float(float(roi.x), float(roi.y), float(roi.w), float(roi.h))

    def _frame_to_widget_rect_float(self, x: float, y: float, w: float, h: float) -> QRectF:
        image_rect = self._image_rect()
        sx = image_rect.width() / FRAME_W
        sy = image_rect.height() / FRAME_H
        return QRectF(
            image_rect.left() + (float(x) * sx),
            image_rect.top() + (float(y) * sy),
            max(1.0, float(w) * sx),
            max(1.0, float(h) * sy),
        )


class ImageCanvas(QWidget):
    fullscreenRequested = Signal(str)

    def __init__(self, view_name: str = "output") -> None:
        super().__init__()
        self.setMinimumSize(160, 90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image: QImage | None = None
        self._image_backing: np.ndarray | None = None
        self._view_name = view_name

    def set_image(self, image: QImage, backing: np.ndarray | None = None) -> None:
        self._image = image
        self._image_backing = backing
        self.update()

    def paintEvent(self, event) -> None:
        del event
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        if self._image is None:
            return
        p.drawImage(self._image_rect(), self._image)

    def _image_rect(self) -> QRectF:
        if self.width() <= 1 or self.height() <= 1:
            return QRectF(0, 0, 1, 1)
        return QRectF(0.0, 0.0, float(self.width()), float(self.height()))

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        super().mouseDoubleClickEvent(event)


class VideoProcessorController:
    def __init__(self, module) -> None:
        self._module = module
        self.enable_basic_scaling = True
        self.deinterlace_enabled = True
        self.reinterlace_enabled = False
        self.basic_scaling_method = "bilinear_sharp"
        self.deinterlace_method = "bob"
        self.denoise_method = "off"
        self.denoise_strength = 0.35
        self.max_auto_basic_scaling = 4
        self.basic_scaling_manual = 4
        self.basic_scaling_auto_mode = True
        self.basic_scaling_method_supported = False
        self.color_space = _normalize_color_space_name(os.environ.get("VP_COLOR_SPACE", "rec709"))
        self.color_range = _normalize_color_range_name(os.environ.get("VP_COLOR_RANGE", "limited"))
        self.ai_sr_enabled = False
        self.ai_sr_active = False
        self.ai_sr_model_path = ""
        self.ai_sr_error: str | None = None
        self.ai_sr_provider = "auto"
        self.ai_sr_trt_precision = "fp16"
        self.ai_sr_require_gpu = True
        self.ai_sr_frame_interval = 30
        self.ai_sr_strict = False
        self.ai_sr_input_align = 2
        self.ai_sr_roi_overscan_percent = 0.0
        self.ai_sr_inference_divisor = 0
        self.ai_sr_detail_preserve_percent = 0.0
        self.ai_sr_post_denoise_method = "off"
        self.ai_sr_post_denoise_strength = 0.0
        self.ai_sr_post_artifact_reduction_method = "off"
        self.ai_sr_post_artifact_reduction_strength = 0.0
        self.ai_sr_post_exaggeration_enabled = False
        self.ai_sr_post_exaggeration_gain = 2.0
        self.ai_sr_max_inflight = 1
        self.ai_sr_info: dict[str, object] | None = None
        self.rtx_vsr_enabled = False
        self.rtx_vsr_active = False
        self.rtx_vsr_quality = "high"
        self.rtx_vsr_scale = 2
        self.rtx_vsr_post_scale_method = "bicubic"
        self.rtx_thdr_enabled = False
        self.rtx_thdr_contrast = 50
        self.rtx_thdr_saturation = 50
        self.rtx_thdr_middle_gray = 50
        self.rtx_thdr_max_luminance = 1000
        self.decklink_output_buffer_frames = 2
        self.worker_process_priority = _normalize_worker_priority_name(
            os.environ.get("VP_WORKER_PROCESS_PRIORITY", "above_normal")
        )
        self.rtx_vsr_error: str | None = None
        self.rtx_vsr_info: dict[str, object] | None = None
        self.processor = None
        self._zeroed_output_warning_emitted = False

    def create(self, roi: Roi) -> None:
        sr_scale = 0 if self.basic_scaling_auto_mode else self.basic_scaling_manual
        self.processor = self._module.VideoProcessor(
            width=FRAME_W,
            height=FRAME_H,
            roi_x=roi.x,
            roi_y=roi.y,
            roi_w=roi.w,
            roi_h=roi.h,
            enable_placeholder_sr=self.enable_basic_scaling,
            sr_scale=sr_scale,
        )
        self.processor.set_max_auto_sr_scale(self.max_auto_basic_scaling)
        self.basic_scaling_method_supported = hasattr(self.processor, "set_sr_flavor")
        if self.basic_scaling_method_supported:
            self.processor.set_sr_flavor(self.basic_scaling_method)
        if hasattr(self.processor, "set_color_space"):
            self.processor.set_color_space(self.color_space)
        if hasattr(self.processor, "set_color_range"):
            self.processor.set_color_range(self.color_range)
        self.processor.set_deinterlace_enabled(self.deinterlace_enabled)
        if hasattr(self.processor, "set_deinterlace_method"):
            self.processor.set_deinterlace_method(self.deinterlace_method)
        if hasattr(self.processor, "set_denoise_method"):
            self.processor.set_denoise_method(self.denoise_method)
        if hasattr(self.processor, "set_denoise_strength"):
            self.processor.set_denoise_strength(self.denoise_strength)

    def set_roi(self, roi: Roi) -> bool:
        if self.processor is not None:
            self.processor.set_roi(roi.x, roi.y, roi.w, roi.h)
        return True

    def set_roi_position(self, roi_x: int, roi_y: int) -> bool:
        if self.processor is not None and hasattr(self.processor, "set_roi_position"):
            self.processor.set_roi_position(int(roi_x), int(roi_y))
        return True

    def set_auto_basic_scaling(self) -> None:
        self.basic_scaling_auto_mode = True
        if self.processor is not None and self.enable_basic_scaling:
            self.processor.set_sr_mode_auto()

    def set_manual_basic_scaling(self, scale: int) -> None:
        self.basic_scaling_manual = scale
        self.basic_scaling_auto_mode = False
        if self.processor is not None and self.enable_basic_scaling:
            self.processor.set_sr_scale_manual(scale)

    def effective_scale(self) -> int:
        if self.processor is None or not self.enable_basic_scaling:
            return 1
        return int(self.processor.get_effective_sr_scale())

    @property
    def enable_placeholder_sr(self) -> bool:
        return bool(self.enable_basic_scaling)

    @enable_placeholder_sr.setter
    def enable_placeholder_sr(self, value: bool) -> None:
        self.enable_basic_scaling = bool(value)

    def set_deinterlace_enabled(self, enabled: bool) -> None:
        self.deinterlace_enabled = enabled
        if self.processor is not None:
            self.processor.set_deinterlace_enabled(enabled)

    def set_reinterlace_enabled(self, enabled: bool) -> None:
        self.reinterlace_enabled = bool(enabled)

    def set_deinterlace_method(self, method: str) -> None:
        self.deinterlace_method = str(method)
        if self.processor is not None and hasattr(self.processor, "set_deinterlace_method"):
            self.processor.set_deinterlace_method(method)

    def set_denoise_settings(self, method: str, strength: float) -> None:
        self.denoise_method = str(method)
        self.denoise_strength = max(0.0, min(1.0, float(strength)))
        if self.processor is not None:
            if hasattr(self.processor, "set_denoise_method"):
                self.processor.set_denoise_method(self.denoise_method)
            if hasattr(self.processor, "set_denoise_strength"):
                self.processor.set_denoise_strength(self.denoise_strength)

    def set_max_auto_basic_scaling(self, scale: int) -> None:
        self.max_auto_basic_scaling = scale
        if self.processor is not None:
            self.processor.set_max_auto_sr_scale(scale)

    def set_basic_scaling_method(self, basic_scaling_method: str) -> None:
        self.basic_scaling_method = basic_scaling_method
        if self.processor is not None and hasattr(self.processor, "set_sr_flavor"):
            self.basic_scaling_method_supported = True
            self.processor.set_sr_flavor(basic_scaling_method)

    def set_color_space(self, color_space: str) -> None:
        self.color_space = _normalize_color_space_name(color_space)
        if self.processor is not None and hasattr(self.processor, "set_color_space"):
            self.processor.set_color_space(self.color_space)

    def set_color_range(self, color_range: str) -> None:
        self.color_range = _normalize_color_range_name(color_range)
        if self.processor is not None and hasattr(self.processor, "set_color_range"):
            self.processor.set_color_range(self.color_range)

    # Backward-compatible aliases for existing call sites.
    def set_auto_sr(self) -> None:
        self.set_auto_basic_scaling()

    def set_manual_sr(self, scale: int) -> None:
        self.set_manual_basic_scaling(scale)

    def set_max_auto_sr_scale(self, scale: int) -> None:
        self.set_max_auto_basic_scaling(scale)

    def set_sr_flavor(self, sr_flavor: str) -> None:
        self.set_basic_scaling_method(sr_flavor)

    @property
    def sr_flavor(self) -> str:
        return self.basic_scaling_method

    @sr_flavor.setter
    def sr_flavor(self, value: str) -> None:
        self.basic_scaling_method = str(value)

    @property
    def max_auto_sr_scale(self) -> int:
        return int(self.max_auto_basic_scaling)

    @max_auto_sr_scale.setter
    def max_auto_sr_scale(self, value: int) -> None:
        self.max_auto_basic_scaling = int(value)

    @property
    def sr_manual_scale(self) -> int:
        return int(self.basic_scaling_manual)

    @sr_manual_scale.setter
    def sr_manual_scale(self, value: int) -> None:
        self.basic_scaling_manual = int(value)

    @property
    def sr_auto_mode(self) -> bool:
        return bool(self.basic_scaling_auto_mode)

    @sr_auto_mode.setter
    def sr_auto_mode(self, value: bool) -> None:
        self.basic_scaling_auto_mode = bool(value)

    @property
    def sr_flavor_supported(self) -> bool:
        return bool(self.basic_scaling_method_supported)

    @sr_flavor_supported.setter
    def sr_flavor_supported(self, value: bool) -> None:
        self.basic_scaling_method_supported = bool(value)

    def process_frame(self, frame_bytes: bytes) -> bytes:
        if self.processor is None:
            raise RuntimeError("VideoProcessor is not initialized")
        output = self.processor.process_frame(frame_bytes)
        if looks_zeroed_uyvy_frame(output):
            if not self._zeroed_output_warning_emitted:
                LOGGER.warning("GPU processing produced an all-zero UYVY frame; using passthrough fallback")
                self._zeroed_output_warning_emitted = True
            return frame_bytes
        return output

    def close(self) -> None:
        self.processor = None

    def set_ai_sr_enabled(self, enabled: bool, wait_for_ack: bool = False, timeout_seconds: float = 3.0) -> None:
        self.ai_sr_enabled = bool(enabled)
        self.ai_sr_active = False
        self.ai_sr_error = "AI SR is only available with worker backend"

    def set_ai_sr_model_path(self, model_path: str, wait_for_ack: bool = False, timeout_seconds: float = 3.0) -> None:
        self.ai_sr_model_path = str(model_path)
        self.ai_sr_active = False
        self.ai_sr_error = "AI SR is only available with worker backend"

    def set_ai_sr_settings(
        self,
        provider: str,
        require_gpu: bool,
        inference_fps: int,
        trt_precision: str,
        strict: bool,
        input_align: int,
        roi_overscan_percent: float,
        inference_divisor: int,
        detail_preserve_percent: float,
        post_denoise_method: str,
        post_denoise_strength: float,
        post_artifact_reduction_method: str,
        post_artifact_reduction_strength: float,
        post_exaggeration_enabled: bool,
        post_exaggeration_gain: float,
        max_inflight: int | None = None,
        wait_for_ack: bool = False,
        timeout_seconds: float = 3.0,
    ) -> None:
        self.ai_sr_provider = str(provider)
        trt_precision_name = str(trt_precision).strip().lower()
        self.ai_sr_trt_precision = "int8" if trt_precision_name == "int8" else "fp16"
        self.ai_sr_require_gpu = bool(require_gpu)
        self.ai_sr_frame_interval = max(1, min(60, int(inference_fps)))
        self.ai_sr_strict = bool(strict)
        self.ai_sr_input_align = max(1, int(input_align))
        self.ai_sr_roi_overscan_percent = max(0.0, float(roi_overscan_percent))
        self.ai_sr_inference_divisor = max(0, int(inference_divisor))
        self.ai_sr_detail_preserve_percent = max(0.0, float(detail_preserve_percent))
        self.ai_sr_post_denoise_method = str(post_denoise_method).strip().lower()
        self.ai_sr_post_denoise_strength = max(0.0, min(1.0, float(post_denoise_strength)))
        self.ai_sr_post_artifact_reduction_method = str(post_artifact_reduction_method).strip().lower()
        self.ai_sr_post_artifact_reduction_strength = max(0.0, min(1.0, float(post_artifact_reduction_strength)))
        self.ai_sr_post_exaggeration_enabled = bool(post_exaggeration_enabled)
        self.ai_sr_post_exaggeration_gain = max(1.0, min(4.0, float(post_exaggeration_gain)))
        if max_inflight is not None:
            self.ai_sr_max_inflight = max(1, min(4, int(max_inflight)))
        self.ai_sr_active = False
        self.ai_sr_error = "AI SR is only available with worker backend"

    def set_rtx_vsr_enabled(self, enabled: bool) -> None:
        self.rtx_vsr_enabled = bool(enabled)
        self.rtx_vsr_active = False
        self.rtx_vsr_error = "RTX VSR is only available with worker backend"

    def set_rtx_vsr_settings(
        self,
        quality: str,
        scale: int,
        post_scale_method: str,
        thdr_enabled: bool,
        thdr_contrast: int,
        thdr_saturation: int,
        thdr_middle_gray: int,
        thdr_max_luminance: int,
    ) -> None:
        self.rtx_vsr_quality = str(quality).strip().lower()
        self.rtx_vsr_scale = max(1, int(scale))
        self.rtx_vsr_post_scale_method = str(post_scale_method).strip().lower() or "bicubic"
        self.rtx_thdr_enabled = bool(thdr_enabled)
        self.rtx_thdr_contrast = max(0, int(thdr_contrast))
        self.rtx_thdr_saturation = max(0, int(thdr_saturation))
        self.rtx_thdr_middle_gray = max(0, int(thdr_middle_gray))
        self.rtx_thdr_max_luminance = max(0, int(thdr_max_luminance))
        self.rtx_vsr_active = False
        self.rtx_vsr_error = "RTX VSR is only available with worker backend"

    def start_decklink(self, in_device: int, in_mode: object, out_device: int, out_mode: object, enable_format_detection: bool) -> None:
        raise RuntimeError("DeckLink capture/output in worker is unavailable for in-process backend")

    def stop_decklink(self) -> None:
        return

    def decklink_tick(self, timeout_ms: int = 50) -> tuple[bytes, bytes] | None:
        raise RuntimeError("DeckLink worker tick is unavailable for in-process backend")

    def decklink_processed_counter(self) -> int:
        return 0

    def decklink_output_nominal_fps(self) -> float:
        return 0.0

    def decklink_output_is_interlaced(self) -> bool:
        return False

    def decklink_transition_units_per_output_frame(self) -> float:
        return 1.0

    def decklink_output_buffer_health(self) -> dict[str, object]:
        return {}

    def decklink_pipeline_timing_health(self) -> dict[str, object]:
        return {}

    def decklink_applied_roi(self) -> Roi | None:
        return None

    def decklink_roi_transition_state(self) -> dict[str, object]:
        return {}

    def decklink_timecode_info(self) -> dict[str, object]:
        return {}

    def set_preview_fps(self, preview_fps: float) -> None:
        # In-process backend does not use worker tick preview throttling.
        _ = preview_fps

    def set_decklink_output_buffer_frames(self, buffer_frames: int) -> None:
        # In-process backend does not use worker DeckLink output buffering.
        self.decklink_output_buffer_frames = max(0, min(10, int(buffer_frames)))

    def set_worker_process_priority(self, priority_name: str) -> None:
        # In-process backend does not launch a worker process.
        self.worker_process_priority = _normalize_worker_priority_name(priority_name)

    def set_roi_subpixel_shift(self, shift_x: float, shift_y: float) -> None:
        if self.processor is not None and hasattr(self.processor, "set_subpixel_shift"):
            self.processor.set_subpixel_shift(float(shift_x), float(shift_y))

    def set_roi_with_subpixel(self, roi: Roi, shift_x: float, shift_y: float, manual_drag: bool = False) -> bool:
        _ = manual_drag
        clamped = clamp_roi(roi)
        if self.processor is not None:
            moving_only = hasattr(self.processor, "set_roi_position")
            if moving_only:
                try:
                    prev_roi = self.processor.get_roi() if hasattr(self.processor, "get_roi") else None
                except Exception:
                    prev_roi = None
                if isinstance(prev_roi, tuple) and len(prev_roi) == 4:
                    moving_only = (int(prev_roi[2]) == int(clamped.w) and int(prev_roi[3]) == int(clamped.h))
            if moving_only and hasattr(self.processor, "set_roi_position"):
                self.processor.set_roi_position(int(clamped.x), int(clamped.y))
            else:
                self.processor.set_roi(int(clamped.x), int(clamped.y), int(clamped.w), int(clamped.h))
            if hasattr(self.processor, "set_subpixel_shift"):
                self.processor.set_subpixel_shift(float(shift_x), float(shift_y))
        return True

    def set_roi_manual_drag_hold_seconds(self, hold_seconds: float) -> None:
        _ = hold_seconds
        return

    def set_interlaced_field2_phase_fraction(self, fraction: float) -> None:
        _ = fraction
        return

    def start_roi_microstep_transition(
        self,
        start_roi: Roi,
        target_roi: Roi,
        duration_frames: int,
        interpolation_mode: str,
        overscan_percent: float,
        start_from_current: bool = False,
        enforce_full_frame_scale_1x: bool = False,
    ) -> None:
        _ = (
            start_roi,
            target_roi,
            duration_frames,
            interpolation_mode,
            overscan_percent,
            start_from_current,
            enforce_full_frame_scale_1x,
        )

    def cancel_roi_microstep_transition(self, reset_subpixel_shift: bool = True) -> None:
        _ = reset_subpixel_shift
        return


class ProcessVideoProcessorController:
    def __init__(self) -> None:
        self.enable_basic_scaling = True
        self.deinterlace_enabled = True
        self.reinterlace_enabled = os.environ.get("VP_REINTERLACE_ENABLE", "0") == "1"
        self.basic_scaling_method = "bilinear_sharp"
        self.deinterlace_method = "bob"
        self.denoise_method = "off"
        self.denoise_strength = 0.35
        self.max_auto_basic_scaling = 4
        self.basic_scaling_manual = 4
        self.basic_scaling_auto_mode = True
        self.basic_scaling_method_supported = True
        self.color_space = _normalize_color_space_name(os.environ.get("VP_COLOR_SPACE", "rec709"))
        self.color_range = _normalize_color_range_name(os.environ.get("VP_COLOR_RANGE", "limited"))
        self.ai_sr_model_path = os.environ.get("VP_AI_SR_MODEL", "")
        self.ai_sr_enabled = os.environ.get("VP_AI_SR_ENABLE", "0") == "1"
        self.ai_sr_provider = os.environ.get("VP_AI_SR_PROVIDER", "auto")
        self.ai_sr_trt_precision = os.environ.get("VP_AI_SR_TRT_PRECISION", "fp16").strip().lower() or "fp16"
        if self.ai_sr_trt_precision not in {"fp16", "int8"}:
            self.ai_sr_trt_precision = "fp16"
        self.ai_sr_require_gpu = os.environ.get("VP_AI_SR_REQUIRE_GPU", "1") == "1"
        explicit_ai_fps = os.environ.get("VP_AI_SR_INFERENCE_FPS")
        if explicit_ai_fps is not None:
            self.ai_sr_frame_interval = _clamp_ai_inference_fps(int(explicit_ai_fps))
        else:
            legacy_interval = int(os.environ.get("VP_AI_SR_FRAME_INTERVAL", "1"))
            self.ai_sr_frame_interval = _legacy_ai_frame_interval_to_fps(legacy_interval)
        self.ai_sr_strict = os.environ.get("VP_AI_SR_STRICT", "0") == "1"
        self.ai_sr_input_align = max(1, int(os.environ.get("VP_AI_SR_INPUT_ALIGN", "2")))
        self.ai_sr_roi_overscan_percent = max(0.0, float(os.environ.get("VP_AI_SR_ROI_OVERSCAN_PCT", "0")))
        self.ai_sr_inference_divisor = max(0, int(os.environ.get("VP_AI_SR_INFERENCE_DIVISOR", "0")))
        self.ai_sr_detail_preserve_percent = max(0.0, float(os.environ.get("VP_AI_SR_DETAIL_PRESERVE_PCT", "0")))
        self.ai_sr_post_denoise_method = str(os.environ.get("VP_AI_SR_POST_DENOISE_METHOD", "off")).strip().lower() or "off"
        self.ai_sr_post_denoise_strength = max(0.0, min(1.0, float(os.environ.get("VP_AI_SR_POST_DENOISE_STRENGTH", "0.0"))))
        self.ai_sr_post_artifact_reduction_method = str(
            os.environ.get("VP_AI_SR_POST_ARTIFACT_REDUCTION_METHOD", "off")
        ).strip().lower() or "off"
        self.ai_sr_post_artifact_reduction_strength = max(
            0.0,
            min(1.0, float(os.environ.get("VP_AI_SR_POST_ARTIFACT_REDUCTION_STRENGTH", "0.0"))),
        )
        self.ai_sr_post_exaggeration_enabled = os.environ.get("VP_AI_SR_POST_EXAGGERATION_ENABLED", "0") == "1"
        self.ai_sr_post_exaggeration_gain = max(
            1.0,
            min(4.0, float(os.environ.get("VP_AI_SR_POST_EXAGGERATION_GAIN", "2.0"))),
        )
        self.ai_sr_hold_last_frame = os.environ.get("VP_AI_SR_HOLD_LAST_FRAME", "1") == "1"
        self.ai_sr_max_hold_ms = max(0.0, float(os.environ.get("VP_AI_SR_MAX_HOLD_MS", "0")))
        self.ai_sr_max_inflight = max(1, min(4, int(os.environ.get("VP_AI_SR_MAX_INFLIGHT", "1"))))
        self.ai_sr_active = False
        self.ai_sr_error: str | None = None
        self.ai_sr_info: dict[str, object] | None = None
        self.ai_sr_last_warning: str | None = None
        self.rtx_vsr_enabled = os.environ.get("VP_RTX_VSR_ENABLE", "0") == "1"
        self.rtx_vsr_quality = os.environ.get("VP_RTX_VSR_QUALITY", "high").strip().lower() or "high"
        self.rtx_vsr_scale = max(1, int(os.environ.get("VP_RTX_VSR_SCALE", "2")))
        self.rtx_vsr_post_scale_method = os.environ.get("VP_RTX_VSR_POST_SCALE_METHOD", "bicubic").strip().lower() or "bicubic"
        self.rtx_thdr_enabled = os.environ.get("VP_RTX_THDR_ENABLE", "0") == "1"
        self.rtx_thdr_contrast = max(0, int(os.environ.get("VP_RTX_THDR_CONTRAST", "50")))
        self.rtx_thdr_saturation = max(0, int(os.environ.get("VP_RTX_THDR_SATURATION", "50")))
        self.rtx_thdr_middle_gray = max(0, int(os.environ.get("VP_RTX_THDR_MIDDLE_GRAY", "50")))
        self.rtx_thdr_max_luminance = max(0, int(os.environ.get("VP_RTX_THDR_MAX_LUMINANCE", "1000")))
        self.decklink_output_buffer_frames = max(0, min(10, int(os.environ.get("VP_DECKLINK_OUTPUT_BUFFER_FRAMES", "2"))))
        self.interlaced_field2_phase_fraction = _clamp_interlaced_field2_phase_fraction(
            float(os.environ.get("VP_INTERLACED_FIELD2_PHASE_FRACTION", "0.50"))
        )
        self.worker_process_priority = _normalize_worker_priority_name(
            os.environ.get("VP_WORKER_PROCESS_PRIORITY", "above_normal")
        )
        self.worker_process_priority_error: str | None = None
        self.rtx_vsr_active = False
        self.rtx_vsr_error: str | None = None
        self.rtx_vsr_info: dict[str, object] | None = None

        self._ctx = mp.get_context("spawn")
        self._request_queue = None
        self._response_queue = None
        self._process = None
        self._roi_telemetry_shared = None
        self._roi_telemetry_seq = None
        self._roi_telemetry_last_seq = -1

        self._next_frame_id = 1
        self._latest_output_frame: bytes | None = None
        self._latest_decklink_frame: tuple[bytes, bytes] | None = None
        self._decklink_frame_updated = False
        self._latest_effective_scale = 1
        self._decklink_no_frame_reason: str | None = None
        self._decklink_processed_counter = 0
        self._decklink_processed_fps = 0.0
        self._decklink_output_nominal_fps = 0.0
        self._decklink_output_is_interlaced = False
        self._decklink_transition_units_per_output_frame = 1.0
        self._decklink_processed_counter_last = 0
        self._decklink_processed_counter_last_ts = 0.0
        self._decklink_processed_fps_smoothed = 0.0
        self._decklink_ai_applied_frames = 0
        self._decklink_ai_reused_frames = 0
        self._decklink_ai_passthrough_frames = 0
        self._decklink_ai_completed_frames = 0
        self._decklink_ai_completed_last = 0
        self._decklink_ai_completed_last_ts = 0.0
        self._decklink_ai_refresh_fps = 0.0
        self._decklink_ai_latest_age_ms = -1.0
        self._decklink_ai_timing_ms: dict[str, object] = {}
        self._decklink_rtx_vsr_applied = False
        self._decklink_rtx_effect_mean_abs_luma = 0.0
        self._decklink_stage_enable_flags: dict[str, bool] = {
            "preprocess": False,
            "basic_scaling": False,
            "ai_sr": False,
            "rtx_vsr": False,
        }
        self._decklink_stage_last_applied: dict[str, bool] = {
            "preprocess": False,
            "basic_scaling": False,
            "ai_sr": False,
            "rtx_vsr": False,
        }
        self._decklink_stage_apply_counts: dict[str, int] = {
            "preprocess": 0,
            "basic_scaling": 0,
            "ai_sr": 0,
            "rtx_vsr": 0,
            "passthrough": 0,
        }
        self._decklink_tick_pending = False
        self._decklink_tick_pending_since = 0.0
        self._decklink_preview_interval = max(1, int(os.environ.get("VP_DECKLINK_PREVIEW_INTERVAL", "3")))
        self._decklink_tick_counter = 0
        self._gpu_live_mode = os.environ.get("VP_GPU_LIVE_MODE", "1") == "1"
        self._preview_fps = max(0.0, float(os.environ.get("VP_PREVIEW_FPS", "90")))
        self._last_preview_request_ts = 0.0
        self._control_send_stats = {
            "attempted": 0,
            "sent": 0,
            "dropped": 0,
            "queue_full": 0,
            "compactions": 0,
            "compaction_roi_dropped": 0,
            "fast_path_hits": 0,
            "total_send_ms": 0.0,
            "max_send_ms": 0.0,
        }
        self._control_send_stats_by_cmd: dict[str, dict[str, float]] = {}
        self._decklink_stage_queue_depths: dict[str, int] = {
            "capture_to_preprocess": 0,
            "preprocess_to_upscale": 0,
            "upscale_to_output": 0,
        }
        self._decklink_stage_drop_counts: dict[str, int] = {
            "capture": 0,
            "preprocess": 0,
            "upscale": 0,
        }
        self._decklink_output_buffer_health: dict[str, object] = {}
        self._decklink_pipeline_timing_health: dict[str, object] = {}
        self._decklink_timecode_info: dict[str, object] = {
            "present": False,
            "text": "",
            "format_code": 0,
            "format_name": "",
        }
        self._decklink_applied_roi: Roi | None = None
        self._decklink_roi_transition_state: dict[str, object] = {}
        self._last_interlaced_phase_log_signature: str = ""

    def _reset_decklink_fps_tracking(self) -> None:
        self._decklink_processed_counter = 0
        self._decklink_processed_fps = 0.0
        self._decklink_output_nominal_fps = 0.0
        self._decklink_output_is_interlaced = False
        self._decklink_transition_units_per_output_frame = 1.0
        self._decklink_output_buffer_health = {}
        self._decklink_pipeline_timing_health = {}
        self._decklink_processed_counter_last = 0
        self._decklink_processed_counter_last_ts = 0.0
        self._decklink_processed_fps_smoothed = 0.0
        self._decklink_ai_applied_frames = 0
        self._decklink_ai_reused_frames = 0
        self._decklink_ai_passthrough_frames = 0
        self._decklink_ai_completed_frames = 0
        self._decklink_ai_completed_last = 0
        self._decklink_ai_completed_last_ts = 0.0
        self._decklink_ai_refresh_fps = 0.0
        self._decklink_ai_latest_age_ms = -1.0
        self._decklink_ai_timing_ms = {}
        self._decklink_timecode_info = {
            "present": False,
            "text": "",
            "format_code": 0,
            "format_name": "",
        }
        self._decklink_applied_roi = None
        self._decklink_roi_transition_state = {}

    def _decode_interp_mode_code(self, mode_code: int) -> str:
        code = int(mode_code)
        if code == 1:
            return "ease_in_out"
        if code == 2:
            return "ease_out"
        return "linear"

    def _read_shared_roi_telemetry(self, force: bool = False) -> None:
        shared = self._roi_telemetry_shared
        seq = self._roi_telemetry_seq
        if shared is None or seq is None:
            return
        try:
            current_seq = int(seq.value)
        except Exception:
            return
        if (not force) and current_seq == self._roi_telemetry_last_seq:
            return

        try:
            with shared.get_lock():
                snapshot = [float(shared[i]) for i in range(_ROI_TELEMETRY_SLOT_COUNT)]
        except Exception:
            return

        self._roi_telemetry_last_seq = current_seq
        try:
            self._decklink_applied_roi = clamp_roi(
                Roi(
                    int(round(snapshot[_ROI_TM_APPLIED_X])),
                    int(round(snapshot[_ROI_TM_APPLIED_Y])),
                    int(round(snapshot[_ROI_TM_APPLIED_W])),
                    int(round(snapshot[_ROI_TM_APPLIED_H])),
                )
            )
        except Exception:
            pass

        active = bool(snapshot[_ROI_TM_ACTIVE] >= 0.5)
        total_frames = max(0, int(round(snapshot[_ROI_TM_TOTAL_FRAMES])))
        frame_progress = max(0.0, float(snapshot[_ROI_TM_FRAME_PROGRESS]))
        prev_transition_state = self._decklink_roi_transition_state if isinstance(self._decklink_roi_transition_state, dict) else {}
        prev_interlaced_phase = prev_transition_state.get("interlaced_field_phase") if isinstance(prev_transition_state, dict) else None
        self._decklink_roi_transition_state = {
            "active": active,
            "frame_progress": frame_progress,
            "total_frames": total_frames,
            "interpolation_mode": self._decode_interp_mode_code(int(round(snapshot[_ROI_TM_INTERP_MODE_CODE]))),
            "start_roi": {
                "x": int(round(snapshot[_ROI_TM_START_X])),
                "y": int(round(snapshot[_ROI_TM_START_Y])),
                "w": int(round(snapshot[_ROI_TM_START_W])),
                "h": int(round(snapshot[_ROI_TM_START_H])),
            },
            "target_roi": {
                "x": int(round(snapshot[_ROI_TM_TARGET_X])),
                "y": int(round(snapshot[_ROI_TM_TARGET_Y])),
                "w": int(round(snapshot[_ROI_TM_TARGET_W])),
                "h": int(round(snapshot[_ROI_TM_TARGET_H])),
            },
        }
        if active and isinstance(prev_interlaced_phase, dict):
            self._decklink_roi_transition_state["interlaced_field_phase"] = dict(prev_interlaced_phase)

    def _apply_decklink_frame_message(self, message: dict[str, object]) -> None:
        self._latest_effective_scale = int(message.get("effective_sr_scale", self._latest_effective_scale))
        if "input_frame_bytes" in message and "output_frame_bytes" in message:
            self._latest_decklink_frame = (
                message["input_frame_bytes"],
                message["output_frame_bytes"],
            )
            self._decklink_frame_updated = True

        new_counter = int(message.get("processed_frame_counter", self._decklink_processed_counter))
        worker_reported_fps = float(message.get("processed_fps", self._decklink_processed_fps))
        now = time.perf_counter()
        local_fps = None

        if self._decklink_processed_counter_last_ts > 0.0 and new_counter >= self._decklink_processed_counter_last:
            dt = now - self._decklink_processed_counter_last_ts
            dc = new_counter - self._decklink_processed_counter_last
            if dt > 1e-4:
                local_fps = float(dc) / dt

        self._decklink_processed_counter = new_counter
        self._decklink_processed_counter_last = new_counter
        self._decklink_processed_counter_last_ts = now

        if local_fps is None:
            effective_fps = worker_reported_fps
        elif self._decklink_processed_fps_smoothed <= 0.0:
            effective_fps = local_fps
        else:
            # Smooth short-term jitter while preserving real output-rate changes.
            alpha = 0.35
            effective_fps = (1.0 - alpha) * self._decklink_processed_fps_smoothed + alpha * local_fps

        self._decklink_processed_fps_smoothed = max(0.0, float(effective_fps))
        self._decklink_processed_fps = self._decklink_processed_fps_smoothed
        self._decklink_output_nominal_fps = max(
            0.0,
            float(message.get("output_nominal_fps", self._decklink_output_nominal_fps)),
        )
        if self._decklink_output_nominal_fps > 0.0:
            self._decklink_processed_fps_smoothed = min(
                self._decklink_processed_fps_smoothed,
                self._decklink_output_nominal_fps,
            )
            self._decklink_processed_fps = self._decklink_processed_fps_smoothed
        self._decklink_output_is_interlaced = bool(
            message.get("output_mode_is_interlaced", self._decklink_output_is_interlaced)
        )
        self._decklink_transition_units_per_output_frame = max(
            0.1,
            float(
                message.get(
                    "output_transition_units_per_frame",
                    self._decklink_transition_units_per_output_frame,
                )
            ),
        )

        self._decklink_ai_applied_frames = int(message.get("ai_sr_applied_frames", self._decklink_ai_applied_frames))
        self._decklink_ai_reused_frames = int(message.get("ai_sr_reused_frames", self._decklink_ai_reused_frames))
        self._decklink_ai_passthrough_frames = int(message.get("ai_sr_passthrough_frames", self._decklink_ai_passthrough_frames))
        new_ai_completed = int(message.get("ai_sr_completed_frames", self._decklink_ai_completed_frames))
        self._decklink_ai_latest_age_ms = float(message.get("ai_sr_latest_age_ms", self._decklink_ai_latest_age_ms))
        ai_refresh_local_fps = None
        if self._decklink_ai_completed_last_ts > 0.0 and new_ai_completed >= self._decklink_ai_completed_last:
            dt_ai = now - self._decklink_ai_completed_last_ts
            dc_ai = new_ai_completed - self._decklink_ai_completed_last
            if dt_ai > 1e-4:
                ai_refresh_local_fps = float(dc_ai) / dt_ai
        self._decklink_ai_completed_frames = new_ai_completed
        self._decklink_ai_completed_last = new_ai_completed
        self._decklink_ai_completed_last_ts = now
        if ai_refresh_local_fps is not None:
            if self._decklink_ai_refresh_fps <= 0.0:
                self._decklink_ai_refresh_fps = ai_refresh_local_fps
            else:
                alpha_ai = 0.40
                self._decklink_ai_refresh_fps = ((1.0 - alpha_ai) * self._decklink_ai_refresh_fps) + (alpha_ai * ai_refresh_local_fps)
        self._decklink_ai_timing_ms = dict(message.get("ai_sr_timing_ms", self._decklink_ai_timing_ms))

        self._decklink_rtx_vsr_applied = bool(message.get("rtx_vsr_applied", self._decklink_rtx_vsr_applied))
        self._decklink_rtx_effect_mean_abs_luma = float(
            message.get("rtx_effect_mean_abs_luma", self._decklink_rtx_effect_mean_abs_luma)
        )
        self._decklink_stage_enable_flags = dict(message.get("stage_enable_flags", self._decklink_stage_enable_flags))
        self._decklink_stage_last_applied = dict(message.get("stage_last_applied", self._decklink_stage_last_applied))
        self._decklink_stage_apply_counts = dict(message.get("stage_apply_counts", self._decklink_stage_apply_counts))
        self._decklink_stage_queue_depths = dict(message.get("stage_queue_depths", self._decklink_stage_queue_depths))
        self._decklink_stage_drop_counts = dict(message.get("stage_drop_counts", self._decklink_stage_drop_counts))
        self._decklink_output_buffer_health = dict(
            message.get("output_buffer_health", self._decklink_output_buffer_health)
        )
        self._decklink_pipeline_timing_health = dict(
            message.get("pipeline_timing_health", self._decklink_pipeline_timing_health)
        )
        self._decklink_timecode_info = dict(message.get("timecode_info", self._decklink_timecode_info))

        roi_payload = message.get("roi_applied")
        if isinstance(roi_payload, dict):
            try:
                self._decklink_applied_roi = clamp_roi(
                    Roi(
                        int(roi_payload.get("x", 0)),
                        int(roi_payload.get("y", 0)),
                        int(roi_payload.get("w", FRAME_W)),
                        int(roi_payload.get("h", FRAME_H)),
                    )
                )
            except Exception:
                self._decklink_applied_roi = None

        transition_payload = message.get("roi_transition")
        if isinstance(transition_payload, dict):
            self._decklink_roi_transition_state = dict(transition_payload)

        self._decklink_no_frame_reason = None
        self._decklink_tick_pending = False
        self._decklink_tick_pending_since = 0.0

    def decklink_applied_roi(self) -> Roi | None:
        self._read_shared_roi_telemetry()
        return self._decklink_applied_roi

    def decklink_roi_transition_state(self) -> dict[str, object]:
        self._read_shared_roi_telemetry()
        return dict(self._decklink_roi_transition_state)

    def _record_control_send_result(
        self,
        cmd: str,
        sent: bool,
        elapsed_ms: float,
        queue_full: bool = False,
        compaction_run: bool = False,
        compaction_roi_dropped: int = 0,
        fast_path_hit: bool = False,
    ) -> None:
        stats = self._control_send_stats
        stats["attempted"] += 1
        if sent:
            stats["sent"] += 1
        else:
            stats["dropped"] += 1
        if queue_full:
            stats["queue_full"] += 1
        if compaction_run:
            stats["compactions"] += 1
            stats["compaction_roi_dropped"] += max(0, int(compaction_roi_dropped))
        if fast_path_hit:
            stats["fast_path_hits"] += 1

        elapsed = max(0.0, float(elapsed_ms))
        stats["total_send_ms"] += elapsed
        if elapsed > float(stats["max_send_ms"]):
            stats["max_send_ms"] = elapsed

        cmd_stats = self._control_send_stats_by_cmd.setdefault(
            cmd,
            {
                "attempted": 0.0,
                "sent": 0.0,
                "dropped": 0.0,
                "queue_full": 0.0,
                "total_send_ms": 0.0,
                "max_send_ms": 0.0,
            },
        )
        cmd_stats["attempted"] += 1
        if sent:
            cmd_stats["sent"] += 1
        else:
            cmd_stats["dropped"] += 1
        if queue_full:
            cmd_stats["queue_full"] += 1
        cmd_stats["total_send_ms"] += elapsed
        if elapsed > cmd_stats["max_send_ms"]:
            cmd_stats["max_send_ms"] = elapsed

    def control_send_stats_snapshot(self, reset: bool = False) -> dict[str, object]:
        stats = dict(self._control_send_stats)
        attempted = max(1, int(stats.get("attempted", 0)))
        stats["avg_send_ms"] = float(stats.get("total_send_ms", 0.0)) / float(attempted)
        stats["by_cmd"] = {k: dict(v) for k, v in self._control_send_stats_by_cmd.items()}

        if reset:
            self._control_send_stats = {
                "attempted": 0,
                "sent": 0,
                "dropped": 0,
                "queue_full": 0,
                "compactions": 0,
                "compaction_roi_dropped": 0,
                "fast_path_hits": 0,
                "total_send_ms": 0.0,
                "max_send_ms": 0.0,
            }
            self._control_send_stats_by_cmd = {}
        return stats

    def decklink_queue_telemetry(self) -> tuple[dict[str, int], dict[str, int]]:
        return dict(self._decklink_stage_queue_depths), dict(self._decklink_stage_drop_counts)

    def decklink_output_buffer_health(self) -> dict[str, object]:
        return dict(self._decklink_output_buffer_health)

    def decklink_pipeline_timing_health(self) -> dict[str, object]:
        return dict(self._decklink_pipeline_timing_health)

    def decklink_timecode_info(self) -> dict[str, object]:
        return dict(self._decklink_timecode_info)

    def create(self, roi: Roi) -> None:
        self.close()

        if run_processor_worker is None:
            raise RuntimeError("Process worker module is unavailable")

        effective_basic_scaling_enabled = bool(self.enable_basic_scaling) and not bool(self.ai_sr_enabled)
        sr_scale = 0 if self.basic_scaling_auto_mode else self.basic_scaling_manual
        project_root = str(Path(__file__).resolve().parents[1])
        startup_config = {
            "project_root": project_root,
            "width": FRAME_W,
            "height": FRAME_H,
            "roi_x": roi.x,
            "roi_y": roi.y,
            "roi_w": roi.w,
            "roi_h": roi.h,
            "enable_basic_scaling": effective_basic_scaling_enabled,
            "sr_scale": sr_scale,
            "basic_scaling_auto_mode": self.basic_scaling_auto_mode,
            "basic_scaling_manual": self.basic_scaling_manual,
            "basic_scaling_method": self.basic_scaling_method,
            "color_space": self.color_space,
            "color_range": self.color_range,
            "max_auto_basic_scaling": self.max_auto_basic_scaling,
            "deinterlace_enabled": self.deinterlace_enabled,
            "reinterlace_enabled": bool(self.reinterlace_enabled),
            "deinterlace_method": self.deinterlace_method,
            "denoise_method": self.denoise_method,
            "denoise_strength": self.denoise_strength,
            "ai_sr_enabled": self.ai_sr_enabled,
            "ai_sr_model_path": self.ai_sr_model_path,
            "ai_sr_provider": self.ai_sr_provider,
            "ai_sr_trt_precision": self.ai_sr_trt_precision,
            "ai_sr_require_gpu": self.ai_sr_require_gpu,
            "ai_sr_frame_interval": self.ai_sr_frame_interval,
            "ai_sr_inference_fps": self.ai_sr_frame_interval,
            "ai_sr_strict": self.ai_sr_strict,
            "ai_sr_input_align": self.ai_sr_input_align,
            "ai_sr_roi_overscan_percent": self.ai_sr_roi_overscan_percent,
            "ai_sr_inference_divisor": self.ai_sr_inference_divisor,
            "ai_sr_detail_preserve_percent": self.ai_sr_detail_preserve_percent,
            "ai_sr_post_denoise_method": self.ai_sr_post_denoise_method,
            "ai_sr_post_denoise_strength": self.ai_sr_post_denoise_strength,
            "ai_sr_post_artifact_reduction_method": self.ai_sr_post_artifact_reduction_method,
            "ai_sr_post_artifact_reduction_strength": self.ai_sr_post_artifact_reduction_strength,
            "ai_sr_post_exaggeration_enabled": self.ai_sr_post_exaggeration_enabled,
            "ai_sr_post_exaggeration_gain": self.ai_sr_post_exaggeration_gain,
            "ai_sr_hold_last_frame": bool(self.ai_sr_hold_last_frame),
            "ai_sr_max_hold_ms": float(self.ai_sr_max_hold_ms),
            "ai_sr_max_inflight": int(self.ai_sr_max_inflight),
            "rtx_vsr_enabled": self.rtx_vsr_enabled,
            "rtx_vsr_quality": self.rtx_vsr_quality,
            "rtx_vsr_scale": self.rtx_vsr_scale,
            "rtx_vsr_post_scale_method": self.rtx_vsr_post_scale_method,
            "rtx_thdr_enabled": self.rtx_thdr_enabled,
            "rtx_thdr_contrast": self.rtx_thdr_contrast,
            "rtx_thdr_saturation": self.rtx_thdr_saturation,
            "rtx_thdr_middle_gray": self.rtx_thdr_middle_gray,
            "rtx_thdr_max_luminance": self.rtx_thdr_max_luminance,
            "decklink_output_buffer_frames": self.decklink_output_buffer_frames,
            "interlaced_field2_phase_fraction": float(self.interlaced_field2_phase_fraction),
            "worker_process_priority": self.worker_process_priority,
            "rtx_video_sdk_root": os.environ.get("RTX_VIDEO_SDK_ROOT", r"C:\Coding Projects\sdks\NVidia video SDK"),
        }

        # Keep request queue larger than response queue so bursty UI events
        # (ROI drag, tick polling) do not trip queue.Full in the GUI thread.
        self._request_queue = self._ctx.Queue(maxsize=32)
        self._response_queue = self._ctx.Queue(maxsize=64)
        self._roi_telemetry_shared = self._ctx.Array("d", _ROI_TELEMETRY_SLOT_COUNT)
        self._roi_telemetry_seq = self._ctx.Value("i", 0)
        self._roi_telemetry_last_seq = -1
        self._process = self._ctx.Process(
            target=run_processor_worker,
            args=(
                self._request_queue,
                self._response_queue,
                startup_config,
                self._roi_telemetry_shared,
                self._roi_telemetry_seq,
            ),
            daemon=True,
            name="video-processor-worker",
        )
        self._process.start()

        self._latest_output_frame = None
        self._latest_decklink_frame = None
        self._latest_effective_scale = 1
        self._next_frame_id = 1
        self._decklink_tick_pending = False
        self._decklink_tick_pending_since = 0.0
        self._reset_decklink_fps_tracking()
        self._wait_for_ready(timeout_seconds=5.0)

    def _wait_for_ready(self, timeout_seconds: float) -> None:
        if self._response_queue is None:
            raise RuntimeError("Worker response queue is not initialized")

        deadline = time.perf_counter() + timeout_seconds
        while time.perf_counter() < deadline:
            self._assert_worker_alive()
            try:
                message = self._response_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            message_type = message.get("type")
            if message_type == "ready":
                self.basic_scaling_method_supported = bool(message.get("basic_scaling_method_supported", message.get("sr_flavor_supported", True)))
                self.ai_sr_enabled = bool(message.get("ai_sr_enabled", self.ai_sr_enabled))
                self.ai_sr_active = bool(message.get("ai_sr_active", False))
                self.ai_sr_error = message.get("ai_sr_error")
                self.ai_sr_info = message.get("ai_sr_info")
                self.rtx_vsr_enabled = bool(message.get("rtx_vsr_enabled", self.rtx_vsr_enabled))
                self.rtx_vsr_active = bool(message.get("rtx_vsr_active", self.rtx_vsr_active))
                self.rtx_vsr_error = message.get("rtx_vsr_error")
                self.rtx_vsr_info = message.get("rtx_vsr_info")
                self.color_space = _normalize_color_space_name(str(message.get("color_space", self.color_space)))
                self.color_range = _normalize_color_range_name(str(message.get("color_range", self.color_range)))
                self.worker_process_priority = _normalize_worker_priority_name(
                    str(message.get("worker_process_priority", self.worker_process_priority))
                )
                self.worker_process_priority_error = (
                    str(message.get("worker_process_priority_error", "")).strip() or None
                )
                return
            if message_type == "error":
                raise RuntimeError(
                    f"Worker startup failed: {message.get('error')}\n{message.get('traceback', '')}"
                )

        raise RuntimeError("Timed out waiting for worker startup")

    def _assert_worker_alive(self) -> None:
        if self._process is None:
            raise RuntimeError("Worker process is not started")
        if not self._process.is_alive():
            exit_code = self._process.exitcode
            if exit_code is None:
                raise RuntimeError("Worker process exited unexpectedly")
            raise RuntimeError(f"Worker process exited unexpectedly (exit_code={exit_code})")

    def _send_control(self, command: dict[str, object]) -> bool:
        self._assert_worker_alive()
        if self._request_queue is None:
            raise RuntimeError("Worker request queue is not initialized")

        started = time.perf_counter()
        cmd = str(command.get("cmd", ""))
        latest_wins_roi_cmds = {
            "set_roi",
            "set_roi_position",
            "set_roi_with_subpixel",
        }
        drop_when_roi_cmds = {
            "decklink_tick",
        }
        best_effort_cmds = {
            "decklink_tick",
            # Live ROI interaction commands should never block the GUI thread.
            "set_roi",
            "set_roi_position",
            "set_roi_subpixel_shift",
            "set_roi_with_subpixel",
        }

        if cmd in latest_wins_roi_cmds:
            # Fast path: avoid queue compaction work on every ROI update.
            try:
                self._request_queue.put_nowait(command)
                self._record_control_send_result(
                    cmd,
                    sent=True,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    fast_path_hit=True,
                )
                return True
            except queue.Full:
                pass

            preserved_commands: list[dict[str, object]] = []
            roi_compaction_drops = 0
            while True:
                try:
                    pending = self._request_queue.get_nowait()
                except queue.Empty:
                    break

                pending_cmd = str(pending.get("cmd", ""))
                if pending_cmd in latest_wins_roi_cmds:
                    roi_compaction_drops += 1
                    continue
                if pending_cmd in drop_when_roi_cmds:
                    continue
                preserved_commands.append(pending)

            for pending in preserved_commands:
                try:
                    self._request_queue.put_nowait(pending)
                except queue.Full:
                    # Preserve pipeline control integrity over stale drag events.
                    break

            try:
                self._request_queue.put_nowait(command)
                self._record_control_send_result(
                    cmd,
                    sent=True,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    queue_full=True,
                    compaction_run=True,
                    compaction_roi_dropped=roi_compaction_drops,
                )
                return True
            except queue.Full:
                if cmd in best_effort_cmds:
                    self._record_control_send_result(
                        cmd,
                        sent=False,
                        elapsed_ms=(time.perf_counter() - started) * 1000.0,
                        queue_full=True,
                        compaction_run=True,
                        compaction_roi_dropped=roi_compaction_drops,
                    )
                    return False

        try:
            self._request_queue.put_nowait(command)
            self._record_control_send_result(
                cmd,
                sent=True,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
            return True
        except queue.Full:
            # Never evict pending critical commands. Drop only the best-effort
            # command itself (e.g. tick) and preserve queued state updates.
            if cmd in best_effort_cmds:
                self._record_control_send_result(
                    cmd,
                    sent=False,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    queue_full=True,
                )
                return False

            # For critical commands, wait briefly for queue capacity instead of
            # removing existing requests that may contain user settings changes.
            try:
                self._request_queue.put(command, timeout=0.25)
            except queue.Full:
                self._record_control_send_result(
                    cmd,
                    sent=False,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    queue_full=True,
                )
                raise RuntimeError(f"Worker request queue saturated while sending '{cmd}'")
            self._record_control_send_result(
                cmd,
                sent=True,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                queue_full=True,
            )
            return True

    def _drain_responses(self) -> None:
        if self._response_queue is None:
            return

        while True:
            try:
                message = self._response_queue.get_nowait()
            except queue.Empty:
                return

            message_type = message.get("type")
            if message_type == "frame":
                self._latest_output_frame = message["frame_bytes"]
                self._latest_effective_scale = int(message.get("effective_sr_scale", self._latest_effective_scale))
                continue

            if message_type == "decklink_frame":
                self._apply_decklink_frame_message(message)
                continue

            if message_type == "decklink_no_frame":
                self._latest_decklink_frame = None
                self._decklink_frame_updated = False
                self._decklink_timecode_info = {
                    "present": False,
                    "text": "",
                    "format_code": 0,
                    "format_name": "",
                }
                self._decklink_no_frame_reason = str(message.get("reason", "unknown"))
                self._decklink_tick_pending = False
                self._decklink_tick_pending_since = 0.0
                continue

            if message_type == "ack":
                ack_cmd = str(message.get("cmd", ""))
                if ack_cmd in {"set_basic_scaling_method", "set_sr_flavor"}:
                    self.basic_scaling_method = str(message.get("basic_scaling_method", message.get("sr_flavor", self.basic_scaling_method)))
                elif ack_cmd == "set_deinterlace_enabled":
                    self.deinterlace_enabled = bool(message.get("deinterlace_enabled", self.deinterlace_enabled))
                elif ack_cmd == "set_reinterlace_enabled":
                    self.reinterlace_enabled = bool(message.get("reinterlace_enabled", self.reinterlace_enabled))
                elif ack_cmd == "set_deinterlace_method":
                    self.deinterlace_method = str(message.get("deinterlace_method", self.deinterlace_method))
                elif ack_cmd == "set_denoise_settings":
                    self.denoise_method = str(message.get("denoise_method", self.denoise_method))
                    self.denoise_strength = float(message.get("denoise_strength", self.denoise_strength))
                elif ack_cmd in {"set_ai_sr_enabled", "set_ai_sr_model_path", "set_ai_sr_settings"}:
                    self.ai_sr_enabled = bool(message.get("ai_sr_enabled", self.ai_sr_enabled))
                    self.ai_sr_active = bool(message.get("ai_sr_active", self.ai_sr_active))
                    self.ai_sr_error = message.get("ai_sr_error")
                    self.ai_sr_info = message.get("ai_sr_info")
                elif ack_cmd in {"set_rtx_vsr_enabled", "set_rtx_vsr_settings"}:
                    self.rtx_vsr_enabled = bool(message.get("rtx_vsr_enabled", self.rtx_vsr_enabled))
                    self.rtx_vsr_active = bool(message.get("rtx_vsr_active", self.rtx_vsr_active))
                    self.rtx_vsr_error = message.get("rtx_vsr_error")
                    self.rtx_vsr_info = message.get("rtx_vsr_info")
                elif ack_cmd == "set_color_space":
                    self.color_space = _normalize_color_space_name(str(message.get("color_space", self.color_space)))
                elif ack_cmd == "set_color_range":
                    self.color_range = _normalize_color_range_name(str(message.get("color_range", self.color_range)))
                elif ack_cmd == "set_decklink_output_buffer_frames":
                    self.decklink_output_buffer_frames = max(
                        0,
                        min(10, int(message.get("decklink_output_buffer_frames", self.decklink_output_buffer_frames))),
                    )
                elif ack_cmd == "set_worker_process_priority":
                    self.worker_process_priority = _normalize_worker_priority_name(
                        str(message.get("worker_process_priority", self.worker_process_priority))
                    )
                    self.worker_process_priority_error = (
                        str(message.get("worker_process_priority_error", "")).strip() or None
                    )
                elif ack_cmd == "set_interlaced_field2_phase_fraction":
                    self.interlaced_field2_phase_fraction = _clamp_interlaced_field2_phase_fraction(
                        float(message.get("interlaced_field2_phase_fraction", self.interlaced_field2_phase_fraction))
                    )
                continue

            if message_type == "warning":
                warning_text = str(message.get("warning", ""))
                if warning_text:
                    self.ai_sr_last_warning = warning_text
                continue

            if message_type == "error":
                raise RuntimeError(
                    f"Worker runtime failure: {message.get('error')}\n{message.get('traceback', '')}"
                )

    def set_roi(self, roi: Roi) -> bool:
        return self._send_control({"cmd": "set_roi", "x": roi.x, "y": roi.y, "w": roi.w, "h": roi.h})

    def set_roi_position(self, roi_x: int, roi_y: int) -> bool:
        return self._send_control({"cmd": "set_roi_position", "x": int(roi_x), "y": int(roi_y)})

    def set_roi_subpixel_shift(self, shift_x: float, shift_y: float) -> None:
        self._send_control(
            {
                "cmd": "set_roi_subpixel_shift",
                "shift_x": float(shift_x),
                "shift_y": float(shift_y),
            }
        )

    def set_roi_with_subpixel(self, roi: Roi, shift_x: float, shift_y: float, manual_drag: bool = False) -> bool:
        return self._send_control(
            {
                "cmd": "set_roi_with_subpixel",
                "x": int(roi.x),
                "y": int(roi.y),
                "w": int(roi.w),
                "h": int(roi.h),
                "shift_x": float(shift_x),
                "shift_y": float(shift_y),
                "manual_drag": bool(manual_drag),
            }
        )

    def set_roi_manual_drag_hold_seconds(self, hold_seconds: float) -> None:
        self._send_control(
            {
                "cmd": "set_roi_manual_drag_hold_seconds",
                "hold_seconds": float(hold_seconds),
            }
        )

    def set_interlaced_field2_phase_fraction(self, fraction: float) -> None:
        clamped = _clamp_interlaced_field2_phase_fraction(float(fraction))
        self.interlaced_field2_phase_fraction = clamped
        self._send_control(
            {
                "cmd": "set_interlaced_field2_phase_fraction",
                "fraction": clamped,
            }
        )
        self._wait_for_ack("set_interlaced_field2_phase_fraction", timeout_seconds=1.0)

    def start_roi_microstep_transition(
        self,
        start_roi: Roi,
        target_roi: Roi,
        duration_frames: int,
        interpolation_mode: str,
        overscan_percent: float,
        start_from_current: bool = False,
        enforce_full_frame_scale_1x: bool = False,
    ) -> None:
        self._send_control(
            {
                "cmd": "start_roi_microstep_transition",
                "start_x": int(start_roi.x),
                "start_y": int(start_roi.y),
                "start_w": int(start_roi.w),
                "start_h": int(start_roi.h),
                "target_x": int(target_roi.x),
                "target_y": int(target_roi.y),
                "target_w": int(target_roi.w),
                "target_h": int(target_roi.h),
                "duration_frames": int(duration_frames),
                "interpolation_mode": str(interpolation_mode),
                "overscan_percent": float(overscan_percent),
                "start_from_current": bool(start_from_current),
                "enforce_full_frame_scale_1x": bool(enforce_full_frame_scale_1x),
            }
        )

    def cancel_roi_microstep_transition(self, reset_subpixel_shift: bool = True) -> None:
        self._send_control(
            {
                "cmd": "cancel_roi_microstep_transition",
                "reset_subpixel_shift": bool(reset_subpixel_shift),
            }
        )

    def set_auto_basic_scaling(self) -> None:
        self.basic_scaling_auto_mode = True
        if self.enable_basic_scaling:
            self._send_control({"cmd": "set_basic_scaling_mode_auto"})
            self._wait_for_ack("set_basic_scaling_mode_auto", timeout_seconds=1.0)

    def set_manual_basic_scaling(self, scale: int) -> None:
        self.basic_scaling_manual = scale
        self.basic_scaling_auto_mode = False
        if self.enable_basic_scaling:
            self._send_control({"cmd": "set_basic_scaling_manual", "scale": int(scale)})
            self._wait_for_ack("set_basic_scaling_manual", timeout_seconds=1.0)

    def effective_scale(self) -> int:
        return max(1, int(self._latest_effective_scale))

    def set_deinterlace_enabled(self, enabled: bool) -> None:
        self.deinterlace_enabled = enabled
        self._send_control({"cmd": "set_deinterlace_enabled", "enabled": bool(enabled)})
        self._wait_for_ack("set_deinterlace_enabled", timeout_seconds=1.0)

    def set_reinterlace_enabled(self, enabled: bool) -> None:
        self.reinterlace_enabled = bool(enabled)
        self._send_control({"cmd": "set_reinterlace_enabled", "enabled": bool(enabled)})
        self._wait_for_ack("set_reinterlace_enabled", timeout_seconds=1.0)

    def set_deinterlace_method(self, method: str) -> None:
        self.deinterlace_method = str(method)
        self._send_control({"cmd": "set_deinterlace_method", "method": self.deinterlace_method})
        self._wait_for_ack("set_deinterlace_method", timeout_seconds=1.0)

    def set_denoise_settings(self, method: str, strength: float) -> None:
        self.denoise_method = str(method)
        self.denoise_strength = max(0.0, min(1.0, float(strength)))
        self._send_control(
            {
                "cmd": "set_denoise_settings",
                "method": self.denoise_method,
                "strength": self.denoise_strength,
            }
        )
        self._wait_for_ack("set_denoise_settings", timeout_seconds=1.0)

    def set_max_auto_basic_scaling(self, scale: int) -> None:
        self.max_auto_basic_scaling = scale
        self._send_control({"cmd": "set_max_auto_basic_scaling", "scale": int(scale)})

    def set_basic_scaling_method(self, basic_scaling_method: str) -> None:
        self.basic_scaling_method = basic_scaling_method
        if self.basic_scaling_method_supported:
            self._send_control({"cmd": "set_basic_scaling_method", "basic_scaling_method": str(basic_scaling_method)})
            self._wait_for_ack("set_basic_scaling_method", timeout_seconds=1.0)

    def set_color_space(self, color_space: str) -> None:
        self.color_space = _normalize_color_space_name(color_space)
        self._send_control({"cmd": "set_color_space", "color_space": self.color_space})
        self._wait_for_ack("set_color_space", timeout_seconds=1.0)

    def set_color_range(self, color_range: str) -> None:
        self.color_range = _normalize_color_range_name(color_range)
        self._send_control({"cmd": "set_color_range", "color_range": self.color_range})
        self._wait_for_ack("set_color_range", timeout_seconds=1.0)

    # Backward-compatible aliases for existing call sites.
    def set_auto_sr(self) -> None:
        self.set_auto_basic_scaling()

    def set_manual_sr(self, scale: int) -> None:
        self.set_manual_basic_scaling(scale)

    def set_max_auto_sr_scale(self, scale: int) -> None:
        self.set_max_auto_basic_scaling(scale)

    def set_sr_flavor(self, sr_flavor: str) -> None:
        self.set_basic_scaling_method(sr_flavor)

    def _wait_for_ack(self, expected_cmd: str, timeout_seconds: float = 3.0) -> None:
        if self._response_queue is None:
            raise RuntimeError("Worker response queue is not initialized")

        deadline = time.perf_counter() + timeout_seconds
        last_warning: str | None = None
        while time.perf_counter() < deadline:
            self._assert_worker_alive()
            try:
                message = self._response_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            message_type = message.get("type")
            if message_type == "ack" and str(message.get("cmd")) == expected_cmd:
                if expected_cmd == "start_decklink":
                    started = bool(message.get("decklink_started", True))
                    if not started:
                        decklink_error = str(message.get("decklink_error", "DeckLink start failed")).strip()
                        if decklink_error:
                            raise RuntimeError(decklink_error)
                        raise RuntimeError("DeckLink start failed")
                    LOGGER.info(
                        (
                            "DeckLink output mode resolved | name=%s | mode=%s | interlaced=%s | "
                            "field_dominance_code=%s | field_dominance_name=%s"
                        ),
                        str(message.get("output_mode_name", "")),
                        str(message.get("output_mode_value", "")),
                        bool(message.get("output_mode_is_interlaced", False)),
                        str(message.get("output_field_dominance_code", "")),
                        str(message.get("output_field_dominance_name", "")),
                    )
                if expected_cmd == "set_basic_scaling_method":
                    self.basic_scaling_method = str(message.get("basic_scaling_method", message.get("sr_flavor", self.basic_scaling_method)))
                if expected_cmd == "set_sr_flavor":
                    self.basic_scaling_method = str(message.get("basic_scaling_method", message.get("sr_flavor", self.basic_scaling_method)))
                if expected_cmd == "set_deinterlace_method":
                    self.deinterlace_method = str(message.get("deinterlace_method", self.deinterlace_method))
                if expected_cmd == "set_deinterlace_enabled":
                    self.deinterlace_enabled = bool(message.get("deinterlace_enabled", self.deinterlace_enabled))
                if expected_cmd == "set_reinterlace_enabled":
                    self.reinterlace_enabled = bool(message.get("reinterlace_enabled", self.reinterlace_enabled))
                if expected_cmd == "set_denoise_settings":
                    self.denoise_method = str(message.get("denoise_method", self.denoise_method))
                    self.denoise_strength = float(message.get("denoise_strength", self.denoise_strength))
                if expected_cmd in {"set_ai_sr_enabled", "set_ai_sr_model_path", "set_ai_sr_settings"}:
                    self.ai_sr_enabled = bool(message.get("ai_sr_enabled", self.ai_sr_enabled))
                    self.ai_sr_active = bool(message.get("ai_sr_active", self.ai_sr_active))
                    self.ai_sr_error = message.get("ai_sr_error")
                    self.ai_sr_info = message.get("ai_sr_info")
                if expected_cmd in {"set_rtx_vsr_enabled", "set_rtx_vsr_settings"}:
                    self.rtx_vsr_enabled = bool(message.get("rtx_vsr_enabled", self.rtx_vsr_enabled))
                    self.rtx_vsr_active = bool(message.get("rtx_vsr_active", self.rtx_vsr_active))
                    self.rtx_vsr_error = message.get("rtx_vsr_error")
                    self.rtx_vsr_info = message.get("rtx_vsr_info")
                if expected_cmd == "set_color_space":
                    self.color_space = _normalize_color_space_name(str(message.get("color_space", self.color_space)))
                if expected_cmd == "set_color_range":
                    self.color_range = _normalize_color_range_name(str(message.get("color_range", self.color_range)))
                return
            if message_type == "error":
                raise RuntimeError(
                    f"Worker runtime failure: {message.get('error')}\n{message.get('traceback', '')}"
                )
            if message_type == "frame":
                self._latest_output_frame = message["frame_bytes"]
                self._latest_effective_scale = int(message.get("effective_sr_scale", self._latest_effective_scale))
                continue
            if message_type == "decklink_frame":
                self._apply_decklink_frame_message(message)
                continue
            if message_type == "decklink_no_frame":
                self._latest_decklink_frame = None
                self._decklink_frame_updated = False
                self._decklink_timecode_info = {
                    "present": False,
                    "text": "",
                    "format_code": 0,
                    "format_name": "",
                }
                self._decklink_no_frame_reason = str(message.get("reason", "unknown"))
                self._decklink_tick_pending = False
                self._decklink_tick_pending_since = 0.0
                continue
            if message_type == "warning":
                warning_text = str(message.get("warning", ""))
                if warning_text:
                    self.ai_sr_last_warning = warning_text
                    last_warning = warning_text
                continue

        diag_parts = [f"expected_cmd={expected_cmd}"]
        if self.ai_sr_error:
            diag_parts.append(f"ai_sr_error={self.ai_sr_error}")
        if last_warning:
            diag_parts.append(f"last_warning={last_warning}")
        if self._decklink_no_frame_reason:
            diag_parts.append(f"decklink_no_frame_reason={self._decklink_no_frame_reason}")
        raise RuntimeError(f"Timed out waiting for worker ack: {expected_cmd} | {' | '.join(diag_parts)}")

    def start_decklink(self, in_device: int, in_mode: object, out_device: int, out_mode: object, enable_format_detection: bool) -> None:
        self._drain_responses()
        self._latest_decklink_frame = None
        self._decklink_frame_updated = False
        self._decklink_no_frame_reason = None
        self._decklink_tick_pending = False
        self._decklink_tick_pending_since = 0.0
        self._decklink_tick_counter = 0
        self._last_preview_request_ts = 0.0
        self._reset_decklink_fps_tracking()
        self._send_control(
            {
                "cmd": "start_decklink",
                "in_device": int(in_device),
                "in_mode": in_mode,
                "out_device": int(out_device),
                "out_mode": out_mode,
                "enable_format_detection": bool(enable_format_detection),
                "decklink_output_buffer_frames": int(self.decklink_output_buffer_frames),
            }
        )
        self._wait_for_ack("start_decklink", timeout_seconds=12.0)

    def stop_decklink(self) -> None:
        if self._process is None:
            return
        self._send_control({"cmd": "stop_decklink"})
        try:
            self._wait_for_ack("stop_decklink", timeout_seconds=1.5)
        except Exception:
            pass
        self._latest_decklink_frame = None
        self._decklink_frame_updated = False
        self._decklink_tick_pending = False
        self._decklink_tick_pending_since = 0.0
        self._decklink_tick_counter = 0
        self._last_preview_request_ts = 0.0
        self._reset_decklink_fps_tracking()

    def decklink_tick(self, timeout_ms: int = 50) -> tuple[bytes, bytes] | None:
        self._drain_responses()

        if self._decklink_tick_pending and self._decklink_tick_pending_since > 0.0:
            if (time.perf_counter() - self._decklink_tick_pending_since) >= 0.75:
                self._decklink_tick_pending = False
                self._decklink_tick_pending_since = 0.0
                self._decklink_no_frame_reason = "tick_request_stalled"

        if not self._decklink_tick_pending:
            # Keep at most one in-flight tick request so stale tick commands cannot
            # build up and push preview display several seconds behind live processing.
            self._decklink_tick_counter += 1
            include_frames = True
            if self._gpu_live_mode:
                now = time.perf_counter()
                include_frames = False
                if self._latest_decklink_frame is None:
                    include_frames = True
                elif self._preview_fps > 0.0 and (now - self._last_preview_request_ts) >= (1.0 / self._preview_fps):
                    include_frames = True
                if include_frames:
                    self._last_preview_request_ts = now
            else:
                include_frames = (self._decklink_tick_counter % self._decklink_preview_interval) == 0
                if self._latest_decklink_frame is None:
                    include_frames = True
            sent = self._send_control(
                {
                    "cmd": "decklink_tick",
                    "timeout_ms": int(timeout_ms),
                    "include_frames": bool(include_frames),
                }
            )
            if sent:
                self._decklink_tick_pending = True
                self._decklink_tick_pending_since = time.perf_counter()
            else:
                self._decklink_tick_pending = False
                self._decklink_tick_pending_since = 0.0
                self._decklink_no_frame_reason = "tick_dropped_queue_full"
        self._drain_responses()
        if self._decklink_no_frame_reason in {"tick_request_stalled", "tick_dropped_queue_full"}:
            return None
        return self._latest_decklink_frame

    def decklink_no_frame_reason(self) -> str | None:
        return self._decklink_no_frame_reason

    def decklink_processed_counter(self) -> int:
        return int(self._decklink_processed_counter)

    def decklink_processed_fps(self) -> float:
        return float(self._decklink_processed_fps)

    def decklink_output_nominal_fps(self) -> float:
        return float(self._decklink_output_nominal_fps)

    def decklink_output_is_interlaced(self) -> bool:
        return bool(self._decklink_output_is_interlaced)

    def decklink_transition_units_per_output_frame(self) -> float:
        return float(self._decklink_transition_units_per_output_frame)

    def set_preview_fps(self, preview_fps: float) -> None:
        self._preview_fps = max(0.0, float(preview_fps))
        # Allow an immediate preview request after a user-adjusted FPS change.
        self._last_preview_request_ts = 0.0

    def set_decklink_output_buffer_frames(self, buffer_frames: int) -> None:
        self.decklink_output_buffer_frames = max(0, min(10, int(buffer_frames)))
        self._send_control(
            {
                "cmd": "set_decklink_output_buffer_frames",
                "decklink_output_buffer_frames": int(self.decklink_output_buffer_frames),
            }
        )

    def set_worker_process_priority(self, priority_name: str) -> None:
        normalized = _normalize_worker_priority_name(priority_name)
        self.worker_process_priority = normalized
        self._send_control(
            {
                "cmd": "set_worker_process_priority",
                "worker_process_priority": normalized,
            }
        )

    def consume_decklink_frame_updated(self) -> bool:
        updated = bool(self._decklink_frame_updated)
        self._decklink_frame_updated = False
        return updated

    def process_frame(self, frame_bytes: bytes) -> bytes:
        self._drain_responses()
        self._assert_worker_alive()
        if self._request_queue is None:
            raise RuntimeError("Worker request queue is not initialized")

        frame_id = self._next_frame_id
        self._next_frame_id += 1

        frame_message = {
            "cmd": "process_frame",
            "frame_id": frame_id,
            "frame_bytes": frame_bytes,
        }

        try:
            self._request_queue.put_nowait(frame_message)
        except queue.Full:
            try:
                self._request_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._request_queue.put_nowait(frame_message)
            except queue.Full:
                # Keep GUI responsive when worker is saturated; reuse latest output.
                return self._latest_output_frame if self._latest_output_frame is not None else frame_bytes

        self._drain_responses()
        if self._latest_output_frame is None:
            return frame_bytes
        return self._latest_output_frame

    def close(self) -> None:
        try:
            self.stop_decklink()
        except Exception:
            pass

        if self._request_queue is not None:
            try:
                self._request_queue.put_nowait({"cmd": "shutdown"})
            except Exception:
                pass

        if self._process is not None:
            self._process.join(timeout=1.5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)

        self._process = None
        self._request_queue = None
        self._response_queue = None
        self._roi_telemetry_shared = None
        self._roi_telemetry_seq = None
        self._roi_telemetry_last_seq = -1
        self._decklink_tick_pending = False
        self._decklink_tick_pending_since = 0.0

    def set_ai_sr_enabled(self, enabled: bool, wait_for_ack: bool = False, timeout_seconds: float = 3.0) -> None:
        self.ai_sr_enabled = bool(enabled)
        self._send_control({"cmd": "set_ai_sr_enabled", "enabled": bool(enabled)})
        if wait_for_ack:
            self._wait_for_ack("set_ai_sr_enabled", timeout_seconds=max(0.5, float(timeout_seconds)))
        # Default behavior remains non-blocking for interactive toggles.

    def set_ai_sr_model_path(self, model_path: str, wait_for_ack: bool = False, timeout_seconds: float = 3.0) -> None:
        self.ai_sr_model_path = str(model_path)
        self._send_control({"cmd": "set_ai_sr_model_path", "model_path": self.ai_sr_model_path})
        if wait_for_ack:
            self._wait_for_ack("set_ai_sr_model_path", timeout_seconds=max(0.5, float(timeout_seconds)))
        # Default behavior remains non-blocking for interactive updates.

    def set_ai_sr_settings(
        self,
        provider: str,
        require_gpu: bool,
        inference_fps: int,
        trt_precision: str,
        strict: bool,
        input_align: int,
        roi_overscan_percent: float,
        inference_divisor: int,
        detail_preserve_percent: float,
        post_denoise_method: str,
        post_denoise_strength: float,
        post_artifact_reduction_method: str,
        post_artifact_reduction_strength: float,
        post_exaggeration_enabled: bool,
        post_exaggeration_gain: float,
        max_inflight: int | None = None,
        wait_for_ack: bool = False,
        timeout_seconds: float = 3.0,
    ) -> None:
        self.ai_sr_provider = str(provider)
        trt_precision_name = str(trt_precision).strip().lower()
        self.ai_sr_trt_precision = "int8" if trt_precision_name == "int8" else "fp16"
        self.ai_sr_require_gpu = bool(require_gpu)
        self.ai_sr_frame_interval = max(1, min(60, int(inference_fps)))
        self.ai_sr_strict = bool(strict)
        self.ai_sr_input_align = max(1, int(input_align))
        self.ai_sr_roi_overscan_percent = max(0.0, float(roi_overscan_percent))
        self.ai_sr_inference_divisor = max(0, int(inference_divisor))
        self.ai_sr_detail_preserve_percent = max(0.0, float(detail_preserve_percent))
        self.ai_sr_post_denoise_method = str(post_denoise_method).strip().lower()
        self.ai_sr_post_denoise_strength = max(0.0, min(1.0, float(post_denoise_strength)))
        self.ai_sr_post_artifact_reduction_method = str(post_artifact_reduction_method).strip().lower()
        self.ai_sr_post_artifact_reduction_strength = max(0.0, min(1.0, float(post_artifact_reduction_strength)))
        self.ai_sr_post_exaggeration_enabled = bool(post_exaggeration_enabled)
        self.ai_sr_post_exaggeration_gain = max(1.0, min(4.0, float(post_exaggeration_gain)))
        if max_inflight is not None:
            self.ai_sr_max_inflight = max(1, min(4, int(max_inflight)))
        self._send_control(
            {
                "cmd": "set_ai_sr_settings",
                "provider": self.ai_sr_provider,
                "trt_precision": self.ai_sr_trt_precision,
                "require_gpu": self.ai_sr_require_gpu,
                "frame_interval": self.ai_sr_frame_interval,
                "inference_fps": self.ai_sr_frame_interval,
                "strict": self.ai_sr_strict,
                "input_align": self.ai_sr_input_align,
                "roi_overscan_percent": self.ai_sr_roi_overscan_percent,
                "inference_divisor": self.ai_sr_inference_divisor,
                "detail_preserve_percent": self.ai_sr_detail_preserve_percent,
                "post_denoise_method": self.ai_sr_post_denoise_method,
                "post_denoise_strength": self.ai_sr_post_denoise_strength,
                "post_artifact_reduction_method": self.ai_sr_post_artifact_reduction_method,
                "post_artifact_reduction_strength": self.ai_sr_post_artifact_reduction_strength,
                "post_exaggeration_enabled": self.ai_sr_post_exaggeration_enabled,
                "post_exaggeration_gain": self.ai_sr_post_exaggeration_gain,
                "max_inflight": self.ai_sr_max_inflight,
            }
        )
        if wait_for_ack:
            self._wait_for_ack("set_ai_sr_settings", timeout_seconds=max(0.5, float(timeout_seconds)))

    def set_rtx_vsr_enabled(self, enabled: bool) -> None:
        self.rtx_vsr_enabled = bool(enabled)
        self._send_control({"cmd": "set_rtx_vsr_enabled", "enabled": self.rtx_vsr_enabled})

    def set_rtx_vsr_settings(
        self,
        quality: str,
        scale: int,
        post_scale_method: str,
        thdr_enabled: bool,
        thdr_contrast: int,
        thdr_saturation: int,
        thdr_middle_gray: int,
        thdr_max_luminance: int,
    ) -> None:
        self.rtx_vsr_quality = str(quality).strip().lower()
        self.rtx_vsr_scale = max(1, int(scale))
        self.rtx_vsr_post_scale_method = str(post_scale_method).strip().lower() or "bicubic"
        self.rtx_thdr_enabled = bool(thdr_enabled)
        self.rtx_thdr_contrast = max(0, int(thdr_contrast))
        self.rtx_thdr_saturation = max(0, int(thdr_saturation))
        self.rtx_thdr_middle_gray = max(0, int(thdr_middle_gray))
        self.rtx_thdr_max_luminance = max(0, int(thdr_max_luminance))
        self._send_control(
            {
                "cmd": "set_rtx_vsr_settings",
                "quality": self.rtx_vsr_quality,
                "scale": self.rtx_vsr_scale,
                "post_scale_method": self.rtx_vsr_post_scale_method,
                "thdr_enabled": self.rtx_thdr_enabled,
                "thdr_contrast": self.rtx_thdr_contrast,
                "thdr_saturation": self.rtx_thdr_saturation,
                "thdr_middle_gray": self.rtx_thdr_middle_gray,
                "thdr_max_luminance": self.rtx_thdr_max_luminance,
            }
        )

    @property
    def enable_placeholder_sr(self) -> bool:
        return bool(self.enable_basic_scaling)

    @enable_placeholder_sr.setter
    def enable_placeholder_sr(self, value: bool) -> None:
        self.enable_basic_scaling = bool(value)

    def decklink_ai_sr_counts(self) -> tuple[int, int, int]:
        return (
            int(self._decklink_ai_applied_frames),
            int(self._decklink_ai_reused_frames),
            int(self._decklink_ai_passthrough_frames),
        )

    def decklink_ai_refresh_stats(self) -> tuple[float, float, int]:
        return (
            float(self._decklink_ai_refresh_fps),
            float(self._decklink_ai_latest_age_ms),
            int(self._decklink_ai_completed_frames),
        )

    def decklink_ai_timing_stats(self) -> dict[str, object]:
        return dict(self._decklink_ai_timing_ms)

    def decklink_rtx_stats(self) -> tuple[bool, float]:
        return bool(self._decklink_rtx_vsr_applied), float(self._decklink_rtx_effect_mean_abs_luma)

    def decklink_stage_telemetry(self) -> tuple[dict[str, bool], dict[str, bool], dict[str, int]]:
        return (
            dict(self._decklink_stage_enable_flags),
            dict(self._decklink_stage_last_applied),
            dict(self._decklink_stage_apply_counts),
        )

    @property
    def sr_flavor(self) -> str:
        return self.basic_scaling_method

    @sr_flavor.setter
    def sr_flavor(self, value: str) -> None:
        self.basic_scaling_method = str(value)

    @property
    def max_auto_sr_scale(self) -> int:
        return int(self.max_auto_basic_scaling)

    @max_auto_sr_scale.setter
    def max_auto_sr_scale(self, value: int) -> None:
        self.max_auto_basic_scaling = int(value)

    @property
    def sr_manual_scale(self) -> int:
        return int(self.basic_scaling_manual)

    @sr_manual_scale.setter
    def sr_manual_scale(self, value: int) -> None:
        self.basic_scaling_manual = int(value)

    @property
    def sr_auto_mode(self) -> bool:
        return bool(self.basic_scaling_auto_mode)

    @sr_auto_mode.setter
    def sr_auto_mode(self, value: bool) -> None:
        self.basic_scaling_auto_mode = bool(value)

    @property
    def sr_flavor_supported(self) -> bool:
        return bool(self.basic_scaling_method_supported)

    @sr_flavor_supported.setter
    def sr_flavor_supported(self, value: bool) -> None:
        self.basic_scaling_method_supported = bool(value)


class MainWindow(QMainWindow):
    def __init__(self, module) -> None:
        super().__init__()
        self.setWindowTitle("video_processor GUI Test Harness")

        self._module = module
        self._source = SyntheticUyvySource()
        self._input_canvas = RoiCanvas(view_name="input")
        self._output_canvas = ImageCanvas(view_name="output")
        self._controller_backend = "in-process"
        self._module = module
        self._controller = self._create_processor_controller(module)
        self._roi = Roi(0, 0, FRAME_W, FRAME_H)
        try:
            self._controller.create(self._roi)
        except Exception as exc:
            LOGGER.warning("Primary controller create failed (%s); switching to in-process backend", exc)
            self._controller = VideoProcessorController(self._module)
            self._controller_backend = "in-process"
            self._controller.create(self._roi)
        self._source_mode = "Blackmagic DeckLink"
        self._capture_session = None
        self._output_session = None
        self._decklink_sessions_running = False
        self._last_frame_error: str | None = None
        self._no_frame_counter = 0
        self._decklink_timecode_display_text = "Timecode: --"
        self._roi_smoothing_percent = 4
        self._roi_latency_smoothing_percent = 0
        self._roi_drag_x_hysteresis_px = max(0.10, min(1.20, float(os.environ.get("VP_ROI_DRAG_X_HYSTERESIS_PX", "0.45"))))
        self._roi_manual_drag_hold_s = max(0.05, min(0.50, float(os.environ.get("VP_ROI_MANUAL_DRAG_HOLD_S", "0.24"))))
        self._roi_min_drag_nudge = max(0.0, min(0.50, float(os.environ.get("VP_ROI_MIN_DRAG_NUDGE", "0.035"))))
        self._interlaced_field2_phase_fraction = _clamp_interlaced_field2_phase_fraction(
            float(os.environ.get("VP_INTERLACED_FIELD2_PHASE_FRACTION", "0.50"))
        )
        self._manual_drag_worker_send_hz = max(60.0, min(120.0, float(os.environ.get("VP_MANUAL_DRAG_WORKER_SEND_HZ", "90"))))
        self._manual_drag_worker_send_interval_ms = int(round(1000.0 / self._manual_drag_worker_send_hz))
        self._manual_roi_frame_lock_to_output = os.environ.get("VP_MANUAL_ROI_FRAME_LOCK", "1") != "0"
        self._manual_roi_last_send_ts = 0.0
        self._manual_roi_target_history: list[tuple[float, float, float]] = []
        self._roi_keyframes: dict[int, RoiKeyframe] = {}
        self._roi_keyframe_slots = (1, 2, 3, 4)
        self._roi_key_save_armed = False
        self._roi_keyframe_transition_default_frames = 30
        self._roi_keyframe_transition: dict[str, object] | None = None
        self._roi_keyframe_last_step_ts = 0.0
        self._roi_keyframe_target_fps = 60.0
        self._roi_keyframe_transition_overscan_percent = 2.0
        self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        self._controller_filtered_target_roi: Roi | None = None
        self._last_status_text: str | None = None
        self._last_status_log_ts = 0.0
        self._status_repeat_log_interval_s = 5.0
        self._input_canvas.set_smoothing_percent(self._roi_smoothing_percent)
        self._input_canvas.set_latency_smoothing_percent(self._roi_latency_smoothing_percent)
        self._input_canvas.set_drag_x_hysteresis_px(self._roi_drag_x_hysteresis_px)

        self._last_stat_time = time.perf_counter()
        self._frame_count = 0
        self._perf_stage_sums_ms = {
            "acquire": 0.0,
            "process": 0.0,
            "output": 0.0,
            "convert_in": 0.0,
            "convert_out": 0.0,
            "tick": 0.0,
        }
        self._perf_stage_counts = {
            "acquire": 0,
            "process": 0,
            "output": 0,
            "convert_in": 0,
            "convert_out": 0,
            "tick": 0,
        }
        self._perf_stage_peaks_ms = {
            "acquire": 0.0,
            "process": 0.0,
            "output": 0.0,
            "convert_in": 0.0,
            "convert_out": 0.0,
            "tick": 0.0,
        }
        self._perf_guard_enabled = False
        self._perf_guard_low_fps_seconds = 0
        self._perf_guard_last_action = ""
        self._health_drop_events_total = 0
        self._health_drop_events_interpolation = 0
        self._health_buffer_warn_events = 0
        self._health_drop_active = False
        self._health_drop_active_interpolation = False
        self._health_last_output_fps = 0.0
        self._health_last_output_nominal_fps = 0.0
        self._health_last_buffer_starvation = 0
        self._health_last_buffer_overflow = 0
        self._health_last_buffer_reprime = 0
        self._decklink_buffer_guard_enabled = os.environ.get("VP_DECKLINK_BUFFER_GUARD", "1") != "0"
        self._decklink_buffer_guard_floor_frames = max(
            1,
            min(10, int(os.environ.get("VP_DECKLINK_BUFFER_GUARD_FLOOR", "2"))),
        )
        self._decklink_buffer_guard_transition_floor_frames = max(
            int(self._decklink_buffer_guard_floor_frames),
            min(10, int(os.environ.get("VP_DECKLINK_BUFFER_GUARD_TRANSITION_FLOOR", "4"))),
        )
        self._decklink_buffer_guard_engage_miss_ratio = max(
            0.0,
            min(1.0, float(os.environ.get("VP_DECKLINK_BUFFER_GUARD_ENGAGE_MISS_RATIO", "0.06"))),
        )
        self._decklink_buffer_guard_release_miss_ratio = max(
            0.0,
            min(1.0, float(os.environ.get("VP_DECKLINK_BUFFER_GUARD_RELEASE_MISS_RATIO", "0.01"))),
        )
        self._decklink_buffer_guard_release_windows_needed = max(
            1,
            int(os.environ.get("VP_DECKLINK_BUFFER_GUARD_RELEASE_WINDOWS", "4")),
        )
        self._decklink_buffer_guard_active = False
        self._decklink_buffer_guard_stable_windows = 0
        self._worker_process_priority = _normalize_worker_priority_name(
            getattr(self._controller, "worker_process_priority", os.environ.get("VP_WORKER_PROCESS_PRIORITY", "above_normal"))
        )
        self._updating_controls = False
        self._controller_roi_target: Roi | None = None
        self._controller_roi_applied = self._roi
        self._manual_live_target_roi: Roi | None = None
        self._pending_manual_controller_roi: Roi | None = None
        self._manual_drag_interp_start_overlay: tuple[float, float, float, float] | None = None
        self._manual_drag_interp_end_overlay: tuple[float, float, float, float] | None = None
        self._manual_drag_interp_started_ts = 0.0
        self._manual_drag_interp_duration_s = 1.0 / 90.0
        self._manual_drag_last_event_ts = 0.0
        self._pending_roi_controls_sync: Roi | None = None
        self._last_manual_roi_update_ts = 0.0
        self._manual_roi_preview_reduce_scale = max(
            0.35,
            min(1.0, float(os.environ.get("VP_MANUAL_ROI_PREVIEW_SCALE", "0.60"))),
        )
        self._roi_diag_canvas_events = 0
        self._roi_diag_controller_send_attempts = 0
        self._roi_diag_controller_send_success = 0
        self._roi_diag_controller_send_drops = 0
        self._roi_diag_controller_send_ms_sum = 0.0
        self._roi_diag_controller_send_ms_max = 0.0
        self._last_interlaced_phase_log_signature = ""
        self._fullscreen_view_name: str | None = None
        self._splitter_initialized = False
        self._main_splitter_initialized = False
        self._is_closing = False
        self._pending_persisted_input_device = None
        self._pending_persisted_output_device = None
        self._pending_persisted_input_mode_text = ""
        self._pending_persisted_output_mode_text = ""
        self._has_persisted_deinterlace_method = False
        self._deinterlace_method_user_selected = False
        self._windowed_geometry_before_fullscreen: QRect | None = None
        self._windowed_was_maximized_before_fullscreen = False
        self._settings_path = Path(__file__).resolve().parent / "app_settings.json"
        self._settings_save_timer = QTimer(self)
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.setInterval(250)
        self._settings_save_timer.timeout.connect(self._save_settings)
        self._decklink_buffer_reapply_timer = QTimer(self)
        self._decklink_buffer_reapply_timer.setSingleShot(True)
        self._decklink_buffer_reapply_timer.setInterval(250)
        self._decklink_buffer_reapply_timer.timeout.connect(self._reapply_decklink_after_buffer_change)
        self._decklink_color_reapply_timer = QTimer(self)
        self._decklink_color_reapply_timer.setSingleShot(True)
        self._decklink_color_reapply_timer.setInterval(300)
        self._decklink_color_reapply_timer.timeout.connect(self._reapply_decklink_after_color_change)
        self._ai_sr_profiles_path = Path(__file__).resolve().parent / "ai_sr_profiles.json"
        self._ai_sr_profiles = self._load_ai_sr_profiles()
        self._preview_downsample_factor = self._normalize_preview_downsample_factor(
            float(os.environ.get("VP_PREVIEW_DOWNSAMPLE", "0.25"))
        )
        self._decklink_tick_poll_fps = max(1.0, float(os.environ.get("VP_DECKLINK_TICK_POLL_FPS", "90")))
        self._decklink_output_buffer_frames = max(
            0,
            min(10, int(getattr(self._controller, "decklink_output_buffer_frames", 2))),
        )
        self._decklink_output_buffer_user_target_frames = int(self._decklink_output_buffer_frames)

        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        self._fullscreen_keyframe_toolbars: dict[str, QWidget] = {}
        self._fullscreen_keyframe_side_panels: dict[str, QWidget] = {}
        self._fullscreen_roi_save_key_buttons: dict[str, QPushButton] = {}
        self._fullscreen_roi_key_slot_buttons: dict[str, tuple[QPushButton, QPushButton, QPushButton, QPushButton]] = {}
        self._fullscreen_roi_transition_labels: dict[str, QLabel] = {}
        self._fullscreen_roi_transition_rate_spins: dict[str, QSpinBox] = {}
        self._fullscreen_roi_duration_override_buttons: dict[str, QPushButton] = {}
        self._fullscreen_decklink_output_buffer_spins: dict[str, QSpinBox] = {}
        self._fullscreen_enter_buttons: dict[str, QPushButton] = {}
        viewers = QWidget()
        viewers_layout = QVBoxLayout(viewers)
        viewers_layout.setContentsMargins(0, 0, 0, 0)
        viewers_layout.setSpacing(4)

        self._input_panel = QWidget()
        input_layout = QVBoxLayout(self._input_panel)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(2)
        self._input_header = QWidget()
        input_header_layout = QHBoxLayout(self._input_header)
        input_header_layout.setContentsMargins(0, 0, 0, 0)
        input_header_layout.setSpacing(8)
        self._input_title_label = QLabel("Input View")
        input_header_layout.addWidget(self._input_title_label)
        input_fullscreen_btn = QPushButton("Full screen")
        input_fullscreen_btn.clicked.connect(lambda: self._set_fullscreen_view("input"))
        input_header_layout.addWidget(input_fullscreen_btn)
        input_header_layout.addStretch(1)
        self._fullscreen_enter_buttons["input"] = input_fullscreen_btn
        input_layout.addWidget(self._input_header)

        self._input_viewer_row = QWidget()
        input_viewer_row_layout = QHBoxLayout(self._input_viewer_row)
        input_viewer_row_layout.setContentsMargins(0, 0, 0, 0)
        input_viewer_row_layout.setSpacing(12)
        self._input_fullscreen_keyframe_side_panel = self._build_fullscreen_keyframe_side_panel("input")
        input_viewer_row_layout.addWidget(self._input_fullscreen_keyframe_side_panel, 0, alignment=Qt.AlignTop)
        input_viewer_row_layout.addWidget(self._input_canvas, 1, alignment=Qt.AlignCenter)
        input_layout.addWidget(self._input_viewer_row, 1)

        self._input_fullscreen_keyframe_toolbar = self._build_fullscreen_keyframe_toolbar("input")
        input_layout.addWidget(self._input_fullscreen_keyframe_toolbar)

        self._output_panel = QWidget()
        output_layout = QVBoxLayout(self._output_panel)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(2)
        self._output_header = QWidget()
        output_header_layout = QHBoxLayout(self._output_header)
        output_header_layout.setContentsMargins(0, 0, 0, 0)
        output_header_layout.setSpacing(8)
        self._output_title_label = QLabel("Output View")
        output_header_layout.addWidget(self._output_title_label)
        output_fullscreen_btn = QPushButton("Full screen")
        output_fullscreen_btn.clicked.connect(lambda: self._set_fullscreen_view("output"))
        output_header_layout.addWidget(output_fullscreen_btn)
        output_header_layout.addStretch(1)
        self._fullscreen_enter_buttons["output"] = output_fullscreen_btn
        output_layout.addWidget(self._output_header)

        self._output_viewer_row = QWidget()
        output_viewer_row_layout = QHBoxLayout(self._output_viewer_row)
        output_viewer_row_layout.setContentsMargins(0, 0, 0, 0)
        output_viewer_row_layout.setSpacing(12)
        self._output_fullscreen_keyframe_side_panel = self._build_fullscreen_keyframe_side_panel("output")
        output_viewer_row_layout.addWidget(self._output_fullscreen_keyframe_side_panel, 0, alignment=Qt.AlignTop)
        output_viewer_row_layout.addWidget(self._output_canvas, 1, alignment=Qt.AlignCenter)
        output_layout.addWidget(self._output_viewer_row, 1)

        self._output_fullscreen_keyframe_toolbar = self._build_fullscreen_keyframe_toolbar("output")
        output_layout.addWidget(self._output_fullscreen_keyframe_toolbar)

        self._display_splitter = QSplitter(Qt.Vertical)
        self._display_splitter.setChildrenCollapsible(False)
        self._display_splitter.addWidget(self._input_panel)
        self._display_splitter.addWidget(self._output_panel)
        self._display_splitter.setStretchFactor(0, 1)
        self._display_splitter.setStretchFactor(1, 1)
        self._display_splitter.splitterMoved.connect(lambda _pos, _index: self._fit_viewers_to_video_aspect())
        viewers_layout.addWidget(self._display_splitter, 1)

        self._controls_panel = self._build_controls()
        self._controls_scroll = QScrollArea()
        self._controls_scroll.setWidgetResizable(True)
        self._controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._controls_scroll.setWidget(self._controls_panel)
        self._controls_scroll.setMinimumWidth(420)

        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.addWidget(viewers)
        self._main_splitter.addWidget(self._controls_scroll)
        self._main_splitter.setStretchFactor(0, 4)
        self._main_splitter.setStretchFactor(1, 1)
        root.addWidget(self._main_splitter, 1)

        self._input_canvas.set_roi(self._roi)
        self._input_canvas.roiChanged.connect(self._on_roi_from_canvas)
        self._input_canvas.scaleChanged.connect(self._on_scale_from_canvas)
        self._input_canvas.fullscreenRequested.connect(self._on_canvas_fullscreen_requested)
        self._output_canvas.fullscreenRequested.connect(self._on_canvas_fullscreen_requested)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._update_timer_interval()
        self._timer.start()

        self._controller_roi_interp_timer = QTimer(self)
        self._controller_roi_interp_timer.setInterval(16)
        self._controller_roi_interp_timer.timeout.connect(self._step_controller_roi_interpolation)

        self._manual_roi_send_timer = QTimer(self)
        self._manual_roi_send_timer.setSingleShot(True)
        self._manual_roi_send_timer.setInterval(16)
        self._manual_roi_send_timer.timeout.connect(self._flush_pending_manual_controller_roi)

        self._roi_controls_sync_timer = QTimer(self)
        self._roi_controls_sync_timer.setSingleShot(True)
        self._roi_controls_sync_timer.setInterval(33)
        self._roi_controls_sync_timer.timeout.connect(self._flush_pending_roi_controls_sync)

        self._roi_keyframe_transition_timer = QTimer(self)
        self._roi_keyframe_transition_timer.setInterval(16)
        self._roi_keyframe_transition_timer.setTimerType(Qt.PreciseTimer)
        self._roi_keyframe_transition_timer.timeout.connect(self._step_roi_keyframe_transition)

        self._setup_shortcuts()
        self._connect_settings_persistence_signals()
        self.roi_transition_frames_spin.valueChanged.connect(self._sync_fullscreen_transition_rate_from_main)
        self.roi_keyframe_duration_override_btn.toggled.connect(self._sync_fullscreen_override_duration_from_main)
        self.decklink_output_buffer_spin.valueChanged.connect(self._sync_fullscreen_decklink_output_buffer_from_main)
        self._update_roi_key_buttons()
        self._sync_fullscreen_transition_rate_from_main(self.roi_transition_frames_spin.value())
        self._sync_fullscreen_override_duration_from_main(self.roi_keyframe_duration_override_btn.isChecked())
        self._sync_fullscreen_decklink_output_buffer_from_main(self.decklink_output_buffer_spin.value())
        self._sync_fullscreen_button_states()
        self._sync_roi_transition_unit_labels()
        self._sync_controls_from_roi(self._roi)
        self._load_settings()
        self._apply_manual_drag_tuning_to_controller()
        self._apply_interlaced_phase_tuning_to_controller()
        self._sync_ai_sr_basic_scaling_ui(notify=False)
        self._apply_startup_ai_sr_settings()
        self._source_mode = self.source_mode_combo.currentText()
        self._sync_blackmagic_controls_enabled_state()
        self._on_source_mode_changed()
        self._sync_roi_transition_unit_labels()
        self._refresh_ai_sr_runtime_panel()
        self._refresh_rtx_vsr_runtime_panel()
        self._update_status("Ready")
        if self._controller_backend == "worker-process":
            self._update_status("Ready | Processing backend: worker process")
        else:
            self._update_status("Ready | Processing backend: in-process")
            self.decklink_status_label.setText("Worker backend not active; running in GUI process")
        LOGGER.info("GUI initialized; default source mode=%s", self._source_mode)
        QTimer.singleShot(0, self._apply_initial_viewer_layout)

    def _create_processor_controller(self, module):
        if run_processor_worker is not None:
            self._controller_backend = "worker-process"
            LOGGER.info("Using worker-process video processor backend")
            return ProcessVideoProcessorController()

        self._controller_backend = "in-process"
        if _worker_import_error is not None:
            LOGGER.warning("Worker backend import failed; using in-process backend: %s", _worker_import_error)
        else:
            LOGGER.info("Worker backend unavailable; using in-process backend")
        return VideoProcessorController(module)

    def _recreate_worker_controller(self) -> None:
        if self._controller_backend != "worker-process":
            return

        try:
            self._controller.close()
        except Exception:
            pass

        self._controller = self._create_processor_controller(self._module)
        if hasattr(self._controller, "worker_process_priority"):
            self._controller.worker_process_priority = _normalize_worker_priority_name(self._worker_process_priority)
        self._controller.create(self._roi)
        self._sync_ai_sr_basic_scaling_ui(notify=False)
        self._apply_startup_ai_sr_settings()
        self._apply_controller_color_settings_from_ui()
        self._apply_worker_process_priority_to_controller(notify=False)
        self._apply_manual_drag_tuning_to_controller()
        self._apply_interlaced_phase_tuning_to_controller()
        LOGGER.info("Worker controller recreated after unexpected worker exit")

    def _apply_controller_color_settings_from_ui(self) -> None:
        selected_space_label = self.color_space_combo.currentText()
        selected_space_name = COLOR_SPACE_LABEL_TO_NAME.get(selected_space_label, "rec709")
        self._controller.set_color_space(selected_space_name)

        selected_range_label = self.color_range_combo.currentText()
        selected_range_name = COLOR_RANGE_LABEL_TO_NAME.get(selected_range_label, "limited")
        self._controller.set_color_range(selected_range_name)

    def _default_ai_sr_model_path(self) -> str:
        return str(Path(__file__).resolve().parents[1] / "models" / "efrlfn_x2.onnx")

    def _load_ai_sr_profiles(self) -> dict[str, dict[str, object]]:
        try:
            if not self._ai_sr_profiles_path.exists():
                return {}
            raw = json.loads(self._ai_sr_profiles_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return {
                    str(model_path): dict(profile)
                    for model_path, profile in raw.items()
                    if isinstance(model_path, str) and isinstance(profile, dict)
                }
        except Exception as exc:
            LOGGER.warning("Failed to load AI SR profiles: %s", exc)
        return {}

    def _save_ai_sr_profiles(self) -> None:
        try:
            self._ai_sr_profiles_path.write_text(
                json.dumps(self._ai_sr_profiles, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as exc:
            LOGGER.warning("Failed to save AI SR profiles: %s", exc)

    def _connect_settings_persistence_signals(self) -> None:
        combo_widgets = [
            self.preview_downsample_combo,
            self.color_space_combo,
            self.color_range_combo,
            self.sr_mode_combo,
            self.sr_flavor_combo,
            self.sr_manual_combo,
            self.auto_sr_max_combo,
            self.deinterlace_method_combo,
            self.denoise_method_combo,
            self.ai_sr_provider_combo,
            self.ai_sr_trt_precision_combo,
            self.ai_sr_input_align_combo,
            self.ai_sr_post_denoise_method_combo,
            self.ai_sr_post_artifact_reduction_method_combo,
            self.rtx_vsr_quality_combo,
            self.rtx_vsr_scale_combo,
            self.rtx_vsr_post_scale_method_combo,
            self.source_mode_combo,
            self.decklink_input_device_combo,
            self.decklink_output_device_combo,
            self.decklink_input_mode_combo,
            self.decklink_output_mode_combo,
            self.worker_priority_combo,
            self.roi_interp_mode_combo,
        ]
        for combo in combo_widgets:
            combo.currentTextChanged.connect(self._schedule_settings_save)

        checkbox_widgets = [
            self.enable_sr_checkbox,
            self.enable_ai_sr_checkbox,
            self.enable_rtx_vsr_checkbox,
            self.deinterlace_checkbox,
            self.reinterlace_checkbox,
            self.perf_guard_checkbox,
            self.ai_sr_require_gpu_checkbox,
            self.ai_sr_strict_checkbox,
            self.ai_sr_post_exaggeration_checkbox,
            self.rtx_thdr_enable_checkbox,
            self.decklink_auto_detect_devices,
            self.decklink_enable_format_detection,
            self.decklink_fps_priority_guard_checkbox,
            self.roi_keyframe_duration_override_btn,
        ]
        for checkbox in checkbox_widgets:
            checkbox.toggled.connect(self._schedule_settings_save)

        spin_widgets = [
            self.fps_spin,
            self.preview_request_fps_spin,
            self.preview_poll_fps_spin,
            self.decklink_output_buffer_spin,
            self.roi_x_spin,
            self.roi_y_spin,
            self.roi_w_spin,
            self.roi_h_spin,
            self.scale_spin,
            self.roi_smoothing_slider,
            self.roi_latency_smoothing_slider,
            self.roi_drag_x_hysteresis_spin,
            self.roi_manual_drag_hold_spin,
            self.roi_min_drag_nudge_spin,
            self.roi_interlaced_field2_phase_spin,
            self.ai_sr_frame_interval_spin,
            self.ai_sr_overscan_spin,
            self.ai_sr_inference_divisor_spin,
            self.ai_sr_detail_preserve_spin,
            self.ai_sr_post_denoise_strength_spin,
            self.ai_sr_post_artifact_reduction_strength_spin,
            self.ai_sr_post_exaggeration_gain_spin,
            self.denoise_strength_spin,
            self.rtx_thdr_contrast_spin,
            self.rtx_thdr_saturation_spin,
            self.rtx_thdr_middle_gray_spin,
            self.rtx_thdr_max_luminance_spin,
            self.roi_transition_frames_spin,
        ]
        for spin in spin_widgets:
            spin.valueChanged.connect(self._schedule_settings_save)

        self.ai_sr_model_combo.currentTextChanged.connect(self._schedule_settings_save)
        self._display_splitter.splitterMoved.connect(lambda _pos, _index: self._schedule_settings_save())
        self._main_splitter.splitterMoved.connect(lambda _pos, _index: self._schedule_settings_save())

    def _schedule_settings_save(self, *_args) -> None:
        if self._updating_controls:
            return
        self._settings_save_timer.start()

    def _collect_settings_payload(self) -> dict[str, object]:
        keyframe_payload: dict[str, object] = {}
        for slot, keyframe in self._roi_keyframes.items():
            keyframe_payload[str(slot)] = self._serialize_roi_keyframe(keyframe)

        return {
            "version": 1,
            "fps": int(self.fps_spin.value()),
            "preview_request_fps": int(self.preview_request_fps_spin.value()),
            "preview_poll_fps": int(self.preview_poll_fps_spin.value()),
            "decklink_output_buffer_frames": int(self.decklink_output_buffer_spin.value()),
            "preview_downsample": str(self.preview_downsample_combo.currentText()),
            "color_space": str(self.color_space_combo.currentText()),
            "color_range": str(self.color_range_combo.currentText()),
            "roi_smoothing_percent": int(self.roi_smoothing_slider.value()),
            "roi_latency_smoothing_percent": int(self.roi_latency_smoothing_slider.value()),
            "roi_drag_x_hysteresis_px": float(self.roi_drag_x_hysteresis_spin.value()),
            "roi_manual_drag_hold_s": float(self.roi_manual_drag_hold_spin.value()),
            "roi_min_drag_nudge": float(self.roi_min_drag_nudge_spin.value()),
            "interlaced_field2_phase_fraction": float(self.roi_interlaced_field2_phase_spin.value()),
            "roi_transition_duration_frames": int(self.roi_transition_frames_spin.value()),
            "roi_interpolation_mode": str(self.roi_interp_mode_combo.currentText()),
            "roi_keyframe_duration_override": bool(self.roi_keyframe_duration_override_btn.isChecked()),
            "roi_keyframes": keyframe_payload,
            "basic_scaling_mode": str(self.sr_mode_combo.currentText()),
            "basic_scaling_method": str(self.sr_flavor_combo.currentText()),
            "basic_scaling_manual": str(self.sr_manual_combo.currentText()),
            "basic_scaling_auto_max": str(self.auto_sr_max_combo.currentText()),
            "basic_scaling_enabled": bool(self.enable_sr_checkbox.isChecked()) and not bool(self.enable_ai_sr_checkbox.isChecked()),
            "deinterlace_enabled": bool(self.deinterlace_checkbox.isChecked()),
            "reinterlace_enabled": bool(self.reinterlace_checkbox.isChecked()),
            "deinterlace_method": str(self.deinterlace_method_combo.currentText()),
            "denoise_method": str(self.denoise_method_combo.currentText()),
            "denoise_strength": float(self.denoise_strength_spin.value()),
            "perf_guard_enabled": bool(self.perf_guard_checkbox.isChecked()),
            "ai_sr_enabled": bool(self.enable_ai_sr_checkbox.isChecked()),
            "ai_sr_model_path": str(self.ai_sr_model_combo.currentText().strip()),
            "ai_sr_provider": str(self.ai_sr_provider_combo.currentText()),
            "ai_sr_trt_precision": str(self.ai_sr_trt_precision_combo.currentText()),
            "ai_sr_require_gpu": bool(self.ai_sr_require_gpu_checkbox.isChecked()),
            "ai_sr_inference_fps": int(self.ai_sr_frame_interval_spin.value()),
            "ai_sr_strict": bool(self.ai_sr_strict_checkbox.isChecked()),
            "ai_sr_input_align": str(self.ai_sr_input_align_combo.currentText()),
            "ai_sr_roi_overscan_percent": float(self.ai_sr_overscan_spin.value()),
            "ai_sr_inference_divisor": int(self.ai_sr_inference_divisor_spin.value()),
            "ai_sr_detail_preserve_percent": float(self.ai_sr_detail_preserve_spin.value()),
            "ai_sr_post_denoise_method": str(self.ai_sr_post_denoise_method_combo.currentText()),
            "ai_sr_post_denoise_strength": float(self.ai_sr_post_denoise_strength_spin.value()),
            "ai_sr_post_artifact_reduction_method": str(self.ai_sr_post_artifact_reduction_method_combo.currentText()),
            "ai_sr_post_artifact_reduction_strength": float(self.ai_sr_post_artifact_reduction_strength_spin.value()),
            "ai_sr_post_exaggeration_enabled": bool(self.ai_sr_post_exaggeration_checkbox.isChecked()),
            "ai_sr_post_exaggeration_gain": float(self.ai_sr_post_exaggeration_gain_spin.value()),
            "rtx_vsr_enabled": bool(self.enable_rtx_vsr_checkbox.isChecked()),
            "rtx_vsr_quality": str(self.rtx_vsr_quality_combo.currentText()),
            "rtx_vsr_scale": str(self.rtx_vsr_scale_combo.currentText()),
            "rtx_vsr_post_scale_method": str(self.rtx_vsr_post_scale_method_combo.currentText()),
            "rtx_thdr_enabled": bool(self.rtx_thdr_enable_checkbox.isChecked()),
            "rtx_thdr_contrast": int(self.rtx_thdr_contrast_spin.value()),
            "rtx_thdr_saturation": int(self.rtx_thdr_saturation_spin.value()),
            "rtx_thdr_middle_gray": int(self.rtx_thdr_middle_gray_spin.value()),
            "rtx_thdr_max_luminance": int(self.rtx_thdr_max_luminance_spin.value()),
            "source_mode": str(self.source_mode_combo.currentText()),
            "decklink_auto_detect": bool(self.decklink_auto_detect_devices.isChecked()),
            "decklink_input_device": self.decklink_input_device_combo.currentData(),
            "decklink_output_device": self.decklink_output_device_combo.currentData(),
            "decklink_input_mode_text": str(self.decklink_input_mode_combo.currentText()),
            "decklink_output_mode_text": str(self.decklink_output_mode_combo.currentText()),
            "decklink_enable_format_detection": bool(self.decklink_enable_format_detection.isChecked()),
            "decklink_fps_priority_guard": bool(self.decklink_fps_priority_guard_checkbox.isChecked()),
            "worker_process_priority": str(self.worker_priority_combo.currentText()),
            "display_splitter_sizes": list(self._display_splitter.sizes()),
            "main_splitter_sizes": list(self._main_splitter.sizes()),
        }

    def _save_settings(self) -> None:
        try:
            payload = self._collect_settings_payload()
            self._settings_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as exc:
            LOGGER.warning("Failed to save app settings: %s", exc)

    def _load_settings(self) -> None:
        if not self._settings_path.exists():
            return
        try:
            raw = json.loads(self._settings_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Failed to parse app settings: %s", exc)
            return

        if not isinstance(raw, dict):
            return

        self._has_persisted_deinterlace_method = bool(str(raw.get("deinterlace_method", "")).strip())

        self._updating_controls = True
        try:
            self.fps_spin.setValue(max(1, min(60, int(raw.get("fps", self.fps_spin.value())))))
            self.preview_request_fps_spin.setValue(max(1, min(60, int(raw.get("preview_request_fps", self.preview_request_fps_spin.value())))))
            self.preview_poll_fps_spin.setValue(max(1, min(120, int(raw.get("preview_poll_fps", self.preview_poll_fps_spin.value())))))
            self.decklink_output_buffer_spin.setValue(
                max(0, min(10, int(raw.get("decklink_output_buffer_frames", self.decklink_output_buffer_spin.value()))))
            )
            raw_smoothing = int(raw.get("roi_smoothing_percent", self.roi_smoothing_slider.value()))
            if raw_smoothing > 10:
                raw_smoothing = int(round(raw_smoothing / 10.0))
            self.roi_smoothing_slider.setValue(max(0, min(10, raw_smoothing)))
            self.roi_latency_smoothing_slider.setValue(max(0, min(100, int(raw.get("roi_latency_smoothing_percent", self.roi_latency_smoothing_slider.value())))))
            self.roi_drag_x_hysteresis_spin.setValue(
                max(0.10, min(1.20, float(raw.get("roi_drag_x_hysteresis_px", self.roi_drag_x_hysteresis_spin.value()))))
            )
            self.roi_manual_drag_hold_spin.setValue(
                max(0.05, min(0.50, float(raw.get("roi_manual_drag_hold_s", self.roi_manual_drag_hold_spin.value()))))
            )
            self.roi_min_drag_nudge_spin.setValue(
                max(0.0, min(0.50, float(raw.get("roi_min_drag_nudge", self.roi_min_drag_nudge_spin.value()))))
            )
            self.roi_interlaced_field2_phase_spin.setValue(
                _clamp_interlaced_field2_phase_fraction(
                    float(raw.get("interlaced_field2_phase_fraction", self.roi_interlaced_field2_phase_spin.value()))
                )
            )
            self.roi_transition_frames_spin.setValue(
                max(1, min(600, int(raw.get("roi_transition_duration_frames", self.roi_transition_frames_spin.value()))))
            )
            self.roi_interp_mode_combo.setCurrentText(str(raw.get("roi_interpolation_mode", self.roi_interp_mode_combo.currentText())))
            self.roi_keyframe_duration_override_btn.setChecked(
                bool(raw.get("roi_keyframe_duration_override", self.roi_keyframe_duration_override_btn.isChecked()))
            )

            self.preview_downsample_combo.setCurrentText(str(raw.get("preview_downsample", self.preview_downsample_combo.currentText())))
            self.color_space_combo.setCurrentText(str(raw.get("color_space", self.color_space_combo.currentText())))
            self.color_range_combo.setCurrentText(str(raw.get("color_range", self.color_range_combo.currentText())))
            self.sr_mode_combo.setCurrentText(str(raw.get("basic_scaling_mode", self.sr_mode_combo.currentText())))
            self.sr_flavor_combo.setCurrentText(str(raw.get("basic_scaling_method", self.sr_flavor_combo.currentText())))
            self.sr_manual_combo.setCurrentText(str(raw.get("basic_scaling_manual", self.sr_manual_combo.currentText())))
            self.auto_sr_max_combo.setCurrentText(str(raw.get("basic_scaling_auto_max", self.auto_sr_max_combo.currentText())))

            self.enable_sr_checkbox.setChecked(bool(raw.get("basic_scaling_enabled", self.enable_sr_checkbox.isChecked())))
            self.deinterlace_checkbox.setChecked(bool(raw.get("deinterlace_enabled", self.deinterlace_checkbox.isChecked())))
            self.reinterlace_checkbox.setChecked(bool(raw.get("reinterlace_enabled", self.reinterlace_checkbox.isChecked())))
            self.deinterlace_method_combo.setCurrentText(str(raw.get("deinterlace_method", self.deinterlace_method_combo.currentText())))
            self.denoise_method_combo.setCurrentText(str(raw.get("denoise_method", self.denoise_method_combo.currentText())))
            self.denoise_strength_spin.setValue(float(raw.get("denoise_strength", self.denoise_strength_spin.value())))
            self.perf_guard_checkbox.setChecked(bool(raw.get("perf_guard_enabled", self.perf_guard_checkbox.isChecked())))

            self.enable_ai_sr_checkbox.setChecked(bool(raw.get("ai_sr_enabled", self.enable_ai_sr_checkbox.isChecked())))
            self.ai_sr_model_combo.setCurrentText(str(raw.get("ai_sr_model_path", self.ai_sr_model_combo.currentText())))
            persisted_provider = str(raw.get("ai_sr_provider", self.ai_sr_provider_combo.currentText())).strip().lower()
            if persisted_provider == "trt_int8":
                persisted_provider = "trt"
                self.ai_sr_trt_precision_combo.setCurrentText("int8")
            elif persisted_provider == "trt_fp16":
                persisted_provider = "trt"
                self.ai_sr_trt_precision_combo.setCurrentText("fp16")
            self.ai_sr_provider_combo.setCurrentText(persisted_provider)
            persisted_trt_precision = str(raw.get("ai_sr_trt_precision", self.ai_sr_trt_precision_combo.currentText())).strip().lower()
            if persisted_trt_precision not in {"fp16", "int8"}:
                persisted_trt_precision = "fp16"
            self.ai_sr_trt_precision_combo.setCurrentText(persisted_trt_precision)
            self.ai_sr_require_gpu_checkbox.setChecked(bool(raw.get("ai_sr_require_gpu", self.ai_sr_require_gpu_checkbox.isChecked())))
            if "ai_sr_inference_fps" in raw:
                persisted_inference_fps = _clamp_ai_inference_fps(int(raw.get("ai_sr_inference_fps", self.ai_sr_frame_interval_spin.value())))
            elif "ai_sr_frame_interval" in raw:
                persisted_inference_fps = _legacy_ai_frame_interval_to_fps(int(raw.get("ai_sr_frame_interval", 1)))
            else:
                persisted_inference_fps = _clamp_ai_inference_fps(int(self.ai_sr_frame_interval_spin.value()))
            self.ai_sr_frame_interval_spin.setValue(persisted_inference_fps)
            self.ai_sr_strict_checkbox.setChecked(bool(raw.get("ai_sr_strict", self.ai_sr_strict_checkbox.isChecked())))

            persisted_align = str(raw.get("ai_sr_input_align", self.ai_sr_input_align_combo.currentText()))
            if persisted_align and persisted_align not in {self.ai_sr_input_align_combo.itemText(i) for i in range(self.ai_sr_input_align_combo.count())}:
                self.ai_sr_input_align_combo.addItem(persisted_align)
            self.ai_sr_input_align_combo.setCurrentText(persisted_align)

            self.ai_sr_overscan_spin.setValue(float(raw.get("ai_sr_roi_overscan_percent", self.ai_sr_overscan_spin.value())))
            self.ai_sr_inference_divisor_spin.setValue(max(0, int(raw.get("ai_sr_inference_divisor", self.ai_sr_inference_divisor_spin.value()))))
            self.ai_sr_detail_preserve_spin.setValue(float(raw.get("ai_sr_detail_preserve_percent", self.ai_sr_detail_preserve_spin.value())))
            self.ai_sr_post_denoise_method_combo.setCurrentText(
                str(raw.get("ai_sr_post_denoise_method", self.ai_sr_post_denoise_method_combo.currentText()))
            )
            self.ai_sr_post_denoise_strength_spin.setValue(
                max(0.0, min(1.0, float(raw.get("ai_sr_post_denoise_strength", self.ai_sr_post_denoise_strength_spin.value()))))
            )
            self.ai_sr_post_artifact_reduction_method_combo.setCurrentText(
                str(
                    raw.get(
                        "ai_sr_post_artifact_reduction_method",
                        self.ai_sr_post_artifact_reduction_method_combo.currentText(),
                    )
                )
            )
            self.ai_sr_post_artifact_reduction_strength_spin.setValue(
                max(
                    0.0,
                    min(
                        1.0,
                        float(
                            raw.get(
                                "ai_sr_post_artifact_reduction_strength",
                                self.ai_sr_post_artifact_reduction_strength_spin.value(),
                            )
                        ),
                    ),
                )
            )
            self.ai_sr_post_exaggeration_checkbox.setChecked(
                bool(raw.get("ai_sr_post_exaggeration_enabled", self.ai_sr_post_exaggeration_checkbox.isChecked()))
            )
            self.ai_sr_post_exaggeration_gain_spin.setValue(
                max(
                    1.0,
                    min(
                        4.0,
                        float(raw.get("ai_sr_post_exaggeration_gain", self.ai_sr_post_exaggeration_gain_spin.value())),
                    ),
                )
            )

            self.enable_rtx_vsr_checkbox.setChecked(bool(raw.get("rtx_vsr_enabled", self.enable_rtx_vsr_checkbox.isChecked())))
            self.rtx_vsr_quality_combo.setCurrentText(str(raw.get("rtx_vsr_quality", self.rtx_vsr_quality_combo.currentText())))
            self.rtx_vsr_scale_combo.setCurrentText(str(raw.get("rtx_vsr_scale", self.rtx_vsr_scale_combo.currentText())))
            self.rtx_vsr_post_scale_method_combo.setCurrentText(str(raw.get("rtx_vsr_post_scale_method", self.rtx_vsr_post_scale_method_combo.currentText())))
            self.rtx_thdr_enable_checkbox.setChecked(bool(raw.get("rtx_thdr_enabled", self.rtx_thdr_enable_checkbox.isChecked())))
            self.rtx_thdr_contrast_spin.setValue(int(raw.get("rtx_thdr_contrast", self.rtx_thdr_contrast_spin.value())))
            self.rtx_thdr_saturation_spin.setValue(int(raw.get("rtx_thdr_saturation", self.rtx_thdr_saturation_spin.value())))
            self.rtx_thdr_middle_gray_spin.setValue(int(raw.get("rtx_thdr_middle_gray", self.rtx_thdr_middle_gray_spin.value())))
            self.rtx_thdr_max_luminance_spin.setValue(int(raw.get("rtx_thdr_max_luminance", self.rtx_thdr_max_luminance_spin.value())))

            self.source_mode_combo.setCurrentText(str(raw.get("source_mode", self.source_mode_combo.currentText())))
            self.decklink_auto_detect_devices.setChecked(bool(raw.get("decklink_auto_detect", self.decklink_auto_detect_devices.isChecked())))
            self.decklink_enable_format_detection.setChecked(
                bool(raw.get("decklink_enable_format_detection", self.decklink_enable_format_detection.isChecked()))
            )
            self.decklink_fps_priority_guard_checkbox.setChecked(
                bool(raw.get("decklink_fps_priority_guard", self.decklink_fps_priority_guard_checkbox.isChecked()))
            )
            self.worker_priority_combo.setCurrentText(
                str(raw.get("worker_process_priority", self.worker_priority_combo.currentText()))
            )
        finally:
            self._updating_controls = False

        self._preview_downsample_factor = self._normalize_preview_downsample_factor(
            PREVIEW_DOWNSAMPLE_LABEL_TO_FACTOR.get(self.preview_downsample_combo.currentText(), self._preview_downsample_factor)
        )
        self._decklink_tick_poll_fps = float(max(1, self.preview_poll_fps_spin.value()))
        self._decklink_output_buffer_frames = int(self.decklink_output_buffer_spin.value())
        self._worker_process_priority = _normalize_worker_priority_name(
            WORKER_PRIORITY_LABEL_TO_NAME.get(
                str(self.worker_priority_combo.currentText()),
                getattr(self._controller, "worker_process_priority", "above_normal"),
            )
        )
        self._apply_worker_process_priority_to_controller(notify=False)

        persisted_in_device = raw.get("decklink_input_device")
        persisted_out_device = raw.get("decklink_output_device")
        self._pending_persisted_input_device = persisted_in_device
        self._pending_persisted_output_device = persisted_out_device
        if persisted_in_device is not None or persisted_out_device is not None:
            for i in range(self.decklink_input_device_combo.count()):
                if self.decklink_input_device_combo.itemData(i) == persisted_in_device:
                    self.decklink_input_device_combo.setCurrentIndex(i)
                    break
            for i in range(self.decklink_output_device_combo.count()):
                if self.decklink_output_device_combo.itemData(i) == persisted_out_device:
                    self.decklink_output_device_combo.setCurrentIndex(i)
                    break

        input_mode_text = str(raw.get("decklink_input_mode_text", "")).strip()
        output_mode_text = str(raw.get("decklink_output_mode_text", "")).strip()
        self._pending_persisted_input_mode_text = input_mode_text
        self._pending_persisted_output_mode_text = output_mode_text
        if input_mode_text:
            for i in range(self.decklink_input_mode_combo.count()):
                if self.decklink_input_mode_combo.itemText(i) == input_mode_text:
                    self.decklink_input_mode_combo.setCurrentIndex(i)
                    break
        if output_mode_text:
            for i in range(self.decklink_output_mode_combo.count()):
                if self.decklink_output_mode_combo.itemText(i) == output_mode_text:
                    self.decklink_output_mode_combo.setCurrentIndex(i)
                    break

        display_sizes = raw.get("display_splitter_sizes")
        if isinstance(display_sizes, list) and len(display_sizes) >= 2:
            self._display_splitter.setSizes([int(display_sizes[0]), int(display_sizes[1])])

        main_sizes = raw.get("main_splitter_sizes")
        if isinstance(main_sizes, list) and len(main_sizes) >= 2:
            self._main_splitter.setSizes([int(main_sizes[0]), int(main_sizes[1])])

        self._restore_roi_keyframes(raw.get("roi_keyframes"))
        self._update_roi_key_buttons()

        self._update_timer_interval()

    def _current_ai_sr_profile(self) -> dict[str, object]:
        return {
            "provider": self.ai_sr_provider_combo.currentText().strip().lower(),
            "trt_precision": self.ai_sr_trt_precision_combo.currentText().strip().lower(),
            "require_gpu": bool(self.ai_sr_require_gpu_checkbox.isChecked()),
            "inference_fps": int(self.ai_sr_frame_interval_spin.value()),
            "strict": bool(self.ai_sr_strict_checkbox.isChecked()),
            "input_align": int(self.ai_sr_input_align_combo.currentText()),
            "roi_overscan_percent": float(self.ai_sr_overscan_spin.value()),
            "inference_divisor": int(self.ai_sr_inference_divisor_spin.value()),
            "detail_preserve_percent": float(self.ai_sr_detail_preserve_spin.value()),
            "post_denoise_method": AI_SR_POST_DENOISE_LABEL_TO_NAME.get(
                self.ai_sr_post_denoise_method_combo.currentText(),
                "off",
            ),
            "post_denoise_strength": float(self.ai_sr_post_denoise_strength_spin.value()),
            "post_artifact_reduction_method": AI_SR_POST_ARTIFACT_REDUCTION_LABEL_TO_NAME.get(
                self.ai_sr_post_artifact_reduction_method_combo.currentText(),
                "off",
            ),
            "post_artifact_reduction_strength": float(self.ai_sr_post_artifact_reduction_strength_spin.value()),
            "post_exaggeration_enabled": bool(self.ai_sr_post_exaggeration_checkbox.isChecked()),
            "post_exaggeration_gain": float(self.ai_sr_post_exaggeration_gain_spin.value()),
        }

    def _apply_ai_sr_profile(self, profile: dict[str, object]) -> None:
        provider = str(profile.get("provider", getattr(self._controller, "ai_sr_provider", "auto"))).lower()
        if provider == "trt_int8":
            provider = "trt"
            profile["trt_precision"] = "int8"
        elif provider == "trt_fp16":
            provider = "trt"
            profile["trt_precision"] = "fp16"
        if provider not in {"auto", "cuda", "trt", "tensorrt", "cpu"}:
            provider = "auto"
        self.ai_sr_provider_combo.setCurrentText(provider)

        trt_precision = str(profile.get("trt_precision", getattr(self._controller, "ai_sr_trt_precision", "fp16"))).strip().lower()
        if trt_precision not in {"fp16", "int8"}:
            trt_precision = "fp16"
        self.ai_sr_trt_precision_combo.setCurrentText(trt_precision)

        self.ai_sr_require_gpu_checkbox.setChecked(bool(profile.get("require_gpu", getattr(self._controller, "ai_sr_require_gpu", True))))
        target_inference_fps = int(profile.get("inference_fps", profile.get("frame_interval", getattr(self._controller, "ai_sr_frame_interval", 2))))
        self.ai_sr_frame_interval_spin.setValue(max(1, min(60, target_inference_fps)))
        self.ai_sr_strict_checkbox.setChecked(bool(profile.get("strict", getattr(self._controller, "ai_sr_strict", False))))

        input_align = max(1, int(profile.get("input_align", getattr(self._controller, "ai_sr_input_align", 2))))
        if str(input_align) not in {self.ai_sr_input_align_combo.itemText(i) for i in range(self.ai_sr_input_align_combo.count())}:
            self.ai_sr_input_align_combo.addItem(str(input_align))
        self.ai_sr_input_align_combo.setCurrentText(str(input_align))

        overscan = max(0.0, float(profile.get("roi_overscan_percent", getattr(self._controller, "ai_sr_roi_overscan_percent", 0.0))))
        self.ai_sr_overscan_spin.setValue(overscan)

        inference_divisor = max(0, int(profile.get("inference_divisor", getattr(self._controller, "ai_sr_inference_divisor", 0))))
        self.ai_sr_inference_divisor_spin.setValue(inference_divisor)

        detail_preserve = max(0.0, float(profile.get("detail_preserve_percent", getattr(self._controller, "ai_sr_detail_preserve_percent", 0.0))))
        self.ai_sr_detail_preserve_spin.setValue(detail_preserve)

        post_denoise_method = str(profile.get("post_denoise_method", getattr(self._controller, "ai_sr_post_denoise_method", "off"))).strip().lower()
        self.ai_sr_post_denoise_method_combo.setCurrentText(
            AI_SR_POST_DENOISE_NAME_TO_LABEL.get(post_denoise_method, "Off")
        )

        post_denoise_strength = max(0.0, min(1.0, float(profile.get("post_denoise_strength", getattr(self._controller, "ai_sr_post_denoise_strength", 0.0)))))
        self.ai_sr_post_denoise_strength_spin.setValue(post_denoise_strength)

        post_artifact_method = str(
            profile.get(
                "post_artifact_reduction_method",
                getattr(self._controller, "ai_sr_post_artifact_reduction_method", "off"),
            )
        ).strip().lower()
        self.ai_sr_post_artifact_reduction_method_combo.setCurrentText(
            AI_SR_POST_ARTIFACT_REDUCTION_NAME_TO_LABEL.get(post_artifact_method, "Off")
        )

        post_artifact_strength = max(
            0.0,
            min(
                1.0,
                float(
                    profile.get(
                        "post_artifact_reduction_strength",
                        getattr(self._controller, "ai_sr_post_artifact_reduction_strength", 0.0),
                    )
                ),
            ),
        )
        self.ai_sr_post_artifact_reduction_strength_spin.setValue(post_artifact_strength)

        post_exaggeration_enabled = bool(
            profile.get(
                "post_exaggeration_enabled",
                getattr(self._controller, "ai_sr_post_exaggeration_enabled", False),
            )
        )
        self.ai_sr_post_exaggeration_checkbox.setChecked(post_exaggeration_enabled)

        post_exaggeration_gain = max(
            1.0,
            min(
                4.0,
                float(
                    profile.get(
                        "post_exaggeration_gain",
                        getattr(self._controller, "ai_sr_post_exaggeration_gain", 2.0),
                    )
                ),
            ),
        )
        self.ai_sr_post_exaggeration_gain_spin.setValue(post_exaggeration_gain)

    def _resolve_startup_ai_sr_model_path(self) -> str:
        configured = self.ai_sr_model_combo.currentText().strip()
        candidates: list[str] = []
        if configured:
            candidates.append(configured)

        default_model = self._default_ai_sr_model_path().strip()
        if default_model:
            candidates.append(default_model)

        for i in range(self.ai_sr_model_combo.count()):
            item = self.ai_sr_model_combo.itemText(i).strip()
            if item:
                candidates.append(item)

        seen: set[str] = set()
        for candidate in candidates:
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            path_obj = Path(candidate)
            if path_obj.exists() and path_obj.is_file():
                return str(path_obj)

        return configured or default_model

    def _apply_startup_ai_sr_settings(self) -> None:
        model_path = self._resolve_startup_ai_sr_model_path()
        ai_enabled = bool(self.enable_ai_sr_checkbox.isChecked())

        if model_path and model_path != self.ai_sr_model_combo.currentText().strip():
            self.ai_sr_model_combo.blockSignals(True)
            self.ai_sr_model_combo.setCurrentText(model_path)
            self.ai_sr_model_combo.blockSignals(False)

        profile = self._current_ai_sr_profile()

        try:
            startup_timeout_s = 20.0
            self._controller.set_ai_sr_model_path(
                model_path,
                wait_for_ack=True,
                timeout_seconds=startup_timeout_s,
            )
            self._controller.set_ai_sr_settings(
                provider=str(profile["provider"]),
                require_gpu=bool(profile["require_gpu"]),
                inference_fps=int(profile["inference_fps"]),
                trt_precision=str(profile["trt_precision"]),
                strict=bool(profile["strict"]),
                input_align=int(profile["input_align"]),
                roi_overscan_percent=float(profile["roi_overscan_percent"]),
                inference_divisor=int(profile["inference_divisor"]),
                detail_preserve_percent=float(profile["detail_preserve_percent"]),
                post_denoise_method=str(profile["post_denoise_method"]),
                post_denoise_strength=float(profile["post_denoise_strength"]),
                post_artifact_reduction_method=str(profile["post_artifact_reduction_method"]),
                post_artifact_reduction_strength=float(profile["post_artifact_reduction_strength"]),
                post_exaggeration_enabled=bool(profile["post_exaggeration_enabled"]),
                post_exaggeration_gain=float(profile["post_exaggeration_gain"]),
                wait_for_ack=True,
                timeout_seconds=startup_timeout_s,
            )
            self._controller.set_ai_sr_enabled(
                ai_enabled,
                wait_for_ack=True,
                timeout_seconds=startup_timeout_s,
            )

            if ai_enabled and not bool(getattr(self._controller, "ai_sr_active", False)):
                ai_err = str(getattr(self._controller, "ai_sr_error", "AI SR did not become active")).strip()
                if ai_err:
                    raise RuntimeError(ai_err)
                raise RuntimeError("AI SR did not become active")

            LOGGER.info(
                "Applied startup AI SR settings: enabled=%s, model=%s, provider=%s, inference_fps=%s",
                ai_enabled,
                model_path,
                profile["provider"],
                profile["inference_fps"],
            )
        except Exception as exc:
            LOGGER.warning("Failed to apply startup AI SR settings: %s", exc)
            self._update_status(f"Startup AI SR apply failed: {exc}")

    def _discover_ai_sr_model_paths(self) -> list[str]:
        models_root = Path(__file__).resolve().parents[1] / "models"
        if not models_root.exists():
            return []

        discovered = {str(path.resolve()) for path in models_root.rglob("*.onnx") if path.is_file()}
        return sorted(discovered, key=lambda p: p.lower())

    def _refresh_ai_sr_model_options(self, preferred_model_path: str | None = None) -> None:
        current_text = self.ai_sr_model_combo.currentText().strip()
        preferred = (preferred_model_path or current_text or self._default_ai_sr_model_path()).strip()

        options = self._discover_ai_sr_model_paths()
        if preferred and preferred not in options:
            options.insert(0, preferred)

        self.ai_sr_model_combo.blockSignals(True)
        self.ai_sr_model_combo.clear()
        for model_path in options:
            self.ai_sr_model_combo.addItem(model_path)
        self.ai_sr_model_combo.setCurrentText(preferred)
        self.ai_sr_model_combo.blockSignals(False)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        settings_box = QGroupBox("General")
        settings_form = QFormLayout(settings_box)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(20)
        self.fps_spin.valueChanged.connect(self._update_timer_interval)
        settings_form.addRow("FPS", self.fps_spin)

        self.preview_request_fps_spin = QSpinBox()
        self.preview_request_fps_spin.setRange(1, 60)
        initial_preview_fps = int(round(float(getattr(self._controller, "_preview_fps", 30.0))))
        self.preview_request_fps_spin.setValue(max(1, min(60, initial_preview_fps)))
        self.preview_request_fps_spin.valueChanged.connect(self._on_preview_request_fps_changed)
        settings_form.addRow("Preview request FPS", self.preview_request_fps_spin)

        self.preview_poll_fps_spin = QSpinBox()
        self.preview_poll_fps_spin.setRange(1, 120)
        self.preview_poll_fps_spin.setValue(int(round(self._decklink_tick_poll_fps)))
        self.preview_poll_fps_spin.valueChanged.connect(self._on_preview_poll_fps_changed)
        settings_form.addRow("Preview poll FPS cap", self.preview_poll_fps_spin)

        self.preview_downsample_combo = QComboBox()
        self.preview_downsample_combo.addItems(list(PREVIEW_DOWNSAMPLE_LABEL_TO_FACTOR.keys()))
        self.preview_downsample_combo.setCurrentText(
            self._preview_downsample_label_for_factor(self._preview_downsample_factor)
        )
        self.preview_downsample_combo.currentIndexChanged.connect(self._on_preview_downsample_changed)
        settings_form.addRow("Preview downsample", self.preview_downsample_combo)

        self.sr_mode_combo = QComboBox()
        self.sr_mode_combo.addItems(["Auto", "Manual"])
        self.sr_mode_combo.currentIndexChanged.connect(self._on_sr_mode_changed)

        self.sr_flavor_combo = QComboBox()
        self.sr_flavor_combo.addItems(list(SR_FLAVOR_LABEL_TO_NAME.keys()))
        self.sr_flavor_combo.setCurrentText(
            SR_FLAVOR_NAME_TO_LABEL.get(self._controller.basic_scaling_method, "Bicubic (Balanced)")
        )
        self.sr_flavor_combo.currentIndexChanged.connect(self._on_sr_flavor_changed)

        self.sr_manual_combo = QComboBox()
        self.sr_manual_combo.addItems(["2", "4", "8", "16"])
        self.sr_manual_combo.setCurrentText("4")
        self.sr_manual_combo.currentIndexChanged.connect(self._on_sr_manual_changed)

        self.auto_sr_max_combo = QComboBox()
        self.auto_sr_max_combo.addItems(["2", "4", "8", "16"])
        self.auto_sr_max_combo.setCurrentText("4")
        self.auto_sr_max_combo.currentIndexChanged.connect(self._on_auto_sr_max_changed)

        self.enable_sr_checkbox = QCheckBox("Enable basic CUDA scaling")
        self.enable_sr_checkbox.setChecked(True)
        self.enable_sr_checkbox.toggled.connect(self._on_enable_sr_toggled)

        self.enable_ai_sr_checkbox = QCheckBox("Enable AI SR (ONNX model)")
        self.enable_ai_sr_checkbox.setChecked(getattr(self._controller, "ai_sr_enabled", False))
        self.enable_ai_sr_checkbox.toggled.connect(self._on_enable_ai_sr_toggled)

        self.enable_rtx_vsr_checkbox = QCheckBox("Enable RTX VSR (NVIDIA SDK path)")
        self.enable_rtx_vsr_checkbox.setChecked(bool(getattr(self._controller, "rtx_vsr_enabled", False)))
        self.enable_rtx_vsr_checkbox.toggled.connect(self._on_enable_rtx_vsr_toggled)

        self.ai_sr_model_combo = QComboBox()
        self.ai_sr_model_combo.setEditable(True)
        default_model_path = self._default_ai_sr_model_path()
        current_model_path = getattr(self._controller, "ai_sr_model_path", "") or default_model_path
        self._refresh_ai_sr_model_options(preferred_model_path=current_model_path)
        self.ai_sr_model_combo.currentTextChanged.connect(self._on_ai_sr_model_selection_changed)

        ai_sr_model_actions = QWidget()
        ai_sr_model_actions_layout = QHBoxLayout(ai_sr_model_actions)
        ai_sr_model_actions_layout.setContentsMargins(0, 0, 0, 0)
        ai_sr_model_actions_layout.setSpacing(8)

        self.ai_sr_model_apply_btn = QPushButton("Apply AI SR Model Path")
        self.ai_sr_model_apply_btn.clicked.connect(self._on_ai_sr_model_apply_clicked)
        ai_sr_model_actions_layout.addWidget(self.ai_sr_model_apply_btn)

        self.ai_sr_model_lightest_btn = QPushButton("Select Lightest Model")
        self.ai_sr_model_lightest_btn.clicked.connect(self._on_ai_sr_model_lightest_clicked)
        ai_sr_model_actions_layout.addWidget(self.ai_sr_model_lightest_btn)

        self.ai_sr_model_refresh_btn = QPushButton("Refresh Model List")
        self.ai_sr_model_refresh_btn.clicked.connect(self._on_ai_sr_model_refresh_clicked)
        ai_sr_model_actions_layout.addWidget(self.ai_sr_model_refresh_btn)

        self.ai_sr_model_quantize_btn = QPushButton("Create INT8 Model")
        self.ai_sr_model_quantize_btn.clicked.connect(self._on_ai_sr_model_quantize_clicked)
        ai_sr_model_actions_layout.addWidget(self.ai_sr_model_quantize_btn)

        self.ai_sr_provider_combo = QComboBox()
        self.ai_sr_provider_combo.addItems(["auto", "cuda", "trt", "cpu"])
        self.ai_sr_provider_combo.setCurrentText(str(getattr(self._controller, "ai_sr_provider", "auto")).lower())

        self.ai_sr_trt_precision_combo = QComboBox()
        self.ai_sr_trt_precision_combo.addItems(["fp16", "int8"])
        trt_precision_default = str(getattr(self._controller, "ai_sr_trt_precision", "fp16")).strip().lower()
        if trt_precision_default not in {"fp16", "int8"}:
            trt_precision_default = "fp16"
        self.ai_sr_trt_precision_combo.setCurrentText(trt_precision_default)

        self.ai_sr_require_gpu_checkbox = QCheckBox("Require GPU provider")
        self.ai_sr_require_gpu_checkbox.setChecked(bool(getattr(self._controller, "ai_sr_require_gpu", True)))

        self.ai_sr_frame_interval_spin = QSpinBox()
        self.ai_sr_frame_interval_spin.setRange(1, 60)
        self.ai_sr_frame_interval_spin.setValue(int(getattr(self._controller, "ai_sr_frame_interval", 1)))
        self.ai_sr_frame_interval_spin.setToolTip(
            "Target AI inference FPS (1-60). Very low values (1-2) can appear as passthrough because inference updates arrive rarely."
        )

        self.ai_sr_strict_checkbox = QCheckBox("Strict AI SR (blocking)")
        self.ai_sr_strict_checkbox.setChecked(bool(getattr(self._controller, "ai_sr_strict", False)))

        self.ai_sr_input_align_combo = QComboBox()
        self.ai_sr_input_align_combo.addItems(["1", "2", "4", "8"])
        self.ai_sr_input_align_combo.setCurrentText(str(int(getattr(self._controller, "ai_sr_input_align", 2))))

        self.ai_sr_overscan_spin = QDoubleSpinBox()
        self.ai_sr_overscan_spin.setRange(0.0, 50.0)
        self.ai_sr_overscan_spin.setDecimals(1)
        self.ai_sr_overscan_spin.setSingleStep(0.5)
        self.ai_sr_overscan_spin.setValue(float(getattr(self._controller, "ai_sr_roi_overscan_percent", 0.0)))

        self.ai_sr_inference_divisor_spin = QSpinBox()
        self.ai_sr_inference_divisor_spin.setRange(0, 16)
        self.ai_sr_inference_divisor_spin.setValue(int(getattr(self._controller, "ai_sr_inference_divisor", 0)))
        self.ai_sr_inference_divisor_spin.setToolTip("0 uses model-native divisor; lower values can improve quality at higher GPU cost")

        self.ai_sr_detail_preserve_spin = QDoubleSpinBox()
        self.ai_sr_detail_preserve_spin.setRange(0.0, 100.0)
        self.ai_sr_detail_preserve_spin.setDecimals(1)
        self.ai_sr_detail_preserve_spin.setSingleStep(2.5)
        self.ai_sr_detail_preserve_spin.setValue(float(getattr(self._controller, "ai_sr_detail_preserve_percent", 0.0)))
        self.ai_sr_detail_preserve_spin.setToolTip("Blend original ROI detail back into AI output to reduce softness")

        self.ai_sr_post_denoise_method_combo = QComboBox()
        self.ai_sr_post_denoise_method_combo.addItems(list(AI_SR_POST_DENOISE_LABEL_TO_NAME.keys()))
        self.ai_sr_post_denoise_method_combo.setCurrentText(
            AI_SR_POST_DENOISE_NAME_TO_LABEL.get(
                str(getattr(self._controller, "ai_sr_post_denoise_method", "off")).strip().lower(),
                "Off",
            )
        )

        self.ai_sr_post_denoise_strength_spin = QDoubleSpinBox()
        self.ai_sr_post_denoise_strength_spin.setRange(0.0, 1.0)
        self.ai_sr_post_denoise_strength_spin.setDecimals(2)
        self.ai_sr_post_denoise_strength_spin.setSingleStep(0.05)
        self.ai_sr_post_denoise_strength_spin.setValue(float(getattr(self._controller, "ai_sr_post_denoise_strength", 0.0)))

        self.ai_sr_post_artifact_reduction_method_combo = QComboBox()
        self.ai_sr_post_artifact_reduction_method_combo.addItems(list(AI_SR_POST_ARTIFACT_REDUCTION_LABEL_TO_NAME.keys()))
        self.ai_sr_post_artifact_reduction_method_combo.setCurrentText(
            AI_SR_POST_ARTIFACT_REDUCTION_NAME_TO_LABEL.get(
                str(getattr(self._controller, "ai_sr_post_artifact_reduction_method", "off")).strip().lower(),
                "Off",
            )
        )

        self.ai_sr_post_artifact_reduction_strength_spin = QDoubleSpinBox()
        self.ai_sr_post_artifact_reduction_strength_spin.setRange(0.0, 1.0)
        self.ai_sr_post_artifact_reduction_strength_spin.setDecimals(2)
        self.ai_sr_post_artifact_reduction_strength_spin.setSingleStep(0.05)
        self.ai_sr_post_artifact_reduction_strength_spin.setValue(
            float(getattr(self._controller, "ai_sr_post_artifact_reduction_strength", 0.0))
        )

        self.ai_sr_post_exaggeration_checkbox = QCheckBox("Enable exaggerated postprocess")
        self.ai_sr_post_exaggeration_checkbox.setChecked(
            bool(getattr(self._controller, "ai_sr_post_exaggeration_enabled", False))
        )

        self.ai_sr_post_exaggeration_gain_spin = QDoubleSpinBox()
        self.ai_sr_post_exaggeration_gain_spin.setRange(1.0, 4.0)
        self.ai_sr_post_exaggeration_gain_spin.setDecimals(2)
        self.ai_sr_post_exaggeration_gain_spin.setSingleStep(0.25)
        self.ai_sr_post_exaggeration_gain_spin.setValue(
            float(getattr(self._controller, "ai_sr_post_exaggeration_gain", 2.0))
        )

        initial_profile = self._ai_sr_profiles.get(current_model_path)
        if initial_profile is not None:
            self._apply_ai_sr_profile(initial_profile)

        ai_sr_tuning_actions = QWidget()
        ai_sr_tuning_actions_layout = QHBoxLayout(ai_sr_tuning_actions)
        ai_sr_tuning_actions_layout.setContentsMargins(0, 0, 0, 0)
        ai_sr_tuning_actions_layout.setSpacing(8)

        self.ai_sr_tuning_apply_btn = QPushButton("Apply AI SR Tuning")
        self.ai_sr_tuning_apply_btn.clicked.connect(self._on_ai_sr_tuning_apply_clicked)
        ai_sr_tuning_actions_layout.addWidget(self.ai_sr_tuning_apply_btn)

        self.ai_sr_profile_save_btn = QPushButton("Save Model Profile")
        self.ai_sr_profile_save_btn.clicked.connect(self._on_ai_sr_profile_save_clicked)
        ai_sr_tuning_actions_layout.addWidget(self.ai_sr_profile_save_btn)

        self.ai_sr_profile_load_btn = QPushButton("Load Model Profile")
        self.ai_sr_profile_load_btn.clicked.connect(self._on_ai_sr_profile_load_clicked)
        ai_sr_tuning_actions_layout.addWidget(self.ai_sr_profile_load_btn)

        ai_sr_runtime_box = QGroupBox("AI SR Runtime")
        ai_sr_runtime_layout = QVBoxLayout(ai_sr_runtime_box)
        ai_sr_runtime_layout.setContentsMargins(8, 8, 8, 8)
        self.ai_sr_runtime_label = QLabel("AI SR runtime info will appear after worker initialization.")
        self.ai_sr_runtime_label.setWordWrap(True)
        ai_sr_runtime_layout.addWidget(self.ai_sr_runtime_label)

        rtx_vsr_box = QGroupBox("RTX Video SDK (VSR)")
        rtx_vsr_form = QFormLayout(rtx_vsr_box)

        self.rtx_vsr_quality_combo = QComboBox()
        self.rtx_vsr_quality_combo.addItems(["low", "medium", "high", "ultra"])
        self.rtx_vsr_quality_combo.setCurrentText(str(getattr(self._controller, "rtx_vsr_quality", "high")).lower())

        self.rtx_vsr_scale_combo = QComboBox()
        self.rtx_vsr_scale_combo.addItems(["1", "2", "4"])
        self.rtx_vsr_scale_combo.setCurrentText(str(int(getattr(self._controller, "rtx_vsr_scale", 2))))

        self.rtx_vsr_post_scale_method_combo = QComboBox()
        self.rtx_vsr_post_scale_method_combo.addItems(list(RTX_POST_SCALE_METHOD_LABEL_TO_NAME.keys()))
        self.rtx_vsr_post_scale_method_combo.setCurrentText(
            RTX_POST_SCALE_METHOD_NAME_TO_LABEL.get(
                str(getattr(self._controller, "rtx_vsr_post_scale_method", "bicubic")),
                "Bicubic (Balanced)",
            )
        )

        self.rtx_thdr_enable_checkbox = QCheckBox("Enable RTX TrueHDR")
        self.rtx_thdr_enable_checkbox.setChecked(bool(getattr(self._controller, "rtx_thdr_enabled", False)))

        self.rtx_thdr_contrast_spin = QSpinBox()
        self.rtx_thdr_contrast_spin.setRange(0, 1000)
        self.rtx_thdr_contrast_spin.setValue(int(getattr(self._controller, "rtx_thdr_contrast", 50)))

        self.rtx_thdr_saturation_spin = QSpinBox()
        self.rtx_thdr_saturation_spin.setRange(0, 1000)
        self.rtx_thdr_saturation_spin.setValue(int(getattr(self._controller, "rtx_thdr_saturation", 50)))

        self.rtx_thdr_middle_gray_spin = QSpinBox()
        self.rtx_thdr_middle_gray_spin.setRange(0, 1000)
        self.rtx_thdr_middle_gray_spin.setValue(int(getattr(self._controller, "rtx_thdr_middle_gray", 50)))

        self.rtx_thdr_max_luminance_spin = QSpinBox()
        self.rtx_thdr_max_luminance_spin.setRange(0, 10000)
        self.rtx_thdr_max_luminance_spin.setValue(int(getattr(self._controller, "rtx_thdr_max_luminance", 1000)))

        self.rtx_vsr_apply_btn = QPushButton("Apply RTX VSR Settings")
        self.rtx_vsr_apply_btn.clicked.connect(self._on_rtx_vsr_settings_apply_clicked)

        self.rtx_vsr_runtime_label = QLabel("RTX VSR runtime info will appear after worker initialization.")
        self.rtx_vsr_runtime_label.setWordWrap(True)

        rtx_vsr_form.addRow("Quality", self.rtx_vsr_quality_combo)
        rtx_vsr_form.addRow(self.rtx_thdr_enable_checkbox)
        rtx_vsr_form.addRow("THDR contrast", self.rtx_thdr_contrast_spin)
        rtx_vsr_form.addRow("THDR saturation", self.rtx_thdr_saturation_spin)
        rtx_vsr_form.addRow("THDR middle gray", self.rtx_thdr_middle_gray_spin)
        rtx_vsr_form.addRow("THDR max luminance", self.rtx_thdr_max_luminance_spin)
        rtx_vsr_form.addRow(self.rtx_vsr_apply_btn)
        rtx_vsr_form.addRow(self.rtx_vsr_runtime_label)

        self.deinterlace_checkbox = QCheckBox("Enable deinterlace")
        self.deinterlace_checkbox.setChecked(True)
        self.deinterlace_checkbox.toggled.connect(self._on_deinterlace_toggled)

        self.reinterlace_checkbox = QCheckBox("Reinterlace output (interlaced modes)")
        self.reinterlace_checkbox.setChecked(bool(getattr(self._controller, "reinterlace_enabled", False)))
        self.reinterlace_checkbox.toggled.connect(self._on_reinterlace_toggled)

        self.deinterlace_method_combo = QComboBox()
        self.deinterlace_method_combo.addItems(list(DEINTERLACE_METHOD_LABEL_TO_NAME.keys()))
        self.deinterlace_method_combo.setCurrentText(
            DEINTERLACE_METHOD_NAME_TO_LABEL.get(getattr(self._controller, "deinterlace_method", "bob"), "Bob (Fast)")
        )
        self.deinterlace_method_combo.currentIndexChanged.connect(self._on_deinterlace_method_changed)

        self.denoise_method_combo = QComboBox()
        self.denoise_method_combo.addItems(list(DENOISE_METHOD_LABEL_TO_NAME.keys()))
        self.denoise_method_combo.setCurrentText(
            DENOISE_METHOD_NAME_TO_LABEL.get(getattr(self._controller, "denoise_method", "off"), "Off")
        )
        self.denoise_method_combo.currentIndexChanged.connect(self._on_denoise_settings_changed)

        self.denoise_strength_spin = QDoubleSpinBox()
        self.denoise_strength_spin.setRange(0.0, 1.0)
        self.denoise_strength_spin.setDecimals(2)
        self.denoise_strength_spin.setSingleStep(0.05)
        self.denoise_strength_spin.setValue(float(getattr(self._controller, "denoise_strength", 0.35)))
        self.denoise_strength_spin.valueChanged.connect(self._on_denoise_settings_changed)

        deinterlace_box = QGroupBox("De-interlacing")
        deinterlace_form = QFormLayout(deinterlace_box)
        deinterlace_form.addRow(self.deinterlace_checkbox)
        deinterlace_form.addRow(self.reinterlace_checkbox)
        deinterlace_form.addRow("Method", self.deinterlace_method_combo)

        denoise_box = QGroupBox("Noise Reduction")
        denoise_form = QFormLayout(denoise_box)
        denoise_form.addRow("Method", self.denoise_method_combo)
        denoise_form.addRow("Strength", self.denoise_strength_spin)

        upscaling_box = QGroupBox("Upscaling (Basic or AI)")
        upscaling_form = QFormLayout(upscaling_box)
        upscaling_form.addRow(self.enable_sr_checkbox)
        upscaling_form.addRow("Basic scaling mode", self.sr_mode_combo)
        upscaling_form.addRow("Basic scaling method", self.sr_flavor_combo)
        upscaling_form.addRow("Manual basic scaling", self.sr_manual_combo)
        upscaling_form.addRow("Auto basic scaling max", self.auto_sr_max_combo)
        upscaling_form.addRow(self.enable_ai_sr_checkbox)
        upscaling_form.addRow(self.enable_rtx_vsr_checkbox)
        upscaling_form.addRow("AI SR model", self.ai_sr_model_combo)
        upscaling_form.addRow(ai_sr_model_actions)
        upscaling_form.addRow("AI SR provider", self.ai_sr_provider_combo)
        upscaling_form.addRow("TensorRT precision", self.ai_sr_trt_precision_combo)
        upscaling_form.addRow(self.ai_sr_require_gpu_checkbox)
        upscaling_form.addRow("AI inference FPS", self.ai_sr_frame_interval_spin)
        upscaling_form.addRow(self.ai_sr_strict_checkbox)
        upscaling_form.addRow("AI SR input alignment", self.ai_sr_input_align_combo)
        upscaling_form.addRow("AI SR ROI overscan %", self.ai_sr_overscan_spin)
        upscaling_form.addRow("AI SR inference divisor", self.ai_sr_inference_divisor_spin)
        upscaling_form.addRow("AI SR detail preserve %", self.ai_sr_detail_preserve_spin)
        upscaling_form.addRow(ai_sr_tuning_actions)
        upscaling_form.addRow(ai_sr_runtime_box)

        ai_sr_postprocess_box = QGroupBox("AI SR Post Process Noise Reduction")
        ai_sr_postprocess_form = QFormLayout(ai_sr_postprocess_box)
        ai_sr_postprocess_form.addRow("Noise Method", self.ai_sr_post_denoise_method_combo)
        ai_sr_postprocess_form.addRow("Noise Level", self.ai_sr_post_denoise_strength_spin)
        ai_sr_postprocess_form.addRow("Artifact Method", self.ai_sr_post_artifact_reduction_method_combo)
        ai_sr_postprocess_form.addRow("Artifact Level", self.ai_sr_post_artifact_reduction_strength_spin)
        ai_sr_postprocess_form.addRow(self.ai_sr_post_exaggeration_checkbox)
        ai_sr_postprocess_form.addRow("Exaggeration Gain", self.ai_sr_post_exaggeration_gain_spin)

        self.perf_guard_checkbox = QCheckBox("Auto performance guard (reduce SR when overloaded)")
        self.perf_guard_checkbox.setChecked(False)
        self.perf_guard_checkbox.toggled.connect(self._on_perf_guard_toggled)
        settings_form.addRow(self.perf_guard_checkbox)

        decklink_box = QGroupBox("Blackmagic I/O")
        decklink_form = QFormLayout(decklink_box)

        self.source_mode_combo = QComboBox()
        self.source_mode_combo.addItems(["Synthetic", "Blackmagic DeckLink"])
        self.source_mode_combo.currentIndexChanged.connect(self._on_blackmagic_combo_changed)
        decklink_form.addRow("Input source", self.source_mode_combo)

        self.decklink_input_device_combo = QComboBox()
        self.decklink_input_device_combo.currentIndexChanged.connect(self._on_decklink_device_changed)
        decklink_form.addRow("Input device", self.decklink_input_device_combo)

        self.decklink_output_device_combo = QComboBox()
        self.decklink_output_device_combo.currentIndexChanged.connect(self._on_decklink_device_changed)
        decklink_form.addRow("Output device", self.decklink_output_device_combo)

        self.decklink_auto_detect_devices = QCheckBox("Auto-detect input/output devices")
        self.decklink_auto_detect_devices.setChecked(True)
        self.decklink_auto_detect_devices.toggled.connect(self._on_auto_detect_toggled)
        decklink_form.addRow(self.decklink_auto_detect_devices)

        self.decklink_input_mode_combo = QComboBox()
        self.decklink_input_mode_combo.currentIndexChanged.connect(self._on_blackmagic_combo_changed)
        decklink_form.addRow("Input mode", self.decklink_input_mode_combo)

        self.decklink_output_mode_combo = QComboBox()
        self.decklink_output_mode_combo.currentIndexChanged.connect(self._on_blackmagic_combo_changed)
        decklink_form.addRow("Output mode", self.decklink_output_mode_combo)

        self.color_space_combo = QComboBox()
        self.color_space_combo.addItems(list(COLOR_SPACE_LABEL_TO_NAME.keys()))
        self.color_space_combo.setCurrentText(
            COLOR_SPACE_NAME_TO_LABEL.get(getattr(self._controller, "color_space", "rec709"), "Rec.709 (SDR)")
        )
        self.color_space_combo.currentIndexChanged.connect(self._on_blackmagic_combo_changed)
        decklink_form.addRow("Output color space", self.color_space_combo)

        self.color_range_combo = QComboBox()
        self.color_range_combo.addItems(list(COLOR_RANGE_LABEL_TO_NAME.keys()))
        self.color_range_combo.setCurrentText(
            COLOR_RANGE_NAME_TO_LABEL.get(getattr(self._controller, "color_range", "limited"), "Limited (Video)")
        )
        self.color_range_combo.currentIndexChanged.connect(self._on_blackmagic_combo_changed)
        decklink_form.addRow("Output color range", self.color_range_combo)

        self.decklink_enable_format_detection = QCheckBox("Enable input format detection")
        self.decklink_enable_format_detection.setChecked(True)
        decklink_form.addRow(self.decklink_enable_format_detection)

        self.decklink_output_buffer_spin = QSpinBox()
        self.decklink_output_buffer_spin.setRange(0, 10)
        self.decklink_output_buffer_spin.setValue(int(self._decklink_output_buffer_frames))
        self.decklink_output_buffer_spin.setToolTip("DeckLink output startup/steady buffer in frames; larger values can smooth short stalls with added latency.")
        self.decklink_output_buffer_spin.valueChanged.connect(self._on_decklink_output_buffer_changed)
        decklink_form.addRow("DeckLink output buffer (frames)", self.decklink_output_buffer_spin)

        self.decklink_fps_priority_guard_checkbox = QCheckBox(
            "Prioritize output frame rate (temporary buffer boost when needed)"
        )
        self.decklink_fps_priority_guard_checkbox.setChecked(bool(self._decklink_buffer_guard_enabled))
        self.decklink_fps_priority_guard_checkbox.toggled.connect(self._on_decklink_buffer_guard_toggled)
        decklink_form.addRow(self.decklink_fps_priority_guard_checkbox)

        self.worker_priority_combo = QComboBox()
        self.worker_priority_combo.addItems(list(WORKER_PRIORITY_LABEL_TO_NAME.keys()))
        self.worker_priority_combo.setCurrentText(
            WORKER_PRIORITY_NAME_TO_LABEL.get(self._worker_process_priority, "Above Normal")
        )
        self.worker_priority_combo.setToolTip(
            "Worker scheduling priority. Higher levels can reduce timing jitter when the system is busy."
        )
        self.worker_priority_combo.currentIndexChanged.connect(self._on_worker_priority_changed)
        decklink_form.addRow("Worker process priority", self.worker_priority_combo)

        self.decklink_pixel_format_combo = QComboBox()
        self.decklink_pixel_format_combo.addItems(["8-bit YUV (UYVY)"])
        self.decklink_pixel_format_combo.setEnabled(False)
        decklink_form.addRow("Pixel format", self.decklink_pixel_format_combo)

        self.decklink_apply_btn = QPushButton("Apply DeckLink Settings")
        self.decklink_apply_btn.clicked.connect(self._on_apply_decklink_settings)
        decklink_form.addRow(self.decklink_apply_btn)

        self.decklink_refresh_btn = QPushButton("Refresh Devices/Modes")
        self.decklink_refresh_btn.clicked.connect(self._refresh_decklink_catalog)
        decklink_form.addRow(self.decklink_refresh_btn)

        self.decklink_status_label = QLabel()
        self.decklink_status_label.setWordWrap(True)
        decklink_form.addRow(self.decklink_status_label)

        roi_box = QGroupBox("ROI")
        roi_form = QFormLayout(roi_box)

        reset_btn = QPushButton("Reset ROI")
        reset_btn.clicked.connect(self._reset_roi)
        roi_form.addRow(reset_btn)

        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, FRAME_W - 2)
        self.roi_x_spin.valueChanged.connect(self._on_roi_spin_changed)
        roi_form.addRow("x", self.roi_x_spin)

        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, FRAME_H - 2)
        self.roi_y_spin.valueChanged.connect(self._on_roi_spin_changed)
        roi_form.addRow("y", self.roi_y_spin)

        self.roi_w_spin = QSpinBox()
        self.roi_w_spin.setRange(2, FRAME_W)
        self.roi_w_spin.setSingleStep(2)
        self.roi_w_spin.valueChanged.connect(self._on_roi_spin_changed)
        roi_form.addRow("w", self.roi_w_spin)

        self.roi_h_spin = QSpinBox()
        self.roi_h_spin.setRange(2, FRAME_H)
        self.roi_h_spin.valueChanged.connect(self._on_roi_spin_changed)
        roi_form.addRow("h", self.roi_h_spin)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(1.0, 16.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setDecimals(2)
        self.scale_spin.valueChanged.connect(self._on_scale_spin_changed)
        roi_form.addRow("Scale", self.scale_spin)

        self.roi_smoothing_slider = QSlider(Qt.Horizontal)
        self.roi_smoothing_slider.setRange(0, 10)
        self.roi_smoothing_slider.setSingleStep(1)
        self.roi_smoothing_slider.setPageStep(1)
        self.roi_smoothing_slider.setValue(int(self._roi_smoothing_percent))
        self.roi_smoothing_slider.valueChanged.connect(self._on_roi_smoothing_changed)

        self.roi_smoothing_value_label = QLabel(f"{int(self._roi_smoothing_percent)}/10")
        roi_smoothing_row = QWidget()
        roi_smoothing_layout = QHBoxLayout(roi_smoothing_row)
        roi_smoothing_layout.setContentsMargins(0, 0, 0, 0)
        roi_smoothing_layout.setSpacing(8)
        roi_smoothing_layout.addWidget(self.roi_smoothing_slider, 1)
        roi_smoothing_layout.addWidget(self.roi_smoothing_value_label)
        roi_form.addRow("Smoothing", roi_smoothing_row)

        self.roi_latency_smoothing_slider = QSlider(Qt.Horizontal)
        self.roi_latency_smoothing_slider.setRange(0, 100)
        self.roi_latency_smoothing_slider.setSingleStep(1)
        self.roi_latency_smoothing_slider.setPageStep(5)
        self.roi_latency_smoothing_slider.setValue(int(self._roi_latency_smoothing_percent))
        self.roi_latency_smoothing_slider.valueChanged.connect(self._on_roi_latency_smoothing_changed)

        self.roi_latency_smoothing_value_label = QLabel(f"{self._roi_latency_smoothing_percent}%")
        roi_latency_smoothing_row = QWidget()
        roi_latency_smoothing_layout = QHBoxLayout(roi_latency_smoothing_row)
        roi_latency_smoothing_layout.setContentsMargins(0, 0, 0, 0)
        roi_latency_smoothing_layout.setSpacing(8)
        roi_latency_smoothing_layout.addWidget(self.roi_latency_smoothing_slider, 1)
        roi_latency_smoothing_layout.addWidget(self.roi_latency_smoothing_value_label)
        roi_form.addRow("Smoothing+Latency", roi_latency_smoothing_row)

        self.roi_drag_x_hysteresis_spin = QDoubleSpinBox()
        self.roi_drag_x_hysteresis_spin.setRange(0.10, 1.20)
        self.roi_drag_x_hysteresis_spin.setDecimals(2)
        self.roi_drag_x_hysteresis_spin.setSingleStep(0.01)
        self.roi_drag_x_hysteresis_spin.setValue(float(self._roi_drag_x_hysteresis_px))
        self.roi_drag_x_hysteresis_spin.valueChanged.connect(self._on_roi_drag_x_hysteresis_changed)
        self.roi_drag_x_hysteresis_spin.setToolTip("Lower values track faster but can jitter; higher values resist micro-wobble.")
        roi_form.addRow("Drag X hysteresis (px)", self.roi_drag_x_hysteresis_spin)

        self.roi_manual_drag_hold_spin = QDoubleSpinBox()
        self.roi_manual_drag_hold_spin.setRange(0.05, 0.50)
        self.roi_manual_drag_hold_spin.setDecimals(2)
        self.roi_manual_drag_hold_spin.setSingleStep(0.01)
        self.roi_manual_drag_hold_spin.setValue(float(self._roi_manual_drag_hold_s))
        self.roi_manual_drag_hold_spin.valueChanged.connect(self._on_roi_manual_drag_hold_changed)
        self.roi_manual_drag_hold_spin.setToolTip("How long worker keeps softer follow mode after each manual drag update.")
        roi_form.addRow("Manual drag hold (s)", self.roi_manual_drag_hold_spin)

        self.roi_min_drag_nudge_spin = QDoubleSpinBox()
        self.roi_min_drag_nudge_spin.setRange(0.00, 0.50)
        self.roi_min_drag_nudge_spin.setDecimals(3)
        self.roi_min_drag_nudge_spin.setSingleStep(0.005)
        self.roi_min_drag_nudge_spin.setValue(float(self._roi_min_drag_nudge))
        self.roi_min_drag_nudge_spin.valueChanged.connect(self._on_roi_min_drag_nudge_changed)
        self.roi_min_drag_nudge_spin.setToolTip("Minimum subpixel motion while drag is active and carrier ROI is unchanged.")
        roi_form.addRow("Min subpixel nudge", self.roi_min_drag_nudge_spin)

        self.roi_interlaced_field2_phase_spin = QDoubleSpinBox()
        self.roi_interlaced_field2_phase_spin.setRange(INTERLACED_FIELD2_PHASE_MIN, INTERLACED_FIELD2_PHASE_MAX)
        self.roi_interlaced_field2_phase_spin.setDecimals(2)
        self.roi_interlaced_field2_phase_spin.setSingleStep(0.05)
        self.roi_interlaced_field2_phase_spin.setValue(float(self._interlaced_field2_phase_fraction))
        self.roi_interlaced_field2_phase_spin.valueChanged.connect(self._on_interlaced_field2_phase_fraction_changed)
        self.roi_interlaced_field2_phase_spin.setToolTip(
            "2nd field timing within one output frame (-1.00 to 2.00). 0=field1 at current frame, 0.5=half-step, 1.0=next frame."
        )
        roi_form.addRow("Interlaced field2 phase", self.roi_interlaced_field2_phase_spin)

        self.roi_transition_frames_spin = QSpinBox()
        self.roi_transition_frames_spin.setRange(1, 600)
        self.roi_transition_frames_spin.setValue(int(self._roi_keyframe_transition_default_frames))
        self.roi_transition_units_label = QLabel("Transition (frames)")
        roi_form.addRow(self.roi_transition_units_label, self.roi_transition_frames_spin)

        self.roi_interp_mode_combo = QComboBox()
        self.roi_interp_mode_combo.addItems(["Linear", "Ease In/Out", "Ease Out"])
        self.roi_interp_mode_combo.setCurrentText("Ease In/Out")
        roi_form.addRow("Interp Mode", self.roi_interp_mode_combo)

        self.roi_keyframe_duration_override_btn = QPushButton("Override Key Duration")
        self.roi_keyframe_duration_override_btn.setCheckable(True)
        self.roi_keyframe_duration_override_btn.setToolTip("When enabled, uses Transition (frames) instead of keyframe-stored duration during recall.")
        roi_form.addRow(self.roi_keyframe_duration_override_btn)

        keyframe_row = QWidget()
        keyframe_layout = QHBoxLayout(keyframe_row)
        keyframe_layout.setContentsMargins(0, 0, 0, 0)
        keyframe_layout.setSpacing(8)
        key_button_min_height = 120

        self.roi_save_key_btn = QPushButton("SAVE KEY")
        self.roi_save_key_btn.setCheckable(True)
        self.roi_save_key_btn.setMinimumHeight(key_button_min_height)
        self.roi_save_key_btn.toggled.connect(self._on_roi_save_key_toggled)
        keyframe_layout.addWidget(self.roi_save_key_btn)

        self.roi_key1_btn = QPushButton("KEY 1")
        self.roi_key1_btn.setMinimumHeight(key_button_min_height)
        self.roi_key1_btn.clicked.connect(lambda: self._on_roi_key_slot_pressed(1))
        keyframe_layout.addWidget(self.roi_key1_btn)

        self.roi_key2_btn = QPushButton("KEY 2")
        self.roi_key2_btn.setMinimumHeight(key_button_min_height)
        self.roi_key2_btn.clicked.connect(lambda: self._on_roi_key_slot_pressed(2))
        keyframe_layout.addWidget(self.roi_key2_btn)

        self.roi_key3_btn = QPushButton("KEY 3")
        self.roi_key3_btn.setMinimumHeight(key_button_min_height)
        self.roi_key3_btn.clicked.connect(lambda: self._on_roi_key_slot_pressed(3))
        keyframe_layout.addWidget(self.roi_key3_btn)

        self.roi_key4_btn = QPushButton("KEY 4")
        self.roi_key4_btn.setMinimumHeight(key_button_min_height)
        self.roi_key4_btn.clicked.connect(lambda: self._on_roi_key_slot_pressed(4))
        keyframe_layout.addWidget(self.roi_key4_btn)

        roi_form.addRow(keyframe_row)

        timecode_row = QWidget()
        timecode_row_layout = QHBoxLayout(timecode_row)
        timecode_row_layout.setContentsMargins(0, 0, 0, 0)
        timecode_row_layout.setSpacing(8)

        self.decklink_timecode_label = QLabel(self._decklink_timecode_display_text)
        self.decklink_timecode_label.setWordWrap(True)
        timecode_row_layout.addWidget(self.decklink_timecode_label, 1)

        self.decklink_timecode_refresh_btn = QPushButton("Refresh")
        self.decklink_timecode_refresh_btn.setMaximumWidth(84)
        self.decklink_timecode_refresh_btn.clicked.connect(self._on_decklink_timecode_refresh_clicked)
        timecode_row_layout.addWidget(self.decklink_timecode_refresh_btn, 0)

        roi_form.addRow(timecode_row)

        keyframe_spacing_row = QWidget()
        keyframe_spacing_row.setFixedHeight(16)
        roi_form.addRow(keyframe_spacing_row)

        post_vsr_scaling_box = QGroupBox("Post VSR Scaling")
        post_vsr_scaling_form = QFormLayout(post_vsr_scaling_box)

        self.rtx_vsr_scaling_apply_btn = QPushButton("Apply VSR Scaling")
        self.rtx_vsr_scaling_apply_btn.clicked.connect(self._on_rtx_vsr_settings_apply_clicked)

        self.rtx_vsr_scaling_info_label = QLabel("VSR scaling info will appear after worker initialization.")
        self.rtx_vsr_scaling_info_label.setWordWrap(True)

        post_vsr_scaling_form.addRow("Internal scale", self.rtx_vsr_scale_combo)
        post_vsr_scaling_form.addRow("Post-VSR scaling method", self.rtx_vsr_post_scale_method_combo)
        post_vsr_scaling_form.addRow(self.rtx_vsr_scaling_apply_btn)
        post_vsr_scaling_form.addRow(self.rtx_vsr_scaling_info_label)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)

        controls_hint = QLabel(
            "Controls:\n"
            "- Input view only: Mouse drag to move ROI\n"
            "- Input view only: Drag bottom-right handle to resize ROI\n"
            "- Input view only: Wheel/Touchpad to zoom\n"
            "- Input view only: Touch pinch zoom disabled (tablet)\n"
            "- Input view only: Arrow keys move ROI\n"
            "- Input view only: Shift+Arrows resize ROI\n"
            "- +/-: zoom"
        )
        controls_hint.setWordWrap(True)

        layout.addWidget(roi_box)
        layout.addWidget(decklink_box)
        layout.addWidget(deinterlace_box)
        layout.addWidget(denoise_box)
        layout.addWidget(upscaling_box)
        layout.addWidget(rtx_vsr_box)
        layout.addWidget(settings_box)
        layout.addWidget(ai_sr_postprocess_box)
        layout.addWidget(post_vsr_scaling_box)
        layout.addWidget(controls_hint)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        return panel

    def _build_fullscreen_keyframe_toolbar(self, view_name: str) -> QWidget:
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(8)

        save_btn = QPushButton("SAVE KEY")
        save_btn.setCheckable(True)
        save_btn.setMinimumHeight(120)
        save_btn.toggled.connect(self._on_roi_save_key_toggled)
        toolbar_layout.addWidget(save_btn)

        key1_btn = QPushButton("KEY 1")
        key1_btn.setMinimumHeight(120)
        key1_btn.clicked.connect(lambda: self._on_roi_key_slot_pressed(1))
        toolbar_layout.addWidget(key1_btn)

        key2_btn = QPushButton("KEY 2")
        key2_btn.setMinimumHeight(120)
        key2_btn.clicked.connect(lambda: self._on_roi_key_slot_pressed(2))
        toolbar_layout.addWidget(key2_btn)

        key3_btn = QPushButton("KEY 3")
        key3_btn.setMinimumHeight(120)
        key3_btn.clicked.connect(lambda: self._on_roi_key_slot_pressed(3))
        toolbar_layout.addWidget(key3_btn)

        key4_btn = QPushButton("KEY 4")
        key4_btn.setMinimumHeight(120)
        key4_btn.clicked.connect(lambda: self._on_roi_key_slot_pressed(4))
        toolbar_layout.addWidget(key4_btn)

        self._fullscreen_keyframe_toolbars[view_name] = toolbar
        self._fullscreen_roi_save_key_buttons[view_name] = save_btn
        self._fullscreen_roi_key_slot_buttons[view_name] = (key1_btn, key2_btn, key3_btn, key4_btn)
        toolbar.setVisible(False)
        return toolbar

    def _build_fullscreen_keyframe_side_panel(self, view_name: str) -> QWidget:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(10)

        title = QLabel("KEYFRAME")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("QLabel { font-size: 18px; font-weight: 700; }")
        panel_layout.addWidget(title)

        exit_fullscreen_btn = QPushButton("EXIT\nFULL SCREEN")
        exit_fullscreen_btn.setMinimumWidth(150)
        exit_fullscreen_btn.setMinimumHeight(96)
        exit_fullscreen_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: 700; padding: 8px; }")
        exit_fullscreen_btn.clicked.connect(lambda: self._set_fullscreen_view(None))
        panel_layout.addWidget(exit_fullscreen_btn)

        transition_label = QLabel("Transition\n(frames)")
        transition_label.setAlignment(Qt.AlignCenter)
        transition_label.setStyleSheet("QLabel { font-size: 14px; font-weight: 600; }")
        panel_layout.addWidget(transition_label)

        transition_spin = QSpinBox()
        transition_spin.setRange(1, 600)
        transition_spin.setValue(int(self._roi_keyframe_transition_default_frames))
        transition_spin.setMinimumWidth(150)
        transition_spin.setMinimumHeight(72)
        transition_spin.setStyleSheet(
            "QSpinBox { font-size: 22px; font-weight: 700; padding: 8px 12px; }"
            "QSpinBox::up-button, QSpinBox::down-button { width: 34px; }"
        )
        transition_spin.valueChanged.connect(self._on_fullscreen_transition_rate_changed)
        panel_layout.addWidget(transition_spin)

        override_btn = QPushButton("OVERRIDE\nKEY DURATION")
        override_btn.setCheckable(True)
        override_btn.setMinimumWidth(150)
        override_btn.setMinimumHeight(96)
        override_btn.setToolTip("Use Transition (frames) as recall duration instead of the keyframe's stored duration.")
        override_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: 700; padding: 8px; }")
        override_btn.toggled.connect(self._on_fullscreen_override_duration_toggled)
        panel_layout.addWidget(override_btn)

        buffer_label = QLabel("DeckLink buffer\n(frames)")
        buffer_label.setAlignment(Qt.AlignCenter)
        buffer_label.setStyleSheet("QLabel { font-size: 14px; font-weight: 600; }")
        panel_layout.addWidget(buffer_label)

        buffer_spin = QSpinBox()
        buffer_spin.setRange(0, 10)
        buffer_spin.setValue(int(self._decklink_output_buffer_frames))
        buffer_spin.setMinimumWidth(150)
        buffer_spin.setMinimumHeight(72)
        buffer_spin.setStyleSheet(
            "QSpinBox { font-size: 22px; font-weight: 700; padding: 8px 12px; }"
            "QSpinBox::up-button, QSpinBox::down-button { width: 34px; }"
        )
        buffer_spin.setToolTip("DeckLink output startup/steady buffer in frames.")
        buffer_spin.valueChanged.connect(self._on_fullscreen_decklink_output_buffer_changed)
        panel_layout.addWidget(buffer_spin)

        reset_roi_btn = QPushButton("RESET\nROI")
        reset_roi_btn.setMinimumWidth(150)
        reset_roi_btn.setMinimumHeight(96)
        reset_roi_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: 700; padding: 8px; }")
        reset_roi_btn.clicked.connect(self._reset_roi)
        panel_layout.addWidget(reset_roi_btn)

        panel_layout.addStretch(1)

        self._fullscreen_keyframe_side_panels[view_name] = panel
        self._fullscreen_roi_transition_labels[view_name] = transition_label
        self._fullscreen_roi_transition_rate_spins[view_name] = transition_spin
        self._fullscreen_roi_duration_override_buttons[view_name] = override_btn
        self._fullscreen_decklink_output_buffer_spins[view_name] = buffer_spin
        panel.setVisible(False)
        return panel

    def _roi_transition_unit_label_text(self) -> str:
        if self.source_mode_combo.currentText() != "Blackmagic DeckLink":
            return "frames"
        return "fields" if self._decklink_output_mode_is_interlaced() else "frames"

    def _sync_roi_transition_unit_labels(self) -> None:
        unit = self._roi_transition_unit_label_text()
        self.roi_transition_units_label.setText(f"Transition ({unit})")

        for label in self._fullscreen_roi_transition_labels.values():
            label.setText(f"Transition ({unit})")

        self.roi_keyframe_duration_override_btn.setToolTip(
            f"When enabled, uses Transition ({unit}) instead of keyframe-stored duration during recall."
        )
        for button in self._fullscreen_roi_duration_override_buttons.values():
            button.setToolTip(
                f"Use Transition ({unit}) as recall duration instead of the keyframe's stored duration."
            )

    def _on_fullscreen_transition_rate_changed(self, value: int) -> None:
        normalized = max(1, min(600, int(value)))
        if self.roi_transition_frames_spin.value() != normalized:
            self.roi_transition_frames_spin.setValue(normalized)
        else:
            self._sync_fullscreen_transition_rate_from_main(normalized)

    def _on_fullscreen_override_duration_toggled(self, checked: bool) -> None:
        target = bool(checked)
        if self.roi_keyframe_duration_override_btn.isChecked() != target:
            self.roi_keyframe_duration_override_btn.setChecked(target)
        else:
            self._sync_fullscreen_override_duration_from_main(target)

    def _sync_fullscreen_transition_rate_from_main(self, value: int) -> None:
        normalized = max(1, min(600, int(value)))
        for spin in self._fullscreen_roi_transition_rate_spins.values():
            previous_block = spin.blockSignals(True)
            spin.setValue(normalized)
            spin.blockSignals(previous_block)

    def _sync_fullscreen_override_duration_from_main(self, checked: bool) -> None:
        target = bool(checked)
        for button in self._fullscreen_roi_duration_override_buttons.values():
            previous_block = button.blockSignals(True)
            button.setChecked(target)
            button.blockSignals(previous_block)

    def _on_fullscreen_decklink_output_buffer_changed(self, value: int) -> None:
        normalized = max(0, min(10, int(value)))
        if self.decklink_output_buffer_spin.value() != normalized:
            self.decklink_output_buffer_spin.setValue(normalized)
        else:
            self._sync_fullscreen_decklink_output_buffer_from_main(normalized)

    def _sync_fullscreen_decklink_output_buffer_from_main(self, value: int) -> None:
        normalized = max(0, min(10, int(value)))
        for spin in self._fullscreen_decklink_output_buffer_spins.values():
            previous_block = spin.blockSignals(True)
            spin.setValue(normalized)
            spin.blockSignals(previous_block)

    def _setup_shortcuts(self) -> None:
        reset_action = QAction(self)
        reset_action.setShortcut("R")
        reset_action.triggered.connect(self._reset_roi)
        self.addAction(reset_action)

        fullscreen_action = QAction(self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self._toggle_fullscreen_view)
        self.addAction(fullscreen_action)

    def _toggle_fullscreen_view(self) -> None:
        if self._fullscreen_view_name is not None:
            self._set_fullscreen_view(None)
            return

        focus_widget = QApplication.focusWidget()
        if focus_widget is not None and (
            focus_widget is self._output_canvas or self._output_canvas.isAncestorOf(focus_widget)
        ):
            self._set_fullscreen_view("output")
            return

        self._set_fullscreen_view("input")

    def _sync_fullscreen_button_states(self) -> None:
        active_view = self._fullscreen_view_name
        for view_name, button in self._fullscreen_enter_buttons.items():
            button.setEnabled(active_view is None)
            button.setVisible(active_view is None)

    def _capture_windowed_geometry_before_fullscreen(self) -> None:
        self._windowed_was_maximized_before_fullscreen = self.isMaximized()
        self._windowed_geometry_before_fullscreen = self.geometry()

    def _restore_windowed_geometry_after_fullscreen(self) -> None:
        geometry = self._windowed_geometry_before_fullscreen
        was_maximized = self._windowed_was_maximized_before_fullscreen

        self.showNormal()
        if geometry is not None and geometry.isValid():
            self.setGeometry(geometry)
        if was_maximized:
            self.showMaximized()

    def _perf_add(self, stage_name: str, elapsed_ms: float) -> None:
        if stage_name not in self._perf_stage_sums_ms:
            return
        self._perf_stage_sums_ms[stage_name] += elapsed_ms
        self._perf_stage_counts[stage_name] += 1
        if elapsed_ms > self._perf_stage_peaks_ms[stage_name]:
            self._perf_stage_peaks_ms[stage_name] = elapsed_ms

    def _perf_snapshot_and_reset(self) -> dict[str, tuple[float, float]]:
        snapshot: dict[str, tuple[float, float]] = {}
        for stage_name in self._perf_stage_sums_ms:
            count = self._perf_stage_counts[stage_name]
            avg_ms = self._perf_stage_sums_ms[stage_name] / count if count > 0 else 0.0
            peak_ms = self._perf_stage_peaks_ms[stage_name]
            snapshot[stage_name] = (avg_ms, peak_ms)
            self._perf_stage_sums_ms[stage_name] = 0.0
            self._perf_stage_counts[stage_name] = 0
            self._perf_stage_peaks_ms[stage_name] = 0.0
        return snapshot

    def _tick(self) -> None:
        try:
            tick_start = time.perf_counter()

            if self._source_mode == "Blackmagic DeckLink" and self._controller_backend == "worker-process":
                t0 = time.perf_counter()
                decklink_frame = self._controller.decklink_tick(timeout_ms=50)
                self._perf_add("acquire", (time.perf_counter() - t0) * 1000.0)

                if decklink_frame is None:
                    self._no_frame_counter += 1
                    if self._no_frame_counter % 20 == 0:
                        LOGGER.warning("No DeckLink worker frames yet (count=%d)", self._no_frame_counter)
                    reason = None
                    if hasattr(self._controller, "decklink_no_frame_reason"):
                        reason = self._controller.decklink_no_frame_reason()
                    if reason == "sessions_not_started":
                        self._update_status(
                            "DeckLink worker sessions not started",
                            suppress_repeat_window_s=10.0,
                        )
                    elif reason == "tick_dropped_queue_full":
                        self._update_status(
                            "DeckLink worker queue is saturated; dropping preview tick requests",
                            suppress_repeat_window_s=3.0,
                        )
                    elif reason == "tick_request_stalled":
                        self._update_status(
                            "DeckLink worker tick request stalled; retrying",
                            suppress_repeat_window_s=3.0,
                        )
                    else:
                        self._update_status(
                            "DeckLink worker active but no input frames yet; check source signal and input mode",
                            suppress_repeat_window_s=10.0,
                        )
                    self._update_decklink_timecode_from_controller(placeholder="Timecode: waiting for DeckLink frames...")
                    return

                input_frame, output_frame = decklink_frame
                self._no_frame_counter = 0
                self._sync_backend_roi_from_worker()
                self._update_decklink_timecode_from_controller(placeholder="Timecode: none detected")
                preview_updated = True
                if hasattr(self._controller, "consume_decklink_frame_updated"):
                    preview_updated = bool(self._controller.consume_decklink_frame_updated())

                interaction_scale = 1.0
                if self._manual_roi_interaction_active():
                    interaction_scale = self._manual_roi_preview_reduce_scale

                self._perf_add("process", (time.perf_counter() - t0) * 1000.0)

                input_preview_size = self._preview_target_for_view("input")
                input_preview_size = self._scaled_preview_target(input_preview_size, interaction_scale)
                if input_preview_size is not None and preview_updated:
                    t1 = time.perf_counter()
                    input_image, input_backing = uyvy_to_qimage(
                        input_frame,
                        preview_max_w=input_preview_size[0],
                        preview_max_h=input_preview_size[1],
                        color_space=getattr(self._controller, "color_space", "rec709"),
                        color_range=getattr(self._controller, "color_range", "limited"),
                    )
                    self._input_canvas.set_image(input_image, input_backing)
                    self._perf_add("convert_in", (time.perf_counter() - t1) * 1000.0)

                output_preview_size = self._preview_target_for_view("output")
                output_preview_size = self._scaled_preview_target(output_preview_size, interaction_scale)
                if output_preview_size is not None and preview_updated:
                    t1 = time.perf_counter()
                    output_image, output_backing = uyvy_to_qimage(
                        output_frame,
                        preview_max_w=output_preview_size[0],
                        preview_max_h=output_preview_size[1],
                        color_space=getattr(self._controller, "color_space", "rec709"),
                        color_range=getattr(self._controller, "color_range", "limited"),
                    )
                    self._output_canvas.set_image(output_image, output_backing)
                    self._perf_add("convert_out", (time.perf_counter() - t1) * 1000.0)

                self._perf_add("tick", (time.perf_counter() - tick_start) * 1000.0)
                self._frame_count += 1

                now = time.perf_counter()
                dt = now - self._last_stat_time
                if dt >= 1.0:
                    fps = self._frame_count / dt
                    perf = self._perf_snapshot_and_reset()
                    self._frame_count = 0
                    self._last_stat_time = now
                    mode_text = "Auto" if self._controller.basic_scaling_auto_mode else "Manual"
                    flavor_text = SR_FLAVOR_NAME_TO_LABEL.get(self._controller.basic_scaling_method, self._controller.basic_scaling_method)
                    ai_sr_state = "off"
                    ai_sr_detail = ""
                    if getattr(self._controller, "ai_sr_enabled", False):
                        ai_sr_state = "active" if getattr(self._controller, "ai_sr_active", False) else "requested"
                        ai_sr_info = getattr(self._controller, "ai_sr_info", None)
                        ai_sr_error = getattr(self._controller, "ai_sr_error", None)
                        if ai_sr_info and ai_sr_state == "active":
                            provider = ai_sr_info.get("provider", "unknown")
                            strict_text = " strict" if bool(ai_sr_info.get("strict_mode", False)) else " async"
                            ai_sr_detail = f" ({provider},{strict_text})"
                        elif ai_sr_error and ai_sr_state != "active":
                            ai_sr_detail = f" ({ai_sr_error})"
                        elif getattr(self._controller, "ai_sr_last_warning", None):
                            ai_sr_detail = f" ({self._controller.ai_sr_last_warning})"
                    worker_fps = 0.0
                    ai_applied = 0
                    ai_reused = 0
                    ai_passthrough = 0
                    if hasattr(self._controller, "decklink_processed_fps"):
                        worker_fps = float(self._controller.decklink_processed_fps())
                    if hasattr(self._controller, "decklink_ai_sr_counts"):
                        ai_applied, ai_reused, ai_passthrough = self._controller.decklink_ai_sr_counts()
                    ai_counts = f"fresh={ai_applied}, reused={ai_reused}, pass={ai_passthrough}"
                    ai_refresh_fps = 0.0
                    ai_latest_age_ms = -1.0
                    ai_completed = 0
                    if hasattr(self._controller, "decklink_ai_refresh_stats"):
                        ai_refresh_fps, ai_latest_age_ms, ai_completed = self._controller.decklink_ai_refresh_stats()
                    rtx_applied = False
                    rtx_delta = 0.0
                    if hasattr(self._controller, "decklink_rtx_stats"):
                        rtx_applied, rtx_delta = self._controller.decklink_rtx_stats()
                    stage_enable = {"preprocess": False, "basic_scaling": False, "ai_sr": False, "rtx_vsr": False}
                    stage_last = {"preprocess": False, "basic_scaling": False, "ai_sr": False, "rtx_vsr": False}
                    stage_counts = {"preprocess": 0, "basic_scaling": 0, "ai_sr": 0, "rtx_vsr": 0, "passthrough": 0}
                    if hasattr(self._controller, "decklink_stage_telemetry"):
                        stage_enable, stage_last, stage_counts = self._controller.decklink_stage_telemetry()

                    stage_enable_text = (
                        f"P={'1' if stage_enable.get('preprocess', False) else '0'}"
                        f" B={'1' if stage_enable.get('basic_scaling', False) else '0'}"
                        f" A={'1' if stage_enable.get('ai_sr', False) else '0'}"
                        f" R={'1' if stage_enable.get('rtx_vsr', False) else '0'}"
                    )
                    stage_last_text = (
                        f"P={'1' if stage_last.get('preprocess', False) else '0'}"
                        f" B={'1' if stage_last.get('basic_scaling', False) else '0'}"
                        f" A={'1' if stage_last.get('ai_sr', False) else '0'}"
                        f" R={'1' if stage_last.get('rtx_vsr', False) else '0'}"
                    )
                    stage_count_text = (
                        f"P={int(stage_counts.get('preprocess', 0))}"
                        f" B={int(stage_counts.get('basic_scaling', 0))}"
                        f" A={int(stage_counts.get('ai_sr', 0))}"
                        f" R={int(stage_counts.get('rtx_vsr', 0))}"
                        f" X={int(stage_counts.get('passthrough', 0))}"
                    )
                    ai_timing = {}
                    if hasattr(self._controller, "decklink_ai_timing_stats"):
                        ai_timing = dict(self._controller.decklink_ai_timing_stats())

                    health_summary = self._evaluate_frame_and_buffer_health(
                        preview_fps=float(fps),
                        output_fps=float(worker_fps),
                    )

                    ai_stage_timing_text = ""
                    avg_prep = ai_timing.get("avg_prep_ms")
                    avg_infer = ai_timing.get("avg_infer_ms")
                    avg_post = ai_timing.get("avg_post_ms")
                    avg_total = ai_timing.get("avg_total_ms")
                    if isinstance(avg_prep, (int, float)) and isinstance(avg_infer, (int, float)) and isinstance(avg_post, (int, float)):
                        total_text = f"/{float(avg_total):.1f}" if isinstance(avg_total, (int, float)) else ""
                        ai_stage_timing_text = (
                            f" | AI ms p/i/o{('/t' if total_text else '')}="
                            f"{float(avg_prep):.1f}/{float(avg_infer):.1f}/{float(avg_post):.1f}{total_text}"
                        )

                    rtx_vsr_state = "off"
                    rtx_vsr_detail = ""
                    if getattr(self._controller, "rtx_vsr_enabled", False):
                        rtx_vsr_state = "active" if getattr(self._controller, "rtx_vsr_active", False) else "requested"
                        rtx_vsr_info = getattr(self._controller, "rtx_vsr_info", None)
                        rtx_vsr_error = getattr(self._controller, "rtx_vsr_error", None)
                        if rtx_vsr_info and rtx_vsr_state == "active":
                            quality = rtx_vsr_info.get("quality", getattr(self._controller, "rtx_vsr_quality", "high"))
                            thdr_enabled = bool(rtx_vsr_info.get("thdr_enabled", False))
                            if thdr_enabled:
                                rtx_vsr_detail = f" ({quality}, thdr=on)"
                            else:
                                rtx_vsr_detail = f" ({quality}, thdr=off)"
                        elif rtx_vsr_error and rtx_vsr_state != "active":
                            rtx_vsr_detail = f" ({rtx_vsr_error})"
                    if getattr(self._controller, "ai_sr_enabled", False):
                        basic_status_text = "Basic scaling=auto-disabled (AI SR ONNX)"
                    else:
                        basic_status_text = (
                            f"Basic scaling mode={mode_text}"
                            f" | Basic scaling method={flavor_text}"
                            f" | effective scaling={self._controller.effective_scale()}"
                        )
                    self._update_status(
                        f"Running | Preview FPS={fps:.1f} | Output FPS={worker_fps:.1f} | {basic_status_text} | AI SR={ai_sr_state}{ai_sr_detail} | AI refresh FPS={ai_refresh_fps:.2f} | AI age={ai_latest_age_ms:.0f}ms | AI completed={ai_completed} | RTX VSR={rtx_vsr_state}{rtx_vsr_detail} | RTX applied={'yes' if rtx_applied else 'no'} | RTX delta={rtx_delta:.2f} | AI frames {ai_counts}{ai_stage_timing_text} | Stage enabled[{stage_enable_text}] | Stage last[{stage_last_text}] | Stage counts[{stage_count_text}] | {health_summary}"
                    )
                    LOGGER.info(
                        (
                            "PERF | preview_fps=%.1f | worker_fps=%.1f | acquire=%.2f/%.2fms | process=%.2f/%.2fms | "
                            "output=%.2f/%.2fms | conv_in=%.2f/%.2fms | conv_out=%.2f/%.2fms | tick=%.2f/%.2fms"
                        ),
                        fps,
                        worker_fps,
                        perf["acquire"][0],
                        perf["acquire"][1],
                        perf["process"][0],
                        perf["process"][1],
                        perf["output"][0],
                        perf["output"][1],
                        perf["convert_in"][0],
                        perf["convert_in"][1],
                        perf["convert_out"][0],
                        perf["convert_out"][1],
                        perf["tick"][0],
                        perf["tick"][1],
                    )
                    self.decklink_status_label.setText(
                        (
                            f"DeckLink streaming via worker process | preview_fps={fps:.1f} | "
                            f"output_fps={worker_fps:.1f} | {health_summary} | "
                            f"drop_events={self._health_drop_events_total} "
                            f"(interp={self._health_drop_events_interpolation}) | "
                            f"buffer_warn_events={self._health_buffer_warn_events}"
                        )
                    )

                    if self._roi_diag_canvas_events > 0 or self._manual_roi_interaction_active():
                        send_avg = self._roi_diag_controller_send_ms_sum / max(1, self._roi_diag_controller_send_attempts)
                        send_max = self._roi_diag_controller_send_ms_max

                        ctrl_stats: dict[str, object] = {}
                        if hasattr(self._controller, "control_send_stats_snapshot"):
                            try:
                                ctrl_stats = dict(self._controller.control_send_stats_snapshot(reset=True))
                            except Exception:
                                ctrl_stats = {}

                        queue_depths: dict[str, int] = {}
                        queue_drops: dict[str, int] = {}
                        if hasattr(self._controller, "decklink_queue_telemetry"):
                            try:
                                queue_depths, queue_drops = self._controller.decklink_queue_telemetry()
                            except Exception:
                                queue_depths, queue_drops = {}, {}

                        LOGGER.info(
                            (
                                "ROI_DIAG | preview_fps=%.1f | output_fps=%.1f | canvas_events=%d | "
                                "roi_send_attempts=%d | roi_send_ok=%d | roi_send_drop=%d | roi_send_ms=%.2f/%.2f | "
                                "ctrl_attempted=%s | ctrl_sent=%s | ctrl_dropped=%s | ctrl_qfull=%s | "
                                "ctrl_compactions=%s | ctrl_roi_drop=%s | ctrl_send_ms=%.2f/%.2f | "
                                "qdepth[c2p=%s,p2u=%s,u2o=%s] | qdrop[c=%s,p=%s,u=%s]"
                            ),
                            fps,
                            worker_fps,
                            self._roi_diag_canvas_events,
                            self._roi_diag_controller_send_attempts,
                            self._roi_diag_controller_send_success,
                            self._roi_diag_controller_send_drops,
                            send_avg,
                            send_max,
                            ctrl_stats.get("attempted", 0),
                            ctrl_stats.get("sent", 0),
                            ctrl_stats.get("dropped", 0),
                            ctrl_stats.get("queue_full", 0),
                            ctrl_stats.get("compactions", 0),
                            ctrl_stats.get("compaction_roi_dropped", 0),
                            float(ctrl_stats.get("avg_send_ms", 0.0)),
                            float(ctrl_stats.get("max_send_ms", 0.0)),
                            queue_depths.get("capture_to_preprocess", 0),
                            queue_depths.get("preprocess_to_upscale", 0),
                            queue_depths.get("upscale_to_output", 0),
                            queue_drops.get("capture", 0),
                            queue_drops.get("preprocess", 0),
                            queue_drops.get("upscale", 0),
                        )

                    transition_state_for_log: dict[str, object] = {}
                    if hasattr(self._controller, "decklink_roi_transition_state"):
                        try:
                            transition_state_for_log = dict(self._controller.decklink_roi_transition_state())
                        except Exception:
                            transition_state_for_log = {}
                    transition_active_for_log = bool(transition_state_for_log.get("active", False))
                    output_interlaced_raw = getattr(self._controller, "decklink_output_is_interlaced", False)
                    if callable(output_interlaced_raw):
                        try:
                            output_interlaced_raw = output_interlaced_raw()
                        except Exception:
                            output_interlaced_raw = False
                    output_interlaced_for_log = bool(output_interlaced_raw)
                    transition_units_raw = getattr(
                        self._controller,
                        "decklink_transition_units_per_output_frame",
                        0.0,
                    )
                    if callable(transition_units_raw):
                        try:
                            transition_units_raw = transition_units_raw()
                        except Exception:
                            transition_units_raw = 0.0
                    try:
                        transition_units_for_log = float(transition_units_raw)
                    except Exception:
                        transition_units_for_log = 0.0
                    phase_for_log = transition_state_for_log.get("interlaced_field_phase")
                    phase_present_for_log = isinstance(phase_for_log, dict)
                    controller_field2_phase_for_log = _clamp_interlaced_field2_phase_fraction(
                        float(getattr(self._controller, "interlaced_field2_phase_fraction", self._interlaced_field2_phase_fraction))
                    )
                    phase_disabled_for_log = abs(controller_field2_phase_for_log) <= 1e-4
                    LOGGER.info(
                        (
                            "ROI_FIELD_GATE | active=%s | interlaced=%s | phase_present=%s | phase_disabled=%s | "
                            "units=%.2f | phase2=%.2f | progress=%.3f/%d | mode=%s"
                        ),
                        transition_active_for_log,
                        output_interlaced_for_log,
                        phase_present_for_log,
                        phase_disabled_for_log,
                        transition_units_for_log,
                        controller_field2_phase_for_log,
                        float(transition_state_for_log.get("frame_progress", 0.0)),
                        max(0, int(transition_state_for_log.get("total_frames", 0))),
                        str(transition_state_for_log.get("interpolation_mode", "")),
                    )

                    if transition_active_for_log and output_interlaced_for_log:
                        phase = phase_for_log
                        if isinstance(phase, dict):
                            roi0 = phase.get("roi0")
                            roi1 = phase.get("roi1")
                            if isinstance(roi0, (list, tuple)) and len(roi0) >= 4 and isinstance(roi1, (list, tuple)) and len(roi1) >= 4:
                                progress = float(transition_state_for_log.get("frame_progress", 0.0))
                                total = max(1, int(transition_state_for_log.get("total_frames", 1)))
                                signature = (
                                    f"{progress:.3f}|{int(roi0[0])},{int(roi0[1])},{int(roi0[2])},{int(roi0[3])}|"
                                    f"{int(roi1[0])},{int(roi1[1])},{int(roi1[2])},{int(roi1[3])}|"
                                    f"{float(phase.get('field0_x', 0.0)):.4f},{float(phase.get('field0_y', 0.0)):.4f}|"
                                    f"{float(phase.get('field1_x', 0.0)):.4f},{float(phase.get('field1_y', 0.0)):.4f}"
                                )
                                if signature != self._last_interlaced_phase_log_signature:
                                    LOGGER.info(
                                        (
                                            "ROI_FIELD_PHASE | progress=%.3f/%d | "
                                            "field0_roi=(%d,%d,%d,%d) | field1_roi=(%d,%d,%d,%d) | "
                                            "field0_shift=(%.4f,%.4f) | field1_shift=(%.4f,%.4f) | phase2=%.2f"
                                        ),
                                        progress,
                                        total,
                                        int(roi0[0]),
                                        int(roi0[1]),
                                        int(roi0[2]),
                                        int(roi0[3]),
                                        int(roi1[0]),
                                        int(roi1[1]),
                                        int(roi1[2]),
                                        int(roi1[3]),
                                        float(phase.get("field0_x", 0.0)),
                                        float(phase.get("field0_y", 0.0)),
                                        float(phase.get("field1_x", 0.0)),
                                        float(phase.get("field1_y", 0.0)),
                                        controller_field2_phase_for_log,
                                    )
                                    self._last_interlaced_phase_log_signature = signature
                    elif self._last_interlaced_phase_log_signature:
                        self._last_interlaced_phase_log_signature = ""

                    self._roi_diag_canvas_events = 0
                    self._roi_diag_controller_send_attempts = 0
                    self._roi_diag_controller_send_success = 0
                    self._roi_diag_controller_send_drops = 0
                    self._roi_diag_controller_send_ms_sum = 0.0
                    self._roi_diag_controller_send_ms_max = 0.0

                    self._refresh_ai_sr_runtime_panel()
                    self._refresh_rtx_vsr_runtime_panel()
                    self._apply_performance_guard(fps)
                return

            t0 = time.perf_counter()
            input_frame = self._next_input_frame()
            self._perf_add("acquire", (time.perf_counter() - t0) * 1000.0)
            if input_frame is None:
                return

            t0 = time.perf_counter()
            output_frame = self._controller.process_frame(input_frame)
            self._perf_add("process", (time.perf_counter() - t0) * 1000.0)

            if self._source_mode == "Blackmagic DeckLink" and self._output_session is not None:
                t0 = time.perf_counter()
                write_frame_to_output(self._output_session, output_frame)
                self._perf_add("output", (time.perf_counter() - t0) * 1000.0)

            input_preview_size = self._preview_target_for_view("input")
            if input_preview_size is not None:
                t0 = time.perf_counter()
                input_image, input_backing = uyvy_to_qimage(
                    input_frame,
                    preview_max_w=input_preview_size[0],
                    preview_max_h=input_preview_size[1],
                    color_space=getattr(self._controller, "color_space", "rec709"),
                    color_range=getattr(self._controller, "color_range", "limited"),
                )
                self._input_canvas.set_image(input_image, input_backing)
                self._perf_add("convert_in", (time.perf_counter() - t0) * 1000.0)

            output_preview_size = self._preview_target_for_view("output")
            if output_preview_size is not None:
                t0 = time.perf_counter()
                output_image, output_backing = uyvy_to_qimage(
                    output_frame,
                    preview_max_w=output_preview_size[0],
                    preview_max_h=output_preview_size[1],
                    color_space=getattr(self._controller, "color_space", "rec709"),
                    color_range=getattr(self._controller, "color_range", "limited"),
                )
                self._output_canvas.set_image(output_image, output_backing)
                self._perf_add("convert_out", (time.perf_counter() - t0) * 1000.0)

            self._perf_add("tick", (time.perf_counter() - tick_start) * 1000.0)
            self._frame_count += 1

            now = time.perf_counter()
            dt = now - self._last_stat_time
            if dt >= 1.0:
                fps = self._frame_count / dt
                perf = self._perf_snapshot_and_reset()
                self._frame_count = 0
                self._last_stat_time = now
                mode_text = "Auto" if self._controller.basic_scaling_auto_mode else "Manual"
                flavor_text = SR_FLAVOR_NAME_TO_LABEL.get(self._controller.basic_scaling_method, self._controller.basic_scaling_method)
                ai_sr_state = "off"
                ai_sr_detail = ""
                if getattr(self._controller, "ai_sr_enabled", False):
                    ai_sr_state = "active" if getattr(self._controller, "ai_sr_active", False) else "requested"
                    ai_sr_info = getattr(self._controller, "ai_sr_info", None)
                    ai_sr_error = getattr(self._controller, "ai_sr_error", None)
                    if ai_sr_info and ai_sr_state == "active":
                        provider = ai_sr_info.get("provider", "unknown")
                        strict_text = " strict" if bool(ai_sr_info.get("strict_mode", False)) else " async"
                        ai_sr_detail = f" ({provider},{strict_text})"
                    elif ai_sr_error and ai_sr_state != "active":
                        ai_sr_detail = f" ({ai_sr_error})"
                    elif getattr(self._controller, "ai_sr_last_warning", None):
                        ai_sr_detail = f" ({self._controller.ai_sr_last_warning})"
                rtx_vsr_state = "off"
                rtx_vsr_detail = ""
                if getattr(self._controller, "rtx_vsr_enabled", False):
                    rtx_vsr_state = "active" if getattr(self._controller, "rtx_vsr_active", False) else "requested"
                    rtx_vsr_info = getattr(self._controller, "rtx_vsr_info", None)
                    rtx_vsr_error = getattr(self._controller, "rtx_vsr_error", None)
                    if rtx_vsr_info and rtx_vsr_state == "active":
                        quality = rtx_vsr_info.get("quality", getattr(self._controller, "rtx_vsr_quality", "high"))
                        rtx_vsr_detail = f" ({quality})"
                    elif rtx_vsr_error and rtx_vsr_state != "active":
                        rtx_vsr_detail = f" ({rtx_vsr_error})"
                if getattr(self._controller, "ai_sr_enabled", False):
                    basic_status_text = "Basic scaling=auto-disabled (AI SR ONNX)"
                else:
                    basic_status_text = (
                        f"Basic scaling mode={mode_text}"
                        f" | Basic scaling method={flavor_text}"
                        f" | effective scaling={self._controller.effective_scale()}"
                    )
                self._update_status(
                    f"Running | FPS={fps:.1f} | {basic_status_text} | AI SR={ai_sr_state}{ai_sr_detail} | RTX VSR={rtx_vsr_state}{rtx_vsr_detail}"
                )
                LOGGER.info(
                    (
                        "PERF | fps=%.1f | acquire=%.2f/%.2fms | process=%.2f/%.2fms | "
                        "output=%.2f/%.2fms | conv_in=%.2f/%.2fms | conv_out=%.2f/%.2fms | tick=%.2f/%.2fms"
                    ),
                    fps,
                    perf["acquire"][0],
                    perf["acquire"][1],
                    perf["process"][0],
                    perf["process"][1],
                    perf["output"][0],
                    perf["output"][1],
                    perf["convert_in"][0],
                    perf["convert_in"][1],
                    perf["convert_out"][0],
                    perf["convert_out"][1],
                    perf["tick"][0],
                    perf["tick"][1],
                )
                if self._source_mode == "Blackmagic DeckLink":
                    self.decklink_status_label.setText("DeckLink streaming")

                self._refresh_ai_sr_runtime_panel()
                self._refresh_rtx_vsr_runtime_panel()
                self._apply_performance_guard(fps)
        except Exception as exc:
            if self._is_closing:
                return
            self._timer.stop()
            self._update_status(f"Runtime error: {exc}")

    def closeEvent(self, event) -> None:
        self._is_closing = True
        self._save_settings()
        self._controller_roi_target = None
        self._controller_roi_interp_timer.stop()
        self._timer.stop()
        self._controller.close()
        self._stop_decklink_sessions()
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_initial_main_splitter_layout()
        self._apply_initial_viewer_layout()
        self._fit_viewers_to_video_aspect()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_viewers_to_video_aspect()

    def _apply_initial_viewer_layout(self) -> None:
        if self._splitter_initialized:
            return
        if not self.isVisible():
            return

        total_h = self._display_splitter.size().height()
        if total_h <= 2:
            return

        half = max(1, total_h // 2)
        self._display_splitter.setSizes([half, max(1, total_h - half)])
        self._splitter_initialized = True
        self._fit_viewers_to_video_aspect()

    def _apply_initial_main_splitter_layout(self) -> None:
        if self._main_splitter_initialized:
            return
        if not self.isVisible():
            return

        total_w = self._main_splitter.size().width()
        if total_w <= 2:
            return

        half = max(1, total_w // 2)
        self._main_splitter.setSizes([max(1, total_w - half), half])
        self._main_splitter_initialized = True

    def _fit_viewers_to_video_aspect(self) -> None:
        self._fit_canvas_in_panel(
            panel=self._input_panel,
            header_widget=self._input_header,
            canvas=self._input_canvas,
            footer_widget=self._input_fullscreen_keyframe_toolbar,
            side_widget=self._input_fullscreen_keyframe_side_panel,
        )
        self._fit_canvas_in_panel(
            panel=self._output_panel,
            header_widget=self._output_header,
            canvas=self._output_canvas,
            footer_widget=self._output_fullscreen_keyframe_toolbar,
            side_widget=self._output_fullscreen_keyframe_side_panel,
        )

    def _fit_canvas_in_panel(
        self,
        panel: QWidget,
        header_widget: QWidget,
        canvas: QWidget,
        footer_widget: QWidget | None = None,
        side_widget: QWidget | None = None,
    ) -> None:
        if not panel.isVisible() or panel.width() <= 0 or panel.height() <= 0:
            return

        layout = panel.layout()
        if layout is None:
            return

        margins = layout.contentsMargins()
        spacing = max(0, layout.spacing())
        used_h = 0
        if header_widget.isVisible():
            used_h += header_widget.sizeHint().height() + spacing
        if footer_widget is not None and footer_widget.isVisible():
            used_h += footer_widget.sizeHint().height() + spacing

        avail_w = panel.width() - margins.left() - margins.right()
        avail_h = panel.height() - margins.top() - margins.bottom() - used_h
        if side_widget is not None and side_widget.isVisible():
            avail_w -= side_widget.sizeHint().width() + spacing

        # Keep both stacked viewers within the window height budget.
        if self._fullscreen_view_name is None:
            window_half_h = max(1, int(self.height() / 2) - 10)
            avail_h = min(avail_h, window_half_h)

        if avail_w <= 10 or avail_h <= 10:
            return

        target_w = avail_w
        target_h = int(round(target_w * 9.0 / 16.0))
        if target_h > avail_h:
            target_h = avail_h
            target_w = int(round(target_h * 16.0 / 9.0))

        target_w = max(1, min(target_w, avail_w))
        target_h = max(1, min(target_h, avail_h))

        if canvas.width() == target_w and canvas.height() == target_h:
            return

        canvas.setFixedSize(target_w, target_h)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_F11:
            self._toggle_fullscreen_view()
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self._fullscreen_view_name is not None:
            self._set_fullscreen_view(None)
            event.accept()
            return

        key_slot = {
            Qt.Key_1: 1,
            Qt.Key_2: 2,
            Qt.Key_3: 3,
            Qt.Key_4: 4,
        }.get(event.key())
        if key_slot is not None:
            disallowed_mods = Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier | Qt.ShiftModifier
            focused = QApplication.focusWidget()
            is_text_entry = isinstance(focused, (QAbstractSpinBox, QLineEdit, QComboBox))
            if (not event.isAutoRepeat()) and (not (event.modifiers() & disallowed_mods)) and (not is_text_entry):
                self._recall_roi_key_slot(key_slot)
                event.accept()
                return

        super().keyPressEvent(event)

    def _on_canvas_fullscreen_requested(self, view_name: str) -> None:
        if self._fullscreen_view_name == view_name:
            self._set_fullscreen_view(None)
            return
        self._set_fullscreen_view(view_name)

    def _set_fullscreen_view(self, view_name: str | None) -> None:
        previous_view = self._fullscreen_view_name
        if view_name is not None and previous_view is None:
            self._capture_windowed_geometry_before_fullscreen()

        self._fullscreen_view_name = view_name
        self._sync_fullscreen_button_states()
        if view_name is None:
            self._controls_scroll.setVisible(True)
            self._input_panel.setVisible(True)
            self._output_panel.setVisible(True)
            for toolbar in self._fullscreen_keyframe_toolbars.values():
                toolbar.setVisible(False)
            for side_panel in self._fullscreen_keyframe_side_panels.values():
                side_panel.setVisible(False)
            self._input_canvas.setEnabled(True)
            self._output_canvas.setEnabled(True)
            if previous_view is not None:
                self._restore_windowed_geometry_after_fullscreen()
            else:
                self.showNormal()
            self._splitter_initialized = False
            QTimer.singleShot(0, self._apply_initial_viewer_layout)
            return

        self._controls_scroll.setVisible(False)
        self._input_panel.setVisible(view_name == "input")
        self._output_panel.setVisible(view_name == "output")
        for toolbar_view, toolbar in self._fullscreen_keyframe_toolbars.items():
            toolbar.setVisible(toolbar_view == view_name)
        for panel_view, side_panel in self._fullscreen_keyframe_side_panels.items():
            side_panel.setVisible(panel_view == view_name)
        self._input_canvas.setEnabled(view_name == "input")
        self._output_canvas.setEnabled(view_name == "output")
        self.showFullScreen()
        QTimer.singleShot(0, self._fit_viewers_to_video_aspect)

    def _preview_target_for_view(self, view_name: str) -> tuple[int, int] | None:
        if self._fullscreen_view_name is not None and self._fullscreen_view_name != view_name:
            return None

        canvas = self._input_canvas if view_name == "input" else self._output_canvas
        if not canvas.isVisible():
            return None

        canvas_w = max(1, canvas.width())
        canvas_h = max(1, canvas.height())
        if self._fullscreen_view_name is None:
            cap_w = WINDOWED_PREVIEW_MAX_W
            cap_h = WINDOWED_PREVIEW_MAX_H
        else:
            cap_w = FULLSCREEN_PREVIEW_MAX_W
            cap_h = FULLSCREEN_PREVIEW_MAX_H

        base_w = min(canvas_w, cap_w)
        base_h = min(canvas_h, cap_h)
        ds = self._preview_downsample_factor
        preview_w = max(1, int(round(base_w * ds)))
        preview_h = max(1, int(round(base_h * ds)))
        return (preview_w, preview_h)

    def _update_timer_interval(self) -> None:
        fps = max(1, self.fps_spin.value())
        poll_fps = float(fps)
        if self._source_mode == "Blackmagic DeckLink" and self._controller_backend == "worker-process":
            # In worker DeckLink mode, preview cadence should follow the dedicated
            # GUI poll setting instead of camera/output FPS controls.
            poll_fps = float(max(1.0, self._decklink_tick_poll_fps))
        self._timer.setInterval(max(1, int(round(1000.0 / max(1.0, poll_fps)))))

    def _normalize_preview_downsample_factor(self, value: float) -> float:
        candidates = sorted(PREVIEW_DOWNSAMPLE_LABEL_TO_FACTOR.values())
        nearest = min(candidates, key=lambda f: abs(f - float(value)))
        return float(nearest)

    def _preview_downsample_label_for_factor(self, factor: float) -> str:
        normalized = self._normalize_preview_downsample_factor(factor)
        for label, value in PREVIEW_DOWNSAMPLE_LABEL_TO_FACTOR.items():
            if abs(value - normalized) < 1e-6:
                return label
        return "Quarter (1/4)"

    def _on_preview_downsample_changed(self) -> None:
        label = self.preview_downsample_combo.currentText()
        factor = PREVIEW_DOWNSAMPLE_LABEL_TO_FACTOR.get(label, 0.25)
        self._preview_downsample_factor = self._normalize_preview_downsample_factor(factor)
        self._update_status(
            f"Preview downsample set to {label} ({int(round(self._preview_downsample_factor * 100.0))}% linear size)"
        )

    def _on_color_space_changed(self) -> None:
        if self._updating_controls:
            return
        selected_label = self.color_space_combo.currentText()
        selected_name = COLOR_SPACE_LABEL_TO_NAME.get(selected_label, "rec709")
        try:
            self._controller.set_color_space(selected_name)
            applied_name = _normalize_color_space_name(getattr(self._controller, "color_space", selected_name))
            applied_label = COLOR_SPACE_NAME_TO_LABEL.get(applied_name, applied_name)
            self._restart_blackmagic_sessions_for_color_update(f"Color space applied: {applied_label}")
        except Exception as exc:
            self._update_status(f"Color space change failed: {exc}")

    def _on_color_range_changed(self) -> None:
        if self._updating_controls:
            return
        selected_label = self.color_range_combo.currentText()
        selected_name = COLOR_RANGE_LABEL_TO_NAME.get(selected_label, "limited")
        try:
            self._controller.set_color_range(selected_name)
            applied_name = _normalize_color_range_name(getattr(self._controller, "color_range", selected_name))
            applied_label = COLOR_RANGE_NAME_TO_LABEL.get(applied_name, applied_name)
            self._restart_blackmagic_sessions_for_color_update(f"Color range applied: {applied_label}")
        except Exception as exc:
            self._update_status(f"Color range change failed: {exc}")

    def _restart_blackmagic_sessions_for_color_update(self, success_message: str) -> None:
        if self._source_mode != "Blackmagic DeckLink":
            self._update_status(success_message)
            return

        if not self._decklink_sessions_running:
            self._update_status(f"{success_message} | DeckLink restart will apply on next session start")
            return

        self._decklink_color_reapply_timer.start()
        self._update_status(f"{success_message} | DeckLink restart queued...")

    def _reapply_decklink_after_color_change(self) -> None:
        if self._source_mode != "Blackmagic DeckLink":
            return
        if not self._decklink_sessions_running:
            return
        self._update_status("Applying color change: restarting DeckLink I/O...")
        self._on_apply_decklink_settings()

    def _on_preview_request_fps_changed(self) -> None:
        preview_fps = int(self.preview_request_fps_spin.value())
        if hasattr(self._controller, "set_preview_fps"):
            self._controller.set_preview_fps(float(preview_fps))
        self._update_status(f"Preview request FPS set to {preview_fps}")

    def _on_preview_poll_fps_changed(self) -> None:
        self._decklink_tick_poll_fps = float(max(1, self.preview_poll_fps_spin.value()))
        self._update_timer_interval()
        self._update_status(f"Preview poll FPS cap set to {int(self._decklink_tick_poll_fps)}")

    def _on_decklink_output_buffer_changed(self) -> None:
        buffer_frames = max(0, min(10, int(self.decklink_output_buffer_spin.value())))
        self._decklink_output_buffer_user_target_frames = int(buffer_frames)
        if buffer_frames >= self._decklink_buffer_guard_floor_frames:
            self._decklink_buffer_guard_active = False
            self._decklink_buffer_guard_stable_windows = 0
        self._decklink_output_buffer_frames = buffer_frames
        if hasattr(self._controller, "decklink_output_buffer_frames"):
            self._controller.decklink_output_buffer_frames = buffer_frames
        if self._updating_controls:
            return
        if self._source_mode == "Blackmagic DeckLink":
            if self._decklink_buffer_reapply_timer.isActive():
                self._decklink_buffer_reapply_timer.stop()
            self._reapply_decklink_after_buffer_change()
        elif hasattr(self._controller, "set_decklink_output_buffer_frames"):
            self._controller.set_decklink_output_buffer_frames(buffer_frames)
        self._update_status(f"DeckLink output buffer set to {buffer_frames} frame(s); applying")

    def _on_worker_priority_changed(self) -> None:
        if self._updating_controls:
            return
        self._worker_process_priority = _normalize_worker_priority_name(
            WORKER_PRIORITY_LABEL_TO_NAME.get(self.worker_priority_combo.currentText(), "above_normal")
        )
        self._apply_worker_process_priority_to_controller(notify=True)

    def _on_decklink_buffer_guard_toggled(self, checked: bool) -> None:
        self._decklink_buffer_guard_enabled = bool(checked)
        if not self._decklink_buffer_guard_enabled:
            self._decklink_buffer_guard_active = False
            self._decklink_buffer_guard_stable_windows = 0
            requested_frames = int(self._decklink_output_buffer_user_target_frames)
            if self._decklink_output_buffer_frames != requested_frames:
                self._decklink_output_buffer_frames = requested_frames
                try:
                    if hasattr(self._controller, "set_decklink_output_buffer_frames"):
                        self._controller.set_decklink_output_buffer_frames(requested_frames)
                    self._update_status(
                        f"FPS-priority guard disabled; restoring requested DeckLink buffer={requested_frames}"
                    )
                except Exception:
                    LOGGER.exception("Failed to restore requested DeckLink buffer after disabling FPS-priority guard")
            return

        self._update_status(
            "FPS-priority guard enabled; buffer remains user-selected and only temporarily increases during deadline stress"
        )

    def _apply_worker_process_priority_to_controller(self, notify: bool) -> None:
        if not hasattr(self._controller, "set_worker_process_priority"):
            return

        priority_name = _normalize_worker_priority_name(self._worker_process_priority)
        try:
            self._controller.set_worker_process_priority(priority_name)
            warning_text = str(getattr(self._controller, "worker_process_priority_error", "") or "")
            if notify:
                if warning_text:
                    self._update_status(
                        f"Worker process priority requested={priority_name}; warning={warning_text}"
                    )
                else:
                    self._update_status(f"Worker process priority set to {priority_name}")
        except Exception as exc:
            LOGGER.exception("Failed to apply worker process priority")
            if notify:
                self._update_status(f"Worker process priority apply failed: {exc}")

    def _reapply_decklink_after_buffer_change(self) -> None:
        if self._source_mode != "Blackmagic DeckLink":
            return
        try:
            if hasattr(self._controller, "set_decklink_output_buffer_frames"):
                self._controller.set_decklink_output_buffer_frames(int(self._decklink_output_buffer_frames))
        except Exception as exc:
            LOGGER.exception("Failed to apply DeckLink output buffer change")
            self._update_status(f"DeckLink buffer apply failed: {exc}")
            return
        self._update_status(
            f"DeckLink output buffer applied: {int(self._decklink_output_buffer_frames)} frame(s)"
        )

    def _maybe_auto_stabilize_decklink_buffer(
        self,
        *,
        deadline_miss_ratio: float,
        deadline_miss_streak: int,
        starvation_delta: int,
        buffered_count: int,
        interaction_active: bool,
    ) -> None:
        if self._source_mode != "Blackmagic DeckLink":
            return
        if not self._decklink_sessions_running:
            return
        if not self._decklink_buffer_guard_enabled:
            return
        requested_frames = int(self._decklink_output_buffer_user_target_frames)
        guard_floor = int(self._decklink_buffer_guard_floor_frames)
        if interaction_active:
            guard_floor = max(
                guard_floor,
                int(self._decklink_buffer_guard_transition_floor_frames),
            )
        if requested_frames >= guard_floor:
            self._decklink_buffer_guard_active = False
            self._decklink_buffer_guard_stable_windows = 0
            return

        engage = (
            float(deadline_miss_ratio) >= float(self._decklink_buffer_guard_engage_miss_ratio)
            or int(deadline_miss_streak) >= 2
            or int(starvation_delta) > 0
            or int(buffered_count) == 0
        )

        if engage:
            self._decklink_buffer_guard_stable_windows = 0
            if self._decklink_output_buffer_frames < guard_floor:
                self._decklink_buffer_guard_active = True
                self._decklink_output_buffer_frames = guard_floor
                try:
                    if hasattr(self._controller, "set_decklink_output_buffer_frames"):
                        self._controller.set_decklink_output_buffer_frames(int(guard_floor))
                    self._update_status(
                        (
                            "DeckLink buffer guard engaged: requested "
                            f"{requested_frames}, temporarily applying {guard_floor} "
                            f"(dl_miss={deadline_miss_ratio * 100.0:.1f}%, streak={deadline_miss_streak}, "
                            f"interaction={'yes' if interaction_active else 'no'})"
                        )
                    )
                except Exception:
                    LOGGER.exception("DeckLink buffer guard failed to apply raised buffer")
            return

        if not self._decklink_buffer_guard_active:
            return

        release_ready = (
            float(deadline_miss_ratio) <= float(self._decklink_buffer_guard_release_miss_ratio)
            and int(deadline_miss_streak) == 0
            and int(starvation_delta) == 0
        )
        if not release_ready:
            self._decklink_buffer_guard_stable_windows = 0
            return

        self._decklink_buffer_guard_stable_windows += 1
        if self._decklink_buffer_guard_stable_windows < int(self._decklink_buffer_guard_release_windows_needed):
            return

        self._decklink_buffer_guard_active = False
        self._decklink_buffer_guard_stable_windows = 0
        self._decklink_output_buffer_frames = requested_frames
        try:
            if hasattr(self._controller, "set_decklink_output_buffer_frames"):
                self._controller.set_decklink_output_buffer_frames(int(requested_frames))
            self._update_status(
                (
                    "DeckLink buffer guard released: restoring requested buffer "
                    f"{requested_frames} frame(s) after stable timing"
                )
            )
        except Exception:
            LOGGER.exception("DeckLink buffer guard failed to restore requested buffer")

    def _on_roi_from_canvas(self, x: int, y: int, w: int, h: int) -> None:
        self._last_manual_roi_update_ts = time.perf_counter()
        self._roi_diag_canvas_events += 1
        had_active_keyframe_transition = self._roi_keyframe_transition is not None
        if had_active_keyframe_transition:
            transition_state = self._roi_keyframe_transition
            if isinstance(transition_state, dict):
                current_estimate = transition_state.get("current_roi_estimate")
                if isinstance(current_estimate, Roi):
                    self._controller_roi_applied = clamp_roi(current_estimate)
            self._cancel_roi_keyframe_transition()
        self._roi = clamp_roi(Roi(x, y, w, h))

        # Canvas interaction already applies smoothing/throttling; avoid a second
        # controller-side interpolation loop that can keep ROI commands churning
        # after manual motion and starve preview ticks.
        self._controller_roi_target = None
        self._controller_filtered_target_roi = None
        self._controller_roi_interp_timer.stop()

        drag_overlay = self._input_canvas.drag_visual_roi_overlay()

        # Treat manual updates as a continuous live keyframe stream: keep only
        # the latest target and interpolate from applied ROI on each send tick.
        if self._manual_live_target_roi is None:
            self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
            self._manual_roi_target_history = []
        raw_curve_target: Roi | tuple[float, float, float, float]
        if drag_overlay is not None:
            raw_curve_target = tuple(float(v) for v in drag_overlay)
        else:
            raw_curve_target = self._roi
        smoothed_target = self._smooth_manual_roi_target(raw_curve_target)
        self._manual_live_target_roi = smoothed_target

        if drag_overlay is not None:
            now = time.perf_counter()
            overlay = tuple(float(v) for v in drag_overlay)
            prev_end = self._manual_drag_interp_end_overlay
            if prev_end is None:
                self._manual_drag_interp_start_overlay = overlay
                self._manual_drag_interp_end_overlay = overlay
                self._manual_drag_interp_started_ts = now
                self._manual_drag_interp_duration_s = 1.0 / 90.0
            else:
                dt = now - self._manual_drag_last_event_ts if self._manual_drag_last_event_ts > 0.0 else (1.0 / 90.0)
                self._manual_drag_interp_start_overlay = tuple(float(v) for v in prev_end)
                self._manual_drag_interp_end_overlay = overlay
                self._manual_drag_interp_started_ts = now
                self._manual_drag_interp_duration_s = max(1.0 / 180.0, min(1.0 / 45.0, dt * 1.10))
            self._manual_drag_last_event_ts = now
        else:
            self._manual_drag_interp_start_overlay = None
            self._manual_drag_interp_end_overlay = None
            self._manual_drag_last_event_ts = 0.0

        self._pending_manual_controller_roi = smoothed_target
        if not self._manual_roi_send_timer.isActive():
            self._manual_roi_send_timer.start()

        self._schedule_roi_controls_sync(self._roi)

    def _flush_pending_manual_controller_roi(self) -> None:
        pending = self._pending_manual_controller_roi
        self._pending_manual_controller_roi = None
        if pending is not None:
            self._manual_live_target_roi = pending

        target = self._manual_live_target_roi
        if target is None:
            return

        current = self._controller_roi_applied
        use_subpixel_microstep = bool(
            hasattr(self._controller, "set_roi_with_subpixel")
        )
        target_scale = roi_scale_from_roi(target)
        if use_subpixel_microstep:
            step_roi, step_shift_x, step_shift_y = self._manual_roi_step_with_subpixel(current, target)
        else:
            step_roi = self._interpolate_controller_roi_step(current, target)
            step_shift_x = 0.0
            step_shift_y = 0.0

        moving_only = (
            step_roi.w == self._controller_roi_applied.w
            and step_roi.h == self._controller_roi_applied.h
        )

        drag_overlay = self._sample_manual_drag_overlay_target(self._input_canvas.drag_visual_roi_overlay())
        if use_subpixel_microstep and moving_only and drag_overlay is not None:
            step_roi, step_shift_x, step_shift_y = self._manual_roi_step_with_subpixel_float_target(
                current,
                drag_overlay,
            )

        should_close_snap = self._is_controller_roi_close(step_roi, target) and not (
            use_subpixel_microstep and moving_only and drag_overlay is not None
        )
        if should_close_snap:
            step_roi = target
            step_shift_x = 0.0
            step_shift_y = 0.0

        sent = False
        scale_intent_active = (target.w != current.w) or (target.h != current.h)
        frame_gate_interval_ms = self._manual_roi_render_gate_interval_ms()
        if frame_gate_interval_ms is not None:
            elapsed_ms = (time.perf_counter() - float(self._manual_roi_last_send_ts)) * 1000.0
            if elapsed_ms < float(frame_gate_interval_ms) and not should_close_snap and not scale_intent_active:
                wait_ms = max(1, int(round(float(frame_gate_interval_ms) - elapsed_ms)))
                self._manual_roi_send_timer.setInterval(wait_ms)
                self._manual_roi_send_timer.start()
                return

        try:
            started = time.perf_counter()
            self._roi_diag_controller_send_attempts += 1
            if use_subpixel_microstep:
                output_is_interlaced = False
                output_is_interlaced_fn = getattr(self._controller, "decklink_output_is_interlaced", None)
                if callable(output_is_interlaced_fn):
                    try:
                        output_is_interlaced = bool(output_is_interlaced_fn())
                    except Exception:
                        output_is_interlaced = False
                manual_interaction = bool(drag_overlay is not None) or (
                    output_is_interlaced
                    and ((step_roi.w != current.w) or (step_roi.h != current.h))
                )
                sent = bool(
                    self._controller.set_roi_with_subpixel(
                        step_roi,
                        step_shift_x,
                        step_shift_y,
                        manual_drag=manual_interaction,
                    )
                )
            else:
                if moving_only and hasattr(self._controller, "set_roi_position"):
                    sent = bool(self._controller.set_roi_position(step_roi.x, step_roi.y))
                else:
                    sent = bool(self._controller.set_roi(step_roi))

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._roi_diag_controller_send_ms_sum += elapsed_ms
            if elapsed_ms > self._roi_diag_controller_send_ms_max:
                self._roi_diag_controller_send_ms_max = elapsed_ms

            if sent:
                self._roi_diag_controller_send_success += 1
                self._manual_roi_last_send_ts = time.perf_counter()
            else:
                self._roi_diag_controller_send_drops += 1
            self._controller_roi_applied = step_roi
        except Exception as exc:
            self._roi_diag_controller_send_drops += 1
            self._update_status(f"ROI update failed: {exc}")

        # Continue stepping while target is not reached, or while new user
        # updates keep arriving.
        manual_drag_active = self._input_canvas.drag_visual_roi_overlay() is not None
        if self._is_controller_roi_close(self._controller_roi_applied, target) and not manual_drag_active:
            if use_subpixel_microstep:
                try:
                    self._controller.set_roi_with_subpixel(target, 0.0, 0.0, manual_drag=False)
                except Exception:
                    pass
            self._manual_live_target_roi = None
            self._manual_roi_target_history = []
            self._manual_drag_interp_start_overlay = None
            self._manual_drag_interp_end_overlay = None
            self._manual_drag_last_event_ts = 0.0
            self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        if self._pending_manual_controller_roi is not None or self._manual_live_target_roi is not None:
            interval_ms = self._manual_roi_send_interval_ms(target_scale, moving_only=moving_only)
            if not sent and self._controller_backend == "worker-process":
                # Back off on queue pressure; latest-wins ROI compaction keeps motion responsive.
                interval_ms = min(24, interval_ms + 4)
            self._manual_roi_send_timer.setInterval(interval_ms)
            self._manual_roi_send_timer.start()

    def _manual_roi_send_interval_ms(self, zoom_scale: float, moving_only: bool = False) -> int:
        z = max(1.0, float(zoom_scale))
        if z >= 6.0:
            base = 8
        elif z >= 4.0:
            base = 10
        else:
            base = 16

        if moving_only and self._controller_backend == "worker-process":
            base = max(base, int(self._manual_drag_worker_send_interval_ms))

        field_interval_ms = self._decklink_output_field_interval_ms()
        if field_interval_ms is not None:
            # Keep manual control updates at least as responsive as field cadence.
            # Slower-than-field pacing makes drag appear steppy at 1080i rates.
            return min(base, int(field_interval_ms))
        return base

    def _smooth_manual_roi_target(self, raw_target: Roi | tuple[float, float, float, float]) -> Roi:
        sample_count = max(0, min(10, int(self._roi_smoothing_percent)))

        if isinstance(raw_target, tuple):
            raw_x, raw_y, raw_w, _raw_h = [float(v) for v in raw_target]
            clamped_w_f = max(2.0, min(float(FRAME_W), raw_w))
            clamped_h_f = max(2.0, min(float(FRAME_H), clamped_w_f * 9.0 / 16.0))
            if clamped_h_f >= float(FRAME_H):
                clamped_h_f = float(FRAME_H)
                clamped_w_f = max(2.0, min(float(FRAME_W), clamped_h_f * 16.0 / 9.0))
            clamped_x_f = max(0.0, min(float(FRAME_W) - clamped_w_f, raw_x))
            clamped_y_f = max(0.0, min(float(FRAME_H) - clamped_h_f, raw_y))
            sample_cx = clamped_x_f + (clamped_w_f * 0.5)
            sample_cy = clamped_y_f + (clamped_h_f * 0.5)
            clamped_target = clamp_roi(
                Roi(
                    int(round(clamped_x_f)),
                    int(round(clamped_y_f)),
                    max(2, int(round(clamped_w_f)) & ~1),
                    max(2, int(round((max(2, int(round(clamped_w_f)) & ~1) * 9.0) / 16.0))),
                )
            )
            sample_w = clamped_w_f
        else:
            clamped_target = clamp_roi(raw_target)
            sample_cx = float(clamped_target.x) + (float(clamped_target.w) * 0.5)
            sample_cy = float(clamped_target.y) + (float(clamped_target.h) * 0.5)
            sample_w = float(clamped_target.w)

        if sample_count <= 0:
            self._manual_roi_target_history = []
            return clamped_target

        self._manual_roi_target_history.append((sample_cx, sample_cy, sample_w))
        if len(self._manual_roi_target_history) > sample_count:
            self._manual_roi_target_history = self._manual_roi_target_history[-sample_count:]

        history = self._manual_roi_target_history
        if len(history) == 1:
            h_cx, h_cy, h_w = history[0]
            h_w_i = max(2, int(round(h_w)) & ~1)
            h_h_i = max(2, int(round(h_w_i * 9.0 / 16.0)))
            h_x_i = int(round(h_cx - (h_w_i * 0.5)))
            h_y_i = int(round(h_cy - (h_h_i * 0.5)))
            return clamp_roi(Roi(h_x_i, h_y_i, h_w_i, h_h_i))

        # Baseline latest-weighted smoothing (stable when sample_count is small).
        weights = np.arange(1.0, float(len(history)) + 1.0, dtype=np.float64)
        total_weight = float(np.sum(weights))
        if total_weight <= 1e-6:
            return clamped_target
        cxs = np.array([float(sample[0]) for sample in history], dtype=np.float64)
        cys = np.array([float(sample[1]) for sample in history], dtype=np.float64)
        ws = np.array([float(sample[2]) for sample in history], dtype=np.float64)
        base_cx = float(np.dot(cxs, weights) / total_weight)
        base_cy = float(np.dot(cys, weights) / total_weight)
        base_w = float(np.dot(ws, weights) / total_weight)

        # Curve-fit smoothing: fit a quadratic over recent polls and evaluate a
        # short fractional lag behind "now". This yields a smoother trajectory
        # than simple averaging while keeping drag response predictable.
        t = np.arange(float(len(history)), dtype=np.float64)
        degree = 2 if len(history) >= 3 else 1
        lag = min(0.95, 0.09 * float(sample_count))
        t_eval = max(0.0, float(len(history) - 1) - lag)

        def _curve_eval(values: np.ndarray, base_value: float) -> float:
            try:
                coeff = np.polyfit(t, values, deg=degree)
                predicted = float(np.polyval(coeff, t_eval))
            except Exception:
                predicted = base_value

            lo = float(np.min(values))
            hi = float(np.max(values))
            span = max(1.0, hi - lo)
            bounded = max(lo - (0.20 * span), min(hi + (0.20 * span), predicted))
            curve_weight = min(0.92, 0.39 + (0.056 * float(sample_count)))
            if len(history) < 4:
                curve_weight *= 0.80
            return (base_value * (1.0 - curve_weight)) + (bounded * curve_weight)

        smoothed_cx_f = _curve_eval(cxs, base_cx)
        smoothed_cy_f = _curve_eval(cys, base_cy)
        smoothed_w_f = _curve_eval(ws, base_w)

        smoothed_w = max(2, int(round(smoothed_w_f)) & ~1)
        smoothed_h = max(2, int(round(smoothed_w * 9.0 / 16.0)))
        smoothed_x = int(round(smoothed_cx_f - (smoothed_w * 0.5)))
        smoothed_y = int(round(smoothed_cy_f - (smoothed_h * 0.5)))
        return clamp_roi(Roi(smoothed_x, smoothed_y, smoothed_w, smoothed_h))

    def _manual_roi_render_gate_interval_ms(self) -> int | None:
        if not bool(self._manual_roi_frame_lock_to_output):
            return None
        if self._source_mode != "Blackmagic DeckLink":
            return None
        if self._controller_backend != "worker-process":
            return None
        return self._decklink_output_field_interval_ms()

    def _sample_manual_drag_overlay_target(
        self,
        live_overlay: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float, float] | None:
        if live_overlay is None:
            self._manual_drag_interp_start_overlay = None
            self._manual_drag_interp_end_overlay = None
            self._manual_drag_last_event_ts = 0.0
            return None

        start = self._manual_drag_interp_start_overlay
        end = self._manual_drag_interp_end_overlay
        if start is None or end is None:
            overlay = tuple(float(v) for v in live_overlay)
            self._manual_drag_interp_start_overlay = overlay
            self._manual_drag_interp_end_overlay = overlay
            self._manual_drag_interp_started_ts = time.perf_counter()
            self._manual_drag_interp_duration_s = 1.0 / 90.0
            return overlay

        now = time.perf_counter()
        duration = max(1e-4, float(self._manual_drag_interp_duration_s))
        t = max(0.0, min(1.0, (now - float(self._manual_drag_interp_started_ts)) / duration))
        curved_t = self._apply_roi_interpolation_curve(t, "ease_in_out")
        sampled = (
            float(start[0]) + ((float(end[0]) - float(start[0])) * curved_t),
            float(start[1]) + ((float(end[1]) - float(start[1])) * curved_t),
            float(start[2]) + ((float(end[2]) - float(start[2])) * curved_t),
            float(start[3]) + ((float(end[3]) - float(start[3])) * curved_t),
        )
        if t >= 1.0:
            self._manual_drag_interp_start_overlay = end
            self._manual_drag_interp_started_ts = now
        return sampled

    def _manual_roi_step_with_subpixel_float_target(
        self,
        current: Roi,
        target_overlay: tuple[float, float, float, float],
    ) -> tuple[Roi, float, float]:
        target_x, target_y, target_w, target_h = [float(v) for v in target_overlay]

        moving_only = (
            abs(target_w - float(current.w)) <= 1e-3
            and abs(target_h - float(current.h)) <= 1e-3
        )
        zoom_scale = max(1.0, FRAME_W / max(1.0, float(current.w)))
        smoothing = max(0.0, min(1.0, self._roi_smoothing_percent / 10.0))

        if moving_only:
            if zoom_scale >= 6.0:
                alpha_pos = 0.09
            elif zoom_scale >= 4.0:
                alpha_pos = 0.12
            else:
                alpha_pos = 0.16
        else:
            alpha_pos = 0.22
        alpha_size = 0.20

        alpha_scale = 1.15 - (0.55 * smoothing)
        alpha_pos = max(0.05, min(0.32, alpha_pos * alpha_scale))
        alpha_size = max(0.07, min(0.36, alpha_size * alpha_scale))

        current_cx = float(current.x) + (float(current.w) * 0.5)
        current_cy = float(current.y) + (float(current.h) * 0.5)
        target_cx = target_x + (target_w * 0.5)
        target_cy = target_y + (target_h * 0.5)

        desired_cx = current_cx + ((target_cx - current_cx) * alpha_pos)
        desired_cy = current_cy + ((target_cy - current_cy) * alpha_pos)
        desired_w = float(current.w) + ((target_w - float(current.w)) * alpha_size)

        quant_w = max(2, int(round(desired_w)) & ~1)
        quant_h = max(2, int(round(quant_w * 9.0 / 16.0)))
        desired_x = desired_cx - (float(quant_w) * 0.5)
        desired_y = desired_cy - (float(quant_h) * 0.5)

        carrier_roi = clamp_roi(
            Roi(
                int(round(desired_x)),
                int(round(desired_y)),
                quant_w,
                quant_h,
            )
        )

        carrier_cx = float(carrier_roi.x) + (float(carrier_roi.w) * 0.5)
        carrier_cy = float(carrier_roi.y) + (float(carrier_roi.h) * 0.5)
        source_dx = target_cx - carrier_cx
        source_dy = target_cy - carrier_cy

        sx = FRAME_W / max(1.0, float(carrier_roi.w))
        sy = FRAME_H / max(1.0, float(carrier_roi.h))
        max_shift_x = max(2.0, min(48.0, sx * 1.5))
        max_shift_y = max(2.0, min(48.0, sy * 1.5))
        shift_x = max(-max_shift_x, min(max_shift_x, -(source_dx * sx)))
        shift_y = max(-max_shift_y, min(max_shift_y, -(source_dy * sy)))

        # If even-pixel carrier ROI is unchanged during drag, ensure tiny
        # non-zero subpixel motion so output does not appear to pause.
        carrier_static = (
            carrier_roi.x == current.x
            and carrier_roi.y == current.y
            and carrier_roi.w == current.w
            and carrier_roi.h == current.h
        )
        if carrier_static and moving_only:
            min_drag_nudge_x = float(self._roi_min_drag_nudge)
            min_drag_nudge_y = float(self._roi_min_drag_nudge * 0.8)
            if abs(source_dx) > 0.010 and abs(shift_x) < min_drag_nudge_x:
                shift_x = max(-max_shift_x, min(max_shift_x, (-1.0 if source_dx > 0.0 else 1.0) * min_drag_nudge_x))
            if abs(source_dy) > 0.010 and abs(shift_y) < min_drag_nudge_y:
                shift_y = max(-max_shift_y, min(max_shift_y, (-1.0 if source_dy > 0.0 else 1.0) * min_drag_nudge_y))

        return carrier_roi, float(shift_x), float(shift_y)

    def _manual_roi_step_with_subpixel(self, current: Roi, target: Roi) -> tuple[Roi, float, float]:
        moving_only = current.w == target.w and current.h == target.h
        zoom_scale = roi_scale_from_roi(target)
        smoothing = max(0.0, min(1.0, self._roi_smoothing_percent / 10.0))

        if moving_only:
            if zoom_scale >= 6.0:
                alpha_pos = 0.09
            elif zoom_scale >= 4.0:
                alpha_pos = 0.12
            else:
                alpha_pos = 0.16
        else:
            alpha_pos = 0.22
        alpha_size = 0.20

        alpha_scale = 1.15 - (0.55 * smoothing)
        alpha_pos = max(0.05, min(0.32, alpha_pos * alpha_scale))
        alpha_size = max(0.07, min(0.36, alpha_size * alpha_scale))

        current_cx = float(current.x) + (float(current.w) * 0.5)
        current_cy = float(current.y) + (float(current.h) * 0.5)
        target_cx = float(target.x) + (float(target.w) * 0.5)
        target_cy = float(target.y) + (float(target.h) * 0.5)

        desired_cx = current_cx + ((target_cx - current_cx) * alpha_pos)
        desired_cy = current_cy + ((target_cy - current_cy) * alpha_pos)
        desired_w = float(current.w) + ((float(target.w) - float(current.w)) * alpha_size)

        quant_w = max(2, int(round(desired_w)) & ~1)
        quant_h = max(2, int(round(quant_w * 9.0 / 16.0)))
        desired_x = desired_cx - (float(quant_w) * 0.5)
        desired_y = desired_cy - (float(quant_h) * 0.5)

        carrier_roi = clamp_roi(
            Roi(
                int(round(desired_x)),
                int(round(desired_y)),
                quant_w,
                quant_h,
            )
        )

        carrier_cx = float(carrier_roi.x) + (float(carrier_roi.w) * 0.5)
        carrier_cy = float(carrier_roi.y) + (float(carrier_roi.h) * 0.5)
        source_dx = desired_cx - carrier_cx
        source_dy = desired_cy - carrier_cy

        sx = FRAME_W / max(1.0, float(carrier_roi.w))
        sy = FRAME_H / max(1.0, float(carrier_roi.h))
        max_shift_x = max(2.0, min(48.0, sx * 1.5))
        max_shift_y = max(2.0, min(48.0, sy * 1.5))
        shift_x = max(-max_shift_x, min(max_shift_x, -(source_dx * sx)))
        shift_y = max(-max_shift_y, min(max_shift_y, -(source_dy * sy)))

        return carrier_roi, float(shift_x), float(shift_y)

    def _schedule_roi_controls_sync(self, roi: Roi) -> None:
        self._pending_roi_controls_sync = clamp_roi(roi)
        if not self._roi_controls_sync_timer.isActive():
            self._roi_controls_sync_timer.start()

    def _flush_pending_roi_controls_sync(self) -> None:
        pending = self._pending_roi_controls_sync
        self._pending_roi_controls_sync = None
        if pending is None:
            return
        self._sync_controls_from_roi(pending)

    def _manual_roi_interaction_active(self) -> bool:
        if self._roi_keyframe_transition is not None:
            return False
        now = time.perf_counter()
        if (now - self._last_manual_roi_update_ts) <= 0.22:
            return True
        if self._manual_roi_send_timer.isActive():
            return True
        return self._pending_manual_controller_roi is not None

    def _sync_backend_roi_from_worker(self) -> dict[str, object]:
        if self._source_mode != "Blackmagic DeckLink" or self._controller_backend != "worker-process":
            return {}

        state = self._roi_keyframe_transition
        if not (isinstance(state, dict) and bool(state.get("backend_driven", False))):
            return {}

        if not hasattr(self._controller, "decklink_applied_roi"):
            return {}

        try:
            worker_roi = self._controller.decklink_applied_roi()
        except Exception:
            worker_roi = None

        worker_transition_state: dict[str, object] = {}
        if hasattr(self._controller, "decklink_roi_transition_state"):
            try:
                worker_transition_state = dict(self._controller.decklink_roi_transition_state())
            except Exception:
                worker_transition_state = {}
        transition_active = bool(worker_transition_state.get("active", False))
        if transition_active:
            state["worker_transition_seen"] = True

        if not isinstance(worker_roi, Roi):
            return worker_transition_state

        worker_roi = clamp_roi(worker_roi)
        state["current_roi_estimate"] = worker_roi
        # Never snap canvas ROI directly during a backend-driven transition.
        # We render a smoothed visual overlay and commit once complete.
        self._roi = worker_roi

        self._controller_roi_applied = worker_roi
        self._schedule_roi_controls_sync(worker_roi)

        # Render GUI interpolation from worker transition phase so the on-screen
        # ROI appears smooth between quantized applied-ROI steps.
        if transition_active:
            try:
                start_raw = worker_transition_state.get("start_roi", {})
                target_raw = worker_transition_state.get("target_roi", {})
                start_roi = clamp_roi(
                    Roi(
                        int(start_raw.get("x", worker_roi.x)),
                        int(start_raw.get("y", worker_roi.y)),
                        int(start_raw.get("w", worker_roi.w)),
                        int(start_raw.get("h", worker_roi.h)),
                    )
                )
                target_roi = clamp_roi(
                    Roi(
                        int(target_raw.get("x", worker_roi.x)),
                        int(target_raw.get("y", worker_roi.y)),
                        int(target_raw.get("w", worker_roi.w)),
                        int(target_raw.get("h", worker_roi.h)),
                    )
                )
                total_frames = max(1, int(worker_transition_state.get("total_frames", 1)))
                frame_progress = max(0.0, min(float(total_frames), float(worker_transition_state.get("frame_progress", 0.0))))
                t = frame_progress / float(total_frames)
                curve_mode = str(worker_transition_state.get("interpolation_mode", "linear"))
                curved_t = self._apply_roi_interpolation_curve(t, curve_mode)

                start_cx = float(start_roi.x) + (float(start_roi.w) * 0.5)
                start_cy = float(start_roi.y) + (float(start_roi.h) * 0.5)
                target_cx = float(target_roi.x) + (float(target_roi.w) * 0.5)
                target_cy = float(target_roi.y) + (float(target_roi.h) * 0.5)

                overlay_cx = start_cx + ((target_cx - start_cx) * curved_t)
                overlay_cy = start_cy + ((target_cy - start_cy) * curved_t)
                overlay_w = float(start_roi.w) + ((float(target_roi.w) - float(start_roi.w)) * curved_t)
                overlay_h = max(2.0, float(overlay_w * 9.0 / 16.0))
                overlay_x = overlay_cx - (overlay_w * 0.5)
                overlay_y = overlay_cy - (overlay_h * 0.5)
                self._input_canvas.set_visual_roi_overlay(overlay_x, overlay_y, overlay_w, overlay_h)
            except Exception:
                self._input_canvas.clear_visual_roi_overlay()
        else:
            self._input_canvas.clear_visual_roi_overlay()

        return worker_transition_state

    def _scaled_preview_target(self, target: tuple[int, int] | None, scale: float) -> tuple[int, int] | None:
        if target is None:
            return None
        if scale >= 0.999:
            return target
        w = max(160, int(round(target[0] * scale)))
        h = max(90, int(round(target[1] * scale)))
        return (w, h)

    def _queue_controller_roi_target(self, roi: Roi, anchor_to_current: bool = False) -> None:
        raw_target = clamp_roi(roi)
        latency = max(0.0, min(1.0, self._roi_latency_smoothing_percent / 100.0))
        anchor_roi = clamp_roi(self._controller_roi_applied)

        if anchor_to_current:
            self._controller_filtered_target_roi = anchor_roi
            self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}

        if latency > 0.0:
            beta = 1.0 - (0.82 * latency)
            if anchor_to_current:
                prev = anchor_roi
            else:
                prev = self._controller_filtered_target_roi if self._controller_filtered_target_roi is not None else raw_target
            filtered = clamp_roi(
                Roi(
                    int(round(prev.x + (raw_target.x - prev.x) * beta)),
                    int(round(prev.y + (raw_target.y - prev.y) * beta)),
                    int(round(prev.w + (raw_target.w - prev.w) * beta)),
                    int(round(prev.h + (raw_target.h - prev.h) * beta)),
                )
            )
            self._controller_filtered_target_roi = filtered
            self._controller_roi_target = filtered
        else:
            self._controller_filtered_target_roi = raw_target
            self._controller_roi_target = raw_target
        if not self._controller_roi_interp_timer.isActive():
            self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        target_scale = roi_scale_from_roi(self._controller_roi_target)
        smoothing = max(0.0, min(1.0, self._roi_smoothing_percent / 10.0))
        if target_scale >= 6.0:
            base_interval_ms = 8
        elif target_scale >= 4.0:
            base_interval_ms = 10
        else:
            base_interval_ms = 16
        interval_scale = 1.20 - (0.60 * smoothing)
        interval_ms = int(round(base_interval_ms * interval_scale))
        interval_ms = max(6, min(24, interval_ms))
        field_interval_ms = self._decklink_output_field_interval_ms()
        if field_interval_ms is not None:
            interval_ms = min(interval_ms, int(field_interval_ms))
        self._controller_roi_interp_timer.setInterval(interval_ms)
        if not self._controller_roi_interp_timer.isActive():
            self._controller_roi_interp_timer.start()

    def _apply_controller_roi_immediate(self, roi: Roi, reset_subpixel_shift: bool = True) -> None:
        clamped = clamp_roi(roi)
        self._controller_roi_target = None
        self._controller_roi_interp_timer.stop()
        self._controller_filtered_target_roi = None
        self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        if reset_subpixel_shift and hasattr(self._controller, "set_roi_subpixel_shift"):
            self._controller.set_roi_subpixel_shift(0.0, 0.0)
        moving_only = (
            clamped.w == self._controller_roi_applied.w
            and clamped.h == self._controller_roi_applied.h
            and hasattr(self._controller, "set_roi_position")
        )
        if moving_only:
            self._controller.set_roi_position(clamped.x, clamped.y)
        else:
            self._controller.set_roi(clamped)
        self._controller_roi_applied = clamped

    def _step_controller_roi_interpolation(self) -> None:
        target = self._controller_roi_target
        if target is None:
            self._controller_roi_interp_timer.stop()
            self._controller_filtered_target_roi = None
            self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
            return

        current = self._controller_roi_applied
        step_roi = self._interpolate_controller_roi_step(current, target)
        try:
            moving_only = (
                step_roi.w == current.w
                and step_roi.h == current.h
                and hasattr(self._controller, "set_roi_position")
            )
            if moving_only:
                self._controller.set_roi_position(step_roi.x, step_roi.y)
            else:
                self._controller.set_roi(step_roi)
        except Exception as exc:
            self._controller_roi_target = None
            self._controller_roi_interp_timer.stop()
            self._controller_filtered_target_roi = None
            self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
            self._update_status(f"ROI update failed: {exc}")
            return
        self._controller_roi_applied = step_roi

        if self._is_controller_roi_close(step_roi, target):
            if (
                target.x != step_roi.x
                or target.y != step_roi.y
                or target.w != step_roi.w
                or target.h != step_roi.h
            ):
                try:
                    moving_only_finalize = (
                        target.w == step_roi.w
                        and target.h == step_roi.h
                        and hasattr(self._controller, "set_roi_position")
                    )
                    if moving_only_finalize:
                        self._controller.set_roi_position(target.x, target.y)
                    else:
                        self._controller.set_roi(target)
                except Exception as exc:
                    self._update_status(f"ROI finalize failed: {exc}")
                else:
                    self._controller_roi_applied = target
            self._controller_roi_target = None
            self._controller_roi_interp_timer.stop()
            self._controller_filtered_target_roi = None
            self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}

    def _interpolate_controller_roi_step(self, current: Roi, target: Roi) -> Roi:
        moving_only = current.w == target.w and current.h == target.h
        zoom_scale = roi_scale_from_roi(target)
        smoothing = max(0.0, min(1.0, self._roi_smoothing_percent / 10.0))
        # Keep output translation slightly more eased than resize/scale updates.
        if moving_only:
            if zoom_scale >= 6.0:
                alpha_pos = 0.11
            elif zoom_scale >= 4.0:
                alpha_pos = 0.15
            else:
                alpha_pos = 0.18
        else:
            alpha_pos = 0.26
        alpha_size = 0.24

        alpha_scale = 1.20 - (0.60 * smoothing)
        alpha_pos = max(0.07, min(0.38, alpha_pos * alpha_scale))
        alpha_size = max(0.09, min(0.42, alpha_size * alpha_scale))

        if zoom_scale >= 6.0:
            lag_limit = 8
        elif zoom_scale >= 4.0:
            lag_limit = 12
        else:
            lag_limit = 16

        near_target_deadband = 1 + int(round(smoothing * 2.0))

        def _step(c: int, t: int, alpha: float, key: str, low_latency: bool = False) -> int:
            delta = t - c
            if delta == 0:
                self._controller_interp_residual[key] = 0.0
                return c

            abs_delta = abs(delta)
            effective_alpha = alpha
            if low_latency:
                accel = (abs_delta / (abs_delta + 64.0)) * 0.38
                effective_alpha = min(0.72, alpha + accel)

                if abs_delta <= near_target_deadband:
                    self._controller_interp_residual[key] = 0.0
                    return t

            raw_move = (delta * effective_alpha) + float(self._controller_interp_residual[key])
            sign = 1 if raw_move > 0 else -1
            move_abs = int(abs(raw_move))
            move = sign * move_abs if move_abs > 0 else 0
            self._controller_interp_residual[key] = raw_move - float(move)

            if low_latency:
                overshoot = abs_delta - lag_limit
                if overshoot > 0:
                    sign = 1 if delta > 0 else -1
                    min_catch_up = int(math.ceil(overshoot * 0.65))
                    enforced = sign * max(abs(move), min_catch_up)
                    if enforced != move:
                        move = enforced
                        self._controller_interp_residual[key] = 0.0

            if move == 0:
                if abs_delta >= max(2, near_target_deadband + 1):
                    move = 1 if delta > 0 else -1
                    self._controller_interp_residual[key] = 0.0
                else:
                    return c
            return c + move

        return clamp_roi(
            Roi(
                _step(current.x, target.x, alpha_pos, "x", low_latency=moving_only),
                _step(current.y, target.y, alpha_pos, "y", low_latency=moving_only),
                _step(current.w, target.w, alpha_size, "w"),
                _step(current.h, target.h, alpha_size, "h"),
            )
        )

    def _on_roi_smoothing_changed(self, value: int) -> None:
        clamped = max(0, min(10, int(value)))
        self._roi_smoothing_percent = clamped
        self.roi_smoothing_value_label.setText(f"{clamped}/10")
        self._manual_roi_target_history = []
        self._input_canvas.set_smoothing_percent(clamped)

    def _on_roi_latency_smoothing_changed(self, value: int) -> None:
        clamped = max(0, min(100, int(value)))
        self._roi_latency_smoothing_percent = clamped
        self.roi_latency_smoothing_value_label.setText(f"{clamped}%")
        self._input_canvas.set_latency_smoothing_percent(clamped)

    def _on_roi_drag_x_hysteresis_changed(self, value: float) -> None:
        clamped = max(0.10, min(1.20, float(value)))
        self._roi_drag_x_hysteresis_px = clamped
        self._input_canvas.set_drag_x_hysteresis_px(clamped)

    def _on_roi_manual_drag_hold_changed(self, value: float) -> None:
        clamped = max(0.05, min(0.50, float(value)))
        self._roi_manual_drag_hold_s = clamped
        self._apply_manual_drag_tuning_to_controller()

    def _on_roi_min_drag_nudge_changed(self, value: float) -> None:
        clamped = max(0.0, min(0.50, float(value)))
        self._roi_min_drag_nudge = clamped

    def _on_interlaced_field2_phase_fraction_changed(self, value: float) -> None:
        clamped = _clamp_interlaced_field2_phase_fraction(float(value))
        self._interlaced_field2_phase_fraction = clamped
        LOGGER.info("Interlaced field2 phase tuning changed in GUI: fraction=%.2f", clamped)
        self._apply_interlaced_phase_tuning_to_controller()

    def _apply_manual_drag_tuning_to_controller(self) -> None:
        if not hasattr(self._controller, "set_roi_manual_drag_hold_seconds"):
            return
        try:
            self._controller.set_roi_manual_drag_hold_seconds(float(self._roi_manual_drag_hold_s))
        except Exception:
            LOGGER.exception("Failed to apply manual drag hold tuning")

    def _apply_interlaced_phase_tuning_to_controller(self) -> None:
        if not hasattr(self._controller, "set_interlaced_field2_phase_fraction"):
            return
        try:
            self._controller.set_interlaced_field2_phase_fraction(float(self._interlaced_field2_phase_fraction))
        except Exception:
            LOGGER.exception("Failed to apply interlaced field2 phase tuning")

    def _is_controller_roi_close(self, roi_a: Roi, roi_b: Roi) -> bool:
        return (
            abs(roi_a.x - roi_b.x) <= 1
            and abs(roi_a.y - roi_b.y) <= 1
            and abs(roi_a.w - roi_b.w) <= 2
            and abs(roi_a.h - roi_b.h) <= 2
        )

    def _on_scale_from_canvas(self, scale: float) -> None:
        if self._updating_controls:
            return
        self._updating_controls = True
        self.scale_spin.setValue(scale)
        self._updating_controls = False

    def _on_roi_spin_changed(self) -> None:
        if self._updating_controls:
            return

        self._cancel_roi_keyframe_transition()

        sender = self.sender()
        roi_w = self.roi_w_spin.value()
        roi_h = self.roi_h_spin.value()

        if sender is self.roi_h_spin:
            roi_w = int(round(roi_h * 16.0 / 9.0))
        else:
            roi_h = int(round(roi_w * 9.0 / 16.0))

        roi = clamp_roi(
            Roi(
                self.roi_x_spin.value(),
                self.roi_y_spin.value(),
                roi_w,
                roi_h,
            )
        )
        self._roi = roi
        self._input_canvas.set_roi(roi)
        self._apply_controller_roi_immediate(roi)
        self._sync_controls_from_roi(roi)

    def _on_scale_spin_changed(self, value: float) -> None:
        if self._updating_controls:
            return

        self._cancel_roi_keyframe_transition()

        center_x = self._roi.x + (self._roi.w / 2.0)
        center_y = self._roi.y + (self._roi.h / 2.0)
        roi = roi_from_scale(value, center_x, center_y)
        self._roi = roi
        self._input_canvas.set_roi(roi)
        self._apply_controller_roi_immediate(roi)
        self._sync_controls_from_roi(roi)

    def _sync_ai_sr_basic_scaling_ui(self, notify: bool = False, runtime_force_disable: bool = False) -> None:
        ai_sr_selected = bool(self.enable_ai_sr_checkbox.isChecked())
        basic_forced_off = False

        if ai_sr_selected and self.enable_sr_checkbox.isChecked():
            if runtime_force_disable:
                self.enable_sr_checkbox.setChecked(False)
            else:
                self.enable_sr_checkbox.blockSignals(True)
                self.enable_sr_checkbox.setChecked(False)
                self.enable_sr_checkbox.blockSignals(False)
                self._controller.enable_basic_scaling = False
            basic_forced_off = True

        self.enable_sr_checkbox.setEnabled(not ai_sr_selected)

        basic_controls_enabled = bool(self.enable_sr_checkbox.isChecked()) and not ai_sr_selected
        self.sr_mode_combo.setEnabled(basic_controls_enabled)
        self.sr_flavor_combo.setEnabled(basic_controls_enabled)
        self.sr_manual_combo.setEnabled(basic_controls_enabled)
        self.auto_sr_max_combo.setEnabled(basic_controls_enabled)

        if ai_sr_selected:
            self.enable_sr_checkbox.setToolTip("Basic CUDA scaling is disabled while AI SR (ONNX) is enabled.")
        else:
            self.enable_sr_checkbox.setToolTip("")

        if notify and basic_forced_off:
            self._update_status("AI SR ONNX selected: basic CUDA scaling has been disabled automatically")

    def _on_sr_mode_changed(self) -> None:
        if self.enable_ai_sr_checkbox.isChecked():
            return
        mode = self.sr_mode_combo.currentText()
        try:
            if mode == "Auto":
                self._controller.set_auto_basic_scaling()
            else:
                self._controller.set_manual_basic_scaling(int(self.sr_manual_combo.currentText()))
        except Exception as exc:
            self._update_status(f"Basic scaling mode change failed: {exc}")

    def _on_sr_manual_changed(self) -> None:
        if self.sr_mode_combo.currentText() != "Manual":
            return
        try:
            self._controller.set_manual_basic_scaling(int(self.sr_manual_combo.currentText()))
        except Exception as exc:
            self._update_status(f"Manual basic scaling change failed: {exc}")

    def _on_sr_flavor_changed(self) -> None:
        selected_label = self.sr_flavor_combo.currentText()
        selected_name = SR_FLAVOR_LABEL_TO_NAME.get(selected_label, "bilinear_sharp")
        if not getattr(self._controller, "basic_scaling_method_supported", False):
            self._update_status("Basic scaling method is not supported by the loaded video_processor build; rebuild extension to enable")
            return
        try:
            self._controller.set_basic_scaling_method(selected_name)
            applied_name = getattr(self._controller, "basic_scaling_method", selected_name)
            applied_label = SR_FLAVOR_NAME_TO_LABEL.get(applied_name, applied_name)
            effective_sr = int(self._controller.effective_scale()) if hasattr(self._controller, "effective_scale") else 1
            if effective_sr <= 1:
                self._update_status(
                    f"Basic scaling method applied: {applied_label} | effective scaling=1 (set Manual basic scaling to 4 or 8 to see visible method differences)"
                )
            else:
                self._update_status(f"Basic scaling method applied: {applied_label}")
        except Exception as exc:
            self._update_status(f"Basic scaling method change failed: {exc}")

    def _on_auto_sr_max_changed(self) -> None:
        try:
            max_scale = int(self.auto_sr_max_combo.currentText())
            self._controller.set_max_auto_basic_scaling(max_scale)
            if self.sr_mode_combo.currentText() == "Auto":
                self._controller.set_auto_basic_scaling()
            self._update_status(f"Auto basic scaling max set to {max_scale}")
        except Exception as exc:
            self._update_status(f"Auto basic scaling max change failed: {exc}")

    def _on_enable_sr_toggled(self, checked: bool) -> None:
        if self._updating_controls:
            self._controller.enable_basic_scaling = bool(checked)
            self._sync_ai_sr_basic_scaling_ui(notify=False)
            return

        if checked and self.enable_ai_sr_checkbox.isChecked():
            self.enable_sr_checkbox.blockSignals(True)
            self.enable_sr_checkbox.setChecked(False)
            self.enable_sr_checkbox.blockSignals(False)
            self._controller.enable_basic_scaling = False
            self._sync_ai_sr_basic_scaling_ui(notify=False)
            self._update_status("Basic CUDA scaling remains disabled while AI SR ONNX is enabled")
            return

        previous_value = self._controller.enable_basic_scaling
        self._controller.enable_basic_scaling = checked
        try:
            self._controller.create(self._roi)
            if self._source_mode == "Blackmagic DeckLink":
                self._start_decklink_sessions()
            self._sync_ai_sr_basic_scaling_ui(notify=False)
            self._update_status("Recreated processor after basic scaling toggle")
        except Exception as exc:
            # Roll back to previous SR enable state so the app can recover in-place.
            self._controller.enable_basic_scaling = previous_value
            try:
                self._controller.create(self._roi)
                if self._source_mode == "Blackmagic DeckLink":
                    self._start_decklink_sessions()
            except Exception:
                pass

            self.enable_sr_checkbox.blockSignals(True)
            self.enable_sr_checkbox.setChecked(previous_value)
            self.enable_sr_checkbox.blockSignals(False)
            self._sync_ai_sr_basic_scaling_ui(notify=False)
            self._update_status(f"Processor recreate failed: {exc}")

    def _on_deinterlace_toggled(self, checked: bool) -> None:
        try:
            self._controller.set_deinterlace_enabled(checked)
            mode_text = "enabled" if checked else "disabled"
            self._update_status(f"Deinterlace {mode_text}")
        except Exception as exc:
            self._update_status(f"Deinterlace toggle failed: {exc}")

    def _on_reinterlace_toggled(self, checked: bool) -> None:
        try:
            if hasattr(self._controller, "set_reinterlace_enabled"):
                self._controller.set_reinterlace_enabled(checked)
            mode_text = "enabled" if checked else "disabled"
            self._update_status(f"Reinterlace {mode_text}")
        except Exception as exc:
            self._update_status(f"Reinterlace toggle failed: {exc}")

    def _on_deinterlace_method_changed(self) -> None:
        if not self._updating_controls:
            self._deinterlace_method_user_selected = True
        method_label = self.deinterlace_method_combo.currentText()
        method_name = DEINTERLACE_METHOD_LABEL_TO_NAME.get(method_label, "bob")
        try:
            self._controller.set_deinterlace_method(method_name)
            applied_method = getattr(self._controller, "deinterlace_method", method_name)
            self._update_status(f"Deinterlace method applied: {applied_method}")
        except Exception as exc:
            self._update_status(f"Deinterlace method change failed: {exc}")

    def _on_denoise_settings_changed(self) -> None:
        method_label = self.denoise_method_combo.currentText()
        method_name = DENOISE_METHOD_LABEL_TO_NAME.get(method_label, "off")
        strength = float(self.denoise_strength_spin.value())
        try:
            self._controller.set_denoise_settings(method_name, strength)
            applied_method = getattr(self._controller, "denoise_method", method_name)
            applied_strength = float(getattr(self._controller, "denoise_strength", strength))
            self._update_status(f"Denoise applied: {applied_method} (strength={applied_strength:.2f})")
        except Exception as exc:
            self._update_status(f"Denoise setting update failed: {exc}")

    def _on_perf_guard_toggled(self, checked: bool) -> None:
        self._perf_guard_enabled = checked
        self._perf_guard_low_fps_seconds = 0
        self._perf_guard_last_action = ""

    def _on_enable_ai_sr_toggled(self, checked: bool) -> None:
        if self._updating_controls:
            self._controller.ai_sr_enabled = bool(checked)
            self._sync_ai_sr_basic_scaling_ui(notify=False)
            return

        self._sync_ai_sr_basic_scaling_ui(notify=checked, runtime_force_disable=checked)
        try:
            self._controller.set_ai_sr_enabled(checked)
            if checked:
                model_path = self.ai_sr_model_combo.currentText().strip()
                self._update_status(f"AI SR ONNX mode requested | basic CUDA scaling disabled | awaiting worker ack | model={model_path}")
            else:
                self._update_status("AI SR disable requested | awaiting worker ack")
        except Exception as exc:
            self._update_status(f"AI SR toggle failed: {exc}")

    def _on_enable_rtx_vsr_toggled(self, checked: bool) -> None:
        try:
            self._controller.set_rtx_vsr_enabled(checked)
            if checked:
                if bool(getattr(self._controller, "ai_sr_enabled", False)):
                    self._update_status("RTX VSR enable requested | awaiting worker ack | note: with AI SR enabled, RTX runs as fallback when AI is unavailable on a frame")
                else:
                    self._update_status("RTX VSR enable requested | awaiting worker ack")
            else:
                self._update_status("RTX VSR disable requested | awaiting worker ack")
        except Exception as exc:
            self._update_status(f"RTX VSR toggle failed: {exc}")

    def _on_ai_sr_model_path_changed(self, model_path: str) -> None:
        if self._updating_controls:
            return
        if not model_path.strip():
            return
        model_candidate = Path(model_path.strip())
        if not model_candidate.exists():
            self._update_status(f"AI SR model file not found: {model_candidate}")
            return
        try:
            self._controller.set_ai_sr_model_path(model_path.strip())
            self._update_status("AI SR model path update requested | awaiting worker ack")
        except Exception as exc:
            self._update_status(f"AI SR model update failed: {exc}")

    def _on_ai_sr_model_apply_clicked(self) -> None:
        self._on_ai_sr_model_path_changed(self.ai_sr_model_combo.currentText())

    def _on_ai_sr_model_selection_changed(self, model_path: str) -> None:
        model_key = model_path.strip()
        if not model_key:
            return
        profile = self._ai_sr_profiles.get(model_key)
        if profile is None:
            return
        self._apply_ai_sr_profile(profile)
        self._update_status("Loaded saved AI SR profile for selected model")

    def _on_ai_sr_model_refresh_clicked(self) -> None:
        previous_text = self.ai_sr_model_combo.currentText().strip()
        self._refresh_ai_sr_model_options(preferred_model_path=previous_text)
        model_count = self.ai_sr_model_combo.count()
        self._update_status(f"AI SR model list refreshed ({model_count} model{'s' if model_count != 1 else ''})")

    def _on_ai_sr_model_lightest_clicked(self) -> None:
        model_paths = [
            Path(self.ai_sr_model_combo.itemText(i).strip())
            for i in range(self.ai_sr_model_combo.count())
            if self.ai_sr_model_combo.itemText(i).strip()
        ]
        existing = [p for p in model_paths if p.exists() and p.is_file()]
        if not existing:
            self._update_status("No AI SR model files found to rank")
            return

        lightest = min(existing, key=lambda p: p.stat().st_size)
        self.ai_sr_model_combo.setCurrentText(str(lightest))
        size_mb = float(lightest.stat().st_size) / (1024.0 * 1024.0)
        self._update_status(f"Selected lightest AI SR model: {lightest.name} ({size_mb:.1f} MB)")

    def _on_ai_sr_model_quantize_clicked(self) -> None:
        model_text = self.ai_sr_model_combo.currentText().strip()
        if not model_text:
            self._update_status("INT8 quantize failed: AI SR model path is empty")
            return

        model_path = Path(model_text)
        if not model_path.exists() or not model_path.is_file():
            self._update_status(f"INT8 quantize failed: model file not found: {model_path}")
            return

        out_path = model_path.with_name(f"{model_path.stem}_int8{model_path.suffix}")
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(
                str(model_path),
                str(out_path),
                weight_type=QuantType.QInt8,
                per_channel=True,
            )
        except Exception as exc:
            self._update_status(f"INT8 quantize failed: {exc}")
            return

        self._refresh_ai_sr_model_options(preferred_model_path=str(out_path))
        self.ai_sr_provider_combo.setCurrentText("trt")
        self.ai_sr_trt_precision_combo.setCurrentText("int8")
        self.ai_sr_require_gpu_checkbox.setChecked(True)
        self._update_status(f"Created INT8 model: {out_path.name} | provider set to trt/int8")

    def _on_ai_sr_tuning_apply_clicked(self) -> None:
        try:
            profile = self._current_ai_sr_profile()
            self._controller.set_ai_sr_settings(
                provider=str(profile["provider"]),
                require_gpu=bool(profile["require_gpu"]),
                inference_fps=int(profile["inference_fps"]),
                trt_precision=str(profile["trt_precision"]),
                strict=bool(profile["strict"]),
                input_align=int(profile["input_align"]),
                roi_overscan_percent=float(profile["roi_overscan_percent"]),
                inference_divisor=int(profile["inference_divisor"]),
                detail_preserve_percent=float(profile["detail_preserve_percent"]),
                post_denoise_method=str(profile["post_denoise_method"]),
                post_denoise_strength=float(profile["post_denoise_strength"]),
                post_artifact_reduction_method=str(profile["post_artifact_reduction_method"]),
                post_artifact_reduction_strength=float(profile["post_artifact_reduction_strength"]),
                post_exaggeration_enabled=bool(profile["post_exaggeration_enabled"]),
                post_exaggeration_gain=float(profile["post_exaggeration_gain"]),
            )
            self._update_status("AI SR tuning update requested | awaiting worker ack")
        except Exception as exc:
            self._update_status(f"AI SR tuning update failed: {exc}")

    def _on_rtx_vsr_settings_apply_clicked(self) -> None:
        try:
            quality = self.rtx_vsr_quality_combo.currentText().strip().lower()
            scale = int(self.rtx_vsr_scale_combo.currentText())
            post_scale_method = RTX_POST_SCALE_METHOD_LABEL_TO_NAME.get(
                self.rtx_vsr_post_scale_method_combo.currentText(),
                "bicubic",
            )
            self._controller.set_rtx_vsr_settings(
                quality,
                scale,
                post_scale_method,
                bool(self.rtx_thdr_enable_checkbox.isChecked()),
                int(self.rtx_thdr_contrast_spin.value()),
                int(self.rtx_thdr_saturation_spin.value()),
                int(self.rtx_thdr_middle_gray_spin.value()),
                int(self.rtx_thdr_max_luminance_spin.value()),
            )
            if bool(getattr(self._controller, "ai_sr_enabled", False)):
                self._update_status("RTX VSR settings update requested | awaiting worker ack | note: with AI SR enabled, RTX runs as fallback when AI is unavailable on a frame")
            else:
                self._update_status("RTX VSR settings update requested | awaiting worker ack")
        except Exception as exc:
            self._update_status(f"RTX VSR settings update failed: {exc}")

    def _on_ai_sr_profile_save_clicked(self) -> None:
        model_path = self.ai_sr_model_combo.currentText().strip()
        if not model_path:
            self._update_status("Save profile failed: AI SR model path is empty")
            return
        self._ai_sr_profiles[model_path] = self._current_ai_sr_profile()
        self._save_ai_sr_profiles()
        self._update_status("Saved AI SR tuning profile for selected model")

    def _on_ai_sr_profile_load_clicked(self) -> None:
        model_path = self.ai_sr_model_combo.currentText().strip()
        if not model_path:
            self._update_status("Load profile failed: AI SR model path is empty")
            return
        profile = self._ai_sr_profiles.get(model_path)
        if profile is None:
            self._update_status("No saved AI SR profile for selected model")
            return
        self._apply_ai_sr_profile(profile)
        self._refresh_ai_sr_runtime_panel()
        self._refresh_rtx_vsr_runtime_panel()
        self._update_status("Loaded AI SR profile for selected model")

    def _refresh_ai_sr_runtime_panel(self) -> None:
        info = getattr(self._controller, "ai_sr_info", None) or {}
        enabled = bool(getattr(self._controller, "ai_sr_enabled", False))
        active = bool(getattr(self._controller, "ai_sr_active", False))
        error_text = getattr(self._controller, "ai_sr_error", None)
        warning_text = getattr(self._controller, "ai_sr_last_warning", None)

        provider = str(info.get("provider", "n/a"))
        provider_upper = provider.upper()
        gpu_active = provider in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
        gpu_state = "YES" if gpu_active else "NO"
        requested_provider = str(info.get("requested_provider", getattr(self._controller, "ai_sr_provider", "auto")))
        trt_precision = str(info.get("trt_precision", getattr(self._controller, "ai_sr_trt_precision", "fp16"))).lower()

        available = info.get("available_providers", [])
        if isinstance(available, (list, tuple)):
            available_text = ", ".join(str(item) for item in available) if available else "n/a"
        else:
            available_text = str(available)

        inference_fps = int(info.get("inference_fps", info.get("frame_interval", getattr(self._controller, "ai_sr_frame_interval", 1))))
        inference_divisor = int(info.get("inference_divisor", getattr(self._controller, "ai_sr_inference_divisor", 0)))
        hold_last_frame = bool(info.get("hold_last_frame", getattr(self._controller, "ai_sr_hold_last_frame", True)))
        max_hold_ms = float(info.get("max_hold_ms", getattr(self._controller, "ai_sr_max_hold_ms", 0.0)))
        post_denoise_method = str(info.get("post_denoise_method", getattr(self._controller, "ai_sr_post_denoise_method", "off")))
        post_denoise_strength = float(info.get("post_denoise_strength", getattr(self._controller, "ai_sr_post_denoise_strength", 0.0)))
        post_artifact_method = str(
            info.get(
                "post_artifact_reduction_method",
                getattr(self._controller, "ai_sr_post_artifact_reduction_method", "off"),
            )
        )
        post_artifact_strength = float(
            info.get(
                "post_artifact_reduction_strength",
                getattr(self._controller, "ai_sr_post_artifact_reduction_strength", 0.0),
            )
        )
        post_exaggeration_enabled = bool(
            info.get(
                "post_exaggeration_enabled",
                getattr(self._controller, "ai_sr_post_exaggeration_enabled", False),
            )
        )
        post_exaggeration_gain = float(
            info.get(
                "post_exaggeration_gain",
                getattr(self._controller, "ai_sr_post_exaggeration_gain", 2.0),
            )
        )

        ai_applied = 0
        ai_reused = 0
        ai_passthrough = 0
        worker_fps = 0.0
        timing_stats: dict[str, object] = {}
        if hasattr(self._controller, "decklink_ai_sr_counts"):
            ai_applied, ai_reused, ai_passthrough = self._controller.decklink_ai_sr_counts()
        if hasattr(self._controller, "decklink_processed_fps"):
            worker_fps = float(self._controller.decklink_processed_fps())
        if hasattr(self._controller, "decklink_ai_timing_stats"):
            timing_stats = dict(self._controller.decklink_ai_timing_stats())

        avg_prep_ms = timing_stats.get("avg_prep_ms", info.get("avg_prep_ms"))
        avg_infer_ms = timing_stats.get("avg_infer_ms", info.get("avg_infer_ms"))
        avg_post_ms = timing_stats.get("avg_post_ms", info.get("avg_post_ms"))
        avg_total_ms = timing_stats.get("avg_total_ms", info.get("avg_total_ms"))
        timing_warmup_frames = timing_stats.get("timing_warmup_frames", info.get("timing_warmup_frames"))
        timing_warmup_remaining = timing_stats.get("timing_warmup_remaining", info.get("timing_warmup_remaining"))
        avg_prep_text = f"{float(avg_prep_ms):.2f} ms" if isinstance(avg_prep_ms, (int, float)) else "n/a"
        avg_infer_text = f"{float(avg_infer_ms):.2f} ms" if isinstance(avg_infer_ms, (int, float)) else "n/a"
        avg_post_text = f"{float(avg_post_ms):.2f} ms" if isinstance(avg_post_ms, (int, float)) else "n/a"
        avg_total_text = f"{float(avg_total_ms):.2f} ms" if isinstance(avg_total_ms, (int, float)) else "n/a"

        io_binding_enabled = timing_stats.get("io_binding_enabled", info.get("io_binding_enabled", False))
        io_binding_error = timing_stats.get("io_binding_error", info.get("io_binding_error"))
        io_binding_text = "on" if bool(io_binding_enabled) else "off"
        pipeline_order = str(info.get("pipeline_order", "crop/preprocess -> onnx(cuda) -> cuda_postprocess -> uyvy"))
        postprocess_gpu_chain = str(
            info.get("postprocess_gpu_chain", "resize/sharpen -> post_denoise(xN) -> post_artifact_reduction(xN) -> rgb_to_uyvy")
        )
        post_exaggeration_passes = int(info.get("post_exaggeration_passes", 2 if post_exaggeration_enabled else 1))
        onnx_output_copy_to_cpu = bool(info.get("onnx_output_copy_to_cpu", True))
        detail_preserve_note = str(info.get("detail_preserve_note", "")).strip()

        lines = [
            f"Enabled: {enabled} | Active: {active}",
            f"GPU active: {gpu_state} | Provider: {provider_upper} | Requested: {requested_provider}",
            f"TensorRT precision: {trt_precision}",
            f"Available providers: {available_text}",
            f"Model path: {info.get('model_path', getattr(self._controller, 'ai_sr_model_path', 'n/a'))}",
            f"Model scale: {info.get('model_scale', 'n/a')} | Input tensor: {info.get('model_input_w', 'n/a')}x{info.get('model_input_h', 'n/a')} | DType: {info.get('input_dtype', 'n/a')}",
            f"Pipeline: {pipeline_order}",
            f"I/O binding: {io_binding_text}",
            f"AI stage ms (avg): prep={avg_prep_text}, infer={avg_infer_text}, post={avg_post_text}, total={avg_total_text}",
            f"Worker FPS: {worker_fps:.1f}",
            (
                "Tuning: "
                f"inference_fps={inference_fps}, "
                f"strict={info.get('strict_mode', getattr(self._controller, 'ai_sr_strict', False))}, "
                f"align={info.get('input_align', getattr(self._controller, 'ai_sr_input_align', 'n/a'))}, "
                f"overscan={info.get('roi_overscan_percent', getattr(self._controller, 'ai_sr_roi_overscan_percent', 'n/a'))}, "
                f"divisor={inference_divisor}, "
                f"detail={info.get('detail_preserve_percent', getattr(self._controller, 'ai_sr_detail_preserve_percent', 'n/a'))}, "
                f"post_denoise={post_denoise_method}@{post_denoise_strength:.2f}, "
                f"post_artifact={post_artifact_method}@{post_artifact_strength:.2f}, "
                f"post_exaggerated={post_exaggeration_enabled}@{post_exaggeration_gain:.2f}, "
                f"hold_last={hold_last_frame}, max_hold_ms={max_hold_ms:.0f}"
            ),
            f"Frames: fresh={ai_applied}, reused={ai_reused}, passthrough={ai_passthrough}",
            f"Postprocess GPU chain: {postprocess_gpu_chain} (passes={post_exaggeration_passes})",
        ]

        if enabled and active and inference_fps <= 2:
            lines.append(
                "Visibility warning: AI inference FPS is very low (1-2), so output updates can look like passthrough. Increase AI inference FPS."
            )

        if isinstance(timing_warmup_frames, (int, float)) and isinstance(timing_warmup_remaining, (int, float)):
            lines.append(
                f"Timing warmup: excluded first {int(timing_warmup_frames)} sample(s), remaining={max(0, int(timing_warmup_remaining))}"
            )

        if enabled:
            lines.append("Mode: ONNX AI SR only (basic CUDA scaling is disabled while AI SR is enabled).")

        lines.append("Scheduler: worker skips frames and submits inference jobs to match the target AI inference FPS.")
        if onnx_output_copy_to_cpu:
            lines.append("GPU pipeline note: ONNX output is currently copied back to CPU for post/conversion steps.")
        else:
            lines.append("GPU pipeline note: ONNX output stays on GPU and feeds native CUDA postprocess without CPU tensor copy.")
        if detail_preserve_note:
            lines.append(f"Detail preserve: {detail_preserve_note}")
        if inference_fps >= 50:
            lines.append("Throughput tip: very high AI inference FPS targets can still saturate the GPU and cause bursty output cadence.")

        if error_text:
            lines.append(f"Error: {error_text}")
        elif io_binding_error:
            lines.append(f"I/O binding fallback: {io_binding_error}")
        elif warning_text:
            lines.append(f"Warning: {warning_text}")

        self.ai_sr_runtime_label.setText("\n".join(lines))

    def _refresh_rtx_vsr_runtime_panel(self) -> None:
        info = getattr(self._controller, "rtx_vsr_info", None) or {}
        enabled = bool(getattr(self._controller, "rtx_vsr_enabled", False))
        active = bool(getattr(self._controller, "rtx_vsr_active", False))
        error_text = getattr(self._controller, "rtx_vsr_error", None)

        quality = info.get("quality", getattr(self._controller, "rtx_vsr_quality", "high"))
        scale = info.get("scale", getattr(self._controller, "rtx_vsr_scale", 2))
        post_scale_method = str(info.get("post_scale_method", getattr(self._controller, "rtx_vsr_post_scale_method", "bicubic")))
        post_scale_label = RTX_POST_SCALE_METHOD_NAME_TO_LABEL.get(post_scale_method, post_scale_method)
        thdr_enabled = bool(info.get("thdr_enabled", getattr(self._controller, "rtx_thdr_enabled", False)))
        thdr_contrast = int(info.get("thdr_contrast", getattr(self._controller, "rtx_thdr_contrast", 50)))
        thdr_saturation = int(info.get("thdr_saturation", getattr(self._controller, "rtx_thdr_saturation", 50)))
        thdr_middle_gray = int(info.get("thdr_middle_gray", getattr(self._controller, "rtx_thdr_middle_gray", 50)))
        thdr_max_luminance = int(info.get("thdr_max_luminance", getattr(self._controller, "rtx_thdr_max_luminance", 1000)))
        backend = info.get("backend", "n/a")
        input_w = info.get("input_w", "n/a")
        input_h = info.get("input_h", "n/a")
        output_w = info.get("output_w", "n/a")
        output_h = info.get("output_h", "n/a")

        lines = [
            f"Enabled: {enabled} | Active: {active}",
            f"Backend: {backend}",
            f"Quality: {quality} | Scale: {scale} | Post scale: {post_scale_label}",
            f"VSR input resolution: {input_w}x{input_h} | Output resolution: {output_w}x{output_h}",
            (
                "TrueHDR: "
                f"enabled={thdr_enabled}, "
                f"contrast={thdr_contrast}, "
                f"saturation={thdr_saturation}, "
                f"middle_gray={thdr_middle_gray}, "
                f"max_luminance={thdr_max_luminance}"
            ),
        ]
        if enabled and not active and bool(getattr(self._controller, "ai_sr_enabled", False)):
            lines.append("Note: RTX VSR path is bypassed while AI SR is enabled.")
        if error_text:
            lines.append(f"Error: {error_text}")

        self.rtx_vsr_runtime_label.setText("\n".join(lines))
        self.rtx_vsr_scaling_info_label.setText(
            (
                f"Scale: {scale} | Post method: {post_scale_label}\n"
                f"VSR input resolution: {input_w}x{input_h}\n"
                f"Output resolution: {output_w}x{output_h}"
            )
        )

    def _target_fps(self) -> float:
        return float(max(1, self.fps_spin.value()))

    def _evaluate_frame_and_buffer_health(self, preview_fps: float, output_fps: float) -> str:
        _ = preview_fps
        nominal_fps = 0.0
        if hasattr(self._controller, "decklink_output_nominal_fps"):
            try:
                nominal_fps = float(self._controller.decklink_output_nominal_fps())
            except Exception:
                nominal_fps = 0.0
        if nominal_fps <= 0.0:
            nominal_fps = float(max(1.0, self._target_fps()))

        output_ratio = float(output_fps) / max(1.0, nominal_fps)
        interaction_active = bool(self._roi_keyframe_transition is not None or self._manual_roi_interaction_active())

        pipeline_timing_health: dict[str, object] = {}
        if hasattr(self._controller, "decklink_pipeline_timing_health"):
            try:
                pipeline_timing_health = dict(self._controller.decklink_pipeline_timing_health())
            except Exception:
                pipeline_timing_health = {}

        emitted_frames = int(pipeline_timing_health.get("frames_emitted", 0))
        deadline_miss_ratio = float(pipeline_timing_health.get("deadline_miss_ratio", 0.0))
        deadline_miss_streak = int(pipeline_timing_health.get("deadline_miss_streak", 0))
        deadline_miss_max_streak = int(pipeline_timing_health.get("deadline_miss_max_streak", 0))
        deadline_late_ms_ema = float(pipeline_timing_health.get("deadline_late_ms_ema", 0.0))
        e2e_ms_ema = float(pipeline_timing_health.get("e2e_ms_ema", 0.0))
        process_ms_ema = float(pipeline_timing_health.get("process_ms_ema", 0.0))
        capture_queue_ms_ema = float(pipeline_timing_health.get("capture_queue_ms_ema", 0.0))
        output_queue_ms_ema = float(pipeline_timing_health.get("output_queue_ms_ema", 0.0))
        output_wait_ms_ema = float(pipeline_timing_health.get("output_wait_ms_ema", 0.0))
        emit_call_ms_ema = float(pipeline_timing_health.get("emit_call_ms_ema", 0.0))
        timing_last_path = str(pipeline_timing_health.get("last_path", ""))

        output_buffer_health: dict[str, object] = {}
        if hasattr(self._controller, "decklink_output_buffer_health"):
            try:
                output_buffer_health = dict(self._controller.decklink_output_buffer_health())
            except Exception:
                output_buffer_health = {}

        starvation_events = int(output_buffer_health.get("starvation_events", 0))
        overflow_events = int(output_buffer_health.get("overflow_events", 0))
        reprime_events = int(output_buffer_health.get("auto_reprime_events", 0))
        buffered_count = int(output_buffer_health.get("last_buffered_count", -1))
        target_buffer = int(output_buffer_health.get("target_buffer_frames", int(self._decklink_output_buffer_frames)))
        last_reprime_reason = str(output_buffer_health.get("last_reprime_reason", ""))

        starvation_delta = max(0, starvation_events - self._health_last_buffer_starvation)
        overflow_delta = max(0, overflow_events - self._health_last_buffer_overflow)
        reprime_delta = max(0, reprime_events - self._health_last_buffer_reprime)

        self._health_last_buffer_starvation = starvation_events
        self._health_last_buffer_overflow = overflow_events
        self._health_last_buffer_reprime = reprime_events
        self._health_last_output_fps = float(output_fps)
        self._health_last_output_nominal_fps = float(nominal_fps)

        self._maybe_auto_stabilize_decklink_buffer(
            deadline_miss_ratio=deadline_miss_ratio,
            deadline_miss_streak=deadline_miss_streak,
            starvation_delta=starvation_delta,
            buffered_count=buffered_count,
            interaction_active=interaction_active,
        )

        health_level = "ok"
        health_reasons: list[str] = []

        fps_drop = output_ratio < 0.92
        severe_drop = output_ratio < 0.80
        if fps_drop and (not self._health_drop_active):
            self._health_drop_events_total += 1
            self._health_drop_active = True
            self._health_drop_active_interpolation = interaction_active
            if interaction_active:
                self._health_drop_events_interpolation += 1
        elif (not fps_drop) and self._health_drop_active:
            self._health_drop_active = False
            self._health_drop_active_interpolation = False

        if fps_drop:
            health_level = "warn"
            health_reasons.append(f"fps_drop={output_fps:.1f}/{nominal_fps:.1f}")

        if severe_drop:
            health_level = "critical"

        if deadline_miss_ratio >= 0.05 or deadline_miss_streak >= 3:
            if health_level == "ok":
                health_level = "warn"
            health_reasons.append(
                f"deadline_miss={deadline_miss_ratio * 100.0:.1f}% streak={deadline_miss_streak}"
            )

        if deadline_miss_ratio >= 0.15 or deadline_miss_streak >= 8:
            health_level = "critical"

        if starvation_delta > 0 or overflow_delta > 0 or reprime_delta > 0:
            self._health_buffer_warn_events += 1
            health_level = "critical" if starvation_delta > 0 else "warn"
            health_reasons.append(
                f"buffer_events(+s={starvation_delta},+o={overflow_delta},+r={reprime_delta})"
            )

        if buffered_count >= 0 and target_buffer > 0 and buffered_count < max(1, target_buffer - 1):
            if health_level == "ok":
                health_level = "warn"
            health_reasons.append(f"buffer_low={buffered_count}/{target_buffer}")

        if health_reasons:
            LOGGER.warning(
                (
                    "HEALTH | level=%s | output_fps=%.2f | nominal_fps=%.2f | ratio=%.3f | "
                    "interp_active=%s | reasons=%s | "
                    "timing[frames=%d,dl_miss_ratio=%.3f,dl_streak=%d,dl_max=%d,e2e_ema_ms=%.2f,proc_ema_ms=%.2f,cq_ema_ms=%.2f,oq_ema_ms=%.2f,ow_ema_ms=%.2f,emit_ema_ms=%.2f,late_ema_ms=%.2f,path=%s] | "
                    "buffer[target=%d,current=%d,s=%d,o=%d,r=%d,reason=%s]"
                ),
                health_level,
                float(output_fps),
                float(nominal_fps),
                float(output_ratio),
                "yes" if interaction_active else "no",
                ",".join(health_reasons),
                emitted_frames,
                deadline_miss_ratio,
                deadline_miss_streak,
                deadline_miss_max_streak,
                e2e_ms_ema,
                process_ms_ema,
                capture_queue_ms_ema,
                output_queue_ms_ema,
                output_wait_ms_ema,
                emit_call_ms_ema,
                deadline_late_ms_ema,
                timing_last_path,
                target_buffer,
                buffered_count,
                starvation_events,
                overflow_events,
                reprime_events,
                last_reprime_reason,
            )

        interp_tag = "interp" if interaction_active else "steady"
        if not health_reasons:
            return (
                f"health=ok ({interp_tag}) | out={output_fps:.1f}/{nominal_fps:.1f} | "
                f"buf={buffered_count}/{target_buffer}"
            )

        return (
            f"health={health_level} ({interp_tag}) | out={output_fps:.1f}/{nominal_fps:.1f} | "
            f"buf={buffered_count}/{target_buffer} | "
            f"dl_miss={deadline_miss_ratio * 100.0:.1f}% (streak={deadline_miss_streak}) | "
            f"e2e={e2e_ms_ema:.1f}ms proc={process_ms_ema:.1f}ms cq={capture_queue_ms_ema:.1f}ms oq={output_queue_ms_ema:.1f}ms | "
            f"{' ; '.join(health_reasons)}"
        )

    def _apply_performance_guard(self, measured_fps: float) -> None:
        if not self._perf_guard_enabled:
            return

        target_fps = self._target_fps()
        if target_fps <= 0:
            return

        low_threshold = target_fps * 0.80
        severe_threshold = target_fps * 0.65

        if measured_fps >= low_threshold:
            self._perf_guard_low_fps_seconds = 0
            return

        self._perf_guard_low_fps_seconds += 1
        if self._perf_guard_low_fps_seconds < 2:
            return

        # First mitigation: clamp basic-scaling cost by switching to manual x2.
        if self._controller.enable_basic_scaling and (
            self._controller.basic_scaling_auto_mode or self._controller.basic_scaling_manual > 2 or self._controller.effective_scale() > 2
        ):
            self._controller.set_manual_basic_scaling(2)
            self._updating_controls = True
            self.sr_mode_combo.setCurrentText("Manual")
            self.sr_manual_combo.setCurrentText("2")
            self._updating_controls = False
            self._perf_guard_last_action = "manual_x2"
            self._perf_guard_low_fps_seconds = 0
            LOGGER.warning(
                "PERF_GUARD | fps=%.1f target=%.1f | action=force_manual_sr_2",
                measured_fps,
                target_fps,
            )
            self._update_status("Performance guard: forced Manual basic scaling=2 to improve FPS")
            return

        # Second mitigation: disable basic scaling if still significantly below target.
        if (
            self._controller.enable_basic_scaling
            and self._controller.basic_scaling_manual == 2
            and measured_fps < severe_threshold
            and self._perf_guard_last_action != "disable_sr"
        ):
            self.enable_sr_checkbox.setChecked(False)
            self._perf_guard_last_action = "disable_sr"
            self._perf_guard_low_fps_seconds = 0
            LOGGER.warning(
                "PERF_GUARD | fps=%.1f target=%.1f | action=disable_basic_scaling",
                measured_fps,
                target_fps,
            )

    def _on_source_mode_changed(self) -> None:
        self._source_mode = self.source_mode_combo.currentText()
        self._sync_blackmagic_controls_enabled_state()
        self._update_timer_interval()
        self._sync_roi_transition_unit_labels()
        if self._source_mode == "Synthetic":
            self._stop_decklink_sessions()
            self._set_decklink_timecode_display(None, placeholder="Timecode: unavailable in Synthetic mode")
            self.decklink_status_label.setText("Synthetic mode active")
            self._update_fps_control_lock()
            return

        self._update_fps_control_lock()
        self._refresh_decklink_catalog()
        self._on_apply_decklink_settings()

    def _on_blackmagic_combo_changed(self) -> None:
        if self._updating_controls:
            return
        self._apply_mode_aware_deinterlace_default_if_needed()
        self._sync_roi_transition_unit_labels()
        self._sync_blackmagic_controls_enabled_state()
        self._update_fps_control_lock()
        self._update_status("DeckLink settings changed. Click Apply DeckLink Settings to apply.")

    def _sync_blackmagic_controls_enabled_state(self) -> None:
        blackmagic_selected = self.source_mode_combo.currentText() == "Blackmagic DeckLink"
        for widget in [
            self.decklink_input_device_combo,
            self.decklink_output_device_combo,
            self.decklink_auto_detect_devices,
            self.decklink_input_mode_combo,
            self.decklink_output_mode_combo,
            self.color_space_combo,
            self.color_range_combo,
            self.decklink_enable_format_detection,
            self.decklink_fps_priority_guard_checkbox,
            self.worker_priority_combo,
            self.decklink_apply_btn,
            self.decklink_refresh_btn,
        ]:
            widget.setEnabled(blackmagic_selected)

    def _update_fps_control_lock(self) -> None:
        blackmagic_selected = self.source_mode_combo.currentText() == "Blackmagic DeckLink"
        self.fps_spin.setEnabled(not blackmagic_selected)

    def _on_apply_decklink_settings(self) -> None:
        selected_source_mode = self.source_mode_combo.currentText()
        self._source_mode = selected_source_mode
        self._sync_blackmagic_controls_enabled_state()
        self._update_timer_interval()
        self._update_fps_control_lock()

        if self._source_mode != "Blackmagic DeckLink":
            self._stop_decklink_sessions()
            self._set_decklink_timecode_display(None, placeholder="Timecode: unavailable in Synthetic mode")
            self.decklink_status_label.setText("Synthetic mode active")
            self._update_status("Applied source mode: Synthetic")
            return

        try:
            self._apply_controller_color_settings_from_ui()
        except Exception as exc:
            self._update_status(f"DeckLink color settings apply failed: {exc}")
            return

        if d is None:
            self.decklink_status_label.setText("decklink_wrapper is not available in this environment")
            self._update_status("DeckLink unavailable: install or activate decklink_wrapper environment")
            return

        if self.decklink_input_device_combo.count() == 0 or self.decklink_output_device_combo.count() == 0:
            self._refresh_decklink_catalog()

        try:
            self._start_decklink_sessions()
        except Exception as exc:
            LOGGER.exception("DeckLink setup failed")
            self.decklink_status_label.setText(f"DeckLink setup failed: {exc}")
            self._update_status(f"DeckLink setup failed: {exc}")

    def _start_decklink_sessions(self) -> None:
        self._stop_decklink_sessions()

        if self.decklink_auto_detect_devices.isChecked():
            self._apply_auto_detect_device_selection()

        in_device = self._selected_combo_data(self.decklink_input_device_combo)
        out_device = self._selected_combo_data(self.decklink_output_device_combo)
        if in_device is None or out_device is None:
            # One retry after a full catalog refresh to recover from stale/placeholder
            # combo state when devices are present but selection data is null.
            self._refresh_decklink_catalog()
            if self.decklink_auto_detect_devices.isChecked():
                self._apply_auto_detect_device_selection()
            in_device = self._selected_combo_data(self.decklink_input_device_combo)
            out_device = self._selected_combo_data(self.decklink_output_device_combo)
        if in_device is None or out_device is None:
            raise RuntimeError("No compatible DeckLink input/output devices selected")

        in_mode = self._selected_combo_data(self.decklink_input_mode_combo)
        out_mode = self._selected_combo_data(self.decklink_output_mode_combo)
        if in_mode is None or out_mode is None:
            raise RuntimeError("No compatible DeckLink input/output modes selected")

        input_fps = self._resolve_mode_fps(in_device, in_mode, input_side=True)
        output_fps = self._resolve_mode_fps(out_device, out_mode, input_side=False)
        if hasattr(self._controller, "decklink_output_buffer_frames"):
            self._controller.decklink_output_buffer_frames = int(self.decklink_output_buffer_spin.value())
        if hasattr(self._controller, "worker_process_priority"):
            self._controller.worker_process_priority = _normalize_worker_priority_name(
                WORKER_PRIORITY_LABEL_TO_NAME.get(self.worker_priority_combo.currentText(), "above_normal")
            )

        if self._controller_backend == "worker-process":
            try:
                self._controller.start_decklink(
                    in_device=in_device,
                    in_mode=in_mode,
                    out_device=out_device,
                    out_mode=out_mode,
                    enable_format_detection=self.decklink_enable_format_detection.isChecked(),
                )
            except RuntimeError as exc:
                error_text = str(exc)
                recoverable_start_failure = (
                    "Worker process exited unexpectedly" in error_text
                    or "Timed out waiting for worker ack: start_decklink" in error_text
                )
                if not recoverable_start_failure:
                    raise
                LOGGER.warning("DeckLink start hit recoverable worker failure; recreating worker and retrying once: %s", error_text)
                self._recreate_worker_controller()
                self._controller.start_decklink(
                    in_device=in_device,
                    in_mode=in_mode,
                    out_device=out_device,
                    out_mode=out_mode,
                    enable_format_detection=self.decklink_enable_format_detection.isChecked(),
                )
            self._capture_session = None
            self._output_session = None
        else:
            self._capture_session = d.CaptureSession(
                device_index=in_device,
                display_mode=in_mode,
                pixel_format=d.PIXEL_FORMAT_8BIT_YUV,
                max_queue_frames=8,
                enable_format_detection=self.decklink_enable_format_detection.isChecked(),
            )

            self._output_session = d.OutputSession(
                device_index=out_device,
                display_mode=out_mode,
                pixel_format=d.PIXEL_FORMAT_8BIT_YUV,
            )

            self._capture_session.start()
            self._output_session.start()

        selected_fps = self._select_decklink_fps(input_fps, output_fps)
        if selected_fps is not None:
            self.fps_spin.setValue(int(round(selected_fps)))
            self._update_timer_interval()

        fps_text = "n/a"
        if input_fps is not None and output_fps is not None:
            fps_text = f"in={input_fps:.2f}, out={output_fps:.2f}, selected={selected_fps:.2f}" if selected_fps is not None else "n/a"
        elif selected_fps is not None:
            fps_text = f"selected={selected_fps:.2f}"

        input_name = f"device {in_device}"
        output_name = f"device {out_device}"
        in_mode_name = self.decklink_input_mode_combo.currentText()
        out_mode_name = self.decklink_output_mode_combo.currentText()
        input_label = self.decklink_input_device_combo.currentText()
        output_label = self.decklink_output_device_combo.currentText()
        if input_label:
            input_name = input_label
        if output_label:
            output_name = output_label

        backend_text = "worker process" if self._controller_backend == "worker-process" else "GUI process"
        self.decklink_status_label.setText(
            "DeckLink configured: "
            f"in={input_name} mode='{in_mode_name}' ({in_mode}); "
            f"out={output_name} mode='{out_mode_name}' ({out_mode}); "
            f"fps={fps_text}; backend={backend_text}"
        )
        self._set_decklink_timecode_display(None, placeholder="Timecode: waiting for DeckLink frames...")
        self._decklink_sessions_running = True
        self._sync_roi_transition_unit_labels()
        LOGGER.info(
            "DeckLink started: input=%s mode=%s output=%s mode=%s fps=%s",
            input_name,
            in_mode_name,
            output_name,
            out_mode_name,
            fps_text,
        )

    def _resolve_mode_fps(self, device_index: int, mode_value: object, input_side: bool) -> float | None:
        modes = (
            _call_decklink_api("list_input_display_modes", device_index)
            if input_side
            else _call_decklink_api("list_output_display_modes", device_index)
        )
        for mode in modes:
            if mode.mode != mode_value:
                continue
            frame_duration = float(getattr(mode, "frame_duration", 0))
            time_scale = float(getattr(mode, "time_scale", 0))
            if frame_duration <= 0 or time_scale <= 0:
                return None
            return time_scale / frame_duration
        return None

    def _select_decklink_fps(self, input_fps: float | None, output_fps: float | None) -> float | None:
        if input_fps is not None and output_fps is not None:
            return min(input_fps, output_fps)
        if input_fps is not None:
            return input_fps
        return output_fps

    def _refresh_decklink_catalog(self) -> None:
        if d is None:
            self.decklink_status_label.setText("decklink_wrapper is not available in this environment")
            LOGGER.error("DeckLink catalog refresh failed: wrapper unavailable")
            return

        try:
            devices = _call_decklink_api("list_devices")
        except Exception as exc:
            LOGGER.exception("DeckLink catalog refresh failed while listing devices")
            self.decklink_status_label.setText(f"DeckLink refresh failed: {exc}")
            self._update_status(f"DeckLink refresh failed: {exc}")
            self.decklink_input_device_combo.clear()
            self.decklink_output_device_combo.clear()
            self.decklink_input_mode_combo.clear()
            self.decklink_output_mode_combo.clear()
            self.decklink_input_device_combo.addItem("DeckLink refresh failed", None)
            self.decklink_output_device_combo.addItem("DeckLink refresh failed", None)
            return

        LOGGER.info("DeckLink refresh: detected %d device(s)", len(devices))

        self.decklink_input_device_combo.blockSignals(True)
        self.decklink_output_device_combo.blockSignals(True)
        self.decklink_input_device_combo.clear()
        self.decklink_output_device_combo.clear()

        input_count = 0
        output_count = 0
        for dev in devices:
            label = f"{dev.display_name} [{dev.model_name}] (index={dev.index})"
            if dev.supports_input:
                self.decklink_input_device_combo.addItem(label, int(dev.index))
                input_count += 1
            if dev.supports_output:
                self.decklink_output_device_combo.addItem(label, int(dev.index))
                output_count += 1

        self.decklink_input_device_combo.blockSignals(False)
        self.decklink_output_device_combo.blockSignals(False)

        if self.decklink_input_device_combo.count() == 0:
            self.decklink_input_device_combo.addItem("No input-capable devices", None)
        if self.decklink_output_device_combo.count() == 0:
            self.decklink_output_device_combo.addItem("No output-capable devices", None)

        LOGGER.info("DeckLink refresh: input devices=%d output devices=%d", input_count, output_count)

        if self.decklink_auto_detect_devices.isChecked():
            self._apply_auto_detect_device_selection()
        else:
            if self._pending_persisted_input_device is not None:
                for i in range(self.decklink_input_device_combo.count()):
                    if self.decklink_input_device_combo.itemData(i) == self._pending_persisted_input_device:
                        self.decklink_input_device_combo.setCurrentIndex(i)
                        break
            if self._pending_persisted_output_device is not None:
                for i in range(self.decklink_output_device_combo.count()):
                    if self.decklink_output_device_combo.itemData(i) == self._pending_persisted_output_device:
                        self.decklink_output_device_combo.setCurrentIndex(i)
                        break

        self._populate_mode_combos()

    def _apply_auto_detect_device_selection(self) -> None:
        def _select_first_valid(combo: QComboBox) -> None:
            for i in range(combo.count()):
                if combo.itemData(i) is not None:
                    combo.setCurrentIndex(i)
                    return
            if combo.count() > 0:
                combo.setCurrentIndex(0)

        _select_first_valid(self.decklink_input_device_combo)
        _select_first_valid(self.decklink_output_device_combo)

    def _on_auto_detect_toggled(self, checked: bool) -> None:
        if checked:
            self._apply_auto_detect_device_selection()
            self._populate_mode_combos()

    def _on_decklink_device_changed(self) -> None:
        self._populate_mode_combos()
        self._on_blackmagic_combo_changed()

    def _populate_mode_combos(self) -> None:
        if d is None:
            return

        in_device = self._selected_combo_data(self.decklink_input_device_combo)
        out_device = self._selected_combo_data(self.decklink_output_device_combo)

        self.decklink_input_mode_combo.clear()
        self.decklink_output_mode_combo.clear()

        if in_device is not None:
            try:
                input_modes = _call_decklink_api("list_input_display_modes", in_device)
            except Exception:
                LOGGER.exception("Failed listing input modes for device %s", in_device)
                input_modes = []
            for mode in input_modes:
                fps = self._fps_from_mode(mode)
                label = f"{mode.name} ({mode.width}x{mode.height}, {fps:.2f}fps)"
                self.decklink_input_mode_combo.addItem(label, mode.mode)

        if out_device is not None:
            try:
                output_modes = _call_decklink_api("list_output_display_modes", out_device)
            except Exception:
                LOGGER.exception("Failed listing output modes for device %s", out_device)
                output_modes = []
            for mode in output_modes:
                fps = self._fps_from_mode(mode)
                label = f"{mode.name} ({mode.width}x{mode.height}, {fps:.2f}fps)"
                self.decklink_output_mode_combo.addItem(label, mode.mode)

        self._select_default_mode(self.decklink_input_mode_combo, INPUT_MODE_QUERY_DEFAULT)
        self._select_default_mode(self.decklink_output_mode_combo, OUTPUT_MODE_QUERY_DEFAULT)

        if self._pending_persisted_input_mode_text:
            for i in range(self.decklink_input_mode_combo.count()):
                if self.decklink_input_mode_combo.itemText(i) == self._pending_persisted_input_mode_text:
                    self.decklink_input_mode_combo.setCurrentIndex(i)
                    break
        if self._pending_persisted_output_mode_text:
            for i in range(self.decklink_output_mode_combo.count()):
                if self.decklink_output_mode_combo.itemText(i) == self._pending_persisted_output_mode_text:
                    self.decklink_output_mode_combo.setCurrentIndex(i)
                    break

        self._apply_mode_aware_deinterlace_default_if_needed()
        self._sync_roi_transition_unit_labels()

        self._apply_mode_aware_deinterlace_default_if_needed()

    def _default_deinterlace_method_name_for_source_mode(self) -> str:
        if self.source_mode_combo.currentText() == "Blackmagic DeckLink":
            if _mode_name_is_interlaced(self.decklink_input_mode_combo.currentText()):
                return INTERLACED_DEFAULT_DEINTERLACE_METHOD
        return PROGRESSIVE_DEFAULT_DEINTERLACE_METHOD

    def _apply_mode_aware_deinterlace_default_if_needed(self) -> None:
        if self._has_persisted_deinterlace_method or self._deinterlace_method_user_selected:
            return

        method_name = self._default_deinterlace_method_name_for_source_mode()
        method_label = DEINTERLACE_METHOD_NAME_TO_LABEL.get(method_name)
        if not method_label:
            return

        if self.deinterlace_method_combo.currentText() != method_label:
            self._updating_controls = True
            try:
                self.deinterlace_method_combo.setCurrentText(method_label)
            finally:
                self._updating_controls = False

        try:
            self._controller.set_deinterlace_method(method_name)
        except Exception:
            LOGGER.exception("Failed to apply mode-aware deinterlace default")

    def _fps_from_mode(self, mode: object) -> float:
        frame_duration = float(getattr(mode, "frame_duration", 0))
        time_scale = float(getattr(mode, "time_scale", 0))
        if frame_duration <= 0 or time_scale <= 0:
            return 0.0
        return time_scale / frame_duration

    def _decklink_output_mode_is_interlaced(self) -> bool:
        if self._source_mode == "Blackmagic DeckLink" and hasattr(self._controller, "decklink_output_is_interlaced"):
            try:
                return bool(self._controller.decklink_output_is_interlaced())
            except Exception:
                pass
        return _mode_name_is_interlaced(self.decklink_output_mode_combo.currentText())

    def _decklink_output_field_interval_ms(self) -> int | None:
        field_rate_hz = self._decklink_output_effective_field_rate_fps()
        if field_rate_hz <= 1.0:
            return None
        return max(1, int(round(1000.0 / field_rate_hz)))

    def _decklink_output_effective_field_rate_fps(self) -> float:
        if self._source_mode != "Blackmagic DeckLink":
            return 0.0

        nominal_fps = 0.0
        if hasattr(self._controller, "decklink_output_nominal_fps"):
            try:
                nominal_fps = float(self._controller.decklink_output_nominal_fps())
            except Exception:
                nominal_fps = 0.0

        if nominal_fps <= 1.0:
            out_device = self._selected_combo_data(self.decklink_output_device_combo)
            out_mode = self._selected_combo_data(self.decklink_output_mode_combo)
            if out_device is not None and out_mode is not None:
                resolved_fps = self._resolve_mode_fps(int(out_device), out_mode, input_side=False)
                if resolved_fps is not None:
                    nominal_fps = float(resolved_fps)

        if nominal_fps <= 1.0:
            return 0.0

        if hasattr(self._controller, "decklink_transition_units_per_output_frame"):
            try:
                units_per_frame = float(self._controller.decklink_transition_units_per_output_frame())
            except Exception:
                units_per_frame = 1.0
            if units_per_frame > 0.1:
                return nominal_fps * units_per_frame

        if self._decklink_output_mode_is_interlaced() and nominal_fps < 45.0:
            return nominal_fps * 2.0
        return nominal_fps

    def _select_default_mode(self, combo: QComboBox, preferred_name: str) -> None:
        if combo.count() == 0:
            return
        for i in range(combo.count()):
            text = combo.itemText(i)
            if preferred_name.lower() in text.lower():
                combo.setCurrentIndex(i)
                return
        combo.setCurrentIndex(0)

    def _selected_combo_data(self, combo: QComboBox):
        current = combo.currentData()
        if current is not None:
            return current

        # Recover from stale placeholder selections by choosing the first
        # concrete device/mode entry when available.
        for i in range(combo.count()):
            candidate = combo.itemData(i)
            if candidate is not None:
                combo.setCurrentIndex(i)
                return candidate
        return None

    def _stop_decklink_sessions(self) -> None:
        if self._decklink_buffer_reapply_timer.isActive():
            self._decklink_buffer_reapply_timer.stop()
        if self._decklink_color_reapply_timer.isActive():
            self._decklink_color_reapply_timer.stop()
        if self._controller_backend == "worker-process":
            try:
                self._controller.stop_decklink()
            except Exception:
                pass

        if self._output_session is not None:
            try:
                self._output_session.stop()
            except Exception:
                pass
            clear_output_schedule_state(self._output_session)
            self._output_session = None

        if self._capture_session is not None:
            try:
                self._capture_session.stop()
            except Exception:
                pass
            self._capture_session = None

        self._decklink_sessions_running = False
        self._set_decklink_timecode_display(None, placeholder="Timecode: --")

        LOGGER.info("DeckLink sessions stopped")

    def _next_input_frame(self) -> bytes | None:
        if self._source_mode == "Synthetic":
            return self._source.next_frame()

        if self._capture_session is None:
            if self._last_frame_error != "DeckLink session not started":
                self._last_frame_error = "DeckLink session not started"
                self._update_status("DeckLink selected but session not started")
            return None

        frame = self._capture_session.acquire(timeout_ms=50)
        if frame is None:
            self._no_frame_counter += 1
            if self._no_frame_counter % 20 == 0:
                LOGGER.warning("No DeckLink input frames yet (count=%d)", self._no_frame_counter)
            self._set_decklink_timecode_display(None, placeholder="Timecode: waiting for DeckLink frames...")
            if self._last_frame_error != "No input signal frames received":
                self._last_frame_error = "No input signal frames received"
                self._update_status("DeckLink connected but no input frames yet; check source signal and input mode")
            return None
        self._set_decklink_timecode_display(_extract_decklink_frame_timecode_info(frame), placeholder="Timecode: none detected")
        frame_bytes = tight_uyvy_bytes(frame)

        self._no_frame_counter = 0
        self._last_frame_error = None
        return frame_bytes

    def _reset_roi(self) -> None:
        self._cancel_roi_keyframe_transition()
        self._roi = Roi(0, 0, FRAME_W, FRAME_H)
        self._input_canvas.set_roi(self._roi)
        self._apply_controller_roi_immediate(self._roi)
        self._sync_controls_from_roi(self._roi)

    def _serialize_roi_keyframe(self, keyframe: RoiKeyframe) -> dict[str, object]:
        return {
            "roi": [
                int(keyframe.roi.x),
                int(keyframe.roi.y),
                int(keyframe.roi.w),
                int(keyframe.roi.h),
            ],
            "duration_frames": int(keyframe.duration_frames),
            "interpolation_mode": str(keyframe.interpolation_mode),
        }

    def _roi_interp_mode_name(self) -> str:
        label = str(self.roi_interp_mode_combo.currentText()).strip().lower()
        if label == "ease in/out":
            return "ease_in_out"
        if label == "ease out":
            return "ease_out"
        return "linear"

    def _roi_interp_mode_label(self, mode_name: str) -> str:
        if str(mode_name).strip().lower() == "ease_in_out":
            return "Ease In/Out"
        if str(mode_name).strip().lower() == "ease_out":
            return "Ease Out"
        return "Linear"

    def _restore_roi_keyframes(self, raw: object) -> None:
        restored: dict[int, RoiKeyframe] = {}
        if not isinstance(raw, dict):
            self._roi_keyframes = restored
            return

        for slot in self._roi_keyframe_slots:
            slot_raw = raw.get(str(slot))
            if not isinstance(slot_raw, dict):
                continue
            roi_list = slot_raw.get("roi")
            if not isinstance(roi_list, list) or len(roi_list) != 4:
                continue
            try:
                roi = clamp_roi(
                    Roi(
                        int(roi_list[0]),
                        int(roi_list[1]),
                        int(roi_list[2]),
                        int(roi_list[3]),
                    )
                )
                duration = max(1, min(600, int(slot_raw.get("duration_frames", self.roi_transition_frames_spin.value()))))
                interp_mode = str(slot_raw.get("interpolation_mode", "linear")).strip().lower()
                if interp_mode not in {"linear", "ease_in_out", "ease_out"}:
                    interp_mode = "linear"
            except Exception:
                continue
            restored[slot] = RoiKeyframe(roi=roi, duration_frames=duration, interpolation_mode=interp_mode)

        self._roi_keyframes = restored

    def _on_roi_save_key_toggled(self, checked: bool) -> None:
        self._roi_key_save_armed = bool(checked)
        self._update_roi_key_buttons()

    def _all_roi_save_key_buttons(self) -> list[QPushButton]:
        return [self.roi_save_key_btn, *self._fullscreen_roi_save_key_buttons.values()]

    def _all_roi_slot_buttons(self, slot: int) -> list[QPushButton]:
        if slot == 1:
            return [
                self.roi_key1_btn,
                *[buttons[0] for buttons in self._fullscreen_roi_key_slot_buttons.values()],
            ]
        if slot == 2:
            return [
                self.roi_key2_btn,
                *[buttons[1] for buttons in self._fullscreen_roi_key_slot_buttons.values()],
            ]
        if slot == 3:
            return [
                self.roi_key3_btn,
                *[buttons[2] for buttons in self._fullscreen_roi_key_slot_buttons.values()],
            ]
        if slot == 4:
            return [
                self.roi_key4_btn,
                *[buttons[3] for buttons in self._fullscreen_roi_key_slot_buttons.values()],
            ]
        return []

    def _on_roi_key_slot_pressed(self, slot: int) -> None:
        if slot not in self._roi_keyframe_slots:
            return

        if self._roi_key_save_armed:
            duration_frames = max(1, min(600, int(self.roi_transition_frames_spin.value())))
            interpolation_mode = self._roi_interp_mode_name()
            self._roi_keyframes[slot] = RoiKeyframe(
                roi=clamp_roi(self._roi),
                duration_frames=duration_frames,
                interpolation_mode=interpolation_mode,
            )
            self.roi_save_key_btn.setChecked(False)
            self._schedule_settings_save()
            self._update_status(
                f"Stored ROI KEY {slot} ({duration_frames} frames, {self._roi_interp_mode_label(interpolation_mode)})"
            )
            return

        self._recall_roi_key_slot(slot)

    def _recall_roi_key_slot(self, slot: int) -> None:
        keyframe = self._roi_keyframes.get(slot)
        if keyframe is None:
            self._update_status(f"KEY {slot} is empty. Arm SAVE KEY to store it.")
            return

        override_duration = bool(self.roi_keyframe_duration_override_btn.isChecked())
        if override_duration:
            requested_duration_frames = max(1, min(600, int(self.roi_transition_frames_spin.value())))
        else:
            requested_duration_frames = max(1, min(600, int(keyframe.duration_frames)))

        duration_frames = self._effective_roi_keyframe_duration_frames(keyframe.roi, requested_duration_frames)
        self._start_roi_keyframe_transition(keyframe.roi, duration_frames, keyframe.interpolation_mode)

        if override_duration:
            self._update_status(
                f"Recalling KEY {slot} over {duration_frames} frames ({self._roi_interp_mode_label(keyframe.interpolation_mode)}, override)"
            )
        else:
            self._update_status(
                f"Recalling KEY {slot} over {duration_frames} frames ({self._roi_interp_mode_label(keyframe.interpolation_mode)})"
            )

    def _effective_roi_keyframe_duration_frames(self, target_roi: Roi, requested_frames: int) -> int:
        _ = target_roi
        requested = max(1, min(600, int(requested_frames)))
        # Keep explicit requested units; worker-side stepping handles
        # field-vs-frame progression.
        return requested

    def _update_roi_key_buttons(self) -> None:
        for save_button in self._all_roi_save_key_buttons():
            previous_block = save_button.blockSignals(True)
            save_button.setChecked(self._roi_key_save_armed)
            save_button.blockSignals(previous_block)

        if self._roi_key_save_armed:
            for save_button in self._all_roi_save_key_buttons():
                save_button.setStyleSheet("QPushButton { background-color: #f1c40f; font-weight: 700; }")
            for slot in self._roi_keyframe_slots:
                for key_button in self._all_roi_slot_buttons(slot):
                    key_button.setText(f"KEY {slot} (STORE)")
        else:
            for save_button in self._all_roi_save_key_buttons():
                save_button.setStyleSheet("")
            for slot in self._roi_keyframe_slots:
                for key_button in self._all_roi_slot_buttons(slot):
                    key_button.setText(f"KEY {slot}")

        for slot in self._roi_keyframe_slots:
            for button in self._all_roi_slot_buttons(slot):
                if slot in self._roi_keyframes:
                    button.setStyleSheet("QPushButton { font-weight: 600; }")
                    button.setToolTip(
                        (
                            f"Stored ({self._roi_keyframes[slot].duration_frames} frames, "
                            f"{self._roi_interp_mode_label(self._roi_keyframes[slot].interpolation_mode)}). Click to recall."
                        )
                    )
                else:
                    button.setStyleSheet("")
                    button.setToolTip("No keyframe stored. Arm SAVE KEY then click to store.")

    def _set_decklink_timecode_display(self, info: dict[str, object] | None, placeholder: str | None = None) -> None:
        display_text = placeholder or "Timecode: --"
        if isinstance(info, dict) and bool(info.get("present", False)):
            timecode_text = str(info.get("text", "")).strip()
            format_name = str(info.get("format_name", "")).strip()
            if timecode_text:
                display_text = f"Timecode: {timecode_text}"
                if format_name:
                    display_text += f" ({format_name})"

        self._decklink_timecode_display_text = display_text
        if hasattr(self, "decklink_timecode_label"):
            self.decklink_timecode_label.setText(display_text)

    def _update_decklink_timecode_from_controller(self, placeholder: str | None = None) -> None:
        info: dict[str, object] | None = None
        if hasattr(self._controller, "decklink_timecode_info"):
            try:
                info = dict(self._controller.decklink_timecode_info())
            except Exception:
                info = None
        self._set_decklink_timecode_display(info, placeholder=placeholder)

    def _on_decklink_timecode_refresh_clicked(self) -> None:
        if self._source_mode != "Blackmagic DeckLink":
            self._set_decklink_timecode_display(None, placeholder="Timecode: unavailable in Synthetic mode")
            return

        if self._controller_backend == "worker-process":
            if not self._decklink_sessions_running:
                self._set_decklink_timecode_display(None, placeholder="Timecode: DeckLink sessions not running")
                return
            try:
                self._controller.decklink_tick(timeout_ms=5)
            except Exception:
                LOGGER.exception("Manual timecode refresh failed during worker tick")
            self._update_decklink_timecode_from_controller(placeholder="Timecode: none detected")
            return

        if self._capture_session is None:
            self._set_decklink_timecode_display(None, placeholder="Timecode: DeckLink sessions not running")
            return

        try:
            frame = self._capture_session.acquire(timeout_ms=5)
        except Exception:
            LOGGER.exception("Manual timecode refresh failed during capture poll")
            frame = None

        if frame is None:
            self._set_decklink_timecode_display(None, placeholder="Timecode: waiting for DeckLink frames...")
            return

        self._set_decklink_timecode_display(
            _extract_decklink_frame_timecode_info(frame),
            placeholder="Timecode: none detected",
        )

    def _cancel_roi_keyframe_transition(self, reset_subpixel_shift: bool = True, notify_backend: bool = True) -> None:
        previous_state = self._roi_keyframe_transition
        if isinstance(previous_state, dict):
            current_estimate = previous_state.get("current_roi_estimate")
            if isinstance(current_estimate, Roi):
                self._controller_roi_applied = clamp_roi(current_estimate)
        self._roi_keyframe_transition = None
        self._roi_keyframe_transition_timer.stop()
        self._roi_keyframe_last_step_ts = 0.0
        self._input_canvas.clear_visual_roi_overlay()
        self._controller_roi_target = None
        self._controller_filtered_target_roi = None
        self._controller_roi_interp_timer.stop()
        self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        if notify_backend and hasattr(self._controller, "cancel_roi_microstep_transition"):
            try:
                self._controller.cancel_roi_microstep_transition(reset_subpixel_shift=reset_subpixel_shift)
            except Exception:
                pass
        if reset_subpixel_shift and hasattr(self._controller, "set_roi_subpixel_shift"):
            self._controller.set_roi_subpixel_shift(0.0, 0.0)
        self._update_timer_interval()

    def _start_roi_keyframe_transition(self, target_roi: Roi, duration_frames: int, interpolation_mode: str) -> None:
        previous_state = self._roi_keyframe_transition
        if isinstance(previous_state, dict):
            current_estimate = previous_state.get("current_roi_estimate")
            if isinstance(current_estimate, Roi):
                self._roi = clamp_roi(current_estimate)

        if hasattr(self._input_canvas, "cancel_pending_interaction_updates"):
            self._input_canvas.cancel_pending_interaction_updates()

        # Clear manual ROI streaming state so stale move/resize commands do not
        # cancel the worker-side keyframe transition right after recall starts.
        self._manual_roi_send_timer.stop()
        self._manual_live_target_roi = None
        self._pending_manual_controller_roi = None

        target = clamp_roi(target_roi)
        total_frames = max(1, min(600, int(duration_frames)))
        mode_name = str(interpolation_mode).strip().lower()
        if mode_name not in {"linear", "ease_in_out", "ease_out"}:
            mode_name = "linear"
        # Retarget without sending a standalone cancel command first. In worker
        # mode, start_roi_microstep_transition can take over in-place from the
        # current rendered ROI+shift, avoiding a one-frame jump between moves.
        self._cancel_roi_keyframe_transition(reset_subpixel_shift=False, notify_backend=False)

        if total_frames <= 1:
            self._roi = target
            self._input_canvas.set_roi(target)
            self._input_canvas.clear_visual_roi_overlay()
            self._apply_controller_roi_immediate(target)
            self._sync_controls_from_roi(target)
            return

        self._roi_keyframe_transition = {
            "start": clamp_roi(self._roi),
            "target": target,
            "total_frames": total_frames,
            "frame_progress": 0.0,
            "interpolation_mode": mode_name,
            "quant_residual": {"x": 0.0, "y": 0.0, "w": 0.0},
            "last_roi": clamp_roi(self._roi),
            "last_subpixel_shift": {"x": 0.0, "y": 0.0},
            "pending_frame_advance": 0.0,
            "worker_transition_seen": False,
        }

        backend_driven = bool(
            self._source_mode == "Blackmagic DeckLink"
            and self._controller_backend == "worker-process"
            and hasattr(self._controller, "start_roi_microstep_transition")
        )
        self._roi_keyframe_transition["backend_driven"] = backend_driven

        # Ensure no background controller interpolation remains active while
        # keyframe transition drives ROI updates directly.
        self._controller_roi_target = None
        self._controller_filtered_target_roi = None
        self._controller_roi_interp_timer.stop()
        self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        self._controller_roi_applied = clamp_roi(self._roi)

        use_worker_clock = bool(
            self._source_mode == "Blackmagic DeckLink"
            and self._controller_backend == "worker-process"
            and hasattr(self._controller, "decklink_processed_counter")
            and hasattr(self._controller, "decklink_tick")
        )
        self._roi_keyframe_transition["use_worker_clock"] = use_worker_clock
        if use_worker_clock:
            try:
                self._roi_keyframe_transition["last_frame_counter"] = int(self._controller.decklink_processed_counter())
            except Exception:
                self._roi_keyframe_transition["last_frame_counter"] = None
        else:
            self._roi_keyframe_transition["last_frame_counter"] = None

        self._roi_keyframe_last_step_ts = time.perf_counter()

        # Match transition polling cadence to target interpolation FPS to reduce
        # control-loop pressure and queue churn during animated recalls.
        target_tick_interval_ms = int(round(1000.0 / max(1.0, self._roi_keyframe_transition_fps())))
        self._roi_keyframe_transition_timer.setInterval(max(8, min(100, target_tick_interval_ms)))

        if backend_driven:
            try:
                self._controller.start_roi_microstep_transition(
                    start_roi=clamp_roi(self._roi),
                    target_roi=target,
                    duration_frames=total_frames,
                    interpolation_mode=mode_name,
                    overscan_percent=float(self._roi_keyframe_transition_overscan_percent),
                    start_from_current=True,
                    enforce_full_frame_scale_1x=(
                        int(target.x) == 0
                        and int(target.y) == 0
                        and int(target.w) == FRAME_W
                        and int(target.h) == FRAME_H
                        and not (
                            int(self._roi.x) == 0
                            and int(self._roi.y) == 0
                            and int(self._roi.w) == FRAME_W
                            and int(self._roi.h) == FRAME_H
                        )
                    ),
                )
            except Exception as exc:
                self._update_status(f"Worker ROI microstep transition start failed: {exc}")
                self._roi_keyframe_transition["backend_driven"] = False
        self._roi_keyframe_transition_timer.start()
        self._update_timer_interval()

    def _apply_roi_interpolation_curve(self, t: float, interpolation_mode: str) -> float:
        clamped_t = max(0.0, min(1.0, float(t)))
        if str(interpolation_mode).strip().lower() == "ease_in_out":
            # Smoothstep for gentle acceleration/deceleration.
            return clamped_t * clamped_t * (3.0 - (2.0 * clamped_t))
        if str(interpolation_mode).strip().lower() == "ease_out":
            return 1.0 - ((1.0 - clamped_t) * (1.0 - clamped_t))
        return clamped_t

    def _roi_keyframe_transition_fps(self) -> float:
        # Clock transition ticks to effective DeckLink cadence. Interlaced
        # modes resolve to field rate when nominal reporting is frame-based.
        decklink_rate = self._decklink_output_effective_field_rate_fps()
        if decklink_rate > 1.0:
            return decklink_rate
        return float(self._roi_keyframe_target_fps)

    def _step_roi_keyframe_transition(self) -> None:
        state = self._roi_keyframe_transition
        if state is None:
            return

        backend_driven = bool(state.get("backend_driven", False))

        start_roi = state["start"]
        target_roi = state["target"]
        total_frames = int(state["total_frames"])
        interpolation_mode = str(state.get("interpolation_mode", "linear"))

        now = time.perf_counter()
        if self._roi_keyframe_last_step_ts <= 0.0:
            self._roi_keyframe_last_step_ts = now

        dt = max(0.0, now - self._roi_keyframe_last_step_ts)
        self._roi_keyframe_last_step_ts = now

        frame_progress = float(state.get("frame_progress", 0.0))
        frame_advance = 0.0

        if bool(state.get("use_worker_clock", False)):
            current_counter = None
            try:
                current_counter = int(self._controller.decklink_processed_counter())
            except Exception:
                current_counter = None

            last_counter = state.get("last_frame_counter")
            if isinstance(current_counter, int):
                state["last_frame_counter"] = current_counter
                if isinstance(last_counter, int) and current_counter >= last_counter:
                    state["pending_frame_advance"] = float(state.get("pending_frame_advance", 0.0)) + float(current_counter - last_counter)

            pending = max(0.0, float(state.get("pending_frame_advance", 0.0)))
            if pending > 0.0:
                # Consume at most one frame-worth per transition tick to avoid
                # visible jumps when GUI polling misses one or more frame-count updates.
                frame_advance = min(1.0, pending)
                state["pending_frame_advance"] = pending - frame_advance

            # Worker-clock mode should only step when a new processed frame has
            # landed. Skipping no-op ticks avoids repeated interpolation math and
            # redundant control sends between frame arrivals.
            if frame_advance <= 0.0 and (not backend_driven):
                return

        if frame_advance <= 0.0:
            # Keep GUI interpolation alive on every timer tick. In backend-driven
            # mode, worker snapshots reconcile phase but should not freeze motion.
            frame_advance = dt * self._roi_keyframe_transition_fps()

        frame_progress += frame_advance
        frame_progress = min(float(total_frames), frame_progress)
        state["frame_progress"] = frame_progress
        is_final_frame = frame_progress >= float(total_frames)

        t = min(1.0, frame_progress / float(max(1, total_frames)))
        curved_t = self._apply_roi_interpolation_curve(t, interpolation_mode)

        # Subpixel interpolation in center/width space reduces coupled x/y/w/h
        # quantization jitter, especially at high zoom where ROI dimensions are small.
        start_cx = float(start_roi.x) + (float(start_roi.w) * 0.5)
        start_cy = float(start_roi.y) + (float(start_roi.h) * 0.5)
        target_cx = float(target_roi.x) + (float(target_roi.w) * 0.5)
        target_cy = float(target_roi.y) + (float(target_roi.h) * 0.5)

        ideal_cx = start_cx + ((target_cx - start_cx) * curved_t)
        ideal_cy = start_cy + ((target_cy - start_cy) * curved_t)
        ideal_w = float(start_roi.w) + ((float(target_roi.w) - float(start_roi.w)) * curved_t)

        if backend_driven:
            worker_transition_state = self._sync_backend_roi_from_worker()

            worker_transition_active = False
            worker_transition_seen = bool(state.get("worker_transition_seen", False))
            if worker_transition_state:
                worker_transition_active = bool(worker_transition_state.get("active", False))
                worker_progress = float(worker_transition_state.get("frame_progress", frame_progress))
                worker_total_frames = max(
                    1,
                    int(worker_transition_state.get("total_frames", total_frames)),
                )
                # Reconcile local GUI phase toward worker phase gradually to avoid
                # abrupt jumps when worker telemetry arrives in coarse intervals.
                if worker_progress > frame_progress:
                    max_catch_up = max(1.0, dt * self._roi_keyframe_transition_fps() * 1.5)
                    frame_progress = min(float(worker_total_frames), frame_progress + min(worker_progress - frame_progress, max_catch_up))
                else:
                    frame_progress = min(float(worker_total_frames), frame_progress)
                state["frame_progress"] = frame_progress
                is_final_frame = frame_progress >= float(worker_total_frames)
            else:
                # Fallback for older worker payloads: preserve smooth local
                # interpolation at GUI preview cadence instead of snapping.
                worker_transition_active = bool(not is_final_frame)

            if not worker_transition_seen:
                display_w = max(2.0, float(ideal_w))
                display_h = max(2.0, float(display_w * 9.0 / 16.0))
                display_x = float(ideal_cx - (display_w * 0.5))
                display_y = float(ideal_cy - (display_h * 0.5))
                self._input_canvas.set_visual_roi_overlay(display_x, display_y, display_w, display_h)

            transition_complete = (not worker_transition_active) or is_final_frame
            if transition_complete:
                self._roi_keyframe_transition = None
                self._roi_keyframe_transition_timer.stop()
                self._input_canvas.clear_visual_roi_overlay()
                final_roi = state.get("current_roi_estimate", self._roi)
                if isinstance(final_roi, Roi):
                    final_roi = clamp_roi(final_roi)
                if not isinstance(final_roi, Roi):
                    final_roi = target_roi
                if not self._is_controller_roi_close(final_roi, target_roi):
                    # Worker transition state may be unavailable (for example,
                    # stale worker build); fall back to requested target.
                    final_roi = target_roi

                self._roi = clamp_roi(final_roi)
                self._input_canvas.set_roi(self._roi)
                self._controller_roi_applied = self._roi
                self._controller_roi_target = None
                self._controller_filtered_target_roi = None
                self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}

                self._sync_controls_from_roi(self._roi)
                self._roi_keyframe_last_step_ts = 0.0
                self._update_timer_interval()
            return

        residual = state.get("quant_residual")
        if not isinstance(residual, dict):
            residual = {"x": 0.0, "y": 0.0, "w": 0.0}
            state["quant_residual"] = residual

        desired_cx = ideal_cx + float(residual.get("x", 0.0))
        desired_cy = ideal_cy + float(residual.get("y", 0.0))
        desired_w = ideal_w + float(residual.get("w", 0.0))

        display_w = max(2.0, float(desired_w))
        display_h = max(2.0, float(display_w * 9.0 / 16.0))
        display_x = float(desired_cx - (display_w * 0.5))
        display_y = float(desired_cy - (display_h * 0.5))
        self._input_canvas.set_visual_roi_overlay(display_x, display_y, display_w, display_h)

        # Apply slight temporary overscan in the backend ROI path to make high-zoom
        # motion feel less quantized while keeping final framing exact.
        target_scale = roi_scale_from_roi(target_roi)
        overscan_pct = float(self._roi_keyframe_transition_overscan_percent)
        if target_scale >= 4.0 and overscan_pct > 0.0:
            # Bell envelope: 0 at start/end, highest mid-transition.
            # This avoids a first-frame zoom jolt while still reducing quantization in the middle.
            overscan_weight = max(0.0, 4.0 * float(curved_t) * (1.0 - float(curved_t)))
            desired_w_backend = desired_w * (1.0 + ((overscan_pct / 100.0) * overscan_weight))
        else:
            desired_w_backend = desired_w

        dx = int(target_roi.x) - int(start_roi.x)
        dy = int(target_roi.y) - int(start_roi.y)
        dw = int(target_roi.w) - int(start_roi.w)

        def _quantize_directional(value: float, delta: int, quantum: int) -> int:
            q = max(1, int(quantum))
            scaled = value / float(q)
            if delta > 0:
                return int(math.floor(scaled)) * q
            if delta < 0:
                return int(math.ceil(scaled)) * q
            return int(round(scaled)) * q

        quant_w = _quantize_directional(desired_w_backend, dw, 2)
        quant_w = max(2, quant_w & ~1)
        quant_h = max(2, int(round(quant_w * 9.0 / 16.0)))

        desired_x = desired_cx - (quant_w * 0.5)
        desired_y = desired_cy - (quant_h * 0.5)

        quant_x = _quantize_directional(desired_x, dx, 2)
        quant_y = _quantize_directional(desired_y, dy, 1)

        interpolated = clamp_roi(
            Roi(
                quant_x,
                quant_y,
                quant_w,
                quant_h,
            )
        )

        last_roi = state.get("last_roi")
        if not isinstance(last_roi, Roi):
            last_roi = self._roi

        mono_x = interpolated.x
        mono_y = interpolated.y
        mono_w = interpolated.w
        mono_h = interpolated.h

        if dx > 0:
            mono_x = max(mono_x, last_roi.x)
        elif dx < 0:
            mono_x = min(mono_x, last_roi.x)

        if dy > 0:
            mono_y = max(mono_y, last_roi.y)
        elif dy < 0:
            mono_y = min(mono_y, last_roi.y)

        if dw > 0:
            mono_w = max(mono_w, last_roi.w)
        elif dw < 0:
            mono_w = min(mono_w, last_roi.w)

        target_h_delta = int(target_roi.h) - int(start_roi.h)
        if target_h_delta > 0:
            mono_h = max(mono_h, last_roi.h)
        elif target_h_delta < 0:
            mono_h = min(mono_h, last_roi.h)

        interpolated = clamp_roi(Roi(mono_x, mono_y, mono_w, mono_h))
        state["last_roi"] = interpolated

        # Compute residual/compensation from the final carrier ROI that will be
        # sent to backend. Doing this before monotonic/clamp introduces mismatch
        # and visible staircase artifacts at very slow transitions.
        interp_cx = float(interpolated.x) + (float(interpolated.w) * 0.5)
        interp_cy = float(interpolated.y) + (float(interpolated.h) * 0.5)
        residual["x"] = desired_cx - interp_cx
        residual["y"] = desired_cy - interp_cy
        residual["w"] = desired_w_backend - float(interpolated.w)

        if hasattr(self._controller, "set_roi_subpixel_shift"):
            source_dx = ideal_cx - interp_cx
            source_dy = ideal_cy - interp_cy
            sx = FRAME_W / max(1.0, float(interpolated.w))
            sy = FRAME_H / max(1.0, float(interpolated.h))
            # ROI moving right shifts scene content left in output.
            max_shift_x = max(2.0, min(48.0, sx * 1.5))
            max_shift_y = max(2.0, min(48.0, sy * 1.5))
            target_shift_x = max(-max_shift_x, min(max_shift_x, -(source_dx * sx)))
            target_shift_y = max(-max_shift_y, min(max_shift_y, -(source_dy * sy)))
        else:
            target_shift_x = 0.0
            target_shift_y = 0.0

        if is_final_frame:
            # Avoid a terminal-frame hard snap. Keep interpolated carrier/shift
            # and let completion criteria finalize with a bounded tolerance.
            residual["x"] = desired_cx - interp_cx
            residual["y"] = desired_cy - interp_cy
            residual["w"] = desired_w_backend - float(interpolated.w)

        roi_changed = (
            interpolated.x != self._roi.x
            or interpolated.y != self._roi.y
            or interpolated.w != self._roi.w
            or interpolated.h != self._roi.h
        )

        transition_complete = (
            interpolated.x == target_roi.x
            and interpolated.y == target_roi.y
            and interpolated.w == target_roi.w
            and interpolated.h == target_roi.h
        ) or (
            is_final_frame
            and abs(interpolated.x - target_roi.x) <= 2
            and abs(interpolated.y - target_roi.y) <= 1
            and abs(interpolated.w - target_roi.w) <= 2
            and abs(interpolated.h - target_roi.h) <= 2
        )

        if roi_changed:
            self._roi = interpolated
            self._input_canvas.set_roi(interpolated)

        if not backend_driven:
            last_shift_state = state.get("last_subpixel_shift")
            if not isinstance(last_shift_state, dict):
                last_shift_state = {"x": 0.0, "y": 0.0}
                state["last_subpixel_shift"] = last_shift_state
            last_shift_x = float(last_shift_state.get("x", 0.0))
            last_shift_y = float(last_shift_state.get("y", 0.0))
            shift_changed = (
                abs(target_shift_x - last_shift_x) > 0.02
                or abs(target_shift_y - last_shift_y) > 0.02
            )

            if hasattr(self._controller, "set_roi_with_subpixel"):
                if roi_changed or shift_changed or transition_complete:
                    self._controller.set_roi_with_subpixel(interpolated, target_shift_x, target_shift_y)
                    self._controller_roi_applied = interpolated
                    state["last_subpixel_shift"] = {
                        "x": float(target_shift_x),
                        "y": float(target_shift_y),
                    }
            elif hasattr(self._controller, "set_roi_subpixel_shift"):
                if shift_changed or transition_complete:
                    self._controller.set_roi_subpixel_shift(target_shift_x, target_shift_y)
                    state["last_subpixel_shift"] = {
                        "x": float(target_shift_x),
                        "y": float(target_shift_y),
                    }
                if roi_changed:
                    self._apply_controller_roi_immediate(interpolated, reset_subpixel_shift=False)
            elif roi_changed:
                self._apply_controller_roi_immediate(interpolated, reset_subpixel_shift=False)

        if transition_complete:
            self._roi_keyframe_transition = None
            self._roi_keyframe_transition_timer.stop()
            self._input_canvas.clear_visual_roi_overlay()
            if (not backend_driven) and hasattr(self._controller, "set_roi_subpixel_shift"):
                self._controller.set_roi_subpixel_shift(0.0, 0.0)
            if (
                interpolated.x != target_roi.x
                or interpolated.y != target_roi.y
                or interpolated.w != target_roi.w
                or interpolated.h != target_roi.h
            ):
                if not backend_driven:
                    self._roi = target_roi
                    self._input_canvas.set_roi(target_roi)
            # Always finalize backend ROI and control values at transition end.
            if not backend_driven:
                self._apply_controller_roi_immediate(self._roi)

            self._sync_controls_from_roi(self._roi)
            self._roi_keyframe_last_step_ts = 0.0

    def _sync_controls_from_roi(self, roi: Roi) -> None:
        self._updating_controls = True
        if self.roi_x_spin.value() != roi.x:
            self.roi_x_spin.setValue(roi.x)
        if self.roi_y_spin.value() != roi.y:
            self.roi_y_spin.setValue(roi.y)
        if self.roi_w_spin.value() != roi.w:
            self.roi_w_spin.setValue(roi.w)
        if self.roi_h_spin.value() != roi.h:
            self.roi_h_spin.setValue(roi.h)
        target_scale = roi_scale_from_roi(roi)
        if abs(float(self.scale_spin.value()) - float(target_scale)) > 1e-6:
            self.scale_spin.setValue(target_scale)
        self._updating_controls = False

    def _update_status(self, text: str, suppress_repeat_window_s: float | None = None) -> None:
        now = time.perf_counter()
        window_s = self._status_repeat_log_interval_s if suppress_repeat_window_s is None else max(0.0, float(suppress_repeat_window_s))
        unchanged = text == self._last_status_text
        within_window = (now - self._last_status_log_ts) < window_s

        if unchanged and within_window:
            return

        self.status_label.setText(text)
        LOGGER.info("STATUS: %s", text)
        self._last_status_text = text
        self._last_status_log_ts = now


def load_video_processor_module():
    project_root = Path(__file__).resolve().parents[1]

    venv_site = project_root / "venv" / "Lib" / "site-packages"
    if venv_site.exists():
        site.addsitedir(str(venv_site))

    # Keep Release highest priority and Debug last to avoid accidental slow debug imports.
    preferred_paths = [
        project_root / "build" / "src" / "Release",
        project_root / "build" / "src" / "RelWithDebInfo",
        project_root / "build" / "src" / "Debug",
    ]
    path_prefixes = [
        project_root / "venv" / "Scripts",
        *preferred_paths,
    ]
    existing_path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    existing_path_keys = {part.lower() for part in existing_path_parts}
    prepended_paths: list[str] = []
    for candidate in path_prefixes:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str.lower() not in existing_path_keys:
            prepended_paths.append(candidate_str)
            existing_path_keys.add(candidate_str.lower())
    if prepended_paths:
        os.environ["PATH"] = os.pathsep.join(prepended_paths + existing_path_parts)

    for candidate in reversed(preferred_paths):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    import video_processor
    LOGGER.info("Loaded video_processor from %s", getattr(video_processor, "__file__", "<unknown>"))

    return video_processor


def main() -> int:
    app = QApplication(sys.argv)
    initialize_com_for_decklink()

    try:
        module = load_video_processor_module()
    except Exception as exc:
        print(f"Failed to import video_processor module: {exc}")
        return 1

    window = MainWindow(module)
    screen = app.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        target_w = min(int(available.width()), max(900, int(available.width() * 0.92)))
        target_h = min(int(available.height()), max(520, int(available.height() * 0.92)))
        window.resize(target_w, target_h)
        window.setMaximumSize(available.size())
    else:
        window.resize(1400, 860)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
