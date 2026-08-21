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
}
DEINTERLACE_METHOD_NAME_TO_LABEL = {value: key for key, value in DEINTERLACE_METHOD_LABEL_TO_NAME.items()}

DENOISE_METHOD_LABEL_TO_NAME = {
    "Off": "off",
    "Luma Gaussian 3x3 (Balanced)": "luma_gaussian3x3",
    "Luma Median 3x3 (Stronger)": "luma_median3x3",
    "Luma Bilateral 3x3 (Artifact Cleaner)": "luma_bilateral3x3",
    "Luma Bilateral 5x5 (Still Image Heavy)": "luma_bilateral5x5",
    "Field Temporal Luma (Advanced)": "field_temporal_luma",
}
DENOISE_METHOD_NAME_TO_LABEL = {value: key for key, value in DENOISE_METHOD_LABEL_TO_NAME.items()}

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
        self._visual_roi_overlay: tuple[float, float, float, float] | None = None

        self._drag_mode = "none"
        self._drag_start_pos = QPointF()
        self._drag_start_roi = self._roi

        self._last_touch_emit_ts = 0.0
        self._touch_emit_interval_s = 1.0 / 45.0
        self._touch_emit_pending = False
        self._touch_emit_pending_scale = False
        self._smoothing_percent = 60
        self._latency_smoothing_percent = 0
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
        self._visual_roi_overlay = None
        self._apply_roi_local(roi)

    def set_visual_roi_overlay(self, x: float, y: float, w: float, h: float) -> None:
        self._visual_roi_overlay = (float(x), float(y), float(w), float(h))
        self.update()

    def clear_visual_roi_overlay(self) -> None:
        if self._visual_roi_overlay is None:
            return
        self._visual_roi_overlay = None
        self.update()

    def roi(self) -> Roi:
        return self._roi

    def set_smoothing_percent(self, value: int) -> None:
        self._smoothing_percent = max(0, min(100, int(value)))

    def set_latency_smoothing_percent(self, value: int) -> None:
        self._latency_smoothing_percent = max(0, min(100, int(value)))

    def paintEvent(self, event) -> None:
        del event
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)

        image_rect = self._image_rect()
        if self._image is not None:
            p.drawImage(image_rect, self._image)

        if self._visual_roi_overlay is not None:
            overlay_x, overlay_y, overlay_w, overlay_h = self._visual_roi_overlay
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

        # Keep the resize handle fully inside the ROI and make it easier to grab.
        handle_size = 48
        p.fillRect(
            QRectF(
                roi_rect_w.right() - handle_size,
                roi_rect_w.bottom() - handle_size,
                handle_size,
                handle_size,
            ),
            Qt.yellow,
        )

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
        handle_size = 48
        handle_rect = QRectF(
            roi_rect.right() - handle_size,
            roi_rect.bottom() - handle_size,
            handle_size,
            handle_size,
        )

        if handle_rect.contains(event.position()):
            self._drag_mode = "resize"
        elif roi_rect.contains(event.position()):
            self._drag_mode = "move"
        else:
            self._drag_mode = "none"

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
            new_roi = Roi(
                self._drag_start_roi.x + int(round(dx * sx)),
                self._drag_start_roi.y + int(round(dy * sy)),
                self._drag_start_roi.w,
                self._drag_start_roi.h,
            )
        else:
            dw_x = int(round(dx * sx))
            dw_y = int(round(dy * sy * (16.0 / 9.0)))
            dw = dw_x if abs(dw_x) >= abs(dw_y) else dw_y
            # Resize around center so the whole ROI scales symmetrically.
            new_w = self._drag_start_roi.w + (2 * dw)
            new_h = int(round(new_w * 9.0 / 16.0))
            center_x = self._drag_start_roi.x + (self._drag_start_roi.w / 2.0)
            center_y = self._drag_start_roi.y + (self._drag_start_roi.h / 2.0)
            new_roi = Roi(
                int(round(center_x - (new_w / 2.0))),
                int(round(center_y - (new_h / 2.0))),
                new_w,
                new_h,
            )

        target_roi = clamp_roi(new_roi)
        emit_scale = self._drag_mode != "move"
        self._queue_interpolated_roi(target_roi, emit_scale=emit_scale, anchor_to_current=True)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        del event
        self._drag_mode = "none"
        self._flush_interaction_emit()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.fullscreenRequested.emit(self._view_name)
            event.accept()
            return
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

    def _queue_interpolated_roi(self, target_roi: Roi, emit_scale: bool = True, anchor_to_current: bool = False) -> None:
        raw_target = clamp_roi(target_roi)
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
        smoothing = max(0.0, min(1.0, self._smoothing_percent / 100.0))
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
        smoothing = max(0.0, min(1.0, self._smoothing_percent / 100.0))
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
        if event.button() == Qt.LeftButton:
            self.fullscreenRequested.emit(self._view_name)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class VideoProcessorController:
    def __init__(self, module) -> None:
        self._module = module
        self.enable_basic_scaling = True
        self.deinterlace_enabled = True
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
        self.ai_sr_max_inflight = 2
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

    def set_preview_fps(self, preview_fps: float) -> None:
        # In-process backend does not use worker tick preview throttling.
        _ = preview_fps

    def set_decklink_output_buffer_frames(self, buffer_frames: int) -> None:
        # In-process backend does not use worker DeckLink output buffering.
        self.decklink_output_buffer_frames = max(0, min(10, int(buffer_frames)))

    def set_roi_subpixel_shift(self, shift_x: float, shift_y: float) -> None:
        _ = (shift_x, shift_y)

    def set_roi_with_subpixel(self, roi: Roi, shift_x: float, shift_y: float) -> None:
        _ = (roi, shift_x, shift_y)

    def start_roi_microstep_transition(
        self,
        start_roi: Roi,
        target_roi: Roi,
        duration_frames: int,
        interpolation_mode: str,
        overscan_percent: float,
        start_from_current: bool = False,
    ) -> None:
        _ = (start_roi, target_roi, duration_frames, interpolation_mode, overscan_percent, start_from_current)

    def cancel_roi_microstep_transition(self, reset_subpixel_shift: bool = True) -> None:
        _ = reset_subpixel_shift
        return


class ProcessVideoProcessorController:
    def __init__(self) -> None:
        self.enable_basic_scaling = True
        self.deinterlace_enabled = True
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
        self.ai_sr_hold_last_frame = os.environ.get("VP_AI_SR_HOLD_LAST_FRAME", "1") == "1"
        self.ai_sr_max_hold_ms = max(0.0, float(os.environ.get("VP_AI_SR_MAX_HOLD_MS", "0")))
        self.ai_sr_max_inflight = max(1, min(4, int(os.environ.get("VP_AI_SR_MAX_INFLIGHT", "2"))))
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
        self.rtx_vsr_active = False
        self.rtx_vsr_error: str | None = None
        self.rtx_vsr_info: dict[str, object] | None = None

        self._ctx = mp.get_context("spawn")
        self._request_queue = None
        self._response_queue = None
        self._process = None

        self._next_frame_id = 1
        self._latest_output_frame: bytes | None = None
        self._latest_decklink_frame: tuple[bytes, bytes] | None = None
        self._decklink_frame_updated = False
        self._latest_effective_scale = 1
        self._decklink_no_frame_reason: str | None = None
        self._decklink_processed_counter = 0
        self._decklink_processed_fps = 0.0
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
        self._preview_fps = max(0.0, float(os.environ.get("VP_PREVIEW_FPS", "30")))
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

    def _reset_decklink_fps_tracking(self) -> None:
        self._decklink_processed_counter = 0
        self._decklink_processed_fps = 0.0
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
        self._decklink_no_frame_reason = None
        self._decklink_tick_pending = False
        self._decklink_tick_pending_since = 0.0

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
            "rtx_video_sdk_root": os.environ.get("RTX_VIDEO_SDK_ROOT", r"C:\Coding Projects\sdks\NVidia video SDK"),
        }

        # Keep request queue larger than response queue so bursty UI events
        # (ROI drag, tick polling) do not trip queue.Full in the GUI thread.
        self._request_queue = self._ctx.Queue(maxsize=32)
        self._response_queue = self._ctx.Queue(maxsize=64)
        self._process = self._ctx.Process(
            target=run_processor_worker,
            args=(self._request_queue, self._response_queue, startup_config),
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
            "set_decklink_output_buffer_frames",
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

    def set_roi_with_subpixel(self, roi: Roi, shift_x: float, shift_y: float) -> None:
        self._send_control(
            {
                "cmd": "set_roi_with_subpixel",
                "x": int(roi.x),
                "y": int(roi.y),
                "w": int(roi.w),
                "h": int(roi.h),
                "shift_x": float(shift_x),
                "shift_y": float(shift_y),
            }
        )

    def start_roi_microstep_transition(
        self,
        start_roi: Roi,
        target_roi: Roi,
        duration_frames: int,
        interpolation_mode: str,
        overscan_percent: float,
        start_from_current: bool = False,
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
                if expected_cmd == "set_basic_scaling_method":
                    self.basic_scaling_method = str(message.get("basic_scaling_method", message.get("sr_flavor", self.basic_scaling_method)))
                if expected_cmd == "set_sr_flavor":
                    self.basic_scaling_method = str(message.get("basic_scaling_method", message.get("sr_flavor", self.basic_scaling_method)))
                if expected_cmd == "set_deinterlace_method":
                    self.deinterlace_method = str(message.get("deinterlace_method", self.deinterlace_method))
                if expected_cmd == "set_deinterlace_enabled":
                    self.deinterlace_enabled = bool(message.get("deinterlace_enabled", self.deinterlace_enabled))
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
        self._roi_smoothing_percent = 60
        self._roi_latency_smoothing_percent = 0
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
        self._updating_controls = False
        self._controller_roi_target: Roi | None = None
        self._controller_roi_applied = self._roi
        self._manual_live_target_roi: Roi | None = None
        self._pending_manual_controller_roi: Roi | None = None
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
        self._fullscreen_view_name: str | None = None
        self._splitter_initialized = False
        self._main_splitter_initialized = False
        self._is_closing = False
        self._pending_persisted_input_device = None
        self._pending_persisted_output_device = None
        self._pending_persisted_input_mode_text = ""
        self._pending_persisted_output_mode_text = ""
        self._settings_path = Path(__file__).resolve().parent / "app_settings.json"
        self._settings_save_timer = QTimer(self)
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.setInterval(250)
        self._settings_save_timer.timeout.connect(self._save_settings)
        self._decklink_buffer_reapply_timer = QTimer(self)
        self._decklink_buffer_reapply_timer.setSingleShot(True)
        self._decklink_buffer_reapply_timer.setInterval(250)
        self._decklink_buffer_reapply_timer.timeout.connect(self._reapply_decklink_after_buffer_change)
        self._ai_sr_profiles_path = Path(__file__).resolve().parent / "ai_sr_profiles.json"
        self._ai_sr_profiles = self._load_ai_sr_profiles()
        self._preview_downsample_factor = self._normalize_preview_downsample_factor(
            float(os.environ.get("VP_PREVIEW_DOWNSAMPLE", "0.25"))
        )
        self._decklink_tick_poll_fps = max(1.0, float(os.environ.get("VP_DECKLINK_TICK_POLL_FPS", "30")))
        self._decklink_output_buffer_frames = max(
            0,
            min(10, int(getattr(self._controller, "decklink_output_buffer_frames", 2))),
        )

        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        self._fullscreen_keyframe_toolbars: dict[str, QWidget] = {}
        self._fullscreen_roi_save_key_buttons: dict[str, QPushButton] = {}
        self._fullscreen_roi_key_slot_buttons: dict[str, tuple[QPushButton, QPushButton, QPushButton, QPushButton]] = {}
        viewers = QWidget()
        viewers_layout = QVBoxLayout(viewers)
        viewers_layout.setContentsMargins(0, 0, 0, 0)
        viewers_layout.setSpacing(4)

        self._input_panel = QWidget()
        input_layout = QVBoxLayout(self._input_panel)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(2)
        self._input_title_label = QLabel("Input View (ROI controls are locked to this view)")
        input_layout.addWidget(self._input_title_label)
        input_layout.addWidget(self._input_canvas, 1, alignment=Qt.AlignCenter)
        self._input_fullscreen_keyframe_toolbar = self._build_fullscreen_keyframe_toolbar("input")
        input_layout.addWidget(self._input_fullscreen_keyframe_toolbar)

        self._output_panel = QWidget()
        output_layout = QVBoxLayout(self._output_panel)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(2)
        self._output_title_label = QLabel("Output View (processed result only)")
        output_layout.addWidget(self._output_title_label)
        output_layout.addWidget(self._output_canvas, 1, alignment=Qt.AlignCenter)
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
        self._roi_keyframe_transition_timer.setInterval(8)
        self._roi_keyframe_transition_timer.setTimerType(Qt.PreciseTimer)
        self._roi_keyframe_transition_timer.timeout.connect(self._step_roi_keyframe_transition)

        self._setup_shortcuts()
        self._connect_settings_persistence_signals()
        self._update_roi_key_buttons()
        self._sync_controls_from_roi(self._roi)
        self._load_settings()
        self._sync_ai_sr_basic_scaling_ui(notify=False)
        self._apply_startup_ai_sr_settings()
        self._source_mode = self.source_mode_combo.currentText()
        self._sync_blackmagic_controls_enabled_state()
        self._on_source_mode_changed()
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
        self._controller.create(self._roi)
        self._sync_ai_sr_basic_scaling_ui(notify=False)
        self._apply_startup_ai_sr_settings()
        LOGGER.info("Worker controller recreated after unexpected worker exit")

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
            self.rtx_vsr_quality_combo,
            self.rtx_vsr_scale_combo,
            self.rtx_vsr_post_scale_method_combo,
            self.source_mode_combo,
            self.decklink_input_device_combo,
            self.decklink_output_device_combo,
            self.decklink_input_mode_combo,
            self.decklink_output_mode_combo,
            self.roi_interp_mode_combo,
        ]
        for combo in combo_widgets:
            combo.currentTextChanged.connect(self._schedule_settings_save)

        checkbox_widgets = [
            self.enable_sr_checkbox,
            self.enable_ai_sr_checkbox,
            self.enable_rtx_vsr_checkbox,
            self.deinterlace_checkbox,
            self.perf_guard_checkbox,
            self.ai_sr_require_gpu_checkbox,
            self.ai_sr_strict_checkbox,
            self.rtx_thdr_enable_checkbox,
            self.decklink_auto_detect_devices,
            self.decklink_enable_format_detection,
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
            self.ai_sr_frame_interval_spin,
            self.ai_sr_overscan_spin,
            self.ai_sr_inference_divisor_spin,
            self.ai_sr_detail_preserve_spin,
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

        self._updating_controls = True
        try:
            self.fps_spin.setValue(max(1, min(60, int(raw.get("fps", self.fps_spin.value())))))
            self.preview_request_fps_spin.setValue(max(1, min(60, int(raw.get("preview_request_fps", self.preview_request_fps_spin.value())))))
            self.preview_poll_fps_spin.setValue(max(1, min(120, int(raw.get("preview_poll_fps", self.preview_poll_fps_spin.value())))))
            self.decklink_output_buffer_spin.setValue(
                max(0, min(10, int(raw.get("decklink_output_buffer_frames", self.decklink_output_buffer_spin.value()))))
            )
            self.roi_smoothing_slider.setValue(max(0, min(100, int(raw.get("roi_smoothing_percent", self.roi_smoothing_slider.value())))))
            self.roi_latency_smoothing_slider.setValue(max(0, min(100, int(raw.get("roi_latency_smoothing_percent", self.roi_latency_smoothing_slider.value())))))
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
        finally:
            self._updating_controls = False

        self._preview_downsample_factor = self._normalize_preview_downsample_factor(
            PREVIEW_DOWNSAMPLE_LABEL_TO_FACTOR.get(self.preview_downsample_combo.currentText(), self._preview_downsample_factor)
        )
        self._decklink_tick_poll_fps = float(max(1, self.preview_poll_fps_spin.value()))
        self._decklink_output_buffer_frames = int(self.decklink_output_buffer_spin.value())

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

        self.color_space_combo = QComboBox()
        self.color_space_combo.addItems(list(COLOR_SPACE_LABEL_TO_NAME.keys()))
        self.color_space_combo.setCurrentText(
            COLOR_SPACE_NAME_TO_LABEL.get(getattr(self._controller, "color_space", "rec709"), "Rec.709 (SDR)")
        )
        self.color_space_combo.currentIndexChanged.connect(self._on_color_space_changed)
        settings_form.addRow("Color space", self.color_space_combo)

        self.color_range_combo = QComboBox()
        self.color_range_combo.addItems(list(COLOR_RANGE_LABEL_TO_NAME.keys()))
        self.color_range_combo.setCurrentText(
            COLOR_RANGE_NAME_TO_LABEL.get(getattr(self._controller, "color_range", "limited"), "Limited (Video)")
        )
        self.color_range_combo.currentIndexChanged.connect(self._on_color_range_changed)
        settings_form.addRow("Color range", self.color_range_combo)

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

        self.perf_guard_checkbox = QCheckBox("Auto performance guard (reduce SR when overloaded)")
        self.perf_guard_checkbox.setChecked(False)
        self.perf_guard_checkbox.toggled.connect(self._on_perf_guard_toggled)
        settings_form.addRow(self.perf_guard_checkbox)

        decklink_box = QGroupBox("Blackmagic I/O")
        decklink_form = QFormLayout(decklink_box)

        self.source_mode_combo = QComboBox()
        self.source_mode_combo.addItems(["Synthetic", "Blackmagic DeckLink"])
        self.source_mode_combo.currentIndexChanged.connect(self._on_source_mode_changed)
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
        decklink_form.addRow("Input mode", self.decklink_input_mode_combo)

        self.decklink_output_mode_combo = QComboBox()
        decklink_form.addRow("Output mode", self.decklink_output_mode_combo)

        self.decklink_enable_format_detection = QCheckBox("Enable input format detection")
        self.decklink_enable_format_detection.setChecked(True)
        decklink_form.addRow(self.decklink_enable_format_detection)

        self.decklink_output_buffer_spin = QSpinBox()
        self.decklink_output_buffer_spin.setRange(0, 10)
        self.decklink_output_buffer_spin.setValue(int(self._decklink_output_buffer_frames))
        self.decklink_output_buffer_spin.setToolTip("DeckLink output startup/steady buffer in frames; larger values can smooth short stalls with added latency.")
        self.decklink_output_buffer_spin.valueChanged.connect(self._on_decklink_output_buffer_changed)
        decklink_form.addRow("DeckLink output buffer (frames)", self.decklink_output_buffer_spin)

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
        self.roi_smoothing_slider.setRange(0, 100)
        self.roi_smoothing_slider.setSingleStep(1)
        self.roi_smoothing_slider.setPageStep(5)
        self.roi_smoothing_slider.setValue(int(self._roi_smoothing_percent))
        self.roi_smoothing_slider.valueChanged.connect(self._on_roi_smoothing_changed)

        self.roi_smoothing_value_label = QLabel(f"{self._roi_smoothing_percent}%")
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

        self.roi_transition_frames_spin = QSpinBox()
        self.roi_transition_frames_spin.setRange(1, 600)
        self.roi_transition_frames_spin.setValue(int(self._roi_keyframe_transition_default_frames))
        roi_form.addRow("Transition (frames)", self.roi_transition_frames_spin)

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

        layout.addWidget(decklink_box)
        layout.addWidget(deinterlace_box)
        layout.addWidget(denoise_box)
        layout.addWidget(upscaling_box)
        layout.addWidget(rtx_vsr_box)
        layout.addWidget(settings_box)
        layout.addWidget(roi_box)
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

    def _setup_shortcuts(self) -> None:
        reset_action = QAction(self)
        reset_action.setShortcut("R")
        reset_action.triggered.connect(self._reset_roi)
        self.addAction(reset_action)

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
                    return

                input_frame, output_frame = decklink_frame
                self._no_frame_counter = 0
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
                        f"Running | Preview FPS={fps:.1f} | Output FPS={worker_fps:.1f} | {basic_status_text} | AI SR={ai_sr_state}{ai_sr_detail} | AI refresh FPS={ai_refresh_fps:.2f} | AI age={ai_latest_age_ms:.0f}ms | AI completed={ai_completed} | RTX VSR={rtx_vsr_state}{rtx_vsr_detail} | RTX applied={'yes' if rtx_applied else 'no'} | RTX delta={rtx_delta:.2f} | AI frames {ai_counts}{ai_stage_timing_text} | Stage enabled[{stage_enable_text}] | Stage last[{stage_last_text}] | Stage counts[{stage_count_text}]"
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
                        f"DeckLink streaming via worker process | preview_fps={fps:.1f} | output_fps={worker_fps:.1f}"
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
            title_label=self._input_title_label,
            canvas=self._input_canvas,
            footer_widget=self._input_fullscreen_keyframe_toolbar,
        )
        self._fit_canvas_in_panel(
            panel=self._output_panel,
            title_label=self._output_title_label,
            canvas=self._output_canvas,
            footer_widget=self._output_fullscreen_keyframe_toolbar,
        )

    def _fit_canvas_in_panel(
        self,
        panel: QWidget,
        title_label: QLabel,
        canvas: QWidget,
        footer_widget: QWidget | None = None,
    ) -> None:
        if not panel.isVisible() or panel.width() <= 0 or panel.height() <= 0:
            return

        layout = panel.layout()
        if layout is None:
            return

        margins = layout.contentsMargins()
        spacing = max(0, layout.spacing())
        used_h = 0
        if title_label.isVisible():
            used_h += title_label.sizeHint().height() + spacing
        if footer_widget is not None and footer_widget.isVisible():
            used_h += footer_widget.sizeHint().height() + spacing

        avail_w = panel.width() - margins.left() - margins.right()
        avail_h = panel.height() - margins.top() - margins.bottom() - used_h

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
        if event.key() == Qt.Key_Escape and self._fullscreen_view_name is not None:
            self._set_fullscreen_view(None)
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_canvas_fullscreen_requested(self, view_name: str) -> None:
        if self._fullscreen_view_name == view_name:
            self._set_fullscreen_view(None)
            return
        self._set_fullscreen_view(view_name)

    def _set_fullscreen_view(self, view_name: str | None) -> None:
        self._fullscreen_view_name = view_name
        if view_name is None:
            self._controls_scroll.setVisible(True)
            self._input_panel.setVisible(True)
            self._output_panel.setVisible(True)
            for toolbar in self._fullscreen_keyframe_toolbars.values():
                toolbar.setVisible(False)
            self._input_canvas.setEnabled(True)
            self._output_canvas.setEnabled(True)
            self.showNormal()
            self._splitter_initialized = False
            QTimer.singleShot(0, self._apply_initial_viewer_layout)
            return

        self._controls_scroll.setVisible(False)
        self._input_panel.setVisible(view_name == "input")
        self._output_panel.setVisible(view_name == "output")
        for toolbar_view, toolbar in self._fullscreen_keyframe_toolbars.items():
            toolbar.setVisible(toolbar_view == view_name)
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
            self._update_status(f"Color space applied: {applied_label}")
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
            self._update_status(f"Color range applied: {applied_label}")
        except Exception as exc:
            self._update_status(f"Color range change failed: {exc}")

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
        self._decklink_output_buffer_frames = buffer_frames
        if hasattr(self._controller, "decklink_output_buffer_frames"):
            self._controller.decklink_output_buffer_frames = buffer_frames
        if self._updating_controls:
            return
        if self._source_mode == "Blackmagic DeckLink":
            self._decklink_buffer_reapply_timer.start()
        elif hasattr(self._controller, "set_decklink_output_buffer_frames"):
            self._controller.set_decklink_output_buffer_frames(buffer_frames)
        self._update_status(f"DeckLink output buffer set to {buffer_frames} frame(s); applying")

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

        # Treat manual updates as a continuous live keyframe stream: keep only
        # the latest target and interpolate from applied ROI on each send tick.
        if self._manual_live_target_roi is None:
            self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        self._manual_live_target_roi = self._roi

        self._pending_manual_controller_roi = self._roi
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
            self._controller_backend == "worker-process"
            and hasattr(self._controller, "set_roi_with_subpixel")
        )
        target_scale = roi_scale_from_roi(target)
        if use_subpixel_microstep:
            step_roi, step_shift_x, step_shift_y = self._manual_roi_step_with_subpixel(current, target)
        else:
            step_roi = self._interpolate_controller_roi_step(current, target)
            step_shift_x = 0.0
            step_shift_y = 0.0

        if self._is_controller_roi_close(step_roi, target):
            step_roi = target
            step_shift_x = 0.0
            step_shift_y = 0.0

        try:
            started = time.perf_counter()
            self._roi_diag_controller_send_attempts += 1
            if use_subpixel_microstep:
                self._controller.set_roi_with_subpixel(step_roi, step_shift_x, step_shift_y)
                sent = True
            else:
                moving_only = (
                    step_roi.w == self._controller_roi_applied.w
                    and step_roi.h == self._controller_roi_applied.h
                    and hasattr(self._controller, "set_roi_position")
                )
                if moving_only:
                    sent = bool(self._controller.set_roi_position(step_roi.x, step_roi.y))
                else:
                    sent = bool(self._controller.set_roi(step_roi))

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._roi_diag_controller_send_ms_sum += elapsed_ms
            if elapsed_ms > self._roi_diag_controller_send_ms_max:
                self._roi_diag_controller_send_ms_max = elapsed_ms

            if sent:
                self._roi_diag_controller_send_success += 1
            else:
                self._roi_diag_controller_send_drops += 1
            self._controller_roi_applied = step_roi
        except Exception as exc:
            self._roi_diag_controller_send_drops += 1
            self._update_status(f"ROI update failed: {exc}")

        # Continue stepping while target is not reached, or while new user
        # updates keep arriving.
        if self._is_controller_roi_close(self._controller_roi_applied, target):
            if use_subpixel_microstep:
                try:
                    self._controller.set_roi_with_subpixel(target, 0.0, 0.0)
                except Exception:
                    pass
            self._manual_live_target_roi = None
            self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        if self._pending_manual_controller_roi is not None or self._manual_live_target_roi is not None:
            self._manual_roi_send_timer.setInterval(self._manual_roi_send_interval_ms(target_scale))
            self._manual_roi_send_timer.start()

    def _manual_roi_send_interval_ms(self, zoom_scale: float) -> int:
        z = max(1.0, float(zoom_scale))
        if z >= 6.0:
            return 8
        if z >= 4.0:
            return 10
        return 16

    def _manual_roi_step_with_subpixel(self, current: Roi, target: Roi) -> tuple[Roi, float, float]:
        moving_only = current.w == target.w and current.h == target.h
        zoom_scale = roi_scale_from_roi(target)
        smoothing = max(0.0, min(1.0, self._roi_smoothing_percent / 100.0))

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
        smoothing = max(0.0, min(1.0, self._roi_smoothing_percent / 100.0))
        if target_scale >= 6.0:
            base_interval_ms = 8
        elif target_scale >= 4.0:
            base_interval_ms = 10
        else:
            base_interval_ms = 16
        interval_scale = 1.20 - (0.60 * smoothing)
        interval_ms = int(round(base_interval_ms * interval_scale))
        self._controller_roi_interp_timer.setInterval(max(6, min(24, interval_ms)))
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
        smoothing = max(0.0, min(1.0, self._roi_smoothing_percent / 100.0))
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
        clamped = max(0, min(100, int(value)))
        self._roi_smoothing_percent = clamped
        self.roi_smoothing_value_label.setText(f"{clamped}%")
        self._input_canvas.set_smoothing_percent(clamped)

    def _on_roi_latency_smoothing_changed(self, value: int) -> None:
        clamped = max(0, min(100, int(value)))
        self._roi_latency_smoothing_percent = clamped
        self.roi_latency_smoothing_value_label.setText(f"{clamped}%")
        self._input_canvas.set_latency_smoothing_percent(clamped)

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

    def _on_deinterlace_method_changed(self) -> None:
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
                f"hold_last={hold_last_frame}, max_hold_ms={max_hold_ms:.0f}"
            ),
            f"Frames: fresh={ai_applied}, reused={ai_reused}, passthrough={ai_passthrough}",
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
        if self._source_mode == "Synthetic":
            self._stop_decklink_sessions()
            self.decklink_status_label.setText("Synthetic mode active")
            self._update_fps_control_lock()
            return

        self._update_fps_control_lock()
        self._refresh_decklink_catalog()
        self._on_apply_decklink_settings()

    def _sync_blackmagic_controls_enabled_state(self) -> None:
        blackmagic_selected = self.source_mode_combo.currentText() == "Blackmagic DeckLink"
        for widget in [
            self.decklink_input_device_combo,
            self.decklink_output_device_combo,
            self.decklink_auto_detect_devices,
            self.decklink_input_mode_combo,
            self.decklink_output_mode_combo,
            self.decklink_enable_format_detection,
            self.decklink_apply_btn,
            self.decklink_refresh_btn,
        ]:
            widget.setEnabled(blackmagic_selected)

    def _update_fps_control_lock(self) -> None:
        blackmagic_selected = self.source_mode_combo.currentText() == "Blackmagic DeckLink"
        self.fps_spin.setEnabled(not blackmagic_selected)

    def _on_apply_decklink_settings(self) -> None:
        if self._source_mode != "Blackmagic DeckLink":
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
            raise RuntimeError("No compatible DeckLink input/output devices selected")

        in_mode = self._selected_combo_data(self.decklink_input_mode_combo)
        out_mode = self._selected_combo_data(self.decklink_output_mode_combo)
        if in_mode is None or out_mode is None:
            raise RuntimeError("No compatible DeckLink input/output modes selected")

        input_fps = self._resolve_mode_fps(in_device, in_mode, input_side=True)
        output_fps = self._resolve_mode_fps(out_device, out_mode, input_side=False)
        if hasattr(self._controller, "decklink_output_buffer_frames"):
            self._controller.decklink_output_buffer_frames = int(self.decklink_output_buffer_spin.value())

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
                if "Worker process exited unexpectedly" not in error_text:
                    raise
                LOGGER.warning("DeckLink start hit dead worker; recreating worker and retrying once: %s", error_text)
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
        self._decklink_sessions_running = True
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
        if self.decklink_input_device_combo.count() > 0:
            self.decklink_input_device_combo.setCurrentIndex(0)
        if self.decklink_output_device_combo.count() > 0:
            self.decklink_output_device_combo.setCurrentIndex(0)

    def _on_auto_detect_toggled(self, checked: bool) -> None:
        if checked:
            self._apply_auto_detect_device_selection()
            self._populate_mode_combos()

    def _on_decklink_device_changed(self) -> None:
        self._populate_mode_combos()

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

    def _fps_from_mode(self, mode: object) -> float:
        frame_duration = float(getattr(mode, "frame_duration", 0))
        time_scale = float(getattr(mode, "time_scale", 0))
        if frame_duration <= 0 or time_scale <= 0:
            return 0.0
        return time_scale / frame_duration

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
        return combo.currentData()

    def _stop_decklink_sessions(self) -> None:
        if self._decklink_buffer_reapply_timer.isActive():
            self._decklink_buffer_reapply_timer.stop()
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
            if self._last_frame_error != "No input signal frames received":
                self._last_frame_error = "No input signal frames received"
                self._update_status("DeckLink connected but no input frames yet; check source signal and input mode")
            return None
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
        adaptive_suffix = ""
        if duration_frames != requested_duration_frames:
            adaptive_suffix = f", adaptive from {requested_duration_frames}"

        self._start_roi_keyframe_transition(keyframe.roi, duration_frames, keyframe.interpolation_mode)
        if override_duration:
            self._update_status(
                f"Recalling KEY {slot} over {duration_frames} frames ({self._roi_interp_mode_label(keyframe.interpolation_mode)}, override{adaptive_suffix})"
            )
        else:
            self._update_status(
                f"Recalling KEY {slot} over {duration_frames} frames ({self._roi_interp_mode_label(keyframe.interpolation_mode)}{adaptive_suffix})"
            )

    def _effective_roi_keyframe_duration_frames(self, target_roi: Roi, requested_frames: int) -> int:
        requested = max(1, min(600, int(requested_frames)))
        current_roi = clamp_roi(self._roi)
        target = clamp_roi(target_roi)

        start_scale = max(1.0, roi_scale_from_roi(current_roi))
        target_scale = max(1.0, roi_scale_from_roi(target))
        scale_ratio = max(start_scale, target_scale) / max(1e-6, min(start_scale, target_scale))
        scale_jump = max(0.0, math.log2(max(1.0, scale_ratio)))

        current_cx = float(current_roi.x) + (float(current_roi.w) * 0.5)
        current_cy = float(current_roi.y) + (float(current_roi.h) * 0.5)
        target_cx = float(target.x) + (float(target.w) * 0.5)
        target_cy = float(target.y) + (float(target.h) * 0.5)
        center_distance = math.hypot(target_cx - current_cx, target_cy - current_cy)

        adaptive_floor = int(math.ceil((scale_jump * 14.0) + (center_distance / 180.0)))
        if scale_ratio >= 1.75:
            adaptive_floor = max(adaptive_floor, int(math.ceil(requested * 1.35)))
        elif scale_ratio >= 1.4:
            adaptive_floor = max(adaptive_floor, int(math.ceil(requested * 1.15)))

        return max(requested, min(600, adaptive_floor))

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

    def _cancel_roi_keyframe_transition(self, reset_subpixel_shift: bool = True) -> None:
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
        if hasattr(self._controller, "cancel_roi_microstep_transition"):
            try:
                self._controller.cancel_roi_microstep_transition(reset_subpixel_shift=reset_subpixel_shift)
            except Exception:
                pass
        if reset_subpixel_shift and hasattr(self._controller, "set_roi_subpixel_shift"):
            self._controller.set_roi_subpixel_shift(0.0, 0.0)

    def _start_roi_keyframe_transition(self, target_roi: Roi, duration_frames: int, interpolation_mode: str) -> None:
        previous_state = self._roi_keyframe_transition
        if isinstance(previous_state, dict):
            current_estimate = previous_state.get("current_roi_estimate")
            if isinstance(current_estimate, Roi):
                self._roi = clamp_roi(current_estimate)

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
        self._cancel_roi_keyframe_transition(reset_subpixel_shift=False)

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

        if backend_driven:
            try:
                self._controller.start_roi_microstep_transition(
                    start_roi=clamp_roi(self._roi),
                    target_roi=target,
                    duration_frames=total_frames,
                    interpolation_mode=mode_name,
                    overscan_percent=float(self._roi_keyframe_transition_overscan_percent),
                    start_from_current=True,
                )
            except Exception as exc:
                self._update_status(f"Worker ROI microstep transition start failed: {exc}")
                self._roi_keyframe_transition["backend_driven"] = False
        self._roi_keyframe_transition_timer.start()

    def _apply_roi_interpolation_curve(self, t: float, interpolation_mode: str) -> float:
        clamped_t = max(0.0, min(1.0, float(t)))
        if str(interpolation_mode).strip().lower() == "ease_in_out":
            # Smoothstep for gentle acceleration/deceleration.
            return clamped_t * clamped_t * (3.0 - (2.0 * clamped_t))
        if str(interpolation_mode).strip().lower() == "ease_out":
            return 1.0 - ((1.0 - clamped_t) * (1.0 - clamped_t))
        return clamped_t

    def _roi_keyframe_transition_fps(self) -> float:
        # Keep transition cadence deterministic; modulating by live worker FPS can
        # introduce subtle speed wobble that appears as jitter.
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
            # Keep worker telemetry fresh while a transition is active so
            # processed_frame_counter deltas reflect output cadence.
            try:
                self._controller.decklink_tick(timeout_ms=0)
            except Exception:
                pass

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

        if frame_advance <= 0.0:
            if bool(state.get("use_worker_clock", False)):
                frame_advance = 0.0
            else:
                frame_advance = dt * self._roi_keyframe_transition_fps()

        frame_progress += frame_advance
        frame_progress = min(float(total_frames), frame_progress)
        state["frame_progress"] = frame_progress

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
            display_w = max(2.0, float(ideal_w))
            display_h = max(2.0, float(display_w * 9.0 / 16.0))
            display_x = float(ideal_cx - (display_w * 0.5))
            display_y = float(ideal_cy - (display_h * 0.5))
            self._input_canvas.set_visual_roi_overlay(display_x, display_y, display_w, display_h)

            estimated_scale = FRAME_W / max(1.0, display_w)
            estimated_roi = clamp_roi(roi_from_scale(estimated_scale, ideal_cx, ideal_cy))
            state["current_roi_estimate"] = estimated_roi
            self._roi = estimated_roi
            self._controller_roi_applied = estimated_roi
            self._controller_roi_target = None
            self._controller_filtered_target_roi = None

            transition_complete = frame_progress >= float(total_frames)
            if transition_complete:
                self._roi_keyframe_transition = None
                self._roi_keyframe_transition_timer.stop()
                self._input_canvas.clear_visual_roi_overlay()
                self._roi = target_roi
                self._input_canvas.set_roi(target_roi)
                self._controller_roi_applied = target_roi
                self._controller_roi_target = None
                self._controller_filtered_target_roi = None
                self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
                self._sync_controls_from_roi(target_roi)
                self._roi_keyframe_last_step_ts = 0.0
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

        roi_changed = (
            interpolated.x != self._roi.x
            or interpolated.y != self._roi.y
            or interpolated.w != self._roi.w
            or interpolated.h != self._roi.h
        )

        transition_complete = frame_progress >= float(total_frames) or (
            interpolated.x == target_roi.x
            and interpolated.y == target_roi.y
            and interpolated.w == target_roi.w
            and interpolated.h == target_roi.h
        )

        if roi_changed:
            self._roi = interpolated
            self._input_canvas.set_roi(interpolated)

        if not backend_driven:
            if hasattr(self._controller, "set_roi_with_subpixel"):
                self._controller.set_roi_with_subpixel(interpolated, target_shift_x, target_shift_y)
                self._controller_roi_applied = interpolated
            elif hasattr(self._controller, "set_roi_subpixel_shift"):
                self._controller.set_roi_subpixel_shift(target_shift_x, target_shift_y)
                if roi_changed:
                    self._apply_controller_roi_immediate(interpolated, reset_subpixel_shift=False)
            elif roi_changed:
                self._apply_controller_roi_immediate(interpolated, reset_subpixel_shift=False)

        if transition_complete:
            self._roi_keyframe_transition = None
            self._roi_keyframe_transition_timer.stop()
            self._input_canvas.clear_visual_roi_overlay()
            if hasattr(self._controller, "set_roi_subpixel_shift"):
                self._controller.set_roi_subpixel_shift(0.0, 0.0)
            if (
                interpolated.x != target_roi.x
                or interpolated.y != target_roi.y
                or interpolated.w != target_roi.w
                or interpolated.h != target_roi.h
            ):
                self._roi = target_roi
                self._input_canvas.set_roi(target_roi)
            # Always finalize backend ROI and control values at transition end.
            if backend_driven and hasattr(self._controller, "set_roi_with_subpixel"):
                self._controller.set_roi_with_subpixel(self._roi, 0.0, 0.0)
                self._controller_roi_applied = self._roi
            else:
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
