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

SR_FLAVOR_LABEL_TO_NAME = {
    "Bilinear (Fast)": "bilinear",
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


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("video_processor_gui")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
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


def clamp_roi(roi: Roi, width: int = FRAME_W, height: int = FRAME_H) -> Roi:
    x = max(0, min(roi.x, width - 2))
    y = max(0, min(roi.y, height - 2))

    max_w = max(2, width - x)
    max_h = max(2, height - y)

    # ROI is locked to 16:9 to match input/output display aspect.
    w = max(2, min(roi.w, max_w))
    w &= ~1
    if w < 2:
        w = 2

    h = max(2, int(round(w * 9.0 / 16.0)))
    if h > max_h:
        h = max_h
        w = max(2, int(round(h * 16.0 / 9.0)))
        w = min(w, max_w)
        w &= ~1
        if w < 2:
            w = 2
        h = max(2, int(round(w * 9.0 / 16.0)))

    if y + h > height:
        y = max(0, height - h)

    x &= ~1
    if x + w > width:
        x = max(0, width - w)
        x &= ~1

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


def uyvy_to_qimage(
    frame_bytes: bytes,
    preview_max_w: int | None = None,
    preview_max_h: int | None = None,
) -> tuple[QImage, np.ndarray | None]:
    if len(frame_bytes) != UYVY_FRAME_BYTES:
        raise ValueError("Invalid UYVY frame byte length.")

    if cv2 is not None:
        global _CV2_RGB_RING_INDEX
        if not _CV2_RGB_RING:
            # Two buffers avoid input/output previews aliasing each other within one tick.
            _CV2_RGB_RING.extend(
                [
                    np.empty((FRAME_H, FRAME_W, 3), dtype=np.uint8),
                    np.empty((FRAME_H, FRAME_W, 3), dtype=np.uint8),
                ]
            )

        rgb = _CV2_RGB_RING[_CV2_RGB_RING_INDEX]
        _CV2_RGB_RING_INDEX = (_CV2_RGB_RING_INDEX + 1) % len(_CV2_RGB_RING)

        yuv422 = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 2)
        cv2.cvtColor(yuv422, cv2.COLOR_YUV2RGB_UYVY, dst=rgb)
        image = QImage(rgb.data, FRAME_W, FRAME_H, FRAME_W * 3, QImage.Format_RGB888)
        if preview_max_w is not None and preview_max_h is not None:
            target_w = max(1, min(int(preview_max_w), FRAME_W))
            target_h = max(1, min(int(preview_max_h), FRAME_H))
            if target_w < FRAME_W or target_h < FRAME_H:
                return image.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.FastTransformation), None
        return image, rgb

    data = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(FRAME_H, FRAME_W // 2, 4)

    # UYVY packing: U, Y0, V, Y1 per 2 pixels.
    u = data[:, :, 0].astype(np.float32)
    y0 = data[:, :, 1].astype(np.float32)
    v = data[:, :, 2].astype(np.float32)
    y1 = data[:, :, 3].astype(np.float32)

    y = np.empty((FRAME_H, FRAME_W), dtype=np.float32)
    y[:, 0::2] = y0
    y[:, 1::2] = y1

    u_full = np.repeat(u, 2, axis=1)
    v_full = np.repeat(v, 2, axis=1)

    # BT.709 limited-range YUV->RGB conversion for HD video signals.
    c = y - 16
    d = u_full - 128
    e = v_full - 128

    r = np.clip(1.164383 * c + 1.792741 * e, 0, 255).astype(np.uint8)
    g = np.clip(1.164383 * c - 0.213249 * d - 0.532909 * e, 0, 255).astype(np.uint8)
    b = np.clip(1.164383 * c + 2.112402 * d, 0, 255).astype(np.uint8)

    rgb = np.empty((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    image = QImage(rgb.data, FRAME_W, FRAME_H, FRAME_W * 3, QImage.Format_RGB888)
    if preview_max_w is not None and preview_max_h is not None:
        target_w = max(1, min(int(preview_max_w), FRAME_W))
        target_h = max(1, min(int(preview_max_h), FRAME_H))
        if target_w < FRAME_W or target_h < FRAME_H:
            return image.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.FastTransformation), None
    return image, rgb


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
        self._roi = Roi(480, 270, 960, 540)

        self._drag_mode = "none"
        self._drag_start_pos = QPointF()
        self._drag_start_roi = self._roi

        self._last_touch_center: QPointF | None = None
        self._last_touch_dist: float | None = None
        self._last_touch_emit_ts = 0.0
        self._touch_emit_interval_s = 1.0 / 45.0
        self._touch_emit_pending = False
        self._touch_emit_pending_scale = False

    def set_image(self, image: QImage, backing: np.ndarray | None = None) -> None:
        self._image = image
        self._image_backing = backing
        self.update()

    def set_roi(self, roi: Roi) -> None:
        self._roi = clamp_roi(roi)
        self.update()

    def roi(self) -> Roi:
        return self._roi

    def paintEvent(self, event) -> None:
        del event
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)

        image_rect = self._image_rect()
        if self._image is not None:
            p.drawImage(image_rect, self._image)

        roi_rect_w = self._frame_to_widget_rect(self._roi)
        p.setRenderHint(QPainter.Antialiasing, True)

        p.setPen(QPen(Qt.yellow, 2))
        p.drawRect(roi_rect_w)

        p.setPen(QPen(Qt.green, 1))
        scale = roi_scale_from_roi(self._roi)
        p.drawText(12, 24, f"ROI: x={self._roi.x} y={self._roi.y} w={self._roi.w} h={self._roi.h}")
        p.drawText(12, 44, f"Scale: {scale:.2f}x")

        handle_size = 8
        p.fillRect(
            QRectF(
                roi_rect_w.right() - handle_size,
                roi_rect_w.bottom() - handle_size,
                handle_size * 2,
                handle_size * 2,
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

        self.setFocus(Qt.MouseFocusReason)
        self._drag_start_pos = event.position()
        self._drag_start_roi = self._roi

        roi_rect = self._frame_to_widget_rect(self._roi)
        handle_rect = QRectF(roi_rect.right() - 12, roi_rect.bottom() - 12, 24, 24)

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
            new_w = self._drag_start_roi.w + dw
            new_h = int(round(new_w * 9.0 / 16.0))
            new_roi = Roi(
                self._drag_start_roi.x,
                self._drag_start_roi.y,
                new_w,
                new_h,
            )

        self._set_roi_and_emit_touch_throttled(
            clamp_roi(new_roi),
            emit_scale=(self._drag_mode != "move"),
        )

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        del event
        self._drag_mode = "none"
        self._flush_pending_touch_emit()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.fullscreenRequested.emit(self._view_name)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta == 0:
            return

        factor = 1.0 + (0.1 if delta > 0 else -0.1)
        target_scale = roi_scale_from_roi(self._roi) * factor

        anchor_frame = self._widget_to_frame(event.position())
        self._apply_scale(target_scale, anchor_frame)

    def event(self, event) -> bool:
        et = event.type()
        if et in (QEvent.Type.TouchBegin, QEvent.Type.TouchUpdate, QEvent.Type.TouchEnd):
            self._handle_touch_event(event)
            event.accept()
            return True
        return super().event(event)

    def _handle_touch_event(self, event: QTouchEvent) -> None:
        points = event.points()
        if not points:
            self._flush_pending_touch_emit()
            self._last_touch_center = None
            self._last_touch_dist = None
            return

        if len(points) == 1:
            pos = points[0].position()
            frame = self._widget_to_frame(pos)
            if self._last_touch_center is not None:
                last_frame = self._widget_to_frame(self._last_touch_center)
                dx = int(round(frame.x() - last_frame.x()))
                dy = int(round(frame.y() - last_frame.y()))
                roi = clamp_roi(Roi(self._roi.x + dx, self._roi.y + dy, self._roi.w, self._roi.h))
                self._set_roi_and_emit_touch_throttled(roi)
            self._last_touch_center = pos
            self._last_touch_dist = None
            return

        p0 = points[0].position()
        p1 = points[1].position()
        center = QPointF((p0.x() + p1.x()) / 2.0, (p0.y() + p1.y()) / 2.0)
        dist = math.hypot(p0.x() - p1.x(), p0.y() - p1.y())

        if self._last_touch_center is not None:
            cur_frame = self._widget_to_frame(center)
            prev_frame = self._widget_to_frame(self._last_touch_center)
            dx = int(round(cur_frame.x() - prev_frame.x()))
            dy = int(round(cur_frame.y() - prev_frame.y()))
            moved = clamp_roi(Roi(self._roi.x + dx, self._roi.y + dy, self._roi.w, self._roi.h))
            if self._last_touch_dist is not None and self._last_touch_dist > 0:
                self.set_roi(moved)
            else:
                self._set_roi_and_emit_touch_throttled(moved, emit_scale=False)

        if self._last_touch_dist is not None and self._last_touch_dist > 0:
            ratio = dist / self._last_touch_dist
            self._apply_scale(
                roi_scale_from_roi(self._roi) * ratio,
                self._widget_to_frame(center),
                emit_scale=False,
                touch_throttle=True,
            )

        self._last_touch_center = center
        self._last_touch_dist = dist

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
            self._set_roi_and_emit_touch_throttled(new_roi, emit_scale=emit_scale)
            return
        self._set_roi_and_emit(new_roi, emit_scale=emit_scale)

    def _set_roi_and_emit(self, roi: Roi, emit_scale: bool = True) -> None:
        self.set_roi(roi)
        self.roiChanged.emit(roi.x, roi.y, roi.w, roi.h)
        if emit_scale:
            self.scaleChanged.emit(roi_scale_from_roi(roi))

    def _set_roi_and_emit_touch_throttled(self, roi: Roi, emit_scale: bool = True) -> None:
        self.set_roi(roi)
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
        image_rect = self._image_rect()
        sx = image_rect.width() / FRAME_W
        sy = image_rect.height() / FRAME_H
        return QRectF(
            image_rect.left() + (roi.x * sx),
            image_rect.top() + (roi.y * sy),
            max(1.0, roi.w * sx),
            max(1.0, roi.h * sy),
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
        self.basic_scaling_method = "bicubic"
        self.deinterlace_method = "bob"
        self.denoise_method = "off"
        self.denoise_strength = 0.35
        self.max_auto_basic_scaling = 4
        self.basic_scaling_manual = 4
        self.basic_scaling_auto_mode = True
        self.basic_scaling_method_supported = False
        self.ai_sr_enabled = False
        self.ai_sr_active = False
        self.ai_sr_model_path = ""
        self.ai_sr_error: str | None = None
        self.ai_sr_provider = "auto"
        self.ai_sr_require_gpu = True
        self.ai_sr_frame_interval = 2
        self.ai_sr_strict = False
        self.ai_sr_input_align = 2
        self.ai_sr_roi_overscan_percent = 0.0
        self.ai_sr_inference_divisor = 0
        self.ai_sr_detail_preserve_percent = 0.0
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
        self.rtx_vsr_error: str | None = None
        self.rtx_vsr_info: dict[str, object] | None = None
        self.processor = None

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
        self.processor.set_deinterlace_enabled(self.deinterlace_enabled)
        if hasattr(self.processor, "set_deinterlace_method"):
            self.processor.set_deinterlace_method(self.deinterlace_method)
        if hasattr(self.processor, "set_denoise_method"):
            self.processor.set_denoise_method(self.denoise_method)
        if hasattr(self.processor, "set_denoise_strength"):
            self.processor.set_denoise_strength(self.denoise_strength)

    def set_roi(self, roi: Roi) -> None:
        if self.processor is not None:
            self.processor.set_roi(roi.x, roi.y, roi.w, roi.h)

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
        return self.processor.process_frame(frame_bytes)

    def close(self) -> None:
        self.processor = None

    def set_ai_sr_enabled(self, enabled: bool) -> None:
        self.ai_sr_enabled = bool(enabled)
        self.ai_sr_active = False
        self.ai_sr_error = "AI SR is only available with worker backend"

    def set_ai_sr_model_path(self, model_path: str) -> None:
        self.ai_sr_model_path = str(model_path)
        self.ai_sr_active = False
        self.ai_sr_error = "AI SR is only available with worker backend"

    def set_ai_sr_settings(
        self,
        provider: str,
        require_gpu: bool,
        frame_interval: int,
        strict: bool,
        input_align: int,
        roi_overscan_percent: float,
        inference_divisor: int,
        detail_preserve_percent: float,
    ) -> None:
        self.ai_sr_provider = str(provider)
        self.ai_sr_require_gpu = bool(require_gpu)
        self.ai_sr_frame_interval = max(1, int(frame_interval))
        self.ai_sr_strict = bool(strict)
        self.ai_sr_input_align = max(1, int(input_align))
        self.ai_sr_roi_overscan_percent = max(0.0, float(roi_overscan_percent))
        self.ai_sr_inference_divisor = max(0, int(inference_divisor))
        self.ai_sr_detail_preserve_percent = max(0.0, float(detail_preserve_percent))
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


class ProcessVideoProcessorController:
    def __init__(self) -> None:
        self.enable_basic_scaling = True
        self.deinterlace_enabled = True
        self.basic_scaling_method = "bicubic"
        self.deinterlace_method = "bob"
        self.denoise_method = "off"
        self.denoise_strength = 0.35
        self.max_auto_basic_scaling = 4
        self.basic_scaling_manual = 4
        self.basic_scaling_auto_mode = True
        self.basic_scaling_method_supported = True
        self.ai_sr_model_path = os.environ.get("VP_AI_SR_MODEL", "")
        self.ai_sr_enabled = os.environ.get("VP_AI_SR_ENABLE", "0") == "1"
        self.ai_sr_provider = os.environ.get("VP_AI_SR_PROVIDER", "auto")
        self.ai_sr_require_gpu = os.environ.get("VP_AI_SR_REQUIRE_GPU", "1") == "1"
        self.ai_sr_frame_interval = max(1, int(os.environ.get("VP_AI_SR_FRAME_INTERVAL", "2")))
        self.ai_sr_strict = os.environ.get("VP_AI_SR_STRICT", "0") == "1"
        self.ai_sr_input_align = max(1, int(os.environ.get("VP_AI_SR_INPUT_ALIGN", "2")))
        self.ai_sr_roi_overscan_percent = max(0.0, float(os.environ.get("VP_AI_SR_ROI_OVERSCAN_PCT", "0")))
        self.ai_sr_inference_divisor = max(0, int(os.environ.get("VP_AI_SR_INFERENCE_DIVISOR", "0")))
        self.ai_sr_detail_preserve_percent = max(0.0, float(os.environ.get("VP_AI_SR_DETAIL_PRESERVE_PCT", "0")))
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
        self._latest_effective_scale = 1
        self._decklink_no_frame_reason: str | None = None
        self._decklink_processed_counter = 0
        self._decklink_processed_fps = 0.0
        self._decklink_ai_applied_frames = 0
        self._decklink_ai_passthrough_frames = 0
        self._decklink_rtx_vsr_applied = False
        self._decklink_rtx_effect_mean_abs_luma = 0.0
        self._decklink_tick_pending = False

    def create(self, roi: Roi) -> None:
        self.close()

        if run_processor_worker is None:
            raise RuntimeError("Process worker module is unavailable")

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
            "enable_basic_scaling": self.enable_basic_scaling,
            "sr_scale": sr_scale,
            "basic_scaling_auto_mode": self.basic_scaling_auto_mode,
            "basic_scaling_manual": self.basic_scaling_manual,
            "basic_scaling_method": self.basic_scaling_method,
            "max_auto_basic_scaling": self.max_auto_basic_scaling,
            "deinterlace_enabled": self.deinterlace_enabled,
            "deinterlace_method": self.deinterlace_method,
            "denoise_method": self.denoise_method,
            "denoise_strength": self.denoise_strength,
            "ai_sr_enabled": self.ai_sr_enabled,
            "ai_sr_model_path": self.ai_sr_model_path,
            "ai_sr_provider": self.ai_sr_provider,
            "ai_sr_require_gpu": self.ai_sr_require_gpu,
            "ai_sr_frame_interval": self.ai_sr_frame_interval,
            "ai_sr_strict": self.ai_sr_strict,
            "ai_sr_input_align": self.ai_sr_input_align,
            "ai_sr_roi_overscan_percent": self.ai_sr_roi_overscan_percent,
            "ai_sr_inference_divisor": self.ai_sr_inference_divisor,
            "ai_sr_detail_preserve_percent": self.ai_sr_detail_preserve_percent,
            "rtx_vsr_enabled": self.rtx_vsr_enabled,
            "rtx_vsr_quality": self.rtx_vsr_quality,
            "rtx_vsr_scale": self.rtx_vsr_scale,
            "rtx_vsr_post_scale_method": self.rtx_vsr_post_scale_method,
            "rtx_thdr_enabled": self.rtx_thdr_enabled,
            "rtx_thdr_contrast": self.rtx_thdr_contrast,
            "rtx_thdr_saturation": self.rtx_thdr_saturation,
            "rtx_thdr_middle_gray": self.rtx_thdr_middle_gray,
            "rtx_thdr_max_luminance": self.rtx_thdr_max_luminance,
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
            raise RuntimeError("Worker process exited unexpectedly")

    def _send_control(self, command: dict[str, object]) -> None:
        self._assert_worker_alive()
        if self._request_queue is None:
            raise RuntimeError("Worker request queue is not initialized")

        cmd = str(command.get("cmd", ""))
        best_effort_cmds = {
            "set_roi",
            "decklink_tick",
            "set_deinterlace_enabled",
            "set_deinterlace_method",
            "set_denoise_settings",
            "set_max_auto_basic_scaling",
            "set_max_auto_sr_scale",
            "set_rtx_vsr_enabled",
            "set_rtx_vsr_settings",
        }

        try:
            self._request_queue.put_nowait(command)
        except queue.Full:
            # Prefer fresh control requests over stale queued work.
            try:
                self._request_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._request_queue.put_nowait(command)
            except queue.Full:
                # If still saturated, drop only best-effort commands.
                if cmd in best_effort_cmds:
                    return
                raise RuntimeError(f"Worker request queue saturated while sending '{cmd}'")

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
                self._latest_effective_scale = int(message.get("effective_sr_scale", self._latest_effective_scale))
                self._latest_decklink_frame = (
                    message["input_frame_bytes"],
                    message["output_frame_bytes"],
                )
                self._decklink_processed_counter = int(message.get("processed_frame_counter", self._decklink_processed_counter))
                self._decklink_processed_fps = float(message.get("processed_fps", self._decklink_processed_fps))
                self._decklink_ai_applied_frames = int(message.get("ai_sr_applied_frames", self._decklink_ai_applied_frames))
                self._decklink_ai_passthrough_frames = int(message.get("ai_sr_passthrough_frames", self._decklink_ai_passthrough_frames))
                self._decklink_rtx_vsr_applied = bool(message.get("rtx_vsr_applied", self._decklink_rtx_vsr_applied))
                self._decklink_rtx_effect_mean_abs_luma = float(
                    message.get("rtx_effect_mean_abs_luma", self._decklink_rtx_effect_mean_abs_luma)
                )
                self._decklink_no_frame_reason = None
                self._decklink_tick_pending = False
                continue

            if message_type == "decklink_no_frame":
                self._latest_decklink_frame = None
                self._decklink_no_frame_reason = str(message.get("reason", "unknown"))
                self._decklink_tick_pending = False
                continue

            if message_type == "ack":
                ack_cmd = str(message.get("cmd", ""))
                if ack_cmd in {"set_basic_scaling_method", "set_sr_flavor"}:
                    self.basic_scaling_method = str(message.get("basic_scaling_method", message.get("sr_flavor", self.basic_scaling_method)))
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

    def set_roi(self, roi: Roi) -> None:
        self._send_control({"cmd": "set_roi", "x": roi.x, "y": roi.y, "w": roi.w, "h": roi.h})

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
        while time.perf_counter() < deadline:
            self._assert_worker_alive()
            try:
                message = self._response_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            message_type = message.get("type")
            if message_type == "ack" and str(message.get("cmd")) == expected_cmd:
                if expected_cmd == "set_basic_scaling_method":
                    self.basic_scaling_method = str(message.get("basic_scaling_method", message.get("sr_flavor", self.basic_scaling_method)))
                if expected_cmd == "set_sr_flavor":
                    self.basic_scaling_method = str(message.get("basic_scaling_method", message.get("sr_flavor", self.basic_scaling_method)))
                if expected_cmd == "set_deinterlace_method":
                    self.deinterlace_method = str(message.get("deinterlace_method", self.deinterlace_method))
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
                self._latest_effective_scale = int(message.get("effective_sr_scale", self._latest_effective_scale))
                self._latest_decklink_frame = (
                    message["input_frame_bytes"],
                    message["output_frame_bytes"],
                )
                self._decklink_processed_counter = int(message.get("processed_frame_counter", self._decklink_processed_counter))
                self._decklink_processed_fps = float(message.get("processed_fps", self._decklink_processed_fps))
                self._decklink_ai_applied_frames = int(message.get("ai_sr_applied_frames", self._decklink_ai_applied_frames))
                self._decklink_ai_passthrough_frames = int(message.get("ai_sr_passthrough_frames", self._decklink_ai_passthrough_frames))
                self._decklink_no_frame_reason = None
                self._decklink_tick_pending = False
                continue
            if message_type == "decklink_no_frame":
                self._latest_decklink_frame = None
                self._decklink_no_frame_reason = str(message.get("reason", "unknown"))
                self._decklink_tick_pending = False
                continue
            if message_type == "warning":
                warning_text = str(message.get("warning", ""))
                if warning_text:
                    self.ai_sr_last_warning = warning_text
                continue

        raise RuntimeError(f"Timed out waiting for worker ack: {expected_cmd}")

    def start_decklink(self, in_device: int, in_mode: object, out_device: int, out_mode: object, enable_format_detection: bool) -> None:
        self._drain_responses()
        self._latest_decklink_frame = None
        self._decklink_no_frame_reason = None
        self._decklink_tick_pending = False
        self._send_control(
            {
                "cmd": "start_decklink",
                "in_device": int(in_device),
                "in_mode": in_mode,
                "out_device": int(out_device),
                "out_mode": out_mode,
                "enable_format_detection": bool(enable_format_detection),
            }
        )
        self._wait_for_ack("start_decklink")

    def stop_decklink(self) -> None:
        if self._process is None:
            return
        self._send_control({"cmd": "stop_decklink"})
        try:
            self._wait_for_ack("stop_decklink", timeout_seconds=1.5)
        except Exception:
            pass
        self._latest_decklink_frame = None
        self._decklink_tick_pending = False

    def decklink_tick(self, timeout_ms: int = 50) -> tuple[bytes, bytes] | None:
        self._drain_responses()
        if not self._decklink_tick_pending:
            # Keep at most one in-flight tick request so stale tick commands cannot
            # build up and push preview display several seconds behind live processing.
            self._send_control({"cmd": "decklink_tick", "timeout_ms": int(timeout_ms)})
            self._decklink_tick_pending = True
        self._drain_responses()
        return self._latest_decklink_frame

    def decklink_no_frame_reason(self) -> str | None:
        return self._decklink_no_frame_reason

    def decklink_processed_fps(self) -> float:
        return float(self._decklink_processed_fps)

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

    def set_ai_sr_enabled(self, enabled: bool) -> None:
        self.ai_sr_enabled = bool(enabled)
        self._send_control({"cmd": "set_ai_sr_enabled", "enabled": bool(enabled)})
        # Do not block the GUI thread waiting for worker ack; keep the last known
        # error visible until the ack updates ai_sr_error/ai_sr_info.

    def set_ai_sr_model_path(self, model_path: str) -> None:
        self.ai_sr_model_path = str(model_path)
        self._send_control({"cmd": "set_ai_sr_model_path", "model_path": self.ai_sr_model_path})
        # Avoid synchronous wait here; keep the previous status until the ack arrives.

    def set_ai_sr_settings(
        self,
        provider: str,
        require_gpu: bool,
        frame_interval: int,
        strict: bool,
        input_align: int,
        roi_overscan_percent: float,
        inference_divisor: int,
        detail_preserve_percent: float,
    ) -> None:
        self.ai_sr_provider = str(provider)
        self.ai_sr_require_gpu = bool(require_gpu)
        self.ai_sr_frame_interval = max(1, int(frame_interval))
        self.ai_sr_strict = bool(strict)
        self.ai_sr_input_align = max(1, int(input_align))
        self.ai_sr_roi_overscan_percent = max(0.0, float(roi_overscan_percent))
        self.ai_sr_inference_divisor = max(0, int(inference_divisor))
        self.ai_sr_detail_preserve_percent = max(0.0, float(detail_preserve_percent))
        self._send_control(
            {
                "cmd": "set_ai_sr_settings",
                "provider": self.ai_sr_provider,
                "require_gpu": self.ai_sr_require_gpu,
                "frame_interval": self.ai_sr_frame_interval,
                "strict": self.ai_sr_strict,
                "input_align": self.ai_sr_input_align,
                "roi_overscan_percent": self.ai_sr_roi_overscan_percent,
                "inference_divisor": self.ai_sr_inference_divisor,
                "detail_preserve_percent": self.ai_sr_detail_preserve_percent,
            }
        )

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

    def decklink_ai_sr_counts(self) -> tuple[int, int]:
        return int(self._decklink_ai_applied_frames), int(self._decklink_ai_passthrough_frames)

    def decklink_rtx_stats(self) -> tuple[bool, float]:
        return bool(self._decklink_rtx_vsr_applied), float(self._decklink_rtx_effect_mean_abs_luma)

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
        self._roi = Roi(480, 270, 960, 540)
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
        self._last_frame_error: str | None = None
        self._no_frame_counter = 0

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
        self._pending_controller_roi: Roi | None = None
        self._fullscreen_view_name: str | None = None
        self._splitter_initialized = False
        self._main_splitter_initialized = False
        self._is_closing = False
        self._ai_sr_profiles_path = Path(__file__).resolve().parent / "ai_sr_profiles.json"
        self._ai_sr_profiles = self._load_ai_sr_profiles()

        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
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

        self._output_panel = QWidget()
        output_layout = QVBoxLayout(self._output_panel)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(2)
        self._output_title_label = QLabel("Output View (processed result only)")
        output_layout.addWidget(self._output_title_label)
        output_layout.addWidget(self._output_canvas, 1, alignment=Qt.AlignCenter)

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

        self._roi_push_timer = QTimer(self)
        self._roi_push_timer.setSingleShot(True)
        self._roi_push_timer.setInterval(33)
        self._roi_push_timer.timeout.connect(self._flush_pending_controller_roi)

        self._setup_shortcuts()
        self._sync_controls_from_roi(self._roi)
        self.source_mode_combo.setCurrentText("Blackmagic DeckLink")
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

    def _default_ai_sr_model_path(self) -> str:
        return str(Path(__file__).resolve().parents[1] / "models" / "realesrgan_x4plus.onnx")

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

    def _current_ai_sr_profile(self) -> dict[str, object]:
        return {
            "provider": self.ai_sr_provider_combo.currentText().strip().lower(),
            "require_gpu": bool(self.ai_sr_require_gpu_checkbox.isChecked()),
            "frame_interval": int(self.ai_sr_frame_interval_spin.value()),
            "strict": bool(self.ai_sr_strict_checkbox.isChecked()),
            "input_align": int(self.ai_sr_input_align_combo.currentText()),
            "roi_overscan_percent": float(self.ai_sr_overscan_spin.value()),
            "inference_divisor": int(self.ai_sr_inference_divisor_spin.value()),
            "detail_preserve_percent": float(self.ai_sr_detail_preserve_spin.value()),
        }

    def _apply_ai_sr_profile(self, profile: dict[str, object]) -> None:
        provider = str(profile.get("provider", getattr(self._controller, "ai_sr_provider", "auto"))).lower()
        if provider not in {"auto", "cuda", "trt", "tensorrt", "cpu"}:
            provider = "auto"
        self.ai_sr_provider_combo.setCurrentText(provider)

        self.ai_sr_require_gpu_checkbox.setChecked(bool(profile.get("require_gpu", getattr(self._controller, "ai_sr_require_gpu", True))))
        self.ai_sr_frame_interval_spin.setValue(max(1, int(profile.get("frame_interval", getattr(self._controller, "ai_sr_frame_interval", 2)))))
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

        self.ai_sr_model_refresh_btn = QPushButton("Refresh Model List")
        self.ai_sr_model_refresh_btn.clicked.connect(self._on_ai_sr_model_refresh_clicked)
        ai_sr_model_actions_layout.addWidget(self.ai_sr_model_refresh_btn)

        self.ai_sr_provider_combo = QComboBox()
        self.ai_sr_provider_combo.addItems(["auto", "cuda", "trt", "cpu"])
        self.ai_sr_provider_combo.setCurrentText(str(getattr(self._controller, "ai_sr_provider", "auto")).lower())

        self.ai_sr_require_gpu_checkbox = QCheckBox("Require GPU provider")
        self.ai_sr_require_gpu_checkbox.setChecked(bool(getattr(self._controller, "ai_sr_require_gpu", True)))

        self.ai_sr_frame_interval_spin = QSpinBox()
        self.ai_sr_frame_interval_spin.setRange(1, 120)
        self.ai_sr_frame_interval_spin.setValue(int(getattr(self._controller, "ai_sr_frame_interval", 2)))

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
        upscaling_form.addRow(self.ai_sr_require_gpu_checkbox)
        upscaling_form.addRow("AI SR frame interval", self.ai_sr_frame_interval_spin)
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

        reset_btn = QPushButton("Reset ROI")
        reset_btn.clicked.connect(self._reset_roi)
        roi_form.addRow(reset_btn)

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
            "- Input view only: Touch 1-finger pan, 2-finger pinch\n"
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
                        self._update_status("DeckLink worker sessions not started")
                    else:
                        self._update_status("DeckLink worker active but no input frames yet; check source signal and input mode")
                    return

                input_frame, output_frame = decklink_frame
                self._no_frame_counter = 0

                self._perf_add("process", (time.perf_counter() - t0) * 1000.0)

                input_preview_size = self._preview_target_for_view("input")
                if input_preview_size is not None:
                    t1 = time.perf_counter()
                    input_image, input_backing = uyvy_to_qimage(
                        input_frame,
                        preview_max_w=input_preview_size[0],
                        preview_max_h=input_preview_size[1],
                    )
                    self._input_canvas.set_image(input_image, input_backing)
                    self._perf_add("convert_in", (time.perf_counter() - t1) * 1000.0)

                output_preview_size = self._preview_target_for_view("output")
                if output_preview_size is not None:
                    t1 = time.perf_counter()
                    output_image, output_backing = uyvy_to_qimage(
                        output_frame,
                        preview_max_w=output_preview_size[0],
                        preview_max_h=output_preview_size[1],
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
                    ai_passthrough = 0
                    if hasattr(self._controller, "decklink_processed_fps"):
                        worker_fps = float(self._controller.decklink_processed_fps())
                    if hasattr(self._controller, "decklink_ai_sr_counts"):
                        ai_applied, ai_passthrough = self._controller.decklink_ai_sr_counts()
                    ai_counts = f"{ai_applied}/{ai_passthrough}"
                    rtx_applied = False
                    rtx_delta = 0.0
                    if hasattr(self._controller, "decklink_rtx_stats"):
                        rtx_applied, rtx_delta = self._controller.decklink_rtx_stats()
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
                    self._update_status(
                        f"Running | Preview FPS={fps:.1f} | Worker FPS={worker_fps:.1f} | Basic scaling mode={mode_text} | Basic scaling method={flavor_text} | effective scaling={self._controller.effective_scale()} | AI SR={ai_sr_state}{ai_sr_detail} | RTX VSR={rtx_vsr_state}{rtx_vsr_detail} | RTX applied={'yes' if rtx_applied else 'no'} | RTX delta={rtx_delta:.2f} | AI applied/passthrough={ai_counts}"
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
                        f"DeckLink streaming via worker process | preview_fps={fps:.1f} | worker_fps={worker_fps:.1f}"
                    )
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
                self._update_status(
                    f"Running | FPS={fps:.1f} | Basic scaling mode={mode_text} | Basic scaling method={flavor_text} | effective scaling={self._controller.effective_scale()} | AI SR={ai_sr_state}{ai_sr_detail} | RTX VSR={rtx_vsr_state}{rtx_vsr_detail}"
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
        )
        self._fit_canvas_in_panel(
            panel=self._output_panel,
            title_label=self._output_title_label,
            canvas=self._output_canvas,
        )

    def _fit_canvas_in_panel(self, panel: QWidget, title_label: QLabel, canvas: QWidget) -> None:
        if not panel.isVisible() or panel.width() <= 0 or panel.height() <= 0:
            return

        layout = panel.layout()
        if layout is None:
            return

        margins = layout.contentsMargins()
        spacing = max(0, layout.spacing())
        label_h = title_label.sizeHint().height() if title_label.isVisible() else 0

        avail_w = panel.width() - margins.left() - margins.right()
        avail_h = panel.height() - margins.top() - margins.bottom() - label_h - spacing

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
            self._input_canvas.setEnabled(True)
            self._output_canvas.setEnabled(True)
            self.showNormal()
            self._splitter_initialized = False
            QTimer.singleShot(0, self._apply_initial_viewer_layout)
            return

        self._controls_scroll.setVisible(False)
        self._input_panel.setVisible(view_name == "input")
        self._output_panel.setVisible(view_name == "output")
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

        return (min(canvas_w, cap_w), min(canvas_h, cap_h))

    def _update_timer_interval(self) -> None:
        fps = max(1, self.fps_spin.value())
        self._timer.setInterval(int(1000 / fps))

    def _on_roi_from_canvas(self, x: int, y: int, w: int, h: int) -> None:
        self._roi = clamp_roi(Roi(x, y, w, h))
        self._pending_controller_roi = self._roi
        if not self._roi_push_timer.isActive():
            self._roi_push_timer.start()
        self._sync_controls_from_roi(self._roi)

    def _flush_pending_controller_roi(self) -> None:
        if self._pending_controller_roi is None:
            return
        self._controller.set_roi(self._pending_controller_roi)
        self._pending_controller_roi = None

    def _on_scale_from_canvas(self, scale: float) -> None:
        if self._updating_controls:
            return
        self._updating_controls = True
        self.scale_spin.setValue(scale)
        self._updating_controls = False

    def _on_roi_spin_changed(self) -> None:
        if self._updating_controls:
            return

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
        self._pending_controller_roi = None
        if self._roi_push_timer.isActive():
            self._roi_push_timer.stop()
        self._input_canvas.set_roi(roi)
        self._controller.set_roi(roi)
        self._sync_controls_from_roi(roi)

    def _on_scale_spin_changed(self, value: float) -> None:
        if self._updating_controls:
            return

        center_x = self._roi.x + (self._roi.w / 2.0)
        center_y = self._roi.y + (self._roi.h / 2.0)
        roi = roi_from_scale(value, center_x, center_y)
        self._roi = roi
        self._pending_controller_roi = None
        if self._roi_push_timer.isActive():
            self._roi_push_timer.stop()
        self._input_canvas.set_roi(roi)
        self._controller.set_roi(roi)
        self._sync_controls_from_roi(roi)

    def _on_sr_mode_changed(self) -> None:
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
        selected_name = SR_FLAVOR_LABEL_TO_NAME.get(selected_label, "bicubic")
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
        previous_value = self._controller.enable_basic_scaling
        self._controller.enable_basic_scaling = checked
        try:
            self._controller.create(self._roi)
            if self._source_mode == "Blackmagic DeckLink":
                self._start_decklink_sessions()
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
        try:
            self._controller.set_ai_sr_enabled(checked)
            if checked:
                model_path = self.ai_sr_model_combo.currentText().strip()
                self._update_status(f"AI SR toggle requested | awaiting worker ack | model={model_path}")
            else:
                self._update_status("AI SR disable requested | awaiting worker ack")
        except Exception as exc:
            self._update_status(f"AI SR toggle failed: {exc}")

    def _on_enable_rtx_vsr_toggled(self, checked: bool) -> None:
        try:
            self._controller.set_rtx_vsr_enabled(checked)
            if checked:
                if bool(getattr(self._controller, "ai_sr_enabled", False)):
                    self._update_status("RTX VSR enable requested | awaiting worker ack | note: RTX path is bypassed while AI SR is enabled")
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

    def _on_ai_sr_tuning_apply_clicked(self) -> None:
        try:
            profile = self._current_ai_sr_profile()
            self._controller.set_ai_sr_settings(
                provider=str(profile["provider"]),
                require_gpu=bool(profile["require_gpu"]),
                frame_interval=int(profile["frame_interval"]),
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
                self._update_status("RTX VSR settings update requested | awaiting worker ack | note: AI SR currently overrides RTX path")
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

        available = info.get("available_providers", [])
        if isinstance(available, (list, tuple)):
            available_text = ", ".join(str(item) for item in available) if available else "n/a"
        else:
            available_text = str(available)

        avg_infer_ms = info.get("avg_infer_ms")
        avg_infer_text = f"{float(avg_infer_ms):.2f} ms" if isinstance(avg_infer_ms, (int, float)) else "n/a"

        ai_applied = 0
        ai_passthrough = 0
        worker_fps = 0.0
        if hasattr(self._controller, "decklink_ai_sr_counts"):
            ai_applied, ai_passthrough = self._controller.decklink_ai_sr_counts()
        if hasattr(self._controller, "decklink_processed_fps"):
            worker_fps = float(self._controller.decklink_processed_fps())

        lines = [
            f"Enabled: {enabled} | Active: {active}",
            f"GPU active: {gpu_state} | Provider: {provider_upper} | Requested: {requested_provider}",
            f"Available providers: {available_text}",
            f"Model path: {info.get('model_path', getattr(self._controller, 'ai_sr_model_path', 'n/a'))}",
            f"Model scale: {info.get('model_scale', 'n/a')} | Input tensor: {info.get('model_input_w', 'n/a')}x{info.get('model_input_h', 'n/a')} | DType: {info.get('input_dtype', 'n/a')}",
            f"Avg infer: {avg_infer_text} | Worker FPS: {worker_fps:.1f}",
            (
                "Tuning: "
                f"interval={info.get('frame_interval', getattr(self._controller, 'ai_sr_frame_interval', 'n/a'))}, "
                f"strict={info.get('strict_mode', getattr(self._controller, 'ai_sr_strict', False))}, "
                f"align={info.get('input_align', getattr(self._controller, 'ai_sr_input_align', 'n/a'))}, "
                f"overscan={info.get('roi_overscan_percent', getattr(self._controller, 'ai_sr_roi_overscan_percent', 'n/a'))}, "
                f"divisor={info.get('inference_divisor', getattr(self._controller, 'ai_sr_inference_divisor', 'n/a'))}, "
                f"detail={info.get('detail_preserve_percent', getattr(self._controller, 'ai_sr_detail_preserve_percent', 'n/a'))}"
            ),
            f"Frames: applied={ai_applied}, passthrough={ai_passthrough}",
        ]

        if error_text:
            lines.append(f"Error: {error_text}")
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

        if self._controller_backend == "worker-process":
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
        LOGGER.info(
            "DeckLink started: input=%s mode=%s output=%s mode=%s fps=%s",
            input_name,
            in_mode_name,
            output_name,
            out_mode_name,
            fps_text,
        )

    def _resolve_mode_fps(self, device_index: int, mode_value: object, input_side: bool) -> float | None:
        if d is None:
            return None

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
        self._roi = Roi(480, 270, 960, 540)
        self._pending_controller_roi = None
        if self._roi_push_timer.isActive():
            self._roi_push_timer.stop()
        self._input_canvas.set_roi(self._roi)
        self._controller.set_roi(self._roi)
        self._sync_controls_from_roi(self._roi)

    def _sync_controls_from_roi(self, roi: Roi) -> None:
        self._updating_controls = True
        self.roi_x_spin.setValue(roi.x)
        self.roi_y_spin.setValue(roi.y)
        self.roi_w_spin.setValue(roi.w)
        self.roi_h_spin.setValue(roi.h)
        self.scale_spin.setValue(roi_scale_from_roi(roi))
        self._updating_controls = False

    def _update_status(self, text: str) -> None:
        self.status_label.setText(text)
        LOGGER.info("STATUS: %s", text)


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
