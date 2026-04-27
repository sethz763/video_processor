from __future__ import annotations

import logging
import math
import site
import sys
import time
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

try:
    import decklink_wrapper as d
except Exception:
    d = None

try:
    import cv2
except Exception:
    cv2 = None


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


def initialize_com_for_decklink() -> None:
    if sys.platform != "win32":
        return

    try:
        # decklink_wrapper expects MTA on this machine; using STA triggers 0x80010106 changed-mode failures.
        COINIT_MULTITHREADED = 0x0
        ole32 = ctypes.windll.ole32
        hr = ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
        # S_OK=0, S_FALSE=1 (already initialized on this thread with same model).
        if hr not in (0, 1):
            LOGGER.warning("CoInitializeEx returned hr=0x%08X", hr & 0xFFFFFFFF)
        else:
            LOGGER.info("COM initialized for DeckLink (hr=0x%08X)", hr & 0xFFFFFFFF)
    except Exception:
        LOGGER.exception("Failed to initialize COM for DeckLink")


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

        self._set_roi_and_emit(clamp_roi(new_roi))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        del event
        self._drag_mode = "none"

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
            return True
        return super().event(event)

    def _handle_touch_event(self, event: QTouchEvent) -> None:
        points = event.points()
        if not points:
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
                self._set_roi_and_emit(roi)
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
            self.set_roi(moved)

        if self._last_touch_dist is not None and self._last_touch_dist > 0:
            ratio = dist / self._last_touch_dist
            self._apply_scale(roi_scale_from_roi(self._roi) * ratio, self._widget_to_frame(center), emit_scale=False)

        self._last_touch_center = center
        self._last_touch_dist = dist

    def _apply_scale(self, new_scale: float, anchor_frame: QPointF, emit_scale: bool = True) -> None:
        new_scale = max(1.0, min(new_scale, 16.0))
        center = anchor_frame
        new_roi = roi_from_scale(new_scale, center.x(), center.y())
        self._set_roi_and_emit(new_roi, emit_scale=emit_scale)

    def _set_roi_and_emit(self, roi: Roi, emit_scale: bool = True) -> None:
        self.set_roi(roi)
        self.roiChanged.emit(roi.x, roi.y, roi.w, roi.h)
        if emit_scale:
            self.scaleChanged.emit(roi_scale_from_roi(roi))

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
        self.enable_placeholder_sr = True
        self.deinterlace_enabled = True
        self.max_auto_sr_scale = 4
        self.sr_manual_scale = 4
        self.sr_auto_mode = True
        self.processor = None

    def create(self, roi: Roi) -> None:
        sr_scale = 0 if self.sr_auto_mode else self.sr_manual_scale
        self.processor = self._module.VideoProcessor(
            width=FRAME_W,
            height=FRAME_H,
            roi_x=roi.x,
            roi_y=roi.y,
            roi_w=roi.w,
            roi_h=roi.h,
            enable_placeholder_sr=self.enable_placeholder_sr,
            sr_scale=sr_scale,
        )
        self.processor.set_max_auto_sr_scale(self.max_auto_sr_scale)
        self.processor.set_deinterlace_enabled(self.deinterlace_enabled)

    def set_roi(self, roi: Roi) -> None:
        if self.processor is not None:
            self.processor.set_roi(roi.x, roi.y, roi.w, roi.h)

    def set_auto_sr(self) -> None:
        self.sr_auto_mode = True
        if self.processor is not None and self.enable_placeholder_sr:
            self.processor.set_sr_mode_auto()

    def set_manual_sr(self, scale: int) -> None:
        self.sr_manual_scale = scale
        self.sr_auto_mode = False
        if self.processor is not None and self.enable_placeholder_sr:
            self.processor.set_sr_scale_manual(scale)

    def effective_scale(self) -> int:
        if self.processor is None or not self.enable_placeholder_sr:
            return 1
        return int(self.processor.get_effective_sr_scale())

    def set_deinterlace_enabled(self, enabled: bool) -> None:
        self.deinterlace_enabled = enabled
        if self.processor is not None:
            self.processor.set_deinterlace_enabled(enabled)

    def set_max_auto_sr_scale(self, scale: int) -> None:
        self.max_auto_sr_scale = scale
        if self.processor is not None:
            self.processor.set_max_auto_sr_scale(scale)


class MainWindow(QMainWindow):
    def __init__(self, module) -> None:
        super().__init__()
        self.setWindowTitle("video_processor GUI Test Harness")

        self._module = module
        self._source = SyntheticUyvySource()
        self._input_canvas = RoiCanvas(view_name="input")
        self._output_canvas = ImageCanvas(view_name="output")
        self._controller = VideoProcessorController(module)
        self._roi = Roi(480, 270, 960, 540)
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
        self._fullscreen_view_name: str | None = None
        self._splitter_initialized = False

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

        root.addWidget(viewers, 4)
        self._controls_panel = self._build_controls()
        self._controls_scroll = QScrollArea()
        self._controls_scroll.setWidgetResizable(True)
        self._controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._controls_scroll.setWidget(self._controls_panel)
        self._controls_scroll.setMinimumWidth(320)
        self._controls_scroll.setMaximumWidth(520)
        root.addWidget(self._controls_scroll, 1)

        self._input_canvas.set_roi(self._roi)
        self._input_canvas.roiChanged.connect(self._on_roi_from_canvas)
        self._input_canvas.scaleChanged.connect(self._on_scale_from_canvas)
        self._input_canvas.fullscreenRequested.connect(self._on_canvas_fullscreen_requested)
        self._output_canvas.fullscreenRequested.connect(self._on_canvas_fullscreen_requested)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._update_timer_interval()
        self._timer.start()

        self._setup_shortcuts()
        self._sync_controls_from_roi(self._roi)
        self.source_mode_combo.setCurrentText("Blackmagic DeckLink")
        self._source_mode = self.source_mode_combo.currentText()
        self._sync_blackmagic_controls_enabled_state()
        self._on_source_mode_changed()
        self._update_status("Ready")
        LOGGER.info("GUI initialized; default source mode=%s", self._source_mode)
        QTimer.singleShot(0, self._apply_initial_viewer_layout)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        settings_box = QGroupBox("Settings")
        settings_form = QFormLayout(settings_box)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(20)
        self.fps_spin.valueChanged.connect(self._update_timer_interval)
        settings_form.addRow("FPS", self.fps_spin)

        self.source_mode_combo = QComboBox()
        self.source_mode_combo.addItems(["Synthetic", "Blackmagic DeckLink"])
        self.source_mode_combo.currentIndexChanged.connect(self._on_source_mode_changed)
        settings_form.addRow("Input source", self.source_mode_combo)

        self.sr_mode_combo = QComboBox()
        self.sr_mode_combo.addItems(["Auto", "Manual"])
        self.sr_mode_combo.currentIndexChanged.connect(self._on_sr_mode_changed)
        settings_form.addRow("SR mode", self.sr_mode_combo)

        self.sr_manual_combo = QComboBox()
        self.sr_manual_combo.addItems(["2", "4", "8", "16"])
        self.sr_manual_combo.setCurrentText("4")
        self.sr_manual_combo.currentIndexChanged.connect(self._on_sr_manual_changed)
        settings_form.addRow("Manual SR", self.sr_manual_combo)

        self.auto_sr_max_combo = QComboBox()
        self.auto_sr_max_combo.addItems(["2", "4", "8", "16"])
        self.auto_sr_max_combo.setCurrentText("4")
        self.auto_sr_max_combo.currentIndexChanged.connect(self._on_auto_sr_max_changed)
        settings_form.addRow("Auto SR max", self.auto_sr_max_combo)

        self.enable_sr_checkbox = QCheckBox("Enable placeholder SR")
        self.enable_sr_checkbox.setChecked(True)
        self.enable_sr_checkbox.toggled.connect(self._on_enable_sr_toggled)
        settings_form.addRow(self.enable_sr_checkbox)

        self.deinterlace_checkbox = QCheckBox("Enable Bob deinterlace")
        self.deinterlace_checkbox.setChecked(True)
        self.deinterlace_checkbox.toggled.connect(self._on_deinterlace_toggled)
        settings_form.addRow(self.deinterlace_checkbox)

        self.perf_guard_checkbox = QCheckBox("Auto performance guard (reduce SR when overloaded)")
        self.perf_guard_checkbox.setChecked(False)
        self.perf_guard_checkbox.toggled.connect(self._on_perf_guard_toggled)
        settings_form.addRow(self.perf_guard_checkbox)

        decklink_box = QGroupBox("Blackmagic Video Format")
        decklink_form = QFormLayout(decklink_box)

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

        layout.addWidget(settings_box)
        layout.addWidget(decklink_box)
        layout.addWidget(roi_box)
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

            t0 = time.perf_counter()
            input_frame = self._next_input_frame()
            self._perf_add("acquire", (time.perf_counter() - t0) * 1000.0)
            if input_frame is None:
                return

            t0 = time.perf_counter()
            output_frame = self._controller.processor.process_frame(input_frame)
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
                mode_text = "Auto" if self._controller.sr_auto_mode else "Manual"
                self._update_status(
                    f"Running | FPS={fps:.1f} | SR mode={mode_text} | effective SR={self._controller.effective_scale()}"
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

                self._apply_performance_guard(fps)
        except Exception as exc:
            self._timer.stop()
            self._update_status(f"Runtime error: {exc}")

    def closeEvent(self, event) -> None:
        self._stop_decklink_sessions()
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
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
        self._controller.set_roi(self._roi)
        self._sync_controls_from_roi(self._roi)

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
        self._input_canvas.set_roi(roi)
        self._controller.set_roi(roi)
        self._sync_controls_from_roi(roi)

    def _on_sr_mode_changed(self) -> None:
        mode = self.sr_mode_combo.currentText()
        try:
            if mode == "Auto":
                self._controller.set_auto_sr()
            else:
                self._controller.set_manual_sr(int(self.sr_manual_combo.currentText()))
        except Exception as exc:
            self._update_status(f"SR mode change failed: {exc}")

    def _on_sr_manual_changed(self) -> None:
        if self.sr_mode_combo.currentText() != "Manual":
            return
        try:
            self._controller.set_manual_sr(int(self.sr_manual_combo.currentText()))
        except Exception as exc:
            self._update_status(f"Manual SR change failed: {exc}")

    def _on_auto_sr_max_changed(self) -> None:
        try:
            max_scale = int(self.auto_sr_max_combo.currentText())
            self._controller.set_max_auto_sr_scale(max_scale)
            if self.sr_mode_combo.currentText() == "Auto":
                self._controller.set_auto_sr()
            self._update_status(f"Auto SR max set to {max_scale}")
        except Exception as exc:
            self._update_status(f"Auto SR max change failed: {exc}")

    def _on_enable_sr_toggled(self, checked: bool) -> None:
        self._controller.enable_placeholder_sr = checked
        try:
            self._controller.create(self._roi)
            self._update_status("Recreated processor after placeholder SR toggle")
        except Exception as exc:
            self._update_status(f"Processor recreate failed: {exc}")

    def _on_deinterlace_toggled(self, checked: bool) -> None:
        try:
            self._controller.set_deinterlace_enabled(checked)
            mode_text = "enabled" if checked else "disabled"
            self._update_status(f"Bob deinterlace {mode_text}")
        except Exception as exc:
            self._update_status(f"Deinterlace toggle failed: {exc}")

    def _on_perf_guard_toggled(self, checked: bool) -> None:
        self._perf_guard_enabled = checked
        self._perf_guard_low_fps_seconds = 0
        self._perf_guard_last_action = ""

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

        # First mitigation: clamp SR cost by switching to manual x2.
        if self._controller.enable_placeholder_sr and (
            self._controller.sr_auto_mode or self._controller.sr_manual_scale > 2 or self._controller.effective_scale() > 2
        ):
            self._controller.set_manual_sr(2)
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
            self._update_status("Performance guard: forced Manual SR=2 to improve FPS")
            return

        # Second mitigation: disable placeholder SR if still significantly below target.
        if (
            self._controller.enable_placeholder_sr
            and self._controller.sr_manual_scale == 2
            and measured_fps < severe_threshold
            and self._perf_guard_last_action != "disable_sr"
        ):
            self.enable_sr_checkbox.setChecked(False)
            self._perf_guard_last_action = "disable_sr"
            self._perf_guard_low_fps_seconds = 0
            LOGGER.warning(
                "PERF_GUARD | fps=%.1f target=%.1f | action=disable_placeholder_sr",
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
        for dev in d.list_devices():
            if int(dev.index) == in_device:
                input_name = f"{dev.display_name} ({in_device})"
            if int(dev.index) == out_device:
                output_name = f"{dev.display_name} ({out_device})"

        self.decklink_status_label.setText(
            "DeckLink configured: "
            f"in={input_name} mode='{in_mode_name}' ({in_mode}); "
            f"out={output_name} mode='{out_mode_name}' ({out_mode}); "
            f"fps={fps_text}"
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

        modes = d.list_input_display_modes(device_index) if input_side else d.list_output_display_modes(device_index)
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
            devices = d.list_devices()
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
                input_modes = d.list_input_display_modes(in_device)
            except Exception:
                LOGGER.exception("Failed listing input modes for device %s", in_device)
                input_modes = []
            for mode in input_modes:
                fps = self._fps_from_mode(mode)
                label = f"{mode.name} ({mode.width}x{mode.height}, {fps:.2f}fps)"
                self.decklink_input_mode_combo.addItem(label, mode.mode)

        if out_device is not None:
            try:
                output_modes = d.list_output_display_modes(out_device)
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
    initialize_com_for_decklink()
    app = QApplication(sys.argv)

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
