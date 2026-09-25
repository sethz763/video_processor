from __future__ import annotations

import logging
from copy import deepcopy
import bisect
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
from statistics import median

import numpy as np
from PySide6.QtCore import QByteArray, QBuffer, QEvent, QIODevice, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QColor, QEventPoint, QIcon, QImage, QKeyEvent, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QTouchEvent, QWheelEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)
try:
    from PySide6.QtMultimedia import QMediaDevices
except ImportError:
    QMediaDevices = None

FRAME_W = 1920
FRAME_H = 1080
NODE_CHECKBOX_INDICATOR_STYLE = (
    "QCheckBox::indicator { width: 14px; height: 14px; "
    "background: #171a1f; border: 1px solid #aeb8c4; border-radius: 2px; }"
    "QCheckBox::indicator:checked { image: url(\""
    + (Path(__file__).resolve().parent / "assets" / "checkbox-check-white.svg").as_posix()
    + "\"); }"
    "QCheckBox::indicator:unchecked { image: none; }"
)
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

# Single-select upscaling mode: which stage (if any) sits on top of the
# always-available basic CUDA scaling/passthrough fallback.
SCALING_MODE_BASIC = "Standard CUDA Scaling"
SCALING_MODE_ONNX_SR = "ONNX SR"
SCALING_MODE_RTX_SR = "Nvidia SR"
SCALING_MODE_OPTIONS = [SCALING_MODE_BASIC, SCALING_MODE_ONNX_SR, SCALING_MODE_RTX_SR]

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


_SUPPORTED_SOURCE_CADENCES = (1, 2, 3, 6, 8)
_MAX_SOURCE_CADENCE_RUN_LENGTH = max(_SUPPORTED_SOURCE_CADENCES) * 2


def _detect_source_cadence(
    run_length: int,
    position_step: int = 1,
    delivery_phases: int = 1,
) -> int:
    normalized_run_length = float(max(1, run_length))
    normalized_position_step = float(max(1, position_step))
    normalized_delivery_phases = max(1, int(delivery_phases))
    if normalized_delivery_phases == 1:
        observed = normalized_run_length / normalized_position_step
    else:
        observed = (
            normalized_run_length * float(normalized_delivery_phases)
        ) / normalized_position_step
    observed = max(1.0, observed)
    return min(_SUPPORTED_SOURCE_CADENCES, key=lambda cadence: (abs(cadence - observed), cadence))


def _detect_source_cadence_stable(
    run_length: int,
    position_step: int,
    delivery_phases: int,
    previous_cadence: int,
) -> int:
    detected = _detect_source_cadence(run_length, position_step, delivery_phases)
    previous = int(previous_cadence)
    if previous not in _SUPPORTED_SOURCE_CADENCES:
        return detected
    observed = (
        float(max(1, run_length))
        * float(max(1, delivery_phases))
        / float(max(1, position_step))
    )
    if abs(observed - float(previous)) <= 1.0:
        return previous
    return detected


def _effective_synthesis_cadence(info: dict[str, object], fallback: int = 1) -> int:
    try:
        phase_step = float(info.get("cadence_phase_step", 0.0))
        delivery_phases = max(1, int(info.get("cadence_delivery_phases", 1)))
    except (TypeError, ValueError):
        phase_step = 0.0
        delivery_phases = 1
    if phase_step > 1e-6:
        observed = float(delivery_phases) / phase_step
        return min(_SUPPORTED_SOURCE_CADENCES, key=lambda cadence: (abs(float(cadence) - observed), cadence))
    try:
        return max(1, int(info.get("detected_cadence", fallback)))
    except (TypeError, ValueError):
        return max(1, int(fallback))


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


def _decklink_timecode_format_options() -> list[tuple[str, int]]:
    fallback_codes = {
        "RP188 VITC1": 0x72707631,
        "RP188 VITC2": 0x72703132,
        "RP188 LTC": 0x72706C74,
        "RP188 HFRTC": 0x72706872,
        "VITC": 0x76697463,
        "VITC Field 2": 0x76697432,
        "Serial": 0x73657269,
    }
    attribute_names = {
        "RP188 VITC1": "TIMECODE_FORMAT_RP188_VITC1",
        "RP188 VITC2": "TIMECODE_FORMAT_RP188_VITC2",
        "RP188 LTC": "TIMECODE_FORMAT_RP188_LTC",
        "RP188 HFRTC": "TIMECODE_FORMAT_RP188_HIGH_FRAME_RATE",
        "VITC": "TIMECODE_FORMAT_VITC",
        "VITC Field 2": "TIMECODE_FORMAT_VITC_FIELD2",
        "Serial": "TIMECODE_FORMAT_SERIAL",
    }
    return [
        (label, int(getattr(d, attribute_names[label], fallback_code)))
        for label, fallback_code in fallback_codes.items()
    ]


def _extract_decklink_frame_timecode_info(frame: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "present": False,
        "text": "",
        "format_code": 0,
        "format_name": "",
        "bcd": 0,
        "flags": 0,
        "hours": 0,
        "minutes": 0,
        "seconds": 0,
        "frames": 0,
        "subframe": 0,
        "field_mark": False,
        "drop_frame": False,
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
        payload["bcd"] = int(getattr(frame, "timecode_bcd", 0))
        payload["flags"] = int(getattr(frame, "timecode_flags", 0))
        payload["hours"] = int(getattr(frame, "timecode_hours", 0))
        payload["minutes"] = int(getattr(frame, "timecode_minutes", 0))
        payload["seconds"] = int(getattr(frame, "timecode_seconds", 0))
        payload["frames"] = int(getattr(frame, "timecode_frames", 0))
        payload["subframe"] = int(getattr(frame, "timecode_subframe", 0))
        payload["field_mark"] = bool(getattr(frame, "timecode_field_mark", False))
        payload["drop_frame"] = bool(getattr(frame, "timecode_drop_frame", ";" in timecode_text))
    except Exception:
        return payload
    return payload

try:
    if __package__:
        from .decklink_backend import load_decklink_backend
    else:
        from decklink_backend import load_decklink_backend
    d = load_decklink_backend()
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
    from gui.processor_worker import EffectCaptureDecoder, EffectMediaDecoder, run_processor_worker, effect_layer_updates, bypassed_effects_payload, PreviewMailbox
except Exception as exc_gui_import:
    try:
        from processor_worker import EffectCaptureDecoder, EffectMediaDecoder, run_processor_worker, effect_layer_updates, bypassed_effects_payload, PreviewMailbox
    except Exception as exc_local_import:
        run_processor_worker = None
        _worker_import_error = exc_local_import
    else:
        _worker_import_error = None
else:
    _worker_import_error = None


from gui.cadence_monitor import session_counter
from gui.input_sources import SourcePool
from gui.capture_adapters import create_input_adapter
from gui.roi_source import RoiSourceSession, rgb_to_uyvy
from gui.roi_warmup import warmup_roi_scaling
from gui.gradient_texture import GRADIENT_MODES
from gui.input_source_panel import InputSourcePanel


def _capture_backend(device_id: object) -> str:
    if isinstance(device_id, str):
        if device_id.startswith("source:"):
            return "logical"
        if device_id.startswith("webcam:"):
            return "webcam"
    return ""


def _windows_video_capture_devices() -> list[tuple[str, str]]:
    if QMediaDevices is None:
        return []
    try:
        return [
            (f"Windows Camera: {device.description()}", f"webcam:{index}")
            for index, device in enumerate(QMediaDevices.videoInputs())
        ]
    except Exception:
        LOGGER.exception("Windows camera enumeration failed")
        return []


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


def _effects_source_signature(payload: dict[str, object]) -> tuple[object, ...]:
    layers = _effect_layers_from_payload(payload)
    layer_signatures = []
    for layer in layers:
        raw_image_sources = layer.get("image_sources", [])
        image_sources = raw_image_sources if isinstance(raw_image_sources, list) else []
        image_source_signatures = tuple(
            (
                int(source.get("slot", 0)),
                str(source.get("source_node_id", "")),
                str(source.get("source_kind", "media")),
                str(source.get("media_path", "")),
                bool(source.get("media_playing", False)),
                bool(source.get("media_loop", True)),
                str(source.get("capture_kind", "")),
                source.get("capture_device_index", -1),
                int(source.get("capture_width", 0)),
                int(source.get("capture_height", 0)),
                int(source.get("capture_reload_token", 0)),
                tuple(source.get("matte_rgba", [])) if isinstance(source.get("matte_rgba"), (list, tuple)) else (),
            )
            for source in image_sources
            if isinstance(source, dict)
        )
        layer_signatures.append(
            (
                int(layer.get("layer_index", 2)),
                bool(layer.get("enabled", False)),
                str(layer.get("source_kind", "media")).strip().lower(),
                str(layer.get("capture_kind", "")).strip().lower(),
                layer.get("capture_device_index", -1),
                int(layer.get("capture_width", 0)),
                int(layer.get("capture_height", 0)),
                int(layer.get("capture_reload_token", 0)),
                str(layer.get("media_path", "")).strip(),
                bool(layer.get("media_playing", False)),
                bool(layer.get("media_loop", True)),
                image_source_signatures,
            )
        )
    return (bool(payload.get("enabled", False)), tuple(layer_signatures))


def _effect_layers_from_payload(payload: dict[str, object]) -> list[dict[str, object]]:
    raw_layers = payload.get("layers")
    if isinstance(raw_layers, list):
        layers = [
            dict(layer)
            for layer in raw_layers
            if isinstance(layer, dict) and 1 <= int(layer.get("layer_index", 0)) <= 64
        ]
        if layers:
            return sorted(layers, key=lambda layer: int(layer.get("layer_index", 2)))
    return [
        {
            "layer_index": 2,
            "enabled": bool(payload.get("enabled", False)),
            "source_kind": str(payload.get("source_kind", "media")),
            "media_path": str(payload.get("media_path", "")),
            "media_playing": bool(payload.get("media_playing", False)),
            "media_loop": bool(payload.get("media_loop", True)),
            "capture_kind": str(payload.get("capture_kind", "")),
            "capture_device_index": int(payload.get("capture_device_index", -1)),
            "capture_width": int(payload.get("capture_width", 0)),
            "capture_height": int(payload.get("capture_height", 0)),
            "matte_rgba": payload.get("matte_rgba", [255, 255, 255, 255]),
            "opacity": float(payload.get("opacity", 1.0)),
            "blend_mode": str(payload.get("blend_mode", "normal")),
            "blur_method": str(payload.get("blur_method", "off")),
            "blur_radius": float(payload.get("blur_radius", 0.0)),
            "blur_target": str(payload.get("blur_target", "both")),
            "key": {
                "mode": str(payload.get("key_mode", "off")),
                "color": [
                    int(payload.get("key_color_r", 0)),
                    int(payload.get("key_color_g", 255)),
                    int(payload.get("key_color_b", 0)),
                ],
                "similarity": float(payload.get("key_similarity", 0.25)),
                "softness": float(payload.get("key_softness", 0.10)),
                "spill_suppression": float(payload.get("spill_suppression", 0.25)),
                "luma_low": float(payload.get("luma_low", 0.0)),
                "luma_high": float(payload.get("luma_high", 1.0)),
                "luma_softness": float(payload.get("luma_softness", 0.10)),
                "edge_feather": float(payload.get("key_edge_feather", 0.0)),
                "invert": bool(payload.get("key_invert", False)),
            },
            "effect_color_from_alpha": bool(payload.get("effect_color_from_alpha", False)),
            "effect_alpha_from_color": bool(payload.get("effect_alpha_from_color", False)),
            "mask_pattern": str(payload.get("mask_pattern", "off")),
            "mask_softness": float(payload.get("mask_softness", 0.0)),
            "mask_aspect": float(payload.get("mask_aspect", 1.0)),
            "mask_invert": bool(payload.get("mask_invert", False)),
            "mask_size": float(payload.get("mask_size", 1.0)),
            "mask_x": float(payload.get("mask_x", 0.0)),
            "mask_y": float(payload.get("mask_y", 0.0)),
            "mask_rotation": float(payload.get("mask_rotation", 0.0)),
            "transform_x": float(payload.get("transform_x", 0.0)),
            "transform_y": float(payload.get("transform_y", 0.0)),
            "transform_z": float(payload.get("transform_z", 0.0)),
            "rotate_x": float(payload.get("rotate_x", 0.0)),
            "rotate_y": float(payload.get("rotate_y", 0.0)),
            "rotate_z": float(payload.get("rotate_z", 0.0)),
            "aspect_x": float(payload.get("aspect_x", 1.0)),
            "aspect_y": float(payload.get("aspect_y", 1.0)),
        }
    ]


def _set_native_effect_layer_config(processor: object, layer: dict[str, object]) -> None:
    if hasattr(processor, 'set_effect_layer_composition'):
        processor.set_effect_layer_composition(int(layer.get('layer_index', 2)), int(layer.get('composite_target', 0)), int(layer.get('composite_source', 0)))
        sources = {int(image['slot']): int(image.get('composite_source', 0)) for image in layer.get('image_sources', [])}
        for slot in range(1, 8):
            processor.set_effect_layer_composition_source(int(layer.get('layer_index', 2)), slot, sources.get(slot, 0))
    elif layer.get('composite_target') or int(layer.get('layer_index', 2)) > 8:
        raise RuntimeError('Rebuild the native module to enable composition passes and additional layers')
    key = layer.get("key", {})
    if not isinstance(key, dict):
        key = {}
    color = key.get("color", [0, 255, 0])
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        color = [0, 255, 0]
    processor.set_effect_layer_config(
        int(layer.get("layer_index", 2)),
        bool(layer.get("enabled", False)),
        opacity=float(layer.get("opacity", 1.0)),
        blend_mode=str(layer.get("blend_mode", "normal")),
        blur_method=str(layer.get("blur_method", "off")),
        blur_radius=float(layer.get("blur_radius", 0.0)),
        blur_target=str(layer.get("blur_target", "both")),
        key_mode=str(key.get("mode", "off")),
        key_color_r=int(color[0]),
        key_color_g=int(color[1]),
        key_color_b=int(color[2]),
        key_similarity=float(key.get("similarity", 0.25)),
        key_softness=float(key.get("softness", 0.10)),
        spill_suppression=float(key.get("spill_suppression", 0.25)),
        luma_low=float(key.get("luma_low", 0.0)),
        luma_high=float(key.get("luma_high", 1.0)),
        luma_softness=float(key.get("luma_softness", 0.10)),
        key_invert=bool(key.get("invert", False)),
        key_edge_feather=float(key.get("edge_feather", 0.0)),
        effect_color_from_alpha=bool(layer.get("effect_color_from_alpha", False)),
        effect_alpha_from_color=bool(layer.get("effect_alpha_from_color", False)),
        preserve_color_from_alpha_opacity=bool(layer.get("preserve_color_from_alpha_opacity", False)),
        source_from_effects_input=bool(layer.get("source_kind") == "effects_input"),
        key_alpha_from_effects_input=bool(layer.get("key_alpha_from_effects_input", False)),
        mask_pattern=str(layer.get("mask_pattern", "off")),
        mask_softness=float(layer.get("mask_softness", 0.0)),
        mask_aspect=float(layer.get("mask_aspect", 1.0)),
        mask_invert=bool(layer.get("mask_invert", False)),
        mask_size=float(layer.get("mask_size", 1.0)),
        mask_x=float(layer.get("mask_x", 0.0)),
        mask_y=float(layer.get("mask_y", 0.0)),
        mask_rotation=float(layer.get("mask_rotation", 0.0)),
        transform_x=float(layer.get("transform_x", 0.0)),
        transform_y=float(layer.get("transform_y", 0.0)),
        transform_z=float(layer.get("transform_z", 0.0)),
        rotate_x=float(layer.get("rotate_x", 0.0)),
        rotate_y=float(layer.get("rotate_y", 0.0)),
        rotate_z=float(layer.get("rotate_z", 0.0)),
        aspect_x=float(layer.get("aspect_x", 1.0)),
        aspect_y=float(layer.get("aspect_y", 1.0)),
        materialize_key_alpha=bool(layer.get("materialize_key_alpha", False)),
    )
    alpha_mix_setter = getattr(processor, "set_effect_layer_alpha_mix", None)
    alpha_mix_base = layer.get("alpha_mix_base")
    raw_alpha_mix_ops = layer.get("alpha_mix_ops", []) if bool(layer.get("enabled", False)) else []
    alpha_mix_ops = [dict(op) for op in raw_alpha_mix_ops if isinstance(op, dict)] if isinstance(raw_alpha_mix_ops, list) else []
    if callable(alpha_mix_setter):
        alpha_mix_setter(
            int(layer.get("layer_index", 2)),
            dict(alpha_mix_base) if isinstance(alpha_mix_base, dict) and bool(layer.get("enabled", False)) else None,
            alpha_mix_ops,
        )
    elif (isinstance(alpha_mix_base, dict) and bool(layer.get("enabled", False))) or alpha_mix_ops:
        raise RuntimeError("Loaded video_processor build does not support alpha mix nodes; rebuild the native module")
    layer_index = int(layer.get("layer_index", 2))
    processor.clear_effect_layer_channel_routes(layer_index)
    channel_routes = layer.get("channel_routes", [])
    if isinstance(channel_routes, list):
        for target_channel, route in enumerate(channel_routes[:4]):
            if not isinstance(route, dict):
                continue
            generator_settings = route.get("generator_settings", {})
            if not isinstance(generator_settings, dict):
                generator_settings = {}
            processor.set_effect_layer_channel_route(
                layer_index,
                target_channel,
                int(route.get("source_channel", -1)),
                blur_method=str(route.get("blur_method", "off")),
                blur_radius=float(route.get("blur_radius", 0.0)),
                transform_x=float(route.get("transform_x", 0.0)),
                transform_y=float(route.get("transform_y", 0.0)),
                transform_z=float(route.get("transform_z", 0.0)),
                rotate_x=float(route.get("rotate_x", 0.0)),
                rotate_y=float(route.get("rotate_y", 0.0)),
                rotate_z=float(route.get("rotate_z", 0.0)),
                aspect_x=float(route.get("aspect_x", 1.0)),
                aspect_y=float(route.get("aspect_y", 1.0)),
                generator_type=str(route.get("generator_type", "off")),
                key_color_r=int(generator_settings.get("key_color_r", 0)),
                key_color_g=int(generator_settings.get("key_color_g", 255)),
                key_color_b=int(generator_settings.get("key_color_b", 0)),
                key_similarity=float(generator_settings.get("key_similarity", 0.25)),
                key_softness=float(generator_settings.get("key_softness", 0.10)),
                key_invert=bool(generator_settings.get("key_invert", False)),
                mask_pattern=str(generator_settings.get("pattern", "off")),
                mask_softness=float(generator_settings.get("softness", 0.0)),
                mask_aspect=float(generator_settings.get("aspect", 1.0)),
                mask_invert=bool(generator_settings.get("invert", False)),
                mask_size=float(generator_settings.get("size", 1.0)),
                mask_x=float(generator_settings.get("x", 0.0)),
                mask_y=float(generator_settings.get("y", 0.0)),
                mask_rotation=float(generator_settings.get("rotation", 0.0)),
            )
            route_alpha_mix_setter = getattr(processor, "set_effect_layer_channel_alpha_mix", None)
            route_alpha_mix_base = route.get("alpha_mix_base")
            raw_route_alpha_mix_ops = route.get("alpha_mix_ops", [])
            route_alpha_mix_ops = [
                dict(op) for op in raw_route_alpha_mix_ops if isinstance(op, dict)
            ] if isinstance(raw_route_alpha_mix_ops, list) else []
            if callable(route_alpha_mix_setter):
                route_alpha_mix_setter(
                    layer_index,
                    target_channel,
                    dict(route_alpha_mix_base) if isinstance(route_alpha_mix_base, dict) else None,
                    route_alpha_mix_ops,
                )
            elif isinstance(route_alpha_mix_base, dict) or route_alpha_mix_ops:
                raise RuntimeError(
                    "Loaded video_processor build does not support channel alpha mix routes; rebuild the native module"
                )
    raw_color_stages = layer.get("color_stages", []) if bool(layer.get("enabled", False)) else []
    color_stages = [dict(stage) for stage in raw_color_stages if isinstance(stage, dict)] if isinstance(raw_color_stages, list) else []
    setter = getattr(processor, "set_effect_layer_color_stages", None)
    if callable(setter):
        setter(layer_index, color_stages)
    elif color_stages:
        raise RuntimeError("Loaded video_processor build does not support per-layer color stages; rebuild the native module")


def _set_native_color_stages(processor: object, payload: dict[str, object]) -> None:
    if not hasattr(processor, "set_color_stages"):
        return
    raw_stages = payload.get("color_stages", []) if bool(payload.get("enabled", False)) else []
    stages = [dict(stage) for stage in raw_stages if isinstance(stage, dict)] if isinstance(raw_stages, list) else []
    processor.set_color_stages(stages)


def _set_native_effects_input_transform(processor: object, payload: dict[str, object]) -> None:
    values = (
        float(payload.get("input_transform_x", 0.0)),
        float(payload.get("input_transform_y", 0.0)),
        float(payload.get("input_transform_z", 0.0)),
        float(payload.get("input_rotate_x", 0.0)),
        float(payload.get("input_rotate_y", 0.0)),
        float(payload.get("input_rotate_z", 0.0)),
        float(payload.get("input_aspect_x", 1.0)),
        float(payload.get("input_aspect_y", 1.0)),
    )
    setter = getattr(processor, "set_effects_input_transform", None)
    if not callable(setter):
        if any(abs(value) > 1e-6 for value in values):
            raise RuntimeError("Loaded video_processor build does not support Effects Input transforms; rebuild the native module")
        return
    setter(*values)


def _legacy_effect_source_signature(payload: dict[str, object]) -> tuple[object, ...]:
    matte_rgba = payload.get("matte_rgba", [255, 255, 255, 255])
    if not isinstance(matte_rgba, (list, tuple)):
        matte_rgba = [255, 255, 255, 255]
    return (
        bool(payload.get("enabled", False)),
        str(payload.get("source_kind", "media")).strip().lower(),
        str(payload.get("capture_kind", "")).strip().lower(),
        payload.get("capture_device_index", -1),
        int(payload.get("capture_width", 0)),
        int(payload.get("capture_height", 0)),
        str(payload.get("media_path", "")).strip(),
        bool(payload.get("media_playing", False)),
        bool(payload.get("media_loop", True)),
        tuple(int(value) for value in matte_rgba),
    )


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


def _effects_graph_from_settings(settings: dict[str, object], settings_path: Path) -> object:
    """Recover only a missing graph; never replace the user's saved choice."""
    if "effects_graph" in settings:
        return settings["effects_graph"]
    recovery_path = settings_path.with_name("app_settings.effects-recovery.json")
    try:
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return recovery.get("effects_graph") if isinstance(recovery, dict) else None


def _video_image_rect(width: int, height: int) -> QRectF:
    """Use one aspect-fit rectangle for painting and pointer/ROI mapping."""
    if width <= 1 or height <= 1:
        return QRectF(0, 0, 1, 1)
    scale = min(float(width) / FRAME_W, float(height) / FRAME_H)
    image_width, image_height = FRAME_W * scale, FRAME_H * scale
    return QRectF((width - image_width) / 2, (height - image_height) / 2, image_width, image_height)

_OUTPUT_SCHEDULE_STATE: dict[int, dict[str, object]] = {}
_RPC_E_CHANGED_MODE_HEX = "0x80010106"

_ROI_TELEMETRY_SLOT_COUNT = 16
_MANUAL_ROI_MAILBOX_SLOT_COUNT = 12
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


@dataclass
class TimecodeRoiKeyframe:
    timecode: str
    frame_number: int
    roi: Roi
    interpolation_mode: str
    timecode_format: str = ""
    drop_frame: bool = False
    field_mark: bool = False


def _store_unique_timecode_keyframe(
    keyframes: dict[int, TimecodeRoiKeyframe],
    keyframe: TimecodeRoiKeyframe,
) -> bool:
    normalized_timecode = _normalize_timecode_display(keyframe.timecode)
    matching_frames = [
        frame_number
        for frame_number, existing in keyframes.items()
        if _normalize_timecode_display(existing.timecode) == normalized_timecode
    ]
    replaced = bool(matching_frames)
    for frame_number in matching_frames:
        keyframes.pop(frame_number, None)
    keyframes[keyframe.frame_number] = keyframe
    return replaced


def _timecode_to_frame_number(timecode: str, nominal_fps: int) -> int | None:
    raw_timecode = str(timecode).strip()
    drop_frame = ";" in raw_timecode
    normalized = raw_timecode.replace(";", ":").replace(".", ":")
    parts = normalized.split(":")
    if len(parts) != 4:
        return None
    try:
        hours, minutes, seconds, frames = (int(part) for part in parts)
    except ValueError:
        return None
    fps = max(1, int(nominal_fps))
    if hours < 0 or minutes not in range(60) or seconds not in range(60) or frames not in range(fps):
        return None
    frame_number = (((hours * 60) + minutes) * 60 + seconds) * fps + frames
    if drop_frame and fps in {30, 60}:
        drop_count = 2 if fps == 30 else 4
        total_minutes = (hours * 60) + minutes
        frame_number -= drop_count * (total_minutes - (total_minutes // 10))
    return frame_number


def _timecode_count_fps(timecode_format: str, video_fps: float) -> int:
    rounded_video_fps = max(1, int(round(float(video_fps))))
    if str(timecode_format).strip() == "RP188 HFRTC":
        return rounded_video_fps
    if rounded_video_fps > 30:
        return max(1, int(round(float(video_fps) / 2.0)))
    return rounded_video_fps


def _normalize_timecode_phase_synthesis_mode(mode_name: str) -> str:
    normalized = str(mode_name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"source_cadence", "cadence", "source", "fallback", "synthesized", "synthesize"}:
        return "source_cadence"
    return "strict"


def _timecode_phase_synthesis_options() -> list[tuple[str, str]]:
    return [
        ("Strict hardware phase", "strict"),
        ("Source cadence fallback", "source_cadence"),
    ]


def _timecode_phase_multiplier(video_fps: float, count_fps: int, phase_mode: str) -> int:
    if _normalize_timecode_phase_synthesis_mode(phase_mode) != "source_cadence":
        return 1
    ratio = float(video_fps) / float(max(1, count_fps))
    rounded = int(round(ratio))
    if rounded <= 1:
        return 1
    if abs(ratio - float(rounded)) > 0.15:
        return 1
    return rounded


def _timecode_to_internal_frame_number(
    timecode: str,
    count_fps: int,
    field_mark: bool = False,
) -> int | None:
    base_frame = _timecode_to_frame_number(timecode, count_fps)
    if base_frame is None:
        return None
    if count_fps == 30:
        return (base_frame * 2) + (1 if field_mark else 0)
    if count_fps == 60:
        return base_frame
    return int(round(base_frame * (60.0 / max(1, count_fps))))


def _timecode_position_from_info(
    info: dict[str, object],
    timecode_format: str,
    video_fps: float,
    phase_mode: str,
    phase_tracker: dict[str, object] | None = None,
    source_seq: object | None = None,
    delivery_phases: int = 1,
) -> float | None:
    timecode = str(info.get("text", "")).strip()
    if not bool(info.get("present", False)) or not timecode:
        if isinstance(phase_tracker, dict):
            phase_tracker.clear()
        return None

    drop_frame = bool(info.get("drop_frame", ";" in timecode))
    calculation_timecode = _timecode_with_drop_frame_separator(timecode, drop_frame)
    count_fps = _timecode_count_fps(timecode_format, video_fps)
    field_mark = bool(info.get("field_mark", False))

    phase_multiplier = _timecode_phase_multiplier(video_fps, count_fps, phase_mode)
    normalized_format = str(timecode_format).strip()
    base_frame = _timecode_to_frame_number(calculation_timecode, count_fps)
    if base_frame is None:
        if isinstance(phase_tracker, dict):
            phase_tracker.clear()
        return None

    if count_fps == 30 and normalized_format in {"RP188 LTC", "RP188 VITC1", "RP188 VITC2"}:
        hardware_position = (base_frame * 2) + (1 if field_mark else 0)
        if _normalize_timecode_phase_synthesis_mode(phase_mode) != "source_cadence":
            if isinstance(phase_tracker, dict):
                phase_tracker.clear()
            return hardware_position
        if not isinstance(phase_tracker, dict):
            phase_tracker = {}
        if source_seq is not None and phase_tracker.get("last_source_seq") == source_seq:
            cached = phase_tracker.get("cached_frame_number")
            if isinstance(cached, (int, float)):
                return float(cached)
        last_position = phase_tracker.get("last_hardware_position")
        last_direction = int(phase_tracker.get("last_direction", 1))
        current_run_length = max(1, int(phase_tracker.get("current_run_length", 1)))
        estimated_run_length = max(1, int(phase_tracker.get("estimated_run_length", 1)))
        position_step = max(1, int(phase_tracker.get("position_step", 1)))
        run_frozen = bool(phase_tracker.get("run_frozen", False))
        if not isinstance(last_position, int):
            direction = 1
            run_index = 0
            current_run_length = 1
            run_frozen = False
        elif hardware_position != last_position:
            direction = 1 if hardware_position > last_position else -1
            observed_step = abs(hardware_position - last_position)
            if observed_step in {1, 2}:
                position_step = observed_step
                if run_frozen:
                    detected_cadence = max(1, int(phase_tracker.get("detected_cadence", 1)))
                else:
                    detected_cadence = _detect_source_cadence_stable(
                        current_run_length,
                        position_step,
                        delivery_phases,
                        int(phase_tracker.get("detected_cadence", 1)),
                    )
                estimated_run_length = detected_cadence * position_step
                estimated_run_length = max(
                    1,
                    int(round(float(estimated_run_length) / float(max(1, delivery_phases)))),
                )
            run_index = 0
            current_run_length = 1
            run_frozen = False
        else:
            direction = 1 if last_direction >= 0 else -1
            run_index = current_run_length
            if current_run_length >= _MAX_SOURCE_CADENCE_RUN_LENGTH:
                run_frozen = True
            else:
                current_run_length += 1
        phase_step = float(position_step) / float(estimated_run_length)
        max_phase = float(position_step) - phase_step
        phase = min(max_phase, float(run_index) * phase_step)
        if direction < 0:
            phase = max_phase - phase
        frame_number = hardware_position + phase
        phase_tracker["last_hardware_position"] = hardware_position
        phase_tracker["last_direction"] = direction
        phase_tracker["current_run_length"] = current_run_length
        phase_tracker["estimated_run_length"] = estimated_run_length
        phase_tracker["position_step"] = position_step
        phase_tracker["phase_step"] = phase_step
        phase_tracker["run_frozen"] = run_frozen
        phase_tracker["delivery_phases"] = max(1, int(delivery_phases))
        phase_tracker["detected_cadence"] = _detect_source_cadence(
            estimated_run_length,
            position_step,
            delivery_phases,
        )
        if source_seq is not None:
            phase_tracker["last_source_seq"] = source_seq
        phase_tracker["cached_frame_number"] = frame_number
        return frame_number
    if field_mark or phase_multiplier <= 1:
        frame_number = _timecode_to_internal_frame_number(calculation_timecode, count_fps, field_mark)
        if isinstance(phase_tracker, dict):
            phase_tracker.clear()
            if source_seq is not None:
                phase_tracker["last_source_seq"] = source_seq
                phase_tracker["cached_frame_number"] = frame_number
        return frame_number

    if isinstance(phase_tracker, dict) and source_seq is not None and phase_tracker.get("last_source_seq") == source_seq:
        cached = phase_tracker.get("cached_frame_number")
        if isinstance(cached, int):
            return cached

    if not isinstance(phase_tracker, dict):
        phase_tracker = {}

    last_base = phase_tracker.get("last_base_frame")
    last_direction = int(phase_tracker.get("last_direction", 1))
    last_multiplier = int(phase_tracker.get("last_phase_multiplier", phase_multiplier))
    current_run_length = max(1, int(phase_tracker.get("current_run_length", 1)))
    estimated_run_length = max(1, int(phase_tracker.get("estimated_run_length", phase_multiplier)))

    if not isinstance(last_base, int) or last_multiplier != phase_multiplier:
        direction = 1
        run_index = 0
        current_run_length = 1
        estimated_run_length = phase_multiplier
    elif base_frame != last_base:
        direction = 1 if base_frame > last_base else -1
        if abs(base_frame - last_base) == 1:
            estimated_run_length = current_run_length
        run_index = 0
        current_run_length = 1
    else:
        direction = 1 if last_direction >= 0 else -1
        run_index = current_run_length
        current_run_length += 1

    phase_step = float(phase_multiplier) / float(estimated_run_length)
    max_phase = float(phase_multiplier) - phase_step
    phase = min(max_phase, float(run_index) * phase_step)
    if direction < 0:
        phase = max_phase - phase

    frame_number = (base_frame * phase_multiplier) + phase
    phase_tracker["last_base_frame"] = base_frame
    phase_tracker["last_direction"] = direction
    phase_tracker["last_phase_multiplier"] = phase_multiplier
    phase_tracker["current_run_length"] = current_run_length
    phase_tracker["estimated_run_length"] = estimated_run_length
    if source_seq is not None:
        phase_tracker["last_source_seq"] = source_seq
    phase_tracker["cached_frame_number"] = frame_number
    return frame_number


def _normalize_timecode_display(timecode: str) -> str:
    normalized = str(timecode).strip().replace(";", ":").replace(".", ":")
    parts = normalized.split(":")
    if len(parts) != 4:
        return str(timecode).strip()
    return f"{parts[0]}:{parts[1]}:{parts[2]}.{parts[3]}"


def _timecode_with_drop_frame_separator(timecode: str, drop_frame: bool) -> str:
    normalized = _normalize_timecode_display(timecode)
    if drop_frame and "." in normalized:
        return normalized.rsplit(".", 1)[0] + ";" + normalized.rsplit(".", 1)[1]
    return normalized


def _infer_legacy_drop_frame(timecode: str, stored_frame_number: int, count_fps: int) -> bool:
    for candidate_fps in {max(1, count_fps), max(1, count_fps * 2)}:
        drop_frame_number = _timecode_to_frame_number(
            _timecode_with_drop_frame_separator(timecode, True),
            candidate_fps,
        )
        non_drop_frame_number = _timecode_to_frame_number(timecode, candidate_fps)
        if (
            drop_frame_number is not None
            and non_drop_frame_number is not None
            and drop_frame_number != non_drop_frame_number
            and int(stored_frame_number) == drop_frame_number
        ):
            return True
    return False


def _timecode_interpolation_curve(values: np.ndarray, interpolation_mode: str) -> np.ndarray:
    mode = str(interpolation_mode).strip().lower()
    if mode == "ease_in_out":
        return values * values * (3.0 - (2.0 * values))
    if mode == "ease_out":
        return 1.0 - np.square(1.0 - values)
    return values


def _build_timecode_roi_segment(
    start_key: TimecodeRoiKeyframe,
    end_key: TimecodeRoiKeyframe,
) -> np.ndarray:
    frame_count = max(1, int(end_key.frame_number) - int(start_key.frame_number))
    progress = np.linspace(0.0, 1.0, frame_count + 1, dtype=np.float32)
    curved = _timecode_interpolation_curve(progress, end_key.interpolation_mode)[:, np.newaxis]
    start = np.array(
        [start_key.roi.x, start_key.roi.y, start_key.roi.w, start_key.roi.h],
        dtype=np.float32,
    )
    end = np.array(
        [end_key.roi.x, end_key.roi.y, end_key.roi.w, end_key.roi.h],
        dtype=np.float32,
    )
    return start + ((end - start) * curved)


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
        buffered_count = session_counter(out, "buffered_video_frame_count")
        state = {
            "enabled": callable(schedule_fn) and callable(start_fn),
            "can_query_buffered": buffered_count is not None,
            "started": False,
            "queued_before_start": 0,
            "display_time": 0,
            "frame_duration": int(getattr(out, "frame_duration", 0)) if hasattr(out, "frame_duration") else 0,
            "time_scale": int(getattr(out, "time_scale", 0)) if hasattr(out, "time_scale") else 0,
            "minimum_preroll_frames": max(1, int(getattr(out, "minimum_preroll_frames", 1))),
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
                    state["queued_before_start"] = int(state["queued_before_start"]) + 1
                    required_preroll_frames = int(state["minimum_preroll_frames"])
                    should_start = False
                    if bool(state.get("can_query_buffered", False)):
                        try:
                            buffered_count = int(session_counter(out, "buffered_video_frame_count"))
                            effective_buffered = max(buffered_count, int(state["queued_before_start"]))
                            should_start = effective_buffered >= required_preroll_frames
                        except Exception:
                            state["can_query_buffered"] = False
                            should_start = int(state["queued_before_start"]) >= required_preroll_frames
                    else:
                        should_start = int(state["queued_before_start"]) >= required_preroll_frames

                    if should_start:
                        out.start_scheduled_playback(0, time_scale, 1.0)
                        state["started"] = True
                        state["queued_before_start"] = 0
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
    manualDragEndpointChanged = Signal(float, float, float, float)
    scaleChanged = Signal(float)
    tapCenterRequested = Signal(float, float)
    fullscreenRequested = Signal(str)
    adjustmentStarted = Signal()
    adjustmentFinished = Signal()

    def __init__(self, view_name: str = "input") -> None:
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._view_name = view_name
        self._preferred_canvas_size = QSize(0, 0)

        self._image: QImage | None = None
        self._image_backing: np.ndarray | None = None
        self._roi = Roi(0, 0, FRAME_W, FRAME_H)
        self._visual_roi_overlay_transition: tuple[float, float, float, float] | None = None
        self._visual_roi_overlay_drag: tuple[float, float, float, float] | None = None
        self._visual_roi_overlay_pinch: tuple[float, float, float, float] | None = None
        self._drag_emit_overlay: tuple[float, float, float, float] | None = None
        self._drag_emit_overlay_samples: list[tuple[float, float, float, float]] = []
        self._drag_emit_median_samples = 3

        self._drag_mode = "none"
        self._drag_started = False
        self._pointer_pressed = False
        self._adjustment_active = False
        self._drag_start_pos = QPointF()
        self._drag_start_roi = self._roi
        self._press_started_ts = 0.0
        self._tap_max_duration_s = 0.35
        self._tap_move_threshold_px = 6.0
        self._pointer_input_source = "unknown"
        self._pointer_input_last_ts = 0.0
        self._pointer_input_last_position: QPointF | None = None
        self._pointer_input_dt_ms = 0.0
        self._pointer_input_delta_px = 0.0
        self._pinch_active = False
        self._pinch_start_distance = 0.0
        self._pinch_start_scale = 1.0
        self._pinch_start_roi = self._roi
        self._pinch_start_midpoint_frame = QPointF()
        self._pinch_anchor_fraction = QPointF(0.5, 0.5)
        self._suppress_mouse_until_ts = 0.0

        self._last_touch_emit_ts = 0.0
        drag_emit_hz = max(60.0, min(120.0, float(os.environ.get("VP_MANUAL_DRAG_EMIT_HZ", "60"))))
        self._drag_move_touch_emit_interval_s = 1.0 / drag_emit_hz
        self._default_touch_emit_interval_s = self._drag_move_touch_emit_interval_s
        self._touch_emit_interval_s = self._default_touch_emit_interval_s
        self._touch_emit_pending = False
        self._touch_emit_pending_scale = False
        self._drag_x_hysteresis_px = max(0.10, min(1.20, float(os.environ.get("VP_ROI_DRAG_X_HYSTERESIS_PX", "0.45"))))
        self._interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        self._interaction_filtered_target_roi: Roi | None = None
        self._interaction_target_roi: Roi | None = None
        self._interaction_target_emit_scale = False
        self._interaction_interp_timer = QTimer(self)
        self._interaction_interp_timer.setInterval(16)
        self._interaction_interp_timer.setTimerType(Qt.PreciseTimer)
        self._interaction_interp_timer.timeout.connect(self._on_interaction_interp_tick)
        self._interaction_emit_flush_timer = QTimer(self)
        self._interaction_emit_flush_timer.setSingleShot(True)
        self._interaction_emit_flush_timer.timeout.connect(self._flush_interaction_emit)

    def set_image(self, image: QImage, backing: np.ndarray | None = None) -> None:
        self._image = image
        self._image_backing = backing
        self.update()

    def set_preferred_canvas_size(self, width: int, height: int) -> None:
        preferred = QSize(max(0, int(width)), max(0, int(height)))
        if self._preferred_canvas_size == preferred:
            return
        self._preferred_canvas_size = preferred
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        if self._preferred_canvas_size.width() > 0 and self._preferred_canvas_size.height() > 0:
            return QSize(self._preferred_canvas_size)
        return super().sizeHint()

    def minimumSizeHint(self) -> QSize:
        return QSize(0, 0)

    def set_roi(self, roi: Roi) -> None:
        self._cancel_interaction_interpolation()
        self._visual_roi_overlay_transition = None
        self._visual_roi_overlay_drag = None
        self._drag_emit_overlay = None
        self._drag_emit_overlay_samples.clear()
        self._visual_roi_overlay_pinch = None
        self._apply_roi_local(roi)
        self._interaction_target_roi = roi

    def cancel_pending_interaction_updates(self) -> None:
        # Drop queued interaction/touch emits so old manual gestures cannot
        # override a programmatic keyframe recall endpoint.
        self._cancel_interaction_interpolation()
        self._touch_emit_pending = False
        self._touch_emit_pending_scale = False

    def reset_interaction_state(self) -> None:
        self.cancel_pending_interaction_updates()
        self._pointer_pressed = False
        self._drag_mode = "none"
        self._drag_started = False
        self._pinch_active = False
        self._pinch_start_distance = 0.0
        self._adjustment_active = False
        self._visual_roi_overlay_drag = None
        self._drag_emit_overlay = None
        self._drag_emit_overlay_samples.clear()
        self._visual_roi_overlay_pinch = None
        self._visual_roi_overlay_transition = None
        self.update()

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
        self._drag_emit_overlay_samples.append(self._visual_roi_overlay_drag)
        if len(self._drag_emit_overlay_samples) > self._drag_emit_median_samples:
            del self._drag_emit_overlay_samples[:-self._drag_emit_median_samples]
        coalesced = tuple(
            float(median(sample[index] for sample in self._drag_emit_overlay_samples))
            for index in range(4)
        )
        if coalesced != self._drag_emit_overlay:
            self._drag_emit_overlay = coalesced
            self.manualDragEndpointChanged.emit(*coalesced)
        self.update()

    def _clear_drag_visual_roi_overlay(self) -> None:
        had_overlay = self._visual_roi_overlay_drag is not None
        self._visual_roi_overlay_drag = None
        self._drag_emit_overlay = None
        self._drag_emit_overlay_samples.clear()
        if had_overlay:
            self.update()

    def drag_visual_roi_overlay(self) -> tuple[float, float, float, float] | None:
        if self._visual_roi_overlay_drag is None:
            return None
        return self._drag_emit_overlay or self._visual_roi_overlay_drag

    def is_move_drag_active(self) -> bool:
        return bool(self._drag_started and self._drag_mode == "move")

    def pinch_visual_roi_overlay(self) -> tuple[float, float, float, float] | None:
        return self._visual_roi_overlay_pinch

    def is_pinch_active(self) -> bool:
        return bool(self._pinch_active)

    def pointer_input_diagnostics(self) -> dict[str, object]:
        return {
            "source": str(self._pointer_input_source),
            "event_dt_ms": float(self._pointer_input_dt_ms),
            "event_delta_px": float(self._pointer_input_delta_px),
        }

    def _record_pointer_input(self, source: str, position: QPointF) -> None:
        now = time.perf_counter()
        previous = self._pointer_input_last_position
        self._pointer_input_source = str(source)
        self._pointer_input_dt_ms = (
            (now - float(self._pointer_input_last_ts)) * 1000.0
            if self._pointer_input_last_ts > 0.0
            else 0.0
        )
        self._pointer_input_delta_px = (
            math.hypot(position.x() - previous.x(), position.y() - previous.y())
            if previous is not None
            else 0.0
        )
        self._pointer_input_last_ts = now
        self._pointer_input_last_position = QPointF(position)

    def roi(self) -> Roi:
        return self._roi

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

        overlay = self._visual_roi_overlay_pinch
        if overlay is None:
            overlay = self._visual_roi_overlay_drag
        if overlay is None:
            overlay = self._visual_roi_overlay_transition

        if overlay is not None:
            overlay_x, overlay_y, overlay_w, overlay_h = overlay
            roi_rect_w = self._frame_to_widget_rect_float(overlay_x, overlay_y, overlay_w, overlay_h)
            display_roi = clamp_roi(
                Roi(
                    int(round(overlay_x)),
                    int(round(overlay_y)),
                    int(round(overlay_w)),
                    int(round(overlay_h)),
                )
            )
        else:
            roi_rect_w = self._frame_to_widget_rect(self._roi)
            display_roi = self._roi
        p.setRenderHint(QPainter.Antialiasing, True)

        p.setPen(QPen(Qt.yellow, 2))
        p.drawRect(roi_rect_w)

        p.setPen(QPen(Qt.green, 1))
        scale = roi_scale_from_roi(display_roi)
        p.drawText(12, 24, f"ROI: x={display_roi.x} y={display_roi.y} w={display_roi.w} h={display_roi.h}")
        p.drawText(12, 44, f"Scale: {scale:.2f}x")

        # Keep the handle in the bottom-right corner without consuming tiny ROIs.
        p.fillRect(self._resize_handle_rect(roi_rect_w), Qt.yellow)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        step = 8
        resize_step = 16
        roi = self._roi

        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self._begin_adjustment()
            self._apply_scale(roi_scale_from_roi(roi) * 1.08, self._roi_center())
            return
        if event.key() == Qt.Key_Minus:
            self._begin_adjustment()
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

        self._begin_adjustment()
        self._set_roi_and_emit(clamp_roi(roi))

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        adjustment_keys = {
            Qt.Key_Plus,
            Qt.Key_Equal,
            Qt.Key_Minus,
            Qt.Key_Left,
            Qt.Key_Right,
            Qt.Key_Up,
            Qt.Key_Down,
        }
        if event.key() in adjustment_keys and not event.isAutoRepeat():
            self._finish_adjustment()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton or self._pinch_active or time.perf_counter() < self._suppress_mouse_until_ts:
            return

        mouse_source = getattr(event.source(), "name", str(event.source()))
        self._record_pointer_input(f"mouse:{mouse_source}", event.position())
        self._pointer_press(event.position())
        event.accept()

    def _pointer_press(self, position: QPointF) -> None:
        self._cancel_interaction_interpolation()
        self.setFocus(Qt.MouseFocusReason)
        self._pointer_pressed = True
        self._drag_start_pos = QPointF(position)
        self._drag_start_roi = self._roi
        self._drag_started = False
        self._press_started_ts = time.perf_counter()

        roi_rect = self._frame_to_widget_rect(self._roi)
        handle_rect = self._resize_handle_rect(roi_rect)

        if handle_rect.contains(position):
            self._drag_mode = "resize"
        elif roi_rect.contains(position):
            self._drag_mode = "move"
        else:
            self._drag_mode = "none"

    def _start_pointer_drag(self) -> None:
        if self._drag_mode == "none" or self._drag_started:
            return
        self._drag_started = True
        self._begin_adjustment()
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
        if self._pinch_active or time.perf_counter() < self._suppress_mouse_until_ts:
            return
        mouse_source = getattr(event.source(), "name", str(event.source()))
        self._record_pointer_input(f"mouse:{mouse_source}", event.position())
        self._pointer_move(event.position())
        event.accept()

    def _pointer_move(self, position: QPointF) -> None:
        dx = position.x() - self._drag_start_pos.x()
        dy = position.y() - self._drag_start_pos.y()
        if not self._drag_started:
            if math.hypot(dx, dy) < self._tap_move_threshold_px:
                return
            if self._drag_mode == "none":
                return
            self._start_pointer_drag()

        if self._drag_mode == "none":
            return

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
        )

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return
        if self._pinch_active or time.perf_counter() < self._suppress_mouse_until_ts:
            event.accept()
            return
        self._pointer_release(event.position())
        event.accept()

    def _pointer_release(self, position: QPointF) -> None:
        if not self._pointer_pressed:
            return
        self._pointer_pressed = False
        elapsed = time.perf_counter() - self._press_started_ts
        moved = math.hypot(
            position.x() - self._drag_start_pos.x(),
            position.y() - self._drag_start_pos.y(),
        )
        is_tap = not self._drag_started and elapsed <= self._tap_max_duration_s and moved < self._tap_move_threshold_px
        had_drag = self._drag_started
        finishing_move = had_drag and self._drag_mode == "move"
        if finishing_move:
            self._pointer_move(position)
            self._drag_emit_overlay = self._visual_roi_overlay_drag
            self._drag_emit_overlay_samples.clear()
            if self._drag_emit_overlay is not None:
                self.manualDragEndpointChanged.emit(*self._drag_emit_overlay)
        self._drag_mode = "none"
        self._drag_started = False
        self._touch_emit_interval_s = self._default_touch_emit_interval_s
        if had_drag:
            self._flush_interaction_emit()
        self._clear_drag_visual_roi_overlay()
        if had_drag:
            self._finish_adjustment()
        if is_tap:
            frame_point = self._widget_to_frame(position)
            self.tapCenterRequested.emit(frame_point.x(), frame_point.y())

    def _begin_pinch(self, first: QPointF, second: QPointF) -> None:
        self._cancel_interaction_interpolation()
        self._pointer_pressed = False
        self._drag_mode = "none"
        self._drag_started = False
        self._clear_drag_visual_roi_overlay()
        self._visual_roi_overlay_transition = None
        self._pinch_active = True
        self._pinch_start_distance = max(1.0, math.hypot(second.x() - first.x(), second.y() - first.y()))
        self._pinch_start_roi = self._roi
        self._pinch_start_scale = roi_scale_from_roi(self._pinch_start_roi)
        midpoint = QPointF((first.x() + second.x()) * 0.5, (first.y() + second.y()) * 0.5)
        self._pinch_start_midpoint_frame = self._widget_to_frame(midpoint)
        self._pinch_anchor_fraction = QPointF(
            (self._pinch_start_midpoint_frame.x() - float(self._pinch_start_roi.x))
            / max(1.0, float(self._pinch_start_roi.w)),
            (self._pinch_start_midpoint_frame.y() - float(self._pinch_start_roi.y))
            / max(1.0, float(self._pinch_start_roi.h)),
        )
        self._visual_roi_overlay_pinch = (
            float(self._roi.x),
            float(self._roi.y),
            float(self._roi.w),
            float(self._roi.h),
        )
        self._touch_emit_interval_s = self._default_touch_emit_interval_s
        self._begin_adjustment()
        self.update()

    def _update_pinch(self, first: QPointF, second: QPointF) -> None:
        distance = max(1.0, math.hypot(second.x() - first.x(), second.y() - first.y()))
        midpoint = QPointF((first.x() + second.x()) * 0.5, (first.y() + second.y()) * 0.5)
        target_scale = self._pinch_start_scale * (distance / self._pinch_start_distance)
        midpoint_frame = self._widget_to_frame(midpoint)
        sized_roi = roi_from_scale(max(1.0, min(target_scale, 16.0)), 0.0, 0.0)
        target_roi = clamp_roi(
            Roi(
                int(round(midpoint_frame.x() - (self._pinch_anchor_fraction.x() * float(sized_roi.w)))),
                int(round(midpoint_frame.y() - (self._pinch_anchor_fraction.y() * float(sized_roi.h)))),
                sized_roi.w,
                sized_roi.h,
            )
        )
        self._visual_roi_overlay_pinch = (
            float(target_roi.x),
            float(target_roi.y),
            float(target_roi.w),
            float(target_roi.h),
        )
        self.update()
        self._queue_interpolated_roi(
            target_roi,
            emit_scale=True,
            anchor_to_current=True,
        )

    def _finish_pinch(self) -> None:
        if not self._pinch_active:
            return
        self._pinch_active = False
        self._pinch_start_distance = 0.0
        self._suppress_mouse_until_ts = time.perf_counter() + 0.25
        self._visual_roi_overlay_pinch = None
        self._visual_roi_overlay_transition = None
        self._flush_interaction_emit()
        self.update()
        self._finish_adjustment()

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
        self._begin_adjustment()

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
        if et in (QEvent.Type.TouchBegin, QEvent.Type.TouchUpdate, QEvent.Type.TouchEnd, QEvent.Type.TouchCancel):
            self._suppress_mouse_until_ts = time.perf_counter() + 0.25
            points = [point for point in event.points() if point.state() != QEventPoint.State.Released]
            if len(points) >= 2:
                first = points[0].position()
                second = points[1].position()
                if not self._pinch_active:
                    self._begin_pinch(first, second)
                self._update_pinch(first, second)
            elif self._pinch_active:
                self._finish_pinch()
            elif et == QEvent.Type.TouchBegin and len(points) == 1:
                self._record_pointer_input("touch", points[0].position())
                self._pointer_press(points[0].position())
            elif et == QEvent.Type.TouchUpdate and len(points) == 1:
                self._record_pointer_input("touch", points[0].position())
                self._pointer_move(points[0].position())
            elif et == QEvent.Type.TouchEnd:
                released_points = event.points()
                release_position = released_points[0].position() if released_points else self._drag_start_pos
                self._pointer_release(release_position)
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
    ) -> None:
        raw_target = clamp_roi(target_roi)
        if anchor_to_current:
            self._interaction_filtered_target_roi = self._roi
            self._interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        self._interaction_filtered_target_roi = raw_target
        self._interaction_target_roi = raw_target
        self._interaction_target_emit_scale = self._interaction_target_emit_scale or emit_scale
        if not self._interaction_interp_timer.isActive():
            self._interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        target_scale = roi_scale_from_roi(self._interaction_target_roi)
        if target_scale >= 6.0:
            base_interval_ms = 8
        elif target_scale >= 4.0:
            base_interval_ms = 10
        else:
            base_interval_ms = 16
        self._interaction_interp_timer.setInterval(base_interval_ms)
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

        if zoom_scale >= 6.0:
            lag_limit = 6
        elif zoom_scale >= 4.0:
            lag_limit = 10
        else:
            lag_limit = 14

        near_target_deadband = 1

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

    def _begin_adjustment(self) -> None:
        if self._adjustment_active:
            return
        self._adjustment_active = True
        self.adjustmentStarted.emit()

    def _finish_adjustment(self) -> None:
        if not self._adjustment_active:
            return
        self._adjustment_active = False
        self.adjustmentFinished.emit()

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
        if self._drag_mode == "none" and not self._pinch_active:
            self._finish_adjustment()

    def _schedule_interaction_emit_flush(self) -> None:
        # Trailing-edge flush ensures the final zoom state is always propagated.
        self._interaction_emit_flush_timer.start(40)

    def _roi_center(self) -> QPointF:
        return QPointF(self._roi.x + (self._roi.w / 2.0), self._roi.y + (self._roi.h / 2.0))

    def _image_rect(self) -> QRectF:
        return _video_image_rect(self.width(), self.height())

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
    tapCenterRequested = Signal(float, float)

    def __init__(self, view_name: str = "output") -> None:
        super().__init__()
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._image: QImage | None = None
        self._image_backing: np.ndarray | None = None
        self._view_name = view_name
        self._preferred_canvas_size = QSize(0, 0)
        self._press_position: QPointF | None = None
        self._press_started_ts = 0.0
        self._suppress_mouse_until_ts = 0.0

    def set_image(self, image: QImage, backing: np.ndarray | None = None) -> None:
        self._image = image
        self._image_backing = backing
        self.update()

    def set_preferred_canvas_size(self, width: int, height: int) -> None:
        preferred = QSize(max(0, int(width)), max(0, int(height)))
        if self._preferred_canvas_size == preferred:
            return
        self._preferred_canvas_size = preferred
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        if self._preferred_canvas_size.width() > 0 and self._preferred_canvas_size.height() > 0:
            return QSize(self._preferred_canvas_size)
        return super().sizeHint()

    def minimumSizeHint(self) -> QSize:
        return QSize(0, 0)

    def paintEvent(self, event) -> None:
        del event
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        if self._image is None:
            return
        p.drawImage(self._image_rect(), self._image)

    def _image_rect(self) -> QRectF:
        return _video_image_rect(self.width(), self.height())

    def _emit_tap(self, position: QPointF) -> None:
        image_rect = self._image_rect()
        frame_x = (position.x() - image_rect.left()) * FRAME_W / max(1.0, image_rect.width())
        frame_y = (position.y() - image_rect.top()) * FRAME_H / max(1.0, image_rect.height())
        self.tapCenterRequested.emit(
            max(0.0, min(float(FRAME_W), frame_x)),
            max(0.0, min(float(FRAME_H), frame_y)),
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and time.perf_counter() >= self._suppress_mouse_until_ts:
            self._press_position = QPointF(event.position())
            self._press_started_ts = time.perf_counter()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        press_position = self._press_position
        self._press_position = None
        if event.button() == Qt.LeftButton and press_position is not None:
            elapsed = time.perf_counter() - self._press_started_ts
            moved = math.hypot(event.position().x() - press_position.x(), event.position().y() - press_position.y())
            if elapsed <= 0.35 and moved < 6.0:
                self._emit_tap(event.position())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def event(self, event) -> bool:
        if event.type() in (QEvent.Type.TouchBegin, QEvent.Type.TouchEnd, QEvent.Type.TouchCancel):
            self._suppress_mouse_until_ts = time.perf_counter() + 0.25
            points = event.points()
            if event.type() == QEvent.Type.TouchBegin and len(points) == 1:
                self._press_position = QPointF(points[0].position())
                self._press_started_ts = time.perf_counter()
            elif event.type() == QEvent.Type.TouchEnd and self._press_position is not None and points:
                release_position = points[0].position()
                elapsed = time.perf_counter() - self._press_started_ts
                moved = math.hypot(
                    release_position.x() - self._press_position.x(),
                    release_position.y() - self._press_position.y(),
                )
                self._press_position = None
                if elapsed <= 0.35 and moved < 6.0:
                    self._emit_tap(release_position)
            else:
                self._press_position = None
            event.accept()
            return True
        return super().event(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        super().mouseDoubleClickEvent(event)


class EffectsPortButton(QPushButton):
    dragStarted = Signal(QPointF)
    dragMoved = Signal(QPointF)
    dragFinished = Signal(QPointF, bool)

    def __init__(self, parent: QWidget) -> None:
        super().__init__("", parent)
        self._drag_origin: QPointF | None = None
        self._drag_moved = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_origin = event.globalPosition()
            self._drag_moved = False
            self.dragStarted.emit(event.globalPosition())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin is not None and bool(event.buttons() & Qt.LeftButton):
            delta = event.globalPosition() - self._drag_origin
            self._drag_moved = self._drag_moved or math.hypot(delta.x(), delta.y()) >= QApplication.startDragDistance()
            self.dragMoved.emit(event.globalPosition())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._drag_origin is not None:
            moved = self._drag_moved
            self._drag_origin = None
            self._drag_moved = False
            if not moved:
                self.click()
            self.dragFinished.emit(event.globalPosition(), moved)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ExpandingDoubleSpinBox(QDoubleSpinBox):
    def validate(self, text, position):
        from PySide6.QtGui import QValidator
        raw = text.removesuffix(self.suffix()).strip()
        value, valid = self.locale().toDouble(raw)
        if valid and math.isfinite(value):
            return QValidator.Acceptable, text, position
        return super().validate(text, position)

    def valueFromText(self, text):
        value, valid = self.locale().toDouble(text.removesuffix(self.suffix()).strip())
        if valid and math.isfinite(value):
            self.setRange(min(self.minimum(), value), max(self.maximum(), value))
            return value
        return super().valueFromText(text)


class EffectsNodeWidget(QWidget):
    portClicked = Signal(str, str)
    portDisconnectRequested = Signal(str, str)
    nodeMoved = Signal()
    selectionRequested = Signal(str, bool)
    nodeDragStarted = Signal(str)
    nodeDragFinished = Signal(str)
    settingsRequested = Signal(str)
    deleteRequested = Signal(str)
    blurLevelChanged = Signal(str, float)
    captureDeviceChanged = Signal(str, object, str)
    gradientColorRequested = Signal(str, str)
    captureSettingsRequested = Signal(str)
    captureReloadRequested = Signal(str)
    mediaPathChanged = Signal(str, str)
    mediaPlaybackChanged = Signal(str, bool, bool)
    mediaKeyframePlaybackArmed = Signal(str, str, bool)
    compositorSettingChanged = Signal(str, str, object)
    colorSettingChanged = Signal(str, str, object)
    compositorLayerActionRequested = Signal(str, str)
    portDragStarted = Signal(str, str, QPointF)
    portDragMoved = Signal(str, str, QPointF)
    portDragFinished = Signal(str, str, QPointF, bool)

    def __init__(self, node_id: str, node_type: str, title: str, parent: QWidget) -> None:
        super().__init__(parent)
        self.node_id = node_id
        self.node_type = node_type
        self._drag_offset: QPointF | None = None
        self._selected = False
        node_sizes = {
            "keying": (380, 520),
            "media": (300, 154),
            "capture": (250, 154),
            "mix": (250, 154),
            "gradient": (250, 154),
            "transform_3d": (250, 154),
            "color_splitter": (250, 154),
            "color_recombiner": (280, 184),
        }
        self.setFixedSize(*node_sizes.get(node_type, (250, 124)))
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip("Drag nodes to position. Drag output ports to inputs. Right-click a port to disconnect.")

        reset_all = QPushButton("Reset all", self)
        reset_all.setGeometry(self.width() - 78, 10, 68, 24)
        reset_all.clicked.connect(lambda: parent._reset_node_settings(self.node_id))
        title_label = QLabel(title, self)
        title_label.setGeometry(14, 10, self.width() - 96, 24)
        title_label.setStyleSheet("font-weight: 600; color: #f4f4f4;")

        self._status_label = QLabel(self)
        self._status_label.setGeometry(14, 39, 220, 30)
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #b9c0c8; font-size: 10px;")

        self._ports: dict[str, QPushButton] = {}
        self._port_labels: dict[str, QLabel] = {}
        output_x = self.width() - 20
        if node_type == "color_recombiner":
            self._add_port("red_input", 0, 68, "Red channel input")
            self._add_port("green_input", 0, 98, "Green channel input")
            self._add_port("blue_input", 0, 128, "Blue channel input")
            self._add_port("alpha_input", 0, 158, "Optional alpha channel input")
        elif node_type == "mix":
            self._add_port("a_input", 0, 68, "Primary alpha input")
            self._add_port("b_input", 0, 128, "Secondary alpha input")
        elif node_type in {"chroma_key", "luma_key"}:
            self._add_port("color_input", 0, 98, "Color input")
        elif node_type not in {"effects_input", "keying", "capture", "media", "composition", "matte", "mask", "gradient", "mix"}:
            self._add_port("input", 0, 98, "Input")
        if node_type == "color_splitter":
            self._add_port("red_output", output_x, 68, "Red channel")
            self._add_port("green_output", output_x, 98, "Green channel")
            self._add_port("blue_output", output_x, 128, "Blue channel")
        elif node_type == "color_recombiner":
            self._add_port("output", output_x, 98, "Recombined color")
            self._add_port("alpha_output", output_x, 128, "Recombined alpha")
        elif node_type == "transform_3d":
            self._add_port("alpha_input", 0, 128, "Alpha input")
            self._add_port("output", output_x, 98, "Transformed color")
            self._add_port("alpha_output", output_x, 128, "Generated alpha")
        elif node_type == "mix":
            self._add_port("alpha_output", output_x, 98, "Mixed alpha")
        elif node_type == "gradient":
            self._add_port("output", output_x, 98, "Gradient color")
        elif node_type in {"mask", "chroma_key", "luma_key"}:
            self._add_port("alpha_output", output_x, 98, "Key alpha" if node_type == "chroma_key" else "Mask alpha")
        elif node_type not in {"effects_output", "keying"}:
            self._add_port("output", output_x, 128 if node_type == "media" else 98, "Color output")
        if node_type == "composition":
            self._add_port("alpha_output", output_x, 68, "Composition alpha")
        if node_type == "media":
            self._add_port("alpha_output", output_x, 98, "Source alpha")
            self.setAcceptDrops(True)
            self.media_browse = QPushButton("...", self)
            self.media_browse.setGeometry(30, 72, 32, 24)
            self.media_browse.setToolTip("Choose video, image, or image sequence")
            self.media_browse.clicked.connect(lambda: self.settingsRequested.emit(self.node_id))
            self.media_play = QPushButton("Play", self)
            self.media_play.setGeometry(68, 72, 54, 24)
            self.media_play.setCheckable(True)
            self.media_play.setToolTip("Play or pause media")
            self.media_play.toggled.connect(self._emit_media_playback)
            self.media_loop = QCheckBox("Loop", self)
            self.media_loop.setStyleSheet(
                "QCheckBox { color: #ffffff; background: transparent; }" + NODE_CHECKBOX_INDICATOR_STYLE
            )
            self.media_loop.setGeometry(128, 72, 58, 24)
            self.media_loop.setChecked(True)
            self.media_loop.toggled.connect(self._emit_media_playback)
            self.media_key_play = QPushButton("Key Play", self)
            self.media_key_play.setGeometry(30, 104, 82, 24)
            self.media_key_play.setCheckable(True)
            self.media_key_play.setToolTip("Toggle Play state arming for the next keyframe")
            self.media_key_play.setStyleSheet(
                "QPushButton:checked { background: #805d12; color: white; font-weight: 600; }"
            )
            self.media_key_play.toggled.connect(
                lambda checked: self._set_media_keyframe_playback_arm("play", checked)
            )
            self.media_key_pause = QPushButton("Key Pause", self)
            self.media_key_pause.setGeometry(118, 104, 88, 24)
            self.media_key_pause.setCheckable(True)
            self.media_key_pause.setToolTip("Toggle Pause state arming for the next keyframe")
            self.media_key_pause.setStyleSheet(
                "QPushButton:checked { background: #805d12; color: white; font-weight: 600; }"
            )
            self.media_key_pause.toggled.connect(
                lambda checked: self._set_media_keyframe_playback_arm("pause", checked)
            )
        elif node_type == "matte":
            self._add_port("alpha_output", output_x, 68, "Matte alpha")
            self.matte_swatch = QPushButton(self)
            self.matte_swatch.setGeometry(30, 72, 72, 24)
            self.matte_swatch.setToolTip("Choose matte color")
            self.matte_swatch.clicked.connect(lambda: self.settingsRequested.emit(self.node_id))
        elif node_type in {"mask", "gradient"}:
            self.mask_settings = QPushButton("Pattern...", self)
            if node_type == "gradient":
                self.mask_settings.setText("Gradient...")
                self.gradient_swatches = {}
                for index, key in enumerate(("color_a", "color_b")):
                    swatch = QPushButton(self)
                    swatch.setGeometry(30 + index * 54, 108, 46, 24)
                    swatch.setToolTip("Choose gradient color " + ("A" if index == 0 else "B"))
                    swatch.clicked.connect(lambda checked=False, key=key: self.gradientColorRequested.emit(self.node_id, key))
                    self.gradient_swatches[key] = swatch
            self.mask_settings.setGeometry(30, 72, 92, 24)
            self.mask_settings.clicked.connect(lambda: self.settingsRequested.emit(self.node_id))
        elif node_type == "blur":
            self._add_port("alpha_input", 0, 68, "Alpha input")
            self._add_port("alpha_output", output_x, 68, "Blurred alpha")
            self.blur_level = QDoubleSpinBox(self)
            self.blur_level.setGeometry(55, 72, 100, 24)
            self.blur_level.setRange(0.0, 128.0)
            self.blur_level.setDecimals(2)
            self.blur_level.setSingleStep(0.25)
            self.blur_level.setSuffix(" px")
            self.blur_level.setKeyboardTracking(False)
            self.blur_level.setToolTip("GPU blur radius")
            reset = QPushButton(self)
            reset.setIcon(QIcon(str(Path(__file__).parent / "assets" / "reset.svg")))
            reset.setToolTip("Reset radius")
            reset.setGeometry(160, 72, 26, 24)
            reset.clicked.connect(lambda: self.blur_level.setValue(3.0))
            self.blur_level.valueChanged.connect(
                lambda value: self.blurLevelChanged.emit(self.node_id, float(value))
            )
        elif node_type == "capture":
            self._capture_devices: list[tuple[str, object]] = []
            self._selected_capture_device: object = None
            self.capture_device_label = QLabel("No source selected", self)
            self.capture_device_label.setGeometry(14, 70, 206, 58)
            self.capture_device_label.setWordWrap(True)
            self.capture_device_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.capture_device_label.setStyleSheet("color: #f4f4f4; font-size: 11px;")
        elif node_type in {"proc_amp", "color_corrector", "chroma_key", "luma_key", "transform_3d", "mix"}:
            self.adjust_settings = QPushButton("Adjust...", self)
            self.adjust_settings.setGeometry(76, 72, 98, 26)
            self.adjust_settings.clicked.connect(lambda: self.settingsRequested.emit(self.node_id))
        elif node_type == "keying":
            self.compositor_blends: dict[int, QComboBox] = {}
            self.compositor_opacities: dict[int, QSlider] = {}
            self._ensure_compositor_layers(2)
            self._add_port("output", 360, 250, "Composite color")
            self._add_port("alpha_output", 360, 280, "Composite alpha")
            self.add_layer_top = QPushButton("+ Top", self)
            self.add_layer_top.setGeometry(14, 72, 70, 26)
            self.remove_layer_top = QPushButton("- Top", self)
            self.remove_layer_top.setGeometry(88, 72, 70, 26)
            self.add_layer_bottom = QPushButton("+ Bottom", self)
            self.add_layer_bottom.setGeometry(162, 72, 82, 26)
            self.remove_layer_bottom = QPushButton("- Bottom", self)
            self.remove_layer_bottom.setGeometry(248, 72, 82, 26)
            self.add_layer_top.clicked.connect(
                lambda: self.compositorLayerActionRequested.emit(self.node_id, "add_top")
            )
            self.remove_layer_top.clicked.connect(
                lambda: self.compositorLayerActionRequested.emit(self.node_id, "remove_top")
            )
            self.add_layer_bottom.clicked.connect(
                lambda: self.compositorLayerActionRequested.emit(self.node_id, "add_bottom")
            )
            self.remove_layer_bottom.clicked.connect(
                lambda: self.compositorLayerActionRequested.emit(self.node_id, "remove_bottom")
            )
            self.layer1_blend = self.compositor_blends[1]
            self.layer2_blend = self.compositor_blends[2]
            self.layer1_opacity = self.compositor_opacities[1]
            self.layer2_opacity = self.compositor_opacities[2]
            for slider in self.compositor_opacities.values():
                slider.setRange(0, 100)
                slider.setValue(100)
        self._base_size = self.size()
        self._base_child_geometries = {
            child: child.geometry() for child in self.findChildren(QWidget, options=Qt.FindDirectChildrenOnly)
        }

    def _emit_media_playback(self) -> None:
        play = getattr(self, "media_play", None)
        loop = getattr(self, "media_loop", None)
        if play is None or loop is None:
            return
        play.setText("Pause" if play.isChecked() else "Play")
        self.mediaPlaybackChanged.emit(self.node_id, play.isChecked(), loop.isChecked())

    def _set_media_keyframe_playback_arm(self, state: str, checked: bool) -> None:
        opposite = self.media_key_pause if state == "play" else self.media_key_play
        if checked and opposite.isChecked():
            opposite.blockSignals(True)
            opposite.setChecked(False)
            opposite.blockSignals(False)
        self.mediaKeyframePlaybackArmed.emit(self.node_id, state, bool(checked))

    def set_media_state(self, playing: bool, loop: bool, has_media: bool = True) -> None:
        play = getattr(self, "media_play", None)
        loop_control = getattr(self, "media_loop", None)
        if play is None or loop_control is None:
            return
        play.blockSignals(True)
        loop_control.blockSignals(True)
        play.setChecked(bool(playing))
        play.setText("Pause" if playing else "Play")
        play.setEnabled(bool(has_media))
        loop_control.setChecked(bool(loop))
        play.blockSignals(False)
        loop_control.blockSignals(False)

    def set_media_keyframe_playback_arm(self, state: str | None) -> None:
        key_play = getattr(self, "media_key_play", None)
        key_pause = getattr(self, "media_key_pause", None)
        if key_play is None or key_pause is None:
            return
        key_play.blockSignals(True)
        key_pause.blockSignals(True)
        key_play.setChecked(state == "play")
        key_pause.setChecked(state == "pause")
        key_play.blockSignals(False)
        key_pause.blockSignals(False)

    def _ensure_compositor_layers(self, count: int) -> None:
        blend_modes = ["normal", "multiply", "screen", "overlay", "soft_light", "hard_light", "difference", "additive_alpha"]
        for layer_index in range(len(self.compositor_blends) + 1, count + 1):
            row_y = 112 + ((layer_index - 1) * 48)
            self._add_port(f"layer{layer_index}_color", 0, row_y, f"Layer {layer_index} color")
            self._add_port(f"layer{layer_index}_alpha", 0, row_y + 22, f"Layer {layer_index} alpha")
            blend = QComboBox(self)
            blend.setGeometry(102, row_y - 2, 126, 24)
            blend.addItems(blend_modes)
            blend.currentTextChanged.connect(
                lambda value, index=layer_index: self.compositorSettingChanged.emit(
                    self.node_id, f"layer{index}_blend_mode", value
                )
            )
            opacity = QSlider(Qt.Horizontal, self)
            opacity.setGeometry(102, row_y + 24, 126, 20)
            opacity.setRange(0, 100)
            opacity.setValue(100)
            opacity.valueChanged.connect(
                lambda value, index=layer_index: self.compositorSettingChanged.emit(
                    self.node_id, f"layer{index}_opacity", value / 100.0
                )
            )
            self.compositor_blends[layer_index] = blend
            self.compositor_opacities[layer_index] = opacity
            opacity.sliderReleased.connect(lambda: self.parent()._record_history())
            # New rows use unscaled geometry; retain it for subsequent zooms.
            geometries = getattr(self, '_base_child_geometries', None)
            if geometries is not None:
                for child in (blend, opacity, self._ports[f'layer{layer_index}_color'], self._ports[f'layer{layer_index}_alpha'], self._port_labels[f'layer{layer_index}_color'], self._port_labels[f'layer{layer_index}_alpha']):
                    geometries[child] = child.geometry()

    def set_compositor_state(self, settings: dict[str, object]) -> None:
        if self.node_type != "keying":
            return
        layer_count = max(2, min(64, int(settings.get("layer_count", 2))))
        state = (layer_count, tuple((str(settings.get(f"layer{i}_blend_mode", "normal")),
                                   round(float(settings.get(f"layer{i}_opacity", 1.0)) * 100))
                                  for i in range(1, layer_count + 1)))
        previous = getattr(self, '_compositor_state', None)
        if state == previous:
            return
        if previous is None or previous[0] != layer_count:
            self._ensure_compositor_layers(layer_count)
            self._base_size = QSize(380, max(320, 112 + layer_count * 48))
            self.apply_zoom(getattr(self.parent(), "_zoom", 1.0))
            for index in self.compositor_blends:
                visible = index <= layer_count
                for control in (self.compositor_blends[index], self.compositor_opacities[index],
                                self._ports[f"layer{index}_color"], self._ports[f"layer{index}_alpha"],
                                self._port_labels[f"layer{index}_color"], self._port_labels[f"layer{index}_alpha"]):
                    control.setVisible(visible)
            self.add_layer_top.setEnabled(layer_count < 64)
            self.add_layer_bottom.setEnabled(layer_count < 64)
            self.remove_layer_top.setEnabled(layer_count > 2)
            self.remove_layer_bottom.setEnabled(layer_count > 2)
        for index, (blend, opacity) in enumerate(state[1], 1):
            if previous is not None and index <= previous[0] and previous[1][index - 1] == (blend, opacity):
                continue
            blend_control = self.compositor_blends[index]
            opacity_control = self.compositor_opacities[index]
            blend_control.blockSignals(True)
            opacity_control.blockSignals(True)
            blend_control.setCurrentText(blend)
            opacity_control.setValue(opacity)
            blend_control.blockSignals(False)
            opacity_control.blockSignals(False)
        self._compositor_state = state

    def set_matte_state(self, settings: dict[str, object]) -> None:
        for key, button in getattr(self, "gradient_swatches", {}).items():
            rgb = settings.get(key, [0, 0, 0] if key == "color_a" else [255, 255, 255])
            button.setStyleSheet(f"background-color: rgb({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])}); border: 1px solid #d9dde2;")
        swatch = getattr(self, "matte_swatch", None)
        if swatch is None:
            return
        red = max(0, min(255, int(settings.get("red", 255))))
        green = max(0, min(255, int(settings.get("green", 255))))
        blue = max(0, min(255, int(settings.get("blue", 255))))
        swatch.setStyleSheet(f"background-color: rgb({red}, {green}, {blue}); border: 1px solid #d9dde2;")

    def set_color_state(self, settings: dict[str, object]) -> None:
        del settings

    def dragEnterEvent(self, event) -> None:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if self.node_type == "media" and any(url.isLocalFile() for url in urls):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        if self.node_type == "media" and event.mimeData().hasUrls():
            local_paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            if local_paths:
                self.mediaPathChanged.emit(self.node_id, local_paths[0])
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    def set_blur_level(self, value: float) -> None:
        control = getattr(self, "blur_level", None)
        if control is None:
            return
        control.blockSignals(True)
        control.setValue(float(value))
        control.blockSignals(False)

    def set_capture_devices(self, devices: list[tuple[str, object]], selected_index: object = None) -> None:
        label = getattr(self, "capture_device_label", None)
        if label is None:
            return
        self._capture_devices = list(devices)
        self._selected_capture_device = selected_index
        selected_name = next(
            (name for name, device_index in devices if device_index == selected_index),
            "No source selected",
        )
        label.setText(selected_name)
        label.setToolTip(selected_name)

    def _add_capture_device_menu(self, menu: QMenu) -> QMenu:
        device_menu = menu.addMenu("Source")
        if self._capture_devices:
            for device_name, device_index in self._capture_devices:
                device_action = device_menu.addAction(device_name)
                device_action.setCheckable(True)
                device_action.setChecked(device_index == self._selected_capture_device)
                device_action.triggered.connect(
                    lambda _checked=False, index=device_index, name=device_name: self.captureDeviceChanged.emit(
                        self.node_id, index, name
                    )
                )
        else:
            no_devices_action = device_menu.addAction("No sources configured")
            no_devices_action.setEnabled(False)
        return device_menu

    def apply_zoom(self, zoom: float) -> None:
        scale = max(0.25, min(2.0, float(zoom)))
        self.setFixedSize(round(self._base_size.width() * scale), round(self._base_size.height() * scale))
        for child, geometry in self._base_child_geometries.items():
            child.setGeometry(
                round(geometry.x() * scale),
                round(geometry.y() * scale),
                max(1, round(geometry.width() * scale)),
                max(1, round(geometry.height() * scale)),
            )

    def _add_port(self, name: str, x: int, y: int, tooltip: str) -> None:
        port = EffectsPortButton(self)
        port.setGeometry(x, y, 20, 20)
        port.setToolTip(tooltip)
        port.setCursor(Qt.CrossCursor)
        port.setContextMenuPolicy(Qt.CustomContextMenu)
        port.setStyleSheet(
            "QPushButton { background: #efb73e; border: 2px solid #1c2025; border-radius: 8px; }"
            "QPushButton:hover { background: #fff0a6; }"
        )
        port.clicked.connect(lambda _checked=False, port_name=name: self.portClicked.emit(self.node_id, port_name))
        port.dragStarted.connect(
            lambda position, port_name=name: self.portDragStarted.emit(self.node_id, port_name, position)
        )
        port.dragMoved.connect(
            lambda position, port_name=name: self.portDragMoved.emit(self.node_id, port_name, position)
        )
        port.dragFinished.connect(
            lambda position, moved, port_name=name: self.portDragFinished.emit(
                self.node_id, port_name, position, moved
            )
        )
        port.customContextMenuRequested.connect(
            lambda _position, port_name=name: self.portDisconnectRequested.emit(self.node_id, port_name)
        )
        self._ports[name] = port
        label_names = {
            **{f"layer{index}_color": f"L{index} Color" for index in range(1, 65)},
            **{f"layer{index}_alpha": f"L{index} Alpha" for index in range(1, 65)},
            "a_input": "Alpha A",
            "b_input": "Alpha B",
            "red_input": "Red",
            "green_input": "Green",
            "blue_input": "Blue",
            "red_output": "Red",
            "green_output": "Green",
            "blue_output": "Blue",
        }
        label = QLabel(label_names.get(name, "Alpha" if "alpha" in name else "Color"), self)
        label.setGeometry(x + 24 if x == 0 else x - 80, y - 1, 76, 22)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter if x == 0 else Qt.AlignRight | Qt.AlignVCenter)
        label.setStyleSheet("color: #d9dde2; font-size: 9px;")
        self._port_labels[name] = label

    def port_center(self, name: str) -> QPointF:
        port = self._ports[name]
        return QPointF(self.x() + port.x() + (port.width() / 2.0), self.y() + port.y() + (port.height() / 2.0))

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(Qt.GlobalColor.transparent))
        painter.setBrush(Qt.GlobalColor.darkGray)
        painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 6, 6)
        painter.setPen(QPen(QColor("#efb73e") if self._selected else Qt.GlobalColor.gray, 3 if self._selected else 1))
        painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 6, 6)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.selectionRequested.emit(self.node_id, bool(event.modifiers() & Qt.ShiftModifier))
            self._drag_offset = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            self.raise_()
            self.nodeDragStarted.emit(self.node_id)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is None or not bool(event.buttons() & Qt.LeftButton):
            return
        target = self.mapToParent(event.position().toPoint()) - self._drag_offset.toPoint()
        self.move(max(0, target.x()), max(0, target.y()))
        self.nodeMoved.emit()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._drag_offset is not None:
            self._drag_offset = None
            self.setCursor(Qt.OpenHandCursor)
            self.nodeMoved.emit()
            self.nodeDragFinished.emit(self.node_id)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        device_menu = None
        reload_action = None
        if self.node_type == "capture":
            device_menu = self._add_capture_device_menu(menu)

        settings_action = menu.addAction("Settings...") if self.node_type != "capture" else None
        delete_action = None
        if self.node_type not in {"effects_input", "effects_output"}:
            menu.addSeparator()
            delete_action = menu.addAction("Delete node")
        selected = menu.exec(event.globalPos())
        del device_menu
        if reload_action is not None and selected == reload_action:
            self.captureReloadRequested.emit(self.node_id)
        elif settings_action is not None and selected == settings_action:
            self.settingsRequested.emit(self.node_id)
        elif delete_action is not None and selected == delete_action:
            self.deleteRequested.emit(self.node_id)


class EffectsGraphCanvas(QWidget):
    captureSettingsRequested = Signal(object)
    graphChanged = Signal()

    NODE_TITLES = {
        "effects_input": "Effects Input",
        "effects_output": "Effects Output",
        "denoise": "Noise Reduction",
        "composition": "Composition Source",
        "media": "Video / Image / Sequence",
        "capture": "Source",
        "keying": "Compositor",
        "blur": "Blur",
        "matte": "Matte Color",
        "mask": "Mask",
        "gradient": "Gradient Texture",
        "mix": "Mix",
        "proc_amp": "Basic Proc Amp",
        "color_corrector": "Color Corrector",
        "chroma_key": "Chroma Key",
        "luma_key": "Luma Key",
        "transform_3d": "3D Transform",
        "color_splitter": "Color Splitter",
        "color_recombiner": "Color Recombiner",
    }

    def __init__(self) -> None:
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(900, 360)
        self.setStyleSheet("background: #171a1f;")
        self._nodes: dict[str, dict[str, object]] = {}
        self._widgets: dict[str, EffectsNodeWidget] = {}
        self._connections: list[tuple[str, str, str, str]] = []
        self._pending_port: tuple[str, str] | None = None
        self._drag_port: tuple[str, str] | None = None
        self._drag_position: QPointF | None = None
        self._selected_nodes: set[str] = set()
        self._group_drag_driver: str | None = None
        self._group_drag_origins: dict[str, QPointF] = {}
        self._marquee_origin: QPointF | None = None
        self._marquee_rect: QRectF | None = None
        self._marquee_additive = False
        self._capture_devices: list[tuple[str, object]] = []
        self._capture_reload_tokens: dict[str, int] = {}
        self._media_keyframe_playback_arms: dict[str, str] = {}
        self._key_dialogs: dict[str, QDialog] = {}
        self._zoom = 1.0
        self._next_node_number = 1
        self._effects_enabled = True
        self._create_node("effects_input", 40, 120, "effects_input")
        self._create_node("effects_output", 620, 120, "effects_output")
        self._connections.append(("effects_input", "output", "effects_output", "input"))
        self._history = [self._history_snapshot()]
        self._history_index = 0
        self._restoring_history = False
        self.graphChanged.connect(self._record_history)

    def _history_snapshot(self) -> dict[str, object]:
        state = self.serialize()
        editor = getattr(self, '_editor', None)
        if editor is not None:
            state['_history_keyframes'] = deepcopy({node_id: frames for node_id, frames in editor._node_keyframes.items() if node_id in self._nodes})
            state['_history_duration'] = editor._duration
        return state

    def _record_history(self) -> None:
        if self._restoring_history or self._group_drag_driver is not None:
            return
        if any(slider.isSliderDown() for slider in self.findChildren(QSlider)):
            return
        state = self._history_snapshot()
        if state == self._history[self._history_index]:
            return
        self._history = self._history[:self._history_index + 1]
        self._history.append(state)
        self._history = self._history[-101:]
        self._history_index = len(self._history) - 1

    def _travel_history(self, offset: int) -> None:
        index = self._history_index + offset
        if not 0 <= index < len(self._history):
            return
        self._restoring_history = True
        try:
            self._history_index = index
            editor = getattr(self, '_editor', None)
            if editor is not None:
                editor._node_keyframes = deepcopy(self._history[index].get('_history_keyframes', {}))
                editor._duration = int(self._history[index].get('_history_duration', editor._duration))
                editor._current_frame = min(editor._current_frame, max(0, editor._duration - 1))
            self.restore(deepcopy(self._history[index]))
        finally:
            self._restoring_history = False

    def undo(self) -> None:
        self._travel_history(-1)

    def redo(self) -> None:
        self._travel_history(1)

    def copy_nodes(self) -> None:
        ids = {i for i in self._selected_nodes if self._nodes[i]['type'] not in {'effects_input', 'effects_output'}}
        if ids:
            QApplication.clipboard().setText(json.dumps({
                'node_editor_clipboard': 1,
                'nodes': [self._nodes[i] for i in self._nodes if i in ids],
                'connections': [c for c in self._connections if c[0] in ids and c[2] in ids],
                'keyframes': {node_id: frames for node_id, frames in getattr(getattr(self, '_editor', None), '_node_keyframes', {}).items() if node_id in ids},
            }))

    def paste_nodes(self) -> None:
        try:
            clipboard_text = QApplication.clipboard().text()
            data = json.loads(clipboard_text)
            if not isinstance(data, dict) or data.get('node_editor_clipboard') != 1:
                return
            nodes = data['nodes']
            if not isinstance(nodes, list) or any(not isinstance(n, dict) or n.get('type') not in self.NODE_TITLES or n['type'] in {'effects_input', 'effects_output'} or not isinstance(n.get('settings'), dict) for n in nodes):
                return
            # Validate before changing the graph.
            paste_count = getattr(self, '_paste_count', 0) + 1 if getattr(self, '_clipboard_text', None) == clipboard_text else 1
            copies = [(str(n['id']), n['type'], int(n['x']) + 40 * paste_count, int(n['y']) + 40 * paste_count, deepcopy(n['settings'])) for n in nodes]
            connections = [tuple(c) for c in data.get('connections', []) if isinstance(c, list) and len(c) == 4 and all(isinstance(part, str) for part in c)]
        except (ValueError, TypeError, KeyError):
            return
        remap = {}
        self._clipboard_text, self._paste_count = clipboard_text, paste_count
        for old, kind, x, y, settings in copies:
            new = self._create_node(kind, x, y)
            remap[old] = new
            self._nodes[new]['settings'] = settings
            self._refresh_node_status(new)
        editor = getattr(self, '_editor', None)
        if editor is not None:
            for old, frames in (data.get('keyframes') if isinstance(data.get('keyframes'), dict) else {}).items():
                if old in remap and isinstance(frames, dict):
                    editor._node_keyframes[remap[old]] = {int(frame): deepcopy(settings) for frame, settings in frames.items() if str(frame).isdigit() and isinstance(settings, dict)}
                    editor._duration = max(editor._duration, 1 + max(editor._node_keyframes[remap[old]], default=0))
        self._connections.extend((remap[a], b, remap[c], d) for a, b, c, d in connections if a in remap and c in remap)
        self._set_selected_nodes(set(remap.values()))
        self.update()
        self.graphChanged.emit()

    @staticmethod
    def _default_node_settings(node_type: str) -> dict:
        settings: dict[str, object] = {}
        if node_type == "denoise":
            settings = {"method": "luma_gaussian3x3", "strength": 0.35}
        elif node_type == "media":
            settings = {"path": "", "opacity": 1.0, "playing": False, "loop": True, "runtime_supported": True}
        elif node_type == "keying":
            settings = {
                "layer_count": 2,
                "layer1_blend_mode": "normal", "layer1_opacity": 1.0,
                "layer2_blend_mode": "normal", "layer2_opacity": 1.0,
                "runtime_supported": True,
            }
        elif node_type == "blur":
            settings = {"method": "gaussian", "radius": 3.0, "target": "both"}
        elif node_type == "capture":
            settings = {
                "device_index": None,
                "device_name": "No source selected",
                "capture_width": 0,
                "capture_height": 0,
            }
        elif node_type == "matte":
            settings = {"red": 255, "green": 255, "blue": 255, "alpha": 1.0}
        elif node_type == "gradient":
            settings = {"gradient_type": "linear", "color_a": [0, 0, 0], "color_b": [255, 255, 255], "rotation": 0.0, "aspect": 1.0,
                        "size": 1.0, "x": 0.0, "y": 0.0, "invert": False}
        elif node_type == "mask":
            settings = {
                "pattern": "circle", "softness": 0.0, "aspect": 1.0, "transparency": 0.0,
                "size": 1.0, "x": 0.0, "y": 0.0, "invert": False, "rotation": 0.0,
            }
        elif node_type == "mix":
            settings = {"mode": "add", "factor": 1.0}
        elif node_type == "transform_3d":
            settings = {
                "x": 0.0, "y": 0.0, "z": 0.0, "aspect_x": 1.0, "aspect_y": 1.0,
                "rotate_x": 0.0, "rotate_y": 0.0, "rotate_z": 0.0,
            }
        elif node_type == "proc_amp":
            settings = {"saturation": 1.0, "hue": 0.0, "brightness": 0.0, "contrast": 1.0, "invert": False}
        elif node_type == "color_corrector":
            settings = {
                "gain_r": 1.0, "gain_g": 1.0, "gain_b": 1.0,
                "mid_r": 1.0, "mid_g": 1.0, "mid_b": 1.0,
                "blacks_r": 0.0, "blacks_g": 0.0, "blacks_b": 0.0,
            }
        elif node_type == "chroma_key":
            settings = {
                "key_color_r": 0, "key_color_g": 255, "key_color_b": 0,
                "key_similarity": 0.25, "key_softness": 0.10,
                "key_edge_feather": 0.0, "spill_suppression": 0.25, "key_invert": False,
            }
        if node_type == "luma_key":
            settings = {"clip": 0.0, "gain": 1.0, "invert": False}
        return settings

    def _reset_node_settings(self, node_id: str) -> None:
        node = self._nodes[node_id]
        node['settings'].clear()
        node['settings'].update(self._default_node_settings(node['type']))
        self._refresh_node_status(node_id)
        self.graphChanged.emit()
        self._record_history()

    def _create_node(self, node_type: str, x: int, y: int, node_id: str | None = None) -> str:
        if node_id is None:
            node_id = f"{node_type}_{self._next_node_number}"
            self._next_node_number += 1
            while node_id in self._nodes:
                node_id = f"{node_type}_{self._next_node_number}"
                self._next_node_number += 1
        settings = self._default_node_settings(node_type)
        self._nodes[node_id] = {"id": node_id, "type": node_type, "x": int(x), "y": int(y), "settings": settings}
        widget = EffectsNodeWidget(node_id, node_type, self.NODE_TITLES[node_type], self)
        widget.move(int(x), int(y))
        widget.portClicked.connect(self._on_port_clicked)
        widget.portDragStarted.connect(self._on_port_drag_started)
        widget.portDragMoved.connect(self._on_port_drag_moved)
        widget.portDragFinished.connect(self._on_port_drag_finished)
        widget.portDisconnectRequested.connect(self._disconnect_port)
        widget.nodeMoved.connect(lambda node_key=node_id: self._on_node_moved(node_key))
        widget.selectionRequested.connect(self._select_node)
        widget.nodeDragStarted.connect(self._on_node_drag_started)
        widget.nodeDragFinished.connect(self._on_node_drag_finished)
        widget.settingsRequested.connect(self._edit_node_settings)
        widget.gradientColorRequested.connect(self._choose_gradient_color)
        widget.deleteRequested.connect(self._delete_requested)
        widget.blurLevelChanged.connect(self._on_blur_level_changed)
        widget.captureDeviceChanged.connect(self._on_capture_device_changed)
        widget.captureSettingsRequested.connect(self._on_capture_settings_requested)
        widget.captureReloadRequested.connect(self._on_capture_reload_requested)
        widget.mediaPathChanged.connect(self._on_media_path_changed)
        widget.mediaPlaybackChanged.connect(self._on_media_playback_changed)
        widget.mediaKeyframePlaybackArmed.connect(self._on_media_keyframe_playback_armed)
        widget.compositorSettingChanged.connect(self._on_compositor_setting_changed)
        widget.colorSettingChanged.connect(self._on_color_setting_changed)
        widget.compositorLayerActionRequested.connect(self._on_compositor_layer_action)
        widget.set_blur_level(float(settings.get("radius", 0.0)))
        widget.set_media_state(
            bool(settings.get("playing", False)),
            bool(settings.get("loop", True)),
            bool(str(settings.get("path", "")).strip()),
        )
        widget.set_capture_devices(self._capture_devices, settings.get("device_index"))
        widget.set_compositor_state(settings)
        widget.set_matte_state(settings)
        widget.set_color_state(settings)
        if node_type == "capture" and self._capture_devices and settings.get("device_index") is None:
            settings.update({"device_index": self._capture_devices[0][1], "device_name": self._capture_devices[0][0]})
        widget.apply_zoom(self._zoom)
        widget.move(round(int(x) * self._zoom), round(int(y) * self._zoom))
        widget.show()
        self._widgets[node_id] = widget
        self._refresh_node_status(node_id)
        self._expand_to_nodes()
        return node_id

    def add_node(self, node_type: str) -> str | None:
        if node_type not in {
            "denoise", "media", "capture", "keying", "blur", "matte", "mask", "gradient", "mix",
            "proc_amp", "color_corrector", "chroma_key", "luma_key", "transform_3d",
            "color_splitter", "color_recombiner",
        }:
            return None
        node_x, node_y = self._next_free_node_position(node_type)
        node_id = self._create_node(node_type, node_x, node_y)
        self.update()
        self.graphChanged.emit()
        return node_id

    def _next_free_node_position(self, node_type: str) -> tuple[int, int]:
        if node_type == "keying":
            node_width, node_height = 380, 520
        elif node_type == "media":
            node_width, node_height = 300, 154
        elif node_type in {"transform_3d", "color_splitter"}:
            node_width, node_height = 250, 154
        elif node_type == "color_recombiner":
            node_width, node_height = 280, 184
        else:
            node_width, node_height = 250, 124
        candidate_x = 300
        candidate_y = 70
        while True:
            candidate = QRect(candidate_x - 16, candidate_y - 16, node_width + 32, node_height + 32)
            occupied = False
            for node in self._nodes.values():
                existing_type = str(node.get("type", ""))
                if existing_type == "keying":
                    existing_width, existing_height = 380, 520
                elif existing_type == "media":
                    existing_width, existing_height = 300, 154
                elif existing_type == "transform_3d":
                    existing_width, existing_height = 250, 154
                else:
                    existing_width, existing_height = 250, 124
                existing = QRect(
                    int(node.get("x", 0)), int(node.get("y", 0)), existing_width, existing_height
                )
                if candidate.intersects(existing):
                    occupied = True
                    break
            if not occupied:
                return candidate_x, candidate_y
            candidate_y += 160

    def _first_available_compositor_layer(self, compositor_id: str, channel: str) -> int:
        node = self._nodes.get(compositor_id, {})
        settings = node.get("settings", {}) if isinstance(node, dict) else {}
        layer_count = max(2, min(64, int(settings.get("layer_count", 2)))) if isinstance(settings, dict) else 2
        occupied = {
            target_port
            for _source, _source_port, target, target_port in self._connections
            if target == compositor_id
        }
        if channel == "alpha":
            for layer_index in range(2, layer_count + 1):
                if (
                    f"layer{layer_index}_color" in occupied
                    and f"layer{layer_index}_alpha" not in occupied
                ):
                    return layer_index
        for layer_index in range(2, layer_count + 1):
            if f"layer{layer_index}_{channel}" not in occupied:
                return layer_index
        if layer_count < 64:
            self._on_compositor_layer_action(compositor_id, "add_top")
            return layer_count + 1
        return layer_count

    def migrate_legacy_denoise(self, method: str, strength: float) -> None:
        if method in {"off", "none"} or strength <= 0.0:
            return
        node_id = self.add_node("denoise")
        if node_id is None:
            return
        settings = self._nodes[node_id]["settings"]
        if isinstance(settings, dict):
            settings.update({"method": str(method), "strength": max(0.0, min(1.0, float(strength)))})
        self._refresh_node_status(node_id)

    def _on_node_moved(self, node_id: str) -> None:
        widget = self._widgets[node_id]
        self._nodes[node_id]["x"] = round(widget.x() / self._zoom)
        self._nodes[node_id]["y"] = round(widget.y() / self._zoom)
        if self._group_drag_driver == node_id and node_id in self._group_drag_origins:
            driver_origin = self._group_drag_origins[node_id]
            delta = QPointF(widget.pos()) - driver_origin
            for selected_id, origin in self._group_drag_origins.items():
                if selected_id == node_id or selected_id not in self._widgets:
                    continue
                selected_widget = self._widgets[selected_id]
                selected_widget.move(
                    max(0, round(origin.x() + delta.x())),
                    max(0, round(origin.y() + delta.y())),
                )
                self._nodes[selected_id]["x"] = round(selected_widget.x() / self._zoom)
                self._nodes[selected_id]["y"] = round(selected_widget.y() / self._zoom)
        self._expand_to_nodes()
        self.update()
        self.graphChanged.emit()

    def _set_selected_nodes(self, node_ids: set[str]) -> None:
        self._selected_nodes = {node_id for node_id in node_ids if node_id in self._nodes}
        for node_id, widget in self._widgets.items():
            widget.set_selected(node_id in self._selected_nodes)
        self.update()

    def _select_node(self, node_id: str, additive: bool) -> None:
        self.setFocus(Qt.MouseFocusReason)
        selected = set(self._selected_nodes)
        if additive:
            if node_id in selected:
                selected.remove(node_id)
            else:
                selected.add(node_id)
        else:
            selected = {node_id}
        self._set_selected_nodes(selected)

    def selected_node_ids(self) -> list[str]:
        return [node_id for node_id in self._nodes if node_id in self._selected_nodes]

    def _on_node_drag_started(self, node_id: str) -> None:
        if node_id not in self._selected_nodes:
            self._set_selected_nodes({node_id})
        self._group_drag_driver = node_id
        self._group_drag_origins = {
            selected_id: QPointF(self._widgets[selected_id].pos())
            for selected_id in self._selected_nodes
            if selected_id in self._widgets
        }

    def _on_node_drag_finished(self, node_id: str) -> None:
        if self._group_drag_driver != node_id:
            return
        self._group_drag_driver = None
        self._group_drag_origins.clear()
        self._record_history()

    def _expand_to_nodes(self) -> None:
        if not self._widgets:
            return
        needed_w = max(int(node["x"]) + 290 for node in self._nodes.values())
        needed_h = max(int(node["y"]) + self._widgets[node_id]._base_size.height() + 24 for node_id, node in self._nodes.items())
        self.setMinimumSize(round(max(900, needed_w) * self._zoom), round(max(360, needed_h) * self._zoom))

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        next_zoom = max(0.25, min(2.0, self._zoom * (1.1 if delta > 0 else (1.0 / 1.1))))
        if abs(next_zoom - self._zoom) < 1e-6:
            event.accept()
            return
        self._zoom = next_zoom
        for node_id, widget in self._widgets.items():
            widget.apply_zoom(self._zoom)
            node = self._nodes[node_id]
            widget.move(round(int(node["x"]) * self._zoom), round(int(node["y"]) * self._zoom))
        self._expand_to_nodes()
        self.update()
        event.accept()

    def set_capture_devices(self, devices: list[tuple[str, object]]) -> None:
        self._capture_devices = list(devices)
        for node_id, node in self._nodes.items():
            if node["type"] == "capture" and isinstance(node["settings"], dict):
                selected = node["settings"].get("device_index")
                selected_name = next((label for label, value in devices if value == selected), None)
                if selected_name is not None:
                    node["settings"]["device_name"] = selected_name
                self._widgets[node_id].set_capture_devices(devices, selected)
                self._refresh_node_status(node_id)

    def _on_port_clicked(self, node_id: str, port_name: str) -> None:
        is_output = self._is_output_port(port_name)
        if self._pending_port is None:
            if is_output:
                self._pending_port = (node_id, port_name)
                self._widgets[node_id].setStyleSheet("border: 2px solid #efb73e;")
            return
        source_id, source_port = self._pending_port
        self._widgets[source_id].setStyleSheet("")
        self._pending_port = None
        if is_output or source_id == node_id:
            return
        self._connect_ports(source_id, source_port, node_id, port_name)

    def _connect_ports(self, source_id: str, source_port: str, node_id: str, port_name: str) -> None:
        input_ports = {
            "input", "color_input", "alpha_input", "a_input", "b_input", "red_input", "green_input", "blue_input",
        } | {
            f"layer{layer_index}_{channel}"
            for layer_index in range(1, 65)
            for channel in ("color", "alpha")
        }
        if source_id == node_id or not self._is_output_port(source_port) or port_name not in input_ports:
            return
        candidate = (source_id, source_port, node_id, port_name)
        retained = [connection for connection in self._connections if connection[2:4] != candidate[2:4]]
        retained.append(candidate)
        if self._main_connections_have_cycle(retained):
            return
        self._connections = retained
        self.update()
        self.graphChanged.emit()

    def _on_port_drag_started(self, node_id: str, port_name: str, global_position: QPointF) -> None:
        if not self._is_output_port(port_name):
            return
        if self._pending_port is not None:
            pending_node_id, _pending_port_name = self._pending_port
            self._widgets[pending_node_id].setStyleSheet("")
        self._pending_port = None
        self._drag_port = (node_id, port_name)
        self._drag_position = QPointF(self.mapFromGlobal(global_position.toPoint()))
        self.update()

    def _on_port_drag_moved(self, node_id: str, port_name: str, global_position: QPointF) -> None:
        if self._drag_port != (node_id, port_name):
            return
        self._drag_position = QPointF(self.mapFromGlobal(global_position.toPoint()))
        self.update()

    def _on_port_drag_finished(
        self,
        node_id: str,
        port_name: str,
        global_position: QPointF,
        moved: bool,
    ) -> None:
        source = self._drag_port
        self._drag_port = None
        self._drag_position = None
        if source != (node_id, port_name):
            self.update()
            return
        if not moved:
            self.update()
            return
        for target_id, widget in reversed(list(self._widgets.items())):
            for target_port, port in widget._ports.items():
                if not port.isVisible():
                    continue
                if self._is_output_port(target_port):
                    continue
                if port.rect().contains(port.mapFromGlobal(global_position.toPoint())):
                    self._connect_ports(node_id, port_name, target_id, target_port)
                    return
        self.update()

    @staticmethod
    def _is_output_port(port_name: str) -> bool:
        return port_name == "output" or port_name.endswith("_output")

    def _disconnect_port(self, node_id: str, port_name: str) -> None:
        retained = [
            connection for connection in self._connections
            if connection[0:2] != (node_id, port_name) and connection[2:4] != (node_id, port_name)
        ]
        if len(retained) == len(self._connections):
            return
        self._connections = retained
        self.update()
        self.graphChanged.emit()

    def _on_blur_level_changed(self, node_id: str, value: float) -> None:
        settings = self._nodes.get(node_id, {}).get("settings")
        if not isinstance(settings, dict):
            return
        settings["radius"] = max(0.0, min(128.0, float(value)))
        self._refresh_node_status(node_id)
        self.graphChanged.emit()

    def _on_capture_device_changed(self, node_id: str, device_index: object, device_name: str) -> None:
        settings = self._nodes.get(node_id, {}).get("settings")
        if not isinstance(settings, dict):
            return
        settings.update({"device_index": device_index, "device_name": str(device_name)})
        if node_id not in self._widgets:
            return
        self._widgets[node_id].set_capture_devices(self._capture_devices, device_index)
        self._refresh_node_status(node_id)
        self.graphChanged.emit()

    def _on_capture_settings_requested(self, node_id: str) -> None:
        settings = self._nodes.get(node_id, {}).get("settings")
        if not isinstance(settings, dict):
            return
        self.captureSettingsRequested.emit(settings.get("device_index"))

    def _on_capture_reload_requested(self, node_id: str) -> None:
        settings = self._nodes.get(node_id, {}).get("settings")
        if not isinstance(settings, dict):
            return
        device_index = settings.get("device_index")
        if not isinstance(device_index, str) or not device_index.startswith(("webcam:", "source:")):
            return
        self._capture_reload_tokens[node_id] = self._capture_reload_tokens.get(node_id, 0) + 1
        self.graphChanged.emit()

    def _on_media_path_changed(self, node_id: str, media_path: str) -> None:
        settings = self._nodes.get(node_id, {}).get("settings")
        path = Path(media_path)
        if not isinstance(settings, dict) or not path.is_file():
            return
        settings.update({"path": str(path), "playing": False})
        self._refresh_node_status(node_id)
        self.graphChanged.emit()

    def _on_media_playback_changed(self, node_id: str, playing: bool, loop: bool) -> None:
        settings = self._nodes.get(node_id, {}).get("settings")
        if not isinstance(settings, dict):
            return
        settings.update({"playing": bool(playing), "loop": bool(loop)})
        self.graphChanged.emit()

    def _on_media_keyframe_playback_armed(self, node_id: str, state: str, armed: bool) -> None:
        if armed:
            self._media_keyframe_playback_arms[node_id] = state
            self._set_selected_nodes(self._selected_nodes | {node_id})
        elif self._media_keyframe_playback_arms.get(node_id) == state:
            self._media_keyframe_playback_arms.pop(node_id, None)

    def _on_compositor_setting_changed(self, node_id: str, name: str, value: object) -> None:
        settings = self._nodes.get(node_id, {}).get("settings")
        if not isinstance(settings, dict):
            return
        settings[name] = value
        self._refresh_node_status(node_id)
        self.graphChanged.emit()

    def _on_color_setting_changed(self, node_id: str, name: str, value: object) -> None:
        settings = self._nodes.get(node_id, {}).get("settings")
        if not isinstance(settings, dict):
            return
        settings[name] = value
        self._refresh_node_status(node_id)
        self.graphChanged.emit()

    def _on_compositor_layer_action(self, node_id: str, action: str) -> None:
        node = self._nodes.get(node_id)
        settings = node.get("settings") if isinstance(node, dict) else None
        if not isinstance(settings, dict):
            return
        layer_setting_defaults: dict[str, object] = {"blend_mode": "normal", "opacity": 1.0}

        def layer_setting(source: dict[str, object], layer_index: int, name: str) -> object:
            prefixed_name = f"layer{layer_index}_{name}"
            return source.get(prefixed_name, layer_setting_defaults[name])

        def set_layer_setting(layer_index: int, name: str, value: object) -> None:
            prefixed_name = f"layer{layer_index}_{name}"
            settings[prefixed_name] = value

        def reset_layer_settings(layer_index: int, remove: bool = False) -> None:
            for name, default in layer_setting_defaults.items():
                prefixed_name = f"layer{layer_index}_{name}"
                if remove:
                    settings.pop(prefixed_name, None)
                else:
                    settings[prefixed_name] = default

        layer_count = max(2, min(64, int(settings.get("layer_count", 2))))
        if action.startswith("add_") and layer_count >= 64:
            return
        if action.startswith("remove_") and layer_count <= 2:
            return

        if action == "add_top":
            settings["layer_count"] = layer_count + 1
            reset_layer_settings(layer_count + 1)
        elif action == "remove_top":
            self._connections = [
                connection for connection in self._connections
                if connection[2:4] not in {
                    (node_id, f"layer{layer_count}_color"),
                    (node_id, f"layer{layer_count}_alpha"),
                }
            ]
            reset_layer_settings(layer_count, remove=True)
            settings["layer_count"] = layer_count - 1
        elif action in {"add_bottom", "remove_bottom"}:
            if action == "add_bottom":
                layer_range = range(layer_count, 1, -1)
                offset = 1
                settings["layer_count"] = layer_count + 1
            else:
                self._connections = [
                    connection for connection in self._connections
                    if connection[2:4] not in {
                        (node_id, "layer2_color"),
                        (node_id, "layer2_alpha"),
                    }
                ]
                layer_range = range(3, layer_count + 1)
                offset = -1
                settings["layer_count"] = layer_count - 1
            remapped: list[tuple[str, str, str, str]] = []
            for source, source_port, target, target_port in self._connections:
                if target == node_id:
                    for layer_index in layer_range:
                        for channel in ("color", "alpha"):
                            if target_port == f"layer{layer_index}_{channel}":
                                target_port = f"layer{layer_index + offset}_{channel}"
                                break
                remapped.append((source, source_port, target, target_port))
            self._connections = remapped
            source_settings = dict(settings)
            for layer_index in layer_range:
                target_index = layer_index + offset
                for name in layer_setting_defaults:
                    set_layer_setting(target_index, name, layer_setting(source_settings, layer_index, name))
            if action == "add_bottom":
                reset_layer_settings(2)
            else:
                reset_layer_settings(layer_count, remove=True)
        else:
            return

        self._refresh_node_status(node_id)
        self._expand_to_nodes()
        self.update()
        self.graphChanged.emit()

    def _main_connections_have_cycle(self, connections: list[tuple[str, str, str, str]]) -> bool:
        edges: dict[str, set[str]] = {}
        for source, _source_port, target, _target_port in connections:
            edges.setdefault(source, set()).add(target)

        def visits_cycle(node_id: str, active: set[str], complete: set[str]) -> bool:
            if node_id in active:
                return True
            if node_id in complete:
                return False
            active.add(node_id)
            if any(visits_cycle(target, active, complete) for target in edges.get(node_id, set())):
                return True
            active.remove(node_id)
            complete.add(node_id)
            return False

        complete: set[str] = set()
        for start in edges:
            if visits_cycle(start, set(), complete):
                return True
        return False

    def _delete_node(self, node_id: str, emit_changed: bool = True) -> None:
        node = self._nodes.get(node_id)
        if node is None or node["type"] in {"effects_input", "effects_output"}:
            return
        incoming = next((c for c in self._connections if c[2] == node_id and c[3] in {"input", "layer1_color"}), None)
        outgoing = next((c for c in self._connections if c[0] == node_id and c[1] == "output"), None)
        self._connections = [c for c in self._connections if c[0] != node_id and c[2] != node_id]
        if incoming is not None and outgoing is not None:
            self._connections.append((incoming[0], incoming[1], outgoing[2], outgoing[3]))
        self._widgets.pop(node_id).deleteLater()
        self._nodes.pop(node_id)
        self._selected_nodes.discard(node_id)
        self._media_keyframe_playback_arms.pop(node_id, None)
        self.update()
        if emit_changed:
            self.graphChanged.emit()

    def _delete_selected_nodes(self) -> None:
        deletable = [
            node_id for node_id in self.selected_node_ids()
            if self._nodes[node_id]["type"] not in {"effects_input", "effects_output"}
        ]
        if not deletable:
            return
        for node_id in deletable:
            self._delete_node(node_id, emit_changed=False)
        self._set_selected_nodes(set())
        self._expand_to_nodes()
        self.graphChanged.emit()

    def _delete_requested(self, node_id: str) -> None:
        if node_id in self._selected_nodes and len(self._selected_nodes) > 1:
            self._delete_selected_nodes()
            return
        self._delete_node(node_id)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        for key, callback in ((QKeySequence.Undo, self.undo), (QKeySequence.Redo, self.redo), (QKeySequence.Copy, self.copy_nodes), (QKeySequence.Paste, self.paste_nodes)):
            if event.matches(key):
                callback()
                event.accept()
                return
        if event.key() == Qt.Key_Z and event.modifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
            self.redo()
            event.accept()
            return
        if event.key() == Qt.Key_Delete:
            self._delete_selected_nodes()
            event.accept()
            return
        super().keyPressEvent(event)

    def _edit_node_settings(self, node_id: str) -> None:
        node = self._nodes[node_id]
        node_type = str(node["type"])
        settings = node["settings"]
        if not isinstance(settings, dict):
            return
        if node_type in {"denoise", "media", "matte", "blur"}:
            self._show_basic_node_settings(node_id)
            return
        elif node_type == "capture":
            self._show_capture_settings(node_id)
            return
        elif node_type in {"proc_amp", "color_corrector"}:
            self._show_color_settings(node_id)
            return
        elif node_type == "transform_3d":
            self._show_transform_settings(node_id)
            return
        elif node_type == "luma_key":
            self._show_luma_key_settings(node_id)
            return
        elif node_type == "chroma_key":
            self._show_chroma_key_settings(node_id)
            return
        elif node_type == "mix":
            self._show_mix_settings(node_id)
            return
        elif node_type == "mask":
            self._show_mask_settings(node_id)
        elif node_type == "gradient":
            self._show_gradient_settings(node_id)
            return
        else:
            return
        self._refresh_node_status(node_id)
        self.graphChanged.emit()

    def _show_basic_node_settings(self, node_id):
        node = self._nodes[node_id]
        opened = self._open_settings_dialog(node_id, self.NODE_TITLES[node['type']])
        if opened is None:
            return
        dialog, form = opened
        kind = node['type']
        def combo(label, name, values):
            control = QComboBox(dialog)
            control.addItems(values)
            control.setProperty('setting_name', name)
            control.setCurrentText(str(node['settings'].get(name, values[0])))
            control.currentTextChanged.connect(lambda value: self._on_color_setting_changed(node_id, name, value))
            form.addRow(label, control)
        if kind == 'denoise':
            combo('Method', 'method', list(DENOISE_METHOD_NAME_TO_LABEL))
            self._add_live_slider(form, dialog, node_id, 'Strength', 'strength', 0, 1, 100)
        elif kind == 'blur':
            combo('Method', 'method', ['gaussian', 'box'])
            combo('Channels', 'target', ['color', 'alpha', 'both'])
            self._add_live_slider(form, dialog, node_id, 'Radius', 'radius', 0, 128, 100)
        elif kind == 'matte':
            for name in ('red', 'green', 'blue'):
                self._add_live_slider(form, dialog, node_id, name.title(), name, 0, 255, 1, 0)
            self._add_live_slider(form, dialog, node_id, 'Alpha', 'alpha', 0, 1, 100)
        elif kind == 'media':
            browse = QPushButton('Choose media...', dialog)
            def choose():
                path, _ = QFileDialog.getOpenFileName(dialog, 'Choose media', str(node['settings'].get('path', '')))
                if path:
                    self._on_color_setting_changed(node_id, 'path', path)
            browse.clicked.connect(choose)
            form.addRow(browse)
            self._add_live_slider(form, dialog, node_id, 'Opacity', 'opacity', 0, 1, 100)
        self._finish_settings_dialog(dialog, form)

    def _show_luma_key_settings(self, node_id):
        opened = self._open_settings_dialog(node_id, 'Luma Key')
        if opened is None:
            return
        dialog, form = opened
        self._add_live_slider(form, dialog, node_id, 'Clip', 'clip', 0, 1, 1000, 3)
        self._add_live_slider(form, dialog, node_id, 'Gain', 'gain', 0, 100, 100, 2)
        invert = QCheckBox('Invert', dialog)
        invert.setProperty('setting_name', 'invert')
        invert.setChecked(bool(self._nodes[node_id]['settings'].get('invert', False)))
        invert.toggled.connect(lambda value: self._on_color_setting_changed(node_id, 'invert', value))
        form.addRow(invert)
        self._finish_settings_dialog(dialog, form)

    def _show_capture_settings(self, node_id: str) -> None:
        settings = self._nodes.get(node_id, {}).get("settings")
        if not isinstance(settings, dict):
            return
        self.captureSettingsRequested.emit(settings.get("device_index"))

    def _add_live_slider(
        self,
        form: QFormLayout,
        dialog: QDialog,
        node_id: str,
        label: str,
        name: str,
        minimum: float,
        maximum: float,
        scale: int,
        decimals: int = 2,
        suffix: str = "",
    ) -> None:
        settings = self._nodes[node_id]["settings"]
        if not isinstance(settings, dict):
            return
        row = QWidget(dialog)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Horizontal, row)
        slider.setRange(round(max(minimum, -1000.0) * scale), round(maximum * scale))
        slider.setProperty("setting_name", name)
        slider.setProperty("setting_scale", scale)
        adaptive = self._nodes[node_id]['type'] == 'transform_3d' and name in {'x', 'y', 'z'}
        value_control = ExpandingDoubleSpinBox(row) if adaptive else QDoubleSpinBox(row)
        current = float(settings.get(name, self._default_node_settings(self._nodes[node_id]['type']).get(name, minimum)))
        if adaptive:
            slider.setRange(max(-2147483647, round(min(minimum, current) * scale)), min(2147483647, round(max(maximum, current) * scale)))
        slider.setValue(max(-2147483647, min(2147483647, round(current * scale))))
        value_control.setRange(min(minimum, current) if adaptive else minimum, max(maximum, current) if adaptive else maximum)
        slider.setProperty('adaptive', adaptive)
        value_control.setProperty("setting_name", name)
        value_control.setDecimals(decimals)
        value_control.setSingleStep(1.0 / float(scale))
        value_control.setSuffix(suffix)
        value_control.setValue(current)
        value_control.setKeyboardTracking(not adaptive)
        value_control.setMinimumWidth(120)
        row_layout.addWidget(slider, 1)
        row_layout.addWidget(value_control)
        reset = QPushButton(row)
        reset.setIcon(QIcon(str(Path(__file__).parent / 'assets' / 'reset.svg')))
        reset.setFixedSize(26, 24)
        reset.setToolTip('Reset ' + label)
        reset.clicked.connect(lambda: value_control.setValue(float(self._default_node_settings(self._nodes[node_id]['type']).get(name, minimum))))
        row_layout.addWidget(reset)

        def slider_changed(raw_value: int) -> None:
            value = float(raw_value) / float(scale)
            value_control.blockSignals(True)
            value_control.setValue(value)
            value_control.blockSignals(False)
            self._on_color_setting_changed(node_id, name, value)

        def value_changed(value: float) -> None:
            slider.blockSignals(True)
            if adaptive:
                limit = 2147483647
                slider.setRange(max(-limit, round(min(minimum, value) * scale)), min(limit, round(max(maximum, value) * scale)))
            slider.setValue(max(-2147483647, min(2147483647, round(float(value) * scale))))
            slider.blockSignals(False)
            self._on_color_setting_changed(node_id, name, float(value))

        slider.valueChanged.connect(slider_changed)
        slider.sliderReleased.connect(self._record_history)
        value_control.valueChanged.connect(value_changed)
        value_control.editingFinished.connect(self._record_history)
        form.addRow(label, row)

    def _open_settings_dialog(self, node_id: str, title: str) -> tuple[QDialog, QFormLayout] | None:
        existing = self._key_dialogs.get(node_id)
        if existing is not None:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return None
        dialog = QDialog(self, Qt.Tool)
        dialog.setWindowTitle(title)
        dialog.setProperty("node_id", node_id)
        dialog.setMinimumWidth(430)
        dialog.setStyleSheet(
            "QDialog { background: #20242a; color: #f4f4f4; }"
            "QLabel { color: #d9dde2; }"
            "QPushButton, QAbstractSpinBox, QComboBox, QLineEdit { color: #f4f4f4; background: #292e35; }"
            "QCheckBox { background: #555555; color: #f4f4f4; border-radius: 4px; padding: 4px 6px; }"
            + NODE_CHECKBOX_INDICATOR_STYLE
        )
        form = QFormLayout(dialog)
        dialog.destroyed.connect(lambda: self._key_dialogs.pop(node_id, None))
        self._key_dialogs[node_id] = dialog
        return dialog, form

    def _finish_settings_dialog(self, dialog: QDialog, form: QFormLayout) -> None:
        buttons = QWidget(dialog)
        row = QHBoxLayout(buttons)
        reset = QPushButton('Reset all', buttons)
        reset.clicked.connect(lambda: self._reset_node_settings(dialog.property('node_id')))
        row.addWidget(reset)
        close_button = QPushButton("Close", buttons)
        close_button.clicked.connect(dialog.close)
        row.addWidget(close_button)
        form.addRow(buttons)
        for button in dialog.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)
        dialog.show()

    def _show_color_settings(self, node_id: str) -> None:
        node_type = str(self._nodes[node_id]["type"])
        opened = self._open_settings_dialog(
            node_id,
            "Basic Proc Amp" if node_type == "proc_amp" else "Color Corrector",
        )
        if opened is None:
            return
        dialog, form = opened
        if node_type == "proc_amp":
            for spec in (
                ("Saturation", "saturation", 0.0, 4.0, 100, 2, ""),
                ("Hue", "hue", -180.0, 180.0, 1, 0, " deg"),
                ("Brightness", "brightness", -1.0, 1.0, 100, 2, ""),
                ("Contrast", "contrast", 0.0, 4.0, 100, 2, ""),
            ):
                self._add_live_slider(form, dialog, node_id, *spec)
            settings = self._nodes[node_id]["settings"]
            invert = QCheckBox("Invert", dialog)
            invert.setProperty("setting_name", "invert")
            invert.setChecked(bool(settings.get("invert", False)) if isinstance(settings, dict) else False)
            invert.toggled.connect(lambda value: self._on_color_setting_changed(node_id, "invert", bool(value)))
            form.addRow(invert)
        else:
            for heading, prefix, minimum, maximum, scale in (
                ("Gain", "gain", 0.0, 4.0, 100),
                ("Mid", "mid", 0.1, 4.0, 100),
                ("Blacks", "blacks", -1.0, 1.0, 100),
            ):
                for channel, channel_label in (("r", "Red"), ("g", "Green"), ("b", "Blue")):
                    self._add_live_slider(
                        form, dialog, node_id, f"{heading} {channel_label}",
                        f"{prefix}_{channel}", minimum, maximum, scale,
                    )
        self._finish_settings_dialog(dialog, form)

    def _show_chroma_key_settings(self, node_id: str) -> None:
        opened = self._open_settings_dialog(node_id, "Chroma Key")
        if opened is None:
            return
        dialog, form = opened
        for label, name in (("Key red", "key_color_r"), ("Key green", "key_color_g"), ("Key blue", "key_color_b")):
            self._add_live_slider(form, dialog, node_id, label, name, 0.0, 255.0, 1, 0)
        self._add_live_slider(form, dialog, node_id, "Similarity", "key_similarity", 0.0, 1.0, 1000, 3)
        self._add_live_slider(form, dialog, node_id, "Softness", "key_softness", 0.0, 1.0, 1000, 3)
        self._add_live_slider(form, dialog, node_id, "Edge feather", "key_edge_feather", 0.0, 16.0, 10, 1, " px")
        self._add_live_slider(form, dialog, node_id, "Spill suppression", "spill_suppression", 0.0, 1.0, 1000, 3)
        settings = self._nodes[node_id]["settings"]
        invert = QCheckBox("Invert alpha", dialog)
        invert.setProperty("setting_name", "key_invert")
        invert.setChecked(bool(settings.get("key_invert", False)) if isinstance(settings, dict) else False)
        invert.toggled.connect(lambda value: self._on_color_setting_changed(node_id, "key_invert", bool(value)))
        form.addRow(invert)
        self._finish_settings_dialog(dialog, form)

    def _show_transform_settings(self, node_id: str) -> None:
        opened = self._open_settings_dialog(node_id, "3D Transform")
        if opened is None:
            return
        dialog, form = opened
        for spec in (
            ("X", "x", -100.0, 100.0, 10, 1, "%"),
            ("Y", "y", -100.0, 100.0, 10, 1, "%"),
            ("Z (1000 = 10x)", "z", -100.0, 100.0, 10, 1, "%"),
            ("Horizontal aspect", "aspect_x", 0.01, 10.0, 100, 2, ""),
            ("Vertical aspect", "aspect_y", 0.01, 10.0, 100, 2, ""),
            ("Rotate X", "rotate_x", -180.0, 180.0, 10, 1, " deg"),
            ("Rotate Y", "rotate_y", -180.0, 180.0, 10, 1, " deg"),
            ("Rotate Z", "rotate_z", -180.0, 180.0, 10, 1, " deg"),
        ):
            self._add_live_slider(form, dialog, node_id, *spec)
        self._finish_settings_dialog(dialog, form)

    def _show_mix_settings(self, node_id: str) -> None:
        settings = self._nodes[node_id]["settings"]
        if not isinstance(settings, dict):
            return
        opened = self._open_settings_dialog(node_id, "Mix")
        if opened is None:
            return
        dialog, form = opened
        modes = ["add", "multiply", "subtract", "overlay", "difference"]
        mode_combo = QComboBox(dialog)
        mode_combo.addItems(modes)
        mode_combo.setProperty("setting_name", "mode")
        current_mode = str(settings.get("mode", "add"))
        mode_combo.setCurrentText(current_mode if current_mode in modes else "add")
        mode_combo.currentTextChanged.connect(
            lambda value: self._on_color_setting_changed(node_id, "mode", str(value))
        )
        form.addRow("Mode", mode_combo)
        self._add_live_slider(form, dialog, node_id, "Factor", "factor", 0.0, 1.0, 1000, 3)
        self._finish_settings_dialog(dialog, form)

    def _choose_gradient_color(self, node_id: str, key: str) -> None:
        settings = self._nodes[node_id]["settings"]
        original = list(settings.get(key, [0, 0, 0] if key == "color_a" else [255, 255, 255]))
        picker = QColorDialog(QColor(*original), self)
        picker.setWindowTitle("Gradient Color " + ("A" if key == "color_a" else "B"))
        def update(color):
            if color.isValid():
                self._on_compositor_setting_changed(node_id, key, [color.red(), color.green(), color.blue()])
        picker.currentColorChanged.connect(update)
        if picker.exec() == QDialog.Accepted:
            update(picker.selectedColor())
        else:
            self._on_compositor_setting_changed(node_id, key, original)

    def _show_gradient_settings(self, node_id: str) -> None:
        opened = self._open_settings_dialog(node_id, "Gradient Texture")
        if opened is None:
            return
        dialog, form = opened
        settings = self._nodes[node_id]["settings"]
        mode = QComboBox(dialog)
        for label, value in GRADIENT_MODES:
            mode.addItem(label, value)
        mode.setCurrentIndex(max(0, mode.findData(settings.get("gradient_type", "linear"))))
        mode.setProperty("setting_name", "gradient_type")
        mode.currentIndexChanged.connect(lambda _: self._on_compositor_setting_changed(node_id, "gradient_type", mode.currentData()))
        form.addRow("Type", mode)
        for label, key, lo, hi, factor, decimals in (
            ("Position X (%)", "x", -100, 100, 10, 1),
            ("Position Y (%)", "y", -100, 100, 10, 1),
            ("Size", "size", .1, 4., 100, 2),
            ("Aspect", "aspect", .25, 4., 100, 2),
            ("Diagonal rotation (deg)", "rotation", -180, 180, 10, 1),
        ):
            self._add_live_slider(form, dialog, node_id, label, key, lo, hi, factor, decimals)
        rotation_row = form.itemAt(form.rowCount() - 1, QFormLayout.FieldRole).widget()
        def update_rotation_control():
            rotation_row.setEnabled(mode.currentData() == "diagonal")
        mode.currentIndexChanged.connect(update_rotation_control)
        update_rotation_control()
        invert = QCheckBox("Invert", dialog)
        invert.setProperty("setting_name", "invert")
        invert.setChecked(bool(settings.get("invert", False)))
        invert.toggled.connect(lambda value: self._on_compositor_setting_changed(node_id, "invert", value))
        form.addRow(invert)
        self._finish_settings_dialog(dialog, form)

    def _mask_icon(self, pattern: str) -> QIcon:
        pixmap = QPixmap(56, 56)
        pixmap.fill(QColor("#262b31"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#f1f3f5"))
        if pattern == "circle":
            painter.drawEllipse(QRectF(9, 9, 38, 38))
        elif pattern == 'barndoor_vertical':
            painter.drawRect(QRectF(20, 0, 16, 56))
        elif pattern == 'barndoor_horizontal':
            painter.drawRect(QRectF(0, 20, 56, 16))
        elif pattern == 'horizontal':
            painter.drawRect(QRectF(0, 28, 56, 28))
        elif pattern == "diamond":
            painter.drawPolygon(QPolygonF([QPointF(28, 6), QPointF(50, 28), QPointF(28, 50), QPointF(6, 28)]))
        else:
            painter.drawRect(QRectF(9, 9, 38, 38))
        painter.end()
        return QIcon(pixmap)

    def _show_mask_settings(self, node_id: str) -> None:
        settings = self._nodes[node_id]["settings"]
        if not isinstance(settings, dict):
            return
        opened = self._open_settings_dialog(node_id, 'Mask')
        if opened is None:
            return
        dialog, form = opened
        grid = QGridLayout()
        patterns = ('square', 'circle', 'diamond', 'barndoor_vertical', 'barndoor_horizontal', 'horizontal')
        for index, pattern in enumerate(patterns):
            label = {'barndoor_vertical': 'Vertical\nBarndoor', 'barndoor_horizontal': 'Horizontal\nBarndoor'}.get(pattern, pattern.title())
            button = QPushButton(label, dialog)
            button.setCheckable(True)
            button.setProperty('mask_pattern', pattern)
            button.setChecked(settings.get('pattern', 'circle') == pattern)
            button.setStyleSheet('QPushButton:checked { background: #346fa8; color: white; border: 2px solid #8fcaff; }')
            button.setIcon(self._mask_icon(pattern))
            button.setIconSize(QSize(56, 56))
            button.setMinimumHeight(70)
            button.setMinimumWidth(150)
            button.clicked.connect(lambda checked=False, value=pattern: self._on_color_setting_changed(node_id, 'pattern', value))
            grid.addWidget(button, index // 3, index % 3)
        form.addRow(grid)
        for label, name, low, high, scale, decimals in (
            ('Softness', 'softness', 0, 1, 100, 2),
            ('Transparency (%)', 'transparency', 0, 100, 10, 1),
            ('Aspect', 'aspect', .25, 4, 100, 2),
            ('Size', 'size', .1, 4, 100, 2),
            ('X position (%)', 'x', -100, 100, 10, 1),
            ('Y position (%)', 'y', -100, 100, 10, 1),
            ('Rotate (deg)', 'rotation', -180, 180, 10, 1),
        ):
            self._add_live_slider(form, dialog, node_id, label, name, low, high, scale, decimals)
        invert = QCheckBox('Invert mask', dialog)
        invert.setProperty('setting_name', 'invert')
        invert.setChecked(bool(settings.get('invert', False)))
        invert.toggled.connect(lambda value: self._on_color_setting_changed(node_id, 'invert', value))
        form.addRow(invert)
        self._finish_settings_dialog(dialog, form)

    def _refresh_node_status(self, node_id: str) -> None:
        node = self._nodes[node_id]
        settings = node["settings"] if isinstance(node["settings"], dict) else {}
        node_type = str(node["type"])
        if node_type == "effects_input":
            status = "GPU frame source"
        elif node_type == "composition":
            status = str(settings.get("name", "Composition")) + "\nColor + alpha"
        elif node_type == "effects_output":
            status = "GPU color + alpha destination"
        elif node_type == "denoise":
            status = f"{settings.get('method', 'off')}  {float(settings.get('strength', 0.0)):.2f}"
        elif node_type == "media":
            source_name = Path(str(settings.get("path", ""))).name or "No source selected"
            status = f"{source_name}\nGPU color + alpha"
        elif node_type == "keying":
            status = f"L2 {settings.get('layer2_blend_mode', 'normal')}  {float(settings.get('layer2_opacity', 1.0)):.2f}"
        elif node_type == "capture":
            width = max(0, int(settings.get("capture_width", 0)))
            height = max(0, int(settings.get("capture_height", 0)))
            status = "Select an input from the Source menu"
        elif node_type == "matte":
            status = "#{:02X}{:02X}{:02X}  alpha {:.2f}".format(
                int(settings.get("red", 255)), int(settings.get("green", 255)),
                int(settings.get("blue", 255)), float(settings.get("alpha", 1.0)),
            )
        elif node_type == "gradient":
            status = str(settings.get("gradient_type", "linear")).replace("_", " ").title()
        elif node_type == "mask":
            status = "{}{}  size {:.2f}\nX {:.0f}%  Y {:.0f}%".format(
                str(settings.get("pattern", "circle")).title(),
                " inverted" if bool(settings.get("invert", False)) else "",
                float(settings.get("size", 1.0)),
                float(settings.get("x", 0.0)), float(settings.get("y", 0.0)),
            )
        elif node_type == "mix":
            status = (
                f"Alpha mix\n{str(settings.get('mode', 'add')).replace('_', ' ').title()}  "
                f"{float(settings.get('factor', 1.0)):.2f}"
            )
        elif node_type == "transform_3d":
            status = "XYZ {:.0f}, {:.0f}, {:.0f}%\nRot {:.0f}, {:.0f}, {:.0f} deg".format(
                float(settings.get("x", 0.0)), float(settings.get("y", 0.0)), float(settings.get("z", 0.0)),
                float(settings.get("rotate_x", 0.0)), float(settings.get("rotate_y", 0.0)),
                float(settings.get("rotate_z", 0.0)),
            )
        elif node_type == "proc_amp":
            status = "Sat {:.2f}  Hue {:.1f}\nBright {:.2f}  Contrast {:.2f}{}".format(
                float(settings.get("saturation", 1.0)), float(settings.get("hue", 0.0)),
                float(settings.get("brightness", 0.0)), float(settings.get("contrast", 1.0)),
                "  Invert" if bool(settings.get("invert", False)) else "",
            )
        elif node_type == "color_corrector":
            status = "RGB lift / gamma / gain"
        elif node_type == "luma_key":
            status = f"Clip {settings.get('clip', 0):.3f}  Gain {settings.get('gain', 1):.2f}"
        elif node_type == "chroma_key":
            status = "#{:02X}{:02X}{:02X}  sim {:.3f}\nSoft {:.3f}  Feather {:.1f}px{}".format(
                int(settings.get("key_color_r", 0)), int(settings.get("key_color_g", 255)),
                int(settings.get("key_color_b", 0)), float(settings.get("key_similarity", 0.25)),
                float(settings.get("key_softness", 0.10)), float(settings.get("key_edge_feather", 0.0)),
                "  Invert" if bool(settings.get("key_invert", False)) else "",
            )
        else:
            status = f"{settings.get('method', 'gaussian')} {settings.get('target', 'both')}"
        self._widgets[node_id].set_status(status)
        self._widgets[node_id].set_blur_level(float(settings.get("radius", 0.0)))
        self._widgets[node_id].set_compositor_state(settings)
        self._widgets[node_id].set_matte_state(settings)
        self._widgets[node_id].set_color_state(settings)
        self._widgets[node_id].set_media_state(
            bool(settings.get("playing", False)),
            bool(settings.get("loop", True)),
            bool(str(settings.get("path", "")).strip()),
        )
        dialog = self._key_dialogs.get(node_id)
        if dialog is not None:
            for button in dialog.findChildren(QPushButton):
                pattern = button.property('mask_pattern')
                if pattern:
                    button.setChecked(pattern == settings.get('pattern', 'circle'))
            for combo in dialog.findChildren(QComboBox):
                name = combo.property('setting_name')
                if name in settings:
                    combo.blockSignals(True)
                    index = combo.findData(settings[name])
                    combo.setCurrentIndex(index if index >= 0 else combo.findText(str(settings[name])))
                    combo.blockSignals(False)
            for checkbox in dialog.findChildren(QCheckBox):
                setting_name = checkbox.property("setting_name")
                if isinstance(setting_name, str) and setting_name in settings:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(bool(settings[setting_name]))
                    checkbox.blockSignals(False)
            for value_control in dialog.findChildren(QDoubleSpinBox):
                setting_name = value_control.property("setting_name")
                if isinstance(setting_name, str) and setting_name in settings:
                    value_control.blockSignals(True)
                    if isinstance(value_control, ExpandingDoubleSpinBox):
                        value_control.setRange(min(-100, float(settings[setting_name])), max(100, float(settings[setting_name])))
                    value_control.setValue(float(settings[setting_name]))
                    value_control.blockSignals(False)
            for slider in dialog.findChildren(QSlider):
                setting_name = slider.property("setting_name")
                setting_scale = slider.property("setting_scale")
                if isinstance(setting_name, str) and setting_name in settings and isinstance(setting_scale, int):
                    slider.blockSignals(True)
                    value = float(settings[setting_name])
                    if slider.property('adaptive'):
                        slider.setRange(max(-2147483647, round(min(-100, value) * setting_scale)), min(2147483647, round(max(100, value) * setting_scale)))
                    slider.setValue(max(-2147483647, min(2147483647, round(value * setting_scale))))
                    slider.blockSignals(False)
        if node_type == "capture":
            self._widgets[node_id].set_capture_devices(self._capture_devices, settings.get("device_index"))

    def active_capture_device(self) -> int | None:
        edges = {
            source: target for source, source_port, target, target_port in self._connections
            if source_port == "output" and target_port == "input"
        }
        for node_id, node in self._nodes.items():
            if node["type"] != "capture" or edges.get(node_id) is None or not isinstance(node["settings"], dict):
                continue
            current = node_id
            visited: set[str] = set()
            while current in edges and current not in visited:
                visited.add(current)
                current = edges[current]
                if current == "effects_output":
                    device_index = node["settings"].get("device_index")
                    if isinstance(device_index, str) and device_index.startswith(("webcam:", "source:")):
                        return None
                    return int(device_index) if device_index is not None else None
        return None

    def active_denoise_settings(self) -> tuple[str, float]:
        if not self._effects_enabled:
            return "off", 0.0
        edges = {
            source: target for source, source_port, target, target_port in self._connections
            if source_port == "output" and target_port in {"input", "layer1_color"}
        }
        current = self._active_main_source()
        visited: set[str] = set()
        active: tuple[str, float] = ("off", 0.0)
        while current in edges and current not in visited:
            visited.add(current)
            current = edges[current]
            node = self._nodes.get(current)
            if node is not None and node["type"] == "denoise" and isinstance(node["settings"], dict):
                settings = node["settings"]
                active = (str(settings.get("method", "off")), float(settings.get("strength", 0.0)))
            if current == "effects_output":
                return active
        return "off", 0.0

    def _active_main_source(self) -> str:
        capture_sources = [
            node_id for node_id, node in self._nodes.items()
            if node["type"] == "capture" and any(
                connection[0] == node_id and connection[1] == "output"
                and connection[3] in {"input", "layer1_color"}
                for connection in self._connections
            )
        ]
        return capture_sources[-1] if capture_sources else "effects_input"

    def native_effects_payload(self) -> dict[str, object]:
        if not self._effects_enabled:
            return bypassed_effects_payload()
        if any(n['type'] in {'composition', 'gradient', 'mask', 'luma_key'} for n in self._nodes.values()) or sum(n['type'] == 'keying' for n in self._nodes.values()) > 1:
            try:
                from .composition_graph import compile_compositions
            except ImportError:
                from composition_graph import compile_compositions
            return compile_compositions(self, EffectsGraphCanvas._single_pass_payload)
        return self._single_pass_payload()

    def _single_pass_payload(self) -> dict[str, object]:
        mix_modes = {"add", "multiply", "subtract", "overlay", "difference"}

        def resolve_generated_alpha(
            source_id: str | None,
            source_port: str,
        ) -> tuple[str | None, str, dict[str, object]]:
            if source_id is None or source_port != "alpha_output":
                return source_id, source_port, {}
            source_node = self._nodes.get(source_id, {})
            source_type = source_node.get("type")
            settings = source_node.get("settings", {})
            if not isinstance(settings, dict):
                settings = {}
            if source_type == "mask":
                return source_id, "alpha_output", {"type": "mask", "settings": dict(settings)}
            if source_type not in {"chroma_key", "luma_key"}:
                return source_id, source_port, {}
            incoming = next(
                (candidate for candidate in self._connections if candidate[2:4] == (source_id, "color_input")),
                None,
            )
            if incoming is None:
                return None, "", {}
            return incoming[0], "alpha_output", {"type": source_type, "settings": dict(settings)}

        def resolve_transform_source(
            connection: tuple[str, str, str, str] | None,
        ) -> tuple[str | None, str, dict[str, object]]:
            if connection is None:
                return None, "", {}
            source_id, source_port = connection[0], connection[1]
            transform_settings: dict[str, object] = {}
            visited: set[str] = set()

            while source_id is not None and source_id not in visited:
                visited.add(source_id)
                source_node = self._nodes.get(source_id, {})
                source_type = source_node.get("type")

                if source_type == "transform_3d":
                    candidate_settings = source_node.get("settings", {})
                    if isinstance(candidate_settings, dict):
                        transform_settings = dict(candidate_settings)
                    preferred_inputs = (
                        ("alpha_input", "input")
                        if source_port == "alpha_output"
                        else ("input", "alpha_input")
                    )
                    incoming = next(
                        (
                            candidate
                            for input_port in preferred_inputs
                            for candidate in self._connections
                            if candidate[2:4] == (source_id, input_port)
                        ),
                        None,
                    )
                    if incoming is None:
                        return None, "", transform_settings
                    source_id = incoming[0]
                    source_port = (
                        "alpha_output"
                        if source_port == "alpha_output"
                        or incoming[1] == "alpha_output"
                        or incoming[3] == "alpha_input"
                        else "output"
                    )
                    continue

                if source_type in {"proc_amp", "color_corrector"}:
                    incoming = next(
                        (candidate for candidate in self._connections if candidate[2:4] == (source_id, "input")),
                        None,
                    )
                    if incoming is None:
                        return None, "", transform_settings
                    source_id, source_port = incoming[0], incoming[1]
                    continue

                return source_id, source_port, transform_settings

            return None, "", transform_settings

        def build_color_stage_payload(
            node_type: str,
            settings: dict[str, object],
        ) -> dict[str, object] | None:
            if node_type == "proc_amp":
                return {
                    "type": node_type,
                    "params": [
                        float(settings.get("saturation", 1.0)),
                        float(settings.get("hue", 0.0)),
                        float(settings.get("brightness", 0.0)),
                        float(settings.get("contrast", 1.0)),
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "invert": bool(settings.get("invert", False)),
                }
            if node_type == "color_corrector":
                return {
                    "type": node_type,
                    "params": [
                        float(settings.get("gain_r", 1.0)),
                        float(settings.get("gain_g", 1.0)),
                        float(settings.get("gain_b", 1.0)),
                        float(settings.get("mid_r", 1.0)),
                        float(settings.get("mid_g", 1.0)),
                        float(settings.get("mid_b", 1.0)),
                        float(settings.get("blacks_r", 0.0)),
                        float(settings.get("blacks_g", 0.0)),
                        float(settings.get("blacks_b", 0.0)),
                    ],
                    "invert": False,
                }
            return None

        def resolve_effect_chain(
            connection: tuple[str, str, str, str] | None,
        ) -> tuple[str | None, str, dict[str, object], dict[str, object], list[dict[str, object]]]:
            if connection is None:
                return None, "", {}, {}, []

            source_id, source_port = connection[0], connection[1]
            transform_settings: dict[str, object] = {}
            blur_settings: dict[str, object] = {}
            color_stage_payloads: list[dict[str, object]] = []
            visited: set[str] = set()

            while source_id is not None and source_id not in visited:
                visited.add(source_id)
                source_node = self._nodes.get(source_id, {})
                source_type = source_node.get("type")
                settings = source_node.get("settings", {})
                if not isinstance(settings, dict):
                    settings = {}

                if source_type == "transform_3d":
                    if not transform_settings:
                        transform_settings = dict(settings)
                    preferred_inputs = (
                        ("alpha_input", "input")
                        if source_port == "alpha_output"
                        else ("input", "alpha_input")
                    )
                    incoming = next(
                        (
                            candidate
                            for input_port in preferred_inputs
                            for candidate in self._connections
                            if candidate[2:4] == (source_id, input_port)
                        ),
                        None,
                    )
                    if incoming is None:
                        return None, "", transform_settings, blur_settings, color_stage_payloads
                    source_id = incoming[0]
                    source_port = (
                        "alpha_output"
                        if source_port == "alpha_output"
                        or incoming[1] == "alpha_output"
                        or incoming[3] == "alpha_input"
                        else "output"
                    )
                    continue

                if source_type == "blur":
                    if not blur_settings:
                        blur_settings = dict(settings)
                    preferred_input = "alpha_input" if source_port == "alpha_output" else "input"
                    incoming = next(
                        (candidate for candidate in self._connections if candidate[2:4] == (source_id, preferred_input)),
                        None,
                    )
                    if incoming is None:
                        return None, "", transform_settings, blur_settings, color_stage_payloads
                    source_id, source_port = incoming[0], incoming[1]
                    continue

                if source_type in {"proc_amp", "color_corrector"}:
                    stage_payload = build_color_stage_payload(str(source_type), settings)
                    if stage_payload is not None:
                        color_stage_payloads.insert(0, stage_payload)
                    incoming = next(
                        (candidate for candidate in self._connections if candidate[2:4] == (source_id, "input")),
                        None,
                    )
                    if incoming is None:
                        return None, "", transform_settings, blur_settings, color_stage_payloads
                    source_id, source_port = incoming[0], incoming[1]
                    continue

                return source_id, source_port, transform_settings, blur_settings, color_stage_payloads

            return None, "", transform_settings, blur_settings, color_stage_payloads

        def resolve_associated_alpha_connection(
            connection: tuple[str, str, str, str] | None,
        ) -> tuple[str, str, str, str] | None:
            current = connection
            visited: set[str] = set()
            while current is not None and current[0] not in visited:
                source_id, source_port = current[0], current[1]
                visited.add(source_id)
                source_type = self._nodes.get(source_id, {}).get("type")
                if source_type == "transform_3d" and source_port == "output":
                    return next(
                        (
                            candidate for candidate in self._connections
                            if candidate[2:4] == (source_id, "alpha_input")
                        ),
                        None,
                    )
                if source_type in {"blur", "proc_amp", "color_corrector"}:
                    current = next(
                        (
                            candidate for candidate in self._connections
                            if candidate[2:4] == (source_id, "input")
                        ),
                        None,
                    )
                    continue
                break
            return None

        def resolve_channel_recombiner(
            connection: tuple[str, str, str, str] | None,
        ) -> tuple[str | None, list[dict[str, object]], list[dict[str, object]]]:
            if connection is None or self._nodes.get(connection[0], {}).get("type") != "color_recombiner":
                return None, [], []
            recombiner_id = connection[0]
            source_ids: set[str] = set()
            routes: list[dict[str, object]] = []
            alpha_generators: list[dict[str, object]] = []
            component_ports = ("red_input", "green_input", "blue_input", "alpha_input")
            splitter_channels = {"red_output": 0, "green_output": 1, "blue_output": 2}
            for component_index, component_port in enumerate(component_ports):
                current = next(
                    (candidate for candidate in self._connections if candidate[2:4] == (recombiner_id, component_port)),
                    None,
                )
                route: dict[str, object] = {
                    "source_channel": -1,
                    "generator_type": "off",
                    "blur_method": "off",
                    "blur_radius": 0.0,
                    "transform_x": 0.0,
                    "transform_y": 0.0,
                    "transform_z": 0.0,
                    "rotate_x": 0.0,
                    "rotate_y": 0.0,
                    "rotate_z": 0.0,
                }
                visited: set[str] = set()
                while current is not None and current[0] not in visited:
                    source_id, source_port = current[0], current[1]
                    visited.add(source_id)
                    source_type = self._nodes.get(source_id, {}).get("type")
                    settings = self._nodes.get(source_id, {}).get("settings", {})
                    if not isinstance(settings, dict):
                        settings = {}
                    if source_type == "transform_3d":
                        for name in ("x", "y", "z", "rotate_x", "rotate_y", "rotate_z", "aspect_x", "aspect_y"):
                            route["transform_" + name if name in {"x", "y", "z"} else name] = float(settings.get(name, 1.0 if name.startswith("aspect_") else 0.0))
                        preferred_input = "alpha_input" if source_port == "alpha_output" else "input"
                        current = next(
                            (candidate for candidate in self._connections if candidate[2:4] == (source_id, preferred_input)),
                            None,
                        )
                        continue
                    if source_type == "blur":
                        route["blur_method"] = str(settings.get("method", "off"))
                        route["blur_radius"] = float(settings.get("radius", 0.0))
                        preferred_input = "alpha_input" if source_port == "alpha_output" else "input"
                        current = next(
                            (candidate for candidate in self._connections if candidate[2:4] == (source_id, preferred_input)),
                            None,
                        )
                        continue
                    if source_type == "color_splitter":
                        current = next(
                            (candidate for candidate in self._connections if candidate[2:4] == (source_id, "input")),
                            None,
                        )
                        if current is not None:
                            route["source_channel"] = (
                                3 if current[1] == "alpha_output"
                                else splitter_channels.get(source_port, component_index)
                            )
                            source_ids.add(current[0])
                        break
                    if source_type == "mix" and source_port == "alpha_output":
                        alpha_mix_base, alpha_mix_ops, alpha_mix_sources, alpha_mix_has_mask = resolve_alpha_mix_chain(current)
                        if alpha_mix_base is not None:
                            route["source_channel"] = 3
                            route["generator_type"] = "alpha_mix"
                            route["alpha_mix_base"] = alpha_mix_base
                            route["alpha_mix_ops"] = list(alpha_mix_ops)
                            route["alpha_mix_mask_only"] = not alpha_mix_sources and alpha_mix_has_mask
                            source_ids.update(
                                source
                                for source in alpha_mix_sources
                                if source is not None and self._nodes.get(source, {}).get("type") != "mask"
                            )
                            if bool(route["alpha_mix_mask_only"]):
                                alpha_generators.append({"type": "mask", "settings": {}})
                            break
                    resolved_id, resolved_port, alpha_generator = resolve_generated_alpha(source_id, source_port)
                    if alpha_generator:
                        route["source_channel"] = 3
                        route["generator_type"] = str(alpha_generator.get("type", "off"))
                        generator_settings = alpha_generator.get("settings", {})
                        if isinstance(generator_settings, dict):
                            route["generator_settings"] = dict(generator_settings)
                        alpha_generators.append(alpha_generator)
                        if resolved_id is not None and self._nodes.get(resolved_id, {}).get("type") != "mask":
                            source_ids.add(resolved_id)
                        break
                    route["source_channel"] = 3 if source_port == "alpha_output" else component_index
                    source_ids.add(source_id)
                    break
                routes.append(route)
            if not source_ids and alpha_generators and all(generator.get("type") == "mask" for generator in alpha_generators):
                source_ids.add(recombiner_id)
            return (next(iter(source_ids)) if len(source_ids) == 1 else None), routes, alpha_generators

        def build_alpha_mix_operand(
            connection: tuple[str, str, str, str] | None,
        ) -> tuple[dict[str, object] | None, set[str], bool]:
            (
                operand_source_id,
                operand_source_port,
                operand_transform_settings,
                operand_blur_settings,
                operand_color_stages,
            ) = resolve_effect_chain(connection)
            source_channel = 3
            splitter_channels = {"red_output": 0, "green_output": 1, "blue_output": 2}
            if (
                self._nodes.get(operand_source_id, {}).get("type") == "color_splitter"
                and operand_source_port in splitter_channels
            ):
                splitter_input = next(
                    (
                        candidate for candidate in self._connections
                        if candidate[2:4] == (operand_source_id, "input")
                    ),
                    None,
                )
                source_channel = splitter_channels[operand_source_port]
                (
                    operand_source_id,
                    operand_source_port,
                    upstream_transform_settings,
                    upstream_blur_settings,
                    upstream_color_stages,
                ) = resolve_effect_chain(splitter_input)
                if not operand_transform_settings:
                    operand_transform_settings = upstream_transform_settings
                if not operand_blur_settings:
                    operand_blur_settings = upstream_blur_settings
                operand_color_stages = upstream_color_stages + operand_color_stages
            resolved_operand_id, resolved_operand_port, alpha_generator = resolve_generated_alpha(
                operand_source_id,
                operand_source_port,
            )
            operand: dict[str, object] = {
                "blur_method": str(operand_blur_settings.get("method", "off")),
                "blur_radius": float(operand_blur_settings.get("radius", 0.0)),
                "transform_x": float(operand_transform_settings.get("x", 0.0)),
                "transform_y": float(operand_transform_settings.get("y", 0.0)),
                "transform_z": float(operand_transform_settings.get("z", 0.0)),
                "rotate_x": float(operand_transform_settings.get("rotate_x", 0.0)),
                "rotate_y": float(operand_transform_settings.get("rotate_y", 0.0)),
                "rotate_z": float(operand_transform_settings.get("rotate_z", 0.0)),
                "aspect_x": float(operand_transform_settings.get("aspect_x", 1.0)),
                "aspect_y": float(operand_transform_settings.get("aspect_y", 1.0)),
                "color_stages": list(operand_color_stages),
            }
            operand_source_ids: set[str] = set()
            if alpha_generator:
                generator_settings = alpha_generator.get("settings", {})
                if not isinstance(generator_settings, dict):
                    generator_settings = {}
                generator_type = str(alpha_generator.get("type", "off"))
                if generator_type == "mask":
                    operand.update(
                        {
                            "type": "mask",
                            "pattern": str(generator_settings.get("pattern", "off")),
                            "softness": float(generator_settings.get("softness", 0.0)),
                            "aspect": float(generator_settings.get("aspect", 1.0)),
                            "invert": bool(generator_settings.get("invert", False)),
                            "size": float(generator_settings.get("size", 1.0)),
                            "x": float(generator_settings.get("x", 0.0)),
                            "y": float(generator_settings.get("y", 0.0)),
                            "rotation": float(generator_settings.get("rotation", 0.0)),
                        }
                    )
                    return operand, operand_source_ids, True
                if generator_type == 'luma_key':
                    operand.update(type='luma_key', source_from_effects_input=resolved_operand_id == 'effects_input',
                                   key_similarity=float(generator_settings.get('clip', 0.0)),
                                   key_softness=float(generator_settings.get('gain', 1.0)),
                                   key_invert=bool(generator_settings.get('invert', False)))
                    if resolved_operand_id is not None:
                        operand_source_ids.add(resolved_operand_id)
                    return operand, operand_source_ids, False
                if generator_type == "chroma_key":
                    operand.update(
                        {
                            "type": "chroma_key",
                            "source_from_effects_input": resolved_operand_id == "effects_input",
                            "key_color_r": int(generator_settings.get("key_color_r", 0)),
                            "key_color_g": int(generator_settings.get("key_color_g", 255)),
                            "key_color_b": int(generator_settings.get("key_color_b", 0)),
                            "key_similarity": float(generator_settings.get("key_similarity", 0.25)),
                            "key_softness": float(generator_settings.get("key_softness", 0.10)),
                            "key_edge_feather": float(generator_settings.get("key_edge_feather", 0.0)),
                            "key_invert": bool(generator_settings.get("key_invert", False)),
                        }
                    )
                    if resolved_operand_id is not None and self._nodes.get(resolved_operand_id, {}).get("type") != "mask":
                        operand_source_ids.add(resolved_operand_id)
                    return operand, operand_source_ids, False
                return None, operand_source_ids, False
            if resolved_operand_id is None or resolved_operand_port not in {"output", "alpha_output"}:
                return None, operand_source_ids, False
            operand["type"] = (
                "source_alpha"
                if resolved_operand_port == "alpha_output" or source_channel != 3
                else "source_luma"
            )
            operand["source_from_effects_input"] = resolved_operand_id == "effects_input"
            operand["source_node_id"] = resolved_operand_id
            operand["source_channel"] = source_channel
            operand_source_ids.add(resolved_operand_id)
            return operand, operand_source_ids, False

        def resolve_alpha_mix_chain(
            connection: tuple[str, str, str, str] | None,
        ) -> tuple[dict[str, object] | None, list[dict[str, object]], set[str], bool]:
            if connection is None:
                return None, [], set(), False
            mix_node = self._nodes.get(connection[0], {}) if connection[1] == "alpha_output" else {}
            if mix_node.get("type") != "mix":
                operand, operand_sources, operand_is_mask = build_alpha_mix_operand(connection)
                return operand, [], operand_sources, operand_is_mask
            mix_id = str(connection[0])
            left_connection = next(
                (candidate for candidate in self._connections if candidate[2:4] == (mix_id, "a_input")),
                None,
            )
            right_connection = next(
                (candidate for candidate in self._connections if candidate[2:4] == (mix_id, "b_input")),
                None,
            )
            base_operand, mix_ops, operand_sources, has_mask_operand = resolve_alpha_mix_chain(left_connection)
            right_operand, right_sources, right_is_mask = build_alpha_mix_operand(right_connection)
            if base_operand is None or right_operand is None:
                return None, [], set(), False
            mix_settings = mix_node.get("settings", {})
            if not isinstance(mix_settings, dict):
                mix_settings = {}
            mode = str(mix_settings.get("mode", "add")).strip().lower().replace(" ", "_")
            if mode not in mix_modes:
                mode = "add"
            mix_ops.append(
                {
                    "mode": mode,
                    "factor": max(0.0, min(1.0, float(mix_settings.get("factor", 1.0)))),
                    "operand": right_operand,
                }
            )
            return base_operand, mix_ops, operand_sources | right_sources, has_mask_operand or right_is_mask

        edges = {
            source: target
            for source, source_port, target, target_port in self._connections
            if source_port == "output" and target_port in {"input", "layer1_color"}
        }
        active_nodes: list[str] = []
        main_source_id = self._active_main_source()
        current = main_source_id
        main_capture_id = current if self._nodes.get(current, {}).get("type") == "capture" else None
        visited: set[str] = set()
        while current in edges and current not in visited:
            visited.add(current)
            current = edges[current]
            active_nodes.append(current)
            if current == "effects_output":
                break
        output_source_connection = next(
            (
                connection for connection in self._connections
                if connection[2] == "effects_output" and connection[3] == "input"
            ),
            None,
        )
        explicit_compositor_id = (
            output_source_connection[0]
            if output_source_connection is not None
            and self._nodes.get(output_source_connection[0], {}).get("type") == "keying"
            else None
        )
        if explicit_compositor_id is not None:
            if explicit_compositor_id in active_nodes:
                compositor_position = active_nodes.index(explicit_compositor_id)
                active_nodes = active_nodes[: compositor_position + 1]
                if not active_nodes or active_nodes[-1] != "effects_output":
                    active_nodes.append("effects_output")
            else:
                active_nodes = [explicit_compositor_id, "effects_output"]
        direct_source_id, direct_source_port, direct_transform_settings = resolve_transform_source(
            output_source_connection
        )
        if self._nodes.get(direct_source_id, {}).get("type") not in {"media", "matte"}:
            direct_source_id = None
        if direct_source_id is not None:
            active_nodes = [direct_source_id, "effects_output"]
        compositor_ids = [node_id for node_id in active_nodes if self._nodes.get(node_id, {}).get("type") == "keying"]
        compositor_id = compositor_ids[-1] if compositor_ids else None
        compositor_position = active_nodes.index(compositor_id) if compositor_id is not None else -1
        base_transform_ids = [
            node_id for position, node_id in enumerate(active_nodes)
            if compositor_position < 0
            and main_source_id == "effects_input"
            and self._nodes.get(node_id, {}).get("type") == "transform_3d"
        ]
        base_transform_settings = (
            self._nodes[base_transform_ids[-1]].get("settings", {})
            if base_transform_ids
            else {}
        )
        if not isinstance(base_transform_settings, dict):
            base_transform_settings = {}
        base_transform_active = any(abs(float(base_transform_settings.get(name, 1.0)) - 1.0) > 1e-6 for name in ("aspect_x", "aspect_y")) or any(
            abs(float(base_transform_settings.get(name, 0.0))) > 1e-6
            for name in ("x", "y", "z", "rotate_x", "rotate_y", "rotate_z")
        )
        color_stages: list[dict[str, object]] = []
        for position, node_id in enumerate(active_nodes):
            node = self._nodes.get(node_id, {})
            node_type = str(node.get("type", ""))
            settings = node.get("settings", {})
            if node_type not in {"proc_amp", "color_corrector"} or not isinstance(settings, dict):
                continue
            if compositor_position >= 0 and position <= compositor_position:
                continue
            stage_payload = build_color_stage_payload(node_type, settings)
            if stage_payload is None:
                continue
            stage_payload["after_composite"] = compositor_position >= 0 and position > compositor_position
            color_stages.append(stage_payload)
        layer2_media_id = next(
            (
                source for source, source_port, target, target_port in self._connections
                if compositor_id is not None and target == compositor_id and target_port == "layer2_color"
                and self._nodes.get(source, {}).get("type") == "media"
            ),
            None,
        )
        layer2_capture_id = next(
            (
                source for source, source_port, target, target_port in self._connections
                if compositor_id is not None and target == compositor_id and target_port == "layer2_color"
                and source_port == "output" and self._nodes.get(source, {}).get("type") == "capture"
            ),
            None,
        )
        layer2_matte_id = next(
            (
                source for source, _source_port, target, target_port in self._connections
                if compositor_id is not None and target == compositor_id and target_port == "layer2_color"
                and self._nodes.get(source, {}).get("type") == "matte"
            ),
            None,
        )
        layer2_mask_id = next(
            (
                source for source, source_port, target, target_port in self._connections
                if compositor_id is not None and target == compositor_id and target_port == "layer2_alpha"
                and source_port == "alpha_output" and self._nodes.get(source, {}).get("type") == "mask"
            ),
            None,
        )
        capture_source_id = layer2_capture_id or main_capture_id
        capture_settings = self._nodes.get(capture_source_id, {}).get("settings", {})
        if not isinstance(capture_settings, dict):
            capture_settings = {}
        capture_device_id = capture_settings.get("device_index")
        capture_kind = _capture_backend(capture_device_id)
        capture_device_index = int(capture_device_id.split(":", 1)[1]) if capture_kind else -1
        media_nodes = [node_id for node_id in active_nodes if self._nodes.get(node_id, {}).get("type") == "media"]
        if layer2_media_id is not None and layer2_media_id not in media_nodes:
            media_nodes.append(layer2_media_id)
        blur_ids = [node_id for node_id in active_nodes if self._nodes.get(node_id, {}).get("type") == "blur"] if compositor_id is None else []
        if not active_nodes or active_nodes[-1] != "effects_output":
            return {"enabled": False, "output_connected": False}
        media_id = media_nodes[0] if media_nodes else None
        media_settings = self._nodes[media_id]["settings"] if media_id is not None else {}
        if not isinstance(media_settings, dict):
            media_settings = {}
        direct_source_type = self._nodes.get(direct_source_id, {}).get("type")
        matte_id = direct_source_id if direct_source_type == "matte" else layer2_matte_id
        matte_settings = self._nodes.get(matte_id, {}).get("settings", {})
        if not isinstance(matte_settings, dict):
            matte_settings = {}
        mask_settings = self._nodes.get(layer2_mask_id, {}).get("settings", {})
        if not isinstance(mask_settings, dict):
            mask_settings = {}
        blend_mode = "normal"
        key_opacity = 1.0
        compositor_settings: dict[str, object] = {}
        if compositor_id is not None and isinstance(self._nodes[compositor_id]["settings"], dict):
            compositor_settings = self._nodes[compositor_id]["settings"]
            blend_mode = str(compositor_settings.get("layer2_blend_mode", "normal"))
            key_opacity = float(compositor_settings.get("layer2_opacity", 1.0))
        blur_settings: dict[str, object] = {"method": "off", "radius": 0, "target": "both"}
        selected_blur = blur_ids
        if selected_blur:
            candidate = self._nodes[selected_blur[-1]]["settings"]
            if isinstance(candidate, dict):
                blur_settings = dict(candidate)
        direct_uploaded_source = direct_source_id is not None or (capture_kind in {"webcam", "logical"} and compositor_id is None)
        uploaded_source_id = capture_source_id or media_id or matte_id
        effect_color_from_alpha = any(
            source == uploaded_source_id and source_port == "alpha_output"
            and ((target == "effects_output" and target_port == "input")
                 or (target == compositor_id and target_port == "layer2_color"))
            for source, source_port, target, target_port in self._connections
        )
        if direct_source_id is not None:
            effect_color_from_alpha = direct_source_port == "alpha_output"
        effect_alpha_from_color = any(
            source == uploaded_source_id and source_port == "output"
            and target == compositor_id and target_port == "layer2_alpha"
            for source, source_port, target, target_port in self._connections
        )
        layer1_opacity = 0.0 if direct_uploaded_source else float(compositor_settings.get("layer1_opacity", 1.0))
        source_opacity = float(matte_settings.get("alpha", 1.0)) if matte_id is not None else float(media_settings.get("opacity", 1.0))
        layer2_opacity = max(0.0, min(1.0, source_opacity * key_opacity))
        key_config = {
            "mode": "off", "color": [0, 255, 0], "similarity": 0.25,
            "softness": 0.10, "edge_feather": 0.0, "spill_suppression": 0.25,
            "luma_low": 0.0, "luma_high": 1.0, "luma_softness": 0.10, "invert": False,
        }
        layer_payloads: list[dict[str, object]] = []
        layer_count = max(2, min(64, int(compositor_settings.get("layer_count", 2))))
        if compositor_id is not None:
            for layer_index in range(1, layer_count + 1):
                color_connection = next(
                    (
                        connection for connection in self._connections
                        if connection[2:4] == (compositor_id, f"layer{layer_index}_color")
                    ),
                    None,
                )
                alpha_connection = next(
                    (
                        connection for connection in self._connections
                        if connection[2:4] == (compositor_id, f"layer{layer_index}_alpha")
                    ),
                    None,
                )
                if alpha_connection is None:
                    alpha_connection = resolve_associated_alpha_connection(color_connection)
                recombiner_source_id, channel_routes, channel_alpha_generators = resolve_channel_recombiner(color_connection)
                (
                    color_source_id,
                    color_source_port,
                    color_transform_settings,
                    color_blur_settings,
                    layer_color_stages,
                ) = resolve_effect_chain(color_connection)
                (
                    alpha_source_id,
                    alpha_source_port,
                    alpha_transform_settings,
                    alpha_blur_settings,
                    _alpha_color_stages,
                ) = resolve_effect_chain(alpha_connection)
                alpha_mix_base, alpha_mix_ops, alpha_mix_source_ids, alpha_mix_has_mask = resolve_alpha_mix_chain(
                    alpha_connection
                )
                color_mix_to_color = bool(
                    color_connection is not None
                    and color_connection[1] == "alpha_output"
                    and self._nodes.get(color_connection[0], {}).get("type") == "mix"
                )
                color_mix_source_ids: set[str] = set()
                color_mix_has_mask = False
                if color_mix_to_color:
                    (
                        color_mix_base,
                        color_mix_ops,
                        color_mix_source_ids,
                        color_mix_has_mask,
                    ) = resolve_alpha_mix_chain(color_connection)
                    if color_mix_base is not None:
                        channel_routes = [
                            {
                                "source_channel": 3,
                                "generator_type": "alpha_mix",
                                "alpha_mix_base": color_mix_base,
                                "alpha_mix_ops": list(color_mix_ops),
                            }
                            for _target_channel in range(3)
                        ] + [{"source_channel": -1, "generator_type": "off"}]
                color_source_id, color_source_port, direct_alpha_generator = resolve_generated_alpha(
                    color_source_id, color_source_port
                )
                if color_mix_to_color:
                    color_source_id = None
                    color_source_port = "alpha_output"
                resolved_alpha_source_id, resolved_alpha_source_port, opacity_alpha_generator = resolve_generated_alpha(
                    alpha_source_id, alpha_source_port
                )
                alpha_mix_concrete_sources = {
                    source
                    for source in alpha_mix_source_ids | color_mix_source_ids
                    if source is not None and source != "effects_input"
                }
                alpha_mix_source_id = next(iter(alpha_mix_concrete_sources)) if len(alpha_mix_concrete_sources) == 1 else None
                alpha_mix_effects_input = "effects_input" in alpha_mix_source_ids | color_mix_source_ids
                source_id = (
                    recombiner_source_id
                    or color_source_id
                    or alpha_mix_source_id
                    or ("effects_input" if alpha_mix_effects_input else None)
                    or alpha_source_id
                )
                source_port = color_source_port if color_source_id is not None else alpha_source_port
                transform_settings = (
                    color_transform_settings
                    if color_transform_settings
                    else alpha_transform_settings
                )
                layer_blur_settings = (
                    color_blur_settings
                    if color_blur_settings
                    else alpha_blur_settings
                )
                source_node = self._nodes.get(source_id, {})
                source_type = str(source_node.get("type", "")) if isinstance(source_node, dict) else ""
                mask_generator = next(
                    (generator for generator in (direct_alpha_generator,) if generator.get("type") == "mask"),
                    None,
                )
                chroma_generator = next(
                    (
                        generator for generator in (direct_alpha_generator, opacity_alpha_generator)
                        if generator.get("type") == "chroma_key"
                    ),
                    None,
                )
                source_settings = source_node.get("settings", {}) if isinstance(source_node, dict) else {}
                if not isinstance(source_settings, dict):
                    source_settings = {}
                layer_capture_id = source_settings.get("device_index") if source_type == "capture" else None
                layer_capture_kind = _capture_backend(layer_capture_id)
                layer_capture_index = int(layer_capture_id.split(":", 1)[1]) if layer_capture_kind else -1
                layer_media_path = str(source_settings.get("path", "")) if source_type == "media" else ""
                layer_matte_rgba = [
                    int(source_settings.get("red", 255)),
                    int(source_settings.get("green", 255)),
                    int(source_settings.get("blue", 255)),
                    255,
                ]
                layer_source_opacity = (
                    float(source_settings.get("alpha", 1.0))
                    if source_type == "matte"
                    else float(source_settings.get("opacity", 1.0))
                )
                mask_id = None
                if alpha_connection is not None and self._nodes.get(alpha_connection[0], {}).get("type") == "mask":
                    mask_id = alpha_connection[0]
                layer_mask_settings = self._nodes.get(mask_id, {}).get("settings", {})
                if mask_generator is not None:
                    layer_mask_settings = mask_generator.get("settings", {})
                if not isinstance(layer_mask_settings, dict):
                    layer_mask_settings = {}
                prefix = f"layer{layer_index}_"
                chroma_key_id = (
                    alpha_connection[0]
                    if alpha_connection is not None
                    and self._nodes.get(alpha_connection[0], {}).get("type") == "chroma_key"
                    else None
                )
                chroma_key_settings = self._nodes.get(chroma_key_id, {}).get("settings", {})
                if chroma_generator is not None:
                    chroma_key_settings = chroma_generator.get("settings", {})
                if not isinstance(chroma_key_settings, dict):
                    chroma_key_settings = {}
                layer_key = dict(key_config)
                if chroma_key_id is not None or chroma_generator is not None:
                    layer_key.update({
                        "mode": "chroma",
                        "color": [
                            int(chroma_key_settings.get("key_color_r", 0)),
                            int(chroma_key_settings.get("key_color_g", 255)),
                            int(chroma_key_settings.get("key_color_b", 0)),
                        ],
                        "similarity": float(chroma_key_settings.get("key_similarity", 0.25)),
                        "softness": float(chroma_key_settings.get("key_softness", 0.10)),
                        "edge_feather": float(chroma_key_settings.get("key_edge_feather", 0.0)),
                        "spill_suppression": float(chroma_key_settings.get("spill_suppression", 0.25)),
                        "invert": bool(chroma_key_settings.get("key_invert", False)),
                    })
                generated_mask_source = source_type in {"mask", "color_recombiner"} and (
                    mask_generator is not None
                    or any(generator.get("type") == "mask" for generator in channel_alpha_generators)
                )
                if (
                    (alpha_mix_base is not None and alpha_mix_has_mask)
                    or (color_mix_to_color and color_mix_has_mask)
                ) and source_id is None:
                    generated_mask_source = True
                effects_input_source = source_type == "effects_input"
                if alpha_mix_base is not None and source_id == "effects_input":
                    effects_input_source = True
                layer_enabled = (
                    alpha_mix_base is not None
                    or source_type == "matte" or generated_mask_source or effects_input_source
                    or bool(layer_media_path) or layer_capture_kind in {"webcam", "logical"}
                )
                preserve_color_from_alpha_opacity = bool(
                    color_connection is not None
                    and alpha_connection is not None
                    and color_connection[0:2] == alpha_connection[0:2]
                    and color_source_port == "alpha_output"
                )
                image_sources: list[dict[str, object]] = []
                source_slots: dict[str, int] = {}

                def assign_operand_source(operand: dict[str, object] | None) -> None:
                    if not isinstance(operand, dict):
                        return
                    # RGB channel routes can share the same operand dictionary.
                    # Once resolved, retain its slot/live-input identity on later visits.
                    if "source_node_id" not in operand:
                        return
                    operand_node_id = str(operand.pop("source_node_id", ""))
                    if not operand_node_id or operand_node_id == "effects_input":
                        operand["source_from_effects_input"] = operand_node_id == "effects_input"
                        return
                    if operand_node_id == source_id:
                        operand["source_slot"] = 0
                        return
                    source_slot = source_slots.get(operand_node_id)
                    if source_slot is None:
                        source_slot = len(source_slots) + 1
                        if source_slot > 7:
                            raise ValueError("A compositor layer supports at most 7 auxiliary image sources")
                        source_slots[operand_node_id] = source_slot
                        operand_node = self._nodes.get(operand_node_id, {})
                        operand_settings = operand_node.get("settings", {}) if isinstance(operand_node, dict) else {}
                        if not isinstance(operand_settings, dict):
                            operand_settings = {}
                        operand_type = str(operand_node.get("type", "")) if isinstance(operand_node, dict) else ""
                        operand_capture_id = operand_settings.get("device_index") if operand_type == "capture" else None
                        operand_capture_kind = _capture_backend(operand_capture_id)
                        image_sources.append(
                            {
                                "slot": source_slot,
                                "source_node_id": operand_node_id,
                                "source_kind": "matte" if operand_type == "matte" else (operand_capture_kind if operand_capture_kind else "media"),
                                "media_path": str(operand_settings.get("path", "")) if operand_type == "media" else "",
                                "media_playing": operand_capture_kind in {"webcam", "logical"} or bool(operand_settings.get("playing", False)),
                                "media_loop": bool(operand_settings.get("loop", True)),
                                "capture_kind": operand_capture_kind,
                                "capture_device_index": int(operand_capture_id.split(":", 1)[1]) if operand_capture_kind else -1,
                                "capture_width": int(operand_settings.get("capture_width", 0)),
                                "capture_height": int(operand_settings.get("capture_height", 0)),
                                "capture_reload_token": self._capture_reload_tokens.get(operand_node_id, 0),
                                "matte_rgba": [
                                    int(operand_settings.get("red", 255)),
                                    int(operand_settings.get("green", 255)),
                                    int(operand_settings.get("blue", 255)),
                                    255,
                                ],
                            }
                        )
                    operand["source_slot"] = source_slot

                assign_operand_source(alpha_mix_base)
                for alpha_mix_op in alpha_mix_ops:
                    if isinstance(alpha_mix_op, dict):
                        assign_operand_source(alpha_mix_op.get("operand"))
                for channel_route in channel_routes:
                    if not isinstance(channel_route, dict):
                        continue
                    assign_operand_source(channel_route.get("alpha_mix_base"))
                    route_alpha_mix_ops = channel_route.get("alpha_mix_ops", [])
                    if isinstance(route_alpha_mix_ops, list):
                        for route_alpha_mix_op in route_alpha_mix_ops:
                            if isinstance(route_alpha_mix_op, dict):
                                assign_operand_source(route_alpha_mix_op.get("operand"))
                layer_payloads.append(
                    {
                        "layer_index": layer_index,
                        "enabled": layer_enabled,
                        "source_node_id": source_id,
                        "source_kind": (
                            "effects_input" if effects_input_source
                            else "matte" if source_type == "matte" or generated_mask_source
                            else layer_capture_kind if layer_capture_kind
                            else "media"
                        ),
                        "media_path": layer_media_path,
                        "media_playing": layer_capture_kind in {"webcam", "logical"} or bool(source_settings.get("playing", False)),
                        "media_loop": bool(source_settings.get("loop", True)),
                        "capture_kind": layer_capture_kind,
                        "capture_device_index": layer_capture_index,
                        "capture_width": int(source_settings.get("capture_width", 0)),
                        "capture_height": int(source_settings.get("capture_height", 0)),
                        "capture_reload_token": self._capture_reload_tokens.get(source_id, 0),
                        "matte_rgba": [255, 255, 255, 255] if generated_mask_source else layer_matte_rgba,
                        "opacity": max(0.0, min(1.0, layer_source_opacity * float(compositor_settings.get(prefix + "opacity", 1.0)))),
                        "blend_mode": str(compositor_settings.get(prefix + "blend_mode", "normal")),
                        "blur_method": str(layer_blur_settings.get("method", "off")),
                        "blur_radius": float(layer_blur_settings.get("radius", 0.0)),
                        "blur_target": str(layer_blur_settings.get("target", "both")),
                        "color_stages": list(layer_color_stages),
                        "alpha_mix_base": alpha_mix_base,
                        "alpha_mix_ops": list(alpha_mix_ops),
                        "image_sources": image_sources,
                        "key": layer_key,
                        "materialize_key_alpha": alpha_mix_base is None and chroma_generator is not None,
                        "key_alpha_from_effects_input": bool(
                            alpha_mix_base is None and
                            opacity_alpha_generator.get("type") == "chroma_key"
                            and resolved_alpha_source_id == "effects_input"
                            and resolved_alpha_source_port == "alpha_output"
                        ),
                        "effect_color_from_alpha": not color_mix_to_color and color_source_port == "alpha_output",
                        "effect_alpha_from_color": bool(
                            alpha_mix_base is None and
                            alpha_source_id == source_id
                            and alpha_source_port == "output"
                        ),
                        "preserve_color_from_alpha_opacity": preserve_color_from_alpha_opacity,
                        "mask_pattern": (
                            str(layer_mask_settings.get("pattern", "off"))
                            if alpha_mix_base is None and (mask_id is not None or mask_generator is not None)
                            else "off"
                        ),
                        "mask_softness": float(layer_mask_settings.get("softness", 0.0)),
                        "mask_aspect": float(layer_mask_settings.get("aspect", 1.0)),
                        "mask_invert": bool(layer_mask_settings.get("invert", False)),
                        "mask_size": float(layer_mask_settings.get("size", 1.0)),
                        "mask_x": float(layer_mask_settings.get("x", 0.0)),
                        "mask_y": float(layer_mask_settings.get("y", 0.0)),
                        "mask_rotation": float(layer_mask_settings.get("rotation", 0.0)),
                        "transform_x": float(transform_settings.get("x", 0.0)),
                        "transform_y": float(transform_settings.get("y", 0.0)),
                        "transform_z": float(transform_settings.get("z", 0.0)),
                        "rotate_x": float(transform_settings.get("rotate_x", 0.0)),
                        "rotate_y": float(transform_settings.get("rotate_y", 0.0)),
                        "rotate_z": float(transform_settings.get("rotate_z", 0.0)),
                        "aspect_x": float(transform_settings.get("aspect_x", 1.0)),
                        "aspect_y": float(transform_settings.get("aspect_y", 1.0)),
                        "channel_routes": channel_routes,
                    }
                )
        if not layer_payloads:
            layer_payloads.append(
                {
                    "layer_index": 2,
                    "enabled": bool(str(media_settings.get("path", "")).strip()) or capture_kind in {"webcam", "logical"} or matte_id is not None,
                    "source_kind": "matte" if matte_id is not None else (capture_kind if capture_kind else "media"),
                    "media_path": str(media_settings.get("path", "")),
                    "media_playing": capture_kind in {"webcam", "logical"} or bool(media_settings.get("playing", False)),
                    "media_loop": bool(media_settings.get("loop", True)),
                    "capture_kind": capture_kind,
                    "capture_device_index": capture_device_index,
                    "capture_width": int(capture_settings.get("capture_width", 0)),
                    "capture_height": int(capture_settings.get("capture_height", 0)),
                    "capture_reload_token": self._capture_reload_tokens.get(capture_source_id, 0),
                    "matte_rgba": [
                        int(matte_settings.get("red", 255)), int(matte_settings.get("green", 255)),
                        int(matte_settings.get("blue", 255)), 255,
                    ],
                    "opacity": layer2_opacity,
                    "blend_mode": blend_mode,
                    "blur_method": str(blur_settings.get("method", "off")),
                    "blur_radius": float(blur_settings.get("radius", 0.0)),
                    "blur_target": str(blur_settings.get("target", "both")),
                    "key": key_config,
                    "effect_color_from_alpha": effect_color_from_alpha,
                    "effect_alpha_from_color": effect_alpha_from_color,
                    "mask_pattern": str(mask_settings.get("pattern", "off")) if layer2_mask_id is not None else "off",
                    "mask_softness": float(mask_settings.get("softness", 0.0)),
                    "mask_aspect": float(mask_settings.get("aspect", 1.0)),
                    "mask_invert": bool(mask_settings.get("invert", False)),
                    "mask_size": float(mask_settings.get("size", 1.0)),
                    "mask_x": float(mask_settings.get("x", 0.0)),
                    "mask_y": float(mask_settings.get("y", 0.0)),
                    "mask_rotation": float(mask_settings.get("rotation", 0.0)),
                    "transform_x": float(direct_transform_settings.get("x", 0.0)),
                    "transform_y": float(direct_transform_settings.get("y", 0.0)),
                    "transform_z": float(direct_transform_settings.get("z", 0.0)),
                    "rotate_x": float(direct_transform_settings.get("rotate_x", 0.0)),
                    "rotate_y": float(direct_transform_settings.get("rotate_y", 0.0)),
                    "rotate_z": float(direct_transform_settings.get("rotate_z", 0.0)),
                    "aspect_x": float(direct_transform_settings.get("aspect_x", 1.0)),
                    "aspect_y": float(direct_transform_settings.get("aspect_y", 1.0)),
                }
            )
        legacy_layer = layer_payloads[0]
        legacy_key = legacy_layer.get("key", key_config)
        if not isinstance(legacy_key, dict):
            legacy_key = key_config
        return {
            "enabled": self._effects_enabled and (
                any(bool(layer.get("enabled", False)) for layer in layer_payloads)
                or bool(selected_blur)
                or bool(color_stages)
                or base_transform_active
                or compositor_id is not None
            ),
            "output_connected": True,
            "explicit_compositor_layers": self._effects_enabled and compositor_id is not None,
            "media_path": str(legacy_layer.get("media_path", "")),
            "media_playing": bool(legacy_layer.get("media_playing", False)),
            "media_loop": bool(legacy_layer.get("media_loop", True)),
            "capture_kind": str(legacy_layer.get("capture_kind", "")),
            "capture_device_index": int(legacy_layer.get("capture_device_index", -1)),
            "source_kind": str(legacy_layer.get("source_kind", "media")),
            "matte_rgba": list(legacy_layer.get("matte_rgba", [255, 255, 255, 255])),
            "opacity": float(legacy_layer.get("opacity", 1.0)),
            "blend_mode": str(legacy_layer.get("blend_mode", "normal")),
            "blur_method": str(legacy_layer.get("blur_method", "off")),
            "blur_radius": float(legacy_layer.get("blur_radius", 0.0)),
            "blur_target": str(legacy_layer.get("blur_target", "both")),
            "layer1_opacity": layer1_opacity,
            "layer1_blend_mode": str(compositor_settings.get("layer1_blend_mode", "normal")),
            "input_transform_x": float(base_transform_settings.get("x", 0.0)),
            "input_transform_y": float(base_transform_settings.get("y", 0.0)),
            "input_transform_z": float(base_transform_settings.get("z", 0.0)),
            "input_rotate_x": float(base_transform_settings.get("rotate_x", 0.0)),
            "input_rotate_y": float(base_transform_settings.get("rotate_y", 0.0)),
            "input_rotate_z": float(base_transform_settings.get("rotate_z", 0.0)),
            "input_aspect_x": float(base_transform_settings.get("aspect_x", 1.0)),
            "input_aspect_y": float(base_transform_settings.get("aspect_y", 1.0)),
            "layers": layer_payloads,
            "color_stages": color_stages,
            "key_mode": str(legacy_key.get("mode", "off")),
            "key_color_r": int(legacy_key.get("color", [0, 255, 0])[0]),
            "key_color_g": int(legacy_key.get("color", [0, 255, 0])[1]),
            "key_color_b": int(legacy_key.get("color", [0, 255, 0])[2]),
            "key_similarity": float(legacy_key.get("similarity", 0.25)),
            "key_softness": float(legacy_key.get("softness", 0.10)),
            "key_edge_feather": float(legacy_key.get("edge_feather", 0.0)),
            "spill_suppression": float(legacy_key.get("spill_suppression", 0.25)),
            "luma_low": float(legacy_key.get("luma_low", 0.0)),
            "luma_high": float(legacy_key.get("luma_high", 1.0)),
            "luma_softness": float(legacy_key.get("luma_softness", 0.10)),
            "key_invert": bool(legacy_key.get("invert", False)),
            "effect_color_from_alpha": bool(legacy_layer.get("effect_color_from_alpha", False)),
            "effect_alpha_from_color": bool(legacy_layer.get("effect_alpha_from_color", False)),
            "mask_pattern": str(legacy_layer.get("mask_pattern", "off")),
            "mask_softness": float(legacy_layer.get("mask_softness", 0.0)),
            "mask_aspect": float(legacy_layer.get("mask_aspect", 1.0)),
            "mask_invert": bool(legacy_layer.get("mask_invert", False)),
            "mask_size": float(legacy_layer.get("mask_size", 1.0)),
            "mask_x": float(legacy_layer.get("mask_x", 0.0)),
            "mask_y": float(legacy_layer.get("mask_y", 0.0)),
            "mask_rotation": float(legacy_layer.get("mask_rotation", 0.0)),
            "transform_x": float(legacy_layer.get("transform_x", 0.0)),
            "transform_y": float(legacy_layer.get("transform_y", 0.0)),
            "transform_z": float(legacy_layer.get("transform_z", 0.0)),
            "rotate_x": float(legacy_layer.get("rotate_x", 0.0)),
            "rotate_y": float(legacy_layer.get("rotate_y", 0.0)),
            "rotate_z": float(legacy_layer.get("rotate_z", 0.0)),
            "aspect_x": float(legacy_layer.get("aspect_x", 1.0)),
            "aspect_y": float(legacy_layer.get("aspect_y", 1.0)),
        }

    def set_effects_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._effects_enabled:
            return
        self._effects_enabled = enabled
        self.graphChanged.emit()

    def effects_enabled(self) -> bool:
        return self._effects_enabled

    def serialize(self) -> dict[str, object]:
        return {
            "version": 6,
            "enabled": self._effects_enabled,
            "nodes": deepcopy(list(self._nodes.values())),
            "connections": [list(connection) for connection in self._connections],
        }

    def restore(self, payload: object) -> bool:
        if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list):
            return False
        self._effects_enabled = bool(payload.get("enabled", True))
        valid_types = set(self.NODE_TITLES)
        normalized_nodes = [node for node in payload["nodes"] if isinstance(node, dict) and str(node.get("type")) in valid_types]
        if not any(node.get("type") == "effects_input" for node in normalized_nodes) or not any(
            node.get("type") == "effects_output" for node in normalized_nodes
        ):
            return False
        for dialog in list(self._key_dialogs.values()):
            dialog.close()
            dialog.deleteLater()
        self._key_dialogs.clear()
        self._pending_port = self._drag_port = None
        for widget in self._widgets.values():
            widget.hide()
            widget.deleteLater()
        self._nodes.clear()
        self._widgets.clear()
        self._connections.clear()
        self._selected_nodes.clear()
        self._media_keyframe_playback_arms.clear()
        self._capture_reload_tokens.clear()
        for raw_node in normalized_nodes:
            node_id = str(raw_node.get("id", ""))
            if not node_id or node_id in self._nodes:
                continue
            self._create_node(str(raw_node["type"]), int(raw_node.get("x", 0)), int(raw_node.get("y", 0)), node_id)
            if isinstance(raw_node.get("settings"), dict):
                settings = self._nodes[node_id]["settings"]
                if isinstance(settings, dict):
                    settings.update(raw_node["settings"])
                self._refresh_node_status(node_id)
        raw_connections = payload.get("connections", [])
        if isinstance(raw_connections, list):
            for raw_connection in raw_connections:
                if isinstance(raw_connection, list) and len(raw_connection) == 4:
                    connection_parts = [str(part) for part in raw_connection]
                    if connection_parts[1] == "adjustment_output":
                        connection_parts[1] = "alpha_output"
                    if connection_parts[3] == "adjustment_input":
                        connection_parts[3] = "alpha_input"
                    if (self._nodes.get(connection_parts[0], {}).get("type") == "gradient"
                            and connection_parts[1] == "alpha_output"):
                        connection_parts[1] = "output"
                    connection = tuple(connection_parts)
                    if connection[0] in self._nodes and connection[2] in self._nodes:
                        self._connections.append(connection)
        if int(payload.get("version", 1)) < 3:
            self._migrate_keying_nodes_to_compositors()
        if int(payload.get("version", 1)) < 4:
            self._connections = [
                connection for connection in self._connections
                if not (
                    self._nodes.get(connection[2], {}).get("type") == "media"
                    and connection[3] in {"input", "alpha_input"}
                )
            ]
        if int(payload.get("version", 1)) < 6:
            self._migrate_compositor_chroma_keys()
        self._expand_to_nodes()
        self.update()
        self.graphChanged.emit()
        return True

    def _migrate_compositor_chroma_keys(self) -> None:
        key_setting_names = (
            "key_mode", "key_color_r", "key_color_g", "key_color_b",
            "key_similarity", "key_softness", "key_edge_feather", "spill_suppression",
            "luma_low", "luma_high", "luma_softness", "key_invert",
        )
        compositor_ids = [node_id for node_id, node in self._nodes.items() if node["type"] == "keying"]
        for compositor_id in compositor_ids:
            compositor = self._nodes[compositor_id]
            settings = compositor.get("settings", {})
            if not isinstance(settings, dict):
                continue
            layer_count = max(2, min(64, int(settings.get("layer_count", 2))))
            for layer_index in range(2, layer_count + 1):
                prefix = f"layer{layer_index}_"
                mode = str(settings.get(prefix + "key_mode", settings.get("key_mode", "off") if layer_index == 2 else "off"))
                color_connection = next(
                    (connection for connection in self._connections if connection[2:4] == (compositor_id, f"layer{layer_index}_color")),
                    None,
                )
                if mode != "chroma" or color_connection is None:
                    continue
                chroma_id = self._create_node(
                    "chroma_key",
                    max(20, int(compositor.get("x", 0)) - 280),
                    int(compositor.get("y", 0)) + ((layer_index - 2) * 150),
                )
                chroma_settings = self._nodes[chroma_id]["settings"]
                if isinstance(chroma_settings, dict):
                    for name in (
                        "key_color_r", "key_color_g", "key_color_b", "key_similarity",
                        "key_softness", "key_edge_feather", "spill_suppression", "key_invert",
                    ):
                        default = chroma_settings[name]
                        chroma_settings[name] = settings.get(prefix + name, settings.get(name, default))
                    self._refresh_node_status(chroma_id)
                self._connections = [
                    connection for connection in self._connections
                    if connection[2:4] != (compositor_id, f"layer{layer_index}_alpha")
                ]
                self._connections.extend(
                    [
                        (color_connection[0], color_connection[1], chroma_id, "color_input"),
                        (chroma_id, "alpha_output", compositor_id, f"layer{layer_index}_alpha"),
                    ]
                )
            for name in key_setting_names:
                settings.pop(name, None)
                for layer_index in range(2, 9):
                    settings.pop(f"layer{layer_index}_{name}", None)

    def _migrate_keying_nodes_to_compositors(self) -> None:
        compositor_ids = [node_id for node_id, node in self._nodes.items() if node["type"] == "keying"]
        for compositor_id in compositor_ids:
            settings = self._nodes[compositor_id]["settings"]
            if isinstance(settings, dict):
                settings["layer2_blend_mode"] = str(settings.get("blend_mode", settings.get("layer2_blend_mode", "normal")))
                settings["layer2_opacity"] = float(settings.get("opacity", settings.get("layer2_opacity", 1.0)))
                settings["runtime_supported"] = True
            if any(connection[0] == compositor_id and connection[1] == "output" for connection in self._connections):
                continue
            media_id = next(
                (node_id for node_id, node in self._nodes.items() if node["type"] == "media"),
                None,
            )
            if media_id is None:
                continue
            media_input = next((c for c in self._connections if c[2:4] == (media_id, "input")), None)
            media_output = next((c for c in self._connections if c[0:2] == (media_id, "output")), None)
            if media_input is None or media_output is None:
                continue
            self._connections = [
                connection for connection in self._connections
                if connection not in {media_input, media_output}
                and not (connection[0] == compositor_id or connection[2] == compositor_id)
            ]
            self._connections.extend(
                [
                    (media_input[0], media_input[1], compositor_id, "layer1_color"),
                    (media_id, "output", compositor_id, "layer2_color"),
                    (media_id, "alpha_output", compositor_id, "layer2_alpha"),
                    (compositor_id, "output", media_output[2], media_output[3]),
                ]
            )
            self._refresh_node_status(compositor_id)
            self._separate_compositor_nodes(compositor_id)

    def _separate_compositor_nodes(self, compositor_id: str) -> None:
        compositor_widget = self._widgets[compositor_id]
        for media_id, node in self._nodes.items():
            if node["type"] != "media":
                continue
            media_widget = self._widgets[media_id]
            if media_widget.geometry().intersects(compositor_widget.geometry()):
                media_widget.move(
                    max(24, compositor_widget.x() - media_widget.width() - 48),
                    compositor_widget.y() + compositor_widget.height() + 32,
                )
                node["x"] = round(media_widget.x() / self._zoom)
                node["y"] = round(media_widget.y() / self._zoom)
        output_widget = self._widgets["effects_output"]
        minimum_output_x = compositor_widget.x() + compositor_widget.width() + 48
        if output_widget.x() < minimum_output_x:
            output_widget.move(minimum_output_x, output_widget.y())
            self._nodes["effects_output"]["x"] = round(output_widget.x() / self._zoom)
        self._expand_to_nodes()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(Qt.GlobalColor.darkGray, 1, Qt.DotLine))
        for x in range(0, self.width(), 32):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 32):
            painter.drawLine(0, y, self.width(), y)
        painter.setPen(QPen(Qt.GlobalColor.yellow, 3))
        for source_id, source_port, target_id, target_port in self._connections:
            source = self._widgets.get(source_id)
            target = self._widgets.get(target_id)
            if source is None or target is None or source_port not in source._ports or target_port not in target._ports:
                continue
            start = source.port_center(source_port)
            end = target.port_center(target_port)
            bend = max(60.0, abs(end.x() - start.x()) * 0.5)
            path = QPainterPath(start)
            path.cubicTo(start.x() + bend, start.y(), end.x() - bend, end.y(), end.x(), end.y())
            painter.drawPath(path)
        if self._drag_port is not None and self._drag_position is not None:
            source_id, source_port = self._drag_port
            start = self._widgets[source_id].port_center(source_port)
            end = self._drag_position
            bend = max(60.0, abs(end.x() - start.x()) * 0.5)
            path = QPainterPath(start)
            path.cubicTo(start.x() + bend, start.y(), end.x() - bend, end.y(), end.x(), end.y())
            painter.drawPath(path)
        if self._marquee_rect is not None:
            painter.setPen(QPen(QColor("#efb73e"), 1, Qt.DashLine))
            painter.setBrush(QColor(239, 183, 62, 38))
            painter.drawRect(self._marquee_rect)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus(Qt.MouseFocusReason)
        if event.button() == Qt.LeftButton and bool(event.modifiers() & Qt.ShiftModifier):
            self._marquee_origin = event.position()
            self._marquee_rect = QRectF(self._marquee_origin, self._marquee_origin)
            self._marquee_additive = True
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            self._set_selected_nodes(set())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._marquee_origin is not None and bool(event.buttons() & Qt.LeftButton):
            self._marquee_rect = QRectF(self._marquee_origin, event.position()).normalized()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._marquee_origin is not None:
            selection_rect = self._marquee_rect or QRectF(self._marquee_origin, event.position()).normalized()
            selected = set(self._selected_nodes) if self._marquee_additive else set()
            if selection_rect.width() >= 3.0 or selection_rect.height() >= 3.0:
                selected.update(
                    node_id
                    for node_id, widget in self._widgets.items()
                    if selection_rect.intersects(QRectF(widget.geometry()))
                )
            self._marquee_origin = None
            self._marquee_rect = None
            self._set_selected_nodes(selected)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class EffectsTimelineWidget(QWidget):
    frameRequested = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self._duration = 1
        self._current_frame = 0
        self._expanded = True
        self._scrubbing = False
        self._tracks: list[tuple[str, str, list[int]]] = []
        self.setMinimumHeight(74)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_timeline(self, duration: int, current_frame: int, tracks: list[tuple[str, str, list[int]]]) -> None:
        self._duration = max(1, int(duration))
        self._current_frame = max(0, min(int(current_frame), self._duration - 1))
        self._tracks = tracks
        self.setFixedHeight(48 + (24 * len(tracks) if self._expanded else 24))
        self.update()

    def _timeline_left(self) -> int:
        return 180

    def _frame_x(self, frame: int) -> float:
        usable = max(1.0, float(self.width() - self._timeline_left() - 14))
        return float(self._timeline_left()) + (float(frame) / float(max(1, self._duration - 1))) * usable

    def _frame_at_x(self, x: float) -> int:
        usable = max(1.0, float(self.width() - self._timeline_left() - 14))
        ratio = max(0.0, min(1.0, (float(x) - self._timeline_left()) / usable))
        return round(ratio * float(max(1, self._duration - 1)))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        if event.position().x() < 28 and event.position().y() < 28:
            self._expanded = not self._expanded
            self.set_timeline(self._duration, self._current_frame, self._tracks)
        elif event.position().x() >= self._timeline_left():
            self._scrubbing = True
            self.frameRequested.emit(self._frame_at_x(event.position().x()))
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._scrubbing and bool(event.buttons() & Qt.LeftButton):
            self.frameRequested.emit(self._frame_at_x(event.position().x()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._scrubbing:
            self._scrubbing = False
            self.frameRequested.emit(self._frame_at_x(event.position().x()))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#20242a"))
        painter.setPen(QPen(QColor("#555d66"), 1))
        painter.drawLine(self._timeline_left(), 0, self._timeline_left(), self.height())
        visible_tracks = self._tracks if self._expanded else []
        master_frames = sorted({frame for _node_id, _title, frames in self._tracks for frame in frames})
        rows = [("Master Effect", master_frames)] + [(title, frames) for _node_id, title, frames in visible_tracks]
        for row_index, (title, frames) in enumerate(rows):
            center_y = 18 + (row_index * 24)
            painter.setPen(QColor("#d9dde2"))
            prefix = "v " if row_index == 0 and self._expanded else ("> " if row_index == 0 else "  ")
            painter.drawText(8, center_y + 4, prefix + title)
            painter.setPen(QPen(QColor("#6b737c"), 1))
            painter.drawLine(self._timeline_left(), center_y, self.width() - 14, center_y)
            painter.setPen(QPen(QColor("#efb73e"), 3))
            for frame in frames:
                x = round(self._frame_x(frame))
                painter.drawLine(x, center_y - 7, x, center_y + 7)
        playhead_x = round(self._frame_x(self._current_frame))
        painter.setPen(QPen(QColor("#f45b5b"), 2))
        painter.drawLine(playhead_x, 0, playhead_x, self.height())


class EffectsGraphEditor(QGroupBox):
    graphChanged = Signal()
    captureStartRequested = Signal(object)
    captureSettingsRequested = Signal(object)

    NODE_MENU = {
        "Noise Reduction": "denoise",
        "Video Clip / Image / Image Sequence": "media",
        "Source": "capture",
        "Composition Source": "composition",
        "Compositor": "keying",
        "Blur": "blur",
        "Matte Color": "matte",
        "Gradient Texture": "gradient",
        "Mask": "mask",
        "Mix": "mix",
        "Basic Proc Amp": "proc_amp",
        "Color Corrector": "color_corrector",
        "Chroma Key": "chroma_key",
        "Luma Key": "luma_key",
        "3D Transform": "transform_3d",
        "Color Splitter": "color_splitter",
        "Color Recombiner": "color_recombiner",
    }

    def __init__(self) -> None:
        super().__init__("Effects")
        self._node_keyframes: dict[str, dict[int, dict[str, object]]] = {}
        self._palette_entries: list[dict[str, object]] = []
        self._current_palette_index: int | None = None
        self._clean_graph_state: dict[str, object] | None = None
        self._current_frame = 0
        self._duration = 1
        self._playing = False
        self._frame_rate = 30.0
        self._playback_started_at = 0.0
        self._playback_start_frame = 0
        self._external_timecode_text = ""
        self._external_timecode_frame: int | None = None
        self._external_origin_frame: int | None = None
        self._external_timeline_origin = 0
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(16)
        self._play_timer.setTimerType(Qt.PreciseTimer)
        self._play_timer.timeout.connect(self._advance_playback)

        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()
        self.bypass_toggle = QCheckBox("Bypass effects")
        self.bypass_toggle.setChecked(False)
        self.bypass_toggle.setToolTip("Pass video through without processing the effects graph")
        top_bar.addWidget(self.bypass_toggle)
        top_bar.addStretch(1)
        self.previous_key_button = QPushButton("Previous Key")
        self.keyframe_button = QPushButton("Keyframe")
        self.delete_keyframe_button = QPushButton("Delete Key")
        self.next_key_button = QPushButton("Next Key")
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.previous_key_button.clicked.connect(lambda: self._jump_keyframe(-1))
        self.keyframe_button.clicked.connect(self._store_selected_keyframes)
        self.delete_keyframe_button.clicked.connect(self._delete_selected_keyframes)
        self.next_key_button.clicked.connect(lambda: self._jump_keyframe(1))
        self.play_button.toggled.connect(self._set_playing)
        self.keyframe_button.setToolTip("Store selected nodes; replaces only an existing key at this frame")
        self.delete_keyframe_button.setToolTip("Delete selected nodes' keyframes at the current frame")
        for button in (
            self.previous_key_button,
            self.keyframe_button,
            self.delete_keyframe_button,
            self.next_key_button,
            self.play_button,
        ):
            top_bar.addWidget(button)
        self.clock_source_combo = QComboBox()
        self.clock_source_combo.addItems(["Internal", "External Timecode"])
        self.clock_source_combo.setToolTip("Run from the timeline clock or follow incoming source timecode")
        self.clock_source_combo.currentTextChanged.connect(self._on_clock_source_changed)
        top_bar.addWidget(self.clock_source_combo)
        self.timeline_fps_spin = QDoubleSpinBox()
        self.timeline_fps_spin.setRange(1.0, 120.0)
        self.timeline_fps_spin.setDecimals(3)
        self.timeline_fps_spin.setSingleStep(1.0)
        self.timeline_fps_spin.setValue(self._frame_rate)
        self.timeline_fps_spin.setSuffix(" fps")
        self.timeline_fps_spin.setToolTip("Timeline frame rate")
        self.timeline_fps_spin.valueChanged.connect(self.set_frame_rate)
        top_bar.addWidget(self.timeline_fps_spin)
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 999999)
        self.frame_spin.setPrefix("Frame ")
        self.frame_spin.valueChanged.connect(self._seek_frame_expanding)
        top_bar.addWidget(self.frame_spin)
        self.timecode_display = QLabel("00:00:00:00")
        self.timecode_display.setMinimumWidth(92)
        self.timecode_display.setAlignment(Qt.AlignCenter)
        self.timecode_display.setStyleSheet("font-family: Consolas; font-weight: 600; color: #efb73e;")
        top_bar.addWidget(self.timecode_display)
        layout.addLayout(top_bar)

        self.tabs = QTabWidget()
        node_view = QWidget()
        node_layout = QVBoxLayout(node_view)
        toolbar = QHBoxLayout()
        self.node_combo = QComboBox()
        self.node_combo.addItem("Add effect node...")
        self.node_combo.addItems(list(self.NODE_MENU))
        self.node_combo.currentIndexChanged.connect(self._add_selected_node)
        toolbar.addWidget(self.node_combo, 1)
        self.new_graph_button = QPushButton("New Graph")
        self.new_graph_button.setToolTip("Clear the current graph and start a new one")
        self.new_graph_button.clicked.connect(self._new_graph)
        toolbar.addWidget(self.new_graph_button)
        self.current_effect_label = QLabel("Current effect: Unsaved")
        self.current_effect_label.setMinimumWidth(220)
        toolbar.addWidget(self.current_effect_label)
        self.update_effect_button = QPushButton("Update Current")
        self.update_effect_button.setToolTip("Overwrite the currently loaded palette effect")
        self.update_effect_button.clicked.connect(self._update_current_palette_item)
        toolbar.addWidget(self.update_effect_button)
        self.save_effect_button = QPushButton("Save New")
        self.save_effect_button.setToolTip("Save the current graph as a new palette effect")
        self.save_effect_button.clicked.connect(self._save_effect_to_palette)
        toolbar.addWidget(self.save_effect_button)
        node_layout.addLayout(toolbar)

        self.canvas = EffectsGraphCanvas()
        self.canvas._editor = self
        self.canvas._history = [self.canvas._history_snapshot()]
        for label, callback in (("Undo", self.canvas.undo), ("Redo", self.canvas.redo), ("Copy", self.canvas.copy_nodes), ("Paste", self.canvas.paste_nodes)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            toolbar.addWidget(button)
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setMinimumHeight(310)
        scroll.setWidget(self.canvas)
        node_layout.addWidget(scroll)
        self.tabs.addTab(node_view, "Node View")

        palette_page = QWidget()
        palette_layout = QVBoxLayout(palette_page)
        self.palette_list = QListWidget()
        self.palette_list.setIconSize(QPixmap(160, 90).size())
        self.palette_list.currentItemChanged.connect(self._on_palette_selection_changed)
        self.palette_list.itemDoubleClicked.connect(self._recall_palette_item)
        palette_layout.addWidget(self.palette_list)
        palette_actions = QHBoxLayout()
        self.rename_effect_button = QPushButton("Rename Effect")
        self.rename_effect_button.clicked.connect(self._rename_palette_item)
        palette_actions.addWidget(self.rename_effect_button)
        self.delete_effect_button = QPushButton("Delete Effect")
        self.delete_effect_button.clicked.connect(self._delete_palette_item)
        palette_actions.addWidget(self.delete_effect_button)
        palette_layout.addLayout(palette_actions)
        self.tabs.addTab(palette_page, "Effects Palette")
        layout.addWidget(self.tabs)

        self.timeline = EffectsTimelineWidget()
        self.timeline.frameRequested.connect(self.set_current_frame)
        layout.addWidget(self.timeline)
        self.canvas.graphChanged.connect(self._on_canvas_changed)
        self.canvas.captureSettingsRequested.connect(self.captureSettingsRequested.emit)
        self.bypass_toggle.toggled.connect(self._set_effects_bypassed)
        self._mark_graph_clean()
        self._refresh_palette_controls()
        self._refresh_timeline()

    def _on_canvas_changed(self) -> None:
        valid_ids = set(self.canvas._nodes)
        self._node_keyframes = {
            node_id: keyframes for node_id, keyframes in self._node_keyframes.items() if node_id in valid_ids
        }
        self._refresh_timeline()
        self._refresh_palette_controls()
        self.graphChanged.emit()

    def _current_graph_state(self) -> dict[str, object]:
        return {
            "graph": self.canvas.serialize(),
            "keyframes": self._serialize_keyframes(),
        }

    def _mark_graph_clean(self) -> None:
        self._clean_graph_state = self._current_graph_state()

    def _has_unsaved_graph_changes(self) -> bool:
        return self._clean_graph_state != self._current_graph_state()

    def _refresh_timeline(self) -> None:
        tracks = []
        for node_id, node in self.canvas._nodes.items():
            title = self.canvas.NODE_TITLES.get(str(node.get("type", "")), str(node_id))
            tracks.append((node_id, title, sorted(self._node_keyframes.get(node_id, {}))))
        self.timeline.set_timeline(self._duration, self._current_frame, tracks)
        if self.clock_source_combo.currentText() == "External Timecode" and self._external_timecode_text:
            self.timecode_display.setText(self._external_timecode_text)
        else:
            nominal_fps = max(1, int(round(self._frame_rate)))
            total_seconds = self._current_frame // nominal_fps
            frames = self._current_frame % nominal_fps
            self.timecode_display.setText(
                f"{total_seconds // 3600:02d}:{(total_seconds // 60) % 60:02d}:{total_seconds % 60:02d}:{frames:02d}"
            )
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(self._current_frame)
        self.frame_spin.blockSignals(False)

    def _seek_frame_expanding(self, frame: int) -> None:
        self._duration = max(self._duration, int(frame) + 1)
        self.set_current_frame(frame)

    def _store_selected_keyframes(self) -> None:
        selected_ids = self.canvas.selected_node_ids()
        if not selected_ids:
            return
        for node_id in selected_ids:
            settings = self.canvas._nodes.get(node_id, {}).get("settings", {})
            if isinstance(settings, dict):
                snapshot = dict(settings)
                if self.canvas._nodes[node_id].get("type") == "media":
                    playback_arm = self.canvas._media_keyframe_playback_arms.get(node_id)
                    snapshot.pop("playing", None)
                    if playback_arm is not None:
                        snapshot["playing"] = playback_arm == "play"
                self._node_keyframes.setdefault(node_id, {})[self._current_frame] = snapshot
        self._duration = max(self._duration, self._current_frame + 1)
        self.canvas._record_history()
        self._refresh_timeline()
        self.graphChanged.emit()

    def _delete_selected_keyframes(self) -> None:
        deleted = False
        for node_id in self.canvas.selected_node_ids():
            keyframes = self._node_keyframes.get(node_id)
            if keyframes is None or self._current_frame not in keyframes:
                continue
            keyframes.pop(self._current_frame)
            if not keyframes:
                self._node_keyframes.pop(node_id, None)
            deleted = True
        if not deleted:
            return
        self.set_current_frame(self._current_frame)
        self.canvas._record_history()
        self.graphChanged.emit()

    def _sync_media_keyframe_arms(self) -> None:
        for node_id, node in self.canvas._nodes.items():
            if node.get("type") != "media" or node_id not in self.canvas._widgets:
                continue
            playing_keys = [
                (frame, settings["playing"])
                for frame, settings in self._node_keyframes.get(node_id, {}).items()
                if "playing" in settings
            ]
            if playing_keys:
                previous = [item for item in playing_keys if item[0] <= self._current_frame]
                if previous:
                    _frame, playing = max(previous, key=lambda item: item[0])
                else:
                    _frame, playing = min(playing_keys, key=lambda item: item[0])
                state = "play" if bool(playing) else "pause"
                self.canvas._media_keyframe_playback_arms[node_id] = state
            else:
                state = None
                self.canvas._media_keyframe_playback_arms.pop(node_id, None)
            self.canvas._widgets[node_id].set_media_keyframe_playback_arm(state)

    @staticmethod
    def _interpolate_settings(
        left: dict[str, object], right: dict[str, object], progress: float
    ) -> dict[str, object]:
        eased = progress * progress * (3.0 - (2.0 * progress))
        result: dict[str, object] = {}
        for name in left.keys() | right.keys():
            if name not in left:
                if progress >= 1.0:
                    result[name] = right[name]
                continue
            left_value = left[name]
            if name not in right:
                result[name] = left_value
                continue
            right_value = right[name]
            if (
                isinstance(left_value, (int, float)) and not isinstance(left_value, bool)
                and isinstance(right_value, (int, float)) and not isinstance(right_value, bool)
            ):
                value = float(left_value) + ((float(right_value) - float(left_value)) * eased)
                result[name] = round(value) if isinstance(left_value, int) and isinstance(right_value, int) else value
            else:
                result[name] = left_value if progress < 1.0 else right_value
        return result

    @classmethod
    def _settings_at_frame(
        cls,
        keyframes: dict[int, dict[str, object]],
        frame: int,
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        setting_names = {
            name
            for settings in keyframes.values()
            for name in settings
        }
        for name in setting_names:
            value_keys = sorted(
                (key_frame, settings[name])
                for key_frame, settings in keyframes.items()
                if name in settings
            )
            previous = [item for item in value_keys if item[0] <= frame]
            following = [item for item in value_keys if item[0] >= frame]
            left_frame, left_value = previous[-1] if previous else value_keys[0]
            right_frame, right_value = following[0] if following else value_keys[-1]
            if left_frame == right_frame:
                result[name] = left_value
                continue
            progress = (frame - left_frame) / float(right_frame - left_frame)
            result.update(cls._interpolate_settings({name: left_value}, {name: right_value}, progress))
        return result

    @staticmethod
    def _composition_has_animation(nodes, depth: int = 0) -> bool:
        if depth > 32:
            return False
        for node in nodes:
            if node.get('type') != 'composition':
                continue
            settings = node.get('settings', {})
            graph = settings.get('graph', {})
            if not graph.get('enabled', True):
                continue
            if any(len(frames) > 1 for frames in settings.get('keyframes', {}).values()):
                return True
            if EffectsGraphEditor._composition_has_animation(graph.get('nodes', []), depth + 1):
                return True
        return False

    def set_current_frame(self, frame: int) -> None:
        previous_frame = self._current_frame
        self._current_frame = max(0, min(int(frame), self._duration - 1))
        changed = self._current_frame != previous_frame and self._composition_has_animation(self.canvas._nodes.values())
        for node_id, keyframes in self._node_keyframes.items():
            if node_id not in self.canvas._nodes or not keyframes:
                continue
            interpolated = self._settings_at_frame(keyframes, self._current_frame)
            settings = self.canvas._nodes[node_id].get("settings")
            if isinstance(settings, dict) and any(settings.get(name) != value for name, value in interpolated.items()):
                settings.update(interpolated)
                self.canvas._refresh_node_status(node_id)
                if self.canvas._nodes[node_id].get("type") == "media":
                    self.canvas._widgets[node_id].set_media_state(
                        bool(settings.get("playing", False)),
                        bool(settings.get("loop", True)),
                        bool(str(settings.get("path", "")).strip()),
                    )
                changed = True
        self._sync_media_keyframe_arms()
        self._refresh_timeline()
        if changed:
            self.graphChanged.emit()

    def _set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        external = self.clock_source_combo.currentText() == "External Timecode"
        self.play_button.setText("Syncing" if self._playing and external else ("Pause" if self._playing else "Play"))
        if self._playing:
            if external:
                self._external_origin_frame = self._external_timecode_frame
                self._external_timeline_origin = self._current_frame
                self._play_timer.stop()
            else:
                self._playback_started_at = time.perf_counter()
                self._playback_start_frame = self._current_frame
                self._play_timer.start()
        else:
            self._play_timer.stop()

    def _advance_playback(self) -> None:
        elapsed_frames = int((time.perf_counter() - self._playback_started_at) * self._frame_rate)
        target_frame = self._playback_start_frame + elapsed_frames
        if target_frame >= self._duration:
            self.set_current_frame(self._duration - 1)
            self.play_button.setChecked(False)
            return
        if target_frame != self._current_frame:
            self.set_current_frame(target_frame)

    def set_frame_rate(self, frame_rate: float) -> None:
        self._frame_rate = max(1.0, min(120.0, float(frame_rate)))
        if abs(self.timeline_fps_spin.value() - self._frame_rate) > 0.0005:
            self.timeline_fps_spin.blockSignals(True)
            self.timeline_fps_spin.setValue(self._frame_rate)
            self.timeline_fps_spin.blockSignals(False)
        interval_ms = max(1, int(round(500.0 / self._frame_rate)))
        self._play_timer.setInterval(interval_ms)
        if self._playing and self.clock_source_combo.currentText() == "Internal":
            self._playback_started_at = time.perf_counter()
            self._playback_start_frame = self._current_frame
        self._refresh_timeline()

    def _on_clock_source_changed(self, source: str) -> None:
        self._external_origin_frame = None
        self._external_timeline_origin = self._current_frame
        if self._playing:
            self._set_playing(True)
        else:
            self._refresh_timeline()

    def _jump_keyframe(self, direction: int) -> None:
        frames = sorted({frame for keyframes in self._node_keyframes.values() for frame in keyframes})
        candidates = [frame for frame in frames if frame > self._current_frame] if direction > 0 else [frame for frame in frames if frame < self._current_frame]
        if candidates:
            self.set_current_frame(candidates[0] if direction > 0 else candidates[-1])

    def _palette_index_from_item(self, item: QListWidgetItem | None) -> int | None:
        if item is None:
            return None
        try:
            index = int(item.data(Qt.UserRole))
        except (TypeError, ValueError):
            return None
        return index if 0 <= index < len(self._palette_entries) else None

    def _selected_palette_index(self) -> int | None:
        return self._palette_index_from_item(self.palette_list.currentItem())

    def _on_palette_selection_changed(self, _current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        self._refresh_palette_controls()

    def _refresh_palette_controls(self) -> None:
        selected_index = self._selected_palette_index()
        current_index = self._current_palette_index if self._current_palette_index is not None else None
        if current_index is not None and not 0 <= current_index < len(self._palette_entries):
            current_index = None
            self._current_palette_index = None
        self.rename_effect_button.setEnabled(selected_index is not None)
        self.delete_effect_button.setEnabled(selected_index is not None)
        self.update_effect_button.setEnabled(current_index is not None)
        if current_index is None:
            label = "Current effect: Unsaved"
        else:
            current_name = str(self._palette_entries[current_index].get("name", f"Effect {current_index + 1}"))
            label = f"Current effect: {current_name}"
        self.current_effect_label.setText(label + (" *" if self._has_unsaved_graph_changes() else ""))

    def _build_palette_entry(self, name: str) -> dict[str, object]:
        entry = {
            "name": name.strip(),
            "graph": self.canvas.serialize(),
            "keyframes": self._serialize_keyframes(),
            "duration": self._duration,
        }
        thumbnail = self.canvas.grab().scaled(160, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        thumbnail_bytes = QByteArray()
        thumbnail_buffer = QBuffer(thumbnail_bytes)
        if thumbnail_buffer.open(QIODevice.WriteOnly) and thumbnail.save(thumbnail_buffer, "PNG"):
            entry["thumbnail"] = bytes(thumbnail_bytes.toBase64()).decode("ascii")
        return entry

    def _palette_item(self, index: int, entry: dict[str, object]) -> QListWidgetItem:
        icon = QIcon()
        thumbnail_text = entry.get("thumbnail", "")
        if isinstance(thumbnail_text, str) and thumbnail_text:
            thumbnail = QPixmap()
            thumbnail.loadFromData(QByteArray.fromBase64(thumbnail_text.encode("ascii")), "PNG")
            icon = QIcon(thumbnail)
        item = QListWidgetItem(icon, str(entry.get("name", f"Effect {index + 1}")))
        item.setData(Qt.UserRole, index)
        return item

    def _rebuild_palette_list(self, selected_index: int | None = None) -> None:
        self.palette_list.clear()
        for index, entry in enumerate(self._palette_entries):
            self.palette_list.addItem(self._palette_item(index, entry))
        if selected_index is not None and 0 <= selected_index < self.palette_list.count():
            self.palette_list.setCurrentRow(selected_index)
        self._refresh_palette_controls()

    def _save_effect_to_palette(self) -> bool:
        default_name = ""
        if self._current_palette_index is not None and 0 <= self._current_palette_index < len(self._palette_entries):
            default_name = str(self._palette_entries[self._current_palette_index].get("name", ""))
        name, accepted = QInputDialog.getText(self, "Save New Effect", "Effect name", text=default_name)
        if not accepted or not name.strip():
            return False
        entry = self._build_palette_entry(name)
        self._palette_entries.append(entry)
        self._current_palette_index = len(self._palette_entries) - 1
        self._mark_graph_clean()
        self._rebuild_palette_list(self._current_palette_index)
        self.graphChanged.emit()
        return True

    def _update_current_palette_item(self) -> bool:
        if self._current_palette_index is None or not 0 <= self._current_palette_index < len(self._palette_entries):
            return False
        current_name = str(self._palette_entries[self._current_palette_index].get("name", "Effect"))
        self._palette_entries[self._current_palette_index] = self._build_palette_entry(current_name)
        self._mark_graph_clean()
        self._rebuild_palette_list(self._current_palette_index)
        self.graphChanged.emit()
        return True

    def _new_graph(self) -> None:
        if self._has_unsaved_graph_changes():
            response = QMessageBox.warning(
                self,
                "Unsaved Graph Changes",
                "Save changes to the current graph before starting a new graph?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if response == QMessageBox.Cancel:
                return
            if response == QMessageBox.Save:
                saved = (
                    self._update_current_palette_item()
                    if self._current_palette_index is not None
                    else self._save_effect_to_palette()
                )
                if not saved:
                    return
        empty_graph = {
            "version": 6,
            "enabled": not self.bypass_toggle.isChecked(),
            "nodes": [
                {"id": "effects_input", "type": "effects_input", "x": 40, "y": 120, "settings": {}},
                {"id": "effects_output", "type": "effects_output", "x": 620, "y": 120, "settings": {}},
            ],
            "connections": [["effects_input", "output", "effects_output", "input"]],
        }
        if not self.canvas.restore(empty_graph):
            return
        self.canvas._next_node_number = 1
        self._restore_keyframes(None)
        self._current_palette_index = None
        self._mark_graph_clean()
        self._rebuild_palette_list()

    def _rename_palette_item(self) -> None:
        item = self.palette_list.currentItem()
        index = self._palette_index_from_item(item)
        if index is None:
            return
        name, accepted = QInputDialog.getText(self, "Rename Effect", "Effect name", text=item.text())
        if accepted and name.strip():
            self._palette_entries[index]["name"] = name.strip()
            item.setText(name.strip())
            self._refresh_palette_controls()
            self.graphChanged.emit()

    def _delete_palette_item(self) -> None:
        index = self._selected_palette_index()
        if index is None:
            return
        entry_name = str(self._palette_entries[index].get("name", f"Effect {index + 1}"))
        response = QMessageBox.question(
            self,
            "Delete Effect",
            f'Delete "{entry_name}" from the effects palette?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if response != QMessageBox.Yes:
            return
        self._palette_entries.pop(index)
        if self._current_palette_index == index:
            self._current_palette_index = None
        elif self._current_palette_index is not None and index < self._current_palette_index:
            self._current_palette_index -= 1
        next_index = min(index, len(self._palette_entries) - 1) if self._palette_entries else None
        self._rebuild_palette_list(next_index)
        self.graphChanged.emit()

    def _recall_palette_item(self, item: QListWidgetItem, _column: int = 0) -> None:
        index = self._palette_index_from_item(item)
        if index is None:
            return
        entry = self._palette_entries[index]
        graph = deepcopy(entry.get("graph"))
        if isinstance(graph, dict):
            # Bypass is a live operator control, not a recalled preset setting.
            # Set it before restore emits graphChanged so no enabled payload
            # reaches the worker even briefly while loading a bypassed effect.
            graph["enabled"] = not self.bypass_toggle.isChecked()
        if self.canvas.restore(graph):
            self._restore_keyframes(entry.get("keyframes"))
            self._current_palette_index = index
            self._rebuild_palette_list(index)
            self.set_current_frame(0)
            self._mark_graph_clean()
            self._refresh_palette_controls()
            capture_device = self.active_capture_device()
            if capture_device is not None:
                self.captureStartRequested.emit(capture_device)

    def _serialize_keyframes(self) -> dict[str, dict[str, dict[str, object]]]:
        return {
            node_id: {str(frame): dict(settings) for frame, settings in keyframes.items()}
            for node_id, keyframes in self._node_keyframes.items()
        }

    def _restore_keyframes(self, payload: object) -> None:
        restored: dict[str, dict[int, dict[str, object]]] = {}
        if isinstance(payload, dict):
            for node_id, raw_keyframes in payload.items():
                if node_id not in self.canvas._nodes or not isinstance(raw_keyframes, dict):
                    continue
                restored[str(node_id)] = {
                    max(0, int(frame)): dict(settings)
                    for frame, settings in raw_keyframes.items()
                    if isinstance(settings, dict)
                }
        self._node_keyframes = restored
        self._duration = max(
            [1] + [frame + 1 for keyframes in restored.values() for frame in keyframes]
        )
        self._current_frame = min(self._current_frame, self._duration - 1)
        self._sync_media_keyframe_arms()
        self._refresh_timeline()

    def _set_effects_bypassed(self, bypassed: bool) -> None:
        self.canvas.set_effects_enabled(not bypassed)

    def _add_selected_node(self, _index: int = 0) -> None:
        node_type = self.NODE_MENU.get(self.node_combo.currentText())
        if node_type is None:
            return
        if node_type == 'composition':
            entries = self._palette_entries
            if not entries:
                QMessageBox.information(self, 'Composition Source', 'Save a composition to the palette first.')
            else:
                labels = [f"{i + 1}: {e.get('name', 'Composition')}" for i,e in enumerate(entries)]
                label, accepted = QInputDialog.getItem(self, 'Composition Source', 'Saved composition', labels, 0, False)
                if accepted:
                    entry = entries[labels.index(label)]
                    x, y = self.canvas._next_free_node_position('composition')
                    node_id = self.canvas._create_node('composition', x, y)
                    self.canvas._nodes[node_id]['settings'] = {'name': entry.get('name', ''), 'graph': deepcopy(entry['graph']), 'keyframes': deepcopy(entry.get('keyframes', {}))}
                    self._duration = max(self._duration, int(entry.get('duration', 1)), 1 + max((int(frame) for keys in entry.get('keyframes', {}).values() for frame in keys), default=0))
                    self._refresh_timeline()
                    self.canvas._refresh_node_status(node_id)
                    self.canvas.graphChanged.emit()
        else:
            self.canvas.add_node(node_type)
        self.node_combo.setCurrentIndex(0)

    def serialize(self) -> dict[str, object]:
        return {
            "editor_version": 2,
            "graph": self.canvas.serialize(),
            "keyframes": self._serialize_keyframes(),
            "duration": self._duration,
            "frame_rate": self._frame_rate,
            "clock_source": self.clock_source_combo.currentText(),
            "palette": self._palette_entries,
        }

    def restore(self, payload: object) -> bool:
        editor_payload = payload if isinstance(payload, dict) and "graph" in payload else None
        graph_payload = editor_payload.get("graph") if isinstance(editor_payload, dict) else payload
        restored = self.canvas.restore(graph_payload)
        if restored:
            if isinstance(editor_payload, dict):
                self._restore_keyframes(editor_payload.get("keyframes"))
                self._duration = max(self._duration, int(editor_payload.get("duration", 1)))
                self.timeline_fps_spin.setValue(float(editor_payload.get("frame_rate", self._frame_rate)))
                clock_source = str(editor_payload.get("clock_source", "Internal"))
                self.clock_source_combo.setCurrentText(
                    clock_source if clock_source in {"Internal", "External Timecode"} else "Internal"
                )
                raw_palette = editor_payload.get("palette", [])
                self._palette_entries = [
                    dict(entry) for entry in raw_palette
                    if isinstance(entry, dict) and isinstance(entry.get("graph"), dict)
                ] if isinstance(raw_palette, list) else []
                self._current_palette_index = None
            else:
                self._restore_keyframes(None)
                self._palette_entries = []
                self._current_palette_index = None
            self._rebuild_palette_list()
            self.bypass_toggle.blockSignals(True)
            self.bypass_toggle.setChecked(not self.canvas.effects_enabled())
            self.bypass_toggle.blockSignals(False)
            self._refresh_timeline()
            self._mark_graph_clean()
        return restored

    def set_composition_timecode(
        self,
        display_text: str,
        info: dict[str, object] | None = None,
    ) -> None:
        self._external_timecode_text = display_text.strip()
        external_frame: int | None = None
        if isinstance(info, dict) and bool(info.get("present", False)):
            timecode_text = str(info.get("text", "")).strip()
            count_fps = _timecode_count_fps(str(info.get("format_name", "")), self._frame_rate)
            base_frame = _timecode_to_frame_number(
                _timecode_with_drop_frame_separator(timecode_text, bool(info.get("drop_frame", False))),
                count_fps,
            )
            if base_frame is not None:
                rate_ratio = self._frame_rate / float(max(1, count_fps))
                external_frame = round(float(base_frame) * rate_ratio)
                if rate_ratio > 1.5 and bool(info.get("field_mark", False)):
                    external_frame += 1
        self._external_timecode_frame = external_frame
        if self.clock_source_combo.currentText() != "External Timecode":
            return
        if self._playing and external_frame is not None:
            if self._external_origin_frame is None:
                self._external_origin_frame = external_frame
                self._external_timeline_origin = self._current_frame
            target = self._external_timeline_origin + (external_frame - self._external_origin_frame)
            self.set_current_frame(target)
        else:
            self._refresh_timeline()

    def migrate_legacy_denoise(self, method: str, strength: float) -> None:
        self.canvas.migrate_legacy_denoise(method, strength)

    def active_denoise_settings(self) -> tuple[str, float]:
        return self.canvas.active_denoise_settings()

    def active_capture_device(self) -> int | None:
        return self.canvas.active_capture_device()

    def set_capture_devices(self, devices: list[tuple[str, object]]) -> None:
        self.canvas.set_capture_devices(devices)

    def native_effects_payload(self) -> dict[str, object]:
        self.canvas._composition_frame = self._current_frame
        self.canvas._composition_settings_at_frame = self._settings_at_frame
        return self.canvas.native_effects_payload()


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
        self.basic_scaling_max_inflight = 1
        self.color_space = _normalize_color_space_name(os.environ.get("VP_COLOR_SPACE", "rec709"))
        self.color_range = _normalize_color_range_name(os.environ.get("VP_COLOR_RANGE", "limited"))
        self.ai_sr_enabled = False
        self.ai_sr_active = False
        self.ai_sr_loading = False
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
        self.decklink_output_buffer_frames = 0
        self.worker_process_priority = _normalize_worker_priority_name(
            os.environ.get("VP_WORKER_PROCESS_PRIORITY", "above_normal")
        )
        self.rtx_vsr_error: str | None = None
        self.rtx_vsr_info: dict[str, object] | None = None
        self.processor = None
        self.effects_payload: dict[str, object] = {"enabled": False}
        self._effect_media_decoders: dict[tuple[int, int], EffectMediaDecoder | EffectCaptureDecoder] = {}
        self._loaded_effect_source_signature: tuple[object, ...] | None = None
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
        self._loaded_effect_source_signature = None
        self.set_effects_config(self.effects_payload)

    def set_roi(self, roi: Roi) -> bool:
        if self.processor is not None:
            self.processor.set_roi(roi.x, roi.y, roi.w, roi.h)
        return True

    def set_roi_settled(self, roi: Roi) -> bool:
        self.set_roi_subpixel_shift(0.0, 0.0)
        return self.set_roi(roi)

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

    def activate_source(self, logical_id, config):
        if not hasattr(self, "_source_pool"):
            self._source_pool = SourcePool()
        self._source_pool.activate(logical_id, config, lambda: create_input_adapter(config, d, EffectCaptureDecoder, EffectMediaDecoder))

    def deactivate_source(self, logical_id):
        if hasattr(self, "_source_pool"):
            self._source_pool.deactivate(logical_id)

    def show_capture_settings(self, device_index: int) -> None:
        decoder = next(
            (
                decoder for decoder in self._effect_media_decoders.values()
                if isinstance(decoder, EffectCaptureDecoder) and decoder.device_index == int(device_index)
            ),
            None,
        )
        temporary_decoder = None
        try:
            if decoder is None:
                temporary_decoder = EffectCaptureDecoder(int(device_index))
                decoder = temporary_decoder
            decoder.show_settings()
        finally:
            if temporary_decoder is not None:
                temporary_decoder.close()

    def set_effects_config(self, payload: dict[str, object]) -> None:
        self.effects_payload = dict(payload)
        if self.processor is None:
            return
        source_signature = _effects_source_signature(payload)
        reload_source = source_signature != self._loaded_effect_source_signature
        if reload_source:
            for decoder in self._effect_media_decoders.values():
                decoder.close()
            self._effect_media_decoders.clear()
        if reload_source:
            self.processor.clear_effect_media()
        enabled = bool(payload.get("enabled", False))
        layers = _effect_layers_from_payload(payload)
        try:
            if enabled:
                for layer in layers:
                    if not bool(layer.get("enabled", False)):
                        continue
                    layer_index = int(layer.get("layer_index", 2))
                    source_kind = str(layer.get("source_kind", "media")).strip().lower()
                    capture_kind = str(layer.get("capture_kind", "")).strip().lower()
                    media_path = str(layer.get("media_path", "")).strip()
                    if source_kind == "matte":
                        matte_rgba = np.asarray(
                            layer.get("matte_rgba", [255, 255, 255, 255]), dtype=np.uint8
                        ).reshape(1, 1, 4)
                        self.processor.upload_effect_layer_media_rgba(layer_index, matte_rgba, 1, 1)
                    elif reload_source and capture_kind == "logical":
                        if not hasattr(self, "_source_pool"):
                            self._source_pool = SourcePool()
                        self._effect_media_decoders[(layer_index, 0)] = self._source_pool.reader(layer["capture_device_index"])
                    elif reload_source and capture_kind == "webcam":
                        decoder = EffectCaptureDecoder(
                            int(layer.get("capture_device_index", 0)),
                            int(layer.get("capture_width", 0)),
                            int(layer.get("capture_height", 0)),
                        )
                        self._effect_media_decoders[(layer_index, 0)] = decoder
                        rgba = decoder.next_rgba()
                        self.processor.upload_effect_layer_media_rgba(
                            layer_index, rgba, int(rgba.shape[1]), int(rgba.shape[0])
                        )
                    elif reload_source and media_path:
                        decoder = EffectMediaDecoder(media_path)
                        self._effect_media_decoders[(layer_index, 0)] = decoder
                        rgba = decoder.next_rgba(loop=bool(layer.get("media_loop", True)))
                        self.processor.upload_effect_layer_media_rgba(
                            layer_index, rgba, int(rgba.shape[1]), int(rgba.shape[0])
                        )
                    raw_image_sources = layer.get("image_sources", [])
                    image_sources = raw_image_sources if isinstance(raw_image_sources, list) else []
                    for image_source in image_sources:
                        if not isinstance(image_source, dict):
                            continue
                        source_slot = int(image_source.get("slot", 0))
                        source_kind = str(image_source.get("source_kind", "media")).strip().lower()
                        source_capture_kind = str(image_source.get("capture_kind", "")).strip().lower()
                        source_path = str(image_source.get("media_path", "")).strip()
                        source_decoder = None
                        if source_kind == "matte":
                            source_rgba = np.asarray(
                                image_source.get("matte_rgba", [255, 255, 255, 255]), dtype=np.uint8
                            ).reshape(1, 1, 4)
                            self.processor.upload_effect_layer_source_rgba(
                                layer_index, source_slot, source_rgba, 1, 1
                            )
                        elif reload_source and source_capture_kind == "logical":
                            if not hasattr(self, "_source_pool"):
                                self._source_pool = SourcePool()
                            source_decoder = self._source_pool.reader(image_source["capture_device_index"])
                        elif reload_source and source_capture_kind == "webcam":
                            source_decoder = EffectCaptureDecoder(
                                int(image_source.get("capture_device_index", 0)),
                                int(image_source.get("capture_width", 0)),
                                int(image_source.get("capture_height", 0)),
                            )
                        elif reload_source and source_path:
                            source_decoder = EffectMediaDecoder(source_path)
                        if source_decoder is not None:
                            source_rgba = source_decoder.next_rgba(loop=bool(image_source.get("media_loop", True)))
                            self._effect_media_decoders[(layer_index, source_slot)] = source_decoder
                            self.processor.upload_effect_layer_source_rgba(
                                layer_index,
                                source_slot,
                                source_rgba,
                                int(source_rgba.shape[1]),
                                int(source_rgba.shape[0]),
                            )
        except Exception:
            self._loaded_effect_source_signature = None
            self.processor.clear_effect_media()
            self.processor.set_effects_config(False, output_connected=False)
            raise
        self.processor.set_effects_config(
            enabled,
            float(payload.get("opacity", 1.0)),
            str(payload.get("blend_mode", "normal")),
            str(payload.get("blur_method", "off")),
            float(payload.get("blur_radius", 0.0)),
            str(payload.get("blur_target", "both")),
            float(payload.get("layer1_opacity", 1.0)),
            str(payload.get("key_mode", "off")),
            int(payload.get("key_color_r", 0)),
            int(payload.get("key_color_g", 255)),
            int(payload.get("key_color_b", 0)),
            float(payload.get("key_similarity", 0.25)),
            float(payload.get("key_softness", 0.10)),
            float(payload.get("spill_suppression", 0.25)),
            float(payload.get("luma_low", 0.0)),
            float(payload.get("luma_high", 1.0)),
            float(payload.get("luma_softness", 0.10)),
            bool(payload.get("key_invert", False)),
            float(payload.get("key_edge_feather", 0.0)),
            bool(payload.get("output_connected", True)),
            bool(payload.get("effect_color_from_alpha", False)),
            bool(payload.get("effect_alpha_from_color", False)),
            bool(payload.get("explicit_compositor_layers", False)),
        )
        _set_native_effects_input_transform(self.processor, payload)
        layers_by_index = {int(layer.get("layer_index", 2)): layer for layer in layers}
        if max(layers_by_index, default=0) > 8 and not hasattr(self.processor, 'set_effect_layer_composition'):
            raise RuntimeError('Rebuild the native module to enable more than eight layers')
        previous_indices = getattr(self, '_configured_effect_layer_indices', set())
        updates, next_layer_indices = effect_layer_updates(layers, enabled, previous_indices)
        self._configured_effect_layer_indices = previous_indices | next_layer_indices
        for layer in updates:
            _set_native_effect_layer_config(self.processor, layer)
        self._configured_effect_layer_indices = next_layer_indices
        _set_native_color_stages(self.processor, payload)
        self._loaded_effect_source_signature = source_signature

    def set_max_auto_basic_scaling(self, scale: int) -> None:
        self.max_auto_basic_scaling = scale
        if self.processor is not None:
            self.processor.set_max_auto_sr_scale(scale)

    def set_basic_scaling_max_inflight(self, max_inflight: int) -> None:
        # In-process backend has no worker parallel-scaling pool to configure.
        self.basic_scaling_max_inflight = max(1, min(4, int(max_inflight)))

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
        effect_layers = _effect_layers_from_payload(self.effects_payload)
        if (
            bool(self.effects_payload.get("enabled", False))
            and any(
                bool(layer.get("media_playing", False))
                or str(layer.get("capture_kind", "")) in {"webcam", "logical"}
                or any(
                    isinstance(source, dict)
                    and (
                        bool(source.get("media_playing", False))
                        or str(source.get("capture_kind", "")) in {"webcam", "logical"}
                    )
                    for source in (
                        layer.get("image_sources", [])
                        if isinstance(layer.get("image_sources", []), list)
                        else []
                    )
                )
                for layer in effect_layers
            )
            and self._effect_media_decoders
        ):
            layers_by_index = {
                int(layer.get("layer_index", 2)): layer
                for layer in effect_layers
            }
            for source_key, decoder in list(self._effect_media_decoders.items()):
                layer_index, source_slot = source_key
                layer = layers_by_index.get(layer_index, {})
                source_config = layer
                if source_slot > 0:
                    raw_sources = layer.get("image_sources", [])
                    source_config = next(
                        (
                            source for source in raw_sources
                            if isinstance(source, dict) and int(source.get("slot", 0)) == source_slot
                        ),
                        {},
                    ) if isinstance(raw_sources, list) else {}
                if not bool(source_config.get("media_playing", False)) and str(source_config.get("capture_kind", "")) not in {"webcam", "logical"}:
                    continue
                try:
                    rgba = decoder.next_rgba(loop=bool(source_config.get("media_loop", True)))
                    if source_slot == 0:
                        self.processor.upload_effect_layer_media_rgba(
                            layer_index, rgba, int(rgba.shape[1]), int(rgba.shape[0])
                        )
                    else:
                        self.processor.upload_effect_layer_source_rgba(
                            layer_index, source_slot, rgba, int(rgba.shape[1]), int(rgba.shape[0])
                        )
                except Exception:
                    decoder.close()
                    self._effect_media_decoders.pop(source_key, None)
                    self.processor.set_effect_layer_config(layer_index, False)
                    raise
        output = self.processor.process_frame(frame_bytes)
        if looks_zeroed_uyvy_frame(output):
            if not self._zeroed_output_warning_emitted:
                LOGGER.warning("GPU processing produced an all-zero UYVY frame; using passthrough fallback")
                self._zeroed_output_warning_emitted = True
            return frame_bytes
        return output

    def close(self) -> None:
        if hasattr(self, "_source_pool"):
            self._source_pool.close()
        for decoder in self._effect_media_decoders.values():
            decoder.close()
        self._effect_media_decoders.clear()
        self._loaded_effect_source_signature = None
        self.processor = None

    def set_basic_scaling_enabled(self, enabled: bool, wait_for_ack: bool = False, timeout_seconds: float = 3.0) -> None:
        # In-process backend: enable_placeholder_sr is constructor-only, so the
        # caller (MainWindow) recreates the processor via create() as needed.
        self.enable_basic_scaling = bool(enabled)

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

    def start_decklink(
        self,
        in_device: int,
        in_mode: object,
        out_device: int,
        out_mode: object,
        enable_format_detection: bool,
        timecode_format: int,
    ) -> None:
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

    def set_roi_with_subpixel(
        self,
        roi: Roi,
        shift_x: float,
        shift_y: float,
        manual_drag: bool = False,
        suspend_timecode: bool = False,
        motion_input: dict[str, object] | None = None,
    ) -> bool:
        _ = manual_drag, suspend_timecode, motion_input
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
        self._roi_command_sequence = 0
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
        self.basic_scaling_max_inflight = max(1, min(4, int(os.environ.get("VP_BASIC_SCALING_MAX_INFLIGHT", "1"))))
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
        self.ai_sr_loading = False
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
        self.decklink_output_buffer_frames = max(0, min(10, int(os.environ.get("VP_DECKLINK_OUTPUT_BUFFER_FRAMES", "0"))))
        self.interlaced_field2_phase_fraction = _clamp_interlaced_field2_phase_fraction(
            float(os.environ.get("VP_INTERLACED_FIELD2_PHASE_FRACTION", "0.50"))
        )
        self.worker_process_priority = _normalize_worker_priority_name(
            os.environ.get("VP_WORKER_PROCESS_PRIORITY", "above_normal")
        )
        self.worker_process_priority_error: str | None = None
        self.worker_keep_alive_enabled = False
        self.worker_keep_alive_error: str | None = None
        self.rtx_vsr_active = False
        self.rtx_vsr_error: str | None = None
        self.rtx_vsr_info: dict[str, object] | None = None
        self.effects_payload: dict[str, object] = {"enabled": False}
        self._effect_source_configured = False

        self._ctx = mp.get_context("spawn")
        self._preview_mailbox = None
        self._request_queue = None
        self._response_queue = None
        self._process = None
        self._roi_telemetry_shared = None
        self._roi_telemetry_seq = None
        self._roi_telemetry_last_seq = -1
        self._manual_roi_mailbox_shared = None
        self._manual_roi_mailbox_seq = None

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
        self._decklink_rtx_last_error: str | None = None
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
        preview_mailbox = getattr(self, "_preview_mailbox", None)
        if preview_mailbox is not None and message.get("preview_sequence") is not None:
            preview = preview_mailbox.read(message["preview_sequence"])
            if preview is not None:
                message["input_frame_bytes"], message["output_frame_bytes"] = preview
        self._latest_effective_scale = int(message.get("effective_sr_scale", self._latest_effective_scale))
        if "input_frame_bytes" in message and "output_frame_bytes" in message:
            self._latest_decklink_frame = (
                message["input_frame_bytes"],
                message["output_frame_bytes"],
            )
            self._decklink_frame_updated = True

        new_counter = int(message.get("processed_frame_counter", self._decklink_processed_counter))
        worker_reported_fps = float(message.get("processed_fps", self._decklink_processed_fps))
        # GUI painting and ROI controls can delay/batch these messages. Their
        # arrival spacing is not the processing (or hardware presentation) clock.
        now = time.perf_counter()
        self._decklink_processed_counter = new_counter
        self._decklink_processed_counter_last = new_counter
        self._decklink_processed_counter_last_ts = now
        self._decklink_processed_fps_smoothed = max(0.0, worker_reported_fps)
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
        self._decklink_rtx_last_error = message.get("rtx_stage_last_error", self._decklink_rtx_last_error)
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

        # Basic CUDA scaling is always kept available as the worker-side
        # fallback layer beneath AI SR/RTX VSR (toggled at runtime via
        # set_basic_scaling_enabled), so it is not suppressed here anymore.
        effective_basic_scaling_enabled = bool(self.enable_basic_scaling)
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
            "basic_scaling_max_inflight": int(self.basic_scaling_max_inflight),
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

        # Keep request queue larger than response queue so control changes and
        # tick polling do not trip queue.Full in the GUI thread.
        self._request_queue = self._ctx.Queue(maxsize=32)
        self._response_queue = self._ctx.Queue(maxsize=64)
        self._preview_mailbox = PreviewMailbox(self._ctx, FRAME_W * FRAME_H * 2)
        self._roi_telemetry_shared = self._ctx.Array("d", _ROI_TELEMETRY_SLOT_COUNT)
        self._roi_telemetry_seq = self._ctx.Value("i", 0)
        self._roi_telemetry_last_seq = -1
        self._manual_roi_mailbox_shared = self._ctx.Array("d", _MANUAL_ROI_MAILBOX_SLOT_COUNT)
        self._manual_roi_mailbox_seq = self._ctx.Value("i", 0)
        self._process = self._ctx.Process(
            target=run_processor_worker,
            args=(
                self._request_queue,
                self._response_queue,
                startup_config,
                self._roi_telemetry_shared,
                self._roi_telemetry_seq,
                self._manual_roi_mailbox_shared,
                self._manual_roi_mailbox_seq,
                self._preview_mailbox,
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
        if bool(self.effects_payload.get("enabled", False)):
            self.set_effects_config(self.effects_payload)

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
                self.ai_sr_loading = bool(message.get("ai_sr_loading", False))
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
                self.worker_keep_alive_enabled = bool(message.get("worker_keep_alive_enabled", False))
                self.worker_keep_alive_error = (
                    str(message.get("worker_keep_alive_error", "")).strip() or None
                )
                if self.worker_keep_alive_error:
                    LOGGER.warning("Worker keep-alive setup warning: %s", self.worker_keep_alive_error)
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
            "set_roi_settled",
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
            "set_roi_settled",
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

    def _next_roi_command_sequence(self) -> int:
        self._roi_command_sequence += 1
        return self._roi_command_sequence

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
                elif ack_cmd == "set_basic_scaling_enabled":
                    self.enable_basic_scaling = bool(message.get("basic_scaling_enabled", self.enable_basic_scaling))
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
                    self.ai_sr_loading = bool(message.get("ai_sr_loading", False))
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
                elif ack_cmd == "set_basic_scaling_max_inflight":
                    self.basic_scaling_max_inflight = max(
                        1,
                        min(4, int(message.get("basic_scaling_max_inflight", self.basic_scaling_max_inflight))),
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

            if message_type == "ai_sr_engine_update":
                # Background AI SR engine build finished (or failed) after an
                # earlier ack already reported "loading"; refresh final state.
                self.ai_sr_enabled = bool(message.get("ai_sr_enabled", self.ai_sr_enabled))
                self.ai_sr_active = bool(message.get("ai_sr_active", self.ai_sr_active))
                self.ai_sr_loading = bool(message.get("ai_sr_loading", False))
                self.ai_sr_error = message.get("ai_sr_error")
                self.ai_sr_info = message.get("ai_sr_info")
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
        return self._send_control(
            {
                "cmd": "set_roi",
                "x": roi.x,
                "y": roi.y,
                "w": roi.w,
                "h": roi.h,
                "roi_sequence": self._next_roi_command_sequence(),
            }
        )

    def set_roi_settled(self, roi: Roi) -> bool:
        return self._send_control(
            {
                "cmd": "set_roi_settled",
                "x": int(roi.x),
                "y": int(roi.y),
                "w": int(roi.w),
                "h": int(roi.h),
                "roi_sequence": self._next_roi_command_sequence(),
            }
        )

    def set_roi_position(self, roi_x: int, roi_y: int) -> bool:
        return self._send_control(
            {
                "cmd": "set_roi_position",
                "x": int(roi_x),
                "y": int(roi_y),
                "roi_sequence": self._next_roi_command_sequence(),
            }
        )

    def set_roi_subpixel_shift(self, shift_x: float, shift_y: float) -> None:
        self._send_control(
            {
                "cmd": "set_roi_subpixel_shift",
                "shift_x": float(shift_x),
                "shift_y": float(shift_y),
                "roi_sequence": self._next_roi_command_sequence(),
            }
        )

    def set_roi_with_subpixel(
        self,
        roi: Roi,
        shift_x: float,
        shift_y: float,
        manual_drag: bool = False,
        suspend_timecode: bool = False,
        motion_input: dict[str, object] | None = None,
    ) -> bool:
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
                "suspend_timecode": bool(suspend_timecode),
                "motion_input": dict(motion_input) if isinstance(motion_input, dict) else {},
                "roi_sequence": self._next_roi_command_sequence(),
            }
        )

    def publish_manual_roi_endpoint(
        self,
        roi: Roi,
        shift_x: float,
        shift_y: float,
        suspend_timecode: bool = False,
        motion_input: dict[str, object] | None = None,
    ) -> bool:
        shared = self._manual_roi_mailbox_shared
        sequence = self._manual_roi_mailbox_seq
        if shared is None or sequence is None:
            return False

        diagnostics = motion_input if isinstance(motion_input, dict) else {}
        source = str(diagnostics.get("source", "unknown"))
        source_code = float(
            {
                "touch": 1,
                "mouse:MouseEventNotSynthesized": 2,
                "mouse:MouseEventSynthesizedBySystem": 3,
                "mouse:MouseEventSynthesizedByQt": 4,
                "mouse:MouseEventSynthesizedByApplication": 5,
            }.get(source, 0)
        )
        command_sequence = self._next_roi_command_sequence()
        try:
            with sequence.get_lock():
                sequence.value = int(sequence.value) + 1
            try:
                with shared.get_lock():
                    values = (
                        1.0,
                        float(roi.x),
                        float(roi.y),
                        float(roi.w),
                        float(roi.h),
                        float(shift_x),
                        float(shift_y),
                        1.0 if suspend_timecode else 0.0,
                        source_code,
                        float(diagnostics.get("event_dt_ms", 0.0)),
                        float(diagnostics.get("event_delta_px", 0.0)),
                        float(command_sequence),
                    )
                    for index, value in enumerate(values):
                        shared[index] = value
            finally:
                with sequence.get_lock():
                    sequence.value = int(sequence.value) + 1
            return True
        except Exception:
            return False

    def clear_manual_roi_endpoint(self) -> None:
        shared = self._manual_roi_mailbox_shared
        sequence = self._manual_roi_mailbox_seq
        if shared is None or sequence is None:
            return
        command_sequence = self._next_roi_command_sequence()
        try:
            with sequence.get_lock():
                sequence.value = int(sequence.value) + 1
            try:
                with shared.get_lock():
                    shared[0] = 0.0
                    shared[11] = float(command_sequence)
            finally:
                with sequence.get_lock():
                    sequence.value = int(sequence.value) + 1
        except Exception:
            pass

    def set_timecode_roi_keyframes(
        self,
        enabled: bool,
        keyframes: list[dict[str, object]],
        phase_mode: str,
    ) -> None:
        self._send_control(
            {
                "cmd": "set_timecode_roi_keyframes",
                "enabled": bool(enabled),
                "keyframes": keyframes,
                "phase_mode": str(phase_mode),
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
        roi_sequence = self._next_roi_command_sequence()
        self._send_control(
            {
                "cmd": "start_roi_microstep_transition",
                "roi_sequence": roi_sequence,
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

    def set_effects_config(self, payload: dict[str, object]) -> None:
        reload_source = (
            not self._effect_source_configured
            or _effects_source_signature(payload) != _effects_source_signature(self.effects_payload)
        )
        self.effects_payload = dict(payload)
        command = dict(payload)
        command["cmd"] = "set_effects_config"
        command["reload_source"] = reload_source
        self._send_control(command)
        self._wait_for_ack("set_effects_config", timeout_seconds=5.0)
        self._effect_source_configured = True

    def activate_source(self, logical_id, config):
        self._send_control({"cmd": "activate_source", "logical_id": logical_id, "config": config})
        self._wait_for_ack("activate_source", timeout_seconds=15.0)

    def deactivate_source(self, logical_id):
        self._send_control({"cmd": "deactivate_source", "logical_id": logical_id})
        self._wait_for_ack("deactivate_source", timeout_seconds=5.0)

    def show_capture_settings(self, device_index: int) -> None:
        self._send_control({"cmd": "show_capture_settings", "device_index": int(device_index)})
        self._wait_for_ack("show_capture_settings", timeout_seconds=300.0)

    def set_max_auto_basic_scaling(self, scale: int) -> None:
        self.max_auto_basic_scaling = scale
        self._send_control({"cmd": "set_max_auto_basic_scaling", "scale": int(scale)})

    def set_basic_scaling_max_inflight(self, max_inflight: int) -> None:
        self.basic_scaling_max_inflight = max(1, min(4, int(max_inflight)))
        self._send_control(
            {"cmd": "set_basic_scaling_max_inflight", "max_inflight": self.basic_scaling_max_inflight}
        )

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
                if message.get("source_error"):
                    raise RuntimeError(str(message["source_error"]))
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
                if expected_cmd == "set_basic_scaling_enabled":
                    self.enable_basic_scaling = bool(message.get("basic_scaling_enabled", self.enable_basic_scaling))
                if expected_cmd == "set_deinterlace_method":
                    self.deinterlace_method = str(message.get("deinterlace_method", self.deinterlace_method))
                if expected_cmd == "set_deinterlace_enabled":
                    self.deinterlace_enabled = bool(message.get("deinterlace_enabled", self.deinterlace_enabled))
                if expected_cmd == "set_reinterlace_enabled":
                    self.reinterlace_enabled = bool(message.get("reinterlace_enabled", self.reinterlace_enabled))
                if expected_cmd == "set_denoise_settings":
                    self.denoise_method = str(message.get("denoise_method", self.denoise_method))
                    self.denoise_strength = float(message.get("denoise_strength", self.denoise_strength))
                if expected_cmd == "set_effects_config":
                    raw_effects_error = message.get("effects_error")
                    effects_error = str(raw_effects_error).strip() if raw_effects_error is not None else ""
                    if effects_error:
                        raise RuntimeError(effects_error)
                if expected_cmd == "show_capture_settings":
                    raw_settings_error = message.get("capture_settings_error")
                    settings_error = str(raw_settings_error).strip() if raw_settings_error is not None else ""
                    if settings_error:
                        raise RuntimeError(settings_error)
                if expected_cmd in {"set_ai_sr_enabled", "set_ai_sr_model_path", "set_ai_sr_settings"}:
                    self.ai_sr_enabled = bool(message.get("ai_sr_enabled", self.ai_sr_enabled))
                    self.ai_sr_active = bool(message.get("ai_sr_active", self.ai_sr_active))
                    self.ai_sr_loading = bool(message.get("ai_sr_loading", False))
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
            if message_type == "ai_sr_engine_update":
                self.ai_sr_enabled = bool(message.get("ai_sr_enabled", self.ai_sr_enabled))
                self.ai_sr_active = bool(message.get("ai_sr_active", self.ai_sr_active))
                self.ai_sr_loading = bool(message.get("ai_sr_loading", False))
                self.ai_sr_error = message.get("ai_sr_error")
                self.ai_sr_info = message.get("ai_sr_info")
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

    def start_decklink(
        self,
        in_device: int,
        in_mode: object,
        out_device: int,
        out_mode: object,
        enable_format_detection: bool,
        timecode_format: int,
    ) -> None:
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
                "in_device": int(in_device) if in_device is not None else None,
                "in_mode": in_mode,
                "out_device": int(out_device),
                "out_mode": out_mode,
                "enable_format_detection": bool(enable_format_detection),
                "timecode_format": int(timecode_format),
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
        self._preview_mailbox = None
        self._roi_telemetry_shared = None
        self._roi_telemetry_seq = None
        self._roi_telemetry_last_seq = -1
        self._manual_roi_mailbox_shared = None
        self._manual_roi_mailbox_seq = None
        self._decklink_tick_pending = False
        self._decklink_tick_pending_since = 0.0
        self._effect_source_configured = False

    def set_basic_scaling_enabled(self, enabled: bool, wait_for_ack: bool = False, timeout_seconds: float = 3.0) -> None:
        self.enable_basic_scaling = bool(enabled)
        self._send_control({"cmd": "set_basic_scaling_enabled", "enabled": bool(enabled)})
        if wait_for_ack:
            self._wait_for_ack("set_basic_scaling_enabled", timeout_seconds=max(0.5, float(timeout_seconds)))

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

    def decklink_rtx_last_error(self) -> str | None:
        return self._decklink_rtx_last_error

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
        self._roi_drag_x_hysteresis_px = max(0.10, min(1.20, float(os.environ.get("VP_ROI_DRAG_X_HYSTERESIS_PX", "0.45"))))
        self._roi_manual_drag_hold_s = max(0.05, min(0.50, float(os.environ.get("VP_ROI_MANUAL_DRAG_HOLD_S", "0.24"))))
        self._interlaced_field2_phase_fraction = _clamp_interlaced_field2_phase_fraction(
            float(os.environ.get("VP_INTERLACED_FIELD2_PHASE_FRACTION", "0.50"))
        )
        self._manual_drag_worker_send_hz = max(60.0, min(120.0, float(os.environ.get("VP_MANUAL_DRAG_WORKER_SEND_HZ", "60"))))
        self._manual_drag_worker_send_interval_ms = max(1, int(math.floor(1000.0 / self._manual_drag_worker_send_hz)))
        self._last_effects_payload: dict[str, object] | None = None
        self._manual_roi_frame_lock_to_output = os.environ.get("VP_MANUAL_ROI_FRAME_LOCK", "1") != "0"
        self._manual_roi_last_send_ts = 0.0
        self._roi_keyframes: dict[int, RoiKeyframe] = {}
        self._roi_keyframe_slots = (1, 2, 3, 4)
        self._roi_key_save_armed = False
        self._timecode_keyframing_enabled = False
        self._timecode_playback_enabled = True
        self._timecode_roi_keyframes: dict[int, TimecodeRoiKeyframe] = {}
        self._timecode_adjustment_anchor: TimecodeRoiKeyframe | None = None
        self._timecode_adjustment_paused = False
        self._timecode_adjustment_finish_pending = False
        self._timecode_resume_debounce_ms = max(
            0,
            min(1000, int(float(os.environ.get("VP_TIMECODE_MANUAL_RESUME_DEBOUNCE_MS", "180")))),
        )
        self._timecode_resume_status = ""
        self._timecode_roi_lookup_keyframes: dict[int, TimecodeRoiKeyframe] = {}
        self._timecode_roi_ordered_frames: list[int] = []
        self._timecode_roi_segment_starts: list[int] = []
        self._timecode_roi_segments: list[tuple[int, int, np.ndarray]] = []
        self._timecode_selected_frame: int | None = None
        self._timecode_last_applied_frame: float | None = None
        self._timecode_phase_tracker: dict[str, object] = {}
        self._decklink_timecode_info_sequence = 0
        self._decklink_timecode_info: dict[str, object] = {}
        self._roi_keyframe_transition_default_frames = 30
        self._fullscreen_scale_presets = [200, 300, 400]
        self._fullscreen_selected_scale_index: int | None = None
        self._roi_keyframe_transition: dict[str, object] | None = None
        self._roi_keyframe_last_step_ts = 0.0
        self._roi_keyframe_target_fps = 60.0
        self._roi_keyframe_transition_overscan_percent = 2.0
        self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        self._controller_filtered_target_roi: Roi | None = None
        self._last_status_text: str | None = None
        self._last_status_log_ts = 0.0
        self._status_repeat_log_interval_s = 5.0
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
        self._decklink_buffer_guard_enabled = os.environ.get("VP_DECKLINK_BUFFER_GUARD", "0") == "1"
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
        self._windowed_qt_geometry_before_fullscreen: QByteArray | None = None
        self._windowed_geometry_before_fullscreen: QRect | None = None
        self._windowed_available_geometry_before_fullscreen: QRect | None = None
        self._windowed_was_maximized_before_fullscreen = False
        self._windowed_display_splitter_sizes: list[int] | None = None
        self._windowed_main_splitter_sizes: list[int] | None = None
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
            min(10, int(getattr(self._controller, "decklink_output_buffer_frames", 0))),
        )
        self._decklink_output_buffer_user_target_frames = int(self._decklink_output_buffer_frames)

        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        self._fullscreen_keyframe_toolbars: dict[str, QWidget] = {}
        self._fullscreen_keyframe_side_panels: dict[str, QWidget] = {}
        self._fullscreen_keyframe_title_labels: dict[str, QLabel] = {}
        self._fullscreen_manual_keyframe_rows: dict[str, QWidget] = {}
        self._fullscreen_timecode_keyframe_rows: dict[str, QWidget] = {}
        self._fullscreen_keyframing_mode_buttons: dict[str, QPushButton] = {}
        self._fullscreen_timecode_display_labels: dict[str, QLabel] = {}
        self._fullscreen_timecode_key_labels: dict[str, QLabel] = {}
        self._fullscreen_timecode_delete_buttons: dict[str, QPushButton] = {}
        self._fullscreen_timecode_delete_all_buttons: dict[str, QPushButton] = {}
        self._fullscreen_timecode_previous_buttons: dict[str, QPushButton] = {}
        self._fullscreen_timecode_next_buttons: dict[str, QPushButton] = {}
        self._fullscreen_timecode_playback_buttons: dict[str, QPushButton] = {}
        self._fullscreen_roi_save_key_buttons: dict[str, QPushButton] = {}
        self._fullscreen_roi_key_slot_buttons: dict[str, tuple[QPushButton, QPushButton, QPushButton, QPushButton]] = {}
        self._fullscreen_roi_transition_labels: dict[str, QLabel] = {}
        self._fullscreen_roi_transition_rate_spins: dict[str, QSpinBox] = {}
        self._fullscreen_roi_interp_mode_combos: dict[str, QComboBox] = {}
        self._fullscreen_roi_duration_override_buttons: dict[str, QPushButton] = {}
        self._fullscreen_scale_buttons: dict[str, tuple[QPushButton, QPushButton, QPushButton]] = {}
        self._fullscreen_enter_buttons: dict[str, QPushButton] = {}
        viewers = QWidget()
        viewers.setMinimumWidth(0)
        viewers_layout = QVBoxLayout(viewers)
        viewers_layout.setContentsMargins(0, 0, 0, 0)
        viewers_layout.setSpacing(4)

        self._input_panel = QWidget()
        self._input_panel.setMinimumWidth(0)
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
        self._input_viewer_row.setMinimumWidth(0)
        input_viewer_row_layout = QHBoxLayout(self._input_viewer_row)
        input_viewer_row_layout.setContentsMargins(0, 0, 0, 0)
        input_viewer_row_layout.setSpacing(12)
        self._input_fullscreen_keyframe_side_panel = self._build_fullscreen_keyframe_side_panel("input")
        input_viewer_row_layout.addWidget(self._input_fullscreen_keyframe_side_panel, 0)
        input_viewer_row_layout.addWidget(self._input_canvas, 1, alignment=Qt.AlignCenter)
        input_layout.addWidget(self._input_viewer_row, 1)

        self._input_fullscreen_keyframe_toolbar = self._build_fullscreen_keyframe_toolbar("input")
        input_layout.addWidget(self._input_fullscreen_keyframe_toolbar)

        self._output_panel = QWidget()
        self._output_panel.setMinimumWidth(0)
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
        self._output_viewer_row.setMinimumWidth(0)
        output_viewer_row_layout = QHBoxLayout(self._output_viewer_row)
        output_viewer_row_layout.setContentsMargins(0, 0, 0, 0)
        output_viewer_row_layout.setSpacing(12)
        self._output_fullscreen_keyframe_side_panel = self._build_fullscreen_keyframe_side_panel("output")
        output_viewer_row_layout.addWidget(self._output_fullscreen_keyframe_side_panel, 0)
        output_viewer_row_layout.addWidget(self._output_canvas, 1, alignment=Qt.AlignCenter)
        output_layout.addWidget(self._output_viewer_row, 1)

        self._output_fullscreen_keyframe_toolbar = self._build_fullscreen_keyframe_toolbar("output")
        output_layout.addWidget(self._output_fullscreen_keyframe_toolbar)

        self._display_splitter = QSplitter(Qt.Vertical)
        self._display_splitter.setMinimumWidth(0)
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
        self._controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._controls_scroll.setMinimumWidth(420)

        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.addWidget(viewers)
        self._main_splitter.addWidget(self._controls_scroll)
        self._main_splitter.setStretchFactor(0, 4)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.splitterMoved.connect(lambda _pos, _index: self._fit_viewers_to_video_aspect())
        root.addWidget(self._main_splitter, 1)

        self._input_canvas.set_roi(self._roi)
        self._input_canvas.adjustmentStarted.connect(self._on_roi_adjustment_started)
        self._input_canvas.adjustmentFinished.connect(self._on_roi_adjustment_finished)
        self._input_canvas.manualDragEndpointChanged.connect(self._on_manual_drag_endpoint)
        self._input_canvas.roiChanged.connect(self._on_roi_from_canvas)
        self._input_canvas.scaleChanged.connect(self._on_scale_from_canvas)
        self._input_canvas.tapCenterRequested.connect(self._on_roi_tap_center_requested)
        self._input_canvas.fullscreenRequested.connect(self._on_canvas_fullscreen_requested)
        self._output_canvas.tapCenterRequested.connect(self._on_output_roi_tap_center_requested)
        self._output_canvas.fullscreenRequested.connect(self._on_canvas_fullscreen_requested)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._update_timer_interval()
        self._timer.start()

        self._controller_roi_interp_timer = QTimer(self)
        self._controller_roi_interp_timer.setInterval(16)
        self._controller_roi_interp_timer.setTimerType(Qt.PreciseTimer)
        self._controller_roi_interp_timer.timeout.connect(self._step_controller_roi_interpolation)

        self._manual_roi_send_timer = QTimer(self)
        self._manual_roi_send_timer.setSingleShot(True)
        self._manual_roi_send_timer.setInterval(16)
        self._manual_roi_send_timer.setTimerType(Qt.PreciseTimer)
        self._manual_roi_send_timer.timeout.connect(self._flush_pending_manual_controller_roi)

        self._roi_controls_sync_timer = QTimer(self)
        self._roi_controls_sync_timer.setSingleShot(True)
        self._roi_controls_sync_timer.setInterval(33)
        self._roi_controls_sync_timer.timeout.connect(self._flush_pending_roi_controls_sync)

        self._roi_control_adjustment_timer = QTimer(self)
        self._roi_control_adjustment_timer.setSingleShot(True)
        self._roi_control_adjustment_timer.setInterval(150)
        self._roi_control_adjustment_timer.timeout.connect(self._on_roi_adjustment_finished)

        self._timecode_resume_timer = QTimer(self)
        self._timecode_resume_timer.setSingleShot(True)
        self._timecode_resume_timer.timeout.connect(self._resume_timecode_after_manual_adjustment)

        self._roi_keyframe_transition_timer = QTimer(self)
        self._roi_keyframe_transition_timer.setInterval(16)
        self._roi_keyframe_transition_timer.setTimerType(Qt.PreciseTimer)
        self._roi_keyframe_transition_timer.timeout.connect(self._step_roi_keyframe_transition)

        self._setup_shortcuts()
        self._connect_settings_persistence_signals()
        self.roi_transition_frames_spin.valueChanged.connect(self._sync_fullscreen_transition_rate_from_main)
        self.roi_interp_mode_combo.currentTextChanged.connect(self._sync_fullscreen_interp_mode_from_main)
        self.roi_keyframe_duration_override_btn.toggled.connect(self._sync_fullscreen_override_duration_from_main)
        self._update_roi_key_buttons()
        self._sync_fullscreen_transition_rate_from_main(self.roi_transition_frames_spin.value())
        self._sync_fullscreen_interp_mode_from_main(self.roi_interp_mode_combo.currentText())
        self._sync_fullscreen_override_duration_from_main(self.roi_keyframe_duration_override_btn.isChecked())
        self._sync_fullscreen_button_states()
        self._sync_roi_transition_unit_labels()
        self._sync_controls_from_roi(self._roi)
        self._load_settings()
        self._input_sources_changed()
        self._apply_manual_drag_tuning_to_controller()
        self._apply_interlaced_phase_tuning_to_controller()
        self._apply_startup_ai_sr_settings()
        self._source_mode = self.source_mode_combo.currentText()
        self._sync_blackmagic_controls_enabled_state()
        self._refresh_decklink_catalog()
        self._source_mode = "Synthetic"
        self._sync_roi_transition_unit_labels()
        self._refresh_ai_sr_runtime_panel()
        self._refresh_rtx_vsr_runtime_panel()
        self._update_status("Ready")
        if self._controller_backend == "worker-process":
            self._update_status("Ready | Processing backend: worker process")
        else:
            self._update_status("Ready | Processing backend: in-process")
            self._set_decklink_status("Worker backend not active; running in GUI process")
        LOGGER.info("GUI initialized; default source mode=%s", self._source_mode)
        QTimer.singleShot(0, self._apply_initial_viewer_layout)
        QTimer.singleShot(0, self._restore_active_io)

    def _restore_active_io(self) -> None:
        saved = getattr(self, "_pending_io_restore", None)
        self._pending_io_restore = None
        if not saved or self._is_closing:
            return
        self._restoring_io = True
        failures = []
        try:
            failures.extend(self.input_sources.restore_active(saved["inputs"]))
            if saved["output"]:
                try:
                    device_index = self.decklink_output_device_combo.findData(saved["device"])
                    if saved["device"] is None or device_index < 0:
                        raise RuntimeError("Saved output device is unavailable")
                    self.decklink_output_device_combo.setCurrentIndex(device_index)
                    mode_index = self.decklink_output_mode_combo.findText(saved["mode"])
                    if mode_index < 0:
                        raise RuntimeError("Saved output mode is unavailable")
                    self.decklink_output_mode_combo.setCurrentIndex(mode_index)
                    # Restore the saved device, not auto-detection's first device.
                    auto_detect = self.decklink_auto_detect_devices.isChecked()
                    self.decklink_auto_detect_devices.blockSignals(True)
                    self.decklink_auto_detect_devices.setChecked(False)
                    try:
                        self._on_apply_decklink_settings()
                    finally:
                        self.decklink_auto_detect_devices.setChecked(auto_detect)
                        self.decklink_auto_detect_devices.blockSignals(False)
                    if not self._decklink_sessions_running:
                        failures.append("Output could not restart; see output status")
                except Exception as exc:
                    failures.append(f"Output: {exc}")
        finally:
            self._restoring_io = False
        if failures:
            self._update_status("I/O restore: " + "; ".join(failures))
            LOGGER.warning("I/O restore: %s", "; ".join(failures))
        self._schedule_settings_save()

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
        self._apply_scaling_mode_visibility(self.scaling_mode_combo.currentText())
        self._apply_startup_ai_sr_settings()
        self._apply_controller_color_settings_from_ui()
        self._apply_worker_process_priority_to_controller(notify=False)
        self._apply_manual_drag_tuning_to_controller()
        self._apply_interlaced_phase_tuning_to_controller()
        effects_payload = self.effects_graph.native_effects_payload()
        self._controller.set_effects_config(effects_payload)
        self._last_effects_payload = dict(effects_payload)
        for logical_id in list(self.input_sources.active):
            try:
                self._controller.activate_source(logical_id, self.input_sources.configs[logical_id - 1])
            except Exception as exc:
                self.input_sources.active.discard(logical_id)
                combo, settings, _name, button = self.input_sources.rows[logical_id - 1]
                combo.setEnabled(True)
                settings.setEnabled(True)
                button.setText("Activate")
                self._update_status(f"Source {logical_id} could not restart: {exc}")
        self._input_sources_changed()
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
            self.scaling_mode_combo,
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
            self.decklink_timecode_format_combo,
            self.decklink_timecode_phase_combo,
            self.worker_priority_combo,
            self.roi_interp_mode_combo,
        ]
        for combo in combo_widgets:
            combo.currentTextChanged.connect(self._schedule_settings_save)

        checkbox_widgets = [
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
            self.basic_scaling_max_inflight_spin,
            self.roi_x_spin,
            self.roi_y_spin,
            self.roi_w_spin,
            self.roi_h_spin,
            self.scale_spin,
            self.roi_drag_x_hysteresis_spin,
            self.roi_manual_drag_hold_spin,
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
        if self._updating_controls or getattr(self, "_restoring_io", False) or self._is_closing:
            return
        self._settings_save_timer.start()

    def _collect_settings_payload(self) -> dict[str, object]:
        keyframe_payload: dict[str, object] = {}
        for slot, keyframe in self._roi_keyframes.items():
            keyframe_payload[str(slot)] = self._serialize_roi_keyframe(keyframe)
        timecode_keyframe_payload = [
            {
                "timecode": keyframe.timecode,
                "frame_number": int(keyframe.frame_number),
                "roi": [keyframe.roi.x, keyframe.roi.y, keyframe.roi.w, keyframe.roi.h],
                "interpolation_mode": keyframe.interpolation_mode,
                "timecode_format": keyframe.timecode_format,
                "drop_frame": bool(keyframe.drop_frame),
                "field_mark": bool(keyframe.field_mark),
            }
            for keyframe in sorted(self._timecode_roi_keyframes.values(), key=lambda item: item.frame_number)
        ]

        return {
            "version": 1,
            "input_sources": self.input_sources.serialize(),
            "active_input_sources": sorted(self.input_sources.active),
            "output_active": bool(self._decklink_sessions_running),
            "fps": int(self.fps_spin.value()),
            "preview_request_fps": int(self.preview_request_fps_spin.value()),
            "preview_poll_fps": int(self.preview_poll_fps_spin.value()),
            "decklink_output_buffer_frames": int(self.decklink_output_buffer_spin.value()),
            "preview_downsample": str(self.preview_downsample_combo.currentText()),
            "color_space": str(self.color_space_combo.currentText()),
            "color_range": str(self.color_range_combo.currentText()),
            "roi_drag_x_hysteresis_px": float(self.roi_drag_x_hysteresis_spin.value()),
            "roi_manual_drag_hold_s": float(self.roi_manual_drag_hold_spin.value()),
            "interlaced_field2_phase_fraction": float(self.roi_interlaced_field2_phase_spin.value()),
            "roi_transition_duration_frames": int(self.roi_transition_frames_spin.value()),
            "roi_interpolation_mode": str(self.roi_interp_mode_combo.currentText()),
            "roi_keyframe_duration_override": bool(self.roi_keyframe_duration_override_btn.isChecked()),
            "fullscreen_scale_presets": list(self._fullscreen_scale_presets),
            "fullscreen_selected_scale_index": self._fullscreen_selected_scale_index,
            "roi_keyframes": keyframe_payload,
            "roi_keyframing_mode": "timecode" if self._timecode_keyframing_enabled else "manual",
            "roi_timecode_playback_enabled": bool(self._timecode_playback_enabled),
            "roi_timecode_keyframes": timecode_keyframe_payload,
            "basic_scaling_mode": str(self.sr_mode_combo.currentText()),
            "basic_scaling_method": str(self.sr_flavor_combo.currentText()),
            "basic_scaling_manual": str(self.sr_manual_combo.currentText()),
            "basic_scaling_auto_max": str(self.auto_sr_max_combo.currentText()),
            "basic_scaling_max_inflight": int(self.basic_scaling_max_inflight_spin.value()),
            "scaling_mode": str(self.scaling_mode_combo.currentText()),
            "basic_scaling_enabled": self.scaling_mode_combo.currentText() == SCALING_MODE_BASIC,
            "deinterlace_enabled": bool(self.deinterlace_checkbox.isChecked()),
            "reinterlace_enabled": bool(self.reinterlace_checkbox.isChecked()),
            "deinterlace_method": str(self.deinterlace_method_combo.currentText()),
            "denoise_method": str(self.denoise_method_combo.currentText()),
            "denoise_strength": float(self.denoise_strength_spin.value()),
            "effects_graph": self.effects_graph.serialize(),
            "perf_guard_enabled": bool(self.perf_guard_checkbox.isChecked()),
            "ai_sr_enabled": self.scaling_mode_combo.currentText() == SCALING_MODE_ONNX_SR,
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
            "rtx_vsr_enabled": self.scaling_mode_combo.currentText() == SCALING_MODE_RTX_SR,
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
            "decklink_timecode_format": str(self.decklink_timecode_format_combo.currentText()),
            "decklink_timecode_phase_mode": str(self.decklink_timecode_phase_combo.currentData()),
            "decklink_enable_format_detection": bool(self.decklink_enable_format_detection.isChecked()),
            "decklink_fps_priority_guard": bool(self.decklink_fps_priority_guard_checkbox.isChecked()),
            "worker_process_priority": str(self.worker_priority_combo.currentText()),
            "display_splitter_sizes": list(
                self._windowed_display_splitter_sizes
                if self._fullscreen_view_name is not None and self._windowed_display_splitter_sizes
                else self._display_splitter.sizes()
            ),
            "main_splitter_sizes": list(
                self._windowed_main_splitter_sizes
                if self._fullscreen_view_name is not None and self._windowed_main_splitter_sizes
                else self._main_splitter.sizes()
            ),
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

        active_inputs = raw.get("active_input_sources", [])
        self._pending_io_restore = {
            "inputs": active_inputs if isinstance(active_inputs, list) else [],
            "output": raw.get("output_active") is True,
            "device": raw.get("decklink_output_device"),
            "mode": str(raw.get("decklink_output_mode_text", "")),
        }
        self.input_sources.restore(raw.get("input_sources", []))
        if "input_sources" not in raw and isinstance(raw.get("decklink_input_device"), int):
            self.input_sources.configs[0].update(
                device=f"decklink:{raw['decklink_input_device']}",
                device_name=f"DeckLink {raw['decklink_input_device']}",
                mode_text=raw.get("decklink_input_mode_text", ""),
                timecode_format=self.decklink_timecode_format_combo.itemData(max(0,
                    self.decklink_timecode_format_combo.findText(str(raw.get("decklink_timecode_format", "RP188 VITC1"))))),
                timecode_phase=raw.get("decklink_timecode_phase_mode", "off"),
                format_detection=raw.get("decklink_enable_format_detection", True),
            )
        self._has_persisted_deinterlace_method = bool(str(raw.get("deinterlace_method", "")).strip())

        self._updating_controls = True
        try:
            self.fps_spin.setValue(max(1, min(60, int(raw.get("fps", self.fps_spin.value())))))
            self.preview_request_fps_spin.setValue(max(1, min(60, int(raw.get("preview_request_fps", self.preview_request_fps_spin.value())))))
            self.preview_poll_fps_spin.setValue(max(1, min(120, int(raw.get("preview_poll_fps", self.preview_poll_fps_spin.value())))))
            self.decklink_output_buffer_spin.setValue(
                max(0, min(10, int(raw.get("decklink_output_buffer_frames", self.decklink_output_buffer_spin.value()))))
            )
            self.roi_drag_x_hysteresis_spin.setValue(
                max(0.10, min(1.20, float(raw.get("roi_drag_x_hysteresis_px", self.roi_drag_x_hysteresis_spin.value()))))
            )
            self.roi_manual_drag_hold_spin.setValue(
                max(0.05, min(0.50, float(raw.get("roi_manual_drag_hold_s", self.roi_manual_drag_hold_spin.value()))))
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
            raw_scale_presets = raw.get("fullscreen_scale_presets", self._fullscreen_scale_presets)
            if isinstance(raw_scale_presets, list) and len(raw_scale_presets) == 3:
                self._fullscreen_scale_presets = [max(100, min(1000000, int(value))) for value in raw_scale_presets]
            raw_selected_scale_index = raw.get("fullscreen_selected_scale_index")
            if isinstance(raw_selected_scale_index, int) and 0 <= raw_selected_scale_index < 3:
                self._fullscreen_selected_scale_index = raw_selected_scale_index
            else:
                self._fullscreen_selected_scale_index = None
            self._sync_fullscreen_scale_buttons()

            self.preview_downsample_combo.setCurrentText(str(raw.get("preview_downsample", self.preview_downsample_combo.currentText())))
            self.color_space_combo.setCurrentText(str(raw.get("color_space", self.color_space_combo.currentText())))
            self.color_range_combo.setCurrentText(str(raw.get("color_range", self.color_range_combo.currentText())))
            self.sr_mode_combo.setCurrentText(str(raw.get("basic_scaling_mode", self.sr_mode_combo.currentText())))
            self.sr_flavor_combo.setCurrentText(str(raw.get("basic_scaling_method", self.sr_flavor_combo.currentText())))
            self.sr_manual_combo.setCurrentText(str(raw.get("basic_scaling_manual", self.sr_manual_combo.currentText())))
            self.auto_sr_max_combo.setCurrentText(str(raw.get("basic_scaling_auto_max", self.auto_sr_max_combo.currentText())))
            self.basic_scaling_max_inflight_spin.setValue(
                max(1, min(4, int(raw.get("basic_scaling_max_inflight", self.basic_scaling_max_inflight_spin.value()))))
            )

            if "scaling_mode" in raw:
                restored_scaling_mode = str(raw.get("scaling_mode", SCALING_MODE_BASIC))
            elif bool(raw.get("ai_sr_enabled", False)):
                restored_scaling_mode = SCALING_MODE_ONNX_SR
            elif bool(raw.get("rtx_vsr_enabled", False)):
                restored_scaling_mode = SCALING_MODE_RTX_SR
            else:
                restored_scaling_mode = SCALING_MODE_BASIC
            if restored_scaling_mode not in SCALING_MODE_OPTIONS:
                restored_scaling_mode = SCALING_MODE_BASIC
            self.scaling_mode_combo.setCurrentText(restored_scaling_mode)
            self._apply_scaling_mode_visibility(restored_scaling_mode)
            self.deinterlace_checkbox.setChecked(bool(raw.get("deinterlace_enabled", self.deinterlace_checkbox.isChecked())))
            self.reinterlace_checkbox.setChecked(bool(raw.get("reinterlace_enabled", self.reinterlace_checkbox.isChecked())))
            self.deinterlace_method_combo.setCurrentText(str(raw.get("deinterlace_method", self.deinterlace_method_combo.currentText())))
            self.denoise_method_combo.setCurrentText(str(raw.get("denoise_method", self.denoise_method_combo.currentText())))
            self.denoise_strength_spin.setValue(float(raw.get("denoise_strength", self.denoise_strength_spin.value())))
            graph_restored = self.effects_graph.restore(self.input_sources.migrate_graph(
                _effects_graph_from_settings(raw, self._settings_path)))
            if not graph_restored:
                legacy_method = DENOISE_METHOD_LABEL_TO_NAME.get(self.denoise_method_combo.currentText(), "off")
                self.effects_graph.migrate_legacy_denoise(legacy_method, float(self.denoise_strength_spin.value()))
            self.perf_guard_checkbox.setChecked(bool(raw.get("perf_guard_enabled", self.perf_guard_checkbox.isChecked())))

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

            self.rtx_vsr_quality_combo.setCurrentText(str(raw.get("rtx_vsr_quality", self.rtx_vsr_quality_combo.currentText())))
            self.rtx_vsr_scale_combo.setCurrentText(str(raw.get("rtx_vsr_scale", self.rtx_vsr_scale_combo.currentText())))
            self.rtx_vsr_post_scale_method_combo.setCurrentText(str(raw.get("rtx_vsr_post_scale_method", self.rtx_vsr_post_scale_method_combo.currentText())))
            self.rtx_thdr_enable_checkbox.setChecked(bool(raw.get("rtx_thdr_enabled", self.rtx_thdr_enable_checkbox.isChecked())))
            self.rtx_thdr_contrast_spin.setValue(int(raw.get("rtx_thdr_contrast", self.rtx_thdr_contrast_spin.value())))
            self.rtx_thdr_saturation_spin.setValue(int(raw.get("rtx_thdr_saturation", self.rtx_thdr_saturation_spin.value())))
            self.rtx_thdr_middle_gray_spin.setValue(int(raw.get("rtx_thdr_middle_gray", self.rtx_thdr_middle_gray_spin.value())))
            self.rtx_thdr_max_luminance_spin.setValue(int(raw.get("rtx_thdr_max_luminance", self.rtx_thdr_max_luminance_spin.value())))

            self.source_mode_combo.setCurrentText(str(raw.get("source_mode", self.source_mode_combo.currentText())))
            persisted_timecode_format = str(
                raw.get("decklink_timecode_format", self.decklink_timecode_format_combo.currentText())
            )
            if self.decklink_timecode_format_combo.findText(persisted_timecode_format) >= 0:
                self.decklink_timecode_format_combo.setCurrentText(persisted_timecode_format)
            persisted_phase_mode = _normalize_timecode_phase_synthesis_mode(
                str(raw.get("decklink_timecode_phase_mode", self.decklink_timecode_phase_combo.currentData()))
            )
            phase_index = self.decklink_timecode_phase_combo.findData(persisted_phase_mode)
            if phase_index >= 0:
                self.decklink_timecode_phase_combo.setCurrentIndex(phase_index)
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
            self._splitter_initialized = True

        main_sizes = raw.get("main_splitter_sizes")
        if isinstance(main_sizes, list) and len(main_sizes) >= 2:
            self._main_splitter.setSizes([int(main_sizes[0]), int(main_sizes[1])])
            self._main_splitter_initialized = True

        self._restore_roi_keyframes(raw.get("roi_keyframes"))
        self._restore_timecode_roi_keyframes(raw.get("roi_timecode_keyframes"))
        self._timecode_playback_enabled = bool(raw.get("roi_timecode_playback_enabled", True))
        self._set_roi_keyframing_mode(str(raw.get("roi_keyframing_mode", "manual")) == "timecode", save=False)
        self._update_roi_key_buttons()

        self._update_timer_interval()
        self._on_effects_graph_changed()

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
        scaling_mode = self.scaling_mode_combo.currentText()
        ai_enabled = scaling_mode == SCALING_MODE_ONNX_SR
        rtx_enabled = scaling_mode == SCALING_MODE_RTX_SR

        # Basic CUDA scaling is always kept enabled at the worker as the live
        # fallback layer beneath AI SR/RTX VSR.
        self._set_basic_scaling_enabled_effective(True)

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
                if bool(getattr(self._controller, "ai_sr_loading", False)):
                    # Engine build continues in the background; do not block/fail
                    # startup on it, the runtime panel updates once it is ready.
                    self._update_status(f"AI SR engine loading in background | model={model_path}")
                else:
                    ai_err = str(getattr(self._controller, "ai_sr_error", "AI SR did not become active")).strip()
                    if ai_err:
                        raise RuntimeError(ai_err)
                    raise RuntimeError("AI SR did not become active")

            LOGGER.info(
                "Applied startup AI SR settings: enabled=%s, model=%s, provider=%s, inference_fps=%s, loading=%s",
                ai_enabled,
                model_path,
                profile["provider"],
                profile["inference_fps"],
                bool(getattr(self._controller, "ai_sr_loading", False)),
            )
        except Exception as exc:
            LOGGER.warning("Failed to apply startup AI SR settings: %s", exc)
            self._update_status(f"Startup AI SR apply failed: {exc}")

        try:
            self._controller.set_rtx_vsr_enabled(rtx_enabled)
        except Exception as exc:
            LOGGER.warning("Failed to apply startup RTX VSR enabled state: %s", exc)

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

        self.basic_scaling_max_inflight_spin = QSpinBox()
        self.basic_scaling_max_inflight_spin.setRange(1, 4)
        self.basic_scaling_max_inflight_spin.setValue(int(getattr(self._controller, "basic_scaling_max_inflight", 1)))
        self.basic_scaling_max_inflight_spin.setToolTip(
            "Parallel basic-scaling worker pool depth (opt-in). Only engages for progressive output + "
            "Manual scaling + non-temporal-luma denoise; applies on next DeckLink Apply/start. "
            "Auto mode or interlaced output always run the single-worker path regardless of this value."
        )
        self.basic_scaling_max_inflight_spin.valueChanged.connect(self._on_basic_scaling_max_inflight_changed)

        if bool(getattr(self._controller, "ai_sr_enabled", False)):
            initial_scaling_mode = SCALING_MODE_ONNX_SR
        elif bool(getattr(self._controller, "rtx_vsr_enabled", False)):
            initial_scaling_mode = SCALING_MODE_RTX_SR
        else:
            initial_scaling_mode = SCALING_MODE_BASIC

        self.scaling_mode_combo = QComboBox()
        self.scaling_mode_combo.addItems(SCALING_MODE_OPTIONS)
        self.scaling_mode_combo.setCurrentText(initial_scaling_mode)
        self.scaling_mode_combo.currentTextChanged.connect(self._on_scaling_mode_changed)


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

        self.rtx_vsr_box = QGroupBox("RTX Video SDK (VSR)")
        rtx_vsr_form = QFormLayout(self.rtx_vsr_box)

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

        self.effects_graph = EffectsGraphEditor()
        self.effects_graph.set_frame_rate(float(self.fps_spin.value()))
        self.effects_graph.graphChanged.connect(self._on_effects_graph_changed)
        self.effects_graph.captureStartRequested.connect(self._start_capture_device_from_effect)
        self.effects_graph.captureSettingsRequested.connect(self._open_capture_device_settings)

        deinterlace_box = QGroupBox("De-interlacing")
        deinterlace_form = QFormLayout(deinterlace_box)
        deinterlace_form.addRow(self.deinterlace_checkbox)
        deinterlace_form.addRow(self.reinterlace_checkbox)
        deinterlace_form.addRow("Method", self.deinterlace_method_combo)

        upscaling_box = QGroupBox("Upscaling")
        self.upscaling_form = QFormLayout(upscaling_box)
        self.upscaling_form.addRow("Scaling mode", self.scaling_mode_combo)

        self._basic_scaling_mode_rows = [
            self.sr_mode_combo,
            self.sr_flavor_combo,
            self.sr_manual_combo,
            self.auto_sr_max_combo,
            self.basic_scaling_max_inflight_spin,
        ]
        self.upscaling_form.addRow("Basic scaling mode", self.sr_mode_combo)
        self.upscaling_form.addRow("Basic scaling method", self.sr_flavor_combo)
        self.upscaling_form.addRow("Manual basic scaling", self.sr_manual_combo)
        self.upscaling_form.addRow("Auto basic scaling max", self.auto_sr_max_combo)
        self.upscaling_form.addRow("Basic scaling max inflight", self.basic_scaling_max_inflight_spin)

        self._ai_sr_mode_rows = [
            self.ai_sr_model_combo,
            ai_sr_model_actions,
            self.ai_sr_provider_combo,
            self.ai_sr_trt_precision_combo,
            self.ai_sr_require_gpu_checkbox,
            self.ai_sr_frame_interval_spin,
            self.ai_sr_strict_checkbox,
            self.ai_sr_input_align_combo,
            self.ai_sr_overscan_spin,
            self.ai_sr_inference_divisor_spin,
            self.ai_sr_detail_preserve_spin,
            ai_sr_tuning_actions,
            ai_sr_runtime_box,
        ]
        self.upscaling_form.addRow("AI SR model", self.ai_sr_model_combo)
        self.upscaling_form.addRow(ai_sr_model_actions)
        self.upscaling_form.addRow("AI SR provider", self.ai_sr_provider_combo)
        self.upscaling_form.addRow("TensorRT precision", self.ai_sr_trt_precision_combo)
        self.upscaling_form.addRow(self.ai_sr_require_gpu_checkbox)
        self.upscaling_form.addRow("AI inference FPS", self.ai_sr_frame_interval_spin)
        self.upscaling_form.addRow(self.ai_sr_strict_checkbox)
        self.upscaling_form.addRow("AI SR input alignment", self.ai_sr_input_align_combo)
        self.upscaling_form.addRow("AI SR ROI overscan %", self.ai_sr_overscan_spin)
        self.upscaling_form.addRow("AI SR inference divisor", self.ai_sr_inference_divisor_spin)
        self.upscaling_form.addRow("AI SR detail preserve %", self.ai_sr_detail_preserve_spin)
        self.upscaling_form.addRow(ai_sr_tuning_actions)
        self.upscaling_form.addRow(ai_sr_runtime_box)

        self.ai_sr_postprocess_box = QGroupBox("AI SR Post Process Noise Reduction")
        ai_sr_postprocess_form = QFormLayout(self.ai_sr_postprocess_box)
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

        self.decklink_box = QGroupBox("Blackmagic I/O")
        decklink_form = QFormLayout(self.decklink_box)

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

        self.decklink_timecode_format_combo = QComboBox()
        for label, format_code in _decklink_timecode_format_options():
            self.decklink_timecode_format_combo.addItem(label, format_code)
        self.decklink_timecode_format_combo.setCurrentText("RP188 VITC1")
        self.decklink_timecode_format_combo.currentIndexChanged.connect(self._on_blackmagic_combo_changed)
        decklink_form.addRow("Fallback timecode type", self.decklink_timecode_format_combo)

        self.decklink_timecode_phase_combo = QComboBox()
        for label, mode_name in _timecode_phase_synthesis_options():
            self.decklink_timecode_phase_combo.addItem(label, mode_name)
        self.decklink_timecode_phase_combo.setCurrentIndex(0)
        self.decklink_timecode_phase_combo.setToolTip(
            "Use source-frame cadence to synthesize HFR phase when the capture device cannot provide HFRTC or field-mark metadata."
        )
        self.decklink_timecode_phase_combo.currentIndexChanged.connect(self._on_blackmagic_combo_changed)
        decklink_form.addRow("HFR timecode phase", self.decklink_timecode_phase_combo)

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
        self.decklink_output_buffer_spin.setRange(1, 10)
        self.decklink_output_buffer_spin.setValue(int(self._decklink_output_buffer_frames))
        self.decklink_output_buffer_spin.setToolTip(
            "Absolute software latency limit in output frames. N includes processing and scheduled output wait; "
            "the output queue never exceeds N frames. All settings use the newest capture and drop late updates. "
            "At 59.94p: 1 = 16.68 ms, 2 = 33.37 ms, 3 = 50.05 ms. "
            "Hardware/display delay is additional. A lower limit can cause late or dropped output; "
            "it does not guarantee a stable frame rate."
        )
        self.decklink_output_buffer_spin.valueChanged.connect(self._on_decklink_output_buffer_changed)
        decklink_form.addRow("Software latency limit (frames)", self.decklink_output_buffer_spin)

        self.decklink_fps_priority_guard_checkbox = QCheckBox(
            "Warn when output timing becomes unstable"
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

        self.status_dialog = QDialog(self, Qt.Tool)
        self.status_dialog.setWindowTitle("Runtime Status")
        self.status_dialog.resize(760, 360)
        status_dialog_layout = QVBoxLayout(self.status_dialog)
        self.status_text = QPlainTextEdit(self.status_dialog)
        self.status_text.setReadOnly(True)
        self.status_text.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        status_dialog_layout.addWidget(self.status_text)

        self.status_button = QPushButton("Open Status Window")
        self.status_button.setToolTip("Open detailed runtime and DeckLink status")
        self.status_button.clicked.connect(self._show_status_dialog)
        decklink_form.addRow(self.status_button)

        roi_box = QGroupBox("ROI")
        roi_form = QFormLayout(roi_box)

        reset_btn = QPushButton("Reset ROI")
        reset_btn.clicked.connect(self._reset_roi)
        roi_form.addRow(reset_btn)

        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, FRAME_W - 2)
        self.roi_x_spin.valueChanged.connect(self._on_roi_spin_changed)
        self.roi_x_spin.editingFinished.connect(self._on_roi_adjustment_finished)
        roi_form.addRow("x", self.roi_x_spin)

        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, FRAME_H - 2)
        self.roi_y_spin.valueChanged.connect(self._on_roi_spin_changed)
        self.roi_y_spin.editingFinished.connect(self._on_roi_adjustment_finished)
        roi_form.addRow("y", self.roi_y_spin)

        self.roi_w_spin = QSpinBox()
        self.roi_w_spin.setRange(2, FRAME_W)
        self.roi_w_spin.setSingleStep(2)
        self.roi_w_spin.valueChanged.connect(self._on_roi_spin_changed)
        self.roi_w_spin.editingFinished.connect(self._on_roi_adjustment_finished)
        roi_form.addRow("w", self.roi_w_spin)

        self.roi_h_spin = QSpinBox()
        self.roi_h_spin.setRange(2, FRAME_H)
        self.roi_h_spin.valueChanged.connect(self._on_roi_spin_changed)
        self.roi_h_spin.editingFinished.connect(self._on_roi_adjustment_finished)
        roi_form.addRow("h", self.roi_h_spin)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(1.0, 16.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setDecimals(2)
        self.scale_spin.valueChanged.connect(self._on_scale_spin_changed)
        self.scale_spin.editingFinished.connect(self._on_roi_adjustment_finished)
        roi_form.addRow("Scale", self.scale_spin)

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

        self.roi_manual_keyframe_widget = QWidget()
        manual_keyframe_layout = QVBoxLayout(self.roi_manual_keyframe_widget)
        manual_keyframe_layout.setContentsMargins(0, 0, 0, 0)
        manual_keyframe_layout.setSpacing(8)

        self.roi_timecode_mode_btn = QPushButton("Timecode Based Keyframing")
        self.roi_timecode_mode_btn.clicked.connect(lambda: self._set_roi_keyframing_mode(True))
        manual_keyframe_layout.addWidget(self.roi_timecode_mode_btn)

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

        manual_keyframe_layout.addWidget(keyframe_row)
        roi_form.addRow(self.roi_manual_keyframe_widget)

        self.roi_timecode_keyframe_widget = QWidget()
        timecode_keyframe_layout = QVBoxLayout(self.roi_timecode_keyframe_widget)
        timecode_keyframe_layout.setContentsMargins(0, 0, 0, 0)
        timecode_keyframe_layout.setSpacing(8)

        self.roi_manual_mode_btn = QPushButton("Manual Keyframing")
        self.roi_manual_mode_btn.clicked.connect(lambda: self._set_roi_keyframing_mode(False))
        timecode_keyframe_layout.addWidget(self.roi_manual_mode_btn)

        self.roi_timecode_playback_btn = QPushButton()
        self.roi_timecode_playback_btn.setCheckable(True)
        self.roi_timecode_playback_btn.toggled.connect(self._on_timecode_playback_toggled)
        timecode_keyframe_layout.addWidget(self.roi_timecode_playback_btn)
        self._update_timecode_playback_mode_control()

        timecode_edit_row = QWidget()
        timecode_edit_layout = QHBoxLayout(timecode_edit_row)
        timecode_edit_layout.setContentsMargins(0, 0, 0, 0)
        timecode_edit_layout.setSpacing(8)
        self.roi_timecode_add_btn = QPushButton("Add Keyframe")
        self.roi_timecode_add_btn.clicked.connect(self._on_timecode_add_keyframe)
        timecode_edit_layout.addWidget(self.roi_timecode_add_btn)
        self.roi_timecode_delete_btn = QPushButton("Delete Keyframe")
        self.roi_timecode_delete_btn.clicked.connect(self._on_timecode_delete_keyframe)
        timecode_edit_layout.addWidget(self.roi_timecode_delete_btn)
        self.roi_timecode_delete_all_btn = QPushButton("Delete All Keys")
        self.roi_timecode_delete_all_btn.clicked.connect(self._on_timecode_delete_all_keyframes)
        timecode_edit_layout.addWidget(self.roi_timecode_delete_all_btn)
        timecode_keyframe_layout.addWidget(timecode_edit_row)

        timecode_nav_row = QWidget()
        timecode_nav_layout = QHBoxLayout(timecode_nav_row)
        timecode_nav_layout.setContentsMargins(0, 0, 0, 0)
        timecode_nav_layout.setSpacing(8)
        self.roi_timecode_previous_btn = QPushButton("Previous Keyframe")
        self.roi_timecode_previous_btn.clicked.connect(lambda: self._navigate_timecode_keyframe(-1))
        timecode_nav_layout.addWidget(self.roi_timecode_previous_btn)
        self.roi_timecode_next_btn = QPushButton("Next Keyframe")
        self.roi_timecode_next_btn.clicked.connect(lambda: self._navigate_timecode_keyframe(1))
        timecode_nav_layout.addWidget(self.roi_timecode_next_btn)
        timecode_keyframe_layout.addWidget(timecode_nav_row)

        self.roi_timecode_key_label = QLabel("Key 0/0 :: --:--:--.--")
        self.roi_timecode_key_label.setAlignment(Qt.AlignCenter)
        timecode_keyframe_layout.addWidget(self.roi_timecode_key_label)
        roi_form.addRow(self.roi_timecode_keyframe_widget)
        self.roi_timecode_keyframe_widget.hide()

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

        self.controls_tabs = QTabWidget()
        self.settings_tabs = QTabWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.controls_tabs)

        def page(tabs, title, widgets):
            content = QWidget()
            box = QVBoxLayout(content)
            for widget in widgets:
                box.addWidget(widget)
            box.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(content)
            tabs.addTab(scroll, title)
            return content

        self.input_sources = InputSourcePanel(self)
        self.input_sources.changed.connect(self._input_sources_changed)
        page(self.controls_tabs, "ROI", [roi_box, controls_hint])
        page(self.controls_tabs, "EFFECTS", [self.effects_graph])
        self.controls_tabs.addTab(self.settings_tabs, "SETTINGS")
        page(self.settings_tabs, "INPUTS", [self.input_sources])

        # Keep the legacy binding widgets for the output pipeline, but expose
        # input configuration exclusively through the source catalog dialogs.
        self._input_bindings = QWidget(self)
        bindings_form = QFormLayout(self._input_bindings)
        for widget in (self.source_mode_combo, self.decklink_input_device_combo,
                       self.decklink_auto_detect_devices, self.decklink_input_mode_combo,
                       self.decklink_timecode_format_combo, self.decklink_timecode_phase_combo,
                       self.decklink_enable_format_detection, self.decklink_pixel_format_combo):
            row = decklink_form.takeRow(widget)
            if row.labelItem is not None:
                bindings_form.addRow(row.labelItem.widget(), row.fieldItem.widget())
            else:
                bindings_form.addRow(row.fieldItem.widget())
        self._input_bindings.hide()
        self.decklink_box.setTitle("Blackmagic Output")
        self.decklink_apply_btn.setText("Apply Output Settings")
        hdr_box = QGroupBox("HDR")
        hdr_form = QFormLayout(hdr_box)
        for widget in (self.rtx_thdr_enable_checkbox, self.rtx_thdr_contrast_spin,
                       self.rtx_thdr_saturation_spin, self.rtx_thdr_middle_gray_spin,
                       self.rtx_thdr_max_luminance_spin):
            row = rtx_vsr_form.takeRow(widget)
            if row.labelItem is not None:
                hdr_form.addRow(row.labelItem.widget(), row.fieldItem.widget())
            else:
                hdr_form.addRow(row.fieldItem.widget())
        hdr_apply = QPushButton("Apply HDR Settings")
        hdr_apply.clicked.connect(self._on_rtx_vsr_settings_apply_clicked)
        hdr_form.addRow(hdr_apply)
        page(self.settings_tabs, "OUTPUTS", [self.decklink_box, hdr_box])
        page(self.settings_tabs, "PREVIEW", [settings_box])
        page(self.settings_tabs, "SCALING", [deinterlace_box, upscaling_box,
             self.rtx_vsr_box, self.ai_sr_postprocess_box, post_vsr_scaling_box])
        self._apply_scaling_mode_visibility(self.scaling_mode_combo.currentText())
        return panel

    def _show_status_dialog(self) -> None:
        self.status_dialog.show()
        self.status_dialog.raise_()
        self.status_dialog.activateWindow()

    def _set_decklink_status(self, text: str) -> None:
        self._update_status(str(text))

    def _build_fullscreen_keyframe_toolbar(self, view_name: str) -> QWidget:
        toolbar = QWidget()
        toolbar_layout = QVBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(0)

        manual_row = QWidget()
        manual_layout = QHBoxLayout(manual_row)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(8)

        save_btn = QPushButton("SAVE KEY")
        save_btn.setCheckable(True)
        save_btn.setMinimumHeight(120)
        save_btn.toggled.connect(self._on_roi_save_key_toggled)
        manual_layout.addWidget(save_btn)

        key1_btn = QPushButton("KEY 1")
        key1_btn.setMinimumHeight(120)
        key1_btn.clicked.connect(lambda: self._on_roi_key_slot_pressed(1))
        manual_layout.addWidget(key1_btn)

        key2_btn = QPushButton("KEY 2")
        key2_btn.setMinimumHeight(120)
        key2_btn.clicked.connect(lambda: self._on_roi_key_slot_pressed(2))
        manual_layout.addWidget(key2_btn)

        key3_btn = QPushButton("KEY 3")
        key3_btn.setMinimumHeight(120)
        key3_btn.clicked.connect(lambda: self._on_roi_key_slot_pressed(3))
        manual_layout.addWidget(key3_btn)

        key4_btn = QPushButton("KEY 4")
        key4_btn.setMinimumHeight(120)
        key4_btn.clicked.connect(lambda: self._on_roi_key_slot_pressed(4))
        manual_layout.addWidget(key4_btn)

        timecode_row = QWidget()
        timecode_layout = QHBoxLayout(timecode_row)
        timecode_layout.setContentsMargins(0, 0, 0, 0)
        timecode_layout.setSpacing(8)

        add_btn = QPushButton("ADD\nKEYFRAME")
        add_btn.setMinimumHeight(120)
        add_btn.clicked.connect(self._on_timecode_add_keyframe)
        timecode_layout.addWidget(add_btn)

        delete_btn = QPushButton("DELETE\nKEYFRAME")
        delete_btn.setMinimumHeight(120)
        delete_btn.clicked.connect(self._on_timecode_delete_keyframe)
        timecode_layout.addWidget(delete_btn)

        delete_all_btn = QPushButton("DELETE\nALL KEYS")
        delete_all_btn.setMinimumHeight(120)
        delete_all_btn.clicked.connect(self._on_timecode_delete_all_keyframes)
        timecode_layout.addWidget(delete_all_btn)

        previous_btn = QPushButton("PREVIOUS\nKEYFRAME")
        previous_btn.setMinimumHeight(120)
        previous_btn.clicked.connect(lambda: self._navigate_timecode_keyframe(-1))
        timecode_layout.addWidget(previous_btn)

        next_btn = QPushButton("NEXT\nKEYFRAME")
        next_btn.setMinimumHeight(120)
        next_btn.clicked.connect(lambda: self._navigate_timecode_keyframe(1))
        timecode_layout.addWidget(next_btn)

        playback_btn = QPushButton()
        playback_btn.setCheckable(True)
        playback_btn.setMinimumHeight(120)
        playback_btn.toggled.connect(self._on_timecode_playback_toggled)
        timecode_layout.addWidget(playback_btn)

        toolbar_layout.addWidget(manual_row)
        toolbar_layout.addWidget(timecode_row)

        self._fullscreen_keyframe_toolbars[view_name] = toolbar
        self._fullscreen_manual_keyframe_rows[view_name] = manual_row
        self._fullscreen_timecode_keyframe_rows[view_name] = timecode_row
        self._fullscreen_timecode_delete_buttons[view_name] = delete_btn
        self._fullscreen_timecode_delete_all_buttons[view_name] = delete_all_btn
        self._fullscreen_timecode_previous_buttons[view_name] = previous_btn
        self._fullscreen_timecode_next_buttons[view_name] = next_btn
        self._fullscreen_timecode_playback_buttons[view_name] = playback_btn
        self._fullscreen_roi_save_key_buttons[view_name] = save_btn
        self._fullscreen_roi_key_slot_buttons[view_name] = (key1_btn, key2_btn, key3_btn, key4_btn)
        timecode_row.setVisible(False)
        self._update_timecode_playback_mode_control()
        toolbar.setVisible(False)
        return toolbar

    def _build_fullscreen_keyframe_side_panel(self, view_name: str) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(170)
        panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Ignored)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)

        title = QLabel("MANUAL KEYFRAME")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("QLabel { font-size: 15px; font-weight: 700; }")
        panel_layout.addWidget(title)

        exit_fullscreen_btn = QPushButton("EXIT\nFULL SCREEN")
        exit_fullscreen_btn.setMinimumWidth(150)
        exit_fullscreen_btn.setMinimumHeight(48)
        exit_fullscreen_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: 700; padding: 8px; }")
        exit_fullscreen_btn.clicked.connect(lambda: self._set_fullscreen_view(None))
        panel_layout.addWidget(exit_fullscreen_btn)

        mode_btn = QPushButton("TIMECODE BASED\nKEYFRAMING")
        mode_btn.setMinimumWidth(150)
        mode_btn.setMinimumHeight(48)
        mode_btn.setStyleSheet("QPushButton { font-size: 15px; font-weight: 700; padding: 8px; }")
        mode_btn.clicked.connect(self._toggle_roi_keyframing_mode)
        panel_layout.addWidget(mode_btn)

        timecode_display_label = QLabel(self._decklink_timecode_display_text)
        timecode_display_label.setAlignment(Qt.AlignCenter)
        timecode_display_label.setWordWrap(True)
        timecode_display_label.setMinimumWidth(150)
        timecode_display_label.setStyleSheet("QLabel { font-size: 14px; font-weight: 600; padding: 6px; }")
        panel_layout.addWidget(timecode_display_label)

        timecode_key_label = QLabel("Key 0/0 :: --:--:--.--")
        timecode_key_label.setAlignment(Qt.AlignCenter)
        timecode_key_label.setWordWrap(True)
        timecode_key_label.setMinimumWidth(150)
        timecode_key_label.setStyleSheet("QLabel { font-size: 14px; font-weight: 700; padding: 6px; }")
        panel_layout.addWidget(timecode_key_label)

        transition_label = QLabel("Transition\n(frames)")
        transition_label.setAlignment(Qt.AlignCenter)
        transition_label.setStyleSheet("QLabel { font-size: 14px; font-weight: 600; }")
        panel_layout.addWidget(transition_label)

        transition_spin = QSpinBox()
        transition_spin.setRange(1, 600)
        transition_spin.setValue(int(self._roi_keyframe_transition_default_frames))
        transition_spin.setMinimumWidth(150)
        transition_spin.setMinimumHeight(44)
        transition_spin.setStyleSheet(
            "QSpinBox { font-size: 22px; font-weight: 700; padding: 8px 12px; }"
            "QSpinBox::up-button, QSpinBox::down-button { width: 34px; }"
        )
        transition_spin.valueChanged.connect(self._on_fullscreen_transition_rate_changed)
        panel_layout.addWidget(transition_spin)

        interp_mode_combo = QComboBox()
        interp_mode_combo.addItems(["Linear", "Ease In/Out", "Ease Out"])
        interp_mode_combo.setCurrentText("Ease In/Out")
        interp_mode_combo.setMinimumWidth(150)
        interp_mode_combo.setMinimumHeight(44)
        interp_mode_combo.setStyleSheet("QComboBox { font-size: 16px; font-weight: 600; padding: 6px 8px; }")
        interp_mode_combo.setToolTip("Interpolation path used for ROI transitions.")
        interp_mode_combo.currentTextChanged.connect(self._on_fullscreen_interp_mode_changed)
        panel_layout.addWidget(interp_mode_combo)

        override_btn = QPushButton("OVERRIDE\nKEY DURATION")
        override_btn.setCheckable(True)
        override_btn.setMinimumWidth(150)
        override_btn.setMinimumHeight(48)
        override_btn.setToolTip("Use Transition (frames) as recall duration instead of the keyframe's stored duration.")
        override_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: 700; padding: 8px; }")
        override_btn.toggled.connect(self._on_fullscreen_override_duration_toggled)
        panel_layout.addWidget(override_btn)

        full_scale_btn = QPushButton("100%")
        full_scale_btn.setMinimumWidth(150)
        full_scale_btn.setMinimumHeight(44)
        full_scale_btn.setStyleSheet("QPushButton { font-size: 18px; font-weight: 700; padding: 8px; }")
        full_scale_btn.setToolTip("Interpolate the ROI to the full frame.")
        full_scale_btn.clicked.connect(self._interpolate_roi_to_full_frame)
        panel_layout.addWidget(full_scale_btn)

        scale_buttons: list[QPushButton] = []
        for index in range(3):
            scale_btn = QPushButton()
            scale_btn.setCheckable(True)
            scale_btn.setMinimumWidth(150)
            scale_btn.setMinimumHeight(44)
            scale_btn.setStyleSheet("QPushButton { font-size: 18px; font-weight: 700; padding: 8px; }")
            scale_btn.setToolTip("Select the scale used when tapping the preview. Right-click to change this percentage.")
            scale_btn.clicked.connect(lambda checked, preset_index=index: self._on_fullscreen_scale_toggled(preset_index, checked))
            scale_btn.setContextMenuPolicy(Qt.CustomContextMenu)
            scale_btn.customContextMenuRequested.connect(
                lambda _position, preset_index=index: self._edit_fullscreen_scale_preset(preset_index)
            )
            panel_layout.addWidget(scale_btn)
            scale_buttons.append(scale_btn)

        panel_layout.addStretch(1)

        self._fullscreen_keyframe_side_panels[view_name] = panel
        self._fullscreen_keyframe_title_labels[view_name] = title
        self._fullscreen_keyframing_mode_buttons[view_name] = mode_btn
        self._fullscreen_timecode_display_labels[view_name] = timecode_display_label
        self._fullscreen_timecode_key_labels[view_name] = timecode_key_label
        self._fullscreen_roi_transition_labels[view_name] = transition_label
        self._fullscreen_roi_transition_rate_spins[view_name] = transition_spin
        self._fullscreen_roi_interp_mode_combos[view_name] = interp_mode_combo
        self._fullscreen_roi_duration_override_buttons[view_name] = override_btn
        self._fullscreen_scale_buttons[view_name] = tuple(scale_buttons)
        self._sync_fullscreen_scale_buttons()
        timecode_display_label.setVisible(False)
        timecode_key_label.setVisible(False)
        panel.setVisible(False)
        return panel

    def _toggle_roi_keyframing_mode(self) -> None:
        self._set_roi_keyframing_mode(not self._timecode_keyframing_enabled)

    def _sync_fullscreen_keyframing_mode(self) -> None:
        timecode_enabled = bool(self._timecode_keyframing_enabled)
        for label in self._fullscreen_keyframe_title_labels.values():
            label.setText("TIMECODE KEYFRAME" if timecode_enabled else "MANUAL KEYFRAME")
        for button in self._fullscreen_keyframing_mode_buttons.values():
            button.setText("MANUAL\nKEYFRAMING" if timecode_enabled else "TIMECODE BASED\nKEYFRAMING")
        for row in self._fullscreen_manual_keyframe_rows.values():
            row.setVisible(not timecode_enabled)
        for row in self._fullscreen_timecode_keyframe_rows.values():
            row.setVisible(timecode_enabled)
        for label in self._fullscreen_timecode_display_labels.values():
            label.setVisible(timecode_enabled)
        for label in self._fullscreen_timecode_key_labels.values():
            label.setVisible(timecode_enabled)
        for button in self._fullscreen_roi_duration_override_buttons.values():
            button.setVisible(not timecode_enabled)
        QTimer.singleShot(0, self._fit_viewers_to_video_aspect)

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

    def _on_fullscreen_interp_mode_changed(self, text: str) -> None:
        if self.roi_interp_mode_combo.currentText() != text:
            self.roi_interp_mode_combo.setCurrentText(text)
        else:
            self._sync_fullscreen_interp_mode_from_main(text)

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

    def _sync_fullscreen_interp_mode_from_main(self, text: str) -> None:
        for combo in self._fullscreen_roi_interp_mode_combos.values():
            previous_block = combo.blockSignals(True)
            combo.setCurrentText(text)
            combo.blockSignals(previous_block)

    def _sync_fullscreen_override_duration_from_main(self, checked: bool) -> None:
        target = bool(checked)
        for button in self._fullscreen_roi_duration_override_buttons.values():
            previous_block = button.blockSignals(True)
            button.setChecked(target)
            button.blockSignals(previous_block)

    def _sync_fullscreen_scale_buttons(self) -> None:
        for buttons in self._fullscreen_scale_buttons.values():
            for index, button in enumerate(buttons):
                previous_block = button.blockSignals(True)
                button.setText(f"{self._fullscreen_scale_presets[index]}%")
                button.setChecked(self._fullscreen_selected_scale_index == index)
                button.blockSignals(previous_block)

    def _on_fullscreen_scale_toggled(self, preset_index: int, checked: bool) -> None:
        index = max(0, min(2, int(preset_index)))
        if checked:
            self._fullscreen_selected_scale_index = index
        elif self._fullscreen_selected_scale_index == index:
            self._fullscreen_selected_scale_index = None
        self._sync_fullscreen_scale_buttons()
        self._schedule_settings_save()

    def _edit_fullscreen_scale_preset(self, preset_index: int) -> None:
        index = max(0, min(2, int(preset_index)))
        value, accepted = QInputDialog.getInt(
            self,
            "Set preview scale",
            "Scale percentage:",
            int(self._fullscreen_scale_presets[index]),
            100,
            1000000,
            1,
        )
        if not accepted:
            return
        self._fullscreen_scale_presets[index] = int(value)
        self._sync_fullscreen_scale_buttons()
        self._schedule_settings_save()

    def _interpolate_roi_to_full_frame(self) -> None:
        target = Roi(0, 0, FRAME_W, FRAME_H)
        duration_frames = max(1, min(600, int(self.roi_transition_frames_spin.value())))
        self._start_roi_keyframe_transition(
            target,
            self._effective_roi_keyframe_duration_frames(target, duration_frames),
            self._roi_interp_mode_name(),
            manual_adjustment=True,
        )

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
        self._windowed_display_splitter_sizes = list(self._display_splitter.sizes())
        self._windowed_main_splitter_sizes = list(self._main_splitter.sizes())
        self._windowed_was_maximized_before_fullscreen = self.isMaximized()
        self._windowed_qt_geometry_before_fullscreen = QByteArray(self.saveGeometry())
        geometry = self.normalGeometry() if self._windowed_was_maximized_before_fullscreen else self.geometry()
        if not geometry.isValid():
            geometry = self.geometry()
        self._windowed_geometry_before_fullscreen = QRect(geometry)

        screen = self.screen()
        self._windowed_available_geometry_before_fullscreen = (
            QRect(screen.availableGeometry()) if screen is not None else None
        )

    def _clamp_windowed_geometry_to_screen(self, geometry: QRect) -> QRect:
        available = self._windowed_available_geometry_before_fullscreen
        if available is None or not available.isValid():
            screen = self.screen() or QApplication.primaryScreen()
            available = QRect(screen.availableGeometry()) if screen is not None else QRect()
        if not available.isValid():
            return QRect(geometry)

        width = max(1, min(geometry.width(), available.width()))
        height = max(1, min(geometry.height(), available.height()))
        x = max(available.left(), min(geometry.x(), available.right() - width + 1))
        y = max(available.top(), min(geometry.y(), available.bottom() - height + 1))
        return QRect(x, y, width, height)

    def _restore_windowed_geometry_after_fullscreen(self) -> None:
        qt_geometry = QByteArray(self._windowed_qt_geometry_before_fullscreen or QByteArray())
        geometry = QRect(self._windowed_geometry_before_fullscreen) if self._windowed_geometry_before_fullscreen is not None else None
        saved_available_geometry = (
            QRect(self._windowed_available_geometry_before_fullscreen)
            if self._windowed_available_geometry_before_fullscreen is not None
            else None
        )
        was_maximized = self._windowed_was_maximized_before_fullscreen

        self.setWindowState(self.windowState() & ~Qt.WindowFullScreen)
        self.showNormal()

        def finish_restore() -> None:
            if self._fullscreen_view_name is not None:
                return

            restored = bool(qt_geometry) and self.restoreGeometry(qt_geometry)
            if was_maximized:
                self.showMaximized()
            else:
                self.setWindowState(Qt.WindowNoState)
                if not restored and geometry is not None and geometry.isValid():
                    self.setGeometry(self._clamp_windowed_geometry_to_screen(geometry))
                self.show()

                original_screen_available = bool(
                    saved_available_geometry is not None
                    and saved_available_geometry.isValid()
                    and any(
                        QRect(screen.availableGeometry()) == saved_available_geometry
                        for screen in QApplication.screens()
                    )
                )
                if geometry is not None and geometry.isValid() and original_screen_available:
                    expected_geometry = QRect(geometry)

                    def enforce_saved_geometry() -> None:
                        if self._fullscreen_view_name is None and not self.isMaximized():
                            self.setGeometry(expected_geometry)
                            self._restore_windowed_splitter_layout()

                    QTimer.singleShot(0, enforce_saved_geometry)
            self._restore_windowed_splitter_layout()

        QTimer.singleShot(0, finish_restore)

    def _restore_windowed_splitter_layout(self) -> None:
        if self._fullscreen_view_name is not None:
            return
        if self._windowed_display_splitter_sizes:
            self._display_splitter.setSizes(self._windowed_display_splitter_sizes)
            self._splitter_initialized = True
        if self._windowed_main_splitter_sizes:
            self._main_splitter.setSizes(self._windowed_main_splitter_sizes)
            self._main_splitter_initialized = True
        self._fit_viewers_to_video_aspect()

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
                        if getattr(self._controller, "ai_sr_active", False):
                            ai_sr_state = "active"
                        elif getattr(self._controller, "ai_sr_loading", False):
                            ai_sr_state = "loading"
                        else:
                            ai_sr_state = "requested"
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
                    self._set_decklink_status(
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
                    if getattr(self._controller, "ai_sr_active", False):
                        ai_sr_state = "active"
                    elif getattr(self._controller, "ai_sr_loading", False):
                        ai_sr_state = "loading"
                    else:
                        ai_sr_state = "requested"
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
                    self._set_decklink_status("DeckLink streaming")

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
        self._settings_save_timer.stop()
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

        if avail_w <= 10 or avail_h <= 10:
            return

        target_w = avail_w
        target_h = int(round(target_w * 9.0 / 16.0))
        if target_h > avail_h:
            target_h = avail_h
            target_w = int(round(target_h * 16.0 / 9.0))

        target_w = max(1, min(target_w, avail_w))
        target_h = max(1, min(target_h, avail_h))

        if canvas.maximumWidth() != target_w or canvas.maximumHeight() != target_h:
            canvas.setMaximumSize(target_w, target_h)

        set_preferred_size = getattr(canvas, "set_preferred_canvas_size", None)
        if callable(set_preferred_size):
            set_preferred_size(target_w, target_h)
        else:
            canvas.updateGeometry()

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
            QTimer.singleShot(0, self._restore_windowed_splitter_layout)
            return

        self._controls_scroll.setVisible(False)
        self._input_panel.setVisible(view_name == "input")
        self._output_panel.setVisible(view_name == "output")
        self._sync_fullscreen_keyframing_mode()
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
        if hasattr(self, "effects_graph"):
            self.effects_graph.set_frame_rate(float(fps))
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
        self._update_status(f"Software latency limit set to {buffer_frames} frame(s); applying")

    def _on_worker_priority_changed(self) -> None:
        if self._updating_controls:
            return
        self._worker_process_priority = _normalize_worker_priority_name(
            WORKER_PRIORITY_LABEL_TO_NAME.get(self.worker_priority_combo.currentText(), "above_normal")
        )
        self._apply_worker_process_priority_to_controller(notify=True)

    def _on_decklink_buffer_guard_toggled(self, checked: bool) -> None:
        self._decklink_buffer_guard_enabled = bool(checked)
        self._decklink_buffer_guard_active = False
        self._decklink_buffer_guard_stable_windows = 0
        self._update_status(
            "DeckLink timing warnings enabled; output buffer remains user-selected"
            if self._decklink_buffer_guard_enabled
            else "DeckLink timing warnings disabled; output buffer remains user-selected"
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
            f"Software latency limit applied: {int(self._decklink_output_buffer_frames)} frame(s)"
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

        engage = (
            float(deadline_miss_ratio) >= float(self._decklink_buffer_guard_engage_miss_ratio)
            or int(deadline_miss_streak) >= 2
            or int(starvation_delta) > 0
            or int(buffered_count) == 0
        )

        if engage:
            self._decklink_buffer_guard_stable_windows = 0
            if not self._decklink_buffer_guard_active:
                self._decklink_buffer_guard_active = True
                self._update_status(
                    (
                        "DeckLink timing warning: buffer remains at requested "
                        f"{requested_frames} frame(s) "
                        f"(dl_miss={deadline_miss_ratio * 100.0:.1f}%, streak={deadline_miss_streak}, "
                        f"interaction={'yes' if interaction_active else 'no'})"
                    )
                )
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
        self._update_status(
            f"DeckLink timing stable; output buffer remains at requested {requested_frames} frame(s)"
        )

    def _on_manual_drag_endpoint(self, x: float, y: float, w: float, h: float) -> None:
        if (
            self._controller_backend != "worker-process"
            or not self._input_canvas.is_move_drag_active()
            or not hasattr(self._controller, "publish_manual_roi_endpoint")
        ):
            return

        overlay = (float(x), float(y), float(w), float(h))
        step_roi, step_shift_x, step_shift_y = self._manual_roi_step_with_subpixel_float_target(overlay)

        self._on_roi_adjustment_started()
        if self._roi_keyframe_transition is not None:
            transition_state = self._roi_keyframe_transition
            if isinstance(transition_state, dict):
                current_estimate = transition_state.get("current_roi_estimate")
                if isinstance(current_estimate, Roi):
                    self._controller_roi_applied = clamp_roi(current_estimate)
            self._cancel_roi_keyframe_transition()

        self._last_manual_roi_update_ts = time.perf_counter()
        started = time.perf_counter()
        self._roi_diag_controller_send_attempts += 1
        sent = bool(
            self._controller.publish_manual_roi_endpoint(
                step_roi,
                step_shift_x,
                step_shift_y,
                suspend_timecode=self._timecode_adjustment_paused,
                motion_input=self._input_canvas.pointer_input_diagnostics(),
            )
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._roi_diag_controller_send_ms_sum += elapsed_ms
        self._roi_diag_controller_send_ms_max = max(self._roi_diag_controller_send_ms_max, elapsed_ms)
        if sent:
            self._roi_diag_controller_send_success += 1
            self._manual_roi_last_send_ts = time.perf_counter()
            self._manual_live_target_roi = None
            self._pending_manual_controller_roi = None
            self._manual_roi_send_timer.stop()
        else:
            self._roi_diag_controller_send_drops += 1

    def _on_roi_from_canvas(self, x: int, y: int, w: int, h: int) -> None:
        self._on_roi_adjustment_started()
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

        self._controller_roi_target = None
        self._controller_filtered_target_roi = None
        self._controller_roi_interp_timer.stop()

        drag_overlay = self._input_canvas.drag_visual_roi_overlay()
        pinch_overlay = self._input_canvas.pinch_visual_roi_overlay()
        if self._manual_live_target_roi is None:
            self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        if pinch_overlay is not None:
            live_target = clamp_roi(Roi(*(int(round(value)) for value in pinch_overlay)))
        elif drag_overlay is not None:
            live_target = clamp_roi(Roi(*(int(round(value)) for value in drag_overlay)))
        else:
            live_target = self._roi
        self._manual_live_target_roi = live_target

        mailbox_drag = bool(
            drag_overlay is not None
            and self._input_canvas.is_move_drag_active()
            and self._controller_backend == "worker-process"
            and hasattr(self._controller, "publish_manual_roi_endpoint")
        )
        if mailbox_drag:
            self._manual_live_target_roi = None
            self._pending_manual_controller_roi = None
            self._manual_roi_send_timer.stop()
            self._schedule_roi_controls_sync(self._roi)
            return
        elif hasattr(self._controller, "clear_manual_roi_endpoint"):
            self._controller.clear_manual_roi_endpoint()

        self._pending_manual_controller_roi = live_target
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

        drag_overlay = self._input_canvas.drag_visual_roi_overlay()
        if use_subpixel_microstep and moving_only and drag_overlay is not None:
            step_roi, step_shift_x, step_shift_y = self._manual_roi_step_with_subpixel_float_target(
                drag_overlay,
            )
        single_bucket_drag = bool(use_subpixel_microstep and moving_only and drag_overlay is not None)

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
                        suspend_timecode=self._timecode_adjustment_paused,
                        motion_input=self._input_canvas.pointer_input_diagnostics(),
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
                self._controller_roi_applied = step_roi
            else:
                self._roi_diag_controller_send_drops += 1
        except Exception as exc:
            self._roi_diag_controller_send_drops += 1
            self._update_status(f"ROI update failed: {exc}")

        # Translation drags retire after one successful bucket send. Resize and
        # control changes continue stepping until their target is reached.
        manual_drag_active = self._input_canvas.drag_visual_roi_overlay() is not None
        if sent and single_bucket_drag:
            self._manual_live_target_roi = None
        elif sent and self._is_controller_roi_close(self._controller_roi_applied, target) and not manual_drag_active:
            self._manual_live_target_roi = None
            self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        if self._pending_manual_controller_roi is not None or self._manual_live_target_roi is not None:
            interval_ms = self._manual_roi_send_interval_ms(target_scale, moving_only=moving_only)
            if not sent and self._controller_backend == "worker-process":
                # Back off on queue pressure; latest-wins ROI compaction keeps motion responsive.
                interval_ms = min(24, interval_ms + 4)
            self._manual_roi_send_timer.setInterval(interval_ms)
            self._manual_roi_send_timer.start()
        elif self._timecode_adjustment_finish_pending:
            self._complete_timecode_adjustment()

    def _manual_roi_send_interval_ms(self, zoom_scale: float, moving_only: bool = False) -> int:
        z = max(1.0, float(zoom_scale))
        if z >= 6.0:
            base = 8
        elif z >= 4.0:
            base = 10
        else:
            base = 16

        if moving_only and self._controller_backend == "worker-process":
            base = int(self._manual_drag_worker_send_interval_ms)

        field_interval_ms = self._decklink_output_field_interval_ms()
        if field_interval_ms is not None:
            # Keep manual control updates at least as responsive as field cadence.
            # Slower-than-field pacing makes drag appear steppy at 1080i rates.
            return min(base, int(field_interval_ms))
        return base

    def _manual_roi_render_gate_interval_ms(self) -> int | None:
        if not bool(self._manual_roi_frame_lock_to_output):
            return None
        if self._source_mode != "Blackmagic DeckLink":
            return None
        if self._controller_backend != "worker-process":
            return None
        return self._decklink_output_field_interval_ms()

    def _manual_roi_step_with_subpixel_float_target(
        self,
        target_overlay: tuple[float, float, float, float],
    ) -> tuple[Roi, float, float]:
        target_x, target_y, target_w, target_h = [float(v) for v in target_overlay]

        target_cx = target_x + (target_w * 0.5)
        target_cy = target_y + (target_h * 0.5)
        desired_cx = target_cx
        desired_cy = target_cy
        desired_w = target_w

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

    def _manual_roi_step_with_subpixel(self, current: Roi, target: Roi) -> tuple[Roi, float, float]:
        moving_only = current.w == target.w and current.h == target.h
        zoom_scale = roi_scale_from_roi(target)

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
        backend_transition_active = isinstance(state, dict) and bool(state.get("backend_driven", False))
        if not backend_transition_active:
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
        if backend_transition_active:
            state["current_roi_estimate"] = worker_roi
            # Never snap canvas ROI directly during a backend-driven transition.
            # We render a smoothed visual overlay and commit once complete.
            self._roi = worker_roi
        else:
            self._roi = worker_roi
            self._input_canvas._apply_roi_local(worker_roi)
        self._controller_roi_applied = worker_roi
        self._schedule_roi_controls_sync(worker_roi)

        # Render GUI interpolation from worker transition phase so the on-screen
        # ROI appears smooth between quantized applied-ROI steps.
        if backend_transition_active and transition_active:
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
        elif backend_transition_active:
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
        anchor_roi = clamp_roi(self._controller_roi_applied)

        if anchor_to_current:
            self._controller_filtered_target_roi = anchor_roi
            self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}

        self._controller_filtered_target_roi = raw_target
        self._controller_roi_target = raw_target
        if not self._controller_roi_interp_timer.isActive():
            self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        target_scale = roi_scale_from_roi(self._controller_roi_target)
        if target_scale >= 6.0:
            interval_ms = 8
        elif target_scale >= 4.0:
            interval_ms = 10
        else:
            interval_ms = 16
        field_interval_ms = self._decklink_output_field_interval_ms()
        if field_interval_ms is not None:
            interval_ms = min(interval_ms, int(field_interval_ms))
        self._controller_roi_interp_timer.setInterval(interval_ms)
        if not self._controller_roi_interp_timer.isActive():
            self._controller_roi_interp_timer.start()

    def _apply_controller_roi_immediate(
        self,
        roi: Roi,
        reset_subpixel_shift: bool = True,
        settle_interlaced: bool = False,
    ) -> None:
        clamped = clamp_roi(roi)
        self._controller_roi_target = None
        self._controller_roi_interp_timer.stop()
        self._controller_filtered_target_roi = None
        self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        if settle_interlaced and hasattr(self._controller, "set_roi_settled"):
            self._controller.set_roi_settled(clamped)
            self._controller_roi_applied = clamped
            return
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

        if zoom_scale >= 6.0:
            lag_limit = 8
        elif zoom_scale >= 4.0:
            lag_limit = 12
        else:
            lag_limit = 16

        near_target_deadband = 1

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

    def _on_roi_drag_x_hysteresis_changed(self, value: float) -> None:
        clamped = max(0.10, min(1.20, float(value)))
        self._roi_drag_x_hysteresis_px = clamped
        self._input_canvas.set_drag_x_hysteresis_px(clamped)

    def _on_roi_manual_drag_hold_changed(self, value: float) -> None:
        clamped = max(0.05, min(0.50, float(value)))
        self._roi_manual_drag_hold_s = clamped
        self._apply_manual_drag_tuning_to_controller()

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

        self._on_roi_adjustment_started()
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
        if self._timecode_adjustment_paused:
            self._roi_control_adjustment_timer.start()

    def _on_scale_spin_changed(self, value: float) -> None:
        if self._updating_controls:
            return

        self._on_roi_adjustment_started()
        self._cancel_roi_keyframe_transition()

        center_x = self._roi.x + (self._roi.w / 2.0)
        center_y = self._roi.y + (self._roi.h / 2.0)
        roi = roi_from_scale(value, center_x, center_y)
        self._roi = roi
        self._input_canvas.set_roi(roi)
        self._apply_controller_roi_immediate(roi)
        self._sync_controls_from_roi(roi)
        if self._timecode_adjustment_paused:
            self._roi_control_adjustment_timer.start()

    def _apply_scaling_mode_visibility(self, mode_text: str) -> None:
        is_basic = mode_text == SCALING_MODE_BASIC
        is_ai = mode_text == SCALING_MODE_ONNX_SR
        is_rtx = mode_text == SCALING_MODE_RTX_SR

        for widget in self._basic_scaling_mode_rows:
            self.upscaling_form.setRowVisible(widget, is_basic)
        for widget in self._ai_sr_mode_rows:
            self.upscaling_form.setRowVisible(widget, is_ai)
        self.ai_sr_postprocess_box.setVisible(is_ai)
        self.rtx_vsr_box.setVisible(is_rtx)

    def _set_basic_scaling_enabled_effective(self, enabled: bool) -> None:
        # Worker backend: instant runtime toggle, no processor/session recreation.
        if self._controller_backend == "worker-process" and hasattr(self._controller, "set_basic_scaling_enabled"):
            self._controller.enable_basic_scaling = bool(enabled)
            self._controller.set_basic_scaling_enabled(enabled)
            return

        # In-process backend: basic scaling is a constructor-only native option.
        previous_value = self._controller.enable_basic_scaling
        self._controller.enable_basic_scaling = bool(enabled)
        try:
            self._controller.create(self._roi)
            if self._source_mode == "Blackmagic DeckLink":
                self._start_decklink_sessions()
        except Exception as exc:
            self._controller.enable_basic_scaling = previous_value
            try:
                self._controller.create(self._roi)
                if self._source_mode == "Blackmagic DeckLink":
                    self._start_decklink_sessions()
            except Exception:
                pass
            self._update_status(f"Processor recreate failed: {exc}")

    def _apply_scaling_mode_runtime(self, mode_text: str) -> None:
        want_ai = mode_text == SCALING_MODE_ONNX_SR
        want_rtx = mode_text == SCALING_MODE_RTX_SR

        # Basic CUDA scaling stays enabled at the worker as the live fallback
        # layer (shown until AI SR/RTX VSR actually becomes active), so
        # switching modes never blacks out or requires a processor restart.
        self._set_basic_scaling_enabled_effective(True)

        try:
            if not want_ai and bool(getattr(self._controller, "ai_sr_enabled", False)):
                self._controller.set_ai_sr_enabled(False)
            if not want_rtx and bool(getattr(self._controller, "rtx_vsr_enabled", False)):
                self._controller.set_rtx_vsr_enabled(False)
            if want_ai:
                self._controller.set_ai_sr_enabled(True)
                model_path = self.ai_sr_model_combo.currentText().strip()
                self._update_status(f"ONNX SR mode selected | awaiting worker ack | model={model_path}")
            elif want_rtx:
                self._controller.set_rtx_vsr_enabled(True)
                self._update_status("Nvidia SR (RTX VSR) mode selected | awaiting worker ack")
            else:
                self._update_status("Standard CUDA scaling mode selected")
        except Exception as exc:
            self._update_status(f"Scaling mode change failed: {exc}")

    def _on_scaling_mode_changed(self, mode_text: str) -> None:
        self._apply_scaling_mode_visibility(mode_text)
        if self._updating_controls:
            return
        self._apply_scaling_mode_runtime(mode_text)

    def _on_sr_mode_changed(self) -> None:
        if self.scaling_mode_combo.currentText() != SCALING_MODE_BASIC:
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

    def _on_basic_scaling_max_inflight_changed(self, value: int) -> None:
        if self._updating_controls:
            return
        try:
            self._controller.set_basic_scaling_max_inflight(int(value))
            self._update_status(
                f"Basic scaling max inflight set to {int(value)} | applies on next DeckLink Apply/start "
                "(only engages for progressive + Manual scaling + non-temporal denoise)"
            )
        except Exception as exc:
            self._update_status(f"Basic scaling max inflight change failed: {exc}")

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

    def _on_effects_graph_changed(self) -> None:
        if self._updating_controls:
            return
        method_name, strength = self.effects_graph.active_denoise_settings()
        method_label = DENOISE_METHOD_NAME_TO_LABEL.get(method_name, "Off")
        self.denoise_method_combo.blockSignals(True)
        self.denoise_strength_spin.blockSignals(True)
        self.denoise_method_combo.setCurrentText(method_label)
        self.denoise_strength_spin.setValue(strength)
        self.denoise_method_combo.blockSignals(False)
        self.denoise_strength_spin.blockSignals(False)
        if (
            method_name != str(getattr(self._controller, "denoise_method", "off"))
            or abs(strength - float(getattr(self._controller, "denoise_strength", 0.0))) > 1e-6
        ):
            self._on_denoise_settings_changed()
        capture_device = self.effects_graph.active_capture_device()
        if capture_device is not None:
            self.decklink_auto_detect_devices.blockSignals(True)
            self.decklink_auto_detect_devices.setChecked(False)
            self.decklink_auto_detect_devices.blockSignals(False)
            for index in range(self.decklink_input_device_combo.count()):
                if self.decklink_input_device_combo.itemData(index) == capture_device:
                    self.decklink_input_device_combo.blockSignals(True)
                    self.decklink_input_device_combo.setCurrentIndex(index)
                    self.decklink_input_device_combo.blockSignals(False)
                    self._populate_mode_combos()
                    break
        effects_payload = self.effects_graph.native_effects_payload()
        effects_error = str(effects_payload.get("error", "")).strip()
        if effects_error:
            self._update_status(effects_error)
        elif effects_payload != self._last_effects_payload:
            try:
                self._controller.set_effects_config(effects_payload)
                self._last_effects_payload = dict(effects_payload)
                state = "enabled" if bool(effects_payload.get("enabled", False)) else "disabled"
                self._update_status(f"GPU effects compositor {state}")
            except Exception as exc:
                self._update_status(f"Effects graph update failed: {exc}")
        self._schedule_settings_save()

    def _select_effect_capture_device(self, capture_device: object) -> bool:
        if isinstance(capture_device, str) and capture_device.startswith("webcam:"):
            self._update_status("Windows camera capture cannot drive the DeckLink input session")
            return False
        self.decklink_auto_detect_devices.blockSignals(True)
        self.decklink_auto_detect_devices.setChecked(False)
        self.decklink_auto_detect_devices.blockSignals(False)
        for index in range(self.decklink_input_device_combo.count()):
            if self.decklink_input_device_combo.itemData(index) == capture_device:
                self.decklink_input_device_combo.blockSignals(True)
                self.decklink_input_device_combo.setCurrentIndex(index)
                self.decklink_input_device_combo.blockSignals(False)
                self._populate_mode_combos()
                return True
        self._refresh_decklink_catalog()
        for index in range(self.decklink_input_device_combo.count()):
            if self.decklink_input_device_combo.itemData(index) == capture_device:
                self.decklink_input_device_combo.setCurrentIndex(index)
                return True
        self._update_status(f"Capture device {capture_device!r} is not available")
        return False

    def _start_capture_device_from_effect(self, capture_device: object) -> None:
        self._update_status("Activate inputs in Settings > Inputs")

    def _open_capture_device_settings(self, capture_device: object) -> None:
        self.controls_tabs.setCurrentIndex(2)
        self.settings_tabs.setCurrentIndex(0)
        if isinstance(capture_device, str) and capture_device.startswith("source:"):
            row = int(capture_device.split(":", 1)[1]) - 1
            self.input_sources.table.selectRow(row)

    def _set_input_source_catalog(self, devices):
        normalized = [(label, f"decklink:{device}" if isinstance(device, int) else device)
                      for label, device in devices]
        self.input_sources.set_devices(normalized)

    def _input_sources_changed(self):
        self.effects_graph.set_capture_devices(self.input_sources.labels())
        roi_config = self.input_sources.configs[0]
        roi_settings = tuple(roi_config.get(key) for key in ("device", "mode", "mode_text", "timecode_format", "timecode_phase"))
        if roi_settings != getattr(self, "_last_roi_source_settings", None):
            self._last_roi_source_settings = roi_settings
            for combo, key in ((self.decklink_timecode_format_combo, "timecode_format"),
                               (self.decklink_timecode_phase_combo, "timecode_phase")):
                index = combo.findData(roi_config.get(key))
                if index >= 0:
                    combo.setCurrentIndex(index)
            self._apply_mode_aware_deinterlace_default_if_needed()
            self._sync_roi_transition_unit_labels()
        active = frozenset(self.input_sources.active)
        if active != getattr(self, "_last_active_input_sources", frozenset()):
            self._last_active_input_sources = active
            canvas = self.effects_graph.canvas
            for node_id, node in canvas._nodes.items():
                if node.get("type") == "capture":
                    canvas._capture_reload_tokens[node_id] = canvas._capture_reload_tokens.get(node_id, 0) + 1
            self._on_effects_graph_changed()
        if hasattr(self, "_settings_save_timer"):
            self._schedule_settings_save()

    def _input_source_modes(self, device):
        return [(m.name, m.mode) for m in _call_decklink_api(
            "list_input_display_modes", int(device.split(":", 1)[1]))]

    def _activate_input_source(self, logical_id, config):
        self._controller.activate_source(logical_id, config)
        self._update_status(f"Source {logical_id} active")

    def _deactivate_input_source(self, logical_id):
        self._controller.deactivate_source(logical_id)
        self._update_status(f"Source {logical_id} inactive")

    def _on_perf_guard_toggled(self, checked: bool) -> None:
        self._perf_guard_enabled = checked
        self._perf_guard_low_fps_seconds = 0
        self._perf_guard_last_action = ""

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
        loading = bool(getattr(self._controller, "ai_sr_loading", False))
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
        provider_fallback_note = str(info.get("provider_fallback_note", "") or "").strip()

        lines = [
            f"Enabled: {enabled} | Active: {active} | Loading: {loading}",
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

        if enabled and loading and not active:
            lines.append(
                "AI SR engine is loading in the background (ONNX Runtime/TensorRT session build); "
                "the current mode keeps rendering live until it is ready."
            )

        if provider_fallback_note:
            lines.append(f"Provider fallback: {provider_fallback_note}")

        if isinstance(timing_warmup_frames, (int, float)) and isinstance(timing_warmup_remaining, (int, float)):
            lines.append(
                f"Timing warmup: excluded first {int(timing_warmup_frames)} sample(s), remaining={max(0, int(timing_warmup_remaining))}"
            )

        if enabled and not active:
            lines.append("Mode: basic CUDA scaling remains live as fallback until the AI SR engine becomes active.")
        elif enabled:
            lines.append("Mode: ONNX AI SR active (basic CUDA scaling is bypassed while AI SR is producing output).")

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

        stage_error = None
        if hasattr(self._controller, "decklink_rtx_last_error"):
            stage_error = self._controller.decklink_rtx_last_error()
        if stage_error:
            lines.append(f"Last inference error (falling back to passthrough): {stage_error}")

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
        clock_correction_events = int(output_buffer_health.get("clock_correction_events", 0))
        overflow_events = int(output_buffer_health.get("overflow_events", 0))
        reprime_events = int(output_buffer_health.get("auto_reprime_events", 0))
        buffered_count = int(output_buffer_health.get("last_buffered_count", -1))
        target_buffer = int(output_buffer_health.get("target_buffer_frames", int(self._decklink_output_buffer_frames)))
        last_reprime_reason = str(output_buffer_health.get("last_reprime_reason", ""))

        cadence = pipeline_timing_health.get('cadence')
        if isinstance(cadence, dict):
            signature = (cadence.get('generation'), cadence.get('event_count'), cadence.get('capture_queue_drops'),
                         output_buffer_health.get('latency_drop_events', 0), target_buffer)
            previous = getattr(self, '_last_cadence_log_signature', None)
            if signature != previous:
                report = dict(cadence)
                previous_event = previous[1] if previous and previous[0] == signature[0] else 0
                report['recent_events'] = [event for event in cadence.get('recent_events', []) if event['event'] > (previous_event or 0)]
                report.update(buffered_frames=buffered_count, target_buffer_frames=target_buffer, path=timing_last_path)
                report['output_capacity_wait'] = {
                    key: output_buffer_health.get(key, 0)
                    for key in ('capacity_wait_polls', 'capacity_poll_ms_last', 'capacity_poll_ms_peak')
                }
                report['low_latency'] = {
                    key: output_buffer_health.get(key, 0)
                    for key in ('latency_budget_frames', 'latency_budget_ms', 'latency_drop_events',
                                'strict_frame_age_ms_last', 'accepted_frame_age_ms_last', 'accepted_frame_age_ms_peak')
                }
                report['output_clock'] = {
                    key: output_buffer_health.get(key)
                    for key in ('clock_source', 'clock_query_errors', 'clock_query_ms_last',
                                'clock_query_ms_peak', 'clock_epoch_adjustment_ms_last',
                                'minimum_preroll_frames', 'low_latency_output_enabled')
                }
                LOGGER.info('CADENCE | %s', json.dumps(report, separators=(',', ':')))
                self._last_cadence_log_signature = signature

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
        if isinstance(cadence, dict) and cadence.get('presentation_results_available'):
            completed = cadence.get('output_completion_results', {})
            generation = cadence.get('generation')
            late_count = int(completed.get('late_frames') or 0)
            dropped_count = int(completed.get('dropped_output_frames') or 0)
            previous = getattr(self, '_last_output_completion_health', None)
            self._last_output_completion_health = (generation, late_count, dropped_count)
            if previous is not None and previous[0] == generation:
                late_delta = max(0, late_count - previous[1])
                dropped_delta = max(0, dropped_count - previous[2])
                if late_delta or dropped_delta:
                    health_level = 'warn'
                    health_reasons.append(f"device_output(+late={late_delta},+dropped={dropped_delta})")

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
                    "buffer[target=%d,current=%d,s=%d,c=%d,o=%d,r=%d,reason=%s]"
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
                clock_correction_events,
                overflow_events,
                reprime_events,
                last_reprime_reason,
            )

        interp_tag = "interp" if interaction_active else "steady"
        if not health_reasons:
            return (
                f"health=ok ({interp_tag}) | out={output_fps:.1f}/{nominal_fps:.1f} | "
                f"buf={buffered_count}/{target_buffer} | corrections={clock_correction_events}"
            )

        return (
            f"health={health_level} ({interp_tag}) | out={output_fps:.1f}/{nominal_fps:.1f} | "
            f"buf={buffered_count}/{target_buffer} | corrections={clock_correction_events} | "
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

        if self.scaling_mode_combo.currentText() != SCALING_MODE_BASIC:
            # Basic scaling is only the sole active stage in this mode; AI
            # SR/RTX VSR performance is not affected by these mitigations.
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
            self._set_basic_scaling_enabled_effective(False)
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
            self._set_decklink_status("Synthetic mode active")
            self._update_fps_control_lock()
            return

        self._update_fps_control_lock()
        self._refresh_decklink_catalog()
        self._on_apply_decklink_settings()

    def _on_blackmagic_combo_changed(self) -> None:
        if self._updating_controls:
            return
        if self.sender() is self.decklink_timecode_format_combo:
            self._reindex_timecode_keyframes_for_selected_format()
            self._timecode_last_applied_frame = None
            self._timecode_phase_tracker.clear()
        if self.sender() is self.decklink_timecode_phase_combo:
            self._timecode_last_applied_frame = None
            self._timecode_phase_tracker.clear()
            self._sync_worker_timecode_roi_keyframes()
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
            self.decklink_timecode_format_combo,
            self.color_space_combo,
            self.color_range_combo,
            self.decklink_enable_format_detection,
            self.decklink_fps_priority_guard_checkbox,
            self.worker_priority_combo,
            self.decklink_apply_btn,
            self.decklink_refresh_btn,
        ]:
            widget.setEnabled(True)

    def _update_fps_control_lock(self) -> None:
        blackmagic_selected = self._source_mode == "Blackmagic DeckLink"
        self.fps_spin.setEnabled(not blackmagic_selected)

    def _on_apply_decklink_settings(self) -> None:
        selected = self.input_sources.configs[0]
        self.source_mode_combo.blockSignals(True)
        self.source_mode_combo.setCurrentText("Blackmagic DeckLink")
        self.source_mode_combo.blockSignals(False)
        for combo, key in ((self.decklink_timecode_format_combo, "timecode_format"),
                           (self.decklink_timecode_phase_combo, "timecode_phase")):
            i = combo.findData(selected.get(key))
            if i >= 0:
                combo.setCurrentIndex(i)
        selected_source_mode = self.source_mode_combo.currentText()
        self._source_mode = selected_source_mode
        self._sync_blackmagic_controls_enabled_state()
        self._update_timer_interval()
        self._update_fps_control_lock()

        if self._source_mode != "Blackmagic DeckLink":
            self._stop_decklink_sessions()
            self._set_decklink_timecode_display(None, placeholder="Timecode: unavailable in Synthetic mode")
            self._set_decklink_status("Synthetic mode active")
            self._update_status("Applied source mode: Synthetic")
            return

        try:
            self._apply_controller_color_settings_from_ui()
        except Exception as exc:
            self._update_status(f"DeckLink color settings apply failed: {exc}")
            return

        if d is None:
            self._set_decklink_status("decklink_wrapper is not available in this environment")
            self._update_status("DeckLink unavailable: install or activate decklink_wrapper environment")
            return

        if self.decklink_output_device_combo.count() == 0:
            self._refresh_decklink_catalog()

        try:
            self._start_decklink_sessions()
        except Exception as exc:
            LOGGER.exception("DeckLink setup failed")
            self._set_decklink_status(f"DeckLink setup failed: {exc}")
            self._update_status(f"DeckLink setup failed: {exc}")

    def _start_decklink_sessions(self) -> None:
        self._stop_decklink_sessions()

        if self.decklink_auto_detect_devices.isChecked():
            self._apply_auto_detect_device_selection()

        in_device = None  # ROI reads logical source 1, independent of output setup.
        out_device = self._selected_combo_data(self.decklink_output_device_combo)
        if out_device is None:
            # One retry after a full catalog refresh to recover from stale/placeholder
            # combo state when devices are present but selection data is null.
            self._refresh_decklink_catalog()
            if self.decklink_auto_detect_devices.isChecked():
                self._apply_auto_detect_device_selection()
            out_device = self._selected_combo_data(self.decklink_output_device_combo)
        if out_device is None:
            raise RuntimeError("No compatible DeckLink output device selected")

        in_mode = None
        out_mode = self._selected_combo_data(self.decklink_output_mode_combo)
        if out_mode is None:
            raise RuntimeError("No compatible DeckLink output mode selected")
        timecode_format = self._selected_combo_data(self.decklink_timecode_format_combo)
        if timecode_format is None:
            timecode_format = 0x72707631

        input_fps = None
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
                    timecode_format=int(timecode_format),
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
                    timecode_format=int(timecode_format),
                )
            self._capture_session = None
            self._output_session = None
        else:
            warmup_key = (id(self._controller.processor), self._controller.basic_scaling_method,
                          self._controller.basic_scaling_auto_mode, self._controller.max_auto_basic_scaling,
                          self._controller.basic_scaling_manual)
            if warmup_key != getattr(self, "_roi_warmup_key", None):
                warmup_roi_scaling(self._controller.processor, FRAME_W, FRAME_H,
                                   self._controller.enable_basic_scaling, self._controller.basic_scaling_auto_mode,
                                   self._controller.max_auto_basic_scaling, self._controller.basic_scaling_manual)
                self._roi_warmup_key = warmup_key
            if not hasattr(self._controller, "_source_pool"):
                self._controller._source_pool = SourcePool()
            pool = self._controller._source_pool
            self._output_session = d.OutputSession(
                device_index=out_device,
                display_mode=out_mode,
                pixel_format=d.PIXEL_FORMAT_8BIT_YUV,
                low_latency_output=True,
            )
            self._capture_session = RoiSourceSession(
                pool, FRAME_W, FRAME_H, 1.0 / (output_fps or (60000 / 1001)),
                lambda rgb: rgb_to_uyvy(rgb, self._controller.color_space, self._controller.color_range),
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

        roi_config = self.input_sources.configs[0]
        input_name = "Logical source 1: " + (str(roi_config.get("name", "")).strip() or roi_config.get("device_name") or "Unassigned")
        if 1 not in self.input_sources.active:
            input_name += " (inactive; black)"
        output_name = f"device {out_device}"
        in_mode_name = self._roi_source_mode_text()
        out_mode_name = self.decklink_output_mode_combo.currentText()
        output_label = self.decklink_output_device_combo.currentText()
        if output_label:
            output_name = output_label

        backend_text = "worker process" if self._controller_backend == "worker-process" else "GUI process"
        timecode_format_name = self.decklink_timecode_format_combo.currentText()
        roi_device = str(roi_config.get("device", ""))
        hfrtc_supported = getattr(self, "_decklink_hfrtc_support_by_index", {}).get(int(roi_device.split(":", 1)[1])) if roi_device.startswith("decklink:") else None
        hfrtc_text = "unknown" if hfrtc_supported is None else ("supported" if hfrtc_supported else "unsupported")
        self._set_decklink_status(
            "DeckLink configured: "
            f"ROI={input_name}; "
            f"out={output_name} mode='{out_mode_name}' ({out_mode}); "
            f"fps={fps_text}; timecode={timecode_format_name}; HFRTC={hfrtc_text}; backend={backend_text}"
        )
        self._set_decklink_timecode_display(None, placeholder="Timecode: waiting for DeckLink frames...")
        self._decklink_sessions_running = True
        self._schedule_settings_save()
        self._sync_roi_transition_unit_labels()
        LOGGER.info(
            "DeckLink started: input=%s mode=%s output=%s mode=%s fps=%s timecode=%s HFRTC=%s",
            input_name,
            in_mode_name,
            output_name,
            out_mode_name,
            fps_text,
            timecode_format_name,
            hfrtc_text,
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
        windows_cameras = _windows_video_capture_devices()
        self._set_input_source_catalog(windows_cameras)
        if d is None:
            self._set_decklink_status("decklink_wrapper is not available in this environment")
            LOGGER.error("DeckLink catalog refresh failed: wrapper unavailable")
            return

        try:
            devices = _call_decklink_api("list_devices")
        except Exception as exc:
            LOGGER.exception("DeckLink catalog refresh failed while listing devices")
            self._set_decklink_status(f"DeckLink refresh failed: {exc}")
            self._update_status(f"DeckLink refresh failed: {exc}")
            self.decklink_input_device_combo.clear()
            self.decklink_output_device_combo.clear()
            self.decklink_input_mode_combo.clear()
            self.decklink_output_mode_combo.clear()
            self.decklink_input_device_combo.addItem("DeckLink refresh failed", None)
            self.decklink_output_device_combo.addItem("DeckLink refresh failed", None)
            return

        LOGGER.info("DeckLink refresh: detected %d device(s)", len(devices))
        self._decklink_hfrtc_support_by_index = {
            int(dev.index): bool(getattr(dev, "supports_high_frame_rate_timecode", False))
            for dev in devices
        }

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

        self._set_input_source_catalog(
            [
                (f"{dev.display_name} [{dev.model_name}]", int(dev.index))
                for dev in devices
                if dev.supports_input
            ]
            + windows_cameras
        )

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
            if _mode_name_is_interlaced(self._roi_source_mode_text()):
                return INTERLACED_DEFAULT_DEINTERLACE_METHOD
        return PROGRESSIVE_DEFAULT_DEINTERLACE_METHOD

    def _roi_source_mode_text(self) -> str:
        if not hasattr(self, "input_sources"):
            return self.decklink_input_mode_combo.currentText()
        config = self.input_sources.configs[0]
        if not str(config.get("device", "")).startswith("decklink:"):
            return "Progressive"
        return str(config.get("mode_text", ""))

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
        # QTimer accepts integer milliseconds. Floor the period so 59.94/60 Hz
        # modes never round up to 17 ms (~58.8 updates/s). Interlaced modes use
        # the effective field rate computed below, preserving two updates/frame.
        return max(1, int(math.floor(1000.0 / field_rate_hz)))

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
        self._schedule_settings_save()
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
        self._on_roi_adjustment_started()
        self._cancel_roi_keyframe_transition()
        self._roi = Roi(0, 0, FRAME_W, FRAME_H)
        self._input_canvas.set_roi(self._roi)
        self._apply_controller_roi_immediate(self._roi)
        self._sync_controls_from_roi(self._roi)
        self._on_roi_adjustment_finished()

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

    def _restore_timecode_roi_keyframes(self, raw: object) -> None:
        restored: dict[int, TimecodeRoiKeyframe] = {}
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                roi_values = item.get("roi")
                if not isinstance(roi_values, list) or len(roi_values) != 4:
                    continue
                try:
                    raw_timecode = str(item.get("timecode", ""))
                    timecode = _normalize_timecode_display(raw_timecode)
                    timecode_format = str(item.get("timecode_format", "")).strip()
                    if not timecode_format:
                        timecode_format = self.decklink_timecode_format_combo.currentText()
                    count_fps = _timecode_count_fps(timecode_format, self._timecode_output_fps_for_tracking())
                    if "drop_frame" in item:
                        drop_frame = bool(item.get("drop_frame"))
                    elif ";" in raw_timecode:
                        drop_frame = True
                    else:
                        drop_frame = _infer_legacy_drop_frame(
                            timecode,
                            int(item.get("frame_number", 0)),
                            count_fps,
                        )
                    calculation_timecode = _timecode_with_drop_frame_separator(timecode, drop_frame)
                    field_mark = bool(item.get("field_mark", False))
                    parsed_frame_number = _timecode_to_internal_frame_number(
                        calculation_timecode,
                        count_fps,
                        field_mark,
                    )
                    if parsed_frame_number is None:
                        continue
                    frame_number = max(0, parsed_frame_number)
                    interpolation_mode = str(item.get("interpolation_mode", "linear")).strip().lower()
                    if interpolation_mode not in {"linear", "ease_in_out", "ease_out"}:
                        interpolation_mode = "linear"
                    keyframe = TimecodeRoiKeyframe(
                        timecode=timecode,
                        frame_number=frame_number,
                        roi=clamp_roi(Roi(*(int(value) for value in roi_values))),
                        interpolation_mode=interpolation_mode,
                        timecode_format=timecode_format,
                        drop_frame=drop_frame,
                        field_mark=field_mark,
                    )
                except (TypeError, ValueError):
                    continue
                if keyframe.timecode:
                    _store_unique_timecode_keyframe(restored, keyframe)
        self._timecode_roi_keyframes = restored
        self._rebuild_timecode_roi_lookup()

    def _set_roi_keyframing_mode(self, timecode_enabled: bool, save: bool = True) -> None:
        self._timecode_resume_timer.stop()
        self._timecode_resume_status = ""
        self._manual_roi_send_timer.stop()
        self._pending_manual_controller_roi = None
        self._manual_live_target_roi = None
        self._timecode_adjustment_finish_pending = False
        self._input_canvas.reset_interaction_state()
        self._input_canvas.set_roi(self._roi)
        self._timecode_keyframing_enabled = bool(timecode_enabled)
        if not self._timecode_keyframing_enabled:
            self._timecode_adjustment_paused = False
            self._timecode_adjustment_anchor = None
            self._rebuild_timecode_roi_lookup()
            if hasattr(self._controller, "set_roi_subpixel_shift"):
                self._controller.set_roi_subpixel_shift(0.0, 0.0)
        self.roi_manual_keyframe_widget.setVisible(not self._timecode_keyframing_enabled)
        self.roi_timecode_keyframe_widget.setVisible(self._timecode_keyframing_enabled)
        self.roi_transition_units_label.setVisible(not self._timecode_keyframing_enabled)
        self.roi_transition_frames_spin.setVisible(not self._timecode_keyframing_enabled)
        self.roi_keyframe_duration_override_btn.setVisible(not self._timecode_keyframing_enabled)
        self._sync_fullscreen_keyframing_mode()
        self._update_timecode_playback_mode_control()
        self._sync_worker_timecode_roi_keyframes()
        self._timecode_last_applied_frame = None
        self._update_timecode_keyframe_display()
        if self._timecode_keyframing_enabled and self._timecode_playback_enabled:
            self._apply_timecode_roi_for_current_timecode()
        if save:
            self._schedule_settings_save()

    def _update_timecode_playback_mode_control(self) -> None:
        playback_active = bool(self._timecode_playback_enabled and not self._timecode_adjustment_paused)
        buttons = []
        if hasattr(self, "roi_timecode_playback_btn"):
            buttons.append(self.roi_timecode_playback_btn)
        buttons.extend(getattr(self, "_fullscreen_timecode_playback_buttons", {}).values())
        for button in buttons:
            was_blocked = button.blockSignals(True)
            button.setChecked(playback_active)
            button.setText("Playback Mode" if playback_active else "Edit Mode")
            button.setToolTip(
                "Timecode drives the keyframed ROI" if playback_active
                else "Timecode playback is disabled; stored keyframes can be selected and edited"
            )
            button.blockSignals(was_blocked)

    def _on_timecode_playback_toggled(self, enabled: bool) -> None:
        if not self._timecode_keyframing_enabled:
            return
        self._timecode_resume_timer.stop()
        self._timecode_resume_status = ""
        self._timecode_adjustment_finish_pending = False
        self._timecode_adjustment_paused = False
        self._timecode_playback_enabled = bool(enabled)
        self._update_timecode_playback_mode_control()
        self._sync_worker_timecode_roi_keyframes()
        self._timecode_last_applied_frame = None
        if self._timecode_playback_enabled:
            self._apply_timecode_roi_for_current_timecode()
            self._update_status("Timecode ROI playback enabled")
        else:
            self._input_canvas.clear_visual_roi_overlay()
            self._update_status("Timecode ROI playback disabled; Edit Mode enabled")
        self._schedule_settings_save()

    def _timecode_nominal_fps(self) -> int:
        video_fps = 0.0
        if hasattr(self._controller, "decklink_output_nominal_fps"):
            try:
                video_fps = float(self._controller.decklink_output_nominal_fps())
            except Exception:
                video_fps = 0.0
        if video_fps <= 1.0:
            video_fps = float(max(1, self.fps_spin.value()))
        selected_format = self.decklink_timecode_format_combo.currentText()
        return _timecode_count_fps(selected_format, video_fps)

    def _reindex_timecode_keyframes_for_selected_format(self) -> None:
        reindexed: dict[int, TimecodeRoiKeyframe] = {}
        for keyframe in self._timecode_roi_keyframes.values():
            calculation_timecode = _timecode_with_drop_frame_separator(keyframe.timecode, keyframe.drop_frame)
            timecode_format = keyframe.timecode_format or self.decklink_timecode_format_combo.currentText()
            count_fps = _timecode_count_fps(timecode_format, self._timecode_output_fps_for_tracking())
            frame_number = _timecode_to_internal_frame_number(
                calculation_timecode,
                count_fps,
                keyframe.field_mark,
            )
            if frame_number is None:
                continue
            reindexed_keyframe = TimecodeRoiKeyframe(
                timecode=keyframe.timecode,
                frame_number=frame_number,
                roi=keyframe.roi,
                interpolation_mode=keyframe.interpolation_mode,
                timecode_format=timecode_format,
                drop_frame=keyframe.drop_frame,
                field_mark=keyframe.field_mark,
            )
            _store_unique_timecode_keyframe(reindexed, reindexed_keyframe)
        self._timecode_roi_keyframes = reindexed
        self._timecode_adjustment_anchor = None
        self._timecode_selected_frame = None
        self._rebuild_timecode_roi_lookup()
        self._update_timecode_keyframe_display()

    def _current_timecode_position(self) -> tuple[str, float] | None:
        if not bool(self._decklink_timecode_info.get("present", False)):
            self._timecode_phase_tracker.clear()
            return None
        timecode = str(self._decklink_timecode_info.get("text", "")).strip()
        detected_format = str(self._decklink_timecode_info.get("format_name", "")).strip()
        if not detected_format:
            detected_format = self.decklink_timecode_format_combo.currentText()
        frame_number = _timecode_position_from_info(
            self._decklink_timecode_info,
            detected_format,
            self._timecode_output_fps_for_tracking(),
            str(self.decklink_timecode_phase_combo.currentData()),
            phase_tracker=self._timecode_phase_tracker,
            source_seq=self._decklink_timecode_info.get("_seq"),
            delivery_phases=self._timecode_input_delivery_phases(),
        )
        if frame_number is None:
            return None
        return timecode, frame_number

    def _timecode_output_fps_for_tracking(self) -> float:
        video_fps = 0.0
        if hasattr(self._controller, "decklink_output_nominal_fps"):
            try:
                video_fps = float(self._controller.decklink_output_nominal_fps())
            except Exception:
                video_fps = 0.0
        if video_fps <= 1.0:
            video_fps = float(max(1, self.fps_spin.value()))
        return video_fps

    def _timecode_input_delivery_phases(self) -> int:
        if self._source_mode != "Blackmagic DeckLink":
            return 1
        return 2 if _mode_name_is_interlaced(self._roi_source_mode_text()) else 1

    def _rebuild_timecode_roi_lookup(self, sync_worker: bool = True) -> None:
        effective_keyframes = dict(self._timecode_roi_keyframes)
        if self._timecode_adjustment_anchor is not None:
            effective_keyframes[self._timecode_adjustment_anchor.frame_number] = self._timecode_adjustment_anchor
        self._timecode_roi_lookup_keyframes = effective_keyframes
        ordered = sorted(effective_keyframes.values(), key=lambda item: item.frame_number)
        self._timecode_roi_ordered_frames = [keyframe.frame_number for keyframe in ordered]
        self._timecode_roi_segments = []
        for start_key, end_key in zip(ordered, ordered[1:]):
            if end_key.frame_number <= start_key.frame_number:
                continue
            values = _build_timecode_roi_segment(start_key, end_key)
            self._timecode_roi_segments.append((start_key.frame_number, end_key.frame_number, values))
        self._timecode_roi_segment_starts = [segment[0] for segment in self._timecode_roi_segments]
        self._timecode_last_applied_frame = None
        if sync_worker:
            self._sync_worker_timecode_roi_keyframes()

    def _sync_worker_timecode_roi_keyframes(self) -> None:
        if getattr(self, "_controller_backend", "") != "worker-process":
            return
        controller = getattr(self, "_controller", None)
        if controller is None or not hasattr(controller, "set_timecode_roi_keyframes"):
            return
        keyframes = []
        for frame_number in sorted(self._timecode_roi_lookup_keyframes):
            keyframe = self._timecode_roi_lookup_keyframes[frame_number]
            keyframes.append(
                {
                    "frame_number": int(frame_number),
                    "roi": [
                        float(keyframe.roi.x),
                        float(keyframe.roi.y),
                        float(keyframe.roi.w),
                        float(keyframe.roi.h),
                    ],
                    "interpolation_mode": str(keyframe.interpolation_mode),
                }
            )
        enabled = (
            self._timecode_keyframing_enabled
            and self._timecode_playback_enabled
            and not self._timecode_adjustment_paused
        )
        controller.set_timecode_roi_keyframes(enabled, keyframes, str(self.decklink_timecode_phase_combo.currentData()))

    def _timecode_roi_values_at_frame(self, frame_number: float) -> np.ndarray | None:
        ordered_frames = self._timecode_roi_ordered_frames
        if not ordered_frames:
            return None
        if frame_number <= ordered_frames[0]:
            roi = self._timecode_roi_lookup_keyframes[ordered_frames[0]].roi
            return np.array([roi.x, roi.y, roi.w, roi.h], dtype=np.float32)
        if frame_number >= ordered_frames[-1]:
            roi = self._timecode_roi_lookup_keyframes[ordered_frames[-1]].roi
            return np.array([roi.x, roi.y, roi.w, roi.h], dtype=np.float32)
        segment_index = bisect.bisect_right(self._timecode_roi_segment_starts, frame_number) - 1
        if segment_index < 0:
            return None
        start_frame, end_frame, values = self._timecode_roi_segments[segment_index]
        if frame_number > end_frame:
            return None
        relative_frame = max(0.0, float(frame_number) - float(start_frame))
        lower_index = min(len(values) - 1, int(math.floor(relative_frame)))
        upper_index = min(len(values) - 1, lower_index + 1)
        fraction = relative_frame - float(lower_index)
        return values[lower_index] + ((values[upper_index] - values[lower_index]) * fraction)

    def _timecode_roi_at_frame(self, frame_number: int) -> Roi | None:
        sample = self._timecode_roi_values_at_frame(frame_number)
        if sample is None:
            return None
        return self._timecode_roi_carrier(sample)

    def _timecode_roi_carrier(self, sample: np.ndarray) -> Roi:
        return clamp_roi(
            Roi(
                int(round(float(sample[0]) / 2.0)) * 2,
                int(round(float(sample[1]))),
                max(2, int(round(float(sample[2]) / 2.0)) * 2),
                int(round(float(sample[3]))),
            )
        )

    def _apply_timecode_roi_sample(self, sample: np.ndarray) -> None:
        target_roi = self._timecode_roi_carrier(sample)
        self._roi = target_roi
        self._input_canvas.set_roi(target_roi)

        if hasattr(self._controller, "set_roi_with_subpixel"):
            self._controller_roi_target = None
            self._controller_roi_interp_timer.stop()
            self._controller_filtered_target_roi = None
            self._controller_interp_residual = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
            desired_center_x = float(sample[0]) + (float(sample[2]) * 0.5)
            desired_center_y = float(sample[1]) + (float(sample[3]) * 0.5)
            carrier_center_x = float(target_roi.x) + (float(target_roi.w) * 0.5)
            carrier_center_y = float(target_roi.y) + (float(target_roi.h) * 0.5)
            scale_x = FRAME_W / max(1.0, float(target_roi.w))
            scale_y = FRAME_H / max(1.0, float(target_roi.h))
            max_shift_x = max(2.0, min(48.0, scale_x * 1.5))
            max_shift_y = max(2.0, min(48.0, scale_y * 1.5))
            shift_x = max(-max_shift_x, min(max_shift_x, -((desired_center_x - carrier_center_x) * scale_x)))
            shift_y = max(-max_shift_y, min(max_shift_y, -((desired_center_y - carrier_center_y) * scale_y)))
            self._controller.set_roi_with_subpixel(target_roi, shift_x, shift_y)
            self._controller_roi_applied = target_roi
        else:
            self._apply_controller_roi_immediate(target_roi)

        self._sync_controls_from_roi(target_roi)

    def _on_timecode_add_keyframe(self) -> None:
        position = self._current_timecode_position()
        if position is None:
            self._update_status("Cannot add keyframe: no valid DeckLink timecode is available")
            return
        timecode, frame_number = position
        frame_number = int(round(frame_number))
        drop_frame = bool(self._decklink_timecode_info.get("drop_frame", ";" in timecode))
        field_mark = bool(self._decklink_timecode_info.get("field_mark", False))
        timecode_format = str(self._decklink_timecode_info.get("format_name", "")).strip()
        timecode = _normalize_timecode_display(timecode)
        keyframe = TimecodeRoiKeyframe(
            timecode=timecode,
            frame_number=frame_number,
            roi=clamp_roi(self._roi),
            interpolation_mode=self._roi_interp_mode_name(),
            timecode_format=timecode_format,
            drop_frame=drop_frame,
            field_mark=field_mark,
        )
        replaced = _store_unique_timecode_keyframe(self._timecode_roi_keyframes, keyframe)
        self._timecode_selected_frame = frame_number
        self._rebuild_timecode_roi_lookup()
        self._update_timecode_keyframe_display()
        self._schedule_settings_save()
        action = "Replaced" if replaced else "Stored"
        self._update_status(f"{action} timecode ROI keyframe at {timecode}")

    def _on_timecode_delete_keyframe(self) -> None:
        frame_number = self._timecode_selected_frame
        if frame_number not in self._timecode_roi_keyframes:
            self._update_status("No loaded timecode keyframe to delete")
            return
        deleted = self._timecode_roi_keyframes.pop(frame_number)
        self._timecode_selected_frame = None
        self._rebuild_timecode_roi_lookup()
        self._update_timecode_keyframe_display()
        self._schedule_settings_save()
        self._update_status(f"Deleted timecode ROI keyframe at {deleted.timecode}")

    def _on_timecode_delete_all_keyframes(self) -> None:
        self._timecode_roi_keyframes.clear()
        self._timecode_adjustment_anchor = None
        self._timecode_selected_frame = None
        self._rebuild_timecode_roi_lookup()
        self._update_timecode_keyframe_display()
        self._schedule_settings_save()
        self._update_status("Deleted all timecode ROI keyframes")

    def _navigate_timecode_keyframe(self, direction: int) -> None:
        ordered_frames = sorted(self._timecode_roi_keyframes)
        if not ordered_frames:
            self._update_status("No timecode keyframes are stored")
            return
        if self._timecode_selected_frame in ordered_frames:
            current_index = ordered_frames.index(self._timecode_selected_frame)
            target_index = max(0, min(len(ordered_frames) - 1, current_index + (1 if direction > 0 else -1)))
        else:
            target_index = 0 if direction > 0 else len(ordered_frames) - 1
        if self._timecode_playback_enabled:
            self._on_timecode_playback_toggled(False)
        self._load_timecode_keyframe(ordered_frames[target_index])

    def _load_timecode_keyframe(self, frame_number: int) -> None:
        keyframe = self._timecode_roi_keyframes.get(frame_number)
        if keyframe is None:
            return
        self._timecode_selected_frame = frame_number
        self._roi = clamp_roi(keyframe.roi)
        self._input_canvas.set_roi(self._roi)
        self._apply_controller_roi_immediate(self._roi)
        self._sync_controls_from_roi(self._roi)
        self._update_timecode_keyframe_display()
        self._update_status(f"Loaded timecode ROI keyframe at {keyframe.timecode}")

    def _update_timecode_keyframe_display(self) -> None:
        if not hasattr(self, "roi_timecode_key_label"):
            return
        ordered_frames = sorted(self._timecode_roi_keyframes)
        total = len(ordered_frames)
        if self._timecode_selected_frame in ordered_frames:
            index = ordered_frames.index(self._timecode_selected_frame) + 1
            timecode = self._timecode_roi_keyframes[self._timecode_selected_frame].timecode
            display_text = f"Key {index}/{total} :: {timecode}"
        else:
            display_text = f"Key 0/{total} :: --:--:--.--"
        self.roi_timecode_key_label.setText(display_text)
        for label in self._fullscreen_timecode_key_labels.values():
            label.setText(display_text)
        has_selection = self._timecode_selected_frame in self._timecode_roi_keyframes
        self.roi_timecode_delete_btn.setEnabled(has_selection)
        self.roi_timecode_delete_all_btn.setEnabled(total > 0)
        self.roi_timecode_previous_btn.setEnabled(total > 0)
        self.roi_timecode_next_btn.setEnabled(total > 0)
        for button in self._fullscreen_timecode_delete_buttons.values():
            button.setEnabled(has_selection)
        for button in self._fullscreen_timecode_delete_all_buttons.values():
            button.setEnabled(total > 0)
        for button in self._fullscreen_timecode_previous_buttons.values():
            button.setEnabled(total > 0)
        for button in self._fullscreen_timecode_next_buttons.values():
            button.setEnabled(total > 0)

    def _apply_timecode_roi_for_current_timecode(self) -> None:
        if (
            not self._timecode_keyframing_enabled
            or not self._timecode_playback_enabled
            or self._timecode_adjustment_paused
        ):
            return
        position = self._current_timecode_position()
        if position is None:
            return
        _, frame_number = position
        if frame_number == self._timecode_last_applied_frame:
            return
        sample = self._timecode_roi_values_at_frame(frame_number)
        if sample is None:
            return
        self._timecode_last_applied_frame = frame_number
        if self._controller_backend != "worker-process":
            self._apply_timecode_roi_sample(sample)
        else:
            carrier_roi = self._timecode_roi_carrier(sample)
            self._roi = carrier_roi
            self._input_canvas.set_roi(carrier_roi)
            self._input_canvas.set_visual_roi_overlay(
                float(sample[0]),
                float(sample[1]),
                float(sample[2]),
                float(sample[3]),
            )
            self._schedule_roi_controls_sync(carrier_roi)
        if frame_number in self._timecode_roi_keyframes:
            self._timecode_selected_frame = frame_number
            self._update_timecode_keyframe_display()

    def _on_roi_adjustment_started(self) -> None:
        if not self._timecode_keyframing_enabled:
            return
        self._timecode_resume_timer.stop()
        self._timecode_resume_status = ""
        self._timecode_adjustment_finish_pending = False
        self._input_canvas.clear_visual_roi_overlay()
        was_paused = self._timecode_adjustment_paused
        self._timecode_adjustment_paused = True
        self._update_timecode_playback_mode_control()
        if not was_paused:
            self._sync_worker_timecode_roi_keyframes()

    def _resume_timecode_after_manual_adjustment(self) -> None:
        if not self._timecode_keyframing_enabled or not self._timecode_adjustment_paused:
            return
        position = self._current_timecode_position()
        anchor = self._timecode_adjustment_anchor
        if position is not None and anchor is not None:
            timecode, frame_number = position
            frame_number = int(round(frame_number))
            self._timecode_adjustment_anchor = TimecodeRoiKeyframe(
                timecode=_normalize_timecode_display(timecode),
                frame_number=frame_number,
                roi=anchor.roi,
                interpolation_mode=anchor.interpolation_mode,
                timecode_format=str(self._decklink_timecode_info.get("format_name", "")).strip(),
                drop_frame=bool(self._decklink_timecode_info.get("drop_frame", ";" in timecode)),
                field_mark=bool(self._decklink_timecode_info.get("field_mark", False)),
            )
            self._rebuild_timecode_roi_lookup(sync_worker=False)
            self._timecode_last_applied_frame = frame_number
        else:
            self._timecode_last_applied_frame = None
        self._timecode_adjustment_paused = False
        self._update_timecode_playback_mode_control()
        self._sync_worker_timecode_roi_keyframes()
        if self._timecode_resume_status:
            self._update_status(self._timecode_resume_status)
        self._timecode_resume_status = ""

    def _on_roi_adjustment_finished(self) -> None:
        if not self._timecode_adjustment_paused:
            return
        if hasattr(self, "_roi_control_adjustment_timer"):
            self._roi_control_adjustment_timer.stop()
        final_roi = clamp_roi(self._roi)
        if self._pending_manual_controller_roi is not None or self._manual_live_target_roi is not None:
            self._timecode_adjustment_finish_pending = True
            self._pending_manual_controller_roi = final_roi
            self._manual_live_target_roi = final_roi
            if not self._manual_roi_send_timer.isActive():
                self._manual_roi_send_timer.start()
            return
        self._complete_timecode_adjustment()

    def _complete_timecode_adjustment(self) -> None:
        if not self._timecode_keyframing_enabled or not self._timecode_adjustment_paused:
            self._timecode_adjustment_finish_pending = False
            return
        self._timecode_adjustment_finish_pending = False
        self._manual_roi_send_timer.stop()
        self._pending_manual_controller_roi = None
        self._manual_live_target_roi = None
        self._apply_controller_roi_immediate(self._roi, settle_interlaced=True)
        if not self._timecode_playback_enabled:
            selected_key = self._timecode_roi_keyframes.get(self._timecode_selected_frame)
            if selected_key is not None:
                self._timecode_roi_keyframes[selected_key.frame_number] = TimecodeRoiKeyframe(
                    timecode=selected_key.timecode,
                    frame_number=selected_key.frame_number,
                    roi=clamp_roi(self._roi),
                    interpolation_mode=self._roi_interp_mode_name(),
                    timecode_format=selected_key.timecode_format,
                    drop_frame=selected_key.drop_frame,
                    field_mark=selected_key.field_mark,
                )
                self._timecode_adjustment_anchor = None
                self._timecode_adjustment_paused = False
                self._rebuild_timecode_roi_lookup()
                self._update_timecode_playback_mode_control()
                self._update_timecode_keyframe_display()
                self._schedule_settings_save()
                self._update_status(f"Updated timecode ROI keyframe at {selected_key.timecode}")
                return
            self._timecode_adjustment_paused = False
            self._update_timecode_playback_mode_control()
            self._sync_worker_timecode_roi_keyframes()
            self._update_status("ROI adjusted in Edit Mode; select or add a keyframe to store it")
            return
        position = self._current_timecode_position()
        if position is None:
            self._timecode_resume_status = "ROI adjusted; interpolation will resume when valid timecode is available"
            self._timecode_resume_timer.start(self._timecode_resume_debounce_ms)
            return
        timecode, frame_number = position
        frame_number = int(round(frame_number))
        anchor_roi = clamp_roi(self._roi)
        self._timecode_adjustment_anchor = TimecodeRoiKeyframe(
            timecode=_normalize_timecode_display(timecode),
            frame_number=frame_number,
            roi=anchor_roi,
            interpolation_mode=self._roi_interp_mode_name(),
            timecode_format=str(self._decklink_timecode_info.get("format_name", "")).strip(),
            drop_frame=bool(self._decklink_timecode_info.get("drop_frame", ";" in timecode)),
            field_mark=bool(self._decklink_timecode_info.get("field_mark", False)),
        )
        self._rebuild_timecode_roi_lookup(sync_worker=False)
        self._timecode_last_applied_frame = frame_number
        self._timecode_resume_status = f"ROI interpolation recalculated from {_normalize_timecode_display(timecode)}"
        self._timecode_resume_timer.start(self._timecode_resume_debounce_ms)

    def _on_roi_tap_center_requested(self, frame_x: float, frame_y: float) -> None:
        current = clamp_roi(self._roi)
        if self._fullscreen_view_name is not None and self._fullscreen_selected_scale_index is not None:
            scale_percent = self._fullscreen_scale_presets[self._fullscreen_selected_scale_index]
            target = roi_from_scale(float(scale_percent) / 100.0, float(frame_x), float(frame_y))
        else:
            target = clamp_roi(
                Roi(
                    int(round(float(frame_x) - (current.w * 0.5))),
                    int(round(float(frame_y) - (current.h * 0.5))),
                    current.w,
                    current.h,
                )
            )
        duration_frames = max(1, min(600, int(self.roi_transition_frames_spin.value())))
        self._start_roi_keyframe_transition(
            target,
            self._effective_roi_keyframe_duration_frames(target, duration_frames),
            self._roi_interp_mode_name(),
            manual_adjustment=True,
        )

    def _on_output_roi_tap_center_requested(self, frame_x: float, frame_y: float) -> None:
        if self._fullscreen_view_name == "output":
            self._on_roi_tap_center_requested(frame_x, frame_y)

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
        if isinstance(info, dict):
            self._decklink_timecode_info_sequence += 1
            self._decklink_timecode_info = dict(info)
            self._decklink_timecode_info["_seq"] = self._decklink_timecode_info_sequence
        else:
            self._decklink_timecode_info = {}
            self._timecode_phase_tracker.clear()
        if isinstance(info, dict) and bool(info.get("present", False)):
            timecode_text = str(info.get("text", "")).strip()
            format_name = str(info.get("format_name", "")).strip()
            if format_name in {"RP188 LTC", "RP188 VITC1", "RP188 VITC2"}:
                format_name = "RP188 auto"
            detected_cadence = _effective_synthesis_cadence(
                info,
                int(self._timecode_phase_tracker.get("detected_cadence", 1)),
            )
            if detected_cadence > 1:
                format_name += f", {detected_cadence}x"
            if timecode_text:
                display_text = f"Timecode: {timecode_text}"
                if format_name:
                    display_text += f" ({format_name})"
                internal_position = info.get("internal_position")
                if isinstance(internal_position, (int, float)) and self._timecode_roi_ordered_frames:
                    if float(internal_position) < float(self._timecode_roi_ordered_frames[0]):
                        display_text += " [before first ROI key]"
                    elif float(internal_position) > float(self._timecode_roi_ordered_frames[-1]):
                        display_text += " [after last ROI key]"

        self._decklink_timecode_display_text = display_text
        if hasattr(self, "decklink_timecode_label"):
            self.decklink_timecode_label.setText(display_text)
        if hasattr(self, "effects_graph"):
            effect_timecode = ""
            if isinstance(info, dict) and bool(info.get("present", False)):
                effect_timecode = str(info.get("text", "")).strip()
            self.effects_graph.set_composition_timecode(
                effect_timecode or "00:00:00:00",
                info if isinstance(info, dict) else None,
            )
        for label in self._fullscreen_timecode_display_labels.values():
            label.setText(display_text)
        self._apply_timecode_roi_for_current_timecode()

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
        if isinstance(previous_state, dict) and bool(previous_state.get("manual_adjustment", False)):
            self._on_roi_adjustment_finished()
        self._update_timer_interval()

    def _start_roi_keyframe_transition(
        self,
        target_roi: Roi,
        duration_frames: int,
        interpolation_mode: str,
        manual_adjustment: bool = False,
    ) -> None:
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
        if hasattr(self._controller, "clear_manual_roi_endpoint"):
            self._controller.clear_manual_roi_endpoint()

        target = clamp_roi(target_roi)
        total_frames = max(1, min(600, int(duration_frames)))
        mode_name = str(interpolation_mode).strip().lower()
        if mode_name not in {"linear", "ease_in_out", "ease_out"}:
            mode_name = "linear"
        # Retarget without sending a standalone cancel command first. In worker
        # mode, start_roi_microstep_transition can take over in-place from the
        # current rendered ROI+shift, avoiding a one-frame jump between moves.
        self._cancel_roi_keyframe_transition(reset_subpixel_shift=False, notify_backend=False)
        if manual_adjustment:
            self._on_roi_adjustment_started()

        if total_frames <= 1:
            self._roi = target
            self._input_canvas.set_roi(target)
            self._input_canvas.clear_visual_roi_overlay()
            self._apply_controller_roi_immediate(target)
            self._sync_controls_from_roi(target)
            if manual_adjustment:
                self._on_roi_adjustment_finished()
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
            "manual_adjustment": bool(manual_adjustment),
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
                finish_manual_adjustment = bool(state.get("manual_adjustment", False))
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
                if finish_manual_adjustment:
                    self._on_roi_adjustment_finished()
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
            finish_manual_adjustment = bool(state.get("manual_adjustment", False))
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
            if finish_manual_adjustment:
                self._on_roi_adjustment_finished()

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

        self.status_text.setPlainText(text)
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
    else:
        window.resize(1400, 860)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
