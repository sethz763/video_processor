from __future__ import annotations

import queue
import site
import sys
import threading
import time
import traceback
import os
import importlib
import math
import ctypes
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _bootstrap_project_venv_site() -> None:
    project_root = Path(__file__).resolve().parents[1]
    venv_site = project_root / "venv" / "Lib" / "site-packages"
    if not venv_site.exists():
        return

    venv_site_str = str(venv_site)
    if venv_site_str not in sys.path:
        # Prefer project venv packages over any globally installed packages.
        sys.path.insert(0, venv_site_str)
    site.addsitedir(venv_site_str)


_bootstrap_project_venv_site()


_CUDA_DLL_DIR_HANDLES: list[Any] = []
_CUDA_DLL_DIR_KEYS: set[str] = set()
_RTX_DLL_DIR_HANDLES: list[Any] = []
_RTX_DLL_DIR_KEYS: set[str] = set()
_TRT_PREFLIGHT_CACHE_OK: bool | None = None
_TRT_PREFLIGHT_CACHE_ERROR: str | None = None

_WORKER_PRIORITY_CLASS_MAP: dict[str, int] = {
    "normal": 0x00000020,
    "above_normal": 0x00008000,
    "high": 0x00000080,
}


def _normalize_worker_priority_name(priority_name: str) -> str:
    normalized = str(priority_name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"normal", "default"}:
        return "normal"
    if normalized in {"above_normal", "abovenormal", "high_normal", "highnormal"}:
        return "above_normal"
    if normalized in {"high", "high_priority", "highpriority"}:
        return "high"
    return "above_normal"


def _apply_current_process_priority(priority_name: str) -> tuple[str, str | None]:
    normalized = _normalize_worker_priority_name(priority_name)
    if os.name != "nt":
        return normalized, "process priority override is currently implemented for Windows only"

    class_id = _WORKER_PRIORITY_CLASS_MAP.get(normalized, _WORKER_PRIORITY_CLASS_MAP["above_normal"])
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        set_priority_class = kernel32.SetPriorityClass
        get_current_process.restype = ctypes.c_void_p
        set_priority_class.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        set_priority_class.restype = ctypes.c_int

        handle = get_current_process()
        ok = bool(set_priority_class(handle, ctypes.c_uint32(class_id)))
        if not ok:
            return normalized, str(ctypes.WinError(ctypes.get_last_error()))
        return normalized, None
    except Exception as exc:
        return normalized, str(exc)


def _clamp_interlaced_field2_phase_fraction(value: float) -> float:
    return max(_INTERLACED_FIELD2_PHASE_MIN, min(_INTERLACED_FIELD2_PHASE_MAX, float(value)))


def _candidate_cuda_dll_dirs() -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        key = str(path).lower()
        if key in seen:
            return
        if path.exists() and path.is_dir():
            seen.add(key)
            dirs.append(path)

    # CUDA toolkit locations from environment variables.
    for env_name, env_value in os.environ.items():
        if env_name == "CUDA_PATH" or env_name.startswith("CUDA_PATH_V"):
            _add(Path(env_value) / "bin")

    # Common default CUDA toolkit install location on Windows.
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    cuda_root = program_files / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if cuda_root.exists():
        for candidate in sorted(cuda_root.glob("v12*"), reverse=True):
            _add(candidate / "bin")

    # TensorRT runtime DLLs are often deployed with NVIDIA Video Effects.
    _add(program_files / "NVIDIA Corporation" / "NVIDIA Video Effects")

    # TensorRT installs are commonly rooted under these environment variables.
    for env_name in ("TENSORRT_PATH", "TENSORRT_ROOT", "TRT_LIBPATH"):
        env_value = os.environ.get(env_name, "").strip()
        if not env_value:
            continue
        root = Path(env_value)
        _add(root)
        _add(root / "lib")
        _add(root / "bin")

    # Common default TensorRT install roots on Windows.
    trt_root = program_files / "NVIDIA GPU Computing Toolkit" / "TensorRT"
    if trt_root.exists():
        _add(trt_root)
        _add(trt_root / "lib")
        _add(trt_root / "bin")
        for candidate in sorted(trt_root.glob("10*"), reverse=True):
            _add(candidate)
            _add(candidate / "lib")
            _add(candidate / "bin")

    # Also support pip-installed NVIDIA runtime packages in this venv.
    project_root = Path(__file__).resolve().parents[1]
    nvidia_site = project_root / "venv" / "Lib" / "site-packages" / "nvidia"
    if nvidia_site.exists():
        for pkg_dir in nvidia_site.iterdir():
            if pkg_dir.is_dir():
                _add(pkg_dir / "bin")

    # Honor caller-provided PATH entries (for example custom TensorRT installs)
    # so runtime setup stays environment-driven instead of hardcoded.
    for path_entry in os.environ.get("PATH", "").split(os.pathsep):
        if not path_entry:
            continue
        try:
            candidate = Path(path_entry)
        except Exception:
            continue
        if not candidate.exists() or not candidate.is_dir():
            continue
        lowered = str(candidate).lower()
        if "nvidia" in lowered or "tensorrt" in lowered or "cuda" in lowered:
            _add(candidate)

    return dirs


def _prepare_cuda_runtime_dll_paths() -> None:
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return

    for dll_dir in _candidate_cuda_dll_dirs():
        key = str(dll_dir).lower()
        if key in _CUDA_DLL_DIR_KEYS:
            continue
        try:
            handle = add_dll_directory(str(dll_dir))
            _CUDA_DLL_DIR_HANDLES.append(handle)
            _CUDA_DLL_DIR_KEYS.add(key)

            # Some delayed runtime loads still rely on PATH resolution.
            dll_dir_str = str(dll_dir)
            path_parts = os.environ.get("PATH", "").split(os.pathsep)
            path_keys = {part.lower() for part in path_parts if part}
            if key not in path_keys:
                os.environ["PATH"] = dll_dir_str + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            # Best effort: continue trying remaining directories.
            continue


def _current_windows_dll_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        key = str(path).lower()
        if key in seen:
            return
        if path.exists() and path.is_dir():
            seen.add(key)
            dirs.append(path)

    for base_dir in _candidate_cuda_dll_dirs():
        _add(base_dir)

    if ort is not None:
        try:
            ort_root = Path(getattr(ort, "__file__", "")).resolve().parent
            _add(ort_root)
            _add(ort_root / "capi")
        except Exception:
            pass

    for path_entry in os.environ.get("PATH", "").split(os.pathsep):
        if not path_entry:
            continue
        try:
            _add(Path(path_entry))
        except Exception:
            continue

    return dirs


def _find_dll_in_search_dirs(dll_name: str, search_dirs: list[Path]) -> Path | None:
    needle = str(dll_name).strip()
    for base in search_dirs:
        candidate = base / needle
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _ensure_tensorrt_builder_resource_alias(search_dirs: list[Path]) -> Path | None:
    expected_name = "nvinfer_builder_resource_10.dll"
    existing = _find_dll_in_search_dirs(expected_name, search_dirs)
    if existing is not None:
        return existing

    alt_names = [
        "nvinfer_builder_resource_ptx_10.dll",
    ]
    source: Path | None = None
    for alt_name in alt_names:
        source = _find_dll_in_search_dirs(alt_name, search_dirs)
        if source is not None:
            break
    if source is None:
        return None

    project_root = Path(__file__).resolve().parents[1]
    shim_dir = project_root / ".runtime" / "dll_shims"
    try:
        shim_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    alias_path = shim_dir / expected_name
    try:
        if (not alias_path.exists()) or (alias_path.stat().st_size != source.stat().st_size):
            shutil.copy2(source, alias_path)
    except Exception:
        return None

    shim_key = str(shim_dir).lower()
    if shim_key not in _CUDA_DLL_DIR_KEYS:
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if callable(add_dll_directory):
            try:
                handle = add_dll_directory(str(shim_dir))
                _CUDA_DLL_DIR_HANDLES.append(handle)
                _CUDA_DLL_DIR_KEYS.add(shim_key)
            except Exception:
                pass

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    path_keys = {part.lower() for part in path_parts if part}
    if shim_key not in path_keys:
        os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")

    if all(str(p).lower() != str(shim_dir).lower() for p in search_dirs):
        search_dirs.insert(0, shim_dir)

    return alias_path


def _preflight_tensorrt_runtime() -> None:
    # Guard the TensorRT path before ORT session creation. Missing builder
    # resources can trigger a native crash instead of a Python exception.
    global _TRT_PREFLIGHT_CACHE_OK, _TRT_PREFLIGHT_CACHE_ERROR

    if _TRT_PREFLIGHT_CACHE_OK is True:
        return
    if _TRT_PREFLIGHT_CACHE_OK is False:
        raise RuntimeError(_TRT_PREFLIGHT_CACHE_ERROR or "TensorRT runtime preflight failed")

    if os.name != "nt":
        _TRT_PREFLIGHT_CACHE_OK = True
        _TRT_PREFLIGHT_CACHE_ERROR = None
        return

    search_dirs = _current_windows_dll_search_dirs()
    _ensure_tensorrt_builder_resource_alias(search_dirs)
    required = [
        "nvinfer_10.dll",
        "nvinfer_plugin_10.dll",
        "nvinfer_builder_resource_10.dll",
    ]

    missing: list[str] = []
    found: dict[str, str] = {}
    for dll_name in required:
        resolved = _find_dll_in_search_dirs(dll_name, search_dirs)
        if resolved is None:
            missing.append(dll_name)
        else:
            found[dll_name] = str(resolved)

    if missing:
        err = (
            "TensorRT runtime is incomplete; missing required DLL(s): "
            f"{', '.join(missing)} | found={found}"
        )
        _TRT_PREFLIGHT_CACHE_OK = False
        _TRT_PREFLIGHT_CACHE_ERROR = err
        raise RuntimeError(err)

    try:
        for dll_name in required:
            ctypes.WinDLL(found[dll_name])
    except Exception as exc:
        err = f"TensorRT runtime DLL load preflight failed: {exc}"
        _TRT_PREFLIGHT_CACHE_OK = False
        _TRT_PREFLIGHT_CACHE_ERROR = err
        raise RuntimeError(err) from exc

    _TRT_PREFLIGHT_CACHE_OK = True
    _TRT_PREFLIGHT_CACHE_ERROR = None


def _prepare_rtx_runtime_dll_paths(sdk_root: str, project_root: Path) -> None:
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return

    candidates = [
        project_root / "build" / "src" / "Release",
        project_root / "build" / "src" / "RelWithDebInfo",
        project_root / "build" / "src" / "Debug",
    ]

    if sdk_root:
        sdk_path = Path(sdk_root)
        candidates.extend(
            [
                sdk_path / "bin" / "Windows" / "x64" / "rel",
                sdk_path / "bin" / "Windows" / "x64" / "dev",
            ]
        )

    for dll_dir in candidates:
        key = str(dll_dir).lower()
        if key in _RTX_DLL_DIR_KEYS:
            continue
        if not dll_dir.exists() or not dll_dir.is_dir():
            continue
        try:
            handle = add_dll_directory(str(dll_dir))
            _RTX_DLL_DIR_HANDLES.append(handle)
            _RTX_DLL_DIR_KEYS.add(key)
        except Exception:
            continue

try:
    import cv2
except Exception:
    cv2 = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import decklink_wrapper as d
except Exception:
    d = None

try:
    import rtx_vsr as rtx_vsr_module
except Exception:
    rtx_vsr_module = None


FRAME_W = 1920
FRAME_H = 1080
UYVY_ROW_BYTES = FRAME_W * 2
_SUBPIXEL_SHIFT_APPLY_EPS = max(0.0, float(os.environ.get("VP_SUBPIXEL_SHIFT_APPLY_EPS", "0.03")))
_INTERLACED_FIELD2_PHASE_MIN = -1.0
_INTERLACED_FIELD2_PHASE_MAX = 2.0
_INTERLACED_FIELD2_PHASE_FRACTION = max(
    _INTERLACED_FIELD2_PHASE_MIN,
    min(_INTERLACED_FIELD2_PHASE_MAX, float(os.environ.get("VP_INTERLACED_FIELD2_PHASE_FRACTION", "0.50"))),
)
_AI_SR_TIMING_WARMUP_FRAMES = max(0, int(os.environ.get("VP_AI_SR_TIMING_WARMUP_FRAMES", "8")))
_OUTPUT_SCHEDULE_STARVED_STREAK_THRESHOLD = max(
    1,
    int(os.environ.get("VP_OUTPUT_BUFFER_STARVED_STREAK", "10")),
)
_OUTPUT_SCHEDULE_OVERFLOW_STREAK_THRESHOLD = max(
    1,
    int(os.environ.get("VP_OUTPUT_BUFFER_OVERFLOW_STREAK", "12")),
)
_OUTPUT_SCHEDULE_AUTO_REPRIME_MIN_INTERVAL_S = max(
    0.05,
    float(os.environ.get("VP_OUTPUT_BUFFER_AUTO_REPRIME_MIN_INTERVAL_S", "2.0")),
)
_OUTPUT_SCHEDULE_ENABLE_HEALTH_POLLING = os.environ.get("VP_OUTPUT_BUFFER_HEALTH_POLLING", "0") == "1"
_OUTPUT_SCHEDULE_ENABLE_AUTO_REPRIME = os.environ.get("VP_OUTPUT_BUFFER_AUTO_REPRIME", "0") == "1"
_OUTPUT_SCHEDULE_HEALTH_SAMPLE_EVERY = max(
    1,
    int(os.environ.get("VP_OUTPUT_BUFFER_HEALTH_SAMPLE_EVERY", "30")),
)
_OUTPUT_SCHEDULE_AUTO_REPRIME_ON_OVERFLOW = os.environ.get("VP_OUTPUT_BUFFER_AUTO_REPRIME_ON_OVERFLOW", "0") == "1"
_OUTPUT_SCHEDULE_LOCAL_OVERFLOW_HEADROOM_FRAMES = max(
    1,
    int(os.environ.get("VP_OUTPUT_BUFFER_LOCAL_OVERFLOW_HEADROOM_FRAMES", "2")),
)


def _decklink_timecode_format_name(format_code: object) -> str:
    try:
        code = int(format_code) & 0xFFFFFFFF
    except Exception:
        return ""

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
    return format_map.get(code, f"0x{code:08X}")


def _extract_frame_timecode_info(frame: object) -> dict[str, object]:
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
        return payload
    except Exception:
        return payload
_OUTPUT_SCHEDULE_STATE: dict[int, dict[str, object]] = {}
_OUTPUT_SCHEDULE_TARGET_BUFFER_FRAMES: dict[int, int] = {}


def _looks_zeroed_uyvy_frame(frame_bytes: bytes) -> bool:
    expected_size = UYVY_ROW_BYTES * FRAME_H
    if len(frame_bytes) != expected_size:
        return False

    # Sample sparsely to keep per-frame validation cheap in live mode.
    sample = np.frombuffer(frame_bytes, dtype=np.uint8)[::4096]
    return sample.size > 0 and int(np.count_nonzero(sample)) == 0


def _freeze_frame_bytes(frame_bytes: bytes | bytearray | memoryview) -> bytes:
    if isinstance(frame_bytes, bytes):
        return frame_bytes
    return bytes(frame_bytes)


def _has_effective_subpixel_shift(shift_x: float, shift_y: float) -> bool:
    eps = float(_SUBPIXEL_SHIFT_APPLY_EPS)
    return abs(float(shift_x)) >= eps or abs(float(shift_y)) >= eps

RTX_POST_SCALE_METHOD_TO_CV2_INTERP = {
    "nearest": cv2.INTER_NEAREST if cv2 is not None else 0,
    "bilinear": cv2.INTER_LINEAR if cv2 is not None else 1,
    "bicubic": cv2.INTER_CUBIC if cv2 is not None else 2,
    "lanczos": cv2.INTER_LANCZOS4 if cv2 is not None else 4,
}

AI_SR_POST_DENOISE_METHODS = {
    "off",
    "luma_gaussian3x3",
    "luma_median3x3",
    "luma_bilateral3x3",
    "luma_bilateral5x5",
}

AI_SR_POST_ARTIFACT_REDUCTION_METHODS = {
    "off",
    "luma_bilateral3x3",
    "luma_bilateral5x5",
}


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


def _normalize_ai_sr_post_denoise_method(method_name: str) -> str:
    normalized = str(method_name).strip().lower().replace(" ", "").replace("-", "_")
    if normalized in {"none", "off"}:
        return "off"
    if normalized in AI_SR_POST_DENOISE_METHODS:
        return normalized
    return "off"


def _normalize_ai_sr_post_artifact_reduction_method(method_name: str) -> str:
    normalized = str(method_name).strip().lower().replace(" ", "").replace("-", "_")
    if normalized in {"none", "off"}:
        return "off"
    if normalized in AI_SR_POST_ARTIFACT_REDUCTION_METHODS:
        return normalized
    return "off"


def _clip_u8_round(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8)


def _uyvy_to_rgb_bt709_limited(yuv422: np.ndarray) -> np.ndarray:
    return _uyvy_to_rgb_limited(yuv422, "rec709", "limited")


def _uyvy_to_rgb_limited(yuv422: np.ndarray, color_space: str, color_range: str = "limited") -> np.ndarray:
    if yuv422.ndim != 3 or yuv422.shape[2] != 2:
        raise RuntimeError(f"Expected UYVY array shape [H, W, 2], got {tuple(yuv422.shape)}")

    h, w, _ = yuv422.shape
    if (w & 1) != 0:
        raise RuntimeError(f"UYVY width must be even, got {w}")

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
            r0 = _clip_u8_round(c0 + 1.474600 * e)
            g0 = _clip_u8_round(c0 - 0.164553 * d - 0.571353 * e)
            b0 = _clip_u8_round(c0 + 1.881400 * d)

            r1 = _clip_u8_round(c1 + 1.474600 * e)
            g1 = _clip_u8_round(c1 - 0.164553 * d - 0.571353 * e)
            b1 = _clip_u8_round(c1 + 1.881400 * d)
        else:
            r0 = _clip_u8_round(c0 + 1.574800 * e)
            g0 = _clip_u8_round(c0 - 0.187324 * d - 0.468124 * e)
            b0 = _clip_u8_round(c0 + 1.855600 * d)

            r1 = _clip_u8_round(c1 + 1.574800 * e)
            g1 = _clip_u8_round(c1 - 0.187324 * d - 0.468124 * e)
            b1 = _clip_u8_round(c1 + 1.855600 * d)
    else:
        c0 = y0 - 16.0
        c1 = y1 - 16.0
        if cs == "rec2020_hlg":
            r0 = _clip_u8_round(1.164383 * c0 + 1.678674 * e)
            g0 = _clip_u8_round(1.164383 * c0 - 0.187326 * d - 0.650424 * e)
            b0 = _clip_u8_round(1.164383 * c0 + 2.141772 * d)

            r1 = _clip_u8_round(1.164383 * c1 + 1.678674 * e)
            g1 = _clip_u8_round(1.164383 * c1 - 0.187326 * d - 0.650424 * e)
            b1 = _clip_u8_round(1.164383 * c1 + 2.141772 * d)
        else:
            r0 = _clip_u8_round(1.164383 * c0 + 1.792741 * e)
            g0 = _clip_u8_round(1.164383 * c0 - 0.213249 * d - 0.532909 * e)
            b0 = _clip_u8_round(1.164383 * c0 + 2.112402 * d)

            r1 = _clip_u8_round(1.164383 * c1 + 1.792741 * e)
            g1 = _clip_u8_round(1.164383 * c1 - 0.213249 * d - 0.532909 * e)
            b1 = _clip_u8_round(1.164383 * c1 + 2.112402 * d)

    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:, 0::2, 0] = r0
    rgb[:, 0::2, 1] = g0
    rgb[:, 0::2, 2] = b0
    rgb[:, 1::2, 0] = r1
    rgb[:, 1::2, 1] = g1
    rgb[:, 1::2, 2] = b1
    return rgb


def _rgb_to_uyvy_bt709_limited(rgb: np.ndarray) -> np.ndarray:
    return _rgb_to_uyvy_limited(rgb, "rec709", "limited")


def _rgb_to_uyvy_limited(rgb: np.ndarray, color_space: str, color_range: str = "limited") -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(f"Expected RGB array shape [H, W, 3], got {tuple(rgb.shape)}")

    h, w, _ = rgb.shape
    if (w & 1) != 0:
        raise RuntimeError(f"RGB width must be even for UYVY conversion, got {w}")

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    cs = _normalize_color_space_name(color_space)
    cr = _normalize_color_range_name(color_range)
    if cr == "full":
        if cs == "rec2020_hlg":
            y = _clip_u8_round(0.262700 * r + 0.678000 * g + 0.059300 * b)
            u = np.clip(128.0 - 0.139630 * r - 0.360370 * g + 0.500000 * b, 0.0, 255.0)
            v = np.clip(128.0 + 0.500000 * r - 0.459786 * g - 0.040214 * b, 0.0, 255.0)
        else:
            y = _clip_u8_round(0.212600 * r + 0.715200 * g + 0.072200 * b)
            u = np.clip(128.0 - 0.114572 * r - 0.385428 * g + 0.500000 * b, 0.0, 255.0)
            v = np.clip(128.0 + 0.500000 * r - 0.454153 * g - 0.045847 * b, 0.0, 255.0)
    else:
        if cs == "rec2020_hlg":
            y = _clip_u8_round(16.0 + 0.225613 * r + 0.582282 * g + 0.050928 * b)
            u = np.clip(128.0 - 0.122655 * r - 0.316561 * g + 0.439216 * b, 0.0, 255.0)
            v = np.clip(128.0 + 0.439216 * r - 0.403890 * g - 0.035325 * b, 0.0, 255.0)
        else:
            y = _clip_u8_round(16.0 + 0.182586 * r + 0.614231 * g + 0.062007 * b)
            u = np.clip(128.0 - 0.100644 * r - 0.338572 * g + 0.439216 * b, 0.0, 255.0)
            v = np.clip(128.0 + 0.439216 * r - 0.398942 * g - 0.040274 * b, 0.0, 255.0)

    y0 = y[:, 0::2]
    y1 = y[:, 1::2]
    u_pair = _clip_u8_round((u[:, 0::2] + u[:, 1::2]) * 0.5)
    v_pair = _clip_u8_round((v[:, 0::2] + v[:, 1::2]) * 0.5)

    packed = np.empty((h, w // 2, 4), dtype=np.uint8)
    packed[:, :, 0] = u_pair
    packed[:, :, 1] = y0
    packed[:, :, 2] = v_pair
    packed[:, :, 3] = y1
    return packed.reshape(h, w, 2)

def _apply_subpixel_shift_uyvy(frame_bytes: bytes, shift_x: float, shift_y: float) -> bytes:
    if cv2 is None:
        return frame_bytes
    if not _has_effective_subpixel_shift(shift_x, shift_y):
        return frame_bytes
    if len(frame_bytes) != (UYVY_ROW_BYTES * FRAME_H):
        return frame_bytes

    yuv422 = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 2)
    packed = yuv422.reshape(FRAME_H, FRAME_W // 2, 4)

    y_plane = np.empty((FRAME_H, FRAME_W), dtype=np.uint8)
    y_plane[:, 0::2] = packed[:, :, 1]
    y_plane[:, 1::2] = packed[:, :, 3]
    uv_plane = np.empty((FRAME_H, FRAME_W // 2, 2), dtype=np.uint8)
    uv_plane[:, :, 0] = packed[:, :, 0]
    uv_plane[:, :, 1] = packed[:, :, 2]

    mat_y = np.array([[1.0, 0.0, float(shift_x)], [0.0, 1.0, float(shift_y)]], dtype=np.float32)
    mat_uv = np.array([[1.0, 0.0, float(shift_x) * 0.5], [0.0, 1.0, float(shift_y)]], dtype=np.float32)

    y_shifted = cv2.warpAffine(
        y_plane,
        mat_y,
        (FRAME_W, FRAME_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    uv_shifted = cv2.warpAffine(
        uv_plane,
        mat_uv,
        (FRAME_W // 2, FRAME_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    out = np.empty((FRAME_H, FRAME_W // 2, 4), dtype=np.uint8)
    out[:, :, 0] = uv_shifted[:, :, 0]
    out[:, :, 1] = y_shifted[:, 0::2]
    out[:, :, 2] = uv_shifted[:, :, 1]
    out[:, :, 3] = y_shifted[:, 1::2]
    return out.tobytes()


def _apply_interlaced_field_phase_shift_uyvy(
    frame_bytes: bytes,
    field0_shift_x: float,
    field0_shift_y: float,
    field1_shift_x: float,
    field1_shift_y: float,
) -> bytes:
    if cv2 is None:
        return frame_bytes
    if len(frame_bytes) != (UYVY_ROW_BYTES * FRAME_H):
        return frame_bytes

    yuv422 = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 2)
    packed = yuv422.reshape(FRAME_H, FRAME_W // 2, 4)

    y_plane = np.empty((FRAME_H, FRAME_W), dtype=np.uint8)
    y_plane[:, 0::2] = packed[:, :, 1]
    y_plane[:, 1::2] = packed[:, :, 3]
    uv_plane = np.empty((FRAME_H, FRAME_W // 2, 2), dtype=np.uint8)
    uv_plane[:, :, 0] = packed[:, :, 0]
    uv_plane[:, :, 1] = packed[:, :, 2]

    def _warp_planes(shift_x: float, shift_y: float) -> tuple[np.ndarray, np.ndarray]:
        mat_y = np.array([[1.0, 0.0, float(shift_x)], [0.0, 1.0, float(shift_y)]], dtype=np.float32)
        mat_uv = np.array([[1.0, 0.0, float(shift_x) * 0.5], [0.0, 1.0, float(shift_y)]], dtype=np.float32)
        y_shifted_local = cv2.warpAffine(
            y_plane,
            mat_y,
            (FRAME_W, FRAME_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        uv_shifted_local = cv2.warpAffine(
            uv_plane,
            mat_uv,
            (FRAME_W // 2, FRAME_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return y_shifted_local, uv_shifted_local

    y_f0, uv_f0 = _warp_planes(float(field0_shift_x), float(field0_shift_y))
    y_f1, uv_f1 = _warp_planes(float(field1_shift_x), float(field1_shift_y))

    y_out = np.empty_like(y_plane)
    uv_out = np.empty_like(uv_plane)
    y_out[0::2, :] = y_f0[0::2, :]
    y_out[1::2, :] = y_f1[1::2, :]
    uv_out[0::2, :, :] = uv_f0[0::2, :, :]
    uv_out[1::2, :, :] = uv_f1[1::2, :, :]

    out = np.empty((FRAME_H, FRAME_W // 2, 4), dtype=np.uint8)
    out[:, :, 0] = uv_out[:, :, 0]
    out[:, :, 1] = y_out[:, 0::2]
    out[:, :, 2] = uv_out[:, :, 1]
    out[:, :, 3] = y_out[:, 1::2]
    return out.tobytes()


def _collapse_interlaced_to_single_field_uyvy(frame_bytes: bytes, lower_field_first: bool) -> bytes:
    if len(frame_bytes) != (UYVY_ROW_BYTES * FRAME_H):
        return frame_bytes
    plane = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(FRAME_H, UYVY_ROW_BYTES)
    out = np.empty_like(plane)
    # Neutral interlaced ROI mode: duplicate one temporal field into both line
    # parities so ROI motion does not exhibit field-offset rendering.
    if lower_field_first:
        out[0::2, :] = plane[1::2, :]
        out[1::2, :] = plane[1::2, :]
    else:
        out[0::2, :] = plane[0::2, :]
        out[1::2, :] = plane[0::2, :]
    return out.tobytes()


class AiSrOnnxEngine:
    def __init__(
        self,
        model_path: str,
        provider: str = "cpu",
        trt_precision: str = "fp16",
        trt_engine_cache_path: str | None = None,
        require_gpu: bool = False,
        input_align: int = 2,
        roi_overscan_percent: float = 0.0,
        inference_divisor: int = 0,
        detail_preserve_percent: float = 0.0,
        post_denoise_method: str = "off",
        post_denoise_strength: float = 0.0,
        post_artifact_reduction_method: str = "off",
        post_artifact_reduction_strength: float = 0.0,
        post_exaggeration_enabled: bool = False,
        post_exaggeration_gain: float = 2.0,
        color_space: str = "rec709",
        color_range: str = "limited",
        native_module: Any | None = None,
        native_processor: Any | None = None,
    ) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime is not installed")
        if cv2 is None:
            raise RuntimeError("opencv-python is required for AI SR color conversion")
        if not model_path:
            raise RuntimeError("AI SR model path is empty")

        model_file = Path(model_path)
        if not model_file.exists():
            raise RuntimeError(f"AI SR model file not found: {model_file}")

        provider_name = provider.lower()
        trt_precision_name = str(trt_precision).strip().lower()
        if provider_name == "trt_int8":
            provider_name = "trt"
            trt_precision_name = "int8"
        elif provider_name == "trt_fp16":
            provider_name = "trt"
            trt_precision_name = "fp16"
        if trt_precision_name not in {"fp16", "int8"}:
            trt_precision_name = "fp16"

        if require_gpu or provider_name in {"cuda", "auto", "trt", "tensorrt"}:
            _prepare_cuda_runtime_dll_paths()
            preload_dlls = getattr(ort, "preload_dlls", None)
            if callable(preload_dlls):
                try:
                    preload_dlls()
                except Exception as exc:
                    raise RuntimeError(f"Failed to preload ONNX Runtime CUDA DLLs: {exc}") from exc

        available_providers = set(ort.get_available_providers())
        available_providers_sorted = sorted(available_providers)
        providers: list[object] = ["CPUExecutionProvider"]
        cuda_available = "CUDAExecutionProvider" in available_providers
        trt_available = "TensorrtExecutionProvider" in available_providers

        cuda_provider_options = {
            "do_copy_in_default_stream": "True",
            "cudnn_conv_use_max_workspace": "True",
        }
        trt_provider_options = {
            "trt_fp16_enable": "True" if trt_precision_name != "int8" else "False",
            "trt_int8_enable": "True" if trt_precision_name == "int8" else "False",
            "trt_engine_cache_enable": "True",
            "trt_timing_cache_enable": "True",
        }
        if trt_engine_cache_path:
            trt_cache_dir = Path(trt_engine_cache_path)
            trt_cache_dir.mkdir(parents=True, exist_ok=True)
            trt_provider_options["trt_engine_cache_path"] = str(trt_cache_dir)

        if provider_name in {"trt", "tensorrt"} and not trt_available:
            raise RuntimeError(
                f"TensorrtExecutionProvider is not available in onnxruntime. Available providers: {available_providers_sorted}"
            )

        if provider_name in {"trt", "tensorrt"}:
            _preflight_tensorrt_runtime()

        if provider_name == "cuda" and not cuda_available:
            raise RuntimeError(
                f"CUDAExecutionProvider is not available in onnxruntime. Available providers: {available_providers_sorted}"
            )

        if require_gpu and not cuda_available:
            raise RuntimeError(
                f"GPU is required for AI SR, but CUDAExecutionProvider is unavailable. Available providers: {available_providers_sorted}"
            )

        if require_gpu:
            # Enforce single-provider GPU session creation to prevent hidden fallback.
            if provider_name in {"trt", "tensorrt"} and trt_available:
                providers = [("TensorrtExecutionProvider", trt_provider_options)]
            else:
                providers = [("CUDAExecutionProvider", cuda_provider_options)]
        elif provider_name in {"trt", "tensorrt"}:
            providers = [("TensorrtExecutionProvider", trt_provider_options)]
        elif provider_name == "auto":
            if trt_available:
                _preflight_tensorrt_runtime()
                providers = [("TensorrtExecutionProvider", trt_provider_options)]
            elif cuda_available:
                providers = [("CUDAExecutionProvider", cuda_provider_options)]
            else:
                raise RuntimeError(
                    f"No GPU execution provider is available for AI SR auto mode. Available providers: {available_providers_sorted}"
                )
        elif provider_name == "cuda" and cuda_available:
            providers = [("CUDAExecutionProvider", cuda_provider_options)]
        elif provider_name == "cpu":
            raise RuntimeError("CPU provider is not supported: legacy CPU ONNX output path has been removed")

        first_provider = providers[0]
        first_provider_name = first_provider[0] if isinstance(first_provider, tuple) else str(first_provider)
        if require_gpu and first_provider_name not in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
            raise RuntimeError(
                f"GPU is required for AI SR, but selected provider is '{first_provider_name}'. Available providers: {available_providers_sorted}"
            )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(model_file), providers=providers, sess_options=session_options)
        inputs = self._session.get_inputs()
        if not inputs:
            raise RuntimeError("AI SR model has no inputs")
        self._input_name = inputs[0].name
        input_type = str(getattr(inputs[0], "type", "")).lower()
        if "float16" in input_type:
            self._input_dtype = np.float16
        elif "float" in input_type:
            self._input_dtype = np.float32
        else:
            raise RuntimeError(f"Unsupported AI SR input tensor type: {input_type}")

        outputs = self._session.get_outputs()
        if not outputs:
            raise RuntimeError("AI SR model has no outputs")
        self._output_name = outputs[0].name
        output_type = str(getattr(outputs[0], "type", "")).lower()
        if "float16" in output_type:
            self._output_dtype = np.float16
        elif "float" in output_type:
            self._output_dtype = np.float32
        elif "uint8" in output_type:
            self._output_dtype = np.uint8
        else:
            self._output_dtype = np.float32

        self._model_path = str(model_file)
        session_providers = self._session.get_providers()
        self._provider = session_providers[0] if session_providers else "CPUExecutionProvider"
        if require_gpu and self._provider not in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
            raise RuntimeError(
                f"GPU is required for AI SR, but onnxruntime session selected '{self._provider}'. "
                f"requested_providers={providers}, session_providers={session_providers}"
            )

        if self._provider not in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
            raise RuntimeError(
                "Legacy CPU ONNX output path has been removed. "
                f"Selected provider '{self._provider}' is not GPU-backed."
            )

        self._model_scale = self._detect_model_scale()
        self._input_w = max(1, FRAME_W // self._model_scale)
        self._input_h = max(1, FRAME_H // self._model_scale)
        self._io_binding_enabled = self._provider in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
        self._io_binding_error: str | None = None
        self._avg_infer_ms: float | None = None
        self._avg_prep_ms: float | None = None
        self._avg_post_ms: float | None = None
        self._avg_total_ms: float | None = None
        self._last_stage_ms: dict[str, float] = {
            "prep": 0.0,
            "infer": 0.0,
            "post": 0.0,
            "total": 0.0,
        }
        self._timing_warmup_frames = int(_AI_SR_TIMING_WARMUP_FRAMES)
        self._timing_warmup_remaining = int(_AI_SR_TIMING_WARMUP_FRAMES)
        self._timing_samples = 0
        self._available_providers = available_providers_sorted
        self._requested_provider = provider_name
        self._trt_precision = trt_precision_name
        self._trt_engine_cache_path = trt_provider_options.get("trt_engine_cache_path", "")
        self._require_gpu = bool(require_gpu)
        self._input_align = max(1, int(input_align))
        self._roi_overscan_percent = max(0.0, min(100.0, float(roi_overscan_percent)))
        self._inference_divisor = max(0, int(inference_divisor))
        self._detail_preserve_requested_percent = max(0.0, min(100.0, float(detail_preserve_percent)))
        self._detail_preserve_percent = 0.0
        self._post_denoise_method = _normalize_ai_sr_post_denoise_method(post_denoise_method)
        self._post_denoise_strength = max(0.0, min(1.0, float(post_denoise_strength)))
        self._post_artifact_reduction_method = _normalize_ai_sr_post_artifact_reduction_method(post_artifact_reduction_method)
        self._post_artifact_reduction_strength = max(0.0, min(1.0, float(post_artifact_reduction_strength)))
        self._post_exaggeration_enabled = bool(post_exaggeration_enabled)
        self._post_exaggeration_gain = max(1.0, min(4.0, float(post_exaggeration_gain)))
        self._color_space = _normalize_color_space_name(color_space)
        self._color_range = _normalize_color_range_name(color_range)
        self._native_processor = native_processor
        self._native_preprocess_available = bool(
            native_processor is not None and hasattr(native_processor, "process_frame_preprocess_roi_rgb")
        )
        self._native_gpu_input_available = bool(
            native_processor is not None and hasattr(native_processor, "process_frame_preprocess_roi_tensor_cuda")
        )

        if native_module is None or not hasattr(native_module, "AiSrCudaPostProcessor"):
            raise RuntimeError(
                "video_processor.AiSrCudaPostProcessor is required for zero-copy AI SR output path. "
                "Rebuild the native module with the updated architecture."
            )

        self._cuda_post = native_module.AiSrCudaPostProcessor(
            output_width=FRAME_W,
            output_height=FRAME_H,
            color_space=self._color_space,
            color_range=self._color_range,
        )
        self._cuda_post.set_post_denoise_method(self._post_denoise_method)
        self._cuda_post.set_post_denoise_strength(self._post_denoise_strength)
        self._cuda_post.set_post_artifact_reduction_method(self._post_artifact_reduction_method)
        self._cuda_post.set_post_artifact_reduction_strength(self._post_artifact_reduction_strength)
        self._cuda_post.set_post_exaggeration_enabled(self._post_exaggeration_enabled)
        self._cuda_post.set_post_exaggeration_gain(self._post_exaggeration_gain)
        if hasattr(self._cuda_post, "get_post_denoise_method"):
            self._post_denoise_method = str(self._cuda_post.get_post_denoise_method())
        if hasattr(self._cuda_post, "get_post_denoise_strength"):
            self._post_denoise_strength = float(self._cuda_post.get_post_denoise_strength())
        if hasattr(self._cuda_post, "get_post_artifact_reduction_method"):
            self._post_artifact_reduction_method = str(self._cuda_post.get_post_artifact_reduction_method())
        if hasattr(self._cuda_post, "get_post_artifact_reduction_strength"):
            self._post_artifact_reduction_strength = float(self._cuda_post.get_post_artifact_reduction_strength())
        if hasattr(self._cuda_post, "get_post_exaggeration_enabled"):
            self._post_exaggeration_enabled = bool(self._cuda_post.get_post_exaggeration_enabled())
        if hasattr(self._cuda_post, "get_post_exaggeration_gain"):
            self._post_exaggeration_gain = float(self._cuda_post.get_post_exaggeration_gain())

    def avg_infer_ms(self) -> float | None:
        if self._avg_infer_ms is None:
            return None
        return float(self._avg_infer_ms)

    def uses_native_preprocess(self) -> bool:
        return bool(self._native_preprocess_available)

    def timing_info(self) -> dict[str, object]:
        return {
            "avg_prep_ms": None if self._avg_prep_ms is None else float(self._avg_prep_ms),
            "avg_infer_ms": None if self._avg_infer_ms is None else float(self._avg_infer_ms),
            "avg_post_ms": None if self._avg_post_ms is None else float(self._avg_post_ms),
            "avg_total_ms": None if self._avg_total_ms is None else float(self._avg_total_ms),
            "last_prep_ms": float(self._last_stage_ms.get("prep", 0.0)),
            "last_infer_ms": float(self._last_stage_ms.get("infer", 0.0)),
            "last_post_ms": float(self._last_stage_ms.get("post", 0.0)),
            "last_total_ms": float(self._last_stage_ms.get("total", 0.0)),
            "io_binding_enabled": bool(self._io_binding_enabled),
            "io_binding_error": self._io_binding_error,
            "timing_warmup_frames": int(self._timing_warmup_frames),
            "timing_warmup_remaining": int(self._timing_warmup_remaining),
            "timing_samples": int(self._timing_samples),
        }

    def _update_ema(self, attr_name: str, sample_ms: float) -> None:
        value = max(0.0, float(sample_ms))
        prev = getattr(self, attr_name)
        if prev is None:
            setattr(self, attr_name, value)
            return
        setattr(self, attr_name, (0.9 * float(prev)) + (0.1 * value))

    def _effective_inference_divisor(self) -> int:
        if self._model_scale <= 1:
            return 1
        if self._inference_divisor <= 0:
            # Auto mode must still keep x2/x4/x8 models usable at real-time frame rates.
            # Full-res x2 inference is the current throughput killer; it collapses the
            # frame rate by running the model on ~1920x1080 RGB data instead of a
            # 960x540 (or smaller) working input.
            if self._model_scale >= 8:
                return 4
            if self._model_scale >= 4:
                return 2
            if self._model_scale >= 2:
                return 2
            return 1
        return max(1, min(self._model_scale, self._inference_divisor))

    def _normalize_roi(self, roi: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        roi_x, roi_y, roi_w, roi_h = [int(v) for v in roi]
        roi_w = max(2, min(roi_w, FRAME_W))
        roi_h = max(2, min(roi_h, FRAME_H))

        # UYVY is 4:2:2 packed, so x and width must remain even.
        roi_w &= ~1
        if roi_w < 2:
            roi_w = 2

        max_x = max(0, FRAME_W - roi_w)
        max_y = max(0, FRAME_H - roi_h)
        roi_x = max(0, min(roi_x, max_x))
        roi_y = max(0, min(roi_y, max_y))

        roi_x &= ~1
        if roi_x > max_x:
            roi_x = max(0, max_x & ~1)

        return roi_x, roi_y, roi_w, roi_h

    def _expand_roi_to_model_safe_min(self, roi: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        roi_x, roi_y, roi_w, roi_h = self._normalize_roi(roi)

        # Ensure downscaled model input is not too tiny for architectures with
        # reshape/pixel-unshuffle style constraints.
        divisor = self._effective_inference_divisor()
        min_model_dim = max(8, int(self._input_align))
        min_roi_w = max(2, min(FRAME_W, min_model_dim * divisor))
        min_roi_h = max(2, min(FRAME_H, min_model_dim * divisor))
        min_roi_w &= ~1
        if min_roi_w < 2:
            min_roi_w = 2

        if roi_w >= min_roi_w and roi_h >= min_roi_h:
            return roi_x, roi_y, roi_w, roi_h

        cx = roi_x + (roi_w / 2.0)
        cy = roi_y + (roi_h / 2.0)
        new_w = max(roi_w, min_roi_w)
        new_h = max(roi_h, min_roi_h)

        new_x = int(round(cx - (new_w / 2.0)))
        new_y = int(round(cy - (new_h / 2.0)))

        new_x = max(0, min(new_x, FRAME_W - new_w))
        new_y = max(0, min(new_y, FRAME_H - new_h))
        new_x &= ~1
        if new_x + new_w > FRAME_W:
            new_x = max(0, FRAME_W - new_w)
            new_x &= ~1

        return self._normalize_roi((new_x, new_y, new_w, new_h))

    def _detect_model_scale(self) -> int:
        # Probe with a small tensor to infer the model's upscaling factor.
        probe_h = 64
        probe_w = 64
        x = np.zeros((1, 3, probe_h, probe_w), dtype=self._input_dtype)
        outputs = self._session.run(None, {self._input_name: x})
        if not outputs:
            return 1

        y = outputs[0]
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        if y.ndim == 4:
            out_h = int(y.shape[2])
            out_w = int(y.shape[3])
        elif y.ndim == 3:
            # CHW or HWC
            if y.shape[0] in (1, 3):
                out_h = int(y.shape[1])
                out_w = int(y.shape[2])
            else:
                out_h = int(y.shape[0])
                out_w = int(y.shape[1])
        else:
            return 1

        scale_h = max(1, int(round(out_h / float(probe_h))))
        scale_w = max(1, int(round(out_w / float(probe_w))))
        return max(1, min(scale_h, scale_w))

    def info(self) -> dict[str, object]:
        return {
            "model_path": self._model_path,
            "provider": self._provider,
            "requested_provider": self._requested_provider,
            "trt_precision": self._trt_precision,
            "trt_engine_cache_path": self._trt_engine_cache_path,
            "available_providers": self._available_providers,
            "gpu_required": self._require_gpu,
            "input_dtype": "float16" if self._input_dtype == np.float16 else "float32",
            "model_scale": int(self._model_scale),
            "model_input_w": int(self._input_w),
            "model_input_h": int(self._input_h),
            "io_binding_enabled": bool(self._io_binding_enabled),
            "io_binding_error": self._io_binding_error,
            "avg_prep_ms": None if self._avg_prep_ms is None else float(self._avg_prep_ms),
            "avg_infer_ms": None if self._avg_infer_ms is None else float(self._avg_infer_ms),
            "avg_post_ms": None if self._avg_post_ms is None else float(self._avg_post_ms),
            "avg_total_ms": None if self._avg_total_ms is None else float(self._avg_total_ms),
            "input_align": int(self._input_align),
            "roi_overscan_percent": float(self._roi_overscan_percent),
            "inference_divisor": int(self._effective_inference_divisor()),
            "detail_preserve_percent": float(self._detail_preserve_percent),
            "detail_preserve_requested_percent": float(self._detail_preserve_requested_percent),
            "post_denoise_method": str(self._post_denoise_method),
            "post_denoise_strength": float(self._post_denoise_strength),
            "post_artifact_reduction_method": str(self._post_artifact_reduction_method),
            "post_artifact_reduction_strength": float(self._post_artifact_reduction_strength),
            "post_exaggeration_enabled": bool(self._post_exaggeration_enabled),
            "post_exaggeration_gain": float(self._post_exaggeration_gain),
            "post_exaggeration_passes": 3 if self._post_exaggeration_enabled else 1,
            "postprocess_gpu_chain": "resize/sharpen -> post_denoise(xN) -> post_artifact_reduction(xN) -> rgb_to_uyvy",
            "timing_warmup_frames": int(self._timing_warmup_frames),
            "timing_warmup_remaining": int(self._timing_warmup_remaining),
            "timing_samples": int(self._timing_samples),
            "onnx_output_copy_to_cpu": False,
            "onnx_zero_copy_cuda_postprocess": True,
            "native_gpu_preprocess_enabled": bool(self._native_preprocess_available),
            "onnx_input_gpu_direct_enabled": bool(self._native_gpu_input_available),
        }

    def _record_timing_sample(self, prep_ms: float, infer_ms: float, post_ms: float, total_ms: float) -> None:
        self._last_stage_ms = {
            "prep": float(prep_ms),
            "infer": float(infer_ms),
            "post": float(post_ms),
            "total": float(total_ms),
        }

        self._timing_samples += 1
        if self._timing_warmup_remaining > 0:
            self._timing_warmup_remaining -= 1
            return

        self._update_ema("_avg_prep_ms", prep_ms)
        self._update_ema("_avg_infer_ms", infer_ms)
        self._update_ema("_avg_post_ms", post_ms)
        self._update_ema("_avg_total_ms", total_ms)

    def _run_model_tensor(self, x: np.ndarray) -> object:
        if not self._io_binding_enabled:
            raise RuntimeError("Zero-copy AI SR requires ONNX Runtime CUDA/TensorRT I/O binding")

        try:
            input_ort = ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
            io_binding = self._session.io_binding()
            io_binding.bind_ortvalue_input(self._input_name, input_ort)
            io_binding.bind_output(self._output_name, "cuda", 0)
            self._session.run_with_iobinding(io_binding)

            get_outputs = getattr(io_binding, "get_outputs", None)
            if not callable(get_outputs):
                raise RuntimeError("ONNX Runtime build does not expose io_binding.get_outputs()")

            bound_outputs = get_outputs()
            if not bound_outputs:
                raise RuntimeError("I/O binding returned no outputs")

            self._io_binding_error = None
            return bound_outputs[0]
        except Exception as exc:
            self._io_binding_error = str(exc)
            raise RuntimeError(f"Zero-copy ONNX output binding failed: {exc}") from exc

    def _run_model_cuda_tensor(self, tensor: Any) -> object:
        if not self._io_binding_enabled:
            raise RuntimeError("Zero-copy AI SR requires ONNX Runtime CUDA/TensorRT I/O binding")

        try:
            tensor_width = int(getattr(tensor, "width"))
            tensor_height = int(getattr(tensor, "height"))
            tensor_channels = int(getattr(tensor, "channels"))
            tensor_layout = str(getattr(tensor, "layout")).strip().lower()
            tensor_dtype = str(getattr(tensor, "dtype")).strip().lower()
            tensor_ptr = int(getattr(tensor, "data_ptr"))

            if tensor_ptr == 0:
                raise RuntimeError("Native CUDA tensor pointer is null")
            if tensor_layout not in {"nchw", "chw"}:
                raise RuntimeError(f"Unsupported native CUDA tensor layout for ONNX input: {tensor_layout}")
            if tensor_channels != 3:
                raise RuntimeError(f"Unsupported native CUDA tensor channels for ONNX input: {tensor_channels}")

            if tensor_dtype in {"float16", "fp16", "half"}:
                element_type = np.float16
            elif tensor_dtype in {"float32", "fp32", "float"}:
                element_type = np.float32
            else:
                raise RuntimeError(f"Unsupported native CUDA tensor dtype for ONNX input: {tensor_dtype}")

            expected_dtype = np.float16 if self._input_dtype == np.float16 else np.float32
            if element_type is not expected_dtype:
                raise RuntimeError(
                    f"Native CUDA tensor dtype ({tensor_dtype}) does not match model input dtype "
                    f"({'float16' if expected_dtype == np.float16 else 'float32'})"
                )

            io_binding = self._session.io_binding()
            io_binding.bind_input(
                name=self._input_name,
                device_type="cuda",
                device_id=0,
                element_type=element_type,
                shape=[1, tensor_channels, tensor_height, tensor_width],
                buffer_ptr=tensor_ptr,
            )
            io_binding.bind_output(self._output_name, "cuda", 0)
            self._session.run_with_iobinding(io_binding)

            get_outputs = getattr(io_binding, "get_outputs", None)
            if not callable(get_outputs):
                raise RuntimeError("ONNX Runtime build does not expose io_binding.get_outputs()")

            bound_outputs = get_outputs()
            if not bound_outputs:
                raise RuntimeError("I/O binding returned no outputs")

            self._io_binding_error = None
            return bound_outputs[0]
        except Exception as exc:
            self._io_binding_error = str(exc)
            raise RuntimeError(f"Zero-copy ONNX input/output binding failed: {exc}") from exc

    def _ort_output_descriptor(self, output_ort: object) -> dict[str, object]:
        shape_attr = getattr(output_ort, "shape", None)
        if callable(shape_attr):
            shape = tuple(int(v) for v in shape_attr())
        elif isinstance(shape_attr, (list, tuple)):
            shape = tuple(int(v) for v in shape_attr)
        else:
            raise RuntimeError("Unable to inspect ONNX output shape from OrtValue")

        if not shape:
            raise RuntimeError("ONNX output shape is empty")

        layout = "nchw"
        channels = 3
        out_h = 0
        out_w = 0

        if len(shape) == 4:
            n, c, h, w = [int(v) for v in shape]
            if n != 1:
                raise RuntimeError(f"Unsupported ONNX batch size for zero-copy path: {shape}")
            if c in (1, 3):
                layout = "nchw"
                channels = c
                out_h = h
                out_w = w
            elif int(shape[3]) in (1, 3):
                layout = "hwc"
                channels = int(shape[3])
                out_h = int(shape[1])
                out_w = int(shape[2])
            else:
                raise RuntimeError(f"Unsupported ONNX output shape for zero-copy path: {shape}")
        elif len(shape) == 3:
            if int(shape[0]) in (1, 3) and int(shape[2]) not in (1, 3):
                layout = "nchw"
                channels = int(shape[0])
                out_h = int(shape[1])
                out_w = int(shape[2])
            elif int(shape[2]) in (1, 3):
                layout = "hwc"
                channels = int(shape[2])
                out_h = int(shape[0])
                out_w = int(shape[1])
            else:
                raise RuntimeError(f"Unsupported ONNX output shape for zero-copy path: {shape}")
        else:
            raise RuntimeError(f"Unsupported ONNX output rank for zero-copy path: {shape}")

        dtype_name = "float32"
        normalized_01 = True
        if self._output_dtype == np.float16:
            dtype_name = "float16"
            normalized_01 = True
        elif self._output_dtype == np.float32:
            dtype_name = "float32"
            normalized_01 = True
        elif self._output_dtype == np.uint8:
            dtype_name = "uint8"
            normalized_01 = False

        if out_w <= 0 or out_h <= 0:
            raise RuntimeError(f"Invalid ONNX output dimensions for zero-copy path: {shape}")

        return {
            "shape": shape,
            "layout": layout,
            "channels": channels,
            "width": int(out_w),
            "height": int(out_h),
            "dtype": dtype_name,
            "normalized_01": bool(normalized_01),
        }

    def _run_model_on_rgb(self, model_rgb: np.ndarray, method: str) -> bytes:
        prep_start = time.perf_counter()

        # Some SR models contain reshape/pixel-unshuffle paths that require specific
        # spatial alignment. Align to configured multiples before inference.
        in_h, in_w = int(model_rgb.shape[0]), int(model_rgb.shape[1])
        align = max(1, int(self._input_align))
        aligned_w = max(align, ((in_w + align - 1) // align) * align)
        aligned_h = max(align, ((in_h + align - 1) // align) * align)
        if aligned_w != in_w or aligned_h != in_h:
            model_rgb = cv2.resize(model_rgb, (aligned_w, aligned_h), interpolation=cv2.INTER_AREA)

        x = cv2.dnn.blobFromImage(
            model_rgb,
            scalefactor=(1.0 / 255.0),
            size=(aligned_w, aligned_h),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
            crop=False,
            ddepth=cv2.CV_32F,
        )
        if self._input_dtype == np.float16:
            x = x.astype(np.float16, copy=False)
        else:
            x = x.astype(np.float32, copy=False)

        prep_ms = (time.perf_counter() - prep_start) * 1000.0

        infer_start = time.perf_counter()
        output_ort = self._run_model_tensor(x)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        post_start = time.perf_counter()

        output_desc = self._ort_output_descriptor(output_ort)
        data_ptr_fn = getattr(output_ort, "data_ptr", None)
        if not callable(data_ptr_fn):
            raise RuntimeError("OrtValue does not expose data_ptr() for zero-copy CUDA postprocessing")

        output_ptr = int(data_ptr_fn())
        out_uyvy = self._cuda_post.process_onnx_output_cuda(
            tensor_ptr=output_ptr,
            tensor_width=int(output_desc["width"]),
            tensor_height=int(output_desc["height"]),
            method=str(method).strip().lower(),
            dtype=str(output_desc["dtype"]),
            layout=str(output_desc["layout"]),
            channels=int(output_desc["channels"]),
            normalized_01=bool(output_desc["normalized_01"]),
        )

        post_ms = (time.perf_counter() - post_start) * 1000.0
        total_ms = prep_ms + infer_ms + post_ms
        self._record_timing_sample(prep_ms, infer_ms, post_ms, total_ms)
        return out_uyvy

    def _run_model_on_cuda_tensor(self, model_tensor: Any, method: str) -> bytes:
        prep_start = time.perf_counter()
        prep_ms = (time.perf_counter() - prep_start) * 1000.0

        infer_start = time.perf_counter()
        output_ort = self._run_model_cuda_tensor(model_tensor)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        post_start = time.perf_counter()
        output_desc = self._ort_output_descriptor(output_ort)
        data_ptr_fn = getattr(output_ort, "data_ptr", None)
        if not callable(data_ptr_fn):
            raise RuntimeError("OrtValue does not expose data_ptr() for zero-copy CUDA postprocessing")

        output_ptr = int(data_ptr_fn())
        out_uyvy = self._cuda_post.process_onnx_output_cuda(
            tensor_ptr=output_ptr,
            tensor_width=int(output_desc["width"]),
            tensor_height=int(output_desc["height"]),
            method=str(method).strip().lower(),
            dtype=str(output_desc["dtype"]),
            layout=str(output_desc["layout"]),
            channels=int(output_desc["channels"]),
            normalized_01=bool(output_desc["normalized_01"]),
        )

        post_ms = (time.perf_counter() - post_start) * 1000.0
        total_ms = prep_ms + infer_ms + post_ms
        self._record_timing_sample(prep_ms, infer_ms, post_ms, total_ms)
        return out_uyvy

    def process_uyvy_frame(self, frame_bytes: bytes) -> bytes:
        return self.process_uyvy_frame_roi_to_output(
            frame_bytes,
            (0, 0, FRAME_W, FRAME_H),
            "bicubic",
        )

    def process_uyvy_frame_roi(self, frame_bytes: bytes, roi: tuple[int, int, int, int]) -> bytes:
        # Legacy ROI-only AI SR path removed. Keep method for compatibility and
        # force direct ROI-to-output rendering through the zero-copy path.
        return self.process_uyvy_frame_roi_to_output(frame_bytes, roi, "bicubic")

    def process_uyvy_frame_roi_to_output(self, frame_bytes: bytes, roi: tuple[int, int, int, int], method: str) -> bytes:
        if len(frame_bytes) != UYVY_ROW_BYTES * FRAME_H:
            raise RuntimeError(f"Unexpected UYVY frame size: {len(frame_bytes)}")

        roi_x, roi_y, roi_w, roi_h = self._expand_roi_to_model_safe_min(roi)

        overscan_scale = max(0.0, float(self._roi_overscan_percent)) / 100.0
        pad_x = int(round((roi_w * overscan_scale) * 0.5))
        pad_y = int(round((roi_h * overscan_scale) * 0.5))

        proc_x = max(0, roi_x - pad_x)
        proc_y = max(0, roi_y - pad_y)
        proc_w = min(FRAME_W - proc_x, roi_w + (pad_x * 2))
        proc_h = min(FRAME_H - proc_y, roi_h + (pad_y * 2))
        proc_w = max(2, proc_w & ~1)
        if proc_x + proc_w > FRAME_W:
            proc_x = max(0, FRAME_W - proc_w)
            proc_x &= ~1

        model_in_w = proc_w
        model_in_h = proc_h
        if self._model_scale > 1:
            divisor = self._effective_inference_divisor()
            model_in_w = max(1, (proc_w + divisor - 1) // divisor)
            model_in_h = max(1, (proc_h + divisor - 1) // divisor)

        align = max(1, int(self._input_align))
        aligned_w = max(align, ((int(model_in_w) + align - 1) // align) * align)
        aligned_h = max(align, ((int(model_in_h) + align - 1) // align) * align)

        if self._native_gpu_input_available and self._native_processor is not None:
            model_dtype = "float16" if self._input_dtype == np.float16 else "float32"
            model_tensor = self._native_processor.process_frame_preprocess_roi_tensor_cuda(
                frame_bytes,
                int(proc_x),
                int(proc_y),
                int(proc_w),
                int(proc_h),
                int(aligned_w),
                int(aligned_h),
                model_dtype,
            )
            # Keep tensor owner alive through run_with_iobinding by passing the object.
            return self._run_model_on_cuda_tensor(model_tensor, method)

        model_rgb: np.ndarray
        if self._native_preprocess_available and self._native_processor is not None:
            roi_rgb_bytes = self._native_processor.process_frame_preprocess_roi_rgb(
                frame_bytes,
                int(proc_x),
                int(proc_y),
                int(proc_w),
                int(proc_h),
                int(model_in_w),
                int(model_in_h),
            )
            model_rgb = np.frombuffer(roi_rgb_bytes, dtype=np.uint8).reshape(model_in_h, model_in_w, 3)
        else:
            yuv422 = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 2)
            roi_yuv = np.ascontiguousarray(yuv422[proc_y : proc_y + proc_h, proc_x : proc_x + proc_w, :])
            roi_rgb = _uyvy_to_rgb_limited(roi_yuv, self._color_space, self._color_range)

            model_rgb = roi_rgb
            if model_in_w != proc_w or model_in_h != proc_h:
                model_rgb = cv2.resize(roi_rgb, (model_in_w, model_in_h), interpolation=cv2.INTER_CUBIC)

        # ONNX output stays on GPU and is fed to native CUDA postprocess without
        # CPU tensor copy or OpenCV-based output conversion.
        return self._run_model_on_rgb(model_rgb, method)

def _load_video_processor_module(project_root: Path):
    venv_site = project_root / "venv" / "Lib" / "site-packages"
    if venv_site.exists():
        site.addsitedir(str(venv_site))

    preferred_paths = [
        project_root / "build" / "src" / "Release",
        project_root / "build" / "src" / "RelWithDebInfo",
        project_root / "build" / "src" / "Debug",
    ]

    for candidate in reversed(preferred_paths):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    import video_processor

    return video_processor


def _create_processor(module: Any, cfg: dict[str, Any]):
    enable_basic_scaling = bool(cfg.get("enable_basic_scaling", cfg.get("enable_placeholder_sr", True)))
    basic_scaling_manual = int(cfg.get("basic_scaling_manual", cfg.get("sr_manual_scale", 4)))
    basic_scaling_auto_mode = bool(cfg.get("basic_scaling_auto_mode", cfg.get("sr_auto_mode", True)))
    basic_scaling_method = str(cfg.get("basic_scaling_method", cfg.get("sr_flavor", "bilinear_sharp")))
    color_space = _normalize_color_space_name(str(cfg.get("color_space", "rec709")))
    color_range = _normalize_color_range_name(str(cfg.get("color_range", "limited")))
    max_auto_basic_scaling = int(cfg.get("max_auto_basic_scaling", cfg.get("max_auto_sr_scale", 4)))
    deinterlace_method = str(cfg.get("deinterlace_method", "bob"))
    denoise_method = str(cfg.get("denoise_method", "off"))
    denoise_strength = float(cfg.get("denoise_strength", 0.35))
    processor = module.VideoProcessor(
        width=int(cfg["width"]),
        height=int(cfg["height"]),
        roi_x=int(cfg["roi_x"]),
        roi_y=int(cfg["roi_y"]),
        roi_w=int(cfg["roi_w"]),
        roi_h=int(cfg["roi_h"]),
        enable_placeholder_sr=enable_basic_scaling,
        sr_scale=int(cfg["sr_scale"]),
    )
    processor.set_max_auto_sr_scale(max_auto_basic_scaling)
    basic_scaling_method_supported = hasattr(processor, "set_sr_flavor")
    if basic_scaling_method_supported:
        processor.set_sr_flavor(basic_scaling_method)
    if hasattr(processor, "set_color_space"):
        processor.set_color_space(color_space)
    if hasattr(processor, "set_color_range"):
        processor.set_color_range(color_range)
    processor.set_deinterlace_enabled(bool(cfg["deinterlace_enabled"]))
    if hasattr(processor, "set_deinterlace_method"):
        processor.set_deinterlace_method(deinterlace_method)
    if hasattr(processor, "set_denoise_method"):
        processor.set_denoise_method(denoise_method)
    if hasattr(processor, "set_denoise_strength"):
        processor.set_denoise_strength(max(0.0, min(1.0, denoise_strength)))
    # SR runtime mode APIs are invalid when basic scaling was disabled at construction.
    if enable_basic_scaling:
        if basic_scaling_auto_mode:
            processor.set_sr_mode_auto()
        else:
            processor.set_sr_scale_manual(basic_scaling_manual)
    return processor, basic_scaling_method_supported


def _tight_uyvy_bytes(frame: object) -> bytes:
    row_bytes = int(frame.row_bytes)
    if row_bytes < UYVY_ROW_BYTES:
        raise RuntimeError(f"Captured row_bytes {row_bytes} is smaller than expected {UYVY_ROW_BYTES}")

    raw = memoryview(frame)
    if row_bytes == UYVY_ROW_BYTES:
        return raw.tobytes()

    # Vectorized row-tightening avoids Python per-row copy overhead at 1080i60.
    raw_np = np.frombuffer(raw, dtype=np.uint8)
    expected_total = row_bytes * FRAME_H
    if raw_np.size < expected_total:
        raise RuntimeError(f"Captured frame buffer is smaller than expected ({raw_np.size} < {expected_total})")

    return raw_np[:expected_total].reshape(FRAME_H, row_bytes)[:, :UYVY_ROW_BYTES].tobytes()


def _estimate_output_schedule_buffered_frames(state: dict[str, object], now_ts: float | None = None) -> int:
    if now_ts is None:
        now_ts = time.perf_counter()

    if not bool(state.get("started", False)):
        return max(0, int(state.get("queued_before_start", 0)))

    frame_duration = int(state.get("frame_duration", 0))
    time_scale = int(state.get("time_scale", 0))
    display_time = int(state.get("display_time", 0))
    schedule_epoch_perf_ts = float(state.get("schedule_epoch_perf_ts", 0.0))

    if frame_duration <= 0 or time_scale <= 0:
        return max(0, int(state.get("queued_before_start", 0)))

    if schedule_epoch_perf_ts <= 0.0:
        return max(0, int(math.ceil(float(display_time) / float(frame_duration))))

    elapsed_ticks = max(0.0, (float(now_ts) - schedule_epoch_perf_ts) * float(time_scale))
    queued_ticks = max(0.0, float(display_time) - elapsed_ticks)
    return max(0, int(math.ceil(queued_ticks / float(frame_duration))))


def _write_frame_to_output(out: object, frame_bytes: bytes) -> bool:
    out_id = id(out)
    state = _OUTPUT_SCHEDULE_STATE.get(out_id)
    if state is None:
        schedule_fn = getattr(out, "schedule_frame_copy", None)
        start_fn = getattr(out, "start_scheduled_playback", None)
        buffered_fn = getattr(out, "buffered_video_frame_count", None)
        frame_duration = int(getattr(out, "frame_duration", 0)) if hasattr(out, "frame_duration") else 0
        time_scale = int(getattr(out, "time_scale", 0)) if hasattr(out, "time_scale") else 0
        frame_period_s = (float(frame_duration) / float(time_scale)) if frame_duration > 0 and time_scale > 0 else 0.0
        target_buffer_frames = max(0, min(10, int(_OUTPUT_SCHEDULE_TARGET_BUFFER_FRAMES.get(out_id, 2))))
        state = {
            "enabled": callable(schedule_fn) and callable(start_fn),
            "can_query_buffered": callable(buffered_fn),
            "started": False,
            "queued_before_start": 0,
            "display_time": 0,
            "frame_duration": frame_duration,
            "time_scale": time_scale,
            "frame_period_s": frame_period_s,
            "sync_next_emit_ts": 0.0,
            "schedule_epoch_perf_ts": 0.0,
            "target_buffer_frames": target_buffer_frames,
            "last_clock_resync_ts": 0.0,
            "padded_out": None,
            "last_buffered_count": -1,
            "starved_streak": 0,
            "overflow_streak": 0,
            "starvation_events": 0,
            "overflow_events": 0,
            "auto_reprime_events": 0,
            "last_reprime_reason": "",
            "last_reprime_ts": 0.0,
            "last_auto_reprime_ts": 0.0,
            "scheduled_frames": 0,
        }
        _OUTPUT_SCHEDULE_STATE[out_id] = state

    if out.row_bytes == UYVY_ROW_BYTES:
        payload = frame_bytes
    else:
        if out.row_bytes < UYVY_ROW_BYTES:
            raise RuntimeError(f"Output row_bytes {out.row_bytes} is smaller than expected {UYVY_ROW_BYTES}")

        out_row_bytes = int(out.row_bytes)
        padded = state.get("padded_out")
        if not isinstance(padded, np.ndarray) or padded.shape != (FRAME_H, out_row_bytes):
            padded = np.empty((FRAME_H, out_row_bytes), dtype=np.uint8)
            state["padded_out"] = padded

        src = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(FRAME_H, UYVY_ROW_BYTES)
        padded[:, :UYVY_ROW_BYTES] = src
        if out_row_bytes > UYVY_ROW_BYTES:
            padded[:, UYVY_ROW_BYTES:] = 0
        payload = padded.tobytes()

    if bool(state.get("enabled", False)):
        frame_duration = int(state.get("frame_duration", 0))
        time_scale = int(state.get("time_scale", 0))
        frame_period_s = float(state.get("frame_period_s", 0.0))
        if frame_duration > 0 and time_scale > 0:
            try:
                now_ts = time.perf_counter()
                target_start_frames = max(0, min(10, int(state.get("target_buffer_frames", 2))))
                estimated_buffered_before = _estimate_output_schedule_buffered_frames(state, now_ts)
                state["last_buffered_count"] = estimated_buffered_before

                if bool(state.get("started", False)) and estimated_buffered_before <= 0:
                    state["starved_streak"] = int(state.get("starved_streak", 0)) + 1
                else:
                    state["starved_streak"] = 0

                overflow_threshold = target_start_frames + _OUTPUT_SCHEDULE_LOCAL_OVERFLOW_HEADROOM_FRAMES
                if bool(state.get("started", False)) and estimated_buffered_before > overflow_threshold:
                    state["overflow_events"] = int(state.get("overflow_events", 0)) + 1
                    _reprime_output_schedule(out, reason="local_overflow")
                    state = _OUTPUT_SCHEDULE_STATE.get(out_id, state)

                since_last_reprime = now_ts - float(state.get("last_auto_reprime_ts", 0.0))
                if (
                    bool(state.get("started", False))
                    and target_start_frames > 0
                    and int(state.get("starved_streak", 0)) >= _OUTPUT_SCHEDULE_STARVED_STREAK_THRESHOLD
                    and since_last_reprime >= _OUTPUT_SCHEDULE_AUTO_REPRIME_MIN_INTERVAL_S
                ):
                    state["starvation_events"] = int(state.get("starvation_events", 0)) + 1
                    state["auto_reprime_events"] = int(state.get("auto_reprime_events", 0)) + 1
                    state["last_auto_reprime_ts"] = now_ts
                    _reprime_output_schedule(out, reason="local_starvation")
                    state = _OUTPUT_SCHEDULE_STATE.get(out_id, state)

                display_time = int(state.get("display_time", 0))
                out.schedule_frame_copy(
                    payload,
                    display_time,
                    frame_duration,
                    time_scale,
                )
                state["display_time"] = int(state.get("display_time", 0)) + frame_duration
                state["scheduled_frames"] = int(state.get("scheduled_frames", 0)) + 1

                if not bool(state.get("started", False)):
                    state["queued_before_start"] = int(state.get("queued_before_start", 0)) + 1
                    should_start = False
                    if target_start_frames <= 0:
                        should_start = True
                    if bool(state.get("can_query_buffered", False)):
                        try:
                            buffered_count = int(out.buffered_video_frame_count())
                            queued_before_start = int(state.get("queued_before_start", 0))
                            effective_buffered = max(buffered_count, queued_before_start)
                            should_start = effective_buffered >= target_start_frames
                        except Exception:
                            state["can_query_buffered"] = False
                    else:
                        should_start = int(state.get("queued_before_start", 0)) >= target_start_frames

                    if should_start:
                        out.start_scheduled_playback(0, time_scale, 1.0)
                        state["started"] = True
                        state["queued_before_start"] = 0
                        state["schedule_epoch_perf_ts"] = now_ts
                        state["last_clock_resync_ts"] = now_ts
                        state["last_buffered_count"] = _estimate_output_schedule_buffered_frames(state, now_ts)
                        if frame_period_s > 0.0:
                            state["sync_next_emit_ts"] = 0.0

                if _OUTPUT_SCHEDULE_ENABLE_HEALTH_POLLING and bool(state.get("started", False)):
                    try:
                        if (int(state.get("scheduled_frames", 0)) % _OUTPUT_SCHEDULE_HEALTH_SAMPLE_EVERY) != 0:
                            return True

                        if bool(state.get("can_query_buffered", False)):
                            buffered_count = int(out.buffered_video_frame_count())
                        else:
                            buffered_count = _estimate_output_schedule_buffered_frames(state, time.perf_counter())
                        state["last_buffered_count"] = buffered_count

                        overflow_threshold = max(target_start_frames + 8, 12)

                        if buffered_count <= 0:
                            state["starved_streak"] = int(state.get("starved_streak", 0)) + 1
                        else:
                            state["starved_streak"] = 0

                        if buffered_count >= overflow_threshold:
                            state["overflow_streak"] = int(state.get("overflow_streak", 0)) + 1
                        else:
                            state["overflow_streak"] = 0

                        now_ts = time.perf_counter()
                        since_last_reprime = now_ts - float(state.get("last_auto_reprime_ts", 0.0))
                        can_auto_reprime = since_last_reprime >= _OUTPUT_SCHEDULE_AUTO_REPRIME_MIN_INTERVAL_S

                        if (
                            _OUTPUT_SCHEDULE_ENABLE_AUTO_REPRIME
                            and int(state.get("starved_streak", 0)) >= _OUTPUT_SCHEDULE_STARVED_STREAK_THRESHOLD
                            and can_auto_reprime
                        ):
                            state["starvation_events"] = int(state.get("starvation_events", 0)) + 1
                            state["auto_reprime_events"] = int(state.get("auto_reprime_events", 0)) + 1
                            state["last_auto_reprime_ts"] = now_ts
                            _reprime_output_schedule(out, reason="auto_starvation")
                        elif (
                            _OUTPUT_SCHEDULE_AUTO_REPRIME_ON_OVERFLOW
                            and int(state.get("overflow_streak", 0)) >= _OUTPUT_SCHEDULE_OVERFLOW_STREAK_THRESHOLD
                            and can_auto_reprime
                        ):
                            state["overflow_events"] = int(state.get("overflow_events", 0)) + 1
                            state["auto_reprime_events"] = int(state.get("auto_reprime_events", 0)) + 1
                            state["last_auto_reprime_ts"] = now_ts
                            _reprime_output_schedule(out, reason="auto_overflow")
                    except Exception:
                        state["can_query_buffered"] = False
                return True
            except Exception:
                # Fall back to blocking output if scheduling path errors at runtime.
                state["enabled"] = False

    out.display_frame_sync(payload)
    return True


def _clear_output_schedule_state(out: object | None) -> None:
    if out is None:
        return
    out_id = id(out)
    _OUTPUT_SCHEDULE_STATE.pop(out_id, None)
    _OUTPUT_SCHEDULE_TARGET_BUFFER_FRAMES.pop(out_id, None)


def _set_output_schedule_buffer_frames(out: object, buffer_frames: int) -> None:
    out_id = id(out)
    clamped = max(0, min(10, int(buffer_frames)))
    _OUTPUT_SCHEDULE_TARGET_BUFFER_FRAMES[out_id] = clamped
    state = _OUTPUT_SCHEDULE_STATE.get(out_id)
    if state is not None:
        state["target_buffer_frames"] = clamped


def _reprime_output_schedule(out: object, reason: str = "manual") -> None:
    state = _OUTPUT_SCHEDULE_STATE.get(id(out))
    if state is None:
        return

    stop_fn = getattr(out, "stop_scheduled_playback", None)
    if callable(stop_fn):
        try:
            stop_fn()
        except Exception:
            pass

    state["started"] = False
    state["queued_before_start"] = 0
    state["display_time"] = 0
    state["schedule_epoch_perf_ts"] = 0.0
    state["sync_next_emit_ts"] = 0.0
    state["starved_streak"] = 0
    state["overflow_streak"] = 0
    state["last_reprime_reason"] = str(reason)
    state["last_reprime_ts"] = time.perf_counter()


def _normalize_worker_roi(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    roi_w = max(2, min(int(w), FRAME_W))
    roi_h = max(2, min(int(h), FRAME_H))

    roi_w &= ~1
    if roi_w < 2:
        roi_w = 2

    max_x = max(0, FRAME_W - roi_w)
    max_y = max(0, FRAME_H - roi_h)
    roi_x = max(0, min(int(x), max_x))
    roi_y = max(0, min(int(y), max_y))

    roi_x &= ~1
    if roi_x > max_x:
        roi_x = max(0, max_x & ~1)

    return roi_x, roi_y, roi_w, roi_h


@dataclass
class _StageFrame:
    frame_id: int
    captured_ts: float
    input_bytes: bytes
    process_start_ts: float = 0.0
    process_end_ts: float = 0.0
    output_queue_put_ts: float = 0.0
    output_dequeue_ts: float = 0.0
    preprocess_bytes: bytes | None = None
    output_bytes: bytes | None = None
    shift_x: float = 0.0
    shift_y: float = 0.0
    roi_x: int = 0
    roi_y: int = 0
    roi_w: int = FRAME_W
    roi_h: int = FRAME_H
    effective_sr_scale: int = 1
    ai_applied: bool = False
    rtx_applied: bool = False
    native_shift_applied: bool = False
    interlaced_field_phase: dict[str, object] | None = None
    interlaced_phase_rendered: bool = False


def run_processor_worker(
    request_queue,
    response_queue,
    startup_config: dict[str, Any],
    roi_telemetry_shared=None,
    roi_telemetry_seq=None,
) -> None:
    _FRAME_MESSAGE_TYPES = {"frame", "decklink_frame", "decklink_no_frame"}
    _CONTROL_MESSAGE_TYPES = {"ready", "ack", "warning", "error"}
    _ROI_SLOT_COUNT = 16
    _TM_ACTIVE = 0
    _TM_FRAME_PROGRESS = 1
    _TM_TOTAL_FRAMES = 2
    _TM_INTERP_MODE_CODE = 3
    _TM_APPLIED_X = 4
    _TM_APPLIED_Y = 5
    _TM_APPLIED_W = 6
    _TM_APPLIED_H = 7
    _TM_START_X = 8
    _TM_START_Y = 9
    _TM_START_W = 10
    _TM_START_H = 11
    _TM_TARGET_X = 12
    _TM_TARGET_Y = 13
    _TM_TARGET_W = 14
    _TM_TARGET_H = 15

    reusable_native_into_supported = False
    reusable_process_frame_out: bytearray | None = None
    reusable_process_frame_no_deinterlace_out: bytearray | None = None
    reusable_process_frame_deinterlace_only_out: bytearray | None = None
    reusable_process_frame_preprocess_only_out: bytearray | None = None

    def _run_native_uyvy_process(
        frame_bytes: bytes,
        process_method_name: str,
        process_into_method_name: str,
        reusable_output: bytearray | None,
    ) -> bytes | bytearray:
        method = getattr(processor, process_method_name)
        if reusable_output is not None and hasattr(processor, process_into_method_name):
            into_method = getattr(processor, process_into_method_name)
            into_method(frame_bytes, reusable_output)
            return reusable_output
        return method(frame_bytes)

    def _interp_mode_to_code(interp_mode: str) -> int:
        mode = str(interp_mode).strip().lower()
        if mode == "ease_in_out":
            return 1
        if mode == "ease_out":
            return 2
        return 0

    def _publish_roi_telemetry(
        current_x: int,
        current_y: int,
        current_w: int,
        current_h: int,
        transition_state: dict[str, Any] | None,
    ) -> None:
        if roi_telemetry_shared is None or roi_telemetry_seq is None:
            return

        start_roi = (current_x, current_y, current_w, current_h)
        target_roi = (current_x, current_y, current_w, current_h)
        active = False
        frame_progress = 0.0
        total_frames = 0
        interp_mode_code = 0

        if isinstance(transition_state, dict):
            active = True
            start_roi = tuple(transition_state.get("start", start_roi))
            target_roi = tuple(transition_state.get("target", target_roi))
            frame_progress = float(transition_state.get("frame_progress", 0.0))
            total_frames = int(transition_state.get("total_frames", 0))
            interp_mode_code = _interp_mode_to_code(str(transition_state.get("interpolation_mode", "linear")))

        try:
            with roi_telemetry_shared.get_lock():
                if len(roi_telemetry_shared) < _ROI_SLOT_COUNT:
                    return
                roi_telemetry_shared[_TM_ACTIVE] = 1.0 if active else 0.0
                roi_telemetry_shared[_TM_FRAME_PROGRESS] = float(frame_progress)
                roi_telemetry_shared[_TM_TOTAL_FRAMES] = float(total_frames)
                roi_telemetry_shared[_TM_INTERP_MODE_CODE] = float(interp_mode_code)
                roi_telemetry_shared[_TM_APPLIED_X] = float(current_x)
                roi_telemetry_shared[_TM_APPLIED_Y] = float(current_y)
                roi_telemetry_shared[_TM_APPLIED_W] = float(current_w)
                roi_telemetry_shared[_TM_APPLIED_H] = float(current_h)
                roi_telemetry_shared[_TM_START_X] = float(int(start_roi[0]))
                roi_telemetry_shared[_TM_START_Y] = float(int(start_roi[1]))
                roi_telemetry_shared[_TM_START_W] = float(int(start_roi[2]))
                roi_telemetry_shared[_TM_START_H] = float(int(start_roi[3]))
                roi_telemetry_shared[_TM_TARGET_X] = float(int(target_roi[0]))
                roi_telemetry_shared[_TM_TARGET_Y] = float(int(target_roi[1]))
                roi_telemetry_shared[_TM_TARGET_W] = float(int(target_roi[2]))
                roi_telemetry_shared[_TM_TARGET_H] = float(int(target_roi[3]))
            with roi_telemetry_seq.get_lock():
                roi_telemetry_seq.value = int(roi_telemetry_seq.value) + 1
        except Exception:
            # Never allow telemetry publishing failures to disrupt frame processing.
            return
    worker_process_priority = _normalize_worker_priority_name(
        str(startup_config.get("worker_process_priority", "above_normal"))
    )
    worker_process_priority, worker_process_priority_error = _apply_current_process_priority(worker_process_priority)

    def _safe_put(message: dict[str, Any]) -> None:
        # Prioritize control-plane messages (ready/ack/error) over frame traffic so
        # GUI state never gets stuck waiting for a dropped acknowledgement.
        msg_type = str(message.get("type", ""))
        is_control_message = msg_type in _CONTROL_MESSAGE_TYPES

        try:
            response_queue.put_nowait(message)
            return
        except queue.Full:
            pass

        if is_control_message:
            preserved_messages: list[dict[str, Any]] = []
            dropped_frame = False
            while True:
                try:
                    queued = response_queue.get_nowait()
                except queue.Empty:
                    break

                queued_type = str(queued.get("type", "")) if isinstance(queued, dict) else ""
                if queued_type in _FRAME_MESSAGE_TYPES:
                    dropped_frame = True
                    break
                preserved_messages.append(queued)

            # Restore preserved non-frame messages in FIFO order.
            for queued in preserved_messages:
                try:
                    response_queue.put_nowait(queued)
                except queue.Full:
                    break

            if dropped_frame:
                try:
                    response_queue.put_nowait(message)
                    return
                except queue.Full:
                    pass

        try:
            response_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            response_queue.put_nowait(message)
        except queue.Full:
            # Final fallback: drop this message to keep worker alive.
            return

    processor = None
    project_root_path = Path(startup_config.get("project_root", Path(__file__).resolve().parents[1]))
    rtx_vsr_runtime_module = rtx_vsr_module
    rtx_video_sdk_root = str(startup_config.get("rtx_video_sdk_root", "")).strip()
    if not rtx_video_sdk_root:
        rtx_video_sdk_root = os.environ.get("RTX_VIDEO_SDK_ROOT", r"C:\Coding Projects\sdks\NVidia video SDK").strip()
    if rtx_video_sdk_root:
        os.environ["RTX_VIDEO_SDK_ROOT"] = rtx_video_sdk_root
    _prepare_rtx_runtime_dll_paths(rtx_video_sdk_root, project_root_path)

    def _resolve_rtx_vsr_module():
        nonlocal rtx_vsr_runtime_module
        if rtx_vsr_runtime_module is not None:
            return rtx_vsr_runtime_module, None

        preferred_paths = [
            project_root_path / "build" / "src" / "Release",
            project_root_path / "build" / "src" / "RelWithDebInfo",
            project_root_path / "build" / "src" / "Debug",
        ]
        for candidate in preferred_paths:
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

        try:
            rtx_vsr_runtime_module = importlib.import_module("rtx_vsr")
            return rtx_vsr_runtime_module, None
        except Exception as exc:
            return None, str(exc)
    capture_session = None
    output_session = None
    pipeline_running = False
    pipeline_stop_event = threading.Event()
    capture_thread: threading.Thread | None = None
    preprocess_thread: threading.Thread | None = None
    upscale_thread: threading.Thread | None = None
    upscale_extra_threads: list[threading.Thread] = []
    output_thread: threading.Thread | None = None
    parallel_basic_processors: list[Any] = []
    parallel_basic_worker_count = 1
    parallel_basic_max_inflight = max(1, min(4, int(os.environ.get("VP_BASIC_SCALING_MAX_INFLIGHT", "1"))))
    q_capture_to_preprocess: queue.Queue[_StageFrame] | None = None
    q_preprocess_to_upscale: queue.Queue[_StageFrame] | None = None
    q_upscale_to_output: queue.Queue[_StageFrame] | None = None
    frame_id_counter = 0
    capture_drop_count = 0
    preprocess_drop_count = 0
    upscale_drop_count = 0
    latest_input_frame: bytes | None = None
    latest_output_frame: bytes | None = None
    latest_timecode_info: dict[str, object] = {
        "present": False,
        "text": "",
        "format_code": 0,
        "format_name": "",
    }
    latest_effective_sr_scale = 1
    latest_rtx_vsr_applied = False
    latest_rtx_effect_mean_abs_luma = 0.0
    rtx_effect_sample_counter = 0
    processed_frame_counter = 0
    started_perf_ts = 0.0
    output_nominal_fps = 0.0
    output_frame_period_s = 0.0
    output_mode_is_interlaced = False
    output_field_dominance_code: int | None = None
    output_mode_name = ""
    output_mode_value = ""
    output_field_dominance_name = ""
    output_transition_units_per_frame = 1.0
    stage_preprocess_applied_frames = 0
    stage_basic_applied_frames = 0
    stage_ai_applied_frames = 0
    stage_rtx_applied_frames = 0
    stage_passthrough_frames = 0
    last_stage_preprocess_applied = False
    last_stage_basic_applied = False
    last_stage_ai_applied = False
    last_stage_rtx_applied = False
    last_stage_stack: list[str] = []
    basic_scaling_last_frame_ms = 0.0
    basic_scaling_avg_frame_ms: float | None = None
    basic_scaling_max_frame_ms = 0.0
    basic_scaling_timing_samples = 0
    timing_frames_emitted = 0
    timing_deadline_miss_events = 0
    timing_deadline_miss_streak = 0
    timing_deadline_miss_max_streak = 0
    timing_e2e_ms_last = 0.0
    timing_e2e_ms_ema = 0.0
    timing_e2e_ms_peak = 0.0
    timing_process_ms_ema = 0.0
    timing_process_ms_peak = 0.0
    timing_capture_queue_ms_ema = 0.0
    timing_capture_queue_ms_peak = 0.0
    timing_output_queue_ms_ema = 0.0
    timing_output_queue_ms_peak = 0.0
    timing_output_wait_ms_ema = 0.0
    timing_output_wait_ms_peak = 0.0
    timing_emit_call_ms_ema = 0.0
    timing_emit_call_ms_peak = 0.0
    timing_deadline_late_ms_ema = 0.0
    timing_deadline_late_ms_peak = 0.0
    timing_last_path = ""
    ai_sr_enabled = bool(startup_config.get("ai_sr_enabled", False))
    ai_sr_model_path = str(startup_config.get("ai_sr_model_path", ""))
    ai_sr_provider = str(startup_config.get("ai_sr_provider", "cuda"))
    ai_sr_trt_precision = str(startup_config.get("ai_sr_trt_precision", os.environ.get("VP_AI_SR_TRT_PRECISION", "fp16"))).strip().lower()
    if ai_sr_trt_precision not in {"fp16", "int8"}:
        ai_sr_trt_precision = "fp16"
    trt_cache_root = Path(startup_config.get("project_root", str(Path(__file__).resolve().parents[1]))) / "build" / "trt_engine_cache"
    ai_sr_require_gpu = bool(startup_config.get("ai_sr_require_gpu", False))
    ai_sr_frame_interval = max(1, min(60, int(startup_config.get("ai_sr_inference_fps", startup_config.get("ai_sr_frame_interval", 1)))))
    ai_sr_strict = bool(startup_config.get("ai_sr_strict", False))
    ai_sr_input_align = max(1, int(startup_config.get("ai_sr_input_align", 2)))
    ai_sr_roi_overscan_percent = float(startup_config.get("ai_sr_roi_overscan_percent", 0.0))
    ai_sr_inference_divisor = max(0, int(startup_config.get("ai_sr_inference_divisor", 0)))
    ai_sr_detail_preserve_percent = float(startup_config.get("ai_sr_detail_preserve_percent", 0.0))
    ai_sr_post_denoise_method = _normalize_ai_sr_post_denoise_method(str(startup_config.get("ai_sr_post_denoise_method", "off")))
    ai_sr_post_denoise_strength = max(0.0, min(1.0, float(startup_config.get("ai_sr_post_denoise_strength", 0.0))))
    ai_sr_post_artifact_reduction_method = _normalize_ai_sr_post_artifact_reduction_method(
        str(startup_config.get("ai_sr_post_artifact_reduction_method", "off"))
    )
    ai_sr_post_artifact_reduction_strength = max(
        0.0,
        min(1.0, float(startup_config.get("ai_sr_post_artifact_reduction_strength", 0.0))),
    )
    ai_sr_post_exaggeration_enabled = bool(startup_config.get("ai_sr_post_exaggeration_enabled", False))
    ai_sr_post_exaggeration_gain = max(
        1.0,
        min(4.0, float(startup_config.get("ai_sr_post_exaggeration_gain", 2.0))),
    )
    ai_sr_runtime_note: str | None = None
    ai_sr_engine: AiSrOnnxEngine | None = None
    ai_sr_info: dict[str, object] | None = None
    ai_sr_frame_counter = 0
    ai_sr_latest_output_frame: bytes | None = None
    ai_sr_latest_output_ts = 0.0
    ai_sr_completed_frames = 0
    ai_sr_warmup_pending = False
    ai_sr_hold_last_frame = bool(
        startup_config.get(
            "ai_sr_hold_last_frame",
            os.environ.get("VP_AI_SR_HOLD_LAST_FRAME", "1") == "1",
        )
    )
    ai_sr_max_hold_ms = max(
        0.0,
        float(startup_config.get("ai_sr_max_hold_ms", os.environ.get("VP_AI_SR_MAX_HOLD_MS", "0"))),
    )
    ai_sr_max_inflight = max(
        1,
        min(4, int(startup_config.get("ai_sr_max_inflight", os.environ.get("VP_AI_SR_MAX_INFLIGHT", "1")))),
    )
    ai_sr_submit_spacing_ms = max(
        0.0,
        float(startup_config.get("ai_sr_submit_spacing_ms", os.environ.get("VP_AI_SR_SUBMIT_SPACING_MS", "0"))),
    )
    ai_sr_last_submit_ts = 0.0
    ai_sr_executor: ThreadPoolExecutor | None = None
    ai_sr_futures: list[Future[bytes]] = []
    ai_sr_dropped_frames = 0
    ai_sr_applied_frames = 0
    ai_sr_reused_frames = 0
    ai_sr_passthrough_frames = 0
    zeroed_output_warning_emitted = False
    preprocess_noop_warning_emitted = False
    native_subpixel_warning_emitted = False
    rtx_vsr_enabled = bool(startup_config.get("rtx_vsr_enabled", False))
    rtx_vsr_quality = str(startup_config.get("rtx_vsr_quality", "high")).strip().lower() or "high"
    rtx_vsr_scale = max(1, int(startup_config.get("rtx_vsr_scale", 2)))
    rtx_vsr_post_scale_method = str(startup_config.get("rtx_vsr_post_scale_method", "bicubic")).strip().lower() or "bicubic"
    rtx_thdr_enabled = bool(startup_config.get("rtx_thdr_enabled", False))
    rtx_thdr_contrast = max(0, int(startup_config.get("rtx_thdr_contrast", 50)))
    rtx_thdr_saturation = max(0, int(startup_config.get("rtx_thdr_saturation", 50)))
    rtx_thdr_middle_gray = max(0, int(startup_config.get("rtx_thdr_middle_gray", 50)))
    rtx_thdr_max_luminance = max(0, int(startup_config.get("rtx_thdr_max_luminance", 1000)))
    rtx_vsr_engine = None
    rtx_vsr_info: dict[str, object] | None = None
    rtx_vsr_error: str | None = None
    rtx_roi_rebuild_pending = False
    rtx_roi_rebuild_due_ts = 0.0
    rtx_roi_rebuild_settle_s = 0.25
    current_basic_scaling_method = str(startup_config.get("basic_scaling_method", "bilinear_sharp"))
    current_color_space = _normalize_color_space_name(str(startup_config.get("color_space", "rec709")))
    current_color_range = _normalize_color_range_name(str(startup_config.get("color_range", "limited")))
    current_roi_x = int(startup_config.get("roi_x", 0))
    current_roi_y = int(startup_config.get("roi_y", 0))
    current_roi_w = int(startup_config.get("roi_w", FRAME_W))
    current_roi_h = int(startup_config.get("roi_h", FRAME_H))
    roi_shift_target_x = float(startup_config.get("roi_subpixel_shift_x", 0.0))
    roi_shift_target_y = float(startup_config.get("roi_subpixel_shift_y", 0.0))
    roi_shift_target_lpf_x = float(roi_shift_target_x)
    roi_shift_target_lpf_y = float(roi_shift_target_y)
    roi_shift_applied_x = float(roi_shift_target_x)
    roi_shift_applied_y = float(roi_shift_target_y)
    roi_shift_velocity_x = 0.0
    roi_shift_velocity_y = 0.0
    roi_shift_accel_x = 0.0
    roi_shift_accel_y = 0.0
    interlaced_prev_motion_shift_x: float | None = None
    interlaced_prev_motion_shift_y: float | None = None
    manual_interlaced_phase_state: dict[str, object] | None = None
    manual_interlaced_phase_until_ts = 0.0
    manual_interlaced_phase_pending = False
    roi_manual_drag_until_ts = 0.0
    roi_manual_drag_hold_s = max(0.05, min(0.50, float(os.environ.get("VP_ROI_MANUAL_DRAG_HOLD_S", "0.24"))))
    interlaced_field2_phase_fraction = _clamp_interlaced_field2_phase_fraction(
        float(startup_config.get("interlaced_field2_phase_fraction", _INTERLACED_FIELD2_PHASE_FRACTION))
    )
    roi_microstep_transition: dict[str, object] | None = None
    current_deinterlace_enabled = bool(startup_config.get("deinterlace_enabled", True))
    current_reinterlace_enabled = bool(startup_config.get("reinterlace_enabled", False))
    current_deinterlace_method = str(startup_config.get("deinterlace_method", "bob"))
    current_denoise_method = str(startup_config.get("denoise_method", "off"))
    current_denoise_strength = max(0.0, min(1.0, float(startup_config.get("denoise_strength", 0.35))))
    current_output_buffer_frames = max(0, min(10, int(startup_config.get("decklink_output_buffer_frames", 2))))
    basic_scaling_enabled = bool(startup_config.get("enable_basic_scaling", startup_config.get("enable_placeholder_sr", True)))
    current_basic_scaling_auto_mode = bool(startup_config.get("basic_scaling_auto_mode", True))
    current_basic_scaling_manual_scale = int(startup_config.get("basic_scaling_manual", startup_config.get("sr_scale", 4)))
    current_max_auto_basic_scaling = int(startup_config.get("max_auto_basic_scaling", startup_config.get("max_auto_sr_scale", 4)))
    state_lock = threading.Lock()

    def _is_live_passthrough_mode() -> bool:
        # Passthrough mode is valid only when no stage is expected to modify pixels.
        denoise_enabled = current_denoise_method not in {"off", "none"} and current_denoise_strength > 0.001
        ai_stage_active = ai_sr_enabled and ai_sr_engine is not None
        rtx_stage_active = rtx_vsr_enabled and rtx_vsr_engine is not None
        return (
            (not current_deinterlace_enabled)
            and (not denoise_enabled)
            and (not basic_scaling_enabled)
            and (not ai_stage_active)
            and (not rtx_stage_active)
        )

    def _is_live_basic_scaling_fast_mode() -> bool:
        # Basic scaling fast mode keeps processing in capture thread when the
        # pipeline is native-only (no Python AI/RTX stages). Native process_frame
        # already fuses preprocess, so deinterlace/denoise can remain enabled.
        ai_stage_active = ai_sr_enabled and ai_sr_engine is not None
        rtx_stage_active = rtx_vsr_enabled and rtx_vsr_engine is not None
        return (
            basic_scaling_enabled
            and (not ai_stage_active)
            and (not rtx_stage_active)
        )

    def _step_smoothed_roi_shift() -> tuple[float, float]:
        nonlocal roi_shift_applied_x, roi_shift_applied_y
        nonlocal roi_shift_velocity_x, roi_shift_velocity_y
        nonlocal roi_shift_target_lpf_x, roi_shift_target_lpf_y
        nonlocal roi_shift_accel_x, roi_shift_accel_y
        nonlocal roi_manual_drag_until_ts
        nonlocal roi_microstep_transition
        nonlocal output_transition_units_per_frame

        # During active keyframe transitions, follow the shift target tightly.
        # For interlaced field-unit progression, keep a small LPF to reduce
        # visible per-frame stepping when transition units advance by 2.
        if roi_microstep_transition is not None:
            if float(output_transition_units_per_frame) > 1.0:
                roi_shift_target_lpf_x += (float(roi_shift_target_x) - roi_shift_target_lpf_x) * 0.74
                roi_shift_target_lpf_y += (float(roi_shift_target_y) - roi_shift_target_lpf_y) * 0.72
                roi_shift_applied_x += (roi_shift_target_lpf_x - roi_shift_applied_x) * 0.86
                roi_shift_applied_y += (roi_shift_target_lpf_y - roi_shift_applied_y) * 0.84
            else:
                roi_shift_target_lpf_x = float(roi_shift_target_x)
                roi_shift_target_lpf_y = float(roi_shift_target_y)
                roi_shift_applied_x = float(roi_shift_target_x)
                roi_shift_applied_y = float(roi_shift_target_y)
            roi_shift_velocity_x = 0.0
            roi_shift_velocity_y = 0.0
            roi_shift_accel_x = 0.0
            roi_shift_accel_y = 0.0
            return float(roi_shift_applied_x), float(roi_shift_applied_y)

        # Deterministic microstep follower: predictable per-frame motion with
        # very small minimum steps to avoid perceptible stair-stepping.
        manual_drag_active = time.perf_counter() <= float(roi_manual_drag_until_ts)
        if manual_drag_active:
            # Softer follower during manual drag to hide even-x carrier steps.
            target_alpha = 0.32
            follow_alpha_x = 0.21
            follow_alpha_y = 0.19
            min_step_x = 0.0022
            min_step_y = 0.0019
            max_step_x = 0.28
            max_step_y = 0.24
            settle_eps_x = 0.0010
            settle_eps_y = 0.0009
        else:
            target_alpha = 0.34
            follow_alpha_x = 0.22
            follow_alpha_y = 0.20
            min_step_x = 0.0040
            min_step_y = 0.0035
            max_step_x = 0.30
            max_step_y = 0.26
            settle_eps_x = 0.0012
            settle_eps_y = 0.0010

        roi_shift_target_lpf_x += (float(roi_shift_target_x) - roi_shift_target_lpf_x) * target_alpha
        roi_shift_target_lpf_y += (float(roi_shift_target_y) - roi_shift_target_lpf_y) * target_alpha

        err_x = float(roi_shift_target_lpf_x - roi_shift_applied_x)
        err_y = float(roi_shift_target_lpf_y - roi_shift_applied_y)

        abs_err_x = abs(err_x)
        abs_err_y = abs(err_y)

        if abs_err_x <= settle_eps_x:
            roi_shift_applied_x = float(roi_shift_target_lpf_x)
            roi_shift_velocity_x = 0.0
        else:
            step_x = max(min_step_x, min(max_step_x, abs_err_x * follow_alpha_x))
            roi_shift_applied_x += step_x if err_x > 0.0 else -step_x
            roi_shift_velocity_x = step_x if err_x > 0.0 else -step_x

        if abs_err_y <= settle_eps_y:
            roi_shift_applied_y = float(roi_shift_target_lpf_y)
            roi_shift_velocity_y = 0.0
        else:
            step_y = max(min_step_y, min(max_step_y, abs_err_y * follow_alpha_y))
            roi_shift_applied_y += step_y if err_y > 0.0 else -step_y
            roi_shift_velocity_y = step_y if err_y > 0.0 else -step_y

        roi_shift_accel_x = 0.0
        roi_shift_accel_y = 0.0

        return float(roi_shift_applied_x), float(roi_shift_applied_y)

    def _set_roi_shift_target(next_target_x: float, next_target_y: float) -> None:
        nonlocal roi_shift_target_x, roi_shift_target_y
        nonlocal roi_shift_target_lpf_x, roi_shift_target_lpf_y
        nonlocal roi_shift_applied_x, roi_shift_applied_y
        nonlocal roi_shift_velocity_x, roi_shift_velocity_y
        nonlocal roi_shift_accel_x, roi_shift_accel_y

        clamped_x = max(-48.0, min(48.0, float(next_target_x)))
        clamped_y = max(-48.0, min(48.0, float(next_target_y)))
        delta_target_x = clamped_x - roi_shift_target_x
        delta_target_y = clamped_y - roi_shift_target_y
        roi_shift_target_x = clamped_x
        roi_shift_target_y = clamped_y

        # On larger compensation steps, pre-bias but do not hard snap.
        if abs(delta_target_x) >= 1.0 or abs(delta_target_y) >= 0.8:
            roi_shift_target_lpf_x += max(-0.92, min(0.92, delta_target_x * 0.70))
            roi_shift_target_lpf_y += max(-0.82, min(0.82, delta_target_y * 0.68))
            roi_shift_applied_x += (delta_target_x * 0.18)
            roi_shift_applied_y += (delta_target_y * 0.16)
            roi_shift_applied_x = max(-48.0, min(48.0, roi_shift_applied_x))
            roi_shift_applied_y = max(-48.0, min(48.0, roi_shift_applied_y))
            roi_shift_velocity_x *= 0.25
            roi_shift_velocity_y *= 0.25
            roi_shift_accel_x = 0.0
            roi_shift_accel_y = 0.0

        # Ensure end-of-move recenter settles immediately to avoid a lingering tail.
        if abs(clamped_x) <= 1e-4 and abs(clamped_y) <= 1e-4:
            roi_shift_target_lpf_x = 0.0
            roi_shift_target_lpf_y = 0.0
            roi_shift_applied_x *= 0.45
            roi_shift_applied_y *= 0.45

    def _set_roi_shift_immediate(next_target_x: float, next_target_y: float) -> None:
        nonlocal roi_shift_target_x, roi_shift_target_y
        nonlocal roi_shift_target_lpf_x, roi_shift_target_lpf_y
        nonlocal roi_shift_applied_x, roi_shift_applied_y
        nonlocal roi_shift_velocity_x, roi_shift_velocity_y
        nonlocal roi_shift_accel_x, roi_shift_accel_y

        clamped_x = max(-48.0, min(48.0, float(next_target_x)))
        clamped_y = max(-48.0, min(48.0, float(next_target_y)))
        roi_shift_target_x = clamped_x
        roi_shift_target_y = clamped_y
        roi_shift_target_lpf_x = clamped_x
        roi_shift_target_lpf_y = clamped_y
        roi_shift_applied_x = clamped_x
        roi_shift_applied_y = clamped_y
        roi_shift_velocity_x = 0.0
        roi_shift_velocity_y = 0.0
        roi_shift_accel_x = 0.0
        roi_shift_accel_y = 0.0

    def _apply_interlaced_field_phase_if_needed(
        frame_bytes: bytes,
        base_shift_x: float,
        base_shift_y: float,
        native_shift_applied: bool,
    ) -> bytes:
        nonlocal interlaced_prev_motion_shift_x, interlaced_prev_motion_shift_y
        nonlocal output_mode_is_interlaced, output_transition_units_per_frame
        nonlocal roi_microstep_transition, roi_manual_drag_until_ts
        nonlocal roi_shift_target_x, roi_shift_target_y, roi_shift_applied_x, roi_shift_applied_y
        nonlocal interlaced_field2_phase_fraction

        if not output_mode_is_interlaced:
            interlaced_prev_motion_shift_x = None
            interlaced_prev_motion_shift_y = None
            return frame_bytes

        if cv2 is None or len(frame_bytes) != (UYVY_ROW_BYTES * FRAME_H):
            return frame_bytes

        phase_fraction = _clamp_interlaced_field2_phase_fraction(float(interlaced_field2_phase_fraction))

        transition_state = roi_microstep_transition if isinstance(roi_microstep_transition, dict) else None
        now_ts = time.perf_counter()
        motion_active = bool(
            (roi_microstep_transition is not None)
            or (now_ts <= float(roi_manual_drag_until_ts))
            or (abs(float(roi_shift_target_x) - float(roi_shift_applied_x)) >= 0.01)
            or (abs(float(roi_shift_target_y) - float(roi_shift_applied_y)) >= 0.01)
        )
        if not _interlaced_phase_controls_active():
            # Neutral controls must fully disable field-phase synthesis.
            interlaced_prev_motion_shift_x = float(base_shift_x)
            interlaced_prev_motion_shift_y = float(base_shift_y)
            if motion_active:
                is_lower_field_first = int(output_field_dominance_code) == 1 if output_field_dominance_code is not None else False
                return _collapse_interlaced_to_single_field_uyvy(frame_bytes, is_lower_field_first)
            return frame_bytes

        transition_field_shift = transition_state.get("interlaced_field_shift") if transition_state is not None else None
        if isinstance(transition_field_shift, dict):
            field0_x = float(transition_field_shift.get("field0_x", 0.0))
            field0_y = float(transition_field_shift.get("field0_y", 0.0))
            field1_x = float(transition_field_shift.get("field1_x", field0_x))
            field1_y = float(transition_field_shift.get("field1_y", field0_y))
            if native_shift_applied:
                # Native path already applied field1/base shift to the whole
                # frame. Overlay differential shifts per field.
                base_x = float(base_shift_x)
                base_y = float(base_shift_y)
                out = _apply_interlaced_field_phase_shift_uyvy(
                    frame_bytes,
                    field0_x - base_x,
                    field0_y - base_y,
                    field1_x - base_x,
                    field1_y - base_y,
                )
            else:
                out = _apply_interlaced_field_phase_shift_uyvy(
                    frame_bytes,
                    field0_x,
                    field0_y,
                    field1_x,
                    field1_y,
                )
            interlaced_prev_motion_shift_x = field1_x
            interlaced_prev_motion_shift_y = field1_y
            return out

        if not motion_active:
            interlaced_prev_motion_shift_x = None
            interlaced_prev_motion_shift_y = None
            return frame_bytes

        prev_x = interlaced_prev_motion_shift_x
        prev_y = interlaced_prev_motion_shift_y
        if prev_x is None or prev_y is None:
            prev_x = float(base_shift_x)
            prev_y = float(base_shift_y)

        delta_x = float(base_shift_x) - float(prev_x)
        delta_y = float(base_shift_y) - float(prev_y)
        per_field_dx = delta_x * phase_fraction
        per_field_dy = delta_y * phase_fraction

        # If follower lag hides motion in consecutive frames, derive a small
        # forward microstep from residual target error to avoid repeated fields.
        if abs(per_field_dx) < 0.0008:
            residual_x = float(roi_shift_target_x) - float(base_shift_x)
            per_field_dx = max(-0.75, min(0.75, residual_x * phase_fraction))
        if abs(per_field_dy) < 0.0008:
            residual_y = float(roi_shift_target_y) - float(base_shift_y)
            per_field_dy = max(-0.75, min(0.75, residual_y * phase_fraction))

        if native_shift_applied:
            field0_x = 0.0
            field0_y = 0.0
            field1_x = float(per_field_dx)
            field1_y = float(per_field_dy)
        else:
            field0_x = float(base_shift_x)
            field0_y = float(base_shift_y)
            field1_x = float(base_shift_x) + float(per_field_dx)
            field1_y = float(base_shift_y) + float(per_field_dy)

        out = _apply_interlaced_field_phase_shift_uyvy(
            frame_bytes,
            field0_x,
            field0_y,
            field1_x,
            field1_y,
        )

        interlaced_prev_motion_shift_x = float(base_shift_x) + float(per_field_dx)
        interlaced_prev_motion_shift_y = float(base_shift_y) + float(per_field_dy)
        return out

    def _weave_interlaced_fields(frame0_bytes: bytes, frame1_bytes: bytes) -> bytes:
        if len(frame0_bytes) != (UYVY_ROW_BYTES * FRAME_H) or len(frame1_bytes) != (UYVY_ROW_BYTES * FRAME_H):
            return frame1_bytes
        frame0 = np.frombuffer(frame0_bytes, dtype=np.uint8).reshape(FRAME_H, UYVY_ROW_BYTES)
        frame1 = np.frombuffer(frame1_bytes, dtype=np.uint8).reshape(FRAME_H, UYVY_ROW_BYTES)
        out = np.empty_like(frame1)
        # DeckLink lower-field-first outputs emit odd lines first; upper-field-first
        # emits even lines first. phase0 is always the first temporal field step.
        is_lower_field_first = int(output_field_dominance_code) == 1 if output_field_dominance_code is not None else False
        if is_lower_field_first:
            out[0::2, :] = frame1[0::2, :]
            out[1::2, :] = frame0[1::2, :]
        else:
            out[0::2, :] = frame0[0::2, :]
            out[1::2, :] = frame1[1::2, :]
        return out.tobytes()

    def _apply_processor_roi_for_phase(roi_phase: tuple[int, int, int, int]) -> None:
        phase_x, phase_y, phase_w, phase_h = _normalize_worker_roi(
            int(roi_phase[0]),
            int(roi_phase[1]),
            int(roi_phase[2]),
            int(roi_phase[3]),
        )
        if phase_w == int(current_roi_w) and phase_h == int(current_roi_h):
            processor.set_roi_position(phase_x, phase_y)
        else:
            processor.set_roi(phase_x, phase_y, phase_w, phase_h)

    def _active_interlaced_field_phase_state(consume_manual_snapshot: bool = False) -> dict[str, object] | None:
        nonlocal manual_interlaced_phase_state, manual_interlaced_phase_until_ts, manual_interlaced_phase_pending
        if not output_mode_is_interlaced:
            manual_interlaced_phase_state = None
            manual_interlaced_phase_until_ts = 0.0
            manual_interlaced_phase_pending = False
            return None
        if not _interlaced_phase_controls_active():
            manual_interlaced_phase_state = None
            manual_interlaced_phase_until_ts = 0.0
            manual_interlaced_phase_pending = False
            return None
        phase_fraction = _clamp_interlaced_field2_phase_fraction(float(interlaced_field2_phase_fraction))
        state = roi_microstep_transition
        if not isinstance(state, dict):
            now_ts = time.perf_counter()
            if now_ts > float(manual_interlaced_phase_until_ts):
                manual_interlaced_phase_pending = False
            if (
                isinstance(manual_interlaced_phase_state, dict)
                and now_ts <= float(manual_interlaced_phase_until_ts)
                and bool(manual_interlaced_phase_pending)
            ):
                roi0 = manual_interlaced_phase_state.get("roi0")
                roi1 = manual_interlaced_phase_state.get("roi1")
                if (
                    isinstance(roi0, tuple)
                    and len(roi0) == 4
                    and isinstance(roi1, tuple)
                    and len(roi1) == 4
                ):
                    if consume_manual_snapshot:
                        manual_interlaced_phase_pending = False
                    return manual_interlaced_phase_state
        if not isinstance(state, dict):
            # Manual drag and residual shift path (no keyframe transition):
            # synthesize interlaced field0/field1 shifts from current smoothed
            # shift state so field-phase rendering remains active while dragging.
            now_ts = time.perf_counter()
            motion_active = bool(
                (now_ts <= float(roi_manual_drag_until_ts))
                or (abs(float(roi_shift_target_x) - float(roi_shift_applied_x)) >= 0.01)
                or (abs(float(roi_shift_target_y) - float(roi_shift_applied_y)) >= 0.01)
            )
            if not motion_active:
                return None

            field0_x = float(roi_shift_applied_x)
            field0_y = float(roi_shift_applied_y)
            delta_x = float(roi_shift_target_x) - field0_x
            delta_y = float(roi_shift_target_y) - field0_y
            field1_x = field0_x + (delta_x * phase_fraction)
            field1_y = field0_y + (delta_y * phase_fraction)

            return {
                "roi0": (int(current_roi_x), int(current_roi_y), int(current_roi_w), int(current_roi_h)),
                "roi1": (int(current_roi_x), int(current_roi_y), int(current_roi_w), int(current_roi_h)),
                "field0_x": float(field0_x),
                "field0_y": float(field0_y),
                "field1_x": float(field1_x),
                "field1_y": float(field1_y),
            }
        phase = state.get("interlaced_field_phase")
        if not isinstance(phase, dict):
            return None
        if "roi0" not in phase or "roi1" not in phase:
            return None
        return phase

    def _render_dual_phase_no_deinterlace(
        input_bytes: bytes,
        phase: dict[str, object],
    ) -> tuple[bytes, bool]:
        roi0 = tuple(phase.get("roi0", (current_roi_x, current_roi_y, current_roi_w, current_roi_h)))
        roi1 = tuple(phase.get("roi1", (current_roi_x, current_roi_y, current_roi_w, current_roi_h)))
        field0_x = float(phase.get("field0_x", 0.0))
        field0_y = float(phase.get("field0_y", 0.0))
        field1_x = float(phase.get("field1_x", 0.0))
        field1_y = float(phase.get("field1_y", 0.0))

        _apply_processor_roi_for_phase(roi0)
        native0 = _set_native_subpixel_shift(field0_x, field0_y)
        if hasattr(processor, "process_frame_no_deinterlace"):
            out0 = _run_native_uyvy_process(
                input_bytes,
                "process_frame_no_deinterlace",
                "process_frame_no_deinterlace_into",
                reusable_process_frame_no_deinterlace_out,
            )
        else:
            out0 = _apply_subpixel_shift_uyvy(input_bytes, field0_x, field0_y)

        # Native *_into paths reuse a shared bytearray. Snapshot field0 now so
        # the second render cannot overwrite field0 before weave.
        out0_frozen = bytes(out0)

        _apply_processor_roi_for_phase(roi1)
        native1 = _set_native_subpixel_shift(field1_x, field1_y)
        if hasattr(processor, "process_frame_no_deinterlace"):
            out1 = _run_native_uyvy_process(
                input_bytes,
                "process_frame_no_deinterlace",
                "process_frame_no_deinterlace_into",
                reusable_process_frame_no_deinterlace_out,
            )
        else:
            out1 = _apply_subpixel_shift_uyvy(input_bytes, field1_x, field1_y)

        out1_frozen = bytes(out1)

        return _weave_interlaced_fields(out0_frozen, out1_frozen), bool(native0 and native1)

    def _render_dual_phase_basic_scaling(
        input_bytes: bytes,
        phase: dict[str, object],
    ) -> tuple[bytes, bool, bool, float]:
        roi0 = tuple(phase.get("roi0", (current_roi_x, current_roi_y, current_roi_w, current_roi_h)))
        roi1 = tuple(phase.get("roi1", (current_roi_x, current_roi_y, current_roi_w, current_roi_h)))
        field0_x = float(phase.get("field0_x", 0.0))
        field0_y = float(phase.get("field0_y", 0.0))
        field1_x = float(phase.get("field1_x", 0.0))
        field1_y = float(phase.get("field1_y", 0.0))

        _apply_processor_roi_for_phase(roi0)
        out0, basic0, native0, ms0 = _apply_basic_scaling_stage(
            input_bytes,
            False,
            field0_x,
            field0_y,
        )
        out0_frozen = bytes(out0)

        _apply_processor_roi_for_phase(roi1)
        out1, basic1, native1, ms1 = _apply_basic_scaling_stage(
            input_bytes,
            False,
            field1_x,
            field1_y,
        )
        out1_frozen = bytes(out1)

        return _weave_interlaced_fields(out0_frozen, out1_frozen), bool(basic0 or basic1), bool(native0 and native1), float(ms0 + ms1)

    def _render_dual_phase_full_pipeline(
        input_bytes: bytes,
        phase: dict[str, object],
    ) -> tuple[bytes, bool, bool, bool, bool, bool]:
        roi0 = tuple(phase.get("roi0", (current_roi_x, current_roi_y, current_roi_w, current_roi_h)))
        roi1 = tuple(phase.get("roi1", (current_roi_x, current_roi_y, current_roi_w, current_roi_h)))
        field0_x = float(phase.get("field0_x", 0.0))
        field0_y = float(phase.get("field0_y", 0.0))
        field1_x = float(phase.get("field1_x", 0.0))
        field1_y = float(phase.get("field1_y", 0.0))

        _apply_processor_roi_for_phase(roi0)
        out0, p0, b0, a0, r0, n0 = _process_pipeline_frame(input_bytes, field0_x, field0_y)
        out0_frozen = bytes(out0)

        _apply_processor_roi_for_phase(roi1)
        out1, p1, b1, a1, r1, n1 = _process_pipeline_frame(input_bytes, field1_x, field1_y)
        out1_frozen = bytes(out1)

        return (
            _weave_interlaced_fields(out0_frozen, out1_frozen),
            bool(p0 or p1),
            bool(b0 or b1),
            bool(a0 or a1),
            bool(r0 or r1),
            bool(n0 and n1),
        )

    def _build_static_interlaced_phase_for_reinterlace(shift_x: float, shift_y: float) -> dict[str, object]:
        roi_state = (int(current_roi_x), int(current_roi_y), int(current_roi_w), int(current_roi_h))
        return {
            "roi0": roi_state,
            "roi1": roi_state,
            "field0_x": float(shift_x),
            "field0_y": float(shift_y),
            "field1_x": float(shift_x),
            "field1_y": float(shift_y),
        }

    def _split_interlaced_source_fields(input_bytes: bytes) -> tuple[bytes, bytes]:
        is_lower_field_first = int(output_field_dominance_code) == 1 if output_field_dominance_code is not None else False
        if is_lower_field_first:
            field0_input = _collapse_interlaced_to_single_field_uyvy(input_bytes, True)
            field1_input = _collapse_interlaced_to_single_field_uyvy(input_bytes, False)
        else:
            field0_input = _collapse_interlaced_to_single_field_uyvy(input_bytes, False)
            field1_input = _collapse_interlaced_to_single_field_uyvy(input_bytes, True)
        return field0_input, field1_input

    def _render_dual_phase_full_pipeline_reinterlace(
        input_bytes: bytes,
        phase: dict[str, object],
    ) -> tuple[bytes, bool, bool, bool, bool, bool]:
        field0_input, field1_input = _split_interlaced_source_fields(input_bytes)

        roi0 = tuple(phase.get("roi0", (current_roi_x, current_roi_y, current_roi_w, current_roi_h)))
        roi1 = tuple(phase.get("roi1", (current_roi_x, current_roi_y, current_roi_w, current_roi_h)))
        field0_x = float(phase.get("field0_x", 0.0))
        field0_y = float(phase.get("field0_y", 0.0))
        field1_x = float(phase.get("field1_x", 0.0))
        field1_y = float(phase.get("field1_y", 0.0))

        _apply_processor_roi_for_phase(roi0)
        out0, p0, b0, a0, r0, n0 = _process_pipeline_frame(field0_input, field0_x, field0_y)
        out0_frozen = bytes(out0)

        _apply_processor_roi_for_phase(roi1)
        out1, p1, b1, a1, r1, n1 = _process_pipeline_frame(field1_input, field1_x, field1_y)
        out1_frozen = bytes(out1)

        return (
            _weave_interlaced_fields(out0_frozen, out1_frozen),
            bool(p0 or p1),
            bool(b0 or b1),
            bool(a0 or a1),
            bool(r0 or r1),
            bool(n0 and n1),
        )

    def _render_dual_phase_basic_scaling_reinterlace(
        input_bytes: bytes,
        phase: dict[str, object],
    ) -> tuple[bytes, bool, bool, float] | None:
        if (
            _denoise_enabled()
            or reusable_process_frame_out is None
            or not hasattr(processor, "process_frame_field_phase_into")
        ):
            return None

        roi0 = tuple(phase.get("roi0", (current_roi_x, current_roi_y, current_roi_w, current_roi_h)))
        roi1 = tuple(phase.get("roi1", (current_roi_x, current_roi_y, current_roi_w, current_roi_h)))
        field0_x = float(phase.get("field0_x", 0.0))
        field0_y = float(phase.get("field0_y", 0.0))
        field1_x = float(phase.get("field1_x", 0.0))
        field1_y = float(phase.get("field1_y", 0.0))
        field0_phase = 1 if int(output_field_dominance_code or 0) == 1 else 0
        field1_phase = 1 - field0_phase
        native_shift_applied = True
        stage_start_ts = time.perf_counter()

        _apply_processor_roi_for_phase(roi0)
        native_shift_applied = _set_native_subpixel_shift(field0_x, field0_y) and native_shift_applied
        processor.process_frame_field_phase_into(input_bytes, reusable_process_frame_out, field0_phase)
        out0_frozen = bytes(reusable_process_frame_out)

        _apply_processor_roi_for_phase(roi1)
        native_shift_applied = _set_native_subpixel_shift(field1_x, field1_y) and native_shift_applied
        processor.process_frame_field_phase_into(input_bytes, reusable_process_frame_out, field1_phase)
        out1_frozen = bytes(reusable_process_frame_out)

        stage_ms = max(0.0, (time.perf_counter() - stage_start_ts) * 1000.0)
        return _weave_interlaced_fields(out0_frozen, out1_frozen), True, native_shift_applied, stage_ms

    def _apply_roi_curve(t: float, mode: str) -> float:
        clamped_t = max(0.0, min(1.0, float(t)))
        mode_name = str(mode).strip().lower()
        if mode_name == "ease_in_out":
            # Smootherstep: gentler accel/decel than cubic smoothstep.
            return clamped_t * clamped_t * clamped_t * (clamped_t * ((6.0 * clamped_t) - 15.0) + 10.0)
        if mode_name == "ease_out":
            inv = 1.0 - clamped_t
            return 1.0 - (inv * inv)
        return clamped_t

    def _start_roi_microstep_transition(
        start_roi: tuple[int, int, int, int],
        target_roi: tuple[int, int, int, int],
        duration_frames: int,
        interpolation_mode: str,
        overscan_percent: float,
        enforce_full_frame_scale_1x: bool = False,
    ) -> None:
        nonlocal roi_microstep_transition
        nonlocal roi_shift_applied_x, roi_shift_applied_y

        s_x, s_y, s_w, s_h = _normalize_worker_roi(*start_roi)
        t_x, t_y, t_w, t_h = _normalize_worker_roi(*target_roi)

        prior_transition_state = roi_microstep_transition if isinstance(roi_microstep_transition, dict) else None
        start_shift_x = float(roi_shift_applied_x)
        start_shift_y = float(roi_shift_applied_y)

        # When retargeting an in-flight interlaced transition, anchor from the
        # latest rendered field phase (field1) so recall-to-recall handoff does
        # not step backward before moving toward the new target.
        if output_mode_is_interlaced and isinstance(prior_transition_state, dict):
            prior_phase = prior_transition_state.get("interlaced_field_phase")
            if isinstance(prior_phase, dict):
                prior_roi1 = prior_phase.get("roi1")
                if isinstance(prior_roi1, tuple) and len(prior_roi1) == 4:
                    s_x, s_y, s_w, s_h = _normalize_worker_roi(
                        int(prior_roi1[0]),
                        int(prior_roi1[1]),
                        int(prior_roi1[2]),
                        int(prior_roi1[3]),
                    )
                start_shift_x = float(prior_phase.get("field1_x", start_shift_x))
                start_shift_y = float(prior_phase.get("field1_y", start_shift_y))

        total_frames = max(1, min(600, int(duration_frames)))
        mode_name = str(interpolation_mode).strip().lower()
        if mode_name not in {"linear", "ease_in_out", "ease_out"}:
            mode_name = "linear"

        sx = FRAME_W / max(1.0, float(s_w))
        sy = FRAME_H / max(1.0, float(s_h))
        start_source_dx = -(float(start_shift_x) / sx) if sx > 1e-6 else 0.0
        start_source_dy = -(float(start_shift_y) / sy) if sy > 1e-6 else 0.0

        roi_microstep_transition = {
            "start": (s_x, s_y, s_w, s_h),
            "target": (t_x, t_y, t_w, t_h),
            "total_frames": total_frames,
            "frame_progress": 0.0,
            "interpolation_mode": mode_name,
            # Preserve the currently rendered subpixel-compensated center as
            # the transition start state so the first transition frame is
            # continuous with the pre-recall frame.
            "residual": {"x": start_source_dx, "y": start_source_dy, "w": 0.0},
            "last_roi": (s_x, s_y, s_w, s_h),
            "overscan_percent": max(0.0, float(overscan_percent)),
            "enforce_full_frame_scale_1x": bool(enforce_full_frame_scale_1x),
        }

        if output_mode_is_interlaced and _interlaced_phase_controls_active():
            roi_microstep_transition["interlaced_field_shift"] = {
                "field0_x": float(start_shift_x),
                "field0_y": float(start_shift_y),
                "field1_x": float(start_shift_x),
                "field1_y": float(start_shift_y),
            }
            roi_microstep_transition["interlaced_field_phase"] = {
                "roi0": (int(s_x), int(s_y), int(s_w), int(s_h)),
                "roi1": (int(s_x), int(s_y), int(s_w), int(s_h)),
                "field0_x": float(start_shift_x),
                "field0_y": float(start_shift_y),
                "field1_x": float(start_shift_x),
                "field1_y": float(start_shift_y),
            }

    def _cancel_roi_microstep_transition(reset_shift: bool = True) -> None:
        nonlocal roi_microstep_transition
        roi_microstep_transition = None
        if reset_shift:
            _set_roi_shift_target(0.0, 0.0)

    def _build_manual_interlaced_phase_snapshot(
        prev_roi_state: tuple[int, int, int, int],
        next_roi_state: tuple[int, int, int, int],
        prev_shift_state: tuple[float, float],
        next_shift_state: tuple[float, float],
    ) -> dict[str, object]:
        phase_fraction = _clamp_interlaced_field2_phase_fraction(float(interlaced_field2_phase_fraction))
        # For manual updates, keep interpolation bounded to avoid field-order
        # overstep and visible x/y wobble on resize-only gestures.
        t_field1 = max(0.0, min(1.0, float(phase_fraction)))

        p_x, p_y, p_w, p_h = [int(v) for v in prev_roi_state]
        n_x, n_y, n_w, n_h = [int(v) for v in next_roi_state]

        p_cx = float(p_x) + (float(p_w) * 0.5)
        p_cy = float(p_y) + (float(p_h) * 0.5)
        n_cx = float(n_x) + (float(n_w) * 0.5)
        n_cy = float(n_y) + (float(n_h) * 0.5)

        size_changed = (p_w != n_w) or (p_h != n_h)
        center_dx = n_cx - p_cx
        center_dy = n_cy - p_cy
        scale_only_like = size_changed and abs(center_dx) <= 0.75 and abs(center_dy) <= 0.75

        if scale_only_like:
            center_x = n_cx
            center_y = n_cy
            interp_w = int(round(float(p_w) + ((float(n_w - p_w)) * t_field1)))
            interp_w = max(2, interp_w & ~1)
            interp_h = max(2, int(round(float(interp_w) * 9.0 / 16.0)))

            roi0 = _normalize_worker_roi(
                int(round(center_x - (float(p_w) * 0.5))),
                int(round(center_y - (float(p_h) * 0.5))),
                int(p_w),
                int(p_h),
            )
            roi1 = _normalize_worker_roi(
                int(round(center_x - (float(interp_w) * 0.5))),
                int(round(center_y - (float(interp_h) * 0.5))),
                int(interp_w),
                int(interp_h),
            )
        else:
            roi0 = _normalize_worker_roi(p_x, p_y, p_w, p_h)
            roi1 = _normalize_worker_roi(
                int(round(float(p_x) + ((float(n_x - p_x)) * t_field1))),
                int(round(float(p_y) + ((float(n_y - p_y)) * t_field1))),
                max(2, int(round(float(p_w) + ((float(n_w - p_w)) * t_field1)))) & ~1,
                max(2, int(round(float(p_h) + ((float(n_h - p_h)) * t_field1)))),
            )

        prev_shift_x, prev_shift_y = [float(v) for v in prev_shift_state]
        next_shift_x, next_shift_y = [float(v) for v in next_shift_state]
        field1_x = prev_shift_x + ((next_shift_x - prev_shift_x) * t_field1)
        field1_y = prev_shift_y + ((next_shift_y - prev_shift_y) * t_field1)

        return {
            "roi0": (int(roi0[0]), int(roi0[1]), int(roi0[2]), int(roi0[3])),
            "roi1": (int(roi1[0]), int(roi1[1]), int(roi1[2]), int(roi1[3])),
            "field0_x": float(prev_shift_x),
            "field0_y": float(prev_shift_y),
            "field1_x": float(field1_x),
            "field1_y": float(field1_y),
        }

    def _apply_manual_roi_with_subpixel_compensation(
        req_x: int,
        req_y: int,
        req_w: int,
        req_h: int,
    ) -> None:
        nonlocal current_roi_x, current_roi_y, current_roi_w, current_roi_h, rtx_vsr_error
        nonlocal manual_interlaced_phase_state, manual_interlaced_phase_until_ts, manual_interlaced_phase_pending

        prev_roi_state = (int(current_roi_x), int(current_roi_y), int(current_roi_w), int(current_roi_h))
        prev_shift_state = (float(roi_shift_applied_x), float(roi_shift_applied_y))

        prev_roi_w = current_roi_w
        prev_roi_h = current_roi_h

        # Desired ROI center uses command-space values before UYVY quantization.
        desired_w = float(max(2, min(int(req_w), FRAME_W)))
        desired_h = float(max(2, min(int(req_h), FRAME_H)))
        desired_cx = float(req_x) + (desired_w * 0.5)
        desired_cy = float(req_y) + (desired_h * 0.5)

        current_roi_x, current_roi_y, current_roi_w, current_roi_h = _normalize_worker_roi(
            int(req_x),
            int(req_y),
            int(req_w),
            int(req_h),
        )

        if current_roi_w == prev_roi_w and current_roi_h == prev_roi_h:
            processor.set_roi_position(current_roi_x, current_roi_y)
        else:
            processor.set_roi(current_roi_x, current_roi_y, current_roi_w, current_roi_h)
            if rtx_vsr_enabled:
                if rtx_vsr_engine is None:
                    rtx_vsr_error = _refresh_rtx_vsr_engine()
                elif current_roi_w != prev_roi_w or current_roi_h != prev_roi_h:
                    _schedule_rtx_roi_rebuild()

        interp_cx = float(current_roi_x) + (float(current_roi_w) * 0.5)
        interp_cy = float(current_roi_y) + (float(current_roi_h) * 0.5)
        source_dx = desired_cx - interp_cx
        source_dy = desired_cy - interp_cy

        sx = FRAME_W / max(1.0, float(current_roi_w))
        sy = FRAME_H / max(1.0, float(current_roi_h))
        max_shift_x = max(2.0, min(48.0, sx * 1.5))
        max_shift_y = max(2.0, min(48.0, sy * 1.5))
        shift_x = max(-max_shift_x, min(max_shift_x, -(source_dx * sx)))
        shift_y = max(-max_shift_y, min(max_shift_y, -(source_dy * sy)))
        _set_roi_shift_target(shift_x, shift_y)

        if output_mode_is_interlaced and _interlaced_phase_controls_active():
            manual_interlaced_phase_state = _build_manual_interlaced_phase_snapshot(
                prev_roi_state,
                (int(current_roi_x), int(current_roi_y), int(current_roi_w), int(current_roi_h)),
                prev_shift_state,
                (float(roi_shift_target_x), float(roi_shift_target_y)),
            )
            manual_interlaced_phase_until_ts = time.perf_counter() + 0.12
            manual_interlaced_phase_pending = True
        else:
            manual_interlaced_phase_state = None
            manual_interlaced_phase_until_ts = 0.0
            manual_interlaced_phase_pending = False

    def _advance_roi_microstep_transition_one_frame(progress_units: float | None = None) -> None:
        nonlocal current_roi_x, current_roi_y, current_roi_w, current_roi_h, roi_microstep_transition, output_transition_units_per_frame
        nonlocal interlaced_field2_phase_fraction

        state = roi_microstep_transition
        if state is None:
            return

        start_roi = tuple(state["start"])
        target_roi = tuple(state["target"])
        total_frames = int(state["total_frames"])
        mode_name = str(state.get("interpolation_mode", "linear"))
        residual = state.get("residual")
        if not isinstance(residual, dict):
            residual = {"x": 0.0, "y": 0.0, "w": 0.0}
            state["residual"] = residual

        step_units = float(output_transition_units_per_frame) if progress_units is None else max(0.01, float(progress_units))
        if output_mode_is_interlaced:
            # Interlaced ROI field synthesis is defined as one ROI frame-step per
            # emitted output frame; field2 phase stays fractional within that step.
            step_units = 1.0
        frame_progress = min(float(total_frames), float(state.get("frame_progress", 0.0)) + step_units)
        state["frame_progress"] = frame_progress
        is_final_frame = frame_progress >= float(total_frames)

        t = float(frame_progress) / float(max(1, total_frames))
        curved_t = _apply_roi_curve(t, mode_name)

        prev_progress = max(0.0, frame_progress - step_units)
        phase_fraction = _clamp_interlaced_field2_phase_fraction(float(interlaced_field2_phase_fraction))
        field0_progress = prev_progress
        field1_progress = max(0.0, min(float(total_frames), prev_progress + (step_units * phase_fraction)))
        t_field0 = min(1.0, field0_progress / float(max(1, total_frames)))
        t_field1 = min(1.0, field1_progress / float(max(1, total_frames)))
        curved_t_field0 = _apply_roi_curve(t_field0, mode_name)
        curved_t_field1 = _apply_roi_curve(t_field1, mode_name)

        s_x, s_y, s_w, s_h = [float(v) for v in start_roi]
        t_x, t_y, t_w, t_h = [float(v) for v in target_roi]

        start_cx = s_x + (s_w * 0.5)
        start_cy = s_y + (s_h * 0.5)
        target_cx = t_x + (t_w * 0.5)
        target_cy = t_y + (t_h * 0.5)

        ideal_cx = start_cx + ((target_cx - start_cx) * curved_t)
        ideal_cy = start_cy + ((target_cy - start_cy) * curved_t)
        ideal_w = s_w + ((t_w - s_w) * curved_t)
        ideal_w_field0 = s_w + ((t_w - s_w) * curved_t_field0)
        ideal_w_field1 = s_w + ((t_w - s_w) * curved_t_field1)
        ideal_cx_field0 = start_cx + ((target_cx - start_cx) * curved_t_field0)
        ideal_cy_field0 = start_cy + ((target_cy - start_cy) * curved_t_field0)
        ideal_cx_field1 = start_cx + ((target_cx - start_cx) * curved_t_field1)
        ideal_cy_field1 = start_cy + ((target_cy - start_cy) * curved_t_field1)

        residual_w = float(residual.get("w", 0.0))
        desired_cx = ideal_cx + float(residual.get("x", 0.0))
        desired_cy = ideal_cy + float(residual.get("y", 0.0))
        desired_w = ideal_w + residual_w

        target_scale = FRAME_W / max(1.0, t_w)
        overscan_pct = float(state.get("overscan_percent", 0.0))
        if target_scale >= 4.0 and overscan_pct > 0.0:
            overscan_weight = max(0.0, 4.0 * curved_t * (1.0 - curved_t))
            desired_w_backend = desired_w * (1.0 + ((overscan_pct / 100.0) * overscan_weight))
            overscan_weight_field0 = max(0.0, 4.0 * curved_t_field0 * (1.0 - curved_t_field0))
            overscan_weight_field1 = max(0.0, 4.0 * curved_t_field1 * (1.0 - curved_t_field1))
            desired_w_backend_field0 = (ideal_w_field0 + residual_w) * (1.0 + ((overscan_pct / 100.0) * overscan_weight_field0))
            desired_w_backend_field1 = (ideal_w_field1 + residual_w) * (1.0 + ((overscan_pct / 100.0) * overscan_weight_field1))
        else:
            desired_w_backend = desired_w
            desired_w_backend_field0 = ideal_w_field0 + residual_w
            desired_w_backend_field1 = ideal_w_field1 + residual_w

        d_x = int(target_roi[0]) - int(start_roi[0])
        d_y = int(target_roi[1]) - int(start_roi[1])
        d_w = int(target_roi[2]) - int(start_roi[2])

        def _quantize_directional(value: float, delta: int, quantum: int) -> int:
            q = max(1, int(quantum))
            scaled = value / float(q)
            if delta > 0:
                return int(math.floor(scaled)) * q
            if delta < 0:
                return int(math.ceil(scaled)) * q
            return int(round(scaled)) * q

        quant_w = _quantize_directional(desired_w_backend, d_w, 2)
        quant_w = max(2, quant_w & ~1)
        quant_h = max(2, int(round(quant_w * 9.0 / 16.0)))

        desired_x = desired_cx - (quant_w * 0.5)
        desired_y = desired_cy - (quant_h * 0.5)

        quant_x = _quantize_directional(desired_x, d_x, 2)
        quant_y = _quantize_directional(desired_y, d_y, 1)

        i_x, i_y, i_w, i_h = _normalize_worker_roi(quant_x, quant_y, quant_w, quant_h)

        last_roi = state.get("last_roi")
        if not isinstance(last_roi, tuple) or len(last_roi) != 4:
            last_roi = (current_roi_x, current_roi_y, current_roi_w, current_roi_h)

        mono_x, mono_y, mono_w, mono_h = i_x, i_y, i_w, i_h
        if d_x > 0:
            mono_x = max(mono_x, int(last_roi[0]))
        elif d_x < 0:
            mono_x = min(mono_x, int(last_roi[0]))
        if d_y > 0:
            mono_y = max(mono_y, int(last_roi[1]))
        elif d_y < 0:
            mono_y = min(mono_y, int(last_roi[1]))
        if d_w > 0:
            mono_w = max(mono_w, int(last_roi[2]))
        elif d_w < 0:
            mono_w = min(mono_w, int(last_roi[2]))

        target_h_delta = int(target_roi[3]) - int(start_roi[3])
        if target_h_delta > 0:
            mono_h = max(mono_h, int(last_roi[3]))
        elif target_h_delta < 0:
            mono_h = min(mono_h, int(last_roi[3]))

        i_x, i_y, i_w, i_h = _normalize_worker_roi(mono_x, mono_y, mono_w, mono_h)
        state["last_roi"] = (i_x, i_y, i_w, i_h)

        interp_cx = float(i_x) + (float(i_w) * 0.5)
        interp_cy = float(i_y) + (float(i_h) * 0.5)
        residual["x"] = desired_cx - interp_cx
        residual["y"] = desired_cy - interp_cy
        residual["w"] = desired_w_backend - float(i_w)

        source_dx = ideal_cx - interp_cx
        source_dy = ideal_cy - interp_cy
        sx = FRAME_W / max(1.0, float(i_w))
        sy = FRAME_H / max(1.0, float(i_h))
        max_shift_x = max(2.0, min(48.0, sx * 1.5))
        max_shift_y = max(2.0, min(48.0, sy * 1.5))
        shift_x = max(-max_shift_x, min(max_shift_x, -(source_dx * sx)))
        shift_y = max(-max_shift_y, min(max_shift_y, -(source_dy * sy)))

        quant_w_field0 = _quantize_directional(desired_w_backend_field0, d_w, 2)
        quant_w_field0 = max(2, quant_w_field0 & ~1)
        quant_h_field0 = max(2, int(round(quant_w_field0 * 9.0 / 16.0)))
        quant_w_field1 = _quantize_directional(desired_w_backend_field1, d_w, 2)
        quant_w_field1 = max(2, quant_w_field1 & ~1)
        quant_h_field1 = max(2, int(round(quant_w_field1 * 9.0 / 16.0)))

        desired_x_field0 = ideal_cx_field0 - (quant_w_field0 * 0.5)
        desired_y_field0 = ideal_cy_field0 - (quant_h_field0 * 0.5)
        desired_x_field1 = ideal_cx_field1 - (quant_w_field1 * 0.5)
        desired_y_field1 = ideal_cy_field1 - (quant_h_field1 * 0.5)

        roi0_x = _quantize_directional(desired_x_field0, d_x, 2)
        roi0_y = _quantize_directional(desired_y_field0, d_y, 1)
        roi1_x = _quantize_directional(desired_x_field1, d_x, 2)
        roi1_y = _quantize_directional(desired_y_field1, d_y, 1)
        roi0_x, roi0_y, roi0_w, roi0_h = _normalize_worker_roi(roi0_x, roi0_y, quant_w_field0, quant_h_field0)
        roi1_x, roi1_y, roi1_w, roi1_h = _normalize_worker_roi(roi1_x, roi1_y, quant_w_field1, quant_h_field1)

        interp_cx_field0 = float(roi0_x) + (float(roi0_w) * 0.5)
        interp_cy_field0 = float(roi0_y) + (float(roi0_h) * 0.5)
        interp_cx_field1 = float(roi1_x) + (float(roi1_w) * 0.5)
        interp_cy_field1 = float(roi1_y) + (float(roi1_h) * 0.5)
        source_dx_field0 = ideal_cx_field0 - interp_cx_field0
        source_dy_field0 = ideal_cy_field0 - interp_cy_field0
        source_dx_field1 = ideal_cx_field1 - interp_cx_field1
        source_dy_field1 = ideal_cy_field1 - interp_cy_field1
        sx0 = FRAME_W / max(1.0, float(roi0_w))
        sy0 = FRAME_H / max(1.0, float(roi0_h))
        sx1 = FRAME_W / max(1.0, float(roi1_w))
        sy1 = FRAME_H / max(1.0, float(roi1_h))
        max_shift_x0 = max(2.0, min(48.0, sx0 * 1.5))
        max_shift_y0 = max(2.0, min(48.0, sy0 * 1.5))
        max_shift_x1 = max(2.0, min(48.0, sx1 * 1.5))
        max_shift_y1 = max(2.0, min(48.0, sy1 * 1.5))
        field0_shift_x = max(-max_shift_x0, min(max_shift_x0, -(source_dx_field0 * sx0)))
        field0_shift_y = max(-max_shift_y0, min(max_shift_y0, -(source_dy_field0 * sy0)))
        field1_shift_x = max(-max_shift_x1, min(max_shift_x1, -(source_dx_field1 * sx1)))
        field1_shift_y = max(-max_shift_y1, min(max_shift_y1, -(source_dy_field1 * sy1)))

        phase_fraction_active = abs(_clamp_interlaced_field2_phase_fraction(float(interlaced_field2_phase_fraction))) > 1e-4
        if not phase_fraction_active:
            state.pop("interlaced_field_shift", None)
            state.pop("interlaced_field_phase", None)
            _set_roi_shift_target(shift_x, shift_y)

            roi_changed = (
                i_x != current_roi_x
                or i_y != current_roi_y
                or i_w != current_roi_w
                or i_h != current_roi_h
            )
            if roi_changed:
                prev_roi_w = current_roi_w
                prev_roi_h = current_roi_h
                current_roi_x, current_roi_y, current_roi_w, current_roi_h = i_x, i_y, i_w, i_h
                if current_roi_w == prev_roi_w and current_roi_h == prev_roi_h:
                    processor.set_roi_position(current_roi_x, current_roi_y)
                else:
                    processor.set_roi(current_roi_x, current_roi_y, current_roi_w, current_roi_h)
                    if rtx_vsr_enabled:
                        if rtx_vsr_engine is None:
                            _ = _refresh_rtx_vsr_engine()
                        elif current_roi_w != prev_roi_w or current_roi_h != prev_roi_h:
                            _schedule_rtx_roi_rebuild()

            transition_complete = (
                current_roi_x == int(target_roi[0])
                and current_roi_y == int(target_roi[1])
                and current_roi_w == int(target_roi[2])
                and current_roi_h == int(target_roi[3])
            ) or (
                is_final_frame
                and abs(current_roi_x - int(target_roi[0])) <= 2
                and abs(current_roi_y - int(target_roi[1])) <= 1
                and abs(current_roi_w - int(target_roi[2])) <= 2
                and abs(current_roi_h - int(target_roi[3])) <= 2
            )
            if transition_complete:
                enforce_full_frame_scale = bool(state.get("enforce_full_frame_scale_1x", False))
                if enforce_full_frame_scale and current_roi_x == 0 and current_roi_y == 0 and current_roi_w == FRAME_W and current_roi_h == FRAME_H:
                    try:
                        processor.set_sr_mode_auto()
                    except Exception:
                        pass
                _set_roi_shift_target(0.0, 0.0)
                roi_microstep_transition = None
            return

        state["interlaced_field_shift"] = {
            "field0_x": float(field0_shift_x),
            "field0_y": float(field0_shift_y),
            "field1_x": float(field1_shift_x),
            "field1_y": float(field1_shift_y),
        }

        # Keep per-field phase data available on every transition step so
        # interlaced render paths can always produce distinct field-time samples,
        # including zoom/scale progression per field.
        state["interlaced_field_phase"] = {
            "roi0": (int(roi0_x), int(roi0_y), int(roi0_w), int(roi0_h)),
            "roi1": (int(roi1_x), int(roi1_y), int(roi1_w), int(roi1_h)),
            "field0_x": float(field0_shift_x),
            "field0_y": float(field0_shift_y),
            "field1_x": float(field1_shift_x),
            "field1_y": float(field1_shift_y),
        }
        _set_roi_shift_target(shift_x, shift_y)

        roi_changed = (
            i_x != current_roi_x
            or i_y != current_roi_y
            or i_w != current_roi_w
            or i_h != current_roi_h
        )
        if roi_changed:
            prev_roi_w = current_roi_w
            prev_roi_h = current_roi_h
            current_roi_x, current_roi_y, current_roi_w, current_roi_h = i_x, i_y, i_w, i_h
            if current_roi_w == prev_roi_w and current_roi_h == prev_roi_h:
                processor.set_roi_position(current_roi_x, current_roi_y)
            else:
                processor.set_roi(current_roi_x, current_roi_y, current_roi_w, current_roi_h)
                if rtx_vsr_enabled:
                    if rtx_vsr_engine is None:
                        _ = _refresh_rtx_vsr_engine()
                    elif current_roi_w != prev_roi_w or current_roi_h != prev_roi_h:
                        _schedule_rtx_roi_rebuild()

        transition_complete = (
            current_roi_x == int(target_roi[0])
            and current_roi_y == int(target_roi[1])
            and current_roi_w == int(target_roi[2])
            and current_roi_h == int(target_roi[3])
        ) or (
            is_final_frame
            and abs(current_roi_x - int(target_roi[0])) <= 2
            and abs(current_roi_y - int(target_roi[1])) <= 1
            and abs(current_roi_w - int(target_roi[2])) <= 2
            and abs(current_roi_h - int(target_roi[3])) <= 2
        )
        if transition_complete:
            enforce_full_frame_scale = bool(state.get("enforce_full_frame_scale_1x", False))
            if enforce_full_frame_scale and current_roi_x == 0 and current_roi_y == 0 and current_roi_w == FRAME_W and current_roi_h == FRAME_H:
                try:
                    processor.set_sr_mode_auto()
                except Exception:
                    pass
            _set_roi_shift_target(0.0, 0.0)
            roi_microstep_transition = None

    def _advance_roi_microstep_transition_for_output_frame() -> None:
        nonlocal roi_microstep_transition, output_mode_is_interlaced
        if roi_microstep_transition is None:
            return

        if not output_mode_is_interlaced:
            _advance_roi_microstep_transition_one_frame()
            return

        # Interlaced path advances one transition step per output frame.
        # Field0/field1 in-between timing is synthesized inside
        # _advance_roi_microstep_transition_one_frame using
        # interlaced_field2_phase_fraction (0.5 = true half-step).
        _advance_roi_microstep_transition_one_frame(progress_units=1.0)

    def _cleanup_ai_async() -> None:
        nonlocal ai_sr_executor, ai_sr_futures
        if ai_sr_futures:
            for ai_future in ai_sr_futures:
                ai_future.cancel()
            ai_sr_futures = []
        if ai_sr_executor is not None:
            ai_sr_executor.shutdown(wait=False, cancel_futures=True)
            ai_sr_executor = None

    def _collect_ai_future_result() -> bool:
        nonlocal ai_sr_latest_output_frame, ai_sr_latest_output_ts, ai_sr_completed_frames, ai_sr_futures
        new_result_ready = False
        if not ai_sr_futures:
            return False

        pending_futures: list[Future[bytes]] = []
        for ai_future in ai_sr_futures:
            if not ai_future.done():
                pending_futures.append(ai_future)
                continue
            try:
                ai_sr_latest_output_frame = ai_future.result()
                ai_sr_latest_output_ts = time.perf_counter()
                ai_sr_completed_frames += 1
                new_result_ready = True
            except Exception as ai_exc:
                _safe_put({"type": "warning", "warning": f"AI SR inference failed: {ai_exc}"})

        ai_sr_futures = pending_futures
        return new_result_ready

    def _ai_inference_busy() -> bool:
        _collect_ai_future_result()
        return len(ai_sr_futures) > 0

    def _ai_submit_due(now: float) -> bool:
        target_spacing_ms = 1000.0 / max(1.0, float(ai_sr_frame_interval))
        if ai_sr_submit_spacing_ms > 0.0:
            target_spacing_ms = max(target_spacing_ms, float(ai_sr_submit_spacing_ms))
        if ai_sr_last_submit_ts <= 0.0:
            return True
        return ((now - ai_sr_last_submit_ts) * 1000.0) >= target_spacing_ms

    def _apply_ai_sr_non_blocking(frame_bytes: bytes, roi: tuple[int, int, int, int], method: str) -> tuple[bytes, bool, bool]:
        nonlocal ai_sr_frame_counter, ai_sr_latest_output_frame, ai_sr_latest_output_ts, ai_sr_completed_frames, ai_sr_warmup_pending, ai_sr_futures
        nonlocal ai_sr_hold_last_frame, ai_sr_max_hold_ms, ai_sr_max_inflight
        nonlocal ai_sr_submit_spacing_ms, ai_sr_last_submit_ts
        if ai_sr_engine is None:
            return frame_bytes, False, False

        new_ai_result_ready = _collect_ai_future_result()

        ai_sr_frame_counter += 1
        now = time.perf_counter()
        run_ai_inference = _ai_submit_due(now)

        if run_ai_inference and len(ai_sr_futures) < ai_sr_max_inflight and ai_sr_executor is not None:
            ai_input_bytes = frame_bytes if isinstance(frame_bytes, bytes) else bytes(frame_bytes)
            ai_roi = roi if isinstance(roi, tuple) else tuple(roi)
            ai_method = method if isinstance(method, str) else str(method)
            ai_future = ai_sr_executor.submit(ai_sr_engine.process_uyvy_frame_roi_to_output, ai_input_bytes, ai_roi, ai_method)
            ai_sr_futures.append(ai_future)
            ai_sr_last_submit_ts = now

        # Keep pipeline cadence stable: do not run a synchronous warmup frame.
        # Until the first async result arrives, render a live baseline resize.
        if ai_sr_warmup_pending and ai_sr_latest_output_frame is not None:
            ai_sr_warmup_pending = False

        if ai_sr_latest_output_frame is not None:
            ai_sr_warmup_pending = False
            if new_ai_result_ready:
                return ai_sr_latest_output_frame, True, True

            # Reuse stale AI output only when explicitly requested.
            if ai_sr_hold_last_frame:
                if ai_sr_max_hold_ms > 0.0 and ai_sr_latest_output_ts > 0.0:
                    age_ms = max(0.0, (time.perf_counter() - float(ai_sr_latest_output_ts)) * 1000.0)
                    if age_ms > float(ai_sr_max_hold_ms):
                        return frame_bytes, False, False
                return ai_sr_latest_output_frame, True, False

        # No implicit fallback path: until async AI output exists, pass through.
        return frame_bytes, False, False

    def _apply_ai_sr(frame_bytes: bytes, roi: tuple[int, int, int, int], method: str) -> tuple[bytes, bool]:
        nonlocal ai_sr_frame_counter, ai_sr_applied_frames, ai_sr_reused_frames, ai_sr_passthrough_frames
        nonlocal ai_sr_last_submit_ts
        if ai_sr_engine is None:
            return frame_bytes, False

        if ai_sr_strict:
            ai_sr_frame_counter += 1
            now = time.perf_counter()
            run_ai_inference = _ai_submit_due(now)
            if not run_ai_inference:
                ai_sr_passthrough_frames += 1
                return frame_bytes, False
            try:
                ai_sr_last_submit_ts = now
                out = ai_sr_engine.process_uyvy_frame_roi_to_output(frame_bytes, roi, method)
                ai_sr_applied_frames += 1
                return out, True
            except Exception as ai_exc:
                ai_sr_passthrough_frames += 1
                _safe_put({"type": "warning", "warning": f"AI SR strict inference failed: {ai_exc}"})
                return frame_bytes, False

        output_frame, ai_output_used, ai_fresh_output = _apply_ai_sr_non_blocking(frame_bytes, roi, method)
        if ai_output_used:
            if ai_fresh_output:
                ai_sr_applied_frames += 1
            else:
                ai_sr_reused_frames += 1
        else:
            ai_sr_passthrough_frames += 1
        return output_frame, ai_output_used

    def _apply_rtx_vsr(frame_bytes: bytes, roi: tuple[int, int, int, int]) -> bytes:
        if rtx_vsr_engine is None:
            return frame_bytes

        roi_x, roi_y, roi_w, roi_h = _normalize_worker_roi(int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
        yuv422 = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 2)
        roi_yuv = np.ascontiguousarray(yuv422[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w, :])
        roi_rgb = _uyvy_to_rgb_limited(roi_yuv, current_color_space, current_color_range)
        roi_rgba = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2RGBA)

        engine_in_w = max(2, int(getattr(rtx_vsr_engine, "input_width", roi_w)) & ~1)
        engine_in_h = max(2, int(getattr(rtx_vsr_engine, "input_height", roi_h)))
        if roi_rgba.shape[1] != engine_in_w or roi_rgba.shape[0] != engine_in_h:
            # Keep RTX active while ROI is being resized by adapting the current
            # ROI crop to the engine's fixed input dimensions.
            roi_rgba = cv2.resize(roi_rgba, (engine_in_w, engine_in_h), interpolation=cv2.INTER_CUBIC)

        sr_rgba = rtx_vsr_engine.process_rgba(roi_rgba)
        if not isinstance(sr_rgba, np.ndarray):
            sr_rgba = np.asarray(sr_rgba)

        if sr_rgba.shape[0] != FRAME_H or sr_rgba.shape[1] != FRAME_W:
            interpolation = RTX_POST_SCALE_METHOD_TO_CV2_INTERP.get(rtx_vsr_post_scale_method, cv2.INTER_CUBIC)
            sr_rgba = cv2.resize(sr_rgba, (FRAME_W, FRAME_H), interpolation=interpolation)

        sr_rgb = cv2.cvtColor(sr_rgba, cv2.COLOR_RGBA2RGB)
        sr_yuv422 = _rgb_to_uyvy_limited(sr_rgb, current_color_space, current_color_range)
        return sr_yuv422.tobytes()

    def _apply_ai_sr_performance_profile() -> None:
        nonlocal ai_sr_runtime_note
        ai_sr_runtime_note = None
        # Intentionally no automatic overrides: strict mode, frame interval,
        # and inference divisor remain exactly as configured by the user.

    def _put_latest_stage_frame(stage_queue: queue.Queue[_StageFrame], item: _StageFrame) -> bool:
        # Keep newest frames and drop oldest when saturated to bound latency.
        try:
            stage_queue.put_nowait(item)
            return False
        except queue.Full:
            try:
                stage_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                stage_queue.put_nowait(item)
                return True
            except queue.Full:
                return True

    def _put_stage_frame_fifo_drop_newest(stage_queue: queue.Queue[_StageFrame], item: _StageFrame) -> bool:
        # Preserve FIFO order under saturation by dropping the incoming frame.
        try:
            stage_queue.put_nowait(item)
            return False
        except queue.Full:
            return True

    def _put_stage_frame_blocking(stage_queue: queue.Queue[_StageFrame], item: _StageFrame, timeout_s: float = 0.03) -> bool:
        # Backpressure producers instead of evicting older frames.
        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        while not pipeline_stop_event.is_set():
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                return False
            try:
                stage_queue.put(item, timeout=min(0.005, remaining))
                return True
            except queue.Full:
                continue
        return False

    def _denoise_enabled() -> bool:
        return current_denoise_method not in {"off", "none"} and current_denoise_strength > 0.001

    def _is_valid_uyvy_frame(frame_bytes: bytes | bytearray | memoryview) -> bool:
        return isinstance(frame_bytes, (bytes, bytearray)) and len(frame_bytes) == (UYVY_ROW_BYTES * FRAME_H)

    def _set_native_subpixel_shift(shift_x: float, shift_y: float) -> bool:
        nonlocal native_subpixel_warning_emitted
        if not hasattr(processor, "set_subpixel_shift"):
            return False
        try:
            processor.set_subpixel_shift(float(shift_x), float(shift_y))
            return True
        except Exception as exc:
            if not native_subpixel_warning_emitted:
                _safe_put({"type": "warning", "warning": f"Native subpixel shift unavailable, using OpenCV fallback: {exc}"})
                native_subpixel_warning_emitted = True
            return False

    def _is_preprocess_enabled() -> bool:
        return bool(current_deinterlace_enabled) or _denoise_enabled()

    def _is_any_sr_selected() -> bool:
        ai_stage_active = ai_sr_enabled and ai_sr_engine is not None
        rtx_stage_active = rtx_vsr_enabled and rtx_vsr_engine is not None
        return bool(ai_stage_active or rtx_stage_active)

    def _should_apply_cpu_subpixel_fallback() -> bool:
        # When any SR path is selected, prioritize throughput over tiny
        # subpixel compensation shifts in Python/OpenCV.
        return not _is_any_sr_selected()

    def _interlaced_phase_controls_active() -> bool:
        phase_fraction = _clamp_interlaced_field2_phase_fraction(float(interlaced_field2_phase_fraction))
        return abs(phase_fraction) > 1e-4

    prev_reinterlace_frame_bytes: bytes | None = None

    def _reinterlace_enabled_for_output() -> bool:
        return bool(output_mode_is_interlaced and current_reinterlace_enabled)

    def _apply_reinterlace_from_previous_frame_if_needed(frame_bytes: bytes) -> bytes:
        nonlocal prev_reinterlace_frame_bytes

        if not _reinterlace_enabled_for_output():
            prev_reinterlace_frame_bytes = None
            return frame_bytes

        if len(frame_bytes) != (UYVY_ROW_BYTES * FRAME_H):
            return frame_bytes

        prev_frame = prev_reinterlace_frame_bytes
        prev_reinterlace_frame_bytes = frame_bytes
        if prev_frame is None or len(prev_frame) != (UYVY_ROW_BYTES * FRAME_H):
            return frame_bytes

        return _weave_interlaced_fields(prev_frame, frame_bytes)

    def _basic_scaling_enabled() -> bool:
        if not bool(basic_scaling_enabled):
            return False
        # Disable native CUDA basic-scaling path whenever an SR path is selected.
        return not _is_any_sr_selected()

    def _is_preprocess_stage_enabled() -> bool:
        if not _is_preprocess_enabled():
            return False

        ai_stage_active = ai_sr_enabled and ai_sr_engine is not None
        if ai_stage_active:
            return not bool(getattr(ai_sr_engine, "uses_native_preprocess", lambda: False)())

        basic_stage_active = _basic_scaling_enabled() and (not ai_stage_active)
        # When native basic scaling is active without AI, preprocess is fused into
        # that stage.
        fuse_preprocess_into_basic = basic_stage_active and not ai_stage_active
        return not fuse_preprocess_into_basic

    def _preprocess_stage(frame_bytes: bytes) -> tuple[bytes, bool]:
        nonlocal preprocess_noop_warning_emitted
        if not _is_preprocess_enabled():
            return frame_bytes, False

        native_out = frame_bytes
        if hasattr(processor, "process_frame_preprocess_only"):
            native_out = _run_native_uyvy_process(
                frame_bytes,
                "process_frame_preprocess_only",
                "process_frame_preprocess_only_into",
                reusable_process_frame_preprocess_only_out,
            )
        elif current_deinterlace_enabled and hasattr(processor, "process_frame_deinterlace_only"):
            native_out = _run_native_uyvy_process(
                frame_bytes,
                "process_frame_deinterlace_only",
                "process_frame_deinterlace_only_into",
                reusable_process_frame_deinterlace_only_out,
            )
        else:
            return frame_bytes, False

        native_invalid = (not _is_valid_uyvy_frame(native_out)) or _looks_zeroed_uyvy_frame(native_out)
        if native_invalid:
            if not preprocess_noop_warning_emitted:
                _safe_put(
                    {
                        "type": "warning",
                        "warning": "Native preprocess produced invalid/zero output in GPU-only mode.",
                    }
                )
                preprocess_noop_warning_emitted = True
            return frame_bytes, False

        return native_out, True

    def _apply_basic_scaling_stage(
        frame_bytes: bytes,
        preprocess_already_applied: bool,
        shift_x: float,
        shift_y: float,
    ) -> tuple[bytes, bool, bool, float]:
        nonlocal zeroed_output_warning_emitted
        if not _basic_scaling_enabled():
            return frame_bytes, False, False, 0.0

        native_shift_applied = False
        if _set_native_subpixel_shift(shift_x, shift_y):
            native_shift_applied = _has_effective_subpixel_shift(shift_x, shift_y)

        basic_stage_start_ts = time.perf_counter()
        if preprocess_already_applied and hasattr(processor, "process_frame_no_deinterlace"):
            scaled = _run_native_uyvy_process(
                frame_bytes,
                "process_frame_no_deinterlace",
                "process_frame_no_deinterlace_into",
                reusable_process_frame_no_deinterlace_out,
            )
        else:
            scaled = _run_native_uyvy_process(
                frame_bytes,
                "process_frame",
                "process_frame_into",
                reusable_process_frame_out,
            )
        basic_stage_ms = max(0.0, (time.perf_counter() - basic_stage_start_ts) * 1000.0)

        # Strict GPU-only mode: no CPU fallback.
        if (not _is_valid_uyvy_frame(scaled)) or _looks_zeroed_uyvy_frame(scaled):
            if not zeroed_output_warning_emitted:
                _safe_put(
                    {
                        "type": "warning",
                        "warning": "Native CUDA basic scaling produced invalid/zero output in GPU-only mode.",
                    }
                )
                zeroed_output_warning_emitted = True
            return frame_bytes, False, False, basic_stage_ms

        return scaled, True, native_shift_applied, basic_stage_ms

    def _record_basic_scaling_timing(frame_ms: float) -> None:
        nonlocal basic_scaling_last_frame_ms, basic_scaling_avg_frame_ms, basic_scaling_max_frame_ms, basic_scaling_timing_samples
        sample_ms = max(0.0, float(frame_ms))
        basic_scaling_last_frame_ms = sample_ms
        basic_scaling_timing_samples += 1
        if basic_scaling_avg_frame_ms is None:
            basic_scaling_avg_frame_ms = sample_ms
        else:
            basic_scaling_avg_frame_ms = (0.92 * float(basic_scaling_avg_frame_ms)) + (0.08 * sample_ms)
        basic_scaling_max_frame_ms = max(float(basic_scaling_max_frame_ms), sample_ms)

    def _apply_rtx_stage(frame_bytes: bytes) -> tuple[bytes, bool]:
        if not (rtx_vsr_enabled and rtx_vsr_engine is not None):
            return frame_bytes, False
        try:
            rtx_out = _apply_rtx_vsr(
                frame_bytes,
                (current_roi_x, current_roi_y, current_roi_w, current_roi_h),
            )
            if _looks_zeroed_uyvy_frame(rtx_out):
                return frame_bytes, False
            return rtx_out, True
        except Exception as rtx_exc:
            _safe_put({"type": "warning", "warning": f"RTX VSR inference failed: {rtx_exc}"})
            return frame_bytes, False

    def _build_stage_stack() -> list[str]:
        # Plugin-style stage ordering: each enabled filter is appended in order,
        # and output of one stage becomes input to the next stage.
        stack: list[str] = []
        ai_stage_active = ai_sr_enabled and ai_sr_engine is not None
        basic_stage_active = _basic_scaling_enabled() and (not ai_stage_active)

        if _is_preprocess_stage_enabled():
            stack.append("preprocess")
        if ai_stage_active:
            stack.append("ai_sr")
        if rtx_vsr_enabled and rtx_vsr_engine is not None:
            stack.append("rtx_vsr")
        # In AI basic-cuda chain mode, basic scaling is intentionally kept after
        # AI so final output sizing is handled by native CUDA path.
        if basic_stage_active:
            stack.append("basic_scaling")
        return stack

    def _process_pipeline_frame(frame_bytes: bytes, shift_x: float, shift_y: float) -> tuple[bytes, bool, bool, bool, bool, bool]:
        # Canonical plugin chain:
        # preprocess (deinterlace/denoise) -> AI SR -> RTX VSR -> basic scaling.
        stage_stack = _build_stage_stack()
        if not stage_stack:
            return frame_bytes, False, False, False, False, False

        preprocess_applied = False
        working = frame_bytes
        ai_applied = False
        rtx_applied = False
        basic_applied = False
        native_shift_applied = False

        for stage_name in stage_stack:
            if stage_name == "preprocess":
                working, preprocess_applied = _preprocess_stage(working)
                continue

            if stage_name == "ai_sr":
                ai_out, ai_applied = _apply_ai_sr(
                    working,
                    (current_roi_x, current_roi_y, current_roi_w, current_roi_h),
                    current_basic_scaling_method,
                )
                if _looks_zeroed_uyvy_frame(ai_out):
                    ai_out = working
                    ai_applied = False
                working = ai_out
                continue

            if stage_name == "rtx_vsr":
                working, rtx_applied = _apply_rtx_stage(working)
                continue

            if stage_name == "basic_scaling":
                working, basic_applied, native_shift_applied, basic_stage_ms = _apply_basic_scaling_stage(
                    working,
                    preprocess_applied,
                    shift_x,
                    shift_y,
                )
                if basic_applied:
                    _record_basic_scaling_timing(basic_stage_ms)
                continue

        return working, preprocess_applied, basic_applied, ai_applied, rtx_applied, native_shift_applied

    def _stop_live_pipeline() -> None:
        nonlocal pipeline_running, capture_thread, preprocess_thread, upscale_thread, output_thread
        nonlocal upscale_extra_threads, parallel_basic_processors, parallel_basic_worker_count
        nonlocal latest_timecode_info
        if not pipeline_running:
            return
        pipeline_stop_event.set()
        threads_to_join = [capture_thread, upscale_thread, output_thread, *upscale_extra_threads]
        for thread in threads_to_join:
            if thread is not None:
                thread.join(timeout=1.0)
        capture_thread = None
        preprocess_thread = None
        upscale_thread = None
        upscale_extra_threads = []
        output_thread = None
        parallel_basic_processors = []
        parallel_basic_worker_count = 1
        latest_timecode_info = {
            "present": False,
            "text": "",
            "format_code": 0,
            "format_name": "",
        }
        pipeline_running = False

    def _start_live_pipeline() -> None:
        nonlocal pipeline_running
        nonlocal q_capture_to_preprocess, q_preprocess_to_upscale, q_upscale_to_output
        nonlocal frame_id_counter, capture_drop_count, preprocess_drop_count, upscale_drop_count
        nonlocal capture_thread, preprocess_thread, upscale_thread, output_thread
        nonlocal upscale_extra_threads, parallel_basic_processors, parallel_basic_worker_count
        nonlocal latest_input_frame, latest_output_frame, latest_timecode_info, latest_effective_sr_scale, processed_frame_counter, started_perf_ts
        nonlocal latest_rtx_vsr_applied, latest_rtx_effect_mean_abs_luma
        nonlocal output_nominal_fps, output_frame_period_s, output_mode_is_interlaced, output_transition_units_per_frame
        nonlocal zeroed_output_warning_emitted, preprocess_noop_warning_emitted, native_subpixel_warning_emitted
        nonlocal rtx_effect_sample_counter
        nonlocal stage_preprocess_applied_frames, stage_basic_applied_frames, stage_ai_applied_frames, stage_rtx_applied_frames, stage_passthrough_frames
        nonlocal last_stage_preprocess_applied, last_stage_basic_applied, last_stage_ai_applied, last_stage_rtx_applied
        nonlocal last_stage_stack
        nonlocal basic_scaling_last_frame_ms, basic_scaling_avg_frame_ms, basic_scaling_max_frame_ms, basic_scaling_timing_samples
        nonlocal timing_frames_emitted, timing_deadline_miss_events, timing_deadline_miss_streak, timing_deadline_miss_max_streak
        nonlocal timing_e2e_ms_last, timing_e2e_ms_ema, timing_e2e_ms_peak
        nonlocal timing_process_ms_ema, timing_process_ms_peak
        nonlocal timing_capture_queue_ms_ema, timing_capture_queue_ms_peak
        nonlocal timing_output_queue_ms_ema, timing_output_queue_ms_peak
        nonlocal timing_output_wait_ms_ema, timing_output_wait_ms_peak
        nonlocal timing_emit_call_ms_ema, timing_emit_call_ms_peak
        nonlocal timing_deadline_late_ms_ema, timing_deadline_late_ms_peak
        nonlocal timing_last_path
        if capture_session is None or output_session is None:
            raise RuntimeError("Cannot start pipeline without active DeckLink sessions")

        _stop_live_pipeline()
        pipeline_stop_event.clear()
        parallel_basic_processors = []
        parallel_basic_worker_count = 1
        if parallel_basic_max_inflight > 1 and _is_live_basic_scaling_fast_mode():
            for _ in range(parallel_basic_max_inflight):
                try:
                    parallel_proc, _ = _create_processor(module, startup_config)
                except Exception:
                    parallel_basic_processors = []
                    parallel_basic_worker_count = 1
                    _safe_put(
                        {
                            "type": "warning",
                            "warning": "Parallel basic scaling init failed; falling back to single-worker scaling",
                        }
                    )
                    break
                parallel_basic_processors.append(parallel_proc)
            if parallel_basic_processors:
                parallel_basic_worker_count = len(parallel_basic_processors)

        q_capture_to_preprocess = queue.Queue(maxsize=max(2, parallel_basic_worker_count * 2))
        q_preprocess_to_upscale = None
        q_upscale_to_output = queue.Queue(maxsize=max(1, parallel_basic_worker_count))
        frame_id_counter = 0
        capture_drop_count = 0
        preprocess_drop_count = 0
        upscale_drop_count = 0
        latest_input_frame = None
        latest_output_frame = None
        latest_timecode_info = {
            "present": False,
            "text": "",
            "format_code": 0,
            "format_name": "",
        }
        latest_effective_sr_scale = 1
        latest_rtx_vsr_applied = False
        latest_rtx_effect_mean_abs_luma = 0.0
        rtx_effect_sample_counter = 0
        processed_frame_counter = 0
        started_perf_ts = time.perf_counter()
        frame_duration = int(getattr(output_session, "frame_duration", 0))
        time_scale = int(getattr(output_session, "time_scale", 0))
        if frame_duration > 0 and time_scale > 0:
            output_frame_period_s = max(0.0, float(frame_duration) / float(time_scale))
            output_nominal_fps = float(time_scale) / float(frame_duration)
        else:
            output_frame_period_s = 0.0
            output_nominal_fps = 0.0
        # Canonical transition cadence is one ROI frame-step per emitted output
        # frame for both progressive and interlaced output. Interlaced field0/
        # field1 in-between timing is synthesized via phase fraction.
        output_transition_units_per_frame = 1.0
        stage_preprocess_applied_frames = 0
        stage_basic_applied_frames = 0
        stage_ai_applied_frames = 0
        stage_rtx_applied_frames = 0
        stage_passthrough_frames = 0
        last_stage_preprocess_applied = False
        last_stage_basic_applied = False
        last_stage_ai_applied = False
        last_stage_rtx_applied = False
        last_stage_stack = []
        basic_scaling_last_frame_ms = 0.0
        basic_scaling_avg_frame_ms = None
        basic_scaling_max_frame_ms = 0.0
        basic_scaling_timing_samples = 0
        timing_frames_emitted = 0
        timing_deadline_miss_events = 0
        timing_deadline_miss_streak = 0
        timing_deadline_miss_max_streak = 0
        timing_e2e_ms_last = 0.0
        timing_e2e_ms_ema = 0.0
        timing_e2e_ms_peak = 0.0
        timing_process_ms_ema = 0.0
        timing_process_ms_peak = 0.0
        timing_capture_queue_ms_ema = 0.0
        timing_capture_queue_ms_peak = 0.0
        timing_output_queue_ms_ema = 0.0
        timing_output_queue_ms_peak = 0.0
        timing_output_wait_ms_ema = 0.0
        timing_output_wait_ms_peak = 0.0
        timing_emit_call_ms_ema = 0.0
        timing_emit_call_ms_peak = 0.0
        timing_deadline_late_ms_ema = 0.0
        timing_deadline_late_ms_peak = 0.0
        timing_last_path = ""
        preprocess_noop_warning_emitted = False
        native_subpixel_warning_emitted = False
        upscale_extra_threads = []

        def _record_pipeline_timing(
            path_name: str,
            captured_ts: float,
            process_start_ts: float,
            process_end_ts: float,
            output_dequeue_ts: float,
            emit_start_ts: float,
            emit_end_ts: float,
            output_wait_s: float,
            emitted: bool,
        ) -> None:
            nonlocal timing_frames_emitted, timing_deadline_miss_events, timing_deadline_miss_streak, timing_deadline_miss_max_streak
            nonlocal timing_e2e_ms_last, timing_e2e_ms_ema, timing_e2e_ms_peak
            nonlocal timing_process_ms_ema, timing_process_ms_peak
            nonlocal timing_capture_queue_ms_ema, timing_capture_queue_ms_peak
            nonlocal timing_output_queue_ms_ema, timing_output_queue_ms_peak
            nonlocal timing_output_wait_ms_ema, timing_output_wait_ms_peak
            nonlocal timing_emit_call_ms_ema, timing_emit_call_ms_peak
            nonlocal timing_deadline_late_ms_ema, timing_deadline_late_ms_peak
            nonlocal timing_last_path

            if not emitted:
                return

            with state_lock:
                alpha = 0.16
                e2e_ms = max(0.0, (emit_end_ts - captured_ts) * 1000.0)
                process_ms = max(0.0, (process_end_ts - process_start_ts) * 1000.0)
                capture_queue_ms = max(0.0, (process_start_ts - captured_ts) * 1000.0)
                output_queue_ms = max(0.0, (output_dequeue_ts - process_end_ts) * 1000.0)
                output_wait_ms = max(0.0, float(output_wait_s) * 1000.0)
                emit_call_ms = max(0.0, (emit_end_ts - emit_start_ts) * 1000.0)

                budget_ms = (1000.0 / output_nominal_fps) if output_nominal_fps > 0.0 else 0.0
                deadline_late_ms = max(0.0, e2e_ms - budget_ms) if budget_ms > 0.0 else 0.0
                missed_deadline = deadline_late_ms > 0.0

                timing_frames_emitted += 1
                timing_last_path = str(path_name)
                timing_e2e_ms_last = e2e_ms

                if timing_frames_emitted == 1:
                    timing_e2e_ms_ema = e2e_ms
                    timing_process_ms_ema = process_ms
                    timing_capture_queue_ms_ema = capture_queue_ms
                    timing_output_queue_ms_ema = output_queue_ms
                    timing_output_wait_ms_ema = output_wait_ms
                    timing_emit_call_ms_ema = emit_call_ms
                    timing_deadline_late_ms_ema = deadline_late_ms
                else:
                    timing_e2e_ms_ema = ((1.0 - alpha) * timing_e2e_ms_ema) + (alpha * e2e_ms)
                    timing_process_ms_ema = ((1.0 - alpha) * timing_process_ms_ema) + (alpha * process_ms)
                    timing_capture_queue_ms_ema = ((1.0 - alpha) * timing_capture_queue_ms_ema) + (alpha * capture_queue_ms)
                    timing_output_queue_ms_ema = ((1.0 - alpha) * timing_output_queue_ms_ema) + (alpha * output_queue_ms)
                    timing_output_wait_ms_ema = ((1.0 - alpha) * timing_output_wait_ms_ema) + (alpha * output_wait_ms)
                    timing_emit_call_ms_ema = ((1.0 - alpha) * timing_emit_call_ms_ema) + (alpha * emit_call_ms)
                    timing_deadline_late_ms_ema = ((1.0 - alpha) * timing_deadline_late_ms_ema) + (alpha * deadline_late_ms)

                timing_e2e_ms_peak = max(timing_e2e_ms_peak, e2e_ms)
                timing_process_ms_peak = max(timing_process_ms_peak, process_ms)
                timing_capture_queue_ms_peak = max(timing_capture_queue_ms_peak, capture_queue_ms)
                timing_output_queue_ms_peak = max(timing_output_queue_ms_peak, output_queue_ms)
                timing_output_wait_ms_peak = max(timing_output_wait_ms_peak, output_wait_ms)
                timing_emit_call_ms_peak = max(timing_emit_call_ms_peak, emit_call_ms)
                timing_deadline_late_ms_peak = max(timing_deadline_late_ms_peak, deadline_late_ms)

                if missed_deadline:
                    timing_deadline_miss_events += 1
                    timing_deadline_miss_streak += 1
                    timing_deadline_miss_max_streak = max(timing_deadline_miss_max_streak, timing_deadline_miss_streak)
                else:
                    timing_deadline_miss_streak = 0

        def _capture_worker() -> None:
            nonlocal frame_id_counter, capture_drop_count
            nonlocal latest_input_frame, latest_output_frame, latest_effective_sr_scale, processed_frame_counter
            nonlocal latest_rtx_vsr_applied, latest_rtx_effect_mean_abs_luma
            nonlocal latest_timecode_info
            nonlocal stage_preprocess_applied_frames, stage_basic_applied_frames, stage_ai_applied_frames, stage_rtx_applied_frames, stage_passthrough_frames
            nonlocal last_stage_preprocess_applied, last_stage_basic_applied, last_stage_ai_applied, last_stage_rtx_applied
            nonlocal last_stage_stack
            assert q_capture_to_preprocess is not None
            while not pipeline_stop_event.is_set():
                try:
                    frame = capture_session.acquire(timeout_ms=2) if capture_session is not None else None
                except Exception:
                    frame = None
                if frame is None:
                    continue
                frame_timecode_info = _extract_frame_timecode_info(frame)
                try:
                    input_bytes = _tight_uyvy_bytes(frame)
                except Exception as exc:
                    _safe_put({"type": "warning", "warning": f"Capture frame conversion failed: {exc}"})
                    continue
                frame_captured_ts = time.perf_counter()

                # Advance transition phase before processing this frame so each
                # emitted frame maps to the next field/frame step, avoiding
                # duplicated field phase from post-emit advancement.
                _advance_roi_microstep_transition_for_output_frame()
                interlaced_phase_snapshot = None
                interlaced_phase_active = _active_interlaced_field_phase_state(consume_manual_snapshot=True)
                if isinstance(interlaced_phase_active, dict):
                    interlaced_phase_snapshot = dict(interlaced_phase_active)

                # Keep input preview live even when processing is backlogged.
                with state_lock:
                    latest_input_frame = input_bytes
                    latest_timecode_info = dict(frame_timecode_info)

                if _is_live_passthrough_mode():
                    # In zero-processing mode, avoid staged queueing and preserve output cadence.
                    process_start_ts = time.perf_counter()
                    output_bytes = input_bytes
                    shift_x, shift_y = _step_smoothed_roi_shift()
                    native_shift_applied = False
                    interlaced_phase = interlaced_phase_snapshot
                    if interlaced_phase is not None:
                        output_bytes, native_shift_applied = _render_dual_phase_no_deinterlace(input_bytes, interlaced_phase)
                    else:
                        if _set_native_subpixel_shift(shift_x, shift_y):
                            native_shift_applied = _has_effective_subpixel_shift(shift_x, shift_y)
                        if native_shift_applied and hasattr(processor, "process_frame_no_deinterlace"):
                            output_bytes = _run_native_uyvy_process(
                                output_bytes,
                                "process_frame_no_deinterlace",
                                "process_frame_no_deinterlace_into",
                                reusable_process_frame_no_deinterlace_out,
                            )
                        elif (
                            (not output_mode_is_interlaced)
                            and _should_apply_cpu_subpixel_fallback()
                            and _has_effective_subpixel_shift(shift_x, shift_y)
                        ):
                            output_bytes = _apply_subpixel_shift_uyvy(output_bytes, shift_x, shift_y)
                        output_bytes = _apply_interlaced_field_phase_if_needed(
                            output_bytes,
                            shift_x,
                            shift_y,
                            native_shift_applied=native_shift_applied,
                        )
                    process_end_ts = time.perf_counter()
                    emit_start_ts = time.perf_counter()
                    try:
                        if output_session is not None:
                            emitted = _write_frame_to_output(output_session, output_bytes)
                        else:
                            emitted = False
                    except Exception as exc:
                        _safe_put({"type": "warning", "warning": f"Output stage failed: {exc}"})
                        continue
                    emit_end_ts = time.perf_counter()
                    _record_pipeline_timing(
                        path_name="passthrough_fast",
                        captured_ts=frame_captured_ts,
                        process_start_ts=process_start_ts,
                        process_end_ts=process_end_ts,
                        output_dequeue_ts=process_end_ts,
                        emit_start_ts=emit_start_ts,
                        emit_end_ts=emit_end_ts,
                        output_wait_s=0.0,
                        emitted=bool(emitted),
                    )

                    with state_lock:
                        latest_input_frame = input_bytes
                        latest_output_frame = output_bytes
                        latest_effective_sr_scale = 1
                        latest_rtx_vsr_applied = False
                        latest_rtx_effect_mean_abs_luma = 0.0
                        if emitted:
                            processed_frame_counter += 1
                    if emitted:
                        with state_lock:
                            roi_transition_snapshot = dict(roi_microstep_transition) if isinstance(roi_microstep_transition, dict) else None
                            roi_x_snapshot = int(current_roi_x)
                            roi_y_snapshot = int(current_roi_y)
                            roi_w_snapshot = int(current_roi_w)
                            roi_h_snapshot = int(current_roi_h)
                        _publish_roi_telemetry(
                            roi_x_snapshot,
                            roi_y_snapshot,
                            roi_w_snapshot,
                            roi_h_snapshot,
                            roi_transition_snapshot,
                        )
                    continue

                if _is_live_basic_scaling_fast_mode():
                    if parallel_basic_worker_count > 1:
                        # Parallel basic scaling is handled in dedicated upscale workers.
                        shift_x, shift_y = _step_smoothed_roi_shift()
                        next_frame_id = frame_id_counter + 1
                        item = _StageFrame(
                            frame_id=next_frame_id,
                            captured_ts=frame_captured_ts,
                            input_bytes=input_bytes,
                            shift_x=float(shift_x),
                            shift_y=float(shift_y),
                            roi_x=int(current_roi_x),
                            roi_y=int(current_roi_y),
                            roi_w=int(current_roi_w),
                            roi_h=int(current_roi_h),
                            interlaced_field_phase=interlaced_phase_snapshot,
                        )
                        if _put_stage_frame_fifo_drop_newest(q_capture_to_preprocess, item):
                            capture_drop_count += 1
                        else:
                            frame_id_counter = next_frame_id
                        continue

                    # Keep basic-scaling-only path off the staged queue graph to
                    # reduce Python scheduling overhead at 1080p60.
                    process_start_ts = time.perf_counter()
                    shift_x, shift_y = _step_smoothed_roi_shift()
                    interlaced_phase = interlaced_phase_snapshot
                    try:
                        if _reinterlace_enabled_for_output():
                            reinterlace_phase = interlaced_phase
                            if reinterlace_phase is None:
                                reinterlace_phase = _build_static_interlaced_phase_for_reinterlace(shift_x, shift_y)
                            interlaced_phase = reinterlace_phase
                            direct_field_result = _render_dual_phase_basic_scaling_reinterlace(input_bytes, reinterlace_phase)
                            if direct_field_result is not None:
                                output_bytes, basic_applied, native_shift_applied, basic_stage_ms = direct_field_result
                            else:
                                output_bytes, preprocess_applied, basic_applied, ai_applied, rtx_applied, native_shift_applied = _render_dual_phase_full_pipeline_reinterlace(
                                    input_bytes,
                                    reinterlace_phase,
                                )
                                _ = preprocess_applied, ai_applied, rtx_applied
                                basic_stage_ms = 0.0
                        elif interlaced_phase is not None:
                            output_bytes, basic_applied, native_shift_applied, basic_stage_ms = _render_dual_phase_basic_scaling(
                                input_bytes,
                                interlaced_phase,
                            )
                        else:
                            output_bytes, basic_applied, native_shift_applied, basic_stage_ms = _apply_basic_scaling_stage(
                                input_bytes,
                                False,
                                shift_x,
                                shift_y,
                            )
                    except Exception as exc:
                        _safe_put({"type": "warning", "warning": f"Basic scaling fast path failed: {exc}"})
                        continue

                    if basic_applied:
                        _record_basic_scaling_timing(basic_stage_ms)

                    if (
                        (not native_shift_applied)
                        and (not output_mode_is_interlaced)
                        and _should_apply_cpu_subpixel_fallback()
                        and _has_effective_subpixel_shift(shift_x, shift_y)
                    ):
                        output_bytes = _apply_subpixel_shift_uyvy(output_bytes, shift_x, shift_y)
                    if interlaced_phase is None:
                        output_bytes = _apply_interlaced_field_phase_if_needed(
                            output_bytes,
                            shift_x,
                            shift_y,
                            native_shift_applied=native_shift_applied,
                        )

                    process_end_ts = time.perf_counter()
                    emit_start_ts = time.perf_counter()

                    try:
                        if output_session is not None:
                            emitted = _write_frame_to_output(output_session, output_bytes)
                        else:
                            emitted = False
                    except Exception as exc:
                        _safe_put({"type": "warning", "warning": f"Output stage failed: {exc}"})
                        continue
                    emit_end_ts = time.perf_counter()
                    _record_pipeline_timing(
                        path_name="basic_scaling_fast",
                        captured_ts=frame_captured_ts,
                        process_start_ts=process_start_ts,
                        process_end_ts=process_end_ts,
                        output_dequeue_ts=process_end_ts,
                        emit_start_ts=emit_start_ts,
                        emit_end_ts=emit_end_ts,
                        output_wait_s=0.0,
                        emitted=bool(emitted),
                    )

                    if basic_applied:
                        stage_basic_applied_frames += 1
                    else:
                        stage_passthrough_frames += 1

                    last_stage_preprocess_applied = False
                    last_stage_basic_applied = bool(basic_applied)
                    last_stage_ai_applied = False
                    last_stage_rtx_applied = False
                    last_stage_stack = ["basic_scaling"] if basic_applied else []

                    with state_lock:
                        latest_input_frame = input_bytes
                        latest_output_frame = output_bytes
                        latest_effective_sr_scale = int(processor.get_effective_sr_scale())
                        latest_rtx_vsr_applied = False
                        latest_rtx_effect_mean_abs_luma = 0.0
                        if emitted:
                            processed_frame_counter += 1
                    if emitted:
                        with state_lock:
                            roi_transition_snapshot = dict(roi_microstep_transition) if isinstance(roi_microstep_transition, dict) else None
                            roi_x_snapshot = int(current_roi_x)
                            roi_y_snapshot = int(current_roi_y)
                            roi_w_snapshot = int(current_roi_w)
                            roi_h_snapshot = int(current_roi_h)
                        _publish_roi_telemetry(
                            roi_x_snapshot,
                            roi_y_snapshot,
                            roi_w_snapshot,
                            roi_h_snapshot,
                            roi_transition_snapshot,
                        )
                    continue

                frame_id_counter += 1
                shift_x, shift_y = _step_smoothed_roi_shift()
                item = _StageFrame(
                    frame_id=frame_id_counter,
                    captured_ts=frame_captured_ts,
                    input_bytes=input_bytes,
                    shift_x=float(shift_x),
                    shift_y=float(shift_y),
                    roi_x=int(current_roi_x),
                    roi_y=int(current_roi_y),
                    roi_w=int(current_roi_w),
                    roi_h=int(current_roi_h),
                    interlaced_field_phase=interlaced_phase_snapshot,
                )
                if _put_latest_stage_frame(q_capture_to_preprocess, item):
                    capture_drop_count += 1

        def _upscale_worker(worker_index: int = 0) -> None:
            nonlocal upscale_drop_count, ai_sr_dropped_frames, preprocess_drop_count
            nonlocal stage_preprocess_applied_frames, stage_basic_applied_frames, stage_ai_applied_frames, stage_rtx_applied_frames, stage_passthrough_frames
            nonlocal last_stage_preprocess_applied, last_stage_basic_applied, last_stage_ai_applied, last_stage_rtx_applied
            nonlocal last_stage_stack
            assert q_capture_to_preprocess is not None
            assert q_upscale_to_output is not None

            thread_parallel_proc = None
            if worker_index < len(parallel_basic_processors):
                thread_parallel_proc = parallel_basic_processors[worker_index]

            last_synced_mode_auto: bool | None = None
            last_synced_manual_scale: int | None = None
            last_synced_max_auto_scale: int | None = None
            last_synced_method = ""
            last_synced_deinterlace_enabled: bool | None = None
            last_synced_deinterlace_method = ""
            last_synced_denoise_method = ""
            last_synced_denoise_strength: float | None = None
            last_synced_color_space = ""
            last_synced_color_range = ""

            while not pipeline_stop_event.is_set():
                try:
                    item = q_capture_to_preprocess.get(timeout=0.01)
                except queue.Empty:
                    continue

                parallel_basic_active = (
                    thread_parallel_proc is not None
                    and parallel_basic_worker_count > 1
                    and _is_live_basic_scaling_fast_mode()
                    and not (output_mode_is_interlaced and _reinterlace_enabled_for_output())
                    and not (
                        output_mode_is_interlaced
                        and isinstance(item.interlaced_field_phase, dict)
                    )
                )

                if parallel_basic_active:
                    try:
                        if hasattr(thread_parallel_proc, "set_roi"):
                            thread_parallel_proc.set_roi(int(item.roi_x), int(item.roi_y), int(item.roi_w), int(item.roi_h))

                        if hasattr(thread_parallel_proc, "set_sr_flavor") and current_basic_scaling_method != last_synced_method:
                            thread_parallel_proc.set_sr_flavor(current_basic_scaling_method)
                            last_synced_method = str(current_basic_scaling_method)

                        if hasattr(thread_parallel_proc, "set_max_auto_sr_scale") and current_max_auto_basic_scaling != last_synced_max_auto_scale:
                            thread_parallel_proc.set_max_auto_sr_scale(int(current_max_auto_basic_scaling))
                            last_synced_max_auto_scale = int(current_max_auto_basic_scaling)

                        if current_basic_scaling_auto_mode != last_synced_mode_auto:
                            if current_basic_scaling_auto_mode:
                                thread_parallel_proc.set_sr_mode_auto()
                            else:
                                thread_parallel_proc.set_sr_scale_manual(int(current_basic_scaling_manual_scale))
                            last_synced_mode_auto = bool(current_basic_scaling_auto_mode)

                        if (
                            (not current_basic_scaling_auto_mode)
                            and current_basic_scaling_manual_scale != last_synced_manual_scale
                        ):
                            thread_parallel_proc.set_sr_scale_manual(int(current_basic_scaling_manual_scale))
                            last_synced_manual_scale = int(current_basic_scaling_manual_scale)

                        if current_deinterlace_enabled != last_synced_deinterlace_enabled:
                            thread_parallel_proc.set_deinterlace_enabled(bool(current_deinterlace_enabled))
                            last_synced_deinterlace_enabled = bool(current_deinterlace_enabled)

                        if hasattr(thread_parallel_proc, "set_deinterlace_method") and current_deinterlace_method != last_synced_deinterlace_method:
                            thread_parallel_proc.set_deinterlace_method(current_deinterlace_method)
                            last_synced_deinterlace_method = str(current_deinterlace_method)

                        if hasattr(thread_parallel_proc, "set_denoise_method") and current_denoise_method != last_synced_denoise_method:
                            thread_parallel_proc.set_denoise_method(current_denoise_method)
                            last_synced_denoise_method = str(current_denoise_method)

                        if (
                            hasattr(thread_parallel_proc, "set_denoise_strength")
                            and (last_synced_denoise_strength is None or abs(current_denoise_strength - last_synced_denoise_strength) > 1e-6)
                        ):
                            thread_parallel_proc.set_denoise_strength(float(current_denoise_strength))
                            last_synced_denoise_strength = float(current_denoise_strength)

                        if hasattr(thread_parallel_proc, "set_color_space") and current_color_space != last_synced_color_space:
                            thread_parallel_proc.set_color_space(current_color_space)
                            last_synced_color_space = str(current_color_space)

                        if hasattr(thread_parallel_proc, "set_color_range") and current_color_range != last_synced_color_range:
                            thread_parallel_proc.set_color_range(current_color_range)
                            last_synced_color_range = str(current_color_range)

                        native_shift_applied = False
                        if hasattr(thread_parallel_proc, "set_subpixel_shift"):
                            thread_parallel_proc.set_subpixel_shift(float(item.shift_x), float(item.shift_y))
                            native_shift_applied = _has_effective_subpixel_shift(float(item.shift_x), float(item.shift_y))

                        item.process_start_ts = time.perf_counter()
                        output_bytes = thread_parallel_proc.process_frame(item.input_bytes)
                        item.process_end_ts = time.perf_counter()
                        basic_stage_ms = max(0.0, (item.process_end_ts - item.process_start_ts) * 1000.0)
                        _record_basic_scaling_timing(basic_stage_ms)

                        if (not _is_valid_uyvy_frame(output_bytes)) or _looks_zeroed_uyvy_frame(output_bytes):
                            output_bytes = item.input_bytes
                            basic_applied = False
                            native_shift_applied = False
                        else:
                            basic_applied = True

                        last_stage_preprocess_applied = False
                        last_stage_basic_applied = bool(basic_applied)
                        last_stage_ai_applied = False
                        last_stage_rtx_applied = False
                        last_stage_stack = ["basic_scaling"] if basic_applied else []

                        if basic_applied:
                            stage_basic_applied_frames += 1
                        else:
                            stage_passthrough_frames += 1

                        item.output_bytes = output_bytes
                        if hasattr(thread_parallel_proc, "get_effective_sr_scale"):
                            item.effective_sr_scale = int(thread_parallel_proc.get_effective_sr_scale())
                        else:
                            item.effective_sr_scale = 1
                        item.native_shift_applied = bool(native_shift_applied)
                        item.ai_applied = False
                        item.rtx_applied = False
                        item.output_queue_put_ts = time.perf_counter()
                        if not _put_stage_frame_blocking(q_upscale_to_output, item):
                            upscale_drop_count += 1
                        continue
                    except Exception as exc:
                        _safe_put({"type": "warning", "warning": f"Parallel basic scaling worker failed: {exc}"})
                        continue

                try:
                    item.preprocess_bytes = item.input_bytes
                except Exception as exc:
                    preprocess_drop_count += 1
                    _safe_put({"type": "warning", "warning": f"Preprocess stage failed: {exc}"})
                    continue

                preprocessed = item.preprocess_bytes if item.preprocess_bytes is not None else item.input_bytes
                try:
                    item.process_start_ts = time.perf_counter()
                    interlaced_phase = item.interlaced_field_phase if isinstance(item.interlaced_field_phase, dict) else None
                    if output_mode_is_interlaced and _reinterlace_enabled_for_output():
                        if interlaced_phase is None:
                            interlaced_phase = _build_static_interlaced_phase_for_reinterlace(float(item.shift_x), float(item.shift_y))
                        output_bytes, preprocess_applied, basic_applied, ai_applied, rtx_applied, native_shift_applied = _render_dual_phase_full_pipeline_reinterlace(
                            preprocessed,
                            interlaced_phase,
                        )
                        item.interlaced_phase_rendered = True
                    elif output_mode_is_interlaced and _interlaced_phase_controls_active() and interlaced_phase is not None:
                        output_bytes, preprocess_applied, basic_applied, ai_applied, rtx_applied, native_shift_applied = _render_dual_phase_full_pipeline(
                            preprocessed,
                            interlaced_phase,
                        )
                        item.interlaced_phase_rendered = True
                    else:
                        output_bytes, preprocess_applied, basic_applied, ai_applied, rtx_applied, native_shift_applied = _process_pipeline_frame(
                            preprocessed,
                            float(item.shift_x),
                            float(item.shift_y),
                        )
                        item.interlaced_phase_rendered = False
                    item.process_end_ts = time.perf_counter()
                    stage_applied = preprocess_applied or basic_applied or ai_applied or rtx_applied
                except Exception as exc:
                    _safe_put({"type": "warning", "warning": f"Upscale stage failed: {exc}"})
                    continue

                if ai_sr_engine is not None and not stage_applied and _ai_inference_busy():
                    ai_sr_dropped_frames += 1

                if preprocess_applied:
                    stage_preprocess_applied_frames += 1
                if basic_applied:
                    stage_basic_applied_frames += 1
                if ai_applied:
                    stage_ai_applied_frames += 1
                if rtx_applied:
                    stage_rtx_applied_frames += 1
                if not stage_applied:
                    stage_passthrough_frames += 1

                last_stage_preprocess_applied = bool(preprocess_applied)
                last_stage_basic_applied = bool(basic_applied)
                last_stage_ai_applied = bool(ai_applied)
                last_stage_rtx_applied = bool(rtx_applied)
                last_stage_stack = _build_stage_stack()

                item.output_bytes = output_bytes
                item.effective_sr_scale = int(processor.get_effective_sr_scale()) if hasattr(processor, "get_effective_sr_scale") else 1
                item.ai_applied = bool(ai_applied)
                item.rtx_applied = bool(rtx_applied)
                item.native_shift_applied = bool(native_shift_applied)
                item.output_queue_put_ts = time.perf_counter()
                if _put_latest_stage_frame(q_upscale_to_output, item):
                    upscale_drop_count += 1

        def _output_worker() -> None:
            nonlocal latest_input_frame, latest_output_frame, latest_effective_sr_scale, processed_frame_counter
            nonlocal latest_rtx_vsr_applied, latest_rtx_effect_mean_abs_luma
            nonlocal rtx_effect_sample_counter
            assert q_upscale_to_output is not None

            # Preserve capture order when parallel upscale workers complete out-of-order.
            reorder_pending: dict[int, _StageFrame] = {}
            next_frame_id = 1
            max_reorder_buffer = max(2, parallel_basic_worker_count * 2)
            max_reorder_wait_s = 0.012

            def _emit_output_item(item: _StageFrame) -> None:
                nonlocal latest_input_frame, latest_output_frame, latest_effective_sr_scale, processed_frame_counter
                nonlocal latest_rtx_vsr_applied, latest_rtx_effect_mean_abs_luma
                nonlocal rtx_effect_sample_counter

                output_bytes = item.output_bytes if item.output_bytes is not None else item.input_bytes
                shift_x = float(item.shift_x)
                shift_y = float(item.shift_y)
                if (
                    (not item.native_shift_applied)
                    and (not output_mode_is_interlaced)
                    and _should_apply_cpu_subpixel_fallback()
                    and _has_effective_subpixel_shift(shift_x, shift_y)
                ):
                    output_bytes = _apply_subpixel_shift_uyvy(output_bytes, shift_x, shift_y)
                if not bool(item.interlaced_phase_rendered):
                    output_bytes = _apply_interlaced_field_phase_if_needed(
                        output_bytes,
                        shift_x,
                        shift_y,
                        native_shift_applied=bool(item.native_shift_applied),
                    )
                if (not bool(item.interlaced_phase_rendered)) and _reinterlace_enabled_for_output():
                    output_bytes = _apply_reinterlace_from_previous_frame_if_needed(output_bytes)
                if output_mode_is_interlaced and (not _interlaced_phase_controls_active()) and (not _reinterlace_enabled_for_output()):
                    is_lower_field_first = int(output_field_dominance_code) == 1 if output_field_dominance_code is not None else False
                    output_bytes = _collapse_interlaced_to_single_field_uyvy(output_bytes, is_lower_field_first)
                sampled_delta = 0.0
                if item.rtx_applied:
                    rtx_effect_sample_counter += 1
                    if (rtx_effect_sample_counter % 12) == 0:
                        try:
                            in_arr = np.frombuffer(item.input_bytes, dtype=np.uint8)
                            out_arr = np.frombuffer(output_bytes, dtype=np.uint8)
                            if in_arr.size == out_arr.size and in_arr.size > 0:
                                step = 32
                                sampled_delta = float(
                                    np.mean(
                                        np.abs(
                                            in_arr[::step].astype(np.int16)
                                            - out_arr[::step].astype(np.int16)
                                        )
                                    )
                                )
                        except Exception:
                            sampled_delta = 0.0

                emit_start_ts = time.perf_counter()
                output_wait_s = max(0.0, emit_start_ts - item.output_dequeue_ts) if item.output_dequeue_ts > 0.0 else 0.0
                try:
                    if output_session is not None:
                        emitted = _write_frame_to_output(output_session, output_bytes)
                    else:
                        emitted = False
                except Exception as exc:
                    _safe_put({"type": "warning", "warning": f"Output stage failed: {exc}"})
                    return
                emit_end_ts = time.perf_counter()

                process_start_ts = item.process_start_ts if item.process_start_ts > 0.0 else item.captured_ts
                process_end_ts = item.process_end_ts if item.process_end_ts > 0.0 else process_start_ts
                output_dequeue_ts = item.output_dequeue_ts if item.output_dequeue_ts > 0.0 else emit_start_ts
                _record_pipeline_timing(
                    path_name="staged_pipeline",
                    captured_ts=item.captured_ts,
                    process_start_ts=process_start_ts,
                    process_end_ts=process_end_ts,
                    output_dequeue_ts=output_dequeue_ts,
                    emit_start_ts=emit_start_ts,
                    emit_end_ts=emit_end_ts,
                    output_wait_s=output_wait_s,
                    emitted=bool(emitted),
                )

                with state_lock:
                    latest_input_frame = item.input_bytes
                    latest_output_frame = output_bytes
                    latest_effective_sr_scale = int(item.effective_sr_scale)
                    latest_rtx_vsr_applied = bool(item.rtx_applied)
                    latest_rtx_effect_mean_abs_luma = sampled_delta
                    if emitted:
                        processed_frame_counter += 1
                if emitted:
                    with state_lock:
                        roi_transition_snapshot = dict(roi_microstep_transition) if isinstance(roi_microstep_transition, dict) else None
                        roi_x_snapshot = int(current_roi_x)
                        roi_y_snapshot = int(current_roi_y)
                        roi_w_snapshot = int(current_roi_w)
                        roi_h_snapshot = int(current_roi_h)
                    _publish_roi_telemetry(
                        roi_x_snapshot,
                        roi_y_snapshot,
                        roi_w_snapshot,
                        roi_h_snapshot,
                        roi_transition_snapshot,
                    )

            while not pipeline_stop_event.is_set():
                try:
                    item = q_upscale_to_output.get(timeout=0.01)
                except queue.Empty:
                    item = None

                now_ts = time.perf_counter()
                if item is not None:
                    item.output_dequeue_ts = now_ts
                    reorder_pending[int(item.frame_id)] = item

                while reorder_pending:
                    if next_frame_id in reorder_pending:
                        ready_item = reorder_pending.pop(next_frame_id)
                        next_frame_id += 1
                        _emit_output_item(ready_item)
                        continue

                    min_pending_id = min(reorder_pending.keys())
                    oldest_pending = min(reorder_pending.values(), key=lambda f: f.output_dequeue_ts)
                    oldest_wait_s = max(0.0, now_ts - oldest_pending.output_dequeue_ts)
                    if len(reorder_pending) >= max_reorder_buffer or oldest_wait_s >= max_reorder_wait_s:
                        next_frame_id = min_pending_id
                        continue
                    break

        capture_thread = threading.Thread(target=_capture_worker, name="vp-capture", daemon=True)
        preprocess_thread = None
        upscale_thread = threading.Thread(target=lambda: _upscale_worker(0), name="vp-upscale-0", daemon=True)
        for worker_index in range(1, parallel_basic_worker_count):
            extra_thread = threading.Thread(
                target=lambda i=worker_index: _upscale_worker(i),
                name=f"vp-upscale-{worker_index}",
                daemon=True,
            )
            upscale_extra_threads.append(extra_thread)
        output_thread = threading.Thread(target=_output_worker, name="vp-output", daemon=True)

        capture_thread.start()
        upscale_thread.start()
        for thread in upscale_extra_threads:
            thread.start()
        output_thread.start()
        pipeline_running = True

    def _refresh_ai_sr_engine() -> str | None:
        nonlocal ai_sr_engine, ai_sr_info, ai_sr_frame_counter, ai_sr_latest_output_frame, ai_sr_latest_output_ts, ai_sr_completed_frames, ai_sr_warmup_pending, ai_sr_executor, ai_sr_futures, ai_sr_dropped_frames, ai_sr_applied_frames, ai_sr_reused_frames, ai_sr_passthrough_frames, ai_sr_runtime_note, ai_sr_max_inflight
        nonlocal ai_sr_submit_spacing_ms, ai_sr_last_submit_ts
        _cleanup_ai_async()

        if not ai_sr_enabled:
            ai_sr_engine = None
            ai_sr_info = None
            ai_sr_frame_counter = 0
            ai_sr_latest_output_frame = None
            ai_sr_latest_output_ts = 0.0
            ai_sr_completed_frames = 0
            ai_sr_warmup_pending = False
            ai_sr_dropped_frames = 0
            ai_sr_applied_frames = 0
            ai_sr_reused_frames = 0
            ai_sr_passthrough_frames = 0
            return None

        try:
            _apply_ai_sr_performance_profile()

            ai_sr_engine = AiSrOnnxEngine(
                ai_sr_model_path,
                provider=ai_sr_provider,
                trt_precision=ai_sr_trt_precision,
                trt_engine_cache_path=str(trt_cache_root),
                require_gpu=ai_sr_require_gpu,
                input_align=ai_sr_input_align,
                roi_overscan_percent=ai_sr_roi_overscan_percent,
                inference_divisor=ai_sr_inference_divisor,
                detail_preserve_percent=ai_sr_detail_preserve_percent,
                post_denoise_method=ai_sr_post_denoise_method,
                post_denoise_strength=ai_sr_post_denoise_strength,
                post_artifact_reduction_method=ai_sr_post_artifact_reduction_method,
                post_artifact_reduction_strength=ai_sr_post_artifact_reduction_strength,
                post_exaggeration_enabled=ai_sr_post_exaggeration_enabled,
                post_exaggeration_gain=ai_sr_post_exaggeration_gain,
                color_space=current_color_space,
                color_range=current_color_range,
                native_module=module,
                native_processor=processor,
            )
            ai_sr_info = ai_sr_engine.info()
            ai_sr_info["strict_mode"] = bool(ai_sr_strict)
            ai_sr_info["async_mode"] = not bool(ai_sr_strict)
            ai_sr_info["frame_interval"] = int(ai_sr_frame_interval)
            ai_sr_info["inference_fps"] = int(ai_sr_frame_interval)
            ai_sr_info["discard_while_busy"] = False
            ai_sr_info["requested_provider"] = str(ai_sr_provider)
            ai_sr_info["trt_precision"] = str(ai_sr_trt_precision)
            ai_sr_info["gpu_required"] = bool(ai_sr_require_gpu)
            ai_sr_info["runtime_profile_note"] = ai_sr_runtime_note
            ai_sr_info["max_hold_ms"] = float(ai_sr_max_hold_ms)
            ai_sr_info["hold_last_frame"] = bool(ai_sr_hold_last_frame)
            ai_sr_info["max_inflight"] = int(ai_sr_max_inflight)
            ai_sr_info["submit_spacing_ms"] = float(ai_sr_submit_spacing_ms)
            ai_sr_info["post_denoise_method"] = str(ai_sr_post_denoise_method)
            ai_sr_info["post_denoise_strength"] = float(ai_sr_post_denoise_strength)
            ai_sr_info["post_artifact_reduction_method"] = str(ai_sr_post_artifact_reduction_method)
            ai_sr_info["post_artifact_reduction_strength"] = float(ai_sr_post_artifact_reduction_strength)
            ai_sr_info["post_exaggeration_enabled"] = bool(ai_sr_post_exaggeration_enabled)
            ai_sr_info["post_exaggeration_gain"] = float(ai_sr_post_exaggeration_gain)
            ai_sr_info["post_exaggeration_passes"] = 3 if ai_sr_post_exaggeration_enabled else 1
            ai_sr_info["postprocess_gpu_chain"] = "resize/sharpen -> post_denoise(xN) -> post_artifact_reduction(xN) -> rgb_to_uyvy"
            ai_sr_info["basic_cuda_post_scale_enabled"] = False
            ai_sr_info["basic_cuda_post_scale_active"] = False
            ai_sr_info["pipeline_order"] = "crop/preprocess -> onnx(cuda) -> cuda_postprocess -> uyvy"
            if float(ai_sr_detail_preserve_percent) > 0.0:
                ai_sr_info["detail_preserve_note"] = (
                    "detail_preserve is disabled in zero-copy mode to keep output fully GPU-resident"
                )
            ai_sr_frame_counter = 0
            ai_sr_latest_output_frame = None
            ai_sr_latest_output_ts = 0.0
            ai_sr_completed_frames = 0
            ai_sr_warmup_pending = True
            ai_sr_last_submit_ts = 0.0
            ai_sr_dropped_frames = 0
            ai_sr_applied_frames = 0
            ai_sr_reused_frames = 0
            ai_sr_passthrough_frames = 0
            ai_sr_executor = ThreadPoolExecutor(max_workers=ai_sr_max_inflight, thread_name_prefix="ai-sr")
            ai_sr_futures = []
            if ai_sr_strict:
                _cleanup_ai_async()
            if ai_sr_runtime_note is not None:
                _safe_put({"type": "warning", "warning": f"AI SR throughput profile applied: {ai_sr_runtime_note}"})
            return None
        except Exception as ai_exc:
            ai_sr_engine = None
            ai_sr_info = None
            ai_sr_frame_counter = 0
            ai_sr_latest_output_frame = None
            ai_sr_latest_output_ts = 0.0
            ai_sr_completed_frames = 0
            ai_sr_warmup_pending = False
            ai_sr_dropped_frames = 0
            ai_sr_applied_frames = 0
            ai_sr_reused_frames = 0
            ai_sr_passthrough_frames = 0
            error_text = str(ai_exc)
            if ort is not None:
                try:
                    ort_providers = ort.get_available_providers()
                except Exception:
                    ort_providers = []
                ort_module = getattr(ort, "__file__", "unknown")
                error_text = (
                    f"{error_text} | onnxruntime_module={ort_module} | "
                    f"available_providers={ort_providers}"
                )
            return error_text

    def _refresh_rtx_vsr_engine() -> str | None:
        nonlocal rtx_vsr_engine, rtx_vsr_info, rtx_roi_rebuild_pending

        rtx_roi_rebuild_pending = False

        if rtx_vsr_engine is not None:
            try:
                rtx_vsr_engine.close()
            except Exception:
                pass
            rtx_vsr_engine = None
        rtx_vsr_info = None

        if not rtx_vsr_enabled:
            return None

        resolved_rtx_module, resolved_rtx_error = _resolve_rtx_vsr_module()
        if resolved_rtx_module is None:
            return (
                "rtx_vsr module is not available"
                f" | import_error={resolved_rtx_error}"
                f" | sdk_root={rtx_video_sdk_root or 'unset'}"
                f" | worker_python={sys.executable}"
            )
        if cv2 is None:
            return "opencv-python is required for RTX VSR color conversion"

        try:
            in_w = max(2, int(current_roi_w) & ~1)
            in_h = max(2, int(current_roi_h))
            rtx_vsr_engine = resolved_rtx_module.RTXVideoSR(
                in_w,
                in_h,
                FRAME_W,
                FRAME_H,
                quality=rtx_vsr_quality,
                thdr_enabled=rtx_thdr_enabled,
                thdr_contrast=rtx_thdr_contrast,
                thdr_saturation=rtx_thdr_saturation,
                thdr_middle_gray=rtx_thdr_middle_gray,
                thdr_max_luminance=rtx_thdr_max_luminance,
            )
            engine_input_w = int(getattr(rtx_vsr_engine, "input_width", in_w))
            engine_input_h = int(getattr(rtx_vsr_engine, "input_height", in_h))
            engine_output_w = int(getattr(rtx_vsr_engine, "output_width", FRAME_W))
            engine_output_h = int(getattr(rtx_vsr_engine, "output_height", FRAME_H))
            rtx_vsr_info = {
                "backend": "nvidia_rtx_video_sdk",
                "quality": rtx_vsr_quality,
                "scale": int(rtx_vsr_scale),
                "post_scale_method": rtx_vsr_post_scale_method,
                "thdr_enabled": bool(rtx_thdr_enabled),
                "thdr_contrast": int(rtx_thdr_contrast),
                "thdr_saturation": int(rtx_thdr_saturation),
                "thdr_middle_gray": int(rtx_thdr_middle_gray),
                "thdr_max_luminance": int(rtx_thdr_max_luminance),
                "input_w": engine_input_w,
                "input_h": engine_input_h,
                "output_w": engine_output_w,
                "output_h": engine_output_h,
            }
            return None
        except Exception as rtx_exc:
            rtx_vsr_engine = None
            rtx_vsr_info = None
            module_file = getattr(resolved_rtx_module, "__file__", "unknown")
            return (
                f"{rtx_exc}"
                f" | roi={current_roi_w}x{current_roi_h}"
                f" | engine_in={in_w}x{in_h}"
                f" | engine_out={FRAME_W}x{FRAME_H}"
                f" | quality={rtx_vsr_quality}"
                f" | thdr_enabled={rtx_thdr_enabled}"
                f" | thdr_contrast={rtx_thdr_contrast}"
                f" | thdr_saturation={rtx_thdr_saturation}"
                f" | thdr_middle_gray={rtx_thdr_middle_gray}"
                f" | thdr_max_luminance={rtx_thdr_max_luminance}"
                f" | sdk_root={rtx_video_sdk_root or 'unset'}"
                f" | module={module_file}"
            )

    def _schedule_rtx_roi_rebuild() -> None:
        nonlocal rtx_roi_rebuild_pending, rtx_roi_rebuild_due_ts
        rtx_roi_rebuild_pending = True
        rtx_roi_rebuild_due_ts = time.perf_counter() + rtx_roi_rebuild_settle_s

    def _maybe_run_pending_rtx_roi_rebuild() -> None:
        nonlocal rtx_vsr_error, rtx_roi_rebuild_pending
        if not rtx_roi_rebuild_pending:
            return
        if roi_microstep_transition is not None:
            # Avoid rebuilding RTX engine during active ROI keyframe scaling;
            # defer until transition settles to prevent visible render hitches.
            return
        if not rtx_vsr_enabled:
            rtx_roi_rebuild_pending = False
            return
        if time.perf_counter() < rtx_roi_rebuild_due_ts:
            return

        rtx_vsr_error = _refresh_rtx_vsr_engine()
        if rtx_vsr_error:
            _safe_put({"type": "warning", "warning": f"RTX VSR ROI resize reconfigure failed: {rtx_vsr_error}"})

    def _close_rtx_vsr_engine() -> None:
        nonlocal rtx_vsr_engine
        if rtx_vsr_engine is None:
            return
        try:
            rtx_vsr_engine.close()
        except Exception:
            pass
        rtx_vsr_engine = None

    def _stop_sessions() -> None:
        nonlocal capture_session, output_session

        _stop_live_pipeline()

        if output_session is not None:
            _clear_output_schedule_state(output_session)
            try:
                output_session.stop()
            except Exception:
                pass
            output_session = None

        if capture_session is not None:
            try:
                capture_session.stop()
            except Exception:
                pass
            capture_session = None

    def _start_sessions(message: dict[str, Any]) -> None:
        nonlocal capture_session, output_session, current_output_buffer_frames, output_mode_is_interlaced
        nonlocal output_field_dominance_code
        nonlocal output_mode_name, output_mode_value, output_field_dominance_name
        if d is None:
            raise RuntimeError("decklink_wrapper is not available in worker process")

        requested_format_detection = bool(message["enable_format_detection"])
        requested_in_mode = message["in_mode"]
        requested_out_mode = message["out_mode"]
        current_output_buffer_frames = max(
            0,
            min(10, int(message.get("decklink_output_buffer_frames", current_output_buffer_frames))),
        )

        def _normalize_mode_key(value: object) -> str:
            if value is None:
                return ""
            try:
                return str(int(value))
            except Exception:
                pass
            return str(value).strip().lower()

        def _mode_name_is_interlaced(name: str) -> bool:
            mode_name = str(name).strip().lower()
            if not mode_name:
                return False
            if "progressive" in mode_name or "psf" in mode_name:
                return False
            if "interlace" in mode_name:
                return True
            for idx, ch in enumerate(mode_name):
                if ch == "i" and idx > 0 and (idx + 1) < len(mode_name):
                    if mode_name[idx - 1].isdigit() and mode_name[idx + 1].isdigit():
                        return True
            return False

        def _mode_name_is_progressive(name: str) -> bool:
            mode_name = str(name).strip().lower()
            if not mode_name:
                return False
            if "progressive" in mode_name or "psf" in mode_name:
                return True
            for idx, ch in enumerate(mode_name):
                if ch == "p" and idx > 0 and (idx + 1) < len(mode_name):
                    if mode_name[idx - 1].isdigit() and mode_name[idx + 1].isdigit():
                        return True
            return False

        def _resolve_display_mode_entry(device_index: int, requested_mode: object, input_side: bool) -> object:
            list_modes = d.list_input_display_modes if input_side else d.list_output_display_modes
            mode_entries = list(list_modes(int(device_index)))
            if not mode_entries:
                side = "input" if input_side else "output"
                raise RuntimeError(f"No DeckLink {side} display modes found for device index={device_index}")

            requested_key = _normalize_mode_key(requested_mode)
            requested_text = str(requested_mode).strip().lower()

            for entry in mode_entries:
                entry_mode = getattr(entry, "mode", None)
                if entry_mode == requested_mode:
                    return entry
                if requested_key and _normalize_mode_key(entry_mode) == requested_key:
                    return entry
                if requested_text and str(getattr(entry, "name", "")).strip().lower() == requested_text:
                    return entry

            side = "input" if input_side else "output"
            available = [
                {
                    "name": str(getattr(entry, "name", "")),
                    "mode": str(getattr(entry, "mode", "")),
                    "w": int(getattr(entry, "width", 0)),
                    "h": int(getattr(entry, "height", 0)),
                }
                for entry in mode_entries
            ]
            raise RuntimeError(
                f"Requested DeckLink {side} mode was not found on worker side | "
                f"requested={requested_mode!r} | available={available}"
            )

        resolved_in_entry = _resolve_display_mode_entry(int(message["in_device"]), requested_in_mode, input_side=True)
        resolved_out_entry = _resolve_display_mode_entry(int(message["out_device"]), requested_out_mode, input_side=False)
        resolved_in_mode = getattr(resolved_in_entry, "mode", None)
        resolved_out_mode = getattr(resolved_out_entry, "mode", None)
        if resolved_in_mode is None or resolved_out_mode is None:
            raise RuntimeError("DeckLink display mode resolution failed: missing mode value")
        output_mode_name = str(getattr(resolved_out_entry, "name", ""))
        output_mode_value = str(getattr(resolved_out_entry, "mode", ""))
        output_field_dominance_name = str(getattr(resolved_out_entry, "field_dominance_name", ""))
        output_mode_is_interlaced = _mode_name_is_interlaced(str(getattr(resolved_out_entry, "name", "")))
        output_field_dominance_code = None
        try:
            output_modes = list(d.list_output_display_modes(int(message["out_device"])))
            progressive_field_dominance_codes = {
                int(getattr(mode_entry, "field_dominance"))
                for mode_entry in output_modes
                if _mode_name_is_progressive(str(getattr(mode_entry, "name", "")))
                and getattr(mode_entry, "field_dominance", None) is not None
            }
            interlaced_field_dominance_codes = {
                int(getattr(mode_entry, "field_dominance"))
                for mode_entry in output_modes
                if _mode_name_is_interlaced(str(getattr(mode_entry, "name", "")))
                and getattr(mode_entry, "field_dominance", None) is not None
            }
            resolved_field_dominance = getattr(resolved_out_entry, "field_dominance", None)
            if resolved_field_dominance is not None:
                output_field_dominance_code = int(resolved_field_dominance)

            resolved_field_dominance_name = str(
                getattr(resolved_out_entry, "field_dominance_name", "")
            ).strip().lower()
            output_field_dominance_name = str(getattr(resolved_out_entry, "field_dominance_name", ""))
            if resolved_field_dominance_name:
                if "progressive" in resolved_field_dominance_name:
                    output_mode_is_interlaced = False
                elif (
                    ("lower" in resolved_field_dominance_name)
                    or ("upper" in resolved_field_dominance_name)
                    or ("interlace" in resolved_field_dominance_name)
                ):
                    output_mode_is_interlaced = True
            elif resolved_field_dominance is not None:
                resolved_field_code = int(resolved_field_dominance)
                if (
                    resolved_field_code in interlaced_field_dominance_codes
                    and resolved_field_code not in progressive_field_dominance_codes
                ):
                    output_mode_is_interlaced = True
                elif (
                    resolved_field_code in progressive_field_dominance_codes
                    and resolved_field_code not in interlaced_field_dominance_codes
                ):
                    output_mode_is_interlaced = False
        except Exception:
            pass

        def _open_sessions(enable_format_detection: bool) -> None:
            nonlocal capture_session, output_session
            _stop_sessions()

            capture_session = d.CaptureSession(
                device_index=int(message["in_device"]),
                display_mode=resolved_in_mode,
                pixel_format=d.PIXEL_FORMAT_8BIT_YUV,
                max_queue_frames=8,
                enable_format_detection=bool(enable_format_detection),
            )
            output_session = d.OutputSession(
                device_index=int(message["out_device"]),
                display_mode=resolved_out_mode,
                pixel_format=d.PIXEL_FORMAT_8BIT_YUV,
            )

            try:
                capture_session.start()
                output_session.start()
                _set_output_schedule_buffer_frames(output_session, current_output_buffer_frames)
                _start_live_pipeline()
            except Exception:
                _stop_sessions()
                raise

        try:
            _open_sessions(requested_format_detection)
        except Exception as first_exc:
            first_text = str(first_exc)
            should_retry_without_detection = (
                requested_format_detection
                and "EnableVideoInput" in first_text
            )
            if not should_retry_without_detection:
                raise

            _safe_put(
                {
                    "type": "warning",
                    "warning": (
                        "DeckLink start retry: EnableVideoInput failed with format detection enabled; "
                        "retrying with format detection disabled"
                    ),
                }
            )
            _open_sessions(False)
    try:
        project_root = Path(startup_config["project_root"])
        module = _load_video_processor_module(project_root)
        processor, basic_scaling_method_supported = _create_processor(module, startup_config)
        reusable_native_into_supported = (
            hasattr(processor, "process_frame_into")
            and hasattr(processor, "process_frame_no_deinterlace_into")
            and hasattr(processor, "process_frame_deinterlace_only_into")
            and hasattr(processor, "process_frame_preprocess_only_into")
        )
        if reusable_native_into_supported:
            frame_bytes = UYVY_ROW_BYTES * FRAME_H
            reusable_process_frame_out = bytearray(frame_bytes)
            reusable_process_frame_no_deinterlace_out = bytearray(frame_bytes)
            reusable_process_frame_deinterlace_only_out = bytearray(frame_bytes)
            reusable_process_frame_preprocess_only_out = bytearray(frame_bytes)
        ai_sr_error = _refresh_ai_sr_engine()
        rtx_vsr_error = _refresh_rtx_vsr_engine()
        _safe_put(
            {
                "type": "ready",
                "basic_scaling_method_supported": bool(basic_scaling_method_supported),
                "sr_flavor_supported": bool(basic_scaling_method_supported),
                "ai_sr_enabled": bool(ai_sr_enabled),
                "ai_sr_active": bool(ai_sr_engine is not None),
                "ai_sr_error": ai_sr_error,
                "ai_sr_info": ai_sr_info,
                "rtx_vsr_enabled": bool(rtx_vsr_enabled),
                "rtx_vsr_active": bool(rtx_vsr_engine is not None and not (ai_sr_enabled and ai_sr_engine is not None)),
                "rtx_vsr_error": rtx_vsr_error,
                "rtx_vsr_info": rtx_vsr_info,
                "color_space": current_color_space,
                "color_range": current_color_range,
                "worker_process_priority": worker_process_priority,
                "worker_process_priority_error": worker_process_priority_error,
            }
        )

        while True:
            _maybe_run_pending_rtx_roi_rebuild()

            message = None
            try:
                message = request_queue.get_nowait()
            except queue.Empty:
                message = None

            if message is None:
                if capture_session is None or output_session is None:
                    # Idle backoff when no active DeckLink sessions and no control message.
                    time.sleep(0.002)
                else:
                    time.sleep(0.001)
                continue

            command = message.get("cmd")
            if command == "decklink_tick":
                latest_roi_message = None
                preserved_messages: list[dict[str, object]] = []
                while True:
                    try:
                        pending = request_queue.get_nowait()
                    except queue.Empty:
                        break

                    pending_cmd = pending.get("cmd")
                    if pending_cmd in {"set_roi", "set_roi_position", "set_roi_with_subpixel"}:
                        latest_roi_message = pending
                        continue
                    if pending_cmd == "decklink_tick":
                        continue
                    preserved_messages.append(pending)

                for pending in preserved_messages:
                    try:
                        request_queue.put_nowait(pending)
                    except queue.Full:
                        break

                # Keep one latest ROI update queued, but always service this tick.
                # Replacing tick with ROI here can leave GUI tick requests pending
                # until timeout, collapsing preview FPS while output keeps running.
                if latest_roi_message is not None:
                    try:
                        request_queue.put_nowait(latest_roi_message)
                    except queue.Full:
                        pass

            if command in {"set_roi", "set_roi_position", "set_roi_with_subpixel", "set_roi_subpixel_shift"}:
                latest_roi_message = message
                latest_tick_message: dict[str, object] | None = None
                preserved_messages: list[dict[str, object]] = []

                while True:
                    try:
                        pending = request_queue.get_nowait()
                    except queue.Empty:
                        break

                    pending_cmd = pending.get("cmd")
                    if pending_cmd in {"set_roi", "set_roi_position", "set_roi_with_subpixel", "set_roi_subpixel_shift"}:
                        latest_roi_message = pending
                        continue
                    if pending_cmd == "decklink_tick":
                        latest_tick_message = pending
                        continue
                    preserved_messages.append(pending)

                for pending in preserved_messages:
                    try:
                        request_queue.put_nowait(pending)
                    except queue.Full:
                        break

                if latest_tick_message is not None:
                    try:
                        request_queue.put_nowait(latest_tick_message)
                    except queue.Full:
                        pass

                message = latest_roi_message
                command = message.get("cmd")

            if command == "shutdown":
                _stop_sessions()
                _cleanup_ai_async()
                _close_rtx_vsr_engine()
                return

            if command == "start_decklink":
                try:
                    _start_sessions(message)
                    _safe_put(
                        {
                            "type": "ack",
                            "cmd": "start_decklink",
                            "decklink_started": True,
                            "decklink_error": None,
                            "output_mode_name": str(output_mode_name),
                            "output_mode_value": str(output_mode_value),
                            "output_mode_is_interlaced": bool(output_mode_is_interlaced),
                            "output_field_dominance_code": output_field_dominance_code,
                            "output_field_dominance_name": str(output_field_dominance_name),
                        }
                    )
                except Exception as decklink_exc:
                    _stop_sessions()
                    _safe_put(
                        {
                            "type": "ack",
                            "cmd": "start_decklink",
                            "decklink_started": False,
                            "decklink_error": str(decklink_exc),
                        }
                    )
                continue

            if command == "set_decklink_output_buffer_frames":
                current_output_buffer_frames = max(
                    0,
                    min(10, int(message.get("decklink_output_buffer_frames", current_output_buffer_frames))),
                )
                if output_session is not None:
                    _set_output_schedule_buffer_frames(output_session, current_output_buffer_frames)
                    _reprime_output_schedule(output_session, reason="manual_buffer_frames_change")
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_decklink_output_buffer_frames",
                        "decklink_output_buffer_frames": int(current_output_buffer_frames),
                    }
                )
                continue

            if command == "set_worker_process_priority":
                requested_priority = str(message.get("worker_process_priority", worker_process_priority))
                worker_process_priority, worker_process_priority_error = _apply_current_process_priority(requested_priority)
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_worker_process_priority",
                        "worker_process_priority": worker_process_priority,
                        "worker_process_priority_error": worker_process_priority_error,
                    }
                )
                continue

            if command == "stop_decklink":
                _stop_sessions()
                _safe_put({"type": "ack", "cmd": "stop_decklink"})
                continue

            if command == "decklink_tick":
                if capture_session is None or output_session is None:
                    _safe_put({"type": "decklink_no_frame", "reason": "sessions_not_started"})
                    continue

                include_frames = bool(message.get("include_frames", True))

                with state_lock:
                    current_input = latest_input_frame
                    current_output = latest_output_frame
                    current_scale = int(latest_effective_sr_scale)
                    current_counter = int(processed_frame_counter)
                    current_rtx_applied = bool(latest_rtx_vsr_applied)
                    current_rtx_delta = float(latest_rtx_effect_mean_abs_luma)
                    current_timecode_info = dict(latest_timecode_info)
                    timing_frames_emitted_snapshot = int(timing_frames_emitted)
                    timing_deadline_miss_events_snapshot = int(timing_deadline_miss_events)
                    timing_deadline_miss_streak_snapshot = int(timing_deadline_miss_streak)
                    timing_deadline_miss_max_streak_snapshot = int(timing_deadline_miss_max_streak)
                    timing_deadline_late_ms_ema_snapshot = float(timing_deadline_late_ms_ema)
                    timing_deadline_late_ms_peak_snapshot = float(timing_deadline_late_ms_peak)
                    timing_e2e_ms_last_snapshot = float(timing_e2e_ms_last)
                    timing_e2e_ms_ema_snapshot = float(timing_e2e_ms_ema)
                    timing_e2e_ms_peak_snapshot = float(timing_e2e_ms_peak)
                    timing_process_ms_ema_snapshot = float(timing_process_ms_ema)
                    timing_process_ms_peak_snapshot = float(timing_process_ms_peak)
                    timing_capture_queue_ms_ema_snapshot = float(timing_capture_queue_ms_ema)
                    timing_capture_queue_ms_peak_snapshot = float(timing_capture_queue_ms_peak)
                    timing_output_queue_ms_ema_snapshot = float(timing_output_queue_ms_ema)
                    timing_output_queue_ms_peak_snapshot = float(timing_output_queue_ms_peak)
                    timing_output_wait_ms_ema_snapshot = float(timing_output_wait_ms_ema)
                    timing_output_wait_ms_peak_snapshot = float(timing_output_wait_ms_peak)
                    timing_emit_call_ms_ema_snapshot = float(timing_emit_call_ms_ema)
                    timing_emit_call_ms_peak_snapshot = float(timing_emit_call_ms_peak)
                    timing_last_path_snapshot = str(timing_last_path)
                    current_roi_x_snapshot = int(current_roi_x)
                    current_roi_y_snapshot = int(current_roi_y)
                    current_roi_w_snapshot = int(current_roi_w)
                    current_roi_h_snapshot = int(current_roi_h)
                    roi_shift_applied_x_snapshot = float(roi_shift_applied_x)
                    roi_shift_applied_y_snapshot = float(roi_shift_applied_y)
                    roi_shift_target_x_snapshot = float(roi_shift_target_x)
                    roi_shift_target_y_snapshot = float(roi_shift_target_y)
                    roi_transition_state_snapshot = dict(roi_microstep_transition) if isinstance(roi_microstep_transition, dict) else None

                if current_input is None or current_output is None:
                    _safe_put({"type": "decklink_no_frame", "reason": "no_input_signal"})
                    continue

                elapsed = max(0.0001, time.perf_counter() - started_perf_ts)
                processed_fps = float(current_counter) / elapsed
                if output_nominal_fps > 0.0:
                    processed_fps = min(processed_fps, output_nominal_fps)
                stage_depths = {
                    "capture_to_preprocess": 0 if q_capture_to_preprocess is None else q_capture_to_preprocess.qsize(),
                    "preprocess_to_upscale": 0 if q_preprocess_to_upscale is None else q_preprocess_to_upscale.qsize(),
                    "upscale_to_output": 0 if q_upscale_to_output is None else q_upscale_to_output.qsize(),
                }
                ai_sr_timing_ms = ai_sr_engine.timing_info() if ai_sr_engine is not None else {}
                output_schedule_stats: dict[str, object] = {}
                if output_session is not None:
                    out_state = _OUTPUT_SCHEDULE_STATE.get(id(output_session))
                    if out_state is not None:
                        output_schedule_stats = {
                            "started": bool(out_state.get("started", False)),
                            "target_buffer_frames": int(out_state.get("target_buffer_frames", 0)),
                            "last_buffered_count": int(out_state.get("last_buffered_count", -1)),
                            "starvation_events": int(out_state.get("starvation_events", 0)),
                            "overflow_events": int(out_state.get("overflow_events", 0)),
                            "auto_reprime_events": int(out_state.get("auto_reprime_events", 0)),
                            "last_reprime_reason": str(out_state.get("last_reprime_reason", "")),
                            "last_reprime_age_ms": (
                                max(0.0, (time.perf_counter() - float(out_state.get("last_reprime_ts", 0.0))) * 1000.0)
                                if float(out_state.get("last_reprime_ts", 0.0)) > 0.0
                                else -1.0
                            ),
                        }
                deadline_miss_ratio = (
                    float(timing_deadline_miss_events_snapshot) / float(timing_frames_emitted_snapshot)
                    if timing_frames_emitted_snapshot > 0
                    else 0.0
                )
                output_budget_ms = (1000.0 / output_nominal_fps) if output_nominal_fps > 0.0 else 0.0
                pipeline_timing_health: dict[str, object] = {
                    "frames_emitted": int(timing_frames_emitted_snapshot),
                    "output_budget_ms": float(output_budget_ms),
                    "deadline_miss_events": int(timing_deadline_miss_events_snapshot),
                    "deadline_miss_ratio": float(deadline_miss_ratio),
                    "deadline_miss_streak": int(timing_deadline_miss_streak_snapshot),
                    "deadline_miss_max_streak": int(timing_deadline_miss_max_streak_snapshot),
                    "deadline_late_ms_ema": float(timing_deadline_late_ms_ema_snapshot),
                    "deadline_late_ms_peak": float(timing_deadline_late_ms_peak_snapshot),
                    "e2e_ms_last": float(timing_e2e_ms_last_snapshot),
                    "e2e_ms_ema": float(timing_e2e_ms_ema_snapshot),
                    "e2e_ms_peak": float(timing_e2e_ms_peak_snapshot),
                    "process_ms_ema": float(timing_process_ms_ema_snapshot),
                    "process_ms_peak": float(timing_process_ms_peak_snapshot),
                    "capture_queue_ms_ema": float(timing_capture_queue_ms_ema_snapshot),
                    "capture_queue_ms_peak": float(timing_capture_queue_ms_peak_snapshot),
                    "output_queue_ms_ema": float(timing_output_queue_ms_ema_snapshot),
                    "output_queue_ms_peak": float(timing_output_queue_ms_peak_snapshot),
                    "output_wait_ms_ema": float(timing_output_wait_ms_ema_snapshot),
                    "output_wait_ms_peak": float(timing_output_wait_ms_peak_snapshot),
                    "emit_call_ms_ema": float(timing_emit_call_ms_ema_snapshot),
                    "emit_call_ms_peak": float(timing_emit_call_ms_peak_snapshot),
                    "last_path": str(timing_last_path_snapshot),
                }
                roi_transition_payload: dict[str, object]
                if isinstance(roi_transition_state_snapshot, dict):
                    start_roi = roi_transition_state_snapshot.get("start", (current_roi_x_snapshot, current_roi_y_snapshot, current_roi_w_snapshot, current_roi_h_snapshot))
                    target_roi = roi_transition_state_snapshot.get("target", (current_roi_x_snapshot, current_roi_y_snapshot, current_roi_w_snapshot, current_roi_h_snapshot))
                    interlaced_field_phase = _active_interlaced_field_phase_state(consume_manual_snapshot=False)
                    roi_transition_payload = {
                        "active": True,
                        "frame_progress": float(roi_transition_state_snapshot.get("frame_progress", 0.0)),
                        "total_frames": int(roi_transition_state_snapshot.get("total_frames", 1)),
                        "interpolation_mode": str(roi_transition_state_snapshot.get("interpolation_mode", "linear")),
                        "start_roi": {
                            "x": int(start_roi[0]),
                            "y": int(start_roi[1]),
                            "w": int(start_roi[2]),
                            "h": int(start_roi[3]),
                        },
                        "target_roi": {
                            "x": int(target_roi[0]),
                            "y": int(target_roi[1]),
                            "w": int(target_roi[2]),
                            "h": int(target_roi[3]),
                        },
                    }
                    if isinstance(interlaced_field_phase, dict):
                        roi_transition_payload["interlaced_field_phase"] = {
                            "roi0": tuple(interlaced_field_phase.get("roi0", (current_roi_x_snapshot, current_roi_y_snapshot, current_roi_w_snapshot, current_roi_h_snapshot))),
                            "roi1": tuple(interlaced_field_phase.get("roi1", (current_roi_x_snapshot, current_roi_y_snapshot, current_roi_w_snapshot, current_roi_h_snapshot))),
                            "field0_x": float(interlaced_field_phase.get("field0_x", 0.0)),
                            "field0_y": float(interlaced_field_phase.get("field0_y", 0.0)),
                            "field1_x": float(interlaced_field_phase.get("field1_x", 0.0)),
                            "field1_y": float(interlaced_field_phase.get("field1_y", 0.0)),
                        }
                else:
                    roi_transition_payload = {
                        "active": False,
                        "frame_progress": 0.0,
                        "total_frames": 0,
                        "interpolation_mode": "",
                    }

                _publish_roi_telemetry(
                    current_roi_x_snapshot,
                    current_roi_y_snapshot,
                    current_roi_w_snapshot,
                    current_roi_h_snapshot,
                    roi_transition_state_snapshot,
                )

                payload: dict[str, object] = {
                    "type": "decklink_frame",
                    "effective_sr_scale": current_scale,
                    "processed_frame_counter": current_counter,
                    "processed_fps": processed_fps,
                    "output_nominal_fps": float(output_nominal_fps),
                    "output_mode_is_interlaced": bool(output_mode_is_interlaced),
                    "output_transition_units_per_frame": float(output_transition_units_per_frame),
                    "ai_sr_applied_frames": int(ai_sr_applied_frames),
                    "ai_sr_reused_frames": int(ai_sr_reused_frames),
                    "ai_sr_passthrough_frames": int(ai_sr_passthrough_frames),
                    "ai_sr_completed_frames": int(ai_sr_completed_frames),
                    "ai_sr_latest_age_ms": max(0.0, (time.perf_counter() - float(ai_sr_latest_output_ts)) * 1000.0)
                    if ai_sr_latest_output_ts > 0.0
                    else -1.0,
                    "ai_sr_timing_ms": ai_sr_timing_ms,
                    "rtx_vsr_applied": current_rtx_applied,
                    "rtx_effect_mean_abs_luma": current_rtx_delta,
                    "stage_enable_flags": {
                        "preprocess": bool(_is_preprocess_stage_enabled()),
                        "basic_scaling": bool(_basic_scaling_enabled()),
                        "ai_sr": bool(ai_sr_enabled and ai_sr_engine is not None),
                        "rtx_vsr": bool(rtx_vsr_enabled and rtx_vsr_engine is not None),
                    },
                    "stage_last_applied": {
                        "preprocess": bool(last_stage_preprocess_applied),
                        "basic_scaling": bool(last_stage_basic_applied),
                        "ai_sr": bool(last_stage_ai_applied),
                        "rtx_vsr": bool(last_stage_rtx_applied),
                    },
                    "stage_stack": list(last_stage_stack),
                    "stage_apply_counts": {
                        "preprocess": int(stage_preprocess_applied_frames),
                        "basic_scaling": int(stage_basic_applied_frames),
                        "ai_sr": int(stage_ai_applied_frames),
                        "rtx_vsr": int(stage_rtx_applied_frames),
                        "passthrough": int(stage_passthrough_frames),
                    },
                    "pipeline_running": bool(pipeline_running),
                    "stage_queue_depths": stage_depths,
                    "stage_drop_counts": {
                        "capture": int(capture_drop_count),
                        "preprocess": int(preprocess_drop_count),
                        "upscale": int(upscale_drop_count),
                    },
                    "basic_scaling_timing_ms": {
                        "last": float(basic_scaling_last_frame_ms),
                        "avg": None if basic_scaling_avg_frame_ms is None else float(basic_scaling_avg_frame_ms),
                        "max": float(basic_scaling_max_frame_ms),
                        "samples": int(basic_scaling_timing_samples),
                    },
                    "roi_applied": {
                        "x": current_roi_x_snapshot,
                        "y": current_roi_y_snapshot,
                        "w": current_roi_w_snapshot,
                        "h": current_roi_h_snapshot,
                    },
                    "roi_subpixel_shift": {
                        "target_x": roi_shift_target_x_snapshot,
                        "target_y": roi_shift_target_y_snapshot,
                        "applied_x": roi_shift_applied_x_snapshot,
                        "applied_y": roi_shift_applied_y_snapshot,
                    },
                    "roi_transition": roi_transition_payload,
                    "output_buffer_health": output_schedule_stats,
                    "pipeline_timing_health": pipeline_timing_health,
                    "timecode_info": current_timecode_info,
                }
                if include_frames:
                    payload["input_frame_bytes"] = _freeze_frame_bytes(current_input)
                    payload["output_frame_bytes"] = _freeze_frame_bytes(current_output)
                _safe_put(payload)
                continue

            if command == "process_frame":
                frame_id = int(message["frame_id"])
                frame_bytes = message["frame_bytes"]
                _advance_roi_microstep_transition_for_output_frame()
                shift_x, shift_y = _step_smoothed_roi_shift()
                interlaced_phase = _active_interlaced_field_phase_state(consume_manual_snapshot=True)
                if output_mode_is_interlaced and _reinterlace_enabled_for_output():
                    if interlaced_phase is None:
                        interlaced_phase = _build_static_interlaced_phase_for_reinterlace(shift_x, shift_y)
                    output_bytes, _, basic_applied, ai_applied, rtx_applied, native_shift_applied = _render_dual_phase_full_pipeline_reinterlace(
                        frame_bytes,
                        interlaced_phase,
                    )
                elif interlaced_phase is not None:
                    output_bytes, _, basic_applied, ai_applied, rtx_applied, native_shift_applied = _render_dual_phase_full_pipeline(
                        frame_bytes,
                        interlaced_phase,
                    )
                else:
                    output_bytes, _, basic_applied, ai_applied, rtx_applied, native_shift_applied = _process_pipeline_frame(
                        frame_bytes,
                        shift_x,
                        shift_y,
                    )
                if (
                    (not native_shift_applied)
                    and (not output_mode_is_interlaced)
                    and _should_apply_cpu_subpixel_fallback()
                    and _has_effective_subpixel_shift(shift_x, shift_y)
                ):
                    output_bytes = _apply_subpixel_shift_uyvy(output_bytes, shift_x, shift_y)
                if interlaced_phase is None:
                    output_bytes = _apply_interlaced_field_phase_if_needed(
                        output_bytes,
                        shift_x,
                        shift_y,
                        native_shift_applied=native_shift_applied,
                    )
                if interlaced_phase is None and _reinterlace_enabled_for_output():
                    output_bytes = _apply_reinterlace_from_previous_frame_if_needed(output_bytes)

                if ai_sr_engine is not None and not ai_applied and _ai_inference_busy():
                    ai_sr_dropped_frames += 1

                latest_output_frame = output_bytes
                last_stage_basic_applied = bool(basic_applied)
                last_stage_ai_applied = bool(ai_applied)
                last_stage_rtx_applied = bool(rtx_applied)
                last_stage_stack = _build_stage_stack()
                with state_lock:
                    _publish_roi_telemetry(
                        int(current_roi_x),
                        int(current_roi_y),
                        int(current_roi_w),
                        int(current_roi_h),
                        dict(roi_microstep_transition) if isinstance(roi_microstep_transition, dict) else None,
                    )
                _safe_put(
                    {
                        "type": "frame",
                        "frame_id": frame_id,
                        "frame_bytes": output_bytes,
                        "effective_sr_scale": int(processor.get_effective_sr_scale()),
                    }
                )
                continue

            if command == "set_roi":
                _cancel_roi_microstep_transition(reset_shift=True)
                _apply_manual_roi_with_subpixel_compensation(
                    int(message["x"]),
                    int(message["y"]),
                    int(message["w"]),
                    int(message["h"]),
                )
                continue

            if command == "set_roi_position":
                _cancel_roi_microstep_transition(reset_shift=True)
                _apply_manual_roi_with_subpixel_compensation(
                    int(message["x"]),
                    int(message["y"]),
                    current_roi_w,
                    current_roi_h,
                )
                continue

            if command == "start_roi_microstep_transition":
                start_from_current = bool(message.get("start_from_current", False))
                if start_from_current:
                    start_roi = (current_roi_x, current_roi_y, current_roi_w, current_roi_h)
                else:
                    start_roi = (
                        int(message.get("start_x", current_roi_x)),
                        int(message.get("start_y", current_roi_y)),
                        int(message.get("start_w", current_roi_w)),
                        int(message.get("start_h", current_roi_h)),
                    )
                _start_roi_microstep_transition(
                    start_roi=start_roi,
                    target_roi=(
                        int(message["target_x"]),
                        int(message["target_y"]),
                        int(message["target_w"]),
                        int(message["target_h"]),
                    ),
                    duration_frames=int(message.get("duration_frames", 1)),
                    interpolation_mode=str(message.get("interpolation_mode", "linear")),
                    overscan_percent=float(message.get("overscan_percent", 0.0)),
                    enforce_full_frame_scale_1x=bool(message.get("enforce_full_frame_scale_1x", False)),
                )
                continue

            if command == "cancel_roi_microstep_transition":
                _cancel_roi_microstep_transition(reset_shift=bool(message.get("reset_subpixel_shift", True)))
                continue

            if command == "set_roi_with_subpixel":
                _cancel_roi_microstep_transition(reset_shift=False)
                prev_roi_state = (int(current_roi_x), int(current_roi_y), int(current_roi_w), int(current_roi_h))
                prev_shift_state = (float(roi_shift_applied_x), float(roi_shift_applied_y))
                if bool(message.get("manual_drag", False)):
                    roi_manual_drag_until_ts = time.perf_counter() + float(roi_manual_drag_hold_s)
                next_roi_x, next_roi_y, next_roi_w, next_roi_h = _normalize_worker_roi(
                    int(message["x"]),
                    int(message["y"]),
                    int(message["w"]),
                    int(message["h"]),
                )
                prev_roi_w = current_roi_w
                prev_roi_h = current_roi_h
                current_roi_x, current_roi_y, current_roi_w, current_roi_h = (
                    next_roi_x,
                    next_roi_y,
                    next_roi_w,
                    next_roi_h,
                )
                if current_roi_w == prev_roi_w and current_roi_h == prev_roi_h:
                    processor.set_roi_position(current_roi_x, current_roi_y)
                else:
                    processor.set_roi(current_roi_x, current_roi_y, current_roi_w, current_roi_h)
                    if rtx_vsr_enabled:
                        if rtx_vsr_engine is None:
                            rtx_vsr_error = _refresh_rtx_vsr_engine()
                        elif current_roi_w != prev_roi_w or current_roi_h != prev_roi_h:
                            _schedule_rtx_roi_rebuild()
                _set_roi_shift_target(
                    float(message.get("shift_x", 0.0)),
                    float(message.get("shift_y", 0.0)),
                )
                if output_mode_is_interlaced and _interlaced_phase_controls_active():
                    manual_interlaced_phase_state = _build_manual_interlaced_phase_snapshot(
                        prev_roi_state,
                        (int(current_roi_x), int(current_roi_y), int(current_roi_w), int(current_roi_h)),
                        prev_shift_state,
                        (float(roi_shift_target_x), float(roi_shift_target_y)),
                    )
                    manual_interlaced_phase_until_ts = time.perf_counter() + 0.12
                    manual_interlaced_phase_pending = True
                else:
                    manual_interlaced_phase_state = None
                    manual_interlaced_phase_until_ts = 0.0
                    manual_interlaced_phase_pending = False
                continue

            if command == "set_roi_subpixel_shift":
                _set_roi_shift_target(
                    float(message.get("shift_x", 0.0)),
                    float(message.get("shift_y", 0.0)),
                )
                continue

            if command == "set_roi_manual_drag_hold_seconds":
                roi_manual_drag_hold_s = max(0.05, min(0.50, float(message.get("hold_seconds", roi_manual_drag_hold_s))))
                continue

            if command == "set_interlaced_phase_shift_scale":
                # Deprecated; retained as a no-op for compatibility with stale
                # clients that may still emit this command.
                _safe_put({"type": "ack", "cmd": "set_interlaced_phase_shift_scale"})
                continue

            if command == "set_interlaced_field2_phase_fraction":
                interlaced_field2_phase_fraction = _clamp_interlaced_field2_phase_fraction(
                    float(message.get("fraction", interlaced_field2_phase_fraction))
                )
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_interlaced_field2_phase_fraction",
                        "interlaced_field2_phase_fraction": float(interlaced_field2_phase_fraction),
                    }
                )
                continue

            if command in {"set_basic_scaling_mode_auto", "set_sr_mode_auto"}:
                current_basic_scaling_auto_mode = True
                processor.set_sr_mode_auto()
                _safe_put({"type": "ack", "cmd": "set_basic_scaling_mode_auto"})
                continue

            if command in {"set_basic_scaling_manual", "set_sr_scale_manual"}:
                current_basic_scaling_auto_mode = False
                current_basic_scaling_manual_scale = int(message["scale"])
                processor.set_sr_scale_manual(int(current_basic_scaling_manual_scale))
                _safe_put({"type": "ack", "cmd": "set_basic_scaling_manual"})
                continue

            if command in {"set_basic_scaling_method", "set_sr_flavor"}:
                applied_basic_scaling_method = str(message.get("basic_scaling_method", message.get("sr_flavor", "bilinear_sharp")))
                if hasattr(processor, "set_sr_flavor"):
                    processor.set_sr_flavor(applied_basic_scaling_method)
                    if hasattr(processor, "get_sr_flavor"):
                        applied_basic_scaling_method = str(processor.get_sr_flavor())
                current_basic_scaling_method = applied_basic_scaling_method
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_basic_scaling_method",
                        "basic_scaling_method": applied_basic_scaling_method,
                        "sr_flavor": applied_basic_scaling_method,
                    }
                )
                continue

            if command == "set_deinterlace_enabled":
                current_deinterlace_enabled = bool(message["enabled"])
                processor.set_deinterlace_enabled(current_deinterlace_enabled)
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_deinterlace_enabled",
                        "deinterlace_enabled": bool(current_deinterlace_enabled),
                    }
                )
                continue

            if command == "set_reinterlace_enabled":
                current_reinterlace_enabled = bool(message.get("enabled", current_reinterlace_enabled))
                if not current_reinterlace_enabled:
                    prev_reinterlace_frame_bytes = None
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_reinterlace_enabled",
                        "reinterlace_enabled": bool(current_reinterlace_enabled),
                    }
                )
                continue

            if command == "set_deinterlace_method":
                current_deinterlace_method = str(message.get("method", current_deinterlace_method)).strip().lower()
                if hasattr(processor, "set_deinterlace_method"):
                    processor.set_deinterlace_method(current_deinterlace_method)
                    if hasattr(processor, "get_deinterlace_method"):
                        current_deinterlace_method = str(processor.get_deinterlace_method())
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_deinterlace_method",
                        "deinterlace_method": current_deinterlace_method,
                    }
                )
                continue

            if command == "set_denoise_settings":
                current_denoise_method = str(message.get("method", current_denoise_method)).strip().lower()
                current_denoise_strength = max(0.0, min(1.0, float(message.get("strength", current_denoise_strength))))
                if hasattr(processor, "set_denoise_method"):
                    processor.set_denoise_method(current_denoise_method)
                    if hasattr(processor, "get_denoise_method"):
                        current_denoise_method = str(processor.get_denoise_method())
                if hasattr(processor, "set_denoise_strength"):
                    processor.set_denoise_strength(current_denoise_strength)
                    if hasattr(processor, "get_denoise_strength"):
                        current_denoise_strength = float(processor.get_denoise_strength())
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_denoise_settings",
                        "denoise_method": current_denoise_method,
                        "denoise_strength": current_denoise_strength,
                    }
                )
                continue

            if command == "set_color_space":
                current_color_space = _normalize_color_space_name(str(message.get("color_space", current_color_space)))
                if hasattr(processor, "set_color_space"):
                    processor.set_color_space(current_color_space)
                    if hasattr(processor, "get_color_space"):
                        current_color_space = _normalize_color_space_name(str(processor.get_color_space()))
                ai_sr_error = _refresh_ai_sr_engine()
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_color_space",
                        "color_space": current_color_space,
                        "ai_sr_enabled": bool(ai_sr_enabled),
                        "ai_sr_active": bool(ai_sr_engine is not None),
                        "ai_sr_error": ai_sr_error,
                        "ai_sr_info": ai_sr_info,
                        "rtx_vsr_enabled": bool(rtx_vsr_enabled),
                        "rtx_vsr_active": bool(rtx_vsr_engine is not None and not (ai_sr_enabled and ai_sr_engine is not None)),
                        "rtx_vsr_error": rtx_vsr_error,
                        "rtx_vsr_info": rtx_vsr_info,
                    }
                )
                continue

            if command == "set_color_range":
                current_color_range = _normalize_color_range_name(str(message.get("color_range", current_color_range)))
                if hasattr(processor, "set_color_range"):
                    processor.set_color_range(current_color_range)
                    if hasattr(processor, "get_color_range"):
                        current_color_range = _normalize_color_range_name(str(processor.get_color_range()))
                ai_sr_error = _refresh_ai_sr_engine()
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_color_range",
                        "color_range": current_color_range,
                        "ai_sr_enabled": bool(ai_sr_enabled),
                        "ai_sr_active": bool(ai_sr_engine is not None),
                        "ai_sr_error": ai_sr_error,
                        "ai_sr_info": ai_sr_info,
                        "rtx_vsr_enabled": bool(rtx_vsr_enabled),
                        "rtx_vsr_active": bool(rtx_vsr_engine is not None and not (ai_sr_enabled and ai_sr_engine is not None)),
                        "rtx_vsr_error": rtx_vsr_error,
                        "rtx_vsr_info": rtx_vsr_info,
                    }
                )
                continue

            if command in {"set_max_auto_basic_scaling", "set_max_auto_sr_scale"}:
                current_max_auto_basic_scaling = int(message["scale"])
                processor.set_max_auto_sr_scale(int(current_max_auto_basic_scaling))
                continue

            if command == "set_ai_sr_enabled":
                ai_sr_enabled = bool(message.get("enabled", False))
                ai_sr_error = _refresh_ai_sr_engine()
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_ai_sr_enabled",
                        "ai_sr_enabled": bool(ai_sr_enabled),
                        "ai_sr_active": bool(ai_sr_engine is not None),
                        "basic_upscale_enabled": not bool(ai_sr_enabled),
                        "ai_sr_error": ai_sr_error,
                        "ai_sr_info": ai_sr_info,
                        "rtx_vsr_enabled": bool(rtx_vsr_enabled),
                        "rtx_vsr_active": bool(rtx_vsr_engine is not None and not (ai_sr_enabled and ai_sr_engine is not None)),
                        "rtx_vsr_error": rtx_vsr_error,
                        "rtx_vsr_info": rtx_vsr_info,
                    }
                )
                continue

            if command == "set_ai_sr_model_path":
                ai_sr_model_path = str(message.get("model_path", ""))
                ai_sr_error = _refresh_ai_sr_engine()
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_ai_sr_model_path",
                        "ai_sr_enabled": bool(ai_sr_enabled),
                        "ai_sr_active": bool(ai_sr_engine is not None),
                        "basic_upscale_enabled": not bool(ai_sr_enabled),
                        "ai_sr_error": ai_sr_error,
                        "ai_sr_info": ai_sr_info,
                        "rtx_vsr_enabled": bool(rtx_vsr_enabled),
                        "rtx_vsr_active": bool(rtx_vsr_engine is not None and not (ai_sr_enabled and ai_sr_engine is not None)),
                        "rtx_vsr_error": rtx_vsr_error,
                        "rtx_vsr_info": rtx_vsr_info,
                    }
                )
                continue

            if command == "set_ai_sr_settings":
                ai_sr_provider = str(message.get("provider", ai_sr_provider))
                ai_sr_trt_precision = str(message.get("trt_precision", ai_sr_trt_precision)).strip().lower()
                if ai_sr_trt_precision not in {"fp16", "int8"}:
                    ai_sr_trt_precision = "fp16"
                ai_sr_require_gpu = bool(message.get("require_gpu", ai_sr_require_gpu))
                ai_sr_frame_interval = max(1, min(60, int(message.get("inference_fps", message.get("frame_interval", ai_sr_frame_interval)))))
                ai_sr_strict = bool(message.get("strict", ai_sr_strict))
                ai_sr_input_align = max(1, int(message.get("input_align", ai_sr_input_align)))
                ai_sr_roi_overscan_percent = float(message.get("roi_overscan_percent", ai_sr_roi_overscan_percent))
                ai_sr_inference_divisor = max(0, int(message.get("inference_divisor", ai_sr_inference_divisor)))
                ai_sr_detail_preserve_percent = float(message.get("detail_preserve_percent", ai_sr_detail_preserve_percent))
                ai_sr_post_denoise_method = _normalize_ai_sr_post_denoise_method(
                    str(message.get("post_denoise_method", ai_sr_post_denoise_method))
                )
                ai_sr_post_denoise_strength = max(
                    0.0,
                    min(1.0, float(message.get("post_denoise_strength", ai_sr_post_denoise_strength))),
                )
                ai_sr_post_artifact_reduction_method = _normalize_ai_sr_post_artifact_reduction_method(
                    str(message.get("post_artifact_reduction_method", ai_sr_post_artifact_reduction_method))
                )
                ai_sr_post_artifact_reduction_strength = max(
                    0.0,
                    min(1.0, float(message.get("post_artifact_reduction_strength", ai_sr_post_artifact_reduction_strength))),
                )
                ai_sr_post_exaggeration_enabled = bool(
                    message.get("post_exaggeration_enabled", ai_sr_post_exaggeration_enabled)
                )
                ai_sr_post_exaggeration_gain = max(
                    1.0,
                    min(4.0, float(message.get("post_exaggeration_gain", ai_sr_post_exaggeration_gain))),
                )
                ai_sr_max_inflight = max(1, min(4, int(message.get("max_inflight", ai_sr_max_inflight))))
                ai_sr_error = _refresh_ai_sr_engine()
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_ai_sr_settings",
                        "ai_sr_enabled": bool(ai_sr_enabled),
                        "ai_sr_active": bool(ai_sr_engine is not None),
                        "basic_upscale_enabled": not bool(ai_sr_enabled),
                        "ai_sr_error": ai_sr_error,
                        "ai_sr_info": ai_sr_info,
                        "rtx_vsr_enabled": bool(rtx_vsr_enabled),
                        "rtx_vsr_active": bool(rtx_vsr_engine is not None and not (ai_sr_enabled and ai_sr_engine is not None)),
                        "rtx_vsr_error": rtx_vsr_error,
                        "rtx_vsr_info": rtx_vsr_info,
                    }
                )
                continue

            if command == "set_rtx_vsr_enabled":
                rtx_vsr_enabled = bool(message.get("enabled", False))
                rtx_vsr_error = _refresh_rtx_vsr_engine()
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_rtx_vsr_enabled",
                        "rtx_vsr_enabled": bool(rtx_vsr_enabled),
                        "rtx_vsr_active": bool(rtx_vsr_engine is not None and not (ai_sr_enabled and ai_sr_engine is not None)),
                        "rtx_vsr_error": rtx_vsr_error,
                        "rtx_vsr_info": rtx_vsr_info,
                    }
                )
                continue

            if command == "set_rtx_vsr_settings":
                rtx_vsr_quality = str(message.get("quality", rtx_vsr_quality)).strip().lower() or "high"
                rtx_vsr_scale = max(1, int(message.get("scale", rtx_vsr_scale)))
                rtx_vsr_post_scale_method = str(message.get("post_scale_method", rtx_vsr_post_scale_method)).strip().lower() or "bicubic"
                rtx_thdr_enabled = bool(message.get("thdr_enabled", rtx_thdr_enabled))
                rtx_thdr_contrast = max(0, int(message.get("thdr_contrast", rtx_thdr_contrast)))
                rtx_thdr_saturation = max(0, int(message.get("thdr_saturation", rtx_thdr_saturation)))
                rtx_thdr_middle_gray = max(0, int(message.get("thdr_middle_gray", rtx_thdr_middle_gray)))
                rtx_thdr_max_luminance = max(0, int(message.get("thdr_max_luminance", rtx_thdr_max_luminance)))
                rtx_vsr_error = _refresh_rtx_vsr_engine()
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_rtx_vsr_settings",
                        "rtx_vsr_enabled": bool(rtx_vsr_enabled),
                        "rtx_vsr_active": bool(rtx_vsr_engine is not None and not (ai_sr_enabled and ai_sr_engine is not None)),
                        "rtx_vsr_error": rtx_vsr_error,
                        "rtx_vsr_info": rtx_vsr_info,
                    }
                )
                continue

    except BaseException as exc:
        _stop_sessions()
        _cleanup_ai_async()
        _close_rtx_vsr_engine()
        try:
            _safe_put(
                {
                    "type": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
