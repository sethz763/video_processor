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

    # Also support pip-installed NVIDIA runtime packages in this venv.
    project_root = Path(__file__).resolve().parents[1]
    nvidia_site = project_root / "venv" / "Lib" / "site-packages" / "nvidia"
    if nvidia_site.exists():
        for pkg_dir in nvidia_site.iterdir():
            if pkg_dir.is_dir():
                _add(pkg_dir / "bin")

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
        except Exception:
            # Best effort: continue trying remaining directories.
            continue


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
_OUTPUT_SCHEDULE_STATE: dict[int, dict[str, object]] = {}
_OUTPUT_SCHEDULE_TARGET_BUFFER_FRAMES: dict[int, int] = {}


def _looks_zeroed_uyvy_frame(frame_bytes: bytes) -> bool:
    expected_size = UYVY_ROW_BYTES * FRAME_H
    if len(frame_bytes) != expected_size:
        return False

    # Sample sparsely to keep per-frame validation cheap in live mode.
    sample = np.frombuffer(frame_bytes, dtype=np.uint8)[::4096]
    return sample.size > 0 and int(np.count_nonzero(sample)) == 0

RTX_POST_SCALE_METHOD_TO_CV2_INTERP = {
    "nearest": cv2.INTER_NEAREST if cv2 is not None else 0,
    "bilinear": cv2.INTER_LINEAR if cv2 is not None else 1,
    "bicubic": cv2.INTER_CUBIC if cv2 is not None else 2,
    "lanczos": cv2.INTER_LANCZOS4 if cv2 is not None else 4,
}


def _uyvy_to_rgb_bt709_limited(yuv422: np.ndarray) -> np.ndarray:
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

    d = u - 128.0
    e = v - 128.0

    c0 = y0 - 16.0
    c1 = y1 - 16.0

    r0 = np.clip(1.164383 * c0 + 1.792741 * e, 0.0, 255.0).astype(np.uint8)
    g0 = np.clip(1.164383 * c0 - 0.213249 * d - 0.532909 * e, 0.0, 255.0).astype(np.uint8)
    b0 = np.clip(1.164383 * c0 + 2.112402 * d, 0.0, 255.0).astype(np.uint8)

    r1 = np.clip(1.164383 * c1 + 1.792741 * e, 0.0, 255.0).astype(np.uint8)
    g1 = np.clip(1.164383 * c1 - 0.213249 * d - 0.532909 * e, 0.0, 255.0).astype(np.uint8)
    b1 = np.clip(1.164383 * c1 + 2.112402 * d, 0.0, 255.0).astype(np.uint8)

    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:, 0::2, 0] = r0
    rgb[:, 0::2, 1] = g0
    rgb[:, 0::2, 2] = b0
    rgb[:, 1::2, 0] = r1
    rgb[:, 1::2, 1] = g1
    rgb[:, 1::2, 2] = b1
    return rgb


def _rgb_to_uyvy_bt709_limited(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(f"Expected RGB array shape [H, W, 3], got {tuple(rgb.shape)}")

    h, w, _ = rgb.shape
    if (w & 1) != 0:
        raise RuntimeError(f"RGB width must be even for UYVY conversion, got {w}")

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    y = np.clip(16.0 + 0.182586 * r + 0.614231 * g + 0.062007 * b, 0.0, 255.0).astype(np.uint8)
    u = np.clip(128.0 - 0.100644 * r - 0.338572 * g + 0.439216 * b, 0.0, 255.0)
    v = np.clip(128.0 + 0.439216 * r - 0.398942 * g - 0.040274 * b, 0.0, 255.0)

    y0 = y[:, 0::2]
    y1 = y[:, 1::2]
    u_pair = ((u[:, 0::2] + u[:, 1::2]) * 0.5).astype(np.uint8)
    v_pair = ((v[:, 0::2] + v[:, 1::2]) * 0.5).astype(np.uint8)

    packed = np.empty((h, w // 2, 4), dtype=np.uint8)
    packed[:, :, 0] = u_pair
    packed[:, :, 1] = y0
    packed[:, :, 2] = v_pair
    packed[:, :, 3] = y1
    return packed.reshape(h, w, 2)

def _apply_subpixel_shift_uyvy(frame_bytes: bytes, shift_x: float, shift_y: float) -> bytes:
    if cv2 is None:
        return frame_bytes
    if abs(float(shift_x)) < 1e-4 and abs(float(shift_y)) < 1e-4:
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


class AiSrOnnxEngine:
    def __init__(
        self,
        model_path: str,
        provider: str = "cpu",
        require_gpu: bool = False,
        input_align: int = 2,
        roi_overscan_percent: float = 0.0,
        inference_divisor: int = 0,
        detail_preserve_percent: float = 0.0,
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
            "do_copy_in_default_stream": "1",
            "cudnn_conv_use_max_workspace": "1",
        }
        trt_provider_options = {
            "trt_fp16_enable": "1",
            "trt_engine_cache_enable": "1",
            "trt_timing_cache_enable": "1",
        }

        if provider_name in {"trt", "tensorrt"} and not trt_available:
            raise RuntimeError(
                f"TensorrtExecutionProvider is not available in onnxruntime. Available providers: {available_providers_sorted}"
            )

        if provider_name == "cuda" and not cuda_available:
            raise RuntimeError(
                f"CUDAExecutionProvider is not available in onnxruntime. Available providers: {available_providers_sorted}"
            )

        if require_gpu and not cuda_available:
            raise RuntimeError(
                f"GPU is required for AI SR, but CUDAExecutionProvider is unavailable. Available providers: {available_providers_sorted}"
            )

        if require_gpu:
            # Enforce CUDA-only session creation when GPU is mandatory so ORT cannot
            # silently initialize a CPU session as a fallback.
            if provider_name in {"trt", "tensorrt"} and trt_available:
                providers = [
                    ("TensorrtExecutionProvider", trt_provider_options),
                    ("CUDAExecutionProvider", cuda_provider_options),
                ]
            else:
                providers = [("CUDAExecutionProvider", cuda_provider_options)]
        elif provider_name in {"trt", "tensorrt"}:
            providers = [
                ("TensorrtExecutionProvider", trt_provider_options),
                ("CUDAExecutionProvider", cuda_provider_options),
                "CPUExecutionProvider",
            ]
        elif provider_name == "auto":
            if trt_available:
                providers = [
                    ("TensorrtExecutionProvider", trt_provider_options),
                    ("CUDAExecutionProvider", cuda_provider_options),
                    "CPUExecutionProvider",
                ]
            elif cuda_available:
                providers = [("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"]
        elif provider_name == "cuda" and cuda_available:
            providers = [("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"]

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
        self._model_path = str(model_file)
        session_providers = self._session.get_providers()
        self._provider = session_providers[0] if session_providers else "CPUExecutionProvider"
        if require_gpu and self._provider not in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
            raise RuntimeError(
                f"GPU is required for AI SR, but onnxruntime session selected '{self._provider}'. "
                f"requested_providers={providers}, session_providers={session_providers}"
            )

        self._model_scale = self._detect_model_scale()
        self._input_w = max(1, FRAME_W // self._model_scale)
        self._input_h = max(1, FRAME_H // self._model_scale)
        self._avg_infer_ms: float | None = None
        self._available_providers = available_providers_sorted
        self._requested_provider = provider_name
        self._require_gpu = bool(require_gpu)
        self._input_align = max(1, int(input_align))
        self._roi_overscan_percent = max(0.0, min(100.0, float(roi_overscan_percent)))
        self._inference_divisor = max(0, int(inference_divisor))
        self._detail_preserve_percent = max(0.0, min(100.0, float(detail_preserve_percent)))

    def _effective_inference_divisor(self) -> int:
        if self._model_scale <= 1:
            return 1
        if self._inference_divisor <= 0:
            # Quality-first auto mode: avoid collapsing x4/x8 models to very small
            # inference inputs, which often looks similar to basic interpolation.
            if self._model_scale >= 8:
                return 4
            if self._model_scale >= 4:
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
            "available_providers": self._available_providers,
            "gpu_required": self._require_gpu,
            "input_dtype": "float16" if self._input_dtype == np.float16 else "float32",
            "model_scale": int(self._model_scale),
            "model_input_w": int(self._input_w),
            "model_input_h": int(self._input_h),
            "avg_infer_ms": None if self._avg_infer_ms is None else float(self._avg_infer_ms),
            "input_align": int(self._input_align),
            "roi_overscan_percent": float(self._roi_overscan_percent),
            "inference_divisor": int(self._effective_inference_divisor()),
            "detail_preserve_percent": float(self._detail_preserve_percent),
        }

    def _run_model_on_rgb(self, model_rgb: np.ndarray) -> np.ndarray:
        # Some SR models contain reshape/pixel-unshuffle paths that require specific
        # spatial alignment. Align to configured multiples before inference.
        in_h, in_w = int(model_rgb.shape[0]), int(model_rgb.shape[1])
        align = max(1, int(self._input_align))
        aligned_w = max(align, ((in_w + align - 1) // align) * align)
        aligned_h = max(align, ((in_h + align - 1) // align) * align)
        if aligned_w != in_w or aligned_h != in_h:
            model_rgb = cv2.resize(model_rgb, (aligned_w, aligned_h), interpolation=cv2.INTER_AREA)

        x = model_rgb.astype(self._input_dtype, copy=False)
        if self._input_dtype == np.float16:
            x = x * np.float16(1.0 / 255.0)
        else:
            x = x * np.float32(1.0 / 255.0)
        x = np.transpose(x, (2, 0, 1))[None, ...]

        infer_start = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: x})
        infer_ms = (time.perf_counter() - infer_start) * 1000.0
        if self._avg_infer_ms is None:
            self._avg_infer_ms = infer_ms
        else:
            self._avg_infer_ms = (0.9 * self._avg_infer_ms) + (0.1 * infer_ms)

        if not outputs:
            raise RuntimeError("AI SR model returned no outputs")

        y = outputs[0]
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        if y.ndim == 4:
            y = y[0]
        if y.ndim != 3:
            raise RuntimeError(f"Unexpected AI SR output shape: {tuple(y.shape)}")

        # Expect CHW or HWC; convert to HWC uint8 RGB.
        if y.shape[0] in (1, 3) and y.shape[-1] not in (1, 3):
            y = np.transpose(y, (1, 2, 0))

        if y.shape[2] == 1:
            y = np.repeat(y, 3, axis=2)

        if y.dtype == np.uint8:
            return y

        y = y.astype(np.float32)
        y_max = float(np.max(y)) if y.size else 0.0
        if y_max <= 1.5:
            y = np.clip(y, 0.0, 1.0)
            return (y * 255.0).astype(np.uint8)

        return np.clip(y, 0.0, 255.0).astype(np.uint8)

    def process_uyvy_frame(self, frame_bytes: bytes) -> bytes:
        if len(frame_bytes) != UYVY_ROW_BYTES * FRAME_H:
            raise RuntimeError(f"Unexpected UYVY frame size: {len(frame_bytes)}")

        yuv422 = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 2)
        rgb = _uyvy_to_rgb_bt709_limited(yuv422)

        # For x2/x4/x8 models, run inference on proportionally downscaled input so output
        # naturally returns near FRAME_W x FRAME_H instead of exploding to 4K/8K.
        model_rgb = rgb
        if self._model_scale > 1:
            divisor = self._effective_inference_divisor()
            model_in_w = max(1, (FRAME_W + divisor - 1) // divisor)
            model_in_h = max(1, (FRAME_H + divisor - 1) // divisor)
            model_rgb = cv2.resize(rgb, (model_in_w, model_in_h), interpolation=cv2.INTER_CUBIC)

        sr_rgb = self._run_model_on_rgb(model_rgb)

        if sr_rgb.shape[0] != FRAME_H or sr_rgb.shape[1] != FRAME_W:
            if sr_rgb.shape[1] > FRAME_W or sr_rgb.shape[0] > FRAME_H:
                resize_interp = cv2.INTER_LANCZOS4
            else:
                resize_interp = cv2.INTER_CUBIC
            sr_rgb = cv2.resize(sr_rgb, (FRAME_W, FRAME_H), interpolation=resize_interp)

        if self._detail_preserve_percent > 0.0:
            preserve = self._detail_preserve_percent / 100.0
            sr_rgb = cv2.addWeighted(sr_rgb, 1.0 - preserve, rgb, preserve, 0.0)

        sr_yuv422 = _rgb_to_uyvy_bt709_limited(sr_rgb)
        return sr_yuv422.tobytes()

    def process_uyvy_frame_roi(self, frame_bytes: bytes, roi: tuple[int, int, int, int]) -> bytes:
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

        yuv422 = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 2)
        roi_yuv = np.ascontiguousarray(yuv422[proc_y : proc_y + proc_h, proc_x : proc_x + proc_w, :])
        roi_rgb = _uyvy_to_rgb_bt709_limited(roi_yuv)

        model_rgb = roi_rgb
        if self._model_scale > 1:
            # Round up so AI SR does not run on a smaller-than-requested effective ROI.
            divisor = self._effective_inference_divisor()
            model_in_w = max(1, (proc_w + divisor - 1) // divisor)
            model_in_h = max(1, (proc_h + divisor - 1) // divisor)
            model_rgb = cv2.resize(roi_rgb, (model_in_w, model_in_h), interpolation=cv2.INTER_CUBIC)

        sr_roi_rgb = self._run_model_on_rgb(model_rgb)

        if sr_roi_rgb.shape[0] != proc_h or sr_roi_rgb.shape[1] != proc_w:
            if sr_roi_rgb.shape[1] > proc_w or sr_roi_rgb.shape[0] > proc_h:
                resize_interp = cv2.INTER_LANCZOS4
            else:
                resize_interp = cv2.INTER_CUBIC
            sr_roi_rgb = cv2.resize(sr_roi_rgb, (proc_w, proc_h), interpolation=resize_interp)

        if self._detail_preserve_percent > 0.0:
            preserve = self._detail_preserve_percent / 100.0
            sr_roi_rgb = cv2.addWeighted(sr_roi_rgb, 1.0 - preserve, roi_rgb, preserve, 0.0)

        sr_roi_yuv = _rgb_to_uyvy_bt709_limited(sr_roi_rgb)
        yuv_out = np.array(yuv422, copy=True)
        yuv_out[proc_y : proc_y + proc_h, proc_x : proc_x + proc_w, :] = sr_roi_yuv
        return yuv_out.tobytes()

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

        yuv422 = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 2)
        roi_yuv = np.ascontiguousarray(yuv422[proc_y : proc_y + proc_h, proc_x : proc_x + proc_w, :])
        roi_rgb = _uyvy_to_rgb_bt709_limited(roi_yuv)

        target_w = FRAME_W
        target_h = FRAME_H
        method_name = str(method).strip().lower()
        upscale_interp = cv2.INTER_LANCZOS4
        if method_name in {"bilinear", "bilinear_sharp"}:
            upscale_interp = cv2.INTER_LINEAR
        elif method_name in {"bicubic", "bicubic_sharpen"}:
            upscale_interp = cv2.INTER_CUBIC

        # Build a baseline zoom from the source ROI so optional detail preserve
        # can retain source edge character while still letting AI drive output.
        baseline_rgb = cv2.resize(roi_rgb, (target_w, target_h), interpolation=upscale_interp)

        model_rgb = roi_rgb
        if self._model_scale > 1:
            divisor = self._effective_inference_divisor()
            model_in_w = max(1, (proc_w + divisor - 1) // divisor)
            model_in_h = max(1, (proc_h + divisor - 1) // divisor)
            model_rgb = cv2.resize(roi_rgb, (model_in_w, model_in_h), interpolation=cv2.INTER_CUBIC)

        sr_rgb = self._run_model_on_rgb(model_rgb)

        if sr_rgb.shape[0] != target_h or sr_rgb.shape[1] != target_w:
            sr_rgb = cv2.resize(sr_rgb, (target_w, target_h), interpolation=upscale_interp)

        if method_name == "bicubic_sharpen":
            sr_rgb = cv2.addWeighted(sr_rgb, 1.35, cv2.GaussianBlur(sr_rgb, (0, 0), 1.0), -0.35, 0)
        elif method_name == "bilinear_sharp":
            sr_rgb = cv2.addWeighted(sr_rgb, 1.20, cv2.GaussianBlur(sr_rgb, (0, 0), 0.8), -0.20, 0)

        if self._detail_preserve_percent > 0.0:
            preserve = self._detail_preserve_percent / 100.0
            sr_rgb = cv2.addWeighted(sr_rgb, 1.0 - preserve, baseline_rgb, preserve, 0.0)

        sr_yuv422 = _rgb_to_uyvy_bt709_limited(sr_rgb)
        return sr_yuv422.tobytes()


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

    out = bytearray(UYVY_ROW_BYTES * FRAME_H)
    for y in range(FRAME_H):
        src_start = y * row_bytes
        src_end = src_start + UYVY_ROW_BYTES
        dst_start = y * UYVY_ROW_BYTES
        out[dst_start : dst_start + UYVY_ROW_BYTES] = raw[src_start:src_end]
    return bytes(out)


def _write_frame_to_output(out: object, frame_bytes: bytes) -> bool:
    if out.row_bytes == UYVY_ROW_BYTES:
        payload = frame_bytes
    else:
        if out.row_bytes < UYVY_ROW_BYTES:
            raise RuntimeError(f"Output row_bytes {out.row_bytes} is smaller than expected {UYVY_ROW_BYTES}")

        padded = np.zeros((FRAME_H, int(out.row_bytes)), dtype=np.uint8)
        src = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(FRAME_H, UYVY_ROW_BYTES)
        padded[:, :UYVY_ROW_BYTES] = src
        payload = padded.tobytes()

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
        }
        _OUTPUT_SCHEDULE_STATE[out_id] = state

    now = time.perf_counter()
    frame_period_s = float(state.get("frame_period_s", 0.0))
    jitter_tolerance_s = min(0.0015, frame_period_s * 0.35) if frame_period_s > 0.0 else 0.001

    if bool(state.get("enabled", False)):
        frame_duration = int(state.get("frame_duration", 0))
        time_scale = int(state.get("time_scale", 0))
        if frame_duration > 0 and time_scale > 0:
            try:
                epoch_ts = float(state.get("schedule_epoch_perf_ts", 0.0))
                if epoch_ts <= 0.0:
                    epoch_ts = now
                    state["schedule_epoch_perf_ts"] = epoch_ts

                elapsed_units = max(0.0, (now - epoch_ts) * float(time_scale))
                display_time = int(state.get("display_time", 0))

                # Keep scheduled preroll bounded so enqueue rate cannot run above the mode frame rate.
                max_preroll_frames = max(0, min(10, int(state.get("target_buffer_frames", 2))))
                max_preroll_units = max_preroll_frames * frame_duration
                if display_time > (elapsed_units + max_preroll_units):
                    return False

                out.schedule_frame_copy(
                    payload,
                    display_time,
                    frame_duration,
                    time_scale,
                )
                state["display_time"] = int(state.get("display_time", 0)) + frame_duration

                if not bool(state.get("started", False)):
                    state["queued_before_start"] = int(state.get("queued_before_start", 0)) + 1
                    should_start = False
                    target_start_frames = max(0, min(10, int(state.get("target_buffer_frames", 2))))
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
                return True
            except Exception:
                # Fall back to blocking output if scheduling path errors at runtime.
                state["enabled"] = False

    next_emit_ts = float(state.get("sync_next_emit_ts", 0.0))
    if frame_period_s > 0.0:
        if next_emit_ts <= 0.0:
            next_emit_ts = now
        elif now + jitter_tolerance_s < next_emit_ts:
            return False

        # If processing was delayed for multiple frame intervals, re-anchor pacing to "now".
        if now - next_emit_ts > (frame_period_s * 3.0):
            next_emit_ts = now

    out.display_frame_sync(payload)
    if frame_period_s > 0.0:
        state["sync_next_emit_ts"] = next_emit_ts + frame_period_s
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


def _reprime_output_schedule(out: object) -> None:
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
    preprocess_bytes: bytes | None = None
    output_bytes: bytes | None = None
    ai_applied: bool = False
    rtx_applied: bool = False
    native_shift_applied: bool = False


def run_processor_worker(request_queue, response_queue, startup_config: dict[str, Any]) -> None:
    _FRAME_MESSAGE_TYPES = {"frame", "decklink_frame", "decklink_no_frame"}
    _CONTROL_MESSAGE_TYPES = {"ready", "ack", "warning", "error"}

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
    output_thread: threading.Thread | None = None
    q_capture_to_preprocess: queue.Queue[_StageFrame] | None = None
    q_preprocess_to_upscale: queue.Queue[_StageFrame] | None = None
    q_upscale_to_output: queue.Queue[_StageFrame] | None = None
    frame_id_counter = 0
    capture_drop_count = 0
    preprocess_drop_count = 0
    upscale_drop_count = 0
    latest_input_frame: bytes | None = None
    latest_output_frame: bytes | None = None
    latest_effective_sr_scale = 1
    latest_rtx_vsr_applied = False
    latest_rtx_effect_mean_abs_luma = 0.0
    rtx_effect_sample_counter = 0
    processed_frame_counter = 0
    started_perf_ts = 0.0
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
    ai_sr_enabled = bool(startup_config.get("ai_sr_enabled", False))
    ai_sr_model_path = str(startup_config.get("ai_sr_model_path", ""))
    ai_sr_provider = str(startup_config.get("ai_sr_provider", "cuda"))
    ai_sr_require_gpu = bool(startup_config.get("ai_sr_require_gpu", False))
    ai_sr_frame_interval = max(1, int(startup_config.get("ai_sr_frame_interval", 2)))
    ai_sr_strict = bool(startup_config.get("ai_sr_strict", False))
    ai_sr_input_align = max(1, int(startup_config.get("ai_sr_input_align", 2)))
    ai_sr_roi_overscan_percent = float(startup_config.get("ai_sr_roi_overscan_percent", 0.0))
    ai_sr_inference_divisor = max(0, int(startup_config.get("ai_sr_inference_divisor", 0)))
    ai_sr_detail_preserve_percent = float(startup_config.get("ai_sr_detail_preserve_percent", 0.0))
    ai_sr_runtime_note: str | None = None
    ai_sr_engine: AiSrOnnxEngine | None = None
    ai_sr_info: dict[str, object] | None = None
    ai_sr_frame_counter = 0
    ai_sr_latest_output_frame: bytes | None = None
    ai_sr_executor: ThreadPoolExecutor | None = None
    ai_sr_future: Future[bytes] | None = None
    ai_sr_dropped_frames = 0
    ai_sr_applied_frames = 0
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
    roi_microstep_transition: dict[str, object] | None = None
    current_deinterlace_enabled = bool(startup_config.get("deinterlace_enabled", True))
    current_deinterlace_method = str(startup_config.get("deinterlace_method", "bob"))
    current_denoise_method = str(startup_config.get("denoise_method", "off"))
    current_denoise_strength = max(0.0, min(1.0, float(startup_config.get("denoise_strength", 0.35))))
    current_output_buffer_frames = max(0, min(10, int(startup_config.get("decklink_output_buffer_frames", 2))))
    basic_scaling_enabled = bool(startup_config.get("enable_basic_scaling", startup_config.get("enable_placeholder_sr", True)))
    state_lock = threading.Lock()

    def _is_live_passthrough_mode() -> bool:
        # Passthrough mode is valid only when no stage is expected to modify pixels.
        denoise_enabled = current_denoise_method not in {"off", "none"} and current_denoise_strength > 0.001
        return (
            (not current_deinterlace_enabled)
            and (not denoise_enabled)
            and (not basic_scaling_enabled)
            and (not ai_sr_enabled)
            and (not rtx_vsr_enabled)
        )

    def _is_live_basic_scaling_fast_mode() -> bool:
        # Basic scaling fast mode keeps processing in capture thread when the
        # pipeline is native-only (no Python AI/RTX/preprocess stages).
        denoise_enabled = current_denoise_method not in {"off", "none"} and current_denoise_strength > 0.001
        return (
            (not current_deinterlace_enabled)
            and (not denoise_enabled)
            and basic_scaling_enabled
            and (not ai_sr_enabled)
            and (not rtx_vsr_enabled)
        )

    def _step_smoothed_roi_shift() -> tuple[float, float]:
        nonlocal roi_shift_applied_x, roi_shift_applied_y
        nonlocal roi_shift_velocity_x, roi_shift_velocity_y
        nonlocal roi_shift_target_lpf_x, roi_shift_target_lpf_y
        nonlocal roi_shift_accel_x, roi_shift_accel_y
        nonlocal roi_microstep_transition

        # During active keyframe transitions, apply the exact per-frame shift
        # target to remove follower lag and achieve dolly-like smoothness.
        if roi_microstep_transition is not None:
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
    ) -> None:
        nonlocal roi_microstep_transition

        s_x, s_y, s_w, s_h = _normalize_worker_roi(*start_roi)
        t_x, t_y, t_w, t_h = _normalize_worker_roi(*target_roi)
        total_frames = max(1, min(600, int(duration_frames)))
        mode_name = str(interpolation_mode).strip().lower()
        if mode_name not in {"linear", "ease_in_out", "ease_out"}:
            mode_name = "linear"

        roi_microstep_transition = {
            "start": (s_x, s_y, s_w, s_h),
            "target": (t_x, t_y, t_w, t_h),
            "total_frames": total_frames,
            "frame_progress": 0,
            "interpolation_mode": mode_name,
            "residual": {"x": 0.0, "y": 0.0, "w": 0.0},
            "last_roi": (s_x, s_y, s_w, s_h),
            "overscan_percent": max(0.0, float(overscan_percent)),
        }

    def _cancel_roi_microstep_transition(reset_shift: bool = True) -> None:
        nonlocal roi_microstep_transition
        roi_microstep_transition = None
        if reset_shift:
            _set_roi_shift_target(0.0, 0.0)

    def _apply_manual_roi_with_subpixel_compensation(
        req_x: int,
        req_y: int,
        req_w: int,
        req_h: int,
    ) -> None:
        nonlocal current_roi_x, current_roi_y, current_roi_w, current_roi_h, rtx_vsr_error

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
        _set_roi_shift_immediate(shift_x, shift_y)

    def _advance_roi_microstep_transition_one_frame() -> None:
        nonlocal current_roi_x, current_roi_y, current_roi_w, current_roi_h, roi_microstep_transition

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

        frame_progress = min(total_frames, int(state.get("frame_progress", 0)) + 1)
        state["frame_progress"] = frame_progress

        t = float(frame_progress) / float(max(1, total_frames))
        curved_t = _apply_roi_curve(t, mode_name)

        s_x, s_y, s_w, s_h = [float(v) for v in start_roi]
        t_x, t_y, t_w, t_h = [float(v) for v in target_roi]

        start_cx = s_x + (s_w * 0.5)
        start_cy = s_y + (s_h * 0.5)
        target_cx = t_x + (t_w * 0.5)
        target_cy = t_y + (t_h * 0.5)

        ideal_cx = start_cx + ((target_cx - start_cx) * curved_t)
        ideal_cy = start_cy + ((target_cy - start_cy) * curved_t)
        ideal_w = s_w + ((t_w - s_w) * curved_t)

        desired_cx = ideal_cx + float(residual.get("x", 0.0))
        desired_cy = ideal_cy + float(residual.get("y", 0.0))
        desired_w = ideal_w + float(residual.get("w", 0.0))

        target_scale = FRAME_W / max(1.0, t_w)
        overscan_pct = float(state.get("overscan_percent", 0.0))
        if target_scale >= 4.0 and overscan_pct > 0.0:
            overscan_weight = max(0.0, 1.0 - curved_t)
            desired_w_backend = desired_w * (1.0 + ((overscan_pct / 100.0) * overscan_weight))
        else:
            desired_w_backend = desired_w

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

        transition_complete = frame_progress >= total_frames or (
            current_roi_x == int(target_roi[0])
            and current_roi_y == int(target_roi[1])
            and current_roi_w == int(target_roi[2])
            and current_roi_h == int(target_roi[3])
        )
        if transition_complete:
            current_roi_x, current_roi_y, current_roi_w, current_roi_h = _normalize_worker_roi(
                int(target_roi[0]), int(target_roi[1]), int(target_roi[2]), int(target_roi[3])
            )
            if (
                current_roi_x != i_x
                or current_roi_y != i_y
                or current_roi_w != i_w
                or current_roi_h != i_h
            ):
                processor.set_roi(current_roi_x, current_roi_y, current_roi_w, current_roi_h)
            _set_roi_shift_target(0.0, 0.0)
            roi_microstep_transition = None

    def _cleanup_ai_async() -> None:
        nonlocal ai_sr_executor, ai_sr_future
        if ai_sr_future is not None:
            ai_sr_future.cancel()
            ai_sr_future = None
        if ai_sr_executor is not None:
            ai_sr_executor.shutdown(wait=False, cancel_futures=True)
            ai_sr_executor = None

    def _collect_ai_future_result() -> None:
        nonlocal ai_sr_latest_output_frame, ai_sr_future
        if ai_sr_future is not None and ai_sr_future.done():
            try:
                ai_sr_latest_output_frame = ai_sr_future.result()
            except Exception as ai_exc:
                _safe_put({"type": "warning", "warning": f"AI SR inference failed: {ai_exc}"})
            finally:
                ai_sr_future = None

    def _ai_inference_busy() -> bool:
        return ai_sr_future is not None and not ai_sr_future.done()

    def _apply_ai_sr_non_blocking(frame_bytes: bytes, roi: tuple[int, int, int, int], method: str) -> tuple[bytes, bool]:
        nonlocal ai_sr_frame_counter, ai_sr_latest_output_frame, ai_sr_future
        if ai_sr_engine is None:
            return frame_bytes, True

        _collect_ai_future_result()

        ai_sr_frame_counter += 1
        run_ai_inference = (ai_sr_frame_counter % ai_sr_frame_interval == 0)
        if run_ai_inference and ai_sr_future is None and ai_sr_executor is not None:
            ai_input_bytes = bytes(frame_bytes)
            ai_roi = tuple(roi)
            ai_method = str(method)
            ai_sr_future = ai_sr_executor.submit(ai_sr_engine.process_uyvy_frame_roi_to_output, ai_input_bytes, ai_roi, ai_method)

        # Keep showing the latest completed AI frame until the next one is ready.
        # This makes AI SR effect persistent on live output rather than one-frame pulses.
        if ai_sr_latest_output_frame is not None:
            return ai_sr_latest_output_frame, True
        return frame_bytes, False

    def _apply_ai_sr(frame_bytes: bytes, roi: tuple[int, int, int, int], method: str) -> tuple[bytes, bool]:
        nonlocal ai_sr_frame_counter, ai_sr_applied_frames, ai_sr_passthrough_frames
        if ai_sr_engine is None:
            return frame_bytes, False

        if ai_sr_strict:
            ai_sr_frame_counter += 1
            run_ai_inference = (ai_sr_frame_counter % ai_sr_frame_interval == 0)
            if not run_ai_inference:
                ai_sr_passthrough_frames += 1
                return frame_bytes, False
            try:
                out = ai_sr_engine.process_uyvy_frame_roi_to_output(frame_bytes, roi, method)
                ai_sr_applied_frames += 1
                return out, True
            except Exception as ai_exc:
                ai_sr_passthrough_frames += 1
                _safe_put({"type": "warning", "warning": f"AI SR strict inference failed: {ai_exc}"})
                return frame_bytes, False

        output_frame, ai_applied = _apply_ai_sr_non_blocking(frame_bytes, roi, method)
        if ai_applied:
            ai_sr_applied_frames += 1
        else:
            ai_sr_passthrough_frames += 1
        return output_frame, ai_applied

    def _apply_rtx_vsr(frame_bytes: bytes, roi: tuple[int, int, int, int]) -> bytes:
        if rtx_vsr_engine is None:
            return frame_bytes

        roi_x, roi_y, roi_w, roi_h = _normalize_worker_roi(int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
        yuv422 = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 2)
        roi_yuv = np.ascontiguousarray(yuv422[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w, :])
        roi_rgb = _uyvy_to_rgb_bt709_limited(roi_yuv)
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
        sr_yuv422 = _rgb_to_uyvy_bt709_limited(sr_rgb)
        return sr_yuv422.tobytes()

    def _apply_ai_sr_performance_profile() -> None:
        nonlocal ai_sr_strict, ai_sr_frame_interval, ai_sr_inference_divisor, ai_sr_runtime_note
        ai_sr_runtime_note = None

        model_name = Path(ai_sr_model_path).name.lower()
        profile_changes: list[str] = []

        # Quality-first throughput policy for heavy models:
        # 1) Never force lower inference resolution automatically.
        # 2) Keep async submission so output cadence stays responsive.
        # 3) Use only a mild cadence adjustment for very heavy models.
        if "x4_fp16" in model_name or "x8" in model_name:
            if ai_sr_strict:
                ai_sr_strict = False
                profile_changes.append("strict->async")

            # Keep high per-frame detail by avoiding forced divisor changes.
            # If the user explicitly set a divisor, preserve it as-is.

            target_interval = 8 if "x8" in model_name else 6
            if ai_sr_frame_interval < target_interval:
                ai_sr_frame_interval = target_interval
                profile_changes.append(f"frame_interval={target_interval}")

            if ai_sr_inference_divisor > 0:
                profile_changes.append(f"inference_divisor=user:{ai_sr_inference_divisor}")
            else:
                profile_changes.append("inference_divisor=auto-quality")

        if profile_changes:
            ai_sr_runtime_note = "; ".join(profile_changes)

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

    def _denoise_enabled() -> bool:
        return current_denoise_method not in {"off", "none"} and current_denoise_strength > 0.001

    def _is_valid_uyvy_frame(frame_bytes: bytes) -> bool:
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

    def _basic_scaling_enabled() -> bool:
        return bool(basic_scaling_enabled)

    def _preprocess_stage(frame_bytes: bytes) -> tuple[bytes, bool]:
        nonlocal preprocess_noop_warning_emitted
        if not _is_preprocess_enabled():
            return frame_bytes, False

        native_out = frame_bytes
        if hasattr(processor, "process_frame_preprocess_only"):
            native_out = processor.process_frame_preprocess_only(frame_bytes)
        elif current_deinterlace_enabled and hasattr(processor, "process_frame_deinterlace_only"):
            native_out = processor.process_frame_deinterlace_only(frame_bytes)
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
    ) -> tuple[bytes, bool, bool]:
        nonlocal zeroed_output_warning_emitted
        if not _basic_scaling_enabled():
            return frame_bytes, False, False

        native_shift_applied = False
        if _set_native_subpixel_shift(shift_x, shift_y):
            native_shift_applied = abs(float(shift_x)) > 1e-4 or abs(float(shift_y)) > 1e-4

        if preprocess_already_applied and hasattr(processor, "process_frame_no_deinterlace"):
            scaled = processor.process_frame_no_deinterlace(frame_bytes)
        else:
            scaled = processor.process_frame(frame_bytes)

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
            return frame_bytes, False, False

        return scaled, True, native_shift_applied

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
        # When native basic scaling is active, preprocess is fused into that
        # stage so effects run in the post-ROI/pre-upscale slot.
        if _is_preprocess_enabled() and not _basic_scaling_enabled():
            stack.append("preprocess")
        if ai_sr_enabled and ai_sr_engine is not None:
            stack.append("ai_sr")
        if rtx_vsr_enabled and rtx_vsr_engine is not None:
            stack.append("rtx_vsr")
        if _basic_scaling_enabled():
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
                working, basic_applied, native_shift_applied = _apply_basic_scaling_stage(
                    working,
                    preprocess_applied,
                    shift_x,
                    shift_y,
                )
                continue

        return working, preprocess_applied, basic_applied, ai_applied, rtx_applied, native_shift_applied

    def _stop_live_pipeline() -> None:
        nonlocal pipeline_running, capture_thread, preprocess_thread, upscale_thread, output_thread
        if not pipeline_running:
            return
        pipeline_stop_event.set()
        for thread in (capture_thread, upscale_thread, output_thread):
            if thread is not None:
                thread.join(timeout=1.0)
        capture_thread = None
        preprocess_thread = None
        upscale_thread = None
        output_thread = None
        pipeline_running = False

    def _start_live_pipeline() -> None:
        nonlocal pipeline_running
        nonlocal q_capture_to_preprocess, q_preprocess_to_upscale, q_upscale_to_output
        nonlocal frame_id_counter, capture_drop_count, preprocess_drop_count, upscale_drop_count
        nonlocal capture_thread, preprocess_thread, upscale_thread, output_thread
        nonlocal latest_input_frame, latest_output_frame, latest_effective_sr_scale, processed_frame_counter, started_perf_ts
        nonlocal latest_rtx_vsr_applied, latest_rtx_effect_mean_abs_luma
        nonlocal zeroed_output_warning_emitted, preprocess_noop_warning_emitted, native_subpixel_warning_emitted
        nonlocal rtx_effect_sample_counter
        nonlocal stage_preprocess_applied_frames, stage_basic_applied_frames, stage_ai_applied_frames, stage_rtx_applied_frames, stage_passthrough_frames
        nonlocal last_stage_preprocess_applied, last_stage_basic_applied, last_stage_ai_applied, last_stage_rtx_applied
        nonlocal last_stage_stack
        if capture_session is None or output_session is None:
            raise RuntimeError("Cannot start pipeline without active DeckLink sessions")

        _stop_live_pipeline()
        pipeline_stop_event.clear()
        q_capture_to_preprocess = queue.Queue(maxsize=2)
        q_preprocess_to_upscale = None
        q_upscale_to_output = queue.Queue(maxsize=1)
        frame_id_counter = 0
        capture_drop_count = 0
        preprocess_drop_count = 0
        upscale_drop_count = 0
        latest_input_frame = None
        latest_output_frame = None
        latest_effective_sr_scale = 1
        latest_rtx_vsr_applied = False
        latest_rtx_effect_mean_abs_luma = 0.0
        rtx_effect_sample_counter = 0
        processed_frame_counter = 0
        started_perf_ts = time.perf_counter()
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
        preprocess_noop_warning_emitted = False
        native_subpixel_warning_emitted = False

        def _capture_worker() -> None:
            nonlocal frame_id_counter, capture_drop_count
            nonlocal latest_input_frame, latest_output_frame, latest_effective_sr_scale, processed_frame_counter
            nonlocal latest_rtx_vsr_applied, latest_rtx_effect_mean_abs_luma
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
                try:
                    input_bytes = _tight_uyvy_bytes(frame)
                except Exception as exc:
                    _safe_put({"type": "warning", "warning": f"Capture frame conversion failed: {exc}"})
                    continue

                # Keep input preview live even when processing is backlogged.
                with state_lock:
                    latest_input_frame = input_bytes

                if _is_live_passthrough_mode():
                    # In zero-processing mode, avoid staged queueing and preserve output cadence.
                    output_bytes = input_bytes
                    shift_x, shift_y = _step_smoothed_roi_shift()
                    native_shift_applied = False
                    if _set_native_subpixel_shift(shift_x, shift_y):
                        native_shift_applied = abs(shift_x) > 1e-4 or abs(shift_y) > 1e-4
                    if native_shift_applied and hasattr(processor, "process_frame_no_deinterlace"):
                        output_bytes = processor.process_frame_no_deinterlace(output_bytes)
                    elif abs(shift_x) > 1e-4 or abs(shift_y) > 1e-4:
                        output_bytes = _apply_subpixel_shift_uyvy(output_bytes, shift_x, shift_y)
                    try:
                        if output_session is not None:
                            emitted = _write_frame_to_output(output_session, output_bytes)
                        else:
                            emitted = False
                    except Exception as exc:
                        _safe_put({"type": "warning", "warning": f"Output stage failed: {exc}"})
                        continue

                    with state_lock:
                        latest_input_frame = input_bytes
                        latest_output_frame = output_bytes
                        latest_effective_sr_scale = 1
                        latest_rtx_vsr_applied = False
                        latest_rtx_effect_mean_abs_luma = 0.0
                        if emitted:
                            processed_frame_counter += 1
                    if emitted:
                        _advance_roi_microstep_transition_one_frame()
                    continue

                if _is_live_basic_scaling_fast_mode():
                    # Keep basic-scaling-only path off the staged queue graph to
                    # reduce Python scheduling overhead at 1080p60.
                    shift_x, shift_y = _step_smoothed_roi_shift()
                    try:
                        output_bytes, basic_applied, native_shift_applied = _apply_basic_scaling_stage(
                            input_bytes,
                            False,
                            shift_x,
                            shift_y,
                        )
                    except Exception as exc:
                        _safe_put({"type": "warning", "warning": f"Basic scaling fast path failed: {exc}"})
                        continue

                    if (not native_shift_applied) and (abs(shift_x) > 1e-4 or abs(shift_y) > 1e-4):
                        output_bytes = _apply_subpixel_shift_uyvy(output_bytes, shift_x, shift_y)

                    try:
                        if output_session is not None:
                            emitted = _write_frame_to_output(output_session, output_bytes)
                        else:
                            emitted = False
                    except Exception as exc:
                        _safe_put({"type": "warning", "warning": f"Output stage failed: {exc}"})
                        continue

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
                        _advance_roi_microstep_transition_one_frame()
                    continue

                frame_id_counter += 1
                item = _StageFrame(frame_id=frame_id_counter, captured_ts=time.perf_counter(), input_bytes=input_bytes)
                if _put_latest_stage_frame(q_capture_to_preprocess, item):
                    capture_drop_count += 1

        def _upscale_worker() -> None:
            nonlocal upscale_drop_count, ai_sr_dropped_frames, preprocess_drop_count
            nonlocal stage_preprocess_applied_frames, stage_basic_applied_frames, stage_ai_applied_frames, stage_rtx_applied_frames, stage_passthrough_frames
            nonlocal last_stage_preprocess_applied, last_stage_basic_applied, last_stage_ai_applied, last_stage_rtx_applied
            nonlocal last_stage_stack
            assert q_capture_to_preprocess is not None
            assert q_upscale_to_output is not None
            while not pipeline_stop_event.is_set():
                try:
                    item = q_capture_to_preprocess.get(timeout=0.01)
                except queue.Empty:
                    continue

                try:
                    item.preprocess_bytes = item.input_bytes
                except Exception as exc:
                    preprocess_drop_count += 1
                    _safe_put({"type": "warning", "warning": f"Preprocess stage failed: {exc}"})
                    continue

                preprocessed = item.preprocess_bytes if item.preprocess_bytes is not None else item.input_bytes
                try:
                    shift_x, shift_y = _step_smoothed_roi_shift()
                    output_bytes, preprocess_applied, basic_applied, ai_applied, rtx_applied, native_shift_applied = _process_pipeline_frame(
                        preprocessed,
                        shift_x,
                        shift_y,
                    )
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
                item.ai_applied = bool(ai_applied)
                item.rtx_applied = bool(rtx_applied)
                item.native_shift_applied = bool(native_shift_applied)
                if _put_latest_stage_frame(q_upscale_to_output, item):
                    upscale_drop_count += 1

        def _output_worker() -> None:
            nonlocal latest_input_frame, latest_output_frame, latest_effective_sr_scale, processed_frame_counter
            nonlocal latest_rtx_vsr_applied, latest_rtx_effect_mean_abs_luma
            nonlocal rtx_effect_sample_counter
            assert q_upscale_to_output is not None
            while not pipeline_stop_event.is_set():
                try:
                    item = q_upscale_to_output.get(timeout=0.01)
                except queue.Empty:
                    continue

                output_bytes = item.output_bytes if item.output_bytes is not None else item.input_bytes
                shift_x, shift_y = _step_smoothed_roi_shift()
                if (not item.native_shift_applied) and (abs(shift_x) > 1e-4 or abs(shift_y) > 1e-4):
                    output_bytes = _apply_subpixel_shift_uyvy(output_bytes, shift_x, shift_y)
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

                try:
                    if output_session is not None:
                        emitted = _write_frame_to_output(output_session, output_bytes)
                    else:
                        emitted = False
                except Exception as exc:
                    _safe_put({"type": "warning", "warning": f"Output stage failed: {exc}"})
                    continue

                with state_lock:
                    latest_input_frame = item.input_bytes
                    latest_output_frame = output_bytes
                    latest_effective_sr_scale = int(processor.get_effective_sr_scale())
                    latest_rtx_vsr_applied = bool(item.rtx_applied)
                    latest_rtx_effect_mean_abs_luma = sampled_delta
                    if emitted:
                        processed_frame_counter += 1
                if emitted:
                    _advance_roi_microstep_transition_one_frame()

        capture_thread = threading.Thread(target=_capture_worker, name="vp-capture", daemon=True)
        preprocess_thread = None
        upscale_thread = threading.Thread(target=_upscale_worker, name="vp-upscale", daemon=True)
        output_thread = threading.Thread(target=_output_worker, name="vp-output", daemon=True)

        capture_thread.start()
        upscale_thread.start()
        output_thread.start()
        pipeline_running = True

    def _refresh_ai_sr_engine() -> str | None:
        nonlocal ai_sr_engine, ai_sr_info, ai_sr_frame_counter, ai_sr_latest_output_frame, ai_sr_executor, ai_sr_future, ai_sr_dropped_frames, ai_sr_applied_frames, ai_sr_passthrough_frames, ai_sr_runtime_note
        _cleanup_ai_async()

        if not ai_sr_enabled:
            ai_sr_engine = None
            ai_sr_info = None
            ai_sr_frame_counter = 0
            ai_sr_latest_output_frame = None
            ai_sr_dropped_frames = 0
            ai_sr_applied_frames = 0
            ai_sr_passthrough_frames = 0
            return None

        try:
            _apply_ai_sr_performance_profile()

            ai_sr_engine = AiSrOnnxEngine(
                ai_sr_model_path,
                provider=ai_sr_provider,
                require_gpu=ai_sr_require_gpu,
                input_align=ai_sr_input_align,
                roi_overscan_percent=ai_sr_roi_overscan_percent,
                inference_divisor=ai_sr_inference_divisor,
                detail_preserve_percent=ai_sr_detail_preserve_percent,
            )
            ai_sr_info = ai_sr_engine.info()
            ai_sr_info["strict_mode"] = bool(ai_sr_strict)
            ai_sr_info["async_mode"] = not bool(ai_sr_strict)
            ai_sr_info["frame_interval"] = int(ai_sr_frame_interval)
            ai_sr_info["discard_while_busy"] = False
            ai_sr_info["requested_provider"] = str(ai_sr_provider)
            ai_sr_info["gpu_required"] = bool(ai_sr_require_gpu)
            ai_sr_info["runtime_profile_note"] = ai_sr_runtime_note
            ai_sr_frame_counter = 0
            ai_sr_latest_output_frame = None
            ai_sr_dropped_frames = 0
            ai_sr_applied_frames = 0
            ai_sr_passthrough_frames = 0
            ai_sr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai-sr")
            ai_sr_future = None
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
            ai_sr_dropped_frames = 0
            ai_sr_applied_frames = 0
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
        nonlocal capture_session, output_session, current_output_buffer_frames
        if d is None:
            raise RuntimeError("decklink_wrapper is not available in worker process")

        _stop_sessions()
        current_output_buffer_frames = max(
            0,
            min(10, int(message.get("decklink_output_buffer_frames", current_output_buffer_frames))),
        )

        capture_session = d.CaptureSession(
            device_index=int(message["in_device"]),
            display_mode=message["in_mode"],
            pixel_format=d.PIXEL_FORMAT_8BIT_YUV,
            max_queue_frames=8,
            enable_format_detection=bool(message["enable_format_detection"]),
        )
        output_session = d.OutputSession(
            device_index=int(message["out_device"]),
            display_mode=message["out_mode"],
            pixel_format=d.PIXEL_FORMAT_8BIT_YUV,
        )

        capture_session.start()
        output_session.start()
        _set_output_schedule_buffer_frames(output_session, current_output_buffer_frames)

        _start_live_pipeline()
    try:
        project_root = Path(startup_config["project_root"])
        module = _load_video_processor_module(project_root)
        processor, basic_scaling_method_supported = _create_processor(module, startup_config)
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
            if command == "shutdown":
                _stop_sessions()
                _cleanup_ai_async()
                _close_rtx_vsr_engine()
                return

            if command == "start_decklink":
                _start_sessions(message)
                _safe_put({"type": "ack", "cmd": "start_decklink"})
                continue

            if command == "set_decklink_output_buffer_frames":
                current_output_buffer_frames = max(
                    0,
                    min(10, int(message.get("decklink_output_buffer_frames", current_output_buffer_frames))),
                )
                if output_session is not None:
                    _set_output_schedule_buffer_frames(output_session, current_output_buffer_frames)
                    _reprime_output_schedule(output_session)
                _safe_put(
                    {
                        "type": "ack",
                        "cmd": "set_decklink_output_buffer_frames",
                        "decklink_output_buffer_frames": int(current_output_buffer_frames),
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

                if current_input is None or current_output is None:
                    _safe_put({"type": "decklink_no_frame", "reason": "no_input_signal"})
                    continue

                elapsed = max(0.0001, time.perf_counter() - started_perf_ts)
                processed_fps = float(current_counter) / elapsed
                stage_depths = {
                    "capture_to_preprocess": 0 if q_capture_to_preprocess is None else q_capture_to_preprocess.qsize(),
                    "preprocess_to_upscale": 0 if q_preprocess_to_upscale is None else q_preprocess_to_upscale.qsize(),
                    "upscale_to_output": 0 if q_upscale_to_output is None else q_upscale_to_output.qsize(),
                }
                payload: dict[str, object] = {
                    "type": "decklink_frame",
                    "effective_sr_scale": current_scale,
                    "processed_frame_counter": current_counter,
                    "processed_fps": processed_fps,
                    "ai_sr_applied_frames": int(ai_sr_applied_frames),
                    "ai_sr_passthrough_frames": int(ai_sr_passthrough_frames),
                    "rtx_vsr_applied": current_rtx_applied,
                    "rtx_effect_mean_abs_luma": current_rtx_delta,
                    "stage_enable_flags": {
                        "preprocess": bool(_is_preprocess_enabled()),
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
                }
                if include_frames:
                    payload["input_frame_bytes"] = current_input
                    payload["output_frame_bytes"] = current_output
                _safe_put(payload)
                continue

            if command == "process_frame":
                frame_id = int(message["frame_id"])
                frame_bytes = message["frame_bytes"]
                _advance_roi_microstep_transition_one_frame()
                shift_x, shift_y = _step_smoothed_roi_shift()
                output_bytes, _, basic_applied, ai_applied, rtx_applied, native_shift_applied = _process_pipeline_frame(
                    frame_bytes,
                    shift_x,
                    shift_y,
                )
                if (not native_shift_applied) and (abs(shift_x) > 1e-4 or abs(shift_y) > 1e-4):
                    output_bytes = _apply_subpixel_shift_uyvy(output_bytes, shift_x, shift_y)

                if ai_sr_engine is not None and not ai_applied and _ai_inference_busy():
                    ai_sr_dropped_frames += 1

                latest_output_frame = output_bytes
                last_stage_basic_applied = bool(basic_applied)
                last_stage_ai_applied = bool(ai_applied)
                last_stage_rtx_applied = bool(rtx_applied)
                last_stage_stack = _build_stage_stack()
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
                _cancel_roi_microstep_transition(reset_shift=False)
                _apply_manual_roi_with_subpixel_compensation(
                    int(message["x"]),
                    int(message["y"]),
                    int(message["w"]),
                    int(message["h"]),
                )
                continue

            if command == "set_roi_position":
                _cancel_roi_microstep_transition(reset_shift=False)
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
                )
                continue

            if command == "cancel_roi_microstep_transition":
                _cancel_roi_microstep_transition(reset_shift=bool(message.get("reset_subpixel_shift", True)))
                continue

            if command == "set_roi_with_subpixel":
                _cancel_roi_microstep_transition(reset_shift=False)
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
                continue

            if command == "set_roi_subpixel_shift":
                _set_roi_shift_target(
                    float(message.get("shift_x", 0.0)),
                    float(message.get("shift_y", 0.0)),
                )
                continue

            if command in {"set_basic_scaling_mode_auto", "set_sr_mode_auto"}:
                processor.set_sr_mode_auto()
                _safe_put({"type": "ack", "cmd": "set_basic_scaling_mode_auto"})
                continue

            if command in {"set_basic_scaling_manual", "set_sr_scale_manual"}:
                processor.set_sr_scale_manual(int(message["scale"]))
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

            if command in {"set_max_auto_basic_scaling", "set_max_auto_sr_scale"}:
                processor.set_max_auto_sr_scale(int(message["scale"]))
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
                ai_sr_require_gpu = bool(message.get("require_gpu", ai_sr_require_gpu))
                ai_sr_frame_interval = max(1, int(message.get("frame_interval", ai_sr_frame_interval)))
                ai_sr_strict = bool(message.get("strict", ai_sr_strict))
                ai_sr_input_align = max(1, int(message.get("input_align", ai_sr_input_align)))
                ai_sr_roi_overscan_percent = float(message.get("roi_overscan_percent", ai_sr_roi_overscan_percent))
                ai_sr_inference_divisor = max(0, int(message.get("inference_divisor", ai_sr_inference_divisor)))
                ai_sr_detail_preserve_percent = float(message.get("detail_preserve_percent", ai_sr_detail_preserve_percent))
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
