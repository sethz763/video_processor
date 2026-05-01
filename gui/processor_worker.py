from __future__ import annotations

import queue
import site
import sys
import threading
import time
import traceback
import os
import importlib
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

RTX_POST_SCALE_METHOD_TO_CV2_INTERP = {
    "nearest": cv2.INTER_NEAREST if cv2 is not None else 0,
    "bilinear": cv2.INTER_LINEAR if cv2 is not None else 1,
    "bicubic": cv2.INTER_CUBIC if cv2 is not None else 2,
    "lanczos": cv2.INTER_LANCZOS4 if cv2 is not None else 4,
}


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
        roi_x = max(0, min(roi_x, FRAME_W - 2))
        roi_y = max(0, min(roi_y, FRAME_H - 2))
        roi_w = max(2, min(roi_w, FRAME_W - roi_x))
        roi_h = max(2, min(roi_h, FRAME_H - roi_y))

        # UYVY is 4:2:2 packed, so x and width must remain even.
        roi_x &= ~1
        roi_w &= ~1
        if roi_w < 2:
            roi_w = 2
        if roi_x + roi_w > FRAME_W:
            roi_x = max(0, FRAME_W - roi_w)
            roi_x &= ~1

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
        rgb = cv2.cvtColor(yuv422, cv2.COLOR_YUV2RGB_UYVY)

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

        sr_yuv422 = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2YUV_UYVY)
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
        roi_rgb = cv2.cvtColor(roi_yuv, cv2.COLOR_YUV2RGB_UYVY)

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

        sr_roi_yuv = cv2.cvtColor(sr_roi_rgb, cv2.COLOR_RGB2YUV_UYVY)
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
        roi_rgb = cv2.cvtColor(roi_yuv, cv2.COLOR_YUV2RGB_UYVY)

        target_w = FRAME_W
        target_h = FRAME_H
        method_name = str(method).strip().lower()
        upscale_interp = cv2.INTER_LANCZOS4
        if method_name == "bilinear":
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

        if self._detail_preserve_percent > 0.0:
            preserve = self._detail_preserve_percent / 100.0
            sr_rgb = cv2.addWeighted(sr_rgb, 1.0 - preserve, baseline_rgb, preserve, 0.0)

        sr_yuv422 = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2YUV_UYVY)
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
    basic_scaling_method = str(cfg.get("basic_scaling_method", cfg.get("sr_flavor", "bicubic")))
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


def _write_frame_to_output(out: object, frame_bytes: bytes) -> None:
    if out.row_bytes == UYVY_ROW_BYTES:
        out.display_frame_sync(frame_bytes)
        return

    if out.row_bytes < UYVY_ROW_BYTES:
        raise RuntimeError(f"Output row_bytes {out.row_bytes} is smaller than expected {UYVY_ROW_BYTES}")

    padded = bytearray(out.row_bytes * out.height)
    for y in range(FRAME_H):
        src_start = y * UYVY_ROW_BYTES
        src_end = src_start + UYVY_ROW_BYTES
        dst_start = y * out.row_bytes
        padded[dst_start : dst_start + UYVY_ROW_BYTES] = frame_bytes[src_start:src_end]

    out.display_frame_sync(padded)


def _normalize_worker_roi(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    roi_x = max(0, min(int(x), FRAME_W - 2))
    roi_y = max(0, min(int(y), FRAME_H - 2))
    roi_w = max(2, min(int(w), FRAME_W - roi_x))
    roi_h = max(2, min(int(h), FRAME_H - roi_y))

    roi_x &= ~1
    roi_w &= ~1
    if roi_w < 2:
        roi_w = 2

    if roi_x + roi_w > FRAME_W:
        roi_x = max(0, FRAME_W - roi_w)
        roi_x &= ~1

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
    processed_frame_counter = 0
    started_perf_ts = 0.0
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
    current_basic_scaling_method = str(startup_config.get("basic_scaling_method", "bicubic"))
    current_roi_x = int(startup_config.get("roi_x", 0))
    current_roi_y = int(startup_config.get("roi_y", 0))
    current_roi_w = int(startup_config.get("roi_w", FRAME_W))
    current_roi_h = int(startup_config.get("roi_h", FRAME_H))
    current_deinterlace_enabled = bool(startup_config.get("deinterlace_enabled", True))
    current_deinterlace_method = str(startup_config.get("deinterlace_method", "bob"))
    current_denoise_method = str(startup_config.get("denoise_method", "off"))
    current_denoise_strength = max(0.0, min(1.0, float(startup_config.get("denoise_strength", 0.35))))
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
        roi_rgb = cv2.cvtColor(roi_yuv, cv2.COLOR_YUV2RGB_UYVY)
        roi_rgba = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2RGBA)

        sr_rgba = rtx_vsr_engine.process_rgba(roi_rgba)
        if not isinstance(sr_rgba, np.ndarray):
            sr_rgba = np.asarray(sr_rgba)

        if sr_rgba.shape[0] != FRAME_H or sr_rgba.shape[1] != FRAME_W:
            interpolation = RTX_POST_SCALE_METHOD_TO_CV2_INTERP.get(rtx_vsr_post_scale_method, cv2.INTER_CUBIC)
            sr_rgba = cv2.resize(sr_rgba, (FRAME_W, FRAME_H), interpolation=interpolation)

        sr_rgb = cv2.cvtColor(sr_rgba, cv2.COLOR_RGBA2RGB)
        sr_yuv422 = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2YUV_UYVY)
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

    def _preprocess_stage(frame_bytes: bytes) -> bytes:
        if not ((ai_sr_enabled and ai_sr_engine is not None) or (rtx_vsr_enabled and rtx_vsr_engine is not None)):
            # Non-AI mode uses a single fused C++ pass in _upscale_stage.
            return frame_bytes

        preprocess_for_ai = current_deinterlace_enabled or (
            current_denoise_method not in {"off", "none"} and current_denoise_strength > 0.001
        )

        if not preprocess_for_ai:
            return frame_bytes

        if hasattr(processor, "process_frame_preprocess_only"):
            return processor.process_frame_preprocess_only(frame_bytes)

        if current_deinterlace_enabled and hasattr(processor, "process_frame_deinterlace_only"):
            return processor.process_frame_deinterlace_only(frame_bytes)

        return frame_bytes

    def _upscale_stage(preprocessed_bytes: bytes) -> tuple[bytes, bool]:
        # AI SR and basic CUDA upscale are mutually exclusive. If AI SR is enabled,
        # always route through AI stage behavior and never invoke basic CUDA upscale.
        if ai_sr_enabled and ai_sr_engine is not None:
            ai_output_bytes, ai_applied = _apply_ai_sr(
                preprocessed_bytes,
                (current_roi_x, current_roi_y, current_roi_w, current_roi_h),
                current_basic_scaling_method,
            )
            return ai_output_bytes, ai_applied

        if rtx_vsr_enabled and rtx_vsr_engine is not None:
            try:
                return _apply_rtx_vsr(
                    preprocessed_bytes,
                    (current_roi_x, current_roi_y, current_roi_w, current_roi_h),
                ), True
            except Exception as rtx_exc:
                _safe_put({"type": "warning", "warning": f"RTX VSR inference failed: {rtx_exc}"})
                return preprocessed_bytes, False

        # Fused path: deinterlace + denoise + basic scaling in one GPU pass.
        return processor.process_frame(preprocessed_bytes), False

    def _process_pipeline_frame(frame_bytes: bytes) -> tuple[bytes, bool]:
        if rtx_vsr_enabled and rtx_vsr_engine is not None and not (ai_sr_enabled and ai_sr_engine is not None):
            preprocessed = _preprocess_stage(frame_bytes)
            return _upscale_stage(preprocessed)

        if not (ai_sr_enabled and ai_sr_engine is not None):
            return processor.process_frame(frame_bytes), False

        frame_for_ai = frame_bytes

        preprocess_for_ai = current_deinterlace_enabled or (
            current_denoise_method not in {"off", "none"} and current_denoise_strength > 0.001
        )

        if preprocess_for_ai:
            if hasattr(processor, "process_frame_preprocess_only"):
                frame_for_ai = processor.process_frame_preprocess_only(frame_for_ai)
            elif current_deinterlace_enabled and hasattr(processor, "process_frame_deinterlace_only"):
                frame_for_ai = processor.process_frame_deinterlace_only(frame_for_ai)

        ai_output_bytes, ai_applied = _apply_ai_sr(
            frame_for_ai,
            (current_roi_x, current_roi_y, current_roi_w, current_roi_h),
            current_basic_scaling_method,
        )

        if ai_applied and ai_sr_engine is not None:
            return ai_output_bytes, True

        if hasattr(processor, "process_frame_no_deinterlace"):
            return processor.process_frame_no_deinterlace(ai_output_bytes), ai_applied

        return processor.process_frame(ai_output_bytes), ai_applied

    def _stop_live_pipeline() -> None:
        nonlocal pipeline_running, capture_thread, preprocess_thread, upscale_thread, output_thread
        if not pipeline_running:
            return
        pipeline_stop_event.set()
        for thread in (capture_thread, preprocess_thread, upscale_thread, output_thread):
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
        if capture_session is None or output_session is None:
            raise RuntimeError("Cannot start pipeline without active DeckLink sessions")

        _stop_live_pipeline()
        pipeline_stop_event.clear()
        q_capture_to_preprocess = queue.Queue(maxsize=2)
        q_preprocess_to_upscale = queue.Queue(maxsize=2)
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
        processed_frame_counter = 0
        started_perf_ts = time.perf_counter()

        def _capture_worker() -> None:
            nonlocal frame_id_counter, capture_drop_count
            nonlocal latest_input_frame, latest_output_frame, latest_effective_sr_scale, processed_frame_counter
            nonlocal latest_rtx_vsr_applied, latest_rtx_effect_mean_abs_luma
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

                if _is_live_passthrough_mode():
                    # In zero-processing mode, avoid staged queueing and preserve output cadence.
                    try:
                        if output_session is not None:
                            _write_frame_to_output(output_session, input_bytes)
                    except Exception as exc:
                        _safe_put({"type": "warning", "warning": f"Output stage failed: {exc}"})
                        continue

                    with state_lock:
                        latest_input_frame = input_bytes
                        latest_output_frame = input_bytes
                        latest_effective_sr_scale = 1
                        latest_rtx_vsr_applied = False
                        latest_rtx_effect_mean_abs_luma = 0.0
                        processed_frame_counter += 1
                    continue

                frame_id_counter += 1
                item = _StageFrame(frame_id=frame_id_counter, captured_ts=time.perf_counter(), input_bytes=input_bytes)
                if _put_latest_stage_frame(q_capture_to_preprocess, item):
                    capture_drop_count += 1

        def _preprocess_worker() -> None:
            nonlocal preprocess_drop_count
            assert q_capture_to_preprocess is not None
            assert q_preprocess_to_upscale is not None
            while not pipeline_stop_event.is_set():
                try:
                    item = q_capture_to_preprocess.get(timeout=0.01)
                except queue.Empty:
                    continue
                try:
                    item.preprocess_bytes = _preprocess_stage(item.input_bytes)
                except Exception as exc:
                    _safe_put({"type": "warning", "warning": f"Preprocess stage failed: {exc}"})
                    continue
                if _put_latest_stage_frame(q_preprocess_to_upscale, item):
                    preprocess_drop_count += 1

        def _upscale_worker() -> None:
            nonlocal upscale_drop_count, ai_sr_dropped_frames
            assert q_preprocess_to_upscale is not None
            assert q_upscale_to_output is not None
            while not pipeline_stop_event.is_set():
                try:
                    item = q_preprocess_to_upscale.get(timeout=0.01)
                except queue.Empty:
                    continue

                preprocessed = item.preprocess_bytes if item.preprocess_bytes is not None else item.input_bytes
                try:
                    output_bytes, ai_applied = _upscale_stage(preprocessed)
                except Exception as exc:
                    _safe_put({"type": "warning", "warning": f"Upscale stage failed: {exc}"})
                    continue

                if ai_sr_engine is not None and not ai_applied and _ai_inference_busy():
                    ai_sr_dropped_frames += 1

                item.output_bytes = output_bytes
                item.ai_applied = ai_applied
                item.rtx_applied = bool(
                    ai_applied
                    and rtx_vsr_enabled
                    and rtx_vsr_engine is not None
                    and not (ai_sr_enabled and ai_sr_engine is not None)
                )
                if _put_latest_stage_frame(q_upscale_to_output, item):
                    upscale_drop_count += 1

        def _output_worker() -> None:
            nonlocal latest_input_frame, latest_output_frame, latest_effective_sr_scale, processed_frame_counter
            nonlocal latest_rtx_vsr_applied, latest_rtx_effect_mean_abs_luma
            assert q_upscale_to_output is not None
            while not pipeline_stop_event.is_set():
                try:
                    item = q_upscale_to_output.get(timeout=0.01)
                except queue.Empty:
                    continue

                output_bytes = item.output_bytes if item.output_bytes is not None else item.input_bytes
                # A lightweight, sampled UYVY byte-delta metric helps verify that
                # RTX VSR is measurably altering output in real time.
                sampled_delta = 0.0
                try:
                    in_arr = np.frombuffer(item.input_bytes, dtype=np.uint8)
                    out_arr = np.frombuffer(output_bytes, dtype=np.uint8)
                    if in_arr.size == out_arr.size and in_arr.size > 0:
                        step = 8
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
                        _write_frame_to_output(output_session, output_bytes)
                except Exception as exc:
                    _safe_put({"type": "warning", "warning": f"Output stage failed: {exc}"})
                    continue

                with state_lock:
                    latest_input_frame = item.input_bytes
                    latest_output_frame = output_bytes
                    latest_effective_sr_scale = int(processor.get_effective_sr_scale())
                    latest_rtx_vsr_applied = bool(item.rtx_applied)
                    latest_rtx_effect_mean_abs_luma = sampled_delta
                    processed_frame_counter += 1

        capture_thread = threading.Thread(target=_capture_worker, name="vp-capture", daemon=True)
        preprocess_thread = threading.Thread(target=_preprocess_worker, name="vp-preprocess", daemon=True)
        upscale_thread = threading.Thread(target=_upscale_worker, name="vp-upscale", daemon=True)
        output_thread = threading.Thread(target=_output_worker, name="vp-output", daemon=True)

        capture_thread.start()
        preprocess_thread.start()
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
        nonlocal rtx_vsr_engine, rtx_vsr_info

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
        nonlocal capture_session, output_session
        if d is None:
            raise RuntimeError("decklink_wrapper is not available in worker process")

        _stop_sessions()

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

            if command == "stop_decklink":
                _stop_sessions()
                _safe_put({"type": "ack", "cmd": "stop_decklink"})
                continue

            if command == "decklink_tick":
                if capture_session is None or output_session is None:
                    _safe_put({"type": "decklink_no_frame", "reason": "sessions_not_started"})
                    continue

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
                _safe_put(
                    {
                        "type": "decklink_frame",
                        "input_frame_bytes": current_input,
                        "output_frame_bytes": current_output,
                        "effective_sr_scale": current_scale,
                        "processed_frame_counter": current_counter,
                        "processed_fps": processed_fps,
                        "ai_sr_applied_frames": int(ai_sr_applied_frames),
                        "ai_sr_passthrough_frames": int(ai_sr_passthrough_frames),
                        "rtx_vsr_applied": current_rtx_applied,
                        "rtx_effect_mean_abs_luma": current_rtx_delta,
                        "pipeline_running": bool(pipeline_running),
                        "stage_queue_depths": stage_depths,
                        "stage_drop_counts": {
                            "capture": int(capture_drop_count),
                            "preprocess": int(preprocess_drop_count),
                            "upscale": int(upscale_drop_count),
                        },
                    }
                )
                continue

            if command == "process_frame":
                frame_id = int(message["frame_id"])
                frame_bytes = message["frame_bytes"]
                output_bytes, ai_applied = _process_pipeline_frame(frame_bytes)

                if ai_sr_engine is not None and not ai_applied and _ai_inference_busy():
                    ai_sr_dropped_frames += 1

                latest_output_frame = output_bytes
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
                current_roi_x, current_roi_y, current_roi_w, current_roi_h = _normalize_worker_roi(
                    int(message["x"]),
                    int(message["y"]),
                    int(message["w"]),
                    int(message["h"]),
                )
                processor.set_roi(current_roi_x, current_roi_y, current_roi_w, current_roi_h)
                if rtx_vsr_enabled:
                    rtx_vsr_error = _refresh_rtx_vsr_engine()
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
                applied_basic_scaling_method = str(message.get("basic_scaling_method", message.get("sr_flavor", "bicubic")))
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
