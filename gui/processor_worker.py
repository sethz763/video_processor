from __future__ import annotations

import queue
import site
import sys
import time
import traceback
from pathlib import Path
from typing import Any

try:
    import decklink_wrapper as d
except Exception:
    d = None


FRAME_W = 1920
FRAME_H = 1080
UYVY_ROW_BYTES = FRAME_W * 2


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
    enable_placeholder_sr = bool(cfg["enable_placeholder_sr"])
    processor = module.VideoProcessor(
        width=int(cfg["width"]),
        height=int(cfg["height"]),
        roi_x=int(cfg["roi_x"]),
        roi_y=int(cfg["roi_y"]),
        roi_w=int(cfg["roi_w"]),
        roi_h=int(cfg["roi_h"]),
        enable_placeholder_sr=enable_placeholder_sr,
        sr_scale=int(cfg["sr_scale"]),
    )
    processor.set_max_auto_sr_scale(int(cfg["max_auto_sr_scale"]))
    processor.set_deinterlace_enabled(bool(cfg["deinterlace_enabled"]))
    # SR runtime mode APIs are invalid when placeholder SR was disabled at construction.
    if enable_placeholder_sr:
        if bool(cfg["sr_auto_mode"]):
            processor.set_sr_mode_auto()
        else:
            processor.set_sr_scale_manual(int(cfg["sr_manual_scale"]))
    return processor


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


def run_processor_worker(request_queue, response_queue, startup_config: dict[str, Any]) -> None:
    def _safe_put(message: dict[str, Any]) -> None:
        try:
            response_queue.put_nowait(message)
        except queue.Full:
            try:
                response_queue.get_nowait()
            except queue.Empty:
                pass
            response_queue.put_nowait(message)

    processor = None
    capture_session = None
    output_session = None
    latest_input_frame: bytes | None = None
    latest_output_frame: bytes | None = None
    latest_effective_sr_scale = 1
    processed_frame_counter = 0
    started_perf_ts = 0.0

    def _stop_sessions() -> None:
        nonlocal capture_session, output_session

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
        nonlocal capture_session, output_session, processed_frame_counter, started_perf_ts
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

        processed_frame_counter = 0
        started_perf_ts = time.perf_counter()
    try:
        project_root = Path(startup_config["project_root"])
        module = _load_video_processor_module(project_root)
        processor = _create_processor(module, startup_config)
        _safe_put({"type": "ready"})

        while True:
            message = None
            try:
                message = request_queue.get_nowait()
            except queue.Empty:
                message = None

            if message is None:
                if capture_session is not None and output_session is not None:
                    frame = capture_session.acquire(timeout_ms=1)
                    if frame is not None:
                        input_bytes = _tight_uyvy_bytes(frame)
                        output_bytes = processor.process_frame(input_bytes)
                        _write_frame_to_output(output_session, output_bytes)

                        latest_input_frame = input_bytes
                        latest_output_frame = output_bytes
                        latest_effective_sr_scale = int(processor.get_effective_sr_scale())
                        processed_frame_counter += 1
                    continue

                # Idle backoff when no active DeckLink sessions and no control message.
                time.sleep(0.002)
                continue

            command = message.get("cmd")
            if command == "shutdown":
                _stop_sessions()
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

                if latest_input_frame is None or latest_output_frame is None:
                    _safe_put({"type": "decklink_no_frame", "reason": "no_input_signal"})
                    continue

                elapsed = max(0.0001, time.perf_counter() - started_perf_ts)
                processed_fps = float(processed_frame_counter) / elapsed
                _safe_put(
                    {
                        "type": "decklink_frame",
                        "input_frame_bytes": latest_input_frame,
                        "output_frame_bytes": latest_output_frame,
                        "effective_sr_scale": int(latest_effective_sr_scale),
                        "processed_frame_counter": int(processed_frame_counter),
                        "processed_fps": processed_fps,
                    }
                )
                continue

            if command == "process_frame":
                frame_id = int(message["frame_id"])
                frame_bytes = message["frame_bytes"]
                output_bytes = processor.process_frame(frame_bytes)
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
                processor.set_roi(
                    int(message["x"]),
                    int(message["y"]),
                    int(message["w"]),
                    int(message["h"]),
                )
                continue

            if command == "set_sr_mode_auto":
                processor.set_sr_mode_auto()
                continue

            if command == "set_sr_scale_manual":
                processor.set_sr_scale_manual(int(message["scale"]))
                continue

            if command == "set_deinterlace_enabled":
                processor.set_deinterlace_enabled(bool(message["enabled"]))
                continue

            if command == "set_max_auto_sr_scale":
                processor.set_max_auto_sr_scale(int(message["scale"]))
                continue

    except BaseException as exc:
        _stop_sessions()
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
