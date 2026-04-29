# GUI Test Harness

PySide6 GUI for testing `video_processor` ROI and SR behavior without DeckLink hardware.

## Features

- Synthetic 1920x1080 UYVY frame source
- Optional Blackmagic DeckLink input/output mode
- Live preview of processed output
- ROI editing via mouse drag/resize, keyboard, wheel/touchpad zoom, and touch gestures
- SR controls: Auto / Manual [2, 4, 8, 16] using live backend toggles
- SR flavor controls: Bilinear / Bicubic / Bicubic+Sharpen
- Optional AI SR via ONNX Runtime in worker backend
- Runtime settings panel (FPS, SR mode, ROI values, scale)
- Blackmagic format settings (device indices, mode queries, format detection)

## Run

From repository root:

```powershell
venv\Scripts\python.exe gui\app.py
```

The app auto-discovers the built extension from:

- `build/src/Release`
- `build/src/RelWithDebInfo`
- `build/src/Debug`

## Notes

- Keep the native module rebuilt after backend changes.
- `Enable placeholder SR` recreates `VideoProcessor` because it is a constructor-time setting.
- In `Synthetic` mode, input/output preview is generated locally.
- In `Blackmagic DeckLink` mode, click `Apply DeckLink Settings` after selecting device indices and mode queries.
- The GUI now attempts an experimental worker-process backend first; in Blackmagic mode this worker owns capture + processing + output, and the GUI only renders previews.
- If worker startup fails, the app automatically falls back to the legacy in-process backend.

## AI SR (ONNX) Quick Start

1. Install ONNX Runtime (CPU or GPU):

```powershell
venv\Scripts\python.exe -m pip install onnxruntime
```

Or for NVIDIA acceleration:

```powershell
venv\Scripts\python.exe -m pip install onnxruntime-gpu
```

2. Place a super-resolution ONNX model (for example a Real-ESRGAN variant) at:

- `models/realesrgan_x4plus.onnx`

3. Launch with environment variables:

```powershell
$env:VP_AI_SR_ENABLE = "1"
$env:VP_AI_SR_MODEL = "C:\Coding Projects\video_processor\models\realesrgan_x4plus.onnx"
# Recommended for best speed: auto-select TensorRT/CUDA when available.
$env:VP_AI_SR_PROVIDER = "auto"
# Run inference asynchronously every 2nd frame by default (better real-time behavior).
$env:VP_AI_SR_STRICT = "0"
$env:VP_AI_SR_FRAME_INTERVAL = "2"
venv\Scripts\python.exe gui\app.py
```

4. In the GUI, enable `Enable AI SR (ONNX model)`.

If the model cannot be loaded, status text will report the exact reason and processing will continue with CUDA placeholder SR only.
