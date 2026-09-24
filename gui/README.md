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
- GPU effects graph with live-input blur, media compositing, and capture-device source nodes

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
- In the effects graph, use the mouse wheel to zoom, right-click a port to disconnect it, and edit blur radius directly on the Blur node.
- Select nodes with Shift-click or Shift-drag, then press Delete to remove all selected effect nodes. `New Graph` clears the editor after prompting about unsaved changes. Mouse-wheel zoom ranges from 25% to 200%.
- Gaussian and box blur use separable CUDA passes so large radii remain practical at full HD.
- Video/Image nodes accept files from the browse button or drag-and-drop. Play/Pause controls frame advancement; Loop either rewinds at end-of-stream or holds the final frame.
- A Video Capture Device node shows the selected DeckLink device or Windows camera as a readable label. Right-click the node to choose a device, `Reload Camera` without reloading the graph, or open `Settings...`, including the explicit webcam resolution selector. Webcam opening and frame reads run on a dedicated decoder thread so effect recall does not block or alter DeckLink control-thread COM state. DirectShow cameras also offer their native device property page; DeckLink devices use the Blackmagic I/O controls.
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

## Node editor

Bypass skips graph compilation and all native composition, transform, and color stages, and disables graph denoise. Editing or animating a bypassed graph keeps the worker's effects configuration unchanged; saved settings are retained for re-enabling. Independent basic scaling and video I/O still run.

- Undo: Ctrl+Z. Redo: Ctrl+Y or Ctrl+Shift+Z. Toolbar buttons provide the same actions. History includes node settings, positions, connections, additions, and deletions (up to 100 edits).
- Copy/Paste: Ctrl+C / Ctrl+V copies selected nodes, their keyframes, and internal connections with new IDs and an offset. Undoing a deletion restores animation too. Input/output boundary nodes are excluded.
- Compositors accept up to 64 layers and can feed other compositors. Nested graphs share a budget of 64 rendered layers. Color and alpha are preserved between GPU passes, including separate alpha connections.
- Add **Composition Source** to select a saved palette composition. This embeds a snapshot, including its keyframes, evaluated at the parent timeline frame. Later palette changes do not alter the embedded snapshot.
- An unconnected background stays transparent. Compositor and composition source nodes expose alpha ports; the native `get_effects_rgba_output()` method returns straight-alpha RGBA bytes for the most recently processed frame. Existing UYVY video outputs retain their format and show transparent regions as black.
- 3D Transform X/Y extend to ±10,000%. Z retains its original depth behavior through +90; +1000 reaches 10× magnification. Negative Z can be typed down to −100,000,000, where the image becomes invisibly small. The slider covers −1000 to +1000; the numeric field provides the larger negative range.

Rebuild the native module after updating these features, then restart the application.

### Diagnosing regular physical-output skips

`gui/logs/app.log` now includes `CADENCE` records when cadence events or capture
queue-drop counters change. They contain SDK input sequence/timestamp gaps,
host capture-delivery gaps, output-submission gaps with processing/submit times,
and the output buffer depth. Recent events are bounded to eight per snapshot;
timestamps are seconds since that pipeline run began. A delivery/submission gap
is an observation, not proof of a displayed-frame drop. The installed DeckLink
wrapper exposes completion callback counts but not late/dropped completion results.

To investigate a roughly 30-second skip, run the same feed for at least two
minutes and note the wall-clock times of visible skips. The corresponding cadence
records distinguish upstream gaps from host processing/submission stalls without
changing output buffering or playback scheduling.

Worker output now recovers an overdue schedule when the SDK confirms the buffer
is empty and the next presentation time is at least one full frame behind the
host playback clock. It advances to a future frame boundary with the configured
buffer lead, without stopping playback or changing already queued frames. The
last displayed frame may be held during recovery. Health telemetry exposes
`clock_correction_events` and `skipped_schedule_slots` (timestamp slots, not SDK
drop counts). The host clock is an estimate because this wrapper does not expose
the device stream clock; this recovery does not eliminate processing stalls or
guarantee synchronization between independently clocked input/output devices.
