# GUI Test Harness

PySide6 GUI for testing `video_processor` ROI and SR behavior without DeckLink hardware.

## Features

- Synthetic 1920x1080 UYVY frame source
- Optional Blackmagic DeckLink input/output mode
- Live preview of processed output
- ROI editing via mouse drag/resize, keyboard, wheel/touchpad zoom, and touch gestures
- SR controls: Auto / Manual [2, 4, 8, 16] using live backend toggles
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
