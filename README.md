# video_processor Milestone 1

Windows-first C++/CUDA/pybind11 proof-of-concept Python extension.

## v1 pipeline

1. Input: 1920x1080 interlaced UYVY bytes
2. Upload to GPU
3. Bob deinterlace to progressive frame
4. Optional placeholder SR: bicubic internal upscale (default auto)
5. Crop + zoom back to 1920x1080
6. Convert RGB back to UYVY
7. Return bytes to Python

## Folder structure

- CMakeLists.txt
- src/CMakeLists.txt
- src/bindings/module.cpp
- src/core/video_processor.hpp
- src/core/video_processor.cpp
- src/cuda/kernels.cuh
- src/cuda/kernels.cu
- examples/live_pipeline_example.py

## Build on Windows (VSCode + Visual Studio)

Prerequisites:
- Visual Studio 2022 C++ toolchain (MSVC v143 + Windows SDK)
- NVIDIA driver
- CUDA Toolkit 12.x or newer
- CMake 3.24+
- Git (needed if pybind11 is auto-fetched by CMake)

Configure:

cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2" -DCMAKE_BUILD_TYPE=Release

Build:

cmake --build build --config Release

If pybind11 is already installed with CMake config files, CMake will use that copy.
If not, CMake will fetch pybind11 automatically during configure.

Locate module:
- build/src/Release/video_processor.cp<python-abi>-win_amd64.pyd (name may vary by Python version)

Run example:

python examples/live_pipeline_example.py

Run example with local PC preview window:

pip install opencv-python numpy
python examples/live_pipeline_example.py --viewer

Run without placeholder SR (A/B test path):

python examples/live_pipeline_example.py --disable-sr

Run with explicit SR scale override:

python examples/live_pipeline_example.py --sr-scale 4

Use auto SR scale selection (default):

python examples/live_pipeline_example.py --sr-scale 0

Auto SR thresholds (based on max of ROI width ratio and ROI height ratio):
- ratio > 0.5 -> SR 2
- 0.25 < ratio <= 0.5 -> SR 4
- 0.125 < ratio <= 0.25 -> SR 8
- ratio <= 0.125 -> SR 16 (falls back to lower scale if GPU memory is insufficient)

If no viewer window appears:
- Ensure you are running in an interactive desktop session (not headless/remote service context).
- Ensure you installed `opencv-python`, not `opencv-python-headless`.

## Python usage

import video_processor

processor = video_processor.VideoProcessor(
    width=1920,
    height=1080,
    roi_x=480,
    roi_y=270,
    roi_w=960,
    roi_h=540,
    enable_placeholder_sr=True,
    sr_scale=0,
)

output_bytes = processor.process_frame(input_uyvy_bytes)

Live SR toggling at runtime (no processor recreation):

processor.set_sr_mode_auto()
processor.set_sr_scale_manual(4)
effective = processor.get_effective_sr_scale()
is_auto = processor.sr_auto_mode

Run backend smoke test for live SR toggle:

python examples/smoke_live_sr_toggle.py

## GUI test harness

PySide6 GUI is available for interactive ROI/scale testing without DeckLink hardware:

python gui/app.py

GUI capabilities:
- live processed preview from synthetic 1920x1080 UYVY frames
- optional Blackmagic DeckLink capture/output path with format settings
- ROI/scale control from mouse, keyboard, wheel/touchpad, and touchscreen gestures
- SR mode toggle (Auto/Manual) using runtime backend APIs
- manual SR value selection [2, 4, 8, 16]

## Zero allocations per frame notes

Native side is allocation-free inside process_frame:
- Device buffers allocated once in constructor
- CUDA stream created once in constructor
- Output host buffer allocated once in constructor

Per-frame behavior:
- Async H2D copy
- Kernel launches on a single stream
- Async D2H copy
- Stream sync

Note: Returning Python bytes creates a Python object per frame. For absolute zero Python-side allocations, the next milestone should switch to writable preallocated buffers or memoryview/NumPy output.
"# video_processor" 
