from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [
    PROJECT_ROOT / "build" / "src" / "Release",
    PROJECT_ROOT / "build" / "src" / "Debug",
    PROJECT_ROOT / "build" / "src" / "RelWithDebInfo",
]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _import_module():
    import rtx_vsr  # type: ignore

    return rtx_vsr


def test_basic_process():
    rtx_vsr = _import_module()

    in_w, in_h = 320, 180
    out_w, out_h = 640, 360

    sr = rtx_vsr.RTXVideoSR(in_w, in_h, out_w, out_h, quality="high")

    frame = np.zeros((in_h, in_w, 4), dtype=np.uint8)
    frame[..., 0] = 32
    frame[..., 1] = 96
    frame[..., 2] = 160
    frame[..., 3] = 255

    out = sr.process_rgba(frame)

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.shape == (out_h, out_w, 4)

    sr.close()


def test_invalid_shape_raises():
    rtx_vsr = _import_module()

    sr = rtx_vsr.RTXVideoSR(64, 64, 128, 128)
    bad = np.zeros((64, 64, 3), dtype=np.uint8)

    try:
        sr.process_rgba(bad)
        raise AssertionError("Expected ValueError for invalid channel count")
    except ValueError:
        pass
    finally:
        sr.close()


def test_invalid_dtype_raises():
    rtx_vsr = _import_module()

    sr = rtx_vsr.RTXVideoSR(64, 64, 128, 128)
    bad = np.zeros((64, 64, 4), dtype=np.float32)

    try:
        sr.process_rgba(bad)
        raise AssertionError("Expected ValueError for invalid dtype")
    except ValueError:
        pass
    finally:
        sr.close()


if __name__ == "__main__":
    tests = [
        test_basic_process,
        test_invalid_shape_raises,
        test_invalid_dtype_raises,
    ]

    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failures += 1
            print(f"FAIL: {test.__name__}: {exc}")

    if failures:
        raise SystemExit(1)

    print("All RTX VSR smoke tests passed")
