from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


PROBE_AXIS_VALUES = [
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    11,
    12,
    13,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
]

SPECIAL_PAIRS = [
    (2, 342),
    (344, 344),
    (342, 342),
]


def _map_ort_dtype(type_text: str) -> np.dtype[Any]:
    tt = type_text.lower()
    if "float16" in tt:
        return np.float16
    if "float" in tt:
        return np.float32
    if "uint8" in tt:
        return np.uint8
    if "int8" in tt:
        return np.int8
    if "int64" in tt:
        return np.int64
    if "int32" in tt:
        return np.int32
    return np.float32


def _dim_to_int(dim: Any) -> int | None:
    if isinstance(dim, int):
        return dim
    return None


@dataclass
class AxisSummary:
    min_ok: int | None
    max_ok: int | None
    strongest_divisor: int
    even_only: bool
    odd_supported: bool
    failures: list[int]


def _summarize_axis(tested: list[int], ok_values: list[int]) -> AxisSummary:
    tested_set = sorted(set(int(v) for v in tested))
    ok_set = sorted(set(int(v) for v in ok_values))
    fail_set = sorted(v for v in tested_set if v not in set(ok_set))

    min_ok = ok_set[0] if ok_set else None
    max_ok = ok_set[-1] if ok_set else None
    odd_supported = any((v % 2) == 1 for v in ok_set)
    even_only = bool(ok_set) and not odd_supported and any((v % 2) == 1 for v in tested_set)

    strongest_divisor = 1
    for d in (2, 4, 8, 16, 32, 64):
        if ok_set and all((v % d) == 0 for v in ok_set):
            strongest_divisor = d

    return AxisSummary(
        min_ok=min_ok,
        max_ok=max_ok,
        strongest_divisor=strongest_divisor,
        even_only=even_only,
        odd_supported=odd_supported,
        failures=fail_set,
    )


def _build_input_tensor(
    input_shape: list[Any],
    input_dtype: np.dtype[Any],
    h: int,
    w: int,
) -> np.ndarray:
    rank = len(input_shape)
    if rank != 4:
        raise ValueError(f"Unsupported input rank for probing: {rank}")

    n = _dim_to_int(input_shape[0]) or 1
    c = _dim_to_int(input_shape[1]) or 3
    h_fixed = _dim_to_int(input_shape[2]) or h
    w_fixed = _dim_to_int(input_shape[3]) or w

    if np.issubdtype(input_dtype, np.floating):
        arr = np.zeros((n, c, h_fixed, w_fixed), dtype=input_dtype)
    else:
        arr = np.zeros((n, c, h_fixed, w_fixed), dtype=input_dtype)
    return arr


def _probe_model(model_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": str(model_path),
        "ok": False,
        "error": None,
    }

    available_providers = ort.get_available_providers()
    preferred_order = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in preferred_order if p in available_providers]
    if not providers:
        providers = ["CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as exc:
        result["error"] = f"Failed to create session: {exc}"
        return result

    inputs = session.get_inputs()
    if not inputs:
        result["error"] = "Model has no inputs"
        return result

    meta = inputs[0]
    in_name = meta.name
    in_shape = list(meta.shape)
    in_type = str(getattr(meta, "type", ""))
    in_dtype = _map_ort_dtype(in_type)

    result["input_name"] = in_name
    result["input_shape"] = in_shape
    result["input_type"] = in_type
    result["session_providers"] = session.get_providers()

    if len(in_shape) != 4:
        result["error"] = f"Unsupported input rank for probe: {len(in_shape)}"
        return result

    h_fixed = _dim_to_int(in_shape[2])
    w_fixed = _dim_to_int(in_shape[3])
    rank_dynamic = [d is None or not isinstance(d, int) for d in in_shape]

    h_probe_ref = h_fixed or 64
    w_probe_ref = w_fixed or 64

    tested_h: list[int] = []
    ok_h: list[int] = []
    tested_w: list[int] = []
    ok_w: list[int] = []
    pair_tests: list[dict[str, Any]] = []
    error_samples: list[str] = []

    for h in PROBE_AXIS_VALUES:
        tested_h.append(h)
        try:
            x = _build_input_tensor(in_shape, in_dtype, h=h, w=w_probe_ref)
            _ = session.run(None, {in_name: x})
            ok_h.append(h)
        except Exception as exc:
            msg = str(exc).splitlines()[0][:180]
            if msg not in error_samples and len(error_samples) < 8:
                error_samples.append(msg)

    for w in PROBE_AXIS_VALUES:
        tested_w.append(w)
        try:
            x = _build_input_tensor(in_shape, in_dtype, h=h_probe_ref, w=w)
            _ = session.run(None, {in_name: x})
            ok_w.append(w)
        except Exception as exc:
            msg = str(exc).splitlines()[0][:180]
            if msg not in error_samples and len(error_samples) < 8:
                error_samples.append(msg)

    for h, w in SPECIAL_PAIRS:
        ok_pair = False
        pair_err = None
        try:
            x = _build_input_tensor(in_shape, in_dtype, h=h, w=w)
            _ = session.run(None, {in_name: x})
            ok_pair = True
        except Exception as exc:
            pair_err = str(exc).splitlines()[0][:180]
            if pair_err not in error_samples and len(error_samples) < 8:
                error_samples.append(pair_err)
        pair_tests.append({"h": h, "w": w, "ok": ok_pair, "error": pair_err})

    h_summary = _summarize_axis(tested_h, ok_h)
    w_summary = _summarize_axis(tested_w, ok_w)

    result.update(
        {
            "ok": True,
            "rank_dynamic": rank_dynamic,
            "h_fixed": h_fixed,
            "w_fixed": w_fixed,
            "tested_h": tested_h,
            "ok_h": ok_h,
            "tested_w": tested_w,
            "ok_w": ok_w,
            "h_summary": {
                "min_ok": h_summary.min_ok,
                "max_ok": h_summary.max_ok,
                "strongest_divisor": h_summary.strongest_divisor,
                "even_only": h_summary.even_only,
                "odd_supported": h_summary.odd_supported,
                "failures": h_summary.failures,
            },
            "w_summary": {
                "min_ok": w_summary.min_ok,
                "max_ok": w_summary.max_ok,
                "strongest_divisor": w_summary.strongest_divisor,
                "even_only": w_summary.even_only,
                "odd_supported": w_summary.odd_supported,
                "failures": w_summary.failures,
            },
            "pair_tests": pair_tests,
            "error_samples": error_samples,
        }
    )

    return result


def _rule_text(summary: dict[str, Any], axis_name: str) -> str:
    min_ok = summary.get("min_ok")
    divisor = int(summary.get("strongest_divisor", 1))
    if min_ok is None:
        return f"No accepted {axis_name} values found in probe set."
    if divisor > 1:
        return f"Observed: {axis_name} >= {min_ok} and {axis_name} % {divisor} == 0 (within tested range)."
    return f"Observed: {axis_name} >= {min_ok} (within tested range)."


def _make_markdown(results: list[dict[str, Any]], root: Path) -> str:
    lines: list[str] = []
    lines.append("# ONNX Input Resolution Compatibility Report")
    lines.append("")
    lines.append(f"Scanned root: `{root}`")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Results are empirical (onnxruntime CPU provider).")
    lines.append("- Rules are observed constraints in tested ranges, not symbolic graph proofs.")
    lines.append("")

    for res in results:
        model = Path(res["model"]).as_posix()
        lines.append(f"## {model}")
        if not res.get("ok"):
            lines.append(f"- Status: FAILED - {res.get('error', 'unknown error')}")
            lines.append("")
            continue

        lines.append("- Status: OK")
        lines.append(f"- Input: name={res.get('input_name')} shape={res.get('input_shape')} type={res.get('input_type')}")
        lines.append(f"- Height rule: {_rule_text(res['h_summary'], 'H')}")
        lines.append(f"- Width rule: {_rule_text(res['w_summary'], 'W')}")

        h_fail = res["h_summary"].get("failures", [])
        w_fail = res["w_summary"].get("failures", [])
        if h_fail:
            lines.append(f"- Rejected H samples: {h_fail[:20]}")
        if w_fail:
            lines.append(f"- Rejected W samples: {w_fail[:20]}")

        pair_failures = [p for p in res.get("pair_tests", []) if not p.get("ok")]
        if pair_failures:
            lines.append("- Special pair failures:")
            for p in pair_failures[:8]:
                lines.append(f"  - ({p['h']}, {p['w']}): {p.get('error')}")

        if res.get("error_samples"):
            lines.append("- Error samples:")
            for msg in res["error_samples"][:5]:
                lines.append(f"  - {msg}")

        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    models_root = project_root / "models"

    model_paths = sorted(
        p for p in models_root.rglob("*.onnx") if p.is_file() and not p.name.startswith("_tmp_")
    )

    if not model_paths:
        print("No ONNX models found under models/")
        return 1

    results: list[dict[str, Any]] = []
    for path in model_paths:
        print(f"Probing: {path}")
        results.append(_probe_model(path))

    out_json = models_root / "onnx_input_resolution_report.json"
    out_md = models_root / "onnx_input_resolution_report.md"

    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out_md.write_text(_make_markdown(results, models_root), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
