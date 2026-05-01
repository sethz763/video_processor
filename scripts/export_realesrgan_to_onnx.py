from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

import torch


def _load_rrdbnet(scale: int):
    # realesrgan/basicsr expects an older torchvision import path.
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        try:
            importlib.import_module("torchvision.transforms.functional_tensor")
        except ModuleNotFoundError:
            alt = importlib.import_module("torchvision.transforms._functional_tensor")
            sys.modules["torchvision.transforms.functional_tensor"] = alt

    from basicsr.archs.rrdbnet_arch import RRDBNet

    return RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")

    for key in ("params_ema", "params", "state_dict"):
        maybe_state = checkpoint.get(key)
        if isinstance(maybe_state, dict):
            state = maybe_state
            break
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise TypeError("Could not find a valid state_dict in checkpoint.")

    cleaned: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_name = name[7:] if name.startswith("module.") else name
        cleaned[clean_name] = value

    if not cleaned:
        raise ValueError("No tensor parameters found in checkpoint.")

    return cleaned


def export_onnx(
    input_path: Path,
    output_path: Path,
    scale: int,
    opset: int,
    sample_height: int,
    sample_width: int,
    dynamic: bool,
) -> None:
    model = _load_rrdbnet(scale=scale)

    checkpoint = torch.load(str(input_path), map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        raise RuntimeError(
            "Missing checkpoint parameters for model: " + ", ".join(sorted(missing)[:10])
        )
    if unexpected:
        raise RuntimeError(
            "Unexpected checkpoint parameters for model: "
            + ", ".join(sorted(unexpected)[:10])
        )

    model.eval()

    dummy_input = torch.randn(1, 3, sample_height, sample_width, dtype=torch.float32)
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "out_height", 3: "out_width"},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
        )

    import onnx

    exported = onnx.load(str(output_path))
    onnx.checker.check_model(exported)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RealESRGAN RRDBNet .pth checkpoint to ONNX."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("models/RealESRGAN_x2plus.pth"),
        help="Path to .pth checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/realesrgan_x2plus.onnx"),
        help="Output ONNX file path.",
    )
    parser.add_argument("--scale", type=int, default=2, help="Upscale factor (x2/x4).")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument(
        "--sample-height",
        type=int,
        default=64,
        help="Sample input height used for tracing.",
    )
    parser.add_argument(
        "--sample-width",
        type=int,
        default=64,
        help="Sample input width used for tracing.",
    )
    parser.add_argument(
        "--static-shape",
        action="store_true",
        help="Export with static input shape (disable dynamic axes).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_onnx(
        input_path=args.input,
        output_path=args.output,
        scale=args.scale,
        opset=args.opset,
        sample_height=args.sample_height,
        sample_width=args.sample_width,
        dynamic=not args.static_shape,
    )
    print(f"Export complete: {args.output}")


if __name__ == "__main__":
    main()
