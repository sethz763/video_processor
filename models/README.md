# AI SR Models

Place ONNX super-resolution models here for GUI AI SR testing.

Default path expected by the GUI:
- models/realesrgan_x4plus.onnx

You can also choose any file path from the GUI field `AI SR model`.


current models tested:
RealEsrganONNX
RealEsrganONNX_x2_fp16.onnx
RealEsrganONNX_x2.onnx
RealEsrganONNX_x4_fp16.onnx
RealEsrganONNX_x4.onnx
RealEsrganONNX_x8_fp16.onnx
RealEsrganONNX_x8.onnx

efrlfn_x2.onnx
efrlfn_x4.onnx
realesrgan_x4plus.onnx


## Input resolution rules (tested)

Source data:
- models/onnx_input_resolution_report.md
- models/onnx_input_resolution_report.json

Global baseline:
- Input tensor format is NCHW with dynamic height/width on all tested models.
- Minimum accepted size in tested range is H >= 2 and W >= 2.

Models that require even dimensions:
- realesrgan_x2plus.onnx
- RealEsrganONNX/RealESRGAN_x2.onnx
- RealEsrganONNX/RealESRGAN_x2_fp16.onnx

Rule for these models:
- H must be even (H % 2 == 0)
- W must be even (W % 2 == 0)

Models that accepted odd and even dimensions in tested range:
- efrlfn_x2.onnx
- efrlfn_x4.onnx
- realesrgan_x4plus.onnx
- RealEsrganONNX/RealESRGAN_x4.onnx
- RealEsrganONNX/RealESRGAN_x4_fp16.onnx
- RealEsrganONNX/RealESRGAN_x8.onnx
- RealEsrganONNX/RealESRGAN_x8_fp16.onnx

Rule for these models:
- H >= 2
- W >= 2

Practical UI rule to remove guesswork:
- If model name contains x2 and is RealESRGAN family, force even ROI width and height.
- Otherwise, clamp ROI width/height to at least 2 and to frame bounds.
