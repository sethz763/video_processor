# ONNX Input Resolution Compatibility Report

Scanned root: `C:\Coding Projects\video_processor\models`

Notes:
- Results are empirical (onnxruntime CPU provider).
- Rules are observed constraints in tested ranges, not symbolic graph proofs.

## C:/Coding Projects/video_processor/models/efrlfn_x2.onnx
- Status: OK
- Input: name=input shape=['batch_size', 3, 'height', 'width'] type=tensor(float)
- Height rule: Observed: H >= 2 (within tested range).
- Width rule: Observed: W >= 2 (within tested range).

## C:/Coding Projects/video_processor/models/efrlfn_x4.onnx
- Status: OK
- Input: name=input shape=['batch_size', 3, 'height', 'width'] type=tensor(float)
- Height rule: Observed: H >= 2 (within tested range).
- Width rule: Observed: W >= 2 (within tested range).

## C:/Coding Projects/video_processor/models/realesrgan_x2plus.onnx
- Status: OK
- Input: name=input shape=['batch', 3, 'height', 'width'] type=tensor(float)
- Height rule: Observed: H >= 2 and H % 2 == 0 (within tested range).
- Width rule: Observed: W >= 2 and W % 2 == 0 (within tested range).
- Rejected H samples: [3, 5, 7, 9, 11, 13]
- Rejected W samples: [3, 5, 7, 9, 11, 13]
- Error samples:
  - [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Reshape node. Name:'node_view' Status Message: E:\_work\1\s\onnxruntime\core\providers\cpu\

## C:/Coding Projects/video_processor/models/realesrgan_x4plus.onnx
- Status: OK
- Input: name=input shape=['batch_size', 3, 'height', 'width'] type=tensor(float)
- Height rule: Observed: H >= 2 (within tested range).
- Width rule: Observed: W >= 2 (within tested range).

## C:/Coding Projects/video_processor/models/RealEsrganONNX/RealESRGAN_x2.onnx
- Status: OK
- Input: name=input shape=['batch_size', 3, 'height', 'width'] type=tensor(float)
- Height rule: Observed: H >= 2 and H % 2 == 0 (within tested range).
- Width rule: Observed: W >= 2 and W % 2 == 0 (within tested range).
- Rejected H samples: [3, 5, 7, 9, 11, 13]
- Rejected W samples: [3, 5, 7, 9, 11, 13]
- Error samples:
  - [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Reshape node. Name:'Reshape_29' Status Message: E:\_work\1\s\onnxruntime\core\providers\cpu

## C:/Coding Projects/video_processor/models/RealEsrganONNX/RealESRGAN_x2_fp16.onnx
- Status: OK
- Input: name=input shape=['batch_size', 3, 'height', 'width'] type=tensor(float16)
- Height rule: Observed: H >= 2 and H % 2 == 0 (within tested range).
- Width rule: Observed: W >= 2 and W % 2 == 0 (within tested range).
- Rejected H samples: [3, 5, 7, 9, 11, 13]
- Rejected W samples: [3, 5, 7, 9, 11, 13]
- Error samples:
  - [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Reshape node. Name:'Reshape_29' Status Message: E:\_work\1\s\onnxruntime\core\providers\cpu

## C:/Coding Projects/video_processor/models/RealEsrganONNX/RealESRGAN_x4.onnx
- Status: OK
- Input: name=input shape=['batch_size', 3, 'height', 'width'] type=tensor(float)
- Height rule: Observed: H >= 2 (within tested range).
- Width rule: Observed: W >= 2 (within tested range).

## C:/Coding Projects/video_processor/models/RealEsrganONNX/RealESRGAN_x4_fp16.onnx
- Status: OK
- Input: name=input shape=['batch_size', 3, 'height', 'width'] type=tensor(float16)
- Height rule: Observed: H >= 2 (within tested range).
- Width rule: Observed: W >= 2 (within tested range).

## C:/Coding Projects/video_processor/models/RealEsrganONNX/RealESRGAN_x8.onnx
- Status: OK
- Input: name=input shape=['batch_size', 3, 'height', 'width'] type=tensor(float)
- Height rule: Observed: H >= 2 (within tested range).
- Width rule: Observed: W >= 2 (within tested range).

## C:/Coding Projects/video_processor/models/RealEsrganONNX/RealESRGAN_x8_fp16.onnx
- Status: OK
- Input: name=input shape=['batch_size', 3, 'height', 'width'] type=tensor(float16)
- Height rule: Observed: H >= 2 (within tested range).
- Width rule: Observed: W >= 2 (within tested range).

