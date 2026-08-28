#include <pybind11/pybind11.h>

#include <cstdint>
#include <cctype>
#include <cstring>
#include <string>
#include <utility>

#include "core/ai_sr_cuda_post_processor.hpp"
#include "core/video_processor.hpp"

namespace py = pybind11;

namespace {

std::pair<const std::uint8_t*, std::size_t> GetContiguousByteBuffer(const py::buffer& frame) {
    const py::buffer_info info = frame.request();
    if (info.itemsize != 1) {
        throw py::value_error("Frame buffer must have byte-sized elements");
    }
    if (info.ndim < 1) {
        throw py::value_error("Frame buffer must have at least one dimension");
    }

    std::size_t total_bytes = static_cast<std::size_t>(info.itemsize);
    for (py::ssize_t dim = info.ndim - 1; dim >= 0; --dim) {
        const py::ssize_t shape = info.shape[dim];
        const py::ssize_t stride = info.strides[dim];
        if (shape < 0) {
            throw py::value_error("Frame buffer has invalid shape");
        }
        if (shape > 1 && stride != static_cast<py::ssize_t>(total_bytes)) {
            throw py::value_error("Frame buffer must be C-contiguous");
        }
        total_bytes *= static_cast<std::size_t>(shape);
    }

    return {
        reinterpret_cast<const std::uint8_t*>(info.ptr),
        total_bytes,
    };
}

std::pair<std::uint8_t*, std::size_t> GetWritableContiguousByteBuffer(
    const py::buffer& buffer,
    const char* buffer_name
) {
    const py::buffer_info info = buffer.request();
    if (info.readonly) {
        throw py::value_error(std::string(buffer_name) + " must be writable");
    }
    if (info.itemsize != 1) {
        throw py::value_error(std::string(buffer_name) + " must have byte-sized elements");
    }
    if (info.ndim < 1) {
        throw py::value_error(std::string(buffer_name) + " must have at least one dimension");
    }

    std::size_t total_bytes = static_cast<std::size_t>(info.itemsize);
    for (py::ssize_t dim = info.ndim - 1; dim >= 0; --dim) {
        const py::ssize_t shape = info.shape[dim];
        const py::ssize_t stride = info.strides[dim];
        if (shape < 0) {
            throw py::value_error(std::string(buffer_name) + " has invalid shape");
        }
        if (shape > 1 && stride != static_cast<py::ssize_t>(total_bytes)) {
            throw py::value_error(std::string(buffer_name) + " must be C-contiguous");
        }
        total_bytes *= static_cast<std::size_t>(shape);
    }

    return {
        reinterpret_cast<std::uint8_t*>(info.ptr),
        total_bytes,
    };
}

py::ssize_t CopyOutputToWritableBuffer(
    const std::string& output,
    std::uint8_t* out_ptr,
    std::size_t out_size,
    const char* output_arg_name
) {
    if (output.size() != out_size) {
        throw py::value_error(
            std::string(output_arg_name) +
            " has incorrect size; expected " +
            std::to_string(output.size()) +
            " bytes"
        );
    }
    std::memcpy(out_ptr, output.data(), output.size());
    return static_cast<py::ssize_t>(output.size());
}

}  // namespace

PYBIND11_MODULE(video_processor, m) {
    m.doc() = "CUDA video processor module for UYVY deinterlace/crop/zoom basic scaling";

    py::class_<vp::CudaTensorBuffer>(m, "CudaTensorBuffer")
        .def_property_readonly("data_ptr", &vp::CudaTensorBuffer::DataPtr)
        .def_property_readonly("bytes", &vp::CudaTensorBuffer::Bytes)
        .def_property_readonly("width", &vp::CudaTensorBuffer::Width)
        .def_property_readonly("height", &vp::CudaTensorBuffer::Height)
        .def_property_readonly("channels", &vp::CudaTensorBuffer::Channels)
        .def_property_readonly("dtype", &vp::CudaTensorBuffer::DType)
        .def_property_readonly("layout", &vp::CudaTensorBuffer::Layout)
        .def_property_readonly("normalized_01", &vp::CudaTensorBuffer::Normalized01);

    py::class_<vp::VideoProcessor>(m, "VideoProcessor")
        .def(
            py::init<int, int, int, int, int, int, bool, int>(),
            py::arg("width") = 1920,
            py::arg("height") = 1080,
            py::arg("roi_x") = 0,
            py::arg("roi_y") = 0,
            py::arg("roi_w") = 1920,
            py::arg("roi_h") = 1080,
            py::arg("enable_placeholder_sr") = true,
            py::arg("sr_scale") = 0
        )
        .def(
            "process_frame",
            [](vp::VideoProcessor& self, const py::buffer& frame) {
                const auto [frame_ptr, frame_size] = GetContiguousByteBuffer(frame);
                std::string output;
                {
                    py::gil_scoped_release release;
                    output = self.ProcessFrameBuffer(frame_ptr, frame_size);
                }
                return py::bytes(output);
            },
            py::arg("frame"),
            "Process one 1920x1080 interlaced UYVY frame and return UYVY bytes."
        )
        .def(
            "process_frame_into",
            [](vp::VideoProcessor& self, const py::buffer& frame, const py::buffer& output) {
                const auto [frame_ptr, frame_size] = GetContiguousByteBuffer(frame);
                const auto [out_ptr, out_size] = GetWritableContiguousByteBuffer(output, "output");

                std::string processed;
                {
                    py::gil_scoped_release release;
                    processed = self.ProcessFrameBuffer(frame_ptr, frame_size);
                }
                return CopyOutputToWritableBuffer(processed, out_ptr, out_size, "output");
            },
            py::arg("frame"),
            py::arg("output"),
            "Process one frame and write UYVY bytes into a caller-provided writable output buffer."
        )
        .def(
            "process_frame_no_deinterlace",
            [](vp::VideoProcessor& self, const py::buffer& frame) {
                const auto [frame_ptr, frame_size] = GetContiguousByteBuffer(frame);
                std::string output;
                {
                    py::gil_scoped_release release;
                    output = self.ProcessFrameNoDeinterlaceBuffer(frame_ptr, frame_size);
                }
                return py::bytes(output);
            },
            py::arg("frame"),
            "Process one UYVY frame while skipping Bob deinterlacing in this pass."
        )
        .def(
            "process_frame_no_deinterlace_into",
            [](vp::VideoProcessor& self, const py::buffer& frame, const py::buffer& output) {
                const auto [frame_ptr, frame_size] = GetContiguousByteBuffer(frame);
                const auto [out_ptr, out_size] = GetWritableContiguousByteBuffer(output, "output");

                std::string processed;
                {
                    py::gil_scoped_release release;
                    processed = self.ProcessFrameNoDeinterlaceBuffer(frame_ptr, frame_size);
                }
                return CopyOutputToWritableBuffer(processed, out_ptr, out_size, "output");
            },
            py::arg("frame"),
            py::arg("output"),
            "Process one frame with deinterlace disabled and write UYVY bytes into a writable output buffer."
        )
        .def(
            "process_frame_deinterlace_only",
            [](vp::VideoProcessor& self, const py::buffer& frame) {
                const auto [frame_ptr, frame_size] = GetContiguousByteBuffer(frame);
                std::string output;
                {
                    py::gil_scoped_release release;
                    output = self.ProcessFrameDeinterlaceOnlyBuffer(frame_ptr, frame_size);
                }
                return py::bytes(output);
            },
            py::arg("frame"),
            "Apply Bob deinterlacing and return deinterlaced UYVY bytes without ROI scaling."
        )
        .def(
            "process_frame_deinterlace_only_into",
            [](vp::VideoProcessor& self, const py::buffer& frame, const py::buffer& output) {
                const auto [frame_ptr, frame_size] = GetContiguousByteBuffer(frame);
                const auto [out_ptr, out_size] = GetWritableContiguousByteBuffer(output, "output");

                std::string processed;
                {
                    py::gil_scoped_release release;
                    processed = self.ProcessFrameDeinterlaceOnlyBuffer(frame_ptr, frame_size);
                }
                return CopyOutputToWritableBuffer(processed, out_ptr, out_size, "output");
            },
            py::arg("frame"),
            py::arg("output"),
            "Apply Bob deinterlacing and write UYVY bytes into a writable output buffer."
        )
        .def(
            "process_frame_preprocess_only",
            [](vp::VideoProcessor& self, const py::buffer& frame) {
                const auto [frame_ptr, frame_size] = GetContiguousByteBuffer(frame);
                std::string output;
                {
                    py::gil_scoped_release release;
                    output = self.ProcessFramePreprocessOnlyBuffer(frame_ptr, frame_size);
                }
                return py::bytes(output);
            },
            py::arg("frame"),
            "Apply enabled preprocess stages (deinterlace/denoise) and return UYVY bytes without ROI scaling."
        )
        .def(
            "process_frame_preprocess_only_into",
            [](vp::VideoProcessor& self, const py::buffer& frame, const py::buffer& output) {
                const auto [frame_ptr, frame_size] = GetContiguousByteBuffer(frame);
                const auto [out_ptr, out_size] = GetWritableContiguousByteBuffer(output, "output");

                std::string processed;
                {
                    py::gil_scoped_release release;
                    processed = self.ProcessFramePreprocessOnlyBuffer(frame_ptr, frame_size);
                }
                return CopyOutputToWritableBuffer(processed, out_ptr, out_size, "output");
            },
            py::arg("frame"),
            py::arg("output"),
            "Apply enabled preprocess stages and write UYVY bytes into a writable output buffer."
        )
        .def(
            "process_frame_preprocess_roi_rgb",
            [](vp::VideoProcessor& self,
               const py::buffer& frame,
               int roi_x,
               int roi_y,
               int roi_w,
               int roi_h,
               int out_w,
               int out_h) {
                const auto [frame_ptr, frame_size] = GetContiguousByteBuffer(frame);
                std::string output;
                {
                    py::gil_scoped_release release;
                    output = self.ProcessFramePreprocessRoiRgbBuffer(
                        frame_ptr,
                        frame_size,
                        roi_x,
                        roi_y,
                        roi_w,
                        roi_h,
                        out_w,
                        out_h
                    );
                }
                return py::bytes(output);
            },
            py::arg("frame"),
            py::arg("roi_x"),
            py::arg("roi_y"),
            py::arg("roi_w"),
            py::arg("roi_h"),
            py::arg("out_w"),
            py::arg("out_h"),
            "Apply native preprocess on GPU for the ROI and return RGB bytes sized to out_w/out_h."
        )
        .def(
            "process_frame_preprocess_roi_tensor_cuda",
            [](vp::VideoProcessor& self,
               const py::buffer& frame,
               int roi_x,
               int roi_y,
               int roi_w,
               int roi_h,
               int out_w,
               int out_h,
               const std::string& dtype) {
                const auto [frame_ptr, frame_size] = GetContiguousByteBuffer(frame);
                vp::CudaTensorBuffer output;
                {
                    py::gil_scoped_release release;
                    output = self.ProcessFramePreprocessRoiTensorCudaBuffer(
                        frame_ptr,
                        frame_size,
                        roi_x,
                        roi_y,
                        roi_w,
                        roi_h,
                        out_w,
                        out_h,
                        dtype
                    );
                }
                return output;
            },
            py::arg("frame"),
            py::arg("roi_x"),
            py::arg("roi_y"),
            py::arg("roi_w"),
            py::arg("roi_h"),
            py::arg("out_w"),
            py::arg("out_h"),
            py::arg("dtype") = "float16",
            "Apply native preprocess on GPU for the ROI and return a CUDA NCHW tensor buffer for direct ONNX input binding."
        )
        .def(
            "set_roi",
            &vp::VideoProcessor::SetRoi,
            py::arg("roi_x"),
            py::arg("roi_y"),
            py::arg("roi_w"),
            py::arg("roi_h"),
            "Set ROI rectangle; values are clamped to valid frame bounds."
        )
        .def(
            "set_roi_position",
            &vp::VideoProcessor::SetRoiPosition,
            py::arg("roi_x"),
            py::arg("roi_y"),
            "Set ROI position; size is preserved and full ROI is clamped."
        )
        .def(
            "set_roi_size",
            &vp::VideoProcessor::SetRoiSize,
            py::arg("roi_w"),
            py::arg("roi_h"),
            "Set ROI size; position is preserved and full ROI is clamped."
        )
        .def(
            "get_roi",
            [](const vp::VideoProcessor& self) {
                int roi_x = 0;
                int roi_y = 0;
                int roi_w = 0;
                int roi_h = 0;
                self.GetRoi(roi_x, roi_y, roi_w, roi_h);
                return py::make_tuple(roi_x, roi_y, roi_w, roi_h);
            },
            "Get current ROI as (roi_x, roi_y, roi_w, roi_h)."
        )
        .def("set_sr_mode_auto", &vp::VideoProcessor::SetSrModeAuto, "Enable auto SR scale mode.")
        .def("set_basic_scaling_mode_auto", &vp::VideoProcessor::SetSrModeAuto, "Enable auto basic-scaling mode.")
        .def(
            "set_sr_scale_manual",
            &vp::VideoProcessor::SetSrScaleManual,
            py::arg("sr_scale"),
            "Set manual SR scale to one of [2, 4, 8, 16]; may fall back on low memory."
        )
        .def(
            "set_basic_scaling_manual",
            &vp::VideoProcessor::SetSrScaleManual,
            py::arg("scale"),
            "Set manual basic scaling to one of [2, 4, 8, 16]; may fall back on low memory."
        )
        .def(
            "get_effective_sr_scale",
            &vp::VideoProcessor::GetEffectiveSrScale,
            "Get the currently active SR scale after any fallback."
        )
        .def(
            "get_effective_basic_scaling",
            &vp::VideoProcessor::GetEffectiveSrScale,
            "Get the currently active basic scaling after any fallback."
        )
        .def(
            "set_max_auto_sr_scale",
            &vp::VideoProcessor::SetMaxAutoSrScale,
            py::arg("sr_scale"),
            "Set the maximum allowed auto SR scale to one of [2, 4, 8, 16]."
        )
        .def(
            "set_max_auto_basic_scaling",
            &vp::VideoProcessor::SetMaxAutoSrScale,
            py::arg("scale"),
            "Set the maximum allowed auto basic scaling to one of [2, 4, 8, 16]."
        )
        .def(
            "get_max_auto_sr_scale",
            &vp::VideoProcessor::GetMaxAutoSrScale,
            "Get the configured maximum auto SR scale."
        )
        .def(
            "get_max_auto_basic_scaling",
            &vp::VideoProcessor::GetMaxAutoSrScale,
            "Get the configured maximum auto basic scaling."
        )
        .def(
            "set_sr_flavor",
            &vp::VideoProcessor::SetSrFlavorByName,
            py::arg("sr_flavor"),
            "Set SR flavor to one of [bilinear, bilinear_sharp, bicubic, bicubic_sharpen]."
        )
        .def(
            "set_basic_scaling_method",
            &vp::VideoProcessor::SetSrFlavorByName,
            py::arg("method"),
            "Set basic scaling method to one of [bilinear, bilinear_sharp, bicubic, bicubic_sharpen]."
        )
        .def(
            "get_sr_flavor",
            &vp::VideoProcessor::GetSrFlavorName,
            "Get current SR flavor name."
        )
        .def(
            "get_basic_scaling_method",
            &vp::VideoProcessor::GetSrFlavorName,
            "Get current basic scaling method name."
        )
        .def(
            "set_deinterlace_enabled",
            &vp::VideoProcessor::SetDeinterlaceEnabled,
            py::arg("enabled"),
            "Enable or disable Bob deinterlacing before ROI/crop processing."
        )
        .def(
            "is_deinterlace_enabled",
            &vp::VideoProcessor::IsDeinterlaceEnabled,
            "Return whether Bob deinterlacing is currently enabled."
        )
        .def(
            "set_deinterlace_method",
            &vp::VideoProcessor::SetDeinterlaceMethodByName,
            py::arg("method"),
            "Set deinterlace method to one of [bob, blend, edge_adaptive]."
        )
        .def(
            "get_deinterlace_method",
            &vp::VideoProcessor::GetDeinterlaceMethodName,
            "Get current deinterlace method name."
        )
        .def(
            "set_denoise_method",
            &vp::VideoProcessor::SetDenoiseMethodByName,
            py::arg("method"),
            "Set denoise method to one of [off, luma_gaussian3x3, luma_median3x3, luma_bilateral3x3, luma_bilateral5x5, field_temporal_luma]."
        )
        .def(
            "get_denoise_method",
            &vp::VideoProcessor::GetDenoiseMethodName,
            "Get current denoise method name."
        )
        .def(
            "set_denoise_strength",
            &vp::VideoProcessor::SetDenoiseStrength,
            py::arg("strength"),
            "Set denoise strength in [0.0, 1.0]."
        )
        .def(
            "get_denoise_strength",
            &vp::VideoProcessor::GetDenoiseStrength,
            "Get denoise strength in [0.0, 1.0]."
        )
        .def(
            "set_subpixel_shift",
            &vp::VideoProcessor::SetSubpixelShift,
            py::arg("shift_x"),
            py::arg("shift_y"),
            "Set native UYVY output subpixel shift in pixels."
        )
        .def(
            "set_color_space",
            &vp::VideoProcessor::SetColorSpaceByName,
            py::arg("color_space"),
            "Set color space to one of [rec709, rec2020_hlg]."
        )
        .def(
            "get_color_space",
            &vp::VideoProcessor::GetColorSpaceName,
            "Get current color space name."
        )
        .def(
            "set_color_range",
            &vp::VideoProcessor::SetColorRangeByName,
            py::arg("color_range"),
            "Set color range to one of [limited, full]."
        )
        .def(
            "get_color_range",
            &vp::VideoProcessor::GetColorRangeName,
            "Get current color range name."
        )
        .def(
            "get_subpixel_shift",
            [](const vp::VideoProcessor& self) {
                float shift_x = 0.0f;
                float shift_y = 0.0f;
                self.GetSubpixelShift(shift_x, shift_y);
                return py::make_tuple(shift_x, shift_y);
            },
            "Get native UYVY output subpixel shift as (shift_x, shift_y)."
        )
        .def_property_readonly("width", &vp::VideoProcessor::width)
        .def_property_readonly("height", &vp::VideoProcessor::height)
        .def_property_readonly("sr_scale", &vp::VideoProcessor::sr_scale)
        .def_property_readonly("sr_auto_mode", &vp::VideoProcessor::IsSrAutoMode)
        .def_property_readonly("effective_sr_scale", &vp::VideoProcessor::GetEffectiveSrScale)
        .def_property_readonly("basic_scaling", &vp::VideoProcessor::GetEffectiveSrScale)
        .def_property("sr_flavor", &vp::VideoProcessor::GetSrFlavorName, &vp::VideoProcessor::SetSrFlavorByName)
        .def_property("basic_scaling_method", &vp::VideoProcessor::GetSrFlavorName, &vp::VideoProcessor::SetSrFlavorByName)
        .def_property("max_auto_sr_scale", &vp::VideoProcessor::GetMaxAutoSrScale, &vp::VideoProcessor::SetMaxAutoSrScale)
        .def_property("max_auto_basic_scaling", &vp::VideoProcessor::GetMaxAutoSrScale, &vp::VideoProcessor::SetMaxAutoSrScale)
        .def_property("deinterlace_enabled", &vp::VideoProcessor::IsDeinterlaceEnabled, &vp::VideoProcessor::SetDeinterlaceEnabled)
        .def_property("deinterlace_method", &vp::VideoProcessor::GetDeinterlaceMethodName, &vp::VideoProcessor::SetDeinterlaceMethodByName)
        .def_property("denoise_method", &vp::VideoProcessor::GetDenoiseMethodName, &vp::VideoProcessor::SetDenoiseMethodByName)
        .def_property("denoise_strength", &vp::VideoProcessor::GetDenoiseStrength, &vp::VideoProcessor::SetDenoiseStrength)
        .def_property("color_space", &vp::VideoProcessor::GetColorSpaceName, &vp::VideoProcessor::SetColorSpaceByName)
        .def_property("color_range", &vp::VideoProcessor::GetColorRangeName, &vp::VideoProcessor::SetColorRangeByName)
        .def_property(
            "subpixel_shift",
            [](const vp::VideoProcessor& self) {
                float shift_x = 0.0f;
                float shift_y = 0.0f;
                self.GetSubpixelShift(shift_x, shift_y);
                return py::make_tuple(shift_x, shift_y);
            },
            [](vp::VideoProcessor& self, py::tuple value) {
                if (value.size() != 2) {
                    throw py::value_error("subpixel_shift expects a 2-item tuple (shift_x, shift_y)");
                }
                self.SetSubpixelShift(value[0].cast<float>(), value[1].cast<float>());
            }
        );

    py::class_<vp::AiSrCudaPostProcessor>(m, "AiSrCudaPostProcessor")
        .def(
            py::init<int, int, const std::string&, const std::string&>(),
            py::arg("output_width") = 1920,
            py::arg("output_height") = 1080,
            py::arg("color_space") = "rec709",
            py::arg("color_range") = "limited"
        )
        .def(
            "process_onnx_output_cuda",
            [](vp::AiSrCudaPostProcessor& self,
               std::uint64_t tensor_ptr,
               int tensor_width,
               int tensor_height,
               const std::string& method,
               const std::string& dtype,
               const std::string& layout,
               int channels,
               bool normalized_01) {
                const std::string dtype_name = [&dtype]() {
                    std::string out;
                    out.reserve(dtype.size());
                    for (char c : dtype) {
                        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                    }
                    return out;
                }();
                const std::string layout_name = [&layout]() {
                    std::string out;
                    out.reserve(layout.size());
                    for (char c : layout) {
                        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                    }
                    return out;
                }();

                vp::AiSrCudaPostProcessor::TensorDType tensor_dtype = vp::AiSrCudaPostProcessor::TensorDType::Float32;
                if (dtype_name == "float16" || dtype_name == "fp16") {
                    tensor_dtype = vp::AiSrCudaPostProcessor::TensorDType::Float16;
                } else if (dtype_name == "uint8" || dtype_name == "u8") {
                    tensor_dtype = vp::AiSrCudaPostProcessor::TensorDType::UInt8;
                } else if (dtype_name != "float32" && dtype_name != "fp32") {
                    throw py::value_error("dtype must be one of [float32, float16, uint8]");
                }

                vp::AiSrCudaPostProcessor::TensorLayout tensor_layout = vp::AiSrCudaPostProcessor::TensorLayout::NCHW;
                if (layout_name == "hwc" || layout_name == "nhwc") {
                    tensor_layout = vp::AiSrCudaPostProcessor::TensorLayout::HWC;
                } else if (layout_name != "chw" && layout_name != "nchw") {
                    throw py::value_error("layout must be one of [nchw, hwc]");
                }

                std::string output;
                {
                    py::gil_scoped_release release;
                    output = self.ProcessOnnxOutputCudaPtr(
                        tensor_ptr,
                        tensor_width,
                        tensor_height,
                        channels,
                        tensor_layout,
                        tensor_dtype,
                        normalized_01,
                        method
                    );
                }
                return py::bytes(output);
            },
            py::arg("tensor_ptr"),
            py::arg("tensor_width"),
            py::arg("tensor_height"),
            py::arg("method") = "bicubic",
            py::arg("dtype") = "float32",
            py::arg("layout") = "nchw",
            py::arg("channels") = 3,
            py::arg("normalized_01") = true,
            "Convert ONNX CUDA tensor output to UYVY bytes via GPU-only resize and color conversion."
        )
        .def("set_color_space", &vp::AiSrCudaPostProcessor::SetColorSpaceByName, py::arg("color_space"))
        .def("set_color_range", &vp::AiSrCudaPostProcessor::SetColorRangeByName, py::arg("color_range"))
        .def("get_color_space", &vp::AiSrCudaPostProcessor::GetColorSpaceName)
        .def("get_color_range", &vp::AiSrCudaPostProcessor::GetColorRangeName)
        .def(
            "set_post_denoise_method",
            &vp::AiSrCudaPostProcessor::SetPostDenoiseMethodByName,
            py::arg("method"),
            "Set AI SR postprocess denoise method to one of [off, luma_gaussian3x3, luma_median3x3, luma_bilateral3x3, luma_bilateral5x5]."
        )
        .def(
            "get_post_denoise_method",
            &vp::AiSrCudaPostProcessor::GetPostDenoiseMethodName,
            "Get AI SR postprocess denoise method name."
        )
        .def(
            "set_post_denoise_strength",
            &vp::AiSrCudaPostProcessor::SetPostDenoiseStrength,
            py::arg("strength"),
            "Set AI SR postprocess denoise strength in [0.0, 1.0]."
        )
        .def(
            "get_post_denoise_strength",
            &vp::AiSrCudaPostProcessor::GetPostDenoiseStrength,
            "Get AI SR postprocess denoise strength in [0.0, 1.0]."
        )
        .def(
            "set_post_artifact_reduction_method",
            &vp::AiSrCudaPostProcessor::SetPostArtifactReductionMethodByName,
            py::arg("method"),
            "Set AI SR postprocess artifact reduction method to one of [off, luma_bilateral3x3, luma_bilateral5x5]."
        )
        .def(
            "get_post_artifact_reduction_method",
            &vp::AiSrCudaPostProcessor::GetPostArtifactReductionMethodName,
            "Get AI SR postprocess artifact reduction method name."
        )
        .def(
            "set_post_artifact_reduction_strength",
            &vp::AiSrCudaPostProcessor::SetPostArtifactReductionStrength,
            py::arg("strength"),
            "Set AI SR postprocess artifact reduction strength in [0.0, 1.0]."
        )
        .def(
            "get_post_artifact_reduction_strength",
            &vp::AiSrCudaPostProcessor::GetPostArtifactReductionStrength,
            "Get AI SR postprocess artifact reduction strength in [0.0, 1.0]."
        )
        .def(
            "set_post_exaggeration_enabled",
            &vp::AiSrCudaPostProcessor::SetPostExaggerationEnabled,
            py::arg("enabled"),
            "Enable or disable exaggerated AI SR postprocess debug mode."
        )
        .def(
            "get_post_exaggeration_enabled",
            &vp::AiSrCudaPostProcessor::GetPostExaggerationEnabled,
            "Return whether exaggerated AI SR postprocess debug mode is enabled."
        )
        .def(
            "set_post_exaggeration_gain",
            &vp::AiSrCudaPostProcessor::SetPostExaggerationGain,
            py::arg("gain"),
            "Set exaggerated AI SR postprocess gain in [1.0, 4.0]."
        )
        .def(
            "get_post_exaggeration_gain",
            &vp::AiSrCudaPostProcessor::GetPostExaggerationGain,
            "Get exaggerated AI SR postprocess gain in [1.0, 4.0]."
        )
        .def_property_readonly("output_width", &vp::AiSrCudaPostProcessor::output_width)
        .def_property_readonly("output_height", &vp::AiSrCudaPostProcessor::output_height);
}
