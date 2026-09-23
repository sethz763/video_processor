#include "core/ai_sr_cuda_post_processor.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>

#include "cuda/kernels.cuh"

namespace vp {
namespace {

inline void CheckCuda(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(err));
    }
}

inline std::string NormalizeName(const std::string& value) {
    std::string normalized;
    normalized.reserve(value.size());
    for (char c : value) {
        if (c == ' ' || c == '-') {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return normalized;
}

inline AiSrCudaPostProcessor::ColorSpace ParseColorSpaceName(const std::string& color_space_name) {
    const std::string normalized = NormalizeName(color_space_name);
    if (normalized == "rec709" || normalized == "rec_709" || normalized == "bt709") {
        return AiSrCudaPostProcessor::ColorSpace::Rec709;
    }
    if (normalized == "rec2020_hlg" || normalized == "rec2020hlg" || normalized == "bt2020_hlg" || normalized == "bt2020hlg") {
        return AiSrCudaPostProcessor::ColorSpace::Rec2020Hlg;
    }
    throw std::invalid_argument("Color space must be one of [rec709, rec2020_hlg].");
}

inline AiSrCudaPostProcessor::ColorRange ParseColorRangeName(const std::string& color_range_name) {
    const std::string normalized = NormalizeName(color_range_name);
    if (normalized == "limited" || normalized == "video") {
        return AiSrCudaPostProcessor::ColorRange::Limited;
    }
    if (normalized == "full" || normalized == "data" || normalized == "pc") {
        return AiSrCudaPostProcessor::ColorRange::Full;
    }
    throw std::invalid_argument("Color range must be one of [limited, full].");
}

inline const char* ToColorSpaceName(AiSrCudaPostProcessor::ColorSpace color_space) {
    switch (color_space) {
        case AiSrCudaPostProcessor::ColorSpace::Rec2020Hlg:
            return "rec2020_hlg";
        case AiSrCudaPostProcessor::ColorSpace::Rec709:
        default:
            return "rec709";
    }
}

inline const char* ToColorRangeName(AiSrCudaPostProcessor::ColorRange color_range) {
    switch (color_range) {
        case AiSrCudaPostProcessor::ColorRange::Full:
            return "full";
        case AiSrCudaPostProcessor::ColorRange::Limited:
        default:
            return "limited";
    }
}

inline int ToColorMatrixId(AiSrCudaPostProcessor::ColorSpace color_space) {
    return color_space == AiSrCudaPostProcessor::ColorSpace::Rec2020Hlg ? 1 : 0;
}

inline int ToColorRangeId(AiSrCudaPostProcessor::ColorRange color_range) {
    return color_range == AiSrCudaPostProcessor::ColorRange::Full ? 1 : 0;
}

inline AiSrCudaPostProcessor::ResizeMethod ParseResizeMethod(const std::string& method_name) {
    const std::string normalized = NormalizeName(method_name);
    if (normalized == "bilinear") {
        return AiSrCudaPostProcessor::ResizeMethod::Bilinear;
    }
    if (normalized == "bilinear_sharp" || normalized == "bilinear+sharp") {
        return AiSrCudaPostProcessor::ResizeMethod::BilinearSharp;
    }
    if (normalized == "bicubic_sharpen" || normalized == "bicubic+sharpen") {
        return AiSrCudaPostProcessor::ResizeMethod::BicubicSharpen;
    }
    // Map Lanczos requests to bicubic for GPU-native execution.
    if (normalized == "lanczos") {
        return AiSrCudaPostProcessor::ResizeMethod::Bicubic;
    }
    return AiSrCudaPostProcessor::ResizeMethod::Bicubic;
}

}  // namespace

AiSrCudaPostProcessor::AiSrCudaPostProcessor(
    int output_width,
    int output_height,
    const std::string& color_space,
    const std::string& color_range
)
    : output_width_(output_width),
      output_height_(output_height),
      output_uyvy_bytes_(0),
      color_space_(ParseColorSpaceName(color_space)),
      color_range_(ParseColorRangeName(color_range)),
      stream_(nullptr),
      d_tensor_rgb_(nullptr),
      d_tensor_rgb_capacity_pixels_(0),
      d_output_rgb_(nullptr),
      d_output_tmp_(nullptr),
      d_output_uyvy_(nullptr),
      h_output_pinned_(nullptr) {
    if (output_width_ <= 0 || output_height_ <= 0) {
        throw std::invalid_argument("Output dimensions must be positive.");
    }
    if ((output_width_ & 1) != 0) {
        throw std::invalid_argument("Output width must be even for UYVY output.");
    }

    output_uyvy_bytes_ = static_cast<std::size_t>(output_width_) * static_cast<std::size_t>(output_height_) * 2u;

    CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    InitializeBuffers();
}

AiSrCudaPostProcessor::~AiSrCudaPostProcessor() {
    Cleanup();
}

void AiSrCudaPostProcessor::InitializeBuffers() {
    const std::size_t output_pixels = static_cast<std::size_t>(output_width_) * static_cast<std::size_t>(output_height_);

    CheckCuda(cudaMalloc(&d_output_rgb_, output_pixels * sizeof(uchar3)), "cudaMalloc d_output_rgb_");
    CheckCuda(cudaMalloc(&d_output_tmp_, output_pixels * sizeof(uchar3)), "cudaMalloc d_output_tmp_");
    CheckCuda(cudaMalloc(&d_output_uyvy_, output_uyvy_bytes_), "cudaMalloc d_output_uyvy_");

    if (cudaHostAlloc(&h_output_pinned_, output_uyvy_bytes_, cudaHostAllocDefault) != cudaSuccess) {
        h_output_pinned_ = nullptr;
        host_output_.resize(output_uyvy_bytes_);
    }
}

void AiSrCudaPostProcessor::EnsureTensorRgbCapacityLocked(int tensor_width, int tensor_height) {
    if (tensor_width <= 0 || tensor_height <= 0) {
        throw std::invalid_argument("Tensor dimensions must be positive.");
    }

    const std::size_t tensor_pixels = static_cast<std::size_t>(tensor_width) * static_cast<std::size_t>(tensor_height);
    if (d_tensor_rgb_ != nullptr && d_tensor_rgb_capacity_pixels_ >= tensor_pixels) {
        return;
    }

    uchar3* new_buffer = nullptr;
    CheckCuda(cudaMalloc(&new_buffer, tensor_pixels * sizeof(uchar3)), "cudaMalloc d_tensor_rgb_");

    if (d_tensor_rgb_ != nullptr) {
        cudaFree(d_tensor_rgb_);
    }

    d_tensor_rgb_ = new_buffer;
    d_tensor_rgb_capacity_pixels_ = tensor_pixels;
}

std::string AiSrCudaPostProcessor::ProcessOnnxOutputCudaPtr(
    std::uint64_t tensor_ptr,
    int tensor_width,
    int tensor_height,
    int channels,
    TensorLayout layout,
    TensorDType dtype,
    bool normalized_01,
    const std::string& resize_method
) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (tensor_ptr == 0) {
        throw std::invalid_argument("ONNX tensor pointer is null.");
    }
    if (channels < 1 || channels > 4) {
        throw std::invalid_argument("ONNX tensor channels must be in [1, 4].");
    }

    EnsureTensorRgbCapacityLocked(tensor_width, tensor_height);

    cuda_kernels::LaunchTensorToRgb(
        reinterpret_cast<const void*>(tensor_ptr),
        static_cast<int>(dtype),
        static_cast<int>(layout),
        channels,
        normalized_01,
        d_tensor_rgb_,
        tensor_width,
        tensor_height,
        stream_
    );

    const ResizeMethod method = ParseResizeMethod(resize_method);
    const uchar3* final_rgb = d_output_rgb_;

    switch (method) {
        case ResizeMethod::Bilinear:
            cuda_kernels::LaunchCropZoomBilinear(
                d_tensor_rgb_,
                tensor_width,
                tensor_height,
                d_output_rgb_,
                output_width_,
                output_height_,
                0,
                0,
                tensor_width,
                tensor_height,
                stream_
            );
            break;
        case ResizeMethod::BilinearSharp:
            cuda_kernels::LaunchCropZoomBilinearSharp(
                d_tensor_rgb_,
                tensor_width,
                tensor_height,
                d_output_rgb_,
                output_width_,
                output_height_,
                0,
                0,
                tensor_width,
                tensor_height,
                stream_
            );
            break;
        case ResizeMethod::Bicubic:
            cuda_kernels::LaunchCropZoomBicubic(
                d_tensor_rgb_,
                tensor_width,
                tensor_height,
                d_output_rgb_,
                output_width_,
                output_height_,
                0,
                0,
                tensor_width,
                tensor_height,
                stream_
            );
            break;
        case ResizeMethod::BicubicSharpen:
            cuda_kernels::LaunchCropZoomBicubic(
                d_tensor_rgb_,
                tensor_width,
                tensor_height,
                d_output_rgb_,
                output_width_,
                output_height_,
                0,
                0,
                tensor_width,
                tensor_height,
                stream_
            );
            cuda_kernels::LaunchSharpen3x3(
                d_output_rgb_,
                d_output_tmp_,
                output_width_,
                output_height_,
                false,
                stream_
            );
            final_rgb = d_output_tmp_;
            break;
    }

    const int color_matrix = ToColorMatrixId(color_space_);
    const int color_range = ToColorRangeId(color_range_);
    cuda_kernels::LaunchRgbToUyvy(final_rgb, d_output_uyvy_, output_width_, output_height_, color_matrix, color_range, stream_);

    uint8_t* host_output_ptr = h_output_pinned_ != nullptr ? h_output_pinned_ : host_output_.data();
    CheckCuda(
        cudaMemcpyAsync(host_output_ptr, d_output_uyvy_, output_uyvy_bytes_, cudaMemcpyDeviceToHost, stream_),
        "cudaMemcpyAsync D2H ai sr postprocess"
    );
    CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize ai sr postprocess");

    return std::string(reinterpret_cast<const char*>(host_output_ptr), output_uyvy_bytes_);
}

void AiSrCudaPostProcessor::SetColorSpaceByName(const std::string& color_space_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    color_space_ = ParseColorSpaceName(color_space_name);
}

void AiSrCudaPostProcessor::SetColorRangeByName(const std::string& color_range_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    color_range_ = ParseColorRangeName(color_range_name);
}

std::string AiSrCudaPostProcessor::GetColorSpaceName() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ToColorSpaceName(color_space_);
}

std::string AiSrCudaPostProcessor::GetColorRangeName() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ToColorRangeName(color_range_);
}

void AiSrCudaPostProcessor::Cleanup() {
    if (h_output_pinned_ != nullptr) {
        cudaFreeHost(h_output_pinned_);
        h_output_pinned_ = nullptr;
    }

    if (d_output_uyvy_ != nullptr) {
        cudaFree(d_output_uyvy_);
        d_output_uyvy_ = nullptr;
    }

    if (d_output_tmp_ != nullptr) {
        cudaFree(d_output_tmp_);
        d_output_tmp_ = nullptr;
    }

    if (d_output_rgb_ != nullptr) {
        cudaFree(d_output_rgb_);
        d_output_rgb_ = nullptr;
    }

    if (d_tensor_rgb_ != nullptr) {
        cudaFree(d_tensor_rgb_);
        d_tensor_rgb_ = nullptr;
    }

    d_tensor_rgb_capacity_pixels_ = 0;

    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

}  // namespace vp
