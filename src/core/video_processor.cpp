#include "core/video_processor.hpp"

#include <algorithm>
#include <stdexcept>

#include "cuda/kernels.cuh"

namespace vp {
namespace {

constexpr int kExpectedWidth = 1920;
constexpr int kExpectedHeight = 1080;
constexpr int kUyvyBytesPerPixel = 2;

inline void CheckCuda(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(err));
    }
}

} // namespace

VideoProcessor::VideoProcessor(
    int width,
    int height,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    bool enable_placeholder_sr,
    int sr_scale
)
    : width_(width),
      height_(height),
      roi_x_(roi_x),
      roi_y_(roi_y),
      roi_w_(roi_w),
      roi_h_(roi_h),
      enable_placeholder_sr_(enable_placeholder_sr),
      sr_scale_(sr_scale),
      sr_width_(width),
      sr_height_(height),
      uyvy_bytes_(static_cast<size_t>(width) * static_cast<size_t>(height) * kUyvyBytesPerPixel),
      rgb_pixels_(static_cast<size_t>(width) * static_cast<size_t>(height)),
      stream_(nullptr),
      d_uyvy_in_(nullptr),
      d_uyvy_out_(nullptr),
      d_rgb_full_(nullptr),
      d_rgb_bob_(nullptr),
      d_rgb_sr_(nullptr),
      d_rgb_zoom_(nullptr) {
    ValidateConfiguration();
    ClampRoi();

    if (enable_placeholder_sr_) {
        sr_width_ = width_ * sr_scale_;
        sr_height_ = height_ * sr_scale_;
    }

    host_output_.resize(uyvy_bytes_);

    CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    InitializeBuffers();
}

VideoProcessor::~VideoProcessor() {
    Cleanup();
}

void VideoProcessor::ValidateConfiguration() const {
    if (width_ != kExpectedWidth || height_ != kExpectedHeight) {
        throw std::invalid_argument("Milestone 1 expects 1920x1080 UYVY frames.");
    }

    if (width_ <= 0 || height_ <= 0) {
        throw std::invalid_argument("Invalid frame dimensions.");
    }

    if (enable_placeholder_sr_ && (sr_scale_ < 2 || sr_scale_ > 4)) {
        throw std::invalid_argument("Placeholder SR scale must be in [2, 4].");
    }
}

void VideoProcessor::ClampRoi() {
    if (roi_w_ <= 0 || roi_h_ <= 0) {
        roi_x_ = 0;
        roi_y_ = 0;
        roi_w_ = width_;
        roi_h_ = height_;
    }

    roi_x_ = std::clamp(roi_x_, 0, width_ - 2);
    roi_y_ = std::clamp(roi_y_, 0, height_ - 2);

    roi_w_ = std::clamp(roi_w_, 2, width_ - roi_x_);
    roi_h_ = std::clamp(roi_h_, 2, height_ - roi_y_);

    // UYVY packs chroma for 2 horizontal pixels, so enforce even start and width.
    roi_x_ &= ~1;
    roi_w_ &= ~1;
    if (roi_w_ < 2) {
        roi_w_ = 2;
    }
}

void VideoProcessor::InitializeBuffers() {
    CheckCuda(cudaMalloc(&d_uyvy_in_, uyvy_bytes_), "cudaMalloc d_uyvy_in_");
    CheckCuda(cudaMalloc(&d_uyvy_out_, uyvy_bytes_), "cudaMalloc d_uyvy_out_");

    CheckCuda(cudaMalloc(&d_rgb_full_, rgb_pixels_ * sizeof(uchar3)), "cudaMalloc d_rgb_full_");
    CheckCuda(cudaMalloc(&d_rgb_bob_, rgb_pixels_ * sizeof(uchar3)), "cudaMalloc d_rgb_bob_");
    CheckCuda(cudaMalloc(&d_rgb_zoom_, rgb_pixels_ * sizeof(uchar3)), "cudaMalloc d_rgb_zoom_");

    if (enable_placeholder_sr_) {
        const size_t sr_pixels = static_cast<size_t>(sr_width_) * static_cast<size_t>(sr_height_);
        CheckCuda(cudaMalloc(&d_rgb_sr_, sr_pixels * sizeof(uchar3)), "cudaMalloc d_rgb_sr_");
    }
}

std::string VideoProcessor::ProcessFrame(const std::string& input_frame) {
    if (input_frame.size() != uyvy_bytes_) {
        throw std::invalid_argument("Invalid frame size; expected 1920*1080*2 bytes in UYVY.");
    }

    CheckCuda(
        cudaMemcpyAsync(d_uyvy_in_, input_frame.data(), uyvy_bytes_, cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync H2D"
    );

    cuda_kernels::LaunchUyvyToRgb(d_uyvy_in_, d_rgb_full_, width_, height_, stream_);
    cuda_kernels::LaunchBobDeinterlace(d_rgb_full_, d_rgb_bob_, width_, height_, stream_);

    const uchar3* crop_input = d_rgb_bob_;
    int crop_src_w = width_;
    int crop_src_h = height_;
    int crop_roi_x = roi_x_;
    int crop_roi_y = roi_y_;
    int crop_roi_w = roi_w_;
    int crop_roi_h = roi_h_;

    if (enable_placeholder_sr_) {
        cuda_kernels::LaunchUpscaleBicubic(
            d_rgb_bob_,
            width_,
            height_,
            d_rgb_sr_,
            sr_width_,
            sr_height_,
            stream_
        );

        crop_input = d_rgb_sr_;
        crop_src_w = sr_width_;
        crop_src_h = sr_height_;
        crop_roi_x = roi_x_ * sr_scale_;
        crop_roi_y = roi_y_ * sr_scale_;
        crop_roi_w = roi_w_ * sr_scale_;
        crop_roi_h = roi_h_ * sr_scale_;
    }

    cuda_kernels::LaunchCropZoomBilinear(
        crop_input,
        crop_src_w,
        crop_src_h,
        d_rgb_zoom_,
        width_,
        height_,
        crop_roi_x,
        crop_roi_y,
        crop_roi_w,
        crop_roi_h,
        stream_
    );

    cuda_kernels::LaunchRgbToUyvy(d_rgb_zoom_, d_uyvy_out_, width_, height_, stream_);

    CheckCuda(
        cudaMemcpyAsync(host_output_.data(), d_uyvy_out_, uyvy_bytes_, cudaMemcpyDeviceToHost, stream_),
        "cudaMemcpyAsync D2H"
    );

    CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

    return std::string(reinterpret_cast<const char*>(host_output_.data()), host_output_.size());
}

void VideoProcessor::Cleanup() {
    if (d_rgb_sr_ != nullptr) {
        cudaFree(d_rgb_sr_);
        d_rgb_sr_ = nullptr;
    }

    if (d_rgb_zoom_ != nullptr) {
        cudaFree(d_rgb_zoom_);
        d_rgb_zoom_ = nullptr;
    }

    if (d_rgb_bob_ != nullptr) {
        cudaFree(d_rgb_bob_);
        d_rgb_bob_ = nullptr;
    }

    if (d_rgb_full_ != nullptr) {
        cudaFree(d_rgb_full_);
        d_rgb_full_ = nullptr;
    }

    if (d_uyvy_out_ != nullptr) {
        cudaFree(d_uyvy_out_);
        d_uyvy_out_ = nullptr;
    }

    if (d_uyvy_in_ != nullptr) {
        cudaFree(d_uyvy_in_);
        d_uyvy_in_ = nullptr;
    }

    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

} // namespace vp
