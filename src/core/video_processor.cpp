#include "core/video_processor.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>

#include "cuda/kernels.cuh"

namespace vp {
namespace {

constexpr int kExpectedWidth = 1920;
constexpr int kExpectedHeight = 1080;
constexpr int kUyvyBytesPerPixel = 2;
constexpr std::array<int, 4> kSupportedSrScales = {16, 8, 4, 2};

inline void CheckCuda(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(err));
    }
}

inline bool IsSupportedSrScale(int sr_scale) {
    for (const int value : kSupportedSrScales) {
        if (value == sr_scale) {
            return true;
        }
    }
    return false;
}

inline int ClampToSupportedSrScale(int sr_scale) {
    for (const int value : kSupportedSrScales) {
        if (sr_scale >= value) {
            return value;
        }
    }
    return 2;
}

inline int SelectAutoSrScale(int width, int height, int roi_w, int roi_h, int max_auto_sr_scale) {
    const float rw = static_cast<float>(roi_w) / static_cast<float>(width);
    const float rh = static_cast<float>(roi_h) / static_cast<float>(height);
    const float ratio = std::max(rw, rh);

    const int capped_max = ClampToSupportedSrScale(max_auto_sr_scale);

    // For large ROIs, placeholder SR adds significant cost with limited benefit,
    // so auto mode can bypass SR entirely to preserve real-time throughput.
    if (ratio > 0.66f) {
        return 1;
    }

    int selected = 16;
    if (ratio > 0.5f) {
        selected = 2;
    } else if (ratio > 0.25f) {
        selected = 4;
    } else if (ratio > 0.125f) {
        selected = 8;
    }

    selected = std::min(selected, capped_max);
    return ClampToSupportedSrScale(selected);
}

inline const char* ToSrFlavorName(SrFlavor sr_flavor) {
    switch (sr_flavor) {
        case SrFlavor::Bilinear:
            return "bilinear";
        case SrFlavor::Bicubic:
            return "bicubic";
        case SrFlavor::BicubicSharpen:
            return "bicubic_sharpen";
    }
    return "bicubic";
}

inline const char* ToDeinterlaceMethodName(DeinterlaceMethod method) {
    switch (method) {
        case DeinterlaceMethod::Bob:
            return "bob";
        case DeinterlaceMethod::Blend:
            return "blend";
        case DeinterlaceMethod::EdgeAdaptive:
            return "edge_adaptive";
    }
    return "bob";
}

inline DeinterlaceMethod ParseDeinterlaceMethodName(const std::string& method_name) {
    std::string normalized;
    normalized.reserve(method_name.size());
    for (char c : method_name) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (normalized == "bob") {
        return DeinterlaceMethod::Bob;
    }
    if (normalized == "blend" || normalized == "weave_blend") {
        return DeinterlaceMethod::Blend;
    }
    if (normalized == "edge_adaptive" || normalized == "ela" || normalized == "edge") {
        return DeinterlaceMethod::EdgeAdaptive;
    }

    throw std::invalid_argument("Deinterlace method must be one of [bob, blend, edge_adaptive].");
}

inline const char* ToDenoiseMethodName(DenoiseMethod method) {
    switch (method) {
        case DenoiseMethod::Off:
            return "off";
        case DenoiseMethod::LumaGaussian3x3:
            return "luma_gaussian3x3";
        case DenoiseMethod::LumaMedian3x3:
            return "luma_median3x3";
        case DenoiseMethod::FieldTemporalLuma:
            return "field_temporal_luma";
    }
    return "off";
}

inline DenoiseMethod ParseDenoiseMethodName(const std::string& method_name) {
    std::string normalized;
    normalized.reserve(method_name.size());
    for (char c : method_name) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (normalized == "off" || normalized == "none") {
        return DenoiseMethod::Off;
    }
    if (normalized == "luma_gaussian3x3" || normalized == "gaussian" || normalized == "gaussian3x3") {
        return DenoiseMethod::LumaGaussian3x3;
    }
    if (normalized == "luma_median3x3" || normalized == "median" || normalized == "median3x3") {
        return DenoiseMethod::LumaMedian3x3;
    }
    if (normalized == "field_temporal_luma" || normalized == "temporal" || normalized == "field_temporal") {
        return DenoiseMethod::FieldTemporalLuma;
    }

    throw std::invalid_argument("Denoise method must be one of [off, luma_gaussian3x3, luma_median3x3, field_temporal_luma].");
}

inline SrFlavor ParseSrFlavorName(const std::string& sr_flavor_name) {
    std::string normalized;
    normalized.reserve(sr_flavor_name.size());
    for (char c : sr_flavor_name) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (normalized == "bilinear") {
        return SrFlavor::Bilinear;
    }
    if (normalized == "bicubic") {
        return SrFlavor::Bicubic;
    }
    if (normalized == "bicubic_sharpen" || normalized == "bicubic+sharpen") {
        return SrFlavor::BicubicSharpen;
    }

    throw std::invalid_argument("SR flavor must be one of [bilinear, bicubic, bicubic_sharpen].");
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
    enable_deinterlace_(true),
    deinterlace_method_(DeinterlaceMethod::Bob),
    denoise_method_(DenoiseMethod::Off),
    denoise_strength_(0.35f),
    sr_flavor_(SrFlavor::Bicubic),
    auto_sr_scale_(enable_placeholder_sr && sr_scale == 0),
    max_auto_sr_scale_(8),
    sr_requested_scale_(sr_scale),
      sr_scale_(sr_scale),
      sr_width_(width),
      sr_height_(height),
    sr_buffer_scale_capacity_(0),
      uyvy_bytes_(static_cast<size_t>(width) * static_cast<size_t>(height) * kUyvyBytesPerPixel),
      rgb_pixels_(static_cast<size_t>(width) * static_cast<size_t>(height)),
      stream_(nullptr),
      d_uyvy_in_(nullptr),
      d_uyvy_out_(nullptr),
      d_rgb_full_(nullptr),
      d_rgb_bob_(nullptr),
    d_rgb_denoise_(nullptr),
            d_rgb_prev_full_(nullptr),
      d_rgb_sr_(nullptr),
            d_rgb_zoom_(nullptr),
            has_prev_rgb_full_(false) {
    ValidateConfiguration();
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        ClampRoi();
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

    if (enable_placeholder_sr_ && sr_scale_ != 0 && !IsSupportedSrScale(sr_scale_)) {
        throw std::invalid_argument("Placeholder SR scale must be 0(auto) or one of [2, 4, 8, 16].");
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

void VideoProcessor::SetRoi(int roi_x, int roi_y, int roi_w, int roi_h) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    roi_x_ = roi_x;
    roi_y_ = roi_y;
    roi_w_ = roi_w;
    roi_h_ = roi_h;
    ClampRoi();
}

void VideoProcessor::SetRoiPosition(int roi_x, int roi_y) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    roi_x_ = roi_x;
    roi_y_ = roi_y;
    ClampRoi();
}

void VideoProcessor::SetRoiSize(int roi_w, int roi_h) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    roi_w_ = roi_w;
    roi_h_ = roi_h;
    ClampRoi();
}

void VideoProcessor::GetRoi(int& roi_x, int& roi_y, int& roi_w, int& roi_h) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    roi_x = roi_x_;
    roi_y = roi_y_;
    roi_w = roi_w_;
    roi_h = roi_h_;
}

void VideoProcessor::SetSrModeAuto() {
    if (!enable_placeholder_sr_) {
        throw std::runtime_error("Placeholder SR is disabled.");
    }

    std::lock_guard<std::mutex> process_lock(process_mutex_);
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize SetSrModeAuto");
    ConfigureSrScaleLocked(0, true);
}

void VideoProcessor::SetMaxAutoSrScale(int sr_scale) {
    if (!IsSupportedSrScale(sr_scale)) {
        throw std::invalid_argument("Max auto SR scale must be one of [2, 4, 8, 16].");
    }

    std::lock_guard<std::mutex> process_lock(process_mutex_);
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    max_auto_sr_scale_ = sr_scale;
    if (enable_placeholder_sr_ && auto_sr_scale_) {
        CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize SetMaxAutoSrScale");
        ConfigureSrScaleLocked(0, true);
    }
}

int VideoProcessor::GetMaxAutoSrScale() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return max_auto_sr_scale_;
}

void VideoProcessor::SetSrFlavor(SrFlavor sr_flavor) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    sr_flavor_ = sr_flavor;
}

void VideoProcessor::SetSrFlavorByName(const std::string& sr_flavor_name) {
    SetSrFlavor(ParseSrFlavorName(sr_flavor_name));
}

SrFlavor VideoProcessor::GetSrFlavor() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return sr_flavor_;
}

std::string VideoProcessor::GetSrFlavorName() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return ToSrFlavorName(sr_flavor_);
}

void VideoProcessor::SetSrScaleManual(int sr_scale) {
    if (!enable_placeholder_sr_) {
        throw std::runtime_error("Placeholder SR is disabled.");
    }
    if (!IsSupportedSrScale(sr_scale)) {
        throw std::invalid_argument("Manual SR scale must be one of [2, 4, 8, 16].");
    }

    std::lock_guard<std::mutex> process_lock(process_mutex_);
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize SetSrScaleManual");
    ConfigureSrScaleLocked(sr_scale, false);
}

int VideoProcessor::GetEffectiveSrScale() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return sr_scale_;
}

bool VideoProcessor::IsSrAutoMode() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return auto_sr_scale_;
}

void VideoProcessor::SetDeinterlaceEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    enable_deinterlace_ = enabled;
}

bool VideoProcessor::IsDeinterlaceEnabled() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return enable_deinterlace_;
}

void VideoProcessor::SetDeinterlaceMethod(DeinterlaceMethod method) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    deinterlace_method_ = method;
}

void VideoProcessor::SetDeinterlaceMethodByName(const std::string& method_name) {
    SetDeinterlaceMethod(ParseDeinterlaceMethodName(method_name));
}

DeinterlaceMethod VideoProcessor::GetDeinterlaceMethod() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return deinterlace_method_;
}

std::string VideoProcessor::GetDeinterlaceMethodName() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return ToDeinterlaceMethodName(deinterlace_method_);
}

void VideoProcessor::SetDenoiseMethod(DenoiseMethod method) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    denoise_method_ = method;
}

void VideoProcessor::SetDenoiseMethodByName(const std::string& method_name) {
    SetDenoiseMethod(ParseDenoiseMethodName(method_name));
}

DenoiseMethod VideoProcessor::GetDenoiseMethod() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return denoise_method_;
}

std::string VideoProcessor::GetDenoiseMethodName() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return ToDenoiseMethodName(denoise_method_);
}

void VideoProcessor::SetDenoiseStrength(float strength) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    denoise_strength_ = std::clamp(strength, 0.0f, 1.0f);
}

float VideoProcessor::GetDenoiseStrength() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return denoise_strength_;
}

int VideoProcessor::sr_scale() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return sr_scale_;
}

bool VideoProcessor::EnsureSrBufferCapacityLocked(int target_scale, cudaError_t& last_error) {
    if (d_rgb_sr_ != nullptr && sr_buffer_scale_capacity_ >= target_scale) {
        return true;
    }

    const int candidate_w = width_ * target_scale;
    const int candidate_h = height_ * target_scale;
    const size_t sr_pixels = static_cast<size_t>(candidate_w) * static_cast<size_t>(candidate_h);

    uchar3* new_buffer = nullptr;
    const cudaError_t err = cudaMalloc(&new_buffer, sr_pixels * sizeof(uchar3));
    if (err != cudaSuccess) {
        last_error = err;
        return false;
    }

    if (d_rgb_sr_ != nullptr) {
        cudaFree(d_rgb_sr_);
    }

    d_rgb_sr_ = new_buffer;
    sr_buffer_scale_capacity_ = target_scale;
    return true;
}

void VideoProcessor::ConfigureSrScaleLocked(int requested_scale, bool auto_mode) {
    int effective_requested_scale = requested_scale;
    if (auto_mode) {
        effective_requested_scale = SelectAutoSrScale(width_, height_, roi_w_, roi_h_, max_auto_sr_scale_);
    }

    if (effective_requested_scale == 1) {
        auto_sr_scale_ = auto_mode;
        sr_requested_scale_ = auto_mode ? 0 : requested_scale;
        sr_scale_ = 1;
        sr_width_ = width_;
        sr_height_ = height_;
        return;
    }

    if (!IsSupportedSrScale(effective_requested_scale)) {
        throw std::invalid_argument("SR scale must resolve to one of [2, 4, 8, 16].");
    }

    cudaError_t last_error = cudaSuccess;

    for (const int candidate_scale : kSupportedSrScales) {
        if (candidate_scale > effective_requested_scale) {
            continue;
        }

        if (EnsureSrBufferCapacityLocked(candidate_scale, last_error)) {
            auto_sr_scale_ = auto_mode;
            sr_requested_scale_ = auto_mode ? 0 : requested_scale;
            sr_scale_ = candidate_scale;
            sr_width_ = width_ * candidate_scale;
            sr_height_ = height_ * candidate_scale;
            return;
        }

        // Continue fallback ladder only for allocation pressure.
        if (last_error != cudaErrorMemoryAllocation) {
            break;
        }
    }

    throw std::runtime_error(
        std::string("cudaMalloc d_rgb_sr_ failed: ") + cudaGetErrorString(last_error)
    );
}

void VideoProcessor::InitializeBuffers() {
    CheckCuda(cudaMalloc(&d_uyvy_in_, uyvy_bytes_), "cudaMalloc d_uyvy_in_");
    CheckCuda(cudaMalloc(&d_uyvy_out_, uyvy_bytes_), "cudaMalloc d_uyvy_out_");

    CheckCuda(cudaMalloc(&d_rgb_full_, rgb_pixels_ * sizeof(uchar3)), "cudaMalloc d_rgb_full_");
    CheckCuda(cudaMalloc(&d_rgb_bob_, rgb_pixels_ * sizeof(uchar3)), "cudaMalloc d_rgb_bob_");
    CheckCuda(cudaMalloc(&d_rgb_denoise_, rgb_pixels_ * sizeof(uchar3)), "cudaMalloc d_rgb_denoise_");
    CheckCuda(cudaMalloc(&d_rgb_prev_full_, rgb_pixels_ * sizeof(uchar3)), "cudaMalloc d_rgb_prev_full_");
    CheckCuda(cudaMalloc(&d_rgb_zoom_, rgb_pixels_ * sizeof(uchar3)), "cudaMalloc d_rgb_zoom_");

    if (enable_placeholder_sr_) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        ConfigureSrScaleLocked(sr_requested_scale_, auto_sr_scale_);
    }
}

std::string VideoProcessor::ProcessFrame(const std::string& input_frame) {
    return ProcessFrameInternal(input_frame, false, false, false);
}

std::string VideoProcessor::ProcessFrameNoDeinterlace(const std::string& input_frame) {
    return ProcessFrameInternal(input_frame, false, false, true);
}

std::string VideoProcessor::ProcessFrameDeinterlaceOnly(const std::string& input_frame) {
    return ProcessFrameInternal(input_frame, true, true, false);
}

std::string VideoProcessor::ProcessFramePreprocessOnly(const std::string& input_frame) {
    return ProcessFrameInternal(input_frame, true, false, false);
}

std::string VideoProcessor::ProcessFrameInternal(
    const std::string& input_frame,
    bool deinterlace_only,
    bool force_deinterlace,
    bool force_disable_deinterlace
) {
    std::lock_guard<std::mutex> process_lock(process_mutex_);

    if (input_frame.size() != uyvy_bytes_) {
        throw std::invalid_argument("Invalid frame size; expected 1920*1080*2 bytes in UYVY.");
    }

    int roi_x = 0;
    int roi_y = 0;
    int roi_w = 0;
    int roi_h = 0;
    int sr_scale = 1;
    SrFlavor sr_flavor = SrFlavor::Bicubic;
    int sr_width = width_;
    int sr_height = height_;
    bool deinterlace_enabled = true;
    DeinterlaceMethod deinterlace_method = DeinterlaceMethod::Bob;
    DenoiseMethod denoise_method = DenoiseMethod::Off;
    float denoise_strength = 0.0f;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (enable_placeholder_sr_ && auto_sr_scale_) {
            const int desired_scale = SelectAutoSrScale(width_, height_, roi_w_, roi_h_, max_auto_sr_scale_);
            if (desired_scale != sr_scale_) {
                ConfigureSrScaleLocked(0, true);
            }
        }

        roi_x = roi_x_;
        roi_y = roi_y_;
        roi_w = roi_w_;
        roi_h = roi_h_;
        sr_scale = sr_scale_;
        sr_flavor = sr_flavor_;
        sr_width = sr_width_;
        sr_height = sr_height_;
        deinterlace_enabled = enable_deinterlace_;
        deinterlace_method = deinterlace_method_;
        denoise_method = denoise_method_;
        denoise_strength = denoise_strength_;

        if (force_deinterlace) {
            deinterlace_enabled = true;
        }
        if (force_disable_deinterlace) {
            deinterlace_enabled = false;
        }
    }

    CheckCuda(
        cudaMemcpyAsync(d_uyvy_in_, input_frame.data(), uyvy_bytes_, cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync H2D"
    );

    cuda_kernels::LaunchUyvyToRgb(d_uyvy_in_, d_rgb_full_, width_, height_, stream_);

    const uchar3* crop_input = d_rgb_full_;
    int crop_src_w = width_;
    int crop_src_h = height_;
    int crop_roi_x = roi_x;
    int crop_roi_y = roi_y;
    int crop_roi_w = roi_w;
    int crop_roi_h = roi_h;

    if (denoise_method == DenoiseMethod::FieldTemporalLuma && denoise_strength > 0.001f) {
        if (has_prev_rgb_full_) {
            cuda_kernels::LaunchDenoiseFieldTemporalLuma(
                d_rgb_full_,
                d_rgb_prev_full_,
                d_rgb_denoise_,
                width_,
                height_,
                denoise_strength,
                stream_
            );
        } else {
            CheckCuda(
                cudaMemcpyAsync(
                    d_rgb_denoise_,
                    d_rgb_full_,
                    rgb_pixels_ * sizeof(uchar3),
                    cudaMemcpyDeviceToDevice,
                    stream_
                ),
                "cudaMemcpyAsync D2D field temporal warmup"
            );
        }
        crop_input = d_rgb_denoise_;
    }

    if (deinterlace_enabled) {
        switch (deinterlace_method) {
            case DeinterlaceMethod::Blend:
                cuda_kernels::LaunchBlendDeinterlace(crop_input, d_rgb_bob_, width_, height_, stream_);
                break;
            case DeinterlaceMethod::EdgeAdaptive:
                cuda_kernels::LaunchEdgeAdaptiveDeinterlace(crop_input, d_rgb_bob_, width_, height_, stream_);
                break;
            case DeinterlaceMethod::Bob:
            default:
                cuda_kernels::LaunchBobDeinterlace(crop_input, d_rgb_bob_, width_, height_, stream_);
                break;
        }
        crop_input = d_rgb_bob_;
    }

    if (denoise_method != DenoiseMethod::Off && denoise_method != DenoiseMethod::FieldTemporalLuma && denoise_strength > 0.001f) {
        if (denoise_method == DenoiseMethod::LumaMedian3x3) {
            cuda_kernels::LaunchDenoiseLumaMedian3x3(crop_input, d_rgb_denoise_, width_, height_, denoise_strength, stream_);
        } else {
            cuda_kernels::LaunchDenoiseLumaGaussian3x3(crop_input, d_rgb_denoise_, width_, height_, denoise_strength, stream_);
        }
        crop_input = d_rgb_denoise_;
    }

    if (deinterlace_only) {
        cuda_kernels::LaunchRgbToUyvy(crop_input, d_uyvy_out_, width_, height_, stream_);

        CheckCuda(
            cudaMemcpyAsync(host_output_.data(), d_uyvy_out_, uyvy_bytes_, cudaMemcpyDeviceToHost, stream_),
            "cudaMemcpyAsync D2H"
        );

        CheckCuda(
            cudaMemcpyAsync(
                d_rgb_prev_full_,
                d_rgb_full_,
                rgb_pixels_ * sizeof(uchar3),
                cudaMemcpyDeviceToDevice,
                stream_
            ),
            "cudaMemcpyAsync D2D update prev rgb"
        );
        has_prev_rgb_full_ = true;

        CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

        return std::string(reinterpret_cast<const char*>(host_output_.data()), host_output_.size());
    }

    if (enable_placeholder_sr_ && sr_scale > 1) {
        int sr_roi_w = std::max(2, roi_w * sr_scale);
        int sr_roi_h = std::max(2, roi_h * sr_scale);

        // Avoid building very large intermediates that are immediately downscaled
        // back to output resolution; this is a major cost on mobile GPUs.
        sr_roi_w = std::min(sr_roi_w, width_);
        sr_roi_h = std::min(sr_roi_h, height_);

        const bool sr_pass_is_redundant = (sr_roi_w <= roi_w) && (sr_roi_h <= roi_h);
        if (!sr_pass_is_redundant) {
            const uchar3* sr_output = d_rgb_sr_;

            // Upscale only the selected ROI region rather than the full frame.
            switch (sr_flavor) {
                case SrFlavor::Bilinear:
                    cuda_kernels::LaunchCropZoomBilinear(
                        crop_input,
                        width_,
                        height_,
                        d_rgb_sr_,
                        sr_roi_w,
                        sr_roi_h,
                        roi_x,
                        roi_y,
                        roi_w,
                        roi_h,
                        stream_
                    );
                    break;
                case SrFlavor::Bicubic:
                    cuda_kernels::LaunchCropZoomBicubic(
                        crop_input,
                        width_,
                        height_,
                        d_rgb_sr_,
                        sr_roi_w,
                        sr_roi_h,
                        roi_x,
                        roi_y,
                        roi_w,
                        roi_h,
                        stream_
                    );
                    break;
                case SrFlavor::BicubicSharpen:
                    cuda_kernels::LaunchCropZoomBicubic(
                        crop_input,
                        width_,
                        height_,
                        d_rgb_sr_,
                        sr_roi_w,
                        sr_roi_h,
                        roi_x,
                        roi_y,
                        roi_w,
                        roi_h,
                        stream_
                    );
                    // Reuse d_rgb_bob_ as scratch for sharpened ROI before final zoom-to-output.
                    cuda_kernels::LaunchSharpen3x3(
                        d_rgb_sr_,
                        d_rgb_bob_,
                        sr_roi_w,
                        sr_roi_h,
                        stream_
                    );
                    sr_output = d_rgb_bob_;
                    break;
            }

            crop_input = sr_output;
            crop_src_w = sr_roi_w;
            crop_src_h = sr_roi_h;
            crop_roi_x = 0;
            crop_roi_y = 0;
            crop_roi_w = sr_roi_w;
            crop_roi_h = sr_roi_h;
        }
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

    CheckCuda(
        cudaMemcpyAsync(
            d_rgb_prev_full_,
            d_rgb_full_,
            rgb_pixels_ * sizeof(uchar3),
            cudaMemcpyDeviceToDevice,
            stream_
        ),
        "cudaMemcpyAsync D2D update prev rgb"
    );
    has_prev_rgb_full_ = true;

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

    if (d_rgb_denoise_ != nullptr) {
        cudaFree(d_rgb_denoise_);
        d_rgb_denoise_ = nullptr;
    }

    if (d_rgb_prev_full_ != nullptr) {
        cudaFree(d_rgb_prev_full_);
        d_rgb_prev_full_ = nullptr;
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
