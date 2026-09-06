#include "core/video_processor.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <stdexcept>

#include "cuda/kernels.cuh"

namespace vp {
namespace {

constexpr int kExpectedWidth = 1920;
constexpr int kExpectedHeight = 1080;
constexpr int kUyvyBytesPerPixel = 2;
constexpr size_t kRgbBytesPerPixel = sizeof(uchar3);
constexpr std::array<int, 4> kSupportedSrScales = {16, 8, 4, 2};
constexpr int kAutoSrScaleSettleFrames = 6;
constexpr float kSubpixelShiftEpsilon = 1e-4f;

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

    // Full-frame (or effectively full-frame) ROI should remain 1x so
    // reset/startup framing does not appear zoomed.
    if (ratio >= 0.98f) {
        return 1;
    }

    // For non-full ROIs, keep auto mode visibly active with a conservative 2x
    // baseline on large crops.
    if (ratio > 0.66f) {
        return 2;
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

inline bool HasSubpixelShift(float shift_x, float shift_y) {
    return std::fabs(shift_x) >= kSubpixelShiftEpsilon || std::fabs(shift_y) >= kSubpixelShiftEpsilon;
}

inline const char* ToSrFlavorName(SrFlavor sr_flavor) {
    switch (sr_flavor) {
        case SrFlavor::Bilinear:
            return "bilinear";
        case SrFlavor::BilinearSharp:
            return "bilinear_sharp";
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
        case DenoiseMethod::LumaBilateral3x3:
            return "luma_bilateral3x3";
        case DenoiseMethod::LumaBilateral5x5:
            return "luma_bilateral5x5";
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
    if (normalized == "luma_bilateral3x3" || normalized == "bilateral" || normalized == "bilateral3x3") {
        return DenoiseMethod::LumaBilateral3x3;
    }
    if (normalized == "luma_bilateral5x5" || normalized == "bilateral5x5" || normalized == "artifact_reduce") {
        return DenoiseMethod::LumaBilateral5x5;
    }
    if (normalized == "field_temporal_luma" || normalized == "temporal" || normalized == "field_temporal") {
        return DenoiseMethod::FieldTemporalLuma;
    }

    throw std::invalid_argument("Denoise method must be one of [off, luma_gaussian3x3, luma_median3x3, luma_bilateral3x3, luma_bilateral5x5, field_temporal_luma].");
}

inline const char* ToColorSpaceName(ColorSpace color_space) {
    switch (color_space) {
        case ColorSpace::Rec709:
            return "rec709";
        case ColorSpace::Rec2020Hlg:
            return "rec2020_hlg";
    }
    return "rec709";
}

inline ColorSpace ParseColorSpaceName(const std::string& color_space_name) {
    std::string normalized;
    normalized.reserve(color_space_name.size());
    for (char c : color_space_name) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (normalized == "rec709" || normalized == "rec_709" || normalized == "bt709") {
        return ColorSpace::Rec709;
    }
    if (normalized == "rec2020_hlg" || normalized == "rec2020-hlg" || normalized == "bt2020_hlg") {
        return ColorSpace::Rec2020Hlg;
    }

    throw std::invalid_argument("Color space must be one of [rec709, rec2020_hlg].");
}

inline int ToColorMatrixId(ColorSpace color_space) {
    return color_space == ColorSpace::Rec2020Hlg ? 1 : 0;
}

inline const char* ToColorRangeName(ColorRange color_range) {
    switch (color_range) {
        case ColorRange::Limited:
            return "limited";
        case ColorRange::Full:
            return "full";
    }
    return "limited";
}

inline ColorRange ParseColorRangeName(const std::string& color_range_name) {
    std::string normalized;
    normalized.reserve(color_range_name.size());
    for (char c : color_range_name) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (normalized == "full" || normalized == "data" || normalized == "pc") {
        return ColorRange::Full;
    }
    if (normalized == "limited" || normalized == "video") {
        return ColorRange::Limited;
    }

    throw std::invalid_argument("Color range must be one of [limited, full].");
}

inline int ToColorRangeId(ColorRange color_range) {
    return color_range == ColorRange::Full ? 1 : 0;
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
    if (normalized == "bilinear_sharp" || normalized == "bilinear+sharp" || normalized == "realtime") {
        return SrFlavor::BilinearSharp;
    }
    if (normalized == "bicubic") {
        return SrFlavor::Bicubic;
    }
    if (normalized == "bicubic_sharpen" || normalized == "bicubic+sharpen") {
        return SrFlavor::BicubicSharpen;
    }

    throw std::invalid_argument("SR flavor must be one of [bilinear, bilinear_sharp, bicubic, bicubic_sharpen].");
}

inline int ParseTensorDTypeCode(const std::string& dtype_name) {
    std::string normalized;
    normalized.reserve(dtype_name.size());
    for (char c : dtype_name) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (normalized == "float" || normalized == "float32" || normalized == "fp32") {
        return 0;
    }
    if (normalized == "float16" || normalized == "fp16" || normalized == "half") {
        return 1;
    }
    if (normalized == "uint8" || normalized == "u8") {
        return 2;
    }

    throw std::invalid_argument("Tensor dtype must be one of [float16, float32, uint8].");
}

inline std::size_t TensorElementSizeBytes(int tensor_dtype) {
    switch (tensor_dtype) {
        case 0:
            return sizeof(float);
        case 1:
            return sizeof(std::uint16_t);
        case 2:
            return sizeof(std::uint8_t);
        default:
            throw std::invalid_argument("Unsupported tensor dtype code.");
    }
}

} // namespace

CudaTensorBuffer::CudaTensorBuffer()
    : data_(nullptr),
      bytes_(0),
      width_(0),
      height_(0),
      channels_(0),
      dtype_(""),
      layout_(""),
      normalized_01_(false) {}

CudaTensorBuffer::CudaTensorBuffer(
    void* data,
    std::size_t bytes,
    int width,
    int height,
    int channels,
    const std::string& dtype,
    const std::string& layout,
    bool normalized_01
)
    : data_(data),
      bytes_(bytes),
      width_(width),
      height_(height),
      channels_(channels),
      dtype_(dtype),
      layout_(layout),
      normalized_01_(normalized_01) {}

CudaTensorBuffer::~CudaTensorBuffer() {
    if (data_ != nullptr) {
        cudaFree(data_);
        data_ = nullptr;
    }
}

CudaTensorBuffer::CudaTensorBuffer(CudaTensorBuffer&& other) noexcept
    : data_(other.data_),
      bytes_(other.bytes_),
      width_(other.width_),
      height_(other.height_),
      channels_(other.channels_),
      dtype_(std::move(other.dtype_)),
      layout_(std::move(other.layout_)),
      normalized_01_(other.normalized_01_) {
    other.data_ = nullptr;
    other.bytes_ = 0;
    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 0;
    other.normalized_01_ = false;
}

CudaTensorBuffer& CudaTensorBuffer::operator=(CudaTensorBuffer&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    if (data_ != nullptr) {
        cudaFree(data_);
    }

    data_ = other.data_;
    bytes_ = other.bytes_;
    width_ = other.width_;
    height_ = other.height_;
    channels_ = other.channels_;
    dtype_ = std::move(other.dtype_);
    layout_ = std::move(other.layout_);
    normalized_01_ = other.normalized_01_;

    other.data_ = nullptr;
    other.bytes_ = 0;
    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 0;
    other.normalized_01_ = false;

    return *this;
}

std::uint64_t CudaTensorBuffer::DataPtr() const {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data_));
}

std::size_t CudaTensorBuffer::Bytes() const {
    return bytes_;
}

int CudaTensorBuffer::Width() const {
    return width_;
}

int CudaTensorBuffer::Height() const {
    return height_;
}

int CudaTensorBuffer::Channels() const {
    return channels_;
}

const std::string& CudaTensorBuffer::DType() const {
    return dtype_;
}

const std::string& CudaTensorBuffer::Layout() const {
    return layout_;
}

bool CudaTensorBuffer::Normalized01() const {
    return normalized_01_;
}

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
    sr_flavor_(SrFlavor::BilinearSharp),
    auto_sr_scale_(enable_placeholder_sr && sr_scale == 0),
    max_auto_sr_scale_(8),
    sr_requested_scale_(sr_scale),
      sr_scale_(sr_scale),
      sr_width_(width),
      sr_height_(height),
    sr_buffer_scale_capacity_(0),
            auto_sr_pending_scale_(-1),
            auto_sr_pending_frames_(0),
            auto_sr_settle_frames_(kAutoSrScaleSettleFrames),
                subpixel_shift_x_(0.0f),
                subpixel_shift_y_(0.0f),
        color_space_(ColorSpace::Rec709),
            color_range_(ColorRange::Limited),
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
        has_prev_rgb_full_(false),
    h_output_pinned_(nullptr),
    h_rgb_output_pinned_(nullptr),
    h_rgb_output_capacity_bytes_(0) {
    ValidateConfiguration();
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        ClampRoi();
    }

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

    roi_w_ = std::clamp(roi_w_, 2, width_);
    roi_h_ = std::clamp(roi_h_, 2, height_);

    // UYVY packs chroma for 2 horizontal pixels, so enforce even start and width.
    roi_w_ &= ~1;
    if (roi_w_ < 2) {
        roi_w_ = 2;
    }

    const int max_x = std::max(0, width_ - roi_w_);
    const int max_y = std::max(0, height_ - roi_h_);
    roi_x_ = std::clamp(roi_x_, 0, max_x);
    roi_y_ = std::clamp(roi_y_, 0, max_y);

    roi_x_ &= ~1;
    if (roi_x_ > max_x) {
        roi_x_ = std::max(0, max_x & ~1);
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
    auto_sr_pending_scale_ = -1;
    auto_sr_pending_frames_ = 0;
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
        auto_sr_pending_scale_ = -1;
        auto_sr_pending_frames_ = 0;
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
    auto_sr_pending_scale_ = -1;
    auto_sr_pending_frames_ = 0;
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

void VideoProcessor::SetSubpixelShift(float shift_x, float shift_y) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    subpixel_shift_x_ = std::fabs(shift_x) < kSubpixelShiftEpsilon ? 0.0f : shift_x;
    subpixel_shift_y_ = std::fabs(shift_y) < kSubpixelShiftEpsilon ? 0.0f : shift_y;
}

void VideoProcessor::GetSubpixelShift(float& shift_x, float& shift_y) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    shift_x = subpixel_shift_x_;
    shift_y = subpixel_shift_y_;
}

void VideoProcessor::SetColorSpace(ColorSpace color_space) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    color_space_ = color_space;
}

void VideoProcessor::SetColorSpaceByName(const std::string& color_space_name) {
    SetColorSpace(ParseColorSpaceName(color_space_name));
}

ColorSpace VideoProcessor::GetColorSpace() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return color_space_;
}

std::string VideoProcessor::GetColorSpaceName() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return ToColorSpaceName(color_space_);
}

void VideoProcessor::SetColorRange(ColorRange color_range) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    color_range_ = color_range;
}

void VideoProcessor::SetColorRangeByName(const std::string& color_range_name) {
    SetColorRange(ParseColorRangeName(color_range_name));
}

ColorRange VideoProcessor::GetColorRange() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return color_range_;
}

std::string VideoProcessor::GetColorRangeName() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return ToColorRangeName(color_range_);
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
    const cudaError_t err = cudaMalloc(&new_buffer, sr_pixels * kRgbBytesPerPixel);
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
        auto_sr_pending_scale_ = -1;
        auto_sr_pending_frames_ = 0;
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
            auto_sr_pending_scale_ = -1;
            auto_sr_pending_frames_ = 0;
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

    CheckCuda(cudaMalloc(&d_rgb_full_, rgb_pixels_ * kRgbBytesPerPixel), "cudaMalloc d_rgb_full_");
    CheckCuda(cudaMalloc(&d_rgb_bob_, rgb_pixels_ * kRgbBytesPerPixel), "cudaMalloc d_rgb_bob_");
    CheckCuda(cudaMalloc(&d_rgb_denoise_, rgb_pixels_ * kRgbBytesPerPixel), "cudaMalloc d_rgb_denoise_");
    CheckCuda(cudaMalloc(&d_rgb_prev_full_, rgb_pixels_ * kRgbBytesPerPixel), "cudaMalloc d_rgb_prev_full_");
    CheckCuda(cudaMalloc(&d_rgb_zoom_, rgb_pixels_ * kRgbBytesPerPixel), "cudaMalloc d_rgb_zoom_");

    if (cudaHostAlloc(&h_output_pinned_, uyvy_bytes_, cudaHostAllocDefault) != cudaSuccess) {
        h_output_pinned_ = nullptr;
        host_output_.resize(uyvy_bytes_);
    }

    h_rgb_output_capacity_bytes_ = rgb_pixels_ * kRgbBytesPerPixel;
    if (cudaHostAlloc(&h_rgb_output_pinned_, h_rgb_output_capacity_bytes_, cudaHostAllocDefault) != cudaSuccess) {
        h_rgb_output_pinned_ = nullptr;
        host_rgb_output_.resize(h_rgb_output_capacity_bytes_);
    }

    if (enable_placeholder_sr_) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        ConfigureSrScaleLocked(sr_requested_scale_, auto_sr_scale_);
    }
}

std::string VideoProcessor::ProcessFrame(const std::string& input_frame) {
    return ProcessFrameBuffer(
        reinterpret_cast<const uint8_t*>(input_frame.data()),
        input_frame.size()
    );
}

std::string VideoProcessor::ProcessFrameNoDeinterlace(const std::string& input_frame) {
    return ProcessFrameNoDeinterlaceBuffer(
        reinterpret_cast<const uint8_t*>(input_frame.data()),
        input_frame.size()
    );
}

std::string VideoProcessor::ProcessFrameDeinterlaceOnly(const std::string& input_frame) {
    return ProcessFrameDeinterlaceOnlyBuffer(
        reinterpret_cast<const uint8_t*>(input_frame.data()),
        input_frame.size()
    );
}

std::string VideoProcessor::ProcessFramePreprocessOnly(const std::string& input_frame) {
    return ProcessFramePreprocessOnlyBuffer(
        reinterpret_cast<const uint8_t*>(input_frame.data()),
        input_frame.size()
    );
}

std::string VideoProcessor::ProcessFramePreprocessRoiRgb(
    const std::string& input_frame,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    int out_w,
    int out_h
) {
    return ProcessFramePreprocessRoiRgbBuffer(
        reinterpret_cast<const uint8_t*>(input_frame.data()),
        input_frame.size(),
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        out_w,
        out_h
    );
}

CudaTensorBuffer VideoProcessor::ProcessFramePreprocessRoiTensorCuda(
    const std::string& input_frame,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    int out_w,
    int out_h,
    const std::string& dtype_name
) {
    return ProcessFramePreprocessRoiTensorCudaBuffer(
        reinterpret_cast<const uint8_t*>(input_frame.data()),
        input_frame.size(),
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        out_w,
        out_h,
        dtype_name
    );
}

std::string VideoProcessor::ProcessFrameBuffer(const uint8_t* input_frame, size_t input_size) {
    return ProcessFrameInternal(input_frame, input_size, false, false, false);
}

std::string VideoProcessor::ProcessFrameFieldPhaseBuffer(
    const uint8_t* input_frame,
    size_t input_size,
    int field_phase
) {
    return ProcessFrameInternal(input_frame, input_size, false, true, false, field_phase & 1);
}

std::string VideoProcessor::ProcessFrameNoDeinterlaceBuffer(const uint8_t* input_frame, size_t input_size) {
    return ProcessFrameInternal(input_frame, input_size, false, false, true);
}

std::string VideoProcessor::ProcessFrameDeinterlaceOnlyBuffer(const uint8_t* input_frame, size_t input_size) {
    return ProcessFrameInternal(input_frame, input_size, true, true, false);
}

std::string VideoProcessor::ProcessFramePreprocessOnlyBuffer(const uint8_t* input_frame, size_t input_size) {
    return ProcessFrameInternal(input_frame, input_size, true, false, false);
}

std::string VideoProcessor::ProcessFramePreprocessRoiRgbBuffer(
    const uint8_t* input_frame,
    size_t input_size,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    int out_w,
    int out_h
) {
    std::lock_guard<std::mutex> process_lock(process_mutex_);

    if (input_frame == nullptr) {
        throw std::invalid_argument("Input frame pointer is null.");
    }
    if (input_size != uyvy_bytes_) {
        throw std::invalid_argument("Invalid frame size; expected 1920*1080*2 bytes in UYVY.");
    }
    if (out_w <= 0 || out_h <= 0) {
        throw std::invalid_argument("Output dimensions must be positive.");
    }

    // UYVY packs chroma for 2 horizontal pixels.
    if ((roi_w & 1) != 0) {
        roi_w -= 1;
    }
    if ((roi_x & 1) != 0) {
        roi_x -= 1;
    }
    if (roi_w < 2) {
        roi_w = 2;
    }
    if (roi_h < 2) {
        roi_h = 2;
    }

    roi_w = std::clamp(roi_w, 2, width_);
    roi_h = std::clamp(roi_h, 2, height_);

    const int max_x = std::max(0, width_ - roi_w);
    const int max_y = std::max(0, height_ - roi_h);
    roi_x = std::clamp(roi_x, 0, max_x);
    roi_y = std::clamp(roi_y, 0, max_y);
    roi_x &= ~1;

    out_w = std::clamp(out_w, 1, width_);
    out_h = std::clamp(out_h, 1, height_);

    bool deinterlace_enabled = true;
    DeinterlaceMethod deinterlace_method = DeinterlaceMethod::Bob;
    DenoiseMethod denoise_method = DenoiseMethod::Off;
    float denoise_strength = 0.0f;
    ColorSpace color_space = ColorSpace::Rec709;
    ColorRange color_range = ColorRange::Limited;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        deinterlace_enabled = enable_deinterlace_;
        deinterlace_method = deinterlace_method_;
        denoise_method = denoise_method_;
        denoise_strength = denoise_strength_;
        color_space = color_space_;
        color_range = color_range_;
    }

    CheckCuda(
        cudaMemcpyAsync(d_uyvy_in_, input_frame, uyvy_bytes_, cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync H2D preprocess roi rgb"
    );

    const int color_matrix = ToColorMatrixId(color_space);
    const int color_range_id = ToColorRangeId(color_range);
    cuda_kernels::LaunchUyvyToRgb(d_uyvy_in_, d_rgb_full_, width_, height_, color_matrix, color_range_id, stream_);

    cuda_kernels::LaunchCropCopyRgb(
        d_rgb_full_,
        width_,
        height_,
        d_rgb_zoom_,
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        stream_
    );

    const uchar3* pre_input = d_rgb_zoom_;
    const int preprocess_w = roi_w;
    const int preprocess_h = roi_h;
    const int preprocess_field_phase = roi_y & 1;
    const bool temporal_denoise_active =
        denoise_method == DenoiseMethod::FieldTemporalLuma && denoise_strength > 0.001f;

    if (temporal_denoise_active) {
        cuda_kernels::LaunchCropCopyRgb(
            d_rgb_prev_full_,
            width_,
            height_,
            d_rgb_sr_,
            roi_x,
            roi_y,
            roi_w,
            roi_h,
            stream_
        );

        if (has_prev_rgb_full_) {
            cuda_kernels::LaunchDenoiseFieldTemporalLuma(
                pre_input,
                d_rgb_sr_,
                d_rgb_denoise_,
                preprocess_w,
                preprocess_h,
                denoise_strength,
                stream_
            );
        } else {
            CheckCuda(
                cudaMemcpyAsync(
                    d_rgb_denoise_,
                    pre_input,
                    static_cast<size_t>(preprocess_w) * static_cast<size_t>(preprocess_h) * kRgbBytesPerPixel,
                    cudaMemcpyDeviceToDevice,
                    stream_
                ),
                "cudaMemcpyAsync D2D preprocess temporal warmup"
            );
        }
        pre_input = d_rgb_denoise_;
    }

    if (deinterlace_enabled) {
        switch (deinterlace_method) {
            case DeinterlaceMethod::Blend:
                cuda_kernels::LaunchBlendDeinterlace(pre_input, d_rgb_bob_, preprocess_w, preprocess_h, stream_);
                break;
            case DeinterlaceMethod::EdgeAdaptive:
                cuda_kernels::LaunchEdgeAdaptiveDeinterlace(
                    pre_input,
                    d_rgb_bob_,
                    preprocess_w,
                    preprocess_h,
                    preprocess_field_phase,
                    stream_
                );
                break;
            case DeinterlaceMethod::Bob:
            default:
                cuda_kernels::LaunchBobDeinterlace(
                    pre_input,
                    d_rgb_bob_,
                    preprocess_w,
                    preprocess_h,
                    preprocess_field_phase,
                    stream_
                );
                break;
        }
        pre_input = d_rgb_bob_;
    }

    if (denoise_method != DenoiseMethod::Off &&
        denoise_method != DenoiseMethod::FieldTemporalLuma &&
        denoise_strength > 0.001f) {
        switch (denoise_method) {
            case DenoiseMethod::LumaMedian3x3:
                cuda_kernels::LaunchDenoiseLumaMedian3x3(pre_input, d_rgb_denoise_, preprocess_w, preprocess_h, denoise_strength, stream_);
                break;
            case DenoiseMethod::LumaBilateral3x3:
                cuda_kernels::LaunchDenoiseLumaBilateral3x3(pre_input, d_rgb_denoise_, preprocess_w, preprocess_h, denoise_strength, stream_);
                break;
            case DenoiseMethod::LumaBilateral5x5:
                cuda_kernels::LaunchDenoiseLumaBilateral5x5(pre_input, d_rgb_denoise_, preprocess_w, preprocess_h, denoise_strength, stream_);
                break;
            case DenoiseMethod::LumaGaussian3x3:
            default:
                cuda_kernels::LaunchDenoiseLumaGaussian3x3(pre_input, d_rgb_denoise_, preprocess_w, preprocess_h, denoise_strength, stream_);
                break;
        }
        pre_input = d_rgb_denoise_;
    }

    if (out_w != preprocess_w || out_h != preprocess_h) {
        cuda_kernels::LaunchCropZoomBicubic(
            pre_input,
            preprocess_w,
            preprocess_h,
            d_rgb_zoom_,
            out_w,
            out_h,
            0,
            0,
            preprocess_w,
            preprocess_h,
            stream_
        );
        pre_input = d_rgb_zoom_;
    }

    const size_t out_bytes = static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * kRgbBytesPerPixel;
    if (out_bytes > h_rgb_output_capacity_bytes_) {
        throw std::runtime_error("RGB output size exceeds preallocated host buffer capacity.");
    }

    uint8_t* host_rgb_ptr = h_rgb_output_pinned_ != nullptr ? h_rgb_output_pinned_ : host_rgb_output_.data();
    CheckCuda(
        cudaMemcpyAsync(host_rgb_ptr, pre_input, out_bytes, cudaMemcpyDeviceToHost, stream_),
        "cudaMemcpyAsync D2H preprocess roi rgb"
    );

    if (temporal_denoise_active) {
        CheckCuda(
            cudaMemcpyAsync(
                d_rgb_prev_full_,
                d_rgb_full_,
                rgb_pixels_ * kRgbBytesPerPixel,
                cudaMemcpyDeviceToDevice,
                stream_
            ),
            "cudaMemcpyAsync D2D update prev rgb preprocess roi"
        );
        has_prev_rgb_full_ = true;
    } else {
        has_prev_rgb_full_ = false;
    }

    CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize preprocess roi rgb");
    return std::string(reinterpret_cast<const char*>(host_rgb_ptr), out_bytes);
}

CudaTensorBuffer VideoProcessor::ProcessFramePreprocessRoiTensorCudaBuffer(
    const uint8_t* input_frame,
    size_t input_size,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    int out_w,
    int out_h,
    const std::string& dtype_name
) {
    std::lock_guard<std::mutex> process_lock(process_mutex_);

    if (input_frame == nullptr) {
        throw std::invalid_argument("Input frame pointer is null.");
    }
    if (input_size != uyvy_bytes_) {
        throw std::invalid_argument("Invalid frame size; expected 1920*1080*2 bytes in UYVY.");
    }
    if (out_w <= 0 || out_h <= 0) {
        throw std::invalid_argument("Output dimensions must be positive.");
    }

    const int tensor_dtype = ParseTensorDTypeCode(dtype_name);
    const int tensor_layout = 0;  // NCHW
    const int tensor_channels = 3;
    const bool tensor_normalized_01 = true;

    // UYVY packs chroma for 2 horizontal pixels.
    if ((roi_w & 1) != 0) {
        roi_w -= 1;
    }
    if ((roi_x & 1) != 0) {
        roi_x -= 1;
    }
    if (roi_w < 2) {
        roi_w = 2;
    }
    if (roi_h < 2) {
        roi_h = 2;
    }

    roi_w = std::clamp(roi_w, 2, width_);
    roi_h = std::clamp(roi_h, 2, height_);

    const int max_x = std::max(0, width_ - roi_w);
    const int max_y = std::max(0, height_ - roi_h);
    roi_x = std::clamp(roi_x, 0, max_x);
    roi_y = std::clamp(roi_y, 0, max_y);
    roi_x &= ~1;

    out_w = std::clamp(out_w, 1, width_);
    out_h = std::clamp(out_h, 1, height_);

    bool deinterlace_enabled = true;
    DeinterlaceMethod deinterlace_method = DeinterlaceMethod::Bob;
    DenoiseMethod denoise_method = DenoiseMethod::Off;
    float denoise_strength = 0.0f;
    ColorSpace color_space = ColorSpace::Rec709;
    ColorRange color_range = ColorRange::Limited;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        deinterlace_enabled = enable_deinterlace_;
        deinterlace_method = deinterlace_method_;
        denoise_method = denoise_method_;
        denoise_strength = denoise_strength_;
        color_space = color_space_;
        color_range = color_range_;
    }

    CheckCuda(
        cudaMemcpyAsync(d_uyvy_in_, input_frame, uyvy_bytes_, cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync H2D preprocess roi tensor"
    );

    const int color_matrix = ToColorMatrixId(color_space);
    const int color_range_id = ToColorRangeId(color_range);
    cuda_kernels::LaunchUyvyToRgb(d_uyvy_in_, d_rgb_full_, width_, height_, color_matrix, color_range_id, stream_);

    cuda_kernels::LaunchCropCopyRgb(
        d_rgb_full_,
        width_,
        height_,
        d_rgb_zoom_,
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        stream_
    );

    const uchar3* pre_input = d_rgb_zoom_;
    const int preprocess_w = roi_w;
    const int preprocess_h = roi_h;
    const int preprocess_field_phase = roi_y & 1;
    const bool temporal_denoise_active =
        denoise_method == DenoiseMethod::FieldTemporalLuma && denoise_strength > 0.001f;

    if (temporal_denoise_active) {
        cuda_kernels::LaunchCropCopyRgb(
            d_rgb_prev_full_,
            width_,
            height_,
            d_rgb_sr_,
            roi_x,
            roi_y,
            roi_w,
            roi_h,
            stream_
        );

        if (has_prev_rgb_full_) {
            cuda_kernels::LaunchDenoiseFieldTemporalLuma(
                pre_input,
                d_rgb_sr_,
                d_rgb_denoise_,
                preprocess_w,
                preprocess_h,
                denoise_strength,
                stream_
            );
        } else {
            CheckCuda(
                cudaMemcpyAsync(
                    d_rgb_denoise_,
                    pre_input,
                    static_cast<size_t>(preprocess_w) * static_cast<size_t>(preprocess_h) * kRgbBytesPerPixel,
                    cudaMemcpyDeviceToDevice,
                    stream_
                ),
                "cudaMemcpyAsync D2D preprocess temporal warmup tensor"
            );
        }
        pre_input = d_rgb_denoise_;
    }

    if (deinterlace_enabled) {
        switch (deinterlace_method) {
            case DeinterlaceMethod::Blend:
                cuda_kernels::LaunchBlendDeinterlace(pre_input, d_rgb_bob_, preprocess_w, preprocess_h, stream_);
                break;
            case DeinterlaceMethod::EdgeAdaptive:
                cuda_kernels::LaunchEdgeAdaptiveDeinterlace(
                    pre_input,
                    d_rgb_bob_,
                    preprocess_w,
                    preprocess_h,
                    preprocess_field_phase,
                    stream_
                );
                break;
            case DeinterlaceMethod::Bob:
            default:
                cuda_kernels::LaunchBobDeinterlace(
                    pre_input,
                    d_rgb_bob_,
                    preprocess_w,
                    preprocess_h,
                    preprocess_field_phase,
                    stream_
                );
                break;
        }
        pre_input = d_rgb_bob_;
    }

    if (denoise_method != DenoiseMethod::Off &&
        denoise_method != DenoiseMethod::FieldTemporalLuma &&
        denoise_strength > 0.001f) {
        switch (denoise_method) {
            case DenoiseMethod::LumaMedian3x3:
                cuda_kernels::LaunchDenoiseLumaMedian3x3(pre_input, d_rgb_denoise_, preprocess_w, preprocess_h, denoise_strength, stream_);
                break;
            case DenoiseMethod::LumaBilateral3x3:
                cuda_kernels::LaunchDenoiseLumaBilateral3x3(pre_input, d_rgb_denoise_, preprocess_w, preprocess_h, denoise_strength, stream_);
                break;
            case DenoiseMethod::LumaBilateral5x5:
                cuda_kernels::LaunchDenoiseLumaBilateral5x5(pre_input, d_rgb_denoise_, preprocess_w, preprocess_h, denoise_strength, stream_);
                break;
            case DenoiseMethod::LumaGaussian3x3:
            default:
                cuda_kernels::LaunchDenoiseLumaGaussian3x3(pre_input, d_rgb_denoise_, preprocess_w, preprocess_h, denoise_strength, stream_);
                break;
        }
        pre_input = d_rgb_denoise_;
    }

    if (out_w != preprocess_w || out_h != preprocess_h) {
        cuda_kernels::LaunchCropZoomBicubic(
            pre_input,
            preprocess_w,
            preprocess_h,
            d_rgb_zoom_,
            out_w,
            out_h,
            0,
            0,
            preprocess_w,
            preprocess_h,
            stream_
        );
        pre_input = d_rgb_zoom_;
    }

    const std::size_t tensor_elements =
        static_cast<std::size_t>(tensor_channels) * static_cast<std::size_t>(out_w) * static_cast<std::size_t>(out_h);
    const std::size_t tensor_bytes = tensor_elements * TensorElementSizeBytes(tensor_dtype);
    void* d_tensor = nullptr;
    CheckCuda(cudaMalloc(&d_tensor, tensor_bytes), "cudaMalloc preprocess roi tensor");

    try {
        cuda_kernels::LaunchRgbToTensor(
            pre_input,
            d_tensor,
            tensor_dtype,
            tensor_layout,
            tensor_channels,
            tensor_normalized_01,
            out_w,
            out_h,
            stream_
        );

        if (temporal_denoise_active) {
            CheckCuda(
                cudaMemcpyAsync(
                    d_rgb_prev_full_,
                    d_rgb_full_,
                    rgb_pixels_ * kRgbBytesPerPixel,
                    cudaMemcpyDeviceToDevice,
                    stream_
                ),
                "cudaMemcpyAsync D2D update prev rgb preprocess roi tensor"
            );
            has_prev_rgb_full_ = true;
        } else {
            has_prev_rgb_full_ = false;
        }

        CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize preprocess roi tensor");
    } catch (...) {
        cudaFree(d_tensor);
        throw;
    }

    return CudaTensorBuffer(
        d_tensor,
        tensor_bytes,
        out_w,
        out_h,
        tensor_channels,
        (tensor_dtype == 1 ? std::string("float16") : (tensor_dtype == 0 ? std::string("float32") : std::string("uint8"))),
        "nchw",
        tensor_normalized_01
    );
}

std::string VideoProcessor::ProcessFrameInternal(
    const uint8_t* input_frame,
    size_t input_size,
    bool deinterlace_only,
    bool force_deinterlace,
    bool force_disable_deinterlace,
    int field_phase_override
) {
    std::lock_guard<std::mutex> process_lock(process_mutex_);

    if (input_frame == nullptr) {
        throw std::invalid_argument("Input frame pointer is null.");
    }

    if (input_size != uyvy_bytes_) {
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
    float subpixel_shift_x = 0.0f;
    float subpixel_shift_y = 0.0f;
    ColorSpace color_space = ColorSpace::Rec709;
    ColorRange color_range = ColorRange::Limited;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (enable_placeholder_sr_ && auto_sr_scale_) {
            const int desired_scale = SelectAutoSrScale(width_, height_, roi_w_, roi_h_, max_auto_sr_scale_);
            if (desired_scale != sr_scale_) {
                if (auto_sr_pending_scale_ != desired_scale) {
                    auto_sr_pending_scale_ = desired_scale;
                    auto_sr_pending_frames_ = 1;
                } else {
                    auto_sr_pending_frames_ += 1;
                }

                if (auto_sr_pending_frames_ >= auto_sr_settle_frames_) {
                    ConfigureSrScaleLocked(0, true);
                }
            } else {
                auto_sr_pending_scale_ = -1;
                auto_sr_pending_frames_ = 0;
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
        subpixel_shift_x = subpixel_shift_x_;
        subpixel_shift_y = subpixel_shift_y_;
        color_space = color_space_;
        color_range = color_range_;

        if (force_deinterlace) {
            deinterlace_enabled = true;
        }
        if (force_disable_deinterlace) {
            deinterlace_enabled = false;
        }
    }

    // Fast no-op path: when no stage modifies pixels and ROI is full-frame,
    // skip GPU work entirely.
    if (!deinterlace_only && !deinterlace_enabled && denoise_method == DenoiseMethod::Off &&
        (!enable_placeholder_sr_ || sr_scale <= 1) &&
        roi_x == 0 && roi_y == 0 && roi_w == width_ && roi_h == height_ &&
        !HasSubpixelShift(subpixel_shift_x, subpixel_shift_y)) {
        return std::string(reinterpret_cast<const char*>(input_frame), uyvy_bytes_);
    }

    uint8_t* host_output_ptr = h_output_pinned_ != nullptr ? h_output_pinned_ : host_output_.data();

    CheckCuda(
        cudaMemcpyAsync(d_uyvy_in_, input_frame, uyvy_bytes_, cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync H2D"
    );

    const bool denoise_active = denoise_method != DenoiseMethod::Off && denoise_strength > 0.001f;
    const bool temporal_denoise_active =
        denoise_method == DenoiseMethod::FieldTemporalLuma && denoise_strength > 0.001f;
    const bool sr_inactive = (!enable_placeholder_sr_) || (sr_scale <= 1);
    const bool full_frame_roi = (roi_x == 0 && roi_y == 0 && roi_w == width_ && roi_h == height_);
    if (!deinterlace_only && !deinterlace_enabled && denoise_active &&
        denoise_method != DenoiseMethod::FieldTemporalLuma &&
        sr_inactive && full_frame_roi) {
        CheckCuda(
            cudaMemcpyAsync(d_uyvy_out_, d_uyvy_in_, uyvy_bytes_, cudaMemcpyDeviceToDevice, stream_),
            "cudaMemcpyAsync D2D uyvy denoise prep"
        );

        switch (denoise_method) {
            case DenoiseMethod::LumaMedian3x3:
                cuda_kernels::LaunchDenoiseUyvyLumaMedian3x3(d_uyvy_in_, d_uyvy_out_, width_, height_, denoise_strength, stream_);
                break;
            case DenoiseMethod::LumaBilateral3x3:
                cuda_kernels::LaunchDenoiseUyvyLumaBilateral3x3(d_uyvy_in_, d_uyvy_out_, width_, height_, denoise_strength, stream_);
                break;
            case DenoiseMethod::LumaBilateral5x5:
                cuda_kernels::LaunchDenoiseUyvyLumaBilateral5x5(d_uyvy_in_, d_uyvy_out_, width_, height_, denoise_strength, stream_);
                break;
            case DenoiseMethod::LumaGaussian3x3:
            default:
                cuda_kernels::LaunchDenoiseUyvyLumaGaussian3x3(d_uyvy_in_, d_uyvy_out_, width_, height_, denoise_strength, stream_);
                break;
        }

        const uint8_t* final_uyvy = d_uyvy_out_;
        if (HasSubpixelShift(subpixel_shift_x, subpixel_shift_y)) {
            cuda_kernels::LaunchUyvySubpixelShift(
                d_uyvy_out_,
                d_uyvy_in_,
                width_,
                height_,
                subpixel_shift_x,
                subpixel_shift_y,
                stream_
            );
            final_uyvy = d_uyvy_in_;
        }

        CheckCuda(
            cudaMemcpyAsync(host_output_ptr, final_uyvy, uyvy_bytes_, cudaMemcpyDeviceToHost, stream_),
            "cudaMemcpyAsync D2H uyvy denoise fast path"
        );
        CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize uyvy denoise fast path");
        return std::string(reinterpret_cast<const char*>(host_output_ptr), uyvy_bytes_);
    }

    const bool use_uyvy_scaling_fast_path =
        enable_placeholder_sr_ &&
        sr_scale > 1 &&
        !deinterlace_enabled &&
        denoise_method == DenoiseMethod::Off &&
        sr_flavor == SrFlavor::Bilinear;

    if (use_uyvy_scaling_fast_path) {
        // Interlaced-safe scaling path: preserve field parity while sampling
        // UYVY directly to avoid vertical field blending artifacts.
        if (roi_w == width_ && roi_h == height_) {
            int zoom_roi_w = std::max(2, width_ / sr_scale);
            int zoom_roi_h = std::max(2, height_ / sr_scale);
            zoom_roi_w &= ~1;
            if (zoom_roi_w < 2) {
                zoom_roi_w = 2;
            }

            roi_x = std::max(0, (width_ - zoom_roi_w) / 2);
            roi_y = std::max(0, (height_ - zoom_roi_h) / 2);
            roi_x &= ~1;
            roi_w = zoom_roi_w;
            roi_h = zoom_roi_h;
        }

        cuda_kernels::LaunchUyvyCropZoomNearest(
            d_uyvy_in_,
            width_,
            height_,
            d_uyvy_out_,
            width_,
            height_,
            roi_x,
            roi_y,
            roi_w,
            roi_h,
            true,
            stream_
        );

        const uint8_t* final_uyvy = d_uyvy_out_;
        if (HasSubpixelShift(subpixel_shift_x, subpixel_shift_y)) {
            cuda_kernels::LaunchUyvySubpixelShift(
                d_uyvy_out_,
                d_uyvy_in_,
                width_,
                height_,
                subpixel_shift_x,
                subpixel_shift_y,
                stream_
            );
            final_uyvy = d_uyvy_in_;
        }

        CheckCuda(
            cudaMemcpyAsync(host_output_ptr, final_uyvy, uyvy_bytes_, cudaMemcpyDeviceToHost, stream_),
            "cudaMemcpyAsync D2H fast path"
        );
        CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize fast path");
        return std::string(reinterpret_cast<const char*>(host_output_ptr), uyvy_bytes_);
    }

    const int color_matrix = ToColorMatrixId(color_space);
    const int color_range_id = ToColorRangeId(color_range);
    if (field_phase_override >= 0) {
        cuda_kernels::LaunchUyvyFieldToRgb(
            d_uyvy_in_,
            d_rgb_full_,
            width_,
            height_,
            field_phase_override,
            color_matrix,
            color_range_id,
            stream_
        );
    } else {
        cuda_kernels::LaunchUyvyToRgb(d_uyvy_in_, d_rgb_full_, width_, height_, color_matrix, color_range_id, stream_);
    }

    const uchar3* crop_input = d_rgb_full_;
    int crop_src_w = width_;
    int crop_src_h = height_;
    int crop_roi_x = roi_x;
    int crop_roi_y = roi_y;
    int crop_roi_w = roi_w;
    int crop_roi_h = roi_h;
    const bool denoise_non_temporal_active =
        denoise_method != DenoiseMethod::Off &&
        denoise_method != DenoiseMethod::FieldTemporalLuma &&
        denoise_strength > 0.001f;
    const bool sr_enabled = enable_placeholder_sr_ && sr_scale > 1;
    const bool roi_smaller_than_full = roi_w < width_ || roi_h < height_;
    const bool preprocess_active = deinterlace_enabled || denoise_method != DenoiseMethod::Off;
    const bool use_roi_preprocess_for_scaling = !deinterlace_only && sr_enabled && preprocess_active && roi_smaller_than_full;

    if (use_roi_preprocess_for_scaling) {
        cuda_kernels::LaunchCropCopyRgb(
            d_rgb_full_,
            width_,
            height_,
            d_rgb_zoom_,
            roi_x,
            roi_y,
            roi_w,
            roi_h,
            stream_
        );
        crop_input = d_rgb_zoom_;
        crop_src_w = roi_w;
        crop_src_h = roi_h;
        crop_roi_x = 0;
        crop_roi_y = 0;
        crop_roi_w = roi_w;
        crop_roi_h = roi_h;
    }

    int preprocess_w = use_roi_preprocess_for_scaling ? roi_w : width_;
    int preprocess_h = use_roi_preprocess_for_scaling ? roi_h : height_;
    const int preprocess_field_phase = use_roi_preprocess_for_scaling ? (roi_y & 1) : 0;

    if (temporal_denoise_active) {
        const uchar3* temporal_prev = d_rgb_prev_full_;
        if (use_roi_preprocess_for_scaling) {
            cuda_kernels::LaunchCropCopyRgb(
                d_rgb_prev_full_,
                width_,
                height_,
                d_rgb_sr_,
                roi_x,
                roi_y,
                roi_w,
                roi_h,
                stream_
            );
            temporal_prev = d_rgb_sr_;
        }

        if (has_prev_rgb_full_) {
            cuda_kernels::LaunchDenoiseFieldTemporalLuma(
                crop_input,
                temporal_prev,
                d_rgb_denoise_,
                preprocess_w,
                preprocess_h,
                denoise_strength,
                stream_
            );
        } else {
            CheckCuda(
                cudaMemcpyAsync(
                    d_rgb_denoise_,
                    crop_input,
                    static_cast<size_t>(preprocess_w) * static_cast<size_t>(preprocess_h) * kRgbBytesPerPixel,
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
                cuda_kernels::LaunchBlendDeinterlace(crop_input, d_rgb_bob_, preprocess_w, preprocess_h, stream_);
                break;
            case DeinterlaceMethod::EdgeAdaptive:
                cuda_kernels::LaunchEdgeAdaptiveDeinterlace(
                    crop_input,
                    d_rgb_bob_,
                    preprocess_w,
                    preprocess_h,
                    preprocess_field_phase,
                    stream_
                );
                break;
            case DeinterlaceMethod::Bob:
            default:
                cuda_kernels::LaunchBobDeinterlace(
                    crop_input,
                    d_rgb_bob_,
                    preprocess_w,
                    preprocess_h,
                    preprocess_field_phase,
                    stream_
                );
                break;
        }
        crop_input = d_rgb_bob_;
    }

    if (denoise_non_temporal_active) {
        switch (denoise_method) {
            case DenoiseMethod::LumaMedian3x3:
                cuda_kernels::LaunchDenoiseLumaMedian3x3(crop_input, d_rgb_denoise_, preprocess_w, preprocess_h, denoise_strength, stream_);
                break;
            case DenoiseMethod::LumaBilateral3x3:
                cuda_kernels::LaunchDenoiseLumaBilateral3x3(crop_input, d_rgb_denoise_, preprocess_w, preprocess_h, denoise_strength, stream_);
                break;
            case DenoiseMethod::LumaBilateral5x5:
                cuda_kernels::LaunchDenoiseLumaBilateral5x5(crop_input, d_rgb_denoise_, preprocess_w, preprocess_h, denoise_strength, stream_);
                break;
            case DenoiseMethod::LumaGaussian3x3:
            default:
                cuda_kernels::LaunchDenoiseLumaGaussian3x3(crop_input, d_rgb_denoise_, preprocess_w, preprocess_h, denoise_strength, stream_);
                break;
        }
        crop_input = d_rgb_denoise_;
    }

    if (deinterlace_only) {
        if (use_roi_preprocess_for_scaling) {
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
            cuda_kernels::LaunchRgbToUyvy(d_rgb_zoom_, d_uyvy_out_, width_, height_, color_matrix, color_range_id, stream_);
        } else {
            cuda_kernels::LaunchRgbToUyvy(crop_input, d_uyvy_out_, width_, height_, color_matrix, color_range_id, stream_);
        }

        const uint8_t* final_uyvy = d_uyvy_out_;
        if (HasSubpixelShift(subpixel_shift_x, subpixel_shift_y)) {
            cuda_kernels::LaunchUyvySubpixelShift(
                d_uyvy_out_,
                d_uyvy_in_,
                width_,
                height_,
                subpixel_shift_x,
                subpixel_shift_y,
                stream_
            );
            final_uyvy = d_uyvy_in_;
        }

        CheckCuda(
            cudaMemcpyAsync(host_output_ptr, final_uyvy, uyvy_bytes_, cudaMemcpyDeviceToHost, stream_),
            "cudaMemcpyAsync D2H"
        );

        if (temporal_denoise_active) {
            CheckCuda(
                cudaMemcpyAsync(
                    d_rgb_prev_full_,
                    d_rgb_full_,
                    rgb_pixels_ * kRgbBytesPerPixel,
                    cudaMemcpyDeviceToDevice,
                    stream_
                ),
                "cudaMemcpyAsync D2D update prev rgb"
            );
            has_prev_rgb_full_ = true;
        } else {
            has_prev_rgb_full_ = false;
        }

        CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

        return std::string(reinterpret_cast<const char*>(host_output_ptr), uyvy_bytes_);
    }

    const bool use_single_pass_full_frame_zoom =
        enable_placeholder_sr_ &&
        sr_scale > 1 &&
        !deinterlace_only &&
        !preprocess_active &&
        roi_x == 0 &&
        roi_y == 0 &&
        roi_w == width_ &&
        roi_h == height_;

    if (use_single_pass_full_frame_zoom) {
        int zoom_roi_w = std::max(2, width_ / sr_scale);
        int zoom_roi_h = std::max(2, height_ / sr_scale);
        zoom_roi_w &= ~1;
        if (zoom_roi_w < 2) {
            zoom_roi_w = 2;
        }

        crop_roi_x = std::max(0, (width_ - zoom_roi_w) / 2);
        crop_roi_y = std::max(0, (height_ - zoom_roi_h) / 2);
        crop_roi_x &= ~1;
        crop_roi_w = zoom_roi_w;
        crop_roi_h = zoom_roi_h;
    }

    if (enable_placeholder_sr_ && sr_scale > 1 && !use_single_pass_full_frame_zoom) {
        int sr_roi_w = std::max(2, roi_w * sr_scale);
        int sr_roi_h = std::max(2, roi_h * sr_scale);
        const bool sr_pass_is_redundant = (sr_roi_w <= roi_w) && (sr_roi_h <= roi_h);
        if (!sr_pass_is_redundant) {
            const uchar3* sr_output = d_rgb_sr_;

            if (use_roi_preprocess_for_scaling) {
                switch (sr_flavor) {
                    case SrFlavor::Bilinear:
                        cuda_kernels::LaunchUpscaleBilinear(crop_input, crop_src_w, crop_src_h, d_rgb_sr_, sr_roi_w, sr_roi_h, stream_);
                        break;
                    case SrFlavor::BilinearSharp:
                        cuda_kernels::LaunchUpscaleBilinearSharp(crop_input, crop_src_w, crop_src_h, d_rgb_sr_, sr_roi_w, sr_roi_h, stream_);
                        break;
                    case SrFlavor::Bicubic:
                        cuda_kernels::LaunchUpscaleBicubic(crop_input, crop_src_w, crop_src_h, d_rgb_sr_, sr_roi_w, sr_roi_h, stream_);
                        break;
                    case SrFlavor::BicubicSharpen:
                        cuda_kernels::LaunchUpscaleBicubic(crop_input, crop_src_w, crop_src_h, d_rgb_sr_, sr_roi_w, sr_roi_h, stream_);
                        break;
                }
            } else {
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
                    case SrFlavor::BilinearSharp:
                        cuda_kernels::LaunchCropZoomBilinearSharp(
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
                        break;
                }
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

    const uchar3* final_output = d_rgb_zoom_;
    switch (sr_flavor) {
        case SrFlavor::Bilinear:
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
            break;
        case SrFlavor::BilinearSharp:
            cuda_kernels::LaunchCropZoomBilinearSharp(
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
            break;
        case SrFlavor::Bicubic:
            cuda_kernels::LaunchCropZoomBicubic(
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
            break;
        case SrFlavor::BicubicSharpen:
            cuda_kernels::LaunchCropZoomBicubic(
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
            cuda_kernels::LaunchSharpen3x3(
                d_rgb_zoom_,
                d_rgb_bob_,
                width_,
                height_,
                !deinterlace_enabled,
                stream_
            );
            final_output = d_rgb_bob_;
            break;
    }

    cuda_kernels::LaunchRgbToUyvy(final_output, d_uyvy_out_, width_, height_, color_matrix, color_range_id, stream_);

    const uint8_t* final_uyvy = d_uyvy_out_;
    if (HasSubpixelShift(subpixel_shift_x, subpixel_shift_y)) {
        cuda_kernels::LaunchUyvySubpixelShift(
            d_uyvy_out_,
            d_uyvy_in_,
            width_,
            height_,
            subpixel_shift_x,
            subpixel_shift_y,
            stream_
        );
        final_uyvy = d_uyvy_in_;
    }

    CheckCuda(
        cudaMemcpyAsync(host_output_ptr, final_uyvy, uyvy_bytes_, cudaMemcpyDeviceToHost, stream_),
        "cudaMemcpyAsync D2H"
    );

    if (temporal_denoise_active) {
        CheckCuda(
            cudaMemcpyAsync(
                d_rgb_prev_full_,
                d_rgb_full_,
                rgb_pixels_ * kRgbBytesPerPixel,
                cudaMemcpyDeviceToDevice,
                stream_
            ),
            "cudaMemcpyAsync D2D update prev rgb"
        );
        has_prev_rgb_full_ = true;
    } else {
        has_prev_rgb_full_ = false;
    }

    CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

    return std::string(reinterpret_cast<const char*>(host_output_ptr), uyvy_bytes_);
}

void VideoProcessor::Cleanup() {
    if (h_rgb_output_pinned_ != nullptr) {
        cudaFreeHost(h_rgb_output_pinned_);
        h_rgb_output_pinned_ = nullptr;
    }

    h_rgb_output_capacity_bytes_ = 0;

    if (h_output_pinned_ != nullptr) {
        cudaFreeHost(h_output_pinned_);
        h_output_pinned_ = nullptr;
    }

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
