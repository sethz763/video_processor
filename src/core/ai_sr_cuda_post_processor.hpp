#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace vp {

class AiSrCudaPostProcessor {
public:
    enum class TensorLayout {
        NCHW = 0,
        HWC = 1,
    };

    enum class TensorDType {
        Float32 = 0,
        Float16 = 1,
        UInt8 = 2,
    };

    enum class ColorSpace {
        Rec709,
        Rec2020Hlg,
    };

    enum class ColorRange {
        Limited,
        Full,
    };

    enum class ResizeMethod {
        Bilinear,
        BilinearSharp,
        Bicubic,
        BicubicSharpen,
    };

    enum class PostDenoiseMethod {
        Off,
        LumaGaussian3x3,
        LumaMedian3x3,
        LumaBilateral3x3,
        LumaBilateral5x5,
    };

    enum class PostArtifactReductionMethod {
        Off,
        LumaBilateral3x3,
        LumaBilateral5x5,
    };

    AiSrCudaPostProcessor(
        int output_width = 1920,
        int output_height = 1080,
        const std::string& color_space = "rec709",
        const std::string& color_range = "limited"
    );

    ~AiSrCudaPostProcessor();

    AiSrCudaPostProcessor(const AiSrCudaPostProcessor&) = delete;
    AiSrCudaPostProcessor& operator=(const AiSrCudaPostProcessor&) = delete;

    std::string ProcessOnnxOutputCudaPtr(
        std::uint64_t tensor_ptr,
        int tensor_width,
        int tensor_height,
        int channels,
        TensorLayout layout,
        TensorDType dtype,
        bool normalized_01,
        const std::string& resize_method
    );

    void SetColorSpaceByName(const std::string& color_space_name);
    void SetColorRangeByName(const std::string& color_range_name);
    std::string GetColorSpaceName() const;
    std::string GetColorRangeName() const;

    void SetPostDenoiseMethodByName(const std::string& method_name);
    std::string GetPostDenoiseMethodName() const;
    void SetPostDenoiseStrength(float strength);
    float GetPostDenoiseStrength() const;
    void SetPostArtifactReductionMethodByName(const std::string& method_name);
    std::string GetPostArtifactReductionMethodName() const;
    void SetPostArtifactReductionStrength(float strength);
    float GetPostArtifactReductionStrength() const;
    void SetPostExaggerationEnabled(bool enabled);
    bool GetPostExaggerationEnabled() const;
    void SetPostExaggerationGain(float gain);
    float GetPostExaggerationGain() const;

    int output_width() const { return output_width_; }
    int output_height() const { return output_height_; }

private:
    void InitializeBuffers();
    void EnsureTensorRgbCapacityLocked(int tensor_width, int tensor_height);
    void Cleanup();

    int output_width_;
    int output_height_;
    std::size_t output_uyvy_bytes_;

    ColorSpace color_space_;
    ColorRange color_range_;
    PostDenoiseMethod post_denoise_method_;
    float post_denoise_strength_;
    PostArtifactReductionMethod post_artifact_reduction_method_;
    float post_artifact_reduction_strength_;
    bool post_exaggeration_enabled_;
    float post_exaggeration_gain_;

    cudaStream_t stream_;

    uchar3* d_tensor_rgb_;
    std::size_t d_tensor_rgb_capacity_pixels_;
    uchar3* d_output_rgb_;
    uchar3* d_output_tmp_;
    uint8_t* d_output_uyvy_;

    uint8_t* h_output_pinned_;
    std::vector<uint8_t> host_output_;

    mutable std::mutex mutex_;
};

}  // namespace vp
