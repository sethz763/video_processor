#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace vp {

enum class SrFlavor {
    Bilinear,
    BilinearSharp,
    Bicubic,
    BicubicSharpen,
};

enum class DeinterlaceMethod {
    Bob,
    Blend,
    EdgeAdaptive,
};

enum class DenoiseMethod {
    Off,
    LumaGaussian3x3,
    LumaMedian3x3,
    LumaBilateral3x3,
    LumaBilateral5x5,
    FieldTemporalLuma,
};

enum class ColorSpace {
    Rec709,
    Rec2020Hlg,
};

enum class ColorRange {
    Limited,
    Full,
};

struct ColorStageConfig {
    int type = 0;
    bool after_composite = false;
    std::array<float, 9> params{};
    bool invert = false;
};

struct AlphaMixOperandConfig {
    int type = 0;
    bool source_from_effects_input = false;
    int source_slot = 0;
    int source_channel = 3;
    std::vector<ColorStageConfig> color_stages{};
    int blur_method = 0;
    float blur_radius = 0.0f;
    float blur_aspect = 1.0f;
    float transform_x = 0.0f;
    float transform_y = 0.0f;
    float transform_z = 0.0f;
    float rotate_x = 0.0f;
    float rotate_y = 0.0f;
    float rotate_z = 0.0f;
    float aspect_x = 1.0f;
    float aspect_y = 1.0f;
    uchar3 key_color = make_uchar3(0, 255, 0);
    float key_similarity = 0.25f;
    float key_softness = 0.10f;
    float key_edge_feather = 0.0f;
    bool key_invert = false;
    int mask_pattern_code = 0;
    float mask_softness = 0.0f;
    float mask_aspect = 1.0f;
    bool mask_invert = false;
    float mask_size = 1.0f;
    float mask_x = 0.0f;
    float mask_y = 0.0f;
    float mask_rotation = 0.0f;
};

struct AlphaMixOpConfig {
    int mode = 0;
    float factor = 1.0f;
    AlphaMixOperandConfig operand{};
};

class CudaTensorBuffer {
public:
    CudaTensorBuffer();
    // owning_stream is used for a stream-ordered cudaFreeAsync release; pass nullptr
    // to fall back to a plain (device-synchronizing) cudaFree.
    CudaTensorBuffer(void* data, std::size_t bytes, int width, int height, int channels, const std::string& dtype, const std::string& layout, bool normalized_01, cudaStream_t owning_stream = nullptr);
    ~CudaTensorBuffer();

    CudaTensorBuffer(const CudaTensorBuffer&) = delete;
    CudaTensorBuffer& operator=(const CudaTensorBuffer&) = delete;
    CudaTensorBuffer(CudaTensorBuffer&& other) noexcept;
    CudaTensorBuffer& operator=(CudaTensorBuffer&& other) noexcept;

    std::uint64_t DataPtr() const;
    std::size_t Bytes() const;
    int Width() const;
    int Height() const;
    int Channels() const;
    const std::string& DType() const;
    const std::string& Layout() const;
    bool Normalized01() const;

private:
    void* data_;
    std::size_t bytes_;
    int width_;
    int height_;
    int channels_;
    std::string dtype_;
    std::string layout_;
    bool normalized_01_;
    cudaStream_t owning_stream_;
};

class VideoProcessor {
public:
    VideoProcessor(
        int width,
        int height,
        int roi_x,
        int roi_y,
        int roi_w,
        int roi_h,
        bool enable_placeholder_sr = true,
        int sr_scale = 0
    );

    ~VideoProcessor();

    VideoProcessor(const VideoProcessor&) = delete;
    VideoProcessor& operator=(const VideoProcessor&) = delete;

    std::string ProcessFrame(const std::string& input_frame);
    std::string ProcessFrameNoDeinterlace(const std::string& input_frame);
    std::string ProcessFrameDeinterlaceOnly(const std::string& input_frame);
    std::string ProcessFramePreprocessOnly(const std::string& input_frame);
    std::string ProcessFramePreprocessRoiRgb(
        const std::string& input_frame,
        int roi_x,
        int roi_y,
        int roi_w,
        int roi_h,
        int out_w,
        int out_h
    );
    std::string ProcessFrameBuffer(const uint8_t* input_frame, size_t input_size);
    std::string ProcessFrameFieldPhaseBuffer(const uint8_t* input_frame, size_t input_size, int field_phase);
    std::string ProcessFrameNoDeinterlaceBuffer(const uint8_t* input_frame, size_t input_size);
    std::string ProcessFrameDeinterlaceOnlyBuffer(const uint8_t* input_frame, size_t input_size);
    std::string ProcessFramePreprocessOnlyBuffer(const uint8_t* input_frame, size_t input_size);
    std::string ProcessFramePreprocessRoiRgbBuffer(
        const uint8_t* input_frame,
        size_t input_size,
        int roi_x,
        int roi_y,
        int roi_w,
        int roi_h,
        int out_w,
        int out_h
    );
    CudaTensorBuffer ProcessFramePreprocessRoiTensorCuda(
        const std::string& input_frame,
        int roi_x,
        int roi_y,
        int roi_w,
        int roi_h,
        int out_w,
        int out_h,
        const std::string& dtype_name
    );
    CudaTensorBuffer ProcessFramePreprocessRoiTensorCudaBuffer(
        const uint8_t* input_frame,
        size_t input_size,
        int roi_x,
        int roi_y,
        int roi_w,
        int roi_h,
        int out_w,
        int out_h,
        const std::string& dtype_name
    );

    void SetRoi(int roi_x, int roi_y, int roi_w, int roi_h);
    void SetRoiPosition(int roi_x, int roi_y);
    void SetRoiSize(int roi_w, int roi_h);
    void GetRoi(int& roi_x, int& roi_y, int& roi_w, int& roi_h) const;

    void SetSrModeAuto();
    void SetSrScaleManual(int sr_scale);
    int GetEffectiveSrScale() const;
    bool IsSrAutoMode() const;
    void SetMaxAutoSrScale(int sr_scale);
    int GetMaxAutoSrScale() const;
    void SetSrFlavor(SrFlavor sr_flavor);
    void SetSrFlavorByName(const std::string& sr_flavor_name);
    SrFlavor GetSrFlavor() const;
    std::string GetSrFlavorName() const;
    void SetDeinterlaceEnabled(bool enabled);
    bool IsDeinterlaceEnabled() const;
    void SetDeinterlaceMethod(DeinterlaceMethod method);
    void SetDeinterlaceMethodByName(const std::string& method_name);
    DeinterlaceMethod GetDeinterlaceMethod() const;
    std::string GetDeinterlaceMethodName() const;
    void SetDenoiseMethod(DenoiseMethod method);
    void SetDenoiseMethodByName(const std::string& method_name);
    DenoiseMethod GetDenoiseMethod() const;
    std::string GetDenoiseMethodName() const;
    void SetDenoiseStrength(float strength);
    float GetDenoiseStrength() const;
    void SetSubpixelShift(float shift_x, float shift_y);
    void SetGpuReadyMode(bool enabled);
    void GetSubpixelShift(float& shift_x, float& shift_y) const;
    void SetColorSpace(ColorSpace color_space);
    void SetColorSpaceByName(const std::string& color_space_name);
    ColorSpace GetColorSpace() const;
    std::string GetColorSpaceName() const;
    void SetColorRange(ColorRange color_range);
    void SetColorRangeByName(const std::string& color_range_name);
    ColorRange GetColorRange() const;
    std::string GetColorRangeName() const;
    void SetEffectsConfig(
        bool enabled,
        float opacity,
        const std::string& blend_mode,
        const std::string& blur_method,
        float blur_radius,
        const std::string& blur_target,
        float layer1_opacity = 1.0f,
        const std::string& key_mode = "off",
        int key_color_r = 0,
        int key_color_g = 255,
        int key_color_b = 0,
        float key_similarity = 0.25f,
        float key_softness = 0.10f,
        float spill_suppression = 0.25f,
        float luma_low = 0.0f,
        float luma_high = 1.0f,
        float luma_softness = 0.10f,
        bool key_invert = false,
        float key_edge_feather = 0.0f,
        bool output_connected = true,
        bool effect_color_from_alpha = false,
        bool effect_alpha_from_color = false,
        bool explicit_compositor_layers = false,
        float blur_aspect = 1.0f
    );
    void UploadEffectMediaRgba(const uint8_t* rgba, size_t bytes, int width, int height);
    void SetEffectsInputTransform(
        float transform_x,
        float transform_y,
        float transform_z,
        float rotate_x,
        float rotate_y,
        float rotate_z,
        float aspect_x = 1.0f,
        float aspect_y = 1.0f
    );
    void SetEffectLayerConfig(
        int layer_index,
        bool enabled,
        float opacity,
        const std::string& blend_mode,
        const std::string& blur_method,
        float blur_radius,
        const std::string& blur_target,
        const std::string& key_mode = "off",
        int key_color_r = 0,
        int key_color_g = 255,
        int key_color_b = 0,
        float key_similarity = 0.25f,
        float key_softness = 0.10f,
        float spill_suppression = 0.25f,
        float luma_low = 0.0f,
        float luma_high = 1.0f,
        float luma_softness = 0.10f,
        bool key_invert = false,
        float key_edge_feather = 0.0f,
        bool effect_color_from_alpha = false,
        bool effect_alpha_from_color = false,
        bool preserve_color_from_alpha_opacity = false,
        bool source_from_effects_input = false,
        bool key_alpha_from_effects_input = false,
        const std::string& mask_pattern = "off",
        float mask_softness = 0.0f,
        float mask_aspect = 1.0f,
        bool mask_invert = false,
        float mask_size = 1.0f,
        float mask_x = 0.0f,
        float mask_y = 0.0f,
        float mask_rotation = 0.0f,
        float transform_x = 0.0f,
        float transform_y = 0.0f,
        float transform_z = 0.0f,
        float rotate_x = 0.0f,
        float rotate_y = 0.0f,
        float rotate_z = 0.0f,
        float aspect_x = 1.0f,
        float aspect_y = 1.0f,
        bool materialize_key_alpha = false,
        float blur_aspect = 1.0f
    );
    void UploadEffectLayerMediaRgba(
        int layer_index,
        const uint8_t* rgba,
        size_t bytes,
        int width,
        int height
    );
    void UploadEffectLayerSourceRgba(
        int layer_index,
        int source_slot,
        const uint8_t* rgba,
        size_t bytes,
        int width,
        int height
    );
    void ClearEffectLayerChannelRoutes(int layer_index);
    void SetEffectLayerChannelRoute(
        int layer_index,
        int target_channel,
        int source_channel,
        const std::string& blur_method,
        float blur_radius,
        float transform_x,
        float transform_y,
        float transform_z,
        float rotate_x,
        float rotate_y,
        float rotate_z,
        float aspect_x,
        float aspect_y,
        const std::string& generator_type = "off",
        int key_color_r = 0,
        int key_color_g = 255,
        int key_color_b = 0,
        float key_similarity = 0.25f,
        float key_softness = 0.10f,
        bool key_invert = false,
        const std::string& mask_pattern = "off",
        float mask_softness = 0.0f,
        float mask_aspect = 1.0f,
        bool mask_invert = false,
        float mask_size = 1.0f,
        float mask_x = 0.0f,
        float mask_y = 0.0f,
        float mask_rotation = 0.0f,
        float blur_aspect = 1.0f
    );
    void SetEffectLayerChannelAlphaMix(
        int layer_index,
        int target_channel,
        bool enabled,
        const AlphaMixOperandConfig& base_operand,
        const std::vector<AlphaMixOpConfig>& ops
    );
    void SetEffectLayerTemporal(int layer_index, const std::string& id, const std::string& mode, float duration, float rate, float decay = 0, const std::string& background = "live");
    void SetEffectLayerComposition(int layer_index, int target, int source);
    void SetEffectLayerCompositionSource(int layer_index, int slot, int source);
    std::string GetEffectsRgbaOutput();
    void SetEffectLayerColorStages(int layer_index, const std::vector<ColorStageConfig>& stages);
    void SetEffectLayerAlphaMix(
        int layer_index,
        bool enabled,
        const AlphaMixOperandConfig& base_operand,
        const std::vector<AlphaMixOpConfig>& ops
    );
    void SetColorStages(const std::vector<ColorStageConfig>& stages);
    void ClearEffectMedia();

    int width() const { return width_; }
    int height() const { return height_; }
    int sr_scale() const;

private:
    static constexpr int kFirstEffectLayer = 1;
    static constexpr int kLastEffectLayer = 64;
    static constexpr size_t kEffectLayerCount = kLastEffectLayer - kFirstEffectLayer + 1;

    struct EffectLayerState {
        struct ImageSourceState {
            int width = 0;
            int height = 0;
            size_t capacity_bytes = 0;
            uint8_t* d_rgba = nullptr;
            uchar3* d_color = nullptr;
            uchar3* d_color_temp = nullptr;
            uint8_t* d_alpha = nullptr;
        };

        struct ChannelRouteState {
            int source_channel = -1;
            int blur_method = 0;
            float blur_radius = 0.0f;
            float blur_aspect = 1.0f;
            float transform_x = 0.0f;
            float transform_y = 0.0f;
            float transform_z = 0.0f;
            float rotate_x = 0.0f;
            float rotate_y = 0.0f;
            float rotate_z = 0.0f;
            float aspect_x = 1.0f;
            float aspect_y = 1.0f;
            int generator_type = 0;
            uchar3 key_color = make_uchar3(0, 255, 0);
            float key_similarity = 0.25f;
            float key_softness = 0.10f;
            bool key_invert = false;
            int mask_pattern_code = 0;
            float mask_softness = 0.0f;
            float mask_aspect = 1.0f;
            bool mask_invert = false;
            float mask_size = 1.0f;
            float mask_x = 0.0f;
            float mask_y = 0.0f;
            float mask_rotation = 0.0f;
            bool alpha_mix_enabled = false;
            AlphaMixOperandConfig alpha_mix_base{};
            std::vector<AlphaMixOpConfig> alpha_mix_ops{};
        };

        bool enabled = false;
        float opacity = 1.0f;
        int blend_mode = 0;
        int key_mode = 0;
        uchar3 key_color = make_uchar3(0, 255, 0);
        float key_similarity = 0.25f;
        float key_softness = 0.10f;
        float spill_suppression = 0.25f;
        float luma_low = 0.0f;
        float luma_high = 1.0f;
        float luma_softness = 0.10f;
        bool key_invert = false;
        float key_edge_feather = 0.0f;
        bool materialize_key_alpha = false;
        bool color_from_alpha = false;
        bool alpha_from_color = false;
        bool preserve_color_from_alpha_opacity = false;
        bool source_from_effects_input = false;
        bool key_alpha_from_effects_input = false;
        int blur_method = 0;
        float blur_radius = 0.0f;
        float blur_aspect = 1.0f;
        int blur_target = 0;
        std::string mask_pattern = "off";
        int mask_pattern_code = 0;
        float mask_softness = 0.0f;
        float mask_aspect = 1.0f;
        bool mask_invert = false;
        float mask_size = 1.0f;
        float mask_x = 0.0f;
        float mask_y = 0.0f;
        float mask_rotation = 0.0f;
        float transform_x = 0.0f;
        float transform_y = 0.0f;
        float transform_z = 0.0f;
        float rotate_x = 0.0f;
        float rotate_y = 0.0f;
        float rotate_z = 0.0f;
        float aspect_x = 1.0f;
        float aspect_y = 1.0f;
        bool channel_routing_enabled = false;
        std::array<ChannelRouteState, 4> channel_routes{};
        std::vector<ColorStageConfig> color_stages{};
        bool alpha_mix_enabled = false;
        AlphaMixOperandConfig alpha_mix_base{};
        std::vector<AlphaMixOpConfig> alpha_mix_ops{};
        int media_width = 0;
        int media_height = 0;
        size_t media_capacity_bytes = 0;
        uint8_t* d_media_rgba = nullptr;
        std::array<ImageSourceState, 8> image_sources{};
    };

    std::string ProcessFrameInternal(
        const uint8_t* input_frame,
        size_t input_size,
        bool deinterlace_only,
        bool force_deinterlace,
        bool force_disable_deinterlace,
        int field_phase_override = -1
    );
    void InitializeBuffers();
    void ValidateConfiguration() const;
    void ClampRoi();
    void ConfigureSrScaleLocked(int requested_scale, bool auto_mode);
    bool EnsureSrBufferCapacityLocked(int target_scale, cudaError_t& last_error);
    bool EffectsActiveLocked() const;
    const uchar3* ApplyColorStages(const uchar3* input, bool after_composite);
    const uchar3* ApplyColorStages(const uchar3* input, const std::vector<ColorStageConfig>& stages);
    void Cleanup();

    int width_;
    int height_;

    int roi_x_;
    int roi_y_;
    int roi_w_;
    int roi_h_;
    mutable std::mutex state_mutex_;
    std::mutex process_mutex_;

    bool enable_placeholder_sr_;
    bool enable_deinterlace_;
    DeinterlaceMethod deinterlace_method_;
    DenoiseMethod denoise_method_;
    float denoise_strength_;
    SrFlavor sr_flavor_;
    bool auto_sr_scale_;
    int max_auto_sr_scale_;
    int sr_requested_scale_;
    int sr_scale_;
    int sr_width_;
    int sr_height_;
    int sr_buffer_scale_capacity_;
    int auto_sr_pending_scale_;
    int auto_sr_pending_frames_;
    int auto_sr_settle_frames_;
    float subpixel_shift_x_;
    float subpixel_shift_y_;
    bool gpu_ready_mode_ = false;
    ColorSpace color_space_;
    ColorRange color_range_;
    float effects_layer1_opacity_;
    bool effects_output_connected_;
    bool effects_explicit_compositor_layers_;
    // Lower-level setters remain usable until a caller explicitly bypasses the graph.
    bool effects_bypassed_ = false;
    float effects_input_transform_x_;
    float effects_input_transform_y_;
    float effects_input_transform_z_;
    float effects_input_rotate_x_;
    float effects_input_rotate_y_;
    float effects_input_rotate_z_;
    float effects_input_aspect_x_;
    float effects_input_aspect_y_;
    std::array<EffectLayerState, kEffectLayerCount> effect_layers_;
    std::vector<ColorStageConfig> color_stages_;

    size_t uyvy_bytes_;
    size_t rgb_pixels_;

    cudaStream_t stream_;

    uint8_t* d_uyvy_in_;
    uint8_t* d_uyvy_out_;
    uchar3* d_rgb_full_;
    uchar3* d_rgb_bob_;
    uchar3* d_rgb_denoise_;
    uchar3* d_rgb_prev_full_;
    uchar3* d_rgb_sr_;
    uchar3* d_rgb_zoom_;
    uchar3* d_effect_color_a_;
    uchar3* d_effect_color_b_;
    uchar3* d_effect_operand_color_a_;
    uchar3* d_effect_operand_color_b_;
    uint8_t* d_effect_alpha_a_;
    uint8_t* d_effect_alpha_b_;
    uint8_t* d_effect_channel_a_;
    uint8_t* d_effect_channel_b_;
    struct TemporalConfig { std::string id; int mode = 0; float duration = 0; float rate = 0; float decay = 0; bool black_background = false; };
    struct TemporalState {
        float4* history = nullptr;
        std::string id;
        int mode = 0;
        bool valid = false;
        double phase = 0;
        double elapsed = 0;
        float decay = 0;
        std::chrono::steady_clock::time_point last{};
    };
    std::array<TemporalConfig, 64> temporal_configs_{};
    std::array<TemporalState, 65> temporal_states_{};
    struct CompositionBuffer { uchar3* color = nullptr; uint8_t* alpha = nullptr; };
    std::array<CompositionBuffer, 65> composition_buffers_{};
    std::array<int, 64> composition_targets_{};
    std::array<int, 64> composition_sources_{};
    std::array<std::array<int, 8>, 64> composition_image_sources_{};
    const uchar3* last_effects_color_ = nullptr;
    const uint8_t* last_effects_alpha_ = nullptr;
    uchar3* d_effect_composite_;
    uchar3* d_effect_composite_b_;
    uint8_t* d_effect_composite_alpha_a_;
    uint8_t* d_effect_composite_alpha_b_;
    uchar3* d_color_stage_a_;
    uchar3* d_color_stage_b_;
    bool has_prev_rgb_full_;
    uint8_t* h_output_pinned_;
    uint8_t* h_rgb_output_pinned_;
    size_t h_rgb_output_capacity_bytes_;

    std::vector<uint8_t> host_output_;
    std::vector<uint8_t> host_rgb_output_;
};

} // namespace vp
