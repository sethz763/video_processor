#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace vp::cuda_kernels {

void LaunchUyvyToRgb(
    const uint8_t* d_uyvy,
    uchar3* d_rgb,
    int width,
    int height,
    int color_matrix,
    int color_range,
    cudaStream_t stream
);

void LaunchUyvyFieldToRgb(
    const uint8_t* d_uyvy,
    uchar3* d_rgb,
    int width,
    int height,
    int source_field_phase,
    int color_matrix,
    int color_range,
    cudaStream_t stream
);

void LaunchUyvyCropZoomNearest(
    const uint8_t* d_uyvy_in,
    int src_width,
    int src_height,
    uint8_t* d_uyvy_out,
    int out_width,
    int out_height,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    bool preserve_field_parity,
    cudaStream_t stream
);

void LaunchUyvySubpixelShift(
    const uint8_t* d_uyvy_in,
    uint8_t* d_uyvy_out,
    int width,
    int height,
    float shift_x,
    float shift_y,
    cudaStream_t stream
);

void LaunchBobDeinterlace(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    int field_phase,
    cudaStream_t stream
);

void LaunchBlendDeinterlace(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    cudaStream_t stream
);

void LaunchEdgeAdaptiveDeinterlace(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    int field_phase,
    cudaStream_t stream
);

void LaunchUpscaleBicubic(
    const uchar3* d_rgb_in,
    int in_width,
    int in_height,
    uchar3* d_rgb_out,
    int out_width,
    int out_height,
    cudaStream_t stream
);

void LaunchUpscaleBilinear(
    const uchar3* d_rgb_in,
    int in_width,
    int in_height,
    uchar3* d_rgb_out,
    int out_width,
    int out_height,
    cudaStream_t stream
);

void LaunchUpscaleBilinearSharp(
    const uchar3* d_rgb_in,
    int in_width,
    int in_height,
    uchar3* d_rgb_out,
    int out_width,
    int out_height,
    cudaStream_t stream
);

void LaunchCropZoomBilinear(
    const uchar3* d_rgb_in,
    int src_width,
    int src_height,
    uchar3* d_rgb_out,
    int out_width,
    int out_height,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    cudaStream_t stream
);

void LaunchCropZoomBilinearSharp(
    const uchar3* d_rgb_in,
    int src_width,
    int src_height,
    uchar3* d_rgb_out,
    int out_width,
    int out_height,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    cudaStream_t stream
);

void LaunchCropZoomBicubic(
    const uchar3* d_rgb_in,
    int src_width,
    int src_height,
    uchar3* d_rgb_out,
    int out_width,
    int out_height,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    cudaStream_t stream
);

void LaunchCropCopyRgb(
    const uchar3* d_rgb_in,
    int src_width,
    int src_height,
    uchar3* d_rgb_out,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    cudaStream_t stream
);

void LaunchSharpen3x3(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    bool preserve_field_parity,
    cudaStream_t stream
);

void LaunchDenoiseLumaGaussian3x3(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
);

void LaunchDenoiseLumaMedian3x3(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
);

void LaunchDenoiseLumaBilateral3x3(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
);

void LaunchDenoiseLumaBilateral5x5(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
);

void LaunchDenoiseUyvyLumaGaussian3x3(
    const uint8_t* d_uyvy_in,
    uint8_t* d_uyvy_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
);

void LaunchDenoiseUyvyLumaMedian3x3(
    const uint8_t* d_uyvy_in,
    uint8_t* d_uyvy_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
);

void LaunchDenoiseUyvyLumaBilateral3x3(
    const uint8_t* d_uyvy_in,
    uint8_t* d_uyvy_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
);

void LaunchDenoiseUyvyLumaBilateral5x5(
    const uint8_t* d_uyvy_in,
    uint8_t* d_uyvy_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
);

void LaunchColorAdjustment(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    int stage_type,
    float param0,
    float param1,
    float param2,
    float param3,
    float param4,
    float param5,
    float param6,
    float param7,
    float param8,
    bool invert,
    cudaStream_t stream
);

void LaunchScaleRgbaToColorAlpha(
    const uint8_t* d_rgba,
    int src_width,
    int src_height,
    uchar3* d_color,
    uint8_t* d_alpha,
    int out_width,
    int out_height,
    cudaStream_t stream
);

void LaunchConvertColorAlphaChannels(
    uchar3* d_color,
    uint8_t* d_alpha,
    int width,
    int height,
    bool color_from_alpha,
    bool alpha_from_color,
    cudaStream_t stream
);

void LaunchExtractColorAlphaChannel(
    const uchar3* d_color,
    const uint8_t* d_alpha,
    uint8_t* d_output,
    int width,
    int height,
    int source_channel,
    cudaStream_t stream
);

void LaunchWriteColorAlphaChannel(
    const uint8_t* d_input,
    uchar3* d_color,
    uint8_t* d_alpha,
    int width,
    int height,
    int target_channel,
    cudaStream_t stream
);

void LaunchTransformAlpha3D(
    const uint8_t* d_input,
    uint8_t* d_output,
    int width,
    int height,
    float translate_x,
    float translate_y,
    float translate_z,
    float rotate_x,
    float rotate_y,
    float rotate_z,
    cudaStream_t stream
);

void LaunchTransformColorAlpha3D(
    const uchar3* d_input_color,
    const uint8_t* d_input_alpha,
    uchar3* d_output_color,
    uint8_t* d_output_alpha,
    int width,
    int height,
    float translate_x,
    float translate_y,
    float translate_z,
    float rotate_x,
    float rotate_y,
    float rotate_z,
    cudaStream_t stream
);

void LaunchApplyProceduralAlphaMask(
    uint8_t* d_alpha,
    int width,
    int height,
    int pattern,
    float softness,
    float aspect,
    bool invert,
    float size,
    float position_x,
    float position_y,
    cudaStream_t stream
);

void LaunchGenerateKeyAlpha(
    const uchar3* d_color,
    uint8_t* d_alpha,
    int width,
    int height,
    int key_mode,
    uchar3 key_color,
    float key_similarity,
    float key_softness,
    float luma_low,
    float luma_high,
    float luma_softness,
    bool key_invert,
    cudaStream_t stream
);

void LaunchBlurColor(
    const uchar3* d_input,
    uchar3* d_temp,
    uchar3* d_output,
    int width,
    int height,
    float radius,
    int method,
    cudaStream_t stream
);

void LaunchBlurAlpha(
    const uint8_t* d_input,
    uint8_t* d_temp,
    uint8_t* d_output,
    int width,
    int height,
    float radius,
    int method,
    cudaStream_t stream
);

void LaunchCompositeColorAlpha(
    const uchar3* d_background,
    const uint8_t* d_background_alpha,
    const uchar3* d_foreground,
    const uint8_t* d_alpha,
    uchar3* d_output,
    uint8_t* d_output_alpha,
    int width,
    int height,
    float initial_background_opacity,
    float opacity,
    int blend_mode,
    int key_mode,
    uchar3 key_color,
    float key_similarity,
    float key_softness,
    float spill_suppression,
    float luma_low,
    float luma_high,
    float luma_softness,
    bool key_invert,
    cudaStream_t stream
);

void LaunchMixAlpha(
    const uint8_t* d_base,
    const uint8_t* d_operand,
    uint8_t* d_output,
    int width,
    int height,
    int mode,
    float factor,
    cudaStream_t stream
);

void LaunchDenoiseFieldTemporalLuma(
    const uchar3* d_rgb_in,
    const uchar3* d_rgb_prev,
    uchar3* d_rgb_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
);

void LaunchRgbToUyvy(
    const uchar3* d_rgb,
    uint8_t* d_uyvy,
    int width,
    int height,
    int color_matrix,
    int color_range,
    cudaStream_t stream
);

void LaunchTensorToRgb(
    const void* d_tensor,
    int tensor_dtype,
    int tensor_layout,
    int channels,
    bool normalized_01,
    uchar3* d_rgb,
    int width,
    int height,
    cudaStream_t stream
);

void LaunchRgbToTensor(
    const uchar3* d_rgb,
    void* d_tensor,
    int tensor_dtype,
    int tensor_layout,
    int channels,
    bool normalized_01,
    int width,
    int height,
    cudaStream_t stream
);

} // namespace vp::cuda_kernels
