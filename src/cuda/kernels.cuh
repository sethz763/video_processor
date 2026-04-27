#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace vp::cuda_kernels {

void LaunchUyvyToRgb(
    const uint8_t* d_uyvy,
    uchar3* d_rgb,
    int width,
    int height,
    cudaStream_t stream
);

void LaunchBobDeinterlace(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
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

void LaunchRgbToUyvy(
    const uchar3* d_rgb,
    uint8_t* d_uyvy,
    int width,
    int height,
    cudaStream_t stream
);

} // namespace vp::cuda_kernels
