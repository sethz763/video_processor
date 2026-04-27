#include "cuda/kernels.cuh"

#include <algorithm>
#include <cmath>

namespace vp::cuda_kernels {
namespace {

__device__ inline uint8_t ClampToU8(float v) {
    v = fminf(255.0f, fmaxf(0.0f, v));
    return static_cast<uint8_t>(v);
}

__device__ inline uchar3 MakeRgbFromYuv(uint8_t y, uint8_t u, uint8_t v) {
    const float yf = static_cast<float>(y);
    const float uf = static_cast<float>(u) - 128.0f;
    const float vf = static_cast<float>(v) - 128.0f;

    const float r = yf + 1.402f * vf;
    const float g = yf - 0.344136f * uf - 0.714136f * vf;
    const float b = yf + 1.772f * uf;

    return make_uchar3(ClampToU8(r), ClampToU8(g), ClampToU8(b));
}

__device__ inline void RgbToYuv(const uchar3& rgb, float& y, float& u, float& v) {
    const float r = static_cast<float>(rgb.x);
    const float g = static_cast<float>(rgb.y);
    const float b = static_cast<float>(rgb.z);

    y = 0.299f * r + 0.587f * g + 0.114f * b;
    u = -0.169f * r - 0.331f * g + 0.5f * b + 128.0f;
    v = 0.5f * r - 0.419f * g - 0.081f * b + 128.0f;
}

__global__ void UyvyToRgbKernel(const uint8_t* uyvy, uchar3* rgb, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int pair_x = x >> 1;
    const int pair_index = (y * (width >> 1) + pair_x) * 4;

    const uint8_t u = uyvy[pair_index + 0];
    const uint8_t y0 = uyvy[pair_index + 1];
    const uint8_t v = uyvy[pair_index + 2];
    const uint8_t y1 = uyvy[pair_index + 3];

    const uint8_t luma = (x & 1) ? y1 : y0;
    rgb[y * width + x] = MakeRgbFromYuv(luma, u, v);
}

__global__ void BobDeinterlaceKernel(const uchar3* rgb_in, uchar3* rgb_out, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    if ((y & 1) == 0) {
        rgb_out[y * width + x] = rgb_in[y * width + x];
        return;
    }

    const int y_prev = max(0, y - 1);
    const int y_next = min(height - 1, y + 1);

    const uchar3 a = rgb_in[y_prev * width + x];
    const uchar3 b = rgb_in[y_next * width + x];

    rgb_out[y * width + x] = make_uchar3(
        static_cast<uint8_t>((static_cast<int>(a.x) + static_cast<int>(b.x)) >> 1),
        static_cast<uint8_t>((static_cast<int>(a.y) + static_cast<int>(b.y)) >> 1),
        static_cast<uint8_t>((static_cast<int>(a.z) + static_cast<int>(b.z)) >> 1)
    );
}

__device__ inline float CubicWeight(float x) {
    x = fabsf(x);
    if (x <= 1.0f) {
        return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
    }
    if (x < 2.0f) {
        return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
    }
    return 0.0f;
}

__device__ inline uchar3 SampleBicubic(const uchar3* src, int width, int height, float fx, float fy) {
    const int x = static_cast<int>(floorf(fx));
    const int y = static_cast<int>(floorf(fy));

    float sum_r = 0.0f;
    float sum_g = 0.0f;
    float sum_b = 0.0f;
    float sum_w = 0.0f;

    for (int j = -1; j <= 2; ++j) {
        for (int i = -1; i <= 2; ++i) {
            const int sx = max(0, min(width - 1, x + i));
            const int sy = max(0, min(height - 1, y + j));
            const float wx = CubicWeight(fx - static_cast<float>(x + i));
            const float wy = CubicWeight(fy - static_cast<float>(y + j));
            const float w = wx * wy;
            const uchar3 p = src[sy * width + sx];

            sum_r += w * static_cast<float>(p.x);
            sum_g += w * static_cast<float>(p.y);
            sum_b += w * static_cast<float>(p.z);
            sum_w += w;
        }
    }

    if (sum_w <= 1e-6f) {
        return src[max(0, min(height - 1, y)) * width + max(0, min(width - 1, x))];
    }

    return make_uchar3(
        ClampToU8(sum_r / sum_w),
        ClampToU8(sum_g / sum_w),
        ClampToU8(sum_b / sum_w)
    );
}

__global__ void UpscaleBicubicKernel(
    const uchar3* rgb_in,
    int in_width,
    int in_height,
    uchar3* rgb_out,
    int out_width,
    int out_height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) {
        return;
    }

    const float scale_x = static_cast<float>(in_width) / static_cast<float>(out_width);
    const float scale_y = static_cast<float>(in_height) / static_cast<float>(out_height);

    const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
    const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;

    rgb_out[y * out_width + x] = SampleBicubic(rgb_in, in_width, in_height, src_x, src_y);
}

__device__ inline uchar3 SampleBilinear(const uchar3* src, int width, int height, float x, float y) {
    x = fminf(static_cast<float>(width - 1), fmaxf(0.0f, x));
    y = fminf(static_cast<float>(height - 1), fmaxf(0.0f, y));

    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = min(width - 1, x0 + 1);
    const int y1 = min(height - 1, y0 + 1);

    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);

    const uchar3 p00 = src[y0 * width + x0];
    const uchar3 p10 = src[y0 * width + x1];
    const uchar3 p01 = src[y1 * width + x0];
    const uchar3 p11 = src[y1 * width + x1];

    const float r0 = p00.x + tx * (p10.x - p00.x);
    const float g0 = p00.y + tx * (p10.y - p00.y);
    const float b0 = p00.z + tx * (p10.z - p00.z);

    const float r1 = p01.x + tx * (p11.x - p01.x);
    const float g1 = p01.y + tx * (p11.y - p01.y);
    const float b1 = p01.z + tx * (p11.z - p01.z);

    return make_uchar3(
        ClampToU8(r0 + ty * (r1 - r0)),
        ClampToU8(g0 + ty * (g1 - g0)),
        ClampToU8(b0 + ty * (b1 - b0))
    );
}

__global__ void CropZoomBilinearKernel(
    const uchar3* rgb_in,
    int src_width,
    int src_height,
    uchar3* rgb_out,
    int out_width,
    int out_height,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) {
        return;
    }

    const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(out_width);
    const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(out_height);

    const float src_x = static_cast<float>(roi_x) + u * static_cast<float>(roi_w - 1);
    const float src_y = static_cast<float>(roi_y) + v * static_cast<float>(roi_h - 1);

    rgb_out[y * out_width + x] = SampleBilinear(rgb_in, src_width, src_height, src_x, src_y);
}

__global__ void UpscaleBilinearKernel(
    const uchar3* rgb_in,
    int in_width,
    int in_height,
    uchar3* rgb_out,
    int out_width,
    int out_height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) {
        return;
    }

    const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(out_width);
    const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(out_height);
    const float src_x = u * static_cast<float>(in_width - 1);
    const float src_y = v * static_cast<float>(in_height - 1);

    rgb_out[y * out_width + x] = SampleBilinear(rgb_in, in_width, in_height, src_x, src_y);
}

__global__ void RgbToUyvyKernel(const uchar3* rgb, uint8_t* uyvy, int width, int height) {
    const int pair_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int pairs_per_row = width >> 1;
    if (pair_x >= pairs_per_row || y >= height) {
        return;
    }

    const int x0 = pair_x << 1;
    const int x1 = x0 + 1;

    const uchar3 p0 = rgb[y * width + x0];
    const uchar3 p1 = rgb[y * width + x1];

    float y0, u0, v0;
    float y1, u1, v1;
    RgbToYuv(p0, y0, u0, v0);
    RgbToYuv(p1, y1, u1, v1);

    const uint8_t y0_u8 = ClampToU8(y0);
    const uint8_t y1_u8 = ClampToU8(y1);
    const uint8_t u_u8 = ClampToU8((u0 + u1) * 0.5f);
    const uint8_t v_u8 = ClampToU8((v0 + v1) * 0.5f);

    const int base = (y * pairs_per_row + pair_x) * 4;
    uyvy[base + 0] = u_u8;
    uyvy[base + 1] = y0_u8;
    uyvy[base + 2] = v_u8;
    uyvy[base + 3] = y1_u8;
}

inline dim3 Grid2D(int width, int height, int bx = 16, int by = 16) {
    return dim3((width + bx - 1) / bx, (height + by - 1) / by, 1);
}

} // namespace

void LaunchUyvyToRgb(const uint8_t* d_uyvy, uchar3* d_rgb, int width, int height, cudaStream_t stream) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    UyvyToRgbKernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_uyvy,
        d_rgb,
        width,
        height
    );
}

void LaunchBobDeinterlace(const uchar3* d_rgb_in, uchar3* d_rgb_out, int width, int height, cudaStream_t stream) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    BobDeinterlaceKernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_rgb_in,
        d_rgb_out,
        width,
        height
    );
}

void LaunchUpscaleBicubic(
    const uchar3* d_rgb_in,
    int in_width,
    int in_height,
    uchar3* d_rgb_out,
    int out_width,
    int out_height,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    UpscaleBicubicKernel<<<Grid2D(out_width, out_height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_rgb_in,
        in_width,
        in_height,
        d_rgb_out,
        out_width,
        out_height
    );
}

void LaunchUpscaleBilinear(
    const uchar3* d_rgb_in,
    int in_width,
    int in_height,
    uchar3* d_rgb_out,
    int out_width,
    int out_height,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    UpscaleBilinearKernel<<<Grid2D(out_width, out_height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_rgb_in,
        in_width,
        in_height,
        d_rgb_out,
        out_width,
        out_height
    );
}

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
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    CropZoomBilinearKernel<<<Grid2D(out_width, out_height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_rgb_in,
        src_width,
        src_height,
        d_rgb_out,
        out_width,
        out_height,
        roi_x,
        roi_y,
        roi_w,
        roi_h
    );
}

void LaunchRgbToUyvy(const uchar3* d_rgb, uint8_t* d_uyvy, int width, int height, cudaStream_t stream) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    const dim3 grid((width / 2 + kBlockX - 1) / kBlockX, (height + kBlockY - 1) / kBlockY, 1);
    const dim3 block(kBlockX, kBlockY, 1);

    RgbToUyvyKernel<<<grid, block, 0, stream>>>(d_rgb, d_uyvy, width, height);
}

} // namespace vp::cuda_kernels
