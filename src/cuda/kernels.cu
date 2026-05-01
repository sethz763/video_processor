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

__global__ void BlendDeinterlaceKernel(const uchar3* rgb_in, uchar3* rgb_out, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int y_prev = max(0, y - 1);
    const int y_next = min(height - 1, y + 1);

    const uchar3 p = rgb_in[y * width + x];
    const uchar3 a = rgb_in[y_prev * width + x];
    const uchar3 b = rgb_in[y_next * width + x];

    rgb_out[y * width + x] = make_uchar3(
        static_cast<uint8_t>((2 * static_cast<int>(p.x) + static_cast<int>(a.x) + static_cast<int>(b.x)) >> 2),
        static_cast<uint8_t>((2 * static_cast<int>(p.y) + static_cast<int>(a.y) + static_cast<int>(b.y)) >> 2),
        static_cast<uint8_t>((2 * static_cast<int>(p.z) + static_cast<int>(a.z) + static_cast<int>(b.z)) >> 2)
    );
}

__device__ inline float RgbLuma(const uchar3& p) {
    return 0.299f * static_cast<float>(p.x) + 0.587f * static_cast<float>(p.y) + 0.114f * static_cast<float>(p.z);
}

__global__ void EdgeAdaptiveDeinterlaceKernel(const uchar3* rgb_in, uchar3* rgb_out, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    if ((y & 1) == 0) {
        rgb_out[y * width + x] = rgb_in[y * width + x];
        return;
    }

    const int xm1 = max(0, x - 1);
    const int xp1 = min(width - 1, x + 1);
    const int y_prev = max(0, y - 1);
    const int y_next = min(height - 1, y + 1);

    const uchar3 v_a = rgb_in[y_prev * width + x];
    const uchar3 v_b = rgb_in[y_next * width + x];
    const uchar3 d1_a = rgb_in[y_prev * width + xm1];
    const uchar3 d1_b = rgb_in[y_next * width + xp1];
    const uchar3 d2_a = rgb_in[y_prev * width + xp1];
    const uchar3 d2_b = rgb_in[y_next * width + xm1];

    const float g_v = fabsf(RgbLuma(v_a) - RgbLuma(v_b));
    const float g_d1 = fabsf(RgbLuma(d1_a) - RgbLuma(d1_b));
    const float g_d2 = fabsf(RgbLuma(d2_a) - RgbLuma(d2_b));

    uchar3 out_a = v_a;
    uchar3 out_b = v_b;
    if (g_d1 < g_v && g_d1 <= g_d2) {
        out_a = d1_a;
        out_b = d1_b;
    } else if (g_d2 < g_v && g_d2 < g_d1) {
        out_a = d2_a;
        out_b = d2_b;
    }

    rgb_out[y * width + x] = make_uchar3(
        static_cast<uint8_t>((static_cast<int>(out_a.x) + static_cast<int>(out_b.x)) >> 1),
        static_cast<uint8_t>((static_cast<int>(out_a.y) + static_cast<int>(out_b.y)) >> 1),
        static_cast<uint8_t>((static_cast<int>(out_a.z) + static_cast<int>(out_b.z)) >> 1)
    );
}

__device__ inline uchar3 ApplyLumaDelta(const uchar3& src, float delta) {
    return make_uchar3(
        ClampToU8(static_cast<float>(src.x) + delta),
        ClampToU8(static_cast<float>(src.y) + delta),
        ClampToU8(static_cast<float>(src.z) + delta)
    );
}

__global__ void DenoiseLumaGaussian3x3Kernel(
    const uchar3* rgb_in,
    uchar3* rgb_out,
    int width,
    int height,
    float strength
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int xm1 = max(0, x - 1);
    const int xp1 = min(width - 1, x + 1);
    const int ym1 = max(0, y - 1);
    const int yp1 = min(height - 1, y + 1);

    const uchar3 c = rgb_in[y * width + x];
    const float y_center = RgbLuma(c);

    const float y_nw = RgbLuma(rgb_in[ym1 * width + xm1]);
    const float y_n = RgbLuma(rgb_in[ym1 * width + x]);
    const float y_ne = RgbLuma(rgb_in[ym1 * width + xp1]);
    const float y_w = RgbLuma(rgb_in[y * width + xm1]);
    const float y_e = RgbLuma(rgb_in[y * width + xp1]);
    const float y_sw = RgbLuma(rgb_in[yp1 * width + xm1]);
    const float y_s = RgbLuma(rgb_in[yp1 * width + x]);
    const float y_se = RgbLuma(rgb_in[yp1 * width + xp1]);

    const float y_blur = (
        y_nw + 2.0f * y_n + y_ne +
        2.0f * y_w + 4.0f * y_center + 2.0f * y_e +
        y_sw + 2.0f * y_s + y_se
    ) / 16.0f;

    const float y_new = y_center + strength * (y_blur - y_center);
    rgb_out[y * width + x] = ApplyLumaDelta(c, y_new - y_center);
}

__device__ inline float Median9(float a[9]) {
    for (int i = 1; i < 9; ++i) {
        float key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
    return a[4];
}

__global__ void DenoiseLumaMedian3x3Kernel(
    const uchar3* rgb_in,
    uchar3* rgb_out,
    int width,
    int height,
    float strength
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int xm1 = max(0, x - 1);
    const int xp1 = min(width - 1, x + 1);
    const int ym1 = max(0, y - 1);
    const int yp1 = min(height - 1, y + 1);

    const uchar3 c = rgb_in[y * width + x];
    const float y_center = RgbLuma(c);

    float samples[9] = {
        RgbLuma(rgb_in[ym1 * width + xm1]),
        RgbLuma(rgb_in[ym1 * width + x]),
        RgbLuma(rgb_in[ym1 * width + xp1]),
        RgbLuma(rgb_in[y * width + xm1]),
        y_center,
        RgbLuma(rgb_in[y * width + xp1]),
        RgbLuma(rgb_in[yp1 * width + xm1]),
        RgbLuma(rgb_in[yp1 * width + x]),
        RgbLuma(rgb_in[yp1 * width + xp1]),
    };

    const float y_med = Median9(samples);
    const float y_new = y_center + strength * (y_med - y_center);
    rgb_out[y * width + x] = ApplyLumaDelta(c, y_new - y_center);
}

__global__ void DenoiseFieldTemporalLumaKernel(
    const uchar3* rgb_in,
    const uchar3* rgb_prev,
    uchar3* rgb_out,
    int width,
    int height,
    float strength
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int xm1 = max(0, x - 1);
    const int xp1 = min(width - 1, x + 1);
    const int y2m = max(0, y - 2);
    const int y2p = min(height - 1, y + 2);

    const uchar3 c = rgb_in[y * width + x];
    const uchar3 p = rgb_prev[y * width + x];
    const float curr_luma = RgbLuma(c);
    const float prev_luma = RgbLuma(p);

    const float l_left = RgbLuma(rgb_in[y * width + xm1]);
    const float l_right = RgbLuma(rgb_in[y * width + xp1]);
    const float l_up2 = RgbLuma(rgb_in[y2m * width + x]);
    const float l_dn2 = RgbLuma(rgb_in[y2p * width + x]);
    const float field_spatial = (2.0f * curr_luma + l_left + l_right + l_up2 + l_dn2) / 6.0f;

    const float luma_diff = fabsf(curr_luma - prev_luma);
    const float motion_threshold = 22.0f;
    const float temporal_gate = fminf(1.0f, fmaxf(0.0f, (motion_threshold - luma_diff) / motion_threshold));

    const float spatial_mix = curr_luma + (0.55f * strength) * (field_spatial - curr_luma);
    const float temporal_mix = spatial_mix + (0.75f * strength * temporal_gate) * (prev_luma - spatial_mix);
    rgb_out[y * width + x] = ApplyLumaDelta(c, temporal_mix - curr_luma);
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

__global__ void CropZoomBicubicKernel(
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

    rgb_out[y * out_width + x] = SampleBicubic(rgb_in, src_width, src_height, src_x, src_y);
}

__global__ void Sharpen3x3Kernel(
    const uchar3* rgb_in,
    uchar3* rgb_out,
    int width,
    int height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int xm1 = max(0, x - 1);
    const int xp1 = min(width - 1, x + 1);
    const int ym1 = max(0, y - 1);
    const int yp1 = min(height - 1, y + 1);

    const uchar3 c = rgb_in[y * width + x];
    const uchar3 n = rgb_in[ym1 * width + x];
    const uchar3 s = rgb_in[yp1 * width + x];
    const uchar3 w = rgb_in[y * width + xm1];
    const uchar3 e = rgb_in[y * width + xp1];
    const uchar3 nw = rgb_in[ym1 * width + xm1];
    const uchar3 ne = rgb_in[ym1 * width + xp1];
    const uchar3 sw = rgb_in[yp1 * width + xm1];
    const uchar3 se = rgb_in[yp1 * width + xp1];

    auto sharpen_channel = [](float center, float north, float south, float west, float east, float c_nw, float c_ne, float c_sw, float c_se) {
        const float blur = (
            4.0f * center +
            2.0f * (north + south + west + east) +
            (c_nw + c_ne + c_sw + c_se)
        ) / 16.0f;
        // Keep this intentionally strong for A/B visibility during SR flavor experiments.
        const float amount = 2.4f;
        return ClampToU8(center + amount * (center - blur));
    };

    rgb_out[y * width + x] = make_uchar3(
        sharpen_channel(
            static_cast<float>(c.x),
            static_cast<float>(n.x),
            static_cast<float>(s.x),
            static_cast<float>(w.x),
            static_cast<float>(e.x),
            static_cast<float>(nw.x),
            static_cast<float>(ne.x),
            static_cast<float>(sw.x),
            static_cast<float>(se.x)
        ),
        sharpen_channel(
            static_cast<float>(c.y),
            static_cast<float>(n.y),
            static_cast<float>(s.y),
            static_cast<float>(w.y),
            static_cast<float>(e.y),
            static_cast<float>(nw.y),
            static_cast<float>(ne.y),
            static_cast<float>(sw.y),
            static_cast<float>(se.y)
        ),
        sharpen_channel(
            static_cast<float>(c.z),
            static_cast<float>(n.z),
            static_cast<float>(s.z),
            static_cast<float>(w.z),
            static_cast<float>(e.z),
            static_cast<float>(nw.z),
            static_cast<float>(ne.z),
            static_cast<float>(sw.z),
            static_cast<float>(se.z)
        )
    );
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

void LaunchBlendDeinterlace(const uchar3* d_rgb_in, uchar3* d_rgb_out, int width, int height, cudaStream_t stream) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    BlendDeinterlaceKernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_rgb_in,
        d_rgb_out,
        width,
        height
    );
}

void LaunchEdgeAdaptiveDeinterlace(const uchar3* d_rgb_in, uchar3* d_rgb_out, int width, int height, cudaStream_t stream) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    EdgeAdaptiveDeinterlaceKernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
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
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    CropZoomBicubicKernel<<<Grid2D(out_width, out_height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
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

void LaunchSharpen3x3(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    Sharpen3x3Kernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_rgb_in,
        d_rgb_out,
        width,
        height
    );
}

void LaunchDenoiseLumaGaussian3x3(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    DenoiseLumaGaussian3x3Kernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_rgb_in,
        d_rgb_out,
        width,
        height,
        fminf(1.0f, fmaxf(0.0f, strength))
    );
}

void LaunchDenoiseLumaMedian3x3(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    DenoiseLumaMedian3x3Kernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_rgb_in,
        d_rgb_out,
        width,
        height,
        fminf(1.0f, fmaxf(0.0f, strength))
    );
}

void LaunchDenoiseFieldTemporalLuma(
    const uchar3* d_rgb_in,
    const uchar3* d_rgb_prev,
    uchar3* d_rgb_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    DenoiseFieldTemporalLumaKernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_rgb_in,
        d_rgb_prev,
        d_rgb_out,
        width,
        height,
        fminf(1.0f, fmaxf(0.0f, strength))
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
