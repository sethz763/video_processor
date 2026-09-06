#include "cuda/kernels.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_fp16.h>

namespace vp::cuda_kernels {
namespace {

inline void CheckKernelLaunch(const char* op) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(op) + " failed: " + cudaGetErrorString(err));
    }
}

__device__ inline uint8_t ClampToU8(float v) {
    v = fminf(255.0f, fmaxf(0.0f, v));
    return static_cast<uint8_t>(v + 0.5f);
}

__device__ inline uchar3 MakeRgbFromYuv(uint8_t y, uint8_t u, uint8_t v, int color_matrix, int color_range) {
    // Limited-range conversion (Y:16-235, U/V:16-240).
    const float d = static_cast<float>(u) - 128.0f;
    const float e = static_cast<float>(v) - 128.0f;
    const float yv = static_cast<float>(y);

    if (color_range == 1) {
        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
        if (color_matrix == 1) {
            r = yv + 1.474600f * e;
            g = yv - 0.164553f * d - 0.571353f * e;
            b = yv + 1.881400f * d;
        } else {
            r = yv + 1.574800f * e;
            g = yv - 0.187324f * d - 0.468124f * e;
            b = yv + 1.855600f * d;
        }
        return make_uchar3(ClampToU8(r), ClampToU8(g), ClampToU8(b));
    }

    const float c = yv - 16.0f;

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    if (color_matrix == 1) {
        // BT.2020 non-constant luminance matrix (used with Rec.2020 HLG content).
        r = 1.164383f * c + 1.678674f * e;
        g = 1.164383f * c - 0.187326f * d - 0.650424f * e;
        b = 1.164383f * c + 2.141772f * d;
    } else {
        // BT.709 matrix.
        r = 1.164383f * c + 1.792741f * e;
        g = 1.164383f * c - 0.213249f * d - 0.532909f * e;
        b = 1.164383f * c + 2.112402f * d;
    }

    return make_uchar3(ClampToU8(r), ClampToU8(g), ClampToU8(b));
}

struct YuvF {
    float y;
    float u;
    float v;
};

__device__ inline YuvF RgbToYuv(const uchar3& rgb, int color_matrix, int color_range) {
    const float r = static_cast<float>(rgb.x);
    const float g = static_cast<float>(rgb.y);
    const float b = static_cast<float>(rgb.z);

    if (color_range == 1) {
        if (color_matrix == 1) {
            return {
                0.262700f * r + 0.678000f * g + 0.059300f * b,
                128.0f - 0.139630f * r - 0.360370f * g + 0.500000f * b,
                128.0f + 0.500000f * r - 0.459786f * g - 0.040214f * b,
            };
        }
        return {
            0.212600f * r + 0.715200f * g + 0.072200f * b,
            128.0f - 0.114572f * r - 0.385428f * g + 0.500000f * b,
            128.0f + 0.500000f * r - 0.454153f * g - 0.045847f * b,
        };
    }

    if (color_matrix == 1) {
        // BT.2020 non-constant luminance matrix.
        return {
            16.0f + 0.225613f * r + 0.582282f * g + 0.050928f * b,
            128.0f - 0.122655f * r - 0.316561f * g + 0.439216f * b,
            128.0f + 0.439216f * r - 0.403890f * g - 0.035325f * b,
        };
    }

    // BT.709 limited-range conversion (inverse of MakeRgbFromYuv).
    return {
        16.0f + 0.182586f * r + 0.614231f * g + 0.062007f * b,
        128.0f - 0.100644f * r - 0.338572f * g + 0.439216f * b,
        128.0f + 0.439216f * r - 0.398942f * g - 0.040274f * b,
    };
}

__global__ void UyvyToRgbKernel(const uint8_t* uyvy, uchar3* rgb, int width, int height, int color_matrix, int color_range) {
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
    rgb[y * width + x] = MakeRgbFromYuv(luma, u, v, color_matrix, color_range);
}

__global__ void UyvyFieldToRgbKernel(
    const uint8_t* uyvy,
    uchar3* rgb,
    int width,
    int height,
    int source_field_phase,
    int color_matrix,
    int color_range
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int source_y = min(height - 1, ((y >> 1) << 1) + (source_field_phase & 1));
    const int pair_x = x >> 1;
    const int pair_index = (source_y * (width >> 1) + pair_x) * 4;

    const uint8_t u = uyvy[pair_index + 0];
    const uint8_t y0 = uyvy[pair_index + 1];
    const uint8_t v = uyvy[pair_index + 2];
    const uint8_t y1 = uyvy[pair_index + 3];

    const uint8_t luma = (x & 1) ? y1 : y0;
    rgb[y * width + x] = MakeRgbFromYuv(luma, u, v, color_matrix, color_range);
}

__device__ inline void SampleUyvyPixel(
    const uint8_t* uyvy,
    int width,
    int height,
    int x,
    int y,
    uint8_t& out_y,
    uint8_t& out_u,
    uint8_t& out_v
) {
    const int sx = max(0, min(width - 1, x));
    const int sy = max(0, min(height - 1, y));
    const int pair_x = sx >> 1;
    const int pair_index = (sy * (width >> 1) + pair_x) * 4;

    const uint8_t u = uyvy[pair_index + 0];
    const uint8_t y0 = uyvy[pair_index + 1];
    const uint8_t v = uyvy[pair_index + 2];
    const uint8_t y1 = uyvy[pair_index + 3];

    out_y = (sx & 1) ? y1 : y0;
    out_u = u;
    out_v = v;
}

__global__ void UyvyCropZoomNearestKernel(
    const uint8_t* uyvy_in,
    int src_width,
    int src_height,
    uint8_t* uyvy_out,
    int out_width,
    int out_height,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    int preserve_field_parity
) {
    const int out_pair_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_pairs_per_row = out_width >> 1;
    if (out_pair_x >= out_pairs_per_row || y >= out_height) {
        return;
    }

    const int x0 = out_pair_x << 1;
    const int x1 = x0 + 1;

    const float u0 = (static_cast<float>(x0) + 0.5f) / static_cast<float>(out_width);
    const float u1 = (static_cast<float>(x1) + 0.5f) / static_cast<float>(out_width);
    const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(out_height);

    const int src_x0 = max(0, min(src_width - 1, static_cast<int>(roi_x + u0 * static_cast<float>(roi_w - 1) + 0.5f)));
    const int src_x1 = max(0, min(src_width - 1, static_cast<int>(roi_x + u1 * static_cast<float>(roi_w - 1) + 0.5f)));

    const int roi_y_min = max(0, min(src_height - 1, roi_y));
    const int roi_y_max = max(roi_y_min, min(src_height - 1, roi_y + roi_h - 1));
    const float src_y_f = roi_y + v * static_cast<float>(roi_h - 1);
    int src_y = max(roi_y_min, min(roi_y_max, static_cast<int>(src_y_f + 0.5f)));

    if (preserve_field_parity != 0) {
        // Anchor parity to ROI source phase so per-field ROI Y offsets
        // (e.g. interlaced field2 phase) are preserved through scaling.
        const int desired_parity = (y + roi_y) & 1;
        if ((src_y & 1) != desired_parity) {
            const int below = src_y - 1;
            const int above = src_y + 1;
            const bool below_ok = below >= roi_y_min;
            const bool above_ok = above <= roi_y_max;

            if (below_ok && above_ok) {
                const float below_err = fabsf(src_y_f - static_cast<float>(below));
                const float above_err = fabsf(src_y_f - static_cast<float>(above));
                src_y = (below_err <= above_err) ? below : above;
            } else if (below_ok) {
                src_y = below;
            } else if (above_ok) {
                src_y = above;
            }
        }
    }

    uint8_t y0, u_a, v_a;
    uint8_t y1, u_b, v_b;
    SampleUyvyPixel(uyvy_in, src_width, src_height, src_x0, src_y, y0, u_a, v_a);
    SampleUyvyPixel(uyvy_in, src_width, src_height, src_x1, src_y, y1, u_b, v_b);

    const uint8_t u = static_cast<uint8_t>((static_cast<int>(u_a) + static_cast<int>(u_b)) >> 1);
    const uint8_t vv = static_cast<uint8_t>((static_cast<int>(v_a) + static_cast<int>(v_b)) >> 1);

    const int out_base = (y * out_pairs_per_row + out_pair_x) * 4;
    uyvy_out[out_base + 0] = u;
    uyvy_out[out_base + 1] = y0;
    uyvy_out[out_base + 2] = vv;
    uyvy_out[out_base + 3] = y1;
}

__device__ inline uint8_t SampleUyvyLumaBilinear(const uint8_t* uyvy, int width, int height, float fx, float fy) {
    fx = fminf(static_cast<float>(width - 1), fmaxf(0.0f, fx));
    fy = fminf(static_cast<float>(height - 1), fmaxf(0.0f, fy));

    const int x0 = static_cast<int>(floorf(fx));
    const int y0 = static_cast<int>(floorf(fy));
    const int x1 = min(width - 1, x0 + 1);
    const int y1 = min(height - 1, y0 + 1);

    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);

    uint8_t l00, u00, v00;
    uint8_t l10, u10, v10;
    uint8_t l01, u01, v01;
    uint8_t l11, u11, v11;
    SampleUyvyPixel(uyvy, width, height, x0, y0, l00, u00, v00);
    SampleUyvyPixel(uyvy, width, height, x1, y0, l10, u10, v10);
    SampleUyvyPixel(uyvy, width, height, x0, y1, l01, u01, v01);
    SampleUyvyPixel(uyvy, width, height, x1, y1, l11, u11, v11);

    const float v0 = static_cast<float>(l00) + tx * (static_cast<float>(l10) - static_cast<float>(l00));
    const float v1 = static_cast<float>(l01) + tx * (static_cast<float>(l11) - static_cast<float>(l01));
    return ClampToU8(v0 + ty * (v1 - v0));
}

__device__ inline void SampleUyvyUvBilinear(
    const uint8_t* uyvy,
    int width,
    int height,
    float pair_fx,
    float fy,
    uint8_t& out_u,
    uint8_t& out_v
) {
    const int pairs = width >> 1;
    pair_fx = fminf(static_cast<float>(pairs - 1), fmaxf(0.0f, pair_fx));
    fy = fminf(static_cast<float>(height - 1), fmaxf(0.0f, fy));

    const int x0 = static_cast<int>(floorf(pair_fx));
    const int y0 = static_cast<int>(floorf(fy));
    const int x1 = min(pairs - 1, x0 + 1);
    const int y1 = min(height - 1, y0 + 1);

    const float tx = pair_fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);

    const int base00 = (y0 * pairs + x0) * 4;
    const int base10 = (y0 * pairs + x1) * 4;
    const int base01 = (y1 * pairs + x0) * 4;
    const int base11 = (y1 * pairs + x1) * 4;

    const float u00 = static_cast<float>(uyvy[base00 + 0]);
    const float u10 = static_cast<float>(uyvy[base10 + 0]);
    const float u01 = static_cast<float>(uyvy[base01 + 0]);
    const float u11 = static_cast<float>(uyvy[base11 + 0]);

    const float v00 = static_cast<float>(uyvy[base00 + 2]);
    const float v10 = static_cast<float>(uyvy[base10 + 2]);
    const float v01 = static_cast<float>(uyvy[base01 + 2]);
    const float v11 = static_cast<float>(uyvy[base11 + 2]);

    const float u0 = u00 + tx * (u10 - u00);
    const float u1 = u01 + tx * (u11 - u01);
    const float v0 = v00 + tx * (v10 - v00);
    const float v1 = v01 + tx * (v11 - v01);

    out_u = ClampToU8(u0 + ty * (u1 - u0));
    out_v = ClampToU8(v0 + ty * (v1 - v0));
}

__global__ void UyvySubpixelShiftKernel(
    const uint8_t* uyvy_in,
    uint8_t* uyvy_out,
    int width,
    int height,
    float shift_x,
    float shift_y
) {
    const int pair_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int pairs = width >> 1;
    if (pair_x >= pairs || y >= height) {
        return;
    }

    const int x0 = pair_x << 1;
    const int x1 = x0 + 1;

    const float src_y = static_cast<float>(y) - shift_y;
    const float src_x0 = static_cast<float>(x0) - shift_x;
    const float src_x1 = static_cast<float>(x1) - shift_x;
    const float src_pair_x = static_cast<float>(pair_x) - (shift_x * 0.5f);

    const uint8_t y0 = SampleUyvyLumaBilinear(uyvy_in, width, height, src_x0, src_y);
    const uint8_t y1 = SampleUyvyLumaBilinear(uyvy_in, width, height, src_x1, src_y);

    uint8_t u = 128;
    uint8_t v = 128;
    SampleUyvyUvBilinear(uyvy_in, width, height, src_pair_x, src_y, u, v);

    const int base = (y * pairs + pair_x) * 4;
    uyvy_out[base + 0] = u;
    uyvy_out[base + 1] = y0;
    uyvy_out[base + 2] = v;
    uyvy_out[base + 3] = y1;
}

__global__ void BobDeinterlaceKernel(const uchar3* rgb_in, uchar3* rgb_out, int width, int height, int field_phase) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    if (((y + field_phase) & 1) == 0) {
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

__global__ void EdgeAdaptiveDeinterlaceKernel(const uchar3* rgb_in, uchar3* rgb_out, int width, int height, int field_phase) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    if (((y + field_phase) & 1) == 0) {
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

__device__ inline uchar3 SampleBilinearSharp(const uchar3* src, int width, int height, float x, float y) {
    const uchar3 c = SampleBilinear(src, width, height, x, y);

    const int ix = max(0, min(width - 1, static_cast<int>(x + 0.5f)));
    const int iy = max(0, min(height - 1, static_cast<int>(y + 0.5f)));

    const uchar3 n = src[max(0, iy - 1) * width + ix];
    const uchar3 s = src[min(height - 1, iy + 1) * width + ix];
    const uchar3 w = src[iy * width + max(0, ix - 1)];
    const uchar3 e = src[iy * width + min(width - 1, ix + 1)];

    const float amount = 0.60f;
    const float c_r = static_cast<float>(c.x);
    const float c_g = static_cast<float>(c.y);
    const float c_b = static_cast<float>(c.z);

    const float blur_r = 0.25f * (static_cast<float>(n.x) + static_cast<float>(s.x) + static_cast<float>(w.x) + static_cast<float>(e.x));
    const float blur_g = 0.25f * (static_cast<float>(n.y) + static_cast<float>(s.y) + static_cast<float>(w.y) + static_cast<float>(e.y));
    const float blur_b = 0.25f * (static_cast<float>(n.z) + static_cast<float>(s.z) + static_cast<float>(w.z) + static_cast<float>(e.z));

    return make_uchar3(
        ClampToU8(c_r + amount * (c_r - blur_r)),
        ClampToU8(c_g + amount * (c_g - blur_g)),
        ClampToU8(c_b + amount * (c_b - blur_b))
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

__global__ void CropCopyRgbKernel(
    const uchar3* rgb_in,
    int src_width,
    int src_height,
    uchar3* rgb_out,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= roi_w || y >= roi_h) {
        return;
    }

    const int sx = max(0, min(src_width - 1, roi_x + x));
    const int sy = max(0, min(src_height - 1, roi_y + y));
    rgb_out[y * roi_w + x] = rgb_in[sy * src_width + sx];
}

__global__ void CropZoomBilinearSharpKernel(
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

    rgb_out[y * out_width + x] = SampleBilinearSharp(rgb_in, src_width, src_height, src_x, src_y);
}

__global__ void Sharpen3x3Kernel(
    const uchar3* rgb_in,
    uchar3* rgb_out,
    int width,
    int height,
    int preserve_field_parity
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int xm1 = max(0, x - 1);
    const int xp1 = min(width - 1, x + 1);

    int ym1 = max(0, y - 1);
    int yp1 = min(height - 1, y + 1);
    if (preserve_field_parity != 0) {
        // Preserve interlaced field parity by sampling vertical neighbors from
        // the same field line set (y-2/y+2) rather than adjacent opposite field lines.
        ym1 = y - 2;
        yp1 = y + 2;
        if (ym1 < 0) {
            ym1 = y;
        }
        if (yp1 >= height) {
            yp1 = y;
        }
    }

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
        // Keep sharpening moderate to avoid accentuating interlaced comb artifacts.
        const float amount = 1.35f;
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

__global__ void UpscaleBilinearSharpKernel(
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

    rgb_out[y * out_width + x] = SampleBilinearSharp(rgb_in, in_width, in_height, src_x, src_y);
}

__device__ inline float BilateralRangeWeight(float diff, float inv_sigma_sq) {
    const float v = diff * diff * inv_sigma_sq;
    return __fdividef(1.0f, 1.0f + v);
}

__device__ inline int UyvyYOffset(int width, int x, int y) {
    const int pairs_per_row = width >> 1;
    const int pair_x = x >> 1;
    const int base = (y * pairs_per_row + pair_x) * 4;
    return base + ((x & 1) ? 3 : 1);
}

__device__ inline float UyvyLumaAt(const uint8_t* uyvy, int width, int height, int x, int y) {
    const int clamped_x = max(0, min(width - 1, x));
    const int clamped_y = max(0, min(height - 1, y));
    return static_cast<float>(uyvy[UyvyYOffset(width, clamped_x, clamped_y)]);
}

__global__ void DenoiseUyvyLumaGaussian3x3Kernel(
    const uint8_t* uyvy_in,
    uint8_t* uyvy_out,
    int width,
    int height,
    float strength
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const float y_center = UyvyLumaAt(uyvy_in, width, height, x, y);

    const float y_nw = UyvyLumaAt(uyvy_in, width, height, x - 1, y - 1);
    const float y_n = UyvyLumaAt(uyvy_in, width, height, x, y - 1);
    const float y_ne = UyvyLumaAt(uyvy_in, width, height, x + 1, y - 1);
    const float y_w = UyvyLumaAt(uyvy_in, width, height, x - 1, y);
    const float y_e = UyvyLumaAt(uyvy_in, width, height, x + 1, y);
    const float y_sw = UyvyLumaAt(uyvy_in, width, height, x - 1, y + 1);
    const float y_s = UyvyLumaAt(uyvy_in, width, height, x, y + 1);
    const float y_se = UyvyLumaAt(uyvy_in, width, height, x + 1, y + 1);

    const float y_blur = (
        y_nw + 2.0f * y_n + y_ne +
        2.0f * y_w + 4.0f * y_center + 2.0f * y_e +
        y_sw + 2.0f * y_s + y_se
    ) / 16.0f;

    const float y_new = y_center + strength * (y_blur - y_center);
    uyvy_out[UyvyYOffset(width, x, y)] = ClampToU8(y_new);
}

__global__ void DenoiseUyvyLumaMedian3x3Kernel(
    const uint8_t* uyvy_in,
    uint8_t* uyvy_out,
    int width,
    int height,
    float strength
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const float y_center = UyvyLumaAt(uyvy_in, width, height, x, y);
    float samples[9] = {
        UyvyLumaAt(uyvy_in, width, height, x - 1, y - 1),
        UyvyLumaAt(uyvy_in, width, height, x, y - 1),
        UyvyLumaAt(uyvy_in, width, height, x + 1, y - 1),
        UyvyLumaAt(uyvy_in, width, height, x - 1, y),
        y_center,
        UyvyLumaAt(uyvy_in, width, height, x + 1, y),
        UyvyLumaAt(uyvy_in, width, height, x - 1, y + 1),
        UyvyLumaAt(uyvy_in, width, height, x, y + 1),
        UyvyLumaAt(uyvy_in, width, height, x + 1, y + 1),
    };

    const float y_med = Median9(samples);
    const float y_new = y_center + strength * (y_med - y_center);
    uyvy_out[UyvyYOffset(width, x, y)] = ClampToU8(y_new);
}

__global__ void DenoiseUyvyLumaBilateral3x3Kernel(
    const uint8_t* uyvy_in,
    uint8_t* uyvy_out,
    int width,
    int height,
    float strength
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const float y_center = UyvyLumaAt(uyvy_in, width, height, x, y);
    const float sigma = 8.0f + 52.0f * strength;
    const float inv_sigma_sq = __fdividef(1.0f, sigma * sigma + 1e-3f);

    float weighted_sum = 0.0f;
    float sum_w = 0.0f;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            const float l = UyvyLumaAt(uyvy_in, width, height, x + i, y + j);
            const float spatial_w = static_cast<float>((i == 0 && j == 0) ? 4 : ((i == 0 || j == 0) ? 2 : 1));
            const float w = spatial_w * BilateralRangeWeight(l - y_center, inv_sigma_sq);
            weighted_sum += w * l;
            sum_w += w;
        }
    }

    const float y_filtered = (sum_w > 1e-6f) ? (weighted_sum / sum_w) : y_center;
    const float y_new = y_center + strength * (y_filtered - y_center);
    uyvy_out[UyvyYOffset(width, x, y)] = ClampToU8(y_new);
}

__global__ void DenoiseUyvyLumaBilateral5x5Kernel(
    const uint8_t* uyvy_in,
    uint8_t* uyvy_out,
    int width,
    int height,
    float strength
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int g[5] = {1, 4, 6, 4, 1};

    const float y_center = UyvyLumaAt(uyvy_in, width, height, x, y);
    const float sigma = 10.0f + 68.0f * strength;
    const float inv_sigma_sq = __fdividef(1.0f, sigma * sigma + 1e-3f);

    float weighted_sum = 0.0f;
    float sum_w = 0.0f;
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            const float l = UyvyLumaAt(uyvy_in, width, height, x + i, y + j);
            const float spatial_w = static_cast<float>(g[j + 2] * g[i + 2]);
            const float w = spatial_w * BilateralRangeWeight(l - y_center, inv_sigma_sq);
            weighted_sum += w * l;
            sum_w += w;
        }
    }

    const float y_filtered = (sum_w > 1e-6f) ? (weighted_sum / sum_w) : y_center;
    const float y_new = y_center + strength * (y_filtered - y_center);
    uyvy_out[UyvyYOffset(width, x, y)] = ClampToU8(y_new);
}

__global__ void DenoiseLumaBilateral3x3Kernel(
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

    const uchar3 c = rgb_in[y * width + x];
    const float y_center = RgbLuma(c);
    const float sigma = 8.0f + 52.0f * strength;
    const float inv_sigma_sq = __fdividef(1.0f, sigma * sigma + 1e-3f);

    float weighted_sum = 0.0f;
    float sum_w = 0.0f;
    for (int j = -1; j <= 1; ++j) {
        const int sy = max(0, min(height - 1, y + j));
        for (int i = -1; i <= 1; ++i) {
            const int sx = max(0, min(width - 1, x + i));
            const float l = RgbLuma(rgb_in[sy * width + sx]);
            const float spatial_w = static_cast<float>((i == 0 && j == 0) ? 4 : ((i == 0 || j == 0) ? 2 : 1));
            const float w = spatial_w * BilateralRangeWeight(l - y_center, inv_sigma_sq);
            weighted_sum += w * l;
            sum_w += w;
        }
    }

    const float y_filtered = (sum_w > 1e-6f) ? (weighted_sum / sum_w) : y_center;
    const float y_new = y_center + strength * (y_filtered - y_center);
    rgb_out[y * width + x] = ApplyLumaDelta(c, y_new - y_center);
}

__global__ void DenoiseLumaBilateral5x5Kernel(
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

    const int g[5] = {1, 4, 6, 4, 1};

    const uchar3 c = rgb_in[y * width + x];
    const float y_center = RgbLuma(c);
    const float sigma = 10.0f + 68.0f * strength;
    const float inv_sigma_sq = __fdividef(1.0f, sigma * sigma + 1e-3f);

    float weighted_sum = 0.0f;
    float sum_w = 0.0f;
    for (int j = -2; j <= 2; ++j) {
        const int sy = max(0, min(height - 1, y + j));
        for (int i = -2; i <= 2; ++i) {
            const int sx = max(0, min(width - 1, x + i));
            const float l = RgbLuma(rgb_in[sy * width + sx]);
            const float spatial_w = static_cast<float>(g[j + 2] * g[i + 2]);
            const float w = spatial_w * BilateralRangeWeight(l - y_center, inv_sigma_sq);
            weighted_sum += w * l;
            sum_w += w;
        }
    }

    const float y_filtered = (sum_w > 1e-6f) ? (weighted_sum / sum_w) : y_center;
    const float y_new = y_center + strength * (y_filtered - y_center);
    rgb_out[y * width + x] = ApplyLumaDelta(c, y_new - y_center);
}

__global__ void RgbToUyvyKernel(const uchar3* rgb, uint8_t* uyvy, int width, int height, int color_matrix, int color_range) {
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

    const YuvF yuv0 = RgbToYuv(p0, color_matrix, color_range);
    const YuvF yuv1 = RgbToYuv(p1, color_matrix, color_range);

    const uint8_t y0_u8 = ClampToU8(yuv0.y);
    const uint8_t y1_u8 = ClampToU8(yuv1.y);
    const uint8_t u_u8 = ClampToU8((yuv0.u + yuv1.u) * 0.5f);
    const uint8_t v_u8 = ClampToU8((yuv0.v + yuv1.v) * 0.5f);

    const int base = (y * pairs_per_row + pair_x) * 4;
    uyvy[base + 0] = u_u8;
    uyvy[base + 1] = y0_u8;
    uyvy[base + 2] = v_u8;
    uyvy[base + 3] = y1_u8;
}

__device__ inline float TensorElemToFloat(const float v) {
    return v;
}

__device__ inline float TensorElemToFloat(const __half v) {
    return __half2float(v);
}

__device__ inline float TensorElemToFloat(const uint8_t v) {
    return static_cast<float>(v);
}

template <typename T>
__device__ inline T FloatToTensorElem(float v);

template <>
__device__ inline float FloatToTensorElem<float>(float v) {
    return v;
}

template <>
__device__ inline __half FloatToTensorElem<__half>(float v) {
    return __float2half(v);
}

template <>
__device__ inline uint8_t FloatToTensorElem<uint8_t>(float v) {
    return ClampToU8(v);
}

template <typename T>
__global__ void RgbToTensorKernel(
    const uchar3* rgb,
    int tensor_layout,
    int channels,
    float scale,
    T* tensor,
    int width,
    int height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int pixel_index = y * width + x;
    const int clamped_channels = max(1, min(4, channels));
    const uchar3 px = rgb[pixel_index];

    const float values[4] = {
        static_cast<float>(px.x) * scale,
        static_cast<float>(px.y) * scale,
        static_cast<float>(px.z) * scale,
        0.0f,
    };

    if (tensor_layout == 0) {
        const int plane = width * height;
        for (int c = 0; c < clamped_channels; ++c) {
            tensor[(c * plane) + pixel_index] = FloatToTensorElem<T>(values[c]);
        }
    } else {
        const int base = pixel_index * clamped_channels;
        for (int c = 0; c < clamped_channels; ++c) {
            tensor[base + c] = FloatToTensorElem<T>(values[c]);
        }
    }
}

template <typename T>
__global__ void TensorToRgbKernel(
    const T* tensor,
    int tensor_layout,
    int channels,
    float scale,
    uchar3* rgb,
    int width,
    int height
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int pixel_index = y * width + x;
    const int clamped_channels = max(1, min(4, channels));

    const int c0 = 0;
    const int c1 = (clamped_channels > 1) ? 1 : 0;
    const int c2 = (clamped_channels > 2) ? 2 : 0;

    int r_index = 0;
    int g_index = 0;
    int b_index = 0;
    if (tensor_layout == 0) {
        // NCHW
        const int plane = width * height;
        r_index = (c0 * plane) + pixel_index;
        g_index = (c1 * plane) + pixel_index;
        b_index = (c2 * plane) + pixel_index;
    } else {
        // HWC
        const int base = pixel_index * clamped_channels;
        r_index = base + c0;
        g_index = base + c1;
        b_index = base + c2;
    }

    const float r = TensorElemToFloat(tensor[r_index]) * scale;
    const float g = TensorElemToFloat(tensor[g_index]) * scale;
    const float b = TensorElemToFloat(tensor[b_index]) * scale;

    rgb[pixel_index] = make_uchar3(ClampToU8(r), ClampToU8(g), ClampToU8(b));
}

inline dim3 Grid2D(int width, int height, int bx = 16, int by = 16) {
    return dim3((width + bx - 1) / bx, (height + by - 1) / by, 1);
}

} // namespace

void LaunchUyvyToRgb(
    const uint8_t* d_uyvy,
    uchar3* d_rgb,
    int width,
    int height,
    int color_matrix,
    int color_range,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    UyvyToRgbKernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_uyvy,
        d_rgb,
        width,
        height,
        color_matrix,
        color_range
    );
    CheckKernelLaunch("UyvyToRgbKernel launch");
}

void LaunchUyvyFieldToRgb(
    const uint8_t* d_uyvy,
    uchar3* d_rgb,
    int width,
    int height,
    int source_field_phase,
    int color_matrix,
    int color_range,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    UyvyFieldToRgbKernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_uyvy,
        d_rgb,
        width,
        height,
        source_field_phase,
        color_matrix,
        color_range
    );
    CheckKernelLaunch("UyvyFieldToRgbKernel launch");
}

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
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    const dim3 grid(((out_width >> 1) + kBlockX - 1) / kBlockX, (out_height + kBlockY - 1) / kBlockY, 1);
    const dim3 block(kBlockX, kBlockY, 1);

    UyvyCropZoomNearestKernel<<<grid, block, 0, stream>>>(
        d_uyvy_in,
        src_width,
        src_height,
        d_uyvy_out,
        out_width,
        out_height,
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        preserve_field_parity ? 1 : 0
    );
    CheckKernelLaunch("UyvyCropZoomNearestKernel launch");
}

void LaunchUyvySubpixelShift(
    const uint8_t* d_uyvy_in,
    uint8_t* d_uyvy_out,
    int width,
    int height,
    float shift_x,
    float shift_y,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    const dim3 grid((((width >> 1) + kBlockX - 1) / kBlockX), ((height + kBlockY - 1) / kBlockY), 1);
    const dim3 block(kBlockX, kBlockY, 1);
    UyvySubpixelShiftKernel<<<grid, block, 0, stream>>>(
        d_uyvy_in,
        d_uyvy_out,
        width,
        height,
        shift_x,
        shift_y
    );
    CheckKernelLaunch("UyvySubpixelShiftKernel launch");
}

void LaunchBobDeinterlace(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    int field_phase,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    BobDeinterlaceKernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_rgb_in,
        d_rgb_out,
        width,
        height,
        field_phase & 1
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

void LaunchEdgeAdaptiveDeinterlace(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    int field_phase,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    EdgeAdaptiveDeinterlaceKernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_rgb_in,
        d_rgb_out,
        width,
        height,
        field_phase & 1
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

void LaunchUpscaleBilinearSharp(
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
    UpscaleBilinearSharpKernel<<<Grid2D(out_width, out_height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
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
    CheckKernelLaunch("CropZoomBilinearKernel launch");
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
    CheckKernelLaunch("CropZoomBicubicKernel launch");
}

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
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    CropZoomBilinearSharpKernel<<<Grid2D(out_width, out_height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
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
    CheckKernelLaunch("CropZoomBilinearSharpKernel launch");
}

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
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    CropCopyRgbKernel<<<Grid2D(roi_w, roi_h, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_rgb_in,
        src_width,
        src_height,
        d_rgb_out,
        roi_x,
        roi_y,
        roi_w,
        roi_h
    );
    CheckKernelLaunch("CropCopyRgbKernel launch");
}

void LaunchSharpen3x3(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    bool preserve_field_parity,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    Sharpen3x3Kernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_rgb_in,
        d_rgb_out,
        width,
        height,
        preserve_field_parity ? 1 : 0
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

void LaunchDenoiseLumaBilateral3x3(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    DenoiseLumaBilateral3x3Kernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_rgb_in,
        d_rgb_out,
        width,
        height,
        fminf(1.0f, fmaxf(0.0f, strength))
    );
}

void LaunchDenoiseLumaBilateral5x5(
    const uchar3* d_rgb_in,
    uchar3* d_rgb_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    DenoiseLumaBilateral5x5Kernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_rgb_in,
        d_rgb_out,
        width,
        height,
        fminf(1.0f, fmaxf(0.0f, strength))
    );
}

void LaunchDenoiseUyvyLumaGaussian3x3(
    const uint8_t* d_uyvy_in,
    uint8_t* d_uyvy_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    DenoiseUyvyLumaGaussian3x3Kernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_uyvy_in,
        d_uyvy_out,
        width,
        height,
        fminf(1.0f, fmaxf(0.0f, strength))
    );
}

void LaunchDenoiseUyvyLumaMedian3x3(
    const uint8_t* d_uyvy_in,
    uint8_t* d_uyvy_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    DenoiseUyvyLumaMedian3x3Kernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_uyvy_in,
        d_uyvy_out,
        width,
        height,
        fminf(1.0f, fmaxf(0.0f, strength))
    );
}

void LaunchDenoiseUyvyLumaBilateral3x3(
    const uint8_t* d_uyvy_in,
    uint8_t* d_uyvy_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    DenoiseUyvyLumaBilateral3x3Kernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_uyvy_in,
        d_uyvy_out,
        width,
        height,
        fminf(1.0f, fmaxf(0.0f, strength))
    );
}

void LaunchDenoiseUyvyLumaBilateral5x5(
    const uint8_t* d_uyvy_in,
    uint8_t* d_uyvy_out,
    int width,
    int height,
    float strength,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    DenoiseUyvyLumaBilateral5x5Kernel<<<Grid2D(width, height, kBlockX, kBlockY), dim3(kBlockX, kBlockY), 0, stream>>>(
        d_uyvy_in,
        d_uyvy_out,
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

void LaunchRgbToUyvy(
    const uchar3* d_rgb,
    uint8_t* d_uyvy,
    int width,
    int height,
    int color_matrix,
    int color_range,
    cudaStream_t stream
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    const dim3 grid((width / 2 + kBlockX - 1) / kBlockX, (height + kBlockY - 1) / kBlockY, 1);
    const dim3 block(kBlockX, kBlockY, 1);

    RgbToUyvyKernel<<<grid, block, 0, stream>>>(d_rgb, d_uyvy, width, height, color_matrix, color_range);
    CheckKernelLaunch("RgbToUyvyKernel launch");
}

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
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    const dim3 grid = Grid2D(width, height, kBlockX, kBlockY);
    const dim3 block(kBlockX, kBlockY, 1);
    const float scale = normalized_01 ? 255.0f : 1.0f;

    switch (tensor_dtype) {
        case 0:
            TensorToRgbKernel<float><<<grid, block, 0, stream>>>(
                static_cast<const float*>(d_tensor),
                tensor_layout,
                channels,
                scale,
                d_rgb,
                width,
                height
            );
            break;
        case 1:
            TensorToRgbKernel<__half><<<grid, block, 0, stream>>>(
                static_cast<const __half*>(d_tensor),
                tensor_layout,
                channels,
                scale,
                d_rgb,
                width,
                height
            );
            break;
        case 2:
            TensorToRgbKernel<uint8_t><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t*>(d_tensor),
                tensor_layout,
                channels,
                1.0f,
                d_rgb,
                width,
                height
            );
            break;
        default:
            throw std::runtime_error("Unsupported tensor dtype code for LaunchTensorToRgb");
    }

    CheckKernelLaunch("TensorToRgbKernel launch");
}

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
) {
    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    const dim3 grid = Grid2D(width, height, kBlockX, kBlockY);
    const dim3 block(kBlockX, kBlockY, 1);
    const float scale = normalized_01 ? (1.0f / 255.0f) : 1.0f;

    switch (tensor_dtype) {
        case 0:
            RgbToTensorKernel<float><<<grid, block, 0, stream>>>(
                d_rgb,
                tensor_layout,
                channels,
                scale,
                static_cast<float*>(d_tensor),
                width,
                height
            );
            break;
        case 1:
            RgbToTensorKernel<__half><<<grid, block, 0, stream>>>(
                d_rgb,
                tensor_layout,
                channels,
                scale,
                static_cast<__half*>(d_tensor),
                width,
                height
            );
            break;
        case 2:
            RgbToTensorKernel<uint8_t><<<grid, block, 0, stream>>>(
                d_rgb,
                tensor_layout,
                channels,
                normalized_01 ? 255.0f : 1.0f,
                static_cast<uint8_t*>(d_tensor),
                width,
                height
            );
            break;
        default:
            throw std::runtime_error("Unsupported tensor dtype code for LaunchRgbToTensor");
    }

    CheckKernelLaunch("RgbToTensorKernel launch");
}

} // namespace vp::cuda_kernels
