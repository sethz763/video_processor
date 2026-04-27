#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace vp {

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

    int width() const { return width_; }
    int height() const { return height_; }
    int sr_scale() const { return sr_scale_; }

private:
    void InitializeBuffers();
    void ValidateConfiguration() const;
    void ClampRoi();
    void Cleanup();

    int width_;
    int height_;

    int roi_x_;
    int roi_y_;
    int roi_w_;
    int roi_h_;

    bool enable_placeholder_sr_;
    bool auto_sr_scale_;
    int sr_scale_;
    int sr_width_;
    int sr_height_;

    size_t uyvy_bytes_;
    size_t rgb_pixels_;

    cudaStream_t stream_;

    uint8_t* d_uyvy_in_;
    uint8_t* d_uyvy_out_;
    uchar3* d_rgb_full_;
    uchar3* d_rgb_bob_;
    uchar3* d_rgb_sr_;
    uchar3* d_rgb_zoom_;

    std::vector<uint8_t> host_output_;
};

} // namespace vp
