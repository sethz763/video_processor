#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace vp {

enum class SrFlavor {
    Bilinear,
    Bicubic,
    BicubicSharpen,
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

    int width() const { return width_; }
    int height() const { return height_; }
    int sr_scale() const;

private:
    std::string ProcessFrameInternal(
        const std::string& input_frame,
        bool deinterlace_only,
        bool force_deinterlace,
        bool force_disable_deinterlace
    );
    void InitializeBuffers();
    void ValidateConfiguration() const;
    void ClampRoi();
    void ConfigureSrScaleLocked(int requested_scale, bool auto_mode);
    bool EnsureSrBufferCapacityLocked(int target_scale, cudaError_t& last_error);
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
    SrFlavor sr_flavor_;
    bool auto_sr_scale_;
    int max_auto_sr_scale_;
    int sr_requested_scale_;
    int sr_scale_;
    int sr_width_;
    int sr_height_;
    int sr_buffer_scale_capacity_;

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
