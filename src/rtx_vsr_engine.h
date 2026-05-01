#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11Texture2D;

namespace rtx_vsr {

struct RTXVideoSRSettings {
    std::uint32_t quality_level = 3;
    bool thdr_enabled = false;
    std::uint32_t thdr_contrast = 50;
    std::uint32_t thdr_saturation = 50;
    std::uint32_t thdr_middle_gray = 50;
    std::uint32_t thdr_max_luminance = 1000;
};

class RTXVideoSREngine {
public:
    RTXVideoSREngine(
        int input_width,
        int input_height,
        int output_width,
        int output_height,
        const std::string& quality,
        bool thdr_enabled,
        std::uint32_t thdr_contrast,
        std::uint32_t thdr_saturation,
        std::uint32_t thdr_middle_gray,
        std::uint32_t thdr_max_luminance
    );

    ~RTXVideoSREngine();

    RTXVideoSREngine(const RTXVideoSREngine&) = delete;
    RTXVideoSREngine& operator=(const RTXVideoSREngine&) = delete;

    std::vector<std::uint8_t> ProcessRGBA(const std::uint8_t* input_rgba, std::size_t input_bytes);
    void Close();

    int input_width() const { return input_width_; }
    int input_height() const { return input_height_; }
    int output_width() const { return output_width_; }
    int output_height() const { return output_height_; }
    RTXVideoSRSettings settings() const { return settings_; }

private:
    void EnsureRuntimeDllLoaded();
    void CreateD3D11Device();
    void CreateTextures();
    void InitSDK();
    void ShutdownSDK();
    void ThrowIfClosed() const;

    std::uint32_t ResolveQualityLevel(const std::string& quality) const;

    int input_width_;
    int input_height_;
    int output_width_;
    int output_height_;

    RTXVideoSRSettings settings_;
    bool closed_;
    bool sdk_initialized_;
    std::string loaded_vsr_dll_path_;

    void* vsr_dll_handle_;

    ID3D11Device* device_;
    ID3D11DeviceContext* context_;

    ID3D11Texture2D* input_staging_;
    ID3D11Texture2D* input_texture_;
    ID3D11Texture2D* output_texture_;
    ID3D11Texture2D* output_staging_;
};

}  // namespace rtx_vsr
