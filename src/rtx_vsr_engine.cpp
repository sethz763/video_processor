#include "rtx_vsr_engine.h"

#include <Windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rtx_video_api.h"

namespace {

template <typename T>
void SafeRelease(T*& ptr) {
    if (ptr != nullptr) {
        ptr->Release();
        ptr = nullptr;
    }
}

std::string HrToString(HRESULT hr) {
    std::ostringstream oss;
    oss << "HRESULT=0x" << std::hex << static_cast<unsigned long>(hr);
    return oss.str();
}

void ThrowIfFailed(HRESULT hr, const char* action) {
    if (FAILED(hr)) {
        throw std::runtime_error(std::string(action) + " failed: " + HrToString(hr));
    }
}

std::wstring ToWide(const std::string& s) {
    if (s.empty()) {
        return std::wstring();
    }
    int needed = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (needed <= 0) {
        throw std::runtime_error("Failed converting UTF-8 to UTF-16");
    }
    std::wstring out(static_cast<size_t>(needed - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, out.data(), needed);
    return out;
}

std::string ToUtf8(const std::wstring& s) {
    if (s.empty()) {
        return std::string();
    }
    int needed = WideCharToMultiByte(CP_UTF8, 0, s.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (needed <= 0) {
        return std::string();
    }
    std::string out(static_cast<size_t>(needed - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, s.c_str(), -1, out.data(), needed, nullptr, nullptr);
    return out;
}

}  // namespace

namespace rtx_vsr {

RTXVideoSREngine::RTXVideoSREngine(
    int input_width,
    int input_height,
    int output_width,
    int output_height,
        const std::string& quality,
        bool thdr_enabled,
        std::uint32_t thdr_contrast,
        std::uint32_t thdr_saturation,
        std::uint32_t thdr_middle_gray,
        std::uint32_t thdr_max_luminance)
    : input_width_(input_width),
      input_height_(input_height),
      output_width_(output_width),
      output_height_(output_height),
            settings_{ResolveQualityLevel(quality), thdr_enabled, thdr_contrast, thdr_saturation, thdr_middle_gray, thdr_max_luminance},
      closed_(false),
      sdk_initialized_(false),
      vsr_dll_handle_(nullptr),
      device_(nullptr),
      context_(nullptr),
      input_staging_(nullptr),
      input_texture_(nullptr),
      output_texture_(nullptr),
      output_staging_(nullptr) {
    if (input_width_ <= 0 || input_height_ <= 0 || output_width_ <= 0 || output_height_ <= 0) {
        throw std::invalid_argument("Input and output dimensions must be > 0");
    }

    EnsureRuntimeDllLoaded();
    CreateD3D11Device();
    CreateTextures();
    InitSDK();
}

RTXVideoSREngine::~RTXVideoSREngine() {
    Close();
}

void RTXVideoSREngine::EnsureRuntimeDllLoaded() {
    std::vector<std::wstring> candidates;
    std::vector<std::string> attempted;

    candidates.emplace_back(L"nvngx_vsr.dll");

#ifdef RTX_VIDEO_SDK_ROOT_PATH
    std::wstring sdk_root = ToWide(RTX_VIDEO_SDK_ROOT_PATH);
    if (!sdk_root.empty()) {
        candidates.push_back(sdk_root + L"\\bin\\Windows\\x64\\rel\\nvngx_vsr.dll");
        candidates.push_back(sdk_root + L"\\bin\\Windows\\x64\\dev\\nvngx_vsr.dll");
    }
#endif

    for (const auto& candidate : candidates) {
        attempted.push_back(ToUtf8(candidate));
        HMODULE h = LoadLibraryW(candidate.c_str());
        if (h != nullptr) {
            vsr_dll_handle_ = h;
            std::wstring module_path(1024, L'\0');
            DWORD len = GetModuleFileNameW(h, module_path.data(), static_cast<DWORD>(module_path.size()));
            if (len > 0 && len < module_path.size()) {
                module_path.resize(len);
                loaded_vsr_dll_path_ = ToUtf8(module_path);
            } else {
                loaded_vsr_dll_path_ = ToUtf8(candidate);
            }
            return;
        }
    }

    std::ostringstream detail;
    detail << "Failed to load nvngx_vsr.dll. Tried:";
    for (const auto& item : attempted) {
        detail << "\n - " << item;
    }
    detail << "\nEnsure NVIDIA RTX Video SDK runtime DLL is available (same directory as rtx_vsr.pyd, in SDK bin/Windows/x64/rel, or via RTX_VIDEO_SDK_ROOT).";

    throw std::runtime_error(detail.str());
}

void RTXVideoSREngine::CreateD3D11Device() {
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if defined(_DEBUG)
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL feature_levels[] = {
        D3D_FEATURE_LEVEL_12_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
    };
    D3D_FEATURE_LEVEL created_level = D3D_FEATURE_LEVEL_11_0;

    Microsoft::WRL::ComPtr<IDXGIAdapter1> chosen_adapter;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> fallback_adapter;
    std::string chosen_adapter_name;

    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(factory.GetAddressOf())))) {
        for (UINT index = 0;; ++index) {
            Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
            if (factory->EnumAdapters1(index, adapter.GetAddressOf()) == DXGI_ERROR_NOT_FOUND) {
                break;
            }

            DXGI_ADAPTER_DESC1 desc = {};
            if (FAILED(adapter->GetDesc1(&desc))) {
                continue;
            }
            if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
                continue;
            }

            if (!fallback_adapter) {
                fallback_adapter = adapter;
            }
            if (desc.VendorId == 0x10DE) {
                chosen_adapter = adapter;
                chosen_adapter_name = ToUtf8(desc.Description);
                break;
            }
        }
    }

    if (!chosen_adapter && fallback_adapter) {
        chosen_adapter = fallback_adapter;
        DXGI_ADAPTER_DESC1 desc = {};
        if (SUCCEEDED(chosen_adapter->GetDesc1(&desc))) {
            chosen_adapter_name = ToUtf8(desc.Description);
        }
    }

    HRESULT hr = D3D11CreateDevice(
        chosen_adapter.Get(),
        chosen_adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        flags,
        feature_levels,
        static_cast<UINT>(std::size(feature_levels)),
        D3D11_SDK_VERSION,
        &device_,
        &created_level,
        &context_);

    if (FAILED(hr) && chosen_adapter) {
        // Fallback to system default if adapter-specific creation fails.
        hr = D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            flags,
            feature_levels,
            static_cast<UINT>(std::size(feature_levels)),
            D3D11_SDK_VERSION,
            &device_,
            &created_level,
            &context_);
    }

    ThrowIfFailed(hr, "D3D11CreateDevice");

    if (created_level < D3D_FEATURE_LEVEL_11_0) {
        throw std::runtime_error("Unsupported GPU: D3D11 feature level 11.0+ is required");
    }
}

void RTXVideoSREngine::CreateTextures() {
    D3D11_TEXTURE2D_DESC in_default_desc = {};
    in_default_desc.Width = static_cast<UINT>(input_width_);
    in_default_desc.Height = static_cast<UINT>(input_height_);
    in_default_desc.MipLevels = 1;
    in_default_desc.ArraySize = 1;
    in_default_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    in_default_desc.SampleDesc.Count = 1;
    in_default_desc.Usage = D3D11_USAGE_DEFAULT;
    in_default_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    ThrowIfFailed(device_->CreateTexture2D(&in_default_desc, nullptr, &input_texture_), "CreateTexture2D(input_texture)");

    D3D11_TEXTURE2D_DESC in_staging_desc = in_default_desc;
    in_staging_desc.Usage = D3D11_USAGE_STAGING;
    in_staging_desc.BindFlags = 0;
    in_staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    ThrowIfFailed(device_->CreateTexture2D(&in_staging_desc, nullptr, &input_staging_), "CreateTexture2D(input_staging)");

    D3D11_TEXTURE2D_DESC out_default_desc = {};
    out_default_desc.Width = static_cast<UINT>(output_width_);
    out_default_desc.Height = static_cast<UINT>(output_height_);
    out_default_desc.MipLevels = 1;
    out_default_desc.ArraySize = 1;
    out_default_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    out_default_desc.SampleDesc.Count = 1;
    out_default_desc.Usage = D3D11_USAGE_DEFAULT;
    out_default_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;

    ThrowIfFailed(device_->CreateTexture2D(&out_default_desc, nullptr, &output_texture_), "CreateTexture2D(output_texture)");

    D3D11_TEXTURE2D_DESC out_staging_desc = out_default_desc;
    out_staging_desc.Usage = D3D11_USAGE_STAGING;
    out_staging_desc.BindFlags = 0;
    out_staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

    ThrowIfFailed(device_->CreateTexture2D(&out_staging_desc, nullptr, &output_staging_), "CreateTexture2D(output_staging)");
}

void RTXVideoSREngine::InitSDK() {
    // TODO: If you move away from sample API wrappers, replace with direct NGX init/create calls.
    API_BOOL created = rtx_video_api_dx11_create(
        device_,
        settings_.thdr_enabled ? API_BOOL_SUCCESS : API_BOOL_FAIL,
        API_BOOL_SUCCESS
    );
    if (created != API_BOOL_SUCCESS) {
        std::ostringstream detail;
        detail << "RTX Video SDK initialization failed after loading nvngx_vsr.dll";
        if (!loaded_vsr_dll_path_.empty()) {
            detail << " from: " << loaded_vsr_dll_path_;
        }
        detail << ". This indicates SDK/GPU capability mismatch or runtime initialization failure (not basic file path visibility).";
        throw std::runtime_error(detail.str());
    }
    sdk_initialized_ = true;
}

void RTXVideoSREngine::ShutdownSDK() {
    if (sdk_initialized_) {
        // TODO: Replace with direct NGX teardown if using direct SDK integration.
        rtx_video_api_dx11_shutdown();
        sdk_initialized_ = false;
    }
}

void RTXVideoSREngine::ThrowIfClosed() const {
    if (closed_) {
        throw std::runtime_error("RTXVideoSR engine is already closed");
    }
}

std::uint32_t RTXVideoSREngine::ResolveQualityLevel(const std::string& quality) const {
    std::string q = quality;
    std::transform(q.begin(), q.end(), q.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (q == "1") {
        return 1;
    }
    if (q == "2") {
        return 2;
    }
    if (q == "3") {
        return 3;
    }
    if (q == "4") {
        return 4;
    }

    if (q == "low") {
        return 1;
    }
    if (q == "medium") {
        return 2;
    }
    if (q == "high") {
        return 3;
    }
    if (q == "ultra") {
        return 4;
    }

    throw std::invalid_argument("Invalid quality. Expected one of: 1,2,3,4,low,medium,high,ultra");
}

std::vector<std::uint8_t> RTXVideoSREngine::ProcessRGBA(const std::uint8_t* input_rgba, std::size_t input_bytes) {
    ThrowIfClosed();

    const std::size_t expected_input_bytes = static_cast<std::size_t>(input_width_) * static_cast<std::size_t>(input_height_) * 4;
    if (input_rgba == nullptr || input_bytes != expected_input_bytes) {
        throw std::invalid_argument("Invalid input buffer size for RGBA frame");
    }

    D3D11_MAPPED_SUBRESOURCE mapped_in = {};
    ThrowIfFailed(context_->Map(input_staging_, 0, D3D11_MAP_WRITE, 0, &mapped_in), "Map(input_staging)");

    const std::size_t src_row_bytes = static_cast<std::size_t>(input_width_) * 4;
    auto* src = input_rgba;
    auto* dst = static_cast<std::uint8_t*>(mapped_in.pData);
    for (int y = 0; y < input_height_; ++y) {
        std::memcpy(dst + static_cast<std::size_t>(y) * mapped_in.RowPitch, src + static_cast<std::size_t>(y) * src_row_bytes, src_row_bytes);
    }

    context_->Unmap(input_staging_, 0);
    context_->CopyResource(input_texture_, input_staging_);

    API_RECT input_rect = {0u, 0u, static_cast<std::uint32_t>(input_width_), static_cast<std::uint32_t>(input_height_)};
    API_RECT output_rect = {0u, 0u, static_cast<std::uint32_t>(output_width_), static_cast<std::uint32_t>(output_height_)};
    API_VSR_Setting vsr = {};
    vsr.QualityLevel = settings_.quality_level;
    API_THDR_Setting thdr = {};
    thdr.Contrast = settings_.thdr_contrast;
    thdr.Saturation = settings_.thdr_saturation;
    thdr.MiddleGray = settings_.thdr_middle_gray;
    thdr.MaxLuminance = settings_.thdr_max_luminance;

    // TODO: If SDK headers/API names differ in your installed version, update this evaluate call to match sample code.
    API_BOOL ok = rtx_video_api_dx11_evaluate(
        input_texture_,
        output_texture_,
        input_rect,
        output_rect,
        &vsr,
        settings_.thdr_enabled ? &thdr : nullptr
    );
    if (ok != API_BOOL_SUCCESS) {
        throw std::runtime_error("RTX Video SDK evaluate failed. Check GPU support, driver version, and input/output texture formats.");
    }

    context_->CopyResource(output_staging_, output_texture_);

    D3D11_MAPPED_SUBRESOURCE mapped_out = {};
    ThrowIfFailed(context_->Map(output_staging_, 0, D3D11_MAP_READ, 0, &mapped_out), "Map(output_staging)");

    const std::size_t out_row_bytes = static_cast<std::size_t>(output_width_) * 4;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(output_height_) * out_row_bytes);
    auto* src_out = static_cast<const std::uint8_t*>(mapped_out.pData);
    for (int y = 0; y < output_height_; ++y) {
        std::memcpy(out.data() + static_cast<std::size_t>(y) * out_row_bytes,
                    src_out + static_cast<std::size_t>(y) * mapped_out.RowPitch,
                    out_row_bytes);
    }

    context_->Unmap(output_staging_, 0);
    return out;
}

void RTXVideoSREngine::Close() {
    if (closed_) {
        return;
    }

    ShutdownSDK();

    SafeRelease(output_staging_);
    SafeRelease(output_texture_);
    SafeRelease(input_texture_);
    SafeRelease(input_staging_);
    SafeRelease(context_);
    SafeRelease(device_);

    if (vsr_dll_handle_ != nullptr) {
        FreeLibrary(static_cast<HMODULE>(vsr_dll_handle_));
        vsr_dll_handle_ = nullptr;
    }

    closed_ = true;
}

}  // namespace rtx_vsr
