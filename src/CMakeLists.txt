set(SRC_FILES
    bindings/module.cpp
    core/video_processor.cpp
    cuda/kernels.cu
)

pybind11_add_module(video_processor MODULE ${SRC_FILES})

target_include_directories(video_processor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(video_processor PRIVATE
    _CRT_SECURE_NO_WARNINGS
)

target_compile_options(video_processor PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

set_target_properties(video_processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(video_processor PRIVATE CUDA::cudart)

# RTX Video SDK root can be overridden at configure time.
set(RTX_VIDEO_SDK_ROOT "C:/Coding Projects/sdks/NVidia video SDK" CACHE PATH "Path to NVIDIA RTX Video SDK root")
set(RTX_VIDEO_SDK_INCLUDE_DIR "${RTX_VIDEO_SDK_ROOT}/include")
set(RTX_VIDEO_SDK_LIB_DIR "${RTX_VIDEO_SDK_ROOT}/lib/Windows/x64")
set(RTX_VIDEO_SDK_BIN_REL_DIR "${RTX_VIDEO_SDK_ROOT}/bin/Windows/x64/rel")
set(RTX_VIDEO_SDK_API_SAMPLE_DIR "${RTX_VIDEO_SDK_ROOT}/samples/RTX_Video_API")

if(WIN32)
    if(NOT EXISTS "${RTX_VIDEO_SDK_INCLUDE_DIR}/nvsdk_ngx.h")
        message(FATAL_ERROR "RTX Video SDK headers not found. Set RTX_VIDEO_SDK_ROOT to your SDK path.")
    endif()
    if(NOT EXISTS "${RTX_VIDEO_SDK_LIB_DIR}/nvsdk_ngx_d.lib")
        message(FATAL_ERROR "RTX Video SDK import lib not found: ${RTX_VIDEO_SDK_LIB_DIR}/nvsdk_ngx_d.lib")
    endif()
    if(NOT EXISTS "${RTX_VIDEO_SDK_API_SAMPLE_DIR}/rtx_video_api.h")
        message(FATAL_ERROR "RTX Video SDK sample API header not found: ${RTX_VIDEO_SDK_API_SAMPLE_DIR}/rtx_video_api.h")
    endif()
endif()

set(RTX_VSR_SRC_FILES
    bindings.cpp
    rtx_vsr_engine.cpp
    "${RTX_VIDEO_SDK_API_SAMPLE_DIR}/rtx_video_api_dx11_impl.cpp"
)

pybind11_add_module(rtx_vsr MODULE ${RTX_VSR_SRC_FILES})

target_include_directories(rtx_vsr PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RTX_VIDEO_SDK_INCLUDE_DIR}
    ${RTX_VIDEO_SDK_API_SAMPLE_DIR}
)

target_compile_definitions(rtx_vsr PRIVATE
    _CRT_SECURE_NO_WARNINGS
    RTX_VIDEO_SDK_ROOT_PATH="${RTX_VIDEO_SDK_ROOT}"
)

target_compile_options(rtx_vsr PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2>
)

target_link_directories(rtx_vsr PRIVATE
    ${RTX_VIDEO_SDK_LIB_DIR}
)

target_link_libraries(rtx_vsr PRIVATE
    d3d11
    dxgi
    CUDA::cudart
    nvsdk_ngx_d
)

if(WIN32)
    add_custom_command(TARGET rtx_vsr POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${RTX_VIDEO_SDK_BIN_REL_DIR}/nvngx_vsr.dll"
            "$<TARGET_FILE_DIR:rtx_vsr>/nvngx_vsr.dll"
        COMMENT "Copying nvngx_vsr.dll next to rtx_vsr module"
    )
endif()
