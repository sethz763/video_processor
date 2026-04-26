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
