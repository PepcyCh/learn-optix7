add_library(optix_common
    check_macros.h
    cuda_buffer.cpp
    cuda_buffer.h
    cuda_macro_utils.h
    frame_counter.h
    frame_counter.cpp
    geometry.h
    geometry.cpp
    renderer_base.cpp
    renderer_base.h
    math/vec.h
)

target_compile_features(optix_common
    PUBLIC cxx_std_20
)

target_include_directories(optix_common
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(optix_common
    PUBLIC OptiX::OptiX CUDA::cudart CUDA::cuda_driver fmt::fmt glfw glad imgui::imgui tinyobjloader::tinyobjloader
)