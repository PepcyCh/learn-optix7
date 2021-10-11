add_library(optix_common
    check_macros.h
    cuda_buffer.cpp
    cuda_buffer.h
    cuda_macro_utils.h
    cuda_texture.cpp
    cuda_texture.h
    frame_counter.cpp
    frame_counter.h
    geometry.cpp
    geometry.h
    gl_utils.cpp
    gl_utils.h
    lights.cpp
    lights.h
    light_data.h
    pixel_buffer.cpp
    pixel_buffer.h
    random.h
    renderer_base.cpp
    renderer_base.h
    scene.cpp
    scene.h
    shader_common.h
    texture.cpp
    texture.h
)

target_compile_features(optix_common
    PUBLIC cxx_std_20
)

target_include_directories(optix_common
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(optix_common
    PUBLIC OptiX::OptiX CUDA::cudart CUDA::cuda_driver fmt::fmt pep-cuda-math
    PRIVATE glfw glad imgui::imgui tinyobjloader::tinyobjloader
)

find_path(STB_INCLUDE_DIRS "stb.h")
target_include_directories(optix_common
    PRIVATE ${STB_INCLUDE_DIRS}
)