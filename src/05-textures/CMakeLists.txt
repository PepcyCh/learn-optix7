add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.cu
    COMMAND nvcc ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.cu
        -ptx -o ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx
        -I ${OptiX_ROOT_DIR}/include -I ${PROJECT_SOURCE_DIR}/src/common
)

add_executable(05-textures
    main.cpp
    launch_params.h
    ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx
)

target_compile_features(05-textures
    PRIVATE cxx_std_20
)

target_link_libraries(05-textures
    PRIVATE optix_common
)