add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.cu
    COMMAND nvcc ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.cu
        -ptx -o ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx -std=c++17
        -I ${OptiX_ROOT_DIR}/include -I ${PROJECT_SOURCE_DIR}/src/common
        -I ${PROJECT_SOURCE_DIR}/external/pep-cuda-math/include
)

add_executable(01-init
    main.cpp
    launch_params.h
    ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx
)

target_compile_features(01-init
    PRIVATE cxx_std_20
)

target_link_libraries(01-init
    PRIVATE optix_common
)