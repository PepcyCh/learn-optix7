add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.cu
    COMMAND nvcc ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.cu
        -ptx -o ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx
        -I ${OptiX_ROOT_DIR}/include -I ${PROJECT_SOURCE_DIR}/src/common
)

add_executable(02-ray-payload
    main.cpp
    launch_params.h
    ${CMAKE_CURRENT_SOURCE_DIR}/device_programs.ptx
)

target_compile_features(02-ray-payload
    PRIVATE cxx_std_20
)

target_link_libraries(02-ray-payload
    PRIVATE optix_common
)