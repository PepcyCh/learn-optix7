cmake_minimum_required(VERSION 3.18)

project(learn-optix7)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_subdirectory(external/glad)

add_subdirectory(src)