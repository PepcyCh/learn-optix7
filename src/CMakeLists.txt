find_package(CUDAToolkit REQUIRED)
find_package(OptiX REQUIRED)

find_package(fmt CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)

configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/defines.h.in
	${CMAKE_CURRENT_SOURCE_DIR}/defines.h
)

add_subdirectory(common)

add_subdirectory(01-init)
add_subdirectory(02-ray-payload)
add_subdirectory(03-sbt-data)
add_subdirectory(04-multi-object)
add_subdirectory(05-textures)
add_subdirectory(06-shadow)
add_subdirectory(07-soft-shadow)
add_subdirectory(08-g-buffer)