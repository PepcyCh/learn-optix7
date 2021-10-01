find_package(CUDAToolkit REQUIRED)
find_package(OptiX REQUIRED)

find_package(fmt CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/defines.h.in
	${CMAKE_CURRENT_SOURCE_DIR}/defines.h
)

add_subdirectory(common)

add_subdirectory(01-init)