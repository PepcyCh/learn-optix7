# Learn OptiX 7

A project to learn OptiX 7.

## Dependencies

One may download and install these through site of NVIDIA.

* CUDA
* OptiX 7 SDK

One may install these dependencies manually or use [vcpkg](https://github.com/microsoft/vcpkg).

* [glfw3](https://github.com/glfw/glfw)
* [ImGui](https://github.com/ocornut/imgui)
* [fmt](https://github.com/fmtlib/fmt)

## Contents

1. initialization
   1. initialize CUDA and OptiX
   2. draw come color using raygen program
2. simple vec4 ray payload
   1. add 2 cubes and build the acceleration structure
   2. generate ray and call `optixTrace` in raygen program
   3. show color based on primitive ID in closesthit and miss program