# Learn OptiX 7

A project to learn OptiX 7.

## Dependencies

One may download and install these through site of NVIDIA.

* CUDA
* OptiX 7 SDK

One may install these dependencies manually or use [vcpkg](https://github.com/microsoft/vcpkg).

* [fmt](https://github.com/fmtlib/fmt)
* [glfw3](https://github.com/glfw/glfw)
* [ImGui](https://github.com/ocornut/imgui)
* [stb](https://github.com/nothings/stb)
* [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader)

## Contents

1. initialization
   1. initialize CUDA and OptiX
   2. draw come color using raygen program
2. simple vec4 ray payload
   1. add 2 cubes and build the acceleration structure
   2. generate ray and call `optixTrace` in raygen program
   3. show color based on primitive ID in closesthit and miss program
3. simple SBT data
   1. add a hard-coded color and mesh data (vertices & indices) in hit group record
   2. fetch these data in closesthit program and shade mesh with color * ndotv (assume camera is also a point light)
4. multiple meshes
   1. use multiple hit group record to store data for different mesh
   2. load sponza.obj and show different mesh with random color (the model can be downloaded from [here](https://casual-effects.com/data/))
5. textures
   1. texture map with CUDA texture
6. hard shadow
   1. add second ray type - shadow ray
   2. use miss program to determine whether it is shadowed or not
   3. a moving point light

