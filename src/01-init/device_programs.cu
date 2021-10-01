#include "optix_device.h"

#include "shader_common.h"
#include "launch_params.h"

__constant__ LaunchParams optix_launch_params;

OPTIX_CLOSESTHIT(Radiance)() {}

OPTIX_ANYHIT(Radiance)() {}

OPTIX_MISS(Radiance)() {}

OPTIX_RAYGEN(RenderFrame)() {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int r = ix % 256;
    const int g = iy % 256;
    const int b = (ix + iy) % 256;

    const unsigned int rgba = 0xff000000 | r | (g << 8) | (b << 16);
    const unsigned int buffer_index = ix + iy * optix_launch_params.frame_width;
    optix_launch_params.color_buffer[buffer_index] = rgba;
}