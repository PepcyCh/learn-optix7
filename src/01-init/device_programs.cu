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

    const unsigned int buffer_index = 4 * (ix + iy * optix_launch_params.frame_width);
    optix_launch_params.color_buffer[buffer_index] = r / 255.0f;
    optix_launch_params.color_buffer[buffer_index + 1] = g / 255.0f;
    optix_launch_params.color_buffer[buffer_index + 2] = b / 255.0f;
    optix_launch_params.color_buffer[buffer_index + 3] = 1.0;
}