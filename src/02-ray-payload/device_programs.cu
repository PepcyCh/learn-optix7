#include "optix_device.h"

#include "shader_common.h"
#include "launch_params.h"

__constant__ LaunchParams optix_launch_params;

struct RayPayload {
    pcm::Vec3 color;
};

OPTIX_CLOSESTHIT(Radiance)() {
    RayPayload *payload = GetRayPayload<RayPayload>();
    float val = optixGetPrimitiveIndex() / 32.0f;
    payload->color = pcm::Vec3(val, val, val);
}

OPTIX_ANYHIT(Radiance)() {}

OPTIX_MISS(Radiance)() {
    RayPayload *payload = GetRayPayload<RayPayload>();
    payload->color = pcm::Vec3(1.0f, 1.0f, 1.0f);
}

OPTIX_RAYGEN(RenderFrame)() {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto &fr = optix_launch_params.frame;
    const auto &cam = optix_launch_params.camera;

    const float u = (ix + 0.5f) / fr.width - 0.5f;
    const float v = 0.5f - (iy + 0.5f) / fr.height;

    RayPayload payload;
    payload.color = pcm::Vec3::Zero();

    RayDesc ray;
    ray.origin = cam.position;
    ray.direction = (cam.direction + cam.right * u + cam.up * v).Normalize();
    ray.t_min = 0.0f;
    ray.t_max = 1e20f;

    TraceRay(
        optix_launch_params.traversable,
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0xff,
        0,
        1,
        0,
        ray,
        &payload
    );

    const unsigned int buffer_index = ix + iy * fr.width;
    fr.color_buffer[buffer_index] = pcm::Vec4(payload.color, 1.0f);
}