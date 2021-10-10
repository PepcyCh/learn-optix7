#include "optix_device.h"

#include "shader_common.h"
#include "launch_params.h"

__constant__ LaunchParams optix_launch_params;

struct RayPayload {
    pcm::Vec3 color;
};

OPTIX_CLOSESTHIT(Radiance)() {
    RayPayload *payload = GetRayPayload<RayPayload>();
    const TriMeshData *mesh_data = (const TriMeshData *)optixGetSbtDataPointer();
    const uint32_t prim_id = optixGetPrimitiveIndex();

    const uint32_t i0 = mesh_data->index[prim_id * 3];
    const uint32_t i1 = mesh_data->index[prim_id * 3 + 1];
    const uint32_t i2 = mesh_data->index[prim_id * 3 + 2];
    const pcm::Vec3 &p0 = mesh_data->vertex[i0];
    const pcm::Vec3 &p1 = mesh_data->vertex[i1];
    const pcm::Vec3 &p2 = mesh_data->vertex[i2];
    const pcm::Vec3 norm = (p1 - p0).Cross(p2 - p0).Normalize();
    
    const pcm::Vec3 ray_dir = GetWorldRayDirection();

    const float dot = max(norm.Dot(-ray_dir), 0.0f);
    
    payload->color = mesh_data->color * (dot * 0.9f + 0.1f); // avoid black ...
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