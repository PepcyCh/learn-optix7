#include "optix_device.h"

#include "shader_common.h"
#include "launch_params.h"

__constant__ LaunchParams optix_launch_params;

struct RayPayload {
    Vec3 color;
};

OPTIX_CLOSESTHIT(Radiance)() {
    RayPayload *payload = GetRayPayload<RayPayload>();
    const TriMeshData *data = (const TriMeshData *)optixGetSbtDataPointer();
    const uint32_t prim_id = optixGetPrimitiveIndex();

    const uint32_t i0 = data->index[prim_id * 3];
    const uint32_t i1 = data->index[prim_id * 3 + 1];
    const uint32_t i2 = data->index[prim_id * 3 + 2];

    const float bc_u = optixGetTriangleBarycentrics().x;
    const float bc_v = optixGetTriangleBarycentrics().y;
    const float bc_w = 1.0f - bc_u - bc_v;

    const Vec3 norm = (bc_w * data->normal[i0] + bc_u * data->normal[i1] + bc_v * data->normal[i2]).Normalize();

    Vec3 base_color = data->base_color;
    if (data->base_color_mapped) {
        const Vec2 uv = bc_w * data->uv[i0] + bc_u * data->uv[i1] + bc_v * data->uv[i2];
        float4 tex_val = tex2D<float4>(data->base_color_map, uv.X(), uv.Y());
        base_color *= Vec3(tex_val.x, tex_val.y, tex_val.z);
    }
    
    const Vec3 ray_dir = GetWorldRayDirection();

    const float dot = max(norm.Dot(-ray_dir), 0.0f);
    
    payload->color = base_color * (dot * 0.9f + 0.1f); // avoid black ...
}

OPTIX_ANYHIT(Radiance)() {}

OPTIX_MISS(Radiance)() {
    RayPayload *payload = GetRayPayload<RayPayload>();
    payload->color = Vec3(1.0f, 1.0f, 1.0f);
}

OPTIX_RAYGEN(RenderFrame)() {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto &fr = optix_launch_params.frame;
    const auto &cam = optix_launch_params.camera;

    const float u = (ix + 0.5f) / fr.width - 0.5f;
    const float v = 0.5f - (iy + 0.5f) / fr.height;

    RayPayload payload;
    payload.color = Vec3::ZeroVec();

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
    fr.color_buffer[buffer_index] = Vec4(payload.color, 1.0f);
}