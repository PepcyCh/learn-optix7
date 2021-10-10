#include "optix_device.h"

#include "shader_common.h"
#include "launch_params.h"

__constant__ LaunchParams optix_launch_params;

struct RayPayload {
    pcm::Vec3 color;
};

struct VisibilityRayPayload {
    float visibility;
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

    const pcm::Vec3 pos = bc_w * data->vertex[i0] + bc_u * data->vertex[i1] + bc_v * data->vertex[i2];
    const pcm::Vec3 norm = (bc_w * data->normal[i0] + bc_u * data->normal[i1] + bc_v * data->normal[i2]).Normalize();

    pcm::Vec3 base_color = data->base_color;
    if (data->base_color_mapped) {
        const pcm::Vec2 uv = bc_w * data->uv[i0] + bc_u * data->uv[i1] + bc_v * data->uv[i2];
        float4 tex_val = tex2D<float4>(data->base_color_map, uv.X(), uv.Y());
        base_color *= pcm::Vec3(tex_val.x, tex_val.y, tex_val.z);
    }
    
    const pcm::Vec3 light_vec = optix_launch_params.light.position - pos;
    const float light_dist = light_vec.Length();
    const pcm::Vec3 light_dir = light_vec / light_dist;

    const float dot = max(norm.Dot(light_dir), 0.0f);

    VisibilityRayPayload shadow_payload;
    shadow_payload.visibility = 0.0f;

    RayDesc ray;
    ray.origin = pos;
    ray.direction = light_dir;
    ray.t_min = 0.001f;
    ray.t_max = light_dist - 0.001f;

    TraceRay(
        optix_launch_params.traversable,
        OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        0xff,
        static_cast<uint32_t>(RayType::eShadow),
        static_cast<uint32_t>(RayType::eCount),
        static_cast<uint32_t>(RayType::eShadow),
        ray,
        &shadow_payload
    );

    const pcm::Vec3 light_strength = optix_launch_params.light.strength / light_dist;
    payload->color = dot * shadow_payload.visibility * base_color * light_strength + 0.1f * base_color;
}

OPTIX_CLOSESTHIT(Empty)() {}

OPTIX_ANYHIT(Empty)() {}

OPTIX_MISS(Radiance)() {
    RayPayload *payload = GetRayPayload<RayPayload>();
    payload->color = pcm::Vec3(1.0f, 1.0f, 1.0f);
}

OPTIX_MISS(Shadow)() {
    VisibilityRayPayload *payload = GetRayPayload<VisibilityRayPayload>();
    payload->visibility = 1.0f;
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
        static_cast<uint32_t>(RayType::eRadiance),
        static_cast<uint32_t>(RayType::eCount),
        static_cast<uint32_t>(RayType::eRadiance),
        ray,
        &payload
    );

    const unsigned int buffer_index = ix + iy * fr.width;
    fr.color_buffer[buffer_index] = pcm::Vec4(payload.color, 1.0f);
}