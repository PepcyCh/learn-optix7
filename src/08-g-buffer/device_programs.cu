#include "optix_device.h"

#include "shader_common.h"
#include "random.h"
#include "launch_params.h"

__constant__ LaunchParams optix_launch_params;

struct VisibilityRayPayload {
    float visibility;
};

static __device__ void SampleLight(
    const pcm::Vec3 &pos,
    RandomNumberGenerator &rng,
    pcm::Vec3 &light_dir,
    pcm::Vec3 &light_strength,
    float &light_dist,
    float &sample_pdf
) {
    const auto &lights = optix_launch_params.light;

    float val = rng.NextFloat(0.0f, lights.light_count);
    uint32_t light_index = min(uint32_t(val), lights.light_count - 1);
    val -= light_index;
    if (val > lights.data[light_index].at_probability * lights.light_count) {
        light_index = lights.data[light_index].at_another_index;
    }

    const pcm::IVec3 ind = lights.data[light_index].index;
    const pcm::Vec3 p0 = lights.vertex[ind.X()];
    const pcm::Vec3 p1 = lights.vertex[ind.Y()];
    const pcm::Vec3 p2 = lights.vertex[ind.Z()];
    const pcm::Vec3 cross = (p1 - p0).Cross(p2 - p0);
    const pcm::Vec3 norm = cross.Normalize();

    const float r0 = rng.NextFloat(0.0f, 1.0f);
    const float r0_sqrt = sqrt(r0);
    const float r1 = rng.NextFloat(0.0f, 1.0f);

    const float u = 1.0f - r0_sqrt;
    const float v = r0_sqrt * (1.0f - r1);
    const float w = 1.0f - u - v;
    const pcm::Vec3 light_pos = u * p0 + v * p1 + w * p2;

    const pcm::Vec3 light_vec = light_pos - pos;
    const float light_dist_sqr = light_vec.MagnitudeSqr();
    light_dist = sqrt(light_dist_sqr);
    light_dir = light_vec / light_dist;
    light_strength = lights.data[light_index].strength * max(norm.Dot(-light_dir), 0.0) / light_dist_sqr;

    sample_pdf = lights.data[light_index].at_probability / (cross.Length() * 0.5f);
}

OPTIX_CLOSESTHIT(Empty)() {}

OPTIX_ANYHIT(Empty)() {}

OPTIX_MISS(Shadow)() {
    VisibilityRayPayload *payload = GetRayPayload<VisibilityRayPayload>();
    payload->visibility = 1.0f;
}

OPTIX_RAYGEN(RenderFrame)() {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto &fr = optix_launch_params.frame;

    const uint32_t buffer_index = ix + iy * fr.width;

    RandomNumberGenerator rng;
    rng.Seed(buffer_index + fr.curr_time);

    pcm::Vec3 pos = pcm::Vec3(fr.pos_buffer[buffer_index]);
    pcm::Vec3 norm = pcm::Vec3(fr.norm_buffer[buffer_index]);
    pcm::Vec3 base_color = pcm::Vec3(fr.base_color_buffer[buffer_index]);

    pcm::Vec3 color(1.0f, 1.0f, 1.0f);
    if (norm.MagnitudeSqr() > 0.8f) {
        pcm::Vec3 light_dir;
        pcm::Vec3 light_strength;
        float light_dist;
        float light_pdf;
        SampleLight(pos, rng, light_dir, light_strength, light_dist, light_pdf);

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

        color = dot * shadow_payload.visibility * base_color * light_strength / light_pdf + 0.1f * base_color;
    }

    const pcm::Vec4 accum_result = fr.accum_weight * pcm::Vec4(color, 1.0f)
        + (1.0f - fr.accum_weight) * fr.prev_color_buffer[buffer_index];
    fr.color_buffer[buffer_index] = accum_result;
}