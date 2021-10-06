#pragma once

#include "light_data.h"

struct TriMeshData {
    Vec3 *vertex;
    Vec3 *normal;
    Vec2 *uv;
    uint32_t *index;
    Vec3 base_color;
    bool base_color_mapped;
    cudaTextureObject_t base_color_map;
};

struct LaunchParams {
    struct {
        Vec4 *color_buffer;
        Vec4 *prev_color_buffer;
        float accum_weight;
        int curr_time;
        int width;
        int height;
    } frame;

    struct {
        Vec3 position;
        Vec3 direction;
        Vec3 right;
        Vec3 up;
    } camera;

    struct {
        Vec3 *vertex;
        LightData *data;
        uint32_t light_count;
    } light;

    OptixTraversableHandle traversable;
};

enum class RayType : uint32_t {
    eRadiance,
    eShadow,
    eCount
};