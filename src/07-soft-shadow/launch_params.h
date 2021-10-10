#pragma once

#include "light_data.h"

struct TriMeshData {
    pcm::Vec3 *vertex;
    pcm::Vec3 *normal;
    pcm::Vec2 *uv;
    uint32_t *index;
    pcm::Vec3 base_color;
    bool base_color_mapped;
    cudaTextureObject_t base_color_map;
};

struct LaunchParams {
    struct {
        pcm::Vec4 *color_buffer;
        pcm::Vec4 *prev_color_buffer;
        float accum_weight;
        int curr_time;
        int width;
        int height;
    } frame;

    struct {
        pcm::Vec3 position;
        pcm::Vec3 direction;
        pcm::Vec3 right;
        pcm::Vec3 up;
    } camera;

    struct {
        pcm::Vec3 *vertex;
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