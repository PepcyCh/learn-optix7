#pragma once

#include "light_data.h"

struct LaunchParams {
    struct {
        pcm::Vec4 *color_buffer;
        pcm::Vec4 *prev_color_buffer;
        pcm::Vec4 *pos_buffer;
        pcm::Vec4 *norm_buffer;
        pcm::Vec4 *base_color_buffer;
        float accum_weight;
        int curr_time;
        int width;
        int height;
    } frame;

    struct {
        pcm::Vec3 *vertex;
        LightData *data;
        uint32_t light_count;
    } light;

    OptixTraversableHandle traversable;
};

enum class RayType : uint32_t {
    eShadow,
    eCount
};