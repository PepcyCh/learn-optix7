#pragma once

#include "pcmath/pcmath.h"

struct TriMeshData {
    pcm::Vec3 color;
    pcm::Vec3 *vertex;
    uint32_t *index;
};

struct LaunchParams {
    struct {
        pcm::Vec4 *color_buffer;
        int id;
        int width;
        int height;
    } frame;

    struct {
        pcm::Vec3 position;
        pcm::Vec3 direction;
        pcm::Vec3 right;
        pcm::Vec3 up;
    } camera;

    OptixTraversableHandle traversable;
};