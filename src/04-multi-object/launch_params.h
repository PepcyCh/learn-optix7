#pragma once

#include "math/vec.h"

struct TriMeshData {
    Vec3 color;
    Vec3 *vertex;
    uint32_t *index;
};

struct LaunchParams {
    struct {
        Vec4 *color_buffer;
        int id;
        int width;
        int height;
    } frame;

    struct {
        Vec3 position;
        Vec3 direction;
        Vec3 right;
        Vec3 up;
    } camera;

    OptixTraversableHandle traversable;
};