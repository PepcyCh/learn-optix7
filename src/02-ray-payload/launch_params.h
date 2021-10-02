#pragma once

#include "math/vec.h"

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