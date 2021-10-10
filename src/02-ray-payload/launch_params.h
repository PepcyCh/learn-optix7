#pragma once

#include "pcmath/pcmath.h"

namespace pcm = pep::cuda_math;

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