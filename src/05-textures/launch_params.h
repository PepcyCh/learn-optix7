#pragma once

#include "pcmath/pcmath.h"

namespace pcm = pep::cuda_math;

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