#pragma once

#include "math/vec.h"

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

    struct {
        Vec3 position;
        Vec3 strength;
    } light;

    OptixTraversableHandle traversable;
};

enum class RayType : uint32_t {
    eRadiance,
    eShadow,
    eCount
};