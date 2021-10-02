#pragma once

#include <vector>

#include "math/vec.h"

class GeometryUtils {
public:
    struct Vertex {
        Vec3 pos;
        Vec3 norm;
        Vec3 tan;
        Vec2 uv;
    };

    struct MeshData {
        std::vector<uint32_t> indices;
        std::vector<Vertex> vertices;
    };

    static MeshData Cube(float w, float h, float d);
};