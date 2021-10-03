#pragma once

#include <vector>

#include <string_view>

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

    static std::vector<MeshData> LoadObj(std::string_view &&filename);
};