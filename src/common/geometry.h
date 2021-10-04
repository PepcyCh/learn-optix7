#pragma once

#include <vector>

#include <string_view>

#include "math/vec.h"

class GeometryUtils {
public:
    struct MeshData {
        std::vector<Vec3> positions;
        std::vector<Vec3> normals;
        std::vector<Vec2> uvs;
        std::vector<uint32_t> indices;
    };

    static MeshData Cube(float w, float h, float d);

    static std::vector<MeshData> LoadObj(std::string_view &&filename);
};