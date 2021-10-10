#pragma once

#include <vector>

#include <string_view>

#include "pcmath/pcmath.h"

namespace pcm = pep::cuda_math;

class GeometryUtils {
public:
    struct MeshData {
        std::vector<pcm::Vec3> positions;
        std::vector<pcm::Vec3> normals;
        std::vector<pcm::Vec2> uvs;
        std::vector<uint32_t> indices;
    };

    static MeshData Cube(float w, float h, float d);

    static std::vector<MeshData> LoadObj(std::string_view &&filename);
};