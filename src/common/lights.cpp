#include "lights.h"

#include <algorithm>
#include <iterator>

namespace ranges = std::ranges;

void Lights::AddMesh(const GeometryUtils::MeshData &mesh, const Vec3 &strength, const Vec3 &translate) {
    ranges::transform(mesh.positions, std::back_inserter(vertices),
        [translate](const Vec3 &pos) { return pos + translate; });
    const int first_index = lights.size();
    for (size_t f = 0; f < mesh.indices.size() / 3; f++) {
        LightData data{};
        data.index = IVec3(
            mesh.indices[3 * f] + first_index,
            mesh.indices[3 * f + 1] + first_index,
            mesh.indices[3 * f + 2] + first_index
        );
        data.strength = strength;
        lights.emplace_back(data);
    }
}

void Lights::BuildAliasTable() {
    float sum = 0.0f;
    for (LightData &data : lights) {
        const Vec3 p0 = vertices[data.index.X()];
        const Vec3 p1 = vertices[data.index.Y()];
        const Vec3 p2 = vertices[data.index.Z()];
        const float area2 = Vec3(p1 - p0).Cross(p2 - p0).Length();
        const float gray = 0.299f * data.strength.X() + 0.587f * data.strength.Y() + 0.114f * data.strength.Z();
        data.at_probability = area2 * gray;
        data.at_another_index = -1;
        sum += data.at_probability;
    }
    const float sum_inv = 1.0f / sum;
    for (LightData &data : lights) {
        data.at_probability *= sum_inv;
    }

    std::vector<float> u(lights.size());
    ranges::transform(lights, u.begin(), [&](const LightData &data) { return data.at_probability * u.size(); });
    size_t poor = ranges::find_if(u, [](float u) { return u < 1.0f; }) - u.begin();
    size_t poor_max = poor;
    size_t rich = ranges::find_if(u, [](float u) { return u > 1.0f; }) - u.begin();
    while (poor < u.size() && rich < u.size()) {
        const float diff = 1.0f - u[poor];
        u[rich] -= diff;
        lights[poor].at_another_index = rich;

        if (u[rich] < 1.0 && rich < poor_max) {
            poor = rich;
        } else {
            poor = ranges::find_if(u.begin() + poor_max + 1, u.end(), [](float u) { return u < 1.0f; }) - u.begin();
            poor_max = poor;
        }

        rich = ranges::find_if(u.begin() + rich, u.end(), [](float u) { return u > 1.0f; }) - u.begin();
    }
}
