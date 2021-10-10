#pragma once

#include <vector>

#include "geometry.h"
#include "light_data.h"

struct Lights {
    void AddMesh(const GeometryUtils::MeshData &mesh, const pcm::Vec3 &strength, const pcm::Vec3 &translate);

    void BuildAliasTable();

    std::vector<pcm::Vec3> vertices;
    std::vector<LightData> lights;
};