#pragma once

#include <vector>

#include "geometry.h"
#include "light_data.h"

struct Lights {
    void AddMesh(const GeometryUtils::MeshData &mesh, const Vec3 &strength, const Vec3 &translate);

    void BuildAliasTable();

    std::vector<Vec3> vertices;
    std::vector<LightData> lights;
};