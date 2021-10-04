#pragma once

#include "geometry.h"
#include "texture.h"

struct Mesh {
    GeometryUtils::MeshData data;
    Vec3 base_color;
    int base_color_map_index = -1;
};

struct Scene {
    std::vector<Mesh> meshes;
    std::vector<Texture> textures;

    static Scene LoadObj(std::string_view &&filename);
};