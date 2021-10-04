#include "scene.h"

#include <stdexcept>

#include "fmt/core.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "tiny_obj_loader.h" // implementation macro has been defines in geometry.cpp

namespace {

struct TinyobjIndexComparator {
    bool operator()(const tinyobj::index_t &lhs, const tinyobj::index_t &rhs) const {
        if (lhs.vertex_index < rhs.vertex_index) {
            return true;
        }
        if (lhs.vertex_index > rhs.vertex_index) {
            return false;
        }
        if (lhs.normal_index < rhs.normal_index) {
            return true;
        }
        if (lhs.normal_index > rhs.normal_index) {
            return false;
        }
        if (lhs.texcoord_index < rhs.texcoord_index) {
            return true;
        }
        if (lhs.texcoord_index > rhs.texcoord_index) {
            return false;
        }

        return false;
    }
};

int LoadTexture(
    const std::string &filename,
    const std::string &dirname,
    std::unordered_map<std::string, int> &texture_map,
    Scene &scene
) {
    const std::string fullname = dirname + filename;
    if (texture_map.count(fullname)) {
        return texture_map[fullname];
    }
    
    int width;
    int height;
    int component;
    // I failed to make RGB textures work without converting them to RGBA ...
    uint8_t *data = stbi_load(fullname.c_str(), &width, &height, &component, STBI_rgb_alpha);
    component = 4;

    Texture tex{
        .data = std::vector(data, data + width * height * component),
        .width = width,
        .height = height,
        .component = component,
        .pixel_stride = component * static_cast<int>(sizeof(uint8_t))
    };
    const int pitch = width * tex.pixel_stride;
    for (int i = 0; i < height / 2; i++) {
        for (int j = 0; j < pitch; j++) {
            std::swap(tex.data[i * pitch + j], tex.data[(height - 1 - i) * pitch + j]);
        }
    }

    stbi_image_free(data);

    const int tex_id = scene.textures.size();
    scene.textures.emplace_back(tex);
    texture_map[fullname] = tex_id;
    return tex_id;
}

}

Scene Scene::LoadObj(std::string_view &&filename) {
    const std::string dirname(filename.substr(0, filename.find_last_of('/') + 1));

    tinyobj::attrib_t attribs;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;

    if (!tinyobj::LoadObj(&attribs, &shapes, &materials, &err, &err, filename.data(), dirname.data(), true)) {
        throw std::runtime_error(fmt::format("Failed to load obj file '{}'", filename));
    }

    Scene scene;
    scene.meshes.reserve(shapes.size());
    std::unordered_map<std::string, int> texture_map;
    for (const tinyobj::shape_t &shape : shapes) {
        Mesh mesh{};
        mesh.data.indices.reserve(shape.mesh.indices.size());

        std::map<tinyobj::index_t, uint32_t, TinyobjIndexComparator> index_map;
        for (const tinyobj::index_t &index : shape.mesh.indices) {
            if (index_map.count(index) == 0) {
                mesh.data.positions.emplace_back(
                    attribs.vertices[3 * index.vertex_index],
                    attribs.vertices[3 * index.vertex_index + 1],
                    attribs.vertices[3 * index.vertex_index + 2]
                );
                if (index.normal_index >= 0) {
                    mesh.data.normals.emplace_back(
                        attribs.normals[3 * index.normal_index],
                        attribs.normals[3 * index.normal_index + 1],
                        attribs.normals[3 * index.normal_index + 2]
                    );
                }
                if (index.texcoord_index >= 0) {
                    mesh.data.uvs.emplace_back(
                        attribs.texcoords[2 * index.texcoord_index],
                        attribs.texcoords[2 * index.texcoord_index + 1]
                    );
                }

                index_map[index] = mesh.data.positions.size() - 1;
            }
            mesh.data.indices.emplace_back(index_map[index]);
        }

        const int mat_id = shape.mesh.material_ids[0];
        auto diffuse = materials[mat_id].diffuse;
        mesh.base_color = Vec3(diffuse[0], diffuse[1], diffuse[2]);
        if (materials[mat_id].diffuse_texname.empty()) {
            mesh.base_color_map_index = -1;
        } else {
            mesh.base_color_map_index = LoadTexture(materials[mat_id].diffuse_texname, dirname, texture_map, scene);
        }

        scene.meshes.emplace_back(mesh);
    }

    return scene;
}
