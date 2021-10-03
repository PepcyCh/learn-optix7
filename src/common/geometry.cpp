#include "geometry.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "fmt/core.h"

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

}

GeometryUtils::MeshData GeometryUtils::Cube(float w, float h, float d) {
    MeshData mesh;

    const float hw = w * 0.5f;
    const float hh = h * 0.5f;
    const float hd = d * 0.5f;

    Vertex v[] = {
        Vertex {Vec3(-hw, -hh, -hd), -Vec3::kZ, -Vec3::kX, Vec2(1.0f, 1.0f)},
        Vertex {Vec3(-hw, hh, -hd), -Vec3::kZ, -Vec3::kX, Vec2(1.0f, 0.0f)},
        Vertex {Vec3(hw, hh, -hd), -Vec3::kZ, -Vec3::kX, Vec2(0.0f, 0.0f)},
        Vertex {Vec3(hw, -hh, -hd), -Vec3::kZ, -Vec3::kX, Vec2(0.0f, 1.0f)},

        Vertex {Vec3(-hw, -hh, hd), Vec3::kZ, Vec3::kX, Vec2(0.0f, 1.0f)},
        Vertex {Vec3(hw, -hh, hd), Vec3::kZ, Vec3::kX, Vec2(1.0f, 1.0f)},
        Vertex {Vec3(hw, hh, hd), Vec3::kZ, Vec3::kX, Vec2(1.0f, 0.0f)},
        Vertex {Vec3(-hw, hh, hd), Vec3::kZ, Vec3::kX, Vec2(0.0f, 0.0f)},

        Vertex {Vec3(-hw, -hh, -hd), -Vec3::kY, -Vec3::kZ, Vec2(0.0f, 0.0f)},
        Vertex {Vec3(hw, -hh, -hd), -Vec3::kY, -Vec3::kZ, Vec2(0.0f, 1.0f)},
        Vertex {Vec3(hw, -hh, hd), -Vec3::kY, -Vec3::kZ, Vec2(1.0f, 1.0f)},
        Vertex {Vec3(-hw, -hh, hd), -Vec3::kY, -Vec3::kZ, Vec2(1.0f, 0.0f)},

        Vertex {Vec3(-hw, hh, -hd), Vec3::kY, -Vec3::kZ, Vec2(1.0f, 0.0f)},
        Vertex {Vec3(-hw, hh, hd), Vec3::kY, -Vec3::kZ, Vec2(0.0f, 0.0f)},
        Vertex {Vec3(hw, hh, hd), Vec3::kY, -Vec3::kZ, Vec2(0.0f, 1.0f)},
        Vertex {Vec3(hw, hh, -hd), Vec3::kY, -Vec3::kZ, Vec2(1.0f, 1.0f)},

        Vertex {Vec3(-hw, -hh, -hd), -Vec3::kX, -Vec3::kZ, Vec2(0.0f, 1.0f)},
        Vertex {Vec3(-hw, -hh, hd), -Vec3::kX, -Vec3::kZ, Vec2(1.0f, 1.0f)},
        Vertex {Vec3(-hw, hh, hd), -Vec3::kX, -Vec3::kZ, Vec2(1.0f, 0.0f)},
        Vertex {Vec3(-hw, hh, -hd), -Vec3::kX, -Vec3::kZ, Vec2(0.0f, 0.0f)},

        Vertex {Vec3(hw, -hh, -hd), Vec3::kX, -Vec3::kZ, Vec2(1.0f, 1.0f)},
        Vertex {Vec3(hw, hh, -hd), Vec3::kX, -Vec3::kZ, Vec2(1.0f, 0.0f)},
        Vertex {Vec3(hw, hh, hd), Vec3::kX, -Vec3::kZ, Vec2(0.0f, 0.0f)},
        Vertex {Vec3(hw, -hh, hd), Vec3::kX, -Vec3::kZ, Vec2(0.0f, 1.0f)},
    };

    mesh.vertices.assign(v, v + 24);

    uint32_t ind[36] = {
        0, 1, 2,    0, 2, 3,    4, 5, 6,    4, 6, 7,
        8, 9, 10,   8, 10, 11,  12, 13, 14, 12, 14, 15,
        16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23
    };
    mesh.indices.assign(ind, ind + 36);

    return mesh;
}

std::vector<GeometryUtils::MeshData> GeometryUtils::LoadObj(std::string_view &&filename) {
    const std::string dirname(filename.substr(0, filename.find_last_of('/') + 1));

    tinyobj::attrib_t attribs;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;

    if (!tinyobj::LoadObj(&attribs, &shapes, &materials, &err, &err, filename.data(), dirname.data(), true)) {
        throw std::runtime_error(fmt::format("Failed to load obj file '{}'", filename));
    }

    std::vector<MeshData> meshes;
    meshes.reserve(shapes.size());
    for (const tinyobj::shape_t &shape : shapes) {
        MeshData mesh{};
        mesh.indices.reserve(shape.mesh.indices.size());

        std::map<tinyobj::index_t, uint32_t, TinyobjIndexComparator> index_map;
        for (const tinyobj::index_t &index : shape.mesh.indices) {
            if (index_map.count(index) == 0) {
                Vertex vert{};
                vert.pos = Vec3(
                    attribs.vertices[3 * index.vertex_index],
                    attribs.vertices[3 * index.vertex_index + 1],
                    attribs.vertices[3 * index.vertex_index + 2]
                );
                if (index.normal_index >= 0) {
                    vert.norm = Vec3(
                        attribs.normals[3 * index.normal_index],
                        attribs.normals[3 * index.normal_index + 1],
                        attribs.normals[3 * index.normal_index + 2]
                    );
                }
                if (index.texcoord_index >= 0) {
                    vert.uv = Vec2(
                        attribs.texcoords[2 * index.texcoord_index],
                        attribs.texcoords[2 * index.texcoord_index + 1]
                    );
                }

                index_map[index] = mesh.vertices.size();
                mesh.vertices.push_back(vert);
            }
            mesh.indices.emplace_back(index_map[index]);
        }

        meshes.emplace_back(mesh);
    }

    return meshes;
}
