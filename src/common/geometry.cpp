#include "geometry.h"

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
