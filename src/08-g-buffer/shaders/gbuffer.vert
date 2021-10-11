#version 330 core

layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_norm;
layout (location = 2) in vec2 a_uv;

out vec3 v_pos;
out vec3 v_norm;
out vec2 v_uv;

uniform mat4 u_proj;
uniform mat4 u_view;

void main() {
    v_pos = a_pos;
    v_norm = a_norm;
    v_uv = a_uv;

    gl_Position = u_proj * u_view * vec4(a_pos, 1.0);
}