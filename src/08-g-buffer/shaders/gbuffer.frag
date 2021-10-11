#version 330 core

in vec3 v_pos;
in vec3 v_norm;
in vec2 v_uv;

layout (location = 0) out vec4 o_pos;
layout (location = 1) out vec4 o_norm;
layout (location = 2) out vec4 o_base_color;

uniform vec4 u_base_color;
uniform sampler2D u_base_color_tex;

void main() {
    vec3 base_color = u_base_color.rgb;
    if (u_base_color.a > 0.0) {
        base_color *= texture(u_base_color_tex, v_uv).rgb;
    }

    o_pos = vec4(v_pos, 1.0);
    o_norm = vec4(normalize(v_norm), 1.0);
    o_base_color = vec4(base_color, 1.0);
}