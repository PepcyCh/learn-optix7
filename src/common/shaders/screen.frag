#version 330 core

in vec2 v_texcoords;

out vec4 frag_color;

uniform sampler2D color_buffer_img;

void main() {
    vec4 color = texture(color_buffer_img, v_texcoords);
    frag_color = color;
}