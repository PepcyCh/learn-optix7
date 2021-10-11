#pragma once

#include "pixel_buffer.h"

struct GBuffer {
    void Init(int width, int height);

    void Resize(int width, int height);

    unsigned int fbo;
    unsigned int pos_tex;
    unsigned int norm_tex;
    unsigned int base_color_tex;
    unsigned int depth_rbo;
    PixelPackBuffer pos_buffer;
    PixelPackBuffer norm_buffer;
    PixelPackBuffer base_color_buffer;

private:
    void InitGl(int width, int height);
};