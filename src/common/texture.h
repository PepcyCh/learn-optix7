#pragma once

#include <vector>

struct Texture {
    std::vector<uint8_t> data;
    int width;
    int height;
    int component;
    int pixel_stride;
};