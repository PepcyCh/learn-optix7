#pragma once

#include "cuda_texture.h"

class PixelUnpackBuffer {
public:
    ~PixelUnpackBuffer();

    void Init(int width, int height, int pixel_stride, int channel_format, int pixel_type);

    void UnpackTo(unsigned int tex_id) const;

    void *Map();

    template<typename T>
    T *TypedMap() {
        return static_cast<T *>(Map());
    }

    void Unmap();

    void Resize(int width, int height, int pixel_stride, int channel_format, int pixel_type);

private:
    unsigned int buffer_id_ = 0;
    cudaGraphicsResource_t cuda_res_ = nullptr;

    int width_ = 0;
    int height_ = 0;
    int channel_format_ = 0;
    int pixel_type_ = 0;
};