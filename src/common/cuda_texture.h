#pragma once

#include "texture_types.h"

#include "texture.h"

class CudaTexture {
public:
    CudaTexture(const Texture &tex);

    cudaTextureObject_t CudaObject() const;

private:
    cudaArray_t array_;
    cudaTextureObject_t object_;
};