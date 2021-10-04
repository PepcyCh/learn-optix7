#include "cuda_texture.h"

#include "channel_descriptor.h"

#include "check_macros.h"

CudaTexture::CudaTexture(const Texture &tex) {
    const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
    CUDA_CHECK(cudaMallocArray(&array_, &channel_desc, tex.width, tex.height));

    const int pitch = tex.width * tex.pixel_stride;
    CUDA_CHECK(cudaMemcpy2DToArray(array_, 0, 0, tex.data.data(), pitch, pitch, tex.height, cudaMemcpyHostToDevice));

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array_;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 1.0f;
    tex_desc.sRGB = 0;

    CUDA_CHECK(cudaCreateTextureObject(&object_, &res_desc, &tex_desc, nullptr));
}

cudaTextureObject_t CudaTexture::CudaObject() const {
    return object_;
}

