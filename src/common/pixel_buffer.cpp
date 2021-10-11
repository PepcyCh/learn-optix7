#include "pixel_buffer.h"

#include <cassert>

#include "glad/glad.h"
#include "cuda_gl_interop.h"

#include "check_macros.h"

PixelUnpackBuffer::~PixelUnpackBuffer() {
    cudaGraphicsUnregisterResource(cuda_res_);
    glDeleteBuffers(1, &buffer_id_);
}

void PixelUnpackBuffer::Init(int width, int height, int pixel_stride, int channel_format, int channel_type) {
    width_ = width;
    height_ = height;
    channel_format_ = channel_format;
    channel_type_ = channel_type;

    const size_t buffer_size = static_cast<size_t>(width) * height * pixel_stride;

    glGenBuffers(1, &buffer_id_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer_id_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer_size, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_res_, buffer_id_, cudaGraphicsMapFlagsWriteDiscard));
}

void PixelUnpackBuffer::UnpackTo(unsigned int tex_id) const {
    assert(buffer_id_ && cuda_res_);

    glBindTexture(GL_TEXTURE_2D, tex_id);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer_id_);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, channel_format_, channel_type_, nullptr);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void *PixelUnpackBuffer::Map() {
    assert(buffer_id_ && cuda_res_);

    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_res_, nullptr));
    size_t mapped_size;
    void *ptr;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&ptr, &mapped_size, cuda_res_));
    return ptr;
}

void PixelUnpackBuffer::Unmap() {
    assert(buffer_id_ && cuda_res_);

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_res_, nullptr));
}

void PixelUnpackBuffer::Resize(int width, int height, int pixel_stride, int channel_format, int channel_type) {
    if (cuda_res_) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_res_));
        glDeleteBuffers(1, &buffer_id_);
    }

    Init(width, height, pixel_stride, channel_format, channel_type);
}

PixelPackBuffer::~PixelPackBuffer() {
    cudaGraphicsUnregisterResource(cuda_res_);
    glDeleteBuffers(1, &buffer_id_);
}

void PixelPackBuffer::Init(int width, int height, int pixel_stride, int channel_format, int channel_type) {
    width_ = width;
    height_ = height;
    channel_format_ = channel_format;
    channel_type_ = channel_type;

    const size_t buffer_size = static_cast<size_t>(width) * height * pixel_stride;

    glGenBuffers(1, &buffer_id_);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, buffer_id_);
    glBufferData(GL_PIXEL_PACK_BUFFER, buffer_size, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_res_, buffer_id_, cudaGraphicsMapFlagsReadOnly));
}

void PixelPackBuffer::PackFrom(unsigned int target) const {
    assert(buffer_id_ && cuda_res_);

    glReadBuffer(target);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, buffer_id_);
    glReadPixels(0, 0, width_, height_, channel_format_, channel_type_, nullptr);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    glReadBuffer(0);
}

void *PixelPackBuffer::Map() {
    assert(buffer_id_ && cuda_res_);

    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_res_, nullptr));
    size_t mapped_size;
    void *ptr;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&ptr, &mapped_size, cuda_res_));
    return ptr;
}

void PixelPackBuffer::Unmap() {
    assert(buffer_id_ && cuda_res_);

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_res_, nullptr));
}

void PixelPackBuffer::Resize(int width, int height, int pixel_stride, int channel_format, int channel_type) {
    if (cuda_res_) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_res_));
        glDeleteBuffers(1, &buffer_id_);
    }

    Init(width, height, pixel_stride, channel_format, channel_type);
}
