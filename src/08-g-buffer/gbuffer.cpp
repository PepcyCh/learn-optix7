#include "gbuffer.h"

#include "glad/glad.h"

void GBuffer::Init(int width, int height) {
    InitGl(width, height);

    pos_buffer.Init(width, height, 16, GL_RGBA, GL_FLOAT);
    norm_buffer.Init(width, height, 16, GL_RGBA, GL_FLOAT);
    base_color_buffer.Init(width, height, 16, GL_RGBA, GL_FLOAT);
}

void GBuffer::Resize(int width, int height) {
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &pos_tex);
    glDeleteTextures(1, &norm_tex);
    glDeleteTextures(1, &base_color_tex);
    glDeleteRenderbuffers(1, &depth_rbo);

    InitGl(width, height);

    pos_buffer.Resize(width, height, 16, GL_RGBA, GL_FLOAT);
    norm_buffer.Resize(width, height, 16, GL_RGBA, GL_FLOAT);
    base_color_buffer.Resize(width, height, 16, GL_RGBA, GL_FLOAT);
}

void GBuffer::InitGl(int width, int height) {
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glViewport(0, 0, width, height);

    glGenTextures(1, &pos_tex);
    glBindTexture(GL_TEXTURE_2D, pos_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pos_tex, 0);

    glGenTextures(1, &norm_tex);
    glBindTexture(GL_TEXTURE_2D, norm_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, norm_tex, 0);

    glGenTextures(1, &base_color_tex);
    glBindTexture(GL_TEXTURE_2D, base_color_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, base_color_tex, 0);

    glGenRenderbuffers(1, &depth_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, depth_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo);

    const GLuint color_attachs[] = {
        GL_COLOR_ATTACHMENT0,
        GL_COLOR_ATTACHMENT1,
        GL_COLOR_ATTACHMENT2
    };
    glDrawBuffers(3, color_attachs);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}
