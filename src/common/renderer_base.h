#pragma once

#include <string_view>

#include "cuda.h"
#include "cuda_texture.h"
#include "GLFW/glfw3.h"
#include "optix.h"

#include "frame_counter.h"
#include "pixel_buffer.h"

class OptixApp {
public:
    OptixApp(int width, int height, std::string_view &&title);

    virtual ~OptixApp();

    OptixApp(const OptixApp &) = delete;
    OptixApp &operator=(const OptixApp &) = delete;

    virtual bool Initialize();

    void MainLoop();

protected:
    bool InitOptix() const;

    void CreateContext();

    void InitGl();

    virtual bool Resize(int width, int height);

    virtual void Update() {}

    virtual void Render() = 0;

    virtual void OnMouse(double x, double y, uint8_t state) {}

    friend void GlfwWindowResizeFunc(GLFWwindow *window, int width, int height);
    friend void GlfwWindowCursorPosFunc(GLFWwindow *window, double x, double y);

    CUcontext cuda_context_;
    CUstream cuda_stream_;
    OptixDeviceContext optix_context_;

    int color_buffer_width_;
    int color_buffer_height_;
    PixelUnpackBuffer color_buffer_;

    GLFWwindow *window_;
    unsigned int screen_tex_;
    unsigned int screen_shader_;

    FrameCounter frame_counter_;
};