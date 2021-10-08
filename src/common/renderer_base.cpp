#include "glad/glad.h" // glad header must be put ahead of glfw ones
#include "renderer_base.h"

#include <format>
#include <fstream>

#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "optix_stubs.h"
#include "optix_function_table_definition.h"

#include "../defines.h"
#include "check_macros.h"

namespace {

void OptixLogFunc(unsigned int level, const char *tag, const char *msg, void *) {
    fmt::print(stderr, "OptiX Log: [{}] [{}]: {}\n", level, tag, msg);
}

void GlfwErrorLogFunc(int error, const char *desc) {
    fmt::print(stderr, "GLFW error: {}({})\n", error, desc);
}

unsigned int LoadShader(const std::string &path, GLenum type) {
    std::ifstream fin(path);
    const std::string code((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());

    GLuint id = glCreateShader(type);
    const char *p_code = code.c_str();
    glShaderSource(id, 1, &p_code, nullptr);
    glCompileShader(id);

    return id;
}

}

void GlfwWindowResizeFunc(GLFWwindow *window, int width, int height) {
    auto* app = static_cast<OptixApp *>(glfwGetWindowUserPointer(window));
    app->Resize(width, height);
}

void GlfwWindowCursorPosFunc(GLFWwindow *window, double x, double y) {
    auto *app = static_cast<OptixApp *>(glfwGetWindowUserPointer(window));
    uint8_t state = 0;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        state |= 1;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        state |= 2;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
        state |= 4;
    }
    app->OnMouse(x, y, state);
}

OptixApp::OptixApp(int width, int height, std::string_view &&title) :
        color_buffer_width_(width), color_buffer_height_(height) {
    glfwSetErrorCallback(GlfwErrorLogFunc);
    glfwInit();

    window_ = glfwCreateWindow(width, height, title.data(), nullptr, nullptr);
    if (window_ == nullptr) {
        fmt::print(stderr, "Failed to create glfw window\n");
        return;
    }
    glfwMakeContextCurrent(window_);

    gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));

    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, GlfwWindowResizeFunc);
    glfwSetCursorPosCallback(window_, GlfwWindowCursorPosFunc);

    InitGl();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
}

OptixApp::~OptixApp() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window_);
    glfwTerminate();
}

void OptixApp::InitGl() {
    glViewport(0, 0, color_buffer_width_, color_buffer_height_);

    glActiveTexture(GL_TEXTURE0);

    glGenTextures(1, &screen_tex_);
    glBindTexture(GL_TEXTURE_2D, screen_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, color_buffer_width_, color_buffer_height_,
        0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    const unsigned int screen_vs =
        LoadShader(fmt::format("{}common/shaders/screen.vert", PROJECT_SRC_DIR), GL_VERTEX_SHADER);
    const unsigned int screen_fs =
        LoadShader(fmt::format("{}common/shaders/screen.frag", PROJECT_SRC_DIR), GL_FRAGMENT_SHADER);
    screen_shader_ = glCreateProgram();
    glAttachShader(screen_shader_, screen_vs);
    glAttachShader(screen_shader_, screen_fs);
    glLinkProgram(screen_shader_);
    glDeleteSamplers(1, &screen_vs);
    glDeleteSamplers(1, &screen_fs);

    const int screen_tex_shader_loc = glGetUniformLocation(screen_shader_, "color_buffer_img");
    glUseProgram(screen_shader_);
    glUniform1i(screen_tex_shader_loc, 0);
}

bool OptixApp::Initialize() {
    if (!InitOptix()) {
        return false;
    }

    CreateContext();

    color_buffer_.Init(color_buffer_width_, color_buffer_height_, 4 * sizeof(float), GL_RGBA, GL_FLOAT);

    return true;
}

void OptixApp::MainLoop() {
    frame_counter_.Reset();

    bool imgui_frame_active = true;

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window_, 1);
        }

        frame_counter_.Tick();

        Update();

        Render();

        color_buffer_.UnpackTo(screen_tex_);
        glBindTexture(GL_TEXTURE_2D, screen_tex_);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        if (imgui_frame_active) {
            ImGui::Begin("Frame Stats", &imgui_frame_active);
            ImGui::Text("FPS: %.3f", frame_counter_.Fps());
            ImGui::Text("per frame: %.3f ms", frame_counter_.Mspf());
            ImGui::End();
        }
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window_);
    }
}

bool OptixApp::InitOptix() const {
    cudaFree(nullptr);

    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices == 0) {
        fmt::print(stderr, "Failed to find a cuda device\n");
        return false;
    }
    fmt::print("Find {} cuda device(s)\n", num_devices);

    OPTIX_CHECK(optixInit());

    return true;
}

void OptixApp::CreateContext() {
    const int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    fmt::print("Run on device '{}'\n", device_prop.name);

    CU_CHECK(cuCtxGetCurrent(&cuda_context_));

    OPTIX_CHECK(optixDeviceContextCreate(cuda_context_, nullptr, &optix_context_));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optix_context_, OptixLogFunc, nullptr, 4));
}

bool OptixApp::Resize(int width, int height) {
    if (width > 0 && height > 0 && (width != color_buffer_width_ || height != color_buffer_height_)) {
        color_buffer_width_ = width;
        color_buffer_height_ = height;

        color_buffer_.Resize(width, height, 4 * sizeof(float), GL_RGBA, GL_FLOAT);

        glViewport(0, 0, width, height);

        glDeleteTextures(1, &screen_tex_);
        glGenTextures(1, &screen_tex_);
        glBindTexture(GL_TEXTURE_2D, screen_tex_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        return true;
    }
    return false;
}
