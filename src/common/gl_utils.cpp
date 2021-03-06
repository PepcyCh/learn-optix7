#include "gl_utils.h"

#include <fstream>

#include "glad/glad.h"
#include "fmt/core.h"

bool GlUtils::CheckExtension(std::string_view &&name) {
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (GLint i = 0; i < n; i++) {
        auto *extension = reinterpret_cast<const char *>(glGetStringi(GL_EXTENSIONS, i));
        if (!strncmp(name.data(), extension, name.length())) {
            return true;
        }
    }
    return false;
}


uint32_t GlUtils::LoadShader(const std::string &path, uint32_t type) {
    std::ifstream fin(path);
    const std::string code((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());

    GLuint id = glCreateShader(type);
    const char *p_code = code.c_str();
    glShaderSource(id, 1, &p_code, nullptr);
    glCompileShader(id);

    int ret;
    glGetShaderiv(id, GL_COMPILE_STATUS, &ret);
    if (!ret) {
        char log_info[512];
        glGetShaderInfoLog(id, 512, nullptr, log_info);
        std::string sh_str;
        if (type == GL_VERTEX_SHADER) {
            sh_str = "vertex";
        } else if (type == GL_FRAGMENT_SHADER) {
            sh_str = "fragment";
        }
        fmt::print(stderr, "CE on {} shader: {}\n", sh_str, log_info);
        glDeleteShader(id);
        id = 0;
    }

    return id;
}
