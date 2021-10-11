#pragma once

#include <string>
#include <string_view>

class GlUtils {
public:
    static bool CheckExtension(std::string_view &&name);

    static uint32_t LoadShader(const std::string &path, uint32_t type);
};