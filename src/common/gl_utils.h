#pragma once

#include <string>

class GlUtils {
public:
    static uint32_t LoadShader(const std::string &path, uint32_t type);
};