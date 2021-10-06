#pragma once

#include "math/vec.h"

struct LightData {
    IVec3 index;
    Vec3 strength;
    int at_another_index;
    float at_probability;
};