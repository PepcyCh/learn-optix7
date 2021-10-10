#pragma once

#include "pcmath/pcmath.h"

namespace pcm = pep::cuda_math;

struct LightData {
    pcm::IVec3 index;
    pcm::Vec3 strength;
    int at_another_index;
    float at_probability;
};