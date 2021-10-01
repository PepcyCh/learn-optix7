#pragma once

#define OPTIX_RAYGEN(name) extern "C" __global__ void __raygen__##name
#define OPTIX_MISS(name) extern "C" __global__ void __miss__##name
#define OPTIX_CLOSESTHIT(name) extern "C" __global__ void __closesthit__##name
#define OPTIX_ANYHIT(name) extern "C" __global__ void __anyhit__##name