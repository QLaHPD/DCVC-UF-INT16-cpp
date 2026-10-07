// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

class GPUInfo {
public:
    GPUInfo()
    {
        if (cudaGetDevice(&device) != cudaSuccess) {
            return;
        }

        cudaDeviceProp deviceProp{};
        if (cudaGetDeviceProperties(&deviceProp, device) != cudaSuccess) return;
        computeCapability = deviceProp.major * 10 + deviceProp.minor;
    }
    ~GPUInfo() {}

public:
    int computeCapability{ 0 };
    int device{ -1 };
};

inline GPUInfo& get_gpu_info()
{
    // The outer runtime selects the CUDA device before invoking a proxy.
    // Refresh the cache when a worker changes devices.
    thread_local GPUInfo info;
    int device = -1;
    if (cudaGetDevice(&device) == cudaSuccess && info.device != device) info = GPUInfo{};
    return info;
}

inline int get_gpu_sm()
{
    return get_gpu_info().computeCapability;
}
