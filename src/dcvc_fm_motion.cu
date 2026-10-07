// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <limits>

#include "dcvc_fm_motion_common.hpp"
#include <vector>

namespace uf::fm::detail {
inline __device__ float __tofloat(float a) { return a; }

inline __device__ float __tofloat(__half a) { return __half2float(a); }

inline __device__ float __multiply_add(float a, float b, float c) {
  return __fmaf_rn(a, b, c);
}

inline __device__ __half __multiply_add(__half a, __half b, __half c) {
  return __hfma(a, b, c);
}

template <typename T>
__global__ void block_mc_forward_kernel(GPUTensor<T> out, const GPUTensor<T> im,
                                        const GPUTensor<T> flow, const int B,
                                        const int C, const int H, const int W) {
  const int b = blockIdx.z;
  const int h = blockIdx.y * blockDim.y + threadIdx.y;
  const int w = blockIdx.x * blockDim.x + threadIdx.x;

  if (h < H && w < W) {
    const T x_off = flow.ptr[b * flow.stride[0] + 0 * flow.stride[1] +
                             h * flow.stride[2] + w * flow.stride[3]];
    const T y_off = flow.ptr[b * flow.stride[0] + 1 * flow.stride[1] +
                             h * flow.stride[2] + w * flow.stride[3]];
    float x_pos = __tofloat(x_off) + static_cast<float>(w);
    float y_pos = __tofloat(y_off) + static_cast<float>(h);
    x_pos = min(max(x_pos, 0.f), static_cast<float>(W - 1));
    y_pos = min(max(y_pos, 0.f), static_cast<float>(H - 1));
    int x0 = __float2int_rd(x_pos);
    int x1 = min(x0 + 1, W - 1);
    int y0 = __float2int_rd(y_pos);
    int y1 = min(y0 + 1, H - 1);

    float w_r = x_pos - static_cast<float>(x0);
    float w_l = 1.f - w_r;
    float w_b = y_pos - static_cast<float>(y0);
    float w_t = 1.f - w_b;

    const T wa = __totype<T>(w_l * w_t);
    const T wb = __totype<T>(w_l * w_b);
    const T wc = __totype<T>(w_r * w_t);
    const T wd = __totype<T>(w_r * w_b);

    for (int c = 0; c < C; c++) {
      const int baseOffset = b * im.stride[0] + c * im.stride[1];

      T r = __totype<T>(0.f);
      const T ima = im.ptr[baseOffset + y0 * im.stride[2] + x0 * im.stride[3]];
      r = __multiply_add(ima, wa, r);
      const T imb = im.ptr[baseOffset + y1 * im.stride[2] + x0 * im.stride[3]];
      r = __multiply_add(imb, wb, r);
      const T imc = im.ptr[baseOffset + y0 * im.stride[2] + x1 * im.stride[3]];
      r = __multiply_add(imc, wc, r);
      const T imd = im.ptr[baseOffset + y1 * im.stride[2] + x1 * im.stride[3]];
      r = __multiply_add(imd, wd, r);
      out.ptr[b * out.stride[0] + c * out.stride[1] + h * out.stride[2] +
              w * out.stride[3]] = r;
    }
  }
}

void block_mc_forward_cuda(torch::Tensor &out, const torch::Tensor &im,
                           const torch::Tensor &flow, const int B, const int C,
                           const int H, const int W) {
  TORCH_CHECK(im.is_cuda() && flow.is_cuda() && out.is_cuda(), "FM custom motion requires CUDA tensors");
  TORCH_CHECK(im.device() == flow.device() && im.device() == out.device(), "FM motion devices differ");
  TORCH_CHECK(im.scalar_type() == flow.scalar_type() && im.scalar_type() == out.scalar_type(), "FM motion dtypes differ");
  TORCH_CHECK(im.scalar_type() == torch::kFloat32 || im.scalar_type() == torch::kFloat16, "FM custom motion supports float32/float16");
  TORCH_CHECK(B > 0 && C > 0 && H > 0 && W > 0 && B <= 65535, "Invalid FM motion dimensions");
  TORCH_CHECK(im.dim() == 4 && out.sizes() == im.sizes() && flow.dim() == 4 && flow.size(0) == B && flow.size(1) == 2 &&
              flow.size(2) == H && flow.size(3) == W, "Invalid FM flow shape");
  for (const auto& tensor : {im, flow, out}) {
    std::int64_t maximum_offset = 0;
    for (int d = 0; d < 4; ++d) {
      TORCH_CHECK(tensor.stride(d) >= 0 && tensor.stride(d) <= std::numeric_limits<int>::max(), "FM motion stride exceeds int32");
      const auto span = (tensor.size(d) - 1) * tensor.stride(d);
      TORCH_CHECK(span <= std::numeric_limits<int>::max() - maximum_offset, "FM motion offset exceeds int32");
      maximum_offset += span;
    }
  }
  const c10::cuda::CUDAGuard guard(im.device());
  // Preserve the original kernel's default-stream launch and explicit FMA order.
  const int BLOCK_SIZE = 32;
  const dim3 gridDim((W + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (H + BLOCK_SIZE - 1) / BLOCK_SIZE, B);
  const dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  if (im.element_size() == 4) {
    block_mc_forward_kernel<float>
        <<<gridDim, blockDim>>>(out, im, flow, B, C, H, W);
  } else if (im.element_size() == 2) {
    block_mc_forward_kernel<__half>
        <<<gridDim, blockDim>>>(out, im, flow, B, C, H, W);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace uf::fm::detail
