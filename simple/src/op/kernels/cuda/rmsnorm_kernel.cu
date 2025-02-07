#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_math.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cstdint>
#include "base/base.h"
#include "rmsnorm_kernel.cuh"
#include "tensor/tensor.h"
#include <cub/block/block_reduce.cuh>

namespace kernel {

template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(
    float* input, float* weight, float* output, int size, float eps) {
  const int tid = threadIdx.x;
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* input_pack = reinterpret_cast<float4*>(input);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 input_float4 = *(input_pack + i);
    sum += input_float4.x * input_float4.x;
    sum += input_float4.y * input_float4.y;
    sum += input_float4.z * input_float4.z;
    sum += input_float4.w * input_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += input[i] * input[i];
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* weight_pack = reinterpret_cast<float4*>(weight);
  float4* output_pack = reinterpret_cast<float4*>(output);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 input_float4 = *(input_pack + i);
    float4 weight_float4 = *(weight_pack + i);
    *(output_pack + i) = make_float4(input_float4.x * weight_float4.x, input_float4.y * weight_float4.y, input_float4.z * weight_float4.z, input_float4.w * weight_float4.w);
  }
  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    output[i] = weight[i] * input[i] * scale;
  }
}

void rmsnorm_kernel_cu(
    const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& output,
    void* stream) {
  CHECK(!input.IsEmpty());
  CHECK(!weight.IsEmpty());
  CHECK(!output.IsEmpty());

  CHECK(
      input.GetDeviceType() == base::DeviceType::CUDA &&
      weight.GetDeviceType() == base::DeviceType::CUDA &&
      output.GetDeviceType() == base::DeviceType::CUDA);

#ifdef QWEN2_SUPPORT
  const float eps = 1e-6f;
#endif
  const float eps = 1e-5f;
  const int32_t size = static_cast<int32_t>(input.GetSize());
  float* input_ptr = const_cast<float*>(input.Ptr<float>());
  float* weight_ptr = const_cast<float*>(weight.Ptr<float>());
  float* output_ptr = const_cast<float*>(output.Ptr<float>());
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32<128>
        <<<1, threads_num, 0, stream_>>>(input_ptr, weight_ptr, output_ptr, size, eps);
  } else {
    row_rmsnorm_f32<128><<<1, threads_num>>>(input_ptr, weight_ptr, output_ptr, size, eps);
  }
}
}  // namespace kernel