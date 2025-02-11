#include <cstdint>
#include "add_kernel.cuh"
#include "base/base.h"
#include "tensor/tensor.h"

namespace kernel {

static __global__ void row_add_f32(float* input0, float* input1, float* output, int size) {
  // const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  // float val0 = input0[tid];
  // float val1 = input1[tid];
  // output[tid] = val0 + val1;
  // const int tid = threadIdx.x;
  // constexpr int pack_size = 4;
  // const int pack_num = size / pack_size;
  // const int pack_off = pack_size * pack_num;
  // auto input0_pack = reinterpret_cast<float4*>(input0);
  // auto input1_pack = reinterpret_cast<float4*>(input1);
  // auto output_pack = reinterpret_cast<float4*>(output);
  // for (int i = tid; i < pack_num; i += blockDim.x) {
  //   output_pack[i].x = input0_pack[i].x + input1_pack[i].x;
  //   output_pack[i].y = input0_pack[i].y + input1_pack[i].y;
  //   output_pack[i].z = input0_pack[i].z + input1_pack[i].z;
  //   output_pack[i].w = input0_pack[i].w + input1_pack[i].w;
  // }
  // for (int i = pack_off + tid; i < size; i += blockDim.x) {
  //   output[i] = input0[i] + input1[i];
  // }
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride_spatial = gridDim.x * blockDim.x;
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;
  auto input0_pack = reinterpret_cast<float4*>(input0);
  auto input1_pack = reinterpret_cast<float4*>(input1);
  auto output_pack = reinterpret_cast<float4*>(output);
  for (int i = tid; i < pack_num; i += stride_spatial) {
    output_pack[i] = make_float4(
        input0_pack[i].x + input1_pack[i].x, input0_pack[i].y + input1_pack[i].y,
        input0_pack[i].z + input1_pack[i].z, input0_pack[i].w + input1_pack[i].w);
  }
  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    output[i] = input0[i] + input1[i];
  }
}

void add_kernel_cu(
    const tensor::Tensor& input0, const tensor::Tensor& input1, const tensor::Tensor& output,
    void* stream) {
  CHECK(!input0.IsEmpty());
  CHECK(!input1.IsEmpty());
  CHECK(!output.IsEmpty());

  CHECK(
      input0.GetDeviceType() == base::DeviceType::CUDA &&
      input1.GetDeviceType() == base::DeviceType::CUDA &&
      output.GetDeviceType() == base::DeviceType::CUDA);
  const int32_t input0_size = static_cast<int32_t>(input0.GetSize());
  const int32_t input1_size = static_cast<int32_t>(input1.GetSize());
  const int32_t output_size = static_cast<int32_t>(output.GetSize());
  CHECK(input0_size == input1_size);
  CHECK(input0_size == output_size);
  auto input0_ptr = const_cast<float*>(input0.Ptr<float>());
  auto input1_ptr = const_cast<float*>(input1.Ptr<float>());
  auto output_ptr = const_cast<float*>(output.Ptr<float>());
  constexpr int threads_num = 512;
  const int grids_num = (input0_size + threads_num - 1) / threads_num;
  if (stream) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    row_add_f32<<<grids_num, threads_num, 0, cuda_stream>>>(
        input0_ptr, input1_ptr, output_ptr, input0_size);
  } else {
    row_add_f32<<<grids_num, threads_num>>>(input0_ptr, input1_ptr, output_ptr, input0_size);
  }
}
}  // namespace kernel