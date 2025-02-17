#include <cstdint>
#include "base/base.h"
#include "embedding_kernel.cuh"
#include "tensor/tensor.h"

namespace kernel {

static __global__ void embedding_f32(
    int* input, float* weight, float* output, int vocab_size, int32_t token_num,
    int32_t weight_dim) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }
  int32_t token = input[token_idx];
  if (token >= vocab_size) {
    return;
  }
  float* output_start = output + token_idx * weight_dim;
  const float* weight_start = weight + token * weight_dim;
  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_start[i] = weight_start[i];
  }
}

void emb_kernel_cu(
    const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& output,
    int32_t vocab_size, void* stream) {
  CHECK(!input.IsEmpty());
  CHECK(!weight.IsEmpty());
  CHECK(!output.IsEmpty());
  tensor::Tensor input_cu;
  if (input.GetDeviceType() != base::DeviceType::CUDA) {
    input_cu = input.Clone();
    input_cu.CopyToCuda();
  }
  const int32_t input_num = static_cast<int32_t>(input.GetSize());
  const int32_t weight_num = weight.GetDim(1);
  CHECK(weight.GetDeviceType() == output.GetDeviceType());
  CHECK(output.GetDeviceType() == base::DeviceType::CUDA);

  constexpr int32_t max_seq_len = 512;
  constexpr int32_t thread_num = 128;
  int32_t* input_ptr = input_cu.Ptr<int32_t>();
  float* weight_ptr = const_cast<float*>(weight.Ptr<float>());
  float* output_ptr = const_cast<float*>(output.Ptr<float>());
  if (stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    embedding_f32<<<max_seq_len, thread_num, 0, cuda_stream>>>(
        input_ptr, weight_ptr, output_ptr, vocab_size, input_num, weight_num);
  } else {
    embedding_f32<<<max_seq_len, thread_num>>>(
        input_ptr, weight_ptr, output_ptr, vocab_size, input_num, weight_num);
  }
}
}  // namespace kernel