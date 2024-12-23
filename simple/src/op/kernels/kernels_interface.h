#pragma once

#include <functional>
#include "base/base.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace kernel {
using AddKernel = std::function<void(
    const tensor::Tensor& input0, const tensor::Tensor& input1, const tensor::Tensor& output,
    void* stream)>;

using EmbeddingKernel = std::function<void(
    const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& output,
    int32_t vocab_size, void* stream)>;

AddKernel GetAddKernel(base::DeviceType device_type);

EmbeddingKernel GetEmbKernel(base::DeviceType device_type);
}  // namespace kernel