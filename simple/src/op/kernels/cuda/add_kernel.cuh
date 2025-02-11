
#pragma once
#include "tensor/tensor.h"

namespace kernel {

void add_kernel_cu(
    const tensor::Tensor& input0, const tensor::Tensor& input1, const tensor::Tensor& output,
    void* stream = nullptr);
}  // namespace kernel