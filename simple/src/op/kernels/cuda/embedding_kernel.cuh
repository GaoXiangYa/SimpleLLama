#pragma once

#include <cstdint>
#include "tensor/tensor.h"

namespace kernel {
void emb_kernel_cu(
    const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& output,
    int32_t vocab_size, void* stream = nullptr);
} // namespace kernel