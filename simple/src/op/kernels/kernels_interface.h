#pragma once

#include <functional>
#include "base/base.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace kernel {
using AddKernel = std::function<void(const tensor::Tensor& input0, const tensor::Tensor& input1, const tensor::Tensor& output, void* stream)>;


AddKernel GetAddKernel(base::DeviceType device_type);
}