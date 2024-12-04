#include "op/operator.h"
#include <glog/logging.h>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "base/base.h"
#include "base/buffer.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace op {
BaseOperator::BaseOperator(
    base::DeviceType device_type, OperatorType layer_type, base::DataType data_type,
    std::string layer_name)
    : layer_name_(layer_name),
      layer_type_(layer_type),
      data_type_(data_type),
      device_type_(device_type) {}

base::DataType BaseOperator::GetDataType() const {
  return data_type_;
}

OperatorType BaseOperator::GetLayerType() const {
  return layer_type_;
}

base::Status BaseOperator::SetWeight(
    [[maybe_unused]] int idx, [[maybe_unused]] const tensor::Tensor& weight) {
  return base::error::FunctionNotImplement();
}

base::Status BaseOperator::SetWeight(
    [[maybe_unused]] int idx, [[maybe_unused]] const std::vector<int>& dims,
    [[maybe_unused]] const void* weight_ptr, [[maybe_unused]] base::DeviceType device_type) {
  return base::error::FunctionNotImplement();
}

const std::string& BaseOperator::GetLayerName() const {
  return layer_name_;
}

void BaseOperator::SetLayerName(const std::string& layer_name) {
  layer_name_ = layer_name;
}

base::DeviceType BaseOperator::GetDeviceType() const {
  return device_type_;
}

void BaseOperator::SetDeviceType(base::DeviceType device_type) {
  device_type_ = device_type;
}

OperatorWithOutParams::OperatorWithOutParams(
    base::DeviceType device_type, OperatorType layer_type, std::string layer_name)
    : BaseOperator(device_type, layer_type, base::DataType::Fp32, layer_name) {}

base::Status OperatorWithOutParams::Init() {
  return base::error::Success();
}

base::Status OperatorWithOutParams::Forward() {
  return base::error::FunctionNotImplement("");
}

base::Status OperatorWithOutParams::CheckTensor(
    const tensor::Tensor& tensor, base::DeviceType device_type, base::DataType data_type) const {
  if (tensor.IsEmpty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.GetDeviceType() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.GetDataType() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong ddata type.");
  }
  return base::error::Success();
}

base::Status OperatorWithOutParams::CheckTensorWithDim(
    const tensor::Tensor& tensor, base::DeviceType device_type, base::DataType data_type,
    ...) const {
  std::va_list args;
  if (tensor.IsEmpty()) {
    return base::error::InvalidArgument("The tensor parameter is empty");
  }
  if (tensor.GetDeviceType() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type");
  }
  if (tensor.GetDataType() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }

  va_start(args, data_type);
  int dims = tensor.GetDimsSize();
  for (int i = 0; i < dims; ++i) {
    int dim = va_arg(args, int);
    if (dim != tensor.GetDim(i)) {
      return base::error::InvalidArgument("The tensor has a wrong dim in dim " + std::to_string(i));
    }
  }
  va_end(args);
  return base::error::Success();
}

void OperatorWithOutParams::SetInput(int idx, const tensor::Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  this->inputs_.at(idx) = input;
}

void OperatorWithOutParams::SetOutput(int idx, const tensor::Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  this->outputs_.at(idx) = output;
}

const tensor::Tensor& OperatorWithOutParams::GetInput(int idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

tensor::Tensor& OperatorWithOutParams::GetInput(int idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

tensor::Tensor& OperatorWithOutParams::GetOutput(int idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}
const tensor::Tensor& OperatorWithOutParams::GetOutput(int idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

base::Status OperatorWithOutParams::Check() const {
  return base::error::FunctionNotImplement("The check function is not implement yet");
}

void OperatorWithOutParams::ResetInputSize(size_t size) {
  inputs_.resize(size);
}

void OperatorWithOutParams::ResetOutputSize(size_t size) {
  outputs_.resize(size);
}

void OperatorWithOutParams::ToCuda() {
  for (auto& input : inputs_) {
    if (!input.IsEmpty()) {
      input.ToCuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
  for (auto& output : outputs_) {
    if (!output.IsEmpty()) {
      output.ToCuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
}

void OperatorWithOutParams::SetCudaConfig(std::shared_ptr<kernel::CudaConfig> config) {
  if (!config) return;
  this->cuda_config_ = config;
}

std::shared_ptr<kernel::CudaConfig> OperatorWithOutParams::GetCudaConfig() const {
  return cuda_config_;
}

size_t OperatorWithOutParams::GetInputSize() const {
  return inputs_.size();
}

size_t OperatorWithOutParams::GetOutputSize() const {
  return outputs_.size();
}

OperatorWithParams::OperatorWithParams(
    base::DeviceType device_type, OperatorType layer_type, bool is_quant_layer,
    std::string layer_name)
    : OperatorWithOutParams(device_type, layer_type, layer_name), is_quant_layer_(is_quant_layer) {}

base::Status OperatorWithParams::SetWeight(int idx, const tensor::Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK(weight.GetDataType() == base::DataType::Fp32);
  if (!weight.IsEmpty()) {
    CHECK(weight.GetDeviceType() == device_type_);
  }
  weights_.at(idx) = weight;
  return base::error::Success();
}

const tensor::Tensor& OperatorWithParams::GetWeight(int idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

void OperatorWithParams::ToCuda() {
  OperatorWithOutParams::ToCuda();
  for (auto& weight : weights_) {
    weight.ToCuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
  if (!scales_.IsEmpty()) {
    scales_.ToCuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
}

base::Status OperatorWithParams::SetWeight(
    int idx, const std::vector<int>& dims, const void* weight_ptr, base::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK_NE(weight_ptr, nullptr);

  auto size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies{});
  auto buffer = std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
  if (device_type != base::DeviceType::Unknown) {
    buffer->SetDeviceType(device_type);
  }

  if (!is_quant_layer_) {
    tensor::Tensor weight(base::DataType::Fp32, dims);
    weight.SetDeviceType(device_type);
    CHECK(weight.Assign(buffer));
    weights_.at(idx) = weight;
  } else {
    tensor::Tensor weight(base::DataType::Int8, dims);
    weight.SetDeviceType(device_type);
    CHECK(weight.Assign(buffer));
    weights_.at(idx) = weight;

    const int weight_size = static_cast<int>(weight.GetSize());
    CHECK(weight_size % group_size_ == 0);

    int scales_nums = weight_size / group_size_;
    scales_ = tensor::Tensor{
        base::DataType::Fp32, scales_nums, false, nullptr,
        reinterpret_cast<float*>((int8_t*)weight_ptr + weight_size)};
    scales_.SetDeviceType(device_type);
  }

  return base::error::Success();
}

void OperatorWithParams::SetScales(const tensor::Tensor& scales) {
  CHECK(!scales.IsEmpty());
  this->scales_ = scales;
}

void OperatorWithParams::SetGroupSize(int32_t group_size) {
  this->group_size_ = group_size;
}

int32_t OperatorWithParams::GetScaleNum() const {
  CHECK(!scales_.IsEmpty());
  return static_cast<int32_t>(scales_.GetSize());
}

void OperatorWithParams::ResetWeightSize(size_t size) {
  weights_.resize(size);
}

size_t OperatorWithParams::GetWeightSize() const {
  return weights_.size();
}

tensor::Tensor& OperatorWithParams::GetWeight(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

base::Status OperatorWithOutParams::Forward(
    const tensor::Tensor& input0, const tensor::Tensor& output0) {
  this->SetInput(0, input0);
  this->SetOutput(0, output0);
  return this->Forward();
}

base::Status OperatorWithOutParams::Forward(
    const tensor::Tensor& input0, const tensor::Tensor& input1, const tensor::Tensor& output0) {
  this->SetInput(0, input0);
  this->SetInput(1, input1);

  this->SetOutput(0, output0);
  return this->Forward();
}

base::Status OperatorWithOutParams::Forward(
    const tensor::Tensor& input0, const tensor::Tensor& input1, const tensor::Tensor& input2,
    const tensor::Tensor& output0) {
  this->SetInput(0, input0);
  this->SetInput(1, input1);
  this->SetInput(2, input2);

  this->SetOutput(0, output0);
  return this->Forward();
}

base::Status OperatorWithOutParams::Forward(
    const tensor::Tensor& input0, const tensor::Tensor& input1, const tensor::Tensor& input2,
    const tensor::Tensor& input3, const tensor::Tensor& output0) {
  this->SetInput(0, input0);
  this->SetInput(1, input1);
  this->SetInput(2, input2);
  this->SetInput(3, input3);

  this->SetOutput(0, output0);
  return this->Forward();
}

base::Status OperatorWithOutParams::Forward(
    const tensor::Tensor& input0, const tensor::Tensor& input1, const tensor::Tensor& input2,
    const tensor::Tensor& input3, const tensor::Tensor& input4, const tensor::Tensor& output0) {
  this->SetInput(0, input0);
  this->SetInput(1, input1);
  this->SetInput(2, input2);
  this->SetInput(3, input3);
  this->SetInput(4, input4);

  this->SetOutput(0, output0);
  return this->Forward();
}

}  // namespace op