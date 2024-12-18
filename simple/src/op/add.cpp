#include "op/add.h"
#include "base/base.h"
#include "kernels/kernels_interface.h"
#include "op/operator.h"
#include "tensor/tensor.h"

namespace op {
VecAddOperator::VecAddOperator(base::DeviceType device_type)
    : OperatorWithOutParams(device_type, OperatorType::Add, "Add") {
  ResetInputSize(2);
  ResetOutputSize(1);
}

base::Status VecAddOperator::Check() const {
  auto input0 = this->GetInput(0);
  auto input1 = this->GetInput(1);
  int input0_size = input0.GetSize();
  base::Status status;
  status = CheckTensorWithDim(input0, device_type_, data_type_, input0_size);
  if (!status) {
    LOG(ERROR) << "The input tensor 0 error in the add layer.";
    return status;
  }
  status = CheckTensorWithDim(input1, device_type_, data_type_, input0_size);
  if (!status) {
    LOG(ERROR) << "The input tensor 1 error in the add layer.";
    return status;
  }
  status = CheckTensorWithDim(GetOutput(0), device_type_, data_type_, input0_size);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the add layer.";
    return status;
  }
  return base::error::Success();
}

base::Status VecAddOperator::Forward() {
  auto status = this->Check();
  if (!status) {
    return status;
  }
  auto input0 = this->GetInput(0);
  auto input1 = this->GetInput(1);
  auto output = this->GetOutput(0);
  if (device_type_ == base::DeviceType::CUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::GetAddKernel(device_type_)(
      input0, input1, output, cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}
}  // namespace op