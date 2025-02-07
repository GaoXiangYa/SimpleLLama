#include "op/rmsnorm.h"
#include <cstdint>
#include "base/base.h"
#include "kernels/kernels_interface.h"
#include "op/operator.h"
#include "tensor/tensor.h"

namespace op {

RmsNormOperator::RmsNormOperator(base::DeviceType device_type, int32_t dim)
    : OperatorWithParams(device_type, OperatorType::RMSNorm, false, "RMSNorm"), dim_(dim) {
  ResetInputSize(1);
  ResetInputSize(1);
  ResetWeightSize(1);
}

base::Status RmsNormOperator::Forward() {
  auto status = Check();
  if (!status) {
    return status;
  }
  auto input = this->GetInput(0);
  auto weight = this->GetWeight(0);
  auto output = this->GetOutput(0);
  if (device_type_ == base::DeviceType::CUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::GetRmsNormKernel(device_type_)(
      input, weight, output, cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

base::Status RmsNormOperator::Check() const {
  auto status = CheckTensorWithDim(GetInput(0), device_type_, data_type_, dim_);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the rmsnorm layer";
    return status;
  }
  status = CheckTensorWithDim(GetWeight(0), device_type_, data_type_, dim_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the rmsnorm layer";
    return status;
  }
  status = CheckTensorWithDim(GetOutput(0), device_type_, data_type_, dim_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the rmsnorm layer";
    return status;
  }
  return base::error::Success();
}
}  // namespace op