#include "op/embedding.h"
#include <cstdint>
#include "base/base.h"
#include "kernels/kernels_interface.h"
#include "op/operator.h"
#include "tensor/tensor.h"

namespace op {
EmbeddingOperator::EmbeddingOperator(
    base::DeviceType device_type, int32_t dim, int32_t seq_len, int32_t vocab_size)
    : OperatorWithParams(device_type, OperatorType::Embedding, false, "Embedding"),
      dim_(dim),
      seq_len_(seq_len),
      vocab_size_(vocab_size) {
  ResetWeightSize(1);
  ResetInputSize(2);
  ResetOutputSize(1);
}

base::Status EmbeddingOperator::Check() const {
  auto const& input_tensor = GetInput(0);
  auto const& token_size = GetInput(1).GetSize();
  if (token_size > input_tensor.GetSize()) {
    return base::error::InvalidArgument("The number of input tensor is greater than seq len");
  }
  base::Status status =
      CheckTensorWithDim(input_tensor, device_type_, base::DataType::Int32, token_size);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the embedding operator.";
    return status;
  }
  status = CheckTensorWithDim(GetWeight(0), device_type_, data_type_, vocab_size_, dim_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the embedding operator.";
    return status;
  }
  status = CheckTensorWithDim(GetOutput(0), device_type_, data_type_, token_size, dim_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the embedding operator.";
    return status;
  }
  return base::error::Success();
}

base::Status EmbeddingOperator::Forward() {
  base::Status status = Check();
  if (!status) {
    return status;
  }
  if (device_type_ == base::DeviceType::CUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::GetEmbKernel(device_type_)(
      GetInput(0), GetWeight(0), GetOutput(0), vocab_size_,
      cuda_config_ ? cuda_config_->stream : nullptr);
  return base::kSuccess;
}
}  // namespace op