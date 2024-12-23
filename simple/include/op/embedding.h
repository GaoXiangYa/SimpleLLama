#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include "base/base.h"
#include "operator.h"
#include "tensor/tensor.h"

namespace op {
struct EmbeddingOutput {
  tensor::Tensor input_tokens;
  tensor::Tensor input_embeddings;
  tensor::Tensor input_token_num;
  explicit EmbeddingOutput(
      tensor::Tensor input_tokens, tensor::Tensor input_embeddings, tensor::Tensor input_token_num)
      : input_tokens(std::move(input_tokens)),
        input_embeddings(std::move(input_embeddings)),
        input_token_num(std::move(input_token_num)) {}
};

class EmbeddingOperator : public OperatorWithParams {
public:
  explicit EmbeddingOperator(base::DeviceType device_type, int32_t dim, int32_t seq_len, int32_t vocab_size);

  base::Status Check() const override;

  base::Status Forward() override;

private:
  int32_t dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t vocab_size_ = 0;
};
}  // namespace op