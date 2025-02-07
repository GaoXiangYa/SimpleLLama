#pragma once

#include <cstdint>
#include "base/base.h"
#include "operator.h"

namespace op {

class RmsNormOperator : public OperatorWithParams {
 public:
  explicit RmsNormOperator(base::DeviceType device_type, int32_t dim);

  base::Status Check() const override;

  base::Status Forward() override;
 private:
  int32_t dim_{0};
};

}  // namespace op