#pragma once

#include "base/base.h"
#include "operator.h"

namespace op {

class VecAddOperator : public OperatorWithOutParams {
 public:
  explicit VecAddOperator(base::DeviceType device_type);
  base::Status Check() const override;
  base::Status Forward() override;
};

}  // namespace op