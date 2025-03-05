#pragma once

#include <cstddef>
#include "base/base.h"
#include "sampler.h"

namespace sampler {
class ArgmaxSampler : public Sampler {
public:
  explicit ArgmaxSampler(base::DeviceType device_type) :Sampler(device_type) {}
  size_t Sample(const float* logits, size_t size, void* stream) override;
};
}