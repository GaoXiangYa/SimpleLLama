#include "sampler/argmax_sampler.h"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include "../op/kernels/cuda/argmax_kernel.cuh"
#include "base/base.h"

namespace sampler {
size_t ArgmaxSampler::Sample(const float* logits, size_t size, void* stream) {
  if (device_type_ == base::DeviceType::CPU) {
    size_t next = std::distance(logits, std::max_element(logits, logits + size));
    return next;
  }
  size_t next = kernel::argmax_kernel_cu(logits, size, stream);
  return next;
}
}  // namespace sampler