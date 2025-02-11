#include "kernels_interface.h"
#include "base/base.h"
#include "cuda/add_kernel.cuh"
#include "cuda/rmsnorm_kernel.cuh"

namespace kernel {
AddKernel GetAddKernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::CUDA) {
    return add_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a add kernel.";
    return nullptr;
  }
}

RmsNormKernel GetRmsNormKernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::CUDA) {
    return rmsnorm_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a add kernel.";
    return nullptr;
  }
}
}  // namespace kernel