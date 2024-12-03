#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace kernel {

struct CudaConfig {
  cudaStream_t steam = nullptr;
  ~CudaConfig() {
    if (steam) {
      cudaStreamDestroy(steam);
    }
  }
};

}  // namespace kernel