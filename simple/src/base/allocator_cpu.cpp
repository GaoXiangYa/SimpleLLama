#include "base/allocator_cpu.h"
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include "base/allocator.h"
#include "base/base.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define HAVE_POSIX_MEMALIGN
#endif

namespace base {

CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::CPU) {}

void* CPUDeviceAllocator::Allocate(std::size_t size) const {
  if (!size) {
    return nullptr;
  }
#ifdef HAVE_POSIX_MEMALIGN
  void* data = nullptr;
  const std::size_t alignment = (size >= static_cast<std::size_t>(1024))
                                    ? static_cast<std::size_t>(32)
                                    : static_cast<std::size_t>(16);
  int status = posix_memalign(
      (void**)&data, ((alignment >= sizeof(void*)) ? alignment : sizeof(void*)),
      byte_size);
  if (status != 0) {
    return nullptr;
  }
  return data;
#else
  void* data = malloc(size);
  return data;
#endif
}

void CPUDeviceAllocator::Release(void* ptr) const {
  if (ptr) {
    free(ptr);
  }
}

void CPUDeviceAllocator::Memcpy(const void* src, void* dest, std::size_t size, MemcpyKind memcpy_kind, void* stream, bool need_sync) {
  CHECK_NE(src, nullptr);
  CHECK_NE(dest, nullptr);
  if (size == 0) {
    return;
  }
  cudaStream_t cuda_stream = nullptr;
  if (stream) {
    cuda_stream = static_cast<CUstream_st*>(stream);
  }
  switch (memcpy_kind) {
    case MemcpyKind::MemcpyCPUToCPU:
      std::memcpy(dest, src, size);
      break;
    case MemcpyKind::MemcpyCPUToCUDA:
      if (!cuda_stream) {
        cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
      } else {
        cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice);
      }
      break;
    case MemcpyKind::MemcpyCUDAToCPU:
      if (!cuda_stream) {
        cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
      } else {
        cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost);
      }
      break;
    case MemcpyKind::MemcpyCUDAToCUDA:
      if (!cuda_stream) {
        cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);
      } else {
        cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToDevice);
      }
      break;
  }
  if (need_sync) {
    cudaDeviceSynchronize();
  }
}

void CPUDeviceAllocator::MemsetZero(void* ptr, std::size_t size, void* stream, bool need_sync) {
  CHECK(device_type_ != DeviceType::Unknown);
  if (device_type_ == DeviceType::CPU) {
    std::memset(ptr, 0, size);
  } else {
    if (stream) {
      cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
      cudaMemsetAsync(ptr, 0, size, cuda_stream);
    } else {
      cudaMemset(ptr, 0,  size);
    }
    if (need_sync) {
      cudaDeviceSynchronize();
    }
  }
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance_ = nullptr;
}  // namespace base