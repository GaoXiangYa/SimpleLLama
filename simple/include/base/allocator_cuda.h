#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <vector>
#include "allocator.h"

namespace base {

struct CudaMemoryBuffer {
  void* data;
  size_t byte_size;
  bool busy;
};

class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();

  void* Allocate(std::size_t size) const override;

  void Release(void* ptr) const override;

  void Memcpy(
      const void* src, void* dest, std::size_t byte_size,
      MemcpyKind memcpy_kind = MemcpyKind::MemcpyCPUToCPU,
      void* stream = nullptr, bool need_sync = false) override;
  void MemsetZero(
      void* ptr, std::size_t byte_size, void* stream = nullptr,
      bool need_sync = false) override;
 private:
  mutable std::map<int, size_t> no_busy_cnt_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};

class CUDADeviceAllocatorFactory {
 public:
  static std::shared_ptr<CUDADeviceAllocator> GetInstance() {
    if (instance_ == nullptr) {
      instance_ = std::make_shared<CUDADeviceAllocator>();
    }
    return instance_;
  }
 private:
  static std::shared_ptr<CUDADeviceAllocator> instance_;
};

}  // namespace base