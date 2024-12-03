#pragma once

#include <cstddef>
#include <memory>
#include "allocator.h"
namespace base {

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  ~CPUDeviceAllocator() {}

  void* Allocate(std::size_t size) const override;

  void Release(void* ptr) const override;

  void Memcpy(
      const void* src, void* dest, std::size_t byte_size,
      MemcpyKind memcpy_kind = MemcpyKind::MemcpyCPUToCPU,
      void* stream = nullptr, bool need_sync = false) override;

  void MemsetZero(
      void* ptr, std::size_t byte_size, void* stream = nullptr,
      bool need_sync = false) override;
};

class CPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<CPUDeviceAllocator> GetInstance() {
    if (instance_ == nullptr) {
      instance_ = std::make_shared<CPUDeviceAllocator>();
    }
    return instance_;
  }
 private:
  static std::shared_ptr<CPUDeviceAllocator> instance_;
};

}  // namespace base