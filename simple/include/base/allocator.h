#pragma once

#include <cstddef>
#include <memory>
#include "base.h"

namespace base {

enum class MemcpyKind {
  MemcpyCPUToCPU = 0,
  MemcpyCPUToCUDA = 1,
  MemcpyCUDAToCPU = 2,
  MemcpyCUDAToCUDA = 3,
};

class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType device_type)
      : device_type_(device_type) {}

  virtual DeviceType GetDeviceType() const { return device_type_; }

  // 释放内存
  virtual void Release(void* ptr) const = 0;

  // 申请内存或者显存
  virtual void* Allocate(std::size_t byte_size) const = 0;

  virtual void Memcpy(
      const void* src, void* dest, std::size_t byte_size,
      MemcpyKind memcpy_kind = MemcpyKind::MemcpyCPUToCPU,
      void* stream = nullptr, bool need_sync = false) = 0;

  virtual void MemsetZero(
      void* ptr, std::size_t byte_size, void* stream = nullptr,
      bool need_sync = false) = 0;
  protected:
  DeviceType device_type_{DeviceType::Unknown};
};

}  // namespace base