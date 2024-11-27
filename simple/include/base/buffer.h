#pragma once

#include <cstddef>
#include <memory>
#include "allocator.h"
#include "base.h"

namespace base {

class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
 public:
  explicit Buffer() = default;

  explicit Buffer(
      const std::size_t byte_size,
      std::shared_ptr<DeviceAllocator> allocator = nullptr, void* ptr = nullptr,
      bool use_external = false);

  virtual ~Buffer();

  bool Allocate();

  void CopyFrom(const Buffer& buffer) const;

  void CopyFrom(const Buffer* buffer) const;

  void* Ptr();

  const void* Ptr() const;

  std::size_t Size() const;

  std::shared_ptr<DeviceAllocator> Allocator() const;

  DeviceType GetDeviceType() const;

  void SetDeviceType(DeviceType device_type);

  std::shared_ptr<Buffer> GetSharedFromThis();

  bool IsExternal() const;
 private:
  std::size_t byte_size_ = 0;
  void* ptr_ = nullptr;
  bool use_external_ = false;
  DeviceType device_type_ = DeviceType::Unknown;
  std::shared_ptr<DeviceAllocator> allocator_;
};
}  // namespace base