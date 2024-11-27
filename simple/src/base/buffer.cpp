#include "base/buffer.h"
#include <glog/logging.h>
#include <algorithm>
#include <cstddef>
#include <memory>
#include "base/allocator.h"
#include "base/base.h"

namespace base {

Buffer::Buffer(
    std::size_t byte_size, std::shared_ptr<DeviceAllocator> allocator,
    void* ptr, bool use_external)
    : byte_size_(byte_size),
      ptr_(ptr),
      use_external_(use_external),
      allocator_(allocator) {
  if (!ptr_ && allocator_) {
    device_type_ = allocator_->GetDeviceType();
    use_external_ = false;
    ptr_ = allocator_->Allocate(byte_size_);
  }
}

Buffer::~Buffer() {
  if (!use_external_) {
    if (ptr_ && allocator_) {
      allocator_->Release(ptr_);
      ptr_ = nullptr;
    }
  }
}

void* Buffer::Ptr() {
  return ptr_;
}

const void* Buffer::Ptr() const {
  return ptr_;
}

std::size_t Buffer::Size() const {
  return byte_size_;
}

bool Buffer::Allocate() {
  if (allocator_ == nullptr || byte_size_ == 0) {
    return false;
  }
  use_external_ = false;
  ptr_ = allocator_->Allocate(byte_size_);
  return ptr_ != nullptr;
}

std::shared_ptr<DeviceAllocator> Buffer::Allocator() const {
  return allocator_;
}

void Buffer::CopyFrom(const Buffer& buffer) const {
  CHECK(allocator_ != nullptr);
  CHECK(buffer.Ptr() != nullptr);

  std::size_t byte_size = std::min(byte_size_, buffer.Size());
  auto const& buffer_device = buffer.GetDeviceType();
  auto const& current_device = this->GetDeviceType();
  CHECK(
      buffer_device != DeviceType::Unknown &&
      current_device != DeviceType::Unknown);

  if (buffer_device == DeviceType::CPU && current_device == DeviceType::CPU) {
    return allocator_->Memcpy(buffer.Ptr(), this->ptr_, byte_size);
  } else if (
      buffer_device == DeviceType::CUDA && current_device == DeviceType::CPU) {
    return allocator_->Memcpy(
        buffer.Ptr(), this->ptr_, byte_size, MemcpyKind::MemcpyCUDAToCPU);
  } else if (
      buffer_device == DeviceType::CPU && current_device == DeviceType::CUDA) {
    return allocator_->Memcpy(
        buffer.Ptr(), this->ptr_, byte_size, MemcpyKind::MemcpyCPUToCUDA);
  } else {
    return allocator_->Memcpy(
        buffer.Ptr(), this->ptr_, byte_size, MemcpyKind::MemcpyCUDAToCUDA);
  }
}

void Buffer::CopyFrom(const Buffer* buffer) const {
  CHECK(allocator_ != nullptr);
  CHECK(buffer->Ptr() != nullptr);

  std::size_t byte_size = std::min(byte_size_, buffer->Size());
  auto const& buffer_device = buffer->GetDeviceType();
  auto const& current_device = this->GetDeviceType();
  CHECK(
      buffer_device != DeviceType::Unknown &&
      current_device != DeviceType::Unknown);

  if (buffer_device == DeviceType::CPU && current_device == DeviceType::CPU) {
    return allocator_->Memcpy(buffer->Ptr(), this->ptr_, byte_size);
  } else if (
      buffer_device == DeviceType::CUDA && current_device == DeviceType::CPU) {
    return allocator_->Memcpy(
        buffer->Ptr(), this->ptr_, byte_size, MemcpyKind::MemcpyCUDAToCPU);
  } else if (
      buffer_device == DeviceType::CPU && current_device == DeviceType::CUDA) {
    return allocator_->Memcpy(
        buffer->Ptr(), this->ptr_, byte_size, MemcpyKind::MemcpyCPUToCUDA);
  } else {
    return allocator_->Memcpy(
        buffer->Ptr(), this->ptr_, byte_size, MemcpyKind::MemcpyCUDAToCUDA);
  }
}

DeviceType Buffer::GetDeviceType() const {
  return device_type_;
}

void Buffer::SetDeviceType(DeviceType device_type) {
  device_type_ = device_type;
}

std::shared_ptr<Buffer> Buffer::GetSharedFromThis() {
  return shared_from_this();
}

bool Buffer::IsExternal() const {
  return this->use_external_;
}
}  // namespace base