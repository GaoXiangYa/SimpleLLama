#pragma once

#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <glog/types.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "base/allocator.h"
#include "base/base.h"
#include "base/buffer.h"
namespace tensor {
class Tensor {
 public:
  explicit Tensor() = default;

  explicit Tensor(
      base::DataType data_type, int32_t dim0, bool need_alloc = false,
      std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

  explicit Tensor(
      base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
      std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

  explicit Tensor(
      base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc = false,
      std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

  explicit Tensor(
      base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
      bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
      void* ptr = nullptr);

  explicit Tensor(
      base::DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
      std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

  bool IsEmpty() const;

  base::DeviceType GetDeviceType() const;

  void SetDeviceType(const base::DeviceType&) const;

  bool Assign(std::shared_ptr<base::Buffer> buffer);

  base::DataType GetDataType() const;

  int32_t GetDimsSize() const;

  size_t GetSize() const;

  size_t GetByteSize() const;

  int GetDim(int idx) const;

  const std::vector<int32_t>& GetDims() const;

  std::vector<size_t> GetStrides() const;

  void Reset(base::DataType data_type, const std::vector<int32_t>& dims);

  bool Allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_alloc = false);

  // Copy data from cpu to cuda
  void CopyToCuda(cudaStream_t stream = nullptr);

    // Copy data from cpu
  void CopyToCpu();

  void InitBuffer(
      std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type, bool need_alloc,
      void* ptr);

  void Reshape(const std::vector<int32_t>& dims);

  std::shared_ptr<base::Buffer> GetBuffer() const;

  template <typename T>
  T* Ptr();

  template <typename T>
  const T* Ptr() const;

  template <typename T>
  T* Ptr(int64_t index);

  template <typename T>
  const T* Ptr(int64_t index) const;

  template <typename T>
  T& Index(int64_t offset);

  template <typename T>
  const T& Index(int64_t offset) const;

  tensor::Tensor Clone() const;
 private:
  size_t size_ = 0;
  std::vector<int32_t> dims_;
  std::shared_ptr<base::Buffer> buffer_;
  base::DataType data_type_ = base::DataType::Unknown;
};

template <typename T>
T& Tensor::Index(int64_t offset) {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->GetSize());
  const T& val = *(reinterpret_cast<T*>(buffer_->Ptr()) + offset);
  return val;
}

template <typename T>
const T& Tensor::Index(int64_t offset) const {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->GetSize());
  const T& val = *(reinterpret_cast<T*>(buffer_->Ptr()) + offset);
  return val;
}

template <typename T>
const T* Tensor::Ptr() const {
  if (!buffer_) {
    return nullptr;
  }
  return const_cast<const T*>(reinterpret_cast<T*>(buffer_->Ptr()));
}

template <typename T>
T* Tensor::Ptr() {
  if (!buffer_) {
    return nullptr;
  }
  return reinterpret_cast<T*>(buffer_->Ptr());
}

template <typename T>
T* Tensor::Ptr(int64_t index) {
  CHECK(buffer_ != nullptr && buffer_->Ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return const_cast<T*>(reinterpret_cast<const T*>(buffer_->Ptr())) + index;
}

template <typename T>
const T* Tensor::Ptr(int64_t index) const {
  CHECK(buffer_ != nullptr && buffer_->Ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return reinterpret_cast<const T*>(buffer_->Ptr()) + index;
}

}  // namespace tensor