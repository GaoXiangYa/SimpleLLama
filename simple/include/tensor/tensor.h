#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <vector>
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
  void SetDeviceType(const base::DeviceType&);
  bool Assign(std::shared_ptr<base::Buffer> buffer);
  base::DataType GetDataType() const;
  size_t GetDimsSize() const;
  size_t GetSize() const;
  int GetDim(int idx) const;
  void ToCuda(cudaStream_t stream);
};
}  // namespace tensor