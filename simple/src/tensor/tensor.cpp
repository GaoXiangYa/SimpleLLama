#include "tensor/tensor.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>
#include "base/allocator.h"
#include "base/allocator_cpu.h"
#include "base/allocator_cuda.h"
#include "base/base.h"
#include "base/buffer.h"

namespace tensor {

template <typename T, typename Tp>
static size_t ReduceDimension(T begin, T end, Tp init) {
  if (begin >= end) {
    return 0;
  }
  auto ret = std::accumulate(begin, end, init, std::multiplies{});
  return ret;
}

static size_t GetDataTypeSize(base::DataType data_type) {
  switch (data_type) {
    case base::DataType::Fp32:
      return 4;
    case base::DataType::Int8:
      return 1;
    case base::DataType::Int32:
      return 4;
    default:
      LOG(FATAL) << "Unknown data type size for " << int(data_type);
      return 0;
  }
  return 0;
}

Tensor::Tensor(
    base::DataType data_type, int32_t dim0, bool need_alloc,
    std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  size_ = dim0;
  if (need_alloc && alloc) {
    Allocate(alloc);
  } else {
    if (ptr != nullptr) {
      CHECK(need_alloc == false)
          << "The need_alloc is true when ptr parameter is not a null pointer.";
      InitBuffer(alloc, data_type_, need_alloc, ptr);
    }
  }
}

Tensor::Tensor(
    base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
    std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  size_ = dim0 * dim1;
  if (need_alloc && alloc) {
    Allocate(alloc);
  } else {
    InitBuffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(
    base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc,
    std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  size_ = dim0 * dim1 * dim2;
  if (need_alloc && alloc) {
    Allocate(alloc);
  } else {
    InitBuffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(
    base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
    bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  dims_.push_back(dim3);
  size_ = dim0 * dim1 * dim2 * dim3;
  if (need_alloc && alloc) {
    Allocate(alloc);
  } else {
    InitBuffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(
    base::DataType data_type, std::vector<int32_t> dims, bool need_alloc,
    std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : dims_(std::move(dims)), data_type_(data_type) {
  size_ = ReduceDimension(dims_.begin(), dims_.end(), 1);
  if (need_alloc && alloc) {
    Allocate(alloc);
  } else {
    InitBuffer(alloc, data_type_, need_alloc, ptr);
  }
}

void Tensor::CopyToCuda(cudaStream_t stream) {
  CHECK_NE(buffer_, nullptr);
  auto const device_type = this->GetDeviceType();
  if (device_type == base::DeviceType::Unknown) {
    LOG(ERROR) << "The device type of the tensor is unknown.";
  } else if (device_type == base::DeviceType::CPU) {
    auto byte_size = this->GetByteSize();
    auto cuda_alloc = base::CUDADeviceAllocatorFactory::GetInstance();
    auto cuda_buffer = std::make_shared<base::Buffer>(byte_size, cuda_alloc);
    cuda_alloc->Memcpy(
        buffer_->Ptr(), cuda_buffer->Ptr(), byte_size, base::MemcpyKind::MemcpyCPUToCUDA, stream);
    this->buffer_ = cuda_buffer;
  } else {
    LOG(INFO) << "The device type of the tensor is already in CUDA!.";
  }
}

void Tensor::CopyToCpu() {
  CHECK_NE(buffer_, nullptr);
  auto const device_type = this->GetDeviceType();
  if (device_type == base::DeviceType::Unknown) {
    LOG(ERROR) << "The device type of the tensor is unknown.";
  } else if (device_type == base::DeviceType::CUDA) {
    auto byte_size = this->GetByteSize();
    auto cpu_alloc = base::CPUDeviceAllocatorFactory::GetInstance();
    auto cpu_buffer = std::make_shared<base::Buffer>(byte_size, cpu_alloc);
    cpu_alloc->Memcpy(
        buffer_->Ptr(), cpu_buffer->Ptr(), byte_size, base::MemcpyKind::MemcpyCUDAToCPU);
    this->buffer_ = cpu_buffer;
  } else {
    LOG(INFO) << "The device type of the tensor is already in CPU!.";
  }
}

size_t Tensor::GetSize() const {
  return this->size_;
}

int32_t Tensor::GetDim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, this->dims_.size());
  return this->dims_.at(idx);
}

base::DeviceType Tensor::GetDeviceType() const {
  if (!buffer_) {
    return base::DeviceType::Unknown;
  }
  return buffer_->GetDeviceType();
}

bool Tensor::Assign(std::shared_ptr<base::Buffer> buffer) {
  if (!buffer) {
    LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
    return false;
  }
  if (buffer_) {
    if (buffer_->GetDeviceType() != buffer->GetDeviceType()) {
      LOG(ERROR) << "The device type of the new buffer is different from the original one.";
    }
  }
  auto byte_size = this->GetByteSize();
  if (byte_size > buffer->GetByteSize()) {
    LOG(ERROR) << "The size of buffer is too small for the tensor!";
    return false;
  }
  buffer_ = buffer;
  return true;
}

bool Tensor::Allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {
  if (!allocator) {
    LOG(ERROR) << "The allocator parameter in the allocate function is null pointer!";
    return false;
  }
  auto byte_size = this->GetByteSize();
  if (byte_size == 0) {
    LOG(ERROR) << "The byte_size in the allocate function is equal to zero!";
    return false;
  }
  if (buffer_ && byte_size <= buffer_->GetByteSize()) {
    if (!need_realloc) {
      return true;
    }
  }
  buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
  if (buffer_->Ptr() == nullptr) {
    LOG(ERROR) << "The memory allocated is a null pointer!";
    return false;
  }
  return true;
}

const std::vector<int32_t>& Tensor::GetDims() const {
  return this->dims_;
}

void Tensor::SetDeviceType(const base::DeviceType& device_type) const {
  if (buffer_) {
    buffer_->SetDeviceType(device_type);
  }
}

void Tensor::Reset(base::DataType data_type, const std::vector<int32_t>& dims) {
  this->data_type_ = data_type;
  this->dims_ = dims;
  this->size_ = ReduceDimension(dims.begin(), dims.end(), 1);
  this->buffer_ = nullptr;
}

int32_t Tensor::GetDimsSize() const {
  return static_cast<int32_t>(dims_.size());
}

base::DataType Tensor::GetDataType() const {
  return data_type_;
}

void Tensor::Reshape(const std::vector<int32_t>& dims) {
  auto size = ReduceDimension(dims.begin(), dims.end(), 1);
  if (!buffer_) {
    this->dims_ = dims;
    this->size_ = size;
    return;
  }
  // reshape large size
  if (size > size_) {
    auto new_buffer = std::make_shared<base::Buffer>(
        size * base::GetDataTypeSize(this->data_type_), buffer_->Allocator());
    CHECK(new_buffer->Allocate());
    new_buffer->CopyFrom(buffer_.get());
    this->buffer_ = new_buffer;
  }
  this->dims_ = dims;
  this->size_ = size;
}

std::shared_ptr<base::Buffer> Tensor::GetBuffer() const {
  return buffer_;
}

Tensor Tensor::Clone() const {
  auto new_tensor = *this;
  auto byte_size = this->GetByteSize();
  auto allocator = buffer_->Allocator();
  new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);
  new_tensor.buffer_->CopyFrom(buffer_.get());
  return new_tensor;
}

size_t Tensor::GetByteSize() const {
  return this->GetSize() * tensor::GetDataTypeSize(data_type_);
}

std::vector<size_t> Tensor::GetStrides() const {
  std::vector<size_t> strides;
  if (!dims_.empty()) {
    int dims_size = dims_.size();
    for (int i = 0; i < dims_size; ++i) {
      auto stride = ReduceDimension(dims_.begin() + i + 1, dims_.end(), 1);
      strides.push_back(stride);
    }
    strides.push_back(1);
  }
  return strides;
}

bool Tensor::IsEmpty() const {
  return size_ == 0 || buffer_ == nullptr || buffer_->Ptr() == nullptr;
}

void Tensor::InitBuffer(
    std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type, bool need_alloc,
    void* ptr) {
  if (!alloc && !need_alloc) {
    auto buffer = std::make_shared<base::Buffer>(
        tensor::GetDataTypeSize(data_type) * size_, nullptr, ptr, true);
    this->buffer_ = buffer;
  } else {
    Allocate(alloc, true);
  }
}
}  // namespace tensor