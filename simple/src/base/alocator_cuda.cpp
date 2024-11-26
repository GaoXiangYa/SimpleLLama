#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>
#include "base/allocator.h"
#include "base/allocator_cuda.h"
#include "base/base.h"

namespace base {
CUDADeviceAllocator::CUDADeviceAllocator()
    : DeviceAllocator(DeviceType::CUDA) {}

void* CUDADeviceAllocator::Allocate(std::size_t size) const {
  int id = -1;
  cudaError_t state = cudaGetDevice(&id);
  CHECK(state == cudaSuccess);
  // 分配的内存如果大于1M直接在big_buffer当中分配
  if (size > 1024 * 1024) {
    auto& big_buffers = big_buffers_map_[id];
    int sel_id = -1;
    int big_buffers_size = big_buffers.size();
    for (int i = 0; i < big_buffers_size; ++i) {
      if (big_buffers[i].byte_size >= size && !big_buffers[i].busy &&
          big_buffers[i].byte_size - size < 1 * 1024 * 1024) {
        if (sel_id == -1 ||
            big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
          sel_id = i;
        }
      }
    }
    if (sel_id != -1) {
      big_buffers[sel_id].busy = true;
      return big_buffers[sel_id].data;
    }
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, size);
    if (state != cudaSuccess) {
      char buf[256];
      snprintf(
          buf, 256,
          "Error: CUDA error when allocating %lu MB memory! maybe there's no "
          "enough memory "
          "left on  device.",
          size >> 20);
      LOG(ERROR) << buf;
      return nullptr;
    }
    big_buffers.emplace_back(ptr, size, true);
    return ptr;
  }
  // 小内存在cuda_buffer当中分配
  auto& cuda_buffer = cuda_buffers_map_[id];
  for (auto& buffer : cuda_buffer) {
    if (buffer.byte_size >= size && !buffer.busy) {
      buffer.busy = true;
      no_busy_cnt_[id] -= buffer.byte_size;
      return buffer.data;
    }
  }
  // 如果都没有直接分配显存并存放在buffer当中
  void* ptr = nullptr;
  state = cudaMalloc(&ptr, size);
  if (state != cudaSuccess) {
    char buf[256];
    snprintf(
        buf, 256,
        "Error: CUDA error when allocating %lu MB memory! maybe there's no "
        "enough memory "
        "left on  device.",
        size >> 20);
    LOG(ERROR) << buf;
    return nullptr;
  }
  cuda_buffer.emplace_back(ptr, size, true);
  return ptr;
}

void CUDADeviceAllocator::Release(void* ptr) const {
  if (ptr == nullptr) {
    return;
  }
  if (cuda_buffers_map_.empty()) {
    return;
  }
  cudaError_t state = cudaSuccess;
  for (auto& [device_id, cuda_buffers] : cuda_buffers_map_) {
    if (no_busy_cnt_[device_id] > 1024 * 1024 * 1024) {
      std::vector<CudaMemoryBuffer> temp_buffer;
      for (auto& buffer : cuda_buffers) {
        if (!buffer.busy) {
          state = cudaSetDevice(device_id);
          state = cudaFree(buffer.data);
          CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device" << device_id;
        } else {
          temp_buffer.push_back(buffer);
        }
      }
      cuda_buffers.clear();
      cuda_buffers = temp_buffer;
      no_busy_cnt_[device_id] = 0;
    }
  }

  for (auto& [device_id, cuda_buffers] : cuda_buffers_map_) {
    for (auto& buffer : cuda_buffers) {
      if (buffer.data == ptr) {
        no_busy_cnt_[device_id] += buffer.byte_size;
        buffer.busy = false;
        return;
      }
    }
    auto& big_buffers = big_buffers_map_[device_id];
    for (auto& buffer : big_buffers) {
      if (buffer.data == ptr) {
        buffer.busy = false;
        return;
      }
    }
  }

  // 直接回收
  state = cudaFree(ptr);
  CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}

void CUDADeviceAllocator::Memcpy(const void* src, void* dest, std::size_t size, MemcpyKind memcpy_kind, void* stream, bool need_sync) {
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

void CUDADeviceAllocator::MemsetZero(void* ptr, std::size_t size, void* stream, bool need_sync) {
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

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance_ = nullptr;
}  // namespace base