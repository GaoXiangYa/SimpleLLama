#pragma once

#include <cstddef>
#include <cstdint>
namespace model {

struct RawModelData {
  ~RawModelData();
  int32_t fd = -1;
  size_t file_size = 0;
  void* data = nullptr;
  void* weight_data = nullptr;
  virtual const void* Weight(size_t offset) const = 0;
};

struct RawModelDataFp32 : public RawModelData {
  const void* Weight(std::size_t offset) const override;
};

struct RawModelDataInt8 : public RawModelData {
  const void* Weight(std::size_t offset) const override;
};
}  // namespace model