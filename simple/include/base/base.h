#pragma once

namespace base {
enum class DeviceType {
  Unknown,
  CPU,
  CUDA,
};

class NoCopyable {
 protected:
  NoCopyable() = default;
  ~NoCopyable() = default;
  NoCopyable(const NoCopyable&) = delete;
  NoCopyable& operator=(const NoCopyable&) = delete;
};

}  // namespace base