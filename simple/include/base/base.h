#pragma once

#include <cstdint>
#include <string>
namespace base {
enum class DeviceType : uint8_t {
  Unknown,
  CPU,
  CUDA,
};

enum class DataType : uint8_t { Unknown = 0, Fp32, Int8, Int32 };

class NoCopyable {
 protected:
  NoCopyable() = default;
  ~NoCopyable() = default;
  NoCopyable(const NoCopyable&) = delete;
  NoCopyable& operator=(const NoCopyable&) = delete;
};

enum StatusCode : uint8_t {
  kSuccess = 0,
  kFunctionUnImplement = 1,
  kPathNotValid = 2,
  kModelParseError = 3,
  kInternalError = 5,
  kKeyValueHasExist = 6,
  kInvalidArgument = 7,
};

inline size_t GetDataTypeSize(DataType data_type) {
  if (data_type == DataType::Fp32) {
    return sizeof(float);
  } else if (data_type == DataType::Int8) {
    return sizeof(int8_t);
  } else if (data_type == DataType::Int32) {
    return sizeof(int32_t);
  } else {
    return 0;
  }
}
class Status {
 public:
  Status(int code = StatusCode::kSuccess, std::string err_message = "");

  Status(const Status& other) = default;

  Status& operator=(const Status& other) = default;

  Status& operator=(int code);

  bool operator==(int code) const;

  bool operator!=(int code) const;

  operator int() const;

  operator bool() const;

  int32_t GetErrCode() const;

  const std::string& GetErrMsg() const;

  void SetErrMsg(const std::string& err_msg);
 private:
  int code_ = StatusCode::kSuccess;
  std::string message_;
};

namespace error {
#define STATUS_CHECK(call)                                                                     \
  do {                                                                                         \
    const base::Status& status = call;                                                         \
    if (!status) {                                                                             \
      const size_t buf_size = 512;                                                             \
      char buf[buf_size];                                                                      \
      snprintf(                                                                                \
          buf, buf_size - 1, "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", \
          __FILE__, __LINE__, int(status), status.get_err_msg().c_str());                      \
      LOG(FATAL) << buf;                                                                       \
    }                                                                                          \
  } while (0)

Status Success(const std::string& err_msg = "");

Status FunctionNotImplement(const std::string& err_msg = "");

Status PathNotValid(const std::string& err_msg = "");

Status ModelParseError(const std::string& err_msg = "");

Status InternalError(const std::string& err_msg = "");

Status KeyHasExits(const std::string& err_msg = "");

Status InvalidArgument(const std::string& err_msg = "");

}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x);

}  // namespace base