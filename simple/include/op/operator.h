#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "base/base.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace op {

enum class OperatorType : uint8_t {
  Unknown = 0,
  Linear = 1,
  Encode,
  Embedding,
  RMSNorm,
  MatMul,
  RoPe,
  MHA,
  SoftMax,
  Add,
  Swiglu
};

// 算子的基类，用来提供算子统一的接口
class BaseOperator {
 public:
  explicit BaseOperator(
      base::DeviceType device_type, OperatorType layer_type, base::DataType data_type,
      std::string layer_name = "");

  base::DataType GetDataType() const;

  OperatorType GetLayerType() const;

  virtual base::Status Init() = 0;

  virtual base::Status Forward() = 0;

  virtual base::Status Forward(const tensor::Tensor& input0, const tensor::Tensor& output0) = 0;

  virtual base::Status Forward(
      const tensor::Tensor& input0, const tensor::Tensor& input1,
      const tensor::Tensor& output0) = 0;

  virtual base::Status Forward(
      const tensor::Tensor& input0, const tensor::Tensor& input1, const tensor::Tensor& input2,
      const tensor::Tensor& output0) = 0;

  virtual base::Status Forward(
      const tensor::Tensor& input0, const tensor::Tensor& input1, const tensor::Tensor& input2,
      const tensor::Tensor& input3, const tensor::Tensor& output0) = 0;

  virtual base::Status Forward(
      const tensor::Tensor& input0, const tensor::Tensor& input1, const tensor::Tensor& input2,
      const tensor::Tensor& input3, const tensor::Tensor& input4,
      const tensor::Tensor& output0) = 0;

  virtual void SetInput(int idx, const tensor::Tensor& input) = 0;

  virtual void SetOutput(int idx, const tensor::Tensor& output) = 0;

  virtual std::size_t GetInputSize() const = 0;

  virtual std::size_t GetOutputSize() const = 0;

  virtual base::Status Check() const = 0;

  virtual tensor::Tensor& GetInput(int idx) = 0;

  virtual tensor::Tensor& GetOutput(int idx) = 0;

  virtual const tensor::Tensor& GetInput(int idx) const = 0;

  virtual const tensor::Tensor& GetOutput(int idx) const = 0;

  virtual base::Status SetWeight(int idx, const tensor::Tensor& weight);

  virtual base::Status SetWeight(
      int idx, const std::vector<int>& dims, const void* weight_ptr,
      base::DeviceType device_type = base::DeviceType::Unknown);

  const std::string& GetLayerName() const;

  void SetLayerName(const std::string& layer_name);

  base::DeviceType GetDeviceType() const;

  void SetDeviceType(base::DeviceType device_type);
 protected:
  std::string layer_name_;
  OperatorType layer_type_{OperatorType::Unknown};
  base::DataType data_type_{base::DataType::Unknown};
  base::DeviceType device_type_{base::DeviceType::Unknown};
};

// 带参算子
class OperatorWithParams : public BaseOperator {
 public:
  explicit OperatorWithParams(
      base::DeviceType device_type, OperatorType layer_type, base::DataType data_type,
      std::string layer_name = "");

  base::Status Init() override;

  base::Status Forward() override;

  base::Status Forward(const tensor::Tensor& input0, const tensor::Tensor& output0) override;

  base::Status Forward(
      const tensor::Tensor& input0, const tensor::Tensor& input1,
      const tensor::Tensor& output0) override;

  base::Status Forward(
      const tensor::Tensor& input0, const tensor::Tensor& input1, const tensor::Tensor& input2,
      const tensor::Tensor& output0) override;

  base::Status Forward(
      const tensor::Tensor& input0, const tensor::Tensor& input1, const tensor::Tensor& input2,
      const tensor::Tensor& input3, const tensor::Tensor& output0) override;

  base::Status Forward(
      const tensor::Tensor& input0, const tensor::Tensor& input1, const tensor::Tensor& input2,
      const tensor::Tensor& input3, const tensor::Tensor& input4,
      const tensor::Tensor& output0) override;

  void SetInput(int idx, const tensor::Tensor& input) override;

  void SetOutput(int idx, const tensor::Tensor& output) override;

  std::size_t GetInputSize() const override;

  std::size_t GetOutputSize() const override;

  base::Status Check() const override;

  tensor::Tensor& GetInput(int idx) override;

  tensor::Tensor& GetOutput(int idx) override;

  const tensor::Tensor& GetInput(int idx) const override;

  const tensor::Tensor& GetOutput(int idx) const override;

  base::Status SetWeight(int idx, const tensor::Tensor& weight) override;

  base::Status SetWeight(
      int idx, const std::vector<int>& dims, const void* weight_ptr,
      base::DeviceType device_type = base::DeviceType::Unknown) override;

  base::Status CheckTensor(
      const tensor::Tensor& tensor, base::DeviceType device_type, base::DataType data_type) const;

  base::Status CheckTensorWithDim(
      const tensor::Tensor& tensor, base::DeviceType device_type, base::DataType data_type,
      ...) const;

  void ResetInputSize(std::size_t size);

  void ResetOutputSize(std::size_t size);

  virtual void ToCuda();

  void SetCudaConfig(std::shared_ptr<kernel::CudaConfig> config);

  std::shared_ptr<kernel::CudaConfig> GetCudaConfig() const;
 protected:
  std::vector<tensor::Tensor> inputs_;
  std::vector<tensor::Tensor> outputs_;
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
};

// 不带参数算子
class OperatorWithOutParams : public OperatorWithParams {
 public:
  explicit OperatorWithOutParams(
      base::DeviceType device_type, OperatorType layer_type, base::DataType data_type,
      std::string layer_name = "");
  std::size_t GetWeightSize() const;

  void ResetWeightSize(std::size_t size);

  tensor::Tensor& GetWeight(int idx);

  const tensor::Tensor& GetWeight(int idx) const;

  void ToCuda() override;

  base::Status SetWeight(int idx, const tensor::Tensor& weight) override;

  base::Status SetWeight(
      int idx, const std::vector<int>& dims, const void* weight_ptr,
      base::DeviceType device_type = base::DeviceType::Unknown) override;

  void SetScales(const tensor::Tensor& scales);

  void SetGroupSize(int group_size_);

  int GetScaleNum() const;
 protected:
  int group_size_ = 0;
  bool is_quant_layer_ = false;
  tensor::Tensor scales_;
  std::vector<tensor::Tensor> weights_;
};

}  // namespace op