#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "base/base.h"
#include "model/config.h"
#include "model/raw_model_data.h"
#include "op/embedding.h"
#include "op/encode.h"
#include "sampler/sampler.h"
#include "tensor/tensor.h"

namespace model {
class Model {
 public:
  explicit Model(
      base::TokenizerType tokenizer_type, base::ModelType model_type, const std::string& token_path,
      const std::string& model_path, bool is_quant_model);
  virtual base::Status Init(base::DeviceType device_type) = 0;

  virtual base::Status Predict(
      const tensor::Tensor& input, const tensor::Tensor& pos_tensor, bool is_prompt,
      int& next) const = 0;

  virtual base::Status Forward(
      const tensor::Tensor& input, const tensor::Tensor& pos_tensor, int& next) const = 0;

  base::ModelType GetModelType() const;

  const std::string& GetTokenPath() const;

  const std::string& GetModelPath() const;

  virtual tensor::Tensor& GetBuffer(ModelBufferType buffer_idx);

  virtual const tensor::Tensor& GetBuffer(ModelBufferType buffer_idx) const;

  virtual bool IsSentenceEnding(int32_t token_idx) const;

  virtual std::string Decode(int32_t token_idx) const;

  virtual std::string Decode(const std::vector<int32_t>& token_idxs) const;

  virtual std::vector<int32_t> Encode(const std::string& sentence) const;

  virtual op::EmbeddingOutput Embedding(const std::vector<int>& tokens) const;

  virtual std::pair<tensor::Tensor, tensor::Tensor> SliceKVCache(
      int32_t layer_idx, int32_t token_pos) const;

  virtual tensor::Tensor FillInput(
      const tensor::Tensor& pos_tensor, const op::EmbeddingOutput& embedding_output,
      bool is_prompt) const;
 protected:
  virtual base::Status InsertBuffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor);

  virtual base::Status ReadModelFile();

  virtual base::Status CreateEncodeLayer();

  virtual base::Status GenModelFromFile();

  virtual base::Status GenerateModelInfos(const ModelConfig& config) const;

  virtual int32_t PostProcessing(const tensor::Tensor& pos, bool is_prompt) const;
 private:
  virtual void InitMem() = 0;

  virtual base::Status CreateOperators() = 0;

  virtual void CreateParamOperators();

  virtual void CreateNonParamOperators();

  virtual void CreateParamQuantLayers() = 0;
 protected:
  int group_size_ = 1;
  bool is_quant_model_ = false;
  std::unique_ptr<TransformerConfig> config_;
  std::string token_path_;
  std::string model_path_;
  std::unique_ptr<op::EncoderOperatorBase> encode_operator_;
  std::unordered_map<ModelBufferType, tensor::Tensor> buffers_;
  std::unique_ptr<sampler::Sampler> sampler_;
  std::shared_ptr<RawModelData> raw_model_data_;
  base::DeviceType device_type_{base::DeviceType::Unknown};
  base::ModelType model_type_{base::ModelType::kModelTypeUnknown};
  base::TokenizerType tokenizer_type_{base::TokenizerType::kEncodeUnknown};
};
}  // namespace model