#pragma once

#include "model/model.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include "base/base.h"
#include "base/buffer.h"
#include "model/config.h"
#include "model/raw_model_data.h"
#include "op/embedding.h"
#include "op/encode.h"
#include "tensor/tensor.h"

namespace model {
Model::Model(
    base::TokenizerType tokenizer_type, base::ModelType model_type, const std::string& token_path,
    const std::string& model_path, bool is_quant_model)
    : is_quant_model_(is_quant_model),
      token_path_(token_path),
      model_path_(model_path),
      model_type_(model_type),
      tokenizer_type_(tokenizer_type) {}

base::ModelType Model::GetModelType() const {
  return model_type_;
}

const std::string& Model::GetTokenPath() const {
  return token_path_;
}

const std::string& Model::GetModelPath() const {
  return model_path_;
}

base::Status Model::InsertBuffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  if (tensor.IsEmpty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert_or_assign(buffer_idx, tensor);
  return base::error::Success();
}

tensor::Tensor& Model::GetBuffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << static_cast<int>(buffer_idx);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::GetBuffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0) << static_cast<int>(buffer_idx);
  return buffers_.at(buffer_idx);
}

// 根据模型的路径读取模型
base::Status Model::ReadModelFile() {
  if (model_path_.empty()) {
    return base::error::PathNotValid("Failed to open the weight file, the model path is empty");
  }
  auto fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return base::error::PathNotValid(
        "Failed to open the weight file " + model_path_ + " may be the path does not exitst!");
  }
  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    return base::error::PathNotValid("Failed to open the file. The path may be invalid.");
  }

  auto config = ModelConfig{};
  if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
    return base::error::ModelParseError(
        "Failed to retrieve the configuration information from the model file!");
  }
  if (is_quant_model_) {
    if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
      return base::error::ModelParseError(
          "Failed to retrieve the group size information from the model file!");
    }
  }

  auto gen_status = GenerateModelInfos(config);
  if (!gen_status) {
    return gen_status;
  }

  if (!is_quant_model_) {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();
  } else {
    raw_model_data_ = std::make_shared<RawModelDataInt8>();
  }

  struct stat st;
  if (fstat(fd, &st) == -1) {
    close(fd);
    return base::error::ModelParseError(
        "Failed to retrieve the file size information from the model file.");
  }
  raw_model_data_->file_size = st.st_size;
  LOG(INFO) << "The tokenizer model path: " << token_path_;
  std::string tokenizer_type_str =
      tokenizer_type_ == base::TokenizerType::kEncodeBpe ? "Bpe" : "Spe";
  LOG(INFO) << "The tokenizer type: " << tokenizer_type_str;
  LOG(INFO) << "The model path: " << model_path_;
  LOG(INFO) << "The model file size: " << raw_model_data_->file_size << " byte";
  std::string quant_info = is_quant_model_ ? "quant" : "not quant";
  LOG(INFO) << "The model is " << quant_info << " model";

  if (config_) {
    LOG(INFO) << "\nThe model info: " << *config_;
  }

  raw_model_data_->fd = fd;
  raw_model_data_->data =
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);
  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return base::error::ModelParseError(
        "Failed to map the weight file " + model_path_ + " into memory.");
  }
  if (!is_quant_model_) {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig);
  } else {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig) + sizeof(group_size_);
  }
  if (raw_model_data_ == nullptr) {
    LOG(ERROR);
    return base::error::ModelParseError(
        "Failed to map the weight file " + model_path_ +
        " into memory, the pointer to weight start address is null");
  }
  return base::error::Success();
}

base::Status Model::GenerateModelInfos(const ModelConfig& config) const {
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  config_->seq_len_ = config.seq_len;

  config_->kv_dim_ = (config_->dim_ * config_->kv_head_num_) / config_->head_num_;
  config_->kv_mul_ = config_->head_num_ / config_->kv_head_num_;
  config_->head_size_ = config_->dim_ / config_->head_num_;

  config_->is_shared_weight_ = config_->vocab_size_ > 0 ? true : false;
  config_->vocab_size_ = std::abs(config_->vocab_size_);
  return base::error::Success();
}

base::Status Model::CreateEncodeLayer() {
  if (tokenizer_type_ == base::TokenizerType::kEncodeSpe) {
    encode_operator_ = std::make_unique<op::SpeEncodeOperator>(this->token_path_, true, false);
  } else {
#ifdef LLAMA3_SUPPORT
    encode_operator_ = std::make_unique<op::BpeEncodeOperator>(this->token_path_, true, false);
#endif

#ifdef QWEN2_SUPPORT
    encode_operator_ = std::make_unique<op::QwenEncodeOperator>(this->token_path_, false, false);
#endif
  }
  if (!encode_operator_) {
    return base::error::InternalError("Create the encode layer failed.");
  }
  config_->vocab_size_ = encode_operator_->GetVocabSize();
  if (config_->vocab_size_ <= 0) {
    return base::error::InternalError("The vocab size param read error from the model file!");
  }
  return base::error::Success();
}

base::Status Model::GenModelFromFile() {
  config_ = std::make_unique<TransformerConfig>();
  auto create_encode_status = CreateEncodeLayer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode operator failed! " << create_encode_status.GetErrMsg();
    return create_encode_status;
  }
  // mmap
  auto mmap_status = ReadModelFile();
  if (!mmap_status) {
    LOG(ERROR) << "Read model file " << model_path_ << " failed! " << mmap_status.GetErrMsg();
    return mmap_status;
  }
  auto operator_create_status = CreateOperators();
  if (!operator_create_status) {
    LOG(ERROR) << "Create operators for the model file " << model_path_ << " faliled! "
               << operator_create_status.GetErrMsg();
    return operator_create_status;
  }
  return base::error::Success();
}

std::vector<int> Model::Encode(const std::string& sentence) const {
  CHECK(encode_operator_ != nullptr);
  return encode_operator_->Encode(sentence);
}

bool Model::IsSentenceEnding(int32_t token_idx) const {
  CHECK(this->encode_operator_ != nullptr);
  return this->encode_operator_->IsSentenceEnding(token_idx);
}

std::string Model::Decode(int32_t token_idx) const {
  CHECK(this->encode_operator_ != nullptr);
  return this->encode_operator_->Decode(token_idx);
}

std::string Model::Decode(const std::vector<int32_t>& token_idxs) const {
  CHECK(this->encode_operator_ != nullptr);
  return this->encode_operator_->Decode(token_idxs);
}

std::pair<tensor::Tensor, tensor::Tensor> Model::SliceKVCache(
    int32_t layer_idx, int32_t token_pos) const {
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;
  float* key_cache_ptr =
      const_cast<float*>(GetBuffer(ModelBufferType::kKeyCache).Ptr<float>(cache_offset));
  float* value_cache_ptr =
      const_cast<float*>(GetBuffer(ModelBufferType::kValueCache).Ptr<float>(cache_offset));
  tensor::Tensor key(base::DataType::Fp32, config_->kv_dim_, false, nullptr, key_cache_ptr);
  tensor::Tensor value(base::DataType::Fp32, config_->kv_dim_, false, nullptr, value_cache_ptr);
  key.SetDeviceType(device_type_);
  value.SetDeviceType(device_type_);
  return {key, value};
}

tensor::Tensor Model::FillInput(const tensor::Tensor& pos_tensor,  const op::EmbeddingOutput& embedding_output, bool is_prompt) const {
  const int32_t pos = pos_tensor.Index<int32_t>(0);
  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;
  int32_t index = 0;
  if (is_prompt) {
    index = pos;
  }
  std::shared_ptr<base::Buffer> input_emp_buffer = std::make_shared<base::Buffer>(config_->dim_ * sizeof(float), nullptr, input_embeddings.Ptr<float>(index * config_->dim_), true);
  tensor::Tensor input(base::DataType::Fp32, config_->dim_);
  input.Assign(input_emp_buffer);
  input.SetDeviceType(device_type_);
  return input;
}
}  // namespace model