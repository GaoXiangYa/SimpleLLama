#pragma once

#include <sentencepiece_processor.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "base/base.h"
#include "base/tiktoken.h"
#include "operator.h"

#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT)
#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <absl/strings/str_split.h>
#include "base/tiktoken.h"
#include "base/unordered_dense.h"
#include "nlohmann/json.hpp"
#endif

namespace op {
class EncoderOperatorBase : public OperatorWithOutParams {
 public:
  explicit EncoderOperatorBase(const std::string& token_model_path, bool has_bos, bool has_eos)
      : op::OperatorWithOutParams(base::DeviceType::CPU, OperatorType::Encode, "Encode"),
        has_bos_(has_bos),
        has_eos_(has_eos),
        token_model_path_(std::move(token_model_path)) {}

  virtual std::vector<int32_t> Encode(const std::string& sentence) const = 0;

  virtual std::string Decode(int32_t token_id) const = 0;

  virtual std::string Decode(const std::vector<int32_t>& token_ids) const = 0;

  virtual bool IsSentenceEnding(int32_t token_id) const = 0;

  virtual int32_t GetVocabSize() const = 0;
 protected:
  bool has_bos_ = true;
  bool has_eos_ = false;
  std::string token_model_path_;
};

class SpeEncodeOperator : public EncoderOperatorBase {
 public:
  explicit SpeEncodeOperator(const std::string& token_model_path, bool has_bos, bool has_eos);

  std::vector<int32_t> Encode(const std::string& sentence) const override;

  std::string Decode(int32_t token_id) const override;

  std::string Decode(const std::vector<int32_t>& token_ids) const override;

  bool IsSentenceEnding(int32_t token_id) const override;

  int32_t GetVocabSize() const override;
 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> spe_;
};

class BpeEncodeOperator : public EncoderOperatorBase {
 public:
  explicit BpeEncodeOperator(const std::string& token_model_path, bool has_bos, bool has_eos);

  std::vector<int32_t> Encode(const std::string& sentence) const override;

  std::string Decode(int32_t token_id) const override;

  std::string Decode(const std::vector<int32_t>& token_ids) const override;

  bool IsSentenceEnding(int32_t token_id) const override;

  int32_t GetVocabSize() const override;
 protected:
  int32_t bos_id_ = -1;
  int32_t eos_id_ = -1;
  int32_t stop_token1_ = -1;
  int32_t stop_token2_ = -1;
  int32_t num_token_ = 0;
  std::unique_ptr<tiktoken::tiktoken> tiktoken_;
};

class QwenEncodeOperator : public BpeEncodeOperator {
 public:
  explicit QwenEncodeOperator(const std::string& token_model_path, bool has_bos, bool has_eos);
};
}  // namespace op