#pragma once

#include "loader/safetensors.h"
#include "model/model.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ni {

class Inkling2;
struct Ink2Config;

struct Ink2MtpConfig {
  int hidden = 2304;
  int num_layers = 3;
  int hq = 24;
  int hkv = 8;
  int head_dim = 128;
  int ffn = 9216;
  int swa = 1024;
  int block = 8;
  int markov_rank = 256;
  float rope_theta = 1000000.0f;
  float rms_eps = 1e-6f;
  std::vector<bool> is_local;
  std::vector<int> taps;  // e.g. [5, 23, 41, 48]
};

struct Ink2MtpLayer {
  bool is_local = false;
  const Tensor *in_norm = nullptr, *ff_norm = nullptr, *ctx_norm = nullptr;
  const Tensor *q_norm = nullptr, *k_norm = nullptr, *k_norm_ctx = nullptr;
  const Tensor *q_proj = nullptr, *k_proj = nullptr, *v_proj = nullptr, *o_proj = nullptr;
  const Tensor *k_proj_ctx = nullptr, *v_proj_ctx = nullptr;
  const Tensor *gate_proj = nullptr, *up_proj = nullptr, *down_proj = nullptr;
  Buffer qkv_w, gate_up_w;
  Buffer qkv_q, qkv_s, o_q, o_s, gu_q, gu_s, down_q, down_s;
  uint32_t qkv_offs[4] = {};
};

// Pure CPU speculative acceptance over `K` draft rows (greedy or lossless rejection sampling).
// Reads `tokens[P+1..P+K]` (drafts), `verify_logits [K+1, V]`, `draft_logits [K, V]`, `verify_samples [K+1]`;
// writes the bonus/correction token at `tokens[P + 1 + a]` and returns `a` (0..K).
int ink2_mtp_accept_cpu(int* tokens, int P, int K, int V, const float* verify_logits, const float* draft_logits,
                        const int* verify_samples, const SamplingParams& sp, uint32_t step);

class Ink2Mtp {
 public:
  Ink2Mtp(Inkling2& trunk, const std::string& dir, const Ink2Config& trunk_cfg, const Tensor& unembed, int tmax,
          int ring_rows);
  ~Ink2Mtp();

  static bool exists(const std::string& dir);
  static Ink2MtpConfig read_config(const std::string& dir);
  static size_t plan_bytes(const std::string& dir, const Ink2Config& trunk_cfg);
  static size_t kv_per_pos(const Ink2MtpConfig& c);
  static size_t ring_bytes_per_seq(const Ink2MtpConfig& c, int ring_rows);

  const Ink2MtpConfig& config() const { return cfg_; }
  int block() const { return cfg_.block; }
  bool is_local(int l) const { return cfg_.is_local[size_t(l)]; }
  int ring_rows() const { return ring_rows_; }

  // Commit `T` trunk tap rows starting at `pos0` into draft layers' KV caches and `s.conv[1]`.
  void encode_commit_context(Encoder& e, Seq& s, int pos0, int T, const Buffer& tap0, const Buffer& tap1,
                             const Buffer& tap2, const Buffer& tap3);

  // Draft `K` (1..7) tokens after anchor `s.tok[P]` into `s.tok[P+1..P+K]`, recording draft distributions.
  void encode_draft(Encoder& e, Seq& s, int P, int K, const SamplingParams& sp, uint32_t step);

  // Run greedy or lossless speculative acceptance against `verify_logits [R, V]` and `verify_samples [R]`.
  // Writes the bonus/correction token into `s.tok[P + 1 + a]` and returns `a` (0..K).
  int accept(Seq& s, int P, int K, const Buffer& verify_logits, const Buffer& verify_samples,
             const SamplingParams& sp, uint32_t step) const;

 private:
  enum class WeightMode { Bf16, Q8, Q4 };
  void enc_mv(Encoder& e, const Tensor& w, const Buffer& x, const Buffer& y, int M, const Tensor* bias = nullptr,
              const Buffer* f32_bias = nullptr, uint32_t n_limit = 0, bool add = false, float softcap = 0.0f) const;
  void enc_mv_q(Encoder& e, const Buffer& wq, const Buffer& ws, uint32_t N, uint32_t K, const Buffer& x,
                const Buffer& y, int M, bool add = false) const;
  void enc_rmsnorm(Encoder& e, const Buffer& x, const Tensor& w, const Buffer& out, int T) const;

  Inkling2& trunk_;
  Ink2MtpConfig cfg_;
  int vocab_ = 0;
  int tmax_ = 512;
  int ring_rows_ = 0;
  WeightMode w_mode_ = WeightMode::Q4;
  bool screen_on_ = true;
  float fuse_w_[4][4] = {};

  WeightStore ws_;
  std::vector<Ink2MtpLayer> layers_;
  const Tensor *mask_embed_ = nullptr, *norm_ = nullptr, *rec_norm_ = nullptr;
  const Tensor *markov_w1_ = nullptr, *rnn_proj_w_ = nullptr, *rnn_proj_b_ = nullptr;
  const Tensor *hidden_out_w_ = nullptr, *hidden_out_b_ = nullptr;
  const Tensor *embed_ = nullptr, *embed_norm_ = nullptr, *unembed_ = nullptr;

  std::vector<Buffer> owned_bufs_;
  Buffer fold_w_, fold_q_, fold_s_, fold_b_;  // Pre-folded BF16/Q8_0 [V, rank] + F32 [V]
  Buffer unembed_p2_q_, unembed_p2_s_, fold_p2_q_, fold_p2_s_, base_proxy_;
  Buffer ctx_h_, ctx_k_, ctx_v_;
  Buffer blk_x_, blk_res_, blk_norm_, blk_q_, blk_k_, blk_v_, blk_att_, blk_gate_, blk_up_, blk_out_;
  Buffer part_m_, part_l_, part_o_;
  Buffer state_, latent_, rnn_in_, rnn_gates_;
  Buffer draft_logits_, d_cand_v_, d_cand_i_, d_sprob_;
};

}  // namespace ni
