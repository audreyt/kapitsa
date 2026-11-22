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

struct Ink2Tree {
  uint32_t R;              // number of nodes (1..16), node 0 = anchor s.tok[P]
  uint32_t depth[16];      // depth[0] = 0; depth[r] = depth[parent[r]] + 1
  uint32_t anc_mask[16];   // bit j set iff node j is ancestor of r (or j == r)
  int32_t parent[16];      // parent[0] = -1; parent[r] in [0, r-1]
  int32_t par1[16];        // lag-1 node index (-1 for root -> hist[2])
  int32_t par2[16];        // lag-2 node index (-2 for root -> hist[1])
  int32_t par3[16];        // lag-3 node index (-3 for root -> hist[0])
  int32_t tokens[16];      // tokens[0] = anchor; tokens[r] = drafted token at node r
  uint32_t n_child[16];    // number of direct children of node u (0..15)
  uint32_t child[16][15];  // direct child node indices of u
};

struct Ink2TreeAccept {
  int32_t m;          // committed rows (1..R): path[0..m-1], tokens emitted = m
  int32_t stop_node;  // path[m-1], row from which bonus/correction token is emitted
  int32_t bonus_tok;  // emitted bonus/correction token written to s.tok[P + m]
  int32_t ok;         // 1 when GPU accept completed
  int32_t path[16];   // path[0]=0, path[1], ..., path[m-1]
};

// Pure CPU speculative acceptance over `K` draft rows (greedy or lossless rejection sampling).
// Reads `tokens[P+1..P+K]` (drafts), `verify_logits [K+1, V]`, `draft_logits [K, V]`, `verify_samples [K+1]`;
// writes the bonus/correction token at `tokens[P + 1 + a]` and returns `a` (0..K).
int ink2_mtp_accept_cpu(int* tokens, int P, int K, int V, const float* verify_logits, const float* draft_logits,
                        const int* verify_samples, const SamplingParams& sp, uint32_t step,
                        bool point_mass = false);

// Pure CPU lossless multi-branch point-mass speculative tree verification (0.00000 KLD at all T >= 0).
Ink2TreeAccept ink2_mtp_accept_tree_cpu(int* tokens, int P, const Ink2Tree& tree, int V,
                                        const float* verify_logits, const int* verify_samples,
                                        const SamplingParams& sp, uint32_t step);

class Ink2Mtp {
 public:
  static constexpr int kMaxVerifyRows = 16;

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

  // Returns true if this sampling config executes exact speculative acceptance on GPU (writing accept_out[0..2]).
  static bool can_gpu_accept(const SamplingParams& sp);

  // Encode on-GPU exact speculative acceptance for `sp.temperature > 0` (`verify_logits` [R, V], R = K + 1).
  void encode_accept_gpu(Encoder& e, Seq& s, int P, int R, const Buffer& verify_logits, const Buffer& accept_out,
                         const SamplingParams& sp, uint32_t step, bool point_mass) const;

  // Run greedy or lossless speculative acceptance against `verify_logits [R, V]`, `verify_samples [R]`, `accept_out`.
  // Writes the bonus/correction token into `s.tok[P + 1 + a]` and returns `a` (0..K).
  int accept(Seq& s, int P, int K, const Buffer& verify_logits, const Buffer& verify_samples, const Buffer& accept_out,
             const SamplingParams& sp, uint32_t step, bool point_mass = false, bool gpu_accepted = false) const;

  // Bayes-Optimal W=4 On-GPU Markov Beam Tree drafting & lossless zero-logits-D2H verification.
  bool can_tree() const { return screen_on_ && cfg_.markov_rank == 256 && cfg_.block >= 2; }
  const Buffer& tree_buf() const { return tree_buf_; }
  void encode_draft_tree(Encoder& e, Seq& s, int P, int R_budget, const SamplingParams& sp);
  void encode_accept_tree_gpu(Encoder& e, Seq& s, int P, int R, const Buffer& verify_logits,
                              const Buffer& verify_samples, const Buffer& accept_out,
                              const SamplingParams& sp, uint32_t step) const;
  Ink2TreeAccept accept_tree(Seq& s, int P, const Buffer& verify_logits, const Buffer& verify_samples,
                             const Buffer& accept_out, const SamplingParams& sp, uint32_t step,
                             bool gpu_accepted) const;

 private:
  void encode_block_backbone(Encoder& e, Seq& s, int P);
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
  Buffer acc_stats_, acc_cand_v_, acc_cand_i_, acc_dists_;
  Buffer beam_state_, beam_next_state_, beam_latent_, beam_rnn_in_, beam_rnn_gates_;
  Buffer beam_cand_v_, beam_cand_i_, beam_meta_, cand_pool_, tree_buf_;
};

}  // namespace ni
