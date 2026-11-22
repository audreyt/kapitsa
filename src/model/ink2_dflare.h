// Bundled DFlare / DSpark block drafter for Thinking Machines Inkling-2-Mini (mtp.safetensors).
#pragma once

#include "loader/safetensors.h"
#include "model/inkling2.h"

#include <memory>
#include <string>
#include <vector>

namespace ni {

struct Ink2DFlareSeq {
  std::vector<Buffer> kc, vc;  // per draft layer (3): half [ctx_cap, Hkv * D]
  Buffer rows_f;               // float [H]: last committed markov_head context mix
  int ctx_len = 0;             // committed context positions in ring (clamped to ctx_cap)
};

struct Ink2TreeAccept {
  int32_t m, stop_node, bonus_tok, path[8];
};

class Ink2DFlare {
 public:
  static constexpr int kMaxDrafts = 7;
  static constexpr int kCandK = 32;

  static bool is_available(const std::string& dir);
  Ink2DFlare(Inkling2& target, const std::string& dir);
  ~Ink2DFlare();

  int max_drafts() const { return block_size_ - 1; }
  int block_size() const { return block_size_; }
  int ctx_cap() const { return ctx_cap_; }

  std::unique_ptr<Ink2DFlareSeq> new_seq();
  void reset(Ink2DFlareSeq& ds);

  // Commit `n_rows` target taps ([4, taps_stride_rows, H] float) starting at sequence position `pos0`.
  // `taps_stride_rows` defaults to `n_rows` when 0 (pass the verify batch size T when committing a prefix).
  // Only the newest min(n_rows, ctx_cap_) rows are projected into the rolling ring.
  void encode_commit(Encoder& e, Ink2DFlareSeq& ds, const Buffer& taps, int n_rows, int pos0,
                     int taps_stride_rows = 0);

  // Draft `depth` tokens (1..7) into `s.tok[pos + 1 .. pos + depth]` conditioned on `s.tok[pos]`
  // (the anchor token at `pos`) and the committed context ring `[pos - ctx_len .. pos - 1]`.
  void encode_draft(Encoder& e, Seq& s, Ink2DFlareSeq& ds, int pos, int depth);

  // On-GPU Zero-Logits-D2H lossless acceptance: samples verify rows 0..depth using the target sampler
  // and writes accepted count `a` and bonus/correction token to `s.tok[pos + a + 1]`.
  void encode_accept(Encoder& e, Seq& s, int pos, int depth, const Buffer& verify_logits, const SamplingParams& sp);
  int accepted_count() const { return accept_out_.as<int>()[0]; }

  // Bayes-Optimal W=4 On-GPU Markov Beam Tree drafting & lossless Zero-Logits-D2H tree accept
  const Buffer& tree_buf() const { return tree_buf_; }
  void encode_draft_tree(Encoder& e, Seq& s, Ink2DFlareSeq& ds, int pos, int R_budget, const SamplingParams& sp);
  void encode_accept_tree(Encoder& e, Seq& s, int pos, int R_budget, const Buffer& verify_logits,
                          const SamplingParams& sp);
  Ink2TreeAccept tree_accept() const { return *tree_accept_out_.as<Ink2TreeAccept>(); }

 private:
  struct Q8Mat {
    Buffer q, s;
    uint32_t N = 0, K = 0;
  };
  struct Layer {
    Buffer dflare_norm, input_norm, post_norm;
    Buffer q_norm, k_norm, k_norm_ctx;
    Q8Mat qkv_q, o_proj, gu_q, down_proj;
    uint32_t qkv_offs[4] = {};
    Q8Mat k_proj_ctx, v_proj_ctx;
  };

  void load();
  void alloc_buffers();
  void enc_q8(Encoder& e, const Q8Mat& w, const Buffer& x, const Buffer& y, int M, bool add = false);
  void enc_q8_mv(Encoder& e, const Q8Mat& w, const Buffer& x, const Buffer& y, int M, bool add = false);
  void enc_rmsnorm(Encoder& e, const Buffer& x, const Buffer& w, const Buffer& out, int T);
  void encode_backbone_and_base(Encoder& e, Seq& s, Ink2DFlareSeq& ds, int pos, int depth);

  Inkling2& tg_;
  std::string dir_;
  int H_ = 2304, NL_ = 3, hq_ = 24, hkv_ = 8, D_ = 128, I_ = 9216;
  int block_size_ = 8, ctx_cap_ = 1023, rank_ = 256;
  float eps_ = 1e-6f, theta_ = 1000000.0f;
  float alpha_[4][4] = {};  // softmax(fuse_weight)

  std::vector<Layer> layers_;
  Buffer mask_embed_, norm_w_;
  Buffer mh_dflare_norm_, mh_w1_, mh_rnn_w_, mh_rnn_b_;
  Buffer fold_q_, fold_s_, fold_b_;              // folded unembed @ hidden_out (Q8_0 [V, 256] + F32 [V])
  Buffer unembed_p2_q_, unembed_p2_s_;           // 2-bit proxy for unembed [V, H]
  Buffer fold_p2_q_, fold_p2_s_;                 // 2-bit proxy for folded head [V, 256]
  std::vector<Buffer> owned_bufs_;

  // Reusable scratch buffers
  Buffer ctx_h_, ctx_k_, ctx_v_;                 // commit scratch (up to 64 rows per batch)
  Buffer bx_, bn_, bq_, bk_, bv_, batt_;         // block scratch (B <= 8)
  Buffer bact_, draft_h_;                        // MLP activation + final hidden (B <= 8)
  Buffer base_proxy_, pick_v_, pick_i_;          // 2-bit screening [depth, V] fp16 + top-128 shortlist
  Buffer mh_state_, mh_rnn_in_, mh_latent_;
  Buffer acc_cv_, acc_ci_, acc_samp_, acc_sprob_, acc_dstep_, accept_out_;
  Buffer beam_state_, beam_next_state_, beam_latent_, beam_rnn_in_;
  Buffer beam_cand_v_, beam_cand_i_, beam_meta_, cand_pool_, tree_buf_, kf_tree_, tree_accept_out_;
};

}  // namespace ni
