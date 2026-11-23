// Thinking Machines Inkling-2-Mini (thinkingmachines/Inkling-2-Mini, model_type "inkling2_loop"):
// Text+Vision engine (24 unique transformer+MoE layers evaluated across a 48-visit recurrent loop schedule
// [0..5, 6..17, 6..17, 6..17, 18..23], 200,058 unpadded vocab out of 201,024 padded, AttRes across blocks +
// cross-pass MessageGate controllers + static/dynamic short convolutions + SWA/global sink attention with
// relative position bias + 128-expert top-8 SiTU MoE in NVFP4 L0-19 and MXFP8 L20-23 + final logit softcap 30,
// plus lazy-loaded Ink2ViT vision tower in src/vision/ink2_vit.cpp).
#pragma once

#include "loader/safetensors.h"
#include "model/model.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ni {

class Ink2Mtp;

struct Ink2Config {
  int hidden = 2304;
  int num_layers = 24;
  int vocab_padded = 201024;
  int vocab = 200058;  // unpadded_vocab_size (logits sliced/evaluated over [0, vocab))
  int hq = 24;
  int hkv = 8;
  int head_dim = 128;
  int d_rel = 16;
  int rel_extent = 1024;
  int swa = 1024;
  int block_sz = 6;
  int core_start = 6;
  int core_end = 18;
  int num_passes = 3;
  int n_experts = 128;
  int top_k = 8;
  int moe_I = 1152;
  int gate_rank = 64;
  float rms_eps = 1e-6f;
  float qk_clip = 22.627416997969522f;
  float rel_clip = 8.0f;
  float log_floor = 8192.0f;
  float log_alpha = 0.1f;
  float route_scale = 8.0f;
  float logit_softcap = 30.0f;
  std::vector<bool> is_local;
  std::vector<int> schedule;  // length 48
  std::vector<int> eos;
};

struct Ink2Ctrl {
  const Tensor* down = nullptr;    // [64, 4608]
  const Tensor* forget = nullptr;  // [2304, 64]
  const Tensor* gain = nullptr;    // [2304, 64]
};

struct Ink2Layer {
  bool is_swa = true;
  bool has_ctrl = false;
  bool is_mxfp8 = false;
  // attention projections
  const Tensor *wq_du = nullptr, *wk_dv = nullptr, *wv_dv = nullptr, *wr_du = nullptr;
  const Tensor *attn_sink_proj = nullptr, *gate_attn = nullptr, *wo_ud = nullptr;
  const Tensor *q_scale = nullptr, *rel_scale = nullptr, *rel_proj = nullptr;
  // norms
  const Tensor *attn_norm = nullptr;  // SWA only
  const Tensor *q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr;  // Global only
  const Tensor *mlp_norm = nullptr;
  // short convolutions
  const Tensor *k_sconv = nullptr, *v_sconv = nullptr;  // SWA static [1024, 1, 4]
  const Tensor *k_static = nullptr, *k_proj = nullptr;  // Global dynamic [4, 1024], [32, 2304]
  const Tensor *v_static = nullptr, *v_proj = nullptr;
  const Tensor *attn_sconv = nullptr, *mlp_sconv = nullptr;  // static [2304, 1, 4]
  // cross-pass controllers (core layers 6..17)
  Ink2Ctrl ctrl_attn, ctrl_k, ctrl_v, ctrl_mlp;
  // MoE
  const Tensor *router_w = nullptr, *router_bias = nullptr;
  const Tensor *w13 = nullptr, *w13_sc = nullptr, *w2 = nullptr, *w2_sc = nullptr;
  Buffer w13_outer, w2_outer;  // FP32 [128] = amax[e] / outer_scale_denominator[0] (NVFP4 only)
  Buffer qkvrsg_w;             // Fused BF16 [wq_du; wr_du; attn_sink_proj; gate_attn; wk_dv; wv_dv]
  Buffer qkvrsg_q, qkvrsg_s;   // Q8_0 weights & scales (fast profile T <= 16)
  Buffer wo_q, wo_s;           // Q8_0 wo_ud weights & scales (fast profile T <= 16)
  uint32_t qkvrsg_offs[7] = {};
  uint32_t qkvrsg_lens[6] = {};
};

class Inkling2 : public Model {
 public:
  explicit Inkling2(const std::string& path, int prefill_chunk = 512);
  ~Inkling2() override;

  static bool is_model(const std::string& dir);
  static Ink2Config read_config(const std::string& dir);
  static MemPlan plan(const std::string& dir, int prefill_chunk = 512);
  static size_t kv_per_pos(const Ink2Config& c);
  static size_t conv_bytes_per_seq(const Ink2Config& c);
  static size_t ring_bytes_per_seq(const Ink2Config& c, int tmax);

  const Ink2Config& config() const { return cfg_; }
  int vocab() const override { return cfg_.vocab; }
  const std::vector<int>& eos_ids() const override { return cfg_.eos; }
  int prefill_chunk() const override { return tmax_; }
  bool can_snapshot() const override { return false; }
  bool grouped_passes() const override { return true; }
  bool use_tensor() const { return tensor_pf_; }
  bool quantized_dense() const { return use_q8_dense_ || use_q8_head_ || use_q4_head_; }
  void set_use_tensor(bool on);
  bool has_mtp() const { return mtp_ != nullptr; }
  int mtp_block() const;
  void disable_mtp();

  void set_mem_budget(size_t bytes) override { mem_budget_ = bytes; }
  std::unique_ptr<Seq> new_seq(int limit, int initial = -1) override;
  bool reserve(Seq& s, int need) override;
  size_t kv_grow_held(const Seq& s) const override;
  void drop_kv(Seq& s) override;
  size_t kv_bytes_per_pos() const override;
  void reset(Seq& s) override;
  Checkpoint new_checkpoint() override;
  void save(Encoder& e, const Seq& s, Checkpoint& c, int pos) override;
  void restore(Encoder& e, Seq& s, const Checkpoint& c) override;

  void encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample = true) override;
  void encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                      Checkpoint* snap = nullptr, int snap_row = 0) override;

  // MTP speculative decoding (linear + Bayes-Optimal W=4 Beam Tree)
  bool can_tree_spec() const;
  void encode_mtp_draft(Encoder& e, Seq& s, int P, int K, const SamplingParams& sp, uint32_t step);
  void encode_mtp_verify(Encoder& e, Seq& s, int P, int R, const SamplingParams& sp, uint32_t step,
                         bool point_mass = false);
  int accept_mtp(Seq& s, int P, int K, const SamplingParams& sp, uint32_t step, bool point_mass = false) const;
  void encode_mtp_commit(Encoder& e, Seq& s, int pos0, int m);

  void encode_mtp_draft_tree(Encoder& e, Seq& s, int P, int R, const SamplingParams& sp);
  void encode_mtp_verify_tree(Encoder& e, Seq& s, int P, int R, const SamplingParams& sp, uint32_t step);
  int accept_mtp_tree(Seq& s, int P, const SamplingParams& sp, uint32_t step, int* out_path) const;
  void encode_mtp_commit_tree(Encoder& e, Seq& s, int pos0, const int* path, int m);

  // Diagnostic helpers (tools/ink2_check)
  float* logits() { return logits_.as<float>(); }
  Buffer logits_buf() const { return logits_; }
  Buffer verify_logits_buf() const { return verify_logits_; }
  void encode_logits_rows(Encoder& e, Seq& s, int pos0, int T, const Buffer& out);
  void encode_prefill_diag(Encoder& e, Seq& s, int pos0, int T, const Buffer& diag_rows);
  int debug_tensor_unit_check();

 private:
  friend class Ink2Mtp;
  void load();
  void alloc_buffers();
  bool ensure_rings(Seq& s);
  bool is_swa_slot(size_t slot) const;
  size_t kv_elem_bytes(size_t slot) const { return (use_kv_fp8_ && slot < cfg_.schedule.size()) ? 1u : 2u; }
  size_t kv_grow_unit() const;
  void enc_forward(Encoder& e, Seq& s, int pos0, int T, bool dump_logits, const Buffer* out_logits, bool need_all_logits,
                   const Buffer* diag_rows = nullptr, bool capture_taps = false, bool save_conv_snaps = false,
                   const Buffer* tree_buf = nullptr);
  void enc_mm(Encoder& e, const Tensor& w, const Buffer& x, const Buffer& y, int M, uint32_t N_limit = 0, bool add = false,
              bool exact_f32 = false);
  void enc_q8_mv(Encoder& e, const Buffer& wq, const Buffer& ws, uint32_t N, uint32_t K, const Buffer& x,
                 const Buffer& y, int M, bool add = false, float softcap = 0.0f);
  void enc_q4_mv(Encoder& e, const Buffer& wq, const Buffer& wsb, uint32_t N, uint32_t K, const Buffer& x,
                 const Buffer& y, int M, bool add = false);
  void enc_unembed(Encoder& e, const Buffer& x, const Buffer& y, int M, bool apply_softcap);
  void enc_norm(Encoder& e, const Buffer& x, const Tensor* w, const Buffer& out, int T, uint32_t dim);
  void enc_attn_res(Encoder& e, const Buffer* const* states, int M, const void* query_ptr, bool q_f32, uint32_t Q,
                    const Buffer& out, int T);
  void enc_msg_gate(Encoder& e, const Ink2Ctrl& c, const Buffer& h, const Buffer& m, const Buffer& out, int T);
  void enc_msg_gate3(Encoder& e, const Ink2Layer& L, const Buffer* const* in_ptrs, const Buffer& m0, const Buffer& m1,
                     const Buffer& m2, int T);
  void enc_sample(Encoder& e, Seq& s, const Buffer& lrow, const Buffer& tokens, int out_index, const SamplingParams& sp,
                  uint32_t step);

  std::string path_;
  Ink2Config cfg_;
  WeightStore ws_;
  std::vector<Ink2Layer> layers_;
  const Tensor *embed_ = nullptr, *embed_norm_ = nullptr, *final_norm_ = nullptr, *unembed_ = nullptr;
  std::vector<const Tensor*> attn_queries_;  // [23] (layers 1..23 -> idx 0..22)
  std::vector<const Tensor*> mlp_queries_;   // [24] (layers 0..23)
  const Tensor* output_query_ = nullptr;
  Buffer msg_queries_;                       // FP32 [12][4][2304] interleaved (attn, k, v, mlp)
  std::vector<Buffer> owned_bufs_;

  int tmax_ = 512;
  int ring_rows_ = 0;
  size_t mem_budget_ = 0;
  bool tensor_pf_ = false;
  bool use_q8_dense_ = false;
  bool use_q8_head_ = false;
  bool use_q4_head_ = false;
  bool use_kv_fp8_ = false;
  Buffer unembed_q_, unembed_s_;
  Buffer unembed_q4_, unembed_sb_;

  // Reusable activations sized for `tmax_` rows
  Buffer x_, x_pre_, inp_sa_, ffn_inp_, h_norm_;
  Buffer blocks_[6];        // block boundary states + bank state during middle cycle
  Buffer prelude_blocks_[4];
  Buffer ar_out_;           // attn_res output: up to 4 * tmax * H
  Buffer msg_out_;          // cross-pass messages: 4 * tmax * H
  Buffer pass_msg_out_;     // hoisted cross-pass messages (T <= 16): [12, 4, 16, H]
  Buffer mg_cat_, mg_lat_;  // split MessageGate staging buffers (T <= 16 and T >= 32)
  Buffer mg_f_, mg_g_;
  Buffer n0_, n1_, n2_;     // normed inputs [tmax, H]
  Buffer q_, k_, v_, r_, sinks_, gate_attn_, att_out_, a_out_;
  Buffer dyn_, dyn_v_, rb_, part_m_, part_l_, part_o_;
  Buffer h_mlp_, x_mlp_, rlog_, eid_, ew_, moe_act_, moe_out_;
  Buffer moe_sorted_, moe_rank_, moe_tmp_, moe_act_h_;
  Buffer moe_tiles_, moe_ntiles_, moe_st_tok_, moe_st_pair_, moe_st_ew_, moe_pair_out_;
  Buffer logits_, cand_v_, cand_i_, sprob_, samp_;
  std::unique_ptr<Ink2Mtp> mtp_;
  Buffer tap_[3], verify_conv_, verify_logits_, verify_samples_, verify_accept_, verify_tree_accept_;
  mutable bool last_gpu_accept_ = false;
};

}  // namespace ni
