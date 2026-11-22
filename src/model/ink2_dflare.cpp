#include "model/ink2_dflare.h"

#include "model/inkling2_kernels.h"
#include "kernels/kflash.h"
#include "model/kparams.h"
#include "model/nemotron_h.h"
#include "model/sampler.h"
#include "util/bf16.h"
#include "util/json_file.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace ni {

using json = nlohmann::json;

namespace {

Pipeline dfl_pipe(const std::string& name) {
  static std::mutex mu;
  static std::unordered_map<std::string, Pipeline> cache;
  {
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
  }
  const Pipeline p = ink2::pipeline(name);
  std::lock_guard<std::mutex> lk(mu);
  return cache.emplace(name, p).first->second;
}

template <class P>
void run_k(Encoder& e, const std::string& name, std::initializer_list<Buffer> bufs, const P& p,
           uint32_t gx, uint32_t gy, uint32_t tx) {
  e.use(dfl_pipe(name));
  int i = 0;
  for (const Buffer& b : bufs) e.buf(i++, b);
  e.val(i, p);
  e.dispatch(gx, gy, 1, tx);
}

struct DflMixP {
  uint32_t R, H, row_off, total_rows;
  float a0, a1, a2, a3, eps;
};
struct DflQ8P {
  uint32_t M, N, K, add;
};
struct DflCtxKvP {
  uint32_t R, Hkv, D, pos0, cap;
  float eps, theta;
};
struct DflBlockP {
  uint32_t B, H, anchor_idx, V;
  float eps;
};
struct DflQkRopeP {
  uint32_t B, Hq, Hkv, D, pos0;
  float eps, theta;
};
struct DflAttnP {
  uint32_t B, Hq, Hkv, D, pos0, ctx_len, ctx_cap;
  float scale;
};
struct DflMarkovP {
  uint32_t R, H, K, step, tok_prev_idx, tok_out_idx;
  float softcap, inv_softcap;
};
struct Ink2NormP {
  uint32_t T, H, has_w, w_f32;
  float eps;
};
struct DflQ8MvP { uint32_t M, N, K, nblk, add; };
struct DflQ8Multi3P { uint32_t M, K, N_tot, nblk; uint32_t offs[4]; };
struct DflQuantP { uint32_t total_blocks, w_f32; };
struct DflP2BaseP { uint32_t M, V, H, nblk; };
struct DflP2ScreenP { uint32_t V, R, has_sup, sup_words; };
struct DflRescoreP { uint32_t V, H, R, u_f32, has_sup, sup_words; };
struct DflArgmaxP { uint32_t tok_out_idx; };
struct DflAccP { uint32_t pos, K; };
struct DflBeamPrepP { uint32_t rank, H, V, first_step; };
struct DflRescoreBeamP { uint32_t V, H, R, u_f32, has_sup, sup_words; float softcap, inv_cap; };
struct DflBeamStepP { uint32_t V, R, d, R_budget, last_step; float inv_t; };
struct DflTreeSelP { uint32_t R_budget, pos; };

constexpr int kCommitBatch = 64;
constexpr uint32_t kShortlistK = 128;

}  // namespace

bool Ink2DFlare::is_available(const std::string& dir) {
  namespace fs = std::filesystem;
  if (!fs::is_regular_file(dir + "/mtp.safetensors")) return false;
  const auto jp = json_file(dir + "/config.json");
  if (!jp || !jp->is_object() || !jp->contains("mtp_config") || !(*jp)["mtp_config"].is_object()) return false;
  const json& m = (*jp)["mtp_config"];
  return m.value("markov_head_type", std::string()) == "rnn_hidden";
}

Ink2DFlare::Ink2DFlare(Inkling2& target, const std::string& dir) : tg_(target), dir_(dir) {
  const auto jp = json_file(dir + "/config.json");
  if (!jp || !jp->is_object() || !jp->contains("mtp_config")) fatal("ink2_dflare: missing mtp_config");
  const json& m = (*jp)["mtp_config"];
  H_ = m.value("hidden_size", 2304);
  NL_ = m.value("num_hidden_layers", 3);
  hq_ = m.value("num_attention_heads", 24);
  hkv_ = m.value("num_key_value_heads", 8);
  D_ = m.value("head_dim", 128);
  I_ = m.value("intermediate_size", 9216);
  eps_ = m.value("rms_norm_eps", 1e-6f);
  theta_ = m.value("rope_theta", 1000000.0f);
  block_size_ = m.value("block_size", 8);
  ctx_cap_ = std::max(1, m.value("sliding_window_size", 1024) - 1);
  rank_ = m.value("markov_rank", 256);
  if ((H_ % 32) != 0 || (I_ % 32) != 0 || rank_ != 256 || block_size_ - 1 > kMaxDrafts)
    fatal("ink2_dflare: unsupported shape H=%d I=%d rank=%d block=%d", H_, I_, rank_, block_size_);

  ink2::compile_kernels();
  if (const char* v = getenv("NI_PREWARM"); !(v && !strcmp(v, "0"))) {
    static const char* const kDflNames[] = {
        "dflare_mix_norm", "dflare_q8_mm", "dflare_ctx_kv_write", "dflare_build_block", "dflare_qk_rope",
        "dflare_attn", "dflare_markov_concat", "dflare_markov_gru", "dflare_fold_mm_q8",
        "dflare_quant_q8_bf16", "dflare_quant_p2_w", "dflare_q8_mv8", "dflare_q8_mv8_multi3",
        "dflare_q8_gate_up_swiglu8", "dflare_p2_unembed_base", "dflare_p2_fold_screen", "dflare_rescore_exact",
        "dflare_argmax_128", "dflare_accept_samples", "dflare_beam_init", "dflare_beam_concat",
        "dflare_beam_gru", "dflare_p2_fold_screen_beam", "dflare_rescore_beam", "dflare_beam_step",
        "dflare_tree_select", "dflare_accept_tree_samples",
    };
    for (const char* k : kDflNames) (void)dfl_pipe(k);
    for (const char* k : {"topk_stage1_rows_sg", "sample_stage2_rows", "sample_stage2_rows_tree", "sample_full"})
      (void)Gpu::get().pipeline(k);
  }
  load();
  alloc_buffers();
  Gpu::get().make_resident(owned_bufs_);
}

Ink2DFlare::~Ink2DFlare() {
  Gpu& g = Gpu::get();
  for (Buffer& b : owned_bufs_) g.release(b);
}

void Ink2DFlare::load() {
  const double t0 = now_ms();
  WeightStore ws;
  ws.open_dir(
      dir_, nullptr, [](const std::string& fn) { return fn == "mtp.safetensors"; }, nullptr);
  Gpu& g = Gpu::get();

  auto req = [&](const std::string& name) -> const Tensor& {
    const Tensor* t = ws.find(name);
    if (!t) fatal("ink2_dflare: missing tensor %s in mtp.safetensors", name.c_str());
    return *t;
  };
  auto to_f32_buf = [&](const std::string& name, const char* lbl, bool own = true) -> Buffer {
    const Tensor& t = req(name);
    const size_t n = size_t(t.numel());
    Buffer b = g.alloc(n * 4, lbl);
    if (own) owned_bufs_.push_back(b);
    float* dst = b.as<float>();
    if (t.dtype == DType::F32) {
      memcpy(dst, t.buf.ptr(), n * 4);
    } else if (t.dtype == DType::BF16) {
      const uint16_t* src = t.buf.as<uint16_t>();
      for (size_t i = 0; i < n; ++i) dst[i] = bf16_to_f32(src[i]);
    } else {
      fatal("ink2_dflare: unsupported dtype for %s", name.c_str());
    }
    return b;
  };
  auto copy_bf16_buf = [&](const std::string& name, const char* lbl) -> Buffer {
    const Tensor& t = req(name);
    if (t.dtype != DType::BF16) fatal("ink2_dflare: expected BF16 for %s", name.c_str());
    const size_t bytes = size_t(t.numel()) * 2;
    Buffer b = g.alloc(bytes, lbl);
    owned_bufs_.push_back(b);
    memcpy(b.ptr(), t.buf.ptr(), bytes);
    return b;
  };

  const unsigned nthr = std::max(1u, std::min(12u, std::thread::hardware_concurrency()));
  size_t w_bytes = 0;
  auto e = g.begin();
  auto pack_q8 = [&](std::initializer_list<const char*> names) -> Q8Mat {
    uint32_t N = 0, K = 0;
    std::vector<const Tensor*> ts;
    for (const char* nm : names) {
      const Tensor& t = req(nm);
      if (t.shape.size() != 2 || t.dtype != DType::BF16) fatal("ink2_dflare: expected 2D BF16 for %s", nm);
      ts.push_back(&t);
      N += uint32_t(t.shape[0]); K = uint32_t(t.shape[1]);
    }
    const size_t nblk = size_t(N) * (K / 32u), srow = size_t(K / 32u) * 2u;
    Q8Mat m{g.alloc(size_t(N) * K, "dfl_q8_q"), g.alloc(nblk * 2u, "dfl_q8_s"), N, K};
    owned_bufs_.push_back(m.q); owned_bufs_.push_back(m.s);
    w_bytes += m.q.size + m.s.size;
    size_t roff = 0;
    for (const Tensor* t : ts) {
      const size_t rows = size_t(t->shape[0]);
      const uint32_t blks = uint32_t(rows * (K / 32u));
      run_k(*e, "dflare_quant_q8_bf16",
            {t->buf, m.q.view(roff * K, rows * K), m.s.view(roff * srow, rows * srow)},
            DflQuantP{blks, 0u}, (blks + 3u) / 4u, 1u, 128u);
      roff += rows;
    }
    return m;
  };
  auto to_q8 = [&](const std::string& name) -> Q8Mat {
    const Tensor& t = req(name);
    if (t.shape.size() != 2 || t.dtype != DType::BF16) fatal("ink2_dflare: expected 2D BF16 for %s", name.c_str());
    Q8Mat m;
    m.N = uint32_t(t.shape[0]);
    m.K = uint32_t(t.shape[1]);
    const size_t qsz = size_t(m.N) * m.K;
    const size_t ssz = size_t(m.N) * (m.K / 32) * 2;
    m.q = g.alloc(qsz, "dfl_q8_q");
    m.s = g.alloc(ssz, "dfl_q8_s");
    owned_bufs_.push_back(m.q);
    owned_bufs_.push_back(m.s);
    w_bytes += qsz + ssz;
    const uint16_t* src = t.buf.as<uint16_t>();
    uint8_t* qdst = m.q.as<uint8_t>();
    uint8_t* sdst = m.s.as<uint8_t>();
    const size_t K = m.K, srow = K / 32 * 2;
    auto worker = [&](size_t r0, size_t r1) {
      std::vector<float> frow(K);
      for (size_t r = r0; r < r1; ++r) {
        const uint16_t* sr = src + r * K;
        for (size_t c = 0; c < K; ++c) frow[c] = bf16_to_f32(sr[c]);
        nh_quant_q8(frow.data(), int64_t(K), qdst + r * K, sdst + r * srow);
      }
    };
    const size_t use_thr = std::min<size_t>(nthr, std::max<size_t>(1, m.N / 64));
    if (use_thr <= 1) {
      worker(0, m.N);
    } else {
      std::vector<std::thread> ts;
      for (size_t ti = 0; ti < use_thr; ++ti) ts.emplace_back(worker, m.N * ti / use_thr, m.N * (ti + 1) / use_thr);
      for (auto& th : ts) th.join();
    }
    return m;
  };

  // fuse_weight [4, 4] -> row-wise softmax alpha[4][4]
  {
    const Tensor& fw = req("fuse_weight");
    const uint16_t* p = fw.buf.as<uint16_t>();
    for (int r = 0; r < 4; ++r) {
      float mx = -1e30f, v[4];
      for (int c = 0; c < 4; ++c) {
        v[c] = bf16_to_f32(p[r * 4 + c]);
        mx = std::max(mx, v[c]);
      }
      float sum = 0.0f;
      for (int c = 0; c < 4; ++c) {
        v[c] = std::exp(v[c] - mx);
        sum += v[c];
      }
      for (int c = 0; c < 4; ++c) alpha_[r][c] = v[c] / sum;
    }
  }

  mask_embed_ = to_f32_buf("mask_embed", "dfl_mask_embed");
  norm_w_ = to_f32_buf("norm.weight", "dfl_norm_w");

  const uint32_t QD = uint32_t(hq_ * D_), KVD = uint32_t(hkv_ * D_);
  layers_.resize(size_t(NL_));
  for (int l = 0; l < NL_; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    Layer& L = layers_[size_t(l)];
    L.dflare_norm = to_f32_buf(p + "dflare_norm.weight", "dfl_dn");
    L.input_norm = to_f32_buf(p + "input_layernorm.weight", "dfl_in");
    L.post_norm = to_f32_buf(p + "post_attention_layernorm.weight", "dfl_pn");
    L.q_norm = to_f32_buf(p + "self_attn.q_norm.weight", "dfl_qn");
    L.k_norm = to_f32_buf(p + "self_attn.k_norm.weight", "dfl_kn");
    L.k_norm_ctx = to_f32_buf(p + "self_attn.k_norm_ctx.weight", "dfl_knc");

    L.qkv_offs[0] = 0u; L.qkv_offs[1] = QD; L.qkv_offs[2] = QD + KVD; L.qkv_offs[3] = QD + 2u * KVD;
    const std::string qp_n = p + "self_attn.q_proj.weight", kp_n = p + "self_attn.k_proj.weight",
                      vp_n = p + "self_attn.v_proj.weight", op_n = p + "self_attn.o_proj.weight",
                      gp_n = p + "mlp.gate_proj.weight", up_n = p + "mlp.up_proj.weight",
                      dp_n = p + "mlp.down_proj.weight";
    L.qkv_q = pack_q8({qp_n.c_str(), kp_n.c_str(), vp_n.c_str()});
    L.o_proj = pack_q8({op_n.c_str()});
    L.k_proj_ctx = to_q8(p + "self_attn.k_proj_ctx.weight");
    L.v_proj_ctx = to_q8(p + "self_attn.v_proj_ctx.weight");
    L.gu_q = pack_q8({gp_n.c_str(), up_n.c_str()});
    L.down_proj = pack_q8({dp_n.c_str()});
  }

  mh_dflare_norm_ = to_f32_buf("markov_head.dflare_norm.weight", "dfl_mh_dn");
  mh_w1_ = copy_bf16_buf("markov_head.markov_w1.weight", "dfl_mh_w1");
  mh_rnn_w_ = copy_bf16_buf("markov_head.rnn_proj.weight", "dfl_mh_rnnw");
  mh_rnn_b_ = to_f32_buf("markov_head.rnn_proj.bias", "dfl_mh_rnnb");

  // Fold target unembed @ hidden_out -> Q8_0 + 2-bit proxy [V, 256] + F32 [V] bias, and pack 2-bit proxy unembed
  const Tensor& hw = req("markov_head.hidden_out.weight");
  Buffer hb = to_f32_buf("markov_head.hidden_out.bias", "dfl_mh_hb_tmp", false);
  const uint32_t V = uint32_t(tg_.vocab()), f_blks = V * uint32_t(rank_ / 32), u_blks = V * uint32_t(H_ / 32);
  const uint32_t u_f32 = tg_.unembed_tensor().dtype == DType::F32 ? 1u : 0u;
  fold_q_ = g.alloc(size_t(V) * size_t(rank_), "dfl_fold_q");
  fold_s_ = g.alloc(size_t(f_blks) * 2u, "dfl_fold_s");
  fold_p2_q_ = g.alloc(size_t(f_blks) * 8u, "dfl_fold_p2q");
  fold_p2_s_ = g.alloc(size_t(f_blks) * 2u, "dfl_fold_p2s");
  fold_b_ = g.alloc(size_t(V) * 4u, "dfl_fold_b");
  unembed_p2_q_ = g.alloc(size_t(u_blks) * 8u, "dfl_unembed_p2q");
  unembed_p2_s_ = g.alloc(size_t(u_blks) * 2u, "dfl_unembed_p2s");
  for (const Buffer& b : {fold_q_, fold_s_, fold_p2_q_, fold_p2_s_, fold_b_, unembed_p2_q_, unembed_p2_s_})
    owned_bufs_.push_back(b);

  {
    struct { uint32_t V, H, R, u_f32; } fp{V, uint32_t(H_), uint32_t(rank_), u_f32};
    run_k(*e, "dflare_fold_mm_q8", {tg_.unembed_tensor().buf, hw.buf, fold_q_, fold_s_, fold_p2_q_, fold_p2_s_},
          fp, uint32_t(rank_ / 32), (V + 31u) / 32u, 128u);
    tg_.enc_unembed(*e, hb, fold_b_, 1);
    run_k(*e, "dflare_quant_p2_w", {tg_.unembed_tensor().buf, unembed_p2_q_, unembed_p2_s_},
          DflQuantP{u_blks, u_f32}, (u_blks + 3u) / 4u, 1u, 128u);
  }
  e->commit_and_wait();
  g.release(hb);

  const size_t head_bytes =
      mh_w1_.size + mh_rnn_w_.size + fold_q_.size + fold_s_.size + fold_p2_q_.size + fold_p2_s_.size + fold_b_.size +
      unembed_p2_q_.size + unembed_p2_s_.size;
  logi("loaded Inkling-2 DFlare drafter (%d layers fused Q8, 2-bit proxy screen, zero-D2H accept): %.2f GB in %.0f ms",
       NL_, (w_bytes + head_bytes) / 1e9, now_ms() - t0);
}

void Ink2DFlare::alloc_buffers() {
  Gpu& g = Gpu::get();
  auto alloc = [&](size_t bytes, const char* lbl) {
    Buffer b = g.alloc(std::max<size_t>(bytes, 16), lbl);
    owned_bufs_.push_back(b);
    return b;
  };
  const size_t B = size_t(block_size_);
  const size_t QD = size_t(hq_) * size_t(D_);
  const size_t KVD = size_t(hkv_) * size_t(D_);
  const size_t V = size_t(tg_.vocab());

  ctx_h_ = alloc(size_t(kCommitBatch) * size_t(H_) * 4, "dfl_ctx_h");
  ctx_k_ = alloc(size_t(kCommitBatch) * KVD * 4, "dfl_ctx_k");
  ctx_v_ = alloc(size_t(kCommitBatch) * KVD * 4, "dfl_ctx_v");

  bx_ = alloc(B * size_t(H_) * 4, "dfl_bx");
  bn_ = alloc(B * size_t(H_) * 4, "dfl_bn");
  bq_ = alloc(B * QD * 4, "dfl_bq");
  bk_ = alloc(B * KVD * 4, "dfl_bk");
  bv_ = alloc(B * KVD * 4, "dfl_bv");
  batt_ = alloc(B * QD * 4, "dfl_batt");
  bact_ = alloc(B * size_t(I_) * 4, "dfl_bact");
  draft_h_ = alloc(size_t(kMaxDrafts) * size_t(H_) * 4, "dfl_draft_h");

  base_proxy_ = alloc(size_t(kMaxDrafts) * V * 2, "dfl_base_proxy");
  pick_v_ = alloc(size_t(kShortlistK) * 4, "dfl_pick_v");
  pick_i_ = alloc(size_t(kShortlistK) * 4, "dfl_pick_i");

  mh_state_ = alloc(size_t(rank_) * 4, "dfl_mh_state");
  mh_rnn_in_ = alloc((2 * size_t(rank_) + 2 * size_t(H_)) * 4, "dfl_mh_in");
  mh_latent_ = alloc(size_t(rank_) * 4, "dfl_mh_lat");

  acc_cv_ = alloc(size_t(Inkling2::kMaxVerify) * 4096u * 4u, "dfl_acc_cv");
  acc_ci_ = alloc(size_t(Inkling2::kMaxVerify) * 4096u * 4u, "dfl_acc_ci");
  acc_samp_ = alloc(size_t(Inkling2::kMaxVerify) * 4u, "dfl_acc_samp");
  acc_sprob_ = alloc(16u, "dfl_acc_sprob");
  acc_dstep_ = alloc(size_t(Inkling2::kMaxVerify) * 4u, "dfl_acc_dstep");
  for (int i = 0; i < Inkling2::kMaxVerify; ++i) acc_dstep_.as<int>()[i] = i;
  accept_out_ = alloc(16u, "dfl_accept_out");
  memset(accept_out_.ptr(), 0, accept_out_.size);

  beam_state_ = alloc(4u * size_t(rank_) * 4u, "dfl_b_state");
  beam_next_state_ = alloc(4u * size_t(rank_) * 4u, "dfl_b_nstate");
  beam_latent_ = alloc(4u * size_t(rank_) * 4u, "dfl_b_lat");
  beam_rnn_in_ = alloc(4u * (2u * size_t(rank_) + 2u * size_t(H_)) * 4u, "dfl_b_rin");
  beam_cand_v_ = alloc(4u * 128u * 4u, "dfl_b_cv");
  beam_cand_i_ = alloc(4u * 128u * 4u, "dfl_b_ci");
  beam_meta_ = alloc(64u, "dfl_b_meta");
  cand_pool_ = alloc(128u * 16u, "dfl_b_cpool");
  tree_buf_ = alloc(512u, "dfl_tree");
  kf_tree_ = alloc(size_t(kf::kWords) * 4u, "dfl_kf_tree");
  memset(kf_tree_.ptr(), 0, kf_tree_.size);
  tree_accept_out_ = alloc(sizeof(Ink2TreeAccept), "dfl_tree_acc");
  memset(tree_accept_out_.ptr(), 0, tree_accept_out_.size);
}

std::unique_ptr<Ink2DFlareSeq> Ink2DFlare::new_seq() {
  Gpu& g = Gpu::get();
  auto ds = std::make_unique<Ink2DFlareSeq>();
  const size_t kv_bytes = size_t(ctx_cap_) * size_t(hkv_) * size_t(D_) * 2;
  ds->kc.resize(size_t(NL_));
  ds->vc.resize(size_t(NL_));
  std::vector<Buffer> res;
  for (int l = 0; l < NL_; ++l) {
    ds->kc[size_t(l)] = g.alloc(kv_bytes, "dfl_seq_kc");
    ds->vc[size_t(l)] = g.alloc(kv_bytes, "dfl_seq_vc");
    memset(ds->kc[size_t(l)].ptr(), 0, kv_bytes);
    memset(ds->vc[size_t(l)].ptr(), 0, kv_bytes);
    res.push_back(ds->kc[size_t(l)]);
    res.push_back(ds->vc[size_t(l)]);
  }
  ds->rows_f = g.alloc(size_t(H_) * 4, "dfl_seq_rows_f");
  memset(ds->rows_f.ptr(), 0, ds->rows_f.size);
  res.push_back(ds->rows_f);
  g.make_resident(res);
  return ds;
}

void Ink2DFlare::reset(Ink2DFlareSeq& ds) {
  ds.ctx_len = 0;
  if (ds.rows_f) memset(ds.rows_f.ptr(), 0, ds.rows_f.size);
}

void Ink2DFlare::enc_q8(Encoder& e, const Q8Mat& w, const Buffer& x, const Buffer& y, int M, bool add) {
  const DflQ8P p{uint32_t(M), w.N, w.K, add ? 1u : 0u};
  e.use(dfl_pipe("dflare_q8_mm"));
  e.buf(0, x);
  e.buf(1, w.q);
  e.buf(2, w.s);
  e.buf(3, y);
  e.val(4, p);
  e.dispatch((w.N + 3) / 4, (uint32_t(M) + 7) / 8, 1, 128);
}

void Ink2DFlare::enc_q8_mv(Encoder& e, const Q8Mat& w, const Buffer& x, const Buffer& y, int M, bool add) {
  run_k(e, "dflare_q8_mv8", {x, w.q, w.s, y}, DflQ8MvP{uint32_t(M), w.N, w.K, w.K / 32u, add ? 1u : 0u},
        (w.N + 7u) / 8u, 1u, 128u);
}

void Ink2DFlare::enc_rmsnorm(Encoder& e, const Buffer& x, const Buffer& w, const Buffer& out, int T) {
  const Ink2NormP p{uint32_t(T), uint32_t(H_), 1u, 1u, eps_};
  e.use(dfl_pipe("ink2_rmsnorm"));
  e.buf(0, x);
  e.buf(1, w);
  e.buf(2, out);
  e.val(3, p);
  e.dispatch(uint32_t(T), 1, 1, 512);
}

void Ink2DFlare::encode_commit(Encoder& e, Ink2DFlareSeq& ds, const Buffer& taps, int n_rows, int pos0,
                               int taps_stride_rows) {
  if (n_rows <= 0) return;
  const int stride = taps_stride_rows > 0 ? taps_stride_rows : n_rows;
  const int keep = std::min(n_rows, ctx_cap_);
  const int start_row = n_rows - keep;
  const int start_pos = pos0 + start_row;

  for (int l = 0; l < NL_; ++l) {
    const Layer& L = layers_[size_t(l)];
    for (int off = 0; off < keep; off += kCommitBatch) {
      const int batch = std::min(kCommitBatch, keep - off);
      {
        const DflMixP mp{uint32_t(batch), uint32_t(H_), uint32_t(start_row + off), uint32_t(stride),
                         alpha_[l][0],    alpha_[l][1], alpha_[l][2],              alpha_[l][3],     eps_};
        e.use(dfl_pipe("dflare_mix_norm"));
        e.buf(0, taps);
        e.buf(1, L.dflare_norm);
        e.buf(2, ctx_h_);
        e.val(3, mp);
        e.dispatch(uint32_t(batch), 1, 1, 512);
      }
      e.group_begin();
      enc_q8(e, L.k_proj_ctx, ctx_h_, ctx_k_, batch, false);
      enc_q8(e, L.v_proj_ctx, ctx_h_, ctx_v_, batch, false);
      e.group_end();
      {
        const DflCtxKvP kp{uint32_t(batch), uint32_t(hkv_), uint32_t(D_), uint32_t(start_pos + off),
                           uint32_t(ctx_cap_), eps_, theta_};
        e.use(dfl_pipe("dflare_ctx_kv_write"));
        e.buf(0, ctx_k_);
        e.buf(1, ctx_v_);
        e.buf(2, L.k_norm_ctx);
        e.buf(3, ds.kc[size_t(l)]);
        e.buf(4, ds.vc[size_t(l)]);
        e.val(5, kp);
        e.dispatch(uint32_t(hkv_), uint32_t(batch), 1, uint32_t(D_));
      }
    }
  }
  // Update markov_head rows_f from the last committed row (n_rows - 1)
  {
    const DflMixP mp{1u,          uint32_t(H_), uint32_t(n_rows - 1), uint32_t(stride),
                     alpha_[3][0], alpha_[3][1], alpha_[3][2],         alpha_[3][3], eps_};
    e.use(dfl_pipe("dflare_mix_norm"));
    e.buf(0, taps);
    e.buf(1, mh_dflare_norm_);
    e.buf(2, ds.rows_f);
    e.val(3, mp);
    e.dispatch(1, 1, 1, 512);
  }
  ds.ctx_len = std::min(ctx_cap_, ds.ctx_len + n_rows);
}

void Ink2DFlare::encode_backbone_and_base(Encoder& e, Seq& s, Ink2DFlareSeq& ds, int pos, int depth) {
  if (depth < 1 || depth > kMaxDrafts) fatal("ink2_dflare: depth %d out of range", depth);
  const int B = depth + 1;
  const uint32_t H = uint32_t(H_), V = uint32_t(tg_.vocab());
  const float scale = 1.0f / std::sqrt(float(D_));
  const uint32_t ctx_len = uint32_t(std::min(ds.ctx_len, std::min(pos, ctx_cap_)));

  // 1. Build block [anchor_embed, mask_embed x depth] -> bx_
  {
    const DflBlockP bp{uint32_t(B), H, uint32_t(pos), uint32_t(tg_.embed_tensor().shape[0]), eps_};
    e.use(dfl_pipe("dflare_build_block"));
    e.buf(0, s.tok);
    e.buf(1, tg_.embed_tensor().buf);
    e.buf(2, tg_.embed_norm_tensor().buf);
    e.buf(3, mask_embed_);
    e.buf(4, bx_);
    e.val(5, bp);
    e.dispatch(uint32_t(B), 1, 1, 512);
  }

  // 2. 3 draft transformer layers (Q8_0 fused QKV + SwiGLU)
  for (int l = 0; l < NL_; ++l) {
    const Layer& L = layers_[size_t(l)];
    enc_rmsnorm(e, bx_, L.input_norm, bn_, B);
    {
      const DflQ8Multi3P mp{uint32_t(B), H, L.qkv_offs[3], H / 32u,
                            {L.qkv_offs[0], L.qkv_offs[1], L.qkv_offs[2], L.qkv_offs[3]}};
      run_k(e, "dflare_q8_mv8_multi3", {bn_, L.qkv_q.q, L.qkv_q.s, bq_, bk_, bv_}, mp, (mp.N_tot + 7u) / 8u, 1u, 128u);
    }
    run_k(e, "dflare_qk_rope", {bq_, bk_, L.q_norm, L.k_norm},
          DflQkRopeP{uint32_t(B), uint32_t(hq_), uint32_t(hkv_), uint32_t(D_), uint32_t(pos), eps_, theta_},
          uint32_t(hq_ + hkv_), uint32_t(B), uint32_t(D_));
    run_k(e, "dflare_attn", {bq_, ds.kc[size_t(l)], ds.vc[size_t(l)], bk_, bv_, batt_},
          DflAttnP{uint32_t(B), uint32_t(hq_), uint32_t(hkv_), uint32_t(D_), uint32_t(pos), ctx_len, uint32_t(ctx_cap_), scale},
          uint32_t(hq_), uint32_t(B), 128u);
    enc_q8_mv(e, L.o_proj, batt_, bx_, B, true);

    enc_rmsnorm(e, bx_, L.post_norm, bn_, B);
    run_k(e, "dflare_q8_gate_up_swiglu8", {bn_, L.gu_q.q, L.gu_q.s, bact_},
          DflQ8MvP{uint32_t(B), uint32_t(I_), H, H / 32u, 0u}, (uint32_t(I_) + 3u) / 4u, 1u, 128u);
    enc_q8_mv(e, L.down_proj, bact_, bx_, B, true);
  }

  // 3. Final norm on mask positions [1 .. depth] -> draft_h_ [depth, H]
  enc_rmsnorm(e, bx_.view(size_t(H) * 4, size_t(depth) * size_t(H) * 4), norm_w_, draft_h_, depth);

  // 4. 2-bit proxy unembed base screening [depth, V] fp16
  run_k(e, "dflare_p2_unembed_base", {draft_h_, unembed_p2_q_, unembed_p2_s_, fold_b_, base_proxy_},
        DflP2BaseP{uint32_t(depth), V, H, H / 32u}, (V + 15u) / 16u, 1u, 128u);
}

void Ink2DFlare::encode_draft(Encoder& e, Seq& s, Ink2DFlareSeq& ds, int pos, int depth) {
  encode_backbone_and_base(e, s, ds, pos, depth);
  const uint32_t H = uint32_t(H_), R = uint32_t(rank_), V = uint32_t(tg_.vocab());

  // 5. Left-to-right RNN Markov head: 2-bit full-vocab screen -> exact top-128 rescore -> argmax
  const uint32_t rnn_in_dim = 2u * R + 2u * H, has_sup = s.suppression ? 1u : 0u;
  const uint32_t sup_words = has_sup ? uint32_t(s.suppression->words) : 0u;
  const Buffer& sup_buf = has_sup ? s.suppression->mask : fold_b_;
  const uint32_t u_f32 = tg_.unembed_tensor().dtype == DType::F32 ? 1u : 0u;
  for (int step = 0; step < depth; ++step) {
    const DflMarkovP mp{R, H, uint32_t(kCandK), uint32_t(step), uint32_t(pos + step), uint32_t(pos + step + 1), 0.0f, 0.0f};
    run_k(e, "dflare_markov_concat", {mh_state_, mh_w1_, draft_h_, ds.rows_f, s.tok, mh_rnn_in_}, mp,
          (rnn_in_dim + 255u) / 256u, 1u, 256u);
    run_k(e, "dflare_markov_gru", {mh_rnn_in_, mh_rnn_w_, mh_rnn_b_, mh_state_, mh_latent_}, mp, (R + 3u) / 4u, 1u, 128u);
    run_k(e, "dflare_p2_fold_screen",
          {mh_latent_, fold_p2_q_, fold_p2_s_, base_proxy_.view(size_t(step) * V * 2u, size_t(V) * 2u), sup_buf, pick_v_, pick_i_},
          DflP2ScreenP{V, R, has_sup, sup_words}, 64u, 1u, 256u);
    run_k(e, "dflare_rescore_exact",
          {draft_h_.view(size_t(step) * H * 4u, size_t(H) * 4u), mh_latent_, tg_.unembed_tensor().buf, fold_q_, fold_s_,
           fold_b_, sup_buf, pick_i_, pick_v_},
          DflRescoreP{V, H, R, u_f32, has_sup, sup_words}, 32u, 1u, 128u);
    run_k(e, "dflare_argmax_128", {pick_v_, pick_i_, s.tok}, DflArgmaxP{uint32_t(pos + step + 1)}, 1u, 1u, 128u);
  }
}

void Ink2DFlare::encode_draft_tree(Encoder& e, Seq& s, Ink2DFlareSeq& ds, int pos, int R_budget,
                                   const SamplingParams& sp) {
  const int R_b = std::clamp(R_budget, 2, Inkling2::kMaxVerify);
  const int D_max = std::min(max_drafts(), R_b - 1);
  encode_backbone_and_base(e, s, ds, pos, D_max);
  const uint32_t H = uint32_t(H_), R = uint32_t(rank_), V = uint32_t(tg_.vocab()), tot_in = 2u * R + 2u * H;
  run_k(e, "dflare_beam_init", {s.tok, beam_meta_, cand_pool_}, uint32_t(pos), 1u, 1u, 32u);

  const float t_beam = sp.temperature > 0.0f ? std::clamp(sp.temperature, 0.25f, 2.0f) : 1.0f;
  const float inv_t = 1.0f / t_beam, softcap = tg_.config().logit_softcap, inv_sc = softcap > 0.0f ? 1.0f / softcap : 0.0f;
  const uint32_t has_sup = s.suppression ? 1u : 0u, sup_words = has_sup ? uint32_t(s.suppression->words) : 0u;
  const uint32_t u_f32 = tg_.unembed_tensor().dtype == DType::F32 ? 1u : 0u;
  const Buffer& sup_buf = has_sup ? s.suppression->mask : fold_b_;

  for (int d = 0; d < D_max; ++d) {
    const uint32_t active_b = (d == 0) ? 1u : uint32_t(std::min(4, R_b - 1));
    const Buffer h_row = draft_h_.view(size_t(d) * H * 4u, size_t(H) * 4u);
    const DflBeamPrepP bp{R, H, V, d == 0 ? 1u : 0u};
    run_k(e, "dflare_beam_concat", {beam_meta_, beam_state_, mh_w1_, h_row, ds.rows_f, beam_rnn_in_}, bp,
          (tot_in + 255u) / 256u, active_b, 256u);
    run_k(e, "dflare_beam_gru", {beam_meta_, beam_rnn_in_, mh_rnn_w_, mh_rnn_b_, beam_state_, beam_next_state_, beam_latent_},
          bp, (R + 3u) / 4u, 1u, 128u);
    run_k(e, "dflare_p2_fold_screen_beam",
          {beam_meta_, beam_latent_, fold_p2_q_, fold_p2_s_, base_proxy_.view(size_t(d) * V * 2u, size_t(V) * 2u), sup_buf,
           beam_cand_v_, beam_cand_i_},
          DflP2ScreenP{V, R, has_sup, sup_words}, 64u, 1u, 256u);
    run_k(e, "dflare_rescore_beam",
          {beam_meta_, h_row, beam_latent_, tg_.unembed_tensor().buf, fold_q_, fold_s_, fold_b_, sup_buf, beam_cand_i_,
           beam_cand_v_},
          DflRescoreBeamP{V, H, R, u_f32, has_sup, sup_words, softcap, inv_sc}, 32u, active_b, 128u);
    run_k(e, "dflare_beam_step", {beam_meta_, cand_pool_, beam_cand_v_, beam_cand_i_, beam_next_state_, beam_state_},
          DflBeamStepP{V, R, uint32_t(d), uint32_t(R_b), (d + 1 == D_max) ? 1u : 0u, inv_t}, 1u, 1u, 128u);
  }
  run_k(e, "dflare_tree_select", {beam_meta_, cand_pool_, tree_buf_, kf_tree_, s.tok},
        DflTreeSelP{uint32_t(R_b), uint32_t(pos)}, 1u, 1u, 128u);
}

void Ink2DFlare::encode_accept_tree(Encoder& e, Seq& s, int pos, int R_budget, const Buffer& verify_logits,
                                    const SamplingParams& sp) {
  const uint32_t V = uint32_t(tg_.vocab());
  if (full_vocab(sp)) {
    enc_sample_full_rows(e, verify_logits, V, R_budget, acc_samp_, 0, acc_sprob_, sp, uint32_t(pos), kf_tree_,
                         size_t(kf::kDep) * 4u);
  } else {
    static const Pipeline p_tk = Gpu::get().pipeline("topk_stage1_rows_sg");
    static const Pipeline p_s2 = Gpu::get().pipeline("sample_stage2_rows_tree");
    const uint32_t K = topk_k(sp), chunk = topk_chunk(K), lists = topk_lists(V, K);
    e.use(p_tk);
    e.buf(0, verify_logits); e.buf(1, acc_cv_); e.buf(2, acc_ci_);
    e.val(3, kp::TopkStage{V, K, chunk});
    e.dispatch(lists, uint32_t(R_budget), 1, topk_threads(K));

    e.use(p_s2);
    e.buf(0, acc_cv_); e.buf(1, acc_ci_); e.buf(2, acc_samp_);
    e.val(3, kp::SampleRows{lists * K, K, sp.temperature, sp.top_p, sp.seed, uint32_t(pos), 0u, 0u, 0u, 0u});
    e.buf(4, acc_dstep_);
    e.buf(5, kf_tree_);
    e.dispatch(uint32_t(R_budget), 1, 1, 1024);
  }
  run_k(e, "dflare_accept_tree_samples", {tree_buf_, acc_samp_, s.tok, tree_accept_out_}, uint32_t(pos), 1u, 1u, 32u);
}

void Ink2DFlare::encode_accept(Encoder& e, Seq& s, int pos, int depth, const Buffer& verify_logits,
                               const SamplingParams& sp) {
  const uint32_t V = uint32_t(tg_.vocab());
  const int R = depth + 1;
  if (full_vocab(sp)) {
    enc_sample_full_rows(e, verify_logits, V, R, acc_samp_, 0, acc_sprob_, sp, uint32_t(pos), acc_dstep_, 0);
  } else {
    static const Pipeline p_tk = Gpu::get().pipeline("topk_stage1_rows_sg");
    static const Pipeline p_s2 = Gpu::get().pipeline("sample_stage2_rows");
    const uint32_t K = topk_k(sp), chunk = topk_chunk(K), lists = topk_lists(V, K);
    e.use(p_tk);
    e.buf(0, verify_logits); e.buf(1, acc_cv_); e.buf(2, acc_ci_);
    e.val(3, kp::TopkStage{V, K, chunk});
    e.dispatch(lists, uint32_t(R), 1, topk_threads(K));

    e.use(p_s2);
    e.buf(0, acc_cv_); e.buf(1, acc_ci_); e.buf(2, acc_samp_);
    e.val(3, kp::SampleRows{lists * K, K, sp.temperature, sp.top_p, sp.seed, uint32_t(pos), 0u, 0u, 0u, 0u});
    e.buf(4, acc_dstep_);  // unused when table_from == 0
    e.dispatch(uint32_t(R), 1, 1, 1024);
  }
  run_k(e, "dflare_accept_samples", {s.tok, acc_samp_, accept_out_}, DflAccP{uint32_t(pos), uint32_t(depth)}, 1u, 1u, 32u);
}

}  // namespace ni
