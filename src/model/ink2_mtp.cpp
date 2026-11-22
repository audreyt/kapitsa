#include "model/ink2_mtp.h"

#include "model/inkling2.h"
#include "model/inkling2_kernels.h"
#include "model/kparams.h"
#include "model/sampler.h"
#include "util/bf16.h"
#include "util/json_file.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_map>

namespace ni {

using json = nlohmann::json;

namespace {

Pipeline mtp_pipe(const std::string& name) {
  static std::mutex mu;
  static std::unordered_map<std::string, Pipeline> cache;
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(name);
  if (it == cache.end()) it = cache.emplace(name, ink2::pipeline(name)).first;
  return it->second;
}

Pipeline base_pipe(const std::string& name) {
  static std::mutex mu;
  static std::unordered_map<std::string, Pipeline> cache;
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(name);
  if (it == cache.end()) it = cache.emplace(name, Gpu::get().pipeline(name)).first;
  return it->second;
}

struct Ink2MtpFuseP {
  uint32_t T, H;
  float w0, w1, w2, w3, eps;
};
struct Ink2MtpPrepP {
  uint32_t B, H, pos0;
  float eps;
};
struct Ink2MtpNormP {
  uint32_t T, H;
  float eps;
};
struct Ink2MtpMmP {
  uint32_t M, N, K, x_ld, y_ld, has_bias, bias_f32, add;
  float softcap;
};
struct Ink2MtpMulti3P {
  uint32_t M, K, N_tot;
  uint32_t offs[4];
};
struct Ink2MtpQ8FoldP {
  uint32_t V, R;
  float softcap;
};
struct Ink2MtpQuantQ8P {
  uint32_t total_blocks;
};
struct Ink2MtpQuantP2P {
  uint32_t total_blocks, w_f32;
};
struct Ink2MtpP2BaseP {
  uint32_t M, V, H, nblk;
};
struct Ink2MtpP2ScreenP {
  uint32_t V, R, has_sup, sup_words;
};
struct Ink2MtpRescoreP {
  uint32_t V, H, R, u_f32, has_sup, sup_words;
  float softcap, inv_cap;
};
struct Ink2MtpQ8MmP {
  uint32_t M, N, K, nblk, add;
};
struct Ink2MtpQ8Multi3P {
  uint32_t M, K, N_tot, nblk;
  uint32_t offs[4];
};
struct Ink2MtpQkP {
  uint32_t T, Hq, Hkv, D, pos0, kv_cap, is_swa, mode;
  float eps, theta;
};
struct Ink2MtpAttnP {
  uint32_t B, P, kv_cap, Hq, Hkv, D, is_swa, swa;
  float scale;
};
struct Ink2MtpAttnSplitP {
  uint32_t B, P, kv_cap, Hq, Hkv, D, is_swa, swa, chunk, nchunks;
  float scale;
};
struct Ink2MtpVecP {
  uint32_t n;
};
struct Ink2MtpRnnPrepP {
  uint32_t rank, H, tok_pos, first_step;
};
struct Ink2MtpRnnUpdP {
  uint32_t rank, first_step;
};
struct Ink2MtpFoldP {
  uint32_t V, R, H;
};
struct Ink2MtpArgmaxP {
  uint32_t V;
};

const json* find_mtp_json(const json& root) {
  if (root.contains("mtp_config") && root["mtp_config"].is_object()) return &root["mtp_config"];
  if (root.contains("text_config") && root["text_config"].is_object()) {
    const json& tc = root["text_config"];
    if (tc.contains("mtp_config") && tc["mtp_config"].is_object()) return &tc["mtp_config"];
  }
  return nullptr;
}

float tensor_elem_f32(const Tensor& t, size_t idx) {
  if (t.dtype == DType::F32) return t.buf.as<float>()[idx];
  if (t.dtype == DType::BF16) return bf16_to_f32(t.buf.as<uint16_t>()[idx]);
  if (t.dtype == DType::F16) {
    uint16_t h = t.buf.as<uint16_t>()[idx];
    uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t frac = h & 0x3ffu;
    uint32_t out = 0;
    if (exp == 0) {
      if (frac != 0) {
        int e = -14;
        while ((frac & 0x400u) == 0) {
          frac <<= 1;
          --e;
        }
        frac &= 0x3ffu;
        out = sign | (uint32_t(e + 127) << 23) | (frac << 13);
      } else {
        out = sign;
      }
    } else if (exp == 31) {
      out = sign | 0x7f800000u | (frac << 13);
    } else {
      out = sign | ((exp + 112u) << 23) | (frac << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
  }
  fatal("ink2_mtp: unsupported scalar dtype");
}

uint32_t pcg_hash(uint32_t v) {
  uint32_t state = v * 747796405u + 2891336453u;
  uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

struct SparseDist {
  std::vector<std::pair<uint32_t, float>> items;  // (token_id, prob)
  float prob(uint32_t id) const {
    for (const auto& kv : items)
      if (kv.first == id) return kv.second;
    return 0.0f;
  }
};

SparseDist compute_row_dist(const float* logits, uint32_t V, const SamplingParams& sp) {
  SparseDist d;
  const uint32_t Kc = (!full_vocab(sp)) ? topk_k(sp)
                                        : (sp.top_k > 0 && uint32_t(sp.top_k) < V ? uint32_t(sp.top_k) : V);
  struct Cand {
    float v;
    uint32_t id;
  };
  auto better = [](const Cand& a, const Cand& b) { return a.v > b.v || (a.v == b.v && a.id < b.id); };

  std::vector<Cand> top;
  float max_v = -INFINITY;
  if (Kc < V && Kc <= 256u) {
    // Min-heap of size Kc storing the worst item at heap[0]
    auto heap_cmp = [&](const Cand& a, const Cand& b) { return better(a, b); };
    top.reserve(Kc);
    for (uint32_t i = 0; i < V; ++i) {
      const float lv = std::isnan(logits[i]) ? -INFINITY : logits[i];
      if (lv > max_v) max_v = lv;
      const Cand c{lv, i};
      if (top.size() < Kc) {
        top.push_back(c);
        std::push_heap(top.begin(), top.end(), heap_cmp);
      } else if (better(c, top[0])) {
        std::pop_heap(top.begin(), top.end(), heap_cmp);
        top.back() = c;
        std::push_heap(top.begin(), top.end(), heap_cmp);
      }
    }
    std::sort(top.begin(), top.end(), better);
  } else {
    top.reserve(V);
    for (uint32_t i = 0; i < V; ++i) {
      const float lv = std::isnan(logits[i]) ? -INFINITY : logits[i];
      if (lv > max_v) max_v = lv;
      if (lv > -INFINITY) top.push_back(Cand{lv, i});
    }
    if (top.empty()) {
      d.items.emplace_back(0u, 1.0f);
      return d;
    }
    if (Kc < top.size()) {
      std::partial_sort(top.begin(), top.begin() + Kc, top.end(), better);
      top.resize(Kc);
    } else {
      std::sort(top.begin(), top.end(), better);
    }
  }
  if (!(max_v > -INFINITY) || top.empty()) {
    d.items.emplace_back(0u, 1.0f);
    return d;
  }

  const size_t n = top.size();
  std::vector<float> unt(n);
  double tot = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const float w = std::exp(top[i].v - top[0].v);
    unt[i] = w;
    tot += double(w);
  }
  size_t keep = n;
  double cum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    cum += double(unt[i]) / tot;
    if (cum >= double(sp.top_p)) {
      keep = i + 1;
      break;
    }
  }
  d.items.reserve(keep);
  double kt = 0.0;
  for (size_t i = 0; i < keep; ++i) {
    const float w = std::exp((top[i].v - top[0].v) / sp.temperature);
    d.items.emplace_back(top[i].id, w);
    kt += double(w);
  }
  const float inv = kt > 0.0 ? float(1.0 / kt) : 1.0f;
  for (auto& kv : d.items) kv.second *= inv;
  return d;
}

}  // namespace

int ink2_mtp_accept_cpu(int* tokens, int P, int K, int V, const float* verify_logits, const float* draft_logits,
                        const int* verify_samples, const SamplingParams& sp, uint32_t step) {
  if (sp.temperature <= 0.0f) {
    int a = 0;
    while (a < K && tokens[P + 1 + a] == verify_samples[a]) ++a;
    tokens[P + 1 + a] = verify_samples[a];
    return a;
  }

  const uint32_t Vu = uint32_t(V);
  int a = 0;
  for (; a < K; ++a) {
    const uint32_t dt = uint32_t(tokens[P + 1 + a]);
    const SparseDist p = compute_row_dist(verify_logits + size_t(a) * Vu, Vu, sp);
    const SparseDist q = compute_row_dist(draft_logits + size_t(a) * Vu, Vu, sp);
    const float pa = p.prob(dt);
    const float qa = q.prob(dt);
    const float u_acc =
        float(pcg_hash(sp.seed ^ pcg_hash(step + uint32_t(a)) ^ 0x9e3779b9u) & 0xffffffu) / 16777216.0f;
    if (qa > 0.0f && u_acc * qa < pa) continue;

    // Rejected: draw correction token from residual max(0, p(x) - q(x)) with independent salt distinct from
    // both the draft sampler's uniform hash and u_acc.
    std::vector<std::pair<uint32_t, float>> res;
    res.reserve(p.items.size());
    double rsum = 0.0;
    for (const auto& kv : p.items) {
      const float r = kv.second - q.prob(kv.first);
      if (r > 0.0f) {
        res.emplace_back(kv.first, r);
        rsum += double(r);
      }
    }
    uint32_t chosen = p.items.empty() ? 0u : p.items[0].first;
    const float u_res =
        float(pcg_hash(sp.seed ^ pcg_hash(step + uint32_t(a)) ^ 0x517cc1b7u) & 0xffffffu) / 16777216.0f;
    if (rsum > 0.0) {
      const double target = double(u_res) * rsum;
      double acc = 0.0;
      chosen = res.back().first;
      for (const auto& kv : res) {
        acc += double(kv.second);
        if (target < acc) {
          chosen = kv.first;
          break;
        }
      }
    } else if (!p.items.empty()) {
      double acc = 0.0;
      chosen = p.items.back().first;
      for (const auto& kv : p.items) {
        acc += double(kv.second);
        if (double(u_res) < acc) {
          chosen = kv.first;
          break;
        }
      }
    }
    tokens[P + 1 + a] = int(chosen);
    return a;
  }
  tokens[P + 1 + K] = verify_samples[K];
  return K;
}

bool Ink2Mtp::exists(const std::string& dir) {
  const auto rp = json_file(dir + "/config.json");
  if (!rp || !rp->is_object()) return false;
  if (!find_mtp_json(*rp)) return false;
  std::error_code ec;
  return std::filesystem::is_regular_file(dir + "/mtp.safetensors", ec);
}

Ink2MtpConfig Ink2Mtp::read_config(const std::string& dir) {
  const auto rp = json_file(dir + "/config.json");
  if (!rp || !rp->is_object()) throw std::runtime_error(dir + ": cannot read config.json");
  const json* mp = find_mtp_json(*rp);
  if (!mp) throw std::runtime_error(dir + ": missing mtp_config in config.json");
  const json& mc = *mp;
  Ink2MtpConfig c;
  c.hidden = mc.value("hidden_size", 2304);
  c.num_layers = mc.value("num_hidden_layers", 3);
  c.hq = mc.value("num_attention_heads", 24);
  c.hkv = mc.value("num_key_value_heads", 8);
  c.head_dim = mc.value("head_dim", 128);
  c.ffn = mc.value("intermediate_size", 9216);
  c.swa = mc.value("sliding_window_size", 1024);
  c.block = mc.value("block_size", 8);
  c.markov_rank = mc.value("markov_rank", 256);
  c.rope_theta = mc.value("rope_theta", 1000000.0f);
  c.rms_eps = mc.value("rms_norm_eps", 1e-6f);
  c.is_local.assign(size_t(c.num_layers), false);
  if (mc.contains("local_layer_ids") && mc["local_layer_ids"].is_array()) {
    for (const auto& lid : mc["local_layer_ids"]) {
      const int idx = lid.get<int>();
      if (idx >= 0 && idx < c.num_layers) c.is_local[size_t(idx)] = true;
    }
  }
  if (mc.contains("target_layer_ids") && mc["target_layer_ids"].is_array()) {
    for (const auto& tid : mc["target_layer_ids"]) c.taps.push_back(tid.get<int>());
  } else {
    c.taps = {5, 23, 41, 48};
  }
  return c;
}

size_t Ink2Mtp::kv_per_pos(const Ink2MtpConfig& c) {
  size_t n_global = 0;
  for (int i = 0; i < c.num_layers; ++i)
    if (!c.is_local[size_t(i)]) ++n_global;
  return n_global * 2 * size_t(c.hkv) * size_t(c.head_dim) * 2;
}

size_t Ink2Mtp::ring_bytes_per_seq(const Ink2MtpConfig& c, int ring_rows) {
  size_t n_local = 0;
  for (int i = 0; i < c.num_layers; ++i)
    if (c.is_local[size_t(i)]) ++n_local;
  return n_local * 2 * size_t(ring_rows) * size_t(c.hkv) * size_t(c.head_dim) * 2 + size_t(c.hidden) * 4;
}

size_t Ink2Mtp::plan_bytes(const std::string& dir, const Ink2Config& trunk_cfg) {
  const Ink2MtpConfig c = read_config(dir);
  std::error_code ec;
  const size_t fsz = size_t(std::filesystem::file_size(dir + "/mtp.safetensors", ec));
  const size_t fold = size_t(trunk_cfg.vocab) * size_t(c.markov_rank) * 2 + size_t(trunk_cfg.vocab) * 4;
  const size_t acts = size_t(c.block - 1) * size_t(trunk_cfg.vocab) * 4 + (32u << 20);
  return fsz + fold + acts;
}

Ink2Mtp::Ink2Mtp(Inkling2& trunk, const std::string& dir, const Ink2Config& trunk_cfg, const Tensor& unembed, int tmax,
                 int ring_rows)
    : trunk_(trunk), cfg_(read_config(dir)), vocab_(trunk_cfg.vocab), tmax_(tmax), ring_rows_(ring_rows) {
  ink2::compile_kernels();
  embed_ = trunk.embed_;
  embed_norm_ = trunk.embed_norm_;
  unembed_ = &unembed;

  const double t0 = now_ms();
  ws_.open_dir(dir, nullptr, [](const std::string& fn) { return fn == "mtp.safetensors"; });
  auto req = [&](const std::string& name) -> const Tensor* {
    const Tensor* t = ws_.find(name);
    if (!t) fatal("ink2_mtp: missing tensor %s in %s/mtp.safetensors", name.c_str(), dir.c_str());
    return t;
  };

  const Tensor* fuse = req("fuse_weight");
  if (fuse->numel() != 16) fatal("ink2_mtp: unexpected fuse_weight shape");
  for (int r = 0; r < 4; ++r) {
    float row[4], mx = -INFINITY;
    for (int c = 0; c < 4; ++c) {
      row[c] = tensor_elem_f32(*fuse, size_t(r) * 4 + size_t(c));
      mx = std::max(mx, row[c]);
    }
    float sum = 0.0f;
    for (int c = 0; c < 4; ++c) {
      row[c] = std::exp(row[c] - mx);
      sum += row[c];
    }
    for (int c = 0; c < 4; ++c) fuse_w_[r][c] = row[c] / sum;
  }

  mask_embed_ = req("mask_embed");
  norm_ = req("norm.weight");
  rec_norm_ = req("markov_head.dflare_norm.weight");
  markov_w1_ = req("markov_head.markov_w1.weight");
  rnn_proj_w_ = req("markov_head.rnn_proj.weight");
  rnn_proj_b_ = req("markov_head.rnn_proj.bias");
  hidden_out_w_ = req("markov_head.hidden_out.weight");
  hidden_out_b_ = req("markov_head.hidden_out.bias");

  layers_.resize(size_t(cfg_.num_layers));
  for (int i = 0; i < cfg_.num_layers; ++i) {
    Ink2MtpLayer& L = layers_[size_t(i)];
    const std::string p = "layers." + std::to_string(i) + ".";
    const std::string a = p + "self_attn.";
    L.is_local = cfg_.is_local[size_t(i)];
    L.in_norm = req(p + "input_layernorm.weight");
    L.ff_norm = req(p + "post_attention_layernorm.weight");
    L.ctx_norm = req(p + "dflare_norm.weight");
    L.q_norm = req(a + "q_norm.weight");
    L.k_norm = req(a + "k_norm.weight");
    L.k_norm_ctx = req(a + "k_norm_ctx.weight");
    L.q_proj = req(a + "q_proj.weight");
    L.k_proj = req(a + "k_proj.weight");
    L.v_proj = req(a + "v_proj.weight");
    L.o_proj = req(a + "o_proj.weight");
    L.k_proj_ctx = req(a + "k_proj_ctx.weight");
    L.v_proj_ctx = req(a + "v_proj_ctx.weight");
    L.gate_proj = req(p + "mlp.gate_proj.weight");
    L.up_proj = req(p + "mlp.up_proj.weight");
    L.down_proj = req(p + "mlp.down_proj.weight");
  }

  Gpu& g = Gpu::get();
  auto alloc = [&](size_t bytes, const char* lbl) {
    Buffer b = g.alloc(std::max<size_t>(bytes, 16), lbl);
    owned_bufs_.push_back(b);
    return b;
  };

  const size_t H = size_t(cfg_.hidden);
  const size_t R = size_t(cfg_.markov_rank);
  const size_t V = size_t(vocab_);
  const size_t B = size_t(cfg_.block);
  const size_t QD = size_t(cfg_.hq) * size_t(cfg_.head_dim);
  const size_t KVD = size_t(cfg_.hkv) * size_t(cfg_.head_dim);

  if ((H & 7) == 0 && (QD & 7) == 0 && (KVD & 7) == 0 && (cfg_.ffn & 7) == 0) {
    for (int i = 0; i < cfg_.num_layers; ++i) {
      Ink2MtpLayer& L = layers_[size_t(i)];
      if (L.q_proj->dtype == DType::BF16 && L.k_proj->dtype == DType::BF16 && L.v_proj->dtype == DType::BF16) {
        L.qkv_offs[0] = 0;
        L.qkv_offs[1] = uint32_t(QD);
        L.qkv_offs[2] = uint32_t(QD + KVD);
        L.qkv_offs[3] = uint32_t(QD + 2 * KVD);
        L.qkv_w = alloc(size_t(L.qkv_offs[3]) * H * 2, "ink2_mtp_qkv_w");
        uint8_t* dst = L.qkv_w.as<uint8_t>();
        std::memcpy(dst + size_t(L.qkv_offs[0]) * H * 2, L.q_proj->buf.as<uint8_t>(), QD * H * 2);
        std::memcpy(dst + size_t(L.qkv_offs[1]) * H * 2, L.k_proj->buf.as<uint8_t>(), KVD * H * 2);
        std::memcpy(dst + size_t(L.qkv_offs[2]) * H * 2, L.v_proj->buf.as<uint8_t>(), KVD * H * 2);
      }
      if (L.gate_proj->dtype == DType::BF16 && L.up_proj->dtype == DType::BF16) {
        const size_t F = size_t(cfg_.ffn);
        L.gate_up_w = alloc(2 * F * H * 2, "ink2_mtp_gu_w");
        uint8_t* dst = L.gate_up_w.as<uint8_t>();
        const uint8_t* gs = L.gate_proj->buf.as<const uint8_t>();
        const uint8_t* us = L.up_proj->buf.as<const uint8_t>();
        const size_t row_b = H * 2;
        for (size_t j = 0; j < F; ++j) {
          std::memcpy(dst + (2 * j + 0) * row_b, gs + j * row_b, row_b);
          std::memcpy(dst + (2 * j + 1) * row_b, us + j * row_b, row_b);
        }
      }
    }
  }

  // Optional Q8/Q4 load-time quantization of DSpark 3L dense layers (NI_INK2_MTP_W=q4|q8|bf16).
  w_mode_ = WeightMode::Q4;
  if (const char* we = getenv("NI_INK2_MTP_W"); we && *we) {
    if (!strcmp(we, "bf16")) w_mode_ = WeightMode::Bf16;
    else if (!strcmp(we, "q8")) w_mode_ = WeightMode::Q8;
    else if (!strcmp(we, "q4")) w_mode_ = WeightMode::Q4;
  }
  const char* scr_env = getenv("NI_INK2_MTP_SCREEN");
  screen_on_ = !(scr_env && *scr_env && atoi(scr_env) == 0);
  if (w_mode_ != WeightMode::Bf16 && (H & 255) == 0 && (QD & 255) == 0 && (cfg_.ffn & 255) == 0) {
    auto e_q = g.begin();
    auto quant_mat = [&](const Buffer& src, size_t rows, size_t cols, Buffer& out_q, Buffer& out_s, const char* tag) {
      const uint32_t tblks = uint32_t(rows * (cols / 32));
      out_q = alloc(w_mode_ == WeightMode::Q8 ? rows * cols : size_t(tblks) * 16, tag);
      out_s = alloc(size_t(tblks) * 2, tag);
      const Ink2MtpQuantQ8P qp{tblks};
      e_q->use(mtp_pipe(w_mode_ == WeightMode::Q8 ? "ink2_quant_q8_bf16" : "ink2_quant_q4_bf16"));
      e_q->buf(0, src);
      e_q->buf(1, out_q);
      e_q->buf(2, out_s);
      e_q->val(3, qp);
      e_q->dispatch((tblks + 3) / 4, 1, 1, 128);
    };
    for (int i = 0; i < cfg_.num_layers; ++i) {
      Ink2MtpLayer& L = layers_[size_t(i)];
      if (L.qkv_w) quant_mat(L.qkv_w, L.qkv_offs[3], H, L.qkv_q, L.qkv_s, "ink2_mtp_qkv_q");
      if (L.o_proj && L.o_proj->dtype == DType::BF16) quant_mat(L.o_proj->buf, H, QD, L.o_q, L.o_s, "ink2_mtp_o_q");
      if (L.gate_up_w) quant_mat(L.gate_up_w, 2 * size_t(cfg_.ffn), H, L.gu_q, L.gu_s, "ink2_mtp_gu_q");
      if (L.down_proj && L.down_proj->dtype == DType::BF16)
        quant_mat(L.down_proj->buf, H, size_t(cfg_.ffn), L.down_q, L.down_s, "ink2_mtp_down_q");
    }
    e_q->commit_and_wait();
  } else if (w_mode_ != WeightMode::Bf16) {
    w_mode_ = WeightMode::Bf16;
  }

  // Pre-fold markov_head.hidden_out (weight [H, R] + bias [H]) with unembed [V, H].
  fold_w_ = alloc(V * R * 2, "ink2_mtp_fold_w");
  fold_b_ = alloc(V * 4, "ink2_mtp_fold_b");
  {
    Buffer lat_wt = g.alloc(R * H * 2, "ink2_mtp_tmp_wt");
    Buffer lat_b = g.alloc(H * 4, "ink2_mtp_tmp_b");
    uint16_t* wt_dst = lat_wt.as<uint16_t>();
    if (hidden_out_w_->dtype == DType::BF16) {
      const uint16_t* src = hidden_out_w_->buf.as<uint16_t>();
      for (size_t h = 0; h < H; ++h)
        for (size_t r = 0; r < R; ++r) wt_dst[r * H + h] = src[h * R + r];
    } else {
      for (size_t h = 0; h < H; ++h)
        for (size_t r = 0; r < R; ++r) wt_dst[r * H + h] = f32_to_bf16(tensor_elem_f32(*hidden_out_w_, h * R + r));
    }
    float* b_dst = lat_b.as<float>();
    for (size_t h = 0; h < H; ++h) b_dst[h] = tensor_elem_f32(*hidden_out_b_, h);

    if (unembed.dtype == DType::BF16) {
      auto e = g.begin();
      const Ink2MtpFoldP fp{uint32_t(V), uint32_t(R), uint32_t(H)};
      e->use(mtp_pipe("ink2_mtp_fold_head"));
      e->buf(0, unembed.buf);
      e->buf(1, lat_wt);
      e->buf(2, fold_w_);
      e->val(3, fp);
      e->dispatch((uint32_t(V) + 3) / 4, (uint32_t(R) + 15) / 16, 1, 128);
      trunk_.enc_mm(*e, unembed, lat_b, fold_b_, 1, uint32_t(V), false);
      const char* q8_env = getenv("NI_INK2_MTP_Q8");
      if (R == 256 && !(q8_env && *q8_env && atoi(q8_env) == 0)) {
        const uint32_t tblks = uint32_t(V * (R / 32));
        fold_q_ = alloc(V * R, "ink2_mtp_fold_q");
        fold_s_ = alloc(size_t(tblks) * 2, "ink2_mtp_fold_s");
        const Ink2MtpQuantQ8P qp{tblks};
        e->use(mtp_pipe("ink2_quant_q8_bf16"));
        e->buf(0, fold_w_);
        e->buf(1, fold_q_);
        e->buf(2, fold_s_);
        e->val(3, qp);
        e->dispatch((tblks + 3) / 4, 1, 1, 128);
      }
      e->commit_and_wait();
    } else {
      auto e = g.begin();
      trunk_.enc_mm(*e, unembed, lat_b, fold_b_, 1, uint32_t(V), false);
      e->commit_and_wait();
      uint16_t* dst = fold_w_.as<uint16_t>();
      for (size_t v = 0; v < V; ++v) {
        for (size_t r = 0; r < R; ++r) {
          double acc = 0.0;
          for (size_t h = 0; h < H; ++h)
            acc += double(tensor_elem_f32(unembed, v * H + h)) * double(bf16_to_f32(wt_dst[r * H + h]));
          dst[v * R + r] = f32_to_bf16(float(acc));
        }
      }
    }
    g.release(lat_wt);
    g.release(lat_b);
  }

  // Pack 2-bit symmetric proxies for unembed and fold_w when greedy screening is enabled.
  if (screen_on_ && (H & 31) == 0 && R == 256 && B > 1) {
    const uint32_t u_blks = uint32_t(V * (H / 32));
    const uint32_t f_blks = uint32_t(V * (R / 32));
    unembed_p2_q_ = alloc(size_t(u_blks) * 8, "ink2_mtp_u_p2q");
    unembed_p2_s_ = alloc(size_t(u_blks) * 2, "ink2_mtp_u_p2s");
    fold_p2_q_ = alloc(size_t(f_blks) * 8, "ink2_mtp_f_p2q");
    fold_p2_s_ = alloc(size_t(f_blks) * 2, "ink2_mtp_f_p2s");
    base_proxy_ = alloc((B - 1) * V * 2, "ink2_mtp_base_p2");
    auto ep = g.begin();
    const Ink2MtpQuantP2P up{u_blks, unembed.dtype == DType::F32 ? 1u : 0u};
    ep->use(mtp_pipe("ink2_quant_p2_w"));
    ep->buf(0, unembed.buf);
    ep->buf(1, unembed_p2_q_);
    ep->buf(2, unembed_p2_s_);
    ep->val(3, up);
    ep->dispatch((u_blks + 3) / 4, 1, 1, 128);

    const Ink2MtpQuantP2P fp2{f_blks, 0u};
    ep->use(mtp_pipe("ink2_quant_p2_w"));
    ep->buf(0, fold_w_);
    ep->buf(1, fold_p2_q_);
    ep->buf(2, fold_p2_s_);
    ep->val(3, fp2);
    ep->dispatch((f_blks + 3) / 4, 1, 1, 128);
    ep->commit_and_wait();
  } else {
    screen_on_ = false;
  }

  ctx_h_ = alloc(size_t(tmax_) * H * 4, "ink2_mtp_ctx_h");
  ctx_k_ = alloc(size_t(tmax_) * KVD * 4, "ink2_mtp_ctx_k");
  ctx_v_ = alloc(size_t(tmax_) * KVD * 4, "ink2_mtp_ctx_v");

  blk_x_ = alloc(B * H * 4, "ink2_mtp_blk_x");
  blk_res_ = alloc(B * H * 4, "ink2_mtp_blk_res");
  blk_norm_ = alloc(B * H * 4, "ink2_mtp_blk_norm");
  blk_q_ = alloc(B * QD * 4, "ink2_mtp_blk_q");
  blk_k_ = alloc(B * KVD * 4, "ink2_mtp_blk_k");
  blk_v_ = alloc(B * KVD * 4, "ink2_mtp_blk_v");
  blk_att_ = alloc(B * QD * 4, "ink2_mtp_blk_att");
  blk_gate_ = alloc(B * size_t(cfg_.ffn) * 4, "ink2_mtp_blk_gate");
  blk_up_ = alloc(B * size_t(cfg_.ffn) * 4, "ink2_mtp_blk_up");
  blk_out_ = alloc(B * H * 4, "ink2_mtp_blk_out");
  part_m_ = alloc(B * size_t(cfg_.hq) * 64u * 4u, "ink2_mtp_part_m");
  part_l_ = alloc(B * size_t(cfg_.hq) * 64u * 4u, "ink2_mtp_part_l");
  part_o_ = alloc(B * size_t(cfg_.hq) * 64u * size_t(cfg_.head_dim) * 4u, "ink2_mtp_part_o");

  state_ = alloc(R * 4, "ink2_mtp_state");
  latent_ = alloc(R * 4, "ink2_mtp_latent");
  rnn_in_ = alloc((2 * R + 2 * H) * 4, "ink2_mtp_rnn_in");
  rnn_gates_ = alloc(3 * R * 4, "ink2_mtp_rnn_gates");
  draft_logits_ = alloc((B - 1) * V * 4, "ink2_mtp_draft_logits");
  d_cand_v_ = alloc((V / 2048 + 2) * 64 * 4, "ink2_mtp_cand_v");
  d_cand_i_ = alloc((V / 2048 + 2) * 64 * 4, "ink2_mtp_cand_i");
  d_sprob_ = alloc(64 * 4, "ink2_mtp_sprob");

  g.make_resident(ws_.buffers());
  g.make_resident(owned_bufs_);
  (void)mtp_pipe("ink2_mtp_attn_splitk");
  (void)mtp_pipe("ink2_mtp_attn_reduce");
  logi("Ink2Mtp loaded: 3 layers (block=%d, markov_rank=%d, taps=[%d,%d,%d,%d], w=%s, screen=%s), %.2f GB + %.0f MB folded in %.0f ms",
       cfg_.block, cfg_.markov_rank, cfg_.taps[0], cfg_.taps[1], cfg_.taps[2], cfg_.taps[3],
       w_mode_ == WeightMode::Q4 ? "q4" : (w_mode_ == WeightMode::Q8 ? "q8" : "bf16"),
       screen_on_ ? "p2" : "full",
       ws_.total_bytes() / 1e9, (fold_w_.size + fold_b_.size) / 1e6, now_ms() - t0);
}

Ink2Mtp::~Ink2Mtp() {
  Gpu& g = Gpu::get();
  for (Buffer& b : owned_bufs_) g.release(b);
}

void Ink2Mtp::enc_mv(Encoder& e, const Tensor& w, const Buffer& x, const Buffer& y, int M, const Tensor* bias,
                     const Buffer* f32_bias, uint32_t n_limit, bool add, float softcap) const {
  if (M > 8 && !bias && !f32_bias && !add && softcap == 0.0f) {
    trunk_.enc_mm(e, w, x, y, M, n_limit, false);
    return;
  }
  const uint32_t N = n_limit ? n_limit : uint32_t(w.shape[0]);
  const uint32_t K = uint32_t(w.shape[1]);
  const Buffer& bbuf = f32_bias ? *f32_bias : (bias ? bias->buf : y);
  const uint32_t has_b = (f32_bias || bias) ? 1u : 0u;
  const uint32_t b_f32 = f32_bias ? 1u : (bias && bias->dtype == DType::F32 ? 1u : 0u);
  for (int m0 = 0; m0 < M; m0 += 8) {
    const uint32_t mc = uint32_t(std::min(8, M - m0));
    const Ink2MtpMmP p{mc, N, K, K, N, has_b, b_f32, add ? 1u : 0u, softcap};
    if (mc <= 4u) {
      e.use(mtp_pipe("ink2_mtp_bf16_mv4"));
      e.buf(0, x.view(size_t(m0) * K * 4, size_t(mc) * K * 4));
      e.buf(1, w.buf);
      e.buf(2, bbuf);
      e.buf(3, y.view(size_t(m0) * N * 4, size_t(mc) * N * 4));
      e.val(4, p);
      e.dispatch((N + 15) / 16, 1, 1, 128);
    } else if ((K & 255u) == 0u) {
      e.use(mtp_pipe("ink2_mtp_bf16_mv8"));
      e.buf(0, x.view(size_t(m0) * K * 4, size_t(mc) * K * 4));
      e.buf(1, w.buf);
      e.buf(2, bbuf);
      e.buf(3, y.view(size_t(m0) * N * 4, size_t(mc) * N * 4));
      e.val(4, p);
      e.dispatch((N + 15) / 16, 1, 1, 128);
    } else {
      e.use(mtp_pipe("ink2_mtp_bf16_mv"));
      e.buf(0, x.view(size_t(m0) * K * 4, size_t(mc) * K * 4));
      e.buf(1, w.buf);
      e.buf(2, bbuf);
      e.buf(3, y.view(size_t(m0) * N * 4, size_t(mc) * N * 4));
      e.val(4, p);
      e.dispatch((N + 7) / 8, 1, 1, 128);
    }
  }
}

void Ink2Mtp::enc_mv_q(Encoder& e, const Buffer& wq, const Buffer& ws, uint32_t N, uint32_t K, const Buffer& x,
                       const Buffer& y, int M, bool add) const {
  const char* kname = w_mode_ == WeightMode::Q8 ? "ink2_mtp_q8_mv8" : "ink2_mtp_q4_mv8";
  const uint32_t nblk = K / 32u;
  for (int m0 = 0; m0 < M; m0 += 8) {
    const uint32_t mc = uint32_t(std::min(8, M - m0));
    const Ink2MtpQ8MmP p{mc, N, K, nblk, add ? 1u : 0u};
    e.use(mtp_pipe(kname));
    e.buf(0, x.view(size_t(m0) * K * 4, size_t(mc) * K * 4));
    e.buf(1, wq);
    e.buf(2, ws);
    e.buf(3, y.view(size_t(m0) * N * 4, size_t(mc) * N * 4));
    e.val(4, p);
    e.dispatch((N + 15u) / 16u, 1, 1, 128);
  }
}

void Ink2Mtp::enc_rmsnorm(Encoder& e, const Buffer& x, const Tensor& w, const Buffer& out, int T) const {
  const Ink2MtpNormP p{uint32_t(T), uint32_t(cfg_.hidden), cfg_.rms_eps};
  e.use(mtp_pipe("ink2_mtp_rmsnorm"));
  e.buf(0, x);
  e.buf(1, w.buf);
  e.buf(2, out);
  e.val(3, p);
  e.dispatch(uint32_t(T), 1, 1, 512);
}

void Ink2Mtp::encode_commit_context(Encoder& e, Seq& s, int pos0, int T, const Buffer& tap0, const Buffer& tap1,
                                    const Buffer& tap2, const Buffer& tap3) {
  if (T <= 0) return;
  const uint32_t H = uint32_t(cfg_.hidden);
  const size_t nv = trunk_.cfg_.schedule.size();

  for (int i = 0; i < cfg_.num_layers; ++i) {
    const Ink2MtpLayer& L = layers_[size_t(i)];
    const Ink2MtpFuseP fp{uint32_t(T), H, fuse_w_[i][0], fuse_w_[i][1], fuse_w_[i][2], fuse_w_[i][3], cfg_.rms_eps};
    e.use(mtp_pipe("ink2_mtp_fuse_norm"));
    e.buf(0, tap0);
    e.buf(1, tap1);
    e.buf(2, tap2);
    e.buf(3, tap3);
    e.buf(4, L.ctx_norm->buf);
    e.buf(5, ctx_h_);
    e.val(6, fp);
    e.dispatch(uint32_t(T), 1, 1, 512);

    e.group_begin();
    enc_mv(e, *L.k_proj_ctx, ctx_h_, ctx_k_, T);
    enc_mv(e, *L.v_proj_ctx, ctx_h_, ctx_v_, T);
    e.group_end();

    const uint32_t kv_cap = L.is_local ? uint32_t(ring_rows_) : uint32_t(s.cap);
    const Ink2MtpQkP qp{uint32_t(T),
                        uint32_t(cfg_.hq),
                        uint32_t(cfg_.hkv),
                        uint32_t(cfg_.head_dim),
                        uint32_t(pos0),
                        kv_cap,
                        L.is_local ? 1u : 0u,
                        0u,
                        cfg_.rms_eps,
                        cfg_.rope_theta};
    e.use(mtp_pipe("ink2_mtp_qk_rope_cache"));
    e.buf(0, blk_q_);
    e.buf(1, ctx_k_);
    e.buf(2, ctx_v_);
    e.buf(3, L.q_norm->buf);
    e.buf(4, L.k_norm_ctx->buf);
    e.buf(5, s.kc[nv + size_t(i)]);
    e.buf(6, s.vc[nv + size_t(i)]);
    e.val(7, qp);
    e.dispatch(uint32_t(cfg_.hkv), uint32_t(T), 1, uint32_t(cfg_.head_dim));
  }

  // Last row's RecurrentNorm fused feature -> s.conv[1]
  const size_t last_off = size_t(T - 1) * H * 4;
  const size_t row_bytes = size_t(H) * 4;
  const Ink2MtpFuseP rp{1u, H, fuse_w_[3][0], fuse_w_[3][1], fuse_w_[3][2], fuse_w_[3][3], cfg_.rms_eps};
  e.use(mtp_pipe("ink2_mtp_fuse_norm"));
  e.buf(0, tap0.view(last_off, row_bytes));
  e.buf(1, tap1.view(last_off, row_bytes));
  e.buf(2, tap2.view(last_off, row_bytes));
  e.buf(3, tap3.view(last_off, row_bytes));
  e.buf(4, rec_norm_->buf);
  e.buf(5, s.conv[1]);
  e.val(6, rp);
  e.dispatch(1, 1, 1, 512);
}

void Ink2Mtp::encode_draft(Encoder& e, Seq& s, int P, int K, const SamplingParams& sp, uint32_t step) {
  const int B = cfg_.block;
  const int Kd = std::clamp(K, 1, B - 1);
  const uint32_t H = uint32_t(cfg_.hidden);
  const uint32_t QD = uint32_t(cfg_.hq * cfg_.head_dim);
  const uint32_t R = uint32_t(cfg_.markov_rank);
  const uint32_t V = uint32_t(vocab_);
  const size_t nv = trunk_.cfg_.schedule.size();

  {
    const Ink2MtpPrepP pp{uint32_t(B), H, uint32_t(P), cfg_.rms_eps};
    e.use(mtp_pipe("ink2_mtp_prep_block"));
    e.buf(0, s.tok);
    e.buf(1, embed_->buf);
    e.buf(2, embed_norm_->buf);
    e.buf(3, mask_embed_->buf);
    e.buf(4, blk_x_);
    e.val(5, pp);
    e.dispatch(uint32_t(B), 1, 1, 512);
  }

  const float attn_scale = 1.0f / std::sqrt(float(cfg_.head_dim));
  const uint32_t n4_res = uint32_t(B) * H / 4;
  const uint32_t n4_ffn = uint32_t(B) * uint32_t(cfg_.ffn) / 4;
  for (int i = 0; i < cfg_.num_layers; ++i) {
    const Ink2MtpLayer& L = layers_[size_t(i)];
    enc_rmsnorm(e, blk_x_, *L.in_norm, blk_norm_, B);

    if (L.qkv_q && B <= 8) {
      const Ink2MtpQ8Multi3P mp{uint32_t(B), H, L.qkv_offs[3], H / 32u,
                                {L.qkv_offs[0], L.qkv_offs[1], L.qkv_offs[2], L.qkv_offs[3]}};
      e.use(mtp_pipe(w_mode_ == WeightMode::Q8 ? "ink2_mtp_q8_mv8_multi3" : "ink2_mtp_q4_mv8_multi3"));
      e.buf(0, blk_norm_);
      e.buf(1, L.qkv_q);
      e.buf(2, L.qkv_s);
      e.buf(3, blk_q_);
      e.buf(4, blk_k_);
      e.buf(5, blk_v_);
      e.val(6, mp);
      e.dispatch((mp.N_tot + 15) / 16, 1, 1, 128);
    } else if (L.qkv_w) {
      const Ink2MtpMulti3P mp{uint32_t(B), H, L.qkv_offs[3],
                              {L.qkv_offs[0], L.qkv_offs[1], L.qkv_offs[2], L.qkv_offs[3]}};
      if (B <= 4) {
        e.use(mtp_pipe("ink2_mtp_bf16_mv4_multi3"));
        e.buf(0, blk_norm_);
        e.buf(1, L.qkv_w);
        e.buf(2, blk_q_);
        e.buf(3, blk_k_);
        e.buf(4, blk_v_);
        e.val(5, mp);
        e.dispatch((mp.N_tot + 15) / 16, 1, 1, 128);
      } else if ((H & 255u) == 0u) {
        e.use(mtp_pipe("ink2_mtp_bf16_mv8_multi3"));
        e.buf(0, blk_norm_);
        e.buf(1, L.qkv_w);
        e.buf(2, blk_q_);
        e.buf(3, blk_k_);
        e.buf(4, blk_v_);
        e.val(5, mp);
        e.dispatch((mp.N_tot + 15) / 16, 1, 1, 128);
      } else {
        e.use(mtp_pipe("ink2_mtp_bf16_mv_multi3"));
        e.buf(0, blk_norm_);
        e.buf(1, L.qkv_w);
        e.buf(2, blk_q_);
        e.buf(3, blk_k_);
        e.buf(4, blk_v_);
        e.val(5, mp);
        e.dispatch((mp.N_tot + 7) / 8, 1, 1, 128);
      }
    } else {
      e.group_begin();
      enc_mv(e, *L.q_proj, blk_norm_, blk_q_, B);
      enc_mv(e, *L.k_proj, blk_norm_, blk_k_, B);
      enc_mv(e, *L.v_proj, blk_norm_, blk_v_, B);
      e.group_end();
    }

    const uint32_t kv_cap = L.is_local ? uint32_t(ring_rows_) : uint32_t(s.cap);
    {
      const Ink2MtpQkP qp{uint32_t(B),
                          uint32_t(cfg_.hq),
                          uint32_t(cfg_.hkv),
                          uint32_t(cfg_.head_dim),
                          uint32_t(P),
                          kv_cap,
                          L.is_local ? 1u : 0u,
                          1u,
                          cfg_.rms_eps,
                          cfg_.rope_theta};
      e.use(mtp_pipe("ink2_mtp_qk_rope_cache"));
      e.buf(0, blk_q_);
      e.buf(1, blk_k_);
      e.buf(2, blk_v_);
      e.buf(3, L.q_norm->buf);
      e.buf(4, L.k_norm->buf);
      e.buf(5, s.kc[nv + size_t(i)]);
      e.buf(6, s.vc[nv + size_t(i)]);
      e.val(7, qp);
      e.dispatch(uint32_t(cfg_.hq + cfg_.hkv), uint32_t(B), 1, uint32_t(cfg_.head_dim));
    }

    if (!L.is_local && P >= 256 && B <= 8) {
      const uint32_t lk = uint32_t(P);
      uint32_t chunk = std::max(64u, (lk + 63u) / 64u);
      chunk = (chunk + 3u) / 4u * 4u;
      const uint32_t nchunks = std::max(1u, (lk + chunk - 1u) / chunk);
      const Ink2MtpAttnSplitP ap{uint32_t(B),
                                 uint32_t(P),
                                 kv_cap,
                                 uint32_t(cfg_.hq),
                                 uint32_t(cfg_.hkv),
                                 uint32_t(cfg_.head_dim),
                                 0u,
                                 uint32_t(cfg_.swa),
                                 chunk,
                                 nchunks,
                                 attn_scale};
      e.use(mtp_pipe("ink2_mtp_attn_splitk"));
      e.buf(0, blk_q_);
      e.buf(1, s.kc[nv + size_t(i)]);
      e.buf(2, s.vc[nv + size_t(i)]);
      e.buf(3, blk_k_);
      e.buf(4, blk_v_);
      e.buf(5, part_m_);
      e.buf(6, part_l_);
      e.buf(7, part_o_);
      e.val(8, ap);
      e.dispatch(uint32_t(cfg_.hq), nchunks, 1, 128);

      e.use(mtp_pipe("ink2_mtp_attn_reduce"));
      e.buf(0, part_m_);
      e.buf(1, part_l_);
      e.buf(2, part_o_);
      e.buf(3, blk_att_);
      e.val(4, ap);
      e.dispatch(uint32_t(cfg_.hq), uint32_t(B), 1, 32);
    } else {
      const Ink2MtpAttnP ap{uint32_t(B),
                            uint32_t(P),
                            kv_cap,
                            uint32_t(cfg_.hq),
                            uint32_t(cfg_.hkv),
                            uint32_t(cfg_.head_dim),
                            L.is_local ? 1u : 0u,
                            uint32_t(cfg_.swa),
                            attn_scale};
      e.use(mtp_pipe("ink2_mtp_attn"));
      e.buf(0, blk_q_);
      e.buf(1, s.kc[nv + size_t(i)]);
      e.buf(2, s.vc[nv + size_t(i)]);
      e.buf(3, blk_k_);
      e.buf(4, blk_v_);
      e.buf(5, blk_att_);
      e.val(6, ap);
      e.dispatch(uint32_t(cfg_.hq), uint32_t(B), 1, 128);
    }

    if (L.o_q) enc_mv_q(e, L.o_q, L.o_s, H, QD, blk_att_, blk_out_, B);
    else enc_mv(e, *L.o_proj, blk_att_, blk_out_, B);
    {
      const Ink2MtpVecP vp{uint32_t(B) * H};
      e.use(mtp_pipe("ink2_mtp_res_add"));
      e.buf(0, blk_x_);
      e.buf(1, blk_out_);
      e.buf(2, blk_res_);
      e.val(3, vp);
      e.dispatch((n4_res + 255) / 256, 1, 1, 256);
    }

    enc_rmsnorm(e, blk_res_, *L.ff_norm, blk_norm_, B);
    if (L.gu_q && B <= 8) {
      const Ink2MtpQ8MmP mp{uint32_t(B), uint32_t(cfg_.ffn), H, H / 32u, 0u};
      e.use(mtp_pipe(w_mode_ == WeightMode::Q8 ? "ink2_mtp_q8_gate_up_swiglu8" : "ink2_mtp_q4_gate_up_swiglu8"));
      e.buf(0, blk_norm_);
      e.buf(1, L.gu_q);
      e.buf(2, L.gu_s);
      e.buf(3, blk_gate_);
      e.val(4, mp);
      e.dispatch((uint32_t(cfg_.ffn) + 7) / 8, 1, 1, 128);
    } else if (L.gate_up_w) {
      const Ink2MtpMmP mp{uint32_t(B), uint32_t(cfg_.ffn), H, H, uint32_t(cfg_.ffn), 0u, 0u, 0u, 0.0f};
      if (B >= 5 && B <= 8 && (H & 255u) == 0u) {
        e.use(mtp_pipe("ink2_mtp_gate_up_swiglu8"));
        e.buf(0, blk_norm_);
        e.buf(1, L.gate_up_w);
        e.buf(2, blk_gate_);
        e.val(3, mp);
        e.dispatch((uint32_t(cfg_.ffn) + 7) / 8, 1, 1, 128);
      } else {
        e.use(mtp_pipe("ink2_mtp_gate_up_swiglu"));
        e.buf(0, blk_norm_);
        e.buf(1, L.gate_up_w);
        e.buf(2, blk_gate_);
        e.val(3, mp);
        e.dispatch((uint32_t(cfg_.ffn) + 3) / 4, 1, 1, 128);
      }
    } else {
      e.group_begin();
      enc_mv(e, *L.gate_proj, blk_norm_, blk_gate_, B);
      enc_mv(e, *L.up_proj, blk_norm_, blk_up_, B);
      e.group_end();

      {
        const Ink2MtpVecP vp{uint32_t(B) * uint32_t(cfg_.ffn)};
        e.use(mtp_pipe("ink2_mtp_swiglu_mul"));
        e.buf(0, blk_gate_);
        e.buf(1, blk_up_);
        e.val(2, vp);
        e.dispatch((n4_ffn + 255) / 256, 1, 1, 256);
      }
    }

    if (L.down_q) enc_mv_q(e, L.down_q, L.down_s, H, uint32_t(cfg_.ffn), blk_gate_, blk_out_, B);
    else enc_mv(e, *L.down_proj, blk_gate_, blk_out_, B);
    {
      const Ink2MtpVecP vp{uint32_t(B) * H};
      e.use(mtp_pipe("ink2_mtp_res_add"));
      e.buf(0, blk_res_);
      e.buf(1, blk_out_);
      e.buf(2, blk_x_);
      e.val(3, vp);
      e.dispatch((n4_res + 255) / 256, 1, 1, 256);
    }
  }

  enc_rmsnorm(e, blk_x_, *norm_, blk_norm_, B);

  const bool use_p2 = screen_on_ && sp.temperature <= 0.0f && s.mask_rows == 0;
  if (use_p2) {
    const Ink2MtpP2BaseP bp{uint32_t(Kd), V, H, H / 32u};
    e.use(mtp_pipe("ink2_mtp_p2_unembed_base"));
    e.buf(0, blk_norm_.view(size_t(H) * 4, size_t(Kd) * H * 4));
    e.buf(1, unembed_p2_q_);
    e.buf(2, unembed_p2_s_);
    e.buf(3, fold_b_);
    e.buf(4, base_proxy_);
    e.val(5, bp);
    e.dispatch((V + 15u) / 16u, 1, 1, 128);
  } else {
    // Base unembed rows 1..Kd (raw pre-softcap logits into draft_logits_[0..Kd-1])
    trunk_.enc_unembed(e, blk_norm_.view(size_t(H) * 4, size_t(Kd) * H * 4), draft_logits_, Kd, false);
  }

  const uint32_t tot_in = 2u * R + 2u * H;
  const float softcap = trunk_.cfg_.logit_softcap;
  for (int i = 0; i < Kd; ++i) {
    {
      const Ink2MtpRnnPrepP rp{R, H, uint32_t(P + i), i == 0 ? 1u : 0u};
      e.use(mtp_pipe("ink2_mtp_rnn_prep"));
      e.buf(0, state_);
      e.buf(1, markov_w1_->buf);
      e.buf(2, s.tok);
      e.buf(3, blk_norm_.view(size_t(1 + i) * H * 4, size_t(H) * 4));
      e.buf(4, s.conv[1]);
      e.buf(5, rnn_in_);
      e.val(6, rp);
      e.dispatch((tot_in + 255) / 256, 1, 1, 256);
    }

    enc_mv(e, *rnn_proj_w_, rnn_in_, rnn_gates_, 1, rnn_proj_b_);

    {
      const Ink2MtpRnnUpdP up{R, i == 0 ? 1u : 0u};
      e.use(mtp_pipe("ink2_mtp_rnn_update"));
      e.buf(0, rnn_gates_);
      e.buf(1, state_);
      e.buf(2, latent_);
      e.val(3, up);
      e.dispatch(1, 1, 1, 256);
    }

    if (use_p2) {
      const uint32_t has_sup = s.suppression ? 1u : 0u;
      const uint32_t sup_words = has_sup ? uint32_t(s.suppression->words) : 0u;
      const Buffer& sup_buf = has_sup ? s.suppression->mask : d_cand_i_;

      const Ink2MtpP2ScreenP sp2{V, R, has_sup, sup_words};
      e.use(mtp_pipe("ink2_mtp_p2_fold_screen"));
      e.buf(0, latent_);
      e.buf(1, fold_p2_q_);
      e.buf(2, fold_p2_s_);
      e.buf(3, base_proxy_.view(size_t(i) * V * 2, size_t(V) * 2));
      e.buf(4, sup_buf);
      e.buf(5, d_cand_v_);
      e.buf(6, d_cand_i_);
      e.val(7, sp2);
      e.dispatch(64, 1, 1, 128);

      const Ink2MtpRescoreP rp2{V, H, R, unembed_->dtype == DType::F32 ? 1u : 0u, has_sup, sup_words, softcap,
                                 softcap > 0.0f ? (1.0f / softcap) : 0.0f};
      e.use(mtp_pipe("ink2_mtp_rescore_exact"));
      e.buf(0, blk_norm_.view(size_t(1 + i) * H * 4, size_t(H) * 4));
      e.buf(1, latent_);
      e.buf(2, unembed_->buf);
      e.buf(3, fold_w_);
      e.buf(4, fold_b_);
      e.buf(5, sup_buf);
      e.buf(6, d_cand_i_);
      e.buf(7, d_cand_v_);
      e.val(8, rp2);
      e.dispatch(32, 1, 1, 128);

      e.use(mtp_pipe("ink2_mtp_argmax_128"));
      e.buf(0, d_cand_v_);
      e.buf(1, d_cand_i_);
      e.buf(2, s.tok.view(size_t(P + 1 + i) * 4, 4));
      e.dispatch(1, 1, 1, 128);
      continue;
    }

    const Buffer lrow = draft_logits_.view(size_t(i) * V * 4, size_t(V) * 4);
    if (fold_q_) {
      const Ink2MtpQ8FoldP qp{V, R, softcap};
      e.use(mtp_pipe("ink2_mtp_q8_fold_mv"));
      e.buf(0, latent_);
      e.buf(1, fold_q_);
      e.buf(2, fold_s_);
      e.buf(3, fold_b_);
      e.buf(4, lrow);
      e.val(5, qp);
      e.dispatch((V + 15) / 16, 1, 1, 128);
    } else {
      const Ink2MtpMmP mp{1u, V, R, R, V, 1u, 1u, 1u, softcap};
      e.use(mtp_pipe("ink2_mtp_bf16_mv4"));
      e.buf(0, latent_);
      e.buf(1, fold_w_);
      e.buf(2, fold_b_);
      e.buf(3, lrow);
      e.val(4, mp);
      e.dispatch((V + 15) / 16, 1, 1, 128);
    }

    enc_token_suppression(e, s, lrow, 1, int(V), int(V));

    if (sp.temperature <= 0.0f) {
      const Ink2MtpArgmaxP ap{V};
      e.use(mtp_pipe("ink2_mtp_argmax_stage1"));
      e.buf(0, lrow);
      e.buf(1, d_cand_v_);
      e.buf(2, d_cand_i_);
      e.val(3, ap);
      e.dispatch(64, 1, 1, 256);

      e.use(mtp_pipe("ink2_mtp_argmax_stage2"));
      e.buf(0, d_cand_v_);
      e.buf(1, d_cand_i_);
      e.buf(2, s.tok.view(size_t(P + 1 + i) * 4, 4));
      e.dispatch(1, 1, 1, 64);
    } else if (full_vocab(sp)) {
      enc_sample_full(e, lrow, V, s.tok, P + 1 + i, d_sprob_, sp, step + uint32_t(i));
    } else {
      const uint32_t Ktop = topk_k(sp);
      const kp::TopkStage ts{V, Ktop, topk_chunk(Ktop)};
      const uint32_t ntg = topk_lists(V, Ktop);
      e.use(base_pipe("topk_stage1"));
      e.buf(0, lrow);
      e.buf(1, d_cand_v_);
      e.buf(2, d_cand_i_);
      e.val(3, ts);
      e.dispatch(ntg, 1, 1, topk_threads(Ktop));
      const kp::Sample p2{ntg * Ktop, Ktop, sp.temperature, sp.top_p, sp.seed, step + uint32_t(i), uint32_t(P + 1 + i)};
      e.use(base_pipe("sample_stage2"));
      e.buf(0, d_cand_v_);
      e.buf(1, d_cand_i_);
      e.buf(2, s.tok);
      e.buf(3, d_sprob_);
      e.val(4, p2);
      e.dispatch(1, 1, 1, 1024);
    }
  }
}

int Ink2Mtp::accept(Seq& s, int P, int K, const Buffer& verify_logits, const Buffer& verify_samples,
                    const SamplingParams& sp, uint32_t step) const {
  return ink2_mtp_accept_cpu(s.tokens(), P, K, vocab_, verify_logits.as<float>(), draft_logits_.as<float>(),
                             verify_samples.as<int>(), sp, step);
}

}  // namespace ni
