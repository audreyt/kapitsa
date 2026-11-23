#include "model/inkling2.h"

#include "engine/device_profile.h"
#include "engine/profile.h"
#include "model/ink2_mtp.h"
#include "model/inkling2_kernels.h"
#include "model/kparams.h"
#include "model/kv_grow.h"
#include "model/sampler.h"
#include "util/bf16.h"
#include "util/json_file.h"
#include "util/log.h"
#include "vision/embed.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace ni {

extern const char* const kTensorKernelSource;

using json = nlohmann::json;

namespace {

constexpr uint32_t kMaxAttnChunks = 64;

size_t visit_conv_floats(const Ink2Config& c) {
  const size_t kvd = size_t(c.hkv) * size_t(c.head_dim);
  return 3 * (2 * kvd + 2 * size_t(c.hidden));  // conv_k, conv_v, conv_a, conv_m
}

Pipeline ink2_pipe(const std::string& name) {
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

static bool ink2_tensor_pf_want() {
  if (__builtin_available(macOS 26.0, *)) {
    const bool here = default_on(Knob::kInk2TensorPf, ModelFacts{});
    return profile_switch("NI_INK2_TENSOR", here, "Inkling-2-Mini prefill GEMMs on the matrix units (NI_INK2_TENSOR)");
  }
  return false;
}

bool want_ink2_q8_dense() {
  static const bool on = [] {
    const bool here = default_on(Knob::kInk2Q8DenseFast, ModelFacts{});
    return profile_switch("NI_INK2_Q8_DENSE", here, "Inkling-2 Q8_0 dense MV (NI_INK2_Q8_DENSE)");
  }();
  return on;
}

bool want_ink2_kv_fp8() {
  static const bool on = [] {
    const char* v = getenv("NI_INK2_KV_FP8");
    return v && *v && atoi(v) != 0;
  }();
  return on;
}

Pipeline base_pipe(const std::string& name) {
  static std::mutex mu;
  static std::unordered_map<std::string, Pipeline> cache;
  {
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
  }
  const Pipeline p = Gpu::get().pipeline(name);
  std::lock_guard<std::mutex> lk(mu);
  return cache.emplace(name, p).first->second;
}

struct Ink2Grouped {
  Encoder& e;
  const bool on;
  Ink2Grouped(Encoder& enc, bool want) : e(enc), on(want && !enc.grouped() && !Encoder::profiling()) {
    if (on) e.set_grouped(true);
  }
  ~Ink2Grouped() {
    if (on) e.set_grouped(false);
  }
};

void prewarm_pipelines(bool tensor_pf) {
  if (const char* v = getenv("NI_PREWARM"); v && !strcmp(v, "0")) return;
  static const char* const kInk2Names[] = {
      "ink2_copy_bytes",       "ink2_bf16_mv",           "ink2_bf16_mv4",         "ink2_bf16_mv4_multi6",
      "ink2_bf16_mv8",         "ink2_bf16_mv8_multi6",   "ink2_bf16_mv_smalln",   "ink2_bf16_mm",
      "ink2_rmsnorm",          "ink2_rmsnorm3",          "ink2_attn_res",         "ink2_attn_res_pass",
      "ink2_msg_gate",         "ink2_msg_gate_prep",     "ink2_msg_gate_mix",
      "ink2_msg_gate_prep_s",  "ink2_msg_gate_prep3_s",  "ink2_msg_gate_down_s",
      "ink2_msg_gate_down3_s", "ink2_msg_gate_up_mix_s", "ink2_msg_gate_up_mix3_s","ink2_msg_gate_up_mix_norm_s",
      "ink2_embed_norm",       "ink2_sconv_static",      "ink2_sconv_static_pf",  "ink2_sconv_dynamic",    "ink2_qkvr_prep",         "ink2_rel_bias",         "ink2_rel_bias_pf",
      "ink2_attn_decode",      "ink2_attn_decode_reduce","ink2_attn_decode_1pass","ink2_attn_verify",      "ink2_attn_verify2",     "ink2_attn_verify_reduce","ink2_attn_prefill",     "ink2_moe_route",
      "ink2_attn_decode_gqa_global", "ink2_attn_decode_gqa_swa", "ink2_attn_decode_1pass_gqa_global", "ink2_attn_decode_1pass_gqa_swa", "ink2_attn_verify_gqa",
      "ink2_moe_nvfp4_w13_situ","ink2_moe_nvfp4_w2_acc", "ink2_moe_mxfp8_w13_situ","ink2_moe_mxfp8_w2_acc",
      "ink2_moe_bucket_small", "ink2_moe_nvfp4_w13_g",   "ink2_moe_nvfp4_w2_g",   "ink2_moe_mxfp8_w13_g",
      "ink2_moe_mxfp8_w2_g",   "ink2_moe_reduce8_m3",    "ink2_mtp_argmax_stage1","ink2_mtp_argmax_stage2",
      "ink2_softcap",          "ink2_quant_q8_bf16",     "ink2_q8_mv4",           "ink2_q8_mv8",
      "ink2_q8_mv4_multi6",    "ink2_q8_mv8_multi6",     "ink2_quant_q4g64_bf16", "ink2_q4g64_mv4",
      "ink2_q4g64_mv8",        "ink2_sconv_static_tree", "ink2_sconv_dynamic_tree","ink2_qkvr_prep_tree",
      "ink2_attn_verify_tree_gqa", "ink2_kv_compact",    "ink2_gather_rows",
      "ink2_moe_bucket",       "ink2_moe_situ_half",     "ink2_moe_combine",
  };
  for (const char* n : kInk2Names) (void)ink2_pipe(n);
  if (tensor_pf) {
    static const char* const kTensorNames[] = {
        "ink2_t_bf16_m64",
        "ink2_t_nvfp4_moe_w13_m16", "ink2_t_nvfp4_moe_w13_m32", "ink2_t_nvfp4_moe_w13_m64",
        "ink2_t_nvfp4_moe_w2_m16",  "ink2_t_nvfp4_moe_w2_m32",  "ink2_t_nvfp4_moe_w2_m64",
        "ink2_t_mxfp8_moe_w13_m16", "ink2_t_mxfp8_moe_w13_m32", "ink2_t_mxfp8_moe_w13_m64",
        "ink2_t_mxfp8_moe_w2_m16",  "ink2_t_mxfp8_moe_w2_m32",  "ink2_t_mxfp8_moe_w2_m64",
        "qattn_ink2_s6k32p",        "qattn_ink2_s3k16",
    };
    for (const char* n : kTensorNames) (void)base_pipe(n);
  }
  (void)base_pipe("topk_stage1");
  (void)base_pipe("sample_stage2");
}

struct Ink2EmbedP {
  uint32_t T, H, tok_off, V;
  float eps;
};
struct Ink2NormP {
  uint32_t T, H, has_w, w_f32;
  float eps;
};
struct Ink2AttnResP {
  uint32_t T, H, M, Q, q_f32;
  float eps, scale;
};
struct Ink2MsgGateP {
  uint32_t T, H, R, w_f32;
  float eps;
};
struct Ink2MsgNormP {
  uint32_t T, H, R, w_f32, nw_f32;
  float eps;
};
struct Ink2MmP {
  uint32_t M, N, K, x_ld, y_ld, w_f32, add;
};
struct Ink2MultiMmP {
  uint32_t M, K, N_tot, n_segs;
  uint32_t offs[7];
  uint32_t lens[6];
};
struct Ink2QuantQ8P {
  uint32_t total_blocks;
};
struct Ink2Q8MmP {
  uint32_t M, N, K, nblk, add;
  float cap, inv_cap;
};
struct Ink2Q8Multi6P {
  uint32_t M, K, N_tot, n_segs, nblk;
  uint32_t offs[7];
  uint32_t lens[6];
};
struct Ink2QuantQ4P {
  uint32_t total_groups;
};
struct Ink2Q4MmP {
  uint32_t M, N, K, ngroups, add;
};
struct Ink2SconvStaticP {
  uint32_t T, C, w_f32, save_snap, snap_stride, add_res;
  float scale;
};
struct Ink2SconvDynP {
  uint32_t T, C, n_heads, sw_f32, save_snap, snap_stride;
};
struct Ink2QkvrP {
  uint32_t T, Hq, Hkv, D, d_rel, pos0, kv_cap, is_swa, scale_f32;
  float qk_clip, r_clip, v_eps, log_floor, log_alpha;
  uint32_t kv_fp8;
};
struct Ink2RelP {
  uint32_t T, Hq, d_rel, extent, proj_f32;
};
struct Ink2AttnDecP {
  uint32_t pos, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, chunk, nchunks;
  float scale;
  uint32_t kv_fp8;
};
struct Ink2AttnVerP {
  uint32_t T, pos0, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, chunk, nchunks;
  float scale;
  uint32_t kv_fp8;
};
struct Ink2AttnPfP {
  uint32_t T, pos0, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa;
  float scale;
  uint32_t kv_fp8;
};
struct Ink2RouteP {
  uint32_t T, E, K, bias_f32;
  float route_scale;
};
struct Ink2MoeP {
  uint32_t T, H, I, K_top;
};
struct Ink2SoftcapP {
  uint32_t n;
  float cap, inv_cap;
};
struct Ink2KvCompactP {
  uint32_t pos0, kv_cap, is_swa, kv_fp8, kvd, m;
  uint32_t path[16];
};
struct Ink2GatherP {
  uint32_t m, H;
  uint32_t path[16];
};
struct Ink2BucketP {
  uint32_t P, E, BM;
};
struct Ink2SituP {
  uint32_t n;
};
struct Ink2CombineP {
  uint32_t T, H, K_top;
};
struct Ink2tGemmP {
  uint32_t M, N, K;
  uint32_t x_ld, y_ld;
  uint32_t add;
  uint32_t q_estride, s_estride;
  uint32_t k;
};

const json& text_cfg(const json& root) { return root.contains("text_config") ? root["text_config"] : root; }

float read_elem_f32(const Tensor& t, size_t idx) {
  if (t.dtype == DType::F32) return t.buf.as<float>()[idx];
  if (t.dtype == DType::BF16) return bf16_to_f32(t.buf.as<uint16_t>()[idx]);
  fatal("inkling2: unsupported scalar/vec dtype");
}

void enc_copy_buf(Encoder& e, const Buffer& src, const Buffer& dst, size_t bytes = 0) {
  const uint64_t n = uint64_t(bytes ? bytes : std::min(src.size, dst.size));
  if (!n) return;
  e.use(ink2_pipe("ink2_copy_bytes"));
  e.buf(0, src);
  e.buf(1, dst);
  e.val(2, n);
  const uint32_t threads = uint32_t((n + 15) / 16);
  e.dispatch((threads + 255) / 256, 1, 1, 256);
}

size_t safetensors_llm_bytes(const std::string& dir) {
  namespace fs = std::filesystem;
  std::error_code ec;
  size_t total = 0;
  for (const auto& ent : fs::directory_iterator(dir, ec)) {
    const std::string p = ent.path().string();
    if (p.size() <= 12 || p.compare(p.size() - 12, 12, ".safetensors") != 0) continue;
    std::ifstream in(p, std::ios::binary);
    if (!in) continue;
    uint64_t hlen = 0;
    in.read(reinterpret_cast<char*>(&hlen), 8);
    if (!in || hlen == 0 || hlen > (64u << 20)) continue;
    std::string hdr(size_t(hlen), '\0');
    in.read(hdr.data(), std::streamsize(hlen));
    const json j = json::parse(hdr, nullptr, false);
    if (!j.is_object()) continue;
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (it.key().rfind("model.llm.", 0) != 0 || !it.value().is_object()) continue;
      if (it.value().contains("data_offsets") && it.value()["data_offsets"].is_array() &&
          it.value()["data_offsets"].size() == 2) {
        const size_t a = it.value()["data_offsets"][0].get<size_t>();
        const size_t b = it.value()["data_offsets"][1].get<size_t>();
        if (b >= a) total += (b - a);
      }
    }
  }
  return total;
}

[[noreturn]] __attribute__((format(printf, 1, 2))) void cfg_error(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  throw std::runtime_error(buf);
}

}  // namespace

bool Inkling2::is_model(const std::string& dir) {
  const auto jp = json_file(dir + "/config.json");
  if (!jp || !jp->is_object()) return false;
  if (jp->value("model_type", std::string()) == "inkling2_loop") return true;
  return jp->contains("text_config") && (*jp)["text_config"].is_object() &&
         (*jp)["text_config"].value("model_type", std::string()) == "inkling2_loop_text";
}

Ink2Config Inkling2::read_config(const std::string& dir) {
  const auto rp = json_file(dir + "/config.json");
  if (!rp || !rp->is_object()) cfg_error("%s: cannot read config.json", dir.c_str());
  const json& root = *rp;
  const json& t = text_cfg(root);
  if (!t.is_object()) cfg_error("%s: missing text_config object", dir.c_str());

  auto req_int = [&](const json& obj, const char* key) -> int {
    if (!obj.contains(key) || !obj[key].is_number_integer())
      cfg_error("%s: missing or non-integer %s in config.json", dir.c_str(), key);
    return obj[key].get<int>();
  };
  auto opt_int = [&](const json& obj, const char* key, int def) -> int {
    if (!obj.contains(key)) return def;
    if (!obj[key].is_number_integer()) cfg_error("%s: non-integer %s in config.json", dir.c_str(), key);
    return obj[key].get<int>();
  };
  auto opt_float = [&](const json& obj, const char* key, float def) -> float {
    if (!obj.contains(key)) return def;
    if (!obj[key].is_number()) cfg_error("%s: non-numeric %s in config.json", dir.c_str(), key);
    return obj[key].get<float>();
  };
  auto opt_bool = [&](const json& obj, const char* key, bool def) -> bool {
    if (!obj.contains(key)) return def;
    if (!obj[key].is_boolean()) cfg_error("%s: non-boolean %s in config.json", dir.c_str(), key);
    return obj[key].get<bool>();
  };

  Ink2Config c;
  c.hidden = req_int(t, "hidden_size");
  c.num_layers = req_int(t, "num_hidden_layers");
  c.vocab_padded = req_int(t, "vocab_size");
  c.vocab = opt_int(t, "unpadded_vocab_size", c.vocab_padded);
  c.hq = req_int(t, "num_attention_heads");
  c.hkv = req_int(t, "num_key_value_heads");
  c.head_dim = req_int(t, "head_dim");
  c.d_rel = opt_int(t, "d_rel", 16);
  c.rel_extent = opt_int(t, "rel_extent", 1024);
  c.swa = opt_int(t, "sliding_window_size", 1024);
  c.block_sz = opt_int(t, "n_layers_per_attn_res_block", 6);
  c.n_experts = opt_int(t, "n_routed_experts", 128);
  c.top_k = opt_int(t, "num_experts_per_tok", 8);
  c.moe_I = opt_int(t, "intermediate_size", 1152);
  c.rms_eps = opt_float(t, "rms_norm_eps", 1e-6f);
  c.qk_clip = opt_float(t, "qk_l2_clip_threshold", 22.627416997969522f);
  c.rel_clip = opt_float(t, "rel_l2_clip_threshold", 8.0f);
  c.log_floor = float(opt_int(t, "log_scaling_n_floor", 8192));
  c.log_alpha = opt_float(t, "log_scaling_alpha", 0.1f);
  c.route_scale = opt_float(t, "route_scale", 8.0f);
  c.logit_softcap = opt_float(t, "final_logit_softcapping", 30.0f);

  if (!t.contains("loop_config") || !t["loop_config"].is_object())
    cfg_error("%s: missing loop_config in config.json", dir.c_str());
  const json& loop = t["loop_config"];
  c.core_start = req_int(loop, "core_start");
  c.core_end = req_int(loop, "core_end");
  c.num_passes = req_int(loop, "num_passes");
  c.gate_rank = opt_int(loop, "gate_rank", 64);

  if (opt_bool(t, "q_bias", false) || opt_bool(t, "o_bias", false))
    cfg_error("%s: Inkling-2 with attention q_bias/o_bias is not supported", dir.c_str());
  if (!opt_bool(t, "use_value_norm_no_gamma", true) || !opt_bool(t, "attn_res_normalize_values", true) ||
      !opt_bool(t, "use_gate_bias", true))
    cfg_error("%s: Inkling-2 requires use_value_norm_no_gamma, attn_res_normalize_values, and use_gate_bias", dir.c_str());
  if (opt_bool(loop, "is_varloop", false))
    cfg_error("%s: Inkling-2 varloop is not supported", dir.c_str());
  if (opt_int(t, "sconv_kernel_size", 4) != 4 || opt_int(t, "n_shared_experts", 0) != 0)
    cfg_error("%s: Inkling-2 requires sconv_kernel_size 4 and n_shared_experts 0", dir.c_str());
  if (opt_int(t, "swa_head_dim", c.head_dim) != c.head_dim ||
      opt_int(t, "swa_num_attention_heads", c.hq) != c.hq ||
      opt_int(t, "swa_num_key_value_heads", c.hkv) != c.hkv)
    cfg_error("%s: Inkling-2 heterogeneous SWA head geometry is not supported", dir.c_str());
  if (std::fabs(opt_float(t, "mlp_gate_softcap", 4.0f) - 4.0f) > 1e-4f ||
      std::fabs(opt_float(t, "mlp_up_softcap", 25.0f) - 25.0f) > 1e-4f)
    cfg_error("%s: Inkling-2 requires mlp_gate_softcap 4 and mlp_up_softcap 25", dir.c_str());

  if (c.hidden <= 0 || c.hidden > 2304 || c.hidden % 128 != 0 || c.num_layers < 2 || c.vocab <= 0 ||
      c.vocab > c.vocab_padded || c.head_dim != 128 || c.hq <= 0 || c.hkv <= 0 || c.hq % c.hkv != 0 ||
      c.d_rel <= 0 || c.d_rel > 128 || c.d_rel % 4 != 0 || c.rel_extent <= 0 || c.swa <= 0 ||
      c.n_experts != 128 || c.top_k < 1 || c.top_k > 8 || c.moe_I <= 0 || c.moe_I % 32 != 0 ||
      c.gate_rank != 64 || c.block_sz <= 0 || (c.num_layers + c.block_sz - 1) / c.block_sz > 4 ||
      c.core_start <= 0 || c.core_start % c.block_sz != 0 || c.core_end <= c.core_start ||
      c.core_end > c.num_layers || c.core_end % c.block_sz != 0 ||
      c.core_end - c.core_start != 2 * c.block_sz || c.num_passes < 1 ||
      !(c.rms_eps > 0.0f && c.qk_clip > 0.0f && c.rel_clip > 0.0f && c.log_floor > 0.0f &&
        c.logit_softcap > 0.0f && std::isfinite(c.log_alpha) && std::isfinite(c.route_scale)))
    cfg_error("%s: an Inkling-2 shape Kapitsa does not run (hidden %d, layers %d, heads %d/%d of %d, "
              "%d experts top-%d I=%d, gate_rank=%d, block=%d core=%d..%d, vocab=%d/%d)",
              dir.c_str(), c.hidden, c.num_layers, c.hq, c.hkv, c.head_dim, c.n_experts, c.top_k, c.moe_I,
              c.gate_rank, c.block_sz, c.core_start, c.core_end, c.vocab, c.vocab_padded);

  c.is_local.assign(size_t(c.num_layers), false);
  if (t.contains("local_layer_ids")) {
    if (!t["local_layer_ids"].is_array())
      cfg_error("%s: local_layer_ids must be an array", dir.c_str());
    for (const auto& lid : t["local_layer_ids"]) {
      if (!lid.is_number_integer()) cfg_error("%s: local_layer_ids must contain integers", dir.c_str());
      const int idx = lid.get<int>();
      if (idx < 0 || idx >= c.num_layers)
        cfg_error("%s: local_layer_ids contains out-of-range layer %d", dir.c_str(), idx);
      c.is_local[size_t(idx)] = true;
    }
  }
  if (!c.is_local[0])
    cfg_error("%s: Inkling-2 layer 0 must be a local layer", dir.c_str());

  for (int i = 0; i < c.core_start; ++i) c.schedule.push_back(i);
  for (int pass = 0; pass < c.num_passes; ++pass)
    for (int i = c.core_start; i < c.core_end; ++i) c.schedule.push_back(i);
  for (int i = c.core_end; i < c.num_layers; ++i) c.schedule.push_back(i);

  const auto gp = json_file(dir + "/generation_config.json");
  auto push_eos = [&](const json& v) {
    if (v.is_number_integer()) {
      const int id = v.get<int>();
      if (std::find(c.eos.begin(), c.eos.end(), id) == c.eos.end()) c.eos.push_back(id);
    } else if (v.is_array()) {
      for (const auto& x : v)
        if (x.is_number_integer()) {
          const int id = x.get<int>();
          if (std::find(c.eos.begin(), c.eos.end(), id) == c.eos.end()) c.eos.push_back(id);
        }
    }
  };
  if (gp && gp->is_object()) {
    if (gp->contains("eos_token_id")) push_eos((*gp)["eos_token_id"]);
  }
  if (root.contains("eos_token_id")) push_eos(root["eos_token_id"]);
  if (c.eos.empty()) c.eos = {200006, 199999};
  else if (std::find(c.eos.begin(), c.eos.end(), 199999) == c.eos.end()) c.eos.push_back(199999);
  return c;
}

size_t Inkling2::kv_per_pos(const Ink2Config& c) {
  int n_global = 0;
  for (int ip : c.schedule)
    if (!c.is_local[size_t(ip)]) ++n_global;
  // 2 (K, V) * hkv * head_dim * sizeof(half)
  return size_t(n_global) * 2 * size_t(c.hkv) * size_t(c.head_dim) * 2;
}

size_t Inkling2::conv_bytes_per_seq(const Ink2Config& c) {
  return c.schedule.size() * visit_conv_floats(c) * 4;
}

size_t Inkling2::ring_bytes_per_seq(const Ink2Config& c, int tmax) {
  int n_swa = 0;
  for (int ip : c.schedule)
    if (c.is_local[size_t(ip)]) ++n_swa;
  const int rcap = (c.swa + std::max(1, tmax) + 15) / 16 * 16;
  return size_t(n_swa) * 2 * size_t(rcap) * size_t(c.hkv) * size_t(c.head_dim) * 2;
}

MemPlan Inkling2::plan(const std::string& dir, int prefill_chunk) {
  const Ink2Config c = read_config(dir);
  const int chunk = prefill_chunk > 0 ? prefill_chunk : 512;
  const size_t T = size_t(std::max(8, chunk));
  const size_t H = size_t(c.hidden);
  const size_t QD = size_t(c.hq) * size_t(c.head_dim);
  const size_t KVD = size_t(c.hkv) * size_t(c.head_dim);
  const size_t V = size_t(c.vocab);
  const size_t per_row = 30 * H * 4 + 3 * QD * 4 + 2 * KVD * 4 + size_t(c.hq) * size_t(c.d_rel) * 4 +
                         size_t(c.hq) * 4 + 4 * size_t(c.hkv) * 4 + size_t(c.hq) * size_t(c.rel_extent) * 4 +
                         size_t(c.n_experts) * 4 + 2 * size_t(c.top_k) * 4 + size_t(c.top_k) * size_t(c.moe_I) * 4;
  const size_t fixed = size_t(c.core_end - c.core_start) * 4 * H * 4 +
                       size_t(c.num_layers) * 2 * size_t(c.n_experts) * 4 +
                       size_t(c.hq) * kMaxAttnChunks * (2 + size_t(c.head_dim)) * 4 + V * 4 +
                       2 * (V / 2048 + 2) * 64 * 4 + 2 * 64 * 4;
  const size_t P = T * size_t(c.top_k);
  const size_t moe_tmp_bytes = ink2_tensor_pf_want()
                                   ? (P * std::max(H, size_t(2) * size_t(c.moe_I)) * 4 + P * size_t(c.moe_I) * 2 +
                                      2 * P * 4 + (P / 16 + size_t(c.n_experts) + 2) * 16 + 16)
                                   : 0;
  MemPlan m;
  m.weights = safetensors_llm_bytes(dir);
  m.kv_per_pos = kv_per_pos(c);
  m.slot_state = 3 * (ring_bytes_per_seq(c, chunk) + conv_bytes_per_seq(c));
  const bool has_mtp = Ink2Mtp::exists(dir);
  Ink2MtpConfig mc;
  if (has_mtp) {
    mc = Ink2Mtp::read_config(dir);
    const int rcap = (c.swa + int(T) + 15) / 16 * 16;
    m.mtp = Ink2Mtp::plan_bytes(dir, c);
    m.kv_per_pos += Ink2Mtp::kv_per_pos(mc);
    m.slot_state += 3 * Ink2Mtp::ring_bytes_per_seq(mc, rcap);
  }
  size_t act = T * per_row + fixed + moe_tmp_bytes + (16u << 20);
  if (has_mtp) {
    const size_t B = size_t(std::max(1, mc.block));
    const size_t MQD = size_t(mc.hq) * size_t(mc.head_dim);
    const size_t MKVD = size_t(mc.hkv) * size_t(mc.head_dim);
    const size_t MH = size_t(mc.hidden);
    act += 3 * T * H * 4 + B * (conv_bytes_per_seq(c) + V * 4 + 4);
    act += T * (MH + 2 * MKVD) * 4 + B * (3 * MH + 2 * MQD + 2 * MKVD + 2 * size_t(mc.ffn) + 4) * 4 +
           size_t(std::max<size_t>(1, B > 1 ? B - 1 : 1)) * V * 4;
  }
  m.activations = act;
  m.hidden = c.hidden;
  m.layers = int(c.schedule.size());
  m.formats = "NVFP4 (L0-19) + MXFP8 (L20-23) + BF16 dense";
  return m;
}

Inkling2::Inkling2(const std::string& path, int prefill_chunk) {
  path_ = path;
  while (path_.size() > 1 && path_.back() == '/') path_.pop_back();
  if (!is_model(path_)) fatal("%s: not an Inkling-2 checkpoint folder", path.c_str());
  cfg_ = read_config(path_);
  tmax_ = std::max(8, prefill_chunk > 0 ? prefill_chunk : 512);
  ring_rows_ = (cfg_.swa + tmax_ + 15) / 16 * 16;
  ink2::compile_kernels();
  if (ink2_tensor_pf_want()) {
    if (__builtin_available(macOS 26.0, *)) {
      Gpu::get().compile_extra("tensor", kTensorKernelSource, true);
      tensor_pf_ = Gpu::get().pipeline("ink2_t_bf16_m64", false).pso != nullptr;
    }
  }
  prewarm_pipelines(tensor_pf_);
  use_q8_dense_ = want_ink2_q8_dense() && (cfg_.hidden & 255) == 0 && ((cfg_.hq * cfg_.head_dim) & 255) == 0;
  use_kv_fp8_ = want_ink2_kv_fp8();
  if (const char* q4e = getenv("NI_INK2_Q4_HEAD"); q4e && *q4e) {
    use_q4_head_ = (atoi(q4e) != 0) && (cfg_.hidden & 255) == 0;
    use_q8_head_ = false;
  } else if (const char* he = getenv("NI_INK2_Q8_HEAD"); he && *he) {
    use_q8_head_ = (atoi(he) != 0) && (cfg_.hidden & 255) == 0;
    use_q4_head_ = false;
  } else {
    use_q8_head_ = use_q8_dense_;
    use_q4_head_ = false;
  }
  load();
}

void Inkling2::set_use_tensor(bool on) {
  if (!on) {
    tensor_pf_ = false;
    return;
  }
  if (!moe_tmp_) fatal("inkling2: tensor buffers not allocated (restart with KAPITSA_PROFILE=fast or NI_INK2_TENSOR=1)");
  if (__builtin_available(macOS 26.0, *)) {
    Gpu::get().compile_extra("tensor", kTensorKernelSource, true);
    tensor_pf_ = Gpu::get().pipeline("ink2_t_bf16_m64", false).pso != nullptr;
  }
}

int Inkling2::mtp_block() const { return mtp_ ? mtp_->block() : 1; }

void Inkling2::disable_mtp() {
  if (!mtp_) return;
  mtp_.reset();
  Gpu& g = Gpu::get();
  for (Buffer& b : tap_) g.release(b);
  g.release(verify_conv_);
  g.release(verify_logits_);
  g.release(verify_samples_);
  g.release(verify_accept_);
}

Inkling2::~Inkling2() {
  Gpu& g = Gpu::get();
  for (Buffer& b : owned_bufs_) g.release(b);
}

void Inkling2::load() {
  const double t0 = now_ms();
  ws_.open_dir(path_, [](const std::string& n) { return n.rfind("model.llm.", 0) == 0; });
  auto req_shape = [&](const std::string& name, std::initializer_list<int64_t> want) -> const Tensor* {
    const Tensor* t = ws_.find(name);
    if (!t) fatal("%s: missing tensor %s", path_.c_str(), name.c_str());
    if (t->shape.size() != want.size() || !std::equal(want.begin(), want.end(), t->shape.begin()))
      fatal("%s: %s has an unexpected shape", path_.c_str(), name.c_str());
    return t;
  };
  auto req_float = [&](const std::string& name, std::initializer_list<int64_t> want) -> const Tensor* {
    const Tensor* t = req_shape(name, want);
    if (t->dtype != DType::BF16 && t->dtype != DType::F32)
      fatal("%s: %s must be BF16 or F32", path_.c_str(), name.c_str());
    return t;
  };
  auto req_raw = [&](const std::string& name, DType dtype, std::initializer_list<int64_t> want) -> const Tensor* {
    const Tensor* t = req_shape(name, want);
    if (t->dtype != dtype) fatal("%s: %s has an unexpected dtype", path_.c_str(), name.c_str());
    return t;
  };
  auto req_any = [&](const std::string& name) -> const Tensor* {
    const Tensor* t = ws_.find(name);
    if (!t) fatal("%s: missing tensor %s", path_.c_str(), name.c_str());
    return t;
  };

  const int64_t H64 = cfg_.hidden;
  const int64_t QD64 = int64_t(cfg_.hq) * cfg_.head_dim;
  const int64_t KVD64 = int64_t(cfg_.hkv) * cfg_.head_dim;
  const int64_t E64 = cfg_.n_experts;
  const int64_t I64 = cfg_.moe_I;

  embed_ = req_any("model.llm.embed.weight");
  if (embed_->dtype != DType::BF16 || embed_->shape.size() != 2 || embed_->shape[0] < cfg_.vocab ||
      embed_->shape[1] != H64)
    fatal("%s: model.llm.embed.weight must be BF16 [>=%d, %lld]", path_.c_str(), cfg_.vocab, (long long)H64);
  embed_norm_ = req_raw("model.llm.embed_norm.weight", DType::BF16, {H64});
  final_norm_ = req_float("model.llm.norm.weight", {H64});
  unembed_ = req_any("model.llm.unembed.weight");
  if ((unembed_->dtype != DType::BF16 && unembed_->dtype != DType::F32) || unembed_->shape.size() != 2 ||
      unembed_->shape[0] < cfg_.vocab || unembed_->shape[1] != H64)
    fatal("%s: model.llm.unembed.weight must be BF16/F32 [>=%d, %lld]", path_.c_str(), cfg_.vocab, (long long)H64);

  attn_queries_.resize(size_t(cfg_.num_layers - 1));
  for (int i = 0; i < cfg_.num_layers - 1; ++i) {
    const std::string qn = "model.llm.attn_res.attn_queries." + std::to_string(i);
    attn_queries_[size_t(i)] = cfg_.is_local[size_t(i + 1)] ? req_float(qn, {H64}) : req_float(qn, {3, H64});
  }
  mlp_queries_.resize(size_t(cfg_.num_layers));
  for (int i = 0; i < cfg_.num_layers; ++i)
    mlp_queries_[size_t(i)] = req_float("model.llm.attn_res.mlp_queries." + std::to_string(i), {H64});
  output_query_ = req_float("model.llm.attn_res.output_query", {H64});

  // Pre-interleave [12, 4, H] middle_cycle cross-pass queries: (attn, k, v, mlp)
  const int ncore = cfg_.core_end - cfg_.core_start;
  const Tensor* qam = req_float("model.llm.middle_cycle_cross_pass_queries", {int64_t(2) * ncore, H64});
  const Tensor* qkv = req_float("model.llm.middle_cycle_cross_pass_kv_queries", {int64_t(2) * ncore, H64});
  const size_t H = size_t(cfg_.hidden);
  msg_queries_ = Gpu::get().alloc(size_t(ncore) * 4 * H * 4, "ink2_msg_queries");
  owned_bufs_.push_back(msg_queries_);
  {
    float* dst = msg_queries_.as<float>();
    for (int c = 0; c < ncore; ++c) {
      for (size_t j = 0; j < H; ++j) {
        dst[(size_t(c) * 4 + 0) * H + j] = read_elem_f32(*qam, (size_t(c) * 2 + 0) * H + j);
        dst[(size_t(c) * 4 + 1) * H + j] = read_elem_f32(*qkv, (size_t(c) * 2 + 0) * H + j);
        dst[(size_t(c) * 4 + 2) * H + j] = read_elem_f32(*qkv, (size_t(c) * 2 + 1) * H + j);
        dst[(size_t(c) * 4 + 3) * H + j] = read_elem_f32(*qam, (size_t(c) * 2 + 1) * H + j);
      }
    }
  }

  auto make_outer = [&](const std::string& base) -> Buffer {
    const Tensor* amax = req_float(base + "_amax", {E64});
    const Tensor* denom = req_any(base + "_outer_scale_denominator");
    if ((denom->dtype != DType::BF16 && denom->dtype != DType::F32) || denom->numel() != 1)
      fatal("%s: %s_outer_scale_denominator must be a 1-element BF16/F32 tensor", path_.c_str(), base.c_str());
    const float d = read_elem_f32(*denom, 0);
    if (!std::isfinite(d) || d <= 0.0f)
      fatal("%s: invalid %s_outer_scale_denominator (%g)", path_.c_str(), base.c_str(), d);
    Buffer b = Gpu::get().alloc(size_t(cfg_.n_experts) * 4, "ink2_outer");
    owned_bufs_.push_back(b);
    float* out = b.as<float>();
    for (int e = 0; e < cfg_.n_experts; ++e) out[e] = read_elem_f32(*amax, size_t(e)) / d;
    return b;
  };

  layers_.assign(size_t(cfg_.num_layers), Ink2Layer{});
  for (int i = 0; i < cfg_.num_layers; ++i) {
    Ink2Layer& L = layers_[size_t(i)];
    const std::string p = "model.llm.layers." + std::to_string(i) + ".";
    L.is_swa = cfg_.is_local[size_t(i)];
    L.has_ctrl = (i >= cfg_.core_start && i < cfg_.core_end);
    L.wq_du = req_float(p + "attn.wq_du.weight", {QD64, H64});
    L.wk_dv = req_float(p + "attn.wk_dv.weight", {KVD64, H64});
    L.wv_dv = req_float(p + "attn.wv_dv.weight", {KVD64, H64});
    L.wr_du = req_float(p + "attn.wr_du.weight", {int64_t(cfg_.hq) * cfg_.d_rel, H64});
    L.attn_sink_proj = req_float(p + "attn.attn_sink_proj.weight", {cfg_.hq, H64});
    L.gate_attn = req_float(p + "attn.gate_attn.weight", {QD64, H64});
    L.wo_ud = req_float(p + "attn.wo_ud.weight", {H64, QD64});
    L.q_scale = req_float(p + "attn.q_l2_clip.scale", {cfg_.hq, cfg_.head_dim});
    L.rel_scale = req_float(p + "attn.rel_l2_clip.scale", {cfg_.hq, cfg_.d_rel});
    if (L.q_scale->dtype != L.rel_scale->dtype)
      fatal("%s: layer %d q_l2_clip.scale and rel_l2_clip.scale dtypes differ", path_.c_str(), i);
    L.rel_proj = req_float(p + "attn.rel_logits_proj.proj", {cfg_.d_rel, cfg_.rel_extent});
    L.mlp_norm = req_float(p + "mlp_norm.weight", {H64});
    L.attn_sconv = req_float(p + "attn_sconv.weight", {H64, 1, 4});
    L.mlp_sconv = req_float(p + "mlp_sconv.weight", {H64, 1, 4});
    if (L.is_swa) {
      L.attn_norm = req_float(p + "attn_norm.weight", {H64});
      L.k_sconv = req_float(p + "attn.k_sconv.weight", {KVD64, 1, 4});
      L.v_sconv = req_float(p + "attn.v_sconv.weight", {KVD64, 1, 4});
    } else {
      L.q_norm = req_float(p + "q_input_norm.weight", {H64});
      L.k_norm = req_float(p + "k_input_norm.weight", {H64});
      L.v_norm = req_float(p + "v_input_norm.weight", {H64});
      L.k_static = req_float(p + "attn.k_sconv.static_w", {4, KVD64});
      L.k_proj = req_float(p + "attn.k_sconv.weight_proj", {int64_t(4) * cfg_.hkv, H64});
      L.v_static = req_float(p + "attn.v_sconv.static_w", {4, KVD64});
      L.v_proj = req_float(p + "attn.v_sconv.weight_proj", {int64_t(4) * cfg_.hkv, H64});
    }
    if (L.has_ctrl) {
      auto load_ctrl = [&](const char* name) {
        const std::string c = p + "controllers." + name + ".";
        Ink2Ctrl ctrl{req_float(c + "down_weight", {cfg_.gate_rank, 2 * H64}),
                      req_float(c + "forget_weight", {H64, cfg_.gate_rank}),
                      req_float(c + "gain_weight", {H64, cfg_.gate_rank})};
        if (ctrl.down->dtype != ctrl.forget->dtype || ctrl.down->dtype != ctrl.gain->dtype)
          fatal("%s: layer %d controller %s weight dtypes differ", path_.c_str(), i, name);
        return ctrl;
      };
      L.ctrl_attn = load_ctrl("attn");
      L.ctrl_k = load_ctrl("k");
      L.ctrl_v = load_ctrl("v");
      L.ctrl_mlp = load_ctrl("mlp");
    }
    L.router_w = req_float(p + "mlp.gate.weight", {E64, H64});
    L.router_bias = req_float(p + "mlp.gate.bias", {E64});
    const std::string m = p + "mlp.experts.";
    L.w13 = req_any(m + "w13_weight");
    if (L.w13->dtype == DType::U8) {
      L.is_mxfp8 = false;
      L.w13 = req_raw(m + "w13_weight", DType::U8, {E64, 2 * I64, H64 / 2});
      L.w13_sc = req_raw(m + "w13_weight_scale", DType::F8E4M3, {E64, 2 * I64, H64 / 16});
      L.w2 = req_raw(m + "w2_weight", DType::U8, {E64, H64, I64 / 2});
      L.w2_sc = req_raw(m + "w2_weight_scale", DType::F8E4M3, {E64, H64, I64 / 16});
      L.w13_outer = make_outer(m + "w13_weight");
      L.w2_outer = make_outer(m + "w2_weight");
    } else if (L.w13->dtype == DType::F8E4M3) {
      L.is_mxfp8 = true;
      L.w13 = req_raw(m + "w13_weight", DType::F8E4M3, {E64, 2 * I64, H64});
      L.w13_sc = req_raw(m + "w13_weight_scale", DType::U8, {E64, 2 * I64, H64 / 32});
      L.w2 = req_raw(m + "w2_weight", DType::F8E4M3, {E64, H64, I64});
      L.w2_sc = req_raw(m + "w2_weight_scale", DType::U8, {E64, H64, I64 / 32});
    } else {
      fatal("%s: unexpected w13 dtype in layer %d", path_.c_str(), i);
    }
  }
  if (cfg_.n_experts > 256 || cfg_.top_k > 8)
    fatal("%s: n_experts=%d top_k=%d exceeds route kernel bounds (256, 8)", path_.c_str(), cfg_.n_experts, cfg_.top_k);

  // Pack [wq_du; wr_du; attn_sink_proj; gate_attn; wk_dv; wv_dv] per layer on 16-row boundaries for fused MV dispatches.
  if (!mem_budget_ || (ws_.total_bytes() + 1060ull * 1024ull * 1024ull <= mem_budget_)) {
    for (int i = 0; i < cfg_.num_layers; ++i) {
      Ink2Layer& L = layers_[size_t(i)];
      const Tensor* segs[6] = {L.wq_du, L.wr_du, L.attn_sink_proj, L.gate_attn, L.wk_dv, L.wv_dv};
      bool ok = (cfg_.hidden & 7) == 0;
      uint32_t off = 0;
      for (int s_idx = 0; s_idx < 6; ++s_idx) {
        const Tensor* t = segs[s_idx];
        if (!t || t->dtype != DType::BF16 || (t->shape[0] & 3) != 0 || t->shape[1] != H64) ok = false;
        L.qkvrsg_offs[s_idx] = off;
        L.qkvrsg_lens[s_idx] = t ? uint32_t(t->shape[0]) : 0u;
        off += L.qkvrsg_lens[s_idx];
        off = (off + 15u) & ~15u;
      }
      L.qkvrsg_offs[6] = off;
      if (ok) {
        L.qkvrsg_w = Gpu::get().alloc(size_t(off) * size_t(cfg_.hidden) * 2, "ink2_qkvrsg");
        owned_bufs_.push_back(L.qkvrsg_w);
        memset(L.qkvrsg_w.ptr(), 0, L.qkvrsg_w.size);
        uint8_t* dst = L.qkvrsg_w.as<uint8_t>();
        for (int s_idx = 0; s_idx < 6; ++s_idx) {
          const size_t bytes = size_t(L.qkvrsg_lens[s_idx]) * size_t(cfg_.hidden) * 2;
          std::memcpy(dst + size_t(L.qkvrsg_offs[s_idx]) * size_t(cfg_.hidden) * 2, segs[s_idx]->buf.as<uint8_t>(), bytes);
        }
      }
    }
  }

  Gpu::get().make_resident(ws_.buffers());
  if (use_q8_dense_ || use_q8_head_ || use_q4_head_) {
    auto e_q = Gpu::get().begin();
    auto quant_mat = [&](const Buffer& src, size_t rows, size_t cols, Buffer& out_q, Buffer& out_s, const char* tag) {
      const uint32_t tblks = uint32_t(rows * (cols / 32));
      out_q = Gpu::get().alloc(rows * cols, tag);
      out_s = Gpu::get().alloc(size_t(tblks) * 2, tag);
      owned_bufs_.push_back(out_q);
      owned_bufs_.push_back(out_s);
      const Ink2QuantQ8P qp{tblks};
      e_q->use(ink2_pipe("ink2_quant_q8_bf16"));
      e_q->buf(0, src);
      e_q->buf(1, out_q);
      e_q->buf(2, out_s);
      e_q->val(3, qp);
      e_q->dispatch((tblks + 3) / 4, 1, 1, 128);
    };
    if (use_q8_dense_) {
      const size_t H = size_t(cfg_.hidden);
      const size_t QD = size_t(cfg_.hq) * size_t(cfg_.head_dim);
      for (int i = 0; i < cfg_.num_layers; ++i) {
        Ink2Layer& L = layers_[size_t(i)];
        if (L.qkvrsg_w) quant_mat(L.qkvrsg_w, L.qkvrsg_offs[6], H, L.qkvrsg_q, L.qkvrsg_s, "ink2_qkvrsg_q8");
        if (L.wo_ud && L.wo_ud->dtype == DType::BF16) quant_mat(L.wo_ud->buf, H, QD, L.wo_q, L.wo_s, "ink2_wo_q8");
      }
    }
    if (use_q4_head_ && unembed_ && unembed_->dtype == DType::BF16) {
      const size_t V = size_t(cfg_.vocab);
      const size_t H = size_t(cfg_.hidden);
      const uint32_t tgrps = uint32_t(V * (H / 64));
      unembed_q4_ = Gpu::get().alloc(V * (H / 2), "ink2_head_q4");
      unembed_sb_ = Gpu::get().alloc(size_t(tgrps) * 4, "ink2_head_q4sb");
      owned_bufs_.push_back(unembed_q4_);
      owned_bufs_.push_back(unembed_sb_);
      const Ink2QuantQ4P qp{tgrps};
      e_q->use(ink2_pipe("ink2_quant_q4g64_bf16"));
      e_q->buf(0, unembed_->buf);
      e_q->buf(1, unembed_q4_);
      e_q->buf(2, unembed_sb_);
      e_q->val(3, qp);
      e_q->dispatch((tgrps + 3) / 4, 1, 1, 128);
    } else if (use_q8_head_ && unembed_ && unembed_->dtype == DType::BF16) {
      quant_mat(unembed_->buf, size_t(cfg_.vocab), size_t(cfg_.hidden), unembed_q_, unembed_s_, "ink2_head_q8");
    }
    e_q->commit_and_wait();
  }
  const char* mtp_env = getenv("NI_INK2_MTP");
  const bool want_mtp = !(mtp_env && *mtp_env && atoi(mtp_env) == 0);
  if (want_mtp && Ink2Mtp::exists(path_)) {
    mtp_ = std::make_unique<Ink2Mtp>(*this, path_, cfg_, *unembed_, tmax_, ring_rows_);
  }
  alloc_buffers();
  Gpu::get().make_resident(owned_bufs_);
  logi("loaded %s: Inkling-2-Mini (24 layers / 48 visits, vocab %d, q8_dense=%d, q8_head=%d, q4_head=%d, kv_fp8=%d), %.2f GB text weights in %.0f ms",
       path_.c_str(), cfg_.vocab, use_q8_dense_ ? 1 : 0, use_q8_head_ ? 1 : 0, use_q4_head_ ? 1 : 0, use_kv_fp8_ ? 1 : 0,
       ws_.total_bytes() / 1e9, now_ms() - t0);
}

void Inkling2::alloc_buffers() {
  Gpu& g = Gpu::get();
  const size_t T = size_t(tmax_), H = size_t(cfg_.hidden);
  const size_t QD = size_t(cfg_.hq) * size_t(cfg_.head_dim);
  const size_t KVD = size_t(cfg_.hkv) * size_t(cfg_.head_dim);
  const size_t V = size_t(cfg_.vocab);
  auto alloc = [&](size_t bytes, const char* lbl) {
    Buffer b = g.alloc(std::max<size_t>(bytes, 16), lbl);
    owned_bufs_.push_back(b);
    return b;
  };
  x_ = alloc(T * H * 4, "ink2_x");
  x_pre_ = alloc(T * H * 4, "ink2_x_pre");
  inp_sa_ = alloc(T * H * 4, "ink2_inp_sa");
  ffn_inp_ = alloc(T * H * 4, "ink2_ffn_inp");
  h_norm_ = alloc(T * H * 4, "ink2_h_norm");
  for (int i = 0; i < 6; ++i) blocks_[i] = alloc(T * H * 4, "ink2_block");
  for (int i = 0; i < 4; ++i) prelude_blocks_[i] = alloc(T * H * 4, "ink2_pre_block");
  ar_out_ = alloc(4 * T * H * 4, "ink2_ar");
  msg_out_ = alloc(4 * T * H * 4, "ink2_msg");
  const size_t ncore = size_t(cfg_.core_end - cfg_.core_start);
  pass_msg_out_ = alloc(ncore * 4 * 16 * H * 4, "ink2_pass_msg");
  mg_cat_ = alloc(std::max<size_t>(tensor_pf_ ? T : 0, 3 * 16) * 2 * H * 4, "ink2_mg_cat");
  mg_lat_ = alloc(std::max<size_t>(tensor_pf_ ? T : 0, 3 * 16) * size_t(cfg_.gate_rank) * 4, "ink2_mg_lat");
  if (tensor_pf_) {
    mg_f_ = alloc(T * H * 4, "ink2_mg_f");
    mg_g_ = alloc(T * H * 4, "ink2_mg_g");
    const size_t P = T * size_t(cfg_.top_k);
    moe_sorted_ = alloc(P * 4, "ink2_moe_sorted");
    moe_rank_ = alloc(P * 4, "ink2_moe_rank");
    moe_tmp_ = alloc(P * std::max(H, size_t(2) * size_t(cfg_.moe_I)) * 4, "ink2_moe_tmp");
    moe_act_h_ = alloc(P * size_t(cfg_.moe_I) * 2, "ink2_moe_act_h");
  }
  {
    const size_t P = T * size_t(cfg_.top_k);
    const size_t r_max = 31u * size_t(cfg_.top_k);
    const size_t small_tiles = std::max<size_t>((r_max + 31) / 32 + size_t(cfg_.n_experts) + 4, size_t(cfg_.n_experts) + 4);
    const size_t tile_bytes = tensor_pf_ ? std::max((P / 16 + size_t(cfg_.n_experts) + 4) * 16, small_tiles * 3 * 4)
                                         : (small_tiles * 3 * 4);
    moe_tiles_ = alloc(tile_bytes, "ink2_moe_tiles");
    moe_ntiles_ = alloc(16, "ink2_moe_ntiles");
    moe_st_tok_ = alloc(r_max * 4, "ink2_moe_st_tok");
    moe_st_pair_ = alloc(r_max * 4, "ink2_moe_st_pair");
    moe_st_ew_ = alloc(r_max * 4, "ink2_moe_st_ew");
    moe_pair_out_ = alloc(r_max * H * 4, "ink2_moe_pair_out");
  }
  n0_ = alloc(T * H * 4, "ink2_n0");
  n1_ = alloc(T * H * 4, "ink2_n1");
  n2_ = alloc(T * H * 4, "ink2_n2");
  q_ = alloc(T * QD * 4, "ink2_q");
  k_ = alloc(T * KVD * 4, "ink2_k");
  v_ = alloc(T * KVD * 4, "ink2_v");
  r_ = alloc(T * size_t(cfg_.hq) * size_t(cfg_.d_rel) * 4, "ink2_r");
  sinks_ = alloc(T * size_t(cfg_.hq) * 4, "ink2_sinks");
  gate_attn_ = alloc(T * QD * 4, "ink2_gate_attn");
  att_out_ = alloc(T * QD * 4, "ink2_att_out");
  a_out_ = alloc(T * H * 4, "ink2_a_out");
  dyn_ = alloc(T * 4 * size_t(cfg_.hkv) * 4, "ink2_dyn");
  dyn_v_ = alloc(T * 4 * size_t(cfg_.hkv) * 4, "ink2_dyn_v");
  rb_ = alloc(T * size_t(cfg_.hq) * size_t(cfg_.rel_extent) * 4, "ink2_rb");
  const size_t attn_rows = mtp_ ? size_t(Ink2Mtp::kMaxVerifyRows) : 1u;
  part_m_ = alloc(attn_rows * size_t(cfg_.hq) * kMaxAttnChunks * 4, "ink2_part_m");
  part_l_ = alloc(attn_rows * size_t(cfg_.hq) * kMaxAttnChunks * 4, "ink2_part_l");
  part_o_ = alloc(attn_rows * size_t(cfg_.hq) * kMaxAttnChunks * size_t(cfg_.head_dim) * 4, "ink2_part_o");
  h_mlp_ = alloc(T * H * 4, "ink2_h_mlp");
  x_mlp_ = alloc(T * H * 4, "ink2_x_mlp");
  rlog_ = alloc(T * size_t(cfg_.n_experts) * 4, "ink2_rlog");
  eid_ = alloc(T * size_t(cfg_.top_k) * 4, "ink2_eid");
  ew_ = alloc(T * size_t(cfg_.top_k) * 4, "ink2_ew");
  moe_act_ = alloc(T * size_t(cfg_.top_k) * size_t(cfg_.moe_I) * 4, "ink2_moe_act");
  moe_out_ = alloc(T * H * 4, "ink2_moe_out");
  logits_ = alloc(V * 4, "ink2_logits");
  cand_v_ = alloc((V / 2048 + 2) * 64 * 4, "ink2_cand_v");
  cand_i_ = alloc((V / 2048 + 2) * 64 * 4, "ink2_cand_i");
  sprob_ = alloc(64 * 4, "ink2_sprob");
  samp_ = alloc(64 * 4, "ink2_samp");
  if (mtp_) {
    const size_t B = size_t(std::max(mtp_->block(), Ink2Mtp::kMaxVerifyRows));
    for (int i = 0; i < 3; ++i) tap_[i] = alloc(T * H * 4, "ink2_mtp_tap");
    verify_conv_ = alloc(B * conv_bytes_per_seq(cfg_), "ink2_verify_conv");
    verify_logits_ = alloc(B * V * 4, "ink2_verify_logits");
    verify_samples_ = alloc(B * 4, "ink2_verify_samples");
    verify_accept_ = alloc(16, "ink2_verify_accept");
    verify_tree_accept_ = alloc(sizeof(Ink2TreeAccept), "ink2_verify_tree_acc");
  }
}

bool Inkling2::is_swa_slot(size_t slot) const {
  const size_t nv = cfg_.schedule.size();
  return slot < nv ? cfg_.is_local[size_t(cfg_.schedule[slot])] : (mtp_ ? mtp_->is_local(int(slot - nv)) : false);
}

// ---- sequences, KV caches, checkpoints ----
std::unique_ptr<Seq> Inkling2::new_seq(int limit, int initial) {
  Gpu& g = Gpu::get();
  auto s = std::make_unique<Seq>();
  s->limit = limit;
  s->tok = g.alloc(size_t(limit + 8) * 4, "ink2_tokens");
  const size_t nv = cfg_.schedule.size() + (mtp_ ? size_t(mtp_->config().num_layers) : 0);
  s->kc.resize(nv);
  s->vc.resize(nv);
  s->conv.push_back(g.alloc(conv_bytes_per_seq(cfg_), "ink2_conv"));
  if (mtp_) s->conv.push_back(g.alloc(size_t(cfg_.hidden) * 4, "ink2_mtp_feat"));
  g.make_resident(s->conv);
  reset(*s);
  const int cap = initial < 0 ? limit : std::min(initial, limit);
  if (cap > 0 && !reserve(*s, cap)) fatal("insufficient memory for a %d-token Inkling-2 KV cache", cap);
  return s;
}

bool Inkling2::ensure_rings(Seq& s) {
  Gpu& g = Gpu::get();
  const size_t kvd = size_t(cfg_.hkv) * size_t(cfg_.head_dim);
  const size_t nv = s.kc.size();
  size_t need = 0;
  for (size_t v = 0; v < nv; ++v)
    if (is_swa_slot(v) && !s.kc[v]) need += 2 * kvd * kv_elem_bytes(v) * size_t(ring_rows_);
  if (!need) return true;
  if (mem_budget_ && g.allocated_bytes() + need > mem_budget_) return false;
  std::vector<Buffer> add;
  for (size_t v = 0; v < nv; ++v)
    if (is_swa_slot(v) && !s.kc[v]) {
      const size_t row = kvd * kv_elem_bytes(v);
      s.kc[v] = g.alloc(row * size_t(ring_rows_), "ink2_kv_ring");
      s.vc[v] = g.alloc(row * size_t(ring_rows_), "ink2_kv_ring");
      memset(s.kc[v].ptr(), 0, s.kc[v].size);
      memset(s.vc[v].ptr(), 0, s.vc[v].size);
      add.push_back(s.kc[v]);
      add.push_back(s.vc[v]);
    }
  g.make_resident(add);
  return true;
}

size_t Inkling2::kv_grow_unit() const {
  const size_t kvd = size_t(cfg_.hkv) * size_t(cfg_.head_dim);
  size_t unit = kvd * (use_kv_fp8_ ? 1u : 2u);
  if (mtp_) {
    const size_t mkvd = size_t(mtp_->config().hkv) * size_t(mtp_->config().head_dim);
    unit = std::max(unit, mkvd * 2u);
  }
  return unit;
}

size_t Inkling2::kv_grow_held(const Seq& s) const {
  return kvgrow::held(s.cap, kv_bytes_per_pos(), kv_grow_unit());
}

bool Inkling2::reserve(Seq& s, int need) {
  if (need <= s.cap) return ensure_rings(s);
  if (need > s.limit) return false;
  Gpu& g = Gpu::get();
  const size_t per = kv_bytes_per_pos();
  const size_t kvd = size_t(cfg_.hkv) * size_t(cfg_.head_dim);
  const size_t nv = s.kc.size();
  size_t rings = 0;
  for (size_t v = 0; v < nv; ++v)
    if (is_swa_slot(v) && !s.kc[v]) rings += 2 * kvd * kv_elem_bytes(v) * size_t(ring_rows_);
  const size_t alloc = g.allocated_bytes();
  const size_t room = !mem_budget_ ? SIZE_MAX : (mem_budget_ > alloc + rings ? mem_budget_ - alloc - rings : 0);
  const kvgrow::Plan p = kvgrow::plan(s.cap, need, s.limit, per, kv_grow_unit(), room);
  if (!p.target) return false;
  const int target = p.target;

  const int old = s.cap;
  std::vector<Buffer> olds, news;
  std::unique_ptr<Encoder> e;
  size_t batch = 0;
  auto flush = [&] {
    if (e) e->commit_and_wait();
    e.reset();
    for (Buffer& b : olds) g.release(b);
    olds.clear();
    batch = 0;
  };
  for (size_t v = 0; v < nv; ++v) {
    if (is_swa_slot(v)) continue;
    const size_t row = kvd * kv_elem_bytes(v);
    const size_t nb = row * size_t(target);
    for (Buffer* b : {&s.kc[v], &s.vc[v]}) {
      if (batch && batch + nb > p.batch) flush();
      Buffer nbuf = g.alloc(nb, "ink2_kv_global");
      if (old > 0 && *b) {
        if (!e) e = g.begin();
        enc_copy_buf(*e, *b, nbuf, size_t(old) * row);
      }
      if (*b) olds.push_back(*b);
      *b = nbuf;
      news.push_back(nbuf);
      batch += nb;
    }
  }
  flush();
  s.cap = target;
  g.make_resident(news);
  return ensure_rings(s);
}

size_t Inkling2::kv_bytes_per_pos() const {
  return (use_kv_fp8_ ? kv_per_pos(cfg_) / 2 : kv_per_pos(cfg_)) + (mtp_ ? Ink2Mtp::kv_per_pos(mtp_->config()) : 0);
}

void Inkling2::drop_kv(Seq& s) {
  Gpu& g = Gpu::get();
  for (auto* v : {&s.kc, &s.vc})
    for (Buffer& b : *v) {
      g.release(b);
      b = Buffer{};
    }
  s.cap = 0;
}

void Inkling2::reset(Seq& s) {
  for (Buffer& b : s.conv)
    if (b) memset(b.ptr(), 0, b.size);
}

Checkpoint Inkling2::new_checkpoint() { return Checkpoint{}; }

void Inkling2::save(Encoder& e, const Seq& s, Checkpoint& c, int pos) {
  Gpu& g = Gpu::get();
  c.pos = -1;
  const size_t nv = s.kc.size();
  if (c.conv.size() != s.conv.size()) {
    for (Buffer& b : c.conv) g.release(b);
    c.conv.resize(s.conv.size());
  }
  for (size_t i = 0; i < s.conv.size(); ++i) {
    if (!s.conv[i]) continue;
    if (!c.conv[i]) {
      if (mem_budget_ && g.allocated_bytes() + s.conv[i].size > mem_budget_) return;
      c.conv[i] = g.alloc(s.conv[i].size, i == 0 ? "ink2_ckpt_conv" : "ink2_ckpt_feat");
      g.make_resident({c.conv[i]});
    }
    enc_copy_buf(e, s.conv[i], c.conv[i]);
  }
  if (c.rec.size() != 2 * nv) c.rec.assign(2 * nv, Buffer{});
  for (size_t v = 0; v < nv; ++v) {
    const bool swa = is_swa_slot(v);
    for (int kv = 0; kv < 2; ++kv) {
      const Buffer& src = kv ? s.vc[v] : s.kc[v];
      Buffer& dst = c.rec[2 * v + size_t(kv)];
      if (!swa || !src) {
        if (dst) g.release(dst), dst = Buffer{};
        continue;
      }
      if (!dst) {
        if (mem_budget_ && g.allocated_bytes() + src.size > mem_budget_) return;
        dst = g.alloc(src.size, "ink2_ckpt_ring");
        g.make_resident({dst});
      }
      enc_copy_buf(e, src, dst);
    }
  }
  c.pos = pos;
}

void Inkling2::restore(Encoder& e, Seq& s, const Checkpoint& c) {
  for (size_t i = 0; i < c.conv.size() && i < s.conv.size(); ++i) {
    if (!c.conv[i]) continue;
    if (!s.conv[i] || s.conv[i].size < c.conv[i].size)
      fatal("inkling2: restore missing or undersized conv destination buffer %zu", i);
    enc_copy_buf(e, c.conv[i], s.conv[i], c.conv[i].size);
  }
  const size_t nv = s.kc.size();
  bool need_rings = false;
  for (size_t v = 0; v < nv && 2 * v + 1 < c.rec.size(); ++v) {
    if ((c.rec[2 * v] && !s.kc[v]) || (c.rec[2 * v + 1] && !s.vc[v])) {
      need_rings = true;
      break;
    }
  }
  if (need_rings && !ensure_rings(s)) fatal("inkling2: restore could not allocate missing ring buffers");
  for (size_t v = 0; v < nv && 2 * v + 1 < c.rec.size(); ++v) {
    for (int kv = 0; kv < 2; ++kv) {
      const Buffer& src = c.rec[2 * v + size_t(kv)];
      if (!src) continue;
      const auto& dsts = kv ? s.vc : s.kc;
      if (v >= dsts.size() || !dsts[v] || dsts[v].size < src.size)
        fatal("inkling2: restore missing or undersized ring destination buffer for visit %zu (%s)", v, kv ? "v" : "k");
      enc_copy_buf(e, src, dsts[v], src.size);
    }
  }
}

// ---- primitives ----
void Inkling2::enc_mm(Encoder& e, const Tensor& w, const Buffer& x, const Buffer& y, int M, uint32_t N_limit,
                      bool add, bool exact_f32) {
  const uint32_t N = N_limit ? N_limit : uint32_t(w.shape[0]);
  const uint32_t K = uint32_t(w.shape[1]);
  if (M <= 16 && N <= 256 && (K & 3u) == 0u) {
    for (int m0 = 0; m0 < M; m0 += 8) {
      const uint32_t mc = uint32_t(std::min(8, M - m0));
      const Ink2MmP p{mc, N, K, K, N, w.dtype == DType::F32 ? 1u : 0u, add ? 1u : 0u};
      e.use(ink2_pipe("ink2_bf16_mv_smalln"));
      e.buf(0, x.view(size_t(m0) * K * 4, size_t(mc) * K * 4));
      e.buf(1, w.buf);
      e.buf(2, y.view(size_t(m0) * N * 4, size_t(mc) * N * 4));
      e.val(3, p);
      e.dispatch(N, 1, 1, 128);
    }
    return;
  }
  if (exact_f32 && M > 8 && w.dtype == DType::BF16 && (K & 127u) == 0u) {
    const Ink2MmP p{uint32_t(M), N, K, K, N, 0u, add ? 1u : 0u};
    e.use(ink2_pipe("ink2_bf16_mv8"));
    e.buf(0, x);
    e.buf(1, w.buf);
    e.buf(2, y);
    e.val(3, p);
    e.dispatch((N + 15) / 16, (uint32_t(M) + 7) / 8, 1, 128);
    return;
  }
  if (M <= 32 || exact_f32) {
    for (int m0 = 0; m0 < M; m0 += 8) {
      const uint32_t mc = uint32_t(std::min(8, M - m0));
      const Ink2MmP p{mc, N, K, K, N, w.dtype == DType::F32 ? 1u : 0u, add ? 1u : 0u};
      const bool aligned_k = w.dtype == DType::F32 ? ((K & 3u) == 0u) : ((K & 7u) == 0u);
      if (mc <= 4u && aligned_k) {
        e.use(ink2_pipe("ink2_bf16_mv4"));
        e.buf(0, x.view(size_t(m0) * K * 4, size_t(mc) * K * 4));
        e.buf(1, w.buf);
        e.buf(2, y.view(size_t(m0) * N * 4, size_t(mc) * N * 4));
        e.val(3, p);
        e.dispatch((N + 15) / 16, 1, 1, 128);
      } else if (w.dtype == DType::BF16 && (K & 127u) == 0u) {
        e.use(ink2_pipe("ink2_bf16_mv8"));
        e.buf(0, x.view(size_t(m0) * K * 4, size_t(mc) * K * 4));
        e.buf(1, w.buf);
        e.buf(2, y.view(size_t(m0) * N * 4, size_t(mc) * N * 4));
        e.val(3, p);
        e.dispatch((N + 15) / 16, 1, 1, 128);
      } else {
        e.use(ink2_pipe("ink2_bf16_mv"));
        e.buf(0, x.view(size_t(m0) * K * 4, size_t(mc) * K * 4));
        e.buf(1, w.buf);
        e.buf(2, y.view(size_t(m0) * N * 4, size_t(mc) * N * 4));
        e.val(3, p);
        e.dispatch((N + 3) / 4, 1, 1, 128);
      }
    }
  } else if (tensor_pf_ && w.dtype == DType::BF16 && K % 32 == 0) {
    const Ink2tGemmP p{uint32_t(M), N, K, K, N, add ? 1u : 0u, 0u, 0u, 0u};
    e.use(base_pipe("ink2_t_bf16_m64"));
    e.buf(0, x);
    e.buf(1, w.buf);
    e.buf(2, w.buf);
    e.buf(3, y);
    e.val(4, p);
    e.buf(5, moe_tiles_);
    e.buf(6, moe_ntiles_);
    e.buf(7, moe_sorted_);
    e.buf(8, x);
    e.dispatch((N + 63) / 64, (uint32_t(M) + 63) / 64, 1, 128);
  } else {
    const Ink2MmP p{uint32_t(M), N, K, K, N, w.dtype == DType::F32 ? 1u : 0u, add ? 1u : 0u};
    e.use(ink2_pipe("ink2_bf16_mm"));
    e.buf(0, x);
    e.buf(1, w.buf);
    e.buf(2, y);
    e.val(3, p);
    e.dispatch((N + 63) / 64, (uint32_t(M) + 31) / 32, 1, 128);
  }
}

void Inkling2::enc_norm(Encoder& e, const Buffer& x, const Tensor* w, const Buffer& out, int T, uint32_t dim) {
  const Ink2NormP p{uint32_t(T), dim, w ? 1u : 0u, (w && w->dtype == DType::F32) ? 1u : 0u, cfg_.rms_eps};
  const Pipeline pl = ink2_pipe("ink2_rmsnorm");
  e.use(pl);
  e.buf(0, x);
  e.buf(1, w ? w->buf : x);
  e.buf(2, out);
  e.val(3, p);
  e.dispatch(uint32_t(T), 1, 1, std::min(512u, pl.max_threads));
}

void Inkling2::enc_attn_res(Encoder& e, const Buffer* const* states, int M, const void* query_ptr, bool q_f32,
                            uint32_t Q, const Buffer& out, int T) {
  if (M < 1 || M > 6 || Q < 1 || Q > 4) fatal("inkling2: attn_res M=%d Q=%u out of range", M, Q);
  const Ink2AttnResP p{uint32_t(T),
                       uint32_t(cfg_.hidden),
                       uint32_t(M),
                       Q,
                       q_f32 ? 1u : 0u,
                       cfg_.rms_eps,
                       1.0f / std::sqrt(float(cfg_.hidden))};
  const Pipeline pl = ink2_pipe("ink2_attn_res");
  e.use(pl);
  for (int m = 0; m < 6; ++m) e.buf(m, m < M ? *states[m] : *states[0]);
  e.buf(6, *static_cast<const Buffer*>(query_ptr));
  e.buf(7, out);
  e.val(8, p);
  e.dispatch(uint32_t(T), 1, 1, std::min(512u, pl.max_threads));
}

void Inkling2::enc_msg_gate(Encoder& e, const Ink2Ctrl& c, const Buffer& h, const Buffer& m, const Buffer& out, int T) {
  if (tensor_pf_ && T >= 32 && cfg_.hidden <= 2560 && (cfg_.hidden & 3) == 0) {
    const uint32_t H = uint32_t(cfg_.hidden);
    const struct { uint32_t T, H; float eps; } pp{uint32_t(T), H, cfg_.rms_eps};
    e.use(ink2_pipe("ink2_msg_gate_prep"));
    e.buf(0, h);
    e.buf(1, m);
    e.buf(2, mg_cat_);
    e.val(3, pp);
    e.dispatch(uint32_t(T), 1, 1, 512);
    enc_mm(e, *c.down, mg_cat_, mg_lat_, T);
    e.group_begin();
    enc_mm(e, *c.forget, mg_lat_, mg_f_, T);
    enc_mm(e, *c.gain, mg_lat_, mg_g_, T);
    e.group_end();
    const uint32_t n = uint32_t(T) * H;
    const struct { uint32_t n; } mp{n};
    e.use(ink2_pipe("ink2_msg_gate_mix"));
    e.buf(0, h);
    e.buf(1, m);
    e.buf(2, mg_f_);
    e.buf(3, mg_g_);
    e.buf(4, out);
    e.val(5, mp);
    e.dispatch(((n + 3) / 4 + 255) / 256, 1, 1, 256);
    return;
  }
  const Ink2MsgGateP p{uint32_t(T), uint32_t(cfg_.hidden), uint32_t(cfg_.gate_rank),
                       c.down->dtype == DType::F32 ? 1u : 0u, cfg_.rms_eps};
  if (T <= 16) {
    const uint32_t H = uint32_t(cfg_.hidden), R = uint32_t(cfg_.gate_rank);
    const Pipeline prep_pl = ink2_pipe("ink2_msg_gate_prep_s");
    e.use(prep_pl);
    e.buf(0, h); e.buf(1, m); e.buf(2, mg_cat_); e.val(3, p);
    e.dispatch(uint32_t(T), 1, 1, std::min(512u, prep_pl.max_threads));
    e.use(ink2_pipe("ink2_msg_gate_down_s"));
    e.buf(0, mg_cat_); e.buf(1, c.down->buf); e.buf(2, mg_lat_); e.val(3, p);
    e.dispatch(R, (uint32_t(T) + 7) / 8, 1, 128);
    e.use(ink2_pipe("ink2_msg_gate_up_mix_s"));
    e.buf(0, h); e.buf(1, m); e.buf(2, mg_lat_);
    e.buf(3, c.forget->buf); e.buf(4, c.gain->buf); e.buf(5, out); e.val(6, p);
    e.dispatch((H + 63) / 64, (uint32_t(T) + 7) / 8, 1, 64);
    return;
  }
  const Pipeline mg_pl = ink2_pipe("ink2_msg_gate");
  e.use(mg_pl);
  e.buf(0, h);
  e.buf(1, m);
  e.buf(2, c.down->buf);
  e.buf(3, c.forget->buf);
  e.buf(4, c.gain->buf);
  e.buf(5, out);
  e.val(6, p);
  e.dispatch(uint32_t(T), 1, 1, std::min(512u, mg_pl.max_threads));
}

void Inkling2::enc_msg_gate3(Encoder& e, const Ink2Layer& L, const Buffer* const* in_ptrs, const Buffer& m0,
                             const Buffer& m1, const Buffer& m2, int T) {
  const Ink2Ctrl* cs[3] = {&L.ctrl_attn, &L.ctrl_k, &L.ctrl_v};
  const Buffer* ms[3] = {&m0, &m1, &m2};
  const Buffer* outs[3] = {&n0_, &n1_, &n2_};
  if (T <= 16 && cs[0]->down->dtype == cs[1]->down->dtype && cs[0]->down->dtype == cs[2]->down->dtype) {
    const uint32_t H = uint32_t(cfg_.hidden), R = uint32_t(cfg_.gate_rank);
    const Ink2MsgGateP p{uint32_t(T), H, R, cs[0]->down->dtype == DType::F32 ? 1u : 0u, cfg_.rms_eps};
    e.use(ink2_pipe("ink2_msg_gate_prep3_s"));
    e.buf(0, *in_ptrs[0]); e.buf(1, *in_ptrs[1]); e.buf(2, *in_ptrs[2]);
    e.buf(3, m0); e.buf(4, m1); e.buf(5, m2);
    e.buf(6, mg_cat_); e.val(7, p);
    e.dispatch(uint32_t(T), 3, 1, 256);

    e.use(ink2_pipe("ink2_msg_gate_down3_s"));
    e.buf(0, mg_cat_);
    e.buf(1, cs[0]->down->buf); e.buf(2, cs[1]->down->buf); e.buf(3, cs[2]->down->buf);
    e.buf(4, mg_lat_); e.val(5, p);
    e.dispatch(R, (uint32_t(T) + 7) / 8, 3, 128);

    e.use(ink2_pipe("ink2_msg_gate_up_mix3_s"));
    e.buf(0, *in_ptrs[0]); e.buf(1, *in_ptrs[1]); e.buf(2, *in_ptrs[2]);
    e.buf(3, m0); e.buf(4, m1); e.buf(5, m2);
    e.buf(6, mg_lat_);
    e.buf(7, cs[0]->forget->buf); e.buf(8, cs[1]->forget->buf); e.buf(9, cs[2]->forget->buf);
    e.buf(10, cs[0]->gain->buf); e.buf(11, cs[1]->gain->buf); e.buf(12, cs[2]->gain->buf);
    e.buf(13, n0_); e.buf(14, n1_); e.buf(15, n2_);
    e.val(16, p);
    e.dispatch((H + 63) / 64, (uint32_t(T) + 7) / 8, 3, 64);
    return;
  }
  const bool batched_mg = tensor_pf_ && T >= 32 && cfg_.hidden <= 2560 && (cfg_.hidden & 3) == 0;
  if (!batched_mg) e.group_begin();
  for (int i = 0; i < 3; ++i) enc_msg_gate(e, *cs[i], *in_ptrs[i], *ms[i], *outs[i], T);
  if (!batched_mg) e.group_end();
}

void Inkling2::enc_q8_mv(Encoder& e, const Buffer& wq, const Buffer& ws, uint32_t N, uint32_t K, const Buffer& x,
                         const Buffer& y, int M, bool add, float softcap) {
  const uint32_t nblk = K / 32u;
  const float inv_cap = softcap > 0.0f ? (1.0f / softcap) : 0.0f;
  for (int m0 = 0; m0 < M; m0 += 8) {
    const uint32_t mc = uint32_t(std::min(8, M - m0));
    const Ink2Q8MmP p{mc, N, K, nblk, add ? 1u : 0u, softcap, inv_cap};
    e.use(ink2_pipe(mc <= 4u ? "ink2_q8_mv4" : "ink2_q8_mv8"));
    e.buf(0, x.view(size_t(m0) * K * 4, size_t(mc) * K * 4));
    e.buf(1, wq);
    e.buf(2, ws);
    e.buf(3, y.view(size_t(m0) * N * 4, size_t(mc) * N * 4));
    e.val(4, p);
    e.dispatch((N + 7u) / 8u, 1, 1, 128);
  }
}

void Inkling2::enc_q4_mv(Encoder& e, const Buffer& wq, const Buffer& wsb, uint32_t N, uint32_t K, const Buffer& x,
                         const Buffer& y, int M, bool add) {
  const uint32_t ngrps = K / 64u;
  for (int m0 = 0; m0 < M; m0 += 8) {
    const uint32_t mc = uint32_t(std::min(8, M - m0));
    const Ink2Q4MmP p{mc, N, K, ngrps, add ? 1u : 0u};
    e.use(ink2_pipe(mc <= 4u ? "ink2_q4g64_mv4" : "ink2_q4g64_mv8"));
    e.buf(0, x.view(size_t(m0) * K * 4, size_t(mc) * K * 4));
    e.buf(1, wq);
    e.buf(2, wsb);
    e.buf(3, y.view(size_t(m0) * N * 4, size_t(mc) * N * 4));
    e.val(4, p);
    e.dispatch((N + 15u) / 16u, 1, 1, 128);
  }
}

void Inkling2::enc_unembed(Encoder& e, const Buffer& x, const Buffer& y, int M, bool apply_softcap) {
  const uint32_t V = uint32_t(cfg_.vocab);
  bool softcap_done = false;
  if (M <= 16 && unembed_q4_) {
    enc_q4_mv(e, unembed_q4_, unembed_sb_, V, uint32_t(cfg_.hidden), x, y, M, false);
  } else if (M <= 16 && unembed_q_) {
    enc_q8_mv(e, unembed_q_, unembed_s_, V, uint32_t(cfg_.hidden), x, y, M, false,
              apply_softcap ? cfg_.logit_softcap : 0.0f);
    softcap_done = true;
  } else {
    enc_mm(e, *unembed_, x, y, M, V, false);
  }
  if (apply_softcap && !softcap_done) {
    const Ink2SoftcapP sp{uint32_t(M) * V, cfg_.logit_softcap, 1.0f / cfg_.logit_softcap};
    e.use(ink2_pipe("ink2_softcap"));
    e.buf(0, y);
    e.val(1, sp);
    e.dispatch((sp.n + 255) / 256, 1, 1, 256);
  }
}

void Inkling2::enc_forward(Encoder& e, Seq& s, int pos0, int T, bool dump_logits, const Buffer* out_logits,
                           bool need_all_logits, const Buffer* diag_rows, bool capture_taps, bool save_conv_snaps,
                           const Buffer* tree_buf) {
  Ink2Grouped gp(e, true);
  const uint32_t H = uint32_t(cfg_.hidden);
  const uint32_t QD = uint32_t(cfg_.hq) * uint32_t(cfg_.head_dim);
  const uint32_t KVD = uint32_t(cfg_.hkv) * uint32_t(cfg_.head_dim);
  const size_t row_bytes = size_t(T) * H * 4;
  const Buffer& conv = s.conv[0];
  const float sconv_static_scale = 1.0f / std::sqrt(5.0f);
  const uint32_t snap_stride =
      save_conv_snaps ? uint32_t(cfg_.schedule.size() * visit_conv_floats(cfg_)) : 0u;
  const uint32_t save_snap_flag = save_conv_snaps ? 1u : 0u;

  // Embed + norm -> x_
  {
    const Ink2EmbedP ep{uint32_t(T), H, uint32_t(pos0), uint32_t(embed_->shape[0]), cfg_.rms_eps};
    const Pipeline embed_pl = ink2_pipe("ink2_embed_norm");
    e.use(embed_pl);
    e.buf(0, s.tok);
    e.buf(1, embed_->buf);
    e.buf(2, embed_norm_->buf);
    e.buf(3, x_);
    e.val(4, ep);
    e.dispatch(uint32_t(T), 1, 1, std::min(512u, embed_pl.max_threads));
  }
  if (s.embeds) vision::enc_embed_runs(e, *s.embeds, pos0, T, x_, cfg_.hidden);
  if (diag_rows) enc_copy_buf(e, x_.view(size_t(T - 1) * H * 4, H * 4), diag_rows->view(0, H * 4), H * 4);

  // Ping-pong active residual stream between x_ and inp_sa_ with zero extra copies.
  const Buffer* cur_x = &x_;
  const Buffer* alt_x = &inp_sa_;
  std::vector<const Buffer*> blocks_vec;
  bool have_prelude = false;
  bool have_prev = false;
  int n_prelude_blocks = 0;

  for (size_t v_idx = 0; v_idx < cfg_.schedule.size(); ++v_idx) {
    const int ip = cfg_.schedule[v_idx];
    const Ink2Layer& L = layers_[size_t(ip)];
    const size_t cbase = v_idx * visit_conv_floats(cfg_) * 4;
    const Buffer conv_k = conv.view(cbase + 0, 3 * KVD * 4);
    const Buffer conv_v = conv.view(cbase + 3 * KVD * 4, 3 * KVD * 4);
    const Buffer conv_a = conv.view(cbase + 6 * KVD * 4, 3 * H * 4);
    const Buffer conv_m = conv.view(cbase + 6 * KVD * 4 + 3 * H * 4, 3 * H * 4);

    if (ip == cfg_.core_start) {
      if (!have_prelude) {
        enc_copy_buf(e, *cur_x, x_pre_, row_bytes);
        n_prelude_blocks = int(blocks_vec.size());
        for (int b = 0; b < n_prelude_blocks; ++b) {
          if (blocks_vec[size_t(b)] != &prelude_blocks_[b])
            enc_copy_buf(e, *blocks_vec[size_t(b)], prelude_blocks_[b], row_bytes);
        }
        have_prelude = true;
      } else {
        // previous = (blocks[len(pre_loop[1]) + 1 :], x): blocks[2] into blocks_[4], *cur_x into blocks_[5].
        enc_copy_buf(e, *blocks_vec[size_t(n_prelude_blocks + 1)], blocks_[4], row_bytes);
        enc_copy_buf(e, *cur_x, blocks_[5], row_bytes);
        cur_x = &x_pre_;
        alt_x = &inp_sa_;
        blocks_vec.clear();
        for (int b = 0; b < n_prelude_blocks; ++b) blocks_vec.push_back(&prelude_blocks_[b]);
        have_prev = true;
        if (T <= 16) {
          const Ink2AttnResP mp{uint32_t(T), H, 2u, 4u, 1u, cfg_.rms_eps, 1.0f / std::sqrt(float(H))};
          e.use(ink2_pipe("ink2_attn_res_pass"));
          e.buf(0, blocks_[4]);
          e.buf(1, blocks_[5]);
          e.buf(2, msg_queries_);
          e.buf(3, pass_msg_out_);
          e.val(4, mp);
          e.dispatch(uint32_t(T), uint32_t(cfg_.core_end - cfg_.core_start), 1, 256);
        }
      }
    }

    if (capture_taps && mtp_) {
      const auto& taps = mtp_->config().taps;
      for (int ti = 0; ti < 3 && ti < int(taps.size()); ++ti)
        if (int(v_idx) == taps[size_t(ti)]) enc_copy_buf(e, *cur_x, tap_[ti], row_bytes);
    }

    const bool has_msgs = have_prev && ip < cfg_.core_end;
    const Buffer& msg_src = (has_msgs && T <= 16) ? pass_msg_out_ : msg_out_;
    const size_t msg_base = (has_msgs && T <= 16) ? size_t(ip - cfg_.core_start) * 4u * row_bytes : 0u;
    const Buffer msg0 = msg_src.view(msg_base + 0 * row_bytes, row_bytes);
    const Buffer msg1 = msg_src.view(msg_base + 1 * row_bytes, row_bytes);
    const Buffer msg2 = msg_src.view(msg_base + 2 * row_bytes, row_bytes);
    const Buffer msg3 = msg_src.view(msg_base + 3 * row_bytes, row_bytes);
    if (has_msgs && T > 16) {
      const Buffer* bank_states[2] = {&blocks_[4], &blocks_[5]};
      const Buffer qv = msg_queries_.view(size_t(ip - cfg_.core_start) * 4 * H * 4, 4 * H * 4);
      enc_attn_res(e, bank_states, 2, &qv, true, 4, msg_out_, T);
    }
    // Attention input AttnRes (*cur_x stays untouched until end-of-visit ping-pong write)
    const Buffer ar0 = ar_out_.view(0 * row_bytes, row_bytes);
    const Buffer ar1 = ar_out_.view(1 * row_bytes, row_bytes);
    const Buffer ar2 = ar_out_.view(2 * row_bytes, row_bytes);
    const Buffer* in_ptrs[3];
    if (ip == 0) {
      in_ptrs[0] = in_ptrs[1] = in_ptrs[2] = cur_x;
    } else {
      std::vector<const Buffer*> st = blocks_vec;
      st.push_back(cur_x);
      const Tensor* aq = attn_queries_[size_t(ip - 1)];
      const uint32_t Q = aq->shape.size() == 2 ? uint32_t(aq->shape[0]) : 1u;
      enc_attn_res(e, st.data(), int(st.size()), &aq->buf, aq->dtype == DType::F32, Q, ar_out_, T);
      if (Q == 1) {
        in_ptrs[0] = in_ptrs[1] = in_ptrs[2] = &ar_out_;
      } else {
        in_ptrs[0] = &ar0;
        in_ptrs[1] = &ar1;
        in_ptrs[2] = &ar2;
      }
    }

    if (has_msgs) {
      enc_msg_gate3(e, L, in_ptrs, msg0, msg1, msg2, T);
      const Tensor* w0 = L.is_swa ? L.attn_norm : L.q_norm;
      const Tensor* w1 = L.is_swa ? L.attn_norm : L.k_norm;
      const Tensor* w2 = L.is_swa ? L.attn_norm : L.v_norm;
      if (w0->dtype == w1->dtype && w0->dtype == w2->dtype) {
        const Ink2NormP np{uint32_t(T), H, 1u, w0->dtype == DType::F32 ? 1u : 0u, cfg_.rms_eps};
        e.use(ink2_pipe("ink2_rmsnorm3"));
        e.buf(0, n0_); e.buf(1, n1_); e.buf(2, n2_);
        e.buf(3, w0->buf); e.buf(4, w1->buf); e.buf(5, w2->buf);
        e.val(6, np);
        e.dispatch(uint32_t(T), 3, 1, 256);
      } else {
        e.group_begin();
        enc_norm(e, n0_, w0, n0_, T, H);
        enc_norm(e, n1_, w1, n1_, T, H);
        enc_norm(e, n2_, w2, n2_, T, H);
        e.group_end();
      }
    } else if (L.is_swa) {
      enc_norm(e, *in_ptrs[0], L.attn_norm, n0_, T, H);
    } else {
      e.group_begin();
      enc_norm(e, *in_ptrs[0], L.q_norm, n0_, T, H);
      enc_norm(e, *in_ptrs[1], L.k_norm, n1_, T, H);
      enc_norm(e, *in_ptrs[2], L.v_norm, n2_, T, H);
      e.group_end();
    }

    const Buffer& kn = (has_msgs || !L.is_swa) ? n1_ : n0_;
    const Buffer& vn = (has_msgs || !L.is_swa) ? n2_ : n0_;

    const bool pf_sconv_kv = L.is_swa && T >= 16 && !save_conv_snaps && !tree_buf && KVD <= H;
    const uint32_t HQDREL = uint32_t(cfg_.hq) * uint32_t(cfg_.d_rel);
    if (L.qkvrsg_q && T <= 16 && (H & 255u) == 0u) {
      for (int m0 = 0; m0 < T; m0 += 8) {
        const uint32_t mc = uint32_t(std::min(8, T - m0));
        Ink2Q8Multi6P mp{mc, H, L.qkvrsg_offs[6], 6u, H / 32u, {}, {}};
        for (uint32_t i = 0; i < 7u; ++i) mp.offs[i] = L.qkvrsg_offs[i];
        for (uint32_t i = 0; i < 6u; ++i) mp.lens[i] = L.qkvrsg_lens[i];
        e.use(ink2_pipe(mc <= 4u ? "ink2_q8_mv4_multi6" : "ink2_q8_mv8_multi6"));
        e.buf(0, n0_.view(size_t(m0) * H * 4, size_t(mc) * H * 4));
        e.buf(1, kn.view(size_t(m0) * H * 4, size_t(mc) * H * 4));
        e.buf(2, vn.view(size_t(m0) * H * 4, size_t(mc) * H * 4));
        e.buf(3, L.qkvrsg_q);
        e.buf(4, L.qkvrsg_s);
        e.buf(5, q_.view(size_t(m0) * QD * 4, size_t(mc) * QD * 4));
        e.buf(6, r_.view(size_t(m0) * HQDREL * 4, size_t(mc) * HQDREL * 4));
        e.buf(7, sinks_.view(size_t(m0) * uint32_t(cfg_.hq) * 4, size_t(mc) * uint32_t(cfg_.hq) * 4));
        e.buf(8, gate_attn_.view(size_t(m0) * QD * 4, size_t(mc) * QD * 4));
        e.buf(9, k_.view(size_t(m0) * KVD * 4, size_t(mc) * KVD * 4));
        e.buf(10, v_.view(size_t(m0) * KVD * 4, size_t(mc) * KVD * 4));
        e.val(11, mp);
        e.dispatch((mp.N_tot + 7u) / 8u, 1, 1, 128);
      }
    } else if (L.qkvrsg_w && T <= 16 && (H & 7u) == 0u) {
      for (int m0 = 0; m0 < T; m0 += 8) {
        const uint32_t mc = uint32_t(std::min(8, T - m0));
        Ink2MultiMmP mp{mc, H, L.qkvrsg_offs[6], 6u, {}, {}};
        for (uint32_t i = 0; i < 7u; ++i) mp.offs[i] = L.qkvrsg_offs[i];
        for (uint32_t i = 0; i < 6u; ++i) mp.lens[i] = L.qkvrsg_lens[i];
        e.use(ink2_pipe(mc <= 4u ? "ink2_bf16_mv4_multi6" : "ink2_bf16_mv8_multi6"));
        e.buf(0, n0_.view(size_t(m0) * H * 4, size_t(mc) * H * 4));
        e.buf(1, kn.view(size_t(m0) * H * 4, size_t(mc) * H * 4));
        e.buf(2, vn.view(size_t(m0) * H * 4, size_t(mc) * H * 4));
        e.buf(3, L.qkvrsg_w);
        e.buf(4, q_.view(size_t(m0) * QD * 4, size_t(mc) * QD * 4));
        e.buf(5, r_.view(size_t(m0) * HQDREL * 4, size_t(mc) * HQDREL * 4));
        e.buf(6, sinks_.view(size_t(m0) * uint32_t(cfg_.hq) * 4, size_t(mc) * uint32_t(cfg_.hq) * 4));
        e.buf(7, gate_attn_.view(size_t(m0) * QD * 4, size_t(mc) * QD * 4));
        e.buf(8, k_.view(size_t(m0) * KVD * 4, size_t(mc) * KVD * 4));
        e.buf(9, v_.view(size_t(m0) * KVD * 4, size_t(mc) * KVD * 4));
        e.val(10, mp);
        e.dispatch((mp.N_tot + 15) / 16, 1, 1, 128);
      }
    } else {
      e.group_begin();
      enc_mm(e, *L.wq_du, n0_, q_, T);
      enc_mm(e, *L.wk_dv, kn, pf_sconv_kv ? moe_out_ : k_, T);
      enc_mm(e, *L.wv_dv, vn, pf_sconv_kv ? h_mlp_ : v_, T);
      enc_mm(e, *L.wr_du, n0_, r_, T);
      enc_mm(e, *L.attn_sink_proj, n0_, sinks_, T);
      enc_mm(e, *L.gate_attn, n0_, gate_attn_, T);
      e.group_end();
    }

    // K / V short convolutions
    if (L.is_swa) {
      if (pf_sconv_kv) {
        e.group_begin();
        const Ink2SconvStaticP sp{uint32_t(T), KVD, L.k_sconv->dtype == DType::F32 ? 1u : 0u, 0u, 0u, 0u,
                                  sconv_static_scale};
        e.use(ink2_pipe("ink2_sconv_static_pf"));
        e.buf(0, moe_out_);
        e.buf(1, L.k_sconv->buf);
        e.buf(2, conv_k);
        e.buf(3, k_);
        e.val(4, sp);
        e.buf(5, k_);
        e.dispatch((KVD + 127) / 128, (uint32_t(T) + 15) / 16, 1, 128);

        const Ink2SconvStaticP svp{uint32_t(T), KVD, L.v_sconv->dtype == DType::F32 ? 1u : 0u, 0u, 0u, 0u,
                                   sconv_static_scale};
        e.use(ink2_pipe("ink2_sconv_static_pf"));
        e.buf(0, h_mlp_);
        e.buf(1, L.v_sconv->buf);
        e.buf(2, conv_v);
        e.buf(3, v_);
        e.val(4, svp);
        e.buf(5, v_);
        e.dispatch((KVD + 127) / 128, (uint32_t(T) + 15) / 16, 1, 128);
        e.group_end();

        e.group_begin();
        enc_copy_buf(e, moe_out_.view(size_t(T - 3) * KVD * 4, 3 * size_t(KVD) * 4), conv_k, 3 * size_t(KVD) * 4);
        enc_copy_buf(e, h_mlp_.view(size_t(T - 3) * KVD * 4, 3 * size_t(KVD) * 4), conv_v, 3 * size_t(KVD) * 4);
        e.group_end();
      } else {
        e.group_begin();
        const Ink2SconvStaticP sp{uint32_t(T), KVD, L.k_sconv->dtype == DType::F32 ? 1u : 0u, save_snap_flag,
                                  snap_stride, 0u, sconv_static_scale};
        e.use(ink2_pipe(tree_buf ? "ink2_sconv_static_tree" : "ink2_sconv_static"));
        e.buf(0, k_);
        e.buf(1, L.k_sconv->buf);
        e.buf(2, conv_k);
        e.buf(3, k_);
        e.val(4, sp);
        e.buf(5, save_conv_snaps ? verify_conv_.view(cbase, verify_conv_.size - cbase) : conv_k);
        e.buf(6, k_);
        if (tree_buf) e.buf(7, *tree_buf);
        e.dispatch((KVD + 255) / 256, 1, 1, 256);

        const Ink2SconvStaticP svp{uint32_t(T), KVD, L.v_sconv->dtype == DType::F32 ? 1u : 0u, save_snap_flag,
                                   snap_stride, 0u, sconv_static_scale};
        e.use(ink2_pipe(tree_buf ? "ink2_sconv_static_tree" : "ink2_sconv_static"));
        e.buf(0, v_);
        e.buf(1, L.v_sconv->buf);
        e.buf(2, conv_v);
        e.buf(3, v_);
        e.val(4, svp);
        e.buf(5, save_conv_snaps ? verify_conv_.view(cbase + 3 * KVD * 4, verify_conv_.size - (cbase + 3 * KVD * 4))
                                 : conv_v);
        e.buf(6, v_);
        if (tree_buf) e.buf(7, *tree_buf);
        e.dispatch((KVD + 255) / 256, 1, 1, 256);
        e.group_end();
      }
    } else {
      e.group_begin();
      enc_mm(e, *L.k_proj, kn, dyn_, T);
      enc_mm(e, *L.v_proj, vn, dyn_v_, T);
      e.group_end();

      e.group_begin();
      const Ink2SconvDynP dp{uint32_t(T), KVD, uint32_t(cfg_.hkv), L.k_static->dtype == DType::F32 ? 1u : 0u,
                             save_snap_flag, snap_stride};
      e.use(ink2_pipe(tree_buf ? "ink2_sconv_dynamic_tree" : "ink2_sconv_dynamic"));
      e.buf(0, k_);
      e.buf(1, dyn_);
      e.buf(2, L.k_static->buf);
      e.buf(3, conv_k);
      e.buf(4, k_);
      e.val(5, dp);
      e.buf(6, save_conv_snaps ? verify_conv_.view(cbase, verify_conv_.size - cbase) : conv_k);
      if (tree_buf) e.buf(7, *tree_buf);
      e.dispatch((KVD + 255) / 256, 1, 1, 256);

      const Ink2SconvDynP dvp{uint32_t(T), KVD, uint32_t(cfg_.hkv), L.v_static->dtype == DType::F32 ? 1u : 0u,
                              save_snap_flag, snap_stride};
      e.use(ink2_pipe(tree_buf ? "ink2_sconv_dynamic_tree" : "ink2_sconv_dynamic"));
      e.buf(0, v_);
      e.buf(1, dyn_v_);
      e.buf(2, L.v_static->buf);
      e.buf(3, conv_v);
      e.buf(4, v_);
      e.val(5, dvp);
      e.buf(6, save_conv_snaps ? verify_conv_.view(cbase + 3 * KVD * 4, verify_conv_.size - (cbase + 3 * KVD * 4))
                               : conv_v);
      if (tree_buf) e.buf(7, *tree_buf);
      e.dispatch((KVD + 255) / 256, 1, 1, 256);
      e.group_end();
    }

    // Q/K/R clipping + V norm + log scaling + KV cache write
    const uint32_t kv_cap = L.is_swa ? uint32_t(ring_rows_) : uint32_t(s.cap);
    const uint32_t kv_fp8_flag = use_kv_fp8_ ? 1u : 0u;
    {
      const Ink2QkvrP qp{uint32_t(T),
                         uint32_t(cfg_.hq),
                         uint32_t(cfg_.hkv),
                         uint32_t(cfg_.head_dim),
                         uint32_t(cfg_.d_rel),
                         uint32_t(pos0),
                         kv_cap,
                         L.is_swa ? 1u : 0u,
                         L.q_scale->dtype == DType::F32 ? 1u : 0u,
                         cfg_.qk_clip,
                         cfg_.rel_clip,
                         cfg_.rms_eps,
                         cfg_.log_floor,
                         cfg_.log_alpha,
                         kv_fp8_flag};
      e.use(ink2_pipe(tree_buf ? "ink2_qkvr_prep_tree" : "ink2_qkvr_prep"));
      e.buf(0, q_);
      e.buf(1, k_);
      e.buf(2, v_);
      e.buf(3, r_);
      e.buf(4, L.q_scale->buf);
      e.buf(5, L.rel_scale->buf);
      e.buf(6, s.kc[v_idx]);
      e.buf(7, s.vc[v_idx]);
      e.val(8, qp);
      if (tree_buf) e.buf(9, *tree_buf);
      e.dispatch(uint32_t(2 * cfg_.hq + 2 * cfg_.hkv), uint32_t(T), 1, uint32_t(cfg_.head_dim));
    }

    // Rel bias [T, Hq, rel_extent]
    {
      const Ink2RelP rp{uint32_t(T), uint32_t(cfg_.hq), uint32_t(cfg_.d_rel), uint32_t(cfg_.rel_extent),
                        L.rel_proj->dtype == DType::F32 ? 1u : 0u};
      const uint32_t active_ext = std::min(uint32_t(cfg_.rel_extent), uint32_t(pos0 + T));
      if (cfg_.d_rel == 16 && T >= 8) {
        e.use(ink2_pipe("ink2_rel_bias_pf"));
        e.buf(0, r_);
        e.buf(1, L.rel_proj->buf);
        e.buf(2, rb_);
        e.val(3, rp);
        e.dispatch((active_ext + 31) / 32, (uint32_t(T) + 7) / 8, (uint32_t(cfg_.hq) + 7) / 8, 128);
      } else {
        e.use(ink2_pipe("ink2_rel_bias"));
        e.buf(0, r_);
        e.buf(1, L.rel_proj->buf);
        e.buf(2, rb_);
        e.val(3, rp);
        e.dispatch((active_ext + 31) / 32, uint32_t(cfg_.hq), uint32_t(T), 32);
      }
    }

    // Attention
    const float attn_scale = 1.0f / std::sqrt(float(cfg_.head_dim));
    const bool use_gqa = (cfg_.hq == cfg_.hkv * 3 && cfg_.head_dim == 128);
    if (T == 1 && !tree_buf) {
      const uint32_t lk = L.is_swa ? std::min(uint32_t(pos0 + 1), uint32_t(cfg_.swa)) : uint32_t(pos0 + 1);
      uint32_t chunk =
          (use_gqa && lk <= 32u) ? lk : std::max(use_gqa ? 16u : 32u, (lk + kMaxAttnChunks - 1) / kMaxAttnChunks);
      chunk = (chunk + 3) / 4 * 4;
      const uint32_t nchunks = std::max(1u, (lk + chunk - 1) / chunk);
      const Ink2AttnDecP adp{uint32_t(pos0),
                             kv_cap,
                             uint32_t(cfg_.hq),
                             uint32_t(cfg_.hkv),
                             uint32_t(cfg_.head_dim),
                             uint32_t(cfg_.rel_extent),
                             L.is_swa ? 1u : 0u,
                             uint32_t(cfg_.swa),
                             chunk,
                             nchunks,
                             attn_scale,
                             kv_fp8_flag};
      if (nchunks == 1u) {
        if (use_gqa) {
          e.use(ink2_pipe(L.is_swa ? "ink2_attn_decode_1pass_gqa_swa" : "ink2_attn_decode_1pass_gqa_global"));
        } else {
          e.use(ink2_pipe("ink2_attn_decode_1pass"));
        }
        e.buf(0, q_);
        e.buf(1, s.kc[v_idx]);
        e.buf(2, s.vc[v_idx]);
        e.buf(3, rb_);
        e.buf(4, sinks_);
        e.buf(5, gate_attn_);
        e.buf(6, att_out_);
        e.val(7, adp);
        e.dispatch(use_gqa ? uint32_t(cfg_.hkv) : uint32_t(cfg_.hq), 1, 1, 128);
      } else {
        if (use_gqa) {
          e.use(ink2_pipe(L.is_swa ? "ink2_attn_decode_gqa_swa" : "ink2_attn_decode_gqa_global"));
        } else {
          e.use(ink2_pipe("ink2_attn_decode"));
        }
        e.buf(0, q_);
        e.buf(1, s.kc[v_idx]);
        e.buf(2, s.vc[v_idx]);
        e.buf(3, rb_);
        e.buf(4, part_m_);
        e.buf(5, part_l_);
        e.buf(6, part_o_);
        e.val(7, adp);
        e.dispatch(use_gqa ? uint32_t(cfg_.hkv) : uint32_t(cfg_.hq), nchunks, 1, 128);

        e.use(ink2_pipe("ink2_attn_decode_reduce"));
        e.buf(0, part_m_);
        e.buf(1, part_l_);
        e.buf(2, part_o_);
        e.buf(3, sinks_);
        e.buf(4, gate_attn_);
        e.buf(5, att_out_);
        e.val(6, adp);
        e.dispatch(uint32_t(cfg_.hq), 1, 1, 32);
      }
    } else if ((tree_buf && use_gqa && T <= 16) ||
               ((use_gqa ? (T >= 2 && T <= 16) : (T >= 2 && T <= 8)) && (pos0 + T) >= 64)) {
      const uint32_t k_min = (L.is_swa && pos0 + 1 > cfg_.swa) ? uint32_t(pos0 + 1 - cfg_.swa) : 0u;
      const uint32_t lk = uint32_t(pos0 + T) - k_min;
      uint32_t chunk = std::max(use_gqa ? 16u : 32u, (lk + kMaxAttnChunks - 1) / kMaxAttnChunks);
      chunk = (chunk + 3) / 4 * 4;
      const uint32_t nchunks = std::max(1u, (lk + chunk - 1) / chunk);
      const Ink2AttnVerP avp{uint32_t(T),
                             uint32_t(pos0),
                             kv_cap,
                             uint32_t(cfg_.hq),
                             uint32_t(cfg_.hkv),
                             uint32_t(cfg_.head_dim),
                             uint32_t(cfg_.rel_extent),
                             L.is_swa ? 1u : 0u,
                             uint32_t(cfg_.swa),
                             chunk,
                             nchunks,
                             attn_scale,
                             kv_fp8_flag};
      if (use_gqa) {
        e.use(ink2_pipe(tree_buf ? "ink2_attn_verify_tree_gqa" : "ink2_attn_verify_gqa"));
        e.buf(0, q_);
        e.buf(1, s.kc[v_idx]);
        e.buf(2, s.vc[v_idx]);
        e.buf(3, rb_);
        e.buf(4, part_m_);
        e.buf(5, part_l_);
        e.buf(6, part_o_);
        e.val(7, avp);
        if (tree_buf) e.buf(8, *tree_buf);
        e.dispatch(uint32_t(cfg_.hkv), nchunks, (uint32_t(T) + 3u) / 4u, 192);
      } else {
        e.use(ink2_pipe(T == 2 ? "ink2_attn_verify2" : "ink2_attn_verify"));
        e.buf(0, q_);
        e.buf(1, s.kc[v_idx]);
        e.buf(2, s.vc[v_idx]);
        e.buf(3, rb_);
        e.buf(4, part_m_);
        e.buf(5, part_l_);
        e.buf(6, part_o_);
        e.val(7, avp);
        e.dispatch(uint32_t(cfg_.hq), nchunks, 1, 128);
      }

      e.use(ink2_pipe("ink2_attn_verify_reduce"));
      e.buf(0, part_m_);
      e.buf(1, part_l_);
      e.buf(2, part_o_);
      e.buf(3, sinks_);
      e.buf(4, gate_attn_);
      e.buf(5, att_out_);
      e.val(6, avp);
      e.dispatch(uint32_t(cfg_.hq), uint32_t(T), 1, 32);
    } else {
      const Ink2AttnPfP apf{uint32_t(T),
                            uint32_t(pos0),
                            kv_cap,
                            uint32_t(cfg_.hq),
                            uint32_t(cfg_.hkv),
                            uint32_t(cfg_.head_dim),
                            uint32_t(cfg_.rel_extent),
                            L.is_swa ? 1u : 0u,
                            uint32_t(cfg_.swa),
                            attn_scale,
                            kv_fp8_flag};
      static const bool qat_env = !getenv("NI_INK2_ATTN_T") || atoi(getenv("NI_INK2_ATTN_T")) != 0;
      static const bool qat_small = getenv("MTL_SHADER_VALIDATION") && atoi(getenv("MTL_SHADER_VALIDATION")) != 0;
      const bool use_qat =
          qat_env && !use_kv_fp8_ && tensor_pf_ && T >= 16 && cfg_.head_dim == 128 && cfg_.hq == cfg_.hkv * 3;
      if (use_qat) {
        const uint32_t qat_tok = qat_small ? 16u : 32u;
        e.use(base_pipe(qat_small ? "qattn_ink2_s3k16" : "qattn_ink2_s6k32p"));
        e.buf(0, q_);
        e.buf(1, s.kc[v_idx]);
        e.buf(2, s.vc[v_idx]);
        e.buf(3, rb_);
        e.buf(4, sinks_);
        e.buf(5, gate_attn_);
        e.buf(6, att_out_);
        e.val(7, apf);
        e.dispatch((uint32_t(T) + qat_tok - 1) / qat_tok, uint32_t(cfg_.hkv), 1, qat_small ? 96 : 192);
      } else {
        e.use(ink2_pipe("ink2_attn_prefill"));
        e.buf(0, q_);
        e.buf(1, s.kc[v_idx]);
        e.buf(2, s.vc[v_idx]);
        e.buf(3, rb_);
        e.buf(4, sinks_);
        e.buf(5, gate_attn_);
        e.buf(6, att_out_);
        e.val(7, apf);
        e.dispatch(uint32_t(cfg_.hq), uint32_t(T), 1, 128);
      }
    }

    // wo_ud + attn_sconv (+ *cur_x residual on non-boundary layers -> ffn_inp_)
    if (L.wo_q && T <= 16) enc_q8_mv(e, L.wo_q, L.wo_s, H, QD, att_out_, a_out_, T);
    else enc_mm(e, *L.wo_ud, att_out_, a_out_, T);
    const bool is_boundary = (ip % cfg_.block_sz == 0);
    if (T >= 16 && !save_conv_snaps && !tree_buf) {
      const Ink2SconvStaticP asp{uint32_t(T), H, L.attn_sconv->dtype == DType::F32 ? 1u : 0u, 0u, 0u,
                                 is_boundary ? 0u : 1u, sconv_static_scale};
      e.use(ink2_pipe("ink2_sconv_static_pf"));
      e.buf(0, a_out_);
      e.buf(1, L.attn_sconv->buf);
      e.buf(2, conv_a);
      e.buf(3, ffn_inp_);
      e.val(4, asp);
      e.buf(5, *cur_x);
      e.dispatch((H + 127) / 128, (uint32_t(T) + 15) / 16, 1, 128);
      enc_copy_buf(e, a_out_.view(size_t(T - 3) * H * 4, 3 * size_t(H) * 4), conv_a, 3 * size_t(H) * 4);
    } else {
      const Ink2SconvStaticP asp{uint32_t(T), H, L.attn_sconv->dtype == DType::F32 ? 1u : 0u, save_snap_flag,
                                 snap_stride, is_boundary ? 0u : 1u, sconv_static_scale};
      e.use(ink2_pipe(tree_buf ? "ink2_sconv_static_tree" : "ink2_sconv_static"));
      e.buf(0, a_out_);
      e.buf(1, L.attn_sconv->buf);
      e.buf(2, conv_a);
      e.buf(3, ffn_inp_);
      e.val(4, asp);
      e.buf(5, save_conv_snaps ? verify_conv_.view(cbase + 6 * KVD * 4, verify_conv_.size - (cbase + 6 * KVD * 4))
                               : conv_a);
      e.buf(6, *cur_x);
      if (tree_buf) e.buf(7, *tree_buf);
      e.dispatch((H + 255) / 256, 1, 1, 256);
    }
    if (is_boundary) {
      const int bslot = ip / cfg_.block_sz;
      const Buffer& dst_b = (bslot == 0) ? prelude_blocks_[0] : blocks_[bslot];
      enc_copy_buf(e, *cur_x, dst_b, row_bytes);
      blocks_vec.push_back(&dst_b);
    }

    // MLP AttnRes + optional MessageGate + norm
    {
      std::vector<const Buffer*> st = blocks_vec;
      st.push_back(&ffn_inp_);
      const Tensor* mq = mlp_queries_[size_t(ip)];
      enc_attn_res(e, st.data(), int(st.size()), &mq->buf, mq->dtype == DType::F32, 1, h_mlp_, T);
    }
    if (has_msgs) enc_msg_gate(e, L.ctrl_mlp, h_mlp_, msg3, h_mlp_, T);
    enc_norm(e, h_mlp_, L.mlp_norm, x_mlp_, T, H);

    // Router + MoE (router uses exact FP32 math on all M so prefill top-k matches decode)
    enc_mm(e, *L.router_w, x_mlp_, rlog_, T, 0, false, true);
    {
      const Ink2RouteP rp{uint32_t(T), uint32_t(cfg_.n_experts), uint32_t(cfg_.top_k),
                          L.router_bias->dtype == DType::F32 ? 1u : 0u, cfg_.route_scale};
      e.use(ink2_pipe("ink2_moe_route"));
      e.buf(0, rlog_);
      e.buf(1, L.router_bias->buf);
      e.buf(2, eid_);
      e.buf(3, ew_);
      e.val(4, rp);
      e.dispatch(uint32_t(T), 1, 1, 32);
    }
    {
      if (tensor_pf_ && T >= 32) {
        const uint32_t P = uint32_t(T) * uint32_t(cfg_.top_k);
        const uint32_t E = uint32_t(cfg_.n_experts);
        const uint32_t I = uint32_t(cfg_.moe_I);
        const uint32_t per = P / E;
        const uint32_t bm = per >= 40 ? 64u : per >= 12 ? 32u : 16u;
        const std::string ms = "_m" + std::to_string(bm);
        const uint32_t max_tiles = P / bm + E + 1;
        {
          const Ink2BucketP bp{P, E, bm};
          e.use(ink2_pipe("ink2_moe_bucket"));
          e.buf(0, eid_);
          e.buf(1, moe_sorted_);
          e.buf(2, moe_tiles_);
          e.buf(3, moe_ntiles_);
          e.buf(4, moe_rank_);
          e.val(5, bp);
          e.dispatch(1, 1, 1, 256);
        }
        const uint32_t w13_q_stride = L.is_mxfp8 ? (2 * I * H) : (2 * I * (H / 2));
        const uint32_t w13_s_stride = L.is_mxfp8 ? (2 * I * (H / 32)) : (2 * I * (H / 16));
        {
          const Ink2tGemmP gp{P, 2 * I, H, H, 2 * I, 0u, w13_q_stride, w13_s_stride, uint32_t(cfg_.top_k)};
          e.use(base_pipe((L.is_mxfp8 ? "ink2_t_mxfp8_moe_w13" : "ink2_t_nvfp4_moe_w13") + ms));
          e.buf(0, x_mlp_);
          e.buf(1, L.w13->buf);
          e.buf(2, L.w13_sc->buf);
          e.buf(3, moe_tmp_);
          e.val(4, gp);
          e.buf(5, moe_tiles_);
          e.buf(6, moe_ntiles_);
          e.buf(7, moe_sorted_);
          e.buf(8, L.is_mxfp8 ? x_mlp_ : L.w13_outer);
          e.dispatch((2 * I + 63) / 64, max_tiles, 1, 128);
        }
        {
          const Ink2SituP sp{P * I};
          e.use(ink2_pipe("ink2_moe_situ_half"));
          e.buf(0, moe_tmp_);
          e.buf(1, moe_act_h_);
          e.val(2, sp);
          e.dispatch((sp.n + 255) / 256, 1, 1, 256);
        }
        const uint32_t w2_q_stride = L.is_mxfp8 ? (H * I) : (H * (I / 2));
        const uint32_t w2_s_stride = L.is_mxfp8 ? (H * (I / 32)) : (H * (I / 16));
        {
          const Ink2tGemmP gp{P, H, I, I, H, 0u, w2_q_stride, w2_s_stride, uint32_t(cfg_.top_k)};
          e.use(base_pipe((L.is_mxfp8 ? "ink2_t_mxfp8_moe_w2" : "ink2_t_nvfp4_moe_w2") + ms));
          e.buf(0, moe_act_h_);
          e.buf(1, L.w2->buf);
          e.buf(2, L.w2_sc->buf);
          e.buf(3, moe_tmp_);
          e.val(4, gp);
          e.buf(5, moe_tiles_);
          e.buf(6, moe_ntiles_);
          e.buf(7, moe_sorted_);
          e.buf(8, L.is_mxfp8 ? x_mlp_ : L.w2_outer);
          e.dispatch((H + 63) / 64, max_tiles, 1, 128);
        }
        {
          const Ink2CombineP cp{uint32_t(T), H, uint32_t(cfg_.top_k)};
          e.use(ink2_pipe("ink2_moe_combine"));
          e.buf(0, moe_tmp_);
          e.buf(1, ew_);
          e.buf(2, moe_out_);
          e.val(3, cp);
          e.dispatch(uint32_t(T), 1, 1, 256);
        }
      } else {
        const Ink2MoeP mp{uint32_t(T), H, uint32_t(cfg_.moe_I), uint32_t(cfg_.top_k)};
        if (T >= 2 && T <= 31 && moe_tiles_) {
          const struct {
            uint32_t T, K_top, E;
          } bp{uint32_t(T), uint32_t(cfg_.top_k), uint32_t(cfg_.n_experts)};
          e.use(ink2_pipe("ink2_moe_bucket_small"));
          e.buf(0, eid_);
          e.buf(1, ew_);
          e.buf(2, moe_tiles_);
          e.buf(3, moe_ntiles_);
          e.buf(4, moe_st_tok_);
          e.buf(5, moe_st_pair_);
          e.buf(6, moe_st_ew_);
          e.val(7, bp);
          e.dispatch(1, 1, 1, 128);

          const uint32_t max_e = std::min(uint32_t(cfg_.n_experts), uint32_t(T) * uint32_t(cfg_.top_k));
          if (!L.is_mxfp8) {
            e.use(ink2_pipe("ink2_moe_nvfp4_w13_g"));
            e.buf(0, x_mlp_);
            e.buf(1, L.w13->buf);
            e.buf(2, L.w13_sc->buf);
            e.buf(3, L.w13_outer);
            e.buf(4, moe_tiles_);
            e.buf(5, moe_ntiles_);
            e.buf(6, moe_st_tok_);
            e.buf(7, moe_st_pair_);
            e.buf(8, moe_act_);
            e.val(9, mp);
            e.dispatch((uint32_t(cfg_.moe_I) + 15) / 16, max_e, 1, 128);

            e.use(ink2_pipe("ink2_moe_nvfp4_w2_g"));
            e.buf(0, moe_act_);
            e.buf(1, L.w2->buf);
            e.buf(2, L.w2_sc->buf);
            e.buf(3, L.w2_outer);
            e.buf(4, moe_tiles_);
            e.buf(5, moe_ntiles_);
            e.buf(6, moe_st_pair_);
            e.buf(7, moe_st_ew_);
            e.buf(8, moe_pair_out_);
            e.val(9, mp);
            e.dispatch((H + 15) / 16, max_e, 1, 128);
          } else {
            e.use(ink2_pipe("ink2_moe_mxfp8_w13_g"));
            e.buf(0, x_mlp_);
            e.buf(1, L.w13->buf);
            e.buf(2, L.w13_sc->buf);
            e.buf(3, moe_tiles_);
            e.buf(4, moe_ntiles_);
            e.buf(5, moe_st_tok_);
            e.buf(6, moe_st_pair_);
            e.buf(7, moe_act_);
            e.val(8, mp);
            e.dispatch((uint32_t(cfg_.moe_I) + 15) / 16, max_e, 1, 128);

            e.use(ink2_pipe("ink2_moe_mxfp8_w2_g"));
            e.buf(0, moe_act_);
            e.buf(1, L.w2->buf);
            e.buf(2, L.w2_sc->buf);
            e.buf(3, moe_tiles_);
            e.buf(4, moe_ntiles_);
            e.buf(5, moe_st_pair_);
            e.buf(6, moe_st_ew_);
            e.buf(7, moe_pair_out_);
            e.val(8, mp);
            e.dispatch((H + 31) / 32, max_e, 1, 128);
          }

          e.use(ink2_pipe("ink2_moe_reduce8_m3"));
          e.buf(0, moe_pair_out_);
          e.buf(1, moe_out_);
          e.val(2, mp);
          e.dispatch(((H / 4) + 63) / 64, uint32_t(T), 1, 64);
        } else if (!L.is_mxfp8) {
          e.use(ink2_pipe("ink2_moe_nvfp4_w13_situ"));
          e.buf(0, x_mlp_);
          e.buf(1, eid_);
          e.buf(2, L.w13->buf);
          e.buf(3, L.w13_sc->buf);
          e.buf(4, L.w13_outer);
          e.buf(5, moe_act_);
          e.val(6, mp);
          e.dispatch((uint32_t(cfg_.moe_I) + 15) / 16, uint32_t(cfg_.top_k), uint32_t(T), 128);

          e.use(ink2_pipe("ink2_moe_nvfp4_w2_acc"));
          e.buf(0, moe_act_);
          e.buf(1, eid_);
          e.buf(2, ew_);
          e.buf(3, L.w2->buf);
          e.buf(4, L.w2_sc->buf);
          e.buf(5, L.w2_outer);
          e.buf(6, moe_out_);
          e.val(7, mp);
          e.dispatch((H + 7u) / 8u, uint32_t(T), 1, 128);
        } else {
          e.use(ink2_pipe("ink2_moe_mxfp8_w13_situ"));
          e.buf(0, x_mlp_);
          e.buf(1, eid_);
          e.buf(2, L.w13->buf);
          e.buf(3, L.w13_sc->buf);
          e.buf(4, moe_act_);
          e.val(5, mp);
          e.dispatch((uint32_t(cfg_.moe_I) + 15) / 16, uint32_t(cfg_.top_k), uint32_t(T), 128);

          e.use(ink2_pipe("ink2_moe_mxfp8_w2_acc"));
          e.buf(0, moe_act_);
          e.buf(1, eid_);
          e.buf(2, ew_);
          e.buf(3, L.w2->buf);
          e.buf(4, L.w2_sc->buf);
          e.buf(5, moe_out_);
          e.val(6, mp);
          e.dispatch((H + 15) / 16, uint32_t(T), 1, 128);
        }
      }
    }

    // mlp_sconv + residual add -> *alt_x, then swap(cur_x, alt_x)
    if (T >= 16 && !save_conv_snaps && !tree_buf) {
      const Ink2SconvStaticP msp{uint32_t(T), H, L.mlp_sconv->dtype == DType::F32 ? 1u : 0u, 0u, 0u, 1u,
                                 sconv_static_scale};
      e.use(ink2_pipe("ink2_sconv_static_pf"));
      e.buf(0, moe_out_);
      e.buf(1, L.mlp_sconv->buf);
      e.buf(2, conv_m);
      e.buf(3, *alt_x);
      e.val(4, msp);
      e.buf(5, ffn_inp_);
      e.dispatch((H + 127) / 128, (uint32_t(T) + 15) / 16, 1, 128);
      enc_copy_buf(e, moe_out_.view(size_t(T - 3) * H * 4, 3 * size_t(H) * 4), conv_m, 3 * size_t(H) * 4);
    } else {
      const Ink2SconvStaticP msp{uint32_t(T), H, L.mlp_sconv->dtype == DType::F32 ? 1u : 0u, save_snap_flag,
                                 snap_stride, 1u, sconv_static_scale};
      e.use(ink2_pipe(tree_buf ? "ink2_sconv_static_tree" : "ink2_sconv_static"));
      e.buf(0, moe_out_);
      e.buf(1, L.mlp_sconv->buf);
      e.buf(2, conv_m);
      e.buf(3, *alt_x);
      e.val(4, msp);
      e.buf(5, save_conv_snaps
                   ? verify_conv_.view(cbase + 6 * KVD * 4 + 3 * H * 4, verify_conv_.size - (cbase + 6 * KVD * 4 + 3 * H * 4))
                   : conv_m);
      e.buf(6, ffn_inp_);
      if (tree_buf) e.buf(7, *tree_buf);
      e.dispatch((H + 255) / 256, 1, 1, 256);
    }
    std::swap(cur_x, alt_x);
    if (alt_x == &x_pre_) alt_x = &x_;

    if (diag_rows) {
      int dslot = -1;
      if (v_idx == 0) dslot = 1;
      else if (v_idx == 5) dslot = 2;
      else if (v_idx == 17) dslot = 3;
      else if (v_idx == 18) dslot = 4;
      else if (v_idx == 30) dslot = 5;
      else if (v_idx == 44) dslot = 6;
      if (dslot >= 0)
        enc_copy_buf(e, cur_x->view(size_t(T - 1) * H * 4, H * 4), diag_rows->view(size_t(dslot) * H * 4, H * 4), H * 4);
    }

    if (T >= 256 && (v_idx & 7) == 7 && v_idx + 1 < cfg_.schedule.size()) e.split();
  }

  if (!dump_logits && !need_all_logits && !capture_taps) return;

  // Final AttnRes + RMSNorm
  {
    std::vector<const Buffer*> st = blocks_vec;
    st.push_back(cur_x);
    enc_attn_res(e, st.data(), int(st.size()), &output_query_->buf, output_query_->dtype == DType::F32, 1, ar_out_, T);
  }
  enc_norm(e, ar_out_, final_norm_, h_norm_, T, H);
  if (diag_rows) enc_copy_buf(e, h_norm_.view(size_t(T - 1) * H * 4, H * 4), diag_rows->view(7 * H * 4, H * 4), H * 4);
  if (!dump_logits && !need_all_logits) return;

  const Buffer& dst = out_logits ? *out_logits : logits_;
  const int rows_out = need_all_logits ? T : 1;
  const Buffer h_in = need_all_logits ? h_norm_ : h_norm_.view(size_t(T - 1) * H * 4, size_t(H) * 4);
  enc_unembed(e, h_in, dst, rows_out, true);
}

void Inkling2::enc_sample(Encoder& e, Seq& s, const Buffer& lrow, const Buffer& tokens, int out_index,
                          const SamplingParams& sp, uint32_t step) {
  enc_logit_mask(e, s, lrow, 1, cfg_.vocab, cfg_.vocab);
  if (full_vocab(sp)) {
    enc_sample_full(e, lrow, uint32_t(cfg_.vocab), tokens, out_index, sprob_, sp, step);
    return;
  }
  const uint32_t K = topk_k(sp);
  const kp::TopkStage ts{uint32_t(cfg_.vocab), K, topk_chunk(K)};
  const uint32_t ntg = topk_lists(uint32_t(cfg_.vocab), K);
  e.use(base_pipe("topk_stage1"));
  e.buf(0, lrow);
  e.buf(1, cand_v_);
  e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, 1, 1, topk_threads(K));
  const kp::Sample p2{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, uint32_t(out_index)};
  e.use(base_pipe("sample_stage2"));
  e.buf(0, cand_v_);
  e.buf(1, cand_i_);
  e.buf(2, tokens);
  e.buf(3, sprob_);
  e.val(4, p2);
  e.dispatch(1, 1, 1, 1024);
}

void Inkling2::encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample) {
  if (pos + 1 >= s.cap) fatal("context full (%d)", s.cap);
  enc_forward(e, s, pos, 1, sample, nullptr, false, nullptr, mtp_ != nullptr, false);
  if (mtp_) mtp_->encode_commit_context(e, s, pos, 1, tap_[0], tap_[1], tap_[2], h_norm_);
  if (!sample) return;
  enc_logit_mask(e, s, logits_, 1, cfg_.vocab, cfg_.vocab);
  enc_sample(e, s, logits_, s.tok, pos + 1, sp, step);
}

void Inkling2::encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                              Checkpoint* snap, int) {
  if (T > tmax_) fatal("prefill chunk %d > %d", T, tmax_);
  if (pos0 + T >= s.cap) fatal("context full (%d)", s.cap);
  if (snap) fatal("inkling2: in-pass snapshots are not supported");
  enc_forward(e, s, pos0, T, sample, nullptr, false, nullptr, mtp_ != nullptr, false);
  if (mtp_) mtp_->encode_commit_context(e, s, pos0, T, tap_[0], tap_[1], tap_[2], h_norm_);
  if (!sample) return;
  enc_logit_mask(e, s, logits_, 1, cfg_.vocab, cfg_.vocab);
  enc_sample(e, s, logits_, s.tok, pos0 + T, sp, step);
}

void Inkling2::encode_mtp_draft(Encoder& e, Seq& s, int P, int K, const SamplingParams& sp, uint32_t step) {
  if (!mtp_) fatal("inkling2: MTP drafter not loaded");
  mtp_->encode_draft(e, s, P, K, sp, step);
}

void Inkling2::encode_mtp_verify(Encoder& e, Seq& s, int P, int R, const SamplingParams& sp, uint32_t step,
                                 bool point_mass) {
  if (!mtp_) fatal("inkling2: MTP drafter not loaded");
  const int max_r = point_mass ? Ink2Mtp::kMaxVerifyRows : mtp_->block();
  if (R < 1 || R > max_r) fatal("inkling2: verify rows %d out of range", R);
  if (P + R >= s.cap) fatal("context full (%d)", s.cap);
  enc_forward(e, s, P, R, true, &verify_logits_, true, nullptr, true, true);
  enc_logit_mask(e, s, verify_logits_, R, cfg_.vocab, cfg_.vocab);
  const uint32_t V = uint32_t(cfg_.vocab);
  if (sp.temperature <= 0.0f) {
    last_gpu_accept_ = false;
    const struct {
      uint32_t V;
    } ap{V};
    e.use(ink2_pipe("ink2_mtp_argmax_stage1"));
    e.buf(0, verify_logits_);
    e.buf(1, cand_v_);
    e.buf(2, cand_i_);
    e.val(3, ap);
    e.dispatch(64, uint32_t(R), 1, 256);

    e.use(ink2_pipe("ink2_mtp_argmax_stage2"));
    e.buf(0, cand_v_);
    e.buf(1, cand_i_);
    e.buf(2, verify_samples_);
    e.dispatch(uint32_t(R), 1, 1, 64);
  } else if (Ink2Mtp::can_gpu_accept(sp)) {
    last_gpu_accept_ = true;
    mtp_->encode_accept_gpu(e, s, P, R, verify_logits_, verify_accept_, sp, step, point_mass);
  } else {
    last_gpu_accept_ = false;
    enc_sample(e, s, verify_logits_.view(size_t(R - 1) * V * 4, size_t(V) * 4), verify_samples_, R - 1, sp,
               step + uint32_t(R - 1));
  }
}

int Inkling2::accept_mtp(Seq& s, int P, int K, const SamplingParams& sp, uint32_t step, bool point_mass) const {
  if (!mtp_) fatal("inkling2: MTP drafter not loaded");
  return mtp_->accept(s, P, K, verify_logits_, verify_samples_, verify_accept_, sp, step, point_mass,
                      last_gpu_accept_);
}

void Inkling2::encode_mtp_commit(Encoder& e, Seq& s, int pos0, int m) {
  if (!mtp_ || m <= 0) return;
  const size_t cbytes = conv_bytes_per_seq(cfg_);
  enc_copy_buf(e, verify_conv_.view(size_t(m - 1) * cbytes, cbytes), s.conv[0], cbytes);
  mtp_->encode_commit_context(e, s, pos0, m, tap_[0], tap_[1], tap_[2], h_norm_);
}

bool Inkling2::can_tree_spec() const {
  return mtp_ && mtp_->can_tree() && cfg_.hq == cfg_.hkv * 3 && cfg_.head_dim == 128;
}

void Inkling2::encode_mtp_draft_tree(Encoder& e, Seq& s, int P, int R, const SamplingParams& sp) {
  if (!can_tree_spec()) fatal("inkling2: beam tree speculation unavailable");
  mtp_->encode_draft_tree(e, s, P, R, sp);
}

void Inkling2::encode_mtp_verify_tree(Encoder& e, Seq& s, int P, int R, const SamplingParams& sp, uint32_t step) {
  if (!can_tree_spec()) fatal("inkling2: beam tree speculation unavailable");
  if (R < 1 || R > Ink2Mtp::kMaxVerifyRows) fatal("inkling2: tree verify rows %d out of range", R);
  if (P + R >= s.cap) fatal("context full (%d)", s.cap);
  enc_forward(e, s, P, R, true, &verify_logits_, true, nullptr, true, true, &mtp_->tree_buf());
  enc_logit_mask(e, s, verify_logits_, R, cfg_.vocab, cfg_.vocab);
  const uint32_t V = uint32_t(cfg_.vocab);
  if (sp.temperature <= 0.0f) {
    const struct {
      uint32_t V;
    } ap{V};
    e.use(ink2_pipe("ink2_mtp_argmax_stage1"));
    e.buf(0, verify_logits_);
    e.buf(1, cand_v_);
    e.buf(2, cand_i_);
    e.val(3, ap);
    e.dispatch(64, uint32_t(R), 1, 256);

    e.use(ink2_pipe("ink2_mtp_argmax_stage2"));
    e.buf(0, cand_v_);
    e.buf(1, cand_i_);
    e.buf(2, verify_samples_);
    e.dispatch(uint32_t(R), 1, 1, 64);

    last_gpu_accept_ = true;
    mtp_->encode_accept_tree_gpu(e, s, P, R, verify_logits_, verify_samples_, verify_tree_accept_, sp, step);
  } else if (Ink2Mtp::can_gpu_accept(sp)) {
    last_gpu_accept_ = true;
    mtp_->encode_accept_tree_gpu(e, s, P, R, verify_logits_, verify_samples_, verify_tree_accept_, sp, step);
  } else {
    last_gpu_accept_ = false;
    enc_sample(e, s, verify_logits_.view(0, size_t(V) * 4), verify_samples_, 0, sp, step);
  }
}

int Inkling2::accept_mtp_tree(Seq& s, int P, const SamplingParams& sp, uint32_t step, int* out_path) const {
  if (!can_tree_spec()) fatal("inkling2: beam tree speculation unavailable");
  const Ink2TreeAccept acc = mtp_->accept_tree(s, P, verify_logits_, verify_samples_, verify_tree_accept_, sp, step,
                                               last_gpu_accept_);
  if (out_path) {
    for (int i = 0; i < acc.m; ++i) out_path[i] = acc.path[i];
  }
  return acc.m;
}

void Inkling2::encode_mtp_commit_tree(Encoder& e, Seq& s, int pos0, const int* path, int m) {
  if (!mtp_ || m <= 0) return;
  bool is_identity = true;
  for (int i = 0; i < m; ++i) {
    if (path[i] != i) {
      is_identity = false;
      break;
    }
  }
  const size_t cbytes = conv_bytes_per_seq(cfg_);
  if (is_identity) {
    enc_copy_buf(e, verify_conv_.view(size_t(m - 1) * cbytes, cbytes), s.conv[0], cbytes);
    mtp_->encode_commit_context(e, s, pos0, m, tap_[0], tap_[1], tap_[2], h_norm_);
    return;
  }
  Ink2Grouped gp(e, true);
  e.group_begin();
  enc_copy_buf(e, verify_conv_.view(size_t(path[m - 1]) * cbytes, cbytes), s.conv[0], cbytes);
  const uint32_t kvd = uint32_t(cfg_.hkv) * uint32_t(cfg_.head_dim);
  const uint32_t kv_fp8_flag = use_kv_fp8_ ? 1u : 0u;
  if (m > 1) {
    Ink2KvCompactP kcp{uint32_t(pos0), 0u, 0u, kv_fp8_flag, kvd, uint32_t(m), {}};
    for (int i = 0; i < m; ++i) kcp.path[i] = uint32_t(path[i]);
    const Pipeline cpl = ink2_pipe("ink2_kv_compact");
    const size_t nv = cfg_.schedule.size();
    for (size_t v_idx = 0; v_idx < nv; ++v_idx) {
      const bool swa = is_swa_slot(v_idx);
      kcp.is_swa = swa ? 1u : 0u;
      kcp.kv_cap = swa ? uint32_t(ring_rows_) : uint32_t(s.cap);
      e.use(cpl);
      e.buf(0, s.kc[v_idx]);
      e.buf(1, s.vc[v_idx]);
      e.val(2, kcp);
      e.dispatch(1, 1, 1, kvd >> 2u);
    }
  }
  const uint32_t H = uint32_t(cfg_.hidden);
  Ink2GatherP gtp{uint32_t(m), H, {}};
  for (int i = 0; i < m; ++i) gtp.path[i] = uint32_t(path[i]);
  const Pipeline gpl = ink2_pipe("ink2_gather_rows");
  const Buffer* srcs[4] = {&tap_[0], &tap_[1], &tap_[2], &h_norm_};
  const Buffer* dsts[4] = {&n0_, &n1_, &n2_, &x_};
  for (int ti = 0; ti < 4; ++ti) {
    e.use(gpl);
    e.buf(0, *srcs[ti]);
    e.buf(1, *dsts[ti]);
    e.val(2, gtp);
    e.dispatch(((H >> 2u) + 255u) / 256u, uint32_t(m), 1, 256);
  }
  e.group_end();
  mtp_->encode_commit_context(e, s, pos0, m, n0_, n1_, n2_, x_);
}

void Inkling2::encode_logits_rows(Encoder& e, Seq& s, int pos0, int T, const Buffer& out) {
  if (T > tmax_) fatal("logits_rows chunk %d > %d", T, tmax_);
  if (pos0 + T > s.cap) fatal("context full (%d)", s.cap);
  enc_forward(e, s, pos0, T, true, &out, true);
}

void Inkling2::encode_prefill_diag(Encoder& e, Seq& s, int pos0, int T, const Buffer& diag_rows) {
  if (T > tmax_) fatal("prefill chunk %d > %d", T, tmax_);
  if (pos0 + T >= s.cap) fatal("context full (%d)", s.cap);
  enc_forward(e, s, pos0, T, true, nullptr, false, &diag_rows);
}

int Inkling2::debug_tensor_unit_check() {
  if (!tensor_pf_) return 0;
  Gpu& g = Gpu::get();
  const uint32_t T = 96, H = uint32_t(cfg_.hidden);
  int fails = 0;
  auto cmp = [&](const char* label, const float* got, const float* ref, size_t n, float tol_rel) {
    double max_abs = 0.0, sum_sq = 0.0, ref_sq = 0.0;
    size_t max_i = 0;
    for (size_t i = 0; i < n; ++i) {
      const double d = double(got[i]) - double(ref[i]);
      if (std::abs(d) > max_abs) { max_abs = std::abs(d); max_i = i; }
      sum_sq += d * d;
      ref_sq += double(ref[i]) * double(ref[i]);
    }
    const double rmse = std::sqrt(sum_sq / double(n)), rel = std::sqrt(sum_sq / std::max(1e-12, ref_sq));
    const bool ok = std::isfinite(rel) && rel <= tol_rel;
    printf("  UNIT %-20s max_diff=%.6f (at %zu: %.5f vs ref %.5f) rmse=%.6f rel=%.6f -> %s\n",
           label, max_abs, max_i, got[max_i], ref[max_i], rmse, rel, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  };
  // 1. BF16 GEMM (L0.wq_du)
  {
    const Tensor& w = *layers_[0].wq_du;
    const uint32_t N = uint32_t(w.shape[0]), K = uint32_t(w.shape[1]);
    for (uint32_t i = 0; i < T * K; ++i) x_mlp_.as<float>()[i] = 0.25f * std::sin(float(i) * 0.013f);
    std::vector<float> ref(size_t(T) * N), ten(size_t(T) * N);
    const Ink2MmP mp{T, N, K, K, N, w.dtype == DType::F32 ? 1u : 0u, 0u};
    const Ink2tGemmP gp{T, N, K, K, N, 0u, 0u, 0u, 0u};
    for (int arm = 0; arm < 2; ++arm) {
      auto e = g.begin();
      if (arm) {
        e->use(base_pipe("ink2_t_bf16_m64"));
        e->buf(0, x_mlp_);
        e->buf(1, w.buf);
        e->buf(2, w.buf);
        e->buf(3, q_);
        e->val(4, gp);
        e->buf(5, moe_tiles_);
        e->buf(6, moe_ntiles_);
        e->buf(7, moe_sorted_);
        e->buf(8, x_mlp_);
        e->dispatch((N + 63) / 64, (T + 63) / 64, 1, 128);
      } else {
        e->use(ink2_pipe("ink2_bf16_mm"));
        e->buf(0, x_mlp_);
        e->buf(1, w.buf);
        e->buf(2, q_);
        e->val(3, mp);
        e->dispatch((N + 63) / 64, (T + 31) / 32, 1, 128);
      }
      e->commit_and_wait();
      (arm ? ten : ref).assign(q_.as<float>(), q_.as<float>() + size_t(T) * N);
    }
    fails += cmp("bf16_mm_t", ten.data(), ref.data(), size_t(T) * N, 0.05f);
  }
  // 2 & 3. MoE NVFP4 (L0) and MXFP8 (L20)
  for (uint32_t i = 0; i < T * H; ++i) x_mlp_.as<float>()[i] = 0.25f * std::sin(float(i) * 0.017f);
  {
    const Ink2Layer& L0 = layers_[0];
    auto e = g.begin();
    e->use(ink2_pipe("ink2_bf16_mm"));
    e->buf(0, x_mlp_); e->buf(1, L0.router_w->buf); e->buf(2, rlog_);
    e->val(3, Ink2MmP{T, uint32_t(cfg_.n_experts), H, H, uint32_t(cfg_.n_experts),
                      L0.router_w->dtype == DType::F32 ? 1u : 0u, 0u});
    e->dispatch((uint32_t(cfg_.n_experts) + 63) / 64, (T + 31) / 32, 1, 128);
    e->use(ink2_pipe("ink2_moe_route"));
    e->buf(0, rlog_); e->buf(1, L0.router_bias->buf); e->buf(2, eid_); e->buf(3, ew_);
    e->val(4, Ink2RouteP{T, uint32_t(cfg_.n_experts), uint32_t(cfg_.top_k),
                         L0.router_bias->dtype == DType::F32 ? 1u : 0u, cfg_.route_scale});
    e->dispatch(T, 1, 1, 32);
    e->commit_and_wait();
  }
  auto check_moe = [&](const char* label, const Ink2Layer& L) {
    const uint32_t I = uint32_t(cfg_.moe_I), K_top = uint32_t(cfg_.top_k), E = uint32_t(cfg_.n_experts);
    const Ink2MoeP mp{T, H, I, K_top};
    std::vector<float> ref(T * H), ten(T * H);
    {
      auto e = g.begin();
      int b = 0;
      e->use(ink2_pipe(L.is_mxfp8 ? "ink2_moe_mxfp8_w13_situ" : "ink2_moe_nvfp4_w13_situ"));
      e->buf(b++, x_mlp_); e->buf(b++, eid_); e->buf(b++, L.w13->buf); e->buf(b++, L.w13_sc->buf);
      if (!L.is_mxfp8) e->buf(b++, L.w13_outer);
      e->buf(b++, moe_act_); e->val(b, mp);
      e->dispatch((I + 15) / 16, K_top, T, 128);
      b = 0;
      e->use(ink2_pipe(L.is_mxfp8 ? "ink2_moe_mxfp8_w2_acc" : "ink2_moe_nvfp4_w2_acc"));
      e->buf(b++, moe_act_); e->buf(b++, eid_); e->buf(b++, ew_); e->buf(b++, L.w2->buf); e->buf(b++, L.w2_sc->buf);
      if (!L.is_mxfp8) e->buf(b++, L.w2_outer);
      e->buf(b++, moe_out_); e->val(b, mp);
      e->dispatch((H + (L.is_mxfp8 ? 15u : 7u)) / (L.is_mxfp8 ? 16u : 8u), T, 1, 128);
      e->commit_and_wait();
      ref.assign(moe_out_.as<float>(), moe_out_.as<float>() + T * H);
    }
    {
      const uint32_t P = T * K_top;
      const uint32_t per = P / E;
      const uint32_t bm = per >= 40 ? 64u : per >= 12 ? 32u : 16u;
      const std::string ms = "_m" + std::to_string(bm);
      const uint32_t max_tiles = P / bm + E + 1;
      auto e = g.begin();
      const Ink2BucketP bp{P, E, bm};
      e->use(ink2_pipe("ink2_moe_bucket"));
      e->buf(0, eid_); e->buf(1, moe_sorted_); e->buf(2, moe_tiles_); e->buf(3, moe_ntiles_);
      e->buf(4, moe_rank_); e->val(5, bp);
      e->dispatch(1, 1, 1, 256);

      const uint32_t w13_q_stride = L.is_mxfp8 ? (2 * I * H) : (2 * I * (H / 2));
      const uint32_t w13_s_stride = L.is_mxfp8 ? (2 * I * (H / 32)) : (2 * I * (H / 16));
      const Ink2tGemmP gp13{P, 2 * I, H, H, 2 * I, 0u, w13_q_stride, w13_s_stride, K_top};
      e->use(base_pipe((L.is_mxfp8 ? "ink2_t_mxfp8_moe_w13" : "ink2_t_nvfp4_moe_w13") + ms));
      e->buf(0, x_mlp_); e->buf(1, L.w13->buf); e->buf(2, L.w13_sc->buf); e->buf(3, moe_tmp_);
      e->val(4, gp13); e->buf(5, moe_tiles_); e->buf(6, moe_ntiles_); e->buf(7, moe_sorted_);
      e->buf(8, L.is_mxfp8 ? x_mlp_ : L.w13_outer);
      e->dispatch((2 * I + 63) / 64, max_tiles, 1, 128);

      const Ink2SituP sp{P * I};
      e->use(ink2_pipe("ink2_moe_situ_half"));
      e->buf(0, moe_tmp_); e->buf(1, moe_act_h_); e->val(2, sp);
      e->dispatch((sp.n + 255) / 256, 1, 1, 256);

      const uint32_t w2_q_stride = L.is_mxfp8 ? (H * I) : (H * (I / 2));
      const uint32_t w2_s_stride = L.is_mxfp8 ? (H * (I / 32)) : (H * (I / 16));
      const Ink2tGemmP gp2{P, H, I, I, H, 0u, w2_q_stride, w2_s_stride, K_top};
      e->use(base_pipe((L.is_mxfp8 ? "ink2_t_mxfp8_moe_w2" : "ink2_t_nvfp4_moe_w2") + ms));
      e->buf(0, moe_act_h_); e->buf(1, L.w2->buf); e->buf(2, L.w2_sc->buf); e->buf(3, moe_tmp_);
      e->val(4, gp2); e->buf(5, moe_tiles_); e->buf(6, moe_ntiles_); e->buf(7, moe_sorted_);
      e->buf(8, L.is_mxfp8 ? x_mlp_ : L.w2_outer);
      e->dispatch((H + 63) / 64, max_tiles, 1, 128);

      const Ink2CombineP cp{T, H, K_top};
      e->use(ink2_pipe("ink2_moe_combine"));
      e->buf(0, moe_tmp_); e->buf(1, ew_); e->buf(2, moe_out_); e->val(3, cp);
      e->dispatch(T, 1, 1, 256);
      e->commit_and_wait();
      ten.assign(moe_out_.as<float>(), moe_out_.as<float>() + T * H);
    }
    return cmp(label, ten.data(), ref.data(), T * H, 0.2f);
  };
  fails += check_moe("moe_nvfp4_t", layers_[0]);
  fails += check_moe("moe_mxfp8_t", layers_[20]);
  // 4. Attention Prefill (T=96, SWA)
  {
    const uint32_t QD = uint32_t(cfg_.hq) * uint32_t(cfg_.head_dim), KVD = uint32_t(cfg_.hkv) * uint32_t(cfg_.head_dim);
    Buffer k_fp16 = g.alloc(size_t(ring_rows_) * KVD * 2, "ink2_chk_k");
    Buffer v_fp16 = g.alloc(size_t(ring_rows_) * KVD * 2, "ink2_chk_v");
    for (uint32_t i = 0; i < T * QD; ++i) {
      q_.as<float>()[i] = 0.2f * std::sin(float(i) * 0.017f);
      gate_attn_.as<float>()[i] = 0.5f * std::cos(float(i) * 0.011f);
    }
    for (uint32_t i = 0; i < T * uint32_t(cfg_.hq); ++i) sinks_.as<float>()[i] = 0.1f * std::sin(float(i) * 0.1f);
    for (uint32_t i = 0; i < T * uint32_t(cfg_.hq) * uint32_t(cfg_.rel_extent); ++i)
      rb_.as<float>()[i] = 0.05f * std::sin(float(i) * 0.007f);
    for (uint32_t i = 0; i < T * KVD; ++i) {
      __fp16 kv = __fp16(0.2f * std::cos(float(i) * 0.019f)), vv = __fp16(0.2f * std::sin(float(i) * 0.023f));
      std::memcpy(k_fp16.as<uint16_t>() + i, &kv, 2);
      std::memcpy(v_fp16.as<uint16_t>() + i, &vv, 2);
    }
    const Ink2AttnPfP apf{T, 0u, uint32_t(ring_rows_), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim),
                          uint32_t(cfg_.rel_extent), 1u, uint32_t(cfg_.swa), 1.0f / std::sqrt(float(cfg_.head_dim)), 0u};
    std::vector<float> ref(T * QD), ten(T * QD);
    for (int arm = 0; arm < 2; ++arm) {
      auto e = g.begin();
      e->use(arm ? base_pipe("qattn_ink2_s6k32p") : ink2_pipe("ink2_attn_prefill"));
      e->buf(0, q_); e->buf(1, k_fp16); e->buf(2, v_fp16); e->buf(3, rb_);
      e->buf(4, sinks_); e->buf(5, gate_attn_); e->buf(6, att_out_); e->val(7, apf);
      e->dispatch(arm ? (T + 31) / 32 : uint32_t(cfg_.hq), arm ? uint32_t(cfg_.hkv) : T, 1, arm ? 192 : 128);
      e->commit_and_wait();
      (arm ? ten : ref).assign(att_out_.as<float>(), att_out_.as<float>() + T * QD);
    }
    g.release(k_fp16);
    g.release(v_fp16);
    fails += cmp("attn_prefill_t", ten.data(), ref.data(), T * QD, 0.2f);
  }
  // 5. Batched Cross-Pass MessageGate
  {
    const Ink2Ctrl& c = layers_[size_t(cfg_.core_start)].ctrl_attn;
    for (uint32_t i = 0; i < T * H; ++i) {
      x_mlp_.as<float>()[i] = 0.25f * std::sin(float(i) * 0.013f);
      h_mlp_.as<float>()[i] = 0.25f * std::cos(float(i) * 0.017f);
    }
    const Ink2MsgGateP mgp{T, H, uint32_t(cfg_.gate_rank), c.down->dtype == DType::F32 ? 1u : 0u, cfg_.rms_eps};
    std::vector<float> ref(T * H), ten(T * H);
    {
      auto e = g.begin();
      e->use(ink2_pipe("ink2_msg_gate"));
      e->buf(0, x_mlp_); e->buf(1, h_mlp_); e->buf(2, c.down->buf); e->buf(3, c.forget->buf);
      e->buf(4, c.gain->buf); e->buf(5, n0_); e->val(6, mgp); e->dispatch(T, 1, 1, 512);
      e->commit_and_wait();
      ref.assign(n0_.as<float>(), n0_.as<float>() + T * H);
    }
    {
      auto e = g.begin();
      enc_msg_gate(*e, c, x_mlp_, h_mlp_, n1_, int(T));
      e->commit_and_wait();
      ten.assign(n1_.as<float>(), n1_.as<float>() + T * H);
    }
    fails += cmp("msg_gate_t", ten.data(), ref.data(), T * H, 0.05f);
  }
  return fails;
}

}  // namespace ni
