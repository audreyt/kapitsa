#include "common.h"

// Inkling-2-Mini (thinkingmachines/Inkling-2-Mini, model_type inkling2_loop) Metal kernels.
// Decode, router GEMMs (exact_f32), norms, attention, and MoE kernels run in FP32 matching the PyTorch
// golden reference; non-router prefill linear projections with M > 32 use ink2_bf16_mm (FP16 simdgroup
// inputs with FP32 accumulation):
//  - AttRes over up to 6 states and 4 queries
//  - Low-rank MessageGate controllers (rank 64, tanh gates)
//  - Static (divisor sqrt(5)) and dynamic (divisor 3) causal short convolutions
//  - Q/K L2 clipping, unweighted V RMSNorm, global tau log-scaling, sink-logit attention with rel bias
//  - 128-expert top-8 sigmoid MoE with interleaved gate/up SiTU (4*tanh*sigmoid * 25*tanh):
//      * layers 0..19: NVFP4 (low/high nibble E2M1, F8_E4M3FN block-16 scales, outer scale amax/denom)
//      * layers 20..23: MXFP8 (F8_E4M3FN weights, UE8M0 block-32 scales)

NI_INLINE float4 ink2_bf16x4(uint2 r) {
  return float4(as_type<float>(r.x << 16), as_type<float>(r.x & 0xFFFF0000u),
                as_type<float>(r.y << 16), as_type<float>(r.y & 0xFFFF0000u));
}

constant float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

NI_INLINE float4 ink2_e2m1_4_raw(uint packed16) {
  const uint p0 = ((packed16 & 0x0008u) << 12) | ((packed16 & 0x0007u) << 9) |
                  ((packed16 & 0x0080u) << 24) | ((packed16 & 0x0070u) << 21);
  const uint p1 = ((packed16 & 0x0800u) << 4) | ((packed16 & 0x0700u) << 1) |
                  ((packed16 & 0x8000u) << 16) | ((packed16 & 0x7000u) << 13);
  return float4(float2(as_type<half2>(p0)), float2(as_type<half2>(p1)));
}

NI_INLINE float ink2_e4m3(uint b) {
  const ushort h = ushort(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
  return float(as_type<half>(h)) * 256.0f;
}

NI_INLINE float ink2_e4m3_16384(uint b) {
  const ushort h = ushort(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
  return float(as_type<half>(h)) * 4194304.0f;
}

NI_INLINE float4 ink2_e4m3_4(uint packed32) {
  return float4(ink2_e4m3(packed32 & 0xFFu), ink2_e4m3((packed32 >> 8) & 0xFFu),
                ink2_e4m3((packed32 >> 16) & 0xFFu), ink2_e4m3(packed32 >> 24));
}

NI_INLINE float ink2_ue8m0(uint b) {
  return b == 0u ? 5.8774718e-39f : as_type<float>(b << 23);
}

NI_INLINE uchar ink2_f32_to_e4m3(float x) {
  const float xc = clamp(isfinite(x) ? x : 0.0f, -448.0f, 448.0f);
  const uint hu = uint(as_type<ushort>(half(xc * (1.0f / 256.0f))));
  const uint sgn = (hu >> 8) & 0x80u;
  const uint mag = hu & 0x7FFFu;
  const uint r = (mag + 0x3Fu + ((mag >> 7) & 1u)) >> 7;
  return uchar(sgn | min(r, 0x7Eu));
}

NI_INLINE float4 ink2_load_kv4(const device void* buf, size_t off, uint kv_fp8) {
  if (kv_fp8) return ink2_e4m3_4(*(const device uint*)((const device uchar*)buf + off));
  return float4(*(const device half4*)((const device half*)buf + off));
}

NI_INLINE float ink2_nvfp4_dot(const device uchar* w, size_t wrow, uint half_k, uint blk, const thread float* xf) {
  const uint2 pw = *(const device uint2*)(w + wrow * half_k + blk * 8);
  const thread uchar* bw = (const thread uchar*)&pw;
  float d = 0.0f;
  for (uint b = 0; b < 8; ++b) d += kE2M1[bw[b] & 15u] * xf[2 * b] + kE2M1[bw[b] >> 4] * xf[2 * b + 1];
  return d;
}

// ---- 1. Embedding lookup + RMSNorm (BF16) ----
struct Ink2EmbedP {
  uint T, H, tok_off, V;
  float eps;
};

[[kernel]] void ink2_embed_norm(const device int* tokens [[buffer(0)]], const device bfloat* embed [[buffer(1)]],
                                const device bfloat* norm_w [[buffer(2)]], device float* out [[buffer(3)]],
                                constant Ink2EmbedP& p [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const int raw_id = tokens[p.tok_off + t];
  const uint id = (raw_id >= 0 && uint(raw_id) < p.V) ? uint(raw_id) : 0u;
  const device bfloat* row = embed + size_t(id) * p.H;
  float xs[6];
  float ss = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float v = float(row[c]);
    xs[k] = v;
    ss += v * v;
  }
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
  device float* dst = out + size_t(t) * p.H;
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) dst[c] = xs[k] * inv * float(norm_w[c]);
}

// ---- 2. Generic FP32 RMSNorm (optional BF16/F32 weight, alias-safe) ----
struct Ink2NormP {
  uint T, H, has_w, w_f32;
  float eps;
};

[[kernel]] void ink2_rmsnorm(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                             device float* out [[buffer(2)]], constant Ink2NormP& p [[buffer(3)]],
                             uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* src = x + size_t(t) * p.H;
  device float* dst = out + size_t(t) * p.H;
  float xs[8];
  float ss = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float v = src[c];
    xs[k] = v;
    ss += v * v;
  }
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    float y = xs[k] * inv;
    if (p.has_w) y *= p.w_f32 ? ((const device float*)w)[c] : float(((const device bfloat*)w)[c]);
    dst[c] = y;
  }
}

// ---- 3. Block Attention Residual (M <= 6 states, Q <= 4 queries) ----
struct Ink2AttnResP {
  uint T, H, M, Q, q_f32;
  float eps, scale;
};

[[kernel]] void ink2_attn_res(const device float* s0 [[buffer(0)]], const device float* s1 [[buffer(1)]],
                              const device float* s2 [[buffer(2)]], const device float* s3 [[buffer(3)]],
                              const device float* s4 [[buffer(4)]], const device float* s5 [[buffer(5)]],
                              const device void* queries [[buffer(6)]], device float* out [[buffer(7)]],
                              constant Ink2AttnResP& p [[buffer(8)]], uint t [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  const device float* ptrs[6] = {s0, s1, s2, s3, s4, s5};
  threadgroup float scratch[32];
  threadgroup float tinv[6];
  threadgroup float tsc[6][4];
  threadgroup float tw[6][4];

  float qv[4][5];
  for (uint q = 0; q < p.Q; ++q) {
    uint k = 0;
    for (uint c = tid; c < p.H; c += ntg, ++k) {
      const uint idx = q * p.H + c;
      qv[q][k] = p.q_f32 ? ((const device float*)queries)[idx] : float(((const device bfloat*)queries)[idx]);
    }
  }

  for (uint m = 0; m < p.M; ++m) {
    const device float* sr = ptrs[m] + size_t(t) * p.H;
    float sv[5];
    float ss = 0.0f;
    uint k = 0;
    for (uint c = tid; c < p.H; c += ntg, ++k) {
      const float v = sr[c];
      sv[k] = v;
      ss += v * v;
    }
    ss = tg_sum(ss, scratch, tid, ntg);
    const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
    if (tid == 0) tinv[m] = inv;
    for (uint q = 0; q < p.Q; ++q) {
      float dot = 0.0f;
      for (uint kk = 0; kk < k; ++kk) dot += (sv[kk] * inv) * qv[q][kk];
      dot = tg_sum(dot, scratch, tid, ntg);
      if (tid == 0) tsc[m][q] = dot * p.scale;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < p.Q) {
    float mx = -INFINITY;
    for (uint m = 0; m < p.M; ++m) mx = max(mx, tsc[m][tid]);
    float exps[6];
    float sum = 0.0f;
    for (uint m = 0; m < p.M; ++m) {
      exps[m] = metal::precise::exp(tsc[m][tid] - mx);
      sum += exps[m];
    }
    const float isum = 1.0f / sum;
    for (uint m = 0; m < p.M; ++m) tw[m][tid] = exps[m] * isum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint c = tid; c < p.H; c += ntg) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint m = 0; m < p.M; ++m) {
      const float kv = ptrs[m][size_t(t) * p.H + c] * tinv[m];
      for (uint q = 0; q < p.Q; ++q) acc[q] += tw[m][q] * kv;
    }
    for (uint q = 0; q < p.Q; ++q) out[(size_t(q) * p.T + t) * p.H + c] = acc[q];
  }
}

// ---- 4. Cross-Pass Controller MessageGate (rank 64) ----
struct Ink2MsgGateP {
  uint T, H, R, w_f32;
  float eps;
};

[[kernel]] void ink2_msg_gate(const device float* h [[buffer(0)]], const device float* m [[buffer(1)]],
                              const device void* down_w [[buffer(2)]], const device void* forget_w [[buffer(3)]],
                              const device void* gain_w [[buffer(4)]], device float* out [[buffer(5)]],
                              constant Ink2MsgGateP& p [[buffer(6)]], uint t [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  // Single H-element shared buffer reused across phases (9,600 B total static tgmem, < 32 KB even under MTL_DEBUG_LAYER=1).
  threadgroup float scratch[32];
  threadgroup float th_buf[2304];
  threadgroup float th_g[64];

  const device float* hr = h + size_t(t) * p.H;
  const device float* mr = m + size_t(t) * p.H;
  float hs[5], ms[5];
  float ssh = 0.0f, ssm = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float hv = hr[c], mv = mr[c];
    hs[k] = hv;
    ms[k] = mv;
    ssh += hv * hv;
    ssm += mv * mv;
  }
  ssh = tg_sum(ssh, scratch, tid, ntg);
  ssm = tg_sum(ssm, scratch, tid, ntg);
  const float invh = metal::precise::rsqrt(ssh / float(p.H) + p.eps);
  const float invm = metal::precise::rsqrt(ssm / float(p.H) + p.eps);

  // Phase 1: norm(h) * down_w[:, 0:H] -> th_g
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) th_buf[c] = hs[k] * invh;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint nsg = (ntg + 31) / 32;
  for (uint r = sg; r < p.R; r += nsg) {
    float acc = 0.0f;
    const size_t row_off = size_t(r) * (2 * p.H);
    for (uint c = lane * 4; c < p.H; c += 128) {
      const float4 nh4 = *(const threadgroup float4*)(th_buf + c);
      float4 dw0;
      if (p.w_f32) {
        dw0 = *(const device float4*)((const device float*)down_w + row_off + c);
      } else {
        const device bfloat* b0 = (const device bfloat*)down_w + row_off + c;
        dw0 = float4(b0[0], b0[1], b0[2], b0[3]);
      }
      acc += dot(nh4, dw0);
    }
    acc = simd_sum(acc);
    if (lane == 0) th_g[r] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 2: norm(m) * down_w[:, H:2H] -> accumulate into th_g
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) th_buf[c] = ms[k] * invm;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint r = sg; r < p.R; r += nsg) {
    float acc = 0.0f;
    const size_t row_off = size_t(r) * (2 * p.H) + p.H;
    for (uint c = lane * 4; c < p.H; c += 128) {
      const float4 nm4 = *(const threadgroup float4*)(th_buf + c);
      float4 dw1;
      if (p.w_f32) {
        dw1 = *(const device float4*)((const device float*)down_w + row_off + c);
      } else {
        const device bfloat* b1 = (const device bfloat*)down_w + row_off + c;
        dw1 = float4(b1[0], b1[1], b1[2], b1[3]);
      }
      acc += dot(nm4, dw1);
    }
    acc = simd_sum(acc);
    if (lane == 0) th_g[r] += acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device float* dst = out + size_t(t) * p.H;
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    float df = 0.0f, dg = 0.0f;
    const size_t off = size_t(c) * p.R;
    for (uint r = 0; r < p.R; r += 4) {
      const float4 g4 = *(const threadgroup float4*)(th_g + r);
      float4 fw, gw;
      if (p.w_f32) {
        fw = *(const device float4*)((const device float*)forget_w + off + r);
        gw = *(const device float4*)((const device float*)gain_w + off + r);
      } else {
        const device bfloat* fb = (const device bfloat*)forget_w + off + r;
        const device bfloat* gb = (const device bfloat*)gain_w + off + r;
        fw = float4(fb[0], fb[1], fb[2], fb[3]);
        gw = float4(gb[0], gb[1], gb[2], gb[3]);
      }
      df += dot(g4, fw);
      dg += dot(g4, gw);
    }
    const float fg = metal::precise::tanh(df);
    const float gn = metal::precise::tanh(dg);
    dst[c] = (1.0f - fg) * hs[k] + gn * ms[k];
  }
}

[[kernel]] void ink2_msg_gate_prep_s(const device float* h [[buffer(0)]], const device float* m [[buffer(1)]],
                                     device float* cat [[buffer(2)]], constant Ink2MsgGateP& p [[buffer(3)]],
                                     uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                     uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* hr = h + size_t(t) * p.H;
  const device float* mr = m + size_t(t) * p.H;
  float hs[5], ms[5];
  float ssh = 0.0f, ssm = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float hv = hr[c], mv = mr[c];
    hs[k] = hv; ms[k] = mv;
    ssh += hv * hv; ssm += mv * mv;
  }
  ssh = tg_sum(ssh, scratch, tid, ntg);
  ssm = tg_sum(ssm, scratch, tid, ntg);
  const float invh = metal::precise::rsqrt(ssh / float(p.H) + p.eps);
  const float invm = metal::precise::rsqrt(ssm / float(p.H) + p.eps);
  device float* d0 = cat + size_t(t) * (2u * p.H);
  device float* d1 = d0 + p.H;
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    d0[c] = hs[k] * invh;
    d1[c] = ms[k] * invm;
  }
}

[[kernel]] void ink2_msg_gate_down_s(const device float* cat [[buffer(0)]], const device void* down_w [[buffer(1)]],
                                     device float* lat [[buffer(2)]], constant Ink2MsgGateP& p [[buffer(3)]],
                                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.x * 4u + sg;
  if (r >= p.R) return;
  const uint t0 = tg.y * 4u, mc = min(4u, p.T - t0);
  const size_t row_off0 = size_t(r) * (2u * p.H), row_off1 = row_off0 + p.H;
  float a0[4] = {0, 0, 0, 0}, a1[4] = {0, 0, 0, 0};
  for (uint c = lane * 4u; c < p.H; c += 128u) {
    float4 dw0, dw1;
    if (p.w_f32) {
      dw0 = *(const device float4*)((const device float*)down_w + row_off0 + c);
      dw1 = *(const device float4*)((const device float*)down_w + row_off1 + c);
    } else {
      dw0 = ink2_bf16x4(*(const device uint2*)((const device bfloat*)down_w + row_off0 + c));
      dw1 = ink2_bf16x4(*(const device uint2*)((const device bfloat*)down_w + row_off1 + c));
    }
    for (uint m = 0u; m < mc; ++m) {
      const device float* row = cat + size_t(t0 + m) * (2u * p.H) + c;
      a0[m] += dot(*(const device float4*)(row), dw0);
      a1[m] += dot(*(const device float4*)(row + p.H), dw1);
    }
  }
  for (uint m = 0u; m < mc; ++m) {
    const float s0 = simd_sum(a0[m]), s1 = simd_sum(a1[m]);
    if (lane == 0u) lat[size_t(t0 + m) * p.R + r] = s0 + s1;
  }
}

[[kernel]] void ink2_msg_gate_up_mix_s(const device float* h [[buffer(0)]], const device float* m_buf [[buffer(1)]],
                                       const device float* lat [[buffer(2)]], const device void* forget_w [[buffer(3)]],
                                       const device void* gain_w [[buffer(4)]], device float* out [[buffer(5)]],
                                       constant Ink2MsgGateP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sh_lat[8 * 64];
  const uint t0 = tg.y * 8u, mc = min(8u, p.T - t0);
  for (uint idx = tid; idx < mc * p.R; idx += 128u) sh_lat[idx] = lat[size_t(t0) * p.R + idx];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = tg.x * 128u + tid;
  if (c >= p.H) return;
  float df[8] = {0, 0, 0, 0, 0, 0, 0, 0}, dg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const size_t off = size_t(c) * p.R;
  for (uint r = 0u; r < p.R; r += 4u) {
    float4 fw, gw;
    if (p.w_f32) {
      fw = *(const device float4*)((const device float*)forget_w + off + r);
      gw = *(const device float4*)((const device float*)gain_w + off + r);
    } else {
      fw = ink2_bf16x4(*(const device uint2*)((const device bfloat*)forget_w + off + r));
      gw = ink2_bf16x4(*(const device uint2*)((const device bfloat*)gain_w + off + r));
    }
    for (uint m = 0u; m < mc; ++m) {
      const float4 g4 = *(const threadgroup float4*)(sh_lat + m * p.R + r);
      df[m] += dot(g4, fw);
      dg[m] += dot(g4, gw);
    }
  }
  for (uint m = 0u; m < mc; ++m) {
    const size_t idx = size_t(t0 + m) * p.H + c;
    const float fg = metal::precise::tanh(df[m]), gn = metal::precise::tanh(dg[m]);
    out[idx] = (1.0f - fg) * h[idx] + gn * m_buf[idx];
  }
}

// ---- 5. Linear Projections: BF16/F32 MatVec (M <= 8) and Tiled MMA GEMM (M > 8) ----
struct Ink2MsgNormP {
  uint T, H, R, w_f32, nw_f32;
  float eps;
};

[[kernel]] void ink2_msg_gate_up_mix_norm_s(const device float* h [[buffer(0)]], const device float* m_buf [[buffer(1)]],
                                            const device float* lat [[buffer(2)]], const device void* forget_w [[buffer(3)]],
                                            const device void* gain_w [[buffer(4)]], const device void* norm_w [[buffer(5)]],
                                            device float* out [[buffer(6)]], constant Ink2MsgNormP& p [[buffer(7)]],
                                            uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                            uint ntg [[threads_per_threadgroup]]) {
  threadgroup float sh_lat[64];
  threadgroup float scratch[32];
  if (t >= p.T) return;
  if (tid < p.R) sh_lat[tid] = lat[size_t(t) * p.R + tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device float* hr = h + size_t(t) * p.H;
  const device float* mr = m_buf + size_t(t) * p.H;
  float vals[5];
  float ss = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    float df = 0.0f, dg = 0.0f;
    const size_t off = size_t(c) * p.R;
    for (uint r = 0u; r < p.R; r += 4u) {
      const float4 g4 = *(const threadgroup float4*)(sh_lat + r);
      float4 fw, gw;
      if (p.w_f32) {
        fw = *(const device float4*)((const device float*)forget_w + off + r);
        gw = *(const device float4*)((const device float*)gain_w + off + r);
      } else {
        fw = ink2_bf16x4(*(const device uint2*)((const device bfloat*)forget_w + off + r));
        gw = ink2_bf16x4(*(const device uint2*)((const device bfloat*)gain_w + off + r));
      }
      df += dot(g4, fw);
      dg += dot(g4, gw);
    }
    const float fg = metal::precise::tanh(df);
    const float gn = metal::precise::tanh(dg);
    const float y = (1.0f - fg) * hr[c] + gn * mr[c];
    vals[k] = y;
    ss += y * y;
  }
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
  device float* dst = out + size_t(t) * p.H;
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float nw = p.nw_f32 ? ((const device float*)norm_w)[c] : float(((const device bfloat*)norm_w)[c]);
    dst[c] = vals[k] * inv * nw;
  }
}

struct Ink2MmP {
  uint M, N, K, x_ld, y_ld, w_f32, add;
};

[[kernel]] void ink2_bf16_mv(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                             device float* y [[buffer(2)]], constant Ink2MmP& p [[buffer(3)]],
                             uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg * 4 + sg;
  if (n >= p.N) return;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const size_t woff = size_t(n) * p.K;
  for (uint k = lane * 4; k < p.K; k += 128) {
    float4 wv;
    if (p.w_f32) {
      wv = *(const device float4*)((const device float*)w + woff + k);
    } else {
      const device bfloat* wb = (const device bfloat*)w + woff + k;
      wv = float4(wb[0], wb[1], wb[2], wb[3]);
    }
    for (uint m = 0; m < p.M; ++m) {
      const float4 xv = *(const device float4*)(x + size_t(m) * p.x_ld + k);
      acc[m] += dot(xv, wv);
    }
  }
  for (uint m = 0; m < p.M; ++m) {
    const float s = simd_sum(acc[m]);
    if (lane == 0) {
      device float* dst = y + size_t(m) * p.y_ld + n;
      *dst = p.add ? (*dst + s) : s;
    }
  }
}

[[kernel]] void ink2_bf16_mv4(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                              device float* y [[buffer(2)]], constant Ink2MmP& p [[buffer(3)]],
                              uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4u + sg) * 4u;
  if (n0 >= p.N) return;
  const uint mc = min(4u, p.N - n0);
  const size_t woff0 = size_t(n0) * p.K;
  if (p.M == 1u && !p.w_f32 && mc == 4u) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const device bfloat* wb = (const device bfloat*)w + woff0;
    for (uint k = lane * 4u; k < p.K; k += 128u) {
      const float4 xv = *(const device float4*)(x + k);
      a0 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 0u * size_t(p.K) + k)));
      a1 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 1u * size_t(p.K) + k)));
      a2 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 2u * size_t(p.K) + k)));
      a3 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 3u * size_t(p.K) + k)));
    }
    const float s0 = simd_sum(a0), s1 = simd_sum(a1), s2 = simd_sum(a2), s3 = simd_sum(a3);
    if (lane == 0u) {
      device float* dst = y + n0;
      if (p.add) { dst[0] += s0; dst[1] += s1; dst[2] += s2; dst[3] += s3; }
      else { dst[0] = s0; dst[1] = s1; dst[2] = s2; dst[3] = s3; }
    }
    return;
  }
  float acc[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  for (uint k = lane * 4u; k < p.K; k += 128u) {
    float4 wv[4];
    for (uint q = 0u; q < mc; ++q) {
      const size_t woff = woff0 + size_t(q) * p.K + k;
      wv[q] = p.w_f32 ? *(const device float4*)((const device float*)w + woff)
                      : ink2_bf16x4(*(const device uint2*)((const device bfloat*)w + woff));
    }
    for (uint m = 0u; m < p.M; ++m) {
      const float4 xv = *(const device float4*)(x + size_t(m) * p.x_ld + k);
      for (uint q = 0u; q < mc; ++q) acc[q][m] += dot(xv, wv[q]);
    }
  }
  for (uint m = 0u; m < p.M; ++m) {
    float sv[4];
    for (uint q = 0u; q < mc; ++q) sv[q] = simd_sum(acc[q][m]);
    if (lane == 0u) {
      device float* dst = y + size_t(m) * p.y_ld + n0;
      for (uint q = 0u; q < mc; ++q) dst[q] = p.add ? (dst[q] + sv[q]) : sv[q];
    }
  }
}

struct Ink2MultiMmP {
  uint M, K, N_tot, n_segs;
  uint offs[7];
  uint lens[6];
};

[[kernel]] void ink2_bf16_mv4_multi6(const device float* x0 [[buffer(0)]], const device float* x1 [[buffer(1)]],
                                     const device float* x2 [[buffer(2)]], const device bfloat* w [[buffer(3)]],
                                     device float* y0 [[buffer(4)]], device float* y1 [[buffer(5)]],
                                     device float* y2 [[buffer(6)]], device float* y3 [[buffer(7)]],
                                     device float* y4 [[buffer(8)]], device float* y5 [[buffer(9)]],
                                     constant Ink2MultiMmP& p [[buffer(10)]],
                                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4u + sg) * 4u;
  if (n0 >= p.N_tot) return;
  uint seg = 0u;
  while (seg + 1u < p.n_segs && n0 >= p.offs[seg + 1u]) ++seg;
  if (seg >= p.n_segs) return;
  const uint seg_off = p.offs[seg];
  const uint seg_len = p.lens[seg];
  const uint local_n = n0 - seg_off;
  if (local_n >= seg_len) return;
  const uint mc = min(4u, seg_len - local_n);
  const device float* x = (seg < 4u) ? x0 : (seg == 4u ? x1 : x2);
  const size_t woff0 = size_t(n0) * p.K;
  device float* bases[6] = {y0, y1, y2, y3, y4, y5};
  device float* dst_base = bases[seg] + local_n;
  if (p.M == 1u && mc == 4u) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const device bfloat* wb = w + woff0;
    for (uint k = lane * 4u; k < p.K; k += 128u) {
      const float4 xv = *(const device float4*)(x + k);
      a0 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 0u * size_t(p.K) + k)));
      a1 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 1u * size_t(p.K) + k)));
      a2 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 2u * size_t(p.K) + k)));
      a3 += dot(xv, ink2_bf16x4(*(const device uint2*)(wb + 3u * size_t(p.K) + k)));
    }
    const float s0 = simd_sum(a0), s1 = simd_sum(a1), s2 = simd_sum(a2), s3 = simd_sum(a3);
    if (lane == 0u) { dst_base[0] = s0; dst_base[1] = s1; dst_base[2] = s2; dst_base[3] = s3; }
    return;
  }
  float acc[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  for (uint k = lane * 4u; k < p.K; k += 128u) {
    float4 wv[4];
    for (uint q = 0u; q < mc; ++q)
      wv[q] = ink2_bf16x4(*(const device uint2*)(w + woff0 + size_t(q) * p.K + k));
    for (uint m = 0u; m < p.M; ++m) {
      const float4 xv = *(const device float4*)(x + size_t(m) * p.K + k);
      for (uint q = 0u; q < mc; ++q) acc[q][m] += dot(xv, wv[q]);
    }
  }
  for (uint m = 0u; m < p.M; ++m) {
    float sv[4];
    for (uint q = 0u; q < mc; ++q) sv[q] = simd_sum(acc[q][m]);
    if (lane == 0u) {
      device float* dst = dst_base + size_t(m) * seg_len;
      for (uint q = 0u; q < mc; ++q) dst[q] = sv[q];
    }
  }
}

[[kernel]] void ink2_bf16_mv8_multi6(const device float* x0 [[buffer(0)]], const device float* x1 [[buffer(1)]],
                                     const device float* x2 [[buffer(2)]], const device bfloat* w [[buffer(3)]],
                                     device float* y0 [[buffer(4)]], device float* y1 [[buffer(5)]],
                                     device float* y2 [[buffer(6)]], device float* y3 [[buffer(7)]],
                                     device float* y4 [[buffer(8)]], device float* y5 [[buffer(9)]],
                                     constant Ink2MultiMmP& p [[buffer(10)]],
                                     uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[8][128];
  const uint tg_n0 = tg * 16u;
  if (tg_n0 >= p.N_tot) return;
  uint tg_seg = 0u;
  while (tg_seg + 1u < p.n_segs && tg_n0 >= p.offs[tg_seg + 1u]) ++tg_seg;
  if (tg_seg >= p.n_segs) return;
  const device float* x = (tg_seg < 4u) ? x0 : (tg_seg == 4u ? x1 : x2);

  const uint n0 = (tg * 4u + sg) * 4u;
  const uint seg_off = p.offs[tg_seg];
  const uint seg_len = p.lens[tg_seg];
  const bool valid = (n0 < seg_off + seg_len) && (n0 < p.N_tot);
  const uint mc = valid ? min(4u, seg_off + seg_len - n0) : 0u;
  float acc[4][8] = {};
  const size_t woff0 = size_t(n0) * p.K;
  const uint total_v4 = p.M * 32u;
  for (uint kbase = 0u; kbase < p.K; kbase += 128u) {
    for (uint idx = tid; idx < total_v4; idx += 128u) {
      const uint m = idx >> 5u;
      const uint c4 = (idx & 31u) << 2u;
      *(threadgroup float4*)(xs[m] + c4) = *(const device float4*)(x + size_t(m) * p.K + kbase + c4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
      const uint kl = lane * 4u;
      float4 wv[4];
      for (uint q = 0u; q < mc; ++q)
        wv[q] = ink2_bf16x4(*(const device uint2*)(w + woff0 + size_t(q) * p.K + kbase + kl));
      for (uint m = 0u; m < p.M; ++m) {
        const float4 xv = *(const threadgroup float4*)(xs[m] + kl);
        for (uint q = 0u; q < mc; ++q) acc[q][m] += dot(xv, wv[q]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!valid) return;
  device float* bases[6] = {y0, y1, y2, y3, y4, y5};
  device float* dst_base = bases[tg_seg] + (n0 - seg_off);
  for (uint m = 0u; m < p.M; ++m) {
    float sv[4];
    for (uint q = 0u; q < mc; ++q) sv[q] = simd_sum(acc[q][m]);
    if (lane == 0u) {
      device float* dst = dst_base + size_t(m) * seg_len;
      for (uint q = 0u; q < mc; ++q) dst[q] = sv[q];
    }
  }
}

[[kernel]] void ink2_bf16_mv8(const device float* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                              device float* y [[buffer(2)]], constant Ink2MmP& p [[buffer(3)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[8][128];
  const uint m0 = tg.y * 8u;
  if (m0 >= p.M) return;
  const uint mc_m = min(8u, p.M - m0);
  const uint n0 = (tg.x * 4u + sg) * 4u;
  const bool valid = n0 < p.N;
  const uint mc = valid ? min(4u, p.N - n0) : 0u;
  float acc[4][8] = {};
  const size_t woff0 = size_t(n0) * p.K;
  const uint total_v4 = mc_m * 32u;
  for (uint kbase = 0u; kbase < p.K; kbase += 128u) {
    for (uint idx = tid; idx < total_v4; idx += 128u) {
      const uint m = idx >> 5u;
      const uint c4 = (idx & 31u) << 2u;
      *(threadgroup float4*)(xs[m] + c4) = *(const device float4*)(x + size_t(m0 + m) * p.x_ld + kbase + c4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
      const uint kl = lane * 4u;
      float4 wv[4];
      for (uint q = 0u; q < mc; ++q)
        wv[q] = ink2_bf16x4(*(const device uint2*)(w + woff0 + size_t(q) * p.K + kbase + kl));
      for (uint m = 0u; m < mc_m; ++m) {
        const float4 xv = *(const threadgroup float4*)(xs[m] + kl);
        for (uint q = 0u; q < mc; ++q) acc[q][m] += dot(xv, wv[q]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!valid) return;
  for (uint m = 0u; m < mc_m; ++m) {
    float sv[4];
    for (uint q = 0u; q < mc; ++q) sv[q] = simd_sum(acc[q][m]);
    if (lane == 0u) {
      device float* dst = y + size_t(m0 + m) * p.y_ld + n0;
      for (uint q = 0u; q < mc; ++q) dst[q] = p.add ? (dst[q] + sv[q]) : sv[q];
    }
  }
}

struct Ink2Q8MmP {
  uint M, N, K, nblk, add;
};

struct Ink2Q8Multi6P {
  uint M, K, N_tot, n_segs, nblk;
  uint offs[7];
  uint lens[6];
};

[[kernel]] void ink2_q8_mv4(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                             const device half* ws [[buffer(2)]], device float* y [[buffer(3)]],
                             constant Ink2Q8MmP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4u + sg) * 4u;
  if (n0 >= p.N) return;
  const uint mc = min(4u, p.N - n0);
  const size_t woff0 = size_t(n0) * p.K;
  const size_t soff0 = size_t(n0) * p.nblk;
  float acc[4][4] = {};
  const uint blk_off = lane >> 3u;
  for (uint kbase = 0u; kbase < p.K; kbase += 128u) {
    const uint kl = lane * 4u;
    const uint blk = (kbase >> 5u) + blk_off;
    float4 xv[4];
    for (uint m = 0u; m < p.M; ++m) xv[m] = *(const device float4*)(x + size_t(m) * p.K + kbase + kl);
    for (uint q = 0u; q < mc; ++q) {
      const uint qw = *(const device uint*)(wq + woff0 + size_t(q) * p.K + kbase + kl);
      const float sc = float(ws[soff0 + size_t(q) * p.nblk + blk]);
      const float4 wv = float4(as_type<char4>(qw)) * sc;
      for (uint m = 0u; m < p.M; ++m) acc[q][m] += dot(xv[m], wv);
    }
  }
  for (uint m = 0u; m < p.M; ++m) {
    float sv[4];
    for (uint q = 0u; q < mc; ++q) sv[q] = simd_sum(acc[q][m]);
    if (lane == 0u) {
      device float* dst = y + size_t(m) * p.N + n0;
      for (uint q = 0u; q < mc; ++q) dst[q] = p.add ? (dst[q] + sv[q]) : sv[q];
    }
  }
}

[[kernel]] void ink2_q8_mv8(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                             const device half* ws [[buffer(2)]], device float* y [[buffer(3)]],
                             constant Ink2Q8MmP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[8][256];
  const uint n0 = (tg * 4u + sg) * 4u;
  const bool valid = n0 < p.N;
  const uint mc = valid ? min(4u, p.N - n0) : 0u;
  float acc[4][8] = {};
  const size_t woff0 = size_t(n0) * p.K;
  const size_t soff0 = size_t(n0) * p.nblk;
  const uint total_v4 = p.M * 64u;
  const uint blk_loc = lane >> 2u;
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    for (uint idx = tid; idx < total_v4; idx += 128u) {
      const uint m = idx >> 6u;
      const uint c4 = (idx & 63u) << 2u;
      *(threadgroup float4*)(xs[m] + c4) = *(const device float4*)(x + size_t(m) * p.K + kbase + c4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
      const uint kl = lane * 8u;
      const uint blk = (kbase >> 5u) + blk_loc;
      for (uint q = 0u; q < mc; ++q) {
        const uint2 qw = *(const device uint2*)(wq + woff0 + size_t(q) * p.K + kbase + kl);
        const float sc = float(ws[soff0 + size_t(q) * p.nblk + blk]);
        const float4 wa = float4(as_type<char4>(qw.x)) * sc;
        const float4 wb = float4(as_type<char4>(qw.y)) * sc;
        for (uint m = 0u; m < p.M; ++m) {
          const threadgroup float* xp = xs[m] + kl;
          acc[q][m] += dot(*(const threadgroup float4*)(xp + 0u), wa) + dot(*(const threadgroup float4*)(xp + 4u), wb);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!valid) return;
  for (uint m = 0u; m < p.M; ++m) {
    float sv[4];
    for (uint q = 0u; q < mc; ++q) sv[q] = simd_sum(acc[q][m]);
    if (lane == 0u) {
      device float* dst = y + size_t(m) * p.N + n0;
      for (uint q = 0u; q < mc; ++q) dst[q] = p.add ? (dst[q] + sv[q]) : sv[q];
    }
  }
}

[[kernel]] void ink2_q8_mv4_multi6(const device float* x0 [[buffer(0)]], const device float* x1 [[buffer(1)]],
                                    const device float* x2 [[buffer(2)]], const device uchar* wq [[buffer(3)]],
                                    const device half* ws [[buffer(4)]], device float* y0 [[buffer(5)]],
                                    device float* y1 [[buffer(6)]], device float* y2 [[buffer(7)]],
                                    device float* y3 [[buffer(8)]], device float* y4 [[buffer(9)]],
                                    device float* y5 [[buffer(10)]], constant Ink2Q8Multi6P& p [[buffer(11)]],
                                    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[4][128];
  const uint tg_n0 = tg * 16u;
  if (tg_n0 >= p.N_tot) return;
  uint tg_seg = 0u;
  while (tg_seg + 1u < p.n_segs && tg_n0 >= p.offs[tg_seg + 1u]) ++tg_seg;
  if (tg_seg >= p.n_segs) return;
  const device float* x = (tg_seg < 4u) ? x0 : (tg_seg == 4u ? x1 : x2);

  const uint n0 = (tg * 4u + sg) * 4u;
  const uint seg_off = p.offs[tg_seg];
  const uint seg_len = p.lens[tg_seg];
  const bool valid = (n0 < seg_off + seg_len) && (n0 < p.N_tot);
  const uint mc = valid ? min(4u, seg_off + seg_len - n0) : 0u;
  float acc[4][4] = {};
  const size_t woff0 = size_t(n0) * p.K;
  const size_t soff0 = size_t(n0) * p.nblk;
  const uint total_v4 = p.M * 32u;
  const uint blk_loc = lane >> 3u;
  for (uint kbase = 0u; kbase < p.K; kbase += 128u) {
    for (uint idx = tid; idx < total_v4; idx += 128u) {
      const uint m = idx >> 5u;
      const uint c4 = (idx & 31u) << 2u;
      *(threadgroup float4*)(xs[m] + c4) = *(const device float4*)(x + size_t(m) * p.K + kbase + c4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
      const uint kl = lane * 4u;
      const uint blk = (kbase >> 5u) + blk_loc;
      for (uint q = 0u; q < mc; ++q) {
        const uint qw = *(const device uint*)(wq + woff0 + size_t(q) * p.K + kbase + kl);
        const float sc = float(ws[soff0 + size_t(q) * p.nblk + blk]);
        const float4 wv = float4(as_type<char4>(qw)) * sc;
        for (uint m = 0u; m < p.M; ++m) {
          const float4 xv = *(const threadgroup float4*)(xs[m] + kl);
          acc[q][m] += dot(xv, wv);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!valid) return;
  device float* bases[6] = {y0, y1, y2, y3, y4, y5};
  device float* dst_base = bases[tg_seg] + (n0 - seg_off);
  for (uint m = 0u; m < p.M; ++m) {
    float sv[4];
    for (uint q = 0u; q < mc; ++q) sv[q] = simd_sum(acc[q][m]);
    if (lane == 0u) {
      device float* dst = dst_base + size_t(m) * seg_len;
      for (uint q = 0u; q < mc; ++q) dst[q] = sv[q];
    }
  }
}

[[kernel]] void ink2_q8_mv8_multi6(const device float* x0 [[buffer(0)]], const device float* x1 [[buffer(1)]],
                                    const device float* x2 [[buffer(2)]], const device uchar* wq [[buffer(3)]],
                                    const device half* ws [[buffer(4)]], device float* y0 [[buffer(5)]],
                                    device float* y1 [[buffer(6)]], device float* y2 [[buffer(7)]],
                                    device float* y3 [[buffer(8)]], device float* y4 [[buffer(9)]],
                                    device float* y5 [[buffer(10)]], constant Ink2Q8Multi6P& p [[buffer(11)]],
                                    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[8][256];
  const uint tg_n0 = tg * 16u;
  if (tg_n0 >= p.N_tot) return;
  uint tg_seg = 0u;
  while (tg_seg + 1u < p.n_segs && tg_n0 >= p.offs[tg_seg + 1u]) ++tg_seg;
  if (tg_seg >= p.n_segs) return;
  const device float* x = (tg_seg < 4u) ? x0 : (tg_seg == 4u ? x1 : x2);

  const uint n0 = (tg * 4u + sg) * 4u;
  const uint seg_off = p.offs[tg_seg];
  const uint seg_len = p.lens[tg_seg];
  const bool valid = (n0 < seg_off + seg_len) && (n0 < p.N_tot);
  const uint mc = valid ? min(4u, seg_off + seg_len - n0) : 0u;
  float acc[4][8] = {};
  const size_t woff0 = size_t(n0) * p.K;
  const size_t soff0 = size_t(n0) * p.nblk;
  const uint total_v4 = p.M * 64u;
  const uint blk_loc = lane >> 2u;
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    for (uint idx = tid; idx < total_v4; idx += 128u) {
      const uint m = idx >> 6u;
      const uint c4 = (idx & 63u) << 2u;
      *(threadgroup float4*)(xs[m] + c4) = *(const device float4*)(x + size_t(m) * p.K + kbase + c4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
      const uint kl = lane * 8u;
      const uint blk = (kbase >> 5u) + blk_loc;
      for (uint q = 0u; q < mc; ++q) {
        const uint2 qw = *(const device uint2*)(wq + woff0 + size_t(q) * p.K + kbase + kl);
        const float sc = float(ws[soff0 + size_t(q) * p.nblk + blk]);
        const float4 wa = float4(as_type<char4>(qw.x)) * sc;
        const float4 wb = float4(as_type<char4>(qw.y)) * sc;
        for (uint m = 0u; m < p.M; ++m) {
          const threadgroup float* xp = xs[m] + kl;
          acc[q][m] += dot(*(const threadgroup float4*)(xp + 0u), wa) + dot(*(const threadgroup float4*)(xp + 4u), wb);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!valid) return;
  device float* bases[6] = {y0, y1, y2, y3, y4, y5};
  device float* dst_base = bases[tg_seg] + (n0 - seg_off);
  for (uint m = 0u; m < p.M; ++m) {
    float sv[4];
    for (uint q = 0u; q < mc; ++q) sv[q] = simd_sum(acc[q][m]);
    if (lane == 0u) {
      device float* dst = dst_base + size_t(m) * seg_len;
      for (uint q = 0u; q < mc; ++q) dst[q] = sv[q];
    }
  }
}

[[kernel]] void ink2_bf16_mm(const device float* X [[buffer(0)]], const device void* W [[buffer(1)]],
                             device float* Y [[buffer(2)]], constant Ink2MmP& p [[buffer(3)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BM = 32, BN = 64, BK = 32, PAD = 8;
  constexpr uint TM = 16, TN = 32, FM = 2, FN = 4;
  threadgroup half xs[BM * (BK + PAD)];
  threadgroup half ws[BN * (BK + PAD)];
  const uint m0 = tg.y * BM, n0 = tg.x * BN;
  const uint sgm = sg / 2, sgn = sg % 2;

  simdgroup_float8x8 acc[FM][FN];
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) acc[i][j] = simdgroup_float8x8(0);

  for (uint k0 = 0; k0 < p.K; k0 += BK) {
    // 128 threads load 32 * 32 = 1024 halfs of X (8 per thread)
    const uint xr = tid / 4, xc = (tid % 4) * 8;
    const uint xm = m0 + xr;
    threadgroup half* xdst = xs + xr * (BK + PAD) + xc;
    if (xm < p.M && k0 + xc + 7 < p.K) {
      const float4 v0 = *(const device float4*)(X + size_t(xm) * p.x_ld + k0 + xc);
      const float4 v1 = *(const device float4*)(X + size_t(xm) * p.x_ld + k0 + xc + 4);
      *(threadgroup half4*)(xdst) = half4(v0);
      *(threadgroup half4*)(xdst + 4) = half4(v1);
    } else {
      for (uint j = 0; j < 8; ++j)
        xdst[j] = (xm < p.M && k0 + xc + j < p.K) ? half(X[size_t(xm) * p.x_ld + k0 + xc + j]) : 0.0h;
    }

    // 128 threads load 64 * 32 = 2048 halfs of W (16 per thread)
    const uint wr = tid / 2, wc = (tid % 2) * 16;
    const uint wn = n0 + wr;
    threadgroup half* wdst = ws + wr * (BK + PAD) + wc;
    if (wn < p.N && k0 + wc + 15 < p.K) {
      if (p.w_f32) {
        const device float4* wp = (const device float4*)((const device float*)W + size_t(wn) * p.K + k0 + wc);
        for (int q = 0; q < 4; ++q) *(threadgroup half4*)(wdst + q * 4) = half4(wp[q]);
      } else {
        const device bfloat* wb = (const device bfloat*)W + size_t(wn) * p.K + k0 + wc;
        for (int q = 0; q < 16; ++q) wdst[q] = half(float(wb[q]));
      }
    } else {
      for (uint j = 0; j < 16; ++j) {
        float v = 0.0f;
        if (wn < p.N && k0 + wc + j < p.K) {
          const size_t idx = size_t(wn) * p.K + k0 + wc + j;
          v = p.w_f32 ? ((const device float*)W)[idx] : float(((const device bfloat*)W)[idx]);
        }
        wdst[j] = half(v);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_half8x8 a[FM], b[FN];
      for (int i = 0; i < FM; ++i) simdgroup_load(a[i], xs + (sgm * TM + i * 8) * (BK + PAD) + kk, BK + PAD);
      for (int j = 0; j < FN; ++j)
        simdgroup_load(b[j], ws + (sgn * TN + j * 8) * (BK + PAD) + kk, BK + PAD, ulong2(0, 0), true);
      for (int i = 0; i < FM; ++i)
        for (int j = 0; j < FN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup float stage_buf[4 * 64];
  threadgroup float* stage = stage_buf + sg * 64;
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) {
      simdgroup_store(acc[i][j], stage, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 64; e += 32) {
        const uint r = e / 8, c = e % 8;
        const uint m = m0 + sgm * TM + i * 8 + r;
        const uint n = n0 + sgn * TN + j * 8 + c;
        if (m < p.M && n < p.N) {
          device float* dst = Y + size_t(m) * p.y_ld + n;
          const float v = stage[e];
          *dst = p.add ? (*dst + v) : v;
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---- 6. Short Causal Convolutions (static & dynamic) ----
struct Ink2SconvStaticP {
  uint T, C, w_f32, save_snap, snap_stride, add_res;
  float scale;
};

[[kernel]] void ink2_sconv_static(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                                  device float* hist [[buffer(2)]], device float* out [[buffer(3)]],
                                  constant Ink2SconvStaticP& p [[buffer(4)]], device float* snap [[buffer(5)]],
                                  const device float* res [[buffer(6)]], uint c [[thread_position_in_grid]]) {
  if (c >= p.C) return;
  float w0, w1, w2, w3;
  if (p.w_f32) {
    const float4 wf = *(const device float4*)((const device float*)w + size_t(c) * 4);
    w0 = wf.x; w1 = wf.y; w2 = wf.z; w3 = wf.w;
  } else {
    const device bfloat* wb = (const device bfloat*)w + size_t(c) * 4;
    w0 = float(wb[0]); w1 = float(wb[1]); w2 = float(wb[2]); w3 = float(wb[3]);
  }
  float h0 = hist[0 * p.C + c], h1 = hist[1 * p.C + c], h2 = hist[2 * p.C + c];
  for (uint t = 0; t < p.T; ++t) {
    const size_t idx = size_t(t) * p.C + c;
    const float xt = x[idx];
    const float conv = w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt;
    float y = (conv + xt) * p.scale;
    if (p.add_res) y += res[idx];
    out[idx] = y;
    h0 = h1; h1 = h2; h2 = xt;
    if (p.save_snap) {
      const size_t sb = size_t(t) * size_t(p.snap_stride) + c;
      snap[sb + 0 * p.C] = h0;
      snap[sb + 1 * p.C] = h1;
      snap[sb + 2 * p.C] = h2;
    }
  }
  hist[0 * p.C + c] = h0;
  hist[1 * p.C + c] = h1;
  hist[2 * p.C + c] = h2;
}

[[kernel]] void ink2_sconv_static_pf(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                                     const device float* hist [[buffer(2)]], device float* out [[buffer(3)]],
                                     constant Ink2SconvStaticP& p [[buffer(4)]], const device float* res [[buffer(5)]],
                                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint c = tg.x * 128u + tid;
  if (c >= p.C) return;
  float w0, w1, w2, w3;
  if (p.w_f32) {
    const float4 wf = *(const device float4*)((const device float*)w + size_t(c) * 4u);
    w0 = wf.x; w1 = wf.y; w2 = wf.z; w3 = wf.w;
  } else {
    const device bfloat* wb = (const device bfloat*)w + size_t(c) * 4u;
    w0 = float(wb[0]); w1 = float(wb[1]); w2 = float(wb[2]); w3 = float(wb[3]);
  }
  const uint t0 = tg.y * 16u;
  const uint t1 = min(p.T, t0 + 16u);
  float h0 = (t0 >= 3u) ? x[size_t(t0 - 3u) * p.C + c] : hist[t0 * p.C + c];
  float h1 = (t0 >= 2u) ? x[size_t(t0 - 2u) * p.C + c] : hist[(t0 + 1u) * p.C + c];
  float h2 = (t0 >= 1u) ? x[size_t(t0 - 1u) * p.C + c] : hist[(t0 + 2u) * p.C + c];
  for (uint t = t0; t < t1; ++t) {
    const size_t idx = size_t(t) * p.C + c;
    const float xt = x[idx];
    const float conv = w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt;
    float y = (conv + xt) * p.scale;
    if (p.add_res) y += res[idx];
    out[idx] = y;
    h0 = h1; h1 = h2; h2 = xt;
  }
}

struct Ink2SconvDynP {
  uint T, C, n_heads, sw_f32, save_snap, snap_stride;
};

[[kernel]] void ink2_sconv_dynamic(const device float* x [[buffer(0)]], const device float* dyn [[buffer(1)]],
                                   const device void* static_w [[buffer(2)]], device float* hist [[buffer(3)]],
                                   device float* out [[buffer(4)]], constant Ink2SconvDynP& p [[buffer(5)]],
                                   device float* snap [[buffer(6)]], uint c [[thread_position_in_grid]]) {
  if (c >= p.C) return;
  const uint head_dim = p.C / p.n_heads;
  const uint h = c / head_dim;
  float sw[4];
  for (uint lag = 0; lag < 4; ++lag) {
    const size_t idx = size_t(lag) * p.C + c;
    sw[lag] = p.sw_f32 ? ((const device float*)static_w)[idx] : float(((const device bfloat*)static_w)[idx]);
  }
  float h0 = hist[0 * p.C + c], h1 = hist[1 * p.C + c], h2 = hist[2 * p.C + c];
  for (uint t = 0; t < p.T; ++t) {
    const float xt = x[size_t(t) * p.C + c];
    const float4 d = *(const device float4*)(dyn + (size_t(t) * p.n_heads + h) * 4);
    const float conv = (sw[0] + d.x) * xt + (sw[1] + d.y) * h2 + (sw[2] + d.z) * h1 + (sw[3] + d.w) * h0;
    out[size_t(t) * p.C + c] = (conv + xt) * (1.0f / 3.0f);
    h0 = h1; h1 = h2; h2 = xt;
    if (p.save_snap) {
      const size_t sb = size_t(t) * size_t(p.snap_stride) + c;
      snap[sb + 0 * p.C] = h0;
      snap[sb + 1 * p.C] = h1;
      snap[sb + 2 * p.C] = h2;
    }
  }
  hist[0 * p.C + c] = h0;
  hist[1 * p.C + c] = h1;
  hist[2 * p.C + c] = h2;
}

// ---- 7. Q/K/V/R Clip + V Norm + Tau + KV Cache Write ----
struct Ink2QkvrP {
  uint T, Hq, Hkv, D, d_rel, pos0, kv_cap, is_swa, scale_f32;
  float qk_clip, r_clip, v_eps, log_floor, log_alpha;
  uint kv_fp8;
};

[[kernel]] void ink2_qkvr_prep(device float* q [[buffer(0)]], const device float* k [[buffer(1)]],
                               const device float* v [[buffer(2)]], device float* r [[buffer(3)]],
                               const device void* q_scale [[buffer(4)]], const device void* r_scale [[buffer(5)]],
                               device void* kc [[buffer(6)]], device void* vc [[buffer(7)]],
                               constant Ink2QkvrP& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  const uint kind_h = tg.x, t = tg.y;
  const uint pos = p.pos0 + t;
  const uint slot = p.is_swa ? (pos % p.kv_cap) : pos;
  const float tau = p.is_swa ? 1.0f : (1.0f + p.log_alpha * metal::precise::log(max(1.0f, float(pos + 1) / p.log_floor)));

  if (kind_h < p.Hq) {
    const uint h = kind_h;
    const size_t off = (size_t(t) * p.Hq + h) * p.D + tid;
    const size_t soff = size_t(h) * p.D + tid;
    const float sc = p.scale_f32 ? ((const device float*)q_scale)[soff] : float(((const device bfloat*)q_scale)[soff]);
    const float val = q[off] * sc;
    const float ss = tg_sum(val * val, scratch, tid, p.D);
    const float mult = min(1.0f, p.qk_clip / (metal::precise::sqrt(ss) + 1e-6f)) * tau;
    q[off] = val * mult;
    return;
  }
  uint rem = kind_h - p.Hq;
  if (rem < p.Hkv) {
    const uint h = rem;
    const float val = k[(size_t(t) * p.Hkv + h) * p.D + tid];
    const float ss = tg_sum(val * val, scratch, tid, p.D);
    const float mult = min(1.0f, p.qk_clip / (metal::precise::sqrt(ss) + 1e-6f));
    const size_t idx = (size_t(slot) * p.Hkv + h) * p.D + tid;
    const float kv = val * mult;
    if (p.kv_fp8) ((device uchar*)kc)[idx] = ink2_f32_to_e4m3(kv);
    else ((device half*)kc)[idx] = half(kv);
    return;
  }
  rem -= p.Hkv;
  if (rem < p.Hkv) {
    const uint h = rem;
    const float val = v[(size_t(t) * p.Hkv + h) * p.D + tid];
    const float ss = tg_sum(val * val, scratch, tid, p.D);
    const float inv = metal::precise::rsqrt(ss / float(p.D) + p.v_eps);
    const size_t idx = (size_t(slot) * p.Hkv + h) * p.D + tid;
    const float vv = val * inv;
    if (p.kv_fp8) ((device uchar*)vc)[idx] = ink2_f32_to_e4m3(vv);
    else ((device half*)vc)[idx] = half(vv);
    return;
  }
  const uint h = rem - p.Hkv;
  const size_t off = (size_t(t) * p.Hq + h) * p.d_rel + tid;
  const size_t soff = size_t(h) * p.d_rel + tid;
  float val = 0.0f;
  if (tid < p.d_rel) {
    const float sc = p.scale_f32 ? ((const device float*)r_scale)[soff] : float(((const device bfloat*)r_scale)[soff]);
    val = r[off] * sc;
  }
  const float ss = tg_sum(val * val, scratch, tid, p.D);
  if (tid < p.d_rel) {
    const float mult = min(1.0f, p.r_clip / (metal::precise::sqrt(ss) + 1e-6f)) * tau;
    r[off] = val * mult;
  }
}

// ---- 8. Relative Position Logit Bias: rb [T, Hq, rel_extent] = r @ rel_proj ----
struct Ink2RelP {
  uint T, Hq, d_rel, extent, proj_f32;
};

[[kernel]] void ink2_rel_bias(const device float* r [[buffer(0)]], const device void* rel_proj [[buffer(1)]],
                              device float* rb [[buffer(2)]], constant Ink2RelP& p [[buffer(3)]],
                              uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col = tg.x * 32 + tid;
  if (col >= p.extent) return;
  const uint h = tg.y, t = tg.z;
  const device float* rv = r + (size_t(t) * p.Hq + h) * p.d_rel;
  float acc = 0.0f;
  for (uint d = 0; d < p.d_rel; ++d) {
    const size_t pidx = size_t(d) * p.extent + col;
    const float pv = p.proj_f32 ? ((const device float*)rel_proj)[pidx] : float(((const device bfloat*)rel_proj)[pidx]);
    acc += rv[d] * pv;
  }
  rb[(size_t(t) * p.Hq + h) * p.extent + col] = acc;
}

[[kernel]] void ink2_rel_bias_pf(const device float* r [[buffer(0)]], const device void* rel_proj [[buffer(1)]],
                                  device float* rb [[buffer(2)]], constant Ink2RelP& p [[buffer(3)]],
                                  uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 sr[64][4];
  const uint col = tg.x * 32u + lane;
  const bool col_ok = col < p.extent;
  const uint c_safe = col_ok ? col : (p.extent - 1u);
  float4 p0, p1, p2, p3;
  if (p.proj_f32) {
    const device float* rp = (const device float*)rel_proj + c_safe;
    p0 = float4(rp[0 * p.extent], rp[1 * p.extent], rp[2 * p.extent], rp[3 * p.extent]);
    p1 = float4(rp[4 * p.extent], rp[5 * p.extent], rp[6 * p.extent], rp[7 * p.extent]);
    p2 = float4(rp[8 * p.extent], rp[9 * p.extent], rp[10 * p.extent], rp[11 * p.extent]);
    p3 = float4(rp[12 * p.extent], rp[13 * p.extent], rp[14 * p.extent], rp[15 * p.extent]);
  } else {
    const device bfloat* rp = (const device bfloat*)rel_proj + c_safe;
    p0 = float4(float(rp[0 * p.extent]), float(rp[1 * p.extent]), float(rp[2 * p.extent]), float(rp[3 * p.extent]));
    p1 = float4(float(rp[4 * p.extent]), float(rp[5 * p.extent]), float(rp[6 * p.extent]), float(rp[7 * p.extent]));
    p2 = float4(float(rp[8 * p.extent]), float(rp[9 * p.extent]), float(rp[10 * p.extent]), float(rp[11 * p.extent]));
    p3 = float4(float(rp[12 * p.extent]), float(rp[13 * p.extent]), float(rp[14 * p.extent]), float(rp[15 * p.extent]));
  }
  const uint t0 = tg.y * 8u, h0 = tg.z * 8u;
  for (uint idx = tid; idx < 256u; idx += 128u) {
    const uint row = idx >> 2u, q = idx & 3u;
    const uint t = t0 + (row >> 3u), h = h0 + (row & 7u);
    sr[row][q] = (t < p.T && h < p.Hq) ? *(const device float4*)(r + (size_t(t) * p.Hq + h) * 16u + q * 4u) : float4(0.0f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!col_ok) return;
  for (uint dt = 0u; dt < 2u; ++dt) {
    const uint t = t0 + sg * 2u + dt;
    if (t >= p.T) break;
    const uint row_base = (sg * 2u + dt) << 3u;
    device float* rbt = rb + size_t(t) * p.Hq * p.extent + col;
    for (uint hh = 0u; hh < 8u && (h0 + hh) < p.Hq; ++hh) {
      const uint row = row_base + hh;
      const float4 r0 = sr[row][0], r1 = sr[row][1], r2 = sr[row][2], r3 = sr[row][3];
      float acc = 0.0f;
      for (uint d = 0u; d < 4u; ++d) acc += r0[d] * p0[d];
      for (uint d = 0u; d < 4u; ++d) acc += r1[d] * p1[d];
      for (uint d = 0u; d < 4u; ++d) acc += r2[d] * p2[d];
      for (uint d = 0u; d < 4u; ++d) acc += r3[d] * p3[d];
      rbt[size_t(h0 + hh) * p.extent] = acc;
    }
  }
}

// ---- 9. Sink Attention + Rel Bias + Gate Multiply (Split-K Decode & Prefill) ----
struct Ink2AttnDecP {
  uint pos, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, chunk, nchunks;
  float scale;
  uint kv_fp8;
};

[[kernel]] void ink2_attn_decode(const device float* q [[buffer(0)]], const device void* kc [[buffer(1)]],
                                 const device void* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                 device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
                                 device float* po [[buffer(6)]], constant Ink2AttnDecP& p [[buffer(7)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgm[4], tgl[4], tgo[4][128];
  const uint h = tg.x, cid = tg.y;
  const uint hkv = h / (p.Hq / p.Hkv);
  const uint d0 = lane * 4;
  const float4 qv = *(const device float4*)(q + size_t(h) * p.D + d0);

  const uint lk = p.is_swa ? min(p.pos + 1u, p.swa) : (p.pos + 1u);
  const uint k_first = p.pos + 1u - lk;
  const uint c0 = k_first + cid * p.chunk;
  const uint c1 = min(p.pos + 1u, c0 + p.chunk);

  float mw = -INFINITY, lw = 0.0f;
  float4 ow = float4(0.0f);
  for (uint kp = c0 + sg; kp < c1; kp += 4) {
    const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
    const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = ink2_load_kv4(kc, koff, p.kv_fp8);
    float sc = simd_sum(dot(qv, kv)) * p.scale;
    const uint dist = p.pos - kp;
    if (dist < p.rel_extent) sc += rb[size_t(h) * p.rel_extent + dist];
    const float4 vv = ink2_load_kv4(vc, koff, p.kv_fp8);
    if (lw == 0.0f) {
      mw = sc; lw = 1.0f; ow = vv;
    } else {
      const float nm = max(mw, sc);
      const float e1 = metal::precise::exp(mw - nm), e2 = metal::precise::exp(sc - nm);
      ow = ow * e1 + vv * e2;
      lw = lw * e1 + e2;
      mw = nm;
    }
  }
  if (lane == 0) {
    tgm[sg] = mw;
    tgl[sg] = lw;
  }
  *(threadgroup float4*)(&tgo[sg][d0]) = ow;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    float mm = -INFINITY, ll = 0.0f;
    float4 oo = float4(0.0f);
    for (uint s = 0; s < 4; ++s) {
      const float l2 = tgl[s];
      if (l2 == 0.0f) continue;
      const float m2 = tgm[s];
      const float4 o2 = *(const threadgroup float4*)(&tgo[s][d0]);
      if (ll == 0.0f) {
        mm = m2; ll = l2; oo = o2;
      } else {
        const float nm = max(mm, m2);
        const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
        oo = oo * e1 + o2 * e2;
        ll = ll * e1 + l2 * e2;
        mm = nm;
      }
    }
    const size_t out_idx = size_t(h) * p.nchunks + cid;
    if (lane == 0) {
      pm[out_idx] = mm;
      pl[out_idx] = ll;
    }
    *(device float4*)(po + out_idx * p.D + d0) = oo;
  }
}

[[kernel]] void ink2_attn_decode_reduce(const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]],
                                        const device float* po [[buffer(2)]], const device float* sinks [[buffer(3)]],
                                        const device float* gate [[buffer(4)]], device float* out [[buffer(5)]],
                                        constant Ink2AttnDecP& p [[buffer(6)]], uint h [[threadgroup_position_in_grid]],
                                        uint lane [[thread_index_in_simdgroup]]) {
  const uint d0 = lane * 4;
  float mm = sinks[h], ll = 1.0f;
  float4 oo = float4(0.0f);
  for (uint cid = 0; cid < p.nchunks; ++cid) {
    const size_t idx = size_t(h) * p.nchunks + cid;
    const float l2 = pl[idx];
    if (l2 == 0.0f) continue;
    const float m2 = pm[idx];
    const float4 o2 = *(const device float4*)(po + idx * p.D + d0);
    const float nm = max(mm, m2);
    const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
    oo = oo * e1 + o2 * e2;
    ll = ll * e1 + l2 * e2;
    mm = nm;
  }
  const float4 g4 = *(const device float4*)(gate + size_t(h) * p.D + d0);
  const float4 sig = float4(1.0f) / (float4(1.0f) + float4(metal::precise::exp(-g4.x), metal::precise::exp(-g4.y),
                                                           metal::precise::exp(-g4.z), metal::precise::exp(-g4.w)));
  *(device float4*)(out + size_t(h) * p.D + d0) = (oo / ll) * sig;
}

[[kernel]] void ink2_attn_decode_1pass(const device float* q [[buffer(0)]], const device void* kc [[buffer(1)]],
                                       const device void* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                       const device float* sinks [[buffer(4)]], const device float* gate [[buffer(5)]],
                                       device float* out [[buffer(6)]], constant Ink2AttnDecP& p [[buffer(7)]],
                                       uint h [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgm[4], tgl[4], tgo[4][128];
  const uint hkv = h / (p.Hq / p.Hkv);
  const uint d0 = lane * 4u;
  const float4 qv = *(const device float4*)(q + size_t(h) * p.D + d0);

  const uint lk = p.is_swa ? min(p.pos + 1u, p.swa) : (p.pos + 1u);
  const uint k_first = p.pos + 1u - lk;

  float mw = -INFINITY, lw = 0.0f;
  float4 ow = float4(0.0f);
  for (uint kp = k_first + sg; kp <= p.pos; kp += 4u) {
    const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
    const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = ink2_load_kv4(kc, koff, p.kv_fp8);
    float sc = simd_sum(dot(qv, kv)) * p.scale;
    const uint dist = p.pos - kp;
    if (dist < p.rel_extent) sc += rb[size_t(h) * p.rel_extent + dist];
    const float4 vv = ink2_load_kv4(vc, koff, p.kv_fp8);
    if (lw == 0.0f) {
      mw = sc; lw = 1.0f; ow = vv;
    } else {
      const float nm = max(mw, sc);
      const float e1 = metal::precise::exp(mw - nm), e2 = metal::precise::exp(sc - nm);
      ow = ow * e1 + vv * e2;
      lw = lw * e1 + e2;
      mw = nm;
    }
  }
  if (lane == 0u) {
    tgm[sg] = mw;
    tgl[sg] = lw;
  }
  *(threadgroup float4*)(&tgo[sg][d0]) = ow;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u) {
    float mm = sinks[h], ll = 1.0f;
    float4 oo = float4(0.0f);
    for (uint s = 0u; s < 4u; ++s) {
      const float l2 = tgl[s];
      if (l2 == 0.0f) continue;
      const float m2 = tgm[s];
      const float4 o2 = *(const threadgroup float4*)(&tgo[s][d0]);
      const float nm = max(mm, m2);
      const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
      oo = oo * e1 + o2 * e2;
      ll = ll * e1 + l2 * e2;
      mm = nm;
    }
    const float4 g4 = *(const device float4*)(gate + size_t(h) * p.D + d0);
    const float4 sig = float4(1.0f) / (float4(1.0f) + float4(metal::precise::exp(-g4.x), metal::precise::exp(-g4.y),
                                                             metal::precise::exp(-g4.z), metal::precise::exp(-g4.w)));
    *(device float4*)(out + size_t(h) * p.D + d0) = (oo / ll) * sig;
  }
}

struct Ink2AttnVerP {
  uint T, pos0, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, chunk, nchunks;
  float scale;
  uint kv_fp8;
};

[[kernel]] void ink2_attn_verify(const device float* q [[buffer(0)]], const device void* kc [[buffer(1)]],
                                 const device void* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                 device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
                                 device float* po [[buffer(6)]], constant Ink2AttnVerP& p [[buffer(7)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgm[4], tgl[4], tgo[4][128];
  const uint h = tg.x, cid = tg.y;
  const uint hkv = h / (p.Hq / p.Hkv);
  const uint d0 = lane * 4;
  float4 qv[8], ow[8];
  float mw[8], lw[8];
  for (uint t = 0; t < p.T; ++t) {
    qv[t] = *(const device float4*)(q + (size_t(t) * p.Hq + h) * p.D + d0);
    mw[t] = -INFINITY;
    lw[t] = 0.0f;
    ow[t] = float4(0.0f);
  }
  const uint k_min = (p.is_swa && p.pos0 + 1u > p.swa) ? (p.pos0 + 1u - p.swa) : 0u;
  const uint k_max = p.pos0 + p.T;
  const uint c0 = k_min + cid * p.chunk;
  const uint c1 = min(k_max, c0 + p.chunk);

  for (uint kp = c0 + sg; kp < c1; kp += 4) {
    const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
    const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = ink2_load_kv4(kc, koff, p.kv_fp8);
    const float4 vv = ink2_load_kv4(vc, koff, p.kv_fp8);
    for (uint t = 0; t < p.T; ++t) {
      const uint qpos = p.pos0 + t;
      const uint t_min = (p.is_swa && qpos + 1u > p.swa) ? (qpos + 1u - p.swa) : 0u;
      if (kp < t_min || kp > qpos) continue;
      float sc = simd_sum(dot(qv[t], kv)) * p.scale;
      const uint dist = qpos - kp;
      if (dist < p.rel_extent) sc += rb[(size_t(t) * p.Hq + h) * p.rel_extent + dist];
      if (lw[t] == 0.0f) {
        mw[t] = sc; lw[t] = 1.0f; ow[t] = vv;
      } else {
        const float nm = max(mw[t], sc);
        const float e1 = metal::precise::exp(mw[t] - nm), e2 = metal::precise::exp(sc - nm);
        ow[t] = ow[t] * e1 + vv * e2;
        lw[t] = lw[t] * e1 + e2;
        mw[t] = nm;
      }
    }
  }
  for (uint t = 0; t < p.T; ++t) {
    if (lane == 0) {
      tgm[sg] = mw[t];
      tgl[sg] = lw[t];
    }
    *(threadgroup float4*)(&tgo[sg][d0]) = ow[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float mm = -INFINITY, ll = 0.0f;
      float4 oo = float4(0.0f);
      for (uint s = 0; s < 4; ++s) {
        const float l2 = tgl[s];
        if (l2 == 0.0f) continue;
        const float m2 = tgm[s];
        const float4 o2 = *(const threadgroup float4*)(&tgo[s][d0]);
        if (ll == 0.0f) {
          mm = m2; ll = l2; oo = o2;
        } else {
          const float nm = max(mm, m2);
          const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
          oo = oo * e1 + o2 * e2;
          ll = ll * e1 + l2 * e2;
          mm = nm;
        }
      }
      const size_t out_idx = (size_t(t) * p.Hq + h) * p.nchunks + cid;
      if (lane == 0) {
        pm[out_idx] = mm;
        pl[out_idx] = ll;
      }
      *(device float4*)(po + out_idx * p.D + d0) = oo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

[[kernel]] void ink2_attn_verify2(const device float* q [[buffer(0)]], const device void* kc [[buffer(1)]],
                                  const device void* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                  device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
                                  device float* po [[buffer(6)]], constant Ink2AttnVerP& p [[buffer(7)]],
                                  uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgm[4], tgl[4], tgo[4][128];
  const uint h = tg.x, cid = tg.y;
  const uint hkv = h / (p.Hq / p.Hkv);
  const uint d0 = lane * 4u;
  const float4 qv0 = *(const device float4*)(q + size_t(h) * p.D + d0);
  const float4 qv1 = *(const device float4*)(q + (size_t(p.Hq) + h) * p.D + d0);
  float mw0 = -INFINITY, lw0 = 0.0f, mw1 = -INFINITY, lw1 = 0.0f;
  float4 ow0 = float4(0.0f), ow1 = float4(0.0f);

  const uint qpos0 = p.pos0, qpos1 = p.pos0 + 1u;
  const uint t0_min = (p.is_swa && qpos0 + 1u > p.swa) ? (qpos0 + 1u - p.swa) : 0u;
  const uint t1_min = (p.is_swa && qpos1 + 1u > p.swa) ? (qpos1 + 1u - p.swa) : 0u;
  const uint k_max = p.pos0 + 2u;
  const uint c0 = t0_min + cid * p.chunk;
  const uint c1 = min(k_max, c0 + p.chunk);

  for (uint kp = c0 + sg; kp < c1; kp += 4u) {
    const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
    const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = ink2_load_kv4(kc, koff, p.kv_fp8);
    const float4 vv = ink2_load_kv4(vc, koff, p.kv_fp8);
    const float qk0 = simd_sum(dot(qv0, kv)) * p.scale;
    const float qk1 = simd_sum(dot(qv1, kv)) * p.scale;
    if (kp >= t0_min && kp <= qpos0) {
      float sc = qk0;
      const uint dist = qpos0 - kp;
      if (dist < p.rel_extent) sc += rb[size_t(h) * p.rel_extent + dist];
      if (lw0 == 0.0f) {
        mw0 = sc; lw0 = 1.0f; ow0 = vv;
      } else {
        const float nm = max(mw0, sc);
        const float e1 = metal::precise::exp(mw0 - nm), e2 = metal::precise::exp(sc - nm);
        ow0 = ow0 * e1 + vv * e2;
        lw0 = lw0 * e1 + e2;
        mw0 = nm;
      }
    }
    if (kp >= t1_min && kp <= qpos1) {
      float sc = qk1;
      const uint dist = qpos1 - kp;
      if (dist < p.rel_extent) sc += rb[(size_t(p.Hq) + h) * p.rel_extent + dist];
      if (lw1 == 0.0f) {
        mw1 = sc; lw1 = 1.0f; ow1 = vv;
      } else {
        const float nm = max(mw1, sc);
        const float e1 = metal::precise::exp(mw1 - nm), e2 = metal::precise::exp(sc - nm);
        ow1 = ow1 * e1 + vv * e2;
        lw1 = lw1 * e1 + e2;
        mw1 = nm;
      }
    }
  }

  for (uint t = 0u; t < 2u; ++t) {
    if (lane == 0u) {
      tgm[sg] = t == 0u ? mw0 : mw1;
      tgl[sg] = t == 0u ? lw0 : lw1;
    }
    *(threadgroup float4*)(&tgo[sg][d0]) = t == 0u ? ow0 : ow1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      float mm = -INFINITY, ll = 0.0f;
      float4 oo = float4(0.0f);
      for (uint s = 0u; s < 4u; ++s) {
        const float l2 = tgl[s];
        if (l2 == 0.0f) continue;
        const float m2 = tgm[s];
        const float4 o2 = *(const threadgroup float4*)(&tgo[s][d0]);
        if (ll == 0.0f) {
          mm = m2; ll = l2; oo = o2;
        } else {
          const float nm = max(mm, m2);
          const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
          oo = oo * e1 + o2 * e2;
          ll = ll * e1 + l2 * e2;
          mm = nm;
        }
      }
      const size_t out_idx = (size_t(t) * p.Hq + h) * p.nchunks + cid;
      if (lane == 0u) {
        pm[out_idx] = mm;
        pl[out_idx] = ll;
      }
      *(device float4*)(po + out_idx * p.D + d0) = oo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

[[kernel]] void ink2_attn_verify_reduce(const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]],
                                        const device float* po [[buffer(2)]], const device float* sinks [[buffer(3)]],
                                        const device float* gate [[buffer(4)]], device float* out [[buffer(5)]],
                                        constant Ink2AttnVerP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                        uint lane [[thread_index_in_simdgroup]]) {
  const uint h = tg.x, t = tg.y, d0 = lane * 4;
  float mm = sinks[size_t(t) * p.Hq + h], ll = 1.0f;
  float4 oo = float4(0.0f);
  const size_t row_base = (size_t(t) * p.Hq + h) * p.nchunks;
  for (uint cid = 0; cid < p.nchunks; ++cid) {
    const size_t idx = row_base + cid;
    const float l2 = pl[idx];
    if (l2 == 0.0f) continue;
    const float m2 = pm[idx];
    const float4 o2 = *(const device float4*)(po + idx * p.D + d0);
    const float nm = max(mm, m2);
    const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
    oo = oo * e1 + o2 * e2;
    ll = ll * e1 + l2 * e2;
    mm = nm;
  }
  const size_t out_off = (size_t(t) * p.Hq + h) * p.D + d0;
  const float4 g4 = *(const device float4*)(gate + out_off);
  const float4 sig = float4(1.0f) / (float4(1.0f) + float4(metal::precise::exp(-g4.x), metal::precise::exp(-g4.y),
                                                           metal::precise::exp(-g4.z), metal::precise::exp(-g4.w)));
  *(device float4*)(out + out_off) = (oo / ll) * sig;
}

struct Ink2AttnPfP {
  uint T, pos0, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa;
  float scale;
  uint kv_fp8;
};

[[kernel]] void ink2_attn_prefill(const device float* q [[buffer(0)]], const device void* kc [[buffer(1)]],
                                  const device void* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                  const device float* sinks [[buffer(4)]], const device float* gate [[buffer(5)]],
                                  device float* out [[buffer(6)]], constant Ink2AttnPfP& p [[buffer(7)]],
                                  uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgm[4], tgl[4], tgo[4][128];
  const uint h = tg.x, t = tg.y;
  const uint hkv = h / (p.Hq / p.Hkv);
  const uint qpos = p.pos0 + t;
  const uint d0 = lane * 4;
  const size_t qoff = (size_t(t) * p.Hq + h) * p.D + d0;
  const float4 qv = *(const device float4*)(q + qoff);
  const device float* rbh = rb + (size_t(t) * p.Hq + h) * p.rel_extent;

  const uint lk = p.is_swa ? min(qpos + 1u, p.swa) : (qpos + 1u);
  const uint c0 = qpos + 1u - lk;
  float mw = -INFINITY, lw = 0.0f;
  float4 ow = float4(0.0f);
  for (uint kp = c0 + sg; kp <= qpos; kp += 4) {
    const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
    const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = ink2_load_kv4(kc, koff, p.kv_fp8);
    float sc = simd_sum(dot(qv, kv)) * p.scale;
    const uint dist = qpos - kp;
    if (dist < p.rel_extent) sc += rbh[dist];
    const float4 vv = ink2_load_kv4(vc, koff, p.kv_fp8);
    if (lw == 0.0f) {
      mw = sc; lw = 1.0f; ow = vv;
    } else {
      const float nm = max(mw, sc);
      const float e1 = metal::precise::exp(mw - nm), e2 = metal::precise::exp(sc - nm);
      ow = ow * e1 + vv * e2;
      lw = lw * e1 + e2;
      mw = nm;
    }
  }
  if (lane == 0) {
    tgm[sg] = mw;
    tgl[sg] = lw;
  }
  *(threadgroup float4*)(&tgo[sg][d0]) = ow;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    float mm = sinks[size_t(t) * p.Hq + h], ll = 1.0f;
    float4 oo = float4(0.0f);
    for (uint s = 0; s < 4; ++s) {
      const float l2 = tgl[s];
      if (l2 == 0.0f) continue;
      const float m2 = tgm[s];
      const float4 o2 = *(const threadgroup float4*)(&tgo[s][d0]);
      const float nm = max(mm, m2);
      const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
      oo = oo * e1 + o2 * e2;
      ll = ll * e1 + l2 * e2;
      mm = nm;
    }
    const float4 g4 = *(const device float4*)(gate + qoff);
    const float4 sig = float4(1.0f) / (float4(1.0f) + float4(metal::precise::exp(-g4.x), metal::precise::exp(-g4.y),
                                                             metal::precise::exp(-g4.z), metal::precise::exp(-g4.w)));
    *(device float4*)(out + qoff) = (oo / ll) * sig;
  }
}

// ---- 10. MoE Router (E <= 128, top-k = 8) ----
struct Ink2RouteP {
  uint T, E, K, bias_f32;
  float route_scale;
};

[[kernel]] void ink2_moe_route(const device float* logits [[buffer(0)]], const device void* bias [[buffer(1)]],
                               device uint* eid [[buffer(2)]], device float* ew [[buffer(3)]],
                               constant Ink2RouteP& p [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
  float ss[8], sb[8];
  uint cnt = 0u;
  for (uint e = lane; e < p.E && cnt < 8u; e += 32u, ++cnt) {
    const float l = logits[size_t(t) * p.E + e];
    const float s = 1.0f / (1.0f + metal::precise::exp(-l));
    const float b = p.bias_f32 ? ((const device float*)bias)[e] : float(((const device bfloat*)bias)[e]);
    ss[cnt] = s;
    sb[cnt] = s + b;
  }
  uint chosen[8];
  float c_ss[8];
  float wsum = 0.0f;
  for (uint slot = 0; slot < p.K; ++slot) {
    float lbest = -INFINITY;
    uint lidx = 0xFFFFFFFFu;
    for (uint i = 0; i < cnt; ++i) {
      if (sb[i] > lbest) {
        lbest = sb[i];
        lidx = lane + i * 32u;
      }
    }
    const float gbest = simd_max(lbest);
    const uint cand = (lbest == gbest && isfinite(gbest)) ? lidx : 0xFFFFFFFFu;
    const uint winner = simd_min(cand);
    const uint wlane = winner & 31u;
    const uint wi = (winner < p.E && wlane == lane) ? (winner >> 5) : 0u;
    const float my_s = (winner < p.E && wlane == lane) ? ss[wi] : 0.0f;
    if (winner < p.E && wlane == lane) sb[wi] = -INFINITY;
    const float ws = (winner < p.E) ? simd_broadcast(my_s, ushort(wlane)) : 0.0f;
    if (lane == 0u) {
      chosen[slot] = winner < p.E ? winner : 0u;
      c_ss[slot] = ws;
      wsum += ws;
    }
  }
  if (lane == 0u) {
    const float inv = p.route_scale / (wsum + 1e-20f);
    for (uint slot = 0; slot < p.K; ++slot) {
      eid[size_t(t) * p.K + slot] = chosen[slot];
      ew[size_t(t) * p.K + slot] = c_ss[slot] * inv;
    }
  }
}

// ---- 11. MoE NVFP4 (layers 0..19): vectorized loads with exact FP32 E2M1 accumulation ----
struct Ink2MoeP {
  uint T, H, I, K_top;
};

[[kernel]] void ink2_moe_nvfp4_w13_situ(const device float* x [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                        const device uchar* w13 [[buffer(2)]], const device uchar* sc [[buffer(3)]],
                                        const device float* outer [[buffer(4)]], device float* act [[buffer(5)]],
                                        constant Ink2MoeP& p [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint j0 = (tg.x * 4u + sg) * 4u;
  if (j0 >= p.I) return;
  const uint nr = min(4u, p.I - j0);
  const uint slot = tg.y, t = tg.z;
  const uint e = eid[size_t(t) * p.K_top + slot];
  const uint two_i = 2u * p.I, half_h = p.H / 2u, blocks = p.H / 16u;
  const size_t wg0_row = size_t(e) * two_i + 2u * j0;
  const device float* xp = x + size_t(t) * p.H;

  float ag[4] = {0.0f, 0.0f, 0.0f, 0.0f}, au[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint blk = lane; blk < blocks; blk += 32u) {
    const uint kbase = blk * 16u;
    const float4 x0 = *(const device float4*)(xp + kbase + 0u);
    const float4 x1 = *(const device float4*)(xp + kbase + 4u);
    const float4 x2 = *(const device float4*)(xp + kbase + 8u);
    const float4 x3 = *(const device float4*)(xp + kbase + 12u);
    for (uint q = 0u; q < nr; ++q) {
      {
        const size_t r = wg0_row + 2u * q;
        const float sb = ink2_e4m3_16384(sc[r * blocks + blk]);
        const uint2 pw = *(const device uint2*)(w13 + r * half_h + blk * 8u);
        ag[q] += (dot(x0, ink2_e2m1_4_raw(pw.x)) + dot(x1, ink2_e2m1_4_raw(pw.x >> 16)) +
                  dot(x2, ink2_e2m1_4_raw(pw.y)) + dot(x3, ink2_e2m1_4_raw(pw.y >> 16))) * sb;
      }
      {
        const size_t r = wg0_row + 2u * q + 1u;
        const float sb = ink2_e4m3_16384(sc[r * blocks + blk]);
        const uint2 pw = *(const device uint2*)(w13 + r * half_h + blk * 8u);
        au[q] += (dot(x0, ink2_e2m1_4_raw(pw.x)) + dot(x1, ink2_e2m1_4_raw(pw.x >> 16)) +
                  dot(x2, ink2_e2m1_4_raw(pw.y)) + dot(x3, ink2_e2m1_4_raw(pw.y >> 16))) * sb;
      }
    }
  }
  const float os = outer[e];
  const size_t obase = (size_t(t) * p.K_top + slot) * p.I + j0;
  for (uint q = 0u; q < nr; ++q) {
    const float g = simd_sum(ag[q]) * os, u = simd_sum(au[q]) * os;
    if (lane == 0u) {
      act[obase + q] = (4.0f * metal::precise::tanh(g * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g))) *
                       (25.0f * metal::precise::tanh(u * 0.04f));
    }
  }
}

[[kernel]] void ink2_moe_nvfp4_w2_acc(const device float* act [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                      const device float* ew [[buffer(2)]], const device uchar* w2 [[buffer(3)]],
                                      const device uchar* sc [[buffer(4)]], const device float* outer [[buffer(5)]],
                                      device float* out [[buffer(6)]], constant Ink2MoeP& p [[buffer(7)]],
                                      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg.x * 4u + sg) * 4u;
  if (n0 >= p.H) return;
  const uint nr = min(4u, p.H - n0);
  const uint t = tg.y, half_i = p.I / 2u, blocks = p.I / 16u;
  float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint slot = 0; slot < p.K_top; ++slot) {
    const uint e = eid[size_t(t) * p.K_top + slot];
    const float wt = ew[size_t(t) * p.K_top + slot];
    const size_t wrow0 = size_t(e) * p.H + n0;
    const device float* ap = act + (size_t(t) * p.K_top + slot) * p.I;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint blk = lane; blk < blocks; blk += 32u) {
      const uint kbase = blk * 16u;
      const float4 a0 = *(const device float4*)(ap + kbase + 0u);
      const float4 a1 = *(const device float4*)(ap + kbase + 4u);
      const float4 a2 = *(const device float4*)(ap + kbase + 8u);
      const float4 a3 = *(const device float4*)(ap + kbase + 12u);
      for (uint q = 0u; q < nr; ++q) {
        const size_t r = wrow0 + q;
        const float sb = ink2_e4m3_16384(sc[r * blocks + blk]);
        const uint2 pw = *(const device uint2*)(w2 + r * half_i + blk * 8u);
        sum[q] += (dot(a0, ink2_e2m1_4_raw(pw.x)) + dot(a1, ink2_e2m1_4_raw(pw.x >> 16)) +
                   dot(a2, ink2_e2m1_4_raw(pw.y)) + dot(a3, ink2_e2m1_4_raw(pw.y >> 16))) * sb;
      }
    }
    for (uint q = 0u; q < nr; ++q) total[q] += wt * (simd_sum(sum[q]) * outer[e]);
  }
  if (lane == 0u) {
    device float* dst = out + size_t(t) * p.H + n0;
    for (uint q = 0u; q < nr; ++q) dst[q] = total[q];
  }
}

// ---- 12. MoE MXFP8 (layers 20..23): F8_E4M3 weights + UE8M0 block-32 scales ----
[[kernel]] void ink2_moe_mxfp8_w13_situ(const device float* x [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                        const device uchar* w13 [[buffer(2)]], const device uchar* sc [[buffer(3)]],
                                        device float* act [[buffer(4)]], constant Ink2MoeP& p [[buffer(5)]],
                                        uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
  const uint j0 = (tg.x * 4u + sg) * 2u;
  if (j0 >= p.I) return;
  const bool has1 = (j0 + 1u) < p.I;
  const uint slot = tg.y, t = tg.z;
  const uint e = eid[size_t(t) * p.K_top + slot];
  const uint two_i = 2u * p.I;
  const uint blocks = p.H / 32u;
  const size_t wg0_row = size_t(e) * two_i + 2u * j0;
  const size_t wu0_row = wg0_row + 1u;
  const size_t wg1_row = wg0_row + 2u;
  const size_t wu1_row = wg0_row + 3u;
  const device float* xp = x + size_t(t) * p.H;

  float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f;
  for (uint blk = lane; blk < blocks; blk += 32u) {
    const uint kbase = blk * 32u;
    const float4 x0 = *(const device float4*)(xp + kbase + 0u), x1 = *(const device float4*)(xp + kbase + 4u);
    const float4 x2 = *(const device float4*)(xp + kbase + 8u), x3 = *(const device float4*)(xp + kbase + 12u);
    const float4 x4 = *(const device float4*)(xp + kbase + 16u), x5 = *(const device float4*)(xp + kbase + 20u);
    const float4 x6 = *(const device float4*)(xp + kbase + 24u), x7 = *(const device float4*)(xp + kbase + 28u);
    auto dot32 = [&](size_t r) {
      const float sb = ink2_ue8m0(sc[r * blocks + blk]);
      const uint4 wa = *(const device uint4*)(w13 + r * p.H + kbase + 0u);
      const uint4 wb = *(const device uint4*)(w13 + r * p.H + kbase + 16u);
      return (dot(x0, ink2_e4m3_4(wa.x)) + dot(x1, ink2_e4m3_4(wa.y)) +
              dot(x2, ink2_e4m3_4(wa.z)) + dot(x3, ink2_e4m3_4(wa.w)) +
              dot(x4, ink2_e4m3_4(wb.x)) + dot(x5, ink2_e4m3_4(wb.y)) +
              dot(x6, ink2_e4m3_4(wb.z)) + dot(x7, ink2_e4m3_4(wb.w))) * sb;
    };
    ag0 += dot32(wg0_row);
    au0 += dot32(wu0_row);
    if (has1) {
      ag1 += dot32(wg1_row);
      au1 += dot32(wu1_row);
    }
  }
  const float g0 = simd_sum(ag0), u0 = simd_sum(au0);
  const float g1 = has1 ? simd_sum(ag1) : 0.0f;
  const float u1 = has1 ? simd_sum(au1) : 0.0f;
  if (lane == 0u) {
    const size_t obase = (size_t(t) * p.K_top + slot) * p.I + j0;
    act[obase] = (4.0f * metal::precise::tanh(g0 * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g0))) *
                 (25.0f * metal::precise::tanh(u0 * 0.04f));
    if (has1) {
      act[obase + 1u] = (4.0f * metal::precise::tanh(g1 * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g1))) *
                        (25.0f * metal::precise::tanh(u1 * 0.04f));
    }
  }
}

[[kernel]] void ink2_moe_mxfp8_w2_acc(const device float* act [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                      const device float* ew [[buffer(2)]], const device uchar* w2 [[buffer(3)]],
                                      const device uchar* sc [[buffer(4)]], device float* out [[buffer(5)]],
                                      constant Ink2MoeP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg.x * 4u + sg) * 4u;
  if (n0 >= p.H) return;
  const uint nr = min(4u, p.H - n0);
  const uint t = tg.y;
  const uint blocks = p.I / 32u;
  float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint slot = 0u; slot < p.K_top; ++slot) {
    const uint e = eid[size_t(t) * p.K_top + slot];
    const float wt = ew[size_t(t) * p.K_top + slot];
    const size_t wrow0 = size_t(e) * p.H + n0;
    const device float* ap = act + (size_t(t) * p.K_top + slot) * p.I;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint blk = lane; blk < blocks; blk += 32u) {
      const uint kbase = blk * 32u;
      const float4 a0 = *(const device float4*)(ap + kbase + 0u), a1 = *(const device float4*)(ap + kbase + 4u);
      const float4 a2 = *(const device float4*)(ap + kbase + 8u), a3 = *(const device float4*)(ap + kbase + 12u);
      const float4 a4 = *(const device float4*)(ap + kbase + 16u), a5 = *(const device float4*)(ap + kbase + 20u);
      const float4 a6 = *(const device float4*)(ap + kbase + 24u), a7 = *(const device float4*)(ap + kbase + 28u);
      auto dot32 = [&](size_t r) {
        const float sb = ink2_ue8m0(sc[r * blocks + blk]);
        const uint4 wa = *(const device uint4*)(w2 + r * p.I + kbase + 0u);
        const uint4 wb = *(const device uint4*)(w2 + r * p.I + kbase + 16u);
        return (dot(a0, ink2_e4m3_4(wa.x)) + dot(a1, ink2_e4m3_4(wa.y)) +
                dot(a2, ink2_e4m3_4(wa.z)) + dot(a3, ink2_e4m3_4(wa.w)) +
                dot(a4, ink2_e4m3_4(wb.x)) + dot(a5, ink2_e4m3_4(wb.y)) +
                dot(a6, ink2_e4m3_4(wb.z)) + dot(a7, ink2_e4m3_4(wb.w))) * sb;
      };
      sum[0] += dot32(wrow0);
      if (nr > 1u) sum[1] += dot32(wrow0 + 1u);
      if (nr > 2u) sum[2] += dot32(wrow0 + 2u);
      if (nr > 3u) sum[3] += dot32(wrow0 + 3u);
    }
    for (uint q = 0u; q < nr; ++q) total[q] += wt * simd_sum(sum[q]);
  }
  if (lane == 0u) {
    device float* dst = out + size_t(t) * p.H + n0;
    for (uint q = 0u; q < nr; ++q) dst[q] = total[q];
  }
}

// ---- 13. Buffer Copy, Final Logit Softcap ----
[[kernel]] void ink2_copy_bytes(const device uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
                                constant uint64_t& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  const uint64_t off = uint64_t(i) * 16;
  if (off + 16 <= n) {
    *(device uint4*)(dst + off) = *(const device uint4*)(src + off);
  } else {
    for (uint64_t k = off; k < n; ++k) dst[k] = src[k];
  }
}

struct Ink2SoftcapP {
  uint n;
  float cap, inv_cap;
};

[[kernel]] void ink2_softcap(device float* x [[buffer(0)]], constant Ink2SoftcapP& p [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
  if (i < p.n) x[i] = p.cap * metal::precise::tanh(x[i] * p.inv_cap);
}

// ---- 14. GPU-bucketed Small-Batch MoE (2 <= T <= 31, Metal 3 compute, zero atomics) ----
struct Ink2MoeSmallBucketP {
  uint T, K_top, E;
};

[[kernel]] void ink2_moe_bucket_small(const device uint* eid [[buffer(0)]], const device float* ew [[buffer(1)]],
                                      device uint* tiles [[buffer(2)]], device uint* ntiles [[buffer(3)]],
                                      device uint* st_tok [[buffer(4)]], device uint* st_pair [[buffer(5)]],
                                      device float* st_ew [[buffer(6)]], constant Ink2MoeSmallBucketP& p [[buffer(7)]],
                                      uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint sh_off[4];
  threadgroup uint sh_act[4];
  threadgroup uint sh_eid[64];
  threadgroup float sh_ew[64];
  const uint r_tot = p.T * p.K_top;
  if (r_tot <= 64u && tid < r_tot) {
    sh_eid[tid] = eid[tid];
    sh_ew[tid] = ew[tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint count = 0;
  if (tid < p.E) {
    if (r_tot <= 64u) {
      for (uint idx = 0; idx < r_tot; ++idx) count += (sh_eid[idx] == tid) ? 1u : 0u;
    } else {
      for (uint idx = 0; idx < r_tot; ++idx) count += (eid[idx] == tid) ? 1u : 0u;
    }
  }
  const uint sg = tid >> 5, lane = tid & 31;
  const uint p_excl = simd_prefix_exclusive_sum(count);
  const uint p_sg = simd_sum(count);
  const uint is_act = (count > 0u) ? 1u : 0u;
  const uint e_excl = simd_prefix_exclusive_sum(is_act);
  const uint e_sg = simd_sum(is_act);
  if (lane == 31u) {
    sh_off[sg] = p_sg;
    sh_act[sg] = e_sg;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint p_base = 0, e_base = 0;
  for (uint s = 0; s < sg; ++s) {
    p_base += sh_off[s];
    e_base += sh_act[s];
  }
  const uint p_pos0 = p_base + p_excl;
  const uint e_pos = e_base + e_excl;
  if (is_act) {
    tiles[e_pos * 3 + 0] = tid;
    tiles[e_pos * 3 + 1] = p_pos0;
    tiles[e_pos * 3 + 2] = count;
    uint wp = p_pos0;
    if (r_tot <= 64u) {
      for (uint idx = 0; idx < r_tot; ++idx) {
        if (sh_eid[idx] == tid) {
          st_tok[wp] = idx / p.K_top;
          st_pair[wp] = idx;
          st_ew[wp] = sh_ew[idx];
          ++wp;
        }
      }
    } else {
      for (uint idx = 0; idx < r_tot; ++idx) {
        if (eid[idx] == tid) {
          st_tok[wp] = idx / p.K_top;
          st_pair[wp] = idx;
          st_ew[wp] = ew[idx];
          ++wp;
        }
      }
    }
  }
  if (tid == 0u) ntiles[0] = sh_act[0] + sh_act[1] + sh_act[2] + sh_act[3];
}

[[kernel]] void ink2_moe_nvfp4_w13_g(const device float* x [[buffer(0)]], const device uchar* w13 [[buffer(1)]],
                                     const device uchar* sc [[buffer(2)]], const device float* outer [[buffer(3)]],
                                     const device uint* tiles [[buffer(4)]], const device uint* ntiles [[buffer(5)]],
                                     const device uint* st_tok [[buffer(6)]], const device uint* st_pair [[buffer(7)]],
                                     device float* act [[buffer(8)]], constant Ink2MoeP& p [[buffer(9)]],
                                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint j0 = (tg.x * 4u + sg) * 2u;
  const uint eidx = tg.y;
  if (j0 >= p.I || eidx >= ntiles[0]) return;
  const bool has1 = (j0 + 1u) < p.I;
  const uint e = tiles[eidx * 3 + 0];
  const uint start = tiles[eidx * 3 + 1];
  const uint count = tiles[eidx * 3 + 2];
  const uint two_i = 2 * p.I;
  const uint half_h = p.H / 2;
  const uint blocks = p.H / 16;
  const size_t wg0_row = size_t(e) * two_i + 2 * j0;
  const size_t wu0_row = wg0_row + 1;
  const size_t wg1_row = wg0_row + 2;
  const size_t wu1_row = wg0_row + 3;
  const float os = outer[e];

  if (count == 1u) {
    const uint tok = st_tok[start];
    const uint pair = st_pair[start];
    const device float* xp_base = x + size_t(tok) * p.H;
    float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f;
    for (uint blk = lane; blk < blocks; blk += 32u) {
      const uint kbase = blk * 16u;
      const float4 x0 = *(const device float4*)(xp_base + kbase + 0u);
      const float4 x1 = *(const device float4*)(xp_base + kbase + 4u);
      const float4 x2 = *(const device float4*)(xp_base + kbase + 8u);
      const float4 x3 = *(const device float4*)(xp_base + kbase + 12u);
      {
        const float sb = ink2_e4m3_16384(sc[wg0_row * blocks + blk]);
        const uint2 wv = *(const device uint2*)(w13 + wg0_row * half_h + blk * 8u);
        ag0 += (dot(x0, ink2_e2m1_4_raw(wv.x)) + dot(x1, ink2_e2m1_4_raw(wv.x >> 16)) +
                dot(x2, ink2_e2m1_4_raw(wv.y)) + dot(x3, ink2_e2m1_4_raw(wv.y >> 16))) * sb;
      }
      {
        const float sb = ink2_e4m3_16384(sc[wu0_row * blocks + blk]);
        const uint2 wv = *(const device uint2*)(w13 + wu0_row * half_h + blk * 8u);
        au0 += (dot(x0, ink2_e2m1_4_raw(wv.x)) + dot(x1, ink2_e2m1_4_raw(wv.x >> 16)) +
                dot(x2, ink2_e2m1_4_raw(wv.y)) + dot(x3, ink2_e2m1_4_raw(wv.y >> 16))) * sb;
      }
      if (has1) {
        {
          const float sb = ink2_e4m3_16384(sc[wg1_row * blocks + blk]);
          const uint2 wv = *(const device uint2*)(w13 + wg1_row * half_h + blk * 8u);
          ag1 += (dot(x0, ink2_e2m1_4_raw(wv.x)) + dot(x1, ink2_e2m1_4_raw(wv.x >> 16)) +
                  dot(x2, ink2_e2m1_4_raw(wv.y)) + dot(x3, ink2_e2m1_4_raw(wv.y >> 16))) * sb;
        }
        {
          const float sb = ink2_e4m3_16384(sc[wu1_row * blocks + blk]);
          const uint2 wv = *(const device uint2*)(w13 + wu1_row * half_h + blk * 8u);
          au1 += (dot(x0, ink2_e2m1_4_raw(wv.x)) + dot(x1, ink2_e2m1_4_raw(wv.x >> 16)) +
                  dot(x2, ink2_e2m1_4_raw(wv.y)) + dot(x3, ink2_e2m1_4_raw(wv.y >> 16))) * sb;
        }
      }
    }
    const float g0 = simd_sum(ag0) * os, u0 = simd_sum(au0) * os;
    const float g1 = has1 ? simd_sum(ag1) * os : 0.0f;
    const float u1 = has1 ? simd_sum(au1) * os : 0.0f;
    if (lane == 0u) {
      const size_t obase = size_t(pair) * p.I + j0;
      act[obase] = (4.0f * metal::precise::tanh(g0 * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g0))) *
                   (25.0f * metal::precise::tanh(u0 * 0.04f));
      if (has1) {
        act[obase + 1u] = (4.0f * metal::precise::tanh(g1 * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g1))) *
                          (25.0f * metal::precise::tanh(u1 * 0.04f));
      }
    }
    return;
  }

  for (uint p0 = 0; p0 < count; p0 += 4u) {
    const uint mc = min(4u, count - p0);
    uint toks[4];
    float ag0[4], au0[4], ag1[4], au1[4];
    for (uint m = 0; m < mc; ++m) {
      toks[m] = st_tok[start + p0 + m];
      ag0[m] = 0.0f; au0[m] = 0.0f; ag1[m] = 0.0f; au1[m] = 0.0f;
    }
    for (uint blk = lane; blk < blocks; blk += 32u) {
      const uint kbase = blk * 16u;
      float4 x0[4], x1[4], x2[4], x3[4];
      for (uint m = 0; m < mc; ++m) {
        const device float* xp = x + size_t(toks[m]) * p.H + kbase;
        x0[m] = *(const device float4*)(xp + 0u);
        x1[m] = *(const device float4*)(xp + 4u);
        x2[m] = *(const device float4*)(xp + 8u);
        x3[m] = *(const device float4*)(xp + 12u);
      }
      {
        const float sb = ink2_e4m3_16384(sc[wg0_row * blocks + blk]);
        const uint2 pw = *(const device uint2*)(w13 + wg0_row * half_h + blk * 8u);
        const float4 w0 = ink2_e2m1_4_raw(pw.x), w1 = ink2_e2m1_4_raw(pw.x >> 16);
        const float4 w2v = ink2_e2m1_4_raw(pw.y), w3 = ink2_e2m1_4_raw(pw.y >> 16);
        for (uint m = 0; m < mc; ++m)
          ag0[m] += (dot(x0[m], w0) + dot(x1[m], w1) + dot(x2[m], w2v) + dot(x3[m], w3)) * sb;
      }
      {
        const float sb = ink2_e4m3_16384(sc[wu0_row * blocks + blk]);
        const uint2 pw = *(const device uint2*)(w13 + wu0_row * half_h + blk * 8u);
        const float4 w0 = ink2_e2m1_4_raw(pw.x), w1 = ink2_e2m1_4_raw(pw.x >> 16);
        const float4 w2v = ink2_e2m1_4_raw(pw.y), w3 = ink2_e2m1_4_raw(pw.y >> 16);
        for (uint m = 0; m < mc; ++m)
          au0[m] += (dot(x0[m], w0) + dot(x1[m], w1) + dot(x2[m], w2v) + dot(x3[m], w3)) * sb;
      }
      if (has1) {
        {
          const float sb = ink2_e4m3_16384(sc[wg1_row * blocks + blk]);
          const uint2 pw = *(const device uint2*)(w13 + wg1_row * half_h + blk * 8u);
          const float4 w0 = ink2_e2m1_4_raw(pw.x), w1 = ink2_e2m1_4_raw(pw.x >> 16);
          const float4 w2v = ink2_e2m1_4_raw(pw.y), w3 = ink2_e2m1_4_raw(pw.y >> 16);
          for (uint m = 0; m < mc; ++m)
            ag1[m] += (dot(x0[m], w0) + dot(x1[m], w1) + dot(x2[m], w2v) + dot(x3[m], w3)) * sb;
        }
        {
          const float sb = ink2_e4m3_16384(sc[wu1_row * blocks + blk]);
          const uint2 pw = *(const device uint2*)(w13 + wu1_row * half_h + blk * 8u);
          const float4 w0 = ink2_e2m1_4_raw(pw.x), w1 = ink2_e2m1_4_raw(pw.x >> 16);
          const float4 w2v = ink2_e2m1_4_raw(pw.y), w3 = ink2_e2m1_4_raw(pw.y >> 16);
          for (uint m = 0; m < mc; ++m)
            au1[m] += (dot(x0[m], w0) + dot(x1[m], w1) + dot(x2[m], w2v) + dot(x3[m], w3)) * sb;
        }
      }
    }
    for (uint m = 0; m < mc; ++m) {
      const float g0 = simd_sum(ag0[m]) * os, u0 = simd_sum(au0[m]) * os;
      const float g1 = has1 ? simd_sum(ag1[m]) * os : 0.0f;
      const float u1 = has1 ? simd_sum(au1[m]) * os : 0.0f;
      if (lane == 0u) {
        const size_t obase = size_t(st_pair[start + p0 + m]) * p.I + j0;
        act[obase] = (4.0f * metal::precise::tanh(g0 * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g0))) *
                     (25.0f * metal::precise::tanh(u0 * 0.04f));
        if (has1) {
          act[obase + 1u] = (4.0f * metal::precise::tanh(g1 * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g1))) *
                            (25.0f * metal::precise::tanh(u1 * 0.04f));
        }
      }
    }
  }
}

[[kernel]] void ink2_moe_nvfp4_w2_g(const device float* act [[buffer(0)]], const device uchar* w2 [[buffer(1)]],
                                    const device uchar* sc [[buffer(2)]], const device float* outer [[buffer(3)]],
                                    const device uint* tiles [[buffer(4)]], const device uint* ntiles [[buffer(5)]],
                                    const device uint* st_pair [[buffer(6)]], const device float* st_ew [[buffer(7)]],
                                    device float* pair_out [[buffer(8)]], constant Ink2MoeP& p [[buffer(9)]],
                                    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg.x * 4u + sg) * 4u;
  const uint eidx = tg.y;
  if (n0 >= p.H || eidx >= ntiles[0]) return;
  const uint nr = min(4u, p.H - n0);
  const uint e = tiles[eidx * 3 + 0];
  const uint start = tiles[eidx * 3 + 1];
  const uint count = tiles[eidx * 3 + 2];
  const uint half_i = p.I / 2;
  const uint blocks = p.I / 16;
  const size_t wrow0 = size_t(e) * p.H + n0;
  const float os = outer[e];

  if (count == 1u) {
    const uint pair = st_pair[start];
    const float wos = st_ew[start] * os;
    const device float* ap_base = act + size_t(pair) * p.I;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (uint blk = lane; blk < blocks; blk += 32u) {
      const uint kbase = blk * 16u;
      const float4 a0 = *(const device float4*)(ap_base + kbase + 0u);
      const float4 a1 = *(const device float4*)(ap_base + kbase + 4u);
      const float4 a2 = *(const device float4*)(ap_base + kbase + 8u);
      const float4 a3 = *(const device float4*)(ap_base + kbase + 12u);
      {
        const float sb = ink2_e4m3_16384(sc[wrow0 * blocks + blk]);
        const uint2 pw = *(const device uint2*)(w2 + wrow0 * half_i + blk * 8u);
        s0 += (dot(a0, ink2_e2m1_4_raw(pw.x)) + dot(a1, ink2_e2m1_4_raw(pw.x >> 16)) +
               dot(a2, ink2_e2m1_4_raw(pw.y)) + dot(a3, ink2_e2m1_4_raw(pw.y >> 16))) * sb;
      }
      if (nr > 1u) {
        const size_t r = wrow0 + 1u;
        const float sb = ink2_e4m3_16384(sc[r * blocks + blk]);
        const uint2 pw = *(const device uint2*)(w2 + r * half_i + blk * 8u);
        s1 += (dot(a0, ink2_e2m1_4_raw(pw.x)) + dot(a1, ink2_e2m1_4_raw(pw.x >> 16)) +
               dot(a2, ink2_e2m1_4_raw(pw.y)) + dot(a3, ink2_e2m1_4_raw(pw.y >> 16))) * sb;
      }
      if (nr > 2u) {
        const size_t r = wrow0 + 2u;
        const float sb = ink2_e4m3_16384(sc[r * blocks + blk]);
        const uint2 pw = *(const device uint2*)(w2 + r * half_i + blk * 8u);
        s2 += (dot(a0, ink2_e2m1_4_raw(pw.x)) + dot(a1, ink2_e2m1_4_raw(pw.x >> 16)) +
               dot(a2, ink2_e2m1_4_raw(pw.y)) + dot(a3, ink2_e2m1_4_raw(pw.y >> 16))) * sb;
      }
      if (nr > 3u) {
        const size_t r = wrow0 + 3u;
        const float sb = ink2_e4m3_16384(sc[r * blocks + blk]);
        const uint2 pw = *(const device uint2*)(w2 + r * half_i + blk * 8u);
        s3 += (dot(a0, ink2_e2m1_4_raw(pw.x)) + dot(a1, ink2_e2m1_4_raw(pw.x >> 16)) +
               dot(a2, ink2_e2m1_4_raw(pw.y)) + dot(a3, ink2_e2m1_4_raw(pw.y >> 16))) * sb;
      }
    }
    const float r0 = wos * simd_sum(s0);
    const float r1 = nr > 1u ? (wos * simd_sum(s1)) : 0.0f;
    const float r2 = nr > 2u ? (wos * simd_sum(s2)) : 0.0f;
    const float r3 = nr > 3u ? (wos * simd_sum(s3)) : 0.0f;
    if (lane == 0u) {
      device float* dst = pair_out + size_t(pair) * p.H + n0;
      dst[0] = r0;
      if (nr > 1u) dst[1] = r1;
      if (nr > 2u) dst[2] = r2;
      if (nr > 3u) dst[3] = r3;
    }
    return;
  }

  for (uint p0 = 0; p0 < count; p0 += 4u) {
    const uint mc = min(4u, count - p0);
    uint pairs[4];
    float sum[4][4];
    for (uint m = 0; m < mc; ++m) {
      pairs[m] = st_pair[start + p0 + m];
      sum[0][m] = 0.0f; sum[1][m] = 0.0f; sum[2][m] = 0.0f; sum[3][m] = 0.0f;
    }
    for (uint blk = lane; blk < blocks; blk += 32u) {
      const uint kbase = blk * 16u;
      float4 a0[4], a1[4], a2[4], a3[4];
      for (uint m = 0; m < mc; ++m) {
        const device float* ap = act + size_t(pairs[m]) * p.I + kbase;
        a0[m] = *(const device float4*)(ap + 0u);
        a1[m] = *(const device float4*)(ap + 4u);
        a2[m] = *(const device float4*)(ap + 8u);
        a3[m] = *(const device float4*)(ap + 12u);
      }
      for (uint q = 0u; q < nr; ++q) {
        const size_t r = wrow0 + q;
        const float sb = ink2_e4m3_16384(sc[r * blocks + blk]);
        const uint2 pw = *(const device uint2*)(w2 + r * half_i + blk * 8u);
        const float4 w0 = ink2_e2m1_4_raw(pw.x), w1 = ink2_e2m1_4_raw(pw.x >> 16);
        const float4 w2v = ink2_e2m1_4_raw(pw.y), w3 = ink2_e2m1_4_raw(pw.y >> 16);
        for (uint m = 0; m < mc; ++m)
          sum[q][m] += (dot(a0[m], w0) + dot(a1[m], w1) + dot(a2[m], w2v) + dot(a3[m], w3)) * sb;
      }
    }
    for (uint m = 0; m < mc; ++m) {
      const float wos = st_ew[start + p0 + m] * os;
      for (uint q = 0u; q < nr; ++q) {
        const float rv = wos * simd_sum(sum[q][m]);
        if (lane == 0u) pair_out[size_t(pairs[m]) * p.H + n0 + q] = rv;
      }
    }
  }
}

[[kernel]] void ink2_moe_mxfp8_w13_g(const device float* x [[buffer(0)]], const device uchar* w13 [[buffer(1)]],
                                     const device uchar* sc [[buffer(2)]], const device uint* tiles [[buffer(3)]],
                                     const device uint* ntiles [[buffer(4)]], const device uint* st_tok [[buffer(5)]],
                                     const device uint* st_pair [[buffer(6)]], device float* act [[buffer(7)]],
                                     constant Ink2MoeP& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint j0 = (tg.x * 4u + sg) * 2u;
  const uint eidx = tg.y;
  if (j0 >= p.I || eidx >= ntiles[0]) return;
  const bool has1 = (j0 + 1u) < p.I;
  const uint e = tiles[eidx * 3 + 0];
  const uint start = tiles[eidx * 3 + 1];
  const uint count = tiles[eidx * 3 + 2];
  const uint two_i = 2 * p.I;
  const uint blocks = p.H / 32;
  const size_t wg0_row = size_t(e) * two_i + 2 * j0;
  const size_t wu0_row = wg0_row + 1;
  const size_t wg1_row = wg0_row + 2;
  const size_t wu1_row = wg0_row + 3;

  if (count == 1u) {
    const uint tok = st_tok[start];
    const uint pair = st_pair[start];
    const device float* xp_base = x + size_t(tok) * p.H;
    float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f;
    for (uint blk = lane; blk < blocks; blk += 32u) {
      const uint kbase = blk * 32u;
      const float4 x0 = *(const device float4*)(xp_base + kbase + 0u), x1 = *(const device float4*)(xp_base + kbase + 4u);
      const float4 x2 = *(const device float4*)(xp_base + kbase + 8u), x3 = *(const device float4*)(xp_base + kbase + 12u);
      const float4 x4 = *(const device float4*)(xp_base + kbase + 16u), x5 = *(const device float4*)(xp_base + kbase + 20u);
      const float4 x6 = *(const device float4*)(xp_base + kbase + 24u), x7 = *(const device float4*)(xp_base + kbase + 28u);
      auto dot32 = [&](size_t r) {
        const float sb = ink2_ue8m0(sc[r * blocks + blk]);
        const uint4 wa = *(const device uint4*)(w13 + r * p.H + kbase + 0u);
        const uint4 wb = *(const device uint4*)(w13 + r * p.H + kbase + 16u);
        return (dot(x0, ink2_e4m3_4(wa.x)) + dot(x1, ink2_e4m3_4(wa.y)) +
                dot(x2, ink2_e4m3_4(wa.z)) + dot(x3, ink2_e4m3_4(wa.w)) +
                dot(x4, ink2_e4m3_4(wb.x)) + dot(x5, ink2_e4m3_4(wb.y)) +
                dot(x6, ink2_e4m3_4(wb.z)) + dot(x7, ink2_e4m3_4(wb.w))) * sb;
      };
      ag0 += dot32(wg0_row);
      au0 += dot32(wu0_row);
      if (has1) {
        ag1 += dot32(wg1_row);
        au1 += dot32(wu1_row);
      }
    }
    const float g0 = simd_sum(ag0), u0 = simd_sum(au0);
    const float g1 = has1 ? simd_sum(ag1) : 0.0f;
    const float u1 = has1 ? simd_sum(au1) : 0.0f;
    if (lane == 0u) {
      const size_t obase = size_t(pair) * p.I + j0;
      act[obase] = (4.0f * metal::precise::tanh(g0 * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g0))) *
                   (25.0f * metal::precise::tanh(u0 * 0.04f));
      if (has1) {
        act[obase + 1u] = (4.0f * metal::precise::tanh(g1 * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g1))) *
                          (25.0f * metal::precise::tanh(u1 * 0.04f));
      }
    }
    return;
  }

  for (uint p0 = 0; p0 < count; p0 += 4u) {
    const uint mc = min(4u, count - p0);
    uint toks[4];
    float ag0[4], au0[4], ag1[4], au1[4];
    for (uint m = 0; m < mc; ++m) {
      toks[m] = st_tok[start + p0 + m];
      ag0[m] = 0.0f; au0[m] = 0.0f; ag1[m] = 0.0f; au1[m] = 0.0f;
    }
    for (uint blk = lane; blk < blocks; blk += 32u) {
      const uint kbase = blk * 32u;
      auto accum_row = [&](size_t r, thread float* acc) {
        const float sb = ink2_ue8m0(sc[r * blocks + blk]);
        const uint4 wa = *(const device uint4*)(w13 + r * p.H + kbase + 0u);
        const uint4 wb = *(const device uint4*)(w13 + r * p.H + kbase + 16u);
        const float4 w0 = ink2_e4m3_4(wa.x), w1 = ink2_e4m3_4(wa.y);
        const float4 w2v = ink2_e4m3_4(wa.z), w3 = ink2_e4m3_4(wa.w);
        const float4 w4 = ink2_e4m3_4(wb.x), w5 = ink2_e4m3_4(wb.y);
        const float4 w6 = ink2_e4m3_4(wb.z), w7 = ink2_e4m3_4(wb.w);
        for (uint m = 0; m < mc; ++m) {
          const device float* xp = x + size_t(toks[m]) * p.H + kbase;
          const float4 x0 = *(const device float4*)(xp + 0u), x1 = *(const device float4*)(xp + 4u);
          const float4 x2 = *(const device float4*)(xp + 8u), x3 = *(const device float4*)(xp + 12u);
          const float4 x4 = *(const device float4*)(xp + 16u), x5 = *(const device float4*)(xp + 20u);
          const float4 x6 = *(const device float4*)(xp + 24u), x7 = *(const device float4*)(xp + 28u);
          acc[m] += (dot(x0, w0) + dot(x1, w1) + dot(x2, w2v) + dot(x3, w3) +
                     dot(x4, w4) + dot(x5, w5) + dot(x6, w6) + dot(x7, w7)) * sb;
        }
      };
      accum_row(wg0_row, ag0);
      accum_row(wu0_row, au0);
      if (has1) {
        accum_row(wg1_row, ag1);
        accum_row(wu1_row, au1);
      }
    }
    for (uint m = 0; m < mc; ++m) {
      const float g0 = simd_sum(ag0[m]), u0 = simd_sum(au0[m]);
      const float g1 = has1 ? simd_sum(ag1[m]) : 0.0f;
      const float u1 = has1 ? simd_sum(au1[m]) : 0.0f;
      if (lane == 0u) {
        const size_t obase = size_t(st_pair[start + p0 + m]) * p.I + j0;
        act[obase] = (4.0f * metal::precise::tanh(g0 * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g0))) *
                     (25.0f * metal::precise::tanh(u0 * 0.04f));
        if (has1) {
          act[obase + 1u] = (4.0f * metal::precise::tanh(g1 * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g1))) *
                            (25.0f * metal::precise::tanh(u1 * 0.04f));
        }
      }
    }
  }
}

[[kernel]] void ink2_moe_mxfp8_w2_g(const device float* act [[buffer(0)]], const device uchar* w2 [[buffer(1)]],
                                    const device uchar* sc [[buffer(2)]], const device uint* tiles [[buffer(3)]],
                                    const device uint* ntiles [[buffer(4)]], const device uint* st_pair [[buffer(5)]],
                                    const device float* st_ew [[buffer(6)]], device float* pair_out [[buffer(7)]],
                                    constant Ink2MoeP& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg.x * 4u + sg) * 4u;
  const uint eidx = tg.y;
  if (n0 >= p.H || eidx >= ntiles[0]) return;
  const uint nr = min(4u, p.H - n0);
  const uint e = tiles[eidx * 3 + 0];
  const uint start = tiles[eidx * 3 + 1];
  const uint count = tiles[eidx * 3 + 2];
  const uint blocks = p.I / 32;
  const size_t wrow0 = size_t(e) * p.H + n0;

  if (count == 1u) {
    const uint pair = st_pair[start];
    const float ew_m = st_ew[start];
    const device float* ap_base = act + size_t(pair) * p.I;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (uint blk = lane; blk < blocks; blk += 32u) {
      const uint kbase = blk * 32u;
      const float4 a0 = *(const device float4*)(ap_base + kbase + 0u), a1 = *(const device float4*)(ap_base + kbase + 4u);
      const float4 a2 = *(const device float4*)(ap_base + kbase + 8u), a3 = *(const device float4*)(ap_base + kbase + 12u);
      const float4 a4 = *(const device float4*)(ap_base + kbase + 16u), a5 = *(const device float4*)(ap_base + kbase + 20u);
      const float4 a6 = *(const device float4*)(ap_base + kbase + 24u), a7 = *(const device float4*)(ap_base + kbase + 28u);
      auto dot32 = [&](size_t r) {
        const float sb = ink2_ue8m0(sc[r * blocks + blk]);
        const uint4 wa = *(const device uint4*)(w2 + r * p.I + kbase + 0u);
        const uint4 wb = *(const device uint4*)(w2 + r * p.I + kbase + 16u);
        return (dot(a0, ink2_e4m3_4(wa.x)) + dot(a1, ink2_e4m3_4(wa.y)) +
                dot(a2, ink2_e4m3_4(wa.z)) + dot(a3, ink2_e4m3_4(wa.w)) +
                dot(a4, ink2_e4m3_4(wb.x)) + dot(a5, ink2_e4m3_4(wb.y)) +
                dot(a6, ink2_e4m3_4(wb.z)) + dot(a7, ink2_e4m3_4(wb.w))) * sb;
      };
      s0 += dot32(wrow0);
      if (nr > 1u) s1 += dot32(wrow0 + 1u);
      if (nr > 2u) s2 += dot32(wrow0 + 2u);
      if (nr > 3u) s3 += dot32(wrow0 + 3u);
    }
    const float r0 = ew_m * simd_sum(s0);
    const float r1 = nr > 1u ? (ew_m * simd_sum(s1)) : 0.0f;
    const float r2 = nr > 2u ? (ew_m * simd_sum(s2)) : 0.0f;
    const float r3 = nr > 3u ? (ew_m * simd_sum(s3)) : 0.0f;
    if (lane == 0u) {
      device float* dst = pair_out + size_t(pair) * p.H + n0;
      dst[0] = r0;
      if (nr > 1u) dst[1] = r1;
      if (nr > 2u) dst[2] = r2;
      if (nr > 3u) dst[3] = r3;
    }
    return;
  }

  for (uint p0 = 0; p0 < count; p0 += 4u) {
    const uint mc = min(4u, count - p0);
    uint pairs[4];
    float sum[4][4];
    for (uint m = 0; m < mc; ++m) {
      pairs[m] = st_pair[start + p0 + m];
      sum[0][m] = 0.0f; sum[1][m] = 0.0f; sum[2][m] = 0.0f; sum[3][m] = 0.0f;
    }
    for (uint blk = lane; blk < blocks; blk += 32u) {
      const uint kbase = blk * 32u;
      for (uint q = 0u; q < nr; ++q) {
        const size_t r = wrow0 + q;
        const float sb = ink2_ue8m0(sc[r * blocks + blk]);
        const uint4 wa = *(const device uint4*)(w2 + r * p.I + kbase + 0u);
        const uint4 wb = *(const device uint4*)(w2 + r * p.I + kbase + 16u);
        const float4 w0 = ink2_e4m3_4(wa.x), w1 = ink2_e4m3_4(wa.y);
        const float4 w2v = ink2_e4m3_4(wa.z), w3 = ink2_e4m3_4(wa.w);
        const float4 w4 = ink2_e4m3_4(wb.x), w5 = ink2_e4m3_4(wb.y);
        const float4 w6 = ink2_e4m3_4(wb.z), w7 = ink2_e4m3_4(wb.w);
        for (uint m = 0; m < mc; ++m) {
          const device float* ap = act + size_t(pairs[m]) * p.I + kbase;
          const float4 a0 = *(const device float4*)(ap + 0u), a1 = *(const device float4*)(ap + 4u);
          const float4 a2 = *(const device float4*)(ap + 8u), a3 = *(const device float4*)(ap + 12u);
          const float4 a4 = *(const device float4*)(ap + 16u), a5 = *(const device float4*)(ap + 20u);
          const float4 a6 = *(const device float4*)(ap + 24u), a7 = *(const device float4*)(ap + 28u);
          sum[q][m] += (dot(a0, w0) + dot(a1, w1) + dot(a2, w2v) + dot(a3, w3) +
                        dot(a4, w4) + dot(a5, w5) + dot(a6, w6) + dot(a7, w7)) * sb;
        }
      }
    }
    for (uint m = 0; m < mc; ++m) {
      const float ew_m = st_ew[start + p0 + m];
      for (uint q = 0u; q < nr; ++q) {
        const float rv = ew_m * simd_sum(sum[q][m]);
        if (lane == 0u) pair_out[size_t(pairs[m]) * p.H + n0 + q] = rv;
      }
    }
  }
}

[[kernel]] void ink2_moe_reduce8_m3(const device float4* pair_out [[buffer(0)]], device float4* out [[buffer(1)]],
                                    constant Ink2MoeP& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint h4 = p.H >> 2;
  const uint col = gid.x, t = gid.y;
  if (col >= h4) return;
  float4 acc = float4(0.0f);
  for (uint k = 0; k < p.K_top; ++k) acc += pair_out[size_t(t * p.K_top + k) * h4 + col];
  out[size_t(t) * h4 + col] = acc;
}
