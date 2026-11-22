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
  float xs[10];
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
  float xs[10];
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

[[kernel]] void ink2_rmsnorm3(device float* x0 [[buffer(0)]], device float* x1 [[buffer(1)]],
                              device float* x2 [[buffer(2)]], const device void* w0 [[buffer(3)]],
                              const device void* w1 [[buffer(4)]], const device void* w2 [[buffer(5)]],
                              constant Ink2NormP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  constexpr uint ntg = 256u;
  const uint t = tg.x, which = tg.y;
  device float* row = (which == 0u ? x0 : (which == 1u ? x1 : x2)) + size_t(t) * p.H;
  const device void* wp = (which == 0u ? w0 : (which == 1u ? w1 : w2));
  float xs[10];
  float ss = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float v = row[c];
    xs[k] = v;
    ss += v * v;
  }
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    float y = xs[k] * inv;
    if (p.has_w) y *= p.w_f32 ? ((const device float*)wp)[c] : float(((const device bfloat*)wp)[c]);
    row[c] = y;
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

  float qv[4][10];
  for (uint q = 0; q < p.Q; ++q) {
    uint k = 0;
    for (uint c = tid; c < p.H; c += ntg, ++k) {
      const uint idx = q * p.H + c;
      qv[q][k] = p.q_f32 ? ((const device float*)queries)[idx] : float(((const device bfloat*)queries)[idx]);
    }
  }

  for (uint m = 0; m < p.M; ++m) {
    const device float* sr = ptrs[m] + size_t(t) * p.H;
    float sv[10];
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

[[kernel]] void ink2_attn_res_pass(const device float* s0 [[buffer(0)]], const device float* s1 [[buffer(1)]],
                                   const device float* queries [[buffer(2)]], device float* out [[buffer(3)]],
                                   constant Ink2AttnResP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint ntg = 256u;
  const uint t = tg.x, cidx = tg.y;
  threadgroup float scratch[32];
  threadgroup float tw[2][4];

  const device float* qs = queries + size_t(cidx) * 4u * p.H;
  const device float* sr0 = s0 + size_t(t) * p.H;
  const device float* sr1 = s1 + size_t(t) * p.H;
  float v0[10], v1[10];
  float ss0 = 0.0f, ss1 = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float a = sr0[c], b = sr1[c];
    v0[k] = a; v1[k] = b;
    ss0 += a * a; ss1 += b * b;
  }
  const float inv0 = metal::precise::rsqrt(tg_sum(ss0, scratch, tid, ntg) / float(p.H) + p.eps);
  const float inv1 = metal::precise::rsqrt(tg_sum(ss1, scratch, tid, ntg) / float(p.H) + p.eps);

  for (uint q = 0; q < 4u; ++q) {
    float d0 = 0.0f, d1 = 0.0f;
    const device float* qr = qs + size_t(q) * p.H;
    for (uint c = tid, kk = 0; c < p.H; c += ntg, ++kk) {
      const float qv = qr[c];
      d0 += (v0[kk] * inv0) * qv;
      d1 += (v1[kk] * inv1) * qv;
    }
    const float sc0 = tg_sum(d0, scratch, tid, ntg) * p.scale;
    const float sc1 = tg_sum(d1, scratch, tid, ntg) * p.scale;
    if (tid == 0u) {
      const float mx = max(sc0, sc1);
      const float e0 = metal::precise::exp(sc0 - mx), e1 = metal::precise::exp(sc1 - mx);
      const float isum = 1.0f / (e0 + e1);
      tw[0][q] = e0 * isum;
      tw[1][q] = e1 * isum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device float* dst = out + size_t(cidx) * 4u * size_t(p.T) * p.H + size_t(t) * p.H;
  for (uint c = tid, kk = 0; c < p.H; c += ntg, ++kk) {
    const float n0 = v0[kk] * inv0, n1 = v1[kk] * inv1;
    for (uint q = 0; q < 4u; ++q) {
      dst[size_t(q) * p.T * p.H + c] = tw[0][q] * n0 + tw[1][q] * n1;
    }
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
  float hs[10], ms[10];
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
  float hs[10], ms[10];
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
                                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.x;
  if (r >= p.R) return;
  const uint t0 = tg.y * 8u, mc = min(8u, p.T - t0);
  const size_t row_off0 = size_t(r) * (2u * p.H), row_off1 = row_off0 + p.H;
  float a0[8] = {}, a1[8] = {};
  for (uint c = tid * 4u; c < p.H; c += 512u) {
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
  threadgroup float sh[4][8];
  for (uint m = 0u; m < mc; ++m) {
    const float s = simd_sum(a0[m] + a1[m]);
    if (lane == 0u) sh[sg][m] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < mc) {
    lat[size_t(t0 + tid) * p.R + r] = sh[0][tid] + sh[1][tid] + sh[2][tid] + sh[3][tid];
  }
}

[[kernel]] void ink2_msg_gate_up_mix_s(const device float* h [[buffer(0)]], const device float* m_buf [[buffer(1)]],
                                       const device float* lat [[buffer(2)]], const device void* forget_w [[buffer(3)]],
                                       const device void* gain_w [[buffer(4)]], device float* out [[buffer(5)]],
                                       constant Ink2MsgGateP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sh_lat[8 * 64];
  const uint t0 = tg.y * 8u, mc = min(8u, p.T - t0);
  for (uint idx = tid; idx < mc * p.R; idx += 64u) sh_lat[idx] = lat[size_t(t0) * p.R + idx];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = tg.x * 64u + tid;
  if (c >= p.H) return;
  float df[8] = {}, dg[8] = {};
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

[[kernel]] void ink2_msg_gate_prep3_s(const device float* h0 [[buffer(0)]], const device float* h1 [[buffer(1)]],
                                      const device float* h2 [[buffer(2)]], const device float* m0 [[buffer(3)]],
                                      const device float* m1 [[buffer(4)]], const device float* m2 [[buffer(5)]],
                                      device float* cat [[buffer(6)]], constant Ink2MsgGateP& p [[buffer(7)]],
                                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint ntg = 256u;
  threadgroup float scratch[32];
  const uint t = tg.x, which = tg.y;
  const device float* hr = (which == 0u ? h0 : (which == 1u ? h1 : h2)) + size_t(t) * p.H;
  const device float* mr = (which == 0u ? m0 : (which == 1u ? m1 : m2)) + size_t(t) * p.H;
  float hs[10], ms[10];
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
  device float* d0 = cat + (size_t(which) * p.T + t) * (2u * p.H);
  device float* d1 = d0 + p.H;
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    d0[c] = hs[k] * invh;
    d1[c] = ms[k] * invm;
  }
}

[[kernel]] void ink2_msg_gate_down3_s(const device float* cat [[buffer(0)]], const device void* down0 [[buffer(1)]],
                                      const device void* down1 [[buffer(2)]], const device void* down2 [[buffer(3)]],
                                      device float* lat [[buffer(4)]], constant Ink2MsgGateP& p [[buffer(5)]],
                                      uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.x, t0 = tg.y * 8u, which = tg.z;
  if (r >= p.R) return;
  const uint mc = min(8u, p.T - t0);
  const device void* down_w = (which == 0u ? down0 : (which == 1u ? down1 : down2));
  const device float* cat_w = cat + size_t(which) * p.T * (2u * p.H);
  device float* lat_w = lat + size_t(which) * p.T * p.R;
  const size_t row_off0 = size_t(r) * (2u * p.H), row_off1 = row_off0 + p.H;
  float a0[8] = {}, a1[8] = {};
  for (uint c = tid * 4u; c < p.H; c += 512u) {
    float4 dw0, dw1;
    if (p.w_f32) {
      dw0 = *(const device float4*)((const device float*)down_w + row_off0 + c);
      dw1 = *(const device float4*)((const device float*)down_w + row_off1 + c);
    } else {
      dw0 = ink2_bf16x4(*(const device uint2*)((const device bfloat*)down_w + row_off0 + c));
      dw1 = ink2_bf16x4(*(const device uint2*)((const device bfloat*)down_w + row_off1 + c));
    }
    for (uint m = 0u; m < mc; ++m) {
      const device float* row = cat_w + size_t(t0 + m) * (2u * p.H) + c;
      a0[m] += dot(*(const device float4*)(row), dw0);
      a1[m] += dot(*(const device float4*)(row + p.H), dw1);
    }
  }
  threadgroup float sh[4][8];
  for (uint m = 0u; m < mc; ++m) {
    const float s = simd_sum(a0[m] + a1[m]);
    if (lane == 0u) sh[sg][m] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < mc) {
    lat_w[size_t(t0 + tid) * p.R + r] = sh[0][tid] + sh[1][tid] + sh[2][tid] + sh[3][tid];
  }
}

[[kernel]] void ink2_msg_gate_up_mix3_s(const device float* h0 [[buffer(0)]], const device float* h1 [[buffer(1)]],
                                        const device float* h2 [[buffer(2)]], const device float* m0 [[buffer(3)]],
                                        const device float* m1 [[buffer(4)]], const device float* m2 [[buffer(5)]],
                                        const device float* lat [[buffer(6)]], const device void* forget0 [[buffer(7)]],
                                        const device void* forget1 [[buffer(8)]], const device void* forget2 [[buffer(9)]],
                                        const device void* gain0 [[buffer(10)]], const device void* gain1 [[buffer(11)]],
                                        const device void* gain2 [[buffer(12)]], device float* out0 [[buffer(13)]],
                                        device float* out1 [[buffer(14)]], device float* out2 [[buffer(15)]],
                                        constant Ink2MsgGateP& p [[buffer(16)]], uint3 tg [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sh_lat[8 * 64];
  const uint t0 = tg.y * 8u, which = tg.z, mc = min(8u, p.T - t0);
  const device float* lat_w = lat + size_t(which) * p.T * p.R;
  for (uint idx = tid; idx < mc * p.R; idx += 64u) sh_lat[idx] = lat_w[size_t(t0) * p.R + idx];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = tg.x * 64u + tid;
  if (c >= p.H) return;
  const device float* h = (which == 0u ? h0 : (which == 1u ? h1 : h2));
  const device float* m_buf = (which == 0u ? m0 : (which == 1u ? m1 : m2));
  const device void* forget_w = (which == 0u ? forget0 : (which == 1u ? forget1 : forget2));
  const device void* gain_w = (which == 0u ? gain0 : (which == 1u ? gain1 : gain2));
  device float* out = (which == 0u ? out0 : (which == 1u ? out1 : out2));
  float df[8] = {}, dg[8] = {};
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
  float vals[10];
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

[[kernel]] void ink2_bf16_mv_smalln(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                                    device float* y [[buffer(2)]], constant Ink2MmP& p [[buffer(3)]],
                                    uint n [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (n >= p.N) return;
  threadgroup float sh[4][8];
  float acc[8] = {};
  const size_t row_off = size_t(n) * p.K;
  for (uint k = tid * 4u; k < p.K; k += 512u) {
    const float4 wv = p.w_f32 ? *(const device float4*)((const device float*)w + row_off + k)
                              : ink2_bf16x4(*(const device uint2*)((const device bfloat*)w + row_off + k));
    for (uint m = 0u; m < p.M; ++m) {
      acc[m] += dot(*(const device float4*)(x + size_t(m) * p.x_ld + k), wv);
    }
  }
  for (uint m = 0u; m < p.M; ++m) {
    const float s = simd_sum(acc[m]);
    if (lane == 0u) sh[sg][m] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < p.M) {
    const float s = sh[0][tid] + sh[1][tid] + sh[2][tid] + sh[3][tid];
    device float* dst = y + size_t(tid) * p.y_ld + n;
    *dst = p.add ? (*dst + s) : s;
  }
}

struct Ink2Q8MmP {
  uint M, N, K, nblk, add;
  float cap, inv_cap;
};

struct Ink2Q8Multi6P {
  uint M, K, N_tot, n_segs, nblk;
  uint offs[7];
  uint lens[6];
};

template <uint RM>
NI_INLINE void ink2_q8_mv_impl(const device float* x, const device uchar* wq, const device half* ws,
                               device float* y, constant Ink2Q8MmP& p, uint tg, uint sg, uint lane) {
  const uint n0 = (tg * 4u + sg) * 2u;
  if (n0 >= p.N) return;
  const bool has1 = (n0 + 1u) < p.N;
  const size_t woff0 = size_t(n0) * p.K;
  const size_t woff1 = has1 ? (woff0 + p.K) : woff0;
  const size_t soff0 = size_t(n0) * p.nblk;
  const size_t soff1 = has1 ? (soff0 + p.nblk) : soff0;
  float acc0[RM] = {};
  float acc1[RM] = {};
  const uint kl = lane * 8u;
  const uint blk_loc = lane >> 2u;
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    if (kl < p.K - kbase) {
      const uint blk = (kbase >> 5u) + blk_loc;
      const uint2 qw0 = *(const device uint2*)(wq + woff0 + kbase + kl);
      const float sc0 = float(ws[soff0 + blk]);
      const float4 w0a = float4(as_type<char4>(qw0.x)) * sc0;
      const float4 w0b = float4(as_type<char4>(qw0.y)) * sc0;
      const uint2 qw1 = has1 ? *(const device uint2*)(wq + woff1 + kbase + kl) : qw0;
      const float sc1 = has1 ? float(ws[soff1 + blk]) : sc0;
      const float4 w1a = float4(as_type<char4>(qw1.x)) * sc1;
      const float4 w1b = float4(as_type<char4>(qw1.y)) * sc1;
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < RM; ++m) {
        if (RM == 8u && m >= p.M) continue;
        const device float* xp = x + size_t(m) * p.K + kbase + kl;
        const float4 xv0 = *(const device float4*)(xp + 0u);
        const float4 xv1 = *(const device float4*)(xp + 4u);
        acc0[m] += dot(xv0, w0a) + dot(xv1, w0b);
        acc1[m] += dot(xv0, w1a) + dot(xv1, w1b);
      }
    }
  }
  _Pragma("clang loop unroll(full)")
  for (uint m = 0u; m < RM; ++m) {
    if (RM == 8u && m >= p.M) break;
    const float s0 = simd_sum(acc0[m]);
    const float s1 = simd_sum(acc1[m]);
    if (lane == 0u) {
      device float* dst = y + size_t(m) * p.N + n0;
      float v0 = p.add ? (dst[0] + s0) : s0;
      if (p.cap > 0.0f) v0 = p.cap * metal::precise::tanh(v0 * p.inv_cap);
      dst[0] = v0;
      if (has1) {
        float v1 = p.add ? (dst[1] + s1) : s1;
        if (p.cap > 0.0f) v1 = p.cap * metal::precise::tanh(v1 * p.inv_cap);
        dst[1] = v1;
      }
    }
  }
}

[[kernel]] void ink2_q8_mv4(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                             const device half* ws [[buffer(2)]], device float* y [[buffer(3)]],
                             constant Ink2Q8MmP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  switch (p.M) {
    case 1u: ink2_q8_mv_impl<1u>(x, wq, ws, y, p, tg, sg, lane); break;
    case 2u: ink2_q8_mv_impl<2u>(x, wq, ws, y, p, tg, sg, lane); break;
    case 3u: ink2_q8_mv_impl<3u>(x, wq, ws, y, p, tg, sg, lane); break;
    default: ink2_q8_mv_impl<4u>(x, wq, ws, y, p, tg, sg, lane); break;
  }
}

[[kernel]] void ink2_q8_mv8(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                             const device half* ws [[buffer(2)]], device float* y [[buffer(3)]],
                             constant Ink2Q8MmP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  ink2_q8_mv_impl<8u>(x, wq, ws, y, p, tg, sg, lane);
}

template <uint RM>
NI_INLINE void ink2_q8_multi6_impl(const device float* x0, const device float* x1, const device float* x2,
                                   const device uchar* wq, const device half* ws, device float* y0, device float* y1,
                                   device float* y2, device float* y3, device float* y4, device float* y5,
                                   constant Ink2Q8Multi6P& p, uint tg, uint sg, uint lane) {
  const uint n0 = (tg * 4u + sg) * 2u;
  if (n0 >= p.N_tot) return;
  uint seg = 0u;
  while (seg + 1u < p.n_segs && n0 >= p.offs[seg + 1u]) ++seg;
  if (seg >= p.n_segs) return;
  const uint seg_off = p.offs[seg];
  const uint seg_len = p.lens[seg];
  if (n0 >= seg_off + seg_len) return;
  const bool has1 = (n0 + 1u) < (seg_off + seg_len);
  const device float* x = (seg < 4u) ? x0 : (seg == 4u ? x1 : x2);

  float acc0[RM] = {};
  float acc1[RM] = {};
  const size_t woff0 = size_t(n0) * p.K;
  const size_t woff1 = has1 ? (woff0 + p.K) : woff0;
  const size_t soff0 = size_t(n0) * p.nblk;
  const size_t soff1 = has1 ? (soff0 + p.nblk) : soff0;
  const uint kl = lane * 8u;
  const uint blk_loc = lane >> 2u;
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    if (kl < p.K - kbase) {
      const uint blk = (kbase >> 5u) + blk_loc;
      const uint2 qw0 = *(const device uint2*)(wq + woff0 + kbase + kl);
      const float sc0 = float(ws[soff0 + blk]);
      const float4 w0a = float4(as_type<char4>(qw0.x)) * sc0;
      const float4 w0b = float4(as_type<char4>(qw0.y)) * sc0;
      const uint2 qw1 = has1 ? *(const device uint2*)(wq + woff1 + kbase + kl) : qw0;
      const float sc1 = has1 ? float(ws[soff1 + blk]) : sc0;
      const float4 w1a = float4(as_type<char4>(qw1.x)) * sc1;
      const float4 w1b = float4(as_type<char4>(qw1.y)) * sc1;
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < RM; ++m) {
        if (RM == 8u && m >= p.M) continue;
        const device float* xp = x + size_t(m) * p.K + kbase + kl;
        const float4 xv0 = *(const device float4*)(xp + 0u);
        const float4 xv1 = *(const device float4*)(xp + 4u);
        acc0[m] += dot(xv0, w0a) + dot(xv1, w0b);
        acc1[m] += dot(xv0, w1a) + dot(xv1, w1b);
      }
    }
  }
  device float* bases[6] = {y0, y1, y2, y3, y4, y5};
  device float* dst_base = bases[seg] + (n0 - seg_off);
  _Pragma("clang loop unroll(full)")
  for (uint m = 0u; m < RM; ++m) {
    if (RM == 8u && m >= p.M) break;
    const float s0 = simd_sum(acc0[m]);
    const float s1 = simd_sum(acc1[m]);
    if (lane == 0u) {
      device float* dst = dst_base + size_t(m) * seg_len;
      dst[0] = s0;
      if (has1) dst[1] = s1;
    }
  }
}

[[kernel]] void ink2_q8_mv4_multi6(const device float* x0 [[buffer(0)]], const device float* x1 [[buffer(1)]],
                                    const device float* x2 [[buffer(2)]], const device uchar* wq [[buffer(3)]],
                                    const device half* ws [[buffer(4)]], device float* y0 [[buffer(5)]],
                                    device float* y1 [[buffer(6)]], device float* y2 [[buffer(7)]],
                                    device float* y3 [[buffer(8)]], device float* y4 [[buffer(9)]],
                                    device float* y5 [[buffer(10)]], constant Ink2Q8Multi6P& p [[buffer(11)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  switch (p.M) {
    case 1u: ink2_q8_multi6_impl<1u>(x0, x1, x2, wq, ws, y0, y1, y2, y3, y4, y5, p, tg, sg, lane); break;
    case 2u: ink2_q8_multi6_impl<2u>(x0, x1, x2, wq, ws, y0, y1, y2, y3, y4, y5, p, tg, sg, lane); break;
    case 3u: ink2_q8_multi6_impl<3u>(x0, x1, x2, wq, ws, y0, y1, y2, y3, y4, y5, p, tg, sg, lane); break;
    default: ink2_q8_multi6_impl<4u>(x0, x1, x2, wq, ws, y0, y1, y2, y3, y4, y5, p, tg, sg, lane); break;
  }
}

[[kernel]] void ink2_q8_mv8_multi6(const device float* x0 [[buffer(0)]], const device float* x1 [[buffer(1)]],
                                    const device float* x2 [[buffer(2)]], const device uchar* wq [[buffer(3)]],
                                    const device half* ws [[buffer(4)]], device float* y0 [[buffer(5)]],
                                    device float* y1 [[buffer(6)]], device float* y2 [[buffer(7)]],
                                    device float* y3 [[buffer(8)]], device float* y4 [[buffer(9)]],
                                    device float* y5 [[buffer(10)]], constant Ink2Q8Multi6P& p [[buffer(11)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  ink2_q8_multi6_impl<8u>(x0, x1, x2, wq, ws, y0, y1, y2, y3, y4, y5, p, tg, sg, lane);
}

struct Ink2QuantQ4P {
  uint total_groups;
};

struct Ink2Q4MmP {
  uint M, N, K, ngroups, add;
};

[[kernel]] void ink2_quant_q4g64_bf16(const device bfloat* w [[buffer(0)]], device uint* q [[buffer(1)]],
                                      device half2* sb [[buffer(2)]], constant Ink2QuantQ4P& p [[buffer(3)]],
                                      uint tg [[threadgroup_position_in_grid]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
  const uint g = tg * 4u + sg;
  if (g >= p.total_groups) return;
  const size_t base = size_t(g) * 64u + size_t(lane) * 2u;
  const float v0 = float(w[base + 0u]), v1 = float(w[base + 1u]);
  const float mn = simd_min(min(v0, v1));
  const float mx = simd_max(max(v0, v1));
  const half hsc = half((mx > mn) ? ((mx - mn) / 15.0f) : 0.0f);
  const half hmn = half(mn);
  const float fsc = float(hsc), fmn = float(hmn);
  const float inv = (fsc > 0.0f) ? (1.0f / fsc) : 0.0f;
  const uint q0 = uint(clamp(metal::rint((v0 - fmn) * inv), 0.0f, 15.0f));
  const uint q1 = uint(clamp(metal::rint((v1 - fmn) * inv), 0.0f, 15.0f));
  const uint b = q0 | (q1 << 4u);
  const uint b1 = simd_shuffle_down(b, 1u);
  const uint b2 = simd_shuffle_down(b, 2u);
  const uint b3 = simd_shuffle_down(b, 3u);
  if ((lane & 3u) == 0u) q[size_t(g) * 8u + (lane >> 2u)] = b | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
  if (lane == 0u) sb[g] = half2(hsc, hmn);
}

[[kernel]] void ink2_q4g64_mv4(const device float* x [[buffer(0)]], const device uint* wq [[buffer(1)]],
                                 const device half2* wsb [[buffer(2)]], device float* y [[buffer(3)]],
                                 constant Ink2Q4MmP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[4][256];
  const uint n0 = (tg * 4u + sg) * 4u;
  const bool valid = n0 < p.N;
  const uint mc = valid ? min(4u, p.N - n0) : 0u;
  float acc[4][4] = {};
  const size_t qrow_stride = size_t(p.ngroups) * 8u;
  const size_t qoff0 = size_t(n0) * qrow_stride;
  const size_t soff0 = size_t(n0) * p.ngroups;
  const uint total_v4 = p.M * 64u;
  const uint grp_loc = lane >> 3u;
  const uint wsub = lane & 7u;
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    for (uint idx = tid; idx < total_v4; idx += 128u) {
      const uint m = idx >> 6u;
      const uint c4 = (idx & 63u) << 2u;
      *(threadgroup float4*)(xs[m] + c4) = *(const device float4*)(x + size_t(m) * p.K + kbase + c4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
      const uint kl = lane * 8u;
      const uint grp = (kbase >> 6u) + grp_loc;
      for (uint q = 0u; q < mc; ++q) {
        const uint pk = wq[qoff0 + size_t(q) * qrow_stride + grp * 8u + wsub];
        const half2 sb = wsb[soff0 + size_t(q) * p.ngroups + grp];
        const float4 sc = float4(float(sb.x)), bi = float4(float(sb.y));
        const float4 wa = fma(float4(float(pk & 15u), float((pk >> 4) & 15u), float((pk >> 8) & 15u), float((pk >> 12) & 15u)),
                              sc, bi);
        const float4 wb = fma(float4(float((pk >> 16) & 15u), float((pk >> 20) & 15u), float((pk >> 24) & 15u), float(pk >> 28)),
                              sc, bi);
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

[[kernel]] void ink2_q4g64_mv8(const device float* x [[buffer(0)]], const device uint* wq [[buffer(1)]],
                                 const device half2* wsb [[buffer(2)]], device float* y [[buffer(3)]],
                                 constant Ink2Q4MmP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[8][256];
  const uint n0 = (tg * 4u + sg) * 4u;
  const bool valid = n0 < p.N;
  const uint mc = valid ? min(4u, p.N - n0) : 0u;
  float acc[4][8] = {};
  const size_t qrow_stride = size_t(p.ngroups) * 8u;
  const size_t qoff0 = size_t(n0) * qrow_stride;
  const size_t soff0 = size_t(n0) * p.ngroups;
  const uint total_v4 = p.M * 64u;
  const uint grp_loc = lane >> 3u;
  const uint wsub = lane & 7u;
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    for (uint idx = tid; idx < total_v4; idx += 128u) {
      const uint m = idx >> 6u;
      const uint c4 = (idx & 63u) << 2u;
      *(threadgroup float4*)(xs[m] + c4) = *(const device float4*)(x + size_t(m) * p.K + kbase + c4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
      const uint kl = lane * 8u;
      const uint grp = (kbase >> 6u) + grp_loc;
      for (uint q = 0u; q < mc; ++q) {
        const uint pk = wq[qoff0 + size_t(q) * qrow_stride + grp * 8u + wsub];
        const half2 sb = wsb[soff0 + size_t(q) * p.ngroups + grp];
        const float4 sc = float4(float(sb.x)), bi = float4(float(sb.y));
        const float4 wa = fma(float4(float(pk & 15u), float((pk >> 4) & 15u), float((pk >> 8) & 15u), float((pk >> 12) & 15u)),
                              sc, bi);
        const float4 wb = fma(float4(float((pk >> 16) & 15u), float((pk >> 20) & 15u), float((pk >> 24) & 15u), float(pk >> 28)),
                              sc, bi);
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
struct Ink2Tree {
  uint R;
  uint depth[16];
  uint anc_mask[16];
  int parent[16];
  int par1[16];
  int par2[16];
  int par3[16];
  int tokens[16];
  uint n_child[16];
  uint child[16][15];
};

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

[[kernel]] void ink2_sconv_static_tree(const device float* x [[buffer(0)]], const device void* w [[buffer(1)]],
                                        const device float* hist [[buffer(2)]], device float* out [[buffer(3)]],
                                        constant Ink2SconvStaticP& p [[buffer(4)]], device float* snap [[buffer(5)]],
                                        const device float* res [[buffer(6)]], const device Ink2Tree* tree [[buffer(7)]],
                                        uint c [[thread_position_in_grid]]) {
  if (c >= p.C) return;
  float w0, w1, w2, w3;
  if (p.w_f32) {
    const float4 wf = *(const device float4*)((const device float*)w + size_t(c) * 4u);
    w0 = wf.x; w1 = wf.y; w2 = wf.z; w3 = wf.w;
  } else {
    const device bfloat* wb = (const device bfloat*)w + size_t(c) * 4u;
    w0 = float(wb[0]); w1 = float(wb[1]); w2 = float(wb[2]); w3 = float(wb[3]);
  }
  const float h_init[3] = {hist[0 * p.C + c], hist[1 * p.C + c], hist[2 * p.C + c]};
  const uint R = min(p.T, 16u);
  float xv[16];
  for (uint r = 0u; r < R; ++r) xv[r] = x[size_t(r) * p.C + c];
  for (uint r = 0u; r < R; ++r) {
    const int i3 = tree->par3[r], i2 = tree->par2[r], i1 = tree->par1[r];
    const float h0 = (i3 >= 0) ? xv[i3] : h_init[i3 + 3];
    const float h1 = (i2 >= 0) ? xv[i2] : h_init[i2 + 3];
    const float h2 = (i1 >= 0) ? xv[i1] : h_init[i1 + 3];
    const float xt = xv[r];
    const float conv = w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt;
    const size_t idx = size_t(r) * p.C + c;
    float y = (conv + xt) * p.scale;
    if (p.add_res) y += res[idx];
    out[idx] = y;
    const size_t sb = size_t(r) * size_t(p.snap_stride) + c;
    snap[sb + 0 * p.C] = h1;
    snap[sb + 1 * p.C] = h2;
    snap[sb + 2 * p.C] = xt;
  }
}

[[kernel]] void ink2_sconv_dynamic_tree(const device float* x [[buffer(0)]], const device float* dyn [[buffer(1)]],
                                         const device void* static_w [[buffer(2)]], const device float* hist [[buffer(3)]],
                                         device float* out [[buffer(4)]], constant Ink2SconvDynP& p [[buffer(5)]],
                                         device float* snap [[buffer(6)]], const device Ink2Tree* tree [[buffer(7)]],
                                         uint c [[thread_position_in_grid]]) {
  if (c >= p.C) return;
  const uint head_dim = p.C / p.n_heads;
  const uint h = c / head_dim;
  float sw[4];
  for (uint lag = 0; lag < 4; ++lag) {
    const size_t idx = size_t(lag) * p.C + c;
    sw[lag] = p.sw_f32 ? ((const device float*)static_w)[idx] : float(((const device bfloat*)static_w)[idx]);
  }
  const float h_init[3] = {hist[0 * p.C + c], hist[1 * p.C + c], hist[2 * p.C + c]};
  const uint R = min(p.T, 16u);
  float xv[16];
  for (uint r = 0u; r < R; ++r) xv[r] = x[size_t(r) * p.C + c];
  for (uint r = 0u; r < R; ++r) {
    const int i3 = tree->par3[r], i2 = tree->par2[r], i1 = tree->par1[r];
    const float h0 = (i3 >= 0) ? xv[i3] : h_init[i3 + 3];
    const float h1 = (i2 >= 0) ? xv[i2] : h_init[i2 + 3];
    const float h2 = (i1 >= 0) ? xv[i1] : h_init[i1 + 3];
    const float xt = xv[r];
    const float4 d = *(const device float4*)(dyn + (size_t(r) * p.n_heads + h) * 4);
    const float conv = (sw[0] + d.x) * xt + (sw[1] + d.y) * h2 + (sw[2] + d.z) * h1 + (sw[3] + d.w) * h0;
    out[size_t(r) * p.C + c] = (conv + xt) * (1.0f / 3.0f);
    const size_t sb = size_t(r) * size_t(p.snap_stride) + c;
    snap[sb + 0 * p.C] = h1;
    snap[sb + 1 * p.C] = h2;
    snap[sb + 2 * p.C] = xt;
  }
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

[[kernel]] void ink2_qkvr_prep_tree(device float* q [[buffer(0)]], const device float* k [[buffer(1)]],
                                     const device float* v [[buffer(2)]], device float* r [[buffer(3)]],
                                     const device void* q_scale [[buffer(4)]], const device void* r_scale [[buffer(5)]],
                                     device void* kc [[buffer(6)]], device void* vc [[buffer(7)]],
                                     constant Ink2QkvrP& p [[buffer(8)]], const device Ink2Tree* tree [[buffer(9)]],
                                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  const uint kind_h = tg.x, t = tg.y;
  const uint slot_pos = p.pos0 + t;
  const uint slot = p.is_swa ? (slot_pos % p.kv_cap) : slot_pos;
  const uint qpos = p.pos0 + tree->depth[t];
  const float tau = p.is_swa ? 1.0f : (1.0f + p.log_alpha * metal::precise::log(max(1.0f, float(qpos + 1u) / p.log_floor)));

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

template <bool IsSwa>
NI_INLINE void ink2_attn_decode_gqa_impl(const device float* q, const device void* kc, const device void* vc,
                                         const device float* rb, device float* pm, device float* pl, device float* po,
                                         constant Ink2AttnDecP& p, uint2 tg, uint sg, uint lane,
                                         threadgroup float* tgm, threadgroup float* tgl, threadgroup float* tgo) {
  const uint hkv = tg.x, cid = tg.y;
  const uint h0 = hkv * 3u, d0 = lane * 4u;
  float4 qv[3], ow[3];
  float mw[3], lw[3];
  for (uint g = 0u; g < 3u; ++g) {
    qv[g] = *(const device float4*)(q + size_t(h0 + g) * p.D + d0);
    mw[g] = -INFINITY;
    lw[g] = 0.0f;
    ow[g] = float4(0.0f);
  }
  const uint lk = IsSwa ? min(p.pos + 1u, p.swa) : (p.pos + 1u);
  const uint k_first = p.pos + 1u - lk;
  const uint c0 = k_first + cid * p.chunk;
  const uint c1 = min(p.pos + 1u, c0 + p.chunk);
  for (uint kp = c0 + sg; kp < c1; kp += 4u) {
    const uint slot = IsSwa ? (kp % p.kv_cap) : kp;
    const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = ink2_load_kv4(kc, koff, p.kv_fp8);
    const float4 vv = ink2_load_kv4(vc, koff, p.kv_fp8);
    const uint dist = p.pos - kp;
    const bool has_rb = dist < p.rel_extent;
    for (uint g = 0u; g < 3u; ++g) {
      float sc = simd_sum(dot(qv[g], kv)) * p.scale;
      if (has_rb) sc += rb[size_t(h0 + g) * p.rel_extent + dist];
      if (lw[g] == 0.0f) {
        mw[g] = sc; lw[g] = 1.0f; ow[g] = vv;
      } else {
        const float nm = max(mw[g], sc);
        const float e1 = metal::precise::exp(mw[g] - nm), e2 = metal::precise::exp(sc - nm);
        ow[g] = ow[g] * e1 + vv * e2;
        lw[g] = lw[g] * e1 + e2;
        mw[g] = nm;
      }
    }
  }
  for (uint g = 0u; g < 3u; ++g) {
    if (lane == 0u) { tgm[sg] = mw[g]; tgl[sg] = lw[g]; }
    *(threadgroup float4*)(tgo + sg * 128u + d0) = ow[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      float mm = -INFINITY, ll = 0.0f;
      float4 oo = float4(0.0f);
      for (uint s = 0u; s < 4u; ++s) {
        const float l2 = tgl[s];
        if (l2 == 0.0f) continue;
        const float m2 = tgm[s];
        const float4 o2 = *(const threadgroup float4*)(tgo + s * 128u + d0);
        if (ll == 0.0f) { mm = m2; ll = l2; oo = o2; }
        else {
          const float nm = max(mm, m2);
          const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
          oo = oo * e1 + o2 * e2; ll = ll * e1 + l2 * e2; mm = nm;
        }
      }
      const size_t out_idx = size_t(h0 + g) * p.nchunks + cid;
      if (lane == 0u) { pm[out_idx] = mm; pl[out_idx] = ll; }
      *(device float4*)(po + out_idx * p.D + d0) = oo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

[[kernel]] void ink2_attn_decode_gqa_global(const device float* q [[buffer(0)]], const device void* kc [[buffer(1)]],
                                            const device void* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                            device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
                                            device float* po [[buffer(6)]], constant Ink2AttnDecP& p [[buffer(7)]],
                                            uint2 tg [[threadgroup_position_in_grid]],
                                            uint sg [[simdgroup_index_in_threadgroup]],
                                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgm[4], tgl[4], tgo[4 * 128];
  ink2_attn_decode_gqa_impl<false>(q, kc, vc, rb, pm, pl, po, p, tg, sg, lane, tgm, tgl, tgo);
}

[[kernel]] void ink2_attn_decode_gqa_swa(const device float* q [[buffer(0)]], const device void* kc [[buffer(1)]],
                                         const device void* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                         device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
                                         device float* po [[buffer(6)]], constant Ink2AttnDecP& p [[buffer(7)]],
                                         uint2 tg [[threadgroup_position_in_grid]],
                                         uint sg [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgm[4], tgl[4], tgo[4 * 128];
  ink2_attn_decode_gqa_impl<true>(q, kc, vc, rb, pm, pl, po, p, tg, sg, lane, tgm, tgl, tgo);
}

template <bool IsSwa>
NI_INLINE void ink2_attn_decode_1pass_gqa_impl(const device float* q, const device void* kc, const device void* vc,
                                               const device float* rb, const device float* sinks,
                                               const device float* gate, device float* out,
                                               constant Ink2AttnDecP& p, uint hkv, uint sg, uint lane,
                                               threadgroup float* tgm, threadgroup float* tgl,
                                               threadgroup float* tgo) {
  const uint h0 = hkv * 3u, d0 = lane * 4u;
  float4 qv[3], ow[3];
  float mw[3], lw[3];
  for (uint g = 0u; g < 3u; ++g) {
    qv[g] = *(const device float4*)(q + size_t(h0 + g) * p.D + d0);
    mw[g] = -INFINITY;
    lw[g] = 0.0f;
    ow[g] = float4(0.0f);
  }
  const uint lk = IsSwa ? min(p.pos + 1u, p.swa) : (p.pos + 1u);
  const uint k_first = p.pos + 1u - lk;
  for (uint kp = k_first + sg; kp <= p.pos; kp += 4u) {
    const uint slot = IsSwa ? (kp % p.kv_cap) : kp;
    const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = ink2_load_kv4(kc, koff, p.kv_fp8);
    const float4 vv = ink2_load_kv4(vc, koff, p.kv_fp8);
    const uint dist = p.pos - kp;
    const bool has_rb = dist < p.rel_extent;
    for (uint g = 0u; g < 3u; ++g) {
      float sc = simd_sum(dot(qv[g], kv)) * p.scale;
      if (has_rb) sc += rb[size_t(h0 + g) * p.rel_extent + dist];
      if (lw[g] == 0.0f) {
        mw[g] = sc; lw[g] = 1.0f; ow[g] = vv;
      } else {
        const float nm = max(mw[g], sc);
        const float e1 = metal::precise::exp(mw[g] - nm), e2 = metal::precise::exp(sc - nm);
        ow[g] = ow[g] * e1 + vv * e2;
        lw[g] = lw[g] * e1 + e2;
        mw[g] = nm;
      }
    }
  }
  for (uint g = 0u; g < 3u; ++g) {
    if (lane == 0u) { tgm[sg] = mw[g]; tgl[sg] = lw[g]; }
    *(threadgroup float4*)(tgo + sg * 128u + d0) = ow[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      const uint h = h0 + g;
      float mm = sinks[h], ll = 1.0f;
      float4 oo = float4(0.0f);
      for (uint s = 0u; s < 4u; ++s) {
        const float l2 = tgl[s];
        if (l2 == 0.0f) continue;
        const float m2 = tgm[s];
        const float4 o2 = *(const threadgroup float4*)(tgo + s * 128u + d0);
        const float nm = max(mm, m2);
        const float e1 = metal::precise::exp(mm - nm), e2 = metal::precise::exp(m2 - nm);
        oo = oo * e1 + o2 * e2; ll = ll * e1 + l2 * e2; mm = nm;
      }
      const float4 g4 = *(const device float4*)(gate + size_t(h) * p.D + d0);
      const float4 sig = float4(1.0f) / (float4(1.0f) + float4(metal::precise::exp(-g4.x), metal::precise::exp(-g4.y),
                                                               metal::precise::exp(-g4.z), metal::precise::exp(-g4.w)));
      *(device float4*)(out + size_t(h) * p.D + d0) = (oo / ll) * sig;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

[[kernel]] void ink2_attn_decode_1pass_gqa_global(const device float* q [[buffer(0)]], const device void* kc [[buffer(1)]],
                                                  const device void* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                                  const device float* sinks [[buffer(4)]], const device float* gate [[buffer(5)]],
                                                  device float* out [[buffer(6)]], constant Ink2AttnDecP& p [[buffer(7)]],
                                                  uint hkv [[threadgroup_position_in_grid]],
                                                  uint sg [[simdgroup_index_in_threadgroup]],
                                                  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgm[4], tgl[4], tgo[4 * 128];
  ink2_attn_decode_1pass_gqa_impl<false>(q, kc, vc, rb, sinks, gate, out, p, hkv, sg, lane, tgm, tgl, tgo);
}

[[kernel]] void ink2_attn_decode_1pass_gqa_swa(const device float* q [[buffer(0)]], const device void* kc [[buffer(1)]],
                                               const device void* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                               const device float* sinks [[buffer(4)]], const device float* gate [[buffer(5)]],
                                               device float* out [[buffer(6)]], constant Ink2AttnDecP& p [[buffer(7)]],
                                               uint hkv [[threadgroup_position_in_grid]],
                                               uint sg [[simdgroup_index_in_threadgroup]],
                                               uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgm[4], tgl[4], tgo[4 * 128];
  ink2_attn_decode_1pass_gqa_impl<true>(q, kc, vc, rb, sinks, gate, out, p, hkv, sg, lane, tgm, tgl, tgo);
}

[[kernel]] void ink2_attn_verify_gqa(const device float* q [[buffer(0)]], const device void* kc [[buffer(1)]],
                                     const device void* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                     device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
                                     device float* po [[buffer(6)]], constant Ink2AttnVerP& p [[buffer(7)]],
                                     uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half sh_k[8][128], sh_v[8][128];
  threadgroup float tgm[3][2], tgl[3][2], tgo[3][2][128];
  const uint hkv = tg.x, cid = tg.y, t0 = tg.z * 4u;
  const uint mc = min(4u, p.T - t0);
  const uint g = sg % 3u, w = sg / 3u;
  const uint h = hkv * 3u + g, d0 = lane * 4u;
  float4 qv[4], ow[4];
  float mw[4], lw[4];
  for (uint t = 0u; t < mc; ++t) {
    qv[t] = *(const device float4*)(q + (size_t(t0 + t) * p.Hq + h) * p.D + d0);
    mw[t] = -INFINITY;
    lw[t] = 0.0f;
    ow[t] = float4(0.0f);
  }
  const uint k_min = (p.is_swa && p.pos0 + 1u > p.swa) ? (p.pos0 + 1u - p.swa) : 0u;
  const uint k_max = p.pos0 + p.T;
  const uint c0 = k_min + cid * p.chunk;
  const uint c1 = min(k_max, min(c0 + p.chunk, p.pos0 + t0 + mc));
  for (uint kbase = c0; kbase < c1; kbase += 8u) {
    const uint nvalid = min(8u, c1 - kbase);
    for (uint idx = tid; idx < nvalid * 32u; idx += 192u) {
      const uint ik = idx >> 5u, c4 = (idx & 31u) << 2u;
      const uint kp = kbase + ik;
      const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
      const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + c4;
      if (p.kv_fp8) {
        *(threadgroup half4*)(sh_k[ik] + c4) = half4(ink2_load_kv4(kc, koff, 1u));
        *(threadgroup half4*)(sh_v[ik] + c4) = half4(ink2_load_kv4(vc, koff, 1u));
      } else {
        *(threadgroup half4*)(sh_k[ik] + c4) = *(const device half4*)((const device half*)kc + koff);
        *(threadgroup half4*)(sh_v[ik] + c4) = *(const device half4*)((const device half*)vc + koff);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = w; i < nvalid; i += 2u) {
      const uint kp = kbase + i;
      const float4 kv = float4(*(const threadgroup half4*)(sh_k[i] + d0));
      const float4 vv = float4(*(const threadgroup half4*)(sh_v[i] + d0));
      for (uint t = 0u; t < mc; ++t) {
        const uint qpos = p.pos0 + t0 + t;
        const uint t_min = (p.is_swa && qpos + 1u > p.swa) ? (qpos + 1u - p.swa) : 0u;
        if (kp < t_min || kp > qpos) continue;
        float sc = simd_sum(dot(qv[t], kv)) * p.scale;
        const uint dist = qpos - kp;
        if (dist < p.rel_extent) sc += rb[(size_t(t0 + t) * p.Hq + h) * p.rel_extent + dist];
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint t = 0u; t < mc; ++t) {
    if (lane == 0u) {
      tgm[g][w] = mw[t];
      tgl[g][w] = lw[t];
    }
    *(threadgroup float4*)(&tgo[g][w][d0]) = ow[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (w == 0u) {
      float mm = tgm[g][0], ll = tgl[g][0];
      float4 oo = *(const threadgroup float4*)(&tgo[g][0][d0]);
      const float l2 = tgl[g][1];
      if (l2 != 0.0f) {
        const float m2 = tgm[g][1];
        const float4 o2 = *(const threadgroup float4*)(&tgo[g][1][d0]);
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
      const size_t out_idx = (size_t(t0 + t) * p.Hq + h) * p.nchunks + cid;
      if (lane == 0u) {
        pm[out_idx] = mm;
        pl[out_idx] = ll;
      }
      *(device float4*)(po + out_idx * p.D + d0) = oo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

[[kernel]] void ink2_attn_verify_tree_gqa(const device float* q [[buffer(0)]], const device void* kc [[buffer(1)]],
                                          const device void* vc [[buffer(2)]], const device float* rb [[buffer(3)]],
                                          device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
                                          device float* po [[buffer(6)]], constant Ink2AttnVerP& p [[buffer(7)]],
                                          const device Ink2Tree* tree [[buffer(8)]],
                                          uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half sh_k[8][128], sh_v[8][128];
  threadgroup float tgm[3][2], tgl[3][2], tgo[3][2][128];
  threadgroup uint sh_depth[16];
  if (tid < 16u) sh_depth[tid] = tree->depth[tid];
  const uint hkv = tg.x, cid = tg.y, t0 = tg.z * 4u;
  const uint mc = min(4u, p.T - t0);
  const uint g = sg % 3u, w = sg / 3u;
  const uint h = hkv * 3u + g, d0 = lane * 4u;
  float4 qv[4], ow[4];
  float mw[4], lw[4];
  uint qdepth[4], qanc[4];
  for (uint t = 0u; t < mc; ++t) {
    qv[t] = *(const device float4*)(q + (size_t(t0 + t) * p.Hq + h) * p.D + d0);
    mw[t] = -INFINITY;
    lw[t] = 0.0f;
    ow[t] = float4(0.0f);
    qdepth[t] = tree->depth[t0 + t];
    qanc[t] = tree->anc_mask[t0 + t];
  }
  const uint k_min = (p.is_swa && p.pos0 + 1u > p.swa) ? (p.pos0 + 1u - p.swa) : 0u;
  const uint k_max = p.pos0 + p.T;
  const uint c0 = k_min + cid * p.chunk;
  const uint c1 = min(k_max, min(c0 + p.chunk, p.pos0 + t0 + mc));
  for (uint kbase = c0; kbase < c1; kbase += 8u) {
    const uint nvalid = min(8u, c1 - kbase);
    for (uint idx = tid; idx < nvalid * 32u; idx += 192u) {
      const uint ik = idx >> 5u, c4 = (idx & 31u) << 2u;
      const uint kp = kbase + ik;
      const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
      const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + c4;
      if (p.kv_fp8) {
        *(threadgroup half4*)(sh_k[ik] + c4) = half4(ink2_load_kv4(kc, koff, 1u));
        *(threadgroup half4*)(sh_v[ik] + c4) = half4(ink2_load_kv4(vc, koff, 1u));
      } else {
        *(threadgroup half4*)(sh_k[ik] + c4) = *(const device half4*)((const device half*)kc + koff);
        *(threadgroup half4*)(sh_v[ik] + c4) = *(const device half4*)((const device half*)vc + koff);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = w; i < nvalid; i += 2u) {
      const uint kp = kbase + i;
      const float4 kv = float4(*(const threadgroup half4*)(sh_k[i] + d0));
      const float4 vv = float4(*(const threadgroup half4*)(sh_v[i] + d0));
      for (uint t = 0u; t < mc; ++t) {
        uint dist;
        if (kp < p.pos0) {
          const uint qpos = p.pos0 + qdepth[t];
          const uint t_min = (p.is_swa && qpos + 1u > p.swa) ? (qpos + 1u - p.swa) : 0u;
          if (kp < t_min) continue;
          dist = qpos - kp;
        } else {
          const uint j = kp - p.pos0;
          if (((qanc[t] >> j) & 1u) == 0u) continue;
          dist = qdepth[t] - sh_depth[j];
        }
        float sc = simd_sum(dot(qv[t], kv)) * p.scale;
        if (dist < p.rel_extent) sc += rb[(size_t(t0 + t) * p.Hq + h) * p.rel_extent + dist];
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint t = 0u; t < mc; ++t) {
    if (lane == 0u) {
      tgm[g][w] = mw[t];
      tgl[g][w] = lw[t];
    }
    *(threadgroup float4*)(&tgo[g][w][d0]) = ow[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (w == 0u) {
      float mm = tgm[g][0], ll = tgl[g][0];
      float4 oo = *(const threadgroup float4*)(&tgo[g][0][d0]);
      const float l2 = tgl[g][1];
      if (l2 != 0.0f) {
        const float m2 = tgm[g][1];
        const float4 o2 = *(const threadgroup float4*)(&tgo[g][1][d0]);
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
      const size_t out_idx = (size_t(t0 + t) * p.Hq + h) * p.nchunks + cid;
      if (lane == 0u) {
        pm[out_idx] = mm;
        pl[out_idx] = ll;
      }
      *(device float4*)(po + out_idx * p.D + d0) = oo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
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
                                        uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint swarp = tid >> 3u, slane = lane & 7u;
  const uint j = tg.x * 16u + swarp;
  const bool j_ok = j < p.I;
  const uint slot = tg.y, t = tg.z;
  const uint e = eid[size_t(t) * p.K_top + slot];
  const uint two_i = 2u * p.I, half_h = p.H / 2u, blocks = p.H / 16u;
  const size_t wg_row = size_t(e) * two_i + 2u * j;
  const size_t wu_row = wg_row + 1u;
  const device float* xp_base = x + size_t(t) * p.H;

  float ag = 0.0f, au = 0.0f;
  for (uint kblk = 0u; kblk < blocks; kblk += 16u) {
    const uint blk0 = kblk + slane * 2u;
    const uint blk1 = blk0 + 1u;
    if (j_ok) {
      if (blk1 < blocks) {
        const device float* xp = xp_base + blk0 * 16u;
        const float4 x0 = *(const device float4*)(xp + 0u),  x1 = *(const device float4*)(xp + 4u);
        const float4 x2 = *(const device float4*)(xp + 8u),  x3 = *(const device float4*)(xp + 12u);
        const float4 x4 = *(const device float4*)(xp + 16u), x5 = *(const device float4*)(xp + 20u);
        const float4 x6 = *(const device float4*)(xp + 24u), x7 = *(const device float4*)(xp + 28u);
        auto dot2 = [&](size_t r) {
          const float sb0 = ink2_e4m3_16384(sc[r * blocks + blk0]);
          const float sb1 = ink2_e4m3_16384(sc[r * blocks + blk1]);
          const uint4 pw = *(const device uint4*)(w13 + r * half_h + blk0 * 8u);
          const float d0 = (dot(x0, ink2_e2m1_4_raw(pw.x)) + dot(x1, ink2_e2m1_4_raw(pw.x >> 16)) +
                            dot(x2, ink2_e2m1_4_raw(pw.y)) + dot(x3, ink2_e2m1_4_raw(pw.y >> 16))) * sb0;
          const float d1 = (dot(x4, ink2_e2m1_4_raw(pw.z)) + dot(x5, ink2_e2m1_4_raw(pw.z >> 16)) +
                            dot(x6, ink2_e2m1_4_raw(pw.w)) + dot(x7, ink2_e2m1_4_raw(pw.w >> 16))) * sb1;
          return d0 + d1;
        };
        ag += dot2(wg_row);
        au += dot2(wu_row);
      } else if (blk0 < blocks) {
        const device float* xp = xp_base + blk0 * 16u;
        const float4 x0 = *(const device float4*)(xp + 0u), x1 = *(const device float4*)(xp + 4u);
        const float4 x2 = *(const device float4*)(xp + 8u), x3 = *(const device float4*)(xp + 12u);
        auto dot1 = [&](size_t r) {
          const float sb0 = ink2_e4m3_16384(sc[r * blocks + blk0]);
          const uint2 pw = *(const device uint2*)(w13 + r * half_h + blk0 * 8u);
          return (dot(x0, ink2_e2m1_4_raw(pw.x)) + dot(x1, ink2_e2m1_4_raw(pw.x >> 16)) +
                  dot(x2, ink2_e2m1_4_raw(pw.y)) + dot(x3, ink2_e2m1_4_raw(pw.y >> 16))) * sb0;
        };
        ag += dot1(wg_row);
        au += dot1(wu_row);
      }
    }
  }
  if (j_ok) {
    ag += simd_shuffle_xor(ag, 4u); ag += simd_shuffle_xor(ag, 2u); ag += simd_shuffle_xor(ag, 1u);
    au += simd_shuffle_xor(au, 4u); au += simd_shuffle_xor(au, 2u); au += simd_shuffle_xor(au, 1u);
    if (slane == 0u) {
      const float os = outer[e];
      const float g = ag * os, u = au * os;
      act[(size_t(t) * p.K_top + slot) * p.I + j] =
          (4.0f * metal::precise::tanh(g * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g))) *
          (25.0f * metal::precise::tanh(u * 0.04f));
    }
  }
}

[[kernel]] void ink2_moe_nvfp4_w2_acc(const device float* act [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                      const device float* ew [[buffer(2)]], const device uchar* w2 [[buffer(3)]],
                                      const device uchar* sc [[buffer(4)]], const device float* outer [[buffer(5)]],
                                      device float* out [[buffer(6)]], constant Ink2MoeP& p [[buffer(7)]],
                                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
  const uint swarp = tid >> 4u, slane = lane & 15u;
  const uint n = tg.x * 8u + swarp;
  const bool n_ok = n < p.H;
  const uint t = tg.y, half_i = p.I / 2u, blocks = p.I / 16u;
  float total = 0.0f;
  for (uint slot = 0u; slot < p.K_top; ++slot) {
    const uint e = eid[size_t(t) * p.K_top + slot];
    const float wt = ew[size_t(t) * p.K_top + slot];
    const size_t wrow = size_t(e) * p.H + n;
    const device float* ap_base = act + (size_t(t) * p.K_top + slot) * p.I;
    float sum = 0.0f;
    for (uint kblk = 0u; kblk < blocks; kblk += 16u) {
      const uint blk = kblk + slane;
      if (n_ok && blk < blocks) {
        const device float* ap = ap_base + blk * 16u;
        const float sb = ink2_e4m3_16384(sc[wrow * blocks + blk]);
        const uint2 pw = *(const device uint2*)(w2 + wrow * half_i + blk * 8u);
        sum += (dot(*(const device float4*)(ap + 0u), ink2_e2m1_4_raw(pw.x)) +
                dot(*(const device float4*)(ap + 4u), ink2_e2m1_4_raw(pw.x >> 16)) +
                dot(*(const device float4*)(ap + 8u), ink2_e2m1_4_raw(pw.y)) +
                dot(*(const device float4*)(ap + 12u), ink2_e2m1_4_raw(pw.y >> 16))) *
               sb;
      }
    }
    if (n_ok) {
      sum += simd_shuffle_xor(sum, 8u);
      sum += simd_shuffle_xor(sum, 4u);
      sum += simd_shuffle_xor(sum, 2u);
      sum += simd_shuffle_xor(sum, 1u);
      total += wt * (sum * outer[e]);
    }
  }
  if (n_ok && slane == 0u) out[size_t(t) * p.H + n] = total;
}

// ---- 12. MoE MXFP8 (layers 20..23): F8_E4M3 weights + UE8M0 block-32 scales ----
[[kernel]] void ink2_moe_mxfp8_w13_situ(const device float* x [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                        const device uchar* w13 [[buffer(2)]], const device uchar* sc [[buffer(3)]],
                                        device float* act [[buffer(4)]], constant Ink2MoeP& p [[buffer(5)]],
                                        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
  const uint swarp = tid >> 3u, slane = lane & 7u;
  const uint j = tg.x * 16u + swarp;
  const bool j_ok = j < p.I;
  const uint slot = tg.y, t = tg.z;
  const uint e = eid[size_t(t) * p.K_top + slot];
  const uint two_i = 2u * p.I, blocks = p.H / 32u;
  const size_t wg_row = size_t(e) * two_i + 2u * j;
  const size_t wu_row = wg_row + 1u;
  const device float* xp_base = x + size_t(t) * p.H;

  threadgroup float sh_x[256];
  float ag = 0.0f, au = 0.0f;
  for (uint kblk = 0u; kblk < blocks; kblk += 8u) {
    const uint k_off = kblk * 32u;
    const uint c_v4 = min(256u, p.H - k_off) >> 2u;
    if (tid < c_v4) {
      *(threadgroup float4*)(sh_x + tid * 4u) = *(const device float4*)(xp_base + k_off + tid * 4u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint blk = kblk + slane;
    if (j_ok && blk < blocks) {
      const threadgroup float* xp = sh_x + slane * 32u;
      const float4 x0 = *(const threadgroup float4*)(xp + 0u), x1 = *(const threadgroup float4*)(xp + 4u);
      const float4 x2 = *(const threadgroup float4*)(xp + 8u), x3 = *(const threadgroup float4*)(xp + 12u);
      const float4 x4 = *(const threadgroup float4*)(xp + 16u), x5 = *(const threadgroup float4*)(xp + 20u);
      const float4 x6 = *(const threadgroup float4*)(xp + 24u), x7 = *(const threadgroup float4*)(xp + 28u);
      const uint kbase = blk * 32u;
      auto dot32 = [&](size_t r) {
        const float sb = ink2_ue8m0(sc[r * blocks + blk]);
        const uint4 wa = *(const device uint4*)(w13 + r * p.H + kbase + 0u);
        const uint4 wb = *(const device uint4*)(w13 + r * p.H + kbase + 16u);
        return (dot(x0, ink2_e4m3_4(wa.x)) + dot(x1, ink2_e4m3_4(wa.y)) +
                dot(x2, ink2_e4m3_4(wa.z)) + dot(x3, ink2_e4m3_4(wa.w)) +
                dot(x4, ink2_e4m3_4(wb.x)) + dot(x5, ink2_e4m3_4(wb.y)) +
                dot(x6, ink2_e4m3_4(wb.z)) + dot(x7, ink2_e4m3_4(wb.w))) * sb;
      };
      ag += dot32(wg_row);
      au += dot32(wu_row);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (j_ok) {
    ag += simd_shuffle_xor(ag, 4u); ag += simd_shuffle_xor(ag, 2u); ag += simd_shuffle_xor(ag, 1u);
    au += simd_shuffle_xor(au, 4u); au += simd_shuffle_xor(au, 2u); au += simd_shuffle_xor(au, 1u);
    if (slane == 0u) {
      act[(size_t(t) * p.K_top + slot) * p.I + j] =
          (4.0f * metal::precise::tanh(ag * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-ag))) *
          (25.0f * metal::precise::tanh(au * 0.04f));
    }
  }
}

[[kernel]] void ink2_moe_mxfp8_w2_acc(const device float* act [[buffer(0)]], const device uint* eid [[buffer(1)]],
                                      const device float* ew [[buffer(2)]], const device uchar* w2 [[buffer(3)]],
                                      const device uchar* sc [[buffer(4)]], device float* out [[buffer(5)]],
                                      constant Ink2MoeP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint swarp = tid >> 3u, slane = lane & 7u;
  const uint n = tg.x * 16u + swarp;
  const bool n_ok = n < p.H;
  const uint t = tg.y, blocks = p.I / 32u;
  threadgroup float sh_a[256];
  float total = 0.0f;
  for (uint slot = 0u; slot < p.K_top; ++slot) {
    const uint e = eid[size_t(t) * p.K_top + slot];
    const float wt = ew[size_t(t) * p.K_top + slot];
    const size_t wrow = size_t(e) * p.H + n;
    const device float* ap_base = act + (size_t(t) * p.K_top + slot) * p.I;
    float sum = 0.0f;
    for (uint kblk = 0u; kblk < blocks; kblk += 8u) {
      const uint k_off = kblk * 32u;
      const uint c_v4 = min(256u, p.I - k_off) >> 2u;
      if (tid < c_v4) {
        *(threadgroup float4*)(sh_a + tid * 4u) = *(const device float4*)(ap_base + k_off + tid * 4u);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint blk = kblk + slane;
      if (n_ok && blk < blocks) {
        const threadgroup float* ap = sh_a + slane * 32u;
        const uint kbase = blk * 32u;
        const float sb = ink2_ue8m0(sc[wrow * blocks + blk]);
        const uint4 wa = *(const device uint4*)(w2 + wrow * p.I + kbase + 0u);
        const uint4 wb = *(const device uint4*)(w2 + wrow * p.I + kbase + 16u);
        sum += (dot(*(const threadgroup float4*)(ap + 0u), ink2_e4m3_4(wa.x)) +
                dot(*(const threadgroup float4*)(ap + 4u), ink2_e4m3_4(wa.y)) +
                dot(*(const threadgroup float4*)(ap + 8u), ink2_e4m3_4(wa.z)) +
                dot(*(const threadgroup float4*)(ap + 12u), ink2_e4m3_4(wa.w)) +
                dot(*(const threadgroup float4*)(ap + 16u), ink2_e4m3_4(wb.x)) +
                dot(*(const threadgroup float4*)(ap + 20u), ink2_e4m3_4(wb.y)) +
                dot(*(const threadgroup float4*)(ap + 24u), ink2_e4m3_4(wb.z)) +
                dot(*(const threadgroup float4*)(ap + 28u), ink2_e4m3_4(wb.w))) *
               sb;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (n_ok) {
      sum += simd_shuffle_xor(sum, 4u);
      sum += simd_shuffle_xor(sum, 2u);
      sum += simd_shuffle_xor(sum, 1u);
      total += wt * sum;
    }
  }
  if (n_ok && slane == 0u) out[size_t(t) * p.H + n] = total;
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
  threadgroup uint sh_eid[128];
  threadgroup float sh_ew[128];
  const uint r_tot = p.T * p.K_top;
  if (r_tot <= 128u && tid < r_tot) {
    sh_eid[tid] = eid[tid];
    sh_ew[tid] = ew[tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint count = 0;
  if (tid < p.E) {
    if (r_tot <= 128u) {
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
    if (r_tot <= 128u) {
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
                                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint eidx = tg.y;
  if (eidx >= ntiles[0]) return;
  const uint swarp = tid >> 3u, slane = lane & 7u;
  const uint j = tg.x * 16u + swarp;
  const bool j_ok = j < p.I;
  const uint e = tiles[eidx * 3 + 0];
  const uint start = tiles[eidx * 3 + 1];
  const uint count = tiles[eidx * 3 + 2];
  const uint two_i = 2u * p.I;
  const uint half_h = p.H / 2u;
  const uint blocks = p.H / 16u;
  const size_t wg_row = size_t(e) * two_i + 2u * j;
  const size_t wu_row = wg_row + 1u;
  const float os = outer[e];

  auto situ = [](float g, float u) {
    return (4.0f * metal::precise::tanh(g * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g))) *
           (25.0f * metal::precise::tanh(u * 0.04f));
  };

  if (count == 1u) {
    const uint tok0 = st_tok[start], pair0 = st_pair[start];
    const device float* xp0 = x + size_t(tok0) * p.H;
    float ag0 = 0.0f, au0 = 0.0f;
    for (uint kblk = 0u; kblk < blocks; kblk += 16u) {
      const uint blk0 = kblk + slane * 2u, blk1 = blk0 + 1u;
      if (j_ok) {
        if (blk1 < blocks) {
          const device float* xp = xp0 + blk0 * 16u;
          const float4 x0 = *(const device float4*)(xp + 0u),  x1 = *(const device float4*)(xp + 4u);
          const float4 x2 = *(const device float4*)(xp + 8u),  x3 = *(const device float4*)(xp + 12u);
          const float4 x4 = *(const device float4*)(xp + 16u), x5 = *(const device float4*)(xp + 20u);
          const float4 x6 = *(const device float4*)(xp + 24u), x7 = *(const device float4*)(xp + 28u);
          auto d2 = [&](size_t r) {
            const float sb0 = ink2_e4m3_16384(sc[r * blocks + blk0]);
            const float sb1 = ink2_e4m3_16384(sc[r * blocks + blk1]);
            const uint4 pw = *(const device uint4*)(w13 + r * half_h + blk0 * 8u);
            return (dot(x0, ink2_e2m1_4_raw(pw.x)) + dot(x1, ink2_e2m1_4_raw(pw.x >> 16)) +
                    dot(x2, ink2_e2m1_4_raw(pw.y)) + dot(x3, ink2_e2m1_4_raw(pw.y >> 16))) * sb0 +
                   (dot(x4, ink2_e2m1_4_raw(pw.z)) + dot(x5, ink2_e2m1_4_raw(pw.z >> 16)) +
                    dot(x6, ink2_e2m1_4_raw(pw.w)) + dot(x7, ink2_e2m1_4_raw(pw.w >> 16))) * sb1;
          };
          ag0 += d2(wg_row);
          au0 += d2(wu_row);
        } else if (blk0 < blocks) {
          const device float* xp = xp0 + blk0 * 16u;
          const float4 x0 = *(const device float4*)(xp + 0u), x1 = *(const device float4*)(xp + 4u);
          const float4 x2 = *(const device float4*)(xp + 8u), x3 = *(const device float4*)(xp + 12u);
          auto d1 = [&](size_t r) {
            const float sb0 = ink2_e4m3_16384(sc[r * blocks + blk0]);
            const uint2 pw = *(const device uint2*)(w13 + r * half_h + blk0 * 8u);
            return (dot(x0, ink2_e2m1_4_raw(pw.x)) + dot(x1, ink2_e2m1_4_raw(pw.x >> 16)) +
                    dot(x2, ink2_e2m1_4_raw(pw.y)) + dot(x3, ink2_e2m1_4_raw(pw.y >> 16))) * sb0;
          };
          ag0 += d1(wg_row);
          au0 += d1(wu_row);
        }
      }
    }
    if (j_ok) {
      ag0 += simd_shuffle_xor(ag0, 4u); ag0 += simd_shuffle_xor(ag0, 2u); ag0 += simd_shuffle_xor(ag0, 1u);
      au0 += simd_shuffle_xor(au0, 4u); au0 += simd_shuffle_xor(au0, 2u); au0 += simd_shuffle_xor(au0, 1u);
      if (slane == 0u) act[size_t(pair0) * p.I + j] = situ(ag0 * os, au0 * os);
    }
    return;
  }

  if (count == 2u) {
    const uint tok0 = st_tok[start], tok1 = st_tok[start + 1u];
    const uint pair0 = st_pair[start], pair1 = st_pair[start + 1u];
    const device float* xp0 = x + size_t(tok0) * p.H;
    const device float* xp1 = x + size_t(tok1) * p.H;
    float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;
    for (uint kblk = 0u; kblk < blocks; kblk += 16u) {
      const uint blk0 = kblk + slane * 2u, blk1 = blk0 + 1u;
      if (j_ok) {
        if (blk1 < blocks) {
          const uint xoff = blk0 * 16u;
          auto d2 = [&](size_t r, thread float& a0, thread float& a1) {
            const float sb0 = ink2_e4m3_16384(sc[r * blocks + blk0]);
            const float sb1 = ink2_e4m3_16384(sc[r * blocks + blk1]);
            const uint4 pw = *(const device uint4*)(w13 + r * half_h + blk0 * 8u);
            {
              const float4 w0 = ink2_e2m1_4_raw(pw.x), w1 = ink2_e2m1_4_raw(pw.x >> 16);
              const float4 w2 = ink2_e2m1_4_raw(pw.y), w3 = ink2_e2m1_4_raw(pw.y >> 16);
              const device float* p0x = xp0 + xoff;
              const device float* p1x = xp1 + xoff;
              a0 += (dot(*(const device float4*)(p0x + 0u), w0) + dot(*(const device float4*)(p0x + 4u), w1) +
                     dot(*(const device float4*)(p0x + 8u), w2) + dot(*(const device float4*)(p0x + 12u), w3)) * sb0;
              a1 += (dot(*(const device float4*)(p1x + 0u), w0) + dot(*(const device float4*)(p1x + 4u), w1) +
                     dot(*(const device float4*)(p1x + 8u), w2) + dot(*(const device float4*)(p1x + 12u), w3)) * sb0;
            }
            {
              const float4 w0 = ink2_e2m1_4_raw(pw.z), w1 = ink2_e2m1_4_raw(pw.z >> 16);
              const float4 w2 = ink2_e2m1_4_raw(pw.w), w3 = ink2_e2m1_4_raw(pw.w >> 16);
              const device float* p0x = xp0 + xoff + 16u;
              const device float* p1x = xp1 + xoff + 16u;
              a0 += (dot(*(const device float4*)(p0x + 0u), w0) + dot(*(const device float4*)(p0x + 4u), w1) +
                     dot(*(const device float4*)(p0x + 8u), w2) + dot(*(const device float4*)(p0x + 12u), w3)) * sb1;
              a1 += (dot(*(const device float4*)(p1x + 0u), w0) + dot(*(const device float4*)(p1x + 4u), w1) +
                     dot(*(const device float4*)(p1x + 8u), w2) + dot(*(const device float4*)(p1x + 12u), w3)) * sb1;
            }
          };
          d2(wg_row, ag0, ag1);
          d2(wu_row, au0, au1);
        } else if (blk0 < blocks) {
          const uint xoff = blk0 * 16u;
          auto d1 = [&](size_t r, thread float& a0, thread float& a1) {
            const float sb0 = ink2_e4m3_16384(sc[r * blocks + blk0]);
            const uint2 pw = *(const device uint2*)(w13 + r * half_h + blk0 * 8u);
            const float4 w0 = ink2_e2m1_4_raw(pw.x), w1 = ink2_e2m1_4_raw(pw.x >> 16);
            const float4 w2 = ink2_e2m1_4_raw(pw.y), w3 = ink2_e2m1_4_raw(pw.y >> 16);
            const device float* p0x = xp0 + xoff;
            const device float* p1x = xp1 + xoff;
            a0 += (dot(*(const device float4*)(p0x + 0u), w0) + dot(*(const device float4*)(p0x + 4u), w1) +
                   dot(*(const device float4*)(p0x + 8u), w2) + dot(*(const device float4*)(p0x + 12u), w3)) * sb0;
            a1 += (dot(*(const device float4*)(p1x + 0u), w0) + dot(*(const device float4*)(p1x + 4u), w1) +
                   dot(*(const device float4*)(p1x + 8u), w2) + dot(*(const device float4*)(p1x + 12u), w3)) * sb0;
          };
          d1(wg_row, ag0, ag1);
          d1(wu_row, au0, au1);
        }
      }
    }
    if (j_ok) {
      ag0 += simd_shuffle_xor(ag0, 4u); ag0 += simd_shuffle_xor(ag0, 2u); ag0 += simd_shuffle_xor(ag0, 1u);
      au0 += simd_shuffle_xor(au0, 4u); au0 += simd_shuffle_xor(au0, 2u); au0 += simd_shuffle_xor(au0, 1u);
      ag1 += simd_shuffle_xor(ag1, 4u); ag1 += simd_shuffle_xor(ag1, 2u); ag1 += simd_shuffle_xor(ag1, 1u);
      au1 += simd_shuffle_xor(au1, 4u); au1 += simd_shuffle_xor(au1, 2u); au1 += simd_shuffle_xor(au1, 1u);
      if (slane == 0u) {
        act[size_t(pair0) * p.I + j] = situ(ag0 * os, au0 * os);
        act[size_t(pair1) * p.I + j] = situ(ag1 * os, au1 * os);
      }
    }
    return;
  }

  for (uint p0 = 0u; p0 < count; p0 += 4u) {
    const uint mc = min(4u, count - p0);
    uint tok[4], pair[4];
    _Pragma("clang loop unroll(full)")
    for (uint m = 0u; m < 4u; ++m) {
      const uint mm = (m < mc) ? m : 0u;
      tok[m] = st_tok[start + p0 + mm];
      pair[m] = st_pair[start + p0 + mm];
    }
    float ag[4] = {}, au[4] = {};
    for (uint kblk = 0u; kblk < blocks; kblk += 16u) {
      const uint blk0 = kblk + slane * 2u, blk1 = blk0 + 1u;
      if (j_ok) {
        if (blk1 < blocks) {
          const uint xoff = blk0 * 16u;
          auto d4_2 = [&](size_t r, thread float* acc) {
            const float sb0 = ink2_e4m3_16384(sc[r * blocks + blk0]);
            const float sb1 = ink2_e4m3_16384(sc[r * blocks + blk1]);
            const uint4 pw = *(const device uint4*)(w13 + r * half_h + blk0 * 8u);
            {
              const float4 w0 = ink2_e2m1_4_raw(pw.x), w1 = ink2_e2m1_4_raw(pw.x >> 16);
              const float4 w2 = ink2_e2m1_4_raw(pw.y), w3 = ink2_e2m1_4_raw(pw.y >> 16);
              _Pragma("clang loop unroll(full)")
              for (uint m = 0u; m < 4u; ++m) {
                if (m < mc) {
                  const device float* xp = x + size_t(tok[m]) * p.H + xoff;
                  acc[m] += (dot(*(const device float4*)(xp + 0u), w0) + dot(*(const device float4*)(xp + 4u), w1) +
                             dot(*(const device float4*)(xp + 8u), w2) + dot(*(const device float4*)(xp + 12u), w3)) * sb0;
                }
              }
            }
            {
              const float4 w0 = ink2_e2m1_4_raw(pw.z), w1 = ink2_e2m1_4_raw(pw.z >> 16);
              const float4 w2 = ink2_e2m1_4_raw(pw.w), w3 = ink2_e2m1_4_raw(pw.w >> 16);
              _Pragma("clang loop unroll(full)")
              for (uint m = 0u; m < 4u; ++m) {
                if (m < mc) {
                  const device float* xp = x + size_t(tok[m]) * p.H + xoff + 16u;
                  acc[m] += (dot(*(const device float4*)(xp + 0u), w0) + dot(*(const device float4*)(xp + 4u), w1) +
                             dot(*(const device float4*)(xp + 8u), w2) + dot(*(const device float4*)(xp + 12u), w3)) * sb1;
                }
              }
            }
          };
          d4_2(wg_row, ag);
          d4_2(wu_row, au);
        } else if (blk0 < blocks) {
          const uint xoff = blk0 * 16u;
          auto d4_1 = [&](size_t r, thread float* acc) {
            const float sb0 = ink2_e4m3_16384(sc[r * blocks + blk0]);
            const uint2 pw = *(const device uint2*)(w13 + r * half_h + blk0 * 8u);
            const float4 w0 = ink2_e2m1_4_raw(pw.x), w1 = ink2_e2m1_4_raw(pw.x >> 16);
            const float4 w2 = ink2_e2m1_4_raw(pw.y), w3 = ink2_e2m1_4_raw(pw.y >> 16);
            _Pragma("clang loop unroll(full)")
            for (uint m = 0u; m < 4u; ++m) {
              if (m < mc) {
                const device float* xp = x + size_t(tok[m]) * p.H + xoff;
                acc[m] += (dot(*(const device float4*)(xp + 0u), w0) + dot(*(const device float4*)(xp + 4u), w1) +
                           dot(*(const device float4*)(xp + 8u), w2) + dot(*(const device float4*)(xp + 12u), w3)) * sb0;
              }
            }
          };
          d4_1(wg_row, ag);
          d4_1(wu_row, au);
        }
      }
    }
    if (j_ok) {
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < 4u; ++m) {
        if (m < mc) {
          ag[m] += simd_shuffle_xor(ag[m], 4u); ag[m] += simd_shuffle_xor(ag[m], 2u); ag[m] += simd_shuffle_xor(ag[m], 1u);
          au[m] += simd_shuffle_xor(au[m], 4u); au[m] += simd_shuffle_xor(au[m], 2u); au[m] += simd_shuffle_xor(au[m], 1u);
          if (slane == 0u) act[size_t(pair[m]) * p.I + j] = situ(ag[m] * os, au[m] * os);
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
                                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
  const uint eidx = tg.y;
  if (eidx >= ntiles[0]) return;
  const uint swarp = tid >> 3u, slane = lane & 7u;
  const uint n = tg.x * 16u + swarp;
  const bool n_ok = n < p.H;
  const uint e = tiles[eidx * 3 + 0];
  const uint start = tiles[eidx * 3 + 1];
  const uint count = tiles[eidx * 3 + 2];
  const uint half_i = p.I / 2u;
  const uint blocks = p.I / 16u;
  const size_t wrow = size_t(e) * p.H + n;
  const float os = outer[e];

  if (count == 1u) {
    const uint pair0 = st_pair[start];
    const device float* ap0 = act + size_t(pair0) * p.I;
    float sum0 = 0.0f;
    for (uint kblk = 0u; kblk < blocks; kblk += 16u) {
      const uint blk0 = kblk + slane * 2u, blk1 = blk0 + 1u;
      if (n_ok) {
        if (blk1 < blocks) {
          const float sb0 = ink2_e4m3_16384(sc[wrow * blocks + blk0]);
          const float sb1 = ink2_e4m3_16384(sc[wrow * blocks + blk1]);
          const uint4 pw = *(const device uint4*)(w2 + wrow * half_i + blk0 * 8u);
          const device float* ap = ap0 + blk0 * 16u;
          sum0 += (dot(*(const device float4*)(ap + 0u), ink2_e2m1_4_raw(pw.x)) +
                   dot(*(const device float4*)(ap + 4u), ink2_e2m1_4_raw(pw.x >> 16)) +
                   dot(*(const device float4*)(ap + 8u), ink2_e2m1_4_raw(pw.y)) +
                   dot(*(const device float4*)(ap + 12u), ink2_e2m1_4_raw(pw.y >> 16))) * sb0 +
                  (dot(*(const device float4*)(ap + 16u), ink2_e2m1_4_raw(pw.z)) +
                   dot(*(const device float4*)(ap + 20u), ink2_e2m1_4_raw(pw.z >> 16)) +
                   dot(*(const device float4*)(ap + 24u), ink2_e2m1_4_raw(pw.w)) +
                   dot(*(const device float4*)(ap + 28u), ink2_e2m1_4_raw(pw.w >> 16))) * sb1;
        } else if (blk0 < blocks) {
          const float sb0 = ink2_e4m3_16384(sc[wrow * blocks + blk0]);
          const uint2 pw = *(const device uint2*)(w2 + wrow * half_i + blk0 * 8u);
          const device float* ap = ap0 + blk0 * 16u;
          sum0 += (dot(*(const device float4*)(ap + 0u), ink2_e2m1_4_raw(pw.x)) +
                   dot(*(const device float4*)(ap + 4u), ink2_e2m1_4_raw(pw.x >> 16)) +
                   dot(*(const device float4*)(ap + 8u), ink2_e2m1_4_raw(pw.y)) +
                   dot(*(const device float4*)(ap + 12u), ink2_e2m1_4_raw(pw.y >> 16))) * sb0;
        }
      }
    }
    if (n_ok) {
      sum0 += simd_shuffle_xor(sum0, 4u); sum0 += simd_shuffle_xor(sum0, 2u); sum0 += simd_shuffle_xor(sum0, 1u);
      if (slane == 0u) pair_out[size_t(pair0) * p.H + n] = (st_ew[start] * os) * sum0;
    }
    return;
  }

  if (count == 2u) {
    const uint pair0 = st_pair[start], pair1 = st_pair[start + 1u];
    const device float* ap0 = act + size_t(pair0) * p.I;
    const device float* ap1 = act + size_t(pair1) * p.I;
    float sum0 = 0.0f, sum1 = 0.0f;
    for (uint kblk = 0u; kblk < blocks; kblk += 16u) {
      const uint blk0 = kblk + slane * 2u, blk1 = blk0 + 1u;
      if (n_ok) {
        if (blk1 < blocks) {
          const float sb0 = ink2_e4m3_16384(sc[wrow * blocks + blk0]);
          const float sb1 = ink2_e4m3_16384(sc[wrow * blocks + blk1]);
          const uint4 pw = *(const device uint4*)(w2 + wrow * half_i + blk0 * 8u);
          const uint aoff = blk0 * 16u;
          {
            const float4 w0 = ink2_e2m1_4_raw(pw.x), w1 = ink2_e2m1_4_raw(pw.x >> 16);
            const float4 w2v = ink2_e2m1_4_raw(pw.y), w3 = ink2_e2m1_4_raw(pw.y >> 16);
            const device float* p0a = ap0 + aoff;
            const device float* p1a = ap1 + aoff;
            sum0 += (dot(*(const device float4*)(p0a + 0u), w0) + dot(*(const device float4*)(p0a + 4u), w1) +
                     dot(*(const device float4*)(p0a + 8u), w2v) + dot(*(const device float4*)(p0a + 12u), w3)) * sb0;
            sum1 += (dot(*(const device float4*)(p1a + 0u), w0) + dot(*(const device float4*)(p1a + 4u), w1) +
                     dot(*(const device float4*)(p1a + 8u), w2v) + dot(*(const device float4*)(p1a + 12u), w3)) * sb0;
          }
          {
            const float4 w0 = ink2_e2m1_4_raw(pw.z), w1 = ink2_e2m1_4_raw(pw.z >> 16);
            const float4 w2v = ink2_e2m1_4_raw(pw.w), w3 = ink2_e2m1_4_raw(pw.w >> 16);
            const device float* p0a = ap0 + aoff + 16u;
            const device float* p1a = ap1 + aoff + 16u;
            sum0 += (dot(*(const device float4*)(p0a + 0u), w0) + dot(*(const device float4*)(p0a + 4u), w1) +
                     dot(*(const device float4*)(p0a + 8u), w2v) + dot(*(const device float4*)(p0a + 12u), w3)) * sb1;
            sum1 += (dot(*(const device float4*)(p1a + 0u), w0) + dot(*(const device float4*)(p1a + 4u), w1) +
                     dot(*(const device float4*)(p1a + 8u), w2v) + dot(*(const device float4*)(p1a + 12u), w3)) * sb1;
          }
        } else if (blk0 < blocks) {
          const float sb0 = ink2_e4m3_16384(sc[wrow * blocks + blk0]);
          const uint2 pw = *(const device uint2*)(w2 + wrow * half_i + blk0 * 8u);
          const uint aoff = blk0 * 16u;
          const float4 w0 = ink2_e2m1_4_raw(pw.x), w1 = ink2_e2m1_4_raw(pw.x >> 16);
          const float4 w2v = ink2_e2m1_4_raw(pw.y), w3 = ink2_e2m1_4_raw(pw.y >> 16);
          const device float* p0a = ap0 + aoff;
          const device float* p1a = ap1 + aoff;
          sum0 += (dot(*(const device float4*)(p0a + 0u), w0) + dot(*(const device float4*)(p0a + 4u), w1) +
                   dot(*(const device float4*)(p0a + 8u), w2v) + dot(*(const device float4*)(p0a + 12u), w3)) * sb0;
          sum1 += (dot(*(const device float4*)(p1a + 0u), w0) + dot(*(const device float4*)(p1a + 4u), w1) +
                   dot(*(const device float4*)(p1a + 8u), w2v) + dot(*(const device float4*)(p1a + 12u), w3)) * sb0;
        }
      }
    }
    if (n_ok) {
      sum0 += simd_shuffle_xor(sum0, 4u); sum0 += simd_shuffle_xor(sum0, 2u); sum0 += simd_shuffle_xor(sum0, 1u);
      sum1 += simd_shuffle_xor(sum1, 4u); sum1 += simd_shuffle_xor(sum1, 2u); sum1 += simd_shuffle_xor(sum1, 1u);
      if (slane == 0u) {
        pair_out[size_t(pair0) * p.H + n] = (st_ew[start] * os) * sum0;
        pair_out[size_t(pair1) * p.H + n] = (st_ew[start + 1u] * os) * sum1;
      }
    }
    return;
  }

  for (uint p0 = 0u; p0 < count; p0 += 8u) {
    const uint mc = min(8u, count - p0);
    uint pair[8];
    _Pragma("clang loop unroll(full)")
    for (uint m = 0u; m < 8u; ++m) pair[m] = st_pair[start + p0 + ((m < mc) ? m : 0u)];
    float sum[8] = {};
    for (uint kblk = 0u; kblk < blocks; kblk += 16u) {
      const uint blk0 = kblk + slane * 2u, blk1 = blk0 + 1u;
      if (n_ok) {
        if (blk1 < blocks) {
          const float sb0 = ink2_e4m3_16384(sc[wrow * blocks + blk0]);
          const float sb1 = ink2_e4m3_16384(sc[wrow * blocks + blk1]);
          const uint4 pw = *(const device uint4*)(w2 + wrow * half_i + blk0 * 8u);
          const uint aoff = blk0 * 16u;
          {
            const float4 w0 = ink2_e2m1_4_raw(pw.x), w1 = ink2_e2m1_4_raw(pw.x >> 16);
            const float4 w2v = ink2_e2m1_4_raw(pw.y), w3 = ink2_e2m1_4_raw(pw.y >> 16);
            _Pragma("clang loop unroll(full)")
            for (uint m = 0u; m < 8u; ++m) {
              if (m < mc) {
                const device float* ap = act + size_t(pair[m]) * p.I + aoff;
                sum[m] += (dot(*(const device float4*)(ap + 0u), w0) + dot(*(const device float4*)(ap + 4u), w1) +
                           dot(*(const device float4*)(ap + 8u), w2v) + dot(*(const device float4*)(ap + 12u), w3)) * sb0;
              }
            }
          }
          {
            const float4 w0 = ink2_e2m1_4_raw(pw.z), w1 = ink2_e2m1_4_raw(pw.z >> 16);
            const float4 w2v = ink2_e2m1_4_raw(pw.w), w3 = ink2_e2m1_4_raw(pw.w >> 16);
            _Pragma("clang loop unroll(full)")
            for (uint m = 0u; m < 8u; ++m) {
              if (m < mc) {
                const device float* ap = act + size_t(pair[m]) * p.I + aoff + 16u;
                sum[m] += (dot(*(const device float4*)(ap + 0u), w0) + dot(*(const device float4*)(ap + 4u), w1) +
                           dot(*(const device float4*)(ap + 8u), w2v) + dot(*(const device float4*)(ap + 12u), w3)) * sb1;
              }
            }
          }
        } else if (blk0 < blocks) {
          const float sb0 = ink2_e4m3_16384(sc[wrow * blocks + blk0]);
          const uint2 pw = *(const device uint2*)(w2 + wrow * half_i + blk0 * 8u);
          const uint aoff = blk0 * 16u;
          const float4 w0 = ink2_e2m1_4_raw(pw.x), w1 = ink2_e2m1_4_raw(pw.x >> 16);
          const float4 w2v = ink2_e2m1_4_raw(pw.y), w3 = ink2_e2m1_4_raw(pw.y >> 16);
          _Pragma("clang loop unroll(full)")
          for (uint m = 0u; m < 8u; ++m) {
            if (m < mc) {
              const device float* ap = act + size_t(pair[m]) * p.I + aoff;
              sum[m] += (dot(*(const device float4*)(ap + 0u), w0) + dot(*(const device float4*)(ap + 4u), w1) +
                         dot(*(const device float4*)(ap + 8u), w2v) + dot(*(const device float4*)(ap + 12u), w3)) * sb0;
            }
          }
        }
      }
    }
    if (n_ok) {
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < 8u; ++m) {
        if (m < mc) {
          sum[m] += simd_shuffle_xor(sum[m], 4u); sum[m] += simd_shuffle_xor(sum[m], 2u); sum[m] += simd_shuffle_xor(sum[m], 1u);
          if (slane == 0u) pair_out[size_t(pair[m]) * p.H + n] = (st_ew[start + p0 + m] * os) * sum[m];
        }
      }
    }
  }
}

[[kernel]] void ink2_moe_mxfp8_w13_g(const device float* x [[buffer(0)]], const device uchar* w13 [[buffer(1)]],
                                     const device uchar* sc [[buffer(2)]], const device uint* tiles [[buffer(3)]],
                                     const device uint* ntiles [[buffer(4)]], const device uint* st_tok [[buffer(5)]],
                                     const device uint* st_pair [[buffer(6)]], device float* act [[buffer(7)]],
                                     constant Ink2MoeP& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint eidx = tg.y;
  if (eidx >= ntiles[0]) return;
  const uint swarp = tid >> 3u, slane = lane & 7u;
  const uint j = tg.x * 16u + swarp;
  const bool j_ok = j < p.I;
  const uint e = tiles[eidx * 3 + 0];
  const uint start = tiles[eidx * 3 + 1];
  const uint count = tiles[eidx * 3 + 2];
  const uint two_i = 2u * p.I;
  const uint blocks = p.H / 32u;
  const size_t wg_row = size_t(e) * two_i + 2u * j;
  const size_t wu_row = wg_row + 1u;

  auto situ = [](float g, float u) {
    return (4.0f * metal::precise::tanh(g * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g))) *
           (25.0f * metal::precise::tanh(u * 0.04f));
  };

  if (count == 1u) {
    const uint tok0 = st_tok[start], pair0 = st_pair[start];
    const device float* xp0 = x + size_t(tok0) * p.H;
    float ag0 = 0.0f, au0 = 0.0f;
    for (uint kblk = 0u; kblk < blocks; kblk += 8u) {
      const uint blk = kblk + slane;
      if (j_ok && blk < blocks) {
        const uint kbase = blk * 32u;
        const device float4* xp = (const device float4*)(xp0 + kbase);
        const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
        const float4 x4 = xp[4], x5 = xp[5], x6 = xp[6], x7 = xp[7];
        auto dot32 = [&](size_t r) {
          const float sb = ink2_ue8m0(sc[r * blocks + blk]);
          const uint4 wa = *(const device uint4*)(w13 + r * p.H + kbase + 0u);
          const uint4 wb = *(const device uint4*)(w13 + r * p.H + kbase + 16u);
          return (dot(x0, ink2_e4m3_4(wa.x)) + dot(x1, ink2_e4m3_4(wa.y)) +
                  dot(x2, ink2_e4m3_4(wa.z)) + dot(x3, ink2_e4m3_4(wa.w)) +
                  dot(x4, ink2_e4m3_4(wb.x)) + dot(x5, ink2_e4m3_4(wb.y)) +
                  dot(x6, ink2_e4m3_4(wb.z)) + dot(x7, ink2_e4m3_4(wb.w))) * sb;
        };
        ag0 += dot32(wg_row);
        au0 += dot32(wu_row);
      }
    }
    if (j_ok) {
      ag0 += simd_shuffle_xor(ag0, 4u); ag0 += simd_shuffle_xor(ag0, 2u); ag0 += simd_shuffle_xor(ag0, 1u);
      au0 += simd_shuffle_xor(au0, 4u); au0 += simd_shuffle_xor(au0, 2u); au0 += simd_shuffle_xor(au0, 1u);
      if (slane == 0u) act[size_t(pair0) * p.I + j] = situ(ag0, au0);
    }
    return;
  }

  if (count == 2u) {
    const uint tok0 = st_tok[start], tok1 = st_tok[start + 1u];
    const uint pair0 = st_pair[start], pair1 = st_pair[start + 1u];
    const device float* xp0 = x + size_t(tok0) * p.H;
    const device float* xp1 = x + size_t(tok1) * p.H;
    float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;
    for (uint kblk = 0u; kblk < blocks; kblk += 8u) {
      const uint blk = kblk + slane;
      if (j_ok && blk < blocks) {
        const uint kbase = blk * 32u;
        auto dot32_2 = [&](size_t r, thread float& a0, thread float& a1) {
          const float sb = ink2_ue8m0(sc[r * blocks + blk]);
          const uint4 wa = *(const device uint4*)(w13 + r * p.H + kbase + 0u);
          const uint4 wb = *(const device uint4*)(w13 + r * p.H + kbase + 16u);
          const float4 w0 = ink2_e4m3_4(wa.x), w1 = ink2_e4m3_4(wa.y);
          const float4 w2v = ink2_e4m3_4(wa.z), w3 = ink2_e4m3_4(wa.w);
          const float4 w4 = ink2_e4m3_4(wb.x), w5 = ink2_e4m3_4(wb.y);
          const float4 w6 = ink2_e4m3_4(wb.z), w7 = ink2_e4m3_4(wb.w);
          const device float4* p0x = (const device float4*)(xp0 + kbase);
          const device float4* p1x = (const device float4*)(xp1 + kbase);
          a0 += (dot(p0x[0], w0) + dot(p0x[1], w1) + dot(p0x[2], w2v) + dot(p0x[3], w3) +
                 dot(p0x[4], w4) + dot(p0x[5], w5) + dot(p0x[6], w6) + dot(p0x[7], w7)) * sb;
          a1 += (dot(p1x[0], w0) + dot(p1x[1], w1) + dot(p1x[2], w2v) + dot(p1x[3], w3) +
                 dot(p1x[4], w4) + dot(p1x[5], w5) + dot(p1x[6], w6) + dot(p1x[7], w7)) * sb;
        };
        dot32_2(wg_row, ag0, ag1);
        dot32_2(wu_row, au0, au1);
      }
    }
    if (j_ok) {
      ag0 += simd_shuffle_xor(ag0, 4u); ag0 += simd_shuffle_xor(ag0, 2u); ag0 += simd_shuffle_xor(ag0, 1u);
      au0 += simd_shuffle_xor(au0, 4u); au0 += simd_shuffle_xor(au0, 2u); au0 += simd_shuffle_xor(au0, 1u);
      ag1 += simd_shuffle_xor(ag1, 4u); ag1 += simd_shuffle_xor(ag1, 2u); ag1 += simd_shuffle_xor(ag1, 1u);
      au1 += simd_shuffle_xor(au1, 4u); au1 += simd_shuffle_xor(au1, 2u); au1 += simd_shuffle_xor(au1, 1u);
      if (slane == 0u) {
        act[size_t(pair0) * p.I + j] = situ(ag0, au0);
        act[size_t(pair1) * p.I + j] = situ(ag1, au1);
      }
    }
    return;
  }

  for (uint p0 = 0u; p0 < count; p0 += 4u) {
    const uint mc = min(4u, count - p0);
    uint tok[4], pair[4];
    _Pragma("clang loop unroll(full)")
    for (uint m = 0u; m < 4u; ++m) {
      const uint mm = (m < mc) ? m : 0u;
      tok[m] = st_tok[start + p0 + mm];
      pair[m] = st_pair[start + p0 + mm];
    }
    float ag[4] = {}, au[4] = {};
    for (uint kblk = 0u; kblk < blocks; kblk += 8u) {
      const uint blk = kblk + slane;
      if (j_ok && blk < blocks) {
        const uint kbase = blk * 32u;
        auto accum_row = [&](size_t r, thread float* acc) {
          const float sb = ink2_ue8m0(sc[r * blocks + blk]);
          const uint4 wa = *(const device uint4*)(w13 + r * p.H + kbase + 0u);
          const uint4 wb = *(const device uint4*)(w13 + r * p.H + kbase + 16u);
          const float4 w0 = ink2_e4m3_4(wa.x), w1 = ink2_e4m3_4(wa.y);
          const float4 w2v = ink2_e4m3_4(wa.z), w3 = ink2_e4m3_4(wa.w);
          const float4 w4 = ink2_e4m3_4(wb.x), w5 = ink2_e4m3_4(wb.y);
          const float4 w6 = ink2_e4m3_4(wb.z), w7 = ink2_e4m3_4(wb.w);
          _Pragma("clang loop unroll(full)")
          for (uint m = 0u; m < 4u; ++m) {
            if (m < mc) {
              const device float4* xp = (const device float4*)(x + size_t(tok[m]) * p.H + kbase);
              acc[m] += (dot(xp[0], w0) + dot(xp[1], w1) + dot(xp[2], w2v) + dot(xp[3], w3) +
                         dot(xp[4], w4) + dot(xp[5], w5) + dot(xp[6], w6) + dot(xp[7], w7)) * sb;
            }
          }
        };
        accum_row(wg_row, ag);
        accum_row(wu_row, au);
      }
    }
    if (j_ok) {
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < 4u; ++m) {
        if (m < mc) {
          ag[m] += simd_shuffle_xor(ag[m], 4u); ag[m] += simd_shuffle_xor(ag[m], 2u); ag[m] += simd_shuffle_xor(ag[m], 1u);
          au[m] += simd_shuffle_xor(au[m], 4u); au[m] += simd_shuffle_xor(au[m], 2u); au[m] += simd_shuffle_xor(au[m], 1u);
          if (slane == 0u) act[size_t(pair[m]) * p.I + j] = situ(ag[m], au[m]);
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
                                    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint eidx = tg.y;
  if (eidx >= ntiles[0]) return;
  const uint swarp = tid >> 2u, slane = lane & 3u;
  const uint n = tg.x * 32u + swarp;
  const bool n_ok = n < p.H;
  const uint e = tiles[eidx * 3 + 0];
  const uint start = tiles[eidx * 3 + 1];
  const uint count = tiles[eidx * 3 + 2];
  const uint blocks = p.I / 32u;
  const size_t wrow = size_t(e) * p.H + n;

  if (count == 1u) {
    const uint pair0 = st_pair[start];
    const device float* ap0 = act + size_t(pair0) * p.I;
    float sum0 = 0.0f;
    for (uint kblk = 0u; kblk < blocks; kblk += 4u) {
      const uint blk = kblk + slane;
      if (n_ok && blk < blocks) {
        const float sb = ink2_ue8m0(sc[wrow * blocks + blk]);
        const uint kbase = blk * 32u;
        const uint4 wa = *(const device uint4*)(w2 + wrow * p.I + kbase + 0u);
        const uint4 wb = *(const device uint4*)(w2 + wrow * p.I + kbase + 16u);
        const device float4* ap = (const device float4*)(ap0 + kbase);
        sum0 += (dot(ap[0], ink2_e4m3_4(wa.x)) + dot(ap[1], ink2_e4m3_4(wa.y)) +
                 dot(ap[2], ink2_e4m3_4(wa.z)) + dot(ap[3], ink2_e4m3_4(wa.w)) +
                 dot(ap[4], ink2_e4m3_4(wb.x)) + dot(ap[5], ink2_e4m3_4(wb.y)) +
                 dot(ap[6], ink2_e4m3_4(wb.z)) + dot(ap[7], ink2_e4m3_4(wb.w))) * sb;
      }
    }
    if (n_ok) {
      sum0 += simd_shuffle_xor(sum0, 2u); sum0 += simd_shuffle_xor(sum0, 1u);
      if (slane == 0u) pair_out[size_t(pair0) * p.H + n] = st_ew[start] * sum0;
    }
    return;
  }

  if (count == 2u) {
    const uint pair0 = st_pair[start], pair1 = st_pair[start + 1u];
    const device float* ap0 = act + size_t(pair0) * p.I;
    const device float* ap1 = act + size_t(pair1) * p.I;
    float sum0 = 0.0f, sum1 = 0.0f;
    for (uint kblk = 0u; kblk < blocks; kblk += 4u) {
      const uint blk = kblk + slane;
      if (n_ok && blk < blocks) {
        const float sb = ink2_ue8m0(sc[wrow * blocks + blk]);
        const uint kbase = blk * 32u;
        const uint4 wa = *(const device uint4*)(w2 + wrow * p.I + kbase + 0u);
        const uint4 wb = *(const device uint4*)(w2 + wrow * p.I + kbase + 16u);
        const float4 w0 = ink2_e4m3_4(wa.x), w1 = ink2_e4m3_4(wa.y);
        const float4 w2v = ink2_e4m3_4(wa.z), w3 = ink2_e4m3_4(wa.w);
        const float4 w4 = ink2_e4m3_4(wb.x), w5 = ink2_e4m3_4(wb.y);
        const float4 w6 = ink2_e4m3_4(wb.z), w7 = ink2_e4m3_4(wb.w);
        const device float4* p0a = (const device float4*)(ap0 + kbase);
        const device float4* p1a = (const device float4*)(ap1 + kbase);
        sum0 += (dot(p0a[0], w0) + dot(p0a[1], w1) + dot(p0a[2], w2v) + dot(p0a[3], w3) +
                 dot(p0a[4], w4) + dot(p0a[5], w5) + dot(p0a[6], w6) + dot(p0a[7], w7)) * sb;
        sum1 += (dot(p1a[0], w0) + dot(p1a[1], w1) + dot(p1a[2], w2v) + dot(p1a[3], w3) +
                 dot(p1a[4], w4) + dot(p1a[5], w5) + dot(p1a[6], w6) + dot(p1a[7], w7)) * sb;
      }
    }
    if (n_ok) {
      sum0 += simd_shuffle_xor(sum0, 2u); sum0 += simd_shuffle_xor(sum0, 1u);
      sum1 += simd_shuffle_xor(sum1, 2u); sum1 += simd_shuffle_xor(sum1, 1u);
      if (slane == 0u) {
        pair_out[size_t(pair0) * p.H + n] = st_ew[start] * sum0;
        pair_out[size_t(pair1) * p.H + n] = st_ew[start + 1u] * sum1;
      }
    }
    return;
  }

  for (uint p0 = 0u; p0 < count; p0 += 8u) {
    const uint mc = min(8u, count - p0);
    uint pair[8];
    _Pragma("clang loop unroll(full)")
    for (uint m = 0u; m < 8u; ++m) pair[m] = st_pair[start + p0 + ((m < mc) ? m : 0u)];
    float sum[8] = {};
    for (uint kblk = 0u; kblk < blocks; kblk += 4u) {
      const uint blk = kblk + slane;
      if (n_ok && blk < blocks) {
        const float sb = ink2_ue8m0(sc[wrow * blocks + blk]);
        const uint kbase = blk * 32u;
        const uint4 wa = *(const device uint4*)(w2 + wrow * p.I + kbase + 0u);
        const uint4 wb = *(const device uint4*)(w2 + wrow * p.I + kbase + 16u);
        const float4 w0 = ink2_e4m3_4(wa.x), w1 = ink2_e4m3_4(wa.y);
        const float4 w2v = ink2_e4m3_4(wa.z), w3 = ink2_e4m3_4(wa.w);
        const float4 w4 = ink2_e4m3_4(wb.x), w5 = ink2_e4m3_4(wb.y);
        const float4 w6 = ink2_e4m3_4(wb.z), w7 = ink2_e4m3_4(wb.w);
        _Pragma("clang loop unroll(full)")
        for (uint m = 0u; m < 8u; ++m) {
          if (m < mc) {
            const device float4* ap = (const device float4*)(act + size_t(pair[m]) * p.I + kbase);
            sum[m] += (dot(ap[0], w0) + dot(ap[1], w1) + dot(ap[2], w2v) + dot(ap[3], w3) +
                       dot(ap[4], w4) + dot(ap[5], w5) + dot(ap[6], w6) + dot(ap[7], w7)) * sb;
          }
        }
      }
    }
    if (n_ok) {
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < 8u; ++m) {
        if (m < mc) {
          sum[m] += simd_shuffle_xor(sum[m], 2u); sum[m] += simd_shuffle_xor(sum[m], 1u);
          if (slane == 0u) pair_out[size_t(pair[m]) * p.H + n] = st_ew[start + p0 + m] * sum[m];
        }
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

// ---- 15. Tree-Speculative KV Compaction & Tap Row Gather ----
struct Ink2KvCompactP {
  uint pos0, kv_cap, is_swa, kv_fp8, kvd, m;
  uint path[16];
};

[[kernel]] void ink2_kv_compact(device void* kc [[buffer(0)]], device void* vc [[buffer(1)]],
                                constant Ink2KvCompactP& p [[buffer(2)]],
                                uint tid [[thread_index_in_threadgroup]]) {
  const uint c4 = tid * 4u;
  if (c4 >= p.kvd || p.m <= 1u) return;
  const uint m = min(p.m, 16u);
  // Topological order guarantees path[k] >= k and path[k] strictly increases, so forward streaming is race-free.
  if (p.kv_fp8) {
    for (uint k = 1u; k < m; ++k) {
      if (p.path[k] == k) continue;
      const uint spos = p.pos0 + p.path[k];
      const uint sslot = p.is_swa ? (spos % p.kv_cap) : spos;
      const size_t soff = size_t(sslot) * p.kvd + c4;
      const uint kv = *(const device uint*)((const device uchar*)kc + soff);
      const uint vv = *(const device uint*)((const device uchar*)vc + soff);
      const uint dpos = p.pos0 + k;
      const uint dslot = p.is_swa ? (dpos % p.kv_cap) : dpos;
      const size_t doff = size_t(dslot) * p.kvd + c4;
      *(device uint*)((device uchar*)kc + doff) = kv;
      *(device uint*)((device uchar*)vc + doff) = vv;
    }
  } else {
    for (uint k = 1u; k < m; ++k) {
      if (p.path[k] == k) continue;
      const uint spos = p.pos0 + p.path[k];
      const uint sslot = p.is_swa ? (spos % p.kv_cap) : spos;
      const size_t soff = size_t(sslot) * p.kvd + c4;
      const uint2 kv = *(const device uint2*)((const device half*)kc + soff);
      const uint2 vv = *(const device uint2*)((const device half*)vc + soff);
      const uint dpos = p.pos0 + k;
      const uint dslot = p.is_swa ? (dpos % p.kv_cap) : dpos;
      const size_t doff = size_t(dslot) * p.kvd + c4;
      *(device uint2*)((device half*)kc + doff) = kv;
      *(device uint2*)((device half*)vc + doff) = vv;
    }
  }
}

struct Ink2GatherP {
  uint m, H;
  uint path[16];
};

[[kernel]] void ink2_gather_rows(const device float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                                 constant Ink2GatherP& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint h4 = p.H >> 2u;
  const uint c4 = gid.x, k = gid.y;
  if (c4 >= h4 || k >= p.m) return;
  dst[size_t(k) * h4 + c4] = src[size_t(p.path[k]) * h4 + c4];
}
