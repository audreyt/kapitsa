#include "common.h"

// Inkling-2-Mini MTP DFlare speculative drafter kernels (standard Metal 3 compute primitives,
// ushort/uint2 BF16 bit-shifts, no tensor_ops, no simdgroup_matrix, no atomics).

NI_INLINE float ink2m_bf16(ushort u) { return as_type<float>(uint(u) << 16); }

NI_INLINE float4 ink2m_bf16_4(uint2 r) {
  return float4(as_type<float>(r.x << 16), as_type<float>(r.x & 0xFFFF0000u),
                as_type<float>(r.y << 16), as_type<float>(r.y & 0xFFFF0000u));
}

NI_INLINE ushort ink2m_f32_to_bf16(float x) {
  uint u = as_type<uint>(x);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return ushort((u >> 16) | 0x0040u);
  const uint lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;
  return ushort(u >> 16);
}

NI_INLINE float ink2m_round_bf16(float x) { return ink2m_bf16(ink2m_f32_to_bf16(x)); }

// ---- 1. Tap fusion + DFlare RMSNorm (alias-free for 4 tap buffers + BF16 norm) ----
struct Ink2MtpFuseP {
  uint T, H;
  float w0, w1, w2, w3, eps;
};

[[kernel]] void ink2_mtp_fuse_norm(const device float* f0 [[buffer(0)]], const device float* f1 [[buffer(1)]],
                                   const device float* f2 [[buffer(2)]], const device float* f3 [[buffer(3)]],
                                   const device ushort* norm_w [[buffer(4)]], device float* out [[buffer(5)]],
                                   constant Ink2MtpFuseP& p [[buffer(6)]], uint t [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const size_t base = size_t(t) * p.H;
  float xs[10];
  float ss = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float v = p.w0 * f0[base + c] + p.w1 * f1[base + c] + p.w2 * f2[base + c] + p.w3 * f3[base + c];
    xs[k] = v;
    ss += v * v;
  }
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) out[base + c] = xs[k] * inv * ink2m_bf16(norm_w[c]);
}

// ---- 2. Draft block preparation: row 0 = embedTokens(tokens[pos0]), rows 1..B-1 = mask_embed ----
struct Ink2MtpPrepP {
  uint B, H, pos0;
  float eps;
};

[[kernel]] void ink2_mtp_prep_block(const device int* tokens [[buffer(0)]], const device ushort* embed [[buffer(1)]],
                                    const device ushort* embed_norm [[buffer(2)]],
                                    const device ushort* mask_embed [[buffer(3)]], device float* out [[buffer(4)]],
                                    constant Ink2MtpPrepP& p [[buffer(5)]], uint t [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  device float* dst = out + size_t(t) * p.H;
  if (t == 0u) {
    const int id = tokens[p.pos0];
    const device ushort* row = embed + size_t(id) * p.H;
    float xs[10];
    float ss = 0.0f;
    uint k = 0;
    for (uint c = tid; c < p.H; c += ntg, ++k) {
      const float v = ink2m_bf16(row[c]);
      xs[k] = v;
      ss += v * v;
    }
    ss = tg_sum(ss, scratch, tid, ntg);
    const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
    k = 0;
    for (uint c = tid; c < p.H; c += ntg, ++k) dst[c] = xs[k] * inv * ink2m_bf16(embed_norm[c]);
  } else {
    for (uint c = tid; c < p.H; c += ntg) dst[c] = ink2m_bf16(mask_embed[c]);
  }
}

// ---- 3. Generic RMSNorm with BF16 weight (alias-safe) ----
struct Ink2MtpNormP {
  uint T, H;
  float eps;
};

[[kernel]] void ink2_mtp_rmsnorm(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                                 device float* out [[buffer(2)]], constant Ink2MtpNormP& p [[buffer(3)]],
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
  for (uint c = tid; c < p.H; c += ntg, ++k) dst[c] = xs[k] * inv * ink2m_bf16(w[c]);
}

// ---- 4. Small-batch BF16 MatVec (M <= 8, 2 output rows per simdgroup, optional bias/add/softcap) ----
struct Ink2MtpMmP {
  uint M, N, K, x_ld, y_ld, has_bias, bias_f32, add;
  float softcap;
};

[[kernel]] void ink2_mtp_bf16_mv(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                                 const device void* bias [[buffer(2)]], device float* y [[buffer(3)]],
                                 constant Ink2MtpMmP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4 + sg) * 2;
  if (n0 >= p.N) return;
  const bool has1 = (n0 + 1) < p.N;
  float acc0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float acc1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const size_t woff0 = size_t(n0) * p.K;
  const size_t woff1 = woff0 + p.K;
  if ((p.K & 7u) == 0u) {
    for (uint k = lane * 8; k < p.K; k += 256) {
      const uint4 q0 = *(const device uint4*)(w + woff0 + k);
      const float4 w0a = ink2m_bf16_4(q0.xy), w0b = ink2m_bf16_4(q0.zw);
      float4 w1a = float4(0.0f), w1b = float4(0.0f);
      if (has1) {
        const uint4 q1 = *(const device uint4*)(w + woff1 + k);
        w1a = ink2m_bf16_4(q1.xy);
        w1b = ink2m_bf16_4(q1.zw);
      }
      for (uint m = 0; m < p.M; ++m) {
        const device float* xp = x + size_t(m) * p.x_ld + k;
        const float4 x0 = *(const device float4*)(xp + 0);
        const float4 x1 = *(const device float4*)(xp + 4);
        acc0[m] += dot(x0, w0a) + dot(x1, w0b);
        if (has1) acc1[m] += dot(x0, w1a) + dot(x1, w1b);
      }
    }
  } else {
    const uint k_end4 = p.K & ~3u;
    for (uint k = lane * 4; k < k_end4; k += 128) {
      const float4 wv0 = ink2m_bf16_4(*(const device uint2*)(w + woff0 + k));
      const float4 wv1 = has1 ? ink2m_bf16_4(*(const device uint2*)(w + woff1 + k)) : float4(0.0f);
      for (uint m = 0; m < p.M; ++m) {
        const float4 xv = *(const device float4*)(x + size_t(m) * p.x_ld + k);
        acc0[m] += dot(xv, wv0);
        if (has1) acc1[m] += dot(xv, wv1);
      }
    }
    if (lane < (p.K - k_end4)) {
      const uint k = k_end4 + lane;
      const float fw0 = ink2m_bf16(w[woff0 + k]);
      const float fw1 = has1 ? ink2m_bf16(w[woff1 + k]) : 0.0f;
      for (uint m = 0; m < p.M; ++m) {
        const float xv = x[size_t(m) * p.x_ld + k];
        acc0[m] += xv * fw0;
        if (has1) acc1[m] += xv * fw1;
      }
    }
  }
  const float b0 =
      p.has_bias ? (p.bias_f32 ? ((const device float*)bias)[n0] : ink2m_bf16(((const device ushort*)bias)[n0])) : 0.0f;
  const float b1 = (p.has_bias && has1)
                       ? (p.bias_f32 ? ((const device float*)bias)[n0 + 1] : ink2m_bf16(((const device ushort*)bias)[n0 + 1]))
                       : 0.0f;
  const float inv_cap = p.softcap > 0.0f ? (1.0f / p.softcap) : 0.0f;
  for (uint m = 0; m < p.M; ++m) {
    float s0 = simd_sum(acc0[m]) + b0;
    float s1 = has1 ? (simd_sum(acc1[m]) + b1) : 0.0f;
    if (lane == 0) {
      device float* dst0 = y + size_t(m) * p.y_ld + n0;
      s0 = p.add ? (*dst0 + s0) : s0;
      if (p.softcap > 0.0f) s0 = p.softcap * metal::precise::tanh(s0 * inv_cap);
      *dst0 = s0;
      if (has1) {
        device float* dst1 = dst0 + 1;
        s1 = p.add ? (*dst1 + s1) : s1;
        if (p.softcap > 0.0f) s1 = p.softcap * metal::precise::tanh(s1 * inv_cap);
        *dst1 = s1;
      }
    }
  }
}

[[kernel]] void ink2_mtp_bf16_mv4(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                                  const device void* bias [[buffer(2)]], device float* y [[buffer(3)]],
                                  constant Ink2MtpMmP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4 + sg) * 4;
  if (n0 >= p.N) return;
  const uint mc = min(4u, p.N - n0);
  float acc0[4] = {0, 0, 0, 0}, acc1[4] = {0, 0, 0, 0}, acc2[4] = {0, 0, 0, 0}, acc3[4] = {0, 0, 0, 0};
  const size_t woff0 = size_t(n0) * p.K;
  if ((p.K & 7u) == 0u) {
    if (p.M == 1u) {
      for (uint k = lane * 8; k < p.K; k += 256) {
        const device float* xp = x + k;
        const float4 x0 = *(const device float4*)(xp + 0);
        const float4 x1 = *(const device float4*)(xp + 4);
        const uint4 q0 = *(const device uint4*)(w + woff0 + k);
        const uint4 q1 = mc > 1u ? *(const device uint4*)(w + woff0 + 1 * p.K + k) : uint4(0);
        const uint4 q2 = mc > 2u ? *(const device uint4*)(w + woff0 + 2 * p.K + k) : uint4(0);
        const uint4 q3 = mc > 3u ? *(const device uint4*)(w + woff0 + 3 * p.K + k) : uint4(0);
        acc0[0] += dot(x0, ink2m_bf16_4(q0.xy)) + dot(x1, ink2m_bf16_4(q0.zw));
        if (mc > 1u) acc1[0] += dot(x0, ink2m_bf16_4(q1.xy)) + dot(x1, ink2m_bf16_4(q1.zw));
        if (mc > 2u) acc2[0] += dot(x0, ink2m_bf16_4(q2.xy)) + dot(x1, ink2m_bf16_4(q2.zw));
        if (mc > 3u) acc3[0] += dot(x0, ink2m_bf16_4(q3.xy)) + dot(x1, ink2m_bf16_4(q3.zw));
      }
    } else {
      for (uint k = lane * 8; k < p.K; k += 256) {
        const uint4 q0 = *(const device uint4*)(w + woff0 + k);
        const uint4 q1 = mc > 1u ? *(const device uint4*)(w + woff0 + 1 * p.K + k) : uint4(0);
        const uint4 q2 = mc > 2u ? *(const device uint4*)(w + woff0 + 2 * p.K + k) : uint4(0);
        const uint4 q3 = mc > 3u ? *(const device uint4*)(w + woff0 + 3 * p.K + k) : uint4(0);
        float4 x0[4], x1[4];
        for (uint m = 0; m < p.M; ++m) {
          const device float* xp = x + size_t(m) * p.x_ld + k;
          x0[m] = *(const device float4*)(xp + 0);
          x1[m] = *(const device float4*)(xp + 4);
        }
        {
          const float4 wa = ink2m_bf16_4(q0.xy), wb = ink2m_bf16_4(q0.zw);
          for (uint m = 0; m < p.M; ++m) acc0[m] += dot(x0[m], wa) + dot(x1[m], wb);
        }
        if (mc > 1u) {
          const float4 wa = ink2m_bf16_4(q1.xy), wb = ink2m_bf16_4(q1.zw);
          for (uint m = 0; m < p.M; ++m) acc1[m] += dot(x0[m], wa) + dot(x1[m], wb);
        }
        if (mc > 2u) {
          const float4 wa = ink2m_bf16_4(q2.xy), wb = ink2m_bf16_4(q2.zw);
          for (uint m = 0; m < p.M; ++m) acc2[m] += dot(x0[m], wa) + dot(x1[m], wb);
        }
        if (mc > 3u) {
          const float4 wa = ink2m_bf16_4(q3.xy), wb = ink2m_bf16_4(q3.zw);
          for (uint m = 0; m < p.M; ++m) acc3[m] += dot(x0[m], wa) + dot(x1[m], wb);
        }
      }
    }
  } else {
    const uint k_end4 = p.K & ~3u;
    for (uint k = lane * 4; k < k_end4; k += 128) {
      const float4 w0 = ink2m_bf16_4(*(const device uint2*)(w + woff0 + k));
      const float4 w1 = mc > 1u ? ink2m_bf16_4(*(const device uint2*)(w + woff0 + p.K + k)) : float4(0.0f);
      const float4 w2 = mc > 2u ? ink2m_bf16_4(*(const device uint2*)(w + woff0 + 2 * p.K + k)) : float4(0.0f);
      const float4 w3 = mc > 3u ? ink2m_bf16_4(*(const device uint2*)(w + woff0 + 3 * p.K + k)) : float4(0.0f);
      for (uint m = 0; m < p.M; ++m) {
        const float4 xv = *(const device float4*)(x + size_t(m) * p.x_ld + k);
        acc0[m] += dot(xv, w0);
        if (mc > 1u) acc1[m] += dot(xv, w1);
        if (mc > 2u) acc2[m] += dot(xv, w2);
        if (mc > 3u) acc3[m] += dot(xv, w3);
      }
    }
    if (lane < (p.K - k_end4)) {
      const uint k = k_end4 + lane;
      const float fw0 = ink2m_bf16(w[woff0 + k]);
      const float fw1 = mc > 1u ? ink2m_bf16(w[woff0 + p.K + k]) : 0.0f;
      const float fw2 = mc > 2u ? ink2m_bf16(w[woff0 + 2 * p.K + k]) : 0.0f;
      const float fw3 = mc > 3u ? ink2m_bf16(w[woff0 + 3 * p.K + k]) : 0.0f;
      for (uint m = 0; m < p.M; ++m) {
        const float xv = x[size_t(m) * p.x_ld + k];
        acc0[m] += xv * fw0;
        if (mc > 1u) acc1[m] += xv * fw1;
        if (mc > 2u) acc2[m] += xv * fw2;
        if (mc > 3u) acc3[m] += xv * fw3;
      }
    }
  }
  const float inv_cap = p.softcap > 0.0f ? (1.0f / p.softcap) : 0.0f;
  float b[4] = {0, 0, 0, 0};
  if (p.has_bias) {
    for (uint q = 0; q < mc; ++q)
      b[q] = p.bias_f32 ? ((const device float*)bias)[n0 + q] : ink2m_bf16(((const device ushort*)bias)[n0 + q]);
  }
  for (uint m = 0; m < p.M; ++m) {
    const float s0 = simd_sum(acc0[m]) + b[0];
    const float s1 = mc > 1u ? (simd_sum(acc1[m]) + b[1]) : 0.0f;
    const float s2 = mc > 2u ? (simd_sum(acc2[m]) + b[2]) : 0.0f;
    const float s3 = mc > 3u ? (simd_sum(acc3[m]) + b[3]) : 0.0f;
    if (lane == 0u) {
      const float sv[4] = {s0, s1, s2, s3};
      device float* dst = y + size_t(m) * p.y_ld + n0;
      for (uint q = 0; q < mc; ++q) {
        float val = p.add ? (dst[q] + sv[q]) : sv[q];
        if (p.softcap > 0.0f) val = p.softcap * metal::precise::tanh(val * inv_cap);
        dst[q] = val;
      }
    }
  }
}

[[kernel]] void ink2_mtp_bf16_mv8(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                                  const device void* bias [[buffer(2)]], device float* y [[buffer(3)]],
                                  constant Ink2MtpMmP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[8][256];
  const uint n0 = (tg * 4u + sg) * 4u;
  const bool valid = n0 < p.N;
  const uint mc = valid ? min(4u, p.N - n0) : 0u;
  float acc0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float acc1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float acc2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float acc3[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const size_t woff0 = size_t(n0) * p.K;
  const uint total_v4 = p.M * 64u;
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    for (uint idx = tid; idx < total_v4; idx += 128u) {
      const uint m = idx >> 6u;
      const uint c4 = (idx & 63u) << 2u;
      *(threadgroup float4*)(xs[m] + c4) = *(const device float4*)(x + size_t(m) * p.x_ld + kbase + c4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
      const uint kl = lane * 8u;
      const uint4 q0 = *(const device uint4*)(w + woff0 + kbase + kl);
      const uint4 q1 = mc > 1u ? *(const device uint4*)(w + woff0 + 1 * p.K + kbase + kl) : uint4(0);
      const uint4 q2 = mc > 2u ? *(const device uint4*)(w + woff0 + 2 * p.K + kbase + kl) : uint4(0);
      const uint4 q3 = mc > 3u ? *(const device uint4*)(w + woff0 + 3 * p.K + kbase + kl) : uint4(0);
      {
        const float4 wa = ink2m_bf16_4(q0.xy), wb = ink2m_bf16_4(q0.zw);
        for (uint m = 0u; m < p.M; ++m) {
          const threadgroup float* xp = xs[m] + kl;
          acc0[m] += dot(*(const threadgroup float4*)(xp + 0u), wa) + dot(*(const threadgroup float4*)(xp + 4u), wb);
        }
      }
      if (mc > 1u) {
        const float4 wa = ink2m_bf16_4(q1.xy), wb = ink2m_bf16_4(q1.zw);
        for (uint m = 0u; m < p.M; ++m) {
          const threadgroup float* xp = xs[m] + kl;
          acc1[m] += dot(*(const threadgroup float4*)(xp + 0u), wa) + dot(*(const threadgroup float4*)(xp + 4u), wb);
        }
      }
      if (mc > 2u) {
        const float4 wa = ink2m_bf16_4(q2.xy), wb = ink2m_bf16_4(q2.zw);
        for (uint m = 0u; m < p.M; ++m) {
          const threadgroup float* xp = xs[m] + kl;
          acc2[m] += dot(*(const threadgroup float4*)(xp + 0u), wa) + dot(*(const threadgroup float4*)(xp + 4u), wb);
        }
      }
      if (mc > 3u) {
        const float4 wa = ink2m_bf16_4(q3.xy), wb = ink2m_bf16_4(q3.zw);
        for (uint m = 0u; m < p.M; ++m) {
          const threadgroup float* xp = xs[m] + kl;
          acc3[m] += dot(*(const threadgroup float4*)(xp + 0u), wa) + dot(*(const threadgroup float4*)(xp + 4u), wb);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!valid) return;
  const float inv_cap = p.softcap > 0.0f ? (1.0f / p.softcap) : 0.0f;
  float b[4] = {0, 0, 0, 0};
  if (p.has_bias) {
    for (uint q = 0u; q < mc; ++q)
      b[q] = p.bias_f32 ? ((const device float*)bias)[n0 + q] : ink2m_bf16(((const device ushort*)bias)[n0 + q]);
  }
  for (uint m = 0u; m < p.M; ++m) {
    const float s0 = simd_sum(acc0[m]) + b[0];
    const float s1 = mc > 1u ? (simd_sum(acc1[m]) + b[1]) : 0.0f;
    const float s2 = mc > 2u ? (simd_sum(acc2[m]) + b[2]) : 0.0f;
    const float s3 = mc > 3u ? (simd_sum(acc3[m]) + b[3]) : 0.0f;
    if (lane == 0u) {
      const float sv[4] = {s0, s1, s2, s3};
      device float* dst = y + size_t(m) * p.y_ld + n0;
      for (uint q = 0u; q < mc; ++q) {
        float val = p.add ? (dst[q] + sv[q]) : sv[q];
        if (p.softcap > 0.0f) val = p.softcap * metal::precise::tanh(val * inv_cap);
        dst[q] = val;
      }
    }
  }
}

struct Ink2MtpMulti3P {
  uint M, K, N_tot;
  uint offs[4];
};

[[kernel]] void ink2_mtp_bf16_mv_multi3(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                                        device float* y0 [[buffer(2)]], device float* y1 [[buffer(3)]],
                                        device float* y2 [[buffer(4)]], constant Ink2MtpMulti3P& p [[buffer(5)]],
                                        uint tg [[threadgroup_position_in_grid]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4u + sg) * 2u;
  if (n0 >= p.N_tot) return;
  uint seg = 0u;
  while (seg < 2u && n0 >= p.offs[seg + 1u]) ++seg;
  const uint seg_off = p.offs[seg];
  const uint seg_len = p.offs[seg + 1u] - seg_off;
  const bool has1 = (n0 + 1u) < p.offs[seg + 1u];
  float acc0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float acc1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const size_t woff0 = size_t(n0) * p.K;
  const size_t woff1 = woff0 + p.K;
  for (uint k = lane * 8u; k < p.K; k += 256u) {
    const uint4 q0 = *(const device uint4*)(w + woff0 + k);
    const float4 w0a = ink2m_bf16_4(q0.xy), w0b = ink2m_bf16_4(q0.zw);
    float4 w1a = float4(0.0f), w1b = float4(0.0f);
    if (has1) {
      const uint4 q1 = *(const device uint4*)(w + woff1 + k);
      w1a = ink2m_bf16_4(q1.xy);
      w1b = ink2m_bf16_4(q1.zw);
    }
    for (uint m = 0; m < p.M; ++m) {
      const device float* xp = x + size_t(m) * p.K + k;
      const float4 x0 = *(const device float4*)(xp + 0);
      const float4 x1 = *(const device float4*)(xp + 4);
      acc0[m] += dot(x0, w0a) + dot(x1, w0b);
      if (has1) acc1[m] += dot(x0, w1a) + dot(x1, w1b);
    }
  }
  device float* bases[3] = {y0, y1, y2};
  device float* dst_base = bases[seg] + (n0 - seg_off);
  for (uint m = 0; m < p.M; ++m) {
    const float s0 = simd_sum(acc0[m]);
    const float s1 = has1 ? simd_sum(acc1[m]) : 0.0f;
    if (lane == 0u) {
      device float* dst = dst_base + size_t(m) * seg_len;
      dst[0] = s0;
      if (has1) dst[1] = s1;
    }
  }
}

[[kernel]] void ink2_mtp_bf16_mv4_multi3(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                                         device float* y0 [[buffer(2)]], device float* y1 [[buffer(3)]],
                                         device float* y2 [[buffer(4)]], constant Ink2MtpMulti3P& p [[buffer(5)]],
                                         uint tg [[threadgroup_position_in_grid]],
                                         uint sg [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4u + sg) * 4u;
  if (n0 >= p.N_tot) return;
  uint seg = 0u;
  while (seg < 2u && n0 >= p.offs[seg + 1u]) ++seg;
  const uint seg_off = p.offs[seg];
  const uint seg_len = p.offs[seg + 1u] - seg_off;
  const uint mc = min(4u, p.offs[seg + 1u] - n0);
  float acc0[4] = {0, 0, 0, 0}, acc1[4] = {0, 0, 0, 0}, acc2[4] = {0, 0, 0, 0}, acc3[4] = {0, 0, 0, 0};
  const size_t woff0 = size_t(n0) * p.K;
  for (uint k = lane * 8u; k < p.K; k += 256u) {
    const uint4 q0 = *(const device uint4*)(w + woff0 + k);
    const uint4 q1 = mc > 1u ? *(const device uint4*)(w + woff0 + 1 * p.K + k) : uint4(0);
    const uint4 q2 = mc > 2u ? *(const device uint4*)(w + woff0 + 2 * p.K + k) : uint4(0);
    const uint4 q3 = mc > 3u ? *(const device uint4*)(w + woff0 + 3 * p.K + k) : uint4(0);
    const float4 w0a = ink2m_bf16_4(q0.xy), w0b = ink2m_bf16_4(q0.zw);
    const float4 w1a = ink2m_bf16_4(q1.xy), w1b = ink2m_bf16_4(q1.zw);
    const float4 w2a = ink2m_bf16_4(q2.xy), w2b = ink2m_bf16_4(q2.zw);
    const float4 w3a = ink2m_bf16_4(q3.xy), w3b = ink2m_bf16_4(q3.zw);
    for (uint m = 0; m < p.M; ++m) {
      const device float* xp = x + size_t(m) * p.K + k;
      const float4 x0 = *(const device float4*)(xp + 0);
      const float4 x1 = *(const device float4*)(xp + 4);
      acc0[m] += dot(x0, w0a) + dot(x1, w0b);
      if (mc > 1u) acc1[m] += dot(x0, w1a) + dot(x1, w1b);
      if (mc > 2u) acc2[m] += dot(x0, w2a) + dot(x1, w2b);
      if (mc > 3u) acc3[m] += dot(x0, w3a) + dot(x1, w3b);
    }
  }
  device float* bases[3] = {y0, y1, y2};
  device float* dst_base = bases[seg] + (n0 - seg_off);
  for (uint m = 0; m < p.M; ++m) {
    const float s0 = simd_sum(acc0[m]);
    const float s1 = mc > 1u ? simd_sum(acc1[m]) : 0.0f;
    const float s2 = mc > 2u ? simd_sum(acc2[m]) : 0.0f;
    const float s3 = mc > 3u ? simd_sum(acc3[m]) : 0.0f;
    if (lane == 0u) {
      device float* dst = dst_base + size_t(m) * seg_len;
      dst[0] = s0;
      if (mc > 1u) dst[1] = s1;
      if (mc > 2u) dst[2] = s2;
      if (mc > 3u) dst[3] = s3;
    }
  }
}

[[kernel]] void ink2_mtp_bf16_mv8_multi3(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                                         device float* y0 [[buffer(2)]], device float* y1 [[buffer(3)]],
                                         device float* y2 [[buffer(4)]], constant Ink2MtpMulti3P& p [[buffer(5)]],
                                         uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[8][256];
  const uint n0 = (tg * 4u + sg) * 4u;
  const bool valid = n0 < p.N_tot;
  uint seg = 0u, seg_off = 0u, seg_len = 0u, mc = 0u;
  if (valid) {
    while (seg < 2u && n0 >= p.offs[seg + 1u]) ++seg;
    seg_off = p.offs[seg];
    seg_len = p.offs[seg + 1u] - seg_off;
    mc = min(4u, p.offs[seg + 1u] - n0);
  }
  float acc0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float acc1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float acc2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float acc3[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const size_t woff0 = size_t(n0) * p.K;
  const uint total_v4 = p.M * 64u;
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    for (uint idx = tid; idx < total_v4; idx += 128u) {
      const uint m = idx >> 6u;
      const uint c4 = (idx & 63u) << 2u;
      *(threadgroup float4*)(xs[m] + c4) = *(const device float4*)(x + size_t(m) * p.K + kbase + c4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
      const uint kl = lane * 8u;
      const uint4 q0 = *(const device uint4*)(w + woff0 + kbase + kl);
      const uint4 q1 = mc > 1u ? *(const device uint4*)(w + woff0 + 1 * p.K + kbase + kl) : uint4(0);
      const uint4 q2 = mc > 2u ? *(const device uint4*)(w + woff0 + 2 * p.K + kbase + kl) : uint4(0);
      const uint4 q3 = mc > 3u ? *(const device uint4*)(w + woff0 + 3 * p.K + kbase + kl) : uint4(0);
      {
        const float4 wa = ink2m_bf16_4(q0.xy), wb = ink2m_bf16_4(q0.zw);
        for (uint m = 0u; m < p.M; ++m) {
          const threadgroup float* xp = xs[m] + kl;
          acc0[m] += dot(*(const threadgroup float4*)(xp + 0u), wa) + dot(*(const threadgroup float4*)(xp + 4u), wb);
        }
      }
      if (mc > 1u) {
        const float4 wa = ink2m_bf16_4(q1.xy), wb = ink2m_bf16_4(q1.zw);
        for (uint m = 0u; m < p.M; ++m) {
          const threadgroup float* xp = xs[m] + kl;
          acc1[m] += dot(*(const threadgroup float4*)(xp + 0u), wa) + dot(*(const threadgroup float4*)(xp + 4u), wb);
        }
      }
      if (mc > 2u) {
        const float4 wa = ink2m_bf16_4(q2.xy), wb = ink2m_bf16_4(q2.zw);
        for (uint m = 0u; m < p.M; ++m) {
          const threadgroup float* xp = xs[m] + kl;
          acc2[m] += dot(*(const threadgroup float4*)(xp + 0u), wa) + dot(*(const threadgroup float4*)(xp + 4u), wb);
        }
      }
      if (mc > 3u) {
        const float4 wa = ink2m_bf16_4(q3.xy), wb = ink2m_bf16_4(q3.zw);
        for (uint m = 0u; m < p.M; ++m) {
          const threadgroup float* xp = xs[m] + kl;
          acc3[m] += dot(*(const threadgroup float4*)(xp + 0u), wa) + dot(*(const threadgroup float4*)(xp + 4u), wb);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!valid) return;
  device float* bases[3] = {y0, y1, y2};
  device float* dst_base = bases[seg] + (n0 - seg_off);
  for (uint m = 0u; m < p.M; ++m) {
    const float s0 = simd_sum(acc0[m]);
    const float s1 = mc > 1u ? simd_sum(acc1[m]) : 0.0f;
    const float s2 = mc > 2u ? simd_sum(acc2[m]) : 0.0f;
    const float s3 = mc > 3u ? simd_sum(acc3[m]) : 0.0f;
    if (lane == 0u) {
      device float* dst = dst_base + size_t(m) * seg_len;
      dst[0] = s0;
      if (mc > 1u) dst[1] = s1;
      if (mc > 2u) dst[2] = s2;
      if (mc > 3u) dst[3] = s3;
    }
  }
}

[[kernel]] void ink2_mtp_gate_up_swiglu(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                                        device float* out [[buffer(2)]], constant Ink2MtpMmP& p [[buffer(3)]],
                                        uint tg [[threadgroup_position_in_grid]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
  const uint j = tg * 4u + sg;
  if (j >= p.N) return;
  float ag[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float au[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const size_t woff_g = size_t(2u * j) * p.K;
  const size_t woff_u = woff_g + p.K;
  for (uint k = lane * 8u; k < p.K; k += 256u) {
    const uint4 qg = *(const device uint4*)(w + woff_g + k);
    const uint4 qu = *(const device uint4*)(w + woff_u + k);
    const float4 wga = ink2m_bf16_4(qg.xy), wgb = ink2m_bf16_4(qg.zw);
    const float4 wua = ink2m_bf16_4(qu.xy), wub = ink2m_bf16_4(qu.zw);
    for (uint m = 0; m < p.M; ++m) {
      const device float* xp = x + size_t(m) * p.x_ld + k;
      const float4 x0 = *(const device float4*)(xp + 0);
      const float4 x1 = *(const device float4*)(xp + 4);
      ag[m] += dot(x0, wga) + dot(x1, wgb);
      au[m] += dot(x0, wua) + dot(x1, wub);
    }
  }
  for (uint m = 0; m < p.M; ++m) {
    const float g = simd_sum(ag[m]);
    const float u = simd_sum(au[m]);
    if (lane == 0u) out[size_t(m) * p.y_ld + j] = (g / (1.0f + metal::precise::exp(-g))) * u;
  }
}

[[kernel]] void ink2_mtp_gate_up_swiglu8(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                                         device float* out [[buffer(2)]], constant Ink2MtpMmP& p [[buffer(3)]],
                                         uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[8][256];
  const uint j0 = (tg * 4u + sg) * 2u;
  const bool valid = j0 < p.N;
  const bool has1 = (j0 + 1u) < p.N;
  float ag0[8] = {0, 0, 0, 0, 0, 0, 0, 0}, au0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float ag1[8] = {0, 0, 0, 0, 0, 0, 0, 0}, au1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const size_t woff_g0 = size_t(2u * j0) * p.K;
  const size_t woff_u0 = woff_g0 + p.K;
  const size_t woff_g1 = woff_g0 + 2u * p.K;
  const size_t woff_u1 = woff_g0 + 3u * p.K;
  const uint total_v4 = p.M * 64u;
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    for (uint idx = tid; idx < total_v4; idx += 128u) {
      const uint m = idx >> 6u;
      const uint c4 = (idx & 63u) << 2u;
      *(threadgroup float4*)(xs[m] + c4) = *(const device float4*)(x + size_t(m) * p.x_ld + kbase + c4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
      const uint kl = lane * 8u;
      const uint4 qg0 = *(const device uint4*)(w + woff_g0 + kbase + kl);
      const uint4 qu0 = *(const device uint4*)(w + woff_u0 + kbase + kl);
      const uint4 qg1 = has1 ? *(const device uint4*)(w + woff_g1 + kbase + kl) : uint4(0);
      const uint4 qu1 = has1 ? *(const device uint4*)(w + woff_u1 + kbase + kl) : uint4(0);
      {
        const float4 wa = ink2m_bf16_4(qg0.xy), wb = ink2m_bf16_4(qg0.zw);
        for (uint m = 0u; m < p.M; ++m) {
          const threadgroup float* xp = xs[m] + kl;
          ag0[m] += dot(*(const threadgroup float4*)(xp + 0u), wa) + dot(*(const threadgroup float4*)(xp + 4u), wb);
        }
      }
      {
        const float4 wa = ink2m_bf16_4(qu0.xy), wb = ink2m_bf16_4(qu0.zw);
        for (uint m = 0u; m < p.M; ++m) {
          const threadgroup float* xp = xs[m] + kl;
          au0[m] += dot(*(const threadgroup float4*)(xp + 0u), wa) + dot(*(const threadgroup float4*)(xp + 4u), wb);
        }
      }
      if (has1) {
        {
          const float4 wa = ink2m_bf16_4(qg1.xy), wb = ink2m_bf16_4(qg1.zw);
          for (uint m = 0u; m < p.M; ++m) {
            const threadgroup float* xp = xs[m] + kl;
            ag1[m] += dot(*(const threadgroup float4*)(xp + 0u), wa) + dot(*(const threadgroup float4*)(xp + 4u), wb);
          }
        }
        {
          const float4 wa = ink2m_bf16_4(qu1.xy), wb = ink2m_bf16_4(qu1.zw);
          for (uint m = 0u; m < p.M; ++m) {
            const threadgroup float* xp = xs[m] + kl;
            au1[m] += dot(*(const threadgroup float4*)(xp + 0u), wa) + dot(*(const threadgroup float4*)(xp + 4u), wb);
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!valid) return;
  for (uint m = 0u; m < p.M; ++m) {
    const float g0 = simd_sum(ag0[m]), u0 = simd_sum(au0[m]);
    const float g1 = has1 ? simd_sum(ag1[m]) : 0.0f;
    const float u1 = has1 ? simd_sum(au1[m]) : 0.0f;
    if (lane == 0u) {
      out[size_t(m) * p.y_ld + j0] = (g0 / (1.0f + metal::precise::exp(-g0))) * u0;
      if (has1) out[size_t(m) * p.y_ld + j0 + 1u] = (g1 / (1.0f + metal::precise::exp(-g1))) * u1;
    }
  }
}

struct Ink2MtpQ8FoldP {
  uint V, R;
  float softcap;
};

[[kernel]] void ink2_mtp_q8_fold_mv(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                                    const device half* ws [[buffer(2)]], const device float* bias [[buffer(3)]],
                                    device float* y [[buffer(4)]], constant Ink2MtpQ8FoldP& p [[buffer(5)]],
                                    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sh_x[256];
  for (uint k = tid; k < p.R; k += 128u) sh_x[k] = x[k];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint v = (tg * 4u + sg) * 4u + (lane >> 3);
  const uint blk = lane & 7u;
  float acc = 0.0f;
  if (v < p.V) {
    const size_t woff = size_t(v) * p.R + blk * 32u;
    const uint4 q0 = *(const device uint4*)(wq + woff);
    const uint4 q1 = *(const device uint4*)(wq + woff + 16u);
    const float sc = float(ws[size_t(v) * (p.R >> 5) + blk]);
    const threadgroup float4* txp = (const threadgroup float4*)(sh_x + blk * 32u);
    acc = (dot(txp[0], float4(as_type<char4>(q0.x))) + dot(txp[1], float4(as_type<char4>(q0.y))) +
           dot(txp[2], float4(as_type<char4>(q0.z))) + dot(txp[3], float4(as_type<char4>(q0.w))) +
           dot(txp[4], float4(as_type<char4>(q1.x))) + dot(txp[5], float4(as_type<char4>(q1.y))) +
           dot(txp[6], float4(as_type<char4>(q1.z))) + dot(txp[7], float4(as_type<char4>(q1.w)))) *
          sc;
  }
  acc += simd_shuffle_xor(acc, 1u);
  acc += simd_shuffle_xor(acc, 2u);
  acc += simd_shuffle_xor(acc, 4u);
  if (blk == 0u && v < p.V) {
    float val = y[v] + acc + bias[v];
    if (p.softcap > 0.0f) val = p.softcap * metal::precise::tanh(val / p.softcap);
    y[v] = val;
  }
}

// ---- 5. QK RMSNorm + Split-Half NeoX RoPE (mode=0: context commit K/V, mode=1: draft block Q/K) ----
struct Ink2MtpQkP {
  uint T, Hq, Hkv, D, pos0, kv_cap, is_swa, mode;
  float eps, theta;
};

[[kernel]] void ink2_mtp_qk_rope_cache(device float* q [[buffer(0)]], device float* k [[buffer(1)]],
                                       const device float* v [[buffer(2)]], const device ushort* q_norm [[buffer(3)]],
                                       const device ushort* k_norm [[buffer(4)]], device half* kc [[buffer(5)]],
                                       device half* vc [[buffer(6)]], constant Ink2MtpQkP& p [[buffer(7)]],
                                       uint2 tg [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  threadgroup float sh[128];
  const uint t = tg.y;
  const uint pos = p.pos0 + t;
  const uint half_d = p.D >> 1;
  const uint pair = tid < half_d ? tid : (tid - half_d);
  const float inv_freq = metal::precise::exp(-float(2u * pair) / float(p.D) * metal::precise::log(p.theta));
  const float ang = float(pos) * inv_freq;
  const float ca = metal::precise::cos(ang), sa = metal::precise::sin(ang);

  if (p.mode == 0u) {
    const uint h = tg.x;
    const size_t off = (size_t(t) * p.Hkv + h) * p.D + tid;
    const float kv = k[off];
    const float ss = tg_sum(kv * kv, scratch, tid, p.D);
    sh[tid] = kv * metal::precise::rsqrt(ss / float(p.D) + p.eps) * ink2m_bf16(k_norm[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float cur = sh[tid];
    const float partner = sh[tid < half_d ? (tid + half_d) : (tid - half_d)];
    const float rot = tid < half_d ? (cur * ca - partner * sa) : (partner * sa + cur * ca);
    const uint slot = p.is_swa ? (pos % p.kv_cap) : pos;
    const size_t dst = (size_t(slot) * p.Hkv + h) * p.D + tid;
    kc[dst] = half(rot);
    vc[dst] = half(v[off]);
    return;
  }

  if (tg.x < p.Hq) {
    const uint h = tg.x;
    const size_t off = (size_t(t) * p.Hq + h) * p.D + tid;
    const float qv = q[off];
    const float ss = tg_sum(qv * qv, scratch, tid, p.D);
    sh[tid] = qv * metal::precise::rsqrt(ss / float(p.D) + p.eps) * ink2m_bf16(q_norm[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float cur = sh[tid];
    const float partner = sh[tid < half_d ? (tid + half_d) : (tid - half_d)];
    q[off] = tid < half_d ? (cur * ca - partner * sa) : (partner * sa + cur * ca);
  } else {
    const uint h = tg.x - p.Hq;
    const size_t off = (size_t(t) * p.Hkv + h) * p.D + tid;
    const float kv = k[off];
    const float ss = tg_sum(kv * kv, scratch, tid, p.D);
    sh[tid] = kv * metal::precise::rsqrt(ss / float(p.D) + p.eps) * ink2m_bf16(k_norm[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float cur = sh[tid];
    const float partner = sh[tid < half_d ? (tid + half_d) : (tid - half_d)];
    k[off] = tid < half_d ? (cur * ca - partner * sa) : (partner * sa + cur * ca);
  }
}

// ---- 6. Bidirectional Draft Attention over context KV [k_start..P-1] + block KV [0..B-1] ----
struct Ink2MtpAttnP {
  uint B, P, kv_cap, Hq, Hkv, D, is_swa, swa;
  float scale;
};

[[kernel]] void ink2_mtp_attn(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                              const device half* vc [[buffer(2)]], const device float* k_blk [[buffer(3)]],
                              const device float* v_blk [[buffer(4)]], device float* out [[buffer(5)]],
                              constant Ink2MtpAttnP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgm[4], tgl[4], tgo[4][128];
  const uint h = tg.x, b = tg.y;
  const uint hkv = h / (p.Hq / p.Hkv);
  const uint d0 = lane * 4;
  const size_t qoff = (size_t(b) * p.Hq + h) * p.D + d0;
  const float4 qv = *(const device float4*)(q + qoff);
  const uint k_start = (p.is_swa && p.P > p.swa) ? (p.P - p.swa) : 0u;

  float mw = -INFINITY, lw = 0.0f;
  float4 ow = float4(0.0f);
  for (uint kp = k_start + sg; kp < p.P; kp += 4) {
    const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
    const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = float4(*(const device half4*)(kc + koff));
    const float sc = simd_sum(dot(qv, kv)) * p.scale;
    const float4 vv = float4(*(const device half4*)(vc + koff));
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
  for (uint kb = sg; kb < p.B; kb += 4) {
    const size_t koff = (size_t(kb) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = *(const device float4*)(k_blk + koff);
    const float sc = simd_sum(dot(qv, kv)) * p.scale;
    const float4 vv = *(const device float4*)(v_blk + koff);
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
    *(device float4*)(out + qoff) = oo / ll;
  }
}

struct Ink2MtpAttnSplitP {
  uint B, P, kv_cap, Hq, Hkv, D, is_swa, swa, chunk, nchunks;
  float scale;
};

[[kernel]] void ink2_mtp_attn_splitk(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                     const device half* vc [[buffer(2)]], const device float* k_blk [[buffer(3)]],
                                     const device float* v_blk [[buffer(4)]], device float* pm [[buffer(5)]],
                                     device float* pl [[buffer(6)]], device float* po [[buffer(7)]],
                                     constant Ink2MtpAttnSplitP& p [[buffer(8)]],
                                     uint2 tg [[threadgroup_position_in_grid]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgm[4], tgl[4], tgo[4][128];
  const uint h = tg.x, cid = tg.y;
  const uint hkv = h / (p.Hq / p.Hkv);
  const uint d0 = lane * 4u;
  float4 qv[8], ow[8];
  float mw[8], lw[8];
  for (uint b = 0u; b < p.B; ++b) {
    qv[b] = *(const device float4*)(q + (size_t(b) * p.Hq + h) * p.D + d0);
    mw[b] = -INFINITY;
    lw[b] = 0.0f;
    ow[b] = float4(0.0f);
  }
  const uint k_start = (p.is_swa && p.P > p.swa) ? (p.P - p.swa) : 0u;
  const uint c0 = k_start + cid * p.chunk;
  const uint c1 = min(p.P, c0 + p.chunk);
  for (uint kp = c0 + sg; kp < c1; kp += 4u) {
    const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
    const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = float4(*(const device half4*)(kc + koff));
    const float4 vv = float4(*(const device half4*)(vc + koff));
    for (uint b = 0u; b < p.B; ++b) {
      const float sc = simd_sum(dot(qv[b], kv)) * p.scale;
      if (lw[b] == 0.0f) {
        mw[b] = sc; lw[b] = 1.0f; ow[b] = vv;
      } else {
        const float nm = max(mw[b], sc);
        const float e1 = metal::precise::exp(mw[b] - nm), e2 = metal::precise::exp(sc - nm);
        ow[b] = ow[b] * e1 + vv * e2;
        lw[b] = lw[b] * e1 + e2;
        mw[b] = nm;
      }
    }
  }
  if (cid == p.nchunks - 1u) {
    for (uint kb = sg; kb < p.B; kb += 4u) {
      const size_t koff = (size_t(kb) * p.Hkv + hkv) * p.D + d0;
      const float4 kv = *(const device float4*)(k_blk + koff);
      const float4 vv = *(const device float4*)(v_blk + koff);
      for (uint b = 0u; b < p.B; ++b) {
        const float sc = simd_sum(dot(qv[b], kv)) * p.scale;
        if (lw[b] == 0.0f) {
          mw[b] = sc; lw[b] = 1.0f; ow[b] = vv;
        } else {
          const float nm = max(mw[b], sc);
          const float e1 = metal::precise::exp(mw[b] - nm), e2 = metal::precise::exp(sc - nm);
          ow[b] = ow[b] * e1 + vv * e2;
          lw[b] = lw[b] * e1 + e2;
          mw[b] = nm;
        }
      }
    }
  }
  for (uint b = 0u; b < p.B; ++b) {
    if (lane == 0u) {
      tgm[sg] = mw[b];
      tgl[sg] = lw[b];
    }
    *(threadgroup float4*)(&tgo[sg][d0]) = ow[b];
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
      const size_t out_idx = (size_t(b) * p.Hq + h) * p.nchunks + cid;
      if (lane == 0u) {
        pm[out_idx] = mm;
        pl[out_idx] = ll;
      }
      *(device float4*)(po + out_idx * p.D + d0) = oo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

[[kernel]] void ink2_mtp_attn_reduce(const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]],
                                     const device float* po [[buffer(2)]], device float* out [[buffer(3)]],
                                     constant Ink2MtpAttnSplitP& p [[buffer(4)]],
                                     uint2 tg [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint h = tg.x, b = tg.y, d0 = lane * 4u;
  float mm = -INFINITY, ll = 0.0f;
  float4 oo = float4(0.0f);
  const size_t row_base = (size_t(b) * p.Hq + h) * p.nchunks;
  for (uint cid = 0u; cid < p.nchunks; ++cid) {
    const size_t idx = row_base + cid;
    const float l2 = pl[idx];
    if (l2 == 0.0f) continue;
    const float m2 = pm[idx];
    const float4 o2 = *(const device float4*)(po + idx * p.D + d0);
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
  *(device float4*)(out + (size_t(b) * p.Hq + h) * p.D + d0) = ll > 0.0f ? (oo / ll) : float4(0.0f);
}

[[kernel]] void ink2_mtp_attn_splitk_gqa(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                         const device half* vc [[buffer(2)]], const device float* k_blk [[buffer(3)]],
                                         const device float* v_blk [[buffer(4)]], device float* pm [[buffer(5)]],
                                         device float* pl [[buffer(6)]], device float* po [[buffer(7)]],
                                         constant Ink2MtpAttnSplitP& p [[buffer(8)]],
                                         uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half sh_k[8][128], sh_v[8][128];
  threadgroup float tgm[3][2], tgl[3][2], tgo[3][2][128];
  const uint hkv = tg.x, cid = tg.y, b0 = tg.z * 4u;
  const uint mc = min(4u, p.B - b0);
  const uint g = sg % 3u, w = sg / 3u;
  const uint h = hkv * 3u + g, d0 = lane * 4u;
  float4 qv[4], ow[4];
  float mw[4], lw[4];
  for (uint b = 0u; b < mc; ++b) {
    qv[b] = *(const device float4*)(q + (size_t(b0 + b) * p.Hq + h) * p.D + d0);
    mw[b] = -INFINITY;
    lw[b] = 0.0f;
    ow[b] = float4(0.0f);
  }
  const uint k_start = (p.is_swa && p.P > p.swa) ? (p.P - p.swa) : 0u;
  const uint c0 = k_start + cid * p.chunk;
  const uint c1 = min(p.P, c0 + p.chunk);
  for (uint kbase = c0; kbase < c1; kbase += 8u) {
    const uint nvalid = min(8u, c1 - kbase);
    for (uint idx = tid; idx < nvalid * 32u; idx += 192u) {
      const uint ik = idx >> 5u, c4 = (idx & 31u) << 2u;
      const uint kp = kbase + ik;
      const uint slot = p.is_swa ? (kp % p.kv_cap) : kp;
      const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + c4;
      *(threadgroup half4*)(sh_k[ik] + c4) = *(const device half4*)(kc + koff);
      *(threadgroup half4*)(sh_v[ik] + c4) = *(const device half4*)(vc + koff);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = w; i < nvalid; i += 2u) {
      const float4 kv = float4(*(const threadgroup half4*)(sh_k[i] + d0));
      const float4 vv = float4(*(const threadgroup half4*)(sh_v[i] + d0));
      for (uint b = 0u; b < mc; ++b) {
        const float sc = simd_sum(dot(qv[b], kv)) * p.scale;
        if (lw[b] == 0.0f) {
          mw[b] = sc; lw[b] = 1.0f; ow[b] = vv;
        } else {
          const float nm = max(mw[b], sc);
          const float e1 = metal::precise::exp(mw[b] - nm), e2 = metal::precise::exp(sc - nm);
          ow[b] = ow[b] * e1 + vv * e2;
          lw[b] = lw[b] * e1 + e2;
          mw[b] = nm;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (cid == p.nchunks - 1u) {
    for (uint kb = w; kb < p.B; kb += 2u) {
      const size_t koff = (size_t(kb) * p.Hkv + hkv) * p.D + d0;
      const float4 kv = *(const device float4*)(k_blk + koff);
      const float4 vv = *(const device float4*)(v_blk + koff);
      for (uint b = 0u; b < mc; ++b) {
        const float sc = simd_sum(dot(qv[b], kv)) * p.scale;
        if (lw[b] == 0.0f) {
          mw[b] = sc; lw[b] = 1.0f; ow[b] = vv;
        } else {
          const float nm = max(mw[b], sc);
          const float e1 = metal::precise::exp(mw[b] - nm), e2 = metal::precise::exp(sc - nm);
          ow[b] = ow[b] * e1 + vv * e2;
          lw[b] = lw[b] * e1 + e2;
          mw[b] = nm;
        }
      }
    }
  }
  for (uint b = 0u; b < mc; ++b) {
    if (lane == 0u) {
      tgm[g][w] = mw[b];
      tgl[g][w] = lw[b];
    }
    *(threadgroup float4*)(&tgo[g][w][d0]) = ow[b];
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
      const size_t out_idx = (size_t(b0 + b) * p.Hq + h) * p.nchunks + cid;
      if (lane == 0u) {
        pm[out_idx] = mm;
        pl[out_idx] = ll;
      }
      *(device float4*)(po + out_idx * p.D + d0) = oo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ---- 7. SwiGLU & BF16-rounded residual sum ----
struct Ink2MtpVecP {
  uint n;
};

[[kernel]] void ink2_mtp_swiglu_mul(device float* g [[buffer(0)]], const device float* u [[buffer(1)]],
                                    constant Ink2MtpVecP& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  const uint idx = i * 4;
  if (idx + 3 < p.n) {
    const float4 gv = *(const device float4*)(g + idx);
    const float4 uv = *(const device float4*)(u + idx);
    const float4 s = float4(gv.x / (1.0f + metal::precise::exp(-gv.x)),
                            gv.y / (1.0f + metal::precise::exp(-gv.y)),
                            gv.z / (1.0f + metal::precise::exp(-gv.z)),
                            gv.w / (1.0f + metal::precise::exp(-gv.w)));
    *(device float4*)(g + idx) = s * uv;
  } else {
    for (uint q = idx; q < p.n; ++q) {
      const float gv = g[q];
      g[q] = (gv / (1.0f + metal::precise::exp(-gv))) * u[q];
    }
  }
}

[[kernel]] void ink2_mtp_res_add(const device float* res [[buffer(0)]], const device float* upd [[buffer(1)]],
                                 device float* out [[buffer(2)]], constant Ink2MtpVecP& p [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
  const uint idx = i * 4;
  if (idx + 3 < p.n) {
    const float4 r = *(const device float4*)(res + idx);
    const float4 u = *(const device float4*)(upd + idx);
    const float4 rb(ink2m_round_bf16(r.x), ink2m_round_bf16(r.y), ink2m_round_bf16(r.z), ink2m_round_bf16(r.w));
    *(device float4*)(out + idx) = rb + u;
  } else {
    for (uint q = idx; q < p.n; ++q) out[q] = ink2m_round_bf16(res[q]) + upd[q];
  }
}

// ---- 8. Markov RNN selector prep & update ----
struct Ink2MtpRnnPrepP {
  uint rank, H, tok_pos, first_step;
};

[[kernel]] void ink2_mtp_rnn_prep(const device float* state [[buffer(0)]], const device ushort* markov_w1 [[buffer(1)]],
                                  const device int* tokens [[buffer(2)]], const device float* hidden [[buffer(3)]],
                                  const device float* feat [[buffer(4)]], device float* rnn_in [[buffer(5)]],
                                  constant Ink2MtpRnnPrepP& p [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  const uint tot = 2u * p.rank + 2u * p.H;
  if (i >= tot) return;
  if (i < p.rank) {
    rnn_in[i] = p.first_step ? 0.0f : state[i];
  } else if (i < 2u * p.rank) {
    const int id = tokens[p.tok_pos];
    rnn_in[i] = ink2m_bf16(markov_w1[size_t(id) * p.rank + (i - p.rank)]);
  } else if (i < 2u * p.rank + p.H) {
    rnn_in[i] = hidden[i - 2u * p.rank];
  } else {
    rnn_in[i] = feat[i - (2u * p.rank + p.H)];
  }
}

struct Ink2MtpRnnUpdP {
  uint rank, first_step;
};

[[kernel]] void ink2_mtp_rnn_update(const device float* gates [[buffer(0)]], device float* state [[buffer(1)]],
                                    device float* latent [[buffer(2)]], constant Ink2MtpRnnUpdP& p [[buffer(3)]],
                                    uint c [[thread_position_in_grid]]) {
  if (c >= p.rank) return;
  const float prev_s = p.first_step ? 0.0f : state[c];
  const float k = 1.0f / (1.0f + metal::precise::exp(-gates[c]));
  const float u = metal::precise::tanh(gates[p.rank + c]);
  const float l = metal::precise::tanh(gates[2u * p.rank + c]);
  state[c] = k * prev_s + (1.0f - k) * u;
  latent[c] = l;
}

// ---- 9. Load-time Folded Head: fold_w [V, R] = BF16(unembed [V, H] @ latent_w_t [R, H]^T) ----
struct Ink2MtpFoldP {
  uint V, R, H;
};

[[kernel]] void ink2_mtp_fold_head(const device ushort* unembed [[buffer(0)]],
                                   const device ushort* latent_wt [[buffer(1)]], device ushort* fold_w [[buffer(2)]],
                                   constant Ink2MtpFoldP& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint v = tg.x * 4u + sg;
  const uint r0 = tg.y * 16u;
  if (v >= p.V || r0 >= p.R) return;
  float acc[16];
  for (uint j = 0; j < 16u; ++j) acc[j] = 0.0f;
  const device ushort* up = unembed + size_t(v) * p.H;
  for (uint k = lane * 4u; k < p.H; k += 128u) {
    const float4 uv = ink2m_bf16_4(*(const device uint2*)(up + k));
    for (uint j = 0; j < 16u && (r0 + j) < p.R; ++j) {
      const float4 wv = ink2m_bf16_4(*(const device uint2*)(latent_wt + size_t(r0 + j) * p.H + k));
      acc[j] += dot(uv, wv);
    }
  }
  for (uint j = 0; j < 16u && (r0 + j) < p.R; ++j) {
    const float s = simd_sum(acc[j]);
    if (lane == 0u) fold_w[size_t(v) * p.R + (r0 + j)] = ink2m_f32_to_bf16(s);
  }
}

// ---- 10. Multi-row greedy argmax (R <= 8 rows, exact lowest-ID tie-break) ----
struct Ink2MtpArgmaxP {
  uint V;
};

[[kernel]] void ink2_mtp_argmax_rows(const device float* logits [[buffer(0)]], device int* out [[buffer(1)]],
                                     constant Ink2MtpArgmaxP& p [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  const device float* src = logits + size_t(row) * p.V;
  float bv = -INFINITY;
  uint bi = 0xffffffffu;
  for (uint i = tid; i < p.V; i += ntg) {
    const float v = src[i];
    if (v > bv || (v == bv && i < bi)) {
      bv = v;
      bi = i;
    }
  }
  float mv = simd_max(bv);
  uint mi = simd_min(bv == mv ? bi : 0xffffffffu);
  const uint sg = tid >> 5, lane = tid & 31, nsg = (ntg + 31) >> 5;
  if (lane == 0) {
    sv[sg] = mv;
    si[sg] = mi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    mv = lane < nsg ? sv[lane] : -INFINITY;
    mi = lane < nsg ? si[lane] : 0xffffffffu;
    const float gv = simd_max(mv);
    const uint gi = simd_min(mv == gv ? mi : 0xffffffffu);
    if (lane == 0) out[row] = int(gi);
  }
}

[[kernel]] void ink2_mtp_argmax_stage1(const device float* logits [[buffer(0)]], device float* cand_v [[buffer(1)]],
                                       device uint* cand_i [[buffer(2)]], constant Ink2MtpArgmaxP& p [[buffer(3)]],
                                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint ntg = 256u, nsg = 8u;
  threadgroup float sv[8];
  threadgroup uint si[8];
  const uint row = tg.y, bid = tg.x, nblk = 64u;
  const device float* src = logits + size_t(row) * p.V;
  float bv = -INFINITY;
  uint bi = 0xffffffffu;
  for (uint i = bid * ntg + tid; i < p.V; i += nblk * ntg) {
    const float v = src[i];
    if (v > bv || (v == bv && i < bi)) {
      bv = v;
      bi = i;
    }
  }
  const float mv = simd_max(bv);
  const uint mi = simd_min(bv == mv ? bi : 0xffffffffu);
  const uint sg = tid >> 5, lane = tid & 31;
  if (lane == 0u) {
    sv[sg] = mv;
    si[sg] = mi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    const float m2 = lane < nsg ? sv[lane] : -INFINITY;
    const uint i2 = lane < nsg ? si[lane] : 0xffffffffu;
    const float gv = simd_max(m2);
    const uint gi = simd_min(m2 == gv ? i2 : 0xffffffffu);
    if (lane == 0u) {
      cand_v[size_t(row) * 64u + bid] = gv;
      cand_i[size_t(row) * 64u + bid] = gi;
    }
  }
}

[[kernel]] void ink2_mtp_argmax_stage2(const device float* cand_v [[buffer(0)]], const device uint* cand_i [[buffer(1)]],
                                       device int* out [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sv[2];
  threadgroup uint si[2];
  const size_t base = size_t(row) * 64u;
  const float v = cand_v[base + tid];
  const uint idx = cand_i[base + tid];
  const float mv = simd_max(v);
  const uint mi = simd_min(v == mv ? idx : 0xffffffffu);
  const uint sg = tid >> 5, lane = tid & 31;
  if (lane == 0u) {
    sv[sg] = mv;
    si[sg] = mi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    const float m2 = lane < 2u ? sv[lane] : -INFINITY;
    const uint i2 = lane < 2u ? si[lane] : 0xffffffffu;
    const float gv = simd_max(m2);
    const uint gi = simd_min(m2 == gv ? i2 : 0xffffffffu);
    if (lane == 0u) out[row] = int(gi);
  }
}

struct Ink2QuantQ8P {
  uint total_blocks;
};

[[kernel]] void ink2_quant_q8_bf16(const device bfloat* w [[buffer(0)]], device char* q [[buffer(1)]],
                                     device half* s [[buffer(2)]], constant Ink2QuantQ8P& p [[buffer(3)]],
                                     uint tg [[threadgroup_position_in_grid]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint blk = tg * 4u + sg;
  if (blk >= p.total_blocks) return;
  const uint idx = blk * 32u + lane;
  const float v = float(w[idx]);
  const float amax = simd_max(metal::fabs(v));
  const float d = amax * (1.0f / 127.0f);
  const float id = d > 0.0f ? (1.0f / d) : 0.0f;
  const int qi = clamp(int(metal::rint(v * id)), -127, 127);
  q[idx] = char(qi);
  if (lane == 0u) s[blk] = half(d);
}

// ---- 8. Quantization & 2-Bit Proxy Screening + Quantized DSpark 3L ----
struct Ink2QuantP2P {
  uint total_blocks, w_f32;
};

[[kernel]] void ink2_quant_p2_w(const device void* w [[buffer(0)]], device uint2* q [[buffer(1)]],
                                  device half* s [[buffer(2)]], constant Ink2QuantP2P& p [[buffer(3)]],
                                  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  const uint blk = tg * 4u + sg;
  if (blk >= p.total_blocks) return;
  const uint idx = blk * 32u + lane;
  const float v = p.w_f32 ? ((const device float*)w)[idx] : ink2m_bf16(((const device ushort*)w)[idx]);
  const float alpha = (simd_sum(metal::fabs(v)) * (1.0f / 32.0f)) * 1.25f;
  const uint c = (v >= alpha) ? 3u : ((v >= 0.0f) ? 2u : ((v >= -alpha) ? 1u : 0u));
  const uint shift = 2u * (lane & 15u);
  const uint b0 = (lane < 16u ? c : 0u) << shift;
  const uint b1 = (lane >= 16u ? c : 0u) << shift;
  const uint w0 = simd_sum(b0);
  const uint w1 = simd_sum(b1);
  if (lane == 0u) {
    q[blk] = uint2(w0, w1);
    s[blk] = half(alpha);
  }
}

[[kernel]] void ink2_quant_q4_bf16(const device ushort* w [[buffer(0)]], device uint4* q [[buffer(1)]],
                                     device half* s [[buffer(2)]], constant Ink2QuantQ8P& p [[buffer(3)]],
                                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint blk = tg * 4u + sg;
  if (blk >= p.total_blocks) return;
  const uint idx = blk * 32u + lane;
  const float v = ink2m_bf16(w[idx]);
  const float amax = simd_max(metal::fabs(v));
  const float d = amax * (1.0f / 7.0f);
  const float id = d > 0.0f ? (1.0f / d) : 0.0f;
  const uint qi = uint(clamp(int(metal::rint(v * id)), -8, 7) + 8);
  const uint wsub = lane >> 3u;
  const uint sh = (lane & 7u) * 4u;
  const uint r0 = simd_sum((wsub == 0u ? qi : 0u) << sh);
  const uint r1 = simd_sum((wsub == 1u ? qi : 0u) << sh);
  const uint r2 = simd_sum((wsub == 2u ? qi : 0u) << sh);
  const uint r3 = simd_sum((wsub == 3u ? qi : 0u) << sh);
  if (lane == 0u) {
    q[blk] = uint4(r0, r1, r2, r3);
    s[blk] = half(d);
  }
}

NI_INLINE float ink2m_p2_dot32(uint2 q, float4 x0, float4 x1, float4 x2, float4 x3,
                               float4 x4, float4 x5, float4 x6, float4 x7,
                               float blk_sum, float sc) {
  auto d4 = [&](float4 xv, uint bits) {
    const float4 cw = float4(float(bits & 3u), float((bits >> 2) & 3u),
                             float((bits >> 4) & 3u), float((bits >> 6) & 3u));
    return dot(xv, cw);
  };
  const float s0 = d4(x0, q.x) + d4(x1, q.x >> 8u) + d4(x2, q.x >> 16u) + d4(x3, q.x >> 24u);
  const float s1 = d4(x4, q.y) + d4(x5, q.y >> 8u) + d4(x6, q.y >> 16u) + d4(x7, q.y >> 24u);
  return ((s0 + s1) - 1.5f * blk_sum) * sc;
}

struct Ink2MtpP2BaseP {
  uint M, V, H, nblk;
};

template <uint M_MAX>
NI_INLINE void ink2_mtp_p2_unembed_base_impl(const device float* x, const device uint2* p2_q, const device half* p2_s,
                                             const device float* fold_b, device half* out,
                                             constant Ink2MtpP2BaseP& p, threadgroup float* xs,
                                             threadgroup float* xsum, uint tg, uint sg, uint lane) {
  const uint v = (tg * 4u + sg) * 4u + (lane >> 3u);
  const uint blk_loc = lane & 7u;
  const bool valid = v < p.V;
  float acc[M_MAX] = {};

  for (uint blk0 = 0u; blk0 < p.nblk; blk0 += 8u) {
    _Pragma("clang loop unroll(full)")
    for (uint m = 0u; m < M_MAX; ++m) {
      if (m < p.M) {
        const device float* xm = x + size_t(m) * p.H + blk0 * 32u;
        for (uint pass = 0u; pass < 2u; ++pass) {
          const uint b = sg + pass * 4u;
          const float xv = (blk0 + b < p.nblk) ? xm[b * 32u + lane] : 0.0f;
          xs[m * 256u + b * 32u + lane] = xv;
          const float s = simd_sum(xv);
          if (lane == 0u) xsum[m * 8u + b] = s;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint blk = blk0 + blk_loc;
    if (valid && blk < p.nblk) {
      const size_t idx = size_t(v) * p.nblk + blk;
      const uint2 qv = p2_q[idx];
      const float sc = float(p2_s[idx]);
      const float4 c0 = float4(float(qv.x & 3u), float((qv.x >> 2u) & 3u), float((qv.x >> 4u) & 3u), float((qv.x >> 6u) & 3u));
      const float4 c1 = float4(float((qv.x >> 8u) & 3u), float((qv.x >> 10u) & 3u), float((qv.x >> 12u) & 3u), float((qv.x >> 14u) & 3u));
      const float4 c2 = float4(float((qv.x >> 16u) & 3u), float((qv.x >> 18u) & 3u), float((qv.x >> 20u) & 3u), float((qv.x >> 22u) & 3u));
      const float4 c3 = float4(float((qv.x >> 24u) & 3u), float((qv.x >> 26u) & 3u), float((qv.x >> 28u) & 3u), float(qv.x >> 30u));
      const float4 c4 = float4(float(qv.y & 3u), float((qv.y >> 2u) & 3u), float((qv.y >> 4u) & 3u), float((qv.y >> 6u) & 3u));
      const float4 c5 = float4(float((qv.y >> 8u) & 3u), float((qv.y >> 10u) & 3u), float((qv.y >> 12u) & 3u), float((qv.y >> 14u) & 3u));
      const float4 c6 = float4(float((qv.y >> 16u) & 3u), float((qv.y >> 18u) & 3u), float((qv.y >> 20u) & 3u), float((qv.y >> 22u) & 3u));
      const float4 c7 = float4(float((qv.y >> 24u) & 3u), float((qv.y >> 26u) & 3u), float((qv.y >> 28u) & 3u), float(qv.y >> 30u));
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < M_MAX; ++m) {
        if (m < p.M) {
          const threadgroup float4* xmp = (const threadgroup float4*)(xs + m * 256u + blk_loc * 32u);
          const float s = dot(xmp[0], c0) + dot(xmp[1], c1) + dot(xmp[2], c2) + dot(xmp[3], c3) +
                          dot(xmp[4], c4) + dot(xmp[5], c5) + dot(xmp[6], c6) + dot(xmp[7], c7);
          acc[m] += (s - 1.5f * xsum[m * 8u + blk_loc]) * sc;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  _Pragma("clang loop unroll(full)")
  for (uint m = 0u; m < M_MAX; ++m) {
    if (m >= p.M) break;
    float vdot = acc[m];
    vdot += simd_shuffle_xor(vdot, 1u);
    vdot += simd_shuffle_xor(vdot, 2u);
    vdot += simd_shuffle_xor(vdot, 4u);
    if (blk_loc == 0u && valid) {
      out[size_t(m) * p.V + v] = half(vdot + fold_b[v]);
    }
  }
}

[[kernel]] void ink2_mtp_p2_unembed_base(const device float* x [[buffer(0)]], const device uint2* p2_q [[buffer(1)]],
                                         const device half* p2_s [[buffer(2)]], const device float* fold_b [[buffer(3)]],
                                         device half* out [[buffer(4)]], constant Ink2MtpP2BaseP& p [[buffer(5)]],
                                         uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[7 * 256];
  threadgroup float xsum[7 * 8];
  if (p.M <= 4u) ink2_mtp_p2_unembed_base_impl<4u>(x, p2_q, p2_s, fold_b, out, p, xs, xsum, tg, sg, lane);
  else ink2_mtp_p2_unembed_base_impl<7u>(x, p2_q, p2_s, fold_b, out, p, xs, xsum, tg, sg, lane);
}

struct Ink2MtpP2ScreenP {
  uint V, R, has_sup, sup_words;
};

[[kernel]] void ink2_mtp_p2_fold_screen(const device float* x [[buffer(0)]], const device uint2* fold_p2_q [[buffer(1)]],
                                        const device half* fold_p2_s [[buffer(2)]], const device half* base_proxy [[buffer(3)]],
                                        const device uint* sup_mask [[buffer(4)]], device float* cand_v [[buffer(5)]],
                                        device uint* cand_i [[buffer(6)]], constant Ink2MtpP2ScreenP& p [[buffer(7)]],
                                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sh_x[256];
  threadgroup float sh_sum[8];
  threadgroup float sh_v[32];
  threadgroup uint sh_i[32];

  for (uint pass = 0u; pass < 2u; ++pass) {
    const uint b = sg + pass * 4u;
    const float xv = x[b * 32u + lane];
    sh_x[b * 32u + lane] = xv;
    const float s = simd_sum(xv);
    if (lane == 0u) sh_sum[b] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float b0v = -INFINITY, b1v = -INFINITY;
  uint b0i = 0xffffffffu, b1i = 0xffffffffu;
  const uint ntg_row = (p.V + 15u) / 16u;
  const uint blk = lane & 7u;
  const threadgroup float4* xmp = (const threadgroup float4*)(sh_x + blk * 32u);
  const float x0s = sh_sum[blk];

  for (uint g = tg; g < ntg_row; g += 64u) {
    const uint v = (g * 4u + sg) * 4u + (lane >> 3u);
    const bool valid = v < p.V;
    float acc = 0.0f;
    if (valid) {
      const size_t idx = size_t(v) * 8u + blk;
      acc = ink2m_p2_dot32(fold_p2_q[idx], xmp[0], xmp[1], xmp[2], xmp[3], xmp[4], xmp[5], xmp[6], xmp[7], x0s,
                           float(fold_p2_s[idx]));
    }
    acc += simd_shuffle_xor(acc, 1u);
    acc += simd_shuffle_xor(acc, 2u);
    acc += simd_shuffle_xor(acc, 4u);
    if (blk == 0u && valid) {
      float val = float(base_proxy[v]) + acc;
      if (p.has_sup) {
        const uint widx = v >> 5u;
        if (widx < p.sup_words && ((sup_mask[widx] >> (v & 31u)) & 1u) == 0u) val = -INFINITY;
      }
      if (val > b0v || (val == b0v && v < b0i)) {
        b1v = b0v; b1i = b0i;
        b0v = val; b0i = v;
      } else if (val > b1v || (val == b1v && v < b1i)) {
        b1v = val; b1i = v;
      }
    }
  }

  if (blk == 0u) {
    const uint slot = sg * 4u + (lane >> 3u);
    sh_v[slot * 2u + 0u] = b0v;
    sh_i[slot * 2u + 0u] = b0i;
    sh_v[slot * 2u + 1u] = b1v;
    sh_i[slot * 2u + 1u] = b1i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    float t0v = -INFINITY, t1v = -INFINITY;
    uint t0i = 0xffffffffu, t1i = 0xffffffffu;
    for (uint j = 0u; j < 32u; ++j) {
      const float vv = sh_v[j];
      const uint vi = sh_i[j];
      if (vi == 0xffffffffu) continue;
      if (vv > t0v || (vv == t0v && vi < t0i)) {
        t1v = t0v; t1i = t0i;
        t0v = vv; t0i = vi;
      } else if (vv > t1v || (vv == t1v && vi < t1i)) {
        t1v = vv; t1i = vi;
      }
    }
    cand_v[tg * 2u + 0u] = t0v;
    cand_i[tg * 2u + 0u] = t0i;
    cand_v[tg * 2u + 1u] = t1v;
    cand_i[tg * 2u + 1u] = t1i;
  }
}

struct Ink2MtpRescoreP {
  uint V, H, R, u_f32, has_sup, sup_words;
  float softcap, inv_cap;
};

[[kernel]] void ink2_mtp_rescore_exact(const device float* x_u [[buffer(0)]], const device float* x_f [[buffer(1)]],
                                       const device void* unembed [[buffer(2)]], const device ushort* fold_w [[buffer(3)]],
                                       const device float* fold_b [[buffer(4)]], const device uint* sup_mask [[buffer(5)]],
                                       const device uint* cand_i [[buffer(6)]], device float* cand_v [[buffer(7)]],
                                       constant Ink2MtpRescoreP& p [[buffer(8)]], uint tg [[threadgroup_position_in_grid]],
                                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint c = tg * 4u + sg;
  const uint v = cand_i[c];
  if (v >= p.V) {
    if (lane == 0u) cand_v[c] = -INFINITY;
    return;
  }
  if (p.has_sup) {
    const uint widx = v >> 5u;
    if (widx < p.sup_words && ((sup_mask[widx] >> (v & 31u)) & 1u) == 0u) {
      if (lane == 0u) cand_v[c] = -INFINITY;
      return;
    }
  }
  float acc = 0.0f;
  const size_t uoff = size_t(v) * p.H;
  if (p.u_f32) {
    const device float* uw = (const device float*)unembed + uoff;
    for (uint k = lane * 4u; k < p.H; k += 128u) {
      acc += dot(*(const device float4*)(x_u + k), *(const device float4*)(uw + k));
    }
  } else {
    const device ushort* uw = (const device ushort*)unembed + uoff;
    for (uint k = lane * 8u; k < p.H; k += 256u) {
      const uint4 qu = *(const device uint4*)(uw + k);
      acc += dot(*(const device float4*)(x_u + k + 0u), ink2m_bf16_4(qu.xy)) +
             dot(*(const device float4*)(x_u + k + 4u), ink2m_bf16_4(qu.zw));
    }
  }
  const device ushort* fw = fold_w + size_t(v) * p.R;
  for (uint r = lane * 8u; r < p.R; r += 256u) {
    const uint4 qf = *(const device uint4*)(fw + r);
    acc += dot(*(const device float4*)(x_f + r + 0u), ink2m_bf16_4(qf.xy)) +
           dot(*(const device float4*)(x_f + r + 4u), ink2m_bf16_4(qf.zw));
  }
  float sum = simd_sum(acc);
  if (lane == 0u) {
    float val = sum + fold_b[v];
    if (p.softcap > 0.0f) val = p.softcap * metal::precise::tanh(val * p.inv_cap);
    cand_v[c] = val;
  }
}

[[kernel]] void ink2_mtp_argmax_128(const device float* cand_v [[buffer(0)]], const device uint* cand_i [[buffer(1)]],
                                    device int* out [[buffer(2)]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sv[4];
  threadgroup uint si[4];
  const float v = cand_v[tid];
  const uint idx = cand_i[tid];
  const float mv = simd_max(v);
  const uint mi = simd_min(v == mv ? idx : 0xffffffffu);
  const uint sg = tid >> 5u, lane = tid & 31u;
  if (lane == 0u) {
    sv[sg] = mv;
    si[sg] = mi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    const float m2 = lane < 4u ? sv[lane] : -INFINITY;
    const uint i2 = lane < 4u ? si[lane] : 0xffffffffu;
    const float gv = simd_max(m2);
    const uint gi = simd_min(m2 == gv ? i2 : 0xffffffffu);
    if (lane == 0u) out[0] = (gi == 0xffffffffu) ? 0 : int(gi);
  }
}

struct Ink2MtpQ8MmP {
  uint M, N, K, nblk, add;
};

struct Ink2MtpQ8Multi3P {
  uint M, K, N_tot, nblk;
  uint offs[4];
};

[[kernel]] void ink2_mtp_q8_mv8(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                                const device half* ws [[buffer(2)]], device float* y [[buffer(3)]],
                                constant Ink2MtpQ8MmP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4u + sg) * 2u;
  if (n0 >= p.N) return;
  const bool has1 = (n0 + 1u) < p.N;
  const size_t woff0 = size_t(n0) * p.K;
  const size_t woff1 = has1 ? (woff0 + p.K) : woff0;
  const size_t soff0 = size_t(n0) * p.nblk;
  const size_t soff1 = has1 ? (soff0 + p.nblk) : soff0;
  const uint kl = lane * 8u;
  const uint blk_loc = lane >> 2u;

  if (p.M == 1u) {
    float a0 = 0.0f, a1 = 0.0f;
    for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
      if (kl < p.K - kbase) {
        const uint blk = (kbase >> 5u) + blk_loc;
        const device float* xp = x + kbase + kl;
        const float4 x0 = *(const device float4*)(xp + 0u), x1 = *(const device float4*)(xp + 4u);
        const uint2 qw0 = *(const device uint2*)(wq + woff0 + kbase + kl);
        const float sc0 = float(ws[soff0 + blk]);
        a0 += (dot(x0, float4(as_type<char4>(qw0.x))) + dot(x1, float4(as_type<char4>(qw0.y)))) * sc0;
        if (has1) {
          const uint2 qw1 = *(const device uint2*)(wq + woff1 + kbase + kl);
          const float sc1 = float(ws[soff1 + blk]);
          a1 += (dot(x0, float4(as_type<char4>(qw1.x))) + dot(x1, float4(as_type<char4>(qw1.y)))) * sc1;
        }
      }
    }
    const float s0 = simd_sum(a0), s1 = simd_sum(a1);
    if (lane == 0u) {
      y[n0] = p.add ? (y[n0] + s0) : s0;
      if (has1) y[n0 + 1u] = p.add ? (y[n0 + 1u] + s1) : s1;
    }
    return;
  }

  float acc0[8] = {}, acc1[8] = {};
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    if (kl < p.K - kbase) {
      const uint blk = (kbase >> 5u) + blk_loc;
      const uint2 qw0 = *(const device uint2*)(wq + woff0 + kbase + kl);
      const float sc0 = float(ws[soff0 + blk]);
      const float4 w0a = float4(as_type<char4>(qw0.x)) * sc0, w0b = float4(as_type<char4>(qw0.y)) * sc0;
      const uint2 qw1 = has1 ? *(const device uint2*)(wq + woff1 + kbase + kl) : qw0;
      const float sc1 = has1 ? float(ws[soff1 + blk]) : sc0;
      const float4 w1a = float4(as_type<char4>(qw1.x)) * sc1, w1b = float4(as_type<char4>(qw1.y)) * sc1;
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < 8u; ++m) {
        if (m < p.M) {
          const device float* xp = x + size_t(m) * p.K + kbase + kl;
          const float4 x0 = *(const device float4*)(xp + 0u), x1 = *(const device float4*)(xp + 4u);
          acc0[m] += dot(x0, w0a) + dot(x1, w0b);
          acc1[m] += dot(x0, w1a) + dot(x1, w1b);
        }
      }
    }
  }
  _Pragma("clang loop unroll(full)")
  for (uint m = 0u; m < 8u; ++m) {
    if (m >= p.M) break;
    const float s0 = simd_sum(acc0[m]), s1 = simd_sum(acc1[m]);
    if (lane == 0u) {
      device float* dst = y + size_t(m) * p.N + n0;
      dst[0] = p.add ? (dst[0] + s0) : s0;
      if (has1) dst[1] = p.add ? (dst[1] + s1) : s1;
    }
  }
}

[[kernel]] void ink2_mtp_q8_mv8_multi3(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                                       const device half* ws [[buffer(2)]], device float* y0 [[buffer(3)]],
                                       device float* y1 [[buffer(4)]], device float* y2 [[buffer(5)]],
                                       constant Ink2MtpQ8Multi3P& p [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],
                                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4u + sg) * 2u;
  if (n0 >= p.N_tot) return;
  uint seg = 0u;
  while (seg < 2u && n0 >= p.offs[seg + 1u]) ++seg;
  const uint seg_off = p.offs[seg];
  const uint seg_end = p.offs[seg + 1u];
  const uint seg_len = seg_end - seg_off;
  if (n0 >= seg_end) return;
  const bool has1 = (n0 + 1u) < seg_end;
  const size_t woff0 = size_t(n0) * p.K;
  const size_t woff1 = has1 ? (woff0 + p.K) : woff0;
  const size_t soff0 = size_t(n0) * p.nblk;
  const size_t soff1 = has1 ? (soff0 + p.nblk) : soff0;
  const uint kl = lane * 8u;
  const uint blk_loc = lane >> 2u;
  device float* bases[3] = {y0, y1, y2};
  device float* dst_base = bases[seg] + (n0 - seg_off);

  if (p.M == 1u) {
    float a0 = 0.0f, a1 = 0.0f;
    for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
      if (kl < p.K - kbase) {
        const uint blk = (kbase >> 5u) + blk_loc;
        const device float* xp = x + kbase + kl;
        const float4 x0 = *(const device float4*)(xp + 0u), x1 = *(const device float4*)(xp + 4u);
        const uint2 qw0 = *(const device uint2*)(wq + woff0 + kbase + kl);
        const float sc0 = float(ws[soff0 + blk]);
        a0 += (dot(x0, float4(as_type<char4>(qw0.x))) + dot(x1, float4(as_type<char4>(qw0.y)))) * sc0;
        if (has1) {
          const uint2 qw1 = *(const device uint2*)(wq + woff1 + kbase + kl);
          const float sc1 = float(ws[soff1 + blk]);
          a1 += (dot(x0, float4(as_type<char4>(qw1.x))) + dot(x1, float4(as_type<char4>(qw1.y)))) * sc1;
        }
      }
    }
    const float s0 = simd_sum(a0), s1 = simd_sum(a1);
    if (lane == 0u) {
      dst_base[0] = s0;
      if (has1) dst_base[1] = s1;
    }
    return;
  }

  float acc0[8] = {}, acc1[8] = {};
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    if (kl < p.K - kbase) {
      const uint blk = (kbase >> 5u) + blk_loc;
      const uint2 qw0 = *(const device uint2*)(wq + woff0 + kbase + kl);
      const float sc0 = float(ws[soff0 + blk]);
      const float4 w0a = float4(as_type<char4>(qw0.x)) * sc0, w0b = float4(as_type<char4>(qw0.y)) * sc0;
      const uint2 qw1 = has1 ? *(const device uint2*)(wq + woff1 + kbase + kl) : qw0;
      const float sc1 = has1 ? float(ws[soff1 + blk]) : sc0;
      const float4 w1a = float4(as_type<char4>(qw1.x)) * sc1, w1b = float4(as_type<char4>(qw1.y)) * sc1;
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < 8u; ++m) {
        if (m < p.M) {
          const device float* xp = x + size_t(m) * p.K + kbase + kl;
          const float4 x0 = *(const device float4*)(xp + 0u), x1 = *(const device float4*)(xp + 4u);
          acc0[m] += dot(x0, w0a) + dot(x1, w0b);
          acc1[m] += dot(x0, w1a) + dot(x1, w1b);
        }
      }
    }
  }
  _Pragma("clang loop unroll(full)")
  for (uint m = 0u; m < 8u; ++m) {
    if (m >= p.M) break;
    const float s0 = simd_sum(acc0[m]), s1 = simd_sum(acc1[m]);
    if (lane == 0u) {
      device float* dst = dst_base + size_t(m) * seg_len;
      dst[0] = s0;
      if (has1) dst[1] = s1;
    }
  }
}

[[kernel]] void ink2_mtp_q8_gate_up_swiglu8(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                                            const device half* ws [[buffer(2)]], device float* out [[buffer(3)]],
                                            constant Ink2MtpQ8MmP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint j = tg * 4u + sg;
  if (j >= p.N) return;
  float ag[8] = {}, au[8] = {};
  const size_t woff_g = size_t(2u * j) * p.K, woff_u = woff_g + p.K;
  const size_t soff_g = size_t(2u * j) * p.nblk, soff_u = soff_g + p.nblk;
  const uint kl = lane * 8u;
  const uint blk_loc = lane >> 2u;
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    if (kl < p.K - kbase) {
      const uint blk = (kbase >> 5u) + blk_loc;
      const uint2 qwg = *(const device uint2*)(wq + woff_g + kbase + kl);
      const float scg = float(ws[soff_g + blk]);
      const float4 wga = float4(as_type<char4>(qwg.x)) * scg, wgb = float4(as_type<char4>(qwg.y)) * scg;
      const uint2 qwu = *(const device uint2*)(wq + woff_u + kbase + kl);
      const float scu = float(ws[soff_u + blk]);
      const float4 wua = float4(as_type<char4>(qwu.x)) * scu, wub = float4(as_type<char4>(qwu.y)) * scu;
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < 8u; ++m) {
        if (m < p.M) {
          const device float* xp = x + size_t(m) * p.K + kbase + kl;
          const float4 x0 = *(const device float4*)(xp + 0u), x1 = *(const device float4*)(xp + 4u);
          ag[m] += dot(x0, wga) + dot(x1, wgb);
          au[m] += dot(x0, wua) + dot(x1, wub);
        }
      }
    }
  }
  _Pragma("clang loop unroll(full)")
  for (uint m = 0u; m < 8u; ++m) {
    if (m >= p.M) break;
    const float g = simd_sum(ag[m]), u = simd_sum(au[m]);
    if (lane == 0u) out[size_t(m) * p.N + j] = (g / (1.0f + metal::precise::exp(-g))) * u;
  }
}

[[kernel]] void ink2_mtp_q4_mv8(const device float* x [[buffer(0)]], const device uint* wq [[buffer(1)]],
                                const device half* ws [[buffer(2)]], device float* y [[buffer(3)]],
                                constant Ink2MtpQ8MmP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4u + sg) * 2u;
  if (n0 >= p.N) return;
  const bool has1 = (n0 + 1u) < p.N;
  const size_t qrow_stride = size_t(p.nblk) * 4u;
  const size_t qoff0 = size_t(n0) * qrow_stride;
  const size_t qoff1 = has1 ? (qoff0 + qrow_stride) : qoff0;
  const size_t soff0 = size_t(n0) * p.nblk;
  const size_t soff1 = has1 ? (soff0 + p.nblk) : soff0;
  const uint kl = lane * 8u;
  const uint blk_loc = lane >> 2u;
  const uint wsub = lane & 3u;

  if (p.M == 1u) {
    float a0 = 0.0f, a1 = 0.0f;
    for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
      if (kl < p.K - kbase) {
        const uint blk = (kbase >> 5u) + blk_loc;
        const device float* xp = x + kbase + kl;
        const float4 x0 = *(const device float4*)(xp + 0u), x1 = *(const device float4*)(xp + 4u);
        {
          const uint pk = wq[qoff0 + blk * 4u + wsub];
          const float sc = float(ws[soff0 + blk]);
          const float4 wa = (float4(float(pk & 15u), float((pk >> 4) & 15u), float((pk >> 8) & 15u), float((pk >> 12) & 15u)) - 8.0f) * sc;
          const float4 wb = (float4(float((pk >> 16) & 15u), float((pk >> 20) & 15u), float((pk >> 24) & 15u), float(pk >> 28)) - 8.0f) * sc;
          a0 += dot(x0, wa) + dot(x1, wb);
        }
        if (has1) {
          const uint pk = wq[qoff1 + blk * 4u + wsub];
          const float sc = float(ws[soff1 + blk]);
          const float4 wa = (float4(float(pk & 15u), float((pk >> 4) & 15u), float((pk >> 8) & 15u), float((pk >> 12) & 15u)) - 8.0f) * sc;
          const float4 wb = (float4(float((pk >> 16) & 15u), float((pk >> 20) & 15u), float((pk >> 24) & 15u), float(pk >> 28)) - 8.0f) * sc;
          a1 += dot(x0, wa) + dot(x1, wb);
        }
      }
    }
    const float s0 = simd_sum(a0), s1 = simd_sum(a1);
    if (lane == 0u) {
      y[n0] = p.add ? (y[n0] + s0) : s0;
      if (has1) y[n0 + 1u] = p.add ? (y[n0 + 1u] + s1) : s1;
    }
    return;
  }

  float acc0[8] = {}, acc1[8] = {};
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    if (kl < p.K - kbase) {
      const uint blk = (kbase >> 5u) + blk_loc;
      const uint pk0 = wq[qoff0 + blk * 4u + wsub];
      const float sc0 = float(ws[soff0 + blk]);
      const float4 w0a = (float4(float(pk0 & 15u), float((pk0 >> 4) & 15u), float((pk0 >> 8) & 15u), float((pk0 >> 12) & 15u)) - 8.0f) * sc0;
      const float4 w0b = (float4(float((pk0 >> 16) & 15u), float((pk0 >> 20) & 15u), float((pk0 >> 24) & 15u), float(pk0 >> 28)) - 8.0f) * sc0;
      const uint pk1 = has1 ? wq[qoff1 + blk * 4u + wsub] : pk0;
      const float sc1 = has1 ? float(ws[soff1 + blk]) : sc0;
      const float4 w1a = (float4(float(pk1 & 15u), float((pk1 >> 4) & 15u), float((pk1 >> 8) & 15u), float((pk1 >> 12) & 15u)) - 8.0f) * sc1;
      const float4 w1b = (float4(float((pk1 >> 16) & 15u), float((pk1 >> 20) & 15u), float((pk1 >> 24) & 15u), float(pk1 >> 28)) - 8.0f) * sc1;
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < 8u; ++m) {
        if (m < p.M) {
          const device float* xp = x + size_t(m) * p.K + kbase + kl;
          const float4 x0 = *(const device float4*)(xp + 0u), x1 = *(const device float4*)(xp + 4u);
          acc0[m] += dot(x0, w0a) + dot(x1, w0b);
          acc1[m] += dot(x0, w1a) + dot(x1, w1b);
        }
      }
    }
  }
  _Pragma("clang loop unroll(full)")
  for (uint m = 0u; m < 8u; ++m) {
    if (m >= p.M) break;
    const float s0 = simd_sum(acc0[m]), s1 = simd_sum(acc1[m]);
    if (lane == 0u) {
      device float* dst = y + size_t(m) * p.N + n0;
      dst[0] = p.add ? (dst[0] + s0) : s0;
      if (has1) dst[1] = p.add ? (dst[1] + s1) : s1;
    }
  }
}

[[kernel]] void ink2_mtp_q4_mv8_multi3(const device float* x [[buffer(0)]], const device uint* wq [[buffer(1)]],
                                       const device half* ws [[buffer(2)]], device float* y0 [[buffer(3)]],
                                       device float* y1 [[buffer(4)]], device float* y2 [[buffer(5)]],
                                       constant Ink2MtpQ8Multi3P& p [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],
                                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * 4u + sg) * 2u;
  if (n0 >= p.N_tot) return;
  uint seg = 0u;
  while (seg < 2u && n0 >= p.offs[seg + 1u]) ++seg;
  const uint seg_off = p.offs[seg];
  const uint seg_end = p.offs[seg + 1u];
  const uint seg_len = seg_end - seg_off;
  if (n0 >= seg_end) return;
  const bool has1 = (n0 + 1u) < seg_end;
  const size_t qrow_stride = size_t(p.nblk) * 4u;
  const size_t qoff0 = size_t(n0) * qrow_stride;
  const size_t qoff1 = has1 ? (qoff0 + qrow_stride) : qoff0;
  const size_t soff0 = size_t(n0) * p.nblk;
  const size_t soff1 = has1 ? (soff0 + p.nblk) : soff0;
  const uint kl = lane * 8u;
  const uint blk_loc = lane >> 2u;
  const uint wsub = lane & 3u;
  device float* bases[3] = {y0, y1, y2};
  device float* dst_base = bases[seg] + (n0 - seg_off);

  if (p.M == 1u) {
    float a0 = 0.0f, a1 = 0.0f;
    for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
      if (kl < p.K - kbase) {
        const uint blk = (kbase >> 5u) + blk_loc;
        const device float* xp = x + kbase + kl;
        const float4 x0 = *(const device float4*)(xp + 0u), x1 = *(const device float4*)(xp + 4u);
        {
          const uint pk = wq[qoff0 + blk * 4u + wsub];
          const float sc = float(ws[soff0 + blk]);
          const float4 wa = (float4(float(pk & 15u), float((pk >> 4) & 15u), float((pk >> 8) & 15u), float((pk >> 12) & 15u)) - 8.0f) * sc;
          const float4 wb = (float4(float((pk >> 16) & 15u), float((pk >> 20) & 15u), float((pk >> 24) & 15u), float(pk >> 28)) - 8.0f) * sc;
          a0 += dot(x0, wa) + dot(x1, wb);
        }
        if (has1) {
          const uint pk = wq[qoff1 + blk * 4u + wsub];
          const float sc = float(ws[soff1 + blk]);
          const float4 wa = (float4(float(pk & 15u), float((pk >> 4) & 15u), float((pk >> 8) & 15u), float((pk >> 12) & 15u)) - 8.0f) * sc;
          const float4 wb = (float4(float((pk >> 16) & 15u), float((pk >> 20) & 15u), float((pk >> 24) & 15u), float(pk >> 28)) - 8.0f) * sc;
          a1 += dot(x0, wa) + dot(x1, wb);
        }
      }
    }
    const float s0 = simd_sum(a0), s1 = simd_sum(a1);
    if (lane == 0u) {
      dst_base[0] = s0;
      if (has1) dst_base[1] = s1;
    }
    return;
  }

  float acc0[8] = {}, acc1[8] = {};
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    if (kl < p.K - kbase) {
      const uint blk = (kbase >> 5u) + blk_loc;
      const uint pk0 = wq[qoff0 + blk * 4u + wsub];
      const float sc0 = float(ws[soff0 + blk]);
      const float4 w0a = (float4(float(pk0 & 15u), float((pk0 >> 4) & 15u), float((pk0 >> 8) & 15u), float((pk0 >> 12) & 15u)) - 8.0f) * sc0;
      const float4 w0b = (float4(float((pk0 >> 16) & 15u), float((pk0 >> 20) & 15u), float((pk0 >> 24) & 15u), float(pk0 >> 28)) - 8.0f) * sc0;
      const uint pk1 = has1 ? wq[qoff1 + blk * 4u + wsub] : pk0;
      const float sc1 = has1 ? float(ws[soff1 + blk]) : sc0;
      const float4 w1a = (float4(float(pk1 & 15u), float((pk1 >> 4) & 15u), float((pk1 >> 8) & 15u), float((pk1 >> 12) & 15u)) - 8.0f) * sc1;
      const float4 w1b = (float4(float((pk1 >> 16) & 15u), float((pk1 >> 20) & 15u), float((pk1 >> 24) & 15u), float(pk1 >> 28)) - 8.0f) * sc1;
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < 8u; ++m) {
        if (m < p.M) {
          const device float* xp = x + size_t(m) * p.K + kbase + kl;
          const float4 x0 = *(const device float4*)(xp + 0u), x1 = *(const device float4*)(xp + 4u);
          acc0[m] += dot(x0, w0a) + dot(x1, w0b);
          acc1[m] += dot(x0, w1a) + dot(x1, w1b);
        }
      }
    }
  }
  _Pragma("clang loop unroll(full)")
  for (uint m = 0u; m < 8u; ++m) {
    if (m >= p.M) break;
    const float s0 = simd_sum(acc0[m]), s1 = simd_sum(acc1[m]);
    if (lane == 0u) {
      device float* dst = dst_base + size_t(m) * seg_len;
      dst[0] = s0;
      if (has1) dst[1] = s1;
    }
  }
}

[[kernel]] void ink2_mtp_q4_gate_up_swiglu8(const device float* x [[buffer(0)]], const device uint* wq [[buffer(1)]],
                                            const device half* ws [[buffer(2)]], device float* out [[buffer(3)]],
                                            constant Ink2MtpQ8MmP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint j = tg * 4u + sg;
  if (j >= p.N) return;
  float ag[8] = {}, au[8] = {};
  const size_t qrow_stride = size_t(p.nblk) * 4u;
  const size_t qoff_g = size_t(2u * j) * qrow_stride, qoff_u = qoff_g + qrow_stride;
  const size_t soff_g = size_t(2u * j) * p.nblk, soff_u = soff_g + p.nblk;
  const uint kl = lane * 8u;
  const uint blk_loc = lane >> 2u;
  const uint wsub = lane & 3u;
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    if (kl < p.K - kbase) {
      const uint blk = (kbase >> 5u) + blk_loc;
      const uint pkg = wq[qoff_g + blk * 4u + wsub];
      const float scg = float(ws[soff_g + blk]);
      const float4 wga = (float4(float(pkg & 15u), float((pkg >> 4) & 15u), float((pkg >> 8) & 15u), float((pkg >> 12) & 15u)) - 8.0f) * scg;
      const float4 wgb = (float4(float((pkg >> 16) & 15u), float((pkg >> 20) & 15u), float((pkg >> 24) & 15u), float(pkg >> 28)) - 8.0f) * scg;
      const uint pku = wq[qoff_u + blk * 4u + wsub];
      const float scu = float(ws[soff_u + blk]);
      const float4 wua = (float4(float(pku & 15u), float((pku >> 4) & 15u), float((pku >> 8) & 15u), float((pku >> 12) & 15u)) - 8.0f) * scu;
      const float4 wub = (float4(float((pku >> 16) & 15u), float((pku >> 20) & 15u), float((pku >> 24) & 15u), float(pku >> 28)) - 8.0f) * scu;
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < 8u; ++m) {
        if (m < p.M) {
          const device float* xp = x + size_t(m) * p.K + kbase + kl;
          const float4 x0 = *(const device float4*)(xp + 0u), x1 = *(const device float4*)(xp + 4u);
          ag[m] += dot(x0, wga) + dot(x1, wgb);
          au[m] += dot(x0, wua) + dot(x1, wub);
        }
      }
    }
  }
  _Pragma("clang loop unroll(full)")
  for (uint m = 0u; m < 8u; ++m) {
    if (m >= p.M) break;
    const float g = simd_sum(ag[m]), u = simd_sum(au[m]);
    if (lane == 0u) out[size_t(m) * p.N + j] = (g / (1.0f + metal::precise::exp(-g))) * u;
  }
}

// ---- 12. On-GPU Exact Speculative Acceptance (Leviathan-Chen full-vocab Gumbel-max + top-k/top-p) ----
NI_INLINE uint4 ink2m_philox4x32_10(uint4 c, uint2 k) {
  for (uint r = 0u; r < 10u; ++r) {
    const uint hi0 = mulhi(0xD2511F53u, c.x);
    const uint lo0 = 0xD2511F53u * c.x;
    const uint hi1 = mulhi(0xCD9E8D57u, c.z);
    const uint lo1 = 0xCD9E8D57u * c.z;
    c = uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    k.x += 0x9E3779B9u;
    k.y += 0xBB67AE85u;
  }
  return c;
}

NI_INLINE float ink2m_u01(uint4 r) {
  return clamp((float(r.x >> 9u) + 0.5f) * (1.0f / 8388608.0f), 1e-7f, 0.9999999f);
}

NI_INLINE void ink2m_simd_argmax(thread float& v, thread uint& i) {
  const float mv = simd_max(v);
  const uint mi = simd_min(v == mv ? i : 0xffffffffu);
  v = as_type<float>(simd_or(v == mv && i == mi ? as_type<uint>(v) : 0u));
  i = mi;
}

NI_INLINE void ink2m_tg_argmax(thread float& v, thread uint& i, threadgroup float* sv, threadgroup uint* si, uint tid,
                               uint ntg) {
  const uint lane = tid & 31u, sg = tid >> 5u, nsg = (ntg + 31u) >> 5u;
  ink2m_simd_argmax(v, i);
  if (lane == 0u) {
    sv[sg] = v;
    si[sg] = i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    v = lane < nsg ? sv[lane] : -INFINITY;
    i = lane < nsg ? si[lane] : 0xffffffffu;
    ink2m_simd_argmax(v, i);
    if (lane == 0u) {
      sv[0] = v;
      si[0] = i;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = sv[0];
  i = si[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

struct Ink2MtpAccP {
  uint R, K, V, K_top, point_mass, seed, step, pos, lists, chunk;
  float inv_t, top_p;
};

struct Ink2MtpDist {
  uint keep;
  uint ids[64];
  float probs[64];
};

[[kernel]] void ink2_mtp_row_logz(const device float* verify_logits [[buffer(0)]],
                                  const device float* draft_logits [[buffer(1)]],
                                  device float2* stats [[buffer(2)]], constant Ink2MtpAccP& p [[buffer(3)]],
                                  uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                  uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* src =
      (row < p.R) ? (verify_logits + size_t(row) * p.V) : (draft_logits + size_t(row - p.R) * p.V);
  float mx = -INFINITY;
  for (uint i = tid; i < p.V; i += ntg) {
    const float v = src[i];
    if (!isnan(v)) mx = max(mx, v);
  }
  mx = tg_max(mx, scratch, tid, ntg);
  if (!(mx > -INFINITY)) {
    if (tid == 0u) stats[row] = float2(-INFINITY, -INFINITY);
    return;
  }
  float sum = 0.0f;
  for (uint i = tid; i < p.V; i += ntg) {
    const float v = src[i];
    if (!isnan(v) && v > -INFINITY) sum += metal::precise::exp((v - mx) * p.inv_t);
  }
  sum = tg_sum(sum, scratch, tid, ntg);
  if (tid == 0u) stats[row] = float2(mx, (sum > 0.0f && isfinite(sum)) ? metal::precise::log(sum) : -INFINITY);
}

[[kernel]] void ink2_mtp_accept_full_decide(const device int* tokens [[buffer(0)]],
                                            const device float* verify_logits [[buffer(1)]],
                                            const device float* draft_logits [[buffer(2)]],
                                            const device float2* stats [[buffer(3)]],
                                            device int* accept_out [[buffer(4)]],
                                            constant Ink2MtpAccP& p [[buffer(5)]],
                                            uint tid [[thread_index_in_threadgroup]]) {
  if (tid != 0u) return;
  uint a = 0u;
  for (; a < p.K; ++a) {
    const int dt_i = tokens[p.pos + 1u + a];
    const float2 sp_p = stats[a];
    float pa = 0.0f;
    if (dt_i >= 0 && uint(dt_i) < p.V && isfinite(sp_p.y)) {
      const float lp = verify_logits[size_t(a) * p.V + uint(dt_i)];
      if (!isnan(lp) && lp > -INFINITY) pa = metal::precise::exp((lp - sp_p.x) * p.inv_t - sp_p.y);
    }
    float qa = 1.0f;
    if (!p.point_mass) {
      const float2 sp_q = stats[p.R + a];
      qa = 0.0f;
      if (dt_i >= 0 && uint(dt_i) < p.V && isfinite(sp_q.y)) {
        const float lq = draft_logits[size_t(a) * p.V + uint(dt_i)];
        if (!isnan(lq) && lq > -INFINITY) qa = metal::precise::exp((lq - sp_q.x) * p.inv_t - sp_q.y);
      }
    }
    const float u = ink2m_u01(ink2m_philox4x32_10(uint4(a, 0x9e3779b9u, p.point_mass, 1u), uint2(p.seed, p.step + a)));
    const bool acc = p.point_mass ? (u < pa) : (qa > 0.0f && (pa >= qa || u * qa < pa));
    if (!acc) break;
  }
  accept_out[0] = int(a);
}

[[kernel]] void ink2_mtp_sample_emit_stage1(const device int* tokens [[buffer(0)]],
                                            const device float* verify_logits [[buffer(1)]],
                                            const device float* draft_logits [[buffer(2)]],
                                            const device float2* stats [[buffer(3)]],
                                            const device int* accept_out [[buffer(4)]],
                                            device float* cand_v [[buffer(5)]], device uint* cand_i [[buffer(6)]],
                                            constant Ink2MtpAccP& p [[buffer(7)]],
                                            uint bid [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint ntg = 256u, nsg = 8u;
  threadgroup float sv[8], sfv[8];
  threadgroup uint si[8], sfi[8];
  const uint a = uint(accept_out[0]);
  const uint dt = (a < p.K) ? uint(tokens[p.pos + 1u + a]) : 0xffffffffu;
  const float2 sp_p = stats[a];
  const float2 sp_q = (!p.point_mass && a < p.K) ? stats[p.R + a] : float2(-INFINITY, -INFINITY);
  const device float* vp = verify_logits + size_t(a) * p.V;
  const device float* qp = (!p.point_mass && a < p.K) ? (draft_logits + size_t(a) * p.V) : vp;
  const uint2 key = uint2(p.seed, p.step + a);

  float bv = -INFINITY, fv = -INFINITY;
  uint bi = 0xffffffffu, fi = 0xffffffffu;
  for (uint v = bid * ntg + tid; v < p.V; v += 64u * ntg) {
    const float lp = vp[v];
    if (isnan(lp) || !(lp > -INFINITY)) continue;
    const float u = ink2m_u01(ink2m_philox4x32_10(uint4(v, a, p.point_mass, 0x517cc1b7u), key));
    const float g = -metal::precise::log(-metal::precise::log(u));
    const float s_fb = (lp - sp_p.x) * p.inv_t + g;
    if (s_fb > fv || (s_fb == fv && v < fi)) {
      fv = s_fb;
      fi = v;
    }
    float sc = -INFINITY;
    if (a >= p.K) {
      sc = s_fb;
    } else if (p.point_mass) {
      if (v != dt) sc = s_fb;
    } else {
      const float pa = isfinite(sp_p.y) ? metal::precise::exp((lp - sp_p.x) * p.inv_t - sp_p.y) : 0.0f;
      const float lq = qp[v];
      const float qa =
          (!isnan(lq) && lq > -INFINITY && isfinite(sp_q.y)) ? metal::precise::exp((lq - sp_q.x) * p.inv_t - sp_q.y) : 0.0f;
      const float rw = pa - qa;
      if (rw > 0.0f) sc = metal::precise::log(rw) + g;
    }
    if (sc > bv || (sc == bv && v < bi)) {
      bv = sc;
      bi = v;
    }
  }
  ink2m_simd_argmax(bv, bi);
  ink2m_simd_argmax(fv, fi);
  const uint sg = tid >> 5u, lane = tid & 31u;
  if (lane == 0u) {
    sv[sg] = bv;
    si[sg] = bi;
    sfv[sg] = fv;
    sfi[sg] = fi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float m2 = lane < nsg ? sv[lane] : -INFINITY;
    uint i2 = lane < nsg ? si[lane] : 0xffffffffu;
    ink2m_simd_argmax(m2, i2);
    float fm2 = lane < nsg ? sfv[lane] : -INFINITY;
    uint fi2 = lane < nsg ? sfi[lane] : 0xffffffffu;
    ink2m_simd_argmax(fm2, fi2);
    if (lane == 0u) {
      cand_v[bid] = m2;
      cand_i[bid] = i2;
      cand_v[64u + bid] = fm2;
      cand_i[64u + bid] = fi2;
    }
  }
}

[[kernel]] void ink2_mtp_sample_emit_stage2(const device float* cand_v [[buffer(0)]], const device uint* cand_i [[buffer(1)]],
                                            device int* accept_out [[buffer(2)]], device int* tokens [[buffer(3)]],
                                            constant Ink2MtpAccP& p [[buffer(4)]],
                                            uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sv[2], sfv[2];
  threadgroup uint si[2], sfi[2];
  float v = cand_v[tid], fv = cand_v[64u + tid];
  uint idx = cand_i[tid], fidx = cand_i[64u + tid];
  ink2m_simd_argmax(v, idx);
  ink2m_simd_argmax(fv, fidx);
  const uint sg = tid >> 5u, lane = tid & 31u;
  if (lane == 0u) {
    sv[sg] = v;
    si[sg] = idx;
    sfv[sg] = fv;
    sfi[sg] = fidx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float m2 = lane < 2u ? sv[lane] : -INFINITY;
    uint i2 = lane < 2u ? si[lane] : 0xffffffffu;
    ink2m_simd_argmax(m2, i2);
    float fm2 = lane < 2u ? sfv[lane] : -INFINITY;
    uint fi2 = lane < 2u ? sfi[lane] : 0xffffffffu;
    ink2m_simd_argmax(fm2, fi2);
    if (lane == 0u) {
      const uint pick = (i2 != 0xffffffffu) ? i2 : fi2;
      const int tok_out = (pick == 0xffffffffu) ? 0 : int(pick);
      const uint a = uint(accept_out[0]);
      tokens[p.pos + 1u + a] = tok_out;
      accept_out[1] = tok_out;
      accept_out[2] = 1;
    }
  }
}

[[kernel]] void ink2_mtp_topk_stage1_rows(const device float* verify_logits [[buffer(0)]],
                                          const device float* draft_logits [[buffer(1)]],
                                          device float* cv [[buffer(2)]], device uint* ci [[buffer(3)]],
                                          constant Ink2MtpAccP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  const uint ntg = p.chunk >> 3u;
  const uint row = tg.y;
  const device float* logits =
      (row < p.R) ? (verify_logits + size_t(row) * p.V) : (draft_logits + size_t(row - p.R) * p.V);
  float v[8];
  const uint base = tg.x * p.chunk + tid * 8u;
  for (uint j = 0u; j < 8u; ++j) {
    const float lv = (base + j < p.V) ? logits[base + j] : -INFINITY;
    v[j] = isnan(lv) ? -INFINITY : lv;
  }
  const size_t out_base = (size_t(row) * p.lists + tg.x) * p.K_top;
  for (uint r = 0u; r < p.K_top; ++r) {
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    for (uint j = 0u; j < 8u; ++j) {
      if (v[j] > bv) {
        bv = v[j];
        bi = base + j;
      }
    }
    ink2m_tg_argmax(bv, bi, sv, si, tid, ntg);
    if (bi >= base && bi < base + 8u) v[bi - base] = -INFINITY;
    if (tid == 0u) {
      cv[out_base + r] = bv;
      ci[out_base + r] = bi;
    }
  }
}

[[kernel]] void ink2_mtp_topk_stage2_dists(const device float* cv_all [[buffer(0)]],
                                           const device uint* ci_all [[buffer(1)]],
                                           device Ink2MtpDist* dists [[buffer(2)]],
                                           constant Ink2MtpAccP& p [[buffer(3)]],
                                           uint row [[threadgroup_position_in_grid]],
                                           uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float sv[32], topv[64];
  threadgroup uint si[32], topi[64];
  const uint n = p.lists * p.K_top;
  const device float* cv = cv_all + size_t(row) * n;
  const device uint* ci = ci_all + size_t(row) * n;
  const uint K = max(1u, min(p.K_top, 64u));
  if (K >= 8u && K <= 32u && p.lists <= 128u) {
    if (tid < 32u) {
      float v[4];
      uint id[4], pos[4];
      for (uint j = 0u; j < 4u; ++j) {
        const uint list = tid + j * 32u;
        pos[j] = 0u;
        v[j] = list < p.lists ? cv[list * p.K_top] : -INFINITY;
        id[j] = list < p.lists ? ci[list * p.K_top] : 0xffffffffu;
      }
      uint first_id = 0xffffffffu;
      for (uint rank = 0u; rank < K; ++rank) {
        float bv = -INFINITY;
        uint bi = 0xffffffffu;
        int bj = -1;
        for (uint j = 0u; j < 4u; ++j) {
          if (v[j] > bv || (v[j] == bv && id[j] < bi)) {
            bv = v[j];
            bi = id[j];
            bj = int(j);
          }
        }
        float wv = bv;
        uint wi = bi;
        ink2m_simd_argmax(wv, wi);
        first_id = min(first_id, wi);
        if (wv == -INFINITY) wi = first_id;
        if (tid == 0u) {
          topv[rank] = wv;
          topi[rank] = wi;
        }
        if (bj >= 0 && bi == wi && bv == wv) {
          const uint list = tid + uint(bj) * 32u;
          const bool has_next = ++pos[bj] < p.K_top;
          v[bj] = has_next ? cv[list * p.K_top + pos[bj]] : -INFINITY;
          id[bj] = has_next ? ci[list * p.K_top + pos[bj]] : 0xffffffffu;
        }
      }
    }
  } else {
    constexpr uint PER = 4u;
    float v[4];
    uint id[4];
    for (uint j = 0u; j < PER; ++j) {
      const uint k = tid * PER + j;
      v[j] = k < n ? cv[k] : -INFINITY;
      id[j] = k < n ? ci[k] : 0xffffffffu;
    }
    for (uint r = 0u; r < K; ++r) {
      float bv = -INFINITY;
      uint bi = 0xffffffffu;
      int bj = -1;
      for (uint j = 0u; j < PER; ++j) {
        if (v[j] > bv || (v[j] == bv && id[j] < bi)) {
          bv = v[j];
          bi = id[j];
          bj = int(j);
        }
      }
      float wv = bv;
      uint wi = bi;
      ink2m_tg_argmax(wv, wi, sv, si, tid, ntg);
      if (bj >= 0 && bi == wi && bv == wv) v[bj] = -INFINITY;
      if (tid == 0u) {
        topv[r] = wv;
        topi[r] = wi;
      }
    }
  }
  if (tid == 0u) {
    device Ink2MtpDist& d = dists[row];
    if (!(topv[0] > -INFINITY) || topi[0] == 0xffffffffu) {
      d.keep = 1u;
      d.ids[0] = 0u;
      d.probs[0] = 1.0f;
      return;
    }
    float unt[64], tot = 0.0f;
    for (uint r = 0u; r < K; ++r) {
      unt[r] = (topi[r] != 0xffffffffu && topv[r] > -INFINITY) ? metal::precise::exp(topv[r] - topv[0]) : 0.0f;
      tot += unt[r];
    }
    uint keep = K;
    float cum = 0.0f;
    for (uint r = 0u; r < K; ++r) {
      cum += unt[r] / tot;
      if (cum >= p.top_p) {
        keep = r + 1u;
        break;
      }
    }
    float kt = 0.0f;
    for (uint r = 0u; r < keep; ++r) {
      const float w = (topi[r] != 0xffffffffu && topv[r] > -INFINITY) ? metal::precise::exp((topv[r] - topv[0]) * p.inv_t) : 0.0f;
      d.ids[r] = topi[r];
      d.probs[r] = w;
      kt += w;
    }
    const float inv = kt > 0.0f ? (1.0f / kt) : 1.0f;
    for (uint r = 0u; r < keep; ++r) d.probs[r] *= inv;
    d.keep = keep;
  }
}

[[kernel]] void ink2_mtp_accept_topk_decide(const device Ink2MtpDist* dists [[buffer(0)]],
                                            device int* tokens [[buffer(1)]], device int* accept_out [[buffer(2)]],
                                            constant Ink2MtpAccP& p [[buffer(3)]],
                                            uint tid [[thread_index_in_threadgroup]]) {
  if (tid != 0u) return;
  uint a = 0u;
  for (; a < p.K; ++a) {
    const uint dt = uint(tokens[p.pos + 1u + a]);
    const device Ink2MtpDist& dp = dists[a];
    float pa = 0.0f;
    for (uint j = 0u; j < dp.keep; ++j) {
      if (dp.ids[j] == dt) {
        pa = dp.probs[j];
        break;
      }
    }
    float qa = 1.0f;
    if (!p.point_mass) {
      const device Ink2MtpDist& dq = dists[p.R + a];
      qa = 0.0f;
      for (uint j = 0u; j < dq.keep; ++j) {
        if (dq.ids[j] == dt) {
          qa = dq.probs[j];
          break;
        }
      }
    }
    const float u_acc = ink2m_u01(ink2m_philox4x32_10(uint4(a, 0x9e3779b9u, p.point_mass, 2u), uint2(p.seed, p.step + a)));
    if (p.point_mass ? (u_acc < pa) : (qa > 0.0f && (pa >= qa || u_acc * qa < pa))) continue;

    float rw[64];
    uint rids[64];
    uint rn = 0u;
    float rsum = 0.0f;
    for (uint j = 0u; j < dp.keep; ++j) {
      const uint vid = dp.ids[j];
      float qv = 0.0f;
      if (p.point_mass) {
        qv = (vid == dt) ? 1.0f : 0.0f;
      } else {
        const device Ink2MtpDist& dq = dists[p.R + a];
        for (uint qj = 0u; qj < dq.keep; ++qj) {
          if (dq.ids[qj] == vid) {
            qv = dq.probs[qj];
            break;
          }
        }
      }
      const float r = dp.probs[j] - qv;
      if (r > 0.0f) {
        rids[rn] = vid;
        rw[rn] = r;
        rsum += r;
        ++rn;
      }
    }
    const float u_res = ink2m_u01(ink2m_philox4x32_10(uint4(a, 0x517cc1b7u, p.point_mass, 3u), uint2(p.seed, p.step + a)));
    uint chosen = dp.ids[0];
    if (rsum > 0.0f && rn > 0u) {
      const float target = u_res * rsum;
      float acc = 0.0f;
      chosen = rids[rn - 1u];
      for (uint j = 0u; j < rn; ++j) {
        acc += rw[j];
        if (target < acc) {
          chosen = rids[j];
          break;
        }
      }
    } else {
      float acc = 0.0f;
      chosen = dp.ids[dp.keep - 1u];
      for (uint j = 0u; j < dp.keep; ++j) {
        acc += dp.probs[j];
        if (u_res < acc) {
          chosen = dp.ids[j];
          break;
        }
      }
    }
    tokens[p.pos + 1u + a] = int(chosen);
    accept_out[0] = int(a);
    accept_out[1] = int(chosen);
    accept_out[2] = 1;
    return;
  }
  const device Ink2MtpDist& dk = dists[p.K];
  const float u_b = ink2m_u01(ink2m_philox4x32_10(uint4(p.K, 0x243f6a88u, p.point_mass, 4u), uint2(p.seed, p.step + p.K)));
  float acc = 0.0f;
  uint chosen = dk.ids[dk.keep - 1u];
  for (uint j = 0u; j < dk.keep; ++j) {
    acc += dk.probs[j];
    if (u_b < acc) {
      chosen = dk.ids[j];
      break;
    }
  }
  tokens[p.pos + 1u + p.K] = int(chosen);
  accept_out[0] = int(p.K);
  accept_out[1] = int(chosen);
  accept_out[2] = 1;
}

// ---- 13. Bayes-Optimal W=4 On-GPU Markov Beam Tree Drafting & Lossless Tree Acceptance ----
// (`struct Ink2Tree` is defined once above in ink2.metal.)
struct Ink2TreeAccept {
  int m;
  int stop_node;
  int bonus_tok;
  int ok;
  int path[16];
};

struct Ink2CandNode {
  int tok;
  int parent_pool;
  uint depth;
  float path_logp;
};

struct Ink2BeamState {
  uint n_beam;
  uint pool_count;
  int tok[4];
  uint pool_idx[4];
  float logp[4];
};

struct Ink2MtpBeamPrepP {
  uint rank, H, first_step;
};

struct Ink2MtpBeamUpdP {
  uint rank, first_step;
};

struct Ink2MtpBeamStepP {
  uint V, R, d, R_budget, last_step;
  float inv_t;
};

struct Ink2MtpTreeSelP {
  uint R_budget, pos;
};

struct Ink2MtpTreeAccP {
  uint V, seed, step, pos;
  float inv_t;
};

[[kernel]] void ink2_mtp_beam_init(const device int* tokens [[buffer(0)]], device Ink2BeamState* meta [[buffer(1)]],
                                   device Ink2CandNode* cand_pool [[buffer(2)]], constant uint& pos [[buffer(3)]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  if (tid != 0u) return;
  const int root_tok = tokens[pos];
  meta->n_beam = 1u;
  meta->pool_count = 1u;
  meta->tok[0] = root_tok;
  meta->pool_idx[0] = 0u;
  meta->logp[0] = 0.0f;
  for (uint b = 1u; b < 4u; ++b) {
    meta->tok[b] = 0;
    meta->pool_idx[b] = 0u;
    meta->logp[b] = -INFINITY;
  }
  cand_pool[0].tok = root_tok;
  cand_pool[0].parent_pool = -1;
  cand_pool[0].depth = 0u;
  cand_pool[0].path_logp = 0.0f;
}

[[kernel]] void ink2_mtp_beam_rnn_prep(const device Ink2BeamState* meta [[buffer(0)]],
                                       const device float* beam_state [[buffer(1)]],
                                       const device ushort* markov_w1 [[buffer(2)]],
                                       const device float* hidden [[buffer(3)]],
                                       const device float* feat [[buffer(4)]],
                                       device float* rnn_in [[buffer(5)]],
                                       constant Ink2MtpBeamPrepP& p [[buffer(6)]],
                                       uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x, b = gid.y;
  const uint tot = 2u * p.rank + 2u * p.H;
  if (i >= tot) return;
  const size_t dst = size_t(b) * tot + i;
  if (b >= meta->n_beam) {
    rnn_in[dst] = 0.0f;
    return;
  }
  if (i < p.rank) {
    rnn_in[dst] = p.first_step ? 0.0f : beam_state[size_t(b) * p.rank + i];
  } else if (i < 2u * p.rank) {
    const int id = meta->tok[b];
    rnn_in[dst] = ink2m_bf16(markov_w1[size_t(id) * p.rank + (i - p.rank)]);
  } else if (i < 2u * p.rank + p.H) {
    rnn_in[dst] = hidden[i - 2u * p.rank];
  } else {
    rnn_in[dst] = feat[i - (2u * p.rank + p.H)];
  }
}

[[kernel]] void ink2_mtp_beam_rnn_update(const device Ink2BeamState* meta [[buffer(0)]],
                                         const device float* gates [[buffer(1)]],
                                         const device float* beam_state [[buffer(2)]],
                                         device float* beam_next_state [[buffer(3)]],
                                         device float* beam_latent [[buffer(4)]],
                                         constant Ink2MtpBeamUpdP& p [[buffer(5)]],
                                         uint2 tg [[threadgroup_position_in_grid]],
                                         uint c [[thread_index_in_threadgroup]]) {
  const uint b = tg.y;
  if (c >= p.rank) return;
  const size_t roff = size_t(b) * p.rank + c;
  if (b >= meta->n_beam) {
    beam_next_state[roff] = 0.0f;
    beam_latent[roff] = 0.0f;
    return;
  }
  const device float* gb = gates + size_t(b) * 3u * p.rank;
  const float prev_s = p.first_step ? 0.0f : beam_state[roff];
  const float k = 1.0f / (1.0f + metal::precise::exp(-gb[c]));
  const float u = metal::precise::tanh(gb[p.rank + c]);
  const float l = metal::precise::tanh(gb[2u * p.rank + c]);
  beam_next_state[roff] = k * prev_s + (1.0f - k) * u;
  beam_latent[roff] = l;
}

[[kernel]] void ink2_mtp_p2_fold_screen_beam(const device Ink2BeamState* meta [[buffer(0)]],
                                             const device float* latents [[buffer(1)]],
                                             const device uint2* fold_p2_q [[buffer(2)]],
                                             const device half* fold_p2_s [[buffer(3)]],
                                             const device half* base_proxy [[buffer(4)]],
                                             const device uint* sup_mask [[buffer(5)]],
                                             device float* beam_cand_v [[buffer(6)]],
                                             device uint* beam_cand_i [[buffer(7)]],
                                             constant Ink2MtpP2ScreenP& p [[buffer(8)]],
                                             uint tg [[threadgroup_position_in_grid]],
                                             uint tid [[thread_index_in_threadgroup]],
                                             uint sg [[simdgroup_index_in_threadgroup]],
                                             uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sh_x[4][256];
  threadgroup float sh_sum[4][8];
  threadgroup float sh_v[4][64];
  threadgroup uint sh_i[4][64];

  const uint nb = min(meta->n_beam, 4u);
  for (uint b = 0u; b < nb; ++b) {
    const device float* x = latents + size_t(b) * 256u;
    const float xv = x[sg * 32u + lane];
    sh_x[b][sg * 32u + lane] = xv;
    const float s = simd_sum(xv);
    if (lane == 0u) sh_sum[b][sg] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint ntg_row = (p.V + 31u) / 32u;
  const uint blk = lane & 7u;
  const uint slot = sg * 4u + (lane >> 3u);

  if (nb == 1u) {
    float b0v = -INFINITY, b1v = -INFINITY;
    uint b0i = 0xffffffffu, b1i = 0xffffffffu;
    const threadgroup float4* xmp = (const threadgroup float4*)(sh_x[0] + blk * 32u);
    const float x0s = sh_sum[0][blk];

    for (uint g = tg; g < ntg_row; g += 64u) {
      const uint v = (g * 8u + sg) * 4u + (lane >> 3u);
      const bool valid = v < p.V;
      float acc = 0.0f;
      if (valid) {
        const size_t idx = size_t(v) * 8u + blk;
        acc = ink2m_p2_dot32(fold_p2_q[idx], xmp[0], xmp[1], xmp[2], xmp[3], xmp[4], xmp[5], xmp[6], xmp[7], x0s,
                             float(fold_p2_s[idx]));
      }
      acc += simd_shuffle_xor(acc, 1u);
      acc += simd_shuffle_xor(acc, 2u);
      acc += simd_shuffle_xor(acc, 4u);
      if (blk == 0u && valid) {
        float val = float(base_proxy[v]) + acc;
        if (p.has_sup) {
          const uint widx = v >> 5u;
          if (widx < p.sup_words && ((sup_mask[widx] >> (v & 31u)) & 1u) == 0u) val = -INFINITY;
        }
        if (val > b0v || (val == b0v && v < b0i)) {
          b1v = b0v; b1i = b0i;
          b0v = val; b0i = v;
        } else if (val > b1v || (val == b1v && v < b1i)) {
          b1v = val; b1i = v;
        }
      }
    }
    if (blk == 0u) {
      sh_v[0][slot * 2u + 0u] = b0v;
      sh_i[0][slot * 2u + 0u] = b0i;
      sh_v[0][slot * 2u + 1u] = b1v;
      sh_i[0][slot * 2u + 1u] = b1i;
    }
  } else {
    float b0v[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float b1v[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    uint b0i[4] = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
    uint b1i[4] = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};

    for (uint g = tg; g < ntg_row; g += 64u) {
      const uint v = (g * 8u + sg) * 4u + (lane >> 3u);
      const bool valid = v < p.V;
      bool suppressed = false;
      if (valid && p.has_sup) {
        const uint widx = v >> 5u;
        if (widx < p.sup_words && ((sup_mask[widx] >> (v & 31u)) & 1u) == 0u) suppressed = true;
      }
      uint2 qv = uint2(0u);
      float sc = 0.0f;
      float bp = 0.0f;
      if (valid && !suppressed) {
        const size_t idx = size_t(v) * 8u + blk;
        qv = fold_p2_q[idx];
        sc = float(fold_p2_s[idx]);
        if (blk == 0u) bp = float(base_proxy[v]);
      }
      const float4 c0 = float4(float(qv.x & 3u), float((qv.x >> 2u) & 3u), float((qv.x >> 4u) & 3u), float((qv.x >> 6u) & 3u));
      const float4 c1 = float4(float((qv.x >> 8u) & 3u), float((qv.x >> 10u) & 3u), float((qv.x >> 12u) & 3u), float((qv.x >> 14u) & 3u));
      const float4 c2 = float4(float((qv.x >> 16u) & 3u), float((qv.x >> 18u) & 3u), float((qv.x >> 20u) & 3u), float((qv.x >> 22u) & 3u));
      const float4 c3 = float4(float((qv.x >> 24u) & 3u), float((qv.x >> 26u) & 3u), float((qv.x >> 28u) & 3u), float(qv.x >> 30u));
      const float4 c4 = float4(float(qv.y & 3u), float((qv.y >> 2u) & 3u), float((qv.y >> 4u) & 3u), float((qv.y >> 6u) & 3u));
      const float4 c5 = float4(float((qv.y >> 8u) & 3u), float((qv.y >> 10u) & 3u), float((qv.y >> 12u) & 3u), float((qv.y >> 14u) & 3u));
      const float4 c6 = float4(float((qv.y >> 16u) & 3u), float((qv.y >> 18u) & 3u), float((qv.y >> 20u) & 3u), float((qv.y >> 22u) & 3u));
      const float4 c7 = float4(float((qv.y >> 24u) & 3u), float((qv.y >> 26u) & 3u), float((qv.y >> 28u) & 3u), float(qv.y >> 30u));
      _Pragma("clang loop unroll(full)")
      for (uint b = 0u; b < 4u; ++b) {
        if (b >= nb) break;
        float acc = 0.0f;
        if (valid && !suppressed) {
          const threadgroup float4* xmp = (const threadgroup float4*)(sh_x[b] + blk * 32u);
          const float s = dot(xmp[0], c0) + dot(xmp[1], c1) + dot(xmp[2], c2) + dot(xmp[3], c3) +
                          dot(xmp[4], c4) + dot(xmp[5], c5) + dot(xmp[6], c6) + dot(xmp[7], c7);
          acc = (s - 1.5f * sh_sum[b][blk]) * sc;
        }
        acc += simd_shuffle_xor(acc, 1u);
        acc += simd_shuffle_xor(acc, 2u);
        acc += simd_shuffle_xor(acc, 4u);
        if (blk == 0u && valid && !suppressed) {
          const float val = bp + acc;
          if (val > b0v[b] || (val == b0v[b] && v < b0i[b])) {
            b1v[b] = b0v[b]; b1i[b] = b0i[b];
            b0v[b] = val;    b0i[b] = v;
          } else if (val > b1v[b] || (val == b1v[b] && v < b1i[b])) {
            b1v[b] = val;    b1i[b] = v;
          }
        }
      }
    }

    if (blk == 0u) {
      _Pragma("clang loop unroll(full)")
      for (uint b = 0u; b < 4u; ++b) {
        if (b >= nb) break;
        sh_v[b][slot * 2u + 0u] = b0v[b];
        sh_i[b][slot * 2u + 0u] = b0i[b];
        sh_v[b][slot * 2u + 1u] = b1v[b];
        sh_i[b][slot * 2u + 1u] = b1i[b];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < nb) {
    const uint b = tid;
    float t0v = -INFINITY, t1v = -INFINITY;
    uint t0i = 0xffffffffu, t1i = 0xffffffffu;
    for (uint j = 0u; j < 64u; ++j) {
      const float vv = sh_v[b][j];
      const uint vi = sh_i[b][j];
      if (vi == 0xffffffffu) continue;
      if (vv > t0v || (vv == t0v && vi < t0i)) {
        t1v = t0v; t1i = t0i;
        t0v = vv;  t0i = vi;
      } else if (vv > t1v || (vv == t1v && vi < t1i)) {
        t1v = vv;  t1i = vi;
      }
    }
    beam_cand_v[size_t(b) * 128u + tg * 2u + 0u] = t0v;
    beam_cand_i[size_t(b) * 128u + tg * 2u + 0u] = t0i;
    beam_cand_v[size_t(b) * 128u + tg * 2u + 1u] = t1v;
    beam_cand_i[size_t(b) * 128u + tg * 2u + 1u] = t1i;
  }
}

[[kernel]] void ink2_mtp_rescore_beam(const device Ink2BeamState* meta [[buffer(0)]],
                                      const device float* x_u [[buffer(1)]],
                                      const device float* beam_latent [[buffer(2)]],
                                      const device void* unembed [[buffer(3)]],
                                      const device ushort* fold_w [[buffer(4)]],
                                      const device float* fold_b [[buffer(5)]],
                                      const device uint* sup_mask [[buffer(6)]],
                                      const device uint* beam_cand_i [[buffer(7)]],
                                      device float* beam_cand_v [[buffer(8)]],
                                      constant Ink2MtpRescoreP& p [[buffer(9)]],
                                      uint2 tg [[threadgroup_position_in_grid]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
  const uint b = tg.y;
  if (b >= meta->n_beam) return;
  const uint c_loc = tg.x * 4u + sg;
  const size_t c = size_t(b) * 128u + c_loc;
  const uint v = beam_cand_i[c];
  if (v >= p.V) {
    if (lane == 0u) beam_cand_v[c] = -INFINITY;
    return;
  }
  if (p.has_sup) {
    const uint widx = v >> 5u;
    if (widx < p.sup_words && ((sup_mask[widx] >> (v & 31u)) & 1u) == 0u) {
      if (lane == 0u) beam_cand_v[c] = -INFINITY;
      return;
    }
  }
  float acc = 0.0f;
  const size_t uoff = size_t(v) * p.H;
  if (p.u_f32) {
    const device float* uw = (const device float*)unembed + uoff;
    for (uint k = lane * 4u; k < p.H; k += 128u) {
      acc += dot(*(const device float4*)(x_u + k), *(const device float4*)(uw + k));
    }
  } else {
    const device ushort* uw = (const device ushort*)unembed + uoff;
    for (uint k = lane * 8u; k < p.H; k += 256u) {
      const uint4 qu = *(const device uint4*)(uw + k);
      acc += dot(*(const device float4*)(x_u + k + 0u), ink2m_bf16_4(qu.xy)) +
             dot(*(const device float4*)(x_u + k + 4u), ink2m_bf16_4(qu.zw));
    }
  }
  const device float* x_f = beam_latent + size_t(b) * p.R;
  const device ushort* fw = fold_w + size_t(v) * p.R;
  for (uint r = lane * 8u; r < p.R; r += 256u) {
    const uint4 qf = *(const device uint4*)(fw + r);
    acc += dot(*(const device float4*)(x_f + r + 0u), ink2m_bf16_4(qf.xy)) +
           dot(*(const device float4*)(x_f + r + 4u), ink2m_bf16_4(qf.zw));
  }
  const float sum = simd_sum(acc);
  if (lane == 0u) {
    float val = sum + fold_b[v];
    if (p.softcap > 0.0f) val = p.softcap * metal::precise::tanh(val * p.inv_cap);
    beam_cand_v[c] = val;
  }
}

[[kernel]] void ink2_mtp_beam_step(device Ink2BeamState* meta [[buffer(0)]],
                                   device Ink2CandNode* cand_pool [[buffer(1)]],
                                   const device float* beam_cand_v [[buffer(2)]],
                                   const device uint* beam_cand_i [[buffer(3)]],
                                   const device float* beam_next_state [[buffer(4)]],
                                   device float* beam_state [[buffer(5)]],
                                   constant Ink2MtpBeamStepP& p [[buffer(6)]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sv[4], sh_exp_lp[32];
  threadgroup uint si[4], sh_exp_pidx[32], sh_exp_src[32], sh_nb, sh_src_beam[4];
  threadgroup int sh_exp_tok[32];
  const uint sg = tid >> 5u, lane = tid & 31u;
  const uint nb = min(meta->n_beam, 4u);
  uint exp_cnt = 0u;
  const uint F = (p.d == 0u) ? min(8u, p.R_budget - 1u) : min(4u, p.R_budget - 1u);

  for (uint b = 0u; b < nb; ++b) {
    const uint id = beam_cand_i[size_t(b) * 128u + tid];
    const float raw = (id < p.V) ? beam_cand_v[size_t(b) * 128u + tid] : -INFINITY;

    const float ms = simd_max(raw);
    if (lane == 0u) sv[sg] = ms;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      const float gmx = simd_max(lane < 4u ? sv[lane] : -INFINITY);
      if (lane == 0u) sv[0] = gmx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mx = sv[0];
    if (!isfinite(mx)) continue;

    const float u_w = (id < p.V && isfinite(raw)) ? metal::precise::exp((raw - mx) * p.inv_t) : 0.0f;
    const float ss = simd_sum(u_w);
    if (lane == 0u) sv[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      const float gz = simd_sum(lane < 4u ? sv[lane] : 0.0f);
      if (lane == 0u) sv[0] = gz;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float z = sv[0];
    if (!(z > 0.0f) || !isfinite(z)) continue;

    const float logz = metal::precise::log(z);
    float lp = (id < p.V && isfinite(raw)) ? min(0.0f, (raw - mx) * p.inv_t - logz) : -INFINITY;

    for (uint f = 0u; f < F; ++f) {
      float plp = lp;
      uint pid = id;
      ink2m_tg_argmax(plp, pid, sv, si, tid, 128u);
      if (pid >= p.V || !isfinite(plp)) break;
      if (id == pid) lp = -INFINITY;
      if (tid == 0u && exp_cnt < 32u) {
        const uint p_idx = meta->pool_count;
        const float new_lp = meta->logp[b] + plp;
        if (p_idx < 128u) {
          cand_pool[p_idx].tok = int(pid);
          cand_pool[p_idx].parent_pool = int(meta->pool_idx[b]);
          cand_pool[p_idx].depth = p.d + 1u;
          cand_pool[p_idx].path_logp = new_lp;
          meta->pool_count = p_idx + 1u;
          sh_exp_lp[exp_cnt] = new_lp;
          sh_exp_tok[exp_cnt] = int(pid);
          sh_exp_pidx[exp_cnt] = p_idx;
          sh_exp_src[exp_cnt] = b;
          ++exp_cnt;
        }
      }
    }
  }

  if (!p.last_step) {
    if (tid == 0u) {
      const uint new_nb = min(4u, exp_cnt);
      bool used[32] = {};
      for (uint j = 0u; j < new_nb; ++j) {
        int best = -1;
        for (uint e = 0u; e < exp_cnt; ++e) {
          if (used[e]) continue;
          if (best < 0 || sh_exp_lp[e] > sh_exp_lp[best]) best = int(e);
        }
        used[best] = true;
        meta->tok[j] = sh_exp_tok[best];
        meta->pool_idx[j] = sh_exp_pidx[best];
        meta->logp[j] = sh_exp_lp[best];
        sh_src_beam[j] = sh_exp_src[best];
      }
      meta->n_beam = new_nb;
      sh_nb = new_nb;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint new_nb = sh_nb;
    for (uint j = 0u; j < new_nb; ++j) {
      const uint sb = sh_src_beam[j];
      for (uint c = tid; c < p.R; c += 128u) beam_state[size_t(j) * p.R + c] = beam_next_state[size_t(sb) * p.R + c];
    }
  }
}

[[kernel]] void ink2_mtp_tree_select(const device Ink2BeamState* meta [[buffer(0)]],
                                     const device Ink2CandNode* cand_pool [[buffer(1)]],
                                     device Ink2Tree* tree [[buffer(2)]], device int* tokens [[buffer(3)]],
                                     constant Ink2MtpTreeSelP& p [[buffer(4)]],
                                     uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint sh_idx[128], sh_dep[128];
  threadgroup float sh_lp[128];
  const uint n_pool = min(meta->pool_count, 128u);
  const uint pi = 1u + tid;
  if (pi < n_pool) {
    sh_idx[tid] = pi;
    sh_lp[tid] = cand_pool[pi].path_logp;
    sh_dep[tid] = cand_pool[pi].depth;
  } else {
    sh_idx[tid] = 0xffffffffu;
    sh_lp[tid] = -INFINITY;
    sh_dep[tid] = 999u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint k = 2u; k <= 128u; k <<= 1u) {
    for (uint j = k >> 1u; j > 0u; j >>= 1u) {
      const uint ixj = tid ^ j;
      if (ixj > tid) {
        const bool asc = ((tid & k) == 0u);
        const float lpa = sh_lp[tid], lpb = sh_lp[ixj];
        const uint da = sh_dep[tid], db = sh_dep[ixj];
        const uint ia = sh_idx[tid], ib = sh_idx[ixj];
        const bool a_better = (lpa > lpb) || (lpa == lpb && (da < db || (da == db && ia < ib)));
        if (asc ? !a_better : a_better) {
          sh_lp[tid] = lpb; sh_lp[ixj] = lpa;
          sh_dep[tid] = db; sh_dep[ixj] = da;
          sh_idx[tid] = ib; sh_idx[ixj] = ia;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  if (tid != 0u) return;
  const uint R_lim = clamp(p.R_budget, 1u, 16u);
  uchar sel[128];
  int p2t[128];
  for (uint i = 0u; i < 128u; ++i) {
    sel[i] = 0;
    p2t[i] = -1;
  }
  sel[0] = 1;
  uint sel_cnt = 1u;
  for (uint rank = 0u; rank < 128u && sel_cnt < R_lim; ++rank) {
    const uint idx = sh_idx[rank];
    if (idx >= n_pool || !isfinite(sh_lp[rank])) break;
    const int par = cand_pool[idx].parent_pool;
    if (par >= 0 && par < int(n_pool) && sel[par]) {
      sel[idx] = 1;
      ++sel_cnt;
    }
  }

  tree->depth[0] = 0u;
  tree->anc_mask[0] = 1u;
  tree->parent[0] = -1;
  tree->par1[0] = -1;
  tree->par2[0] = -2;
  tree->par3[0] = -3;
  tree->tokens[0] = cand_pool[0].tok;
  for (uint r = 0u; r < 16u; ++r) tree->n_child[r] = 0u;
  tokens[p.pos] = cand_pool[0].tok;
  p2t[0] = 0;

  uint r_out = 1u;
  for (uint i = 1u; i < n_pool && r_out < R_lim; ++i) {
    if (!sel[i]) continue;
    const int par_p = cand_pool[i].parent_pool;
    const int tp = (par_p >= 0 && par_p < int(n_pool)) ? p2t[par_p] : -1;
    if (tp < 0) continue;
    const uint r = r_out++;
    p2t[i] = int(r);
    tree->parent[r] = tp;
    tree->depth[r] = tree->depth[tp] + 1u;
    tree->anc_mask[r] = tree->anc_mask[tp] | (1u << r);
    tree->par1[r] = tp;
    tree->par2[r] = tree->par1[tp];
    tree->par3[r] = tree->par2[tp];
    tree->tokens[r] = cand_pool[i].tok;
    const uint nc = tree->n_child[tp];
    if (nc < 15u) tree->child[tp][nc] = r, tree->n_child[tp] = nc + 1u;
    tokens[p.pos + r] = cand_pool[i].tok;
  }
  while (r_out < R_lim) {
    const uint r = r_out++;
    tree->parent[r] = 0;
    tree->depth[r] = 1u;
    tree->anc_mask[r] = 1u | (1u << r);
    tree->par1[r] = 0;
    tree->par2[r] = -1;
    tree->par3[r] = -2;
    tree->tokens[r] = cand_pool[0].tok;
    tokens[p.pos + r] = cand_pool[0].tok;
  }
  tree->R = R_lim;
}

[[kernel]] void ink2_mtp_accept_tree_greedy(const device Ink2Tree* tree [[buffer(0)]],
                                            const device int* verify_samples [[buffer(1)]],
                                            device int* tokens [[buffer(2)]],
                                            device Ink2TreeAccept* out [[buffer(3)]],
                                            constant Ink2MtpTreeAccP& p [[buffer(4)]],
                                            uint tid [[thread_index_in_threadgroup]]) {
  if (tid != 0u) return;
  const uint R_lim = min(tree->R, 16u);
  int m = 1;
  uint u = 0u;
  out->path[0] = 0;
  while (uint(m) < R_lim) {
    const int target = verify_samples[u];
    const uint nc = min(tree->n_child[u], 15u);
    int next_u = -1;
    for (uint j = 0u; j < nc; ++j) {
      const uint c = tree->child[u][j];
      if (c < R_lim && tree->tokens[c] == target) {
        next_u = int(c);
        break;
      }
    }
    if (next_u < 0) break;
    tokens[p.pos + uint(m)] = target;
    out->path[m] = next_u;
    u = uint(next_u);
    ++m;
  }
  const int bonus = verify_samples[u];
  tokens[p.pos + uint(m)] = bonus;
  out->m = m;
  out->stop_node = int(u);
  out->bonus_tok = bonus;
  out->ok = 1;
}

[[kernel]] void ink2_mtp_accept_tree_full_decide(const device Ink2Tree* tree [[buffer(0)]],
                                                 const device float* verify_logits [[buffer(1)]],
                                                 const device float2* stats [[buffer(2)]],
                                                 device int* tokens [[buffer(3)]],
                                                 device Ink2TreeAccept* out [[buffer(4)]],
                                                 constant Ink2MtpTreeAccP& p [[buffer(5)]],
                                                 uint tid [[thread_index_in_threadgroup]]) {
  if (tid != 0u) return;
  const uint R_lim = min(tree->R, 16u);
  int m = 1;
  uint u = 0u;
  out->path[0] = 0;
  while (uint(m) < R_lim) {
    const uint nc = min(tree->n_child[u], 15u);
    if (nc == 0u) break;
    const float2 sp_p = stats[u];
    const device float* vp = verify_logits + size_t(u) * p.V;
    const float u_acc = ink2m_u01(ink2m_philox4x32_10(uint4(u, 0x9e3779b9u, 0x7f4a7c15u, 1u), uint2(p.seed, p.step + u)));
    float cum = 0.0f;
    int next_u = -1;
    int next_tok = 0;
    for (uint j = 0u; j < nc; ++j) {
      const uint c = tree->child[u][j];
      if (c >= R_lim) continue;
      const int ctok = tree->tokens[c];
      float pc = 0.0f;
      if (ctok >= 0 && uint(ctok) < p.V && isfinite(sp_p.y)) {
        const float lp = vp[uint(ctok)];
        if (!isnan(lp) && lp > -INFINITY) pc = metal::precise::exp((lp - sp_p.x) * p.inv_t - sp_p.y);
      }
      cum += pc;
      if (u_acc < cum) {
        next_u = int(c);
        next_tok = ctok;
        break;
      }
    }
    if (next_u < 0) break;
    tokens[p.pos + uint(m)] = next_tok;
    out->path[m] = next_u;
    u = uint(next_u);
    ++m;
  }
  out->m = m;
  out->stop_node = int(u);
  out->bonus_tok = 0;
  out->ok = 0;
}

[[kernel]] void ink2_mtp_sample_emit_tree_stage1(const device Ink2Tree* tree [[buffer(0)]],
                                                 const device float* verify_logits [[buffer(1)]],
                                                 const device float2* stats [[buffer(2)]],
                                                 const device Ink2TreeAccept* out [[buffer(3)]],
                                                 device float* cand_v [[buffer(4)]], device uint* cand_i [[buffer(5)]],
                                                 constant Ink2MtpTreeAccP& p [[buffer(6)]],
                                                 uint bid [[threadgroup_position_in_grid]],
                                                 uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint ntg = 256u, nsg = 8u;
  threadgroup float sv[8], sfv[8];
  threadgroup uint si[8], sfi[8];
  const uint u = uint(out->stop_node);
  const uint nc = min(tree->n_child[u], 15u);
  uint rej[15];
  for (uint j = 0u; j < nc; ++j) rej[j] = uint(tree->tokens[tree->child[u][j]]);
  const float2 sp_p = stats[u];
  const device float* vp = verify_logits + size_t(u) * p.V;
  const uint2 key = uint2(p.seed, p.step + u);

  float bv = -INFINITY, fv = -INFINITY;
  uint bi = 0xffffffffu, fi = 0xffffffffu;
  for (uint v = bid * ntg + tid; v < p.V; v += 64u * ntg) {
    const float lp = vp[v];
    if (isnan(lp) || !(lp > -INFINITY)) continue;
    const float u_rng = ink2m_u01(ink2m_philox4x32_10(uint4(v, u, 1u, 0x517cc1b7u), key));
    const float g = -metal::precise::log(-metal::precise::log(u_rng));
    const float s_fb = (lp - sp_p.x) * p.inv_t + g;
    if (s_fb > fv || (s_fb == fv && v < fi)) {
      fv = s_fb;
      fi = v;
    }
    bool rejected = false;
    for (uint j = 0u; j < nc; ++j) {
      if (v == rej[j]) {
        rejected = true;
        break;
      }
    }
    if (!rejected && (s_fb > bv || (s_fb == bv && v < bi))) {
      bv = s_fb;
      bi = v;
    }
  }
  ink2m_simd_argmax(bv, bi);
  ink2m_simd_argmax(fv, fi);
  const uint sg = tid >> 5u, lane = tid & 31u;
  if (lane == 0u) {
    sv[sg] = bv;
    si[sg] = bi;
    sfv[sg] = fv;
    sfi[sg] = fi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float m2 = lane < nsg ? sv[lane] : -INFINITY;
    uint i2 = lane < nsg ? si[lane] : 0xffffffffu;
    ink2m_simd_argmax(m2, i2);
    float fm2 = lane < nsg ? sfv[lane] : -INFINITY;
    uint fi2 = lane < nsg ? sfi[lane] : 0xffffffffu;
    ink2m_simd_argmax(fm2, fi2);
    if (lane == 0u) {
      cand_v[bid] = m2;
      cand_i[bid] = i2;
      cand_v[64u + bid] = fm2;
      cand_i[64u + bid] = fi2;
    }
  }
}

[[kernel]] void ink2_mtp_sample_emit_tree_stage2(const device float* cand_v [[buffer(0)]],
                                                 const device uint* cand_i [[buffer(1)]],
                                                 device Ink2TreeAccept* out [[buffer(2)]],
                                                 device int* tokens [[buffer(3)]],
                                                 constant Ink2MtpTreeAccP& p [[buffer(4)]],
                                                 uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sv[2], sfv[2];
  threadgroup uint si[2], sfi[2];
  float v = cand_v[tid], fv = cand_v[64u + tid];
  uint idx = cand_i[tid], fidx = cand_i[64u + tid];
  ink2m_simd_argmax(v, idx);
  ink2m_simd_argmax(fv, fidx);
  const uint sg = tid >> 5u, lane = tid & 31u;
  if (lane == 0u) {
    sv[sg] = v;
    si[sg] = idx;
    sfv[sg] = fv;
    sfi[sg] = fidx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float m2 = lane < 2u ? sv[lane] : -INFINITY;
    uint i2 = lane < 2u ? si[lane] : 0xffffffffu;
    ink2m_simd_argmax(m2, i2);
    float fm2 = lane < 2u ? sfv[lane] : -INFINITY;
    uint fi2 = lane < 2u ? sfi[lane] : 0xffffffffu;
    ink2m_simd_argmax(fm2, fi2);
    if (lane == 0u) {
      const uint pick = (i2 != 0xffffffffu) ? i2 : fi2;
      const int tok_out = (pick == 0xffffffffu) ? 0 : int(pick);
      tokens[p.pos + uint(out->m)] = tok_out;
      out->bonus_tok = tok_out;
      out->ok = 1;
    }
  }
}

[[kernel]] void ink2_mtp_accept_tree_topk_decide(const device Ink2Tree* tree [[buffer(0)]],
                                                 const device Ink2MtpDist* dists [[buffer(1)]],
                                                 device int* tokens [[buffer(2)]],
                                                 device Ink2TreeAccept* out [[buffer(3)]],
                                                 constant Ink2MtpTreeAccP& p [[buffer(4)]],
                                                 uint tid [[thread_index_in_threadgroup]]) {
  if (tid != 0u) return;
  const uint R_lim = min(tree->R, 16u);
  int m = 1;
  uint u = 0u;
  out->path[0] = 0;
  while (uint(m) < R_lim) {
    const uint nc = min(tree->n_child[u], 15u);
    if (nc == 0u) break;
    const device Ink2MtpDist& dp = dists[u];
    const float u_acc = ink2m_u01(ink2m_philox4x32_10(uint4(u, 0x9e3779b9u, 0x7f4a7c15u, 2u), uint2(p.seed, p.step + u)));
    float cum = 0.0f;
    int next_u = -1;
    int next_tok = 0;
    for (uint k = 0u; k < nc; ++k) {
      const uint c = tree->child[u][k];
      if (c >= R_lim) continue;
      const uint ctok = uint(tree->tokens[c]);
      float pc = 0.0f;
      for (uint j = 0u; j < dp.keep; ++j) {
        if (dp.ids[j] == ctok) {
          pc = dp.probs[j];
          break;
        }
      }
      cum += pc;
      if (u_acc < cum) {
        next_u = int(c);
        next_tok = int(ctok);
        break;
      }
    }
    if (next_u < 0) break;
    tokens[p.pos + uint(m)] = next_tok;
    out->path[m] = next_u;
    u = uint(next_u);
    ++m;
  }

  const uint nc = min(tree->n_child[u], 15u);
  const device Ink2MtpDist& dp = dists[u];
  uint rids[64];
  float rw[64];
  uint rn = 0u;
  float rsum = 0.0f;
  for (uint j = 0u; j < dp.keep; ++j) {
    const uint vid = dp.ids[j];
    bool rej = false;
    for (uint k = 0u; k < nc; ++k) {
      const uint c = tree->child[u][k];
      if (c < R_lim && uint(tree->tokens[c]) == vid) {
        rej = true;
        break;
      }
    }
    if (!rej && dp.probs[j] > 0.0f) {
      rids[rn] = vid;
      rw[rn] = dp.probs[j];
      rsum += dp.probs[j];
      ++rn;
    }
  }
  const float u_res = ink2m_u01(ink2m_philox4x32_10(uint4(u, 0x517cc1b7u, 1u, 3u), uint2(p.seed, p.step + u)));
  uint chosen = dp.ids[0];
  if (rsum > 0.0f && rn > 0u) {
    const float target = u_res * rsum;
    float acc = 0.0f;
    chosen = rids[rn - 1u];
    for (uint j = 0u; j < rn; ++j) {
      acc += rw[j];
      if (target < acc) {
        chosen = rids[j];
        break;
      }
    }
  } else {
    float acc = 0.0f;
    chosen = dp.ids[dp.keep - 1u];
    for (uint j = 0u; j < dp.keep; ++j) {
      acc += dp.probs[j];
      if (u_res < acc) {
        chosen = dp.ids[j];
        break;
      }
    }
  }
  tokens[p.pos + uint(m)] = int(chosen);
  out->m = m;
  out->stop_node = int(u);
  out->bonus_tok = int(chosen);
  out->ok = 1;
}
