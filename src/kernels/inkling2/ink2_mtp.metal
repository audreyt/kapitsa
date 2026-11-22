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
  float xs[6];
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
    float xs[6];
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
  float xs[6];
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
