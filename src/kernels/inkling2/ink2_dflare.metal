#include "common.h"

// Inkling-2-Mini bundled DFlare / DSpark block drafter (mtp.safetensors):
// 3 Qwen3-style layers (RoPE theta=1e6, QK-norm, SwiGLU MLP, sliding context window 1023)
// with per-layer softmax tap fusion (dflare_norm) + bidirectional block attention +
// RNN Markov head (rank 256, top-32 candidate rescoring against target unembed).

struct DflMixP {
  uint R, H, row_off, total_rows;
  float a0, a1, a2, a3, eps;
};

// Mixes 4 target taps [4, total_rows, H] for rows [row_off .. row_off + R) with alpha, RMSNorm -> out [R, H]
[[kernel]] void dflare_mix_norm(const device float* taps [[buffer(0)]], const device float* w [[buffer(1)]],
                                device float* out [[buffer(2)]], constant DflMixP& p [[buffer(3)]],
                                uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const size_t stride = size_t(p.total_rows) * p.H;
  const size_t roff = size_t(p.row_off + r) * p.H;
  const device float* t0 = taps + 0 * stride + roff;
  const device float* t1 = taps + 1 * stride + roff;
  const device float* t2 = taps + 2 * stride + roff;
  const device float* t3 = taps + 3 * stride + roff;
  float xs[8];
  float ss = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float v = p.a0 * t0[c] + p.a1 * t1[c] + p.a2 * t2[c] + p.a3 * t3[c];
    xs[k] = v;
    ss += v * v;
  }
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
  device float* dst = out + size_t(r) * p.H;
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) dst[c] = xs[k] * inv * w[c];
}

struct DflQ8P {
  uint M, N, K, add;
};

// Coalesced Q8_0 matmul for draft context projections (M <= 64, 2D grid: tg.x = n/4, tg.y = m/8).
[[kernel]] void dflare_q8_mm(const device float* x [[buffer(0)]], const device uchar* q [[buffer(1)]],
                             const device half* s [[buffer(2)]], device float* y [[buffer(3)]],
                             constant DflQ8P& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 4u + sg;
  if (n >= p.N) return;
  const uint m0 = tg.y * 8u;
  const uint mc = min(8u, p.M - m0);
  const uint blocks = p.K / 32u;
  const size_t qoff = size_t(n) * p.K;
  const size_t soff = size_t(n) * blocks;
  const uint kl = lane * 8u, blk_loc = lane >> 2u;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    if (kl < p.K - kbase) {
      const uint blk = (kbase >> 5u) + blk_loc;
      const uint2 qw = *(const device uint2*)(q + qoff + kbase + kl);
      const float sc = float(s[soff + blk]);
      const float4 wa = float4(as_type<char4>(qw.x)) * sc, wb = float4(as_type<char4>(qw.y)) * sc;
      for (uint m = 0u; m < mc; ++m) {
        const device float* xp = x + size_t(m0 + m) * p.K + kbase + kl;
        acc[m] += dot(*(const device float4*)(xp), wa) + dot(*(const device float4*)(xp + 4u), wb);
      }
    }
  }
  for (uint m = 0u; m < mc; ++m) {
    const float sum = simd_sum(acc[m]);
    if (lane == 0u) {
      device float* dst = y + size_t(m0 + m) * p.N + n;
      *dst = p.add ? (*dst + sum) : sum;
    }
  }
}

struct DflCtxKvP {
  uint R, Hkv, D, pos0, cap;
  float eps, theta;
};

// Apply k_norm_ctx + NeoX split-half RoPE to ctx_k [R, Hkv, D] and write half kc/vc into ring [cap, Hkv*D]
[[kernel]] void dflare_ctx_kv_write(const device float* kin [[buffer(0)]], const device float* vin [[buffer(1)]],
                                    const device float* knw [[buffer(2)]], device half* kc [[buffer(3)]],
                                    device half* vc [[buffer(4)]], constant DflCtxKvP& p [[buffer(5)]],
                                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  threadgroup float kbuf[128];
  const uint h = tg.x, r = tg.y;
  const uint pos = p.pos0 + r;
  const uint slot = pos % p.cap;
  const size_t in_off = (size_t(r) * p.Hkv + h) * p.D;
  const size_t out_off = (size_t(slot) * p.Hkv + h) * p.D;

  const float kv = kin[in_off + tid];
  const float ss = tg_sum(kv * kv, scratch, tid, p.D);
  const float inv = metal::precise::rsqrt(ss / float(p.D) + p.eps);
  kbuf[tid] = kv * inv * knw[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint half_d = p.D / 2;
  if (tid < half_d) {
    const float freq = metal::precise::pow(p.theta, -float(2 * tid) / float(p.D));
    const float ang = float(pos) * freq;
    const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    const float x0 = kbuf[tid], x1 = kbuf[tid + half_d];
    kc[out_off + tid] = half(x0 * cs - x1 * sn);
    kc[out_off + tid + half_d] = half(x0 * sn + x1 * cs);
  }
  vc[out_off + tid] = half(vin[in_off + tid]);
}

struct DflBlockP {
  uint B, H, anchor_idx, V;
  float eps;
};

// Build draft block input [B, H]: row 0 = embed_norm(embed[tokens[anchor_idx]]), rows 1..B-1 = mask_embed
[[kernel]] void dflare_build_block(const device int* tokens [[buffer(0)]], const device bfloat* embed [[buffer(1)]],
                                   const device bfloat* embed_norm [[buffer(2)]], const device float* mask_emb [[buffer(3)]],
                                   device float* out [[buffer(4)]], constant DflBlockP& p [[buffer(5)]],
                                   uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                   uint ntg [[threads_per_threadgroup]]) {
  device float* dst = out + size_t(b) * p.H;
  if (b > 0) {
    for (uint c = tid; c < p.H; c += ntg) dst[c] = mask_emb[c];
    return;
  }
  threadgroup float scratch[32];
  const int raw_id = tokens[p.anchor_idx];
  const uint id = (raw_id >= 0 && uint(raw_id) < p.V) ? uint(raw_id) : 0u;
  const device bfloat* row = embed + size_t(id) * p.H;
  float xs[8];
  float ss = 0.0f;
  uint k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) {
    const float v = float(row[c]);
    xs[k] = v;
    ss += v * v;
  }
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(p.H) + p.eps);
  k = 0;
  for (uint c = tid; c < p.H; c += ntg, ++k) dst[c] = xs[k] * inv * float(embed_norm[c]);
}

struct DflQkRopeP {
  uint B, Hq, Hkv, D, pos0;
  float eps, theta;
};

// In-place q_norm + NeoX RoPE on q [B, Hq, D] and k_norm + NeoX RoPE on k [B, Hkv, D]
[[kernel]] void dflare_qk_rope(device float* q [[buffer(0)]], device float* k [[buffer(1)]],
                               const device float* qnw [[buffer(2)]], const device float* knw [[buffer(3)]],
                               constant DflQkRopeP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  threadgroup float buf[128];
  const uint kh = tg.x, b = tg.y;
  const bool is_q = kh < p.Hq;
  const uint h = is_q ? kh : (kh - p.Hq);
  device float* vec = (is_q ? q : k) + (size_t(b) * (is_q ? p.Hq : p.Hkv) + h) * p.D;
  const device float* nw = is_q ? qnw : knw;
  const float val = vec[tid];
  const float ss = tg_sum(val * val, scratch, tid, p.D);
  const float inv = metal::precise::rsqrt(ss / float(p.D) + p.eps);
  buf[tid] = val * inv * nw[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint half_d = p.D / 2;
  if (tid < half_d) {
    const float freq = metal::precise::pow(p.theta, -float(2 * tid) / float(p.D));
    const float ang = float(p.pos0 + b) * freq;
    const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    const float x0 = buf[tid], x1 = buf[tid + half_d];
    vec[tid] = x0 * cs - x1 * sn;
    vec[tid + half_d] = x0 * sn + x1 * cs;
  }
}

struct DflAttnP {
  uint B, Hq, Hkv, D, pos0, ctx_len, ctx_cap;
  float scale;
};

// Bidirectional block attention over [ctx_len context ring keys, B block keys] (no causal mask, no sinks)
[[kernel]] void dflare_attn(const device float* q [[buffer(0)]], const device half* ctx_k [[buffer(1)]],
                            const device half* ctx_v [[buffer(2)]], const device float* blk_k [[buffer(3)]],
                            const device float* blk_v [[buffer(4)]], device float* out [[buffer(5)]],
                            constant DflAttnP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgm[4], tgl[4], tgo[4][128];
  const uint h = tg.x, b = tg.y;
  const uint hkv = h / (p.Hq / p.Hkv);
  const uint d0 = lane * 4;
  const size_t qoff = (size_t(b) * p.Hq + h) * p.D + d0;
  const float4 qv = *(const device float4*)(q + qoff);

  const uint c_start = p.pos0 - p.ctx_len;
  float mw = -INFINITY, lw = 0.0f;
  float4 ow = float4(0.0f);

  // Context ring keys
  for (uint i = sg; i < p.ctx_len; i += 4) {
    const uint slot = (c_start + i) % p.ctx_cap;
    const size_t koff = (size_t(slot) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = float4(*(const device half4*)(ctx_k + koff));
    const float sc = simd_sum(dot(qv, kv)) * p.scale;
    const float4 vv = float4(*(const device half4*)(ctx_v + koff));
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
  // Bidirectional block keys (all B positions)
  for (uint j = sg; j < p.B; j += 4) {
    const size_t bkoff = (size_t(j) * p.Hkv + hkv) * p.D + d0;
    const float4 kv = *(const device float4*)(blk_k + bkoff);
    const float sc = simd_sum(dot(qv, kv)) * p.scale;
    const float4 vv = *(const device float4*)(blk_v + bkoff);
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

// ---- RNN Markov Head (rank R=256, 2-bit proxy screening + top-128 exact rescoring) ----
struct DflMarkovP {
  uint R, H, K, step, tok_prev_idx, tok_out_idx;
  float softcap, inv_softcap;
};

// Step 1 of Markov head: build concat([state(R), w1[prev_tok](R), draft_h[step](H), rows_f(H)]) -> [2R+2H]
[[kernel]] void dflare_markov_concat(const device float* state [[buffer(0)]], const device bfloat* w1 [[buffer(1)]],
                                     const device float* draft_h [[buffer(2)]], const device float* rows_f [[buffer(3)]],
                                     const device int* tokens [[buffer(4)]], device float* rnn_in [[buffer(5)]],
                                     constant DflMarkovP& p [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  const uint tot = 2u * p.R + 2u * p.H;
  if (i >= tot) return;
  if (i < p.R) {
    rnn_in[i] = p.step == 0u ? 0.0f : state[i];
  } else if (i < 2u * p.R) {
    const int tok = tokens[p.tok_prev_idx];
    const uint tid = tok >= 0 ? uint(tok) : 0u;
    rnn_in[i] = float(w1[size_t(tid) * p.R + (i - p.R)]);
  } else if (i < 2u * p.R + p.H) {
    rnn_in[i] = draft_h[size_t(p.step) * p.H + (i - 2u * p.R)];
  } else {
    rnn_in[i] = rows_f[i - (2u * p.R + p.H)];
  }
}

// Step 2: rnn_proj [3R, 2R+2H] BF16 + bias -> update state [R] & write latent [R]
[[kernel]] void dflare_markov_gru(const device float* rnn_in [[buffer(0)]], const device bfloat* proj_w [[buffer(1)]],
                                  const device float* proj_b [[buffer(2)]], device float* state [[buffer(3)]],
                                  device float* latent [[buffer(4)]], constant DflMarkovP& p [[buffer(5)]],
                                  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg * 4u + sg;
  if (r >= p.R) return;
  const uint K = 2u * p.R + 2u * p.H;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
  const device bfloat* w0 = proj_w + size_t(r) * K;
  const device bfloat* w1 = proj_w + size_t(r + p.R) * K;
  const device bfloat* w2 = proj_w + size_t(r + 2u * p.R) * K;
  for (uint k = lane * 4u; k < K; k += 128u) {
    const float4 xv = *(const device float4*)(rnn_in + k);
    const uint2 q0 = *(const device uint2*)(w0 + k);
    const uint2 q1 = *(const device uint2*)(w1 + k);
    const uint2 q2 = *(const device uint2*)(w2 + k);
    const float4 f0 = float4(as_type<float>(q0.x << 16), as_type<float>(q0.x & 0xFFFF0000u),
                             as_type<float>(q0.y << 16), as_type<float>(q0.y & 0xFFFF0000u));
    const float4 f1 = float4(as_type<float>(q1.x << 16), as_type<float>(q1.x & 0xFFFF0000u),
                             as_type<float>(q1.y << 16), as_type<float>(q1.y & 0xFFFF0000u));
    const float4 f2 = float4(as_type<float>(q2.x << 16), as_type<float>(q2.x & 0xFFFF0000u),
                             as_type<float>(q2.y << 16), as_type<float>(q2.y & 0xFFFF0000u));
    a0 += dot(xv, f0);
    a1 += dot(xv, f1);
    a2 += dot(xv, f2);
  }
  const float p0 = simd_sum(a0) + proj_b[r];
  const float p1 = simd_sum(a1) + proj_b[r + p.R];
  const float p2 = simd_sum(a2) + proj_b[r + 2u * p.R];
  if (lane == 0) {
    const float st_prev = p.step == 0u ? 0.0f : state[r];
    const float gate = 1.0f / (1.0f + metal::precise::exp(-p0));
    state[r] = gate * st_prev + (1.0f - gate) * metal::precise::tanh(p1);
    latent[r] = metal::precise::tanh(p2);
  }
}

NI_INLINE void dfl_p2_pack(float val, uint lane, thread uint2& q_out, thread float& alpha_out) {
  alpha_out = (simd_sum(fabs(val)) * (1.0f / 32.0f)) * 1.25f;
  const uint c = (val >= alpha_out) ? 3u : ((val >= 0.0f) ? 2u : ((val >= -alpha_out) ? 1u : 0u));
  const uint sh = 2u * (lane & 15u);
  q_out = uint2(simd_sum((lane < 16u ? c : 0u) << sh), simd_sum((lane >= 16u ? c : 0u) << sh));
}

// One-time load kernel: fold unembed [V, H] @ hidden_out.weight [H, 256] -> Q8_0 + 2-bit proxy [V, 256]
struct DflFoldP {
  uint V, H, R, u_f32;
};

[[kernel]] void dflare_fold_mm_q8(const device void* unembed [[buffer(0)]], const device bfloat* hout_w [[buffer(1)]],
                                  device uchar* fold_q [[buffer(2)]], device half* fold_s [[buffer(3)]],
                                  device uint2* fold_p2_q [[buffer(4)]], device half* fold_p2_s [[buffer(5)]],
                                  constant DflFoldP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[32][36];
  threadgroup float ws[32][36];
  const uint blk = tg.x;        // 0..7 (each block = 32 cols of R=256)
  const uint r0 = blk * 32u;
  const uint v0 = tg.y * 32u;
  const uint sgm = sg >> 1;     // 0..1 (16 rows)
  const uint sgn = sg & 1u;     // 0..1 (16 cols)

  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

  for (uint k0 = 0; k0 < p.H; k0 += 32u) {
    for (uint idx = tid; idx < 1024u; idx += 128u) {
      const uint r = idx >> 5, c = idx & 31u;
      const uint v = v0 + r;
      float uv = 0.0f;
      if (v < p.V && k0 + c < p.H) {
        const size_t uidx = size_t(v) * p.H + (k0 + c);
        uv = p.u_f32 ? ((const device float*)unembed)[uidx] : float(((const device bfloat*)unembed)[uidx]);
      }
      xs[r][c] = uv;
      ws[r][c] = (k0 + r < p.H && r0 + c < p.R) ? float(hout_w[size_t(k0 + r) * p.R + (r0 + c)]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 32u; kk += 8u) {
      simdgroup_float8x8 a[2], b[2];
      for (uint i = 0; i < 2; ++i) simdgroup_load(a[i], &xs[sgm * 16 + i * 8][kk], 36);
      for (uint j = 0; j < 2; ++j) simdgroup_load(b[j], &ws[kk][sgn * 16 + j * 8], 36);
      for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j) simdgroup_store(acc[i][j], &xs[sgm * 16 + i * 8][sgn * 16 + j * 8], 36);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint nblk = p.R / 32u;
  for (uint i = 0; i < 8u; ++i) {
    const uint r = sg * 8u + i;
    const uint v = v0 + r;
    if (v >= p.V) continue;
    const float val = xs[r][lane];
    const float amax = simd_max(fabs(val));
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? (1.0f / d) : 0.0f;
    fold_q[size_t(v) * p.R + r0 + lane] = uchar(char(clamp(int(rint(val * id)), -127, 127)));
    uint2 p2q; float alpha;
    dfl_p2_pack(val, lane, p2q, alpha);
    if (lane == 0u) {
      fold_s[size_t(v) * nblk + blk] = half(d);
      fold_p2_q[size_t(v) * nblk + blk] = p2q;
      fold_p2_s[size_t(v) * nblk + blk] = half(alpha);
    }
  }
}

NI_INLINE float4 dfl_bf16_4(uint2 r) {
  return float4(as_type<float>(r.x << 16), as_type<float>(r.x & 0xFFFF0000u),
                as_type<float>(r.y << 16), as_type<float>(r.y & 0xFFFF0000u));
}

NI_INLINE void dfl_p2_unpack(uint2 q, thread float4 c[8]) {
  _Pragma("clang loop unroll(full)")
  for (uint i = 0u; i < 4u; ++i) {
    const uint bx = q.x >> (i * 8u), by = q.y >> (i * 8u);
    c[i] = float4(float(bx & 3u), float((bx >> 2u) & 3u), float((bx >> 4u) & 3u), float((bx >> 6u) & 3u));
    c[i + 4u] = float4(float(by & 3u), float((by >> 2u) & 3u), float((by >> 4u) & 3u), float((by >> 6u) & 3u));
  }
}

NI_INLINE float dfl_p2_dot32(const thread float4 c[8], const threadgroup float4* xmp, float blk_sum, float sc) {
  float s = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint i = 0u; i < 8u; ++i) s += dot(xmp[i], c[i]);
  return (s - 1.5f * blk_sum) * sc;
}

NI_INLINE bool dfl_tok_allowed(const device uint* sup_mask, uint v, uint has_sup, uint sup_words) {
  return !has_sup || (v >> 5u) >= sup_words || (((sup_mask[v >> 5u] >> (v & 31u)) & 1u) != 0u);
}

NI_INLINE void dfl_top2(float v, uint i, thread float& m0v, thread uint& m0i, thread float& m1v, thread uint& m1i) {
  if (v > m0v || (v == m0v && i < m0i)) { m1v = m0v; m1i = m0i; m0v = v; m0i = i; }
  else if (v > m1v || (v == m1v && i < m1i)) { m1v = v; m1i = i; }
}

// ---- GPU Load-Time Quantizers (Q8_0 and 2-bit proxy) ----
struct DflQuantP { uint total_blocks, w_f32; };

[[kernel]] void dflare_quant_q8_bf16(const device bfloat* w [[buffer(0)]], device uchar* q [[buffer(1)]],
                                     device half* s [[buffer(2)]], constant DflQuantP& p [[buffer(3)]],
                                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint blk = tg * 4u + sg;
  if (blk >= p.total_blocks) return;
  const uint idx = blk * 32u + lane;
  const float v = float(w[idx]), d = simd_max(fabs(v)) * (1.0f / 127.0f), id = d > 0.0f ? (1.0f / d) : 0.0f;
  q[idx] = uchar(char(clamp(int(rint(v * id)), -127, 127)));
  if (lane == 0u) s[blk] = half(d);
}

[[kernel]] void dflare_quant_p2_w(const device void* w [[buffer(0)]], device uint2* q [[buffer(1)]],
                                  device half* s [[buffer(2)]], constant DflQuantP& p [[buffer(3)]],
                                  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  const uint blk = tg * 4u + sg;
  if (blk >= p.total_blocks) return;
  const uint idx = blk * 32u + lane;
  const float v = p.w_f32 ? ((const device float*)w)[idx] : float(((const device bfloat*)w)[idx]);
  uint2 p2q; float alpha;
  dfl_p2_pack(v, lane, p2q, alpha);
  if (lane == 0u) { q[blk] = p2q; s[blk] = half(alpha); }
}

// ---- Fused Q8_0 Draft Backbone Kernels (M <= 8) ----
struct DflQ8MvP { uint M, N, K, nblk, add; };
struct DflQ8Multi3P { uint M, K, N_tot, nblk; uint offs[4]; };

template <bool MULTI3, uint M_MAX>
NI_INLINE void dfl_q8_mv_impl(const device float* x, const device uchar* wq, const device half* ws,
                              device float* y0, device float* y1, device float* y2, const uint offs[4],
                              uint M, uint N_tot, uint K, uint nblk, uint add, uint tg, uint sg, uint lane) {
  const uint n0 = (tg * 4u + sg) * 2u;
  if (n0 >= N_tot) return;
  uint seg = 0u;
  if (MULTI3) while (seg < 2u && n0 >= offs[seg + 1u]) ++seg;
  const uint seg_off = MULTI3 ? offs[seg] : 0u, seg_end = MULTI3 ? offs[seg + 1u] : N_tot, seg_len = seg_end - seg_off;
  if (n0 >= seg_end) return;
  const bool has1 = (n0 + 1u) < seg_end;
  const size_t woff0 = size_t(n0) * K, woff1 = has1 ? (woff0 + K) : woff0;
  const size_t soff0 = size_t(n0) * nblk, soff1 = has1 ? (soff0 + nblk) : soff0;
  const uint kl = lane * 8u, blk_loc = lane >> 2u;
  float acc0[M_MAX] = {}, acc1[M_MAX] = {};
  for (uint kbase = 0u; kbase < K; kbase += 256u) {
    if (kl < K - kbase) {
      const uint blk = (kbase >> 5u) + blk_loc;
      const uint2 qw0 = *(const device uint2*)(wq + woff0 + kbase + kl), qw1 = has1 ? *(const device uint2*)(wq + woff1 + kbase + kl) : qw0;
      const float sc0 = float(ws[soff0 + blk]), sc1 = has1 ? float(ws[soff1 + blk]) : sc0;
      const float4 w0a = float4(as_type<char4>(qw0.x)) * sc0, w0b = float4(as_type<char4>(qw0.y)) * sc0;
      const float4 w1a = float4(as_type<char4>(qw1.x)) * sc1, w1b = float4(as_type<char4>(qw1.y)) * sc1;
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < M_MAX; ++m) if (m < M) {
        const device float* xp = x + size_t(m) * K + kbase + kl;
        const float4 x0 = *(const device float4*)(xp + 0u), x1 = *(const device float4*)(xp + 4u);
        acc0[m] += dot(x0, w0a) + dot(x1, w0b);
        if (has1) acc1[m] += dot(x0, w1a) + dot(x1, w1b);
      }
    }
  }
  device float* bases[3] = {y0, y1, y2};
  device float* dst_base = (MULTI3 ? bases[seg] : y0) + (n0 - seg_off);
  _Pragma("clang loop unroll(full)")
  for (uint m = 0u; m < M_MAX; ++m) {
    if (m >= M) break;
    const float s0 = simd_sum(acc0[m]), s1 = has1 ? simd_sum(acc1[m]) : 0.0f;
    if (lane == 0u) {
      device float* dst = dst_base + size_t(m) * seg_len;
      dst[0] = (!MULTI3 && add) ? (dst[0] + s0) : s0;
      if (has1) dst[1] = (!MULTI3 && add) ? (dst[1] + s1) : s1;
    }
  }
}

[[kernel]] void dflare_q8_mv8(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                              const device half* ws [[buffer(2)]], device float* y [[buffer(3)]],
                              constant DflQ8MvP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint dummy[4] = {0u, p.N, p.N, p.N};
  if (p.M <= 4u) dfl_q8_mv_impl<false, 4u>(x, wq, ws, y, y, y, dummy, p.M, p.N, p.K, p.nblk, p.add, tg, sg, lane);
  else dfl_q8_mv_impl<false, 8u>(x, wq, ws, y, y, y, dummy, p.M, p.N, p.K, p.nblk, p.add, tg, sg, lane);
}

[[kernel]] void dflare_q8_mv8_multi3(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                                     const device half* ws [[buffer(2)]], device float* y0 [[buffer(3)]],
                                     device float* y1 [[buffer(4)]], device float* y2 [[buffer(5)]],
                                     constant DflQ8Multi3P& p [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],
                                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint offs[4] = {p.offs[0], p.offs[1], p.offs[2], p.offs[3]};
  if (p.M <= 4u) dfl_q8_mv_impl<true, 4u>(x, wq, ws, y0, y1, y2, offs, p.M, p.N_tot, p.K, p.nblk, 0u, tg, sg, lane);
  else dfl_q8_mv_impl<true, 8u>(x, wq, ws, y0, y1, y2, offs, p.M, p.N_tot, p.K, p.nblk, 0u, tg, sg, lane);
}

template <uint M_MAX>
NI_INLINE void dfl_q8_gu_impl(const device float* x, const device uchar* wq, const device half* ws,
                              device float* out, constant DflQ8MvP& p, uint tg, uint sg, uint lane) {
  const uint j = tg * 4u + sg;
  if (j >= p.N) return;
  float ag[M_MAX] = {}, au[M_MAX] = {};
  const size_t woff_g = size_t(j) * p.K, woff_u = size_t(j + p.N) * p.K;
  const size_t soff_g = size_t(j) * p.nblk, soff_u = size_t(j + p.N) * p.nblk;
  const uint kl = lane * 8u, blk_loc = lane >> 2u;
  for (uint kbase = 0u; kbase < p.K; kbase += 256u) {
    if (kl < p.K - kbase) {
      const uint blk = (kbase >> 5u) + blk_loc;
      const uint2 qwg = *(const device uint2*)(wq + woff_g + kbase + kl), qwu = *(const device uint2*)(wq + woff_u + kbase + kl);
      const float scg = float(ws[soff_g + blk]), scu = float(ws[soff_u + blk]);
      const float4 wga = float4(as_type<char4>(qwg.x)) * scg, wgb = float4(as_type<char4>(qwg.y)) * scg;
      const float4 wua = float4(as_type<char4>(qwu.x)) * scu, wub = float4(as_type<char4>(qwu.y)) * scu;
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < M_MAX; ++m) if (m < p.M) {
        const device float* xp = x + size_t(m) * p.K + kbase + kl;
        const float4 x0 = *(const device float4*)(xp + 0u), x1 = *(const device float4*)(xp + 4u);
        ag[m] += dot(x0, wga) + dot(x1, wgb);
        au[m] += dot(x0, wua) + dot(x1, wub);
      }
    }
  }
  _Pragma("clang loop unroll(full)")
  for (uint m = 0u; m < M_MAX; ++m) {
    if (m >= p.M) break;
    const float g = simd_sum(ag[m]), u = simd_sum(au[m]);
    if (lane == 0u) out[size_t(m) * p.N + j] = (g / (1.0f + metal::precise::exp(-g))) * u;
  }
}

[[kernel]] void dflare_q8_gate_up_swiglu8(const device float* x [[buffer(0)]], const device uchar* wq [[buffer(1)]],
                                          const device half* ws [[buffer(2)]], device float* out [[buffer(3)]],
                                          constant DflQ8MvP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (p.M <= 4u) dfl_q8_gu_impl<4u>(x, wq, ws, out, p, tg, sg, lane);
  else dfl_q8_gu_impl<8u>(x, wq, ws, out, p, tg, sg, lane);
}

// ---- 2-Bit Proxy Screening + Exact Top-128 Rescoring ----
struct DflP2BaseP { uint M, V, H, nblk; };

template <uint M_MAX>
NI_INLINE void dfl_p2_base_impl(const device float* x, const device uint2* p2_q, const device half* p2_s,
                                const device float* fold_b, device half* out, constant DflP2BaseP& p,
                                threadgroup float* xs, threadgroup float* xsum, uint tg, uint sg, uint lane) {
  const uint v = (tg * 4u + sg) * 4u + (lane >> 3u), blk_loc = lane & 7u;
  const bool valid = v < p.V;
  float acc[M_MAX] = {};
  for (uint blk0 = 0u; blk0 < p.nblk; blk0 += 8u) {
    _Pragma("clang loop unroll(full)")
    for (uint m = 0u; m < M_MAX; ++m) if (m < p.M) {
      const device float* xm = x + size_t(m) * p.H + blk0 * 32u;
      for (uint pass = 0u; pass < 2u; ++pass) {
        const uint b = sg + pass * 4u;
        const float xv = (blk0 + b < p.nblk) ? xm[b * 32u + lane] : 0.0f;
        xs[m * 256u + b * 32u + lane] = xv;
        const float s = simd_sum(xv);
        if (lane == 0u) xsum[m * 8u + b] = s;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint blk = blk0 + blk_loc;
    if (valid && blk < p.nblk) {
      const size_t idx = size_t(v) * p.nblk + blk;
      const float sc = float(p2_s[idx]);
      float4 c[8];
      dfl_p2_unpack(p2_q[idx], c);
      _Pragma("clang loop unroll(full)")
      for (uint m = 0u; m < M_MAX; ++m) if (m < p.M)
        acc[m] += dfl_p2_dot32(c, (const threadgroup float4*)(xs + m * 256u + blk_loc * 32u), xsum[m * 8u + blk_loc], sc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  _Pragma("clang loop unroll(full)")
  for (uint m = 0u; m < M_MAX; ++m) {
    if (m >= p.M) break;
    float vdot = acc[m];
    vdot += simd_shuffle_xor(vdot, 1u); vdot += simd_shuffle_xor(vdot, 2u); vdot += simd_shuffle_xor(vdot, 4u);
    if (blk_loc == 0u && valid) out[size_t(m) * p.V + v] = half(vdot + fold_b[v]);
  }
}

[[kernel]] void dflare_p2_unembed_base(const device float* x [[buffer(0)]], const device uint2* p2_q [[buffer(1)]],
                                       const device half* p2_s [[buffer(2)]], const device float* fold_b [[buffer(3)]],
                                       device half* out [[buffer(4)]], constant DflP2BaseP& p [[buffer(5)]],
                                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[7 * 256], xsum[7 * 8];
  if (p.M <= 4u) dfl_p2_base_impl<4u>(x, p2_q, p2_s, fold_b, out, p, xs, xsum, tg, sg, lane);
  else dfl_p2_base_impl<7u>(x, p2_q, p2_s, fold_b, out, p, xs, xsum, tg, sg, lane);
}

struct DflP2ScreenP { uint V, R, has_sup, sup_words; };

[[kernel]] void dflare_p2_fold_screen(const device float* x [[buffer(0)]], const device uint2* fold_p2_q [[buffer(1)]],
                                      const device half* fold_p2_s [[buffer(2)]], const device half* base_proxy [[buffer(3)]],
                                      const device uint* sup_mask [[buffer(4)]], device float* cand_v [[buffer(5)]],
                                      device uint* cand_i [[buffer(6)]], constant DflP2ScreenP& p [[buffer(7)]],
                                      uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sh_x[256], sh_sum[8], sh_v[64];
  threadgroup uint sh_i[64];
  const float xv = x[sg * 32u + lane];
  sh_x[sg * 32u + lane] = xv;
  const float s = simd_sum(xv);
  if (lane == 0u) sh_sum[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float b0v = -INFINITY, b1v = -INFINITY;
  uint b0i = 0xffffffffu, b1i = 0xffffffffu;
  const uint ntg_row = (p.V + 31u) / 32u, blk = lane & 7u;
  const threadgroup float4* xmp = (const threadgroup float4*)(sh_x + blk * 32u);
  const float x0s = sh_sum[blk];
  for (uint g = tg; g < ntg_row; g += 64u) {
    const uint v = (g * 8u + sg) * 4u + (lane >> 3u);
    const bool valid = v < p.V;
    float acc = 0.0f;
    if (valid) {
      float4 c[8]; const size_t idx = size_t(v) * 8u + blk;
      dfl_p2_unpack(fold_p2_q[idx], c); acc = dfl_p2_dot32(c, xmp, x0s, float(fold_p2_s[idx]));
    }
    acc += simd_shuffle_xor(acc, 1u); acc += simd_shuffle_xor(acc, 2u); acc += simd_shuffle_xor(acc, 4u);
    if (blk == 0u && valid) {
      const float val = dfl_tok_allowed(sup_mask, v, p.has_sup, p.sup_words) ? (float(base_proxy[v]) + acc) : -INFINITY;
      dfl_top2(val, v, b0v, b0i, b1v, b1i);
    }
  }
  if (blk == 0u) {
    const uint slot = sg * 4u + (lane >> 3u);
    sh_v[slot * 2u + 0u] = b0v; sh_i[slot * 2u + 0u] = b0i;
    sh_v[slot * 2u + 1u] = b1v; sh_i[slot * 2u + 1u] = b1i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float t0v = -INFINITY, t1v = -INFINITY;
    uint t0i = 0xffffffffu, t1i = 0xffffffffu;
    for (uint j = 0u; j < 64u; ++j) if (sh_i[j] != 0xffffffffu) dfl_top2(sh_v[j], sh_i[j], t0v, t0i, t1v, t1i);
    cand_v[tg * 2u + 0u] = t0v; cand_i[tg * 2u + 0u] = t0i;
    cand_v[tg * 2u + 1u] = t1v; cand_i[tg * 2u + 1u] = t1i;
  }
}

struct DflRescoreP { uint V, H, R, u_f32, has_sup, sup_words; };

[[kernel]] void dflare_rescore_exact(const device float* x_u [[buffer(0)]], const device float* x_f [[buffer(1)]],
                                     const device void* unembed [[buffer(2)]], const device uchar* fold_q [[buffer(3)]],
                                     const device half* fold_s [[buffer(4)]], const device float* fold_b [[buffer(5)]],
                                     const device uint* sup_mask [[buffer(6)]], const device uint* cand_i [[buffer(7)]],
                                     device float* cand_v [[buffer(8)]], constant DflRescoreP& p [[buffer(9)]],
                                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint c = tg * 4u + sg, v = cand_i[c];
  if (v >= p.V || !dfl_tok_allowed(sup_mask, v, p.has_sup, p.sup_words)) { if (lane == 0u) cand_v[c] = -INFINITY; return; }
  float acc = 0.0f;
  const size_t uoff = size_t(v) * p.H;
  if (p.u_f32) {
    const device float* uw = (const device float*)unembed + uoff;
    for (uint k = lane * 4u; k < p.H; k += 128u) acc += dot(*(const device float4*)(x_u + k), *(const device float4*)(uw + k));
  } else {
    const device ushort* uw = (const device ushort*)unembed + uoff;
    for (uint k = lane * 8u; k < p.H; k += 256u) {
      const uint4 qu = *(const device uint4*)(uw + k);
      acc += dot(*(const device float4*)(x_u + k + 0u), dfl_bf16_4(qu.xy)) + dot(*(const device float4*)(x_u + k + 4u), dfl_bf16_4(qu.zw));
    }
  }
  const uint f_nblk = p.R >> 5u;
  for (uint r = lane * 8u; r < p.R; r += 256u) {
    const float sc = float(fold_s[size_t(v) * f_nblk + (r >> 5u)]);
    const uint2 qf = *(const device uint2*)(fold_q + size_t(v) * p.R + r);
    acc += (dot(*(const device float4*)(x_f + r + 0u), float4(as_type<char4>(qf.x))) +
            dot(*(const device float4*)(x_f + r + 4u), float4(as_type<char4>(qf.y)))) * sc;
  }
  const float sum = simd_sum(acc);
  if (lane == 0u) { const float val = sum + fold_b[v]; cand_v[c] = isnan(val) ? -INFINITY : val; }
}

struct DflArgmaxP { uint tok_out_idx; };

[[kernel]] void dflare_argmax_128(const device float* cand_v [[buffer(0)]], const device uint* cand_i [[buffer(1)]],
                                  device int* tokens [[buffer(2)]], constant DflArgmaxP& p [[buffer(3)]],
                                  uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sv[4]; threadgroup uint si[4];
  const float v = cand_v[tid]; const uint idx = cand_i[tid], sg = tid >> 5u, lane = tid & 31u;
  const float mv = simd_max(v); const uint mi = simd_min((v == mv && mv > -INFINITY) ? idx : 0xffffffffu);
  if (lane == 0u) { sv[sg] = mv; si[sg] = mi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    const float m2 = lane < 4u ? sv[lane] : -INFINITY, gv = simd_max(m2);
    const uint i2 = lane < 4u ? si[lane] : 0xffffffffu, gi = simd_min((m2 == gv && gv > -INFINITY) ? i2 : 0xffffffffu);
    if (lane == 0u) tokens[p.tok_out_idx] = (gi == 0xffffffffu) ? 0 : int(gi);
  }
}

// ---- On-GPU Zero-Logits-D2H Speculative Acceptance ----
struct DflAccP { uint pos, K; };

[[kernel]] void dflare_accept_samples(device int* tokens [[buffer(0)]], const device int* samp [[buffer(1)]],
                                      device int* accept_out [[buffer(2)]], constant DflAccP& p [[buffer(3)]],
                                      uint lane [[thread_index_in_simdgroup]]) {
  const bool stop = lane >= p.K || tokens[p.pos + 1u + lane] != samp[lane];
  const uint a = simd_min(stop ? lane : 32u);
  if (lane == 0u) {
    const int bonus = samp[a];
    tokens[p.pos + 1u + a] = bonus;
    accept_out[0] = int(a); accept_out[1] = bonus;
  }
}
