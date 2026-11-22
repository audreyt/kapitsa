# Inkling-2-Mini on Apple Silicon (DSpark + Matrix Prefill)

Architecture notes, CLI usage, and verified M5 Max benchmarks for Thinking Machines' **Inkling-2-Mini** (`inkling2_loop`, 24 unique layers over 48 visits, 128 experts top-8 NVFP4/MXFP8 SiTU MoE, 3-layer MTP head) on Kapitsa `fork:main` (`perf/ink2-basalt-dspark` merged with upstream `main` `8a4ab9a`).

See [docs/usage.md](usage.md) for general CLI options and [docs/benchmarks-m5-max.md](benchmarks-m5-max.md) for historical M5 Max baselines.

## Architecture Summary

The integrated stack combines five coordinated layers across Metal 3 exact decode, Metal 4 tensor-unit prefill, quantized dense GEMVs, on-GPU speculative tree verification, and grammar/token masking:

1. **Exact Metal 3 Decode & Small-Chunk Core ([PR #1393](https://github.com/nace-ai/kapitsa/pull/1393), `perf/ink2-exact-decode` @ `ec6bb54`)**
   - Vectorized FP32 decode/verify GEMVs (`ink2_bf16_mv4`, `ink2_bf16_mv8`, fused `ink2_bf16_mv4_multi6` / `ink2_bf16_mv8_multi6`), vectorized `ink2_moe_{nvfp4,mxfp8}_{w13_situ,w2_acc}`, `ink2_attn_verify_gqa`, `ink2_rmsnorm3`, and `active_ext`-bounded relative bias.
   - Runs as the exact baseline under `--profile precise` (default) and for all `T <= 32` verify/decode paths where FP32 precision is required.

2. **Metal 4 Tensor Prefill & FlashAttention (Upstream `#1387` in `8a4ab9a`)**
   - Activated automatically under `--profile fast` (`NI_INK2_TENSOR`, macOS 26+ on Apple10+ matrix units):
     - Dense BF16 prefill GEMMs (`M > 32`) via `ink2_t_bf16_m64` and `ink2_msg_gate_{prep,mix}`.
     - Dynamic `bm` (`16`/`32`/`64`) bucketed 128-expert NVFP4/MXFP8 MoE (`ink2_moe_bucket`, `ink2_t_*_moe_w13`, `ink2_moe_situ_half`, `ink2_t_*_moe_w2`, `ink2_moe_combine`) for `T >= 32` (`2 <= T <= 31` uses `ink2_moe_bucket_small` + `_g` + `ink2_moe_reduce8_m3`).
     - Prefill FlashAttention (`qattn_ink2_s6k32p`, `NI_INK2_ATTN_T`) for `T >= 16` FP16 KV.

3. **Fast-Profile Q8 Dense & Head (`ink2_q8_dense_fast`, `perf/ink2-q8-mv4-multi5` @ `97157ff`)**
   - Under `--profile fast` (`NI_INK2_Q8_DENSE`, `NI_INK2_Q8_HEAD`), packs decode-heavy BF16 dense matrices (`qkvrsg`, `wo_ud`) and the `unembed` LM head to `Q8_0` (`ink2_q8_mv4`, `ink2_q8_mv8`, `ink2_q8_mv_multi6`), cutting decode memory bandwidth while keeping greedy rollouts matching the golden check.

4. **DSpark 3-Layer MTP & Zero-Logits-D2H (`perf/ink2-dflare-q2-zero-d2h` @ `f8417a2`)**
   - Loads Inkling-2-Mini's 3-layer speculative MTP head (`block=8`, `markov_rank=256`, taps `[5, 23, 41, 48]`, default `w=q4` via `NI_INK2_MTP_W`, 2-pass Top-K screening `screen=p2`).
   - Performs candidate scoring, sampling, and verification on GPU (`ink2_mtp_accept_*`, `ink2_mtp_sample_emit_*`) without copying full-vocabulary logits to CPU when `can_gpu_accept(sp)` holds (`temperature == 0` or `top_k > 0`).

5. **W=4 On-GPU Markov Beam Tree & GPU-Timed Policy (`perf/ink2-dspark-w4-tree-policy` @ `fbef399`)**
   - Expands a W=4 on-GPU Markov beam tree (`ink2_mtp_tree_select`) verified in a single forward pass with tree-aware short convolutions (`ink2_sconv_{static,dynamic}_tree`), KV append (`ink2_qkvr_prep_tree`), logical-depth parity GQA tree verify (`ink2_attn_verify_tree_gqa`), GPU tree acceptance (`ink2_mtp_accept_tree_*`), and `ink2_kv_compact` commit.
   - An online UCB/EMA cost-yield controller (`tree_yield[chan][R] / tree_ms[R]`) clocked with GPU execution timestamps (`e2->gpu_ms()`) selects the optimal budget `R in {2, 3, 4, 6, 8}` per channel (`think`, `text`, `tool`) alongside suffix copy drafting (`NI_INK2_SUFFIX_DRAFT`).
   - Supports structured JSON schemas (Upstream `#1388`, via constrained per-step masks) and `--suppress-tokens` (`NI_SUPPRESS_TOKENS`) across both base and MTP/tree decode.

## Branch & PR Map

| Component | Branch / PR | Head Commit | Target |
| :--- | :--- | :--- | :--- |
| Exact decode/verify Metal 3 | `perf/ink2-exact-decode` ([upstream PR #1393](https://github.com/nace-ai/kapitsa/pull/1393)) | `ec6bb54` | `nace-ai/kapitsa:main` |
| Q8_0 dense & unembed MV | `fork/perf/ink2-q8-mv4-multi5` (follows `#1381`) | `97157ff` | Stacked PR |
| DFlare 3L Q4 MTP + zero D2H | `fork/perf/ink2-dflare-q2-zero-d2h` (follows `#1383`) | `f8417a2` | Stacked PR |
| DSpark W=4 beam tree + policy | `fork/perf/ink2-dspark-w4-tree-policy` (follows `#1383`) | `fbef399` | Stacked PR |
| Integrated deployment branch | `perf/ink2-basalt-dspark` (`fork:main`) | Current branch | Local deploy (`--profile fast`) |

## Usage

Serve Inkling-2-Mini with the full fast-profile acceleration stack (Metal 4 tensor prefill, Q8 dense/head decode, DSpark W=4 tree) and optional token suppression:

```sh
kapitsa serve --model /Volumes/w/models/Inkling-2-Mini --profile fast \
  --suppress-tokens /Volumes/w/models/Inkling-2-Mini/mask_zh-TW_t1.json
```

Useful environment overrides:

- `KAPITSA_PROFILE=fast|precise`: enables/disables `ink2_tensor_pf` and `ink2_q8_dense_fast` (CLI `--profile` takes precedence).
- `NI_INK2_TENSOR=0|1`, `NI_INK2_ATTN_T=0|1`: toggle Metal 4 tensor GEMMs/MoE and `qattn_ink2` FlashAttention.
- `NI_INK2_Q8_DENSE=0|1`, `NI_INK2_Q8_HEAD=0|1`: toggle `Q8_0` dense GEMV and LM-head packing.
- `NI_INK2_MTP_K=0..7`: pin linear speculative depth or disable MTP (`0`), bypassing auto DSpark tree selection.
- `NI_INK2_TREE=0|1`, `NI_INK2_TREE_R=2..8`: force beam tree speculation off/on or pin tree verify budget `R`.
- `NI_INK2_MTP_W=q4|q8|bf16`, `NI_INK2_SUFFIX_DRAFT=0|1`: control MTP weight format and suffix copy drafting.

## Verified M5 Max Benchmarks (`--profile fast`)

Measured on Apple M5 Max (`Mac17,6`, 40 GPU cores, 128 GB unified memory, macOS `27.3` `26C5047e`, AC power, `cached_tokens = 0`) against `/Volumes/w/models/Inkling-2-Mini` (`18.89 GB` text weights + `0.64 GB` DSpark MTP head).

### Kernel & Correctness Checks

- **CTest + Unit Suites**: `ctest` `20/20` passed, `api_test` `28657` passed / `0` failed (`1` chat-template round-trip skipped when invoked without a model directory), `device_profile_test` `22` defaults passed, `think_control_test` `49/49` passed.
- **`ink2_check` (`KAPITSA_PROFILE=precise`, `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`)**: `3/3` golden prompts PASS (`fp16_top10_diff <= 0.00134`, `24/24` rollout each), `22/22` MTP & beam tree tests PASS, `0` Metal validation errors.
- **`ink2_check` (`KAPITSA_PROFILE=fast`, `MTL_DEBUG_LAYER=1`)**: `3/3` golden prompts PASS (`top1` exact match, `24/24` rollout each), `5/5` `T96_tensor_ab` kernel unit checks PASS (`bf16_mm_t` rel `0.000001`, `moe_nvfp4_t` rel `0.000294`, `moe_mxfp8_t` rel `0.000311`, `attn_prefill_t` rel `0.000352`, `msg_gate_t` rel `0.000214`), `22/22` MTP & beam tree tests PASS, `0` Metal API validation errors.
- **`ink2_check` (`KAPITSA_PROFILE=fast`, `NI_INK2_Q8_DENSE=0`)**: strict unquantized tolerance (`min(fp16, fp32) < 0.01`) PASS across `3/3` prompts (`0.00013`, `0.00110`, `0.00160`), `5/5` `T96_tensor_ab` PASS, `22/22` MTP & tree tests PASS.
- **Live Greedy Exactness (`NI_INK2_MTP_K=0` plain vs default W=4 DSpark tree, `--profile fast`)**: **5/5 exact SHA-256 output match** across factual (`18d898...`), Python coding (`e7dc8f...`), math (`9e54fe...`), `zh-TW` (`fe60eb...`), and JSON (`3229e4...`) prompts (`76–128` tokens generated).

### Prefill Throughput (`--profile fast`, `max_tokens = 1`)

| Prompt Length | Engine Prefill Time | Wall Time (incl. HTTP) | Wall Throughput |
| :--- | :--- | :--- | :--- |
| `844` tokens | `544 ms` | `0.5457 s` | **`1546.8 tok/s`** |
| `3047` tokens | `1865 ms` | `1.8773 s` | **`1623.0 tok/s`** |
| `6023` tokens | `3748 ms` | `3.7671 s` | **`1598.9 tok/s`** |

### Decode Throughput (`--profile fast`, `NI_INK2_Q8_DENSE=1` + DSpark W=4)

| Workload | Prompt / Gen Tokens | Sampling | Engine Decode | Acceptance | Cycles |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Short fact (`K=0` baseline) | `43` / `128` | `T=0`, `top_k=1` | `70.9 tok/s` | `1.00 tok/cycle` | `127` |
| Python quicksort (`K=0` baseline) | `49` / `128` | `T=0`, `top_k=1` | `68.4 tok/s` | `1.00 tok/cycle` | `127` |
| Short fact (W=4 DSpark) | `43` / `128` | `T=0`, `top_k=1` | **`82.8 tok/s`** | `1.98 tok/cycle` | `64` |
| Python quicksort (W=4 DSpark) | `49` / `128` | `T=0`, `top_k=1` | **`117.1 tok/s`** | `3.10 tok/cycle` | `41` |
| Math two-step (W=4 DSpark) | `60` / `123` | `T=0`, `top_k=1` | **`111.6 tok/s`** | `3.05 tok/cycle` | `40` |
| `zh-TW` explain (W=4 DSpark) | `54` / `128` | `T=0`, `top_k=1` | **`89.1 tok/s`** | `2.59 tok/cycle` | `49` |
| Primes CSV (W=4 DSpark) | `46` / `76` | `T=0`, `top_k=1` | **`78.2 tok/s`** | `2.14 tok/cycle` | `35` |
| Short coding C++ header | `73` / `256` | `T=0` | **`75.6 tok/s`** | `1.78 tok/cycle` | `143` |
| `5270`-token architectural rewrite | `5270` / `256` | `T=0` | **`66.3 tok/s`** | `1.73 tok/cycle` | `147` |
| Sampled (`T=1.0`, `top_k=32`, GPU zero-D2H) | `63` / `256` | `T=1`, `top_k=32` | **`69.5 tok/s`** | `1.31 tok/cycle` | `195` |
| Sampled (`T=1.0`, `top_k=0`, full vocab) | `63` / `256` | `T=1`, `top_k=0` | `45.7 tok/s` | `1.53 tok/cycle` | `167` |

- **Structured Output (`response_format` `json_schema`)**: HTTP `200`, valid JSON matching schema (`28` tokens in `384 ms`, `70.4 tok/s`, `30` masks @ `29.5 us` each).
- **Traditional Chinese Token Suppression (`--suppress-tokens mask_zh-TW_t1.json`)**: `3382` token IDs suppressed on load, `0` simplified-only characters emitted across both reasoning and final content, running DSpark at `77.6 tok/s` (`192` tokens, `1.80 tok/cycle`) and `75.5 tok/s` (`382` tokens, `1.38 tok/cycle`).
