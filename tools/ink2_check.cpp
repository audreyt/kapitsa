// Parity check for thinkingmachines/Inkling-2-Mini against tests/fixtures/ink2_golden.json.
#include "metal/gpu.h"
#include "model/inkling2.h"
#include "model/sampler.h"
#include "tokenizer/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

using json = nlohmann::json;

#ifndef INK2_GOLDEN_PATH
#define INK2_GOLDEN_PATH "tests/fixtures/ink2_golden.json"
#endif

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: ink2_check <model_dir> [golden_json]\n");
    return 2;
  }
  const std::string model_dir = argv[1];
  std::string golden_path;
  if (argc > 2) {
    golden_path = argv[2];
  } else {
    golden_path = std::ifstream(INK2_GOLDEN_PATH).good() ? INK2_GOLDEN_PATH : "tests/fixtures/ink2_golden.json";
  }

  std::ifstream gf(golden_path);
  if (!gf) {
    fprintf(stderr, "cannot open %s\n", golden_path.c_str());
    return 1;
  }
  const json golden = json::parse(gf);

  ni::Tokenizer tok;
  tok.load(model_dir + "/tokenizer.json");
  ni::Inkling2 model(model_dir, 512);
  ni::Gpu& gpu = ni::Gpu::get();
  const int V = model.vocab();

  ni::SamplingParams greedy;
  greedy.temperature = 0.0f;
  greedy.top_p = 1.0f;
  greedy.top_k = 1;
  greedy.seed = 0;

  int fails = 0;
  for (const auto& p : golden.at("prompts")) {
    const std::string name = p.at("name").get<std::string>();
    const std::string text = p.at("text").get<std::string>();
    const std::vector<int> want_ids = p.at("ids").get<std::vector<int>>();
    const std::vector<int> got_ids = tok.encode(text);
    if (got_ids != want_ids) {
      printf("FAIL %s: tokenizer mismatch (%zu vs %zu)\n", name.c_str(), got_ids.size(), want_ids.size());
      ++fails;
      continue;
    }

    auto seq = model.new_seq(512, 512);
    int* tok_ptr = seq->tokens();
    for (size_t i = 0; i < got_ids.size(); ++i) tok_ptr[i] = got_ids[i];

    const int n_prompt = int(got_ids.size());
    {
      auto e = gpu.begin();
      model.encode_prefill(*e, *seq, 0, n_prompt, greedy, 0, true);
      e->commit_and_wait();
    }

    const float* logits = model.logits();
    const std::vector<int> top10_ids = p.at("top10_ids").get<std::vector<int>>();
    const std::vector<float> top10_vals = p.at("top10_vals").get<std::vector<float>>();
    const std::vector<int> fp16_top10_ids = p.at("fp16_kv_top10_ids").get<std::vector<int>>();
    const std::vector<float> fp16_top10_vals = p.at("fp16_kv_top10_vals").get<std::vector<float>>();
    const std::vector<int> want_rollout = p.at("greedy_24").get<std::vector<int>>();

    int top1_id = 0;
    float top1_val = logits[0];
    for (int v = 1; v < V; ++v) {
      if (logits[v] > top1_val) {
        top1_val = logits[v];
        top1_id = v;
      }
    }
    float max_top10_diff_fp32 = 0.0f;
    for (size_t k = 0; k < top10_ids.size(); ++k) {
      max_top10_diff_fp32 = std::max(max_top10_diff_fp32, std::fabs(logits[top10_ids[k]] - top10_vals[k]));
    }
    float max_top10_diff_fp16 = 0.0f;
    for (size_t k = 0; k < fp16_top10_ids.size(); ++k) {
      max_top10_diff_fp16 = std::max(max_top10_diff_fp16, std::fabs(logits[fp16_top10_ids[k]] - fp16_top10_vals[k]));
    }

    std::vector<int> rollout;
    rollout.push_back(tok_ptr[n_prompt]);
    int pos = n_prompt;
    for (int step = 1; step < 24; ++step) {
      auto e = gpu.begin();
      model.encode_step(*e, *seq, pos, greedy, uint32_t(step), true);
      e->commit_and_wait();
      ++pos;
      rollout.push_back(tok_ptr[pos]);
    }

    int match_n = 0;
    for (size_t i = 0; i < want_rollout.size() && i < rollout.size(); ++i) {
      if (rollout[i] == want_rollout[i]) ++match_n;
      else break;
    }
    const bool ok = (top1_id == top10_ids[0]) && (max_top10_diff_fp16 < 0.01f) && (rollout == want_rollout);
    printf("%s %s: top1=%d (want %d) top1_logit=%.4f (fp16 %.4f, fp32 %.4f) fp16_top10_diff=%.5f fp32_top10_diff=%.5f rollout=%d/24\n",
           ok ? "PASS" : "FAIL", name.c_str(), top1_id, top10_ids[0], top1_val, fp16_top10_vals[0], top10_vals[0],
           max_top10_diff_fp16, max_top10_diff_fp32, match_n);
    if (!ok) {
      ++fails;
      printf("  got  rollout:");
      for (int id : rollout) printf(" %d", id);
      printf("\n  want rollout:");
      for (int id : want_rollout) printf(" %d", id);
      printf("\n");
      if (p.contains("dumps")) {
        auto dseq = model.new_seq(512, 512);
        for (size_t i = 0; i < got_ids.size(); ++i) dseq->tokens()[i] = got_ids[i];
        const int H = model.config().hidden;
        ni::Buffer diag = gpu.alloc(8 * size_t(H) * 4, "ink2_diag");
        {
          auto e = gpu.begin();
          model.encode_prefill_diag(*e, *dseq, 0, n_prompt, diag);
          e->commit_and_wait();
        }
        const char* keys[8] = {"after_embed", "v0_out", "v5_out", "v17_out", "v18_out", "v30_out", "v44_out", "final_norm"};
        for (int sidx = 0; sidx < 8; ++sidx) {
          const float* v = diag.as<float>() + size_t(sidx) * size_t(H);
          double sum = 0.0, sq = 0.0;
          for (int c = 0; c < H; ++c) {
            sum += v[c];
            sq += double(v[c]) * double(v[c]);
          }
          const auto& w = p["dumps"].at(keys[sidx]);
          printf("  diag %-12s got(sum=%.4f l2=%.4f [%.5f,%.5f,%.5f,%.5f]) want(sum=%.4f l2=%.4f [%.5f,%.5f,%.5f,%.5f])\n",
                 keys[sidx], sum, std::sqrt(sq), v[0], v[1], v[2], v[3],
                 w["sum"].get<double>(), w["l2"].get<double>(),
                 w["first8"][0].get<double>(), w["first8"][1].get<double>(),
                 w["first8"][2].get<double>(), w["first8"][3].get<double>());
        }
        gpu.release(diag);
      }
    }
  }
  if (model.use_tensor()) {
    printf("RUN T96_tensor_ab:\n");
    fails += model.debug_tensor_unit_check();
  }
  if (model.has_mtp()) {
    const auto& p0 = golden.at("prompts").at(0);
    const std::vector<int> pids0 = p0.at("ids").get<std::vector<int>>();
    const std::vector<int> g24 = p0.at("greedy_24").get<std::vector<int>>();
    auto run_mtp_rollout = [&](const char* label, const std::vector<int>& prompt, const std::vector<int>& want_toks, int K,
                               ni::SamplingParams sp, std::vector<int>* out_toks = nullptr,
                               bool point_mass = false) -> bool {
      auto seq = model.new_seq(512, 512);
      const int n = int(prompt.size());
      for (int i = 0; i < n; ++i) seq->tokens()[i] = prompt[i];
      auto e = gpu.begin_grouped();
      model.encode_prefill(*e, *seq, 0, n, sp, 0, true);
      e->commit_and_wait();
      std::vector<int> gen;
      gen.push_back(seq->tokens()[n]);
      int P = n;
      int cyc = 0;
      while (int(gen.size()) < int(want_toks.size()) && P + K + 2 < seq->cap) {
        auto ec = gpu.begin_grouped();
        if (point_mass) {
          for (int i = 0; i < K; ++i) seq->tokens()[P + 1 + i] = prompt[size_t(i % n)];
        } else {
          model.encode_mtp_draft(*ec, *seq, P, K, sp, uint32_t(P));
        }
        model.encode_mtp_verify(*ec, *seq, P, K + 1, sp, uint32_t(P), point_mass);
        ec->commit_and_wait();
        const int a = model.accept_mtp(*seq, P, K, sp, uint32_t(P), point_mass);
        const int m = a + 1;
        auto em = gpu.begin_grouped();
        model.encode_mtp_commit(*em, *seq, P, m);
        em->commit_and_wait();
        for (int i = 1; i <= m && int(gen.size()) < int(want_toks.size()); ++i) gen.push_back(seq->tokens()[P + i]);
        P += m;
        ++cyc;
      }
      int match = 0;
      const int total = int(want_toks.size());
      for (int i = 0; i < total && i < int(gen.size()); ++i) {
        if (gen[size_t(i)] == want_toks[size_t(i)]) ++match;
      }
      bool ok = (match == total);
      if (sp.temperature > 0.0f) {
        ok = (int(gen.size()) == total);
        for (int t : gen) {
          if (t < 0 || t >= model.vocab()) ok = false;
        }
        printf("%s %s: K=%d valid=%zu/%d cycles=%d (%.2f tok/cyc)\n", ok ? "PASS" : "FAIL", label, K, gen.size(),
               total, cyc, cyc > 0 ? double(gen.size() - 1) / cyc : 0.0);
      } else {
        printf("%s %s: K=%d match=%d/%d cycles=%d (%.2f tok/cyc)\n", ok ? "PASS" : "FAIL", label, K, match, total, cyc,
               cyc > 0 ? double(gen.size() - 1) / cyc : 0.0);
      }
      if (out_toks) *out_toks = gen;
      return ok;
    };

    for (int K : {3, 5, 7}) {
      char lbl[32];
      snprintf(lbl, sizeof lbl, "mtp_greedy_K%d", K);
      if (!run_mtp_rollout(lbl, pids0, g24, K, greedy)) ++fails;
    }

    // Longer prefix (96 tokens -> triggers KV > 64 split-K attention) plain vs MTP greedy K=5
    {
      std::vector<int> p96;
      p96.reserve(96);
      while (int(p96.size()) < 96) {
        for (int id : pids0) {
          if (int(p96.size()) < 96) p96.push_back(id);
        }
      }
      auto ref_seq = model.new_seq(512, 512);
      for (int i = 0; i < 96; ++i) ref_seq->tokens()[i] = p96[size_t(i)];
      auto ep = gpu.begin_grouped();
      model.encode_prefill(*ep, *ref_seq, 0, 96, greedy, 0, true);
      ep->commit_and_wait();
      std::vector<int> ref24 = {ref_seq->tokens()[96]};
      for (int step = 1; step < 24; ++step) {
        auto es = gpu.begin_grouped();
        model.encode_step(*es, *ref_seq, 96 + step - 1, greedy, uint32_t(step), true);
        es->commit_and_wait();
        ref24.push_back(ref_seq->tokens()[96 + step]);
      }
      if (!run_mtp_rollout("mtp_pos96_splitk_K5", p96, ref24, 5, greedy)) ++fails;
    }

    // Sampled lossless rejection sampling determinism + valid outputs
    {
      ni::SamplingParams sp;
      sp.temperature = 0.7f;
      sp.top_k = 40;
      sp.top_p = 0.95f;
      sp.seed = 12345u;
      std::vector<int> s1, s2;
      const std::vector<int> dummy(24, 0);
      bool ok1 = run_mtp_rollout("mtp_sampled_T0.7_a", pids0, dummy, 5, sp, &s1);
      bool ok2 = run_mtp_rollout("mtp_sampled_T0.7_b", pids0, dummy, 5, sp, &s2);
      const bool det = ok1 && ok2 && (s1 == s2);
      printf("%s mtp_sampled_deterministic: %zu tokens match\n", det ? "PASS" : "FAIL", s1.size());
      if (!det) ++fails;
    }
    {
      ni::SamplingParams sp;
      sp.temperature = 1.0f;
      sp.top_k = 0;
      sp.top_p = 1.0f;
      sp.seed = 42u;
      std::vector<int> s1, s2;
      const std::vector<int> dummy(24, 0);
      bool ok1 = run_mtp_rollout("mtp_sampled_T1_fullvocab_a", pids0, dummy, 5, sp, &s1);
      bool ok2 = run_mtp_rollout("mtp_sampled_T1_fullvocab_b", pids0, dummy, 5, sp, &s2);
      const bool det = ok1 && ok2 && (s1 == s2);
      printf("%s mtp_sampled_T1_fullvocab_det: %zu tokens match\n", det ? "PASS" : "FAIL", s1.size());
      if (!det) ++fails;

      std::vector<int> pm1, pm2;
      bool pok1 = run_mtp_rollout("mtp_pointmass_T1_fullvocab_a", pids0, dummy, 5, sp, &pm1, true);
      bool pok2 = run_mtp_rollout("mtp_pointmass_T1_fullvocab_b", pids0, dummy, 5, sp, &pm2, true);
      const bool pdet = pok1 && pok2 && (pm1 == pm2);
      printf("%s mtp_pointmass_T1_fullvocab_det: %zu tokens match\n", pdet ? "PASS" : "FAIL", pm1.size());
      if (!pdet) ++fails;
    }

    if (model.can_tree_spec()) {
      auto run_tree_rollout = [&](const char* label, const std::vector<int>& prompt, const std::vector<int>& want_toks,
                                  int R, ni::SamplingParams sp, std::vector<int>* out_toks = nullptr) -> bool {
        auto seq = model.new_seq(512, 512);
        const int n = int(prompt.size());
        for (int i = 0; i < n; ++i) seq->tokens()[i] = prompt[i];
        auto e = gpu.begin_grouped();
        model.encode_prefill(*e, *seq, 0, n, sp, 0, true);
        e->commit_and_wait();
        std::vector<int> gen;
        gen.push_back(seq->tokens()[n]);
        int P = n;
        int cyc = 0;
        int path[16] = {};
        while (int(gen.size()) < int(want_toks.size()) && P + R + 2 < seq->cap) {
          auto ec = gpu.begin_grouped();
          model.encode_mtp_draft_tree(*ec, *seq, P, R, sp);
          model.encode_mtp_verify_tree(*ec, *seq, P, R, sp, uint32_t(P));
          ec->commit_and_wait();
          const int m = model.accept_mtp_tree(*seq, P, sp, uint32_t(P), path);
          auto em = gpu.begin_grouped();
          model.encode_mtp_commit_tree(*em, *seq, P, path, m);
          em->commit_and_wait();
          for (int i = 1; i <= m && int(gen.size()) < int(want_toks.size()); ++i) gen.push_back(seq->tokens()[P + i]);
          P += m;
          ++cyc;
        }
        int match = 0;
        const int total = int(want_toks.size());
        for (int i = 0; i < total && i < int(gen.size()); ++i) {
          if (gen[size_t(i)] == want_toks[size_t(i)]) ++match;
        }
        bool ok = (match == total);
        if (sp.temperature > 0.0f) {
          ok = (int(gen.size()) == total);
          for (int t : gen) {
            if (t < 0 || t >= model.vocab()) ok = false;
          }
          printf("%s %s: R=%d valid=%zu/%d cycles=%d (%.2f tok/cyc)\n", ok ? "PASS" : "FAIL", label, R, gen.size(),
                 total, cyc, cyc > 0 ? double(gen.size() - 1) / cyc : 0.0);
        } else {
          printf("%s %s: R=%d match=%d/%d cycles=%d (%.2f tok/cyc)\n", ok ? "PASS" : "FAIL", label, R, match, total,
                 cyc, cyc > 0 ? double(gen.size() - 1) / cyc : 0.0);
        }
        if (out_toks) *out_toks = gen;
        return ok;
      };

      if (!run_tree_rollout("mtp_tree_greedy_R4", pids0, g24, 4, greedy)) ++fails;
      if (!run_tree_rollout("mtp_tree_greedy_R8", pids0, g24, 8, greedy)) ++fails;

      // 96-token prefix tree test (split-K attention + non-trivial path KV compaction + conv snapshot commit)
      {
        std::vector<int> p96;
        p96.reserve(96);
        while (int(p96.size()) < 96) {
          for (int id : pids0) {
            if (int(p96.size()) < 96) p96.push_back(id);
          }
        }
        auto ref_seq = model.new_seq(512, 512);
        for (int i = 0; i < 96; ++i) ref_seq->tokens()[i] = p96[size_t(i)];
        auto ep = gpu.begin_grouped();
        model.encode_prefill(*ep, *ref_seq, 0, 96, greedy, 0, true);
        ep->commit_and_wait();
        std::vector<int> ref24 = {ref_seq->tokens()[96]};
        for (int step = 1; step < 24; ++step) {
          auto es = gpu.begin_grouped();
          model.encode_step(*es, *ref_seq, 96 + step - 1, greedy, uint32_t(step), true);
          es->commit_and_wait();
          ref24.push_back(ref_seq->tokens()[96 + step]);
        }
        if (!run_tree_rollout("mtp_tree_pos96_R8", p96, ref24, 8, greedy)) ++fails;
      }

      {
        ni::SamplingParams sp;
        sp.temperature = 1.0f;
        sp.top_k = 0;
        sp.top_p = 1.0f;
        sp.seed = 42u;
        std::vector<int> t1, t2;
        const std::vector<int> dummy(24, 0);
        bool ok1 = run_tree_rollout("mtp_tree_T1_fullvocab_R8_a", pids0, dummy, 8, sp, &t1);
        bool ok2 = run_tree_rollout("mtp_tree_T1_fullvocab_R8_b", pids0, dummy, 8, sp, &t2);
        const bool det = ok1 && ok2 && (t1 == t2);
        printf("%s mtp_tree_T1_fullvocab_R8_det: %zu tokens match\n", det ? "PASS" : "FAIL", t1.size());
        if (!det) ++fails;
      }
      {
        ni::SamplingParams sp;
        sp.temperature = 0.7f;
        sp.top_k = 40;
        sp.top_p = 0.95f;
        sp.seed = 12345u;
        std::vector<int> t1, t2;
        const std::vector<int> dummy(24, 0);
        bool ok1 = run_tree_rollout("mtp_tree_T0.7_topk40_R8_a", pids0, dummy, 8, sp, &t1);
        bool ok2 = run_tree_rollout("mtp_tree_T0.7_topk40_R8_b", pids0, dummy, 8, sp, &t2);
        const bool det = ok1 && ok2 && (t1 == t2);
        printf("%s mtp_tree_T0.7_topk40_R8_det: %zu tokens match\n", det ? "PASS" : "FAIL", t1.size());
        if (!det) ++fails;
      }
    }
  }
  if (ni::Encoder::profiling()) ni::Encoder::profile_report(1);
  return fails ? 1 : 0;
}
