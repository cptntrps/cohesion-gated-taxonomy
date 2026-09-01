#!/usr/bin/env python3
"""W7: the END-TO-END headline (owner catch, 2026-09-01).

Every extraction number so far conditioned on the governing-law clauses being
already isolated — and isolated by the BENCHMARK'S GOLD LABELS, not by the tool.
On an open corpus the tool must first FIND the concept with its own category-blind
tree + LLM naming, and that step has its own recall, precision and error mixing.

This script measures the production-shaped pipeline:
  find:    take the tool's own tree.json nodes the LLM named "Governing Law"
           (top-most such nodes), union their members. No gold used to select.
  extract: cluster the FOUND set (intruders included), one real LLM call per
           cluster, cohesion gate at the supported 40% read budget, real per-item
           reads for gated items, on a 400-item sample.
  score:   routing recall/precision vs gold concept labels; extraction accuracy
           vs the improved oracle; false-assertion rate on intruders; and the
           honest end-to-end number = routing_recall x extraction_accuracy.

Falsifiable goals, declared before running:
  G9  (routing): recall >= 0.90 and precision >= 0.80, else the find step is the
      bottleneck and no extraction claim may quote isolated-set accuracy.
  G10 (intruders): false-assertion rate on non-GL members <= 10%, else propagation
      invents values for clauses that have none.
  G11 (end-to-end): recall x accuracy >= 0.85, else the honest headline drops to
      that number and the isolated-set 96% may not be quoted alone.
"""
import os, sys, json, pathlib, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from extraction_validate import ask_state, EMB, K
from w3_oracle_audit import ground_state_improved
from cluster import cohesion

HERE = pathlib.Path(__file__).parent
SAMPLE = 400
WORKERS = 8
READ_FRAC = 0.40
SEED = 13

def main():
    from datasets import load_dataset
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    texts = list(ds["text"]); labels = np.array(ds["label"]); gold = ds.features["label"].names
    V = np.load(EMB)["V"].astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    n = min(len(texts), len(V)); texts, labels, V = texts[:n], labels[:n], V[:n]
    gl_label = gold.index("Governing Laws")

    tree = json.load(open(HERE / "tree.json")); members = json.load(open(HERE / "members.json"))
    by_id = {nd["id"]: nd for nd in tree["nodes"]}
    def is_gl_name(nd): return "governing law" in (nd["name"] or "").lower()
    picked = [nd for nd in tree["nodes"]
              if is_gl_name(nd) and not (nd["parent"] and nd["parent"] in by_id and is_gl_name(by_id[nd["parent"]]))]
    found = sorted({i for nd in picked for i in members[nd["id"]] if i < n})
    found = np.array(found, dtype=int)
    print(f"FIND: {len(picked)} top-most 'Governing Law' nodes -> {len(found)} members")

    is_gl = labels[found] == gl_label
    gold_total = int((labels == gl_label).sum())
    recall = is_gl.sum() / gold_total
    precision = is_gl.mean()
    g9 = "PASS" if (recall >= 0.90 and precision >= 0.80) else "FAIL"
    print(f"routing: recall {recall:.3f} ({int(is_gl.sum())}/{gold_total} true GL found), "
          f"precision {precision:.3f} ({int((~is_gl).sum())} intruders) -> G9 {g9}")

    # ---- gated extraction on the found set, exactly as production would run ----
    Vf = V[found]
    km = KMeans(min(K, len(found)), n_init=4, random_state=SEED).fit(Vf)
    clusters = [np.where(km.labels_ == c)[0] for c in range(km.n_clusters)]
    def reps(m):
        c = Vf[m].mean(0); c /= np.linalg.norm(c) + 1e-9
        order = m[np.argsort(-(Vf[m] @ c))]
        return [texts[found[i]] for i in order[:6]]
    print(f"{km.n_clusters} real cluster-level LLM calls...", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        cvals = list(ex.map(lambda m: ask_state(reps(m)), clusters))
    cf = np.empty(len(found), dtype=object); coh = np.zeros(len(found))
    for c, m in enumerate(clusters):
        cf[m] = cvals[c]; coh[m] = cohesion(Vf[m])

    rng = np.random.default_rng(7)
    samp = rng.choice(len(found), min(SAMPLE, len(found)), replace=False)
    thr = np.quantile(coh, READ_FRAC)
    gated = coh[samp] <= thr
    print(f"per-item reads for {int(gated.sum())} gated sample items...", flush=True)
    reads = {}
    todo = [int(j) for j, g in zip(samp, gated) if g]
    with ThreadPoolExecutor(WORKERS) as ex:
        for j, v in zip(todo, ex.map(lambda j: ask_state([texts[found[j]]]), todo)):
            reads[j] = v
    pred = np.array([reads[int(j)] if g else cf[j] for j, g in zip(samp, gated)], dtype=object)

    # ---- scoring ----
    s_isgl = labels[found[samp]] == gl_label
    oracle = np.array([ground_state_improved(texts[found[j]]) if s_isgl[k] else None
                       for k, j in enumerate(samp)], dtype=object)
    resolv = s_isgl & (oracle != None)                                  # noqa: E711
    acc_found = (pred[resolv] == oracle[resolv]).mean()
    intr = ~s_isgl
    false_assert = (pred[intr] != "NONE").mean() if intr.any() else 0.0
    g10 = "PASS" if false_assert <= 0.10 else "FAIL"
    e2e = recall * acc_found
    g11 = "PASS" if e2e >= 0.85 else "FAIL"
    calls_sampled = km.n_clusters + int(gated.sum())
    print(f"\nextraction on FOUND set (sample n={len(samp)}, {int(resolv.sum())} resolvable GL, "
          f"{int(intr.sum())} intruders):")
    print(f"  accuracy on resolvable GL items:   {100*acc_found:.1f}%")
    print(f"  false-assertion rate on intruders: {100*false_assert:.1f}%  -> G10 {g10}")
    print(f"  LLM calls for the sampled slice:   {calls_sampled} (vs {len(samp)} read-everything)")
    print(f"\nEND-TO-END: routing recall {recall:.3f} x extraction {acc_found:.3f} "
          f"= {e2e:.3f} -> G11 {g11}")
    print("(items the find step missed get NO value: counted as missed, not as wrong.)")

    json.dump({"n_found": int(len(found)), "recall": round(float(recall), 3),
               "precision": round(float(precision), 3), "g9": g9,
               "acc_found_pct": round(100*float(acc_found), 1),
               "false_assert_pct": round(100*float(false_assert), 1), "g10": g10,
               "end_to_end": round(float(e2e), 3), "g11": g11,
               "sample_calls": calls_sampled, "read_frac": READ_FRAC},
              open(HERE / "w7_results.json", "w"), indent=1)
    print("wrote w7_results.json")

if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set")
    main()
