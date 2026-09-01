#!/usr/bin/env python3
"""W3 from the validation brief: how much does the regex oracle's blind spot bias
the reported accuracy?

The eval set keeps only clauses the ORIGINAL regex could resolve. RESULTS.md §8
showed the regex silently drops ~16% of clauses that plainly name a state
(case/format brittleness). Those excluded items are plausibly the harder ones, so
accuracy measured on the kept set may be inflated.

Method (deterministic first, LLM second):
  1. Build an IMPROVED oracle: the same patterns run case-insensitively, plus
     title-cased matching for ALL-CAPS clauses. On the overlap with the old oracle
     it must agree ~100% (else the improvement is a different oracle, not a fix).
  2. The newly-resolved items are assigned to the seed-13 clusters by nearest
     centroid, and scored with the CACHED real cluster predictions (zero new
     cluster calls).
  3. A 100-item random sample of newly-resolved items gets real per-item reads.

Falsifiable goals, declared before running:
  G4: improved-oracle agreement with the old oracle on the overlap >= 99%.
      Below that the audit is apparatus-invalid.
  G5: if cluster-first accuracy on newly-resolved items is >= 5 points below its
      accuracy on the original set, the exclusion materially inflated the claim
      (report the corrected pooled number as the honest one).
"""
import os, sys, json, re, pathlib, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from extraction_test import STATES, STATE_RE, ground_state
from extraction_validate import ask_state, EMB, K

HERE = pathlib.Path(__file__).parent
WORKERS = 8
READ_N = 100

STATE_RE_I = re.compile(STATE_RE.pattern, re.IGNORECASE)

def ground_state_improved(text):
    m = STATE_RE_I.search(text)
    if m:
        cand = m.group(1).title()
        if cand in STATES:
            return cand
    tl = text.title()
    present = [s for s in STATES if s in text or s in tl]
    return present[0] if len(present) == 1 else None

def main():
    from datasets import load_dataset
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    texts = list(ds["text"]); labels = np.array(ds["label"]); gold = ds.features["label"].names
    V = np.load(EMB)["V"].astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    n = min(len(texts), len(V)); texts, labels, V = texts[:n], labels[:n], V[:n]
    gl = np.where(labels == gold.index("Governing Laws"))[0]

    old = np.array([ground_state(texts[i]) for i in gl], dtype=object)
    new = np.array([ground_state_improved(texts[i]) for i in gl], dtype=object)
    both = (old != None) & (new != None)                      # noqa: E711
    agree = (old[both] == new[both]).mean()
    only_new = (old == None) & (new != None)                  # noqa: E711
    print(f"governing-law clauses: {len(gl)}")
    print(f"old oracle resolves {int((old != None).sum())}, improved resolves {int((new != None).sum())}, "
          f"newly resolved {int(only_new.sum())}")
    print(f"agreement on overlap: {100*agree:.2f}%  -> G4 {'PASS' if agree >= 0.99 else 'FAIL'}")
    if agree < 0.99:
        for i in np.where(both & (old != new))[0][:10]:
            print(f"  disagree: old={old[i]} new={new[i]} :: {texts[gl[i]][:120]}")
        sys.exit("apparatus-invalid: the improved oracle disagrees with the old one")

    # cluster space from the ORIGINAL kept set (matches all prior experiments)
    keep_old = np.where(old != None)[0]                       # noqa: E711
    idx_old = gl[keep_old]; y_old = old[keep_old]; Vg = V[idx_old]
    km = KMeans(K, n_init=4, random_state=13).fit(Vg)
    cpreds = json.load(open(HERE / "w1_cluster_preds.json"))["13"]
    cf_old = np.array([cpreds[c] for c in km.labels_], dtype=object)
    acc_old = (cf_old == y_old).mean()

    # newly-resolved items -> nearest existing cluster (production propagation path)
    new_idx = gl[np.where(only_new)[0]]
    y_new = new[np.where(only_new)[0]]
    lab_new = km.predict(V[new_idx])
    cf_new = np.array([cpreds[c] for c in lab_new], dtype=object)
    acc_new = (cf_new == y_new).mean()
    drop = 100 * (acc_old - acc_new)
    g5 = "INFLATED" if drop >= 5.0 else "NOT MATERIALLY INFLATED"
    pooled = 100 * float(np.concatenate([(cf_old == y_old), (cf_new == y_new)]).mean())
    print(f"\ncluster-first accuracy: original set {100*acc_old:.1f}% | newly-resolved {100*acc_new:.1f}% "
          f"(drop {drop:.1f} pts) -> G5: exclusion {g5}")
    print(f"pooled (honest) cluster-first accuracy: {pooled:.1f}%")

    # real per-item reads on a sample of the newly-resolved items
    rng = np.random.default_rng(7)
    samp = rng.choice(len(new_idx), min(READ_N, len(new_idx)), replace=False)
    print(f"\nper-item LLM reads on {len(samp)} newly-resolved items...", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        reads = np.array(list(ex.map(lambda j: ask_state([texts[new_idx[j]]]), samp)), dtype=object)
    read_acc = (reads == y_new[samp]).mean()
    print(f"read-everything on newly-resolved sample: {100*read_acc:.1f}% "
          f"(vs 96.5% reported on the original set)")

    json.dump({"n_gl": int(len(gl)), "old_resolved": int((old != None).sum()),      # noqa: E711
               "new_resolved": int((new != None).sum()), "newly": int(only_new.sum()),  # noqa: E711
               "overlap_agreement_pct": round(100*float(agree), 2),
               "cf_acc_original_pct": round(100*float(acc_old), 1),
               "cf_acc_newly_pct": round(100*float(acc_new), 1),
               "cf_acc_pooled_pct": round(pooled, 1),
               "read_acc_newly_pct": round(100*float(read_acc), 1), "g5": g5},
              open(HERE / "w3_results.json", "w"), indent=1)
    print("wrote w3_results.json")

if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set")
    main()
