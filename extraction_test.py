#!/usr/bin/env python3
"""Test the consolidation/extraction hypothesis (owner 2026-08-26):

  ALL CLAUSES -> GOVERNING LAW (concept) -> DELAWARE (value) -> contract.field = Delaware

Hypothesis: to fill the `governing_law` field, CLUSTER the governing-law clauses and
ask the value once per sub-cluster (K LLM calls), and every member inherits it — this
beats an LLM reading each clause/contract (N calls) on accuracy AND cost.

We get ground truth for free: the state is regex-extractable from the clause text.
So we can measure the CEILING of cluster-first extraction = weighted state-purity of
the sub-clusters. High purity => one label per cluster propagates correctly to all
members. Baseline = assign the single most common state to everyone (no reading).
"""
import re, pathlib, collections
import numpy as np
from sklearn.cluster import KMeans

EMB = pathlib.Path("/home/gui/projects/hyperbolic-experiments/crossover/ledgar_emb.npz")
STATES = ["Delaware","New York","California","Nevada","Texas","Illinois","Florida",
          "New Jersey","Massachusetts","Washington","Pennsylvania","Georgia","Ohio",
          "Virginia","Minnesota","Colorado","Connecticut","Maryland","Michigan",
          "Missouri","Wisconsin","Arizona","Tennessee","Indiana","Oregon","Utah",
          "North Carolina","South Carolina","Kansas","Oklahoma","Louisiana","Alabama"]
STATE_RE = re.compile(
    r"(?:laws of|State of|Commonwealth of|internal laws of)(?: the)?(?: State| Commonwealth)?(?: of)? "
    r"([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?)")

def ground_state(text):
    m = STATE_RE.search(text)
    if m and m.group(1) in STATES:
        return m.group(1)
    present = [s for s in STATES if s in text]
    return present[0] if len(present) == 1 else None

def main():
    from datasets import load_dataset
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    texts = list(ds["text"]); labels = np.array(ds["label"])
    gold = ds.features["label"].names
    gl = gold.index("Governing Laws")
    V = np.load(EMB)["V"].astype(np.float32); V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    n = min(len(texts), len(V)); texts, labels, V = texts[:n], labels[:n], V[:n]

    gl_idx = np.where(labels == gl)[0]
    st = [ground_state(texts[i]) for i in gl_idx]
    keep = [j for j, s in enumerate(st) if s]
    idx = gl_idx[keep]; y = np.array([st[j] for j in keep])
    Vg = V[idx]
    dist = collections.Counter(y.tolist())
    maj = dist.most_common(1)[0]
    baseline = 100 * maj[1] / len(y)
    print(f"governing-law clauses with a resolvable state: {len(y)} / {len(gl_idx)}")
    print(f"state distribution (top): {dist.most_common(6)}")
    print(f"BASELINE (assign most common '{maj[0]}' to all, no reading): {baseline:.1f}%\n")

    print(f"{'K sub-clusters':>14} {'LLM calls':>10} {'state-purity%':>14} {'vs baseline':>12}")
    print("-" * 54)
    for k in [6, 10, 16, 24, 40]:
        km = KMeans(n_clusters=k, n_init=4, random_state=13).fit(Vg)
        correct = 0
        for c in range(k):
            m = km.labels_ == c
            if m.sum() == 0:
                continue
            dom = collections.Counter(y[m].tolist()).most_common(1)[0][1]
            correct += dom            # each member gets the sub-cluster's dominant state
        purity = 100 * correct / len(y)
        print(f"{k:14d} {k:10d} {purity:14.1f} {purity-baseline:+11.1f}")
    print(f"\n(read-everything alternative = {len(y)} LLM reads; cluster-first = K calls.)")
    print("Hypothesis supported if sub-cluster state-purity >> baseline at K << N calls.")

if __name__ == "__main__":
    main()
