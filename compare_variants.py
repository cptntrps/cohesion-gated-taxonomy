#!/usr/bin/env python3
"""Judge variants by the RIGHT metric for each mode (owner reframe 2026-08-26):

  MAP    (assign to a target taxonomy) -> maximize gold PURITY. Coarse = failure.
  EMERGE (discover a taxonomy)         -> minimize FALSE MERGES. Coarse != wrong;
          only sticking distinct concepts together is wrong.

Metrics per tree:
  purity%       weighted gold purity of leaves            (MAP yardstick)
  fmerge%       % of leaf clauses in a leaf that still has a clean 2-way seam
                (cosine silhouette >= FM) -> a false merge (EMERGE yardstick)
  mm_mean       mean leaf multimodality (lower = more coherent leaves)
Gold is used only to score purity; fmerge/mm are gold-free.
"""
import os, json, pathlib
import numpy as np

EMB = pathlib.Path(os.environ.get("LEDGAR_EMB", "ledgar_emb.npz"))
FM = 0.10   # cosine-silhouette above this = a real seam = false merge

def load():
    from datasets import load_dataset
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    texts = list(ds["text"]); labels = np.array(ds["label"])
    gold = ds.features["label"].names
    V = np.load(EMB)["V"].astype(np.float32); V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    n = min(len(texts), len(V))
    return texts[:n], labels[:n], gold, V[:n]

def summarize(nodes, links, members, V):
    from cluster import multimodality
    child = {l["source"] for l in links}
    leaves = [n for n in nodes if n["depth"] > 0 and n["id"] not in child and n["count"] > 0]
    w = np.array([l["count"] for l in leaves], dtype=float)
    pur = np.array([l["purity"] for l in leaves], dtype=float)
    mm = np.array([multimodality(V[np.array(members[l["id"]], dtype=int)]) for l in leaves])
    fmerge = 100 * w[mm >= FM].sum() / w.sum()
    return {"leaves": len(leaves), "nodes": len(nodes) - 1,
            "purity": round(float((pur * w).sum() / w.sum()), 1),
            "fmerge": round(float(fmerge), 1), "mm_mean": round(float(mm.mean()), 3)}

def main():
    from cluster import build_taxonomy
    texts, labels, gold, V = load()
    variants = {
        "map baseline":       dict(mode="map"),
        "map + refine":       dict(mode="map", refine=True),
        "emerge":             dict(mode="emerge"),
        "emerge + refine":    dict(mode="emerge", refine=True),
    }
    print(f"{'variant':18} {'leaves':>7} {'nodes':>6} {'purity%':>8} {'fmerge%':>8} {'mm_mean':>8}")
    print("-" * 60)
    for name, kw in variants.items():
        nodes, links, members, report, _ = build_taxonomy(
            V, texts, labels, gold, anchors=None, geometry="hyperbolic",
            do_naming=False, log=lambda m: None, **kw)
        s = summarize(nodes, links, members, V)
        print(f"{name:18} {s['leaves']:7d} {s['nodes']:6d} {s['purity']:8.1f} {s['fmerge']:8.1f} {s['mm_mean']:8.3f}")
    print(f"\nMAP goal: high purity%. EMERGE goal: low fmerge% (few leaves with a live seam).")
    print("Over-pruning that lowers purity but keeps fmerge low is CORRECT for emerge (coarse, not wrong).")

if __name__ == "__main__":
    main()
