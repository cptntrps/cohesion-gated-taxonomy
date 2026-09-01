#!/usr/bin/env python3
"""H3b: the constructive follow-up to h3_anomaly_value.py.

h3 showed no single geometric score separates NOVEL (concept has no home leaf)
from MISASSIGNED (concept lives in another leaf) — yet that is the anomaly
panel's real triage question. Hypothesis: they are different geometries and need
different signals:
  novelty     = distance to the NEAREST leaf centroid, any leaf (far from ALL
                known concepts = the taxonomy lacks this one)
  misassigned = margin by which SOME OTHER leaf beats the item's own leaf
                (strongly attracted elsewhere = filed in the wrong drawer)

Falsifiable goal: each signal beats the shipped hyperbolic score's P@40 on its
own target class (shipped: 5.0% novel in top-40, h3_results.json). Two queues
replace the single conflated panel if both win.
"""
import json, collections, pathlib, os
import numpy as np

HERE = pathlib.Path(__file__).parent
EMB = pathlib.Path(os.environ.get("LEDGAR_EMB", "ledgar_emb.npz"))
TOPK = 40

def main():
    from datasets import load_dataset
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    labels = np.array(ds["label"]); gold = ds.features["label"].names
    V = np.load(EMB)["V"].astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    n = min(len(labels), len(V)); labels, V = labels[:n], V[:n]
    tree = json.load(open(HERE / "tree.json")); members = json.load(open(HERE / "members.json"))
    child_ids = {l["source"] for l in tree["links"]}
    leaves = [nd for nd in tree["nodes"] if nd["depth"] > 0 and nd["id"] not in child_ids and nd["count"] > 0]

    cents, doms, own = [], [], {}
    for nd in leaves:
        idx = np.array([i for i in members[nd["id"]] if i < n], dtype=int)
        if len(idx) == 0: continue
        c = V[idx].mean(0); c /= np.linalg.norm(c) + 1e-9
        cents.append(c); doms.append(collections.Counter(labels[idx].tolist()).most_common(1)[0][0])
        for i in idx: own[i] = len(cents) - 1
    C = np.array(cents); homed = set(doms)
    items = np.array(sorted(own)); ar = np.arange(len(items))
    ownleaf = np.array([own[i] for i in items])
    cls = np.array([0 if labels[i] == doms[ownleaf[k]] else (2 if labels[i] not in homed else 1)
                    for k, i in enumerate(items)])

    S = V[items] @ C.T
    ownv = S[ar, ownleaf]
    S2 = S.copy(); S2[ar, ownleaf] = -2
    novelty = 1 - S.max(1)          # far from every known concept
    margin = S2.max(1) - ownv       # attracted to some other concept

    rng = np.random.default_rng(0)
    def auc(p, q, m=200000):
        a = p[rng.integers(0, len(p), m)]; b = q[rng.integers(0, len(q), m)]
        return float((a > b).mean() + 0.5 * (a == b).mean())

    top_n = np.argsort(-novelty)[:TOPK]; top_m = np.argsort(-margin)[:TOPK]
    out = {
        "novelty_queue": {"auc_novel": round(auc(novelty[cls == 2], novelty[cls != 2]), 3),
                          "p40_novel_pct": round(100 * (cls[top_n] == 2).mean(), 1)},
        "misassign_queue": {"auc_mis": round(auc(margin[cls == 1], margin[cls != 1]), 3),
                            "p40_mis_pct": round(100 * (cls[top_m] == 1).mean(), 1)},
        "shipped_reference_p40_novel_pct": 5.0,
    }
    print(json.dumps(out, indent=1))
    win = (out["novelty_queue"]["p40_novel_pct"] > 5.0 and out["misassign_queue"]["p40_mis_pct"] > 50.0)
    out["verdict"] = "TWO-QUEUE DESIGN WINS — replace the single hyperbolic panel" if win else "no win"
    print("VERDICT:", out["verdict"])
    json.dump(out, open(HERE / "h3b_results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
