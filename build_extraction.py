#!/usr/bin/env python3
"""Produce extraction.json for the extraction view: governing-law clauses ->
sub-clusters -> one LLM-labelled state each -> per-clause field value, plus the
validated scorecard and gold-free confidence (cohesion). ~40 LLM calls."""
import os, sys, json, pathlib, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from extraction_test import ground_state
from extraction_validate import ask_state, EMB, K, WORKERS
from cluster import cohesion

HERE = pathlib.Path(__file__).parent

def main():
    from datasets import load_dataset
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    texts = list(ds["text"]); labels = np.array(ds["label"]); gold = ds.features["label"].names
    V = np.load(EMB)["V"].astype(np.float32); V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    n = min(len(texts), len(V)); texts, labels, V = texts[:n], labels[:n], V[:n]

    gl = np.where(labels == gold.index("Governing Laws"))[0]
    st = [ground_state(texts[i]) for i in gl]
    keep = [j for j, s in enumerate(st) if s]
    idx = gl[keep]; y = np.array([st[j] for j in keep]); Vg = V[idx]; N = len(y)

    km = KMeans(n_clusters=K, n_init=4, random_state=13).fit(Vg)
    clusters = [np.where(km.labels_ == c)[0] for c in range(K)]
    def reps(m):
        c = Vg[m].mean(0); c /= np.linalg.norm(c) + 1e-9
        return [texts[idx[i]] for i in m[np.argsort(-(Vg[m] @ c))][:6]]
    with ThreadPoolExecutor(WORKERS) as ex:
        vals = list(ex.map(lambda m: ask_state(reps(m)), clusters))

    cohs = [cohesion(Vg[m]) for m in clusters]
    med = float(np.median(cohs))
    rows, correct = [], 0
    for c, m in enumerate(clusters):
        gt = collections.Counter(y[m].tolist())
        acc = 100 * gt.get(vals[c], 0) / len(m)          # clauses this label gets right
        correct += gt.get(vals[c], 0)
        rows.append({"cluster": int(c), "value": vals[c], "size": int(len(m)),
                     "cohesion": round(cohs[c], 3),
                     "confidence": "high" if cohs[c] >= med else "low",
                     "accuracy": round(acc, 1),
                     "samples": [texts[idx[i]].strip()[:300] for i in m[:3]]})
    rows.sort(key=lambda r: (-r["size"]))
    cf_acc = round(100 * correct / N, 1)
    field = collections.Counter()
    for c, m in enumerate(clusters):
        field[vals[c]] += len(m)

    val = {}
    p = HERE / "extraction_validate.json"
    if p.exists():
        val = json.load(open(p))
    out = {
        "concept": "Governing Law", "field": "governing_law",
        "n_clauses": int(N), "k": K, "fanout": round(N / K, 0),
        "cluster_first_acc": cf_acc,
        "read_all_acc": val.get("read_all_acc"), "read_all_calls": int(N),
        "hi_coh_acc": val.get("hi_coh_acc"), "lo_coh_acc": val.get("lo_coh_acc"),
        "field_distribution": sorted(field.items(), key=lambda t: -t[1]),
        "rows": rows,
    }
    json.dump(out, open(HERE / "extraction.json", "w"), indent=1)
    print(f"wrote extraction.json: {N} clauses, {K} calls, cluster-first {cf_acc}%, "
          f"read-all {out['read_all_acc']}%")

if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set")
    main()
