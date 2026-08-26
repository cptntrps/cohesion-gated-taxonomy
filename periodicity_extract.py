#!/usr/bin/env python3
"""Attribute -> canonical value extraction (owner architecture 2026-08-26).

periodicity is a cross-concept ATTRIBUTE with a canonical value table:
  {WEEKLY, BI-WEEKLY, MONTHLY, QUARTERLY, SEMI-ANNUALLY, ANNUALLY}.
Cluster the clauses that carry a periodicity, label each sub-cluster with ONE
canonical value (LLM), propagate. Validate vs the regex canonical value.
Skewed field (80% annually), so the point is recovering the MINORITY values.
"""
import os, sys, json, re, pathlib, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from cluster import cohesion
import urllib.request

API = "https://api.deepseek.com/v1/chat/completions"; MODEL = "deepseek-v4-flash"
EMB = pathlib.Path("/home/gui/projects/hyperbolic-experiments/crossover/ledgar_emb.npz")
CANON = {"WEEKLY": r"\bweekly\b", "BI-WEEKLY": r"bi-?weekly|every two weeks|bimonthly",
         "MONTHLY": r"\bmonthly\b|per month|each month", "QUARTERLY": r"\bquarterly\b|each quarter|per quarter",
         "SEMI-ANNUALLY": r"semi-?annual|twice a year|every six months",
         "ANNUALLY": r"\bannual|\byearly\b|per annum|each year|per year"}
VALS = list(CANON) + ["NONE"]; K = 15

def gt(t):
    h = [k for k, p in CANON.items() if re.search(p, t, re.I)]
    return h[0] if len(h) == 1 else None

def ask(clauses):
    key = os.environ["DEEPSEEK_API_KEY"]
    body = json.dumps({"model": MODEL, "temperature": 0.0, "max_tokens": 800, "stream": False,
        "messages": [{"role": "user", "content":
            "What PERIODICITY/frequency do these clauses specify? Reply with ONE of: "
            + ", ".join(VALS) + ".\n\n" + "\n---\n".join(c.strip()[:300] for c in clauses)}]}).encode()
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(API, data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}), timeout=90))
            o = (r["choices"][0]["message"].get("content") or "").upper()
            for v in sorted(VALS, key=len, reverse=True):
                if v in o:
                    return v
            return "NONE"
        except Exception as e:
            sys.stderr.write(f" retry {e}\n")
    return "NONE"

def main():
    from datasets import load_dataset
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    texts = list(ds["text"])
    V = np.load(EMB)["V"].astype(np.float32); V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    n = min(len(texts), len(V)); texts, V = texts[:n], V[:n]
    idx = np.array([i for i in range(n) if gt(texts[i])]); y = np.array([gt(texts[i]) for i in idx])
    Va = V[idx]; N = len(y)
    base = 100 * collections.Counter(y.tolist()).most_common(1)[0][1] / N
    print(f"{N} clauses with a canonical periodicity. majority baseline (always ANNUALLY) = {base:.1f}%\n")

    km = KMeans(K, n_init=4, random_state=13).fit(Va)
    cl = [np.where(km.labels_ == c)[0] for c in range(K)]
    def reps(m):
        c = Va[m].mean(0); c /= np.linalg.norm(c) + 1e-9
        return [texts[idx[i]] for i in m[np.argsort(-(Va[m] @ c))][:6]]
    with ThreadPoolExecutor(8) as ex:
        cval = list(ex.map(lambda m: ask(reps(m)), cl))
    pred = np.empty(N, dtype=object); coh = np.zeros(N)
    for c, m in enumerate(cl):
        pred[m] = cval[c]; coh[m] = cohesion(Va[m])
    acc = 100 * (pred == y).mean()
    print(f"CLUSTER-FIRST: {acc:.1f}% at {K} LLM calls  (baseline {base:.1f}%)\n")
    print("per-value recall (does it recover the minority values?):")
    for v in CANON:
        m = y == v
        if m.sum():
            print(f"  {v:14} n={int(m.sum()):5d}  recall={100*(pred[m]==y[m]).mean():5.1f}%")
    med = np.median(coh)
    print(f"\ncohesion gate: high {100*(pred[coh>=med]==y[coh>=med]).mean():.1f}% | low {100*(pred[coh<med]==y[coh<med]).mean():.1f}%")
    print("\ncanonical value table (extracted distribution):", dict(collections.Counter(pred.tolist())))

if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set")
    main()
