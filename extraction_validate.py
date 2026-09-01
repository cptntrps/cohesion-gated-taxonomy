#!/usr/bin/env python3
"""Validate the extraction hypothesis with REAL LLM calls (not just the ceiling).

  cluster-first: sub-cluster governing-law clauses, ask DeepSeek the state ONCE per
                 sub-cluster (K calls), propagate to members. Real accuracy vs regex GT.
  read-all      : ask DeepSeek per clause on a random sample. Accuracy + extrapolated cost.
  cohesion gate : does low-cohesion isolate the errors? (read only those, trust the rest)

Ground truth = regex state from the clause text (the state is literally written).
"""
import os, sys, json, re, pathlib, collections, urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from extraction_test import STATES, ground_state
from cluster import cohesion

EMB = pathlib.Path(os.environ.get("LEDGAR_EMB", "ledgar_emb.npz"))
API = os.environ.get("EXTRACT_API", "https://api.deepseek.com/v1/chat/completions")
MODEL = os.environ.get("EXTRACT_MODEL", "deepseek-v4-flash")
K = 40
READ_SAMPLE = 150
WORKERS = 8

def ask_state(clauses):
    """One DeepSeek call: given 1+ governing-law clauses, return the US state."""
    key = os.environ["DEEPSEEK_API_KEY"]
    body = json.dumps({"model": MODEL, "temperature": 0.0, "max_tokens": 800, "stream": False,
        "messages": [{"role": "user", "content":
            "The following are contract governing-law clauses. Which single US state's law "
            "governs? Reply with ONLY the state name (e.g. 'Delaware'), or 'NONE'.\n\n"
            + "\n---\n".join(c.strip()[:400] for c in clauses)}]}).encode()
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(API, data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}), timeout=90))
            out = (r["choices"][0]["message"].get("content") or "").strip()
            for s in STATES:
                if s.lower() in out.lower():
                    return s
            return "NONE"
        except Exception as e:
            sys.stderr.write(f"  llm retry: {e}\n")
    return "ERR"

def main():
    from datasets import load_dataset
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    texts = list(ds["text"]); labels = np.array(ds["label"]); gold = ds.features["label"].names
    V = np.load(EMB)["V"].astype(np.float32); V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    n = min(len(texts), len(V)); texts, labels, V = texts[:n], labels[:n], V[:n]

    gl = np.where(labels == gold.index("Governing Laws"))[0]
    st = [ground_state(texts[i]) for i in gl]
    keep = [j for j, s in enumerate(st) if s]
    idx = gl[keep]; y = np.array([st[j] for j in keep]); Vg = V[idx]
    N = len(y)
    print(f"{N} governing-law clauses with regex ground-truth state\n", flush=True)

    # ---- cluster-first: K sub-clusters, 1 LLM call each ----
    km = KMeans(n_clusters=K, n_init=4, random_state=13).fit(Vg)
    def reps(members):
        c = Vg[members].mean(0); c /= np.linalg.norm(c) + 1e-9
        order = members[np.argsort(-(Vg[members] @ c))]
        return [texts[idx[m]] for m in order[:6]]
    clusters = [np.where(km.labels_ == c)[0] for c in range(K)]
    with ThreadPoolExecutor(WORKERS) as ex:
        cvals = list(ex.map(lambda m: ask_state(reps(m)), clusters))
    pred = np.empty(N, dtype=object)
    coh = np.zeros(N)
    for c, m in enumerate(clusters):
        pred[m] = cvals[c]; coh[m] = cohesion(Vg[m])
    cf_acc = 100 * (pred == y).mean()
    print(f"CLUSTER-FIRST: {cf_acc:.1f}% accuracy at {K} LLM calls "
          f"({N} clauses labelled, {N/K:.0f}x fanout)", flush=True)

    # cohesion gate: high vs low cohesion accuracy
    med = np.median(coh)
    hi = coh >= med; lo = ~hi
    print(f"  high-cohesion half: {100*(pred[hi]==y[hi]).mean():.1f}%  |  "
          f"low-cohesion half: {100*(pred[lo]==y[lo]).mean():.1f}%   "
          f"(gate works if low << high)", flush=True)

    # ---- read-everything (sampled) ----
    rng = np.random.default_rng(13)
    samp = rng.choice(N, min(READ_SAMPLE, N), replace=False)
    with ThreadPoolExecutor(WORKERS) as ex:
        rpred = list(ex.map(lambda i: ask_state([texts[idx[i]]]), samp))
    ra_acc = 100 * np.mean([rpred[j] == y[samp[j]] for j in range(len(samp))])
    print(f"\nREAD-EVERYTHING (n={len(samp)} sample): {ra_acc:.1f}% accuracy; "
          f"full run would be {N} calls ({N/K:.0f}x more than cluster-first)", flush=True)

    print(f"\nVERDICT: cluster-first delivers {cf_acc/ra_acc*100:.0f}% of read-everything "
          f"accuracy at {100*K/N:.1f}% of the LLM calls.")
    json.dump({"n": int(N), "K": K, "cluster_first_acc": round(cf_acc,1),
               "read_all_acc": round(ra_acc,1), "read_sample": len(samp),
               "hi_coh_acc": round(100*(pred[hi]==y[hi]).mean(),1),
               "lo_coh_acc": round(100*(pred[lo]==y[lo]).mean(),1)},
              open(pathlib.Path(__file__).parent / "extraction_validate.json", "w"), indent=1)

if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set")
    main()
