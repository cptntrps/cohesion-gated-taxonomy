#!/usr/bin/env python3
"""Extraction on a SEMANTIC field regex can't handle: assignability of a contract.

Field `assignment` in {PROHIBITED, CONSENT, PERMITTED, OTHER}. No stereotyped
surface form -> regex has nothing clean to grab. There is no free ground-truth
oracle here (that was the point of governing-law state), so:
  reference   = per-clause LLM read (read-everything) on a sample -> the expensive truth
  cluster-first = K sub-clusters, 1 LLM label each, propagated
  naive regex = a plausible engineer's rule set
We measure AGREEMENT with the reference. The value prop: cluster-first reproduces
read-everything at ~K/N cost; regex does not.
"""
import os, sys, json, re, pathlib, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from extraction_validate import EMB, WORKERS
import urllib.request

API = "https://api.deepseek.com/v1/chat/completions"; MODEL = "deepseek-v4-flash"
CATS = ["PROHIBITED", "CONSENT", "PERMITTED", "OTHER"]
K = 20; READ_SAMPLE = 200

Q = ("Classify the assignability rule in the following contract clause(s). Reply with "
     "ONE word: PROHIBITED (assignment barred), CONSENT (allowed only with the other "
     "party's consent), PERMITTED (freely assignable), or OTHER.\n\n")

def ask(clauses):
    key = os.environ["DEEPSEEK_API_KEY"]
    body = json.dumps({"model": MODEL, "temperature": 0.0, "max_tokens": 800, "stream": False,
        "messages": [{"role": "user", "content": Q + "\n---\n".join(c.strip()[:400] for c in clauses)}]}).encode()
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(API, data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}), timeout=90))
            out = (r["choices"][0]["message"].get("content") or "").upper()
            for c in CATS:
                if c in out:
                    return c
            return "OTHER"
        except Exception as e:
            sys.stderr.write(f"  retry {e}\n")
    return "OTHER"

def naive_regex(t):
    t = t.lower()
    if re.search(r"may not (be )?assign|shall not (be )?assign|not.{0,12}assignable|no assignment|prohibit", t):
        return "PROHIBITED"
    if re.search(r"prior written consent|without.{0,25}consent|with.{0,12}consent of", t):
        return "CONSENT"
    if re.search(r"freely (assign|transfer)|may (freely )?assign|binding upon.{0,40}assigns|successors and assigns", t):
        return "PERMITTED"
    return "OTHER"

def main():
    from datasets import load_dataset
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    texts = list(ds["text"]); labels = np.array(ds["label"]); gold = ds.features["label"].names
    V = np.load(EMB)["V"].astype(np.float32); V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    n = min(len(texts), len(V)); texts, labels, V = texts[:n], labels[:n], V[:n]

    a = np.where(labels == gold.index("Assignments"))[0]
    Va = V[a]; N = len(a)
    print(f"{N} assignment clauses\n", flush=True)

    # cluster-first
    km = KMeans(K, n_init=4, random_state=13).fit(Va)
    def reps(m):
        c = Va[m].mean(0); c /= np.linalg.norm(c) + 1e-9
        return [texts[a[i]] for i in m[np.argsort(-(Va[m] @ c))][:6]]
    cl = [np.where(km.labels_ == c)[0] for c in range(K)]
    with ThreadPoolExecutor(WORKERS) as ex:
        cval = list(ex.map(lambda m: ask(reps(m)), cl))
    cf = np.empty(N, dtype=object)
    for c, m in enumerate(cl):
        cf[m] = cval[c]

    # reference: per-clause LLM read on a sample
    rng = np.random.default_rng(13); samp = rng.choice(N, min(READ_SAMPLE, N), replace=False)
    with ThreadPoolExecutor(WORKERS) as ex:
        ref = list(ex.map(lambda i: ask([texts[a[i]]]), samp))
    ref = np.array(ref, dtype=object)

    rx = np.array([naive_regex(texts[a[i]]) for i in samp], dtype=object)
    cf_s = cf[samp]
    cf_agree = 100 * (cf_s == ref).mean()
    rx_agree = 100 * (rx == ref).mean()
    print("reference distribution (per-clause LLM):", dict(collections.Counter(ref.tolist())))
    print(f"\nagreement with read-everything reference (n={len(samp)}):")
    print(f"  cluster-first ({K} calls total): {cf_agree:5.1f}%")
    print(f"  naive regex   (0 calls)        : {rx_agree:5.1f}%")
    print(f"\ncluster-first beats regex by {cf_agree-rx_agree:+.1f} points on a field regex can't pattern.")
    print("\n--- 12 clauses: regex vs cluster-first vs read-everything ---")
    for j in samp[:12]:
        k = list(samp).index(j)
        print(f" rx={rx[k]:10} cf={cf_s[k]:10} ref={ref[k]:10} | {texts[a[j]][:120].replace(chr(10),' ')}")
    json.dump({"n": int(N), "k": K, "cf_agree": round(cf_agree,1), "rx_agree": round(rx_agree,1),
               "sample": len(samp)}, open(pathlib.Path(__file__).parent / "extraction_semantic.json", "w"), indent=1)

if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set")
    main()
