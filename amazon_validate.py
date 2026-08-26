#!/usr/bin/env python3
"""Quick cross-domain check: emerge/map on Amazon PRODUCTS (non-legal, non-news).
Balanced subset across the top categories. Product = category hierarchy
(Toys & Games | Games & Accessories | Board Games). Tests whether the same
pipeline recovers a product taxonomy and names product clusters sensibly."""
import os, sys, json, collections, urllib.request, pathlib
import numpy as np
from sklearn.datasets import load_files  # noqa
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "amz_emb.npz"
PER = 500

def top(c):
    for s in ["|", ">", "/"]:
        if s in c: return c.split(s)[0].strip()
    return c.strip()
def mid(c):
    parts = [p.strip() for p in c.replace(">", "|").split("|")]
    return " | ".join(parts[:2]) if len(parts) > 1 else parts[0]

def embed(texts):
    if CACHE.exists(): return np.load(CACHE)["V"]
    V = np.zeros((len(texts), 768), dtype=np.float32); B = 32
    for i in range(0, len(texts), B):
        b = [t.lower()[:800] for t in texts[i:i+B]]
        req = urllib.request.Request("http://localhost:11434/api/embed",
            data=json.dumps({"model": "nomic-embed-text", "input": b}).encode(),
            headers={"Content-Type": "application/json"})
        V[i:i+len(b)] = np.array(json.load(urllib.request.urlopen(req, timeout=600))["embeddings"], dtype=np.float32)
        if (i//B) % 20 == 0: print(f"  embed {i}/{len(texts)}", flush=True)
    np.savez_compressed(CACHE, V=V); return V

def purity(a, y):
    return round(100*sum(collections.Counter(y[a==c].tolist()).most_common(1)[0][1] for c in set(a.tolist()))/len(y), 1)

def name_cluster(samples):
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key: return "?"
    body = json.dumps({"model": "deepseek-v4-flash", "temperature": 0.0, "max_tokens": 700,
        "messages": [{"role": "user", "content": "These are product titles from one cluster. Give the SHORTEST product-category name (1-3 words). Reply ONLY the name.\n\n" + "\n".join("- "+s[:80] for s in samples[:8])}]}).encode()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request("https://api.deepseek.com/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}), timeout=60))
        return ((r["choices"][0]["message"].get("content") or "?").strip().splitlines() or ["?"])[0][:40]
    except Exception: return "?"

def main():
    from datasets import load_dataset
    ds = load_dataset("ckandemir/amazon-products", split="train")
    names, cats, descs = ds["Product Name"], ds["Category"], ds["Description"]
    byc = collections.defaultdict(list)
    for i, c in enumerate(cats):
        if c and names[i]: byc[top(c)].append(i)
    tops = [t for t, v in sorted(byc.items(), key=lambda kv: -len(kv[1])) if len(v) >= PER][:6]
    idx = []
    for t in tops: idx += byc[t][:PER]
    texts = [f"{names[i]}. {descs[i] or ''}" for i in idx]
    y_top = np.array([top(cats[i]) for i in idx]); y_mid = np.array([mid(cats[i]) for i in idx])
    print(f"{len(texts)} products, top cats: {tops}", flush=True)
    V = embed(texts); V = V/(np.linalg.norm(V, axis=1, keepdims=True)+1e-9)
    print("embedded.\n", flush=True)

    kt, km_ = len(set(y_top.tolist())), len(set(y_mid.tolist()))
    bt = round(100*collections.Counter(y_top.tolist()).most_common(1)[0][1]/len(y_top), 1)
    lab_t = KMeans(kt, n_init=4, random_state=13).fit(V).labels_
    lab_m = KMeans(km_, n_init=4, random_state=13).fit(V).labels_
    print("== MAP ==")
    print(f"  top cats  k={kt:2d}: purity {purity(lab_t,y_top)}%  NMI {nmi(y_top,lab_t):.2f}  (baseline {bt}%)")
    print(f"  mid cats  k={km_:2d}: purity {purity(lab_m,y_mid)}%  NMI {nmi(y_mid,lab_m):.2f}")
    # EMERGE 2-level
    top6 = KMeans(kt, n_init=4, random_state=13).fit(V).labels_
    print(f"\n== EMERGE == top-{kt} clusters vs top category: purity {purity(top6,y_top)}%  NMI {nmi(y_top,top6):.2f}")
    print("\n== LLM names for the discovered top clusters (vs their true dominant category) ==")
    for c in range(kt):
        m = np.where(top6 == c)[0]
        dom = collections.Counter(y_top[m].tolist()).most_common(1)[0]
        nm = name_cluster([texts[i] for i in m[:8]])
        print(f"  cluster {c} (n={len(m)}): LLM='{nm}'  true~{dom[0]} {100*dom[1]/len(m):.0f}%")

if __name__ == "__main__":
    main()
