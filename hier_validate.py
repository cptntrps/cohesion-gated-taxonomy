#!/usr/bin/env python3
"""Validate taxonomy EMERGENCE + MAPPING on a NON-legal corpus with a known
hierarchy: 20 Newsgroups (6 super-groups -> 20 groups). (WOS's HF loader is a
deprecated script; 20NG is the equivalent fast hierarchical benchmark.)

MAP:    cluster to k=20 (groups) and k=6 (supers); weighted purity vs gold. Baseline
        = majority class.
EMERGE: top-level KMeans=6 -> does it recover the 6 supers? then split each ->
        do leaves recover the 20 groups? (hierarchical recovery, unsupervised).
Gold labels only score; clustering never sees them.
"""
import json, pathlib, collections, urllib.request
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "ng_emb.npz"
N_SUB = 6000

def super_of(group):
    p = group.split(".")[0]
    return "religion" if p in ("alt", "soc") else p   # alt.atheism + soc.religion -> religion

def embed(texts):
    if CACHE.exists():
        return np.load(CACHE)["V"]
    V = np.zeros((len(texts), 768), dtype=np.float32); B = 32
    for i in range(0, len(texts), B):
        batch = [t.lower()[:1500] for t in texts[i:i+B]]
        req = urllib.request.Request("http://localhost:11434/api/embed",
            data=json.dumps({"model": "nomic-embed-text", "input": batch}).encode(),
            headers={"Content-Type": "application/json"})
        emb = json.load(urllib.request.urlopen(req, timeout=600))["embeddings"]
        V[i:i+len(emb)] = np.array(emb, dtype=np.float32)
        if (i // B) % 30 == 0: print(f"  embed {i}/{len(texts)}", flush=True)
    np.savez_compressed(CACHE, V=V)
    return V

def purity(assign, y):
    tot = 0
    for c in set(assign.tolist()):
        m = assign == c
        tot += collections.Counter(y[m].tolist()).most_common(1)[0][1]
    return round(100 * tot / len(y), 1)

def main():
    d = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    texts, targets, names = d.data, d.target, d.target_names
    groups = np.array([names[t] for t in targets])
    supers = np.array([super_of(g) for g in groups])
    keep = [i for i, t in enumerate(texts) if len(t.strip()) > 40][:N_SUB]
    texts = [texts[i] for i in keep]; groups = groups[keep]; supers = supers[keep]
    print(f"{len(texts)} posts, {len(set(groups))} groups, {len(set(supers))} supers: {sorted(set(supers))}", flush=True)
    V = embed(texts); V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    print("embedded.\n", flush=True)

    ng, ns = len(set(groups)), len(set(supers))
    base_g = round(100 * collections.Counter(groups.tolist()).most_common(1)[0][1] / len(groups), 1)
    base_s = round(100 * collections.Counter(supers.tolist()).most_common(1)[0][1] / len(supers), 1)

    # MAP
    lab_g = KMeans(ng, n_init=4, random_state=13).fit(V).labels_
    lab_s = KMeans(ns, n_init=4, random_state=13).fit(V).labels_
    print("== MAP (cluster count = gold count) ==")
    print(f"  groups k={ng}: purity {purity(lab_g, groups)}%   (baseline {base_g}%)")
    print(f"  supers k={ns}: purity {purity(lab_s, supers)}%   (baseline {base_s}%)")

    # EMERGE (hierarchical recovery): top=6 -> supers, then split each -> groups
    top = KMeans(ns, n_init=4, random_state=13).fit(V).labels_
    print("\n== EMERGE (unsupervised 2-level recovery) ==")
    print(f"  top-{ns} clusters vs SUPER labels: purity {purity(top, supers)}%")
    leaf_correct, leaf_n, distinct_super = 0, 0, set()
    for c in set(top.tolist()):
        m = np.where(top == c)[0]
        distinct_super.add(collections.Counter(supers[m].tolist()).most_common(1)[0][0])
        k = min(4, len(m))
        if k < 2: continue
        sub = KMeans(k, n_init=3, random_state=13).fit(V[m]).labels_
        for s in set(sub.tolist()):
            mm = m[sub == s]
            leaf_correct += collections.Counter(groups[mm].tolist()).most_common(1)[0][1]; leaf_n += len(mm)
    print(f"  distinct supers captured by the {ns} top clusters: {len(distinct_super)}/{ns}")
    print(f"  2-level leaves vs GROUP labels: purity {round(100*leaf_correct/leaf_n,1)}%   (baseline {base_g}%)")
    print("\nemergence works if top-cluster super-purity >> baseline and most supers are distinctly captured.")

if __name__ == "__main__":
    main()
