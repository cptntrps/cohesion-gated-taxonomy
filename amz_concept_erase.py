#!/usr/bin/env python3
"""Concept-direction removal: a FACET is a direction in embedding space. Estimate
the 'kids' direction, show which categories carry it, project it OUT, and measure
whether the kids-confounded categories (Toys, Baby) separate on the residual.
This is the mechanism for parallel facets: project ONTO a direction to organize by
that facet; project it OUT to see the orthogonal structure."""
import json, collections, urllib.request, pathlib
import numpy as np

HERE = pathlib.Path(__file__).parent
PER = 500

def top(c):
    for s in ["|", ">", "/"]:
        if s in c: return c.split(s)[0].strip()
    return c.strip()

def embed(phrases):
    req = urllib.request.Request("http://localhost:11434/api/embed",
        data=json.dumps({"model": "nomic-embed-text", "input": [p.lower() for p in phrases]}).encode(),
        headers={"Content-Type": "application/json"})
    A = np.array(json.load(urllib.request.urlopen(req, timeout=120))["embeddings"], dtype=np.float32)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)

def unit(v): return v / (np.linalg.norm(v) + 1e-9)

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
    y = np.array([top(cats[i]) for i in idx])
    V = np.load(HERE / "amz_emb.npz")["V"]; V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    # estimate the 'kids' direction from anchor phrases (concept, not category)
    pos = embed(["for kids", "for children", "for toddlers", "children's product", "age 3 and up", "kids play"]).mean(0)
    neg = embed(["for adults", "adult professional", "for grown ups", "office use", "for men and women"]).mean(0)
    u = unit(pos - neg)

    def cen(cat): m = y == cat; c = V[m].mean(0); return unit(c)
    print("kids-score (projection onto the kids direction) by category:")
    for t in tops:
        print(f"  {t:28} {float((V[y==t] @ u).mean()):+.3f}")

    # remove the kids direction from every product
    Vr = V - np.outer(V @ u, u); Vr = Vr / (np.linalg.norm(Vr, axis=1, keepdims=True) + 1e-9)
    def cen_r(cat): m = y == cat; c = Vr[m].mean(0); return unit(c)

    print("\ncategory-pair centroid cosine  (BEFORE  ->  AFTER removing kids):")
    def show(a, b):
        before = float(cen(a) @ cen(b)); after = float(cen_r(a) @ cen_r(b))
        tag = "  <- separated" if after < before - 0.02 else ("  <- pulled together" if after > before + 0.02 else "")
        print(f"  {a[:18]:18} <-> {b[:18]:18} {before:+.3f} -> {after:+.3f}{tag}")
    show("Toys & Games", "Baby Products")
    show("Toys & Games", "Home & Kitchen")
    show("Toys & Games", "Sports & Outdoors")   # control: low-kids pair
    show("Clothing, Shoes & Jewelry", "Sports & Outdoors")  # control

    # within Toys: does removing kids let toy SUB-TYPES separate better?
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi
    def mid(c):
        p = [x.strip() for x in c.replace(">", "|").split("|")]; return p[1] if len(p) > 1 else p[0]
    tmask = y == "Toys & Games"; ty = np.array([mid(cats[idx[i]]) for i in np.where(tmask)[0]])
    kk = len(set(ty.tolist()))
    b = nmi(ty, KMeans(kk, n_init=4, random_state=13).fit(V[tmask]).labels_)
    a = nmi(ty, KMeans(kk, n_init=4, random_state=13).fit(Vr[tmask]).labels_)
    print(f"\nwithin Toys, sub-type NMI (board games/puzzles/vehicles/...): {b:.3f} -> {a:.3f} after removing kids")

if __name__ == "__main__":
    main()
