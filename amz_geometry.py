#!/usr/bin/env python3
"""Does hyperbolic (and its cheap analog, mean-centering) restructure the anisotropy
so concept structure lives in ANGLE? Raw cosine crams all categories to ~0.94.
Compare: raw vs mean-centered vs Poincare ball -- on (a) how spread the category
angles are, (b) clustering quality, (c) whether angular concept-removal works."""
import collections, pathlib
import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from datasets import load_dataset

HERE = pathlib.Path(__file__).parent
def top(c):
    for s in ["|", ">", "/"]:
        if s in c: return c.split(s)[0].strip()
    return c.strip()
def unit(v): return v / (np.linalg.norm(v) + 1e-9)

ds = load_dataset("ckandemir/amazon-products", split="train"); cats = ds["Category"]; names = ds["Product Name"]
byc = collections.defaultdict(list)
for i, c in enumerate(cats):
    if c and names[i]: byc[top(c)].append(i)
tops = [t for t, v in sorted(byc.items(), key=lambda kv: -len(kv[1])) if len(v) >= 500][:6]
idx = []
for t in tops: idx += byc[t][:500]
y = np.array([top(cats[i]) for i in idx])
V = np.load(HERE / "amz_emb.npz")["V"]; V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

def pair_cos(A):
    cens = {t: unit(A[y == t].mean(0)) for t in tops}
    return np.mean([cens[a] @ cens[b] for a, b in combinations(tops, 2)])
def pur(a): return round(100*sum(collections.Counter(y[a==c].tolist()).most_common(1)[0][1] for c in set(a.tolist()))/len(y),1)

# raw
print(f"{'space':22}{'mean cat-pair cosine':>22}{'cluster NMI':>13}{'purity':>9}")
print("-"*66)
lab = KMeans(5, n_init=4, random_state=13).fit(V).labels_
print(f"{'raw cosine':22}{pair_cos(V):22.3f}{nmi(y,lab):13.2f}{pur(lab):8.1f}%")

# mean-centered (cheap analog of moving the anisotropic hub to a separate axis)
Vc = V - V.mean(0); Vc = Vc / (np.linalg.norm(Vc, axis=1, keepdims=True) + 1e-9)
lab = KMeans(5, n_init=4, random_state=13).fit(Vc).labels_
print(f"{'mean-centered':22}{pair_cos(Vc):22.3f}{nmi(y,lab):13.2f}{pur(lab):8.1f}%")

# Poincare ball: angle = direction in ball; cluster via hyperbolic k-means
from hyperbolic import make_ball, hyp_kmeans
P, _, _ = make_ball(V)
Pu = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)   # angular part
lab = hyp_kmeans(P, 5, 13)
print(f"{'Poincare ball (angle)':22}{pair_cos(Pu):22.3f}{nmi(y,lab):13.2f}{pur(lab):8.1f}%")

# angular concept removal in the well-behaved (centered) space
def cos_c(A, a, b): return float(unit(A[y==a].mean(0)) @ unit(A[y==b].mean(0)))
kids = unit(Vc[y=="Baby Products"].mean(0) - Vc[y=="Sports & Outdoors"].mean(0))
Vr = Vc - np.outer(Vc @ kids, kids); Vr = Vr/(np.linalg.norm(Vr,axis=1,keepdims=True)+1e-9)
print(f"\nToys<->Baby: raw {cos_c(V,'Toys & Games','Baby Products'):+.3f} | centered "
      f"{cos_c(Vc,'Toys & Games','Baby Products'):+.3f} | after removing kids {cos_c(Vr,'Toys & Games','Baby Products'):+.3f}")
print("(spread cat-pair cosine away from ~0.94 = anisotropy restructured; concept axis now usable)")
