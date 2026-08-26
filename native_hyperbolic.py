#!/usr/bin/env python3
"""Train NATIVE Poincare embeddings (Riemannian, geoopt) on the kNN graph of the
Amazon nomic embeddings, then compare to raw Euclidean.

Question (owner): the anisotropy is a property of the EUCLIDEAN nomic model. Does a
natively-hyperbolic embedding (radius emerges as generality, angle as concept) remove
it AND cluster at least as well? Graph = k-NN from nomic (the practical 're-embed
Euclidean into the ball' path; a from-scratch hyperbolic text model isn't available).
"""
import collections, pathlib
import numpy as np, torch, geoopt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from datasets import load_dataset

HERE = pathlib.Path(__file__).parent
DIM, K, NEG, EPOCHS, LR = 16, 10, 10, 300, 0.05
dev = "cuda" if torch.cuda.is_available() else "cpu"

def top(c):
    for s in ["|", ">", "/"]:
        if s in c: return c.split(s)[0].strip()
    return c.strip()

def main():
    ds = load_dataset("ckandemir/amazon-products", split="train"); cats = ds["Category"]; names = ds["Product Name"]
    byc = collections.defaultdict(list)
    for i, c in enumerate(cats):
        if c and names[i]: byc[top(c)].append(i)
    tops = [t for t, v in sorted(byc.items(), key=lambda kv: -len(kv[1])) if len(v) >= 500][:6]
    idx = []
    for t in tops: idx += byc[t][:500]
    y = np.array([top(cats[i]) for i in idx])
    V = np.load(HERE / "amz_emb.npz")["V"].astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    N = len(V)

    # kNN graph from nomic (positive edges)
    nn = NearestNeighbors(n_neighbors=K+1, metric="cosine").fit(V)
    _, nbr = nn.kneighbors(V)
    edges = np.array([[i, j] for i in range(N) for j in nbr[i, 1:]])
    print(f"{N} nodes, {len(edges)} kNN edges, training Poincare dim={DIM} on {dev}", flush=True)

    ball = geoopt.PoincareBall()
    emb = geoopt.ManifoldParameter(ball.random(N, DIM, std=1e-3).to(dev), manifold=ball)
    opt = geoopt.optim.RiemannianAdam([emb], lr=LR)
    E = torch.tensor(edges, device=dev)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(E), device=dev)
        u, v = E[perm, 0], E[perm, 1]
        neg = torch.randint(0, N, (len(u), NEG), device=dev)
        dp = ball.dist(emb[u], emb[v])                      # (B,)
        dn = ball.dist(emb[u].unsqueeze(1), emb[neg])        # (B, NEG)
        logits = torch.cat([-dp.unsqueeze(1), -dn], dim=1)   # positive first
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(u), dtype=torch.long, device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 60 == 0: print(f"  ep {ep} loss {loss.item():.3f}", flush=True)

    P = emb.detach().cpu().numpy()
    rad = np.linalg.norm(P, axis=1)
    Pu = P / (rad[:, None] + 1e-9)
    from itertools import combinations
    def unit(x): return x / (np.linalg.norm(x) + 1e-9)
    def pair(A):
        cen = {t: unit(A[y == t].mean(0)) for t in tops}
        return float(np.mean([cen[a] @ cen[b] for a, b in combinations(tops, 2)]))
    def pur(a): return round(100*sum(collections.Counter(y[a==c].tolist()).most_common(1)[0][1] for c in set(a.tolist()))/len(y),1)

    labE = KMeans(5, n_init=4, random_state=13).fit(V).labels_
    labH = KMeans(5, n_init=4, random_state=13).fit(P).labels_   # Euclidean KMeans in the ball coords
    print("\n== native Poincare vs Euclidean nomic ==")
    print(f"  Euclidean nomic : cat-pair cos {pair(V):+.3f}  NMI {nmi(y,labE):.2f}  purity {pur(labE)}%")
    print(f"  native Poincare : angle cos {pair(Pu):+.3f}  NMI {nmi(y,labH):.2f}  purity {pur(labH)}%")
    print(f"\nradius range: {rad.min():.3f}..{rad.max():.3f}  (radius = generality if broad cats sit lower)")
    print("mean radius by category (lower = more central/general):")
    for t in sorted(tops, key=lambda t: rad[y==t].mean()):
        print(f"  {t:26} {rad[y==t].mean():.3f}")

if __name__ == "__main__":
    main()
