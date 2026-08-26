#!/usr/bin/env python3
"""Crystallization feedback loop: does re-running hyperbolic on the discovered tree
give info that REFINES the clustering? Each round: build the graph from the current
tree (transitive closure) + item kNN, train Poincare, then REASSIGN each item to its
nearest LEAF node in the ball. Measure NMI per round: refine / converge / degrade?"""
import collections, pathlib
import numpy as np, torch, geoopt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from datasets import load_dataset

HERE = pathlib.Path(__file__).parent
DIM, K, NEG, EPOCHS, LR, K1, K2, ROUNDS = 16, 6, 10, 250, 0.05, 5, 4, 3
dev = "cuda" if torch.cuda.is_available() else "cpu"
def top(c):
    for s in ["|", ">", "/"]:
        if s in c: return c.split(s)[0].strip()
    return c.strip()
def mid(c):
    p = [x.strip() for x in c.replace(">", "|").split("|")]; return p[1] if len(p) > 1 else p[0]

def crystallize(V, leaf, kNN_edges, N):
    L0, S0, R = N, N + K1*K2, N + K1*K2 + K1; NN = R + 1
    sup = leaf // K2
    edges = list(kNN_edges)
    for i in range(N): edges += [[i, L0+leaf[i]], [i, S0+sup[i]], [i, R]]
    for l in range(K1*K2): edges += [[L0+l, S0+l//K2], [L0+l, R]]
    for s in range(K1): edges.append([S0+s, R])
    ball = geoopt.PoincareBall()
    emb = geoopt.ManifoldParameter(ball.random(NN, DIM, std=1e-3).to(dev), manifold=ball)
    opt = geoopt.optim.RiemannianAdam([emb], lr=LR)
    E = torch.tensor(edges, device=dev)
    for ep in range(EPOCHS):
        p = torch.randperm(len(E), device=dev); u, v = E[p,0], E[p,1]
        neg = torch.randint(0, NN, (len(u), NEG), device=dev)
        dp = ball.dist(emb[u], emb[v]); dn = ball.dist(emb[u].unsqueeze(1), emb[neg])
        loss = torch.nn.functional.cross_entropy(torch.cat([-dp.unsqueeze(1), -dn],1),
               torch.zeros(len(u), dtype=torch.long, device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
    # reassign each item to nearest LEAF node in the ball
    P = emb.detach()
    leafP = P[L0:S0]                                    # (20, DIM)
    d = ball.dist(P[:N].unsqueeze(1), leafP.unsqueeze(0))   # (N, 20)
    new_leaf = d.argmin(1).cpu().numpy()
    return new_leaf, np.linalg.norm(P.cpu().numpy()[:N], axis=1)

def main():
    ds = load_dataset("ckandemir/amazon-products", split="train"); cats = ds["Category"]; names = ds["Product Name"]
    byc = collections.defaultdict(list)
    for i, c in enumerate(cats):
        if c and names[i]: byc[top(c)].append(i)
    tops = [t for t, v in sorted(byc.items(), key=lambda kv: -len(kv[1])) if len(v) >= 500][:6]
    idx = []
    for t in tops: idx += byc[t][:500]
    y_top = np.array([top(cats[i]) for i in idx]); y_mid = np.array([mid(cats[i]) for i in idx])
    V = np.load(HERE / "amz_emb.npz")["V"].astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    N = len(V)
    nn = NearestNeighbors(n_neighbors=K+1, metric="cosine").fit(V); _, nbr = nn.kneighbors(V)
    kNN = [[i, j] for i in range(N) for j in nbr[i, 1:]]

    # round 0: Euclidean KMeans tree
    sup0 = KMeans(K1, n_init=4, random_state=13).fit(V).labels_
    leaf = np.zeros(N, dtype=int)
    for s in range(K1):
        m = np.where(sup0 == s)[0]; leaf[m] = s*K2 + KMeans(K2, n_init=3, random_state=13).fit(V[m]).labels_
    print(f"round 0 (Euclidean KMeans): leaf-NMI(mid) {nmi(y_mid,leaf):.3f}  super-NMI(top) {nmi(y_top,leaf//K2):.3f}")
    for r in range(1, ROUNDS+1):
        leaf, rad = crystallize(V, leaf, kNN, N)
        print(f"round {r} (crystallize+reassign): leaf-NMI(mid) {nmi(y_mid,leaf):.3f}  super-NMI(top) {nmi(y_top,leaf//K2):.3f}  radius {rad.min():.2f}..{rad.max():.2f}")
    print("\nrefine if NMI rises across rounds; degrade if it falls toward the 0.17 hyperbolic-cluster level.")

if __name__ == "__main__":
    main()
