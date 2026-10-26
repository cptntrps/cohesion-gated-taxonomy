#!/usr/bin/env python3
"""Does hyperbolic store a large tree without COLLAPSE, where Euclidean can't?
Pure geometry: build b-ary trees growing to ~600k nodes, embed in the Poincare ball
vs Euclidean at the SAME low dimension, and measure collapse:
  separation = dist(node, random other) / dist(node, parent).  High = tree structure
  preserved; ~1 = collapsed (can't tell neighbors from strangers).
  distortion  = 1 - Spearman(embedded dist, true tree-hop dist) on sampled pairs.
Hyperbolic should hold as N grows; Euclidean should collapse (separation -> 1)."""
import numpy as np, torch, geoopt
from scipy.stats import spearmanr

DIM, B, NEG, EPOCHS, EDGES_PER = 8, 5, 8, 120, 80000
dev = "cuda" if torch.cuda.is_available() else "cpu"

def build_tree(target):
    parent = [-1]; depth = [0]; frontier = [0]
    while len(parent) < target:
        nf = []
        for p in frontier:
            for _ in range(B):
                if len(parent) >= target: break
                parent.append(p); depth.append(depth[p]+1); nf.append(len(parent)-1)
            if len(parent) >= target: break
        frontier = nf
        if not frontier: break
    return np.array(parent), np.array(depth)

def tree_dist(a, b, parent, depth):
    da, db = depth[a], depth[b]; d = 0
    while da > db: a = parent[a]; da -= 1; d += 1
    while db > da: b = parent[b]; db -= 1; d += 1
    while a != b: a = parent[a]; b = parent[b]; d += 2
    return d

def embed(parent, geom):
    N = len(parent)
    edges = torch.tensor([[i, parent[i]] for i in range(1, N)], device=dev)
    if geom == "hyperbolic":
        ball = geoopt.PoincareBall()
        X = geoopt.ManifoldParameter(ball.random(N, DIM, std=1e-3).to(dev), manifold=ball)
        opt = geoopt.optim.RiemannianAdam([X], lr=0.05); dist = ball.dist
    else:
        X = torch.nn.Parameter((torch.randn(N, DIM, device=dev)*1e-3))
        opt = torch.optim.Adam([X], lr=0.05); dist = lambda a, b: (a-b).norm(dim=-1)
    for ep in range(EPOCHS):
        s = edges[torch.randint(0, len(edges), (EDGES_PER,), device=dev)]
        u, v = s[:, 0], s[:, 1]; neg = torch.randint(0, N, (len(u), NEG), device=dev)
        dp = dist(X[u], X[v]); dn = dist(X[u].unsqueeze(1), X[neg])
        loss = torch.nn.functional.cross_entropy(torch.cat([-dp.unsqueeze(1), -dn], 1),
               torch.zeros(len(u), dtype=torch.long, device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
    return X.detach()

def evaluate(parent, depth, X, geom):
    N = len(parent); ball = geoopt.PoincareBall()
    dist = ball.dist if geom == "hyperbolic" else (lambda a, b: (a-b).norm(dim=-1))
    rng = np.random.default_rng(0); nodes = rng.integers(1, N, 1500)
    seps = []
    for i in nodes:
        dp = float(dist(X[i:i+1], X[parent[i]:parent[i]+1])[0])
        others = torch.tensor(rng.integers(0, N, 20), device=dev)
        do = float(dist(X[i].unsqueeze(0), X[others]).min())
        if dp > 1e-6: seps.append(do/dp)
    pa = rng.integers(0, N, 800); pb = rng.integers(0, N, 800)
    td = np.array([tree_dist(a, b, parent, depth) for a, b in zip(pa, pb)])
    ed = dist(X[torch.tensor(pa, device=dev)], X[torch.tensor(pb, device=dev)]).cpu().numpy()
    rho = spearmanr(td, ed).correlation
    return float(np.median(seps)), float(rho)

def main():
    print(f"dim={DIM} (low, so crowding shows). separation>1 = structure kept; ~1 = collapse.\n")
    print(f"{'N nodes':>9}{'geom':>12}{'separation':>12}{'dist-corr':>11}")
    for target in [20000, 200000, 600000]:
        parent, depth = build_tree(target); N = len(parent)
        for geom in ["euclidean", "hyperbolic"]:
            X = embed(parent, geom)
            sep, rho = evaluate(parent, depth, X, geom)
            print(f"{N:>9}{geom:>12}{sep:>12.2f}{rho:>11.2f}", flush=True)
        print()

if __name__ == "__main__":
    main()
