#!/usr/bin/env python3
"""CORRECTED synthetic tree-scale test.

tree_scale.py sampled RANDOM node pairs, which in a broad tree nearly all meet only at
the root -> the target tree-distance had almost no variance and Spearman was
meaningless. That is the same defect found in amazon_scale.py. This version uses
STRATIFIED pairs (one partner per ancestor level) plus a validity gate, so the
synthetic and real-data numbers are computed the same way and are comparable.
"""
import collections
import numpy as np, torch, geoopt
from scipy.stats import spearmanr

DIM, B, NEG, EPOCHS, EDGES_PER = 8, 5, 8, 400, 100000
dev = "cuda" if torch.cuda.is_available() else "cpu"

def build_tree(target):
    parent = [-1]; frontier = [0]
    while len(parent) < target:
        nf = []
        for p in frontier:
            for _ in range(B):
                if len(parent) >= target: break
                parent.append(p); nf.append(len(parent) - 1)
            if len(parent) >= target: break
        frontier = nf
        if not frontier: break
    return np.array(parent)

def main():
    print(f"dim={DIM}, stratified pairs + validity gate\n")
    print(f"{'N nodes':>9}{'geom':>12}{'dist-corr':>11}")
    for target in [200000, 600000]:
        parent = build_tree(target); N = len(parent)
        children = collections.defaultdict(list)
        for n, p in enumerate(parent):
            if p != -1: children[p].append(n)
        def chain(n):
            out = []; x = n
            while x != -1: out.append(x); x = parent[x]
            return out
        def under(n, cap=300):
            out = []; st = [n]
            while st and len(out) < cap:
                c = st.pop(); out.append(c); st.extend(children.get(c, []))
            return out
        rng = np.random.default_rng(0)
        pairs = []
        for _ in range(900):
            a = int(rng.integers(0, N)); ch = chain(a)
            for lvl in range(1, len(ch)):
                cand = [x for x in under(ch[lvl]) if x != a]
                if cand: pairs.append((a, int(rng.choice(cand))))
        pa = np.array([p[0] for p in pairs]); pb = np.array([p[1] for p in pairs])
        def td_(a, b):
            ps = set(chain(a)); d = 0; y = b
            while y not in ps: d += 1; y = parent[y]
            x = a; d2 = 0
            while x != y: d2 += 1; x = parent[x]
            return d + d2
        td = np.array([td_(int(a), int(b)) for a, b in zip(pa, pb)])
        cnt = collections.Counter(td.tolist())
        if len(cnt) < 4 or td.std() < 0.5:
            print(f"{N:>9}  INVALID target (distinct {len(cnt)}, std {td.std():.2f}) -- no result")
            continue
        print(f"{N:>9}  [{len(td)} pairs, {len(cnt)} distinct distances, std {td.std():.2f}]")

        edges = torch.tensor([[i, parent[i]] for i in range(1, N)], device=dev)
        for geom in ["euclidean", "hyperbolic"]:
            if geom == "hyperbolic":
                ball = geoopt.PoincareBall()
                X = geoopt.ManifoldParameter(ball.random(N, DIM, std=1e-3).to(dev), manifold=ball)
                opt = geoopt.optim.RiemannianAdam([X], lr=0.05); dist = ball.dist
            else:
                X = torch.nn.Parameter(torch.randn(N, DIM, device=dev) * 1e-3)
                opt = torch.optim.Adam([X], lr=0.05); dist = lambda a, b: (a - b).norm(dim=-1)
            for ep in range(EPOCHS):
                s = edges[torch.randint(0, len(edges), (EDGES_PER,), device=dev)]
                u, v = s[:, 0], s[:, 1]
                neg = torch.randint(0, N, (len(u), NEG), device=dev)
                dp = dist(X[u], X[v]); dn = dist(X[u].unsqueeze(1), X[neg])
                loss = torch.nn.functional.cross_entropy(
                    torch.cat([-dp.unsqueeze(1), -dn], 1),
                    torch.zeros(len(u), dtype=torch.long, device=dev))
                opt.zero_grad(); loss.backward(); opt.step()
            X = X.detach()
            ed = dist(X[torch.tensor(pa, device=dev)], X[torch.tensor(pb, device=dev)]).cpu().numpy()
            print(f"{'':>9}{geom:>12}{spearmanr(td, ed).correlation:>11.3f}", flush=True)
        print()

if __name__ == "__main__":
    main()
