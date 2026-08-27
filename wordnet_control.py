#!/usr/bin/env python3
"""POSITIVE CONTROL for the retracted hyperbolic storage claim.

WordNet nouns are the canonical benchmark where hyperbolic embeddings are published to
win (Nickel & Kiela NeurIPS 2017; Sala et al. ICML 2018 report MAP 0.989 in 2 dims).
It is a REAL, human-curated, deep hierarchy (82k synsets, depth 14) -- unlike our
synthetic trees and clustering-derived trees.

This run uses the SAME corrected protocol as tree_scale_fixed.py / amazon_scale_fixed.py
(stratified pairs + validity gate + equal budget), so it tests our METHOD, not the data.

Interpretation, decided before running:
  - hyperbolic WINS  -> method is sound; the retraction stands, and the finding sharpens
                        to "hyperbolic needs a true taxonomy, not a clustering-derived
                        or synthetic-uniform tree".
  - hyperbolic LOSES -> our measurement contradicts well-replicated published results,
                        so the METHODOLOGY is suspect and the retraction must itself be
                        re-opened.
"""
import collections
import numpy as np, torch, geoopt
from scipy.stats import spearmanr
from nltk.corpus import wordnet as wn

DIM, NEG, EPOCHS, BATCH = 8, 8, 600, 100_000
dev = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # ---- build the real WordNet noun hypernym DAG -> tree (first hypernym path) ----
    syns = list(wn.all_synsets('n'))
    idx = {s.name(): i for i, s in enumerate(syns)}
    N = len(syns)
    parent = np.full(N, -1, dtype=np.int64)
    for s in syns:
        hs = s.hypernyms()
        if hs:
            parent[idx[s.name()]] = idx[hs[0].name()]
    roots = int((parent == -1).sum())
    print(f"WordNet nouns: {N} synsets, {roots} roots", flush=True)

    children = collections.defaultdict(list)
    for n, p in enumerate(parent):
        if p != -1: children[int(p)].append(n)

    def chain(n):
        out = []; x = int(n); seen = set()
        while x != -1 and x not in seen:
            out.append(x); seen.add(x); x = int(parent[x])
        return out
    def under(n, cap=300):
        out = []; st = [int(n)]
        while st and len(out) < cap:
            c = st.pop(); out.append(c); st.extend(children.get(c, []))
        return out

    depths = [len(chain(i)) for i in range(0, N, max(1, N // 2000))]
    print(f"median depth {int(np.median(depths))}, max sampled depth {max(depths)}", flush=True)

    # ---- stratified pairs (same protocol as the corrected runs) ----
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(1500):
        a = int(rng.integers(0, N)); ch = chain(a)
        for lvl in range(1, len(ch)):
            cand = [x for x in under(ch[lvl]) if x != a]
            if cand: pairs.append((a, int(rng.choice(cand))))
    pa = np.array([p[0] for p in pairs]); pb = np.array([p[1] for p in pairs])

    def td_(a, b):
        ps = {};
        for d, x in enumerate(chain(a)): ps[x] = d
        d2 = 0; y = int(b)
        while y not in ps and y != -1:
            d2 += 1; y = int(parent[y])
        return (ps.get(y, 0) + d2) if y != -1 else 99
    td = np.array([td_(a, b) for a, b in zip(pa, pb)])
    keep = td < 99
    pa, pb, td = pa[keep], pb[keep], td[keep]
    cnt = collections.Counter(td.tolist())
    print(f"pairs {len(td)}, distinct distances {len(cnt)}, std {td.std():.2f}", flush=True)
    if len(cnt) < 4 or td.std() < 0.5:
        print("INVALID target -- no result reported."); return
    print("validity gate PASSED\n", flush=True)

    # TRANSITIVE CLOSURE (Nickel & Kiela protocol): every node -> ALL its ancestors.
    # Direct-parent-only edges never teach multi-hop structure and cripple the global
    # hierarchy signal that hyperbolic radius encodes.
    el = []
    for i in range(N):
        for anc in chain(i)[1:]:
            el.append([i, anc])
    edges = torch.tensor(el, device=dev)
    print(f"{len(edges)} transitive-closure edges; training both geometries at dim={DIM}", flush=True)

    res = {}
    print(f"{'geometry':>12}{'tree-dist corr':>16}")
    for geom in ["euclidean", "hyperbolic"]:
        if geom == "hyperbolic":
            ball = geoopt.PoincareBall()
            X = geoopt.ManifoldParameter(ball.random(N, DIM, std=1e-3).to(dev), manifold=ball)
            opt = geoopt.optim.RiemannianAdam([X], lr=0.05); dist = ball.dist
        else:
            X = torch.nn.Parameter(torch.randn(N, DIM, device=dev) * 1e-3)
            opt = torch.optim.Adam([X], lr=0.05); dist = lambda a, b: (a - b).norm(dim=-1)
        for ep in range(EPOCHS):
            s = edges[torch.randint(0, len(edges), (BATCH,), device=dev)]
            u, v = s[:, 0], s[:, 1]
            neg = torch.randint(0, N, (len(u), NEG), device=dev)
            dp = dist(X[u], X[v]); dn = dist(X[u].unsqueeze(1), X[neg])
            loss = torch.nn.functional.cross_entropy(
                torch.cat([-dp.unsqueeze(1), -dn], 1),
                torch.zeros(len(u), dtype=torch.long, device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 200 == 0: print(f"    {geom} ep {ep} loss {loss.item():.3f}", flush=True)
        X = X.detach()
        ed = dist(X[torch.tensor(pa, device=dev)], X[torch.tensor(pb, device=dev)]).cpu().numpy()
        res[geom] = float(spearmanr(td, ed).correlation)
        print(f"{geom:>12}{res[geom]:>16.3f}", flush=True)

    gap = res["hyperbolic"] - res["euclidean"]
    print(f"\nRESULT: hyperbolic {res['hyperbolic']:.3f} vs euclidean {res['euclidean']:.3f}")
    if gap > 0.05:
        print("-> HYPERBOLIC WINS on a true taxonomy. Method is sound; retraction stands and")
        print("   sharpens to: hyperbolic needs a REAL hierarchy, not a derived/uniform one.")
    elif gap < -0.05:
        print("-> HYPERBOLIC LOSES on the canonical benchmark where it is published to win.")
        print("   Our METHODOLOGY is suspect. The retraction must be re-opened.")
    else:
        print("-> TIED. Inconclusive as a control.")

if __name__ == "__main__":
    main()
