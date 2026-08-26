#!/usr/bin/env python3
"""Corrected scale test on the REAL 205k discovered hierarchy (uses cached embeddings).

Two defects in amazon_scale.py, both mine, both fixed here:
 (1) BROKEN TARGET: random item pairs in a wide shallow tree almost all meet only at
     the root, so true tree-distance had near-zero variance and Spearman was
     meaningless. Fixed by STRATIFIED pair sampling: for each anchor, draw one partner
     from each ancestor level (same leaf, same parent, ... , root), which spans the
     full distance range by construction.
 (2) UNDERTRAINED: 150 epochs x 60k samples over ~1M edges leaves each of 234k nodes
     seen ~38 times. Budget now scales with graph size.

VALIDITY GATE (runs before any comparison): if the target has <4 distinct values or
std < 0.5, the measurement is declared INVALID and no correlation is reported. A
degenerate estimand must never be reported as a result.

Goal (unchanged): hyperbolic tree-distance correlation materially exceeds Euclidean,
as it did on synthetic trees (0.42-0.46 vs 0.14-0.21).
"""
import os, json, pathlib, collections
import numpy as np, torch, geoopt
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).parent
CACHE = pathlib.Path(os.environ.get("AMZ_BIG_EMB", HERE / "amz_big_emb.npz"))
N_ITEMS = int(os.environ.get("N_ITEMS", "205000"))
FANOUT, DEPTH = 3, 11
DIM, NEG = 8, 8
dev = "cuda" if torch.cuda.is_available() else "cpu"

def build_tree(V):
    N = len(V); parent = {}; nxt = N; root = nxt; nxt += 1; parent[root] = -1
    frontier = [(np.arange(N), root, 0)]
    while frontier:
        idx, par, d = frontier.pop()
        if d >= DEPTH or len(idx) < FANOUT * 2:
            for i in idx: parent[int(i)] = par
            continue
        k = min(FANOUT, max(2, len(idx) // 2))
        lab = MiniBatchKMeans(k, random_state=13, n_init=3, batch_size=4096).fit(V[idx]).labels_
        for c in range(k):
            sub = idx[lab == c]
            if len(sub) == 0: continue
            nid = nxt; nxt += 1; parent[nid] = par
            frontier.append((sub, nid, d + 1))
    return parent, root, nxt

def main():
    V = np.load(CACHE)["V"][:N_ITEMS]; V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    N = len(V)
    parent, root, NN = build_tree(V)
    print(f"{N} items -> {NN} nodes", flush=True)

    # children index + ancestor chains
    children = collections.defaultdict(list)
    for n, p in parent.items():
        if p != -1: children[p].append(n)
    def chain(n):
        out = []; x = n
        while x != -1: out.append(x); x = parent.get(x, -1)
        return out
    def leaves_under(n, cap=400):
        out = []; stack = [n]
        while stack and len(out) < cap:
            c = stack.pop()
            if c < N: out.append(c)
            else: stack.extend(children.get(c, []))
        return out

    # ---- FIX 1: stratified pairs, one per ancestor level ----
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(1200):
        a = int(rng.integers(0, N)); ch = chain(a)
        for lvl in range(1, len(ch)):                 # climb to each ancestor
            sibs = leaves_under(ch[lvl])
            sibs = [s for s in sibs if s != a]
            if sibs: pairs.append((a, int(rng.choice(sibs))))
    pa = np.array([p[0] for p in pairs]); pb = np.array([p[1] for p in pairs])

    def tdist(a, b):
        pset = set(chain(a)); d = 0; y = b
        while y not in pset: d += 1; y = parent.get(y, -1)
        x = a; d2 = 0
        while x != y: d2 += 1; x = parent.get(x, -1)
        return d + d2
    td = np.array([tdist(int(a), int(b)) for a, b in zip(pa, pb)])

    # ---- VALIDITY GATE ----
    dist_counts = dict(collections.Counter(td.tolist()))
    print(f"target tree-distance distribution over {len(td)} pairs: {dist_counts}")
    print(f"  distinct values {len(dist_counts)}, std {td.std():.3f}")
    if len(dist_counts) < 4 or td.std() < 0.5:
        print("\nINVALID MEASUREMENT: degenerate target (no variance to correlate).")
        print("No correlation reported. Fix the sampling before interpreting anything.")
        return
    print("  validity gate PASSED\n", flush=True)

    # ---- FIX 2: training budget scales with graph size ----
    edges = []
    for n, p in parent.items():
        x = p
        while x != -1: edges.append([n, x]); x = parent.get(x, -1)
    E = torch.tensor(edges, device=dev)
    EPOCHS = 600; BATCH = 100_000
    print(f"{len(edges)} ancestor edges; {EPOCHS} epochs x {BATCH} = "
          f"{EPOCHS*BATCH/len(edges):.1f}x edge coverage", flush=True)

    def train(geom):
        if geom == "hyperbolic":
            ball = geoopt.PoincareBall()
            X = geoopt.ManifoldParameter(ball.random(NN, DIM, std=1e-3).to(dev), manifold=ball)
            opt = geoopt.optim.RiemannianAdam([X], lr=0.05); dist = ball.dist
        else:
            X = torch.nn.Parameter(torch.randn(NN, DIM, device=dev) * 1e-3)
            opt = torch.optim.Adam([X], lr=0.05); dist = lambda a, b: (a - b).norm(dim=-1)
        for ep in range(EPOCHS):
            s = E[torch.randint(0, len(E), (BATCH,), device=dev)]
            u, v = s[:, 0], s[:, 1]
            neg = torch.randint(0, NN, (len(u), NEG), device=dev)
            dp = dist(X[u], X[v]); dn = dist(X[u].unsqueeze(1), X[neg])
            loss = torch.nn.functional.cross_entropy(
                torch.cat([-dp.unsqueeze(1), -dn], 1),
                torch.zeros(len(u), dtype=torch.long, device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 150 == 0: print(f"    {geom} ep {ep} loss {loss.item():.3f}", flush=True)
        return X.detach(), dist

    res = {}
    print(f"{'geometry':>12}{'tree-dist corr':>16}")
    for geom in ["euclidean", "hyperbolic"]:
        X, dist = train(geom)
        ed = dist(X[torch.tensor(pa, device=dev)], X[torch.tensor(pb, device=dev)]).cpu().numpy()
        rho = spearmanr(td, ed).correlation
        res[geom] = float(rho)
        print(f"{geom:>12}{rho:>16.3f}", flush=True)

    gap = res["hyperbolic"] - res["euclidean"]
    verdict = ("CONFIRMED on real discovered hierarchy" if gap > 0.05
               else "NOT confirmed" if gap < -0.05 else "TIED (no material difference)")
    print(f"\nRESULT: hyperbolic {res['hyperbolic']:.3f} vs euclidean {res['euclidean']:.3f} -> {verdict}")
    json.dump({"n_items": N, "n_nodes": NN, "dim": DIM, "n_pairs": len(td),
               "target_distinct": len(dist_counts), "verdict": verdict, **res},
              open(HERE / "amazon_scale_deep.json", "w"), indent=1)

if __name__ == "__main__":
    main()
