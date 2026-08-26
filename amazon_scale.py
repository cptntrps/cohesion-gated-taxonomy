#!/usr/bin/env python3
"""Scale test on REAL data: does hyperbolic store a DISCOVERED hierarchy without
collapse, where Euclidean degrades? (tree_scale.py did this on synthetic k-ary trees;
this uses a hierarchy our own pipeline discovered from real Amazon products.)

Stage 1 (slow, once): embed N real product titles+descriptions locally -> cache.
Stage 2: discover a hierarchy by recursive KMeans (the pipeline's own EMERGE step).
Stage 3: embed that discovered tree (transitive-closure ancestor edges) in the
         Poincare ball vs Euclidean at equal low dim; measure tree-distance
         correlation = the 'no collapse' metric.

Falsifiable goal: hyperbolic tree-distance correlation materially exceeds Euclidean
at this scale, as it did on synthetic trees (0.42-0.46 vs 0.14-0.21).
"""
import os, sys, json, pathlib, collections, urllib.request
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).parent
CACHE = pathlib.Path(os.environ.get("AMZ_BIG_EMB", HERE / "amz_big_emb.npz"))
N_ITEMS = int(os.environ.get("N_ITEMS", "205000"))
FANOUT, DEPTH = 8, 5          # 8^5 = 32768 leaves max
DIM, NEG, EPOCHS = 8, 8, 150

def load_texts():
    from datasets import load_dataset
    ds = load_dataset("smartcat/Amazon-2023-GenQ", split="train", streaming=True)
    texts, cats = [], []
    for r in ds:
        t = (r.get("title") or "").strip()
        if not t: continue
        texts.append((t + ". " + (r.get("description") or "")[:200]).strip())
        cats.append(r.get("main_category") or "?")
        if len(texts) >= N_ITEMS: break
    return texts, np.array(cats)

def embed(texts):
    if CACHE.exists():
        V = np.load(CACHE)["V"]
        if len(V) >= len(texts): return V[:len(texts)]
    V = np.zeros((len(texts), 768), dtype=np.float32); B = 64
    for i in range(0, len(texts), B):
        b = [t.lower()[:600] for t in texts[i:i+B]]
        req = urllib.request.Request("http://localhost:11434/api/embed",
            data=json.dumps({"model": "nomic-embed-text", "input": b}).encode(),
            headers={"Content-Type": "application/json"})
        V[i:i+len(b)] = np.array(json.load(urllib.request.urlopen(req, timeout=900))["embeddings"], dtype=np.float32)
        if (i//B) % 200 == 0:
            print(f"  embed {i}/{len(texts)}", flush=True)
            np.savez_compressed(CACHE, V=V)      # checkpoint
    np.savez_compressed(CACHE, V=V)
    return V

def discover(V):
    """Recursive MiniBatchKMeans -> parent array over items+internal nodes."""
    N = len(V)
    node_parent = {}          # node id -> parent id ; items are 0..N-1
    next_id = N
    root = next_id; next_id += 1
    node_parent[root] = -1
    frontier = [(np.arange(N), root, 0)]
    while frontier:
        idx, parent, d = frontier.pop()
        if d >= DEPTH or len(idx) < FANOUT * 2:
            for i in idx: node_parent[int(i)] = parent
            continue
        k = min(FANOUT, max(2, len(idx) // 2))
        lab = MiniBatchKMeans(k, random_state=13, n_init=3, batch_size=4096).fit(V[idx]).labels_
        for c in range(k):
            sub = idx[lab == c]
            if len(sub) == 0: continue
            nid = next_id; next_id += 1
            node_parent[nid] = parent
            frontier.append((sub, nid, d + 1))
    return node_parent, root, next_id

def main():
    import torch, geoopt
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading up to {N_ITEMS} products...", flush=True)
    texts, cats = load_texts()
    print(f"{len(texts)} products, {len(set(cats.tolist()))} main categories", flush=True)
    V = embed(texts); V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    print("embedded. discovering hierarchy...", flush=True)
    node_parent, root, NN = discover(V)
    depths = {}
    def depth_of(n):
        if n in depths: return depths[n]
        p = node_parent.get(n, -1)
        depths[n] = 0 if p == -1 else depth_of(p) + 1
        return depths[n]
    for n in list(node_parent): depth_of(n)
    print(f"discovered tree: {NN} nodes ({len(texts)} items + {NN-len(texts)} internal), max depth {max(depths.values())}", flush=True)

    edges = []
    for n, p in node_parent.items():
        cur = p
        while cur != -1:                       # transitive closure to all ancestors
            edges.append([n, cur]); cur = node_parent.get(cur, -1)
    E = torch.tensor(edges, device=dev)
    print(f"{len(edges)} ancestor edges. training both geometries at dim={DIM}...", flush=True)

    def train(geom):
        if geom == "hyperbolic":
            ball = geoopt.PoincareBall()
            X = geoopt.ManifoldParameter(ball.random(NN, DIM, std=1e-3).to(dev), manifold=ball)
            opt = geoopt.optim.RiemannianAdam([X], lr=0.05); dist = ball.dist
        else:
            X = torch.nn.Parameter(torch.randn(NN, DIM, device=dev) * 1e-3)
            opt = torch.optim.Adam([X], lr=0.05); dist = lambda a, b: (a - b).norm(dim=-1)
        for ep in range(EPOCHS):
            s = E[torch.randint(0, len(E), (60000,), device=dev)]
            u, v = s[:, 0], s[:, 1]
            neg = torch.randint(0, NN, (len(u), NEG), device=dev)
            dp = dist(X[u], X[v]); dn = dist(X[u].unsqueeze(1), X[neg])
            loss = torch.nn.functional.cross_entropy(torch.cat([-dp.unsqueeze(1), -dn], 1),
                   torch.zeros(len(u), dtype=torch.long, device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 50 == 0: print(f"    {geom} ep {ep} loss {loss.item():.3f}", flush=True)
        return X.detach(), dist

    def tree_dist(a, b):
        pa = set(); x = a
        while x != -1: pa.add(x); x = node_parent.get(x, -1)
        d = 0; y = b
        while y not in pa: d += 1; y = node_parent.get(y, -1)
        x = a; d2 = 0
        while x != y: d2 += 1; x = node_parent.get(x, -1)
        return d + d2

    rng = np.random.default_rng(0)
    pa = rng.integers(0, len(texts), 800); pb = rng.integers(0, len(texts), 800)
    td = np.array([tree_dist(int(a), int(b)) for a, b in zip(pa, pb)])
    print(f"\n{'geometry':>12}{'tree-dist corr':>16}")
    res = {}
    for geom in ["euclidean", "hyperbolic"]:
        X, dist = train(geom)
        ed = dist(X[torch.tensor(pa, device=dev)], X[torch.tensor(pb, device=dev)]).cpu().numpy()
        rho = spearmanr(td, ed).correlation
        res[geom] = float(rho)
        print(f"{geom:>12}{rho:>16.3f}", flush=True)
    print(f"\nGOAL: hyperbolic materially exceeds euclidean (synthetic trees gave 0.42-0.46 vs 0.14-0.21).")
    print(f"RESULT: hyperbolic {res['hyperbolic']:.3f} vs euclidean {res['euclidean']:.3f} -> "
          f"{'CONFIRMED on real discovered hierarchy' if res['hyperbolic'] > res['euclidean'] + 0.05 else 'NOT confirmed'}")
    json.dump({"n_items": len(texts), "n_nodes": NN, "dim": DIM, **res},
              open(HERE / "amazon_scale.json", "w"), indent=1)

if __name__ == "__main__":
    main()
