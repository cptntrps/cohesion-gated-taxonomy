#!/usr/bin/env python3
"""Round 2: seed the graph with the round-1 hierarchy, then re-embed hyperbolically.

Round 1 (flat KMeans) discovers a 2-level tree: 5 supers -> 20 leaves. We add those
tree nodes to the graph (item->leaf->super->root) plus item-item kNN, then train
native Poincare embeddings. Test: does radius NOW encode depth (root center ->
items boundary), which the flat kNN graph failed to do? That is hyperbolic's actual
value proposition, and the crystallization claim: round 1 gives the hierarchy that
round 2's geometry needs.
"""
import collections, pathlib
import numpy as np, torch, geoopt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from datasets import load_dataset

HERE = pathlib.Path(__file__).parent
DIM, K, NEG, EPOCHS, LR = 16, 6, 10, 300, 0.05
K1, K2 = 5, 4
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

    # --- ROUND 1: flat KMeans hierarchy: 5 supers -> 4 leaves each ---
    sup = KMeans(K1, n_init=4, random_state=13).fit(V).labels_
    leaf = np.zeros(N, dtype=int)
    for s in range(K1):
        m = np.where(sup == s)[0]
        leaf[m] = s * K2 + KMeans(K2, n_init=3, random_state=13).fit(V[m]).labels_
    # node ids: items 0..N-1, leaves N..N+19, supers N+20..N+24, root N+25
    LEAF0, SUP0, ROOT = N, N + K1 * K2, N + K1 * K2 + K1
    NN = ROOT + 1

    # --- graph: item-item kNN (local) + hierarchy edges (structure) ---
    nn = NearestNeighbors(n_neighbors=K+1, metric="cosine").fit(V); _, nbr = nn.kneighbors(V)
    edges = [[i, j] for i in range(N) for j in nbr[i, 1:]]
    for i in range(N):                                       # item -> ALL ancestors (transitive closure)
        edges += [[i, LEAF0 + leaf[i]], [i, SUP0 + sup[i]], [i, ROOT]]
    for l in range(K1 * K2):
        edges += [[LEAF0 + l, SUP0 + l // K2], [LEAF0 + l, ROOT]]
    for s in range(K1):
        edges.append([SUP0 + s, ROOT])
    E = torch.tensor(edges, device=dev)
    print(f"{NN} nodes ({N} items + {K1*K2} leaves + {K1} supers + root), {len(edges)} edges on {dev}", flush=True)

    ball = geoopt.PoincareBall()
    emb = geoopt.ManifoldParameter(ball.random(NN, DIM, std=1e-3).to(dev), manifold=ball)
    opt = geoopt.optim.RiemannianAdam([emb], lr=LR)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(E), device=dev); u, v = E[perm, 0], E[perm, 1]
        neg = torch.randint(0, NN, (len(u), NEG), device=dev)
        dp = ball.dist(emb[u], emb[v]); dn = ball.dist(emb[u].unsqueeze(1), emb[neg])
        logits = torch.cat([-dp.unsqueeze(1), -dn], 1)
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(u), dtype=torch.long, device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"  ep {ep} loss {loss.item():.3f}", flush=True)

    P = emb.detach().cpu().numpy(); rad = np.linalg.norm(P, axis=1)
    print("\n== radius by depth (root should be smallest -> items largest) ==")
    print(f"  root        {rad[ROOT]:.3f}")
    print(f"  supers      {rad[SUP0:ROOT].mean():.3f}  (n={K1})")
    print(f"  leaves      {rad[LEAF0:SUP0].mean():.3f}  (n={K1*K2})")
    print(f"  items       {rad[:N].mean():.3f}  (n={N})")
    mono = rad[ROOT] < rad[SUP0:ROOT].mean() < rad[LEAF0:SUP0].mean() < rad[:N].mean()
    print(f"  -> radius encodes depth (monotone root<super<leaf<item): {mono}")

    Pi = P[:N]
    def pur(a): return round(100*sum(collections.Counter(y[a==c].tolist()).most_common(1)[0][1] for c in set(a.tolist()))/len(y),1)
    labH = KMeans(5, n_init=4, random_state=13).fit(Pi).labels_
    labE = KMeans(5, n_init=4, random_state=13).fit(V).labels_
    print(f"\nitem clustering NMI: round2-hyperbolic {nmi(y,labH):.2f} (pur {pur(labH)}%)  vs  flat-kNN-hyperbolic 0.19  vs  Euclidean {nmi(y,labE):.2f}")

if __name__ == "__main__":
    main()
