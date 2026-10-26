#!/usr/bin/env python3
"""Sarkar's combinatorial construction (Sala et al., ICML 2018) — NO optimization.

Sala et al. abandon gradient descent for hyperbolic embeddings precisely because it
suffers poor local minima and initialization sensitivity. Our own SGD/Adam attempts
reproduced that failure exactly: points collapsed to a shell at the projection ceiling
(88.6% at r>0.99) and radius carried no depth signal (rho=0.025), while the loss kept
falling. This script replaces optimization with the deterministic construction.

Algorithm (Poincare disk, complex arithmetic):
  - root at origin; its children evenly spaced on a circle of radius tanh(tau/2)
  - for any node u with parent p: apply the Mobius isometry taking u -> 0, place u's
    children at angles evenly spaced starting opposite the (transported) parent, at
    radius tanh(tau/2), then map back with the inverse isometry.
  tau controls per-level separation; larger tau = better distortion but faster
  approach to the boundary (the precision issue the paper flags).

Falsifiable goal: radius must encode depth (Spearman >> 0), which gradient training
failed to achieve (rho=0.025).
"""
import collections, sys
import numpy as np
from scipy.stats import spearmanr

TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
MAX_NODES = int(sys.argv[2]) if len(sys.argv) > 2 else 40000

def mobius_to_origin(x, z):
    """Isometry taking x -> 0, applied to z."""
    return (z - x) / (1 - np.conj(x) * z)

def mobius_from_origin(x, z):
    """Inverse: takes 0 -> x, applied to z."""
    return (z + x) / (1 + np.conj(x) * z)

def sarkar(parent, children, roots, tau):
    """Return complex positions in the Poincare disk."""
    pos = np.zeros(len(parent), dtype=np.complex128)
    R = np.tanh(tau / 2.0)                      # hyperbolic step -> euclidean radius
    order = []
    for r in roots:
        pos[r] = 0j
        q = collections.deque([r])
        while q:
            u = q.popleft(); order.append(u)
            kids = children.get(u, [])
            if not kids:
                continue
            p = parent[u]
            if p == -1:
                # root: spread children over the full circle
                angles = [2 * np.pi * i / len(kids) for i in range(len(kids))]
                base = 0.0
            else:
                # transport parent to origin-frame of u, place kids opposite it
                zp = mobius_to_origin(pos[u], pos[p])
                base = np.angle(zp)
                angles = [base + np.pi * (1 + 2 * (i + 1) / (len(kids) + 1))
                          for i in range(len(kids))]
            for a, c in zip(angles, kids):
                local = R * np.exp(1j * a)
                pos[c] = mobius_from_origin(pos[u], local)
                q.append(c)
    return pos

def main():
    from nltk.corpus import wordnet as wn
    syns = list(wn.all_synsets('n'))[:MAX_NODES]
    idx = {s.name(): i for i, s in enumerate(syns)}
    N = len(syns)
    parent = np.full(N, -1, dtype=np.int64)
    for s in syns:
        h = s.hypernyms()
        if h and h[0].name() in idx:
            parent[idx[s.name()]] = idx[h[0].name()]
    children = collections.defaultdict(list)
    for n, p in enumerate(parent):
        if p != -1: children[int(p)].append(n)
    roots = [i for i in range(N) if parent[i] == -1]

    def chain(n):
        o = []; x = int(n); seen = set()
        while x != -1 and x not in seen:
            o.append(x); seen.add(x); x = int(parent[x])
        return o
    depth = np.array([len(chain(i)) for i in range(N)])
    print(f"WordNet subset: {N} synsets, {len(roots)} roots, max depth {depth.max()}, tau={TAU}")

    pos = sarkar(parent, children, roots, TAU)
    r = np.abs(pos)
    ok = np.isfinite(r)
    print(f"finite positions: {100*ok.mean():.1f}%")
    r_ok = r[ok]
    print(f"radius: p10 {np.percentile(r_ok,10):.6f}  p50 {np.percentile(r_ok,50):.6f}  "
          f"p90 {np.percentile(r_ok,90):.6f}  max {r_ok.max():.9f}")
    print(f"points at r>0.99: {100*(r_ok>0.99).mean():.1f}%   at r>0.999999: {100*(r_ok>0.999999).mean():.1f}%")

    rho = spearmanr(depth[ok], r[ok]).correlation
    print(f"\nradius vs depth Spearman: {rho:+.3f}   (gradient training gave +0.025)")

    # tree-distance fidelity on stratified pairs, same protocol as our other runs
    def under(n, cap=200):
        out = []; st = [int(n)]
        while st and len(out) < cap:
            c = st.pop(); out.append(c); st.extend(children.get(c, []))
        return out
    rng = np.random.default_rng(0)
    pa, pb = [], []
    for _ in range(1200):
        a = int(rng.integers(0, N)); ch = chain(a)
        for lvl in range(1, len(ch)):
            cand = [x for x in under(ch[lvl]) if x != a]
            if cand: pa.append(a); pb.append(int(rng.choice(cand)))
    pa = np.array(pa); pb = np.array(pb)
    def td_(a, b):
        ps = {x: d for d, x in enumerate(chain(a))}
        d2 = 0; y = int(b)
        while y not in ps and y != -1: d2 += 1; y = int(parent[y])
        return ps.get(y, 0) + d2 if y != -1 else -1
    td = np.array([td_(a, b) for a, b in zip(pa, pb)])
    m = (td >= 0) & np.isfinite(r[pa]) & np.isfinite(r[pb])
    pa, pb, td = pa[m], pb[m], td[m]
    za, zb = pos[pa], pos[pb]
    num = np.abs(za - zb) ** 2
    den = (1 - np.abs(za) ** 2) * (1 - np.abs(zb) ** 2)
    hd = np.arccosh(np.clip(1 + 2 * num / np.clip(den, 1e-300, None), 1, None))
    good = np.isfinite(hd)
    print(f"pairs {good.sum()}, distinct distances {len(set(td[good].tolist()))}")
    print(f"tree-distance Spearman (hyperbolic, constructed): {spearmanr(td[good], hd[good]).correlation:+.3f}")
    print("  (our gradient-trained hyperbolic: -0.351 ; our Euclidean: +0.610)")

if __name__ == "__main__":
    main()
