#!/usr/bin/env python3
"""Hyperbolic cluster-finding in the Poincare ball (owner directive 2026-08-26).

Reuses the crossover exp map (to_ball) and Poincare distance. Unsupervised:
- PCA the 768d clause embeddings to a low ball dimension.
- Scale so points spread across radii (tanh), then map into the ball.
- k-means with Poincare-distance assignment and ball-projected centroids.

Honest note (kept, not hidden): radius here is PCA magnitude, not a trained
hierarchy depth, so this is hyperbolic geometry over cosine structure. cluster.py
reports flat-vs-hyperbolic leaf purity side by side so the geometry is judged, not
asserted.
"""
import numpy as np
from sklearn.decomposition import PCA

def to_ball(X, eps=1e-5):
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    scale = np.tanh(n) / (n + 1e-9)
    return X * scale * (1 - eps)

def make_ball(V, dim=48, target_radius=0.7, seed=13):
    """768d unit embeddings -> points in the `dim`-D Poincare ball."""
    pca = PCA(n_components=min(dim, V.shape[1]), random_state=seed)
    X = pca.fit_transform(V)
    med = np.median(np.linalg.norm(X, axis=1)) + 1e-9
    s = np.arctanh(target_radius) / med          # median norm -> target_radius
    P = to_ball(s * X).astype(np.float32)
    return P, pca, s

def project_new(names_V, pca, s):
    """Map fresh 768d vectors (e.g. anchor names) into the same ball."""
    X = pca.transform(names_V)
    return to_ball(s * X).astype(np.float32)

def poincare_dist_matrix(P, C):
    """P (n,d), C (k,d) in the ball -> (n,k) Poincare distances."""
    Pn2 = (P ** 2).sum(1)
    Cn2 = (C ** 2).sum(1)
    cross = P @ C.T
    diff2 = Pn2[:, None] + Cn2[None, :] - 2 * cross
    denom = np.clip((1 - Pn2)[:, None] * (1 - Cn2)[None, :], 1e-7, None)
    arg = np.clip(1 + 2 * diff2 / denom, 1 + 1e-9, None)
    return np.arccosh(arg)

def _ball_mean(pts):
    m = pts.mean(0)
    nn = np.linalg.norm(m)
    return m / nn * (1 - 1e-4) if nn >= 1 else m

def hyp_kmeans(P, k, seed=13, iters=14):
    """Poincare-ball k-means. Returns integer labels (0..k-1)."""
    k = min(k, len(P))
    if k < 2:
        return np.zeros(len(P), dtype=int)
    rng = np.random.default_rng(seed)
    # k-means++ style seed by Poincare distance
    idx0 = int(rng.integers(len(P)))
    centers = [P[idx0]]
    for _ in range(k - 1):
        d = poincare_dist_matrix(P, np.array(centers)).min(1)
        probs = d ** 2
        tot = probs.sum()
        centers.append(P[int(rng.integers(len(P)))] if tot <= 0
                       else P[int(rng.choice(len(P), p=probs / tot))])
    C = np.array(centers, dtype=np.float32)
    labels = np.zeros(len(P), dtype=int)
    for _ in range(iters):
        labels = poincare_dist_matrix(P, C).argmin(1)
        for j in range(k):
            pts = P[labels == j]
            if len(pts):
                C[j] = _ball_mean(pts)
    return labels
