"""WORK-2283 retrieval arms over a catalog of skills and owner decisions. Hermetic: numpy and scikit-learn only.

Arms (each returns a full ranking of catalog indices for one query):
  bm25        Okapi BM25 over the item text (name + description head).
  flat        cosine on the full embedding.
  flat_pca    cosine on the PCA-reduced embedding (isolates the dimension reduction).
  hyp_tree    the treatment: the catalog tree in a d-dim Poincare ball (Sarkar-style construction with Mobius
              addition, semantic child directions, no gradient training: RESULTS.md sections 12-13); the query
              descends by beam search, is inserted as a child of its landing node, and items are ranked by
              unrounded Poincare distance to it, so the geometry decides every rank.
  tree_cos    the same tree, descent and landing node, ranked by explicit tree hop distance instead of Poincare distance
              (isolates the hyperbolic geometry from the tree).
  popularity  items by gold frequency in the dev split.
  random      a seeded permutation.
The LLM arm lives in llm_arm.py (it needs a model).
"""
import math
import re
from collections import Counter, deque

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text):
    return TOKEN.findall(text.lower().replace("_", " ").replace("-", " "))


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokens(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.docs], dtype=float)
        self.avg = self.len.mean() if len(self.docs) else 0.0
        df = Counter(t for c in self.docs for t in c)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query):
        q = set(tokens(query))
        s = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.len / (self.avg or 1.0))
        for i, c in enumerate(self.docs):
            for t in q:
                f = c.get(t)
                if f:
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + norm[i])
        return s


def unit(X):
    X = np.asarray(X, dtype=np.float64)
    return X / np.clip(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12, None)


def rank_by(scores, tiebreak=None):
    """Indices by descending score; equal scores fall back to tiebreak (descending) then index."""
    scores = np.asarray(scores, dtype=float)
    tb = np.zeros_like(scores) if tiebreak is None else np.asarray(tiebreak, dtype=float)
    return np.lexsort((np.arange(len(scores)), -tb, -scores))


# ------------------------------------------------------------------ the tree

class Tree:
    """Nodes 0..n-1. parent[0] = -1 is the root. leaf_item[node] = catalog index or -1."""

    def __init__(self):
        self.parent, self.children, self.leaf_item, self.label = [-1], [[]], [-1], ["root"]

    def add(self, parent, label, item=-1):
        self.parent.append(parent)
        self.children.append([])
        self.leaf_item.append(item)
        self.label.append(label)
        self.children[parent].append(len(self.parent) - 1)
        return len(self.parent) - 1

    def __len__(self):
        return len(self.parent)

    def depth(self):
        d = [0] * len(self)
        for v in range(1, len(self)):
            d[v] = d[self.parent[v]] + 1
        return d

    def leaves(self):
        return [v for v in range(len(self)) if self.leaf_item[v] >= 0]

    def hops_from(self, src):
        dist = [-1] * len(self)
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for w in self.children[u] + ([self.parent[u]] if self.parent[u] >= 0 else []):
                if dist[w] < 0:
                    dist[w] = dist[u] + 1
                    q.append(w)
        return dist


def build_tree(kinds, groups, V, skill_clusters=14, max_group=24, per_sub=12, seed=13):
    """root -> kind -> group -> [subgroup] -> item.

    Decisions use their real domain as the group. Skills have no usable metadata grouping, so their groups are
    Ward clusters of the embeddings (an imposed level: a caveat). A group larger than max_group is split into
    ceil(n/per_sub) Ward clusters.
    """
    V = unit(V)
    t = Tree()
    for kind in sorted(set(kinds)):
        kn = t.add(0, f"kind:{kind}")
        idx = [i for i, k in enumerate(kinds) if k == kind]
        if kind == "skill" or any(groups[i] is None for i in idx):
            k = min(skill_clusters, len(idx))
            lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(V[idx]) if k > 1 else [0] * len(idx)
            gmap = {}
            for i, l in zip(idx, lab):
                gmap.setdefault(f"cluster{l}", []).append(i)
        else:
            gmap = {}
            for i in idx:
                gmap.setdefault(groups[i], []).append(i)
        for g in sorted(gmap):
            members = gmap[g]
            gn = t.add(kn, f"{kind}/{g}")
            if len(members) > max_group:
                k = math.ceil(len(members) / per_sub)
                lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(V[members])
                subs = {}
                for i, l in zip(members, lab):
                    subs.setdefault(int(l), []).append(i)
                for l in sorted(subs):
                    sn = t.add(gn, f"{kind}/{g}/{l}")
                    for i in subs[l]:
                        t.add(sn, f"item{i}", i)
            else:
                for i in members:
                    t.add(gn, f"item{i}", i)
    return t


# ------------------------------------------------------------------ Sarkar's construction in the Poincare disk

def mobius_add(x, y):
    """Mobius addition x (+) y in the Poincare ball (curvature -1); x (d,) or (n, d), y (n, d) or (d,).
    The same operation as pg_hyperbolic hyp_mobius_add."""
    x, y = np.atleast_2d(x).astype(np.float64), np.atleast_2d(y).astype(np.float64)
    xy = np.sum(x * y, -1, keepdims=True)
    x2 = np.sum(x * x, -1, keepdims=True)
    y2 = np.sum(y * y, -1, keepdims=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    return num / (1 + 2 * xy + x2 * y2)


def poincare(a, b):
    """Poincare-ball distance between real points (last axis = coordinates, broadcasts); pg_hyperbolic
    poincare_distance: arcosh(1 + 2|a-b|^2 / ((1-|a|^2)(1-|b|^2)))."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    num = 2 * np.sum((a - b) ** 2, -1)
    den = np.clip((1 - np.sum(a * a, -1)) * (1 - np.sum(b * b, -1)), 1e-300, None)
    return np.arccosh(1 + num / den)


def child_direction(offset, toward_parent, away=0.5):
    """Unit direction for a child: its semantic offset with the component toward the parent removed, pushed away
    from the parent by `away` (so children sit on the far side, as in Sarkar's construction)."""
    o = np.asarray(offset, dtype=np.float64)
    if toward_parent is not None:
        e = toward_parent / max(np.linalg.norm(toward_parent), 1e-12)
        o = o - (o @ e) * e
        n = np.linalg.norm(o)
        o = (o / n if n > 1e-12 else np.zeros_like(o)) - away * e
    n = np.linalg.norm(o)
    if n < 1e-12:
        o = np.zeros_like(o)
        o[0] = 1.0
        n = 1.0
    return o / n


def build_ball(tree, low, tau, away=0.5):
    """Positions of every tree node in the d-dim Poincare ball (d = low.shape[1]). Root at 0; each child at
    hyperbolic distance tau from its parent: x_c = x_u (+) tanh(tau/2) v_c, with v_c the child's semantic
    direction (descriptor offset from the parent's, pointed away from the grandparent). No optimisation."""
    pos = np.zeros((len(tree), low.shape[1]))
    r = math.tanh(tau / 2.0)
    q = deque([0])
    while q:
        u = q.popleft()
        if not tree.children[u]:
            continue
        toward = None
        if tree.parent[u] >= 0:
            toward = mobius_add(-pos[u], pos[tree.parent[u]])[0]
        for c in tree.children[u]:
            v = child_direction(low[c] - low[u], toward, away)
            pos[c] = mobius_add(pos[u], r * v)[0]
            q.append(c)
    return pos


class HypIndex:
    """The treatment. dim: PCA dimension of the descriptors used for the descent; ball_dim: dimension of the
    Poincare ball; tau: hyperbolic step parent -> child; tau_q: the step at which the query is inserted.

    rank(): the query descends the tree (beam search on descriptor cosine), is inserted into the ball as a new
    child of the best landing node by the same rule as the items, and every item is ranked by its unrounded
    Poincare distance to that point. metric='hops' is the tree_cos arm: the same landing node, items ranked by
    tree hop distance from it, ties by descriptor cosine (no geometry). land_all: use every final beam node (the
    query inserted under each; the minimum distance or hop count wins) instead of the best one only."""

    def __init__(self, tree, V, dim=128, ball_dim=32, tau=2.5, tau_q=None, beam=6, seed=13, away=0.5,
                 land_all=False):
        self.tree, self.beam, self.tau, self.land_all = tree, beam, tau, land_all
        self.tau_q = tau if tau_q is None else tau_q
        self.away = away
        V = unit(V)
        self.pca = PCA(n_components=min(dim, V.shape[1], V.shape[0]), random_state=seed).fit(V)
        self.Xd = unit(self.pca.transform(V))
        self.leaf_of_item = {tree.leaf_item[v]: v for v in tree.leaves()}
        self.item_nodes = np.array([self.leaf_of_item[i] for i in range(len(V))])
        desc = np.zeros((len(tree), self.Xd.shape[1]))
        for v in reversed(range(len(tree))):
            if tree.leaf_item[v] >= 0:
                desc[v] = self.Xd[tree.leaf_item[v]]
            else:
                desc[v] = sum((desc[c] for c in tree.children[v]), np.zeros(self.Xd.shape[1]))
        self.desc = unit(desc)
        k = min(ball_dim, self.desc.shape[1])
        self.ball_pca = PCA(n_components=k, random_state=seed).fit(self.desc)
        self.low = self.ball_pca.transform(self.desc)
        self.pos = build_ball(tree, self.low, tau, away)
        self.leaf_parent = [bool(tree.children[v]) and all(tree.leaf_item[c] >= 0 for c in tree.children[v])
                            for v in range(len(tree))]

    def reduce(self, qv):
        return unit(self.pca.transform(unit(np.atleast_2d(qv))))[0]

    def descend(self, qd):
        beam = [0]
        while not all(self.leaf_parent[v] for v in beam):
            cand = []
            for v in beam:
                cand += [v] if self.leaf_parent[v] else [c for c in self.tree.children[v] if self.tree.leaf_item[c] < 0]
            cand = sorted(set(cand), key=lambda v: (-float(self.desc[v] @ qd), v))
            beam = cand[: self.beam]
        return sorted(beam, key=lambda v: (-float(self.desc[v] @ qd), v))

    def query_point(self, qd, b):
        toward = mobius_add(-self.pos[b], self.pos[self.tree.parent[b]])[0] if self.tree.parent[b] >= 0 else None
        ql = self.ball_pca.transform(qd[None, :])[0]
        v = child_direction(ql - self.low[b], toward, self.away)
        return mobius_add(self.pos[b], math.tanh(self.tau_q / 2.0) * v)[0]

    def rank(self, qv, metric="poincare"):
        qd = self.reduce(qv)
        land = self.descend(qd)
        nodes = land if self.land_all else land[:1]
        if metric == "poincare":
            d = np.min([poincare(self.query_point(qd, b)[None, :], self.pos[self.item_nodes]) for b in nodes], axis=0)
            return rank_by(-d), land
        hops = np.min([np.array(self.tree.hops_from(b))[self.item_nodes] for b in nodes], axis=0).astype(float)
        return rank_by(-hops, self.Xd @ qd), land


def sala_tau(tree, eps=1.0):
    """Sala et al. (2018) step for (1+eps) distortion: tau = (1+eps)/eps * log(4 * deg_max / pi)."""
    deg = max(len(c) + (1 if tree.parent[v] >= 0 else 0) for v, c in enumerate(tree.children))
    return (1 + eps) / eps * math.log(4 * deg / math.pi)


def order_agreement(tree, pos, n_anchors=200, seed=13):
    """Share of leaf triples (x, y, z) with hops(x,y) < hops(x,z) where also d_P(x,y) < d_P(x,z): 1.0 means the
    disk preserves every tree-distance order the ring ranking relies on (positive control)."""
    rng = np.random.default_rng(seed)
    leaves = np.array(tree.leaves())
    agree = total = 0
    for x in rng.choice(leaves, min(n_anchors, len(leaves)), replace=False):
        h = np.array(tree.hops_from(int(x)))[leaves]
        d = poincare(pos[x], pos[leaves])
        for hv in sorted(set(h[h > 0])):
            nearer, farther = d[h == hv], d[h > hv]
            if len(farther):
                agree += \
                    int(np.sum(nearer[:, None] < farther[None, :]))
                total += len(nearer) * len(farther)
    return agree / total if total else float("nan")


def dist_fidelity(tree, pos, n_pairs=4000, seed=13):
    """Spearman rho between Poincare distance and tree hop distance over random leaf pairs (positive control)."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    leaves = tree.leaves()
    a = rng.choice(leaves, n_pairs)
    b = rng.choice(leaves, n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    hops = {}
    hd = []
    for x, y in zip(a, b):
        if x not in hops:
            hops[x] = tree.hops_from(x)
        hd.append(hops[x][y])
    pd = poincare(pos[a], pos[b])
    return float(spearmanr(pd, hd).statistic), int(len(hd))


# ------------------------------------------------------------------ metrics

def first_rank(ranking, gold_idx):
    pos = {int(i): r for r, i in enumerate(ranking)}
    return min(pos[g] for g in gold_idx) + 1 if gold_idx else None


def recall_at(ranking, gold_idx, k):
    top = set(int(i) for i in ranking[:k])
    return sum(g in top for g in gold_idx) / len(gold_idx)
