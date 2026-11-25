"""WORK-2283 retrieval arms over a catalog of skills and owner decisions. Hermetic: numpy and scikit-learn only.

Arms (each returns a full ranking of catalog indices for one query):
  bm25        Okapi BM25 over the item text (name + description head).
  flat        cosine on the full embedding.
  flat_pca    cosine on the PCA-reduced embedding (isolates the dimension reduction).
  hyp_tree    the treatment: a catalog tree, embedded in the Poincare disk by Sarkar's construction (no gradient
              training: the optimizer failure of RESULTS.md §12-13), descended by beam search on PCA-reduced node
              descriptors, then ranked ring by ring by Poincare distance from the landing nodes (the "jumps"),
              ties broken by reduced cosine.
  tree_cos    the same tree and descent, ranked by explicit tree hop distance instead of Poincare distance
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

def sarkar(tree, tau, order=None):
    """Complex positions in the Poincare disk. Root at 0; each node's children evenly spaced around it at
    hyperbolic distance tau, the first opposite the parent (Sala et al., ICML 2018, combinatorial construction).
    order: {node: children in angular order}; default is the tree's insertion order (arbitrary angles)."""
    pos = np.zeros(len(tree), dtype=np.complex128)
    r = math.tanh(tau / 2.0)
    q = deque([0])
    while q:
        u = q.popleft()
        kids = (order or {}).get(u, tree.children[u])
        if not kids:
            continue
        x = pos[u]
        if tree.parent[u] < 0:
            start = 0.0
        else:
            p0 = (pos[tree.parent[u]] - x) / (1 - np.conj(x) * pos[tree.parent[u]])
            start = float(np.angle(p0))
        n = len(kids) + (0 if tree.parent[u] < 0 else 1)
        for j, c in enumerate(kids):
            ang = start + 2 * math.pi * (j + (0 if tree.parent[u] < 0 else 1)) / n
            z = r * np.exp(1j * ang)
            pos[c] = (z + x) / (1 + np.conj(x) * z)
            q.append(c)
    return pos


def poincare(a, b):
    """Poincare distance between complex points (broadcasts)."""
    a, b = np.asarray(a), np.asarray(b)
    num = 2 * np.abs(a - b) ** 2
    den = np.clip((1 - np.abs(a) ** 2) * (1 - np.abs(b) ** 2), 1e-300, None)
    return np.arccosh(1 + num / den)


# ------------------------------------------------------------------ the hierarchical index

def seriation(tree, desc):
    """Children of each node in the order of their projection on the first principal axis of their descriptors,
    so neighbouring angles in the disk hold similar content."""
    order = {}
    for v, kids in enumerate(tree.children):
        if len(kids) >= 3:
            X = desc[kids] - desc[kids].mean(0)
            axis = np.linalg.svd(X, full_matrices=False)[2][0]
            order[v] = [kids[i] for i in np.argsort(X @ axis, kind="stable")]
    return order


class HypIndex:
    """round_dp: Poincare distances are rounded to this many decimals. Near the disk boundary float64 leaves
    ~1e-4 noise in distances that are exactly equal in theory (all children of one node), and rings are several
    units apart, so rounding restores the exact ties without merging rings."""

    def __init__(self, tree, V, dim=32, tau=1.5, beam=2, seed=13, seriate=True, round_dp=2):
        self.tree, self.beam, self.round_dp = tree, beam, round_dp
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
        self.pos = sarkar(tree, tau, seriation(tree, self.desc) if seriate else None)
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
        return beam

    def rank(self, qv, metric="poincare"):
        qd = self.reduce(qv)
        land = self.descend(qd)
        if metric == "poincare":
            d = np.min([poincare(self.pos[b], self.pos[self.item_nodes]) for b in land], axis=0)
            d = np.round(d, self.round_dp)
        else:
            hops = [np.array(self.tree.hops_from(b))[self.item_nodes] for b in land]
            d = np.min(hops, axis=0).astype(float)
        return rank_by(-d, self.Xd @ qd), land


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
