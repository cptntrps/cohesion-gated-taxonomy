# Architecture

## The two-stage shape

```
                 ┌──────────────── O(N), machine, $0 ────────────────┐
  documents ──►  chunk ──► embed (local) ──► cluster ──► cohesion score
                 └───────────────────────────────────────────────────┘
                                    │
                 ┌──────────── O(k), LLM ────────────┐
                 │  name each cluster (k calls)      │
                 │  extract value per sub-cluster    │
                 └───────────────────────────────────┘
                                    │
                 ┌──── O(uncertain residual), LLM ───┐
                 │  per-item read ONLY where         │
                 │  cohesion says "don't trust"      │
                 └───────────────────────────────────┘
                                    │
                 ┌──────────── O(k), human ──────────┐
                 │  rename / add anchor → re-cluster │
                 │  adjudicate flagged items         │
                 └───────────────────────────────────┘
```

The invariant: **nothing that scales with N is paid for by a human or an LLM.**

## Modules

| file | role |
|---|---|
| `cluster.py` | shared builder. Divisive clustering, MAP/EMERGE modes, `cohesion()`, `multimodality()`, `refine_partition()`, anchored rebuild |
| `naming.py` | the LLM's only job: shortest canonical name for a cluster, given its ancestor chain |
| `hyperbolic.py` | Poincaré ball: `make_ball`, `hyp_kmeans`, `poincare_dist_matrix` |
| `build_tree.py` | batch build → `tree.json` + `members.json` |
| `server.py` | stateful interactive server: `/rename`, `/anchor`, `/unanchor`, `/rerun`, `/reset` |
| `index.html` | React (CDN) + 3d-force-graph UI: table view, graph view, novelty panel |
| `extraction*.py` | cluster-first field extraction and its validation |
| `hybrid_paired_test.py` | the paired bootstrap that governs the headline claim |

## Key design decisions and why

### Clustering is Euclidean, not hyperbolic
Hyperbolic lost at clustering in 6 independent tests including a natively-trained
Riemannian embedding (NMI 0.19 vs 0.47). Hyperbolic is retained only for
scale-resilient storage of an *already-discovered* hierarchy, where it holds
tree-distance fidelity flat from 20k → 600k nodes at dim 8.

### Mean-centering, not whitening
Raw embeddings are anisotropic (all category-pair cosines ≈0.91), which makes concept
direction arithmetic impossible. Mean-centering fixes this for free with no clustering
loss. Blind whitening (dropping top PCA components) **destroys** the signal — those
components *are* the category structure (purity 68.9% → 45.4%).

### Depth from the silhouette peak
Splitting past the point where silhouette stops improving produces **sub-groupings**,
not new groups. The silhouette peak coincides with the NMI/AMI peak at the true group
count, giving a label-free depth rule.

### EMERGE splits only at a seam
MAP splits to a fixed depth to maximize purity. EMERGE splits only where a cluster is
genuinely multimodal (2-means cosine silhouette above a threshold) and stops when
unimodal, however coarse. This cut false merges from 66% to 15.9%.

### Anchors are structural
A human rename is not a relabel: the name is embedded and becomes an attractor, and the
**entire corpus re-clusters**. Items may stay, move to another anchor, or fall back to
discovery. This is deliberate — the alternative (freeze membership, change the string)
is available but is a different operation.

### Concept → attribute → value
Sub-structure is one of three kinds, and only one of them is a clustering problem:

| kind | example | handled by |
|---|---|---|
| categorical value | Governing Law → Delaware | sub-clustering + one LLM call per sub-cluster |
| continuous value | Base Salary → $350,000 | per-item extraction (embeddings do not encode magnitude) |
| extra provision | + forum-selection, + annual-review | multi-label tag |

Clustering surfaces the *provision* axis strongly, the *categorical value* axis weakly
(boilerplate dominates the embedding), and the *continuous* axis not at all.

## Data flow

`tree.json` — nodes (id, name, depth, parent, count, cohesion, purity, samples,
`human_named`), links, report, anchors, anomalies.
`members.json` — node id → item indices, so any subtree can be re-clustered without
re-embedding.
