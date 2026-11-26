# catalog-retrieval-hyperbolic-01: premise check, the hyperbolic embedding encodes the catalog tree

Premise: the Poincare-disk positions of the treatment preserve the catalog tree, so ranking by Poincare distance
means ranking by tree position. Without this check a negative result could be an encoding failure (the gradient
optimizer failure of cohesion-gated-taxonomy RESULTS.md sections 12-13), not evidence about the method.

Construction: Sarkar's combinatorial construction (no optimization), step tau from Sala et al. (2018) for (1+eps)
distortion with eps = 1: tau = 2 log(4 deg_max / pi) = 6.921 on this tree. Siblings in angular order of their first
principal axis (seriation), so neighbouring angles hold similar content.

Tree (frozen catalog, 987 items): 1090 nodes, depth 4: root -> kind (skill, decision) -> group (14 Ward clusters of
skills; 22 real decision domains) -> subgroup (Ward split of groups above 24 items) -> item.

Checks (data):
- Order agreement (share of leaf triples x, y, z with hops(x,y) < hops(x,z) and d_P(x,y) < d_P(x,z), 200 anchors):
  0.9954.
- Float64 near the disk boundary (1 - |x| down to 2e-12) leaves up to 1.67e-4 noise in distances that are equal
  in theory (all children of one node at distance tau). Rounded to 2 decimals: spread 0.00, so ties are exact again.
- Smallest gap between the first ring (children of a landing node) and the next ring: 5.67, so rounding to 0.01
  never merges rings.
- Hermetic tests (tests/test_catalog_retrieval.py): children at distance tau, order agreement 1.0 on a synthetic
  tree at the Sala tau and below 1.0 at tau 1.5, the index lands in the query's cluster.

## Amendment v2 checks: the ball of treatment v2 at the frozen configuration (2026-11-25, catalog only)

The v1 disk (2-d, seriated siblings, distances rounded to 2 decimals) is replaced. Premise for v2: the ball positions
keep the catalog tree, so the Poincare distance to the inserted query point orders items by tree position first and
by semantic direction inside a group. Without this check a negative result could be an encoding failure.

Construction (retrieval.py build_ball): root at 0; each child at hyperbolic distance tau from its parent by Mobius
addition, in the child's semantic direction (descriptor offset from the parent, pushed away from the grandparent).
No optimisation. Frozen configuration: dim 256, ball_dim 256, tau 1.5, beam 24, one landing node.

Checks (data; the script reads the catalog and the item embeddings only):
- Tree: 1090 nodes, depth 4, 103 internal nodes (median 9 children, largest 24). 88 leaf-parent nodes are the
  landing candidates; they hold 3 to 24 items (median 10).
- Every child is at tau from its parent: largest |d(parent, child) - tau| = 3.1e-14.
- Largest norm 0.98939 (1 - norm = 0.0106), so every position fits the float32 vectors of pg_hyperbolic (the bar is
  1 - 1e-7). Casting the positions to float32 changes a leaf-pair distance by at most 4.9e-06 (3000 pairs).
- Order agreement (share of leaf triples x, y, z with hops(x,y) < hops(x,z) and d_P(x,y) < d_P(x,z), 200 anchors):
  1.0. Spearman rho between Poincare distance and hop distance: 0.954 (3993 leaf pairs).
- The geometry orders siblings: distances between items under one landing node run from 0.38 to 2.58 (2 tau = 3.0),
  so the ranking inside a group is not a tie.
- pg_hyperbolic cross-check on 40 real leaf pairs (float32 positions): poincare_distance agrees to a relative
  difference of 5.3e-15. The hermetic suite also checks the Mobius identities, the distance formula and that the
  ranking equals the distance order (tests/test_catalog_retrieval.py; the database cross-check skips without PG_DSN).
- The second PCA keeps all the variance at ball_dim = dim (1.0); the descriptor PCA keeps 0.928 of the variance of
  the 1024-d item vectors.

PREMISE: supported
