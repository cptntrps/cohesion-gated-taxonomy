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

PREMISE: supported
