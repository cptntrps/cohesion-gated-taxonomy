# Next steps (deferred backlog)

Ordered by value to the product. Owner: double_gui (registry.product cohesion-gated-taxonomy).

## Measured state (2026-08-26)
- Discovery clustering: hyperbolic leaf purity 41.6% vs flat 50.6% vs 5.3% baseline (PASS >=2x). Flat wins purity.
- Anomaly (hyperbolic): surfaces real misassignments; note cohesion (+0.74) matches the
  hyperbolic misfit (-0.73), so the hyperbolic path adds surface area for no measured gain.
- Retrieval productivity (`scorecard.py`): on PARAPHRASE concepts where keyword recall < 90%
  (Governing Law 1.3%, Confidentiality 89.5%, Severability 13.5%), semantic retrieval reaches 90%
  recall at 3/3 pass >=3x, median 7.7x fewer clauses. Keyword-literal concepts: keyword already works.

## Measured state (2026-10-19, demo fixes)
- Naming failures 6 -> 0: empty reasoning-model replies are retried with a larger token budget, then with
  the non-reasoning `deepseek-chat`, then a deterministic heading-based name marked "(auto)" (none needed).
- Top-level imbalance: size-aware k (about n/1500 children, 2..12) and a 12-way root split. Largest branch
  28,261 -> 16,655 clauses; leaf purity 40.0% -> 41.2% (flat 45.4%). Duplicate sibling names are re-asked
  with the taken names listed (1 remains: two "Term" branches).
- Live scorecard in the UI (item 1 below) is DONE: `GET /api/score?q=` and the Find tab.

## Deferred (do in this order)

1. **Live scorecard in the UI.** `/score {name}` endpoint (semantic rank -> clauses-to-90%-recall +
   precision + keyword baseline). A reviewer types a concept, sees the productivity number. The claim
   is currently CLI-only (`scorecard.py`); this makes it demoable. Highest presentation value.

2. **Poincare-disk layout + jump.** [LOW PRIORITY after the 2026-08-26 retraction: hyperbolic has no measured advantage here.] Render the ACTUAL 2-D Poincare disk (the tree already lives in the
   ball, `hyperbolic.make_ball`) instead of the force graph: radius = depth/generality, geodesic edges.
   "Jump" = click a node, travel the geodesic to concept-neighbors across branches. Presentation, not a
   new claim. This is the dimensionality-reduction strength made visible (many Euclidean dims -> 2-D).

3. **Fix hyperbolic top-level imbalance.** Discovery produces one ~24k blob ("Entire Agreement").
   Balanced / spherical k-means, or split-largest-until-even, would lift hyperbolic purity toward flat
   and make navigation-retrieval viable (it is not, currently: navigation scored worse than semantic rank).

4. **Trained hyperbolic (the spec's density spike).** The 41.6<50.6 loss is UNSUPERVISED (PCA radius,
   no hierarchy). Owner intent: engineer a DENSE clause-similarity graph and train Poincare positions
   (radius = real hierarchy depth) with a RANK-based frame regularizer (MSE already failed, Spearman 0.42).
   Only then does curvature have a shot at purity. Precondition to measure: clause-graph average degree.

5. **Persist flat-vs-hyperbolic purity across interactive re-runs.** Server `rebuild()` drops the flat
   comparison number (only `build_tree.py` computes it). Carry it forward in tree.json.

6. **Reduce naming failures.** ~4/100 clusters return `[naming failed]` (DeepSeek reasoning model empty
   content). Shown loudly, never backfilled. Retry-with-higher-token-budget or a second model would cut it.
