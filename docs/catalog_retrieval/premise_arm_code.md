# catalog-retrieval-hyperbolic-01: premise check, no arm reads the test labels

Premise: no arm can see the gold of the task it ranks.

Checks (code read, experiments/catalog_retrieval):
- llm_arm.py: the accepted ids are the catalog ids in the system prompt (parse(res, pos) with pos from catalog.jsonl),
  never the gold; an id not in the catalog is dropped. The prompt holds the catalog and the task text only.
- run_arms.py popularity_scores: counts gold only for tasks with split == "dev".
- bm25, flat, flat_pca, hyp_tree, tree_cos: inputs are the item texts, the item embeddings and the query; the tree and
  the PCA are fit on the catalog only.
- score.py reads the gold only to score rows after the rankings exist.

## Amendment v2 checks (2026-11-25, code at the design commit)

- llm_arm.py: the prompt numbers the catalog rows; the accepted vocabulary is the row numbers 0..n-1 of the prompt
  (parse(res, len(cat))), never the gold. The ranking is the picks only. A reply with 0 valid picks is a failed row,
  and every raw reply is stored.
- retrieval.py HypIndex: the tree, both PCAs and the ball positions are fit on the catalog only; the query enters
  through reduce(), descend() and query_point(), which read the query vector and the index only.
- run_arms.py choose(): reads the dev grid only. popularity_scores still counts gold for split == "dev" only.
- score.py reading(): BEST_SIMPLE is a constant fixed from dev; the call reads the gold only through the scored rows.

PREMISE: supported
