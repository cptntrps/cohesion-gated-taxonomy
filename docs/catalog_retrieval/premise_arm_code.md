# catalog-retrieval-hyperbolic-01: premise check, no arm reads the test labels

Premise: no arm can see the gold of the task it ranks.

Checks (code read, experiments/catalog_retrieval):
- llm_arm.py: the accepted ids are the catalog ids in the system prompt (parse(res, pos) with pos from catalog.jsonl),
  never the gold; an id not in the catalog is dropped. The prompt holds the catalog and the task text only.
- run_arms.py popularity_scores: counts gold only for tasks with split == "dev".
- bm25, flat, flat_pca, hyp_tree, tree_cos: inputs are the item texts, the item embeddings and the query; the tree and
  the PCA are fit on the catalog only.
- score.py reads the gold only to score rows after the rankings exist.

PREMISE: supported
