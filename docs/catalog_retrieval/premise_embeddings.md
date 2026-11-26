# catalog-retrieval-hyperbolic-01: premise check, bge-m3 embeddings are sound

Premise: the local embedding model (bge-m3 via Ollama on aurora) gives distinct, meaningful vectors for the catalog
and the task texts. Every embedding arm (flat, flat_pca, hyp_tree, tree_cos) depends on it.

Checks (2026-11-25, data, not assumption):
- Distinct capitalised names do not collapse (the nomic-on-Ollama [UNK] failure in memory ollama-embedding-bug):
  cos("Ironclad", "Neon") = 0.458, not 1.0.
- A paraphrase pair about a failed Home Assistant automation: cos 0.858; the same text against a bread recipe: 0.344.
- A Portuguese and an English sentence with one meaning: cos 0.780 (nomic-embed-text: 0.442). Owner texts mix both
  languages, so bge-m3 is the model for all embedding arms.
- Catalog (987 items): 0 vector pairs with cos > 0.9999; largest off-diagonal cos 0.9993 (two near-identical
  decision texts, not a collapse).
- Nearest neighbours of three public skills are on topic: leak-triage -> three security leak decisions;
  ha-automations -> homeassistant-yaml-dry-verifier, ha-connect, ha-external-triggers; scientific-method -> the
  scientific-method-enforced decision, analyze, grounded-citations.
- One query embedding (warm, single text): median 338.5 ms, p95 391.7 ms (20 dev queries). This cost is the same for
  every embedding arm and is reported separately from the ranking latency.

## Amendment v2 checks (2026-11-25, before any test-split run; review finding F11)

- Model identity: Ollama tag bge-m3:latest has digest
  7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab. run_arms.py writes this digest to the meta
  file of every run.
- The cached item vectors come from this digest: a fresh embedding of all 987 item texts equals emb_items.npz
  (largest absolute difference 0.0, smallest row cosine 1.0). The cache key is the sha256 of the item texts.
- Query embedding latency measured again on 20 dev texts of key v3: median 394 ms, p95 637 ms (the GPU was shared).
  score.py keeps the first value (338.5 ms) as the shared embedding cost of every embedding arm; BM25 pays none.

PREMISE: supported
