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

PREMISE: supported
