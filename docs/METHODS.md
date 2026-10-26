# Methods — how to reproduce

## Setup

```bash
pip install numpy scikit-learn scipy datasets torch geoopt
ollama pull nomic-embed-text                  # local embeddings, $0
export DEEPSEEK_API_KEY=...                   # any OpenAI-compatible chat endpoint
export COHESION_TAXONOMY_DATA=/path/to/data   # ledgar_emb.npz, tree.json, members.json live here (cohesion_taxonomy/paths.py); LEDGAR_EMB overrides the embedding file alone
```

Datasets load from HuggingFace on first use: `MAdAiLab/lex_glue_ledgar`,
`ckandemir/amazon-products`; 20 Newsgroups via scikit-learn. Embedding a 60k corpus
takes ~20–30 min once, then it is cached.

## Reproducing each result

| result | command | runtime |
|---|---|---|
| discovery tree + UI | `python -m cohesion_taxonomy.build_tree && python -m cohesion_taxonomy.server` | ~3 min + serve |
| headline extraction claim | `python experiments/hybrid_paired_test.py` | ~2 min (400 LLM calls) |
| cluster-first vs read-all | `python -m cohesion_taxonomy.extraction_validate` | ~3 min (190 calls) |
| gold-free signal study | `python experiments/purity_probe.py` | ~1 min (no LLM) |
| MAP vs EMERGE | `python experiments/compare_variants.py` | ~8 min (no LLM) |
| naming conciseness | `python experiments/naming_conciseness.py` | ~2 min (160 calls) |
| cross-domain: news | `python experiments/hier_validate.py` | ~4 min |
| cross-domain: products | `python experiments/amazon_validate.py` | ~2 min |
| geometry / anisotropy | `python experiments/amz_geometry.py` | ~1 min |
| native hyperbolic | `python experiments/native_hyperbolic.py` | ~3 min (GPU) |
| crystallization loop | `python experiments/crystallize_loop.py` | ~6 min (GPU) |
| 600k-node storage test | `python experiments/tree_scale.py` | ~7 min (GPU) |

Seeds are fixed (`random_state=13`) throughout, so runs are deterministic apart from
LLM sampling (temperature 0, but the provider is not bit-deterministic).

## Measurement rules we hold to

1. **Gold labels never touch clustering or naming.** They are used only to score,
   after the fact. Anchor names are embedded from the *name string*, not from labels.
2. **Declare the falsifiable goal before running.** Each script's docstring states the
   success and failure condition up front; results are reported against it even when
   they fail.
3. **Report purity, NMI and AMI separately.** Purity is a per-item percentage; NMI/AMI
   are information-overlap scores. Never present NMI as "% of items".
4. **Chance-correct.** AMI/ARI alongside NMI; random-assignment floors alongside
   majority-class baselines.
5. **Significance-test any accuracy comparison.** Paired bootstrap on per-item
   differences; if the CI includes zero, the honest wording is "matches", not "beats".
6. **Never assume an LLM read is correct.** Measure it. The retired "beats exhaustive"
   claim came from exactly that assumption.
7. **Baselines must be fair.** Majority-class is the weakest baseline; also report the
   random-at-same-k floor, and a supervised or keyword reference where one exists.

## Metric definitions

- **purity** — fraction of items whose gold label equals their cluster's majority label,
  size-weighted. Per-item percentage.
- **NMI / AMI** — normalized / adjusted mutual information between clustering and gold.
  AMI is chance-corrected. Not percentages of items.
- **cohesion** — mean cosine of a cluster's members to its unit centroid. Label-free.
- **multimodality** — cosine silhouette of the best 2-means split of a cluster. High
  means a real seam (should split); low means coherent (stop, even if coarse).
- **false-merge %** — share of leaf items sitting in a leaf that still has a clean
  2-way seam (silhouette ≥ 0.10). The EMERGE failure metric.
- **tree-distance correlation** — Spearman between embedded distance and true tree-hop
  distance on sampled pairs. The "no collapse" metric for hierarchy storage.

## Cost accounting

LLM calls are counted explicitly in each script. "Read-everything" means one call per
item; "cluster-first" means one call per cluster; "hybrid" means k naming calls plus
per-item reads for gated clusters only. Embedding is local and excluded from LLM cost
(it is $0 and runs once).
