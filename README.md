# Cohesion-gated taxonomy discovery and extraction

Cluster a corpus, let an LLM name only the clusters, and use a **label-free geometric
confidence score to decide where to spend per-item LLM calls**. Measured on legal
clauses, news posts and e-commerce products.

**Prior art, named up front.** Embed → cluster → LLM-names-each-cluster is
[BERTopic](https://arxiv.org/abs/2203.05794) (2022),
[TopicGPT](https://aclanthology.org/2024.naacl-long.164.pdf) (2024),
[TnT-LLM](https://arxiv.org/abs/2403.12173) (2024), Anthropic's
[Clio](https://www.anthropic.com/research/clio) (2024), Nomic Atlas, and Relativity
Analytics (shipping since ~2010). Querying an LLM on cluster representatives and
propagating to members is [arXiv:2607.19704](https://arxiv.org/abs/2607.19704) (2026)
and, in the label setting, Nguyen & Smeulders (ICML 2004). Confidence-gated cascades
are [FrugalGPT](https://arxiv.org/abs/2305.05176) (2023). Seeded/anchored clustering is
Basu et al. (ICML 2002), anchored CorEx, guided BERTopic. Cluster cohesion as a
label-free quality proxy is silhouette (Rousseeuw 1987). Geometry-based budget
allocation is TypiClust (ICML 2022) / ProbCover (NeurIPS 2022). Neighborhood denoising
is Zhu et al. (ICML 2003), Zhou et al. (NeurIPS 2003).

**What we did not find in the literature:** using *whole-cluster cohesion, computed
before any item-level LLM call*, to decide which clusters get exhaustive per-item
inference and which get one cluster-level extraction propagated to members. We say
"we did not find it", not "this is the first".

## Measured results

**Extraction** (LEDGAR governing-law → US state; ground truth = regex on clause text):

| method | accuracy | LLM calls |
|---|---|---|
| read-everything (per-item LLM) | 96.5% | 400 |
| cluster-first only | 92.2% | 0 (+40 naming) |
| **cohesion-gated hybrid** | **96.2%** | **80** |

Paired bootstrap (n=400, 4000 resamples): hybrid − read-everything = **−0.2 points,
95% CI [−2.5, +2.0] — not significant**. The honest claim is **"matches
read-everything at 5× fewer calls"**, never "beats it". An earlier frontier that
appeared to beat exhaustive was an artifact of assuming read items correct; the
`hybrid_paired_test.py` measurement retired that claim.

**Gold-free confidence.** Cluster cohesion predicts gold purity at Spearman **+0.74**
across 83 leaves (multivariate R² 0.61). High-cohesion clusters are 98.0% accurate vs
79.2% for low-cohesion — so cohesion tells you where to spend, with zero model calls.

**Gate ablation** at equal read budget (all gates cost zero model calls):

| items read | cohesion | item-margin | random |
|---|---|---|---|
| 100 | **94.7%** | 93.6% | 92.4% |
| 250 | **96.7%** | 95.5% | 93.0% |
| 500 | 97.2% | **97.7%** | 93.7% |
| 1000 | 98.6% | **99.4%** | 95.3% |

Cohesion wins at low budget; item-level margin wins at high budget. Both beat random.

**Human effort.** Naming 20 clusters organizes 6,000 documents at 57.3% purity
(per-item) — ~300× fewer human decisions than labeling all 6,000. Quality rises with
more namings; NMI/AMI peak at k=20 (the true group count), so splits past the peak are
sub-groupings, not new groups. Note: purity (per-item %), NMI and AMI are distinct
metrics and are reported separately.

**Cross-domain.** Legal (LEDGAR, 100 clause types), news (20 Newsgroups, 6→20), and
e-commerce (Amazon products). Same map-strong pattern in all three.

## Negative results (we falsified our own hypotheses)

- **Hyperbolic loses at clustering, 6 independent times**, including a natively-trained
  Riemannian embedding (geoopt): NMI 0.19 vs Euclidean 0.47. This contradicts
  HypHC (NeurIPS 2020) / gHHC (KDD 2019) on these corpora; treat it as a replication
  failure with a named target, not a refutation.
- **A crystallization feedback loop degrades monotonically**: feeding the discovered
  tree back in and re-embedding drives leaf-NMI 0.522 → 0.284 → 0.227 → 0.172.
- **Hyperbolic does win at storage**: tree-distance correlation stays flat at 0.42–0.46
  from 20k → 600k nodes at dim=8, vs Euclidean 0.14–0.21. This half is already proved
  by [Sala et al. (ICML 2018)](https://proceedings.mlr.press/v80/sala18a.html).
- **Embedding anisotropy** (all category-pair cosines ≈0.91) is fixed for free by
  mean-centering — step one of All-but-the-Top (ICLR 2018) — with no clustering loss.
- **Incidental attributes fail cluster-first extraction**: periodicity (mentioned in
  passing) scored 33.9% vs an 80.1% majority baseline. Cluster-first only works when
  the attribute is the dominant signal of the text.

## Run

```bash
export DEEPSEEK_API_KEY=...       # any OpenAI-compatible chat endpoint; see naming.py
ollama pull nomic-embed-text      # local embeddings, $0, nothing leaves the host
python3 build_tree.py             # discovery tree -> tree.json + members.json
PORT=8799 python3 server.py       # interactive UI: rename/anchor/re-run
# open http://127.0.0.1:8799
```

## Layout

`cluster.py` shared builder (map/emerge modes, cohesion, refine) · `naming.py` LLM
naming · `hyperbolic.py` Poincaré ball · `build_tree.py` batch build · `server.py` +
`index.html` interactive UI · `extraction*.py` field extraction · `hybrid_paired_test.py`
the paired significance test · `purity_probe.py` gold-free signal study ·
`hier_validate.py` / `amazon_validate.py` cross-domain validation ·
`tree_scale.py` 600k-node hyperbolic storage test.

Every script is one committed experiment; the commit message states its finding,
including the ones that failed.
