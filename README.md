# Cohesion-gated taxonomy discovery and extraction

Cluster a corpus, let an LLM name only the clusters, and use a **label-free geometric
confidence score to decide where to spend per-item LLM calls**. Measured on legal
clauses, news posts and e-commerce products.

📄 [PRD](docs/PRD.md) · 🏗 [Architecture](docs/ARCHITECTURE.md) · 📊 [Results](docs/RESULTS.md) · 🔬 [Methods / reproduce](docs/METHODS.md) · 🗺 [Next steps](docs/NEXT_STEPS.md)

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
- **Hyperbolic: our gradient training was broken; the geometry is fine.** Every negative
  hyperbolic result we produced came from gradient-descent Poincaré training, which
  [Sala et al. (ICML 2018)](https://proceedings.mlr.press/v80/sala18a/sala18a.pdf) show
  is unreliable (poor local minima). We reproduced that failure exactly: 88.6% of points
  pinned at the ball's projection ceiling and radius↔depth ρ = 0.025 while the loss fell
  steadily. Replacing the optimizer with **Sarkar's combinatorial construction**, on
  WordNet (82,115 synsets): radius↔depth **+0.774**, tree-distance **+0.864 in TWO
  dimensions** — beating our gradient-trained Euclidean at 8 dimensions (+0.610).
  **All hyperbolic-negative results in this repo measure our optimizer, not the geometry,
  and should not be cited as evidence about hyperbolic embeddings.**
  See [docs/RESULTS.md §11–§13](docs/RESULTS.md). Euclidean-only results are unaffected.
- **Embedding anisotropy** (all category-pair cosines ≈0.91) is fixed for free by
  mean-centering — step one of All-but-the-Top (ICLR 2018) — with no clustering loss.
- **Incidental attributes fail cluster-first extraction**: periodicity (mentioned in
  passing) scored 33.9% vs an 80.1% majority baseline. Cluster-first only works when
  the attribute is the dominant signal of the text.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -e .
export NAMING_API_KEY=...            # any OpenAI-compatible chat endpoint; NAMING_API_BASE defaults to DeepSeek (see cohesion_taxonomy/naming.py)
ollama pull nomic-embed-text         # local embeddings, $0, nothing leaves the host
.venv/bin/python -m cohesion_taxonomy.build_tree    # discovery tree -> <data dir>/tree.json + members.json
.venv/bin/python -m cohesion_taxonomy.server        # interactive UI on http://127.0.0.1:8580
.venv/bin/python -m pytest -q                       # hermetic suite: no model, no corpus
```

Runtime state never lives in the repo: `cohesion_taxonomy/paths.py` resolves the data
directory (`COHESION_TAXONOMY_DATA`, default `./data`, gitignored) for the embeddings, the
built tree and the extraction result. `GET /healthz` reports what the server loaded and
whether the served tree covers the corpus; it is the deploy gate (`deploy.sh`).

On aurora the app runs as the user unit `cohesion-gated-taxonomy.service` (loopback :8580,
data in `~/data/cohesion-gated-taxonomy`, naming through the livingos-api gateway's
neutral DeepSeek lane with its own scoped key from the sops store
`services/cohesion-gated-taxonomy.enc.env`): `systemctl --user status cohesion-gated-taxonomy`.
Product register: `registry.product` slug `cohesion-gated-taxonomy`, steward double_gui.

The UI (2026-10-18) follows [DISCOVERY_FLOW.md](docs/DISCOVERY_FLOW.md): **1 Discover** (the named tree with a
coherence meter per concept, benchmark labels hidden behind a toggle), **2 Correct** (name a concept, rename a
group, seed one from a clause that fits nowhere; every change re-clusters the corpus), **3 Find** (type a concept,
get the clauses ranked by meaning and the live scorecard: clauses to read to reach 90% recall by meaning vs by
keyword, `GET /api/score?q=...`), **4 Extract** (the field-extraction demo). A **How it works** tab explains the mechanism in plain language for a non-technical reader, with its numbers read live from `tree.json` and `extraction.json`; `#how` (or `#discover`, `#correct`, `#find`) opens a tab directly. The previous research console is
kept as `web/index_v1.html`.

## Layout

`cohesion_taxonomy/` is the product: `cluster.py` shared builder (map/emerge modes,
cohesion, refine, anchored rebuild) · `naming.py` LLM naming · `hyperbolic.py` Poincaré
ball · `build_tree.py` batch build · `build_extraction.py` + `extraction_validate.py` +
`extraction_ground.py` field extraction · `server.py` the interactive server · `paths.py`
where state lives. `web/` is the UI (`index.html`, `extraction.html`, the earlier research
console `index_v1.html`). `experiments/` holds every research script and its result file
(`hybrid_paired_test.py` is the paired significance test, `purity_probe.py` the gold-free
signal study, `hier_validate.py` / `amazon_validate.py` the cross-domain validation,
`tree_scale.py` the 600k-node hyperbolic storage test; see `experiments/README.md`).
`tests/` is the hermetic suite. `paper/` the write-up.

Every experiment script is one committed experiment; the commit message states its finding,
including the ones that failed.
