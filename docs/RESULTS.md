# Results

Every number below is produced by a committed script in this repo. Negative results are
included at equal weight — several of them killed our own hypotheses.

Corpora: **LEDGAR** (60,000 contract clauses, 100 provision types), **20 Newsgroups**
(6 super-groups → 20 groups), **Amazon products** (category hierarchy + attributes).
Embedder: `nomic-embed-text` (768d, local). Namer/extractor: `deepseek-v4-flash`.

---

## 1. Extraction — the headline

LEDGAR governing-law → US state. Ground truth: regex on the clause text (the state is
literally written). Paired sample n=400, 4000-resample paired bootstrap.
Script: `hybrid_paired_test.py`.

| method | accuracy | LLM calls | vs read-everything | 95% CI | significant |
|---|---|---|---|---|---|
| read-everything (per item) | 96.5% | 400 | — | — | — |
| cluster-first only | 92.2% | 0 (+40 naming) | — | — | — |
| **cohesion-gated hybrid (20% read)** | **96.2%** | **80** | −0.2 | [−2.5, +2.0] | **no** |
| cohesion-gated hybrid (50% read) | 97.2% | 202 | +0.8 | [−1.0, +2.5] | **no** |

**Claim: matches read-everything at 5× fewer calls.** Not "beats" — the difference is
not statistically distinguishable from zero. An earlier frontier appeared to beat
exhaustive; that was an artifact of *assuming* read items correct rather than measuring
them, and is retired.

## 2. The gold-free confidence signal

Script: `purity_probe.py`. 83 leaves, signals computed with no labels, correlated
against gold purity afterwards.

| signal | Spearman vs purity |
|---|---|
| **cohesion** (mean cosine to centroid) | **+0.74** |
| hyperbolic geodesic misfit | −0.73 |
| dispersion (p90 distance) | −0.71 |
| Poincaré radius spread | −0.57 |
| log cluster size | −0.55 |
| bimodality gap | −0.48 |

Multivariate OLS R² = 0.61. Operationally: high-cohesion clusters are **98.0%** accurate
vs **79.2%** for low-cohesion — the gate separates the errors without labels.

Note: cohesion (+0.74) is at least as strong as the hyperbolic misfit (−0.73), so the
hyperbolic anomaly path is not carrying the signal. Published internal-vs-external
validity correlations of r≈0.65–0.66 bracket our 0.74; the *concept* is not new.

## 3. Gate ablation at equal budget

All three gates cost **zero** model calls to compute.

| items read | cohesion | item-margin | random |
|---|---|---|---|
| 100 | **94.7%** | 93.6% | 92.4% |
| 250 | **96.7%** | 95.5% | 93.0% |
| 500 | 97.2% | **97.7%** | 93.7% |
| 1000 | 98.6% | **99.4%** | 95.3% |

Cohesion wins at low budget, item-level margin at high budget, both beat random.
A confidence-cascade gate (FrugalGPT-style) needs one model call **per item** — a
2,475-call floor here — so the zero-call gates operate below where it can run at all.

## 4. Human effort

Script: `hier_validate.py` (20 Newsgroups, 6,000 posts). Manual alternative = 6,000
labeling decisions.

| human actions (cluster namings) | items organized (purity, per-item) | NMI | vs manual |
|---|---|---|---|
| 6 | 28.7% | 0.47 | 1000× fewer |
| 12 | 47.7% | 0.55 | 500× fewer |
| **20** | **57.3%** | **0.57** | **300× fewer** |
| 40 | 60.7% | 0.55 | 150× fewer |
| 320 | 68.6% | 0.49 | 19× fewer |

Purity, NMI and AMI are **distinct metrics** — purity is a per-item percentage, NMI/AMI
are information-overlap scores. At k=20: purity 57.3%, NMI 0.572, **AMI 0.567**
(chance-corrected). The peak at k=20 = the true group count survives chance correction;
splits past it are sub-groupings.

## 5. Cross-domain generality

| domain | task | result |
|---|---|---|
| Legal (LEDGAR, 100 types) | MAP | purity ~8–10× majority baseline |
| News (20NG, 20 groups) | MAP | purity 57.3%, NMI 0.572, AMI 0.567 (baseline 5.7%) |
| News (20NG, 6 supers) | EMERGE | purity 63.5%, 4/6 supers distinctly captured |
| E-commerce (Amazon, 5 cats) | MAP | purity 68.9%, NMI 0.47 (baseline 20%) |

Same map-strong / emerge-partial pattern in all three — the method is not domain-specific.

## 6. MAP vs EMERGE

Script: `compare_variants.py`. Judged by the metric appropriate to each mode.

| variant | leaves | purity (MAP metric) | false-merge % (EMERGE metric) |
|---|---|---|---|
| map baseline | 100 | 41.6 | 66.2 |
| map + refine | 112 | **46.8** | 74.7 |
| emerge | 600 | 29.9 | **15.9** |
| emerge + refine | 856 | 37.0 | 20.8 |

Gold-agreement is the wrong metric for EMERGE: the system merged `rec.autos` +
`rec.motorcycles` and `mac.hardware` + `ibm.pc.hardware` — arguably a *better* taxonomy
than the reference, scored as error.

Cohesion refinement (greedy) adds **+5.2** purity in MAP mode; the annealed
random-walk variant *hurt* (−2.7) and was not shipped.

## 7. Naming

Script: `naming_conciseness.py`. 80 leaves named two ways.

| metric | verbose prompt | minimal prompt |
|---|---|---|
| avg words per name | 3.06 | **1.40** |
| names containing "and" | 35% | **2%** |
| closeness to canonical gold term | 0.766 | **0.886** |
| duplicate names per repeated concept | 2.29 | **1.35** |

"Benefits and Reimbursement" → "Benefits". Fewer words scored *better*, not merely
shorter. A follow-up (`naming_trypart.py`) tested resolving "A and B" compounds by
picking the part closest to the cluster centroid — it **lost** (0.682 vs 0.733) and was
not shipped.

## 8. Robustness vs regex

On a field with a clean regex pattern, regex **silently dropped 519 of 3,167** clauses
(16%) that plainly named a state — case/format brittleness (ALL-CAPS "STATE OF NEW
YORK"). Only 173 (5%) genuinely name no jurisdiction. The embedding method is
surface-robust and makes the genuinely-undefined items visible as their own cluster
instead of returning nothing.

## 9. Negative results

| hypothesis | outcome | script |
|---|---|---|
| Hyperbolic clusters better | **FALSE, 6×** — incl. natively-trained Riemannian: NMI 0.19 vs Euclidean 0.47 | `native_hyperbolic.py` |
| Crystallization feedback refines | **FALSE** — degrades monotonically 0.522 → 0.284 → 0.227 → 0.172 | `crystallize_loop.py` |
| Hyperbolic stores hierarchy at scale | **RETRACTED — was a measurement artifact.** See §11. | `tree_scale_fixed.py` |
| Whitening improves clustering | **FALSE** — 68.9% → 45.4%; top PCA components *are* the signal | `amz_geometry.py` |
| Mean-centering fixes anisotropy free | **TRUE** — cat-pair cosine 0.91 → −0.24, clustering unchanged | `amz_geometry.py` |
| Removing a concept direction disentangles | **WEAK** — needs mean-centering first; effect small | `amz_concept_erase.py` |
| Cluster-first works on incidental attributes | **FALSE** — periodicity 33.9% vs 80.1% majority baseline | `periodicity_extract.py` |
| Annealed refine beats greedy | **FALSE** — −2.7 purity | `compare_variants.py` |
| Compound names resolvable by centroid | **FALSE** — 0.682 vs 0.733 | `naming_trypart.py` |

The hyperbolic clustering result contradicts HypHC (NeurIPS 2020) and gHHC (KDD 2019)
on these corpora. Treat it as a replication failure with a named target, not a
refutation — the comparison did not match their tuning budgets or protocols.

## 10. Known limitations

- The regex ground-truth oracle is itself imperfect (it dropped 16% of resolvable
  clauses). Headline extraction accuracy should be re-confirmed by blind manual
  adjudication.
- LEDGAR clauses are standalone; the contract-level rollup is untested.
- EMERGE has no validated gold-free quality metric — the open problem in §10 of the PRD.
- Cross-domain runs used balanced subsets (2,500–6,000 items), not full corpora.


---

## 11. Correction — the retracted hyperbolic storage claim (2026-08-26)

**What we published:** hyperbolic embeddings store a large hierarchy without collapse,
tree-distance correlation flat at 0.42–0.46 from 20k→600k nodes at dim 8, vs Euclidean
0.14–0.21 (`tree_scale.py`).

**The defect.** Evaluation pairs were sampled uniformly at random
(`pa = rng.integers(0, N, 800)`). In a broad tree, two random nodes almost always meet
only at the root, so the target tree-distance was nearly constant — in one diagnostic,
643 of 800 pairs shared the identical distance (std 0.89). Spearman correlation against
a near-constant target is not meaningful. The same defect first surfaced on the real
Amazon run, where **both** geometries scored ≈0 (−0.018 / −0.033) — a broken
measurement, not a result.

**The fix.** Stratified pair sampling (one partner drawn per ancestor level, spanning
the full distance range) plus a validity gate that refuses to report any correlation
when the target has <4 distinct values or std <0.5.

**Corrected results** (`tree_scale_fixed.py`, `amazon_scale_fixed.py`):

| hierarchy | nodes | Euclidean | hyperbolic |
|---|---|---|---|
| synthetic k-ary (b=5) | 200,000 | **0.588** | 0.318 |
| synthetic k-ary (b=5) | 600,000 | **0.501** | 0.230 |
| real discovered, wide/shallow (8×5) | 234,828 | **0.840** | 0.082 |
| real discovered, narrow/deep (3×11) | 283,792 | **0.832** | 0.067 |

**Ruled out** as alternative explanations: training budget (re-ran at 5× — synthetic
numbers unchanged) and tree shape (tested wide/shallow and narrow/deep on real data —
same verdict).

**Standing conclusion.** Euclidean preserves tree distance better than hyperbolic at
dim 8 on every hierarchy we tested, synthetic and real. Hyperbolic now has **no**
measured advantage anywhere in this project — 7 independent losses. The honest nuance:
under the original far-pairs-only sampling hyperbolic scored higher, so it may preserve
*maximally distant* relations better; preserving the full distance range is the correct
test for storage, and it loses that.

**Scope.** This does not replicate or refute published hyperbolic-embedding results
(e.g. Sala et al. ICML 2018), which use different dimensions, losses and metrics
(Hits@10, depth–radius correlation, MAP). It retracts *our* measurement only.


---

## 12. SECOND CORRECTION — our hyperbolic arm fails its positive control (2026-08-26)

§11 retracted the hyperbolic storage claim and concluded "Euclidean wins everywhere".
**That conclusion is now also withdrawn.** It was produced by a hyperbolic
implementation that does not pass a benchmark with a known answer.

**The control.** WordNet nouns (82,115 synsets, depth 18) are the canonical benchmark
where hyperbolic embeddings are published to win decisively — Nickel & Kiela (NeurIPS
2017) and Sala et al. (ICML 2018, MAP 0.989 in *two* dimensions). Run under our own
protocol, with the transitive closure of the hypernym relation (603,757 edges,
matching the published setup):

| metric | Euclidean | hyperbolic | published expectation |
|---|---|---|---|
| MRR, parent reconstruction (published-style) | **0.756** | 0.391 | hyperbolic ≈0.98 |
| tree-distance Spearman (our metric) | **0.610** | −0.351 | — |

Hyperbolic loses on **both** metrics, including the one the literature uses. Our
hyperbolic number is far below published values while our Euclidean number is
reasonable. The most likely cause is optimizer configuration — RiemannianAdam at
lr=0.05 with no burn-in phase, saturating points against the ball boundary in float32.
Published protocols use burn-in (10 epochs at one-tenth the learning rate) and tuned
Riemannian SGD.

**Standing position on hyperbolic in this repo: NO VERDICT.**
- The original "hyperbolic stores hierarchies without collapse" claim: **withdrawn**
  (degenerate evaluation sampling, §11).
- The replacement "Euclidean wins everywhere" claim: **also withdrawn** (the hyperbolic
  arm fails its positive control).
- We are not asserting that hyperbolic is better *or* worse. We are asserting that
  **our apparatus cannot currently measure it**, and we know this because it fails a
  benchmark with a known answer.

Anyone reproducing hyperbolic comparisons from this repo should fix the optimizer
setup and re-establish the WordNet control **before** trusting any hyperbolic number
here. `wordnet_control.py` and `wordnet_metric_check.py` are the control harness.

**Note on the rest of the repo.** The Euclidean-only results — the cohesion gate, the
extraction frontier, naming, MAP/EMERGE, cross-domain generality — do not depend on the
hyperbolic arm and are unaffected by this correction.
