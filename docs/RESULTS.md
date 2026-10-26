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


---

## 13. THIRD CORRECTION — hyperbolic works; our OPTIMIZER was the problem (2026-08-27)

§12 said "no verdict — our apparatus cannot measure hyperbolic". We can now measure it,
and **hyperbolic wins**. The fault was never the geometry; it was using gradient descent
at all.

**The pointer.** [Sala et al., ICML 2018](https://proceedings.mlr.press/v80/sala18a/sala18a.pdf)
abandon SGD for hyperbolic embeddings *precisely because* it suffers poor local minima
and severe initialization sensitivity, and instead use a deterministic **combinatorial
construction** (Sarkar's). Our gradient failure reproduced their documented failure mode
exactly: 88.6% of points pinned at the projection ceiling, radius↔depth ρ = 0.025, loss
falling the whole time.

**Sarkar's construction** (`sarkar_construction.py`, no optimization at all): root at the
origin; children placed at even angular intervals at hyperbolic step τ, positioned via
Möbius isometries that carry each parent to the origin and back.

**WordNet nouns, 82,115 synsets, τ=3.0, TWO dimensions:**

| method | dims | radius↔depth | tree-distance ρ |
|---|---|---|---|
| gradient-trained hyperbolic (ours) | 8 | +0.025 | −0.351 |
| gradient-trained Euclidean (ours) | 8 | — | +0.610 |
| **Sarkar-constructed hyperbolic** | **2** | **+0.774** | **+0.864** |

**Hyperbolic in 2 dimensions beats Euclidean in 8** — the dimension-efficiency result the
literature reports, reproduced here once the construction replaces the optimizer.

**τ sweep** (40k subset) shows the expected curve, including the precision wall Sala et al.
flag: τ=0.4 → 0.479, τ=0.8 → 0.557, τ=1.5 → 0.707, **τ=3.0 → 0.880**, τ=5.0 → 0.626
(degrades; 80% of points past r>0.999999, float64 exhausted).

**Standing position on hyperbolic — REVISED AGAIN:**
- Hyperbolic **does** represent hierarchies with far better dimension efficiency, when
  built by combinatorial construction.
- **Gradient-based Poincaré training is unreliable** and was the sole cause of every
  negative hyperbolic result in this repo. Those results measure our optimizer, not the
  geometry, and should not be cited as evidence about hyperbolic embeddings.
- The clustering comparisons (§9, "hyperbolic loses at clustering 6×") used the same
  broken gradient training and are **likewise not evidence about hyperbolic**. They are
  untrustworthy and would need re-running with a constructed embedding.

**Method note for anyone reproducing:** do not gradient-train Poincaré embeddings without
first passing a control where the answer is known. `wordnet_control.py` +
`hyp_diagnose.py` (radius↔depth ρ, boundary-pinning %) are that gate. Loss is *not* a
health signal — ours fell monotonically while the representation carried no hierarchy.

**Scope addendum (owner, 2026-09-01) — the §11 "real discovered" rows have a second,
independent defect.** Those 235k/284k-node hierarchies were imposed by divisive KMeans
on Amazon product embeddings; no one verified the underlying data has intrinsic deep
hierarchy. A storage test on a KMeans-imposed tree measures an artificial hierarchy —
a known input limitation, not evidence about hyperbolic geometry. This argument covers
the "real discovered" rows only: the synthetic k-ary rows were genuinely hierarchical
by construction, so their result is explained by the broken optimizer alone (§12–13).
Any future storage test must first pass the target-distance validity gate (≥4 distinct
values, std ≥0.5) AND establish that the tree reflects real structure, e.g. a
gold-label hierarchy or WordNet, not a clustering artifact.

## Anchors: is a taxonomy in the company's own words measured? (2026-10-25, `anchor_test.py`, `anchor_results.json`)

Pre-registered before the run. LEDGAR's 100 clause-type names stand in for a legal team's playbook terms; 20 drawn at
random (seed 2026); the hidden labels score them. Production path: `embed_names`, `TAU = 0.60`, nearest anchor.

| gate | A: name only | B: name + 5 example clauses (pre-registered after A failed) | pass |
|---|---|---|---|
| A1 median precision | 0.37 FAIL | 0.25 FAIL | >= 0.70 |
| A2 median recall | 0.35 FAIL | 0.81 PASS | >= 0.50 |
| A3 off-domain placebos (10), total | 2 of 60,000 PASS | n/a | < 0.5% |
| A3b legal-sounding placebos (5), max | 0.2% PASS ("crop insurance" took 130 Insurances clauses) | n/a | <= 1% |
| A4 unanchored types still found in discovery | 10 of 14 FAIL | not re-run | >= 90% |
| A5 other-type clauses pulled into anchors | 20.6% FAIL | 65% FAIL | <= 5% |

- Anchors are not a confirmation machine here: fake names attract nothing (unlike the 2026-06-10 Egan experiment).
- Name-only anchors split in two: distinctive names work (Notices 0.98 precision, Change In Control 0.83, Intellectual
  Property 0.82, Liens 0.71); broad names vacuum their neighbours (Amendments took 3,301 clauses, 16% amendments, mostly
  Counterparts; Disclosures swallowed Confidentiality; Payments swallowed Expenses); some attract nothing (Survival 6 of
  1,469, Existence 2). No TAU fixes it: 0.70 gives 0.87 precision at 0.005 recall; 0.50 gives 0.71 recall at 0.11.
- Exemplars lift recall but the leave-one-out threshold is too loose: broad anchors swallow whole neighbourhoods
  (Payments 5,856, Authority 4,744). Distinctive concepts stay good (Arbitration 0.82 / 0.82, IP 0.81 / 0.83,
  Subsidiaries 0.92 precision).
- Caveats: LEDGAR types are heading-derived and overlap (Waivers / No Waivers / Waiver Of Jury Trials), so exact-type
  precision is a lower bound (No Waivers family precision 0.93). A4 is weak: the unanchored baseline itself finds only
  14 of the 80 types at purity >= 50%.
- Standing: "a taxonomy from corporate knowledge" is NOT measured as working for arbitrary playbook terms. It works for
  distinctive concepts and fails for broad or overlapping ones, with up to a fifth of the corpus moved as collateral.
  Not tried (would be a new pre-registration): a per-anchor precision check by a blind judge before an anchor is
  accepted, and competitive assignment that leaves contested clauses in discovery.

### Condition C: blind-judge acceptance gate (2026-10-25, `anchor_judge_test.py`, `anchor_judge_results.json`)

Pre-registered after A and B failed. Name anchors as in A; an anchor is kept only if a blind local judge says yes to
>= 70% of 15 random members ("Is this clause a '<name>' clause? yes/no, unsure = no"). Judge gemma3:12b (amended from
27b before any judgment: the GPU was busy with a holo-splat job and the 27b model timed out).

| gate | result | pass |
|---|---|---|
| C1 median true precision of accepted anchors | 0.37 | >= 0.70 FAIL |
| C2 judge decision agrees with truth | 35% | >= 80% FAIL |
| C3 capturing placebos rejected | 1 of 2 ("patent prosecution" accepted at 0.87) | 2 of 2 FAIL |
| C4 other-type clauses in accepted anchors | 16% | <= 5% FAIL |
| C5 anchors accepted | 18 of 20 | >= 5 PASS |

- The judge is not a rubber stamp: on deliberately wrong pairs (Notices on governing-law clauses, Survival on
  base-salary clauses, 5 pairs x 5 clauses) it said yes 0 of 25 times. It fails on the near cases.
- The failure mode, read off the members: anchors attach to what a clause MENTIONS, not what it DOES. "Amendments"
  captures the governing-law and counterparts clauses of amendment documents ("This Amendment shall be governed by
  ..."); "patent prosecution" captures the counterparts clause of a Patent Security Agreement (and, rightly, real
  patent-filing cooperation clauses: this placebo was not empty-domain). A yes/no question naming one category
  shares the same confusion. Same lesson as 2026-06-10 ("nomic embeds content, not register").
- Standing: anchors as implemented (name or name + exemplars, with or without a yes/no judge) do not give a precise
  taxonomy in the company's own words on LEDGAR. Next hypothesis, not yet run: a forced-choice judge that assigns
  each clause's FUNCTION among all anchors plus "other", instead of a leading yes/no per anchor.

### Condition D: forced-choice function judge (2026-10-25, `anchor_forced_test.py`, `anchor_forced_results.json`)

Pre-registered after C. One change: the judge picks which ONE of the 20 anchors (plus "Other", shuffled) describes
what a sampled clause DOES; an anchor is kept if >= 70% of its 15 sampled members come back to it.

| gate | C (yes/no) | D (forced choice) | pass |
|---|---|---|---|
| C1 median true precision of accepted anchors | 0.37 | 0.44 FAIL | >= 0.70 |
| C2 judge decision agrees with truth | 35% | 55% FAIL | >= 80% |
| C3 capturing placebos rejected | 1 of 2 | 2 of 2 PASS | 2 of 2 |
| C4 other-type clauses in accepted anchors | 16% | 2.3% PASS | <= 5% |
| C5 anchors accepted | 18 | 8 PASS | >= 5 |

- The forced choice fixes safety: both capturing placebos rejected (their members went to Duties, IP, Disclosures),
  collateral down from 16% to 2.3%, the vacuuming anchors rejected (Amendments 0.27, Disclosures 0.27, Payments 0.60,
  Subsidiaries 0.20, Terms 0.47).
- It does not fix selection: it rejects two of the best anchors (Intellectual Property, true 0.82, judged 0.33;
  Change In Control, true 0.83, judged 0.67) and accepts weak ones (Authority, true 0.21, judged 0.80; Duties 0.37).
  The 12B judge over-uses "Duties" (11 of 15 "crop insurance" members). Part of the C1 miss is LEDGAR's overlapping
  types: No Waivers is 0.26 exact but 0.93 by family.
- Line closed, as pre-registered: four conditions (A name, B name + exemplars, C yes/no judge, D forced-choice
  judge), none meets precision. Standing: anchors are safe to SUGGEST vocabulary and to find where a playbook term
  lands (D rejects fakes and keeps collateral at 2%), but they do not yet give a precise taxonomy in the company's
  own words. Untested knob, recorded not run: a stronger judge (27B or frontier) with a larger sample per anchor.

### Condition E: anchor plus human picks (2026-10-25, `anchor_feedback_test.py`, `anchor_feedback_results.json`)

Owner's correction after D: the anchor is not meant to work alone; once set, the human picks which clauses match it
in the company's sense. Pre-registered: simulated human (yes iff exact hidden type), 10 candidates per round (first
round by name similarity, later rounds by a logistic regression on the ticks), clause joins its best anchor at
p >= 0.5. Same 20 random anchors.

| picks per anchor | precision (median) | recall (median) | collateral | random-candidate control: recall |
|---|---|---|---|---|
| 10 | 0.49 | 0.27 | 3.8% | 0.00 (16 of 20 anchors never see a positive) |
| **20 (gated)** | **0.54 FAIL** | **0.37 FAIL** | **3.4% PASS** | 0.00 |
| 40 | 0.57 | 0.51 | 4.3% | 0.00 |

F4 placebos end empty: PASS (2 of 2). Against the machine alone (A: 0.37 / 0.35 / 21%), every measure improves.

- The anchor name is what makes the picks count: with random candidates, the same budget almost never shows the
  human a single positive (recall 0 at every budget).
- At 20 picks, 9 of 20 anchors reach precision >= 0.70 (Notices 0.99, Disclosures 0.96, Terms 0.94, Survival 0.92,
  Subsidiaries 0.85, Change In Control 0.83, IP 0.82, Arbitration 0.78, Liens 0.70); IP, Arbitration, Notices and
  Subsidiaries also reach recall >= 0.50.
- Failures are rare types (Books 23 clauses, Qualifications 47, Existence 258) and overlapping types (No Waivers vs
  Waivers, Duties vs Positions). The simulated human is strict (exact type); a real team that treats Waivers and No
  Waivers as one concept would score higher, so this is a lower bound for overlapping vocabularies.
- Standing: "set an anchor, then pick what matches it in your company's sense" is the right design and is safe
  (collateral under 5%, fakes stay empty), but 20 picks per concept do not reach the pre-registered precision and
  recall; 40 picks reach the recall bar, not the precision bar. Not yet tested: smarter candidate choice (show the
  most uncertain clauses, not only the top-rated), and a real person instead of the label oracle.

### Condition E scored by validity, not label agreement (2026-10-25, `anchor_validity.py`, `anchor_validity_results.json`)

Owner: "the question must not be if it found the same exact as the gold, but if it found valid ones." Pre-registered
before judging: condition E at 20 picks; per anchor up to 8 disputed members (assigned, label differs), 2 label-matching
members and 2 random clauses of unrelated types, 233 items shuffled and blind; two frontier-model judges (Claude
subagents, half each) asked whether a legal team filing under the anchor would accept the clause there, by function,
unclear = invalid; they were told not to open the key.

| gate | result | pass |
|---|---|---|
| V0 judge trust: accepts label matches / rejects random | 94.9% / 95.0% | both >= 90% PASS |
| V1 median validity precision | **0.92** (label precision 0.54) | >= 0.70 PASS |

- 16 of 20 anchors reach >= 0.70 validity precision; No Waivers 0.34 by label is 1.00 valid (its "wrong" members are
  Waivers clauses), Positions 0.25 -> 0.81, Definitions 0.44 -> 0.86, Authority 0.55 -> 0.83.
- Still weak: Qualifications 0.41, Amendments 0.50 (4 members), Books 0.59, Duties 0.62.
- Limits: 8 disputed items per anchor (each rate is +-~17 points); calibration negatives are easy (random unrelated
  types), so the judge's leniency on near misses is not calibrated; the judges are the same model family as the
  experimenter; recall is still measured against labels (0.37 median), and how many valid clauses each anchor missed
  is unknown.
- Standing: with an anchor plus 20 human picks per concept, about 9 in 10 of the clauses Nomen files under a
  company's term are valid uses of that term (median, 16/20 concepts above 70%). The label-agreement scores of
  conditions A-E understate validity because LEDGAR gives each clause one heading-derived type.

## Anchor validity v2: pre-registered out-of-sample replication (2026-10-25, `docs/PREREG_anchor_validity_v2.md`, `anchor_validity_v2.py`, `anchor_validity_v2_results.json`)

Protocol committed before any data (1abd6d8, deviations log f94bad0). 20 anchors drawn at random from the 80 types
not used in v1; treatment E = anchor + 20 simulated human picks; baseline A = anchor name only; 842 blind shuffled
items, every item judged by two independent judges (J1 Claude Opus, J2 Claude Sonnet); primary verdict = both valid.

| hypothesis | measure | result | verdict |
|---|---|---|---|
| H3 judges reliable | J1 accepts gold / rejects random; J2 same; kappa | 94.7% / 100%; 96.5% / 95.0%; 0.757 | PASS |
| H1 E is valid | median validity precision, bootstrap 95% CI | **0.989 [0.914, 1.000]** (label precision 0.608) | PASS |
| H2 human picks matter | E vs A, Wilcoxon one-sided, 17 paired anchors | E 0.989 vs A 0.735, p = 0.002 | PASS |
| H4 E finds most valid clauses | median validity recall | 0.419 | FAIL |

Diagnostics: hard decoys (the nearest other clause type shown under the anchor) accepted by J1 15%, J2 28% (both:
at most 15%), so the judges are somewhat lenient on near misses; far-band non-members judged valid 0% / 7.5%.

- Where name-only anchors fail and picks rescue them: Litigations A 0.07 -> E 0.99, Enforceability 0.16 -> 0.96,
  Capitalization 0.17 -> 0.99, Releases 0.27 -> 0.77.
- Failures: Miscellaneous 0.14 (no coherent function), Construction 0.47 (29 members), Assigns (E produced no
  members: no model fit a positive above 0.5).
- Limitation found by reading members: "Titles" scores 1.00 valid but every disputed member is a Headings clause
  ("headings are for convenience only"); LEDGAR's Titles means title to property, which the simulated human ticked.
  The model drifted to the other sense of the word and the judges, who see only the name, accepted it. Validity by
  name cannot see a drift to another legitimate sense. Scoring Titles at its label precision (0.05) leaves the median
  at 0.98.
- Standing (replicated out of sample): with a company term plus about 20 human picks, the clauses Nomen files under
  the term are valid uses of it at a median of about 99% (95% CI lower bound 91%), significantly better than the
  term alone (74%). It finds about 42% of the valid clauses (median); recall, not precision, is the open problem.
  Not tested: a real person instead of the label oracle; uncertainty sampling for recall; other corpora.
