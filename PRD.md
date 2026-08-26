# PRD — Cohesion-gated taxonomy discovery and extraction

Status: validated prototype. Every number here is measured and reproducible from this
repo; see [docs/RESULTS.md](docs/RESULTS.md) for the full evidence table.

## 1. Problem

Organizing a large document corpus costs **O(N) decisions**. Either a human reads and
labels every item, or an LLM does. Both scale linearly with corpus size, and both are
spent overwhelmingly on the *easy* items.

Two distinct jobs are usually conflated, and they need different tools and different
success criteria:

| job | question | failure mode |
|---|---|---|
| **MAP** | "Assign these to a taxonomy I already have." | wrong node — measured by accuracy against the reference |
| **EMERGE** | "I don't have a taxonomy. Find one." | **false merge** — sticking distinct concepts together. Coarse-but-coherent is *not* an error. |

Existing products (Relativity, Luminance, CLM extraction tools) sell MAP against a
human-authored playbook. EMERGE is under-served: a corpus whose categories nobody has
written down yet.

## 2. Users

- **Operator / analyst (primary).** Domain expert, not an engineer. Needs to organize,
  search, and extract fields from a corpus. Interacts through a UI, never code.
- **Reviewer (secondary).** Adjudicates the items the system flags as low confidence.
- **Downstream consumer.** Queries the resulting structured table.

## 3. Principles

1. **The machine does the O(N) work; the human does the O(k) work.** Clustering
   organizes every item; the human names/corrects clusters only.
2. **The LLM is never spent on the easy majority.** It names clusters (k calls) and
   reads only the items a label-free signal says are uncertain.
3. **No silent failures.** Items that fit nothing become a visible "unfitted" bucket,
   never a null result. Low-confidence cells are flagged, not hidden.
4. **Local by default.** Embeddings run on-host at $0; corpus never leaves the machine.
5. **Human corrections are structural, not cosmetic.** Renaming a cluster re-runs the
   geometry; it does not just relabel.

## 4. Core mechanism

**Cohesion gate.** Cluster cohesion (mean cosine of members to their centroid) is
computed **before any per-item LLM call** and predicts correctness (Spearman **+0.74**
vs gold purity; high-cohesion clusters **98.0%** accurate vs **79.2%** for low). It
decides which clusters are trusted with one propagated cluster-level answer and which
get exhaustive per-item inference.

This is the element we could not find in prior art. Everything else in the pipeline has
named antecedents — see the README.

## 5. User flows

| # | flow | what the human does | what the system does |
|---|---|---|---|
| A | **Emerge a taxonomy** | reviews and renames k cluster names | chunks, embeds, clusters to the silhouette peak, LLM-names each cluster, re-runs on correction |
| B | **Map into a taxonomy** | supplies the target taxonomy | assigns all items, scores cohesion, routes only low-confidence clusters to per-item read |
| C | **Deepen (concept → attribute → value)** | confirms attribute/value tables | classifies sub-structure: categorical value → sub-cluster; continuous → extract field; extra provision → multi-label tag |
| D | **Find / retrieve** | asks for "all items of type X" | semantic retrieval, robust to paraphrase and formatting |
| E | **Grow the taxonomy** | names one flagged novel item | surfaces unfitted items, seeds a new node, re-runs |
| F | **Build the table** | reviews flagged cells | populates contract/item × facet columns with confidence flags |

Flows compose into a loop: A emerges → human verifies → C deepens → E grows as new
items arrive.

## 6. Requirements

**Functional**
- F1. Cluster a corpus with no taxonomy, labels, or category metadata supplied.
- F2. Name every discovered cluster with a short canonical term (1–3 words, no
  "A and B" compounds).
- F3. Compute a label-free confidence score per cluster; expose it in the UI.
- F4. Accept a human cluster name as an anchor and re-cluster the whole corpus around it.
- F5. Surface items that fit no cluster as named-able novelty candidates.
- F6. Extract a structured field per concept, one LLM call per sub-cluster, propagated.
- F7. Route only low-confidence clusters to per-item LLM reads (the gate).
- F8. Two modes with separate stopping rules: MAP (split to depth) and EMERGE (split
  only at a real multimodal seam).

**Non-functional**
- N1. Embeddings local, $0, corpus never leaves the host.
- N2. LLM calls O(k + uncertain residual), never O(N).
- N3. Every result reproducible from a committed script; failures committed too.
- N4. Report purity (per-item %), NMI and AMI as **separate** quantities; never present
  an information-overlap score as a percentage of items.
- N5. Any accuracy comparison ships with a paired significance test.

## 7. Success metrics (all measured — see docs/RESULTS.md)

| metric | target | measured |
|---|---|---|
| extraction accuracy vs exhaustive read | parity | **96.2% @ 80 calls** vs 96.5% @ 400 (diff −0.2, CI [−2.5,+2.0], n.s.) |
| LLM call reduction at parity | ≥5× | **5×** |
| gold-free confidence signal | \|ρ\| ≥ 0.5 | **+0.74** |
| human decisions to organize a corpus | ≥100× fewer | **~300×** (20 namings organize 6,000 docs) |
| false merges in EMERGE mode | reduce materially | **66% → 15.9%** |
| cross-domain generality | ≥3 unrelated domains | legal, news, e-commerce |

## 8. Non-goals (explicit)

- **Not a clustering-accuracy SOTA.** Clustering quality is mid-range and that is
  acceptable; the contribution is *where the expensive calls go*.
- **Not "cheaper AI review".** Per-document AI review has commoditized (~$0.11–0.50;
  RelativityOne bundles aiR at no additional cost from early 2026). An LLM-call saving
  is a margin lever, not a wedge. The wedge is EMERGE + local/private operation.
- **Not a hyperbolic anything.** Hyperbolic geometry lost at clustering in 6
  independent tests, and its one remaining justification — scale-resilient hierarchy
  storage — was **retracted on 2026-08-26** as a measurement artifact (docs/RESULTS.md
  §11). Euclidean wins on every hierarchy tested, synthetic and real. The Poincaré code
  is retained for reproducibility of the negative results, not because it is used.
- **Not a novelty claim.** Every component has prior art; only the specific pre-call
  gating policy was not located. See README.

## 9. Risks

| risk | evidence | mitigation |
|---|---|---|
| Cluster-first fails on **incidental** attributes | periodicity scored 33.9% vs 80.1% baseline | restrict to attributes that dominate the text; give cross-cutting attributes their own projection |
| Gold-agreement misjudges EMERGE | system merged rec.autos+rec.motorcycles — arguably better than the reference, scored as error | judge EMERGE by false merges + coherence, not reference agreement |
| Cohesion gate loses at high budget | item-margin wins at ≥500 reads | use cohesion at low budget, switch to item-level margin at high budget |
| Ground truth may be weaker than the system | regex oracle silently dropped 519/3167 clauses | blind manual adjudication before publishing headline accuracy |
| Commoditization | AI review bundled free | compete on EMERGE and privacy, not cost |

## 10. Open questions

1. Is there a validated **gold-free emergence quality metric** (coherence + distinctness
   + judge, calibrated to human preference)? This is the honest open problem — the field
   routinely dodges it by scoring against a reference taxonomy.
2. Does the cohesion gate beat a per-item-confidence cascade at **equal total budget**
   including the cascade's own N cheap calls? Partial evidence: the gate costs zero
   calls, so it operates below the cascade's N-call floor.
3. Does the pipeline hold on full contracts (contract-level rollup), not just standalone
   clauses?
