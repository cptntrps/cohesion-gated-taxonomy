# catalog-retrieval-hyperbolic-01: design (WORK-2283)

Written 2026-11-25 before any test-split run. Code: cohesion-gated-taxonomy, experiments/catalog_retrieval
(branch work-2283-catalog-retrieval). Data: ~/data/cohesion-gated-taxonomy/catalog_retrieval.

## Question

Does a hyperbolic hierarchical index (Sarkar-constructed Poincare-disk tree, PCA-reduced descriptors, beam descent
and ring-by-ring Poincare-distance jumps) put the gold item of a real agent task in its top 10 more often than flat
bge-m3 cosine?

Decision served: whether to build a hyperbolic index over rules, decisions and skills for agents, or keep a simpler
method. The method must be better or cheaper than the best simple method.

## Lessons applied (memory and repo history)

- Gradient-trained Poincare embeddings failed the WordNet control; Sarkar's construction passed (cohesion-gated-
  taxonomy RESULTS.md sections 12-13). The treatment uses the construction, and a premise check proves the encoding.
- Geometric quality is not task validity (ontological_rigor): the only outcome is the downstream retrieval task.
- Early hyperbolic retrieval wins were scored on the embedding's own training edges (recall lesson): here the gold is
  external (real skill calls and real decision citations), never the tree.
- Uniform-shell zoom retrieval degenerated: the treatment ranks by tree rings from a descended landing node, not by
  radius shells.
- Rearrangement is not validation (anchor-emergence): placebo-like controls (random, popularity) are arms.
- nomic-embed on Ollama collapses capitalised inputs: bge-m3 is used and checked (premise_embeddings.md).

## Corpus and answer key

- Catalog (987 items, catalog.jsonl sha256 43a7c61c...): 214 skills (~/src/agents/skills) and 773 owner decisions
  (one per subject_key whose newest row is decided or executed; rules such as Rule A and Rule G are rows of this
  register). The ops.owner_decisions kind column of WORK-2279 does not exist yet, so rules and decisions are one kind.
- Item text for every arm: "<name>: <first 40 words of the description>".
- Tasks (tasks.jsonl, key v2, sha256 ea343798...): 432 tasks from surviving Claude transcripts (older ones are gone,
  so 333 of 479 skill_invocation rows have no text). Skill tasks: the last user or brief text before a real Skill
  call (125). Decision tasks: briefs and messages that cite owner_decision:<subject> (gold = the cited subjects in the
  catalog). Gold names masked. Redacted on the full text with lib/redact.sh, then cut to 2000 characters.
  Near-duplicates removed, at most 3 tasks per first-gold item, harness notifications and batch-judge prompts excluded.
- Split by session, seed work-2283-v1: dev 115 tasks / 59 sessions (tuning only), test 317 tasks / 146 sessions.
- Persisted: ngc.gold_dataset and ngc.gold_item (dataset catalog-retrieval-work2283-v2), loaded before the run.

## Arms

| arm | role | what it does |
|---|---|---|
| hyp_tree | treatment | tree root -> kind -> group -> [subgroup] -> item in the Poincare disk (Sarkar, tau 6.921, seriated siblings); query -> 128-d PCA; beam 6 descent by cosine to node descriptors; rank items by Poincare distance (2 decimals) from the nearest landing node, ties by 128-d cosine |
| flat | baseline | cosine on the full 1024-d bge-m3 vectors |
| bm25 | control | Okapi BM25 (k1 1.5, b 0.75) on the item text |
| llm_catalog | control | Claude Haiku 5.5 reads the whole catalog (system prompt) and returns 10 ids |
| popularity | control | items by gold frequency in the dev split |
| tree_cos | secondary | hyp_tree with tree hop distance in place of Poincare distance (isolates the geometry) |
| flat_pca | secondary | cosine on the 128-d PCA vectors (isolates the dimension reduction) |
| random | secondary | a seeded permutation per task |

Treatment configuration chosen on dev only: PCA dimension 128 and beam 6 gave the best dev hit@10 of hyp_tree in the
grid dim {16, 32, 64, 128} x beam {1, 2, 3, 4, 6} (dev_tuning_grid.json). Two numerical fixes were made on dev before
this design: rounding (float64 noise broke exact ties) and seriation (arbitrary sibling angles carry no meaning).

## Primary outcome

Set test_sessions: 146 units, one task per session (smallest sha256 of "work-2283-unit:" + task id). Field hit10: the
first gold item of the task is in the arm's top 10. Stratum all.

## Hypotheses

- H1: hyp_tree minus flat > 0 (falsified if <= 0).
- H2, H3, H4: hyp_tree minus bm25, llm_catalog, popularity > 0 (the same prediction and criterion as H1).
- H5 (secondary): hyp_tree minus tree_cos > 0: the Poincare geometry adds retrieval value beyond the tree it encodes.
- H6 (secondary): flat_pca minus flat < 0: reducing 1024 to 128 dimensions loses hit@10.

Competing explanations: a gain of hyp_tree over flat could come from the tree, not the geometry (H5 tests it), or
from lexical cues that BM25 captures better (H2). A loss could come from wrong branch choices in the descent (beam
width) or from the dimension cut (H6).

## Power

Exploratory. With 146 independent sessions per arm, the gate's two-proportion power model (Newcombe interval, joint
AND over the baseline and three controls) needs at least 169 units for FAIL power 0.80 even at a margin of 0.15, and a
true effect near 0.30 for PASS power 0.80 (ngc_eval.stats n-for-power, baseline 0.58 from the dev pilot). The study
reports estimates with 95% intervals and no gate PASS/FAIL. The record counts per unit are self-reported; the
registration proves the design existed before scoring, not before the data.

Preregistered descriptive reading (not a gate verdict, fixed now): on the secondary all-task paired differences
(session-cluster bootstrap, 95%), the method is called POSITIVE if hyp_tree minus each of flat, bm25 and llm_catalog
has a lower bound above 0 on hit10, at equal or lower cost per 1000 queries than the best of them; NEGATIVE if hyp_tree
minus the best of flat, bm25 and llm_catalog has an upper bound below 0; else INCONCLUSIVE.

## Secondary analysis (fixed now, estimate only)

All 317 test tasks: hit5, hit10, MRR (rank cut 50; the LLM ranks only its 10 picks), recall5, recall10 over all gold;
95% intervals by the session-cluster bootstrap of ngc_eval.stats (10000 resamples); paired per-task differences of
hyp_tree minus each arm; breakdowns by kind and by task type (skill_call, security_review, decision_brief);
ranking latency (median, p95) per arm plus the shared 338 ms query embedding; cost per 1000 queries. Every run is
logged to MLflow experiment work-2283-catalog-retrieval. An arm with more than 10% failed rows did not run.

## Cost (dev pilot)

USD per 1000 queries of marginal API spend: llm_catalog 7.56 (Haiku 5.5, cached 106k-token catalog prompt, median
2.0 s per query); every other arm 0 (local CPU and GPU). Ranking latency on dev: bm25 12.6 ms, flat 1.7 ms, flat_pca
0.3 ms, hyp_tree 0.7 ms, tree_cos 1.7 ms. The index holds 987 items; at this size every local arm is far below the
338 ms query embedding.

## Dev pilot (115 tasks, hit@10, not evidence for the test claim)

bm25 0.713, flat 0.583, flat_pca 0.565, llm_catalog 0.548, hyp_tree 0.530, tree_cos 0.530, popularity 0.209,
random 0.017.

## Caveats

- Gold holds only what agents used or cited; relevant but unused items count as misses for every arm.
- 44% of the test units are security-review prompts over diffs that cite decisions in code, which favours lexical
  matching.
- The skill-group level of the tree is imposed (Ward clusters); the decision domains are real.
- Haiku 5.5 is the cheapest Claude tier; a stronger model may do better as the LLM arm.
- One owner, one corpus, one date: no claim beyond this catalog of about 1000 items.
