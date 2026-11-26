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

---

# Amendment v2 (2026-11-25, written before any test-split run)

The first design review (reviewer agent, 2026-11-25, DESIGN 9ed8b0c7..., study/design_review_v1.md) returned FIX-FIRST
with 13 findings. The hierarchy gate (hierarchy_gate.md) was measured and failed. This amendment answers both. It
replaces the v1 sections "Corpus and answer key", "Arms" (the hyp_tree, tree_cos, bm25 and llm_catalog rows and the
configuration paragraph), "Power", the descriptive reading, "Cost" and "Dev pilot". No arm ran on the test split
before this amendment (premise_heldout.md).

## What does not change

- The question, the decision served and the lessons.
- The primary outcome: set test_sessions (one task per session), field hit10 (the first gold item is in the arm's
  top 10), stratum all.
- Hypotheses H1 to H6, their predictions and criteria, the baseline (flat) and the controls (bm25, llm_catalog,
  popularity).
- The split rule and its seed. The study stays exploratory.

## Hierarchy gate result and the scope of the claim

The rule was fixed in commit f17e32f before any edge statistic. Result (study/hierarchy_gate_result.json): FAIL.

- H-G1 coverage: 230 of 987 items have one or more real edges (0.23; the bar is 0.50). The 180 edges (129 decision
  citations, 42 skill references, 9 skill names in decisions) form 68 components.
- H-G2 tree-likeness: the largest component has 31 nodes and 31 edges. Its mean four-point delta is 0.0. 9 of the 20
  degree-preserving shuffles also give 0.0 (shuffle mean 0.026), so the real graph is not more tree-like than chance
  at its degree.

By the preregistered rule:
- No real-edge hyperbolic arm is claimed. The finding of the gate is "this corpus has no genuine tree beyond chance
  at its degree".
- hyp_tree runs on the imposed tree (real decision domains, Ward clusters for skills and for large groups). Its
  result is evidence about an imposed hierarchy only. A negative result says nothing about a corpus with real edges.
- Gui's parent-disambiguation arm is UNTESTED, not falsified. Its precondition is items with two or more real
  parents of one kind (Rule F folders); 0 of 214 skill headers declare a folder. The report names this gap.
- Type discovery ran on dev only (results/dev/type_discovery.json). It proposes; it does not score the test claim.

## Review findings and the action for each

| finding | action |
|---|---|
| F1 CRITICAL: the geometry decided no rank | Option (b), by the lead's ruling, since the question is about the geometry. hyp_tree v2 ranks every item by unrounded Poincare distance to the inserted query point. A hermetic test asserts that the ranking equals the distance order. tree_cos stays its own arm, so H5 (hyp_tree minus tree_cos) isolates the ranking metric. |
| F2 HIGH: gold-key fragments in the texts | Key v3 masks the slug fragments of every catalog decision subject and every 8-digit date, in every task. The scan finds 0 slug fragments of the gold. The gold_fragment stratum is reported (premise_key_handcheck.md). |
| F3 HIGH: lost LLM picks | The prompt numbers the catalog rows and the model answers numbers. Every raw reply is stored. 0 valid picks is a failed row. The run prints the n_picks distribution (dev: 107 of 115 rows with 10 picks, 0 failed). |
| F4 MEDIUM: padded LLM ranking | The ranking is the picks only. |
| F5 MEDIUM: H1 changes two variables | H7 (secondary, estimate only): hyp_tree minus flat_pca > 0. With H6 it splits H1 into the dimension cut and the tree plus ball. |
| F6 MEDIUM: two populations | The reading rule now needs both sets (all tasks and the primary units) and a per-type condition (below). |
| F7 MEDIUM: weak labels | Stays as a limit. The record names the label source as an alternative explanation of H1, H2, H3 and H7. The skill_call type is reported as the nearest proxy for a real retrieval need. No general claim rests on one task type (reading rule). |
| F8 MEDIUM: reading not mechanical | score.py reading() computes the call. "The best simple method" is fixed from dev (BEST_SIMPLE), never from test estimates. The cost report adds the end-to-end latency with the query embedding and the GPU need. INCONCLUSIVE means: keep the simpler method. |
| F9 MEDIUM: hash coverage | prereg_ref carries the sha256 of design.md and the commit of code and design; the record hash covers prereg_ref. |
| F10 LOW: tuning record | The grid ran again on key v3 with the final code (results/dev_tuning_grid_v4.json), with beams up to 32 and both landing modes. BM25 k1 and b are tuned on the same dev split. Edges are stated below. |
| F11 LOW: model identity | The embedding digest is in every run meta, and a fresh embedding of all items equals the cache (premise_embeddings.md). |
| F12 LOW: near-duplicates | Key v3 drops a task with token Jaccard >= 0.75 to a kept task; 0 such unit pairs remain. |
| F13 LOW: power text | Replaced below: with threshold 0 the criterion cannot FAIL under the null. |

## Treatment v2 (replaces the hyp_tree and tree_cos rows)

- Tree: root -> kind -> group -> [subgroup] -> item, as in v1 (real decision domains; 14 Ward clusters of skills; a
  group above 24 items is split into Ward subgroups).
- Descriptors: unit vectors in the `dim`-d PCA space of the catalog; a node's descriptor is the unit sum of its
  children's.
- Ball: a second PCA of the node descriptors gives `ball_dim` coordinates. The root is at 0. Each child is at
  hyperbolic distance `tau` from its parent: x_child = x_parent (+) tanh(tau/2) v, with (+) the Mobius addition and v
  the child's semantic direction (its descriptor offset from the parent, without the component toward the
  grandparent, pushed away from it). No training. The distance and the Mobius addition are those of pg_hyperbolic
  (cross-check test; premise_tree.md).
- Query: the `dim`-d query descends the tree by beam search on descriptor cosine and lands on leaf-parent nodes. The
  query is inserted into the ball as a new child of a landing node, by the same rule as the items.
- hyp_tree: every item is ranked by its unrounded Poincare distance to the query point (with `land_all` 1: the
  smallest distance over the query points under every beam node). Nothing else enters the ranking.
- tree_cos: the same tree, descent and landing nodes; items ranked by hop count from the landing node, ties by
  `dim`-d cosine. It has no hyperbolic geometry.
- H5 reads hyp_tree minus tree_cos. The two arms differ in the ranking only: distance in the ball against hops with
  cosine ties. When `ball_dim` is below `dim`, a loss can also come from the ball's lower dimension; the record
  names this.

## Frozen configuration (dev split only, key v3)

- Grid (run_arms.py --tune, results/dev_tuning_grid_v4.json, 486 tree cells and 30 BM25 cells, final code):
  dim {64, 128} x ball_dim {16, 32, 64, 128, not above dim} x tau {1.5, 2.5, 3.5} x landing {best node, every beam
  node} x beam {1, 2, 4, 6, 8, 12, 16, 24, 32}. The best base cells had ball_dim = dim, the largest dim and the
  smallest tau, so the grid went on to dim 256 and tau 1.0 with ball_dim = dim. BM25: k1 {0.9, 1.2, 1.5, 2.0, 2.5,
  3.0} x b {0.3, 0.5, 0.75, 0.9, 1.0}.
- Selection rule (run_arms.py choose, hermetic test): hyp_tree = the cell with the highest dev hit@10 whose ball
  fits float32; ties go to the higher dev MRR, then the smaller beam, ball_dim and dim, then tau nearest 2.5, then
  one landing node. BM25 = the highest dev hit@10; ties to the higher MRR, then the cell nearest (1.5, 0.75).
- Frozen (run_arms.py FROZEN): dim 256, ball_dim 256, tau 1.5, beam 24, one landing node (land_all 0); BM25 k1 2.5,
  b 0.75. flat_pca uses the same dim (256). tree_cos uses the same tree, descent and landing node.
- Since ball_dim = dim, the ball has the dimension of the descriptors, and H5 compares the ranking metric only.
- Edges, stated: dim 256 is the largest dim tried. The best hyp_tree dev hit@10 per dim is 0.400 (64), 0.417 (128),
  0.426 (256): one or two tasks of 115 per doubling. flat_pca gains more per doubling (0.478, 0.565, 0.617), so the
  gap to flat_pca widens (-0.078, -0.148, -0.191). A larger dim is not expected to close it. Beam: 24 and 32 give
  the same hit@10 (plateau). tau: 1.0 and 1.5 tie on hit@10; 1.5 wins on MRR. BM25: the chosen cell is inside its
  grid, and the whole BM25 grid lies between 0.696 and 0.739 (5 tasks).
- The embedding model digest of the grid is 79076464...; the test run checks that the live model reproduces the
  item cache before it embeds the test tasks.

## Answer key v3 (replaces "Corpus and answer key" where it differs)

- Catalog: unchanged (987 items, catalog.jsonl sha256 43a7c61c...).
- Tasks: key v3, tasks.jsonl sha256 71f49d73..., 424 tasks: dev 115 tasks in 60 sessions, test 309 tasks in 144
  sessions. Persisted as ngc.gold_dataset catalog-retrieval-work2283-v3. Key v2 (432 tasks) was never scored on test.
- New rules (build_key.py): fragment and date masking in every task; the gold_fragment flag; the Jaccard rule. The
  split seed, the cap of 3 tasks per first-gold item and the 2000-character cut are unchanged.
- Residual: 5 test tasks keep the 8-digit date of their gold inside a longer token. The masking also removes generic
  phrases that equal another decision's slug (3 of 20 hand-checked tasks). Both lower lexical cues for every arm.
- The test split is scored with key v3 and no other key.

## Primary outcome and hypotheses

- Primary set: test_sessions, 144 units. Field hit10. Stratum all.
- H1 to H4 as in v1. H5 and H6 as in v1 (secondary). H7 is new (secondary): hyp_tree minus flat_pca > 0, falsified
  if <= 0.
- A failed llm_catalog row (a call that fails twice, or 0 valid picks) counts as a miss of that arm. Above 10%
  failed rows the arm did not run and its hypotheses are not_run. The count of failed rows is reported.

## Descriptive reading v2 (replaces the v1 reading; computed by score.py reading())

Why it changes: the review found that the v1 rule was not mechanical, chose "the best" simple method from test
estimates, and read a population (all tasks) that differs from the primary units (F6, F8).

The paired hit10 differences of hyp_tree minus each of flat, bm25 and llm_catalog are computed on two sets: all 309
test tasks (session-cluster bootstrap, 10000 resamples, 95%) and the 144 primary units (bootstrap over units).

- POSITIVE: on both sets the lower bound is above 0 against each of the three; in no task type with 20 or more
  units is a point estimate negative; and the marginal USD per 1000 queries of hyp_tree is not above that of
  BEST_SIMPLE.
- NEGATIVE: on both sets the upper bound of hyp_tree minus BEST_SIMPLE is below 0; and in no task type with 20 or
  more units is that point estimate positive.
- INCONCLUSIVE: anything else. The action is the same as for NEGATIVE: keep the simpler method. The treatment must
  prove its value.
- BEST_SIMPLE = bm25: the simple method with the highest dev hit@10 on key v3 (dev pilot below). It is fixed now.
  This rule is conservative for NEGATIVE: if flat or llm_catalog beats bm25 on test, a loss against them alone gives
  INCONCLUSIVE.
- CHEAPER (a separate flag, not the call): hyp_tree is non-inferior to llm_catalog on the units (lower bound above
  -0.05) at a lower cost per query in USD and in end-to-end latency.
- Cost clause: the v1 clause compared marginal USD per 1000 queries, and it stays so. A hard latency clause against
  BM25 would make POSITIVE unreachable by construction (every embedding arm pays about 0.34 s for the query
  embedding on a GPU; BM25 pays none). So the report gives, next to the call, the end-to-end latency per arm and
  which arms need the GPU embedding.
- The record's H1 to H7 get intervals from ngc_eval.method score-record (Newcombe, unit counts) and the verdict
  estimate_only, since the study is exploratory. The call above is the only verdict word of the study.

## Power (replaces the v1 text)

Exploratory. 144 independent sessions per arm. With the declared threshold 0, a true difference of 0 gives FAIL
with probability 0.027; no sample size reaches FAIL power 0.80, so the criterion cannot support a confirmatory
claim. One contrast has PASS power 0.41 at a true effect of 0.10, 0.73 at 0.15 and 0.94 at 0.20 (ngc_eval.stats
power, two_proportion, baseline 0.50); the joint AND over the baseline and three controls is lower. The study
reports estimates with 95% intervals and no gate PASS or FAIL. The unit counts are self-reported; the registration
proves that the design existed before scoring, not before the data.

## Run protocol

1. The record (study/record.json) passes check-record with the review verdict MERGE and is registered in
   ngc.record_register before any test output exists.
2. run_arms.py runs the test split once, with FROZEN only; it refuses another configuration and a second run.
   llm_arm.py runs the LLM control on the same tasks. Both refuse a key that is not the frozen key of the record.
3. score.py writes the summary and the unit counts, and logs every arm to MLflow experiment
   work-2283-catalog-retrieval. score-record scores the registered record once.
4. The result is written under this design in docs/RESULTS.md: numbers, intervals, the call, and every deviation.

## Dev pilot on key v3 at the frozen configuration (115 tasks in 60 sessions; not evidence for the test claim)

hit@10, all dev tasks: bm25 0.739, flat_pca 0.617, flat 0.591, llm_catalog 0.530, tree_cos 0.443, hyp_tree 0.426,
popularity 0.209, random 0.017. llm_catalog: 0 failed rows, 107 of 115 rows with 10 picks.

Paired dev differences of hyp_tree (all tasks, session-cluster bootstrap): minus bm25 -0.313 [-0.439, -0.222];
minus flat -0.165 [-0.262, -0.106]; minus llm_catalog -0.104 [-0.250, 0.000]; minus tree_cos -0.017
[-0.060, 0.000]; minus flat_pca -0.191 [-0.304, -0.126]. The dev reading is NEGATIVE; the test split decides.

By task type (dev, hit@10): skill_call (23 tasks) llm_catalog 0.522, bm25 0.261, flat 0.174, hyp_tree 0.087;
security_review (29) bm25 0.828, flat 0.655, hyp_tree 0.379, llm_catalog 0.379; decision_brief (63) bm25 0.873,
flat 0.714, llm_catalog 0.603, hyp_tree 0.571. Fragment stratum (dev): with a gold fragment (30 tasks) bm25 0.933,
flat 0.900, hyp_tree 0.667; without (85) bm25 0.671, flat 0.482, hyp_tree 0.341.

Cost (dev): llm_catalog 1.96 USD per 1000 queries (0.226 USD for 115 calls, cached catalog prompt) and a median of
2.7 s per query; every other arm 0 USD. Ranking latency: bm25 8.2 ms, flat 0.1 ms, flat_pca 0.4 ms, hyp_tree
1.0 ms, tree_cos 0.5 ms. Every embedding arm also pays the query embedding (338.5 ms in the first measure, 394 ms in
the second); BM25 pays none. The test run spends about 0.6 USD on llm_catalog.

Type discovery (dev only, run at 14:40 with the v1 descent, dim 128 and beam 6): 12 uncovered tasks, 1 candidate
type, which passed the blind judge (precision 0.80, placebo 0.00). The known-real controls scored 0.25 to 0.75, so
the judge is strict and the placebo was easy. It was not run again at the frozen configuration, since the GPU had
no room for the judge model. A passed candidate is a proposal only.

## Caveats added

- The treatment is one construction (a training-free ball over an imposed tree) on one corpus of 987 items. The
  study does not test a trained hyperbolic embedding, a real-edge hierarchy or a larger catalog.
- The labels measure a reverse lookup from a paraphrase for most units (F7). The skill_call stratum has 25 units and
  wide intervals.
- The dev split has 115 tasks in 60 sessions. Many grid cells tie; the selection rule breaks ties without the test
  split.
