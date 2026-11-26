# catalog-retrieval-hyperbolic-01: premise check, the test split was unseen when the design was fixed

Premise: no arm was run, embedded or scored on the test split before this design and its review.

Checks (2026-11-25):
- results/ holds only dev/ and dev_tuning_grid.json; there is no results/test directory.
- There is no emb_tasks_test.npz (test task texts were never embedded).
- Every tuning choice (PCA dimension 128 and beam 6 from the dev grid; the tie rounding and the seriation; the key v2
  exclusions) was made on dev or on the key, before any test run.
- The split is by session (sha256 of seed work-2283-v1 and the session id); 0 sessions are in both splits.

## Amendment v2 checks (2026-11-25, before any test-split run)

- results/ holds dev/ and the dev tuning grids only; there is no results/test directory and no emb_tasks_test.npz.
- ngc.record_register holds no row for this study id: no design is registered and nothing is scored.
- The treatment (ball construction, query insertion), the configuration (run_arms.py choose and FROZEN), the BM25
  parameters and the reading rule were changed and chosen on dev or on the design only.
- What did read the test split before this amendment: the first design review read test task texts and labels
  (never an arm output) to count gold-key fragments (F2) and near-duplicate unit pairs (F12). Key v3 answers those
  two findings with rules that apply to every task of both splits. The leak scans and the hand checks also read
  texts and labels only.
- run_arms.py refuses a test run with any configuration other than FROZEN, and refuses a second test run.

PREMISE: supported
