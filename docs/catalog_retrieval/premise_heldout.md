# catalog-retrieval-hyperbolic-01: premise check, the test split was unseen when the design was fixed

Premise: no arm was run, embedded or scored on the test split before this design and its review.

Checks (2026-11-25):
- results/ holds only dev/ and dev_tuning_grid.json; there is no results/test directory.
- There is no emb_tasks_test.npz (test task texts were never embedded).
- Every tuning choice (PCA dimension 128 and beam 6 from the dev grid; the tie rounding and the seriation; the key v2
  exclusions) was made on dev or on the key, before any test run.
- The split is by session (sha256 of seed work-2283-v1 and the session id); 0 sessions are in both splits.

PREMISE: supported
