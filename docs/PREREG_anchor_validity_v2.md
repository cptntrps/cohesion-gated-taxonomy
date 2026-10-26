# Pre-registration: anchor + human picks, scored by validity (v2, out-of-sample replication)

Registered 2026-10-25, committed before any v2 data, assignment or judgment exists. Owner request: "create a
scientifically structured test and do it again". v1 (`anchor_validity.py`, RESULTS "Condition E scored by validity")
found median validity precision 0.92 on the 20 anchors the method was developed on, with five stated limits. v2
removes them.

## Question

When a company sets an anchor (its own term for a clause type) and a person ticks about 20 candidate clauses that
match it, are the clauses Nomen then files under that term valid uses of the term, and is that better than the
anchor name alone?

## Design

| element | v2 |
|---|---|
| corpus | LEDGAR, 60,000 provisions, nomic-embed-text vectors (as in all conditions) |
| anchors | 20 type names drawn at random, seed 4242, from the 80 types NOT used in v1 (out of sample) |
| treatment E | anchor + 20 simulated human picks (`anchor_feedback_test.teach`, ranked, 10 per round); join at p >= 0.5, best anchor |
| baseline A | anchor name only, nearest anchor at cosine >= TAU 0.60 (`anchor_test.assign`) |
| simulated human | ticks yes iff the clause's LEDGAR type is the anchor's type (strict; conservative under a validity score) |
| judges | two independent frontier-model judges (Claude subagents J1 and J2), each sees EVERY item, blind (items shuffled, no kind, no label, told not to open the key) |
| question | would a legal team filing clauses under the anchor accept this clause there? judge function, not mentions; unclear = invalid |
| verdict | primary: valid only if J1 AND J2 say valid; secondary: either |

Items per anchor (seed 4242):

| kind | n | purpose |
|---|---|---|
| E disputed | 15 | E members whose label differs from the anchor |
| A disputed | 10 | A members whose label differs from the anchor |
| gold | 3 | E members whose label matches (calibration positives) |
| random | 3 | clauses of a type outside the 20 anchors (calibration negatives) |
| hard | 3 | clauses of the anchor's nearest other type by centroid cosine, excluding its synonym family (`anchor_test.FAMILIES`), shown under the anchor (leniency on near misses; diagnostic) |
| near miss | 10 | E non-members with E probability in [0.2, 0.5), label differs (for validity recall) |
| far | 2 | E non-members with probability < 0.2, label differs (checks the far band is ~0 valid) |

## Measures

- validity precision of a condition, per anchor = (label-matching members + disputed members x judged-valid rate of
  its disputed sample) / members
- validity recall of E, per anchor = valid members / (valid members + label-matching non-members + near-band
  non-members x their judged-valid rate); the far band is assumed 0 and its sample checks that
- bootstrap: 2,000 resamples of anchors, and within each anchor of its judged items, for the medians
- Cohen's kappa between J1 and J2 on all disputed and near-miss items

## Hypotheses and gates (fixed now)

| id | hypothesis | pass |
|---|---|---|
| H1 | E is valid | median validity precision >= 0.70 AND bootstrap 95% lower bound >= 0.60 |
| H2 | the human picks matter | E > A in validity precision, Wilcoxon signed-rank one-sided p < 0.05 over the anchors that have members in both |
| H3 | the judges are reliable | each judge accepts >= 90% of gold AND rejects >= 90% of random, AND kappa >= 0.60 |
| H4 | E finds most valid clauses | median validity recall >= 0.50 |

If H3 fails, H1, H2 and H4 are reported as not established, whatever their numbers. Diagnostics (no gate): hard-
decoy acceptance per judge, label precision and recall, per-anchor table, the either-judge verdict.

## Deviations

Any deviation from this document is written below with its reason and time, before the scores are computed.

- 2026-10-25, before any verdict was returned or any score computed: J1 runs on Claude Opus and J2 on Claude Sonnet
  (the protocol said only "two Claude subagents"), so the two judges are different models and their agreement is a
  stronger reliability check. Each judge reads all 842 items in 4 batches of about 210 (context size); batches are
  merged per judge before scoring. Items: 842 (E disputed 264, A disputed 171, gold 57, random 60, hard 60, near
  190, far 40; fewer than planned where an anchor had fewer candidates).
