# Pre-registration: anchor + picks at zero LLM calls vs an LLM reading every clause (v3)

Registered 2026-10-26, committed before any v3 data, prediction or judgment exists. Owner: "it is both. it is cheaper
and also the other ... by clustering together, you prevent using the llm" -> "do it". v2 showed validity precision
0.989 for anchor + 20 picks but validity recall 0.419, so "cheaper at the same quality" was not established.

## Question

At zero LLM calls, how many human picks per term does Nomen need to match an LLM that reads every clause, on validity
precision and on validity recall?

## Design

| element | v3 |
|---|---|
| terms | the 20 out-of-sample v2 anchors (`v2_key.json`) |
| test set S | 2,000 clauses drawn at random (seed 7777) from the 60,000; S is excluded from every pick |
| Nomen variants (0 LLM calls) | R20, R40, R80: ranked rounds (10 per round, top-rated unjudged; round 1 by name similarity); U40, U80: "least sure first": after round 1, each round shows the 5 top-rated and the 5 closest to p = 0.5. Simulated human = exact LEDGAR type (as v2). Logistic regression as `anchor_feedback_test.teach`. A test clause gets its best term if p >= 0.5, else none |
| LLM baseline | Claude Sonnet, one call per test clause: given the clause and the 20 terms, pick the ONE term that describes its function, or None. Blind to labels. Batches of 250 |
| pooling | every (clause, term) pair proposed by any system, plus each test clause's label pair when its label is one of the 20, is judged; unproposed, non-label pairs are assumed invalid (standard pooling; stated as a limit) |
| judges | two independent Claude Opus judges (J1, J2), blind to which system proposed a pair; same question and rule as v2; a pair is valid only if both say valid |
| calibration | 40 label pairs (positives) and 40 pairs of a test clause with a term that is neither its label nor in its synonym family (negatives), mixed in blind |

## Measures

- validity precision of a system = valid proposed pairs / proposed pairs (micro, over all terms)
- pooled validity recall = valid proposed pairs / all valid pairs in the pool
- bootstrap: 2,000 resamples of test clauses; differences Nomen - LLM computed within each resample
- cost per 60,000 clauses: LLM calls (baseline 60,000; Nomen 0) and human picks (Nomen: budget x 20)

## Hypotheses and gates

| id | hypothesis | pass |
|---|---|---|
| H7 | judges reliable | each judge accepts >= 90% of calibration positives and rejects >= 90% of negatives; kappa >= 0.60 on pool pairs |
| H5 | Nomen precision is not worse than the LLM's at 0 calls | R20: bootstrap 95% lower bound of (precision R20 - precision LLM) >= -0.05 |
| H6 | some Nomen budget matches the LLM's recall | at least one variant with <= 80 picks per term has bootstrap 95% lower bound of (recall variant - recall LLM) >= -0.05; the smallest such budget is the answer |

If H7 fails, H5 and H6 are not established. Diagnostics: per-term table, either-judge results, LLM "None" rate.

## Deviations

Written below with reason and time, before scores are computed.
