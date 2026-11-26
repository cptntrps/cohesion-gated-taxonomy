# catalog-retrieval-hyperbolic-01: premise check, the primary units are independent

Premise: the 146 primary units of the test split are independent enough to count as units.

Rules: the unit is one session; per session one task (smallest sha256 of "work-2283-unit:" + task id). Near-duplicate
texts were removed (normalised 400-character prefix without paths and numbers) and at most 3 tasks per first-gold
item were kept, in time order.

Checks (test split, labels only):
- 146 units from 317 tasks; 111 distinct first-gold items; the largest item covers 2.1% of the units.
- Kinds: 121 decision, 25 skill. Sources: 104 decision messages, 17 decision briefs, 25 skill calls.
- 0 sessions are in both splits.
Residual: units of one task type (security reviews of diffs, job-repair runs) share templates. The deduplication
removes identical templates; the per-type breakdown shows whether one type drives a difference.

## Amendment v2 checks: key v3 (2026-11-25, before any test-split run; review findings F6 and F12)

- 144 units from 309 test tasks; 111 distinct first-gold items; the largest item covers 2.1% of the units.
- Kinds: 119 decision, 25 skill. Task types of the units: 64 security_review, 55 decision_brief, 25 skill_call.
  Task types of all 309 tasks: 201 decision_brief, 64 security_review, 44 skill_call. The units over-weight
  security reviews (44% of units, 21% of tasks), so the reading rule needs both sets and a per-type condition.
- 0 sessions are in both splits. 52 of the 144 units have a first-gold item that also occurs in the dev gold (the
  popularity control can only score on those).
- Unit pairs with token Jaccard >= 0.75 after the key v3 rule: 0 (the three pairs of finding F12 are gone).

PREMISE: supported
