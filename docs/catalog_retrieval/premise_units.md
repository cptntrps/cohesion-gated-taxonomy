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

PREMISE: supported
