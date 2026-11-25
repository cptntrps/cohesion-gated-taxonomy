# catalog-retrieval-hyperbolic-01: hierarchy gate (preregistered before measurement)

Written 2026-11-25 before any edge graph statistic was computed. Lesson (graph concept kg_domain_spine): hyperbolic
embedding pays off only on a genuine hierarchy with real edges; real edges are the gate, not depth.

## Real edges (no embedding, no clustering)

From the catalog's own texts only (never from the task texts, so held-out tasks never build edges):
- decision -> decision: an owner_decision:<subject> citation in the decision's question, decision, why, note,
  evidence or authority quote; and the supersedes link.
- skill -> decision: an owner_decision:<subject> citation in the SKILL.md body.
- skill -> skill: a reference to another skill's path, link or name-plus-"skill" in the SKILL.md body.
- decision -> skill: a decision text that names a skill (path, or name-plus-"skill").
- Rule F folders (skill -> folder): only if skill headers declare them (0 of 214 at the time of writing).

## Gate

Undirected graph; LCC = largest connected component.
- H-G1 coverage: at least 50% of the 987 catalog items have one or more real edges.
- H-G2 tree-likeness: Gromov four-point delta, mean over 20000 random quadruples of LCC nodes (seed 13). The real
  LCC's mean delta is below every one of 20 degree-preserving shuffles of the same graph (double-edge swaps, 10 x
  edges per replica, LCC of each), and at least 20% below the shuffles' mean. Also reported: max sampled delta,
  delta over the mean distance, cycle rank (edges - nodes + components) per node, and the parent-count distribution.

Pass H-G1 and H-G2: the hyperbolic arm also runs on the real-edge tree (a spanning tree of the real graph, extra
parents kept as copies) and is scored on the held-out test tasks. Fail either: the finding is "this corpus has no
genuine tree beyond chance at its degree"; no real-edge hyperbolic arm is claimed; the imposed-tree arm (hyp_tree)
stays as designed and its result is read as evidence about an imposed hierarchy only.

## Gui's parent-disambiguation arm and the parent gate

Precondition: items with two or more real parents of one kind (Rule F folders). Without them the arm is UNTESTED
(not falsified) and the report names the gap. If the precondition holds, the arm and its own gate accuracy (blind
gold label of the right parent, against a random-parent control) are preregistered in a design amendment before any
test-split run.

## Type discovery

Run on the dev split only (it proposes, it does not score the test claim): tasks whose landing distance exceeds the
preset bound (the 90th percentile of dev landing distances) are pooled with their nearest items; Ward clusters of at
least 3 tasks are candidate types, named by Claude Haiku 5.5. Each candidate passes only a blind precision check by a
judge of another family (GPT through codex) with placebo names (shuffled candidate names) and known-real controls
(existing decision domains), judges defaulting to false. Reported: candidates, how many passed, one example. A passed
candidate is a proposal for the type_decision flow and a Rule F folder for Gui; nothing is created.
