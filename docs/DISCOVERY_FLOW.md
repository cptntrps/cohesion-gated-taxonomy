# The discovery flow (what the product shows, in order)

Audience: someone with a haystack of documents (here 60,000 contract clauses) and no labels.
Each step is one tab of the UI and one measured number. Gold labels (the benchmark) are only
used to score; the product never sees them.

| # | Step | What the person does | What they see | The number that proves it |
|---|------|---------------------|---------------|---------------------------|
| 1 | Discover | opens the corpus | a named tree of concepts: branches, leaves, how many clauses each, how coherent each one is | 60,000 clauses organized into 84 concepts with 0 labels; leaf purity 40% vs 5.3% by guessing (7.5x) |
| 2 | Trust | sorts by coherence | tight clusters are green, loose ones amber; the panel shows the clauses the name came from | coherence predicts purity (Spearman +0.74): high-coherence clusters are 98% right, low ones 79% |
| 3 | Correct | renames a cluster, or names a clause that fits nowhere | the whole geometry re-runs around that word; a new branch appears with its members | one human word reorganizes thousands of clauses (anchors) |
| 4 | Find | types a concept ("governing law") | the clauses ranked by meaning, the leaf each sits in, and the cost to reach 90% recall by meaning vs by keyword | 90% recall after reading 7.7x fewer clauses than keyword search (median over paraphrase-heavy concepts) |
| 5 | Extract | picks a field ("governing_law state") | one LLM call per sub-cluster, propagated to members; the value distribution | 96.2% accuracy at 80 LLM calls vs 96.5% at 400 on the first sample; the W1 audit supports non-inferiority at a 40% read (about 2x fewer calls), not 5x |

Demo script (60 s): open Discover (tree is already there) -> sort by coherence, open one green and one amber leaf -> Correct: rename "[naming failed]" to "Assignment of rights", watch it re-run -> Find: "governing law", read the 7.7x -> Extract: the state distribution.
