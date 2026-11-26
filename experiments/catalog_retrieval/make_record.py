#!/usr/bin/env python3
"""WORK-2283: write the ngc-eval study record (schema ngc-eval/study-record/v1) for the study directory.

  python3 make_record.py <study_dir> --commit <git sha> [--review design_review_v2.md --verdict MERGE]

Design v2 (docs/catalog_retrieval/design.md, amendment v2). Every sha256 is computed from the files in <study_dir>;
the frozen configuration comes from run_arms.FROZEN; nothing is typed by hand. The commit is the code and design
commit of the repository: prereg_ref carries it and the sha256 of design.md, and the record hash covers prereg_ref.
"""
import argparse
import hashlib
import json
from pathlib import Path

import run_arms as A

STUDY = "catalog-retrieval-hyperbolic-01"
SET = "test_sessions"
N_UNITS = 144
LLM_USD_PER_1000 = 1.96   # dev pilot on key v3: 0.226 USD for 115 calls
LABEL_BIAS = ("label source: the gold is what an author cited or an agent invoked, so the texts paraphrase their gold; "
              "this favours lexical and semantic overlap over structure")
DATA = ("test split, one task per session (smallest sha256 of 'work-2283-unit:' + task id), first gold item in the "
        "top 10; key v3")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hyp(hid, statement, arm, vs, alts, op=">", crit="<="):
    return {"id": hid, "statement": statement, "prediction": {"op": op, "value": 0.0},
            "measurement": {"metric": {"set": SET, "arm": arm, "vs": vs, "field": "hit10", "stratum": "all"},
                            "data": DATA},
            "falsification_criterion": {"op": crit, "threshold": 0.0}, "alternative_explanations": alts}


def premise(text, source, how, path, d, depends=True):
    return {"premise": text, "source": source, "produced_by": "double_gui (WORK-2283)", "verified": True, "how": how,
            "evidence": {"path": path, "sha256": sha(d / path)}, "analysis_depends": depends}


def build(d, commit, review=None, verdict=None):
    c = A.FROZEN
    tree = (f"catalog tree root -> kind -> group -> [subgroup] -> item, {c['dim']}-d PCA descriptors, beam-{c['beam']} "
            f"descent by descriptor cosine to one landing node (land_all {c['land_all']})")
    rec = {
        "schema": "ngc-eval/study-record/v1",
        "study_id": STUDY,
        "title": "Hyperbolic hierarchical retrieval of rules, decisions and skills for agent tasks (WORK-2283)",
        "kind": "exploratory",
        "model": "none",
        "prereg_ref": f"design.md in this directory (sha256 {sha(d / 'design.md')}), the copy of cohesion-gated-taxonomy "
                      f"docs/catalog_retrieval/design.md with amendment v2 at commit {commit} (code and design), "
                      "written and reviewed before any test-split run",
        "notes": "Retrieval methods are under study, not a model. Fixed components: bge-m3 embeddings (local Ollama, "
                 "digest in premise_embeddings.md) for flat, flat_pca, hyp_tree and tree_cos; Claude Haiku 5.5 "
                 "(claude-haiku-5-5) for the llm_catalog control only. Gold is rule-based (real Skill calls, real "
                 "owner_decision citations), no model labels. Exploratory: 144 independent sessions cannot reach power "
                 "0.80 (design.md Power). The hierarchy gate failed (hierarchy_gate_result.json): hyp_tree is evidence "
                 "about an imposed hierarchy only, and no real-edge hyperbolic arm is claimed. The descriptive reading "
                 "(POSITIVE, NEGATIVE, INCONCLUSIVE) is computed by score.py reading() at the commit above.",
        "arms": ["hyp_tree", "flat", "bm25", "llm_catalog", "popularity", "tree_cos", "flat_pca", "random"],
        "arm_specs": {
            "hyp_tree": f"{tree}; the tree sits in a {c['ball_dim']}-d Poincare ball (each child at hyperbolic distance "
                        f"tau {c['tau']} from its parent, semantic child directions, no training); the query is inserted "
                        "as a child of the landing node and every item is ranked by its unrounded Poincare distance to "
                        "the query point",
            "flat": "cosine similarity on the full 1024-d bge-m3 vectors",
            "bm25": f"Okapi BM25 with k1 {c['k1']} and b {c['b']} on name plus first 40 description words",
            "llm_catalog": "Claude Haiku 5.5 reads the whole catalog as its system prompt (one numbered line per item) and "
                           "returns 10 row numbers; the ranking is its picks only",
            "popularity": "items ordered by gold frequency in the dev split",
            "tree_cos": f"{tree}; items ranked by tree hop distance from the landing node, ties by {c['dim']}-d cosine "
                        "(no hyperbolic geometry)",
            "flat_pca": f"cosine similarity on the {c['dim']}-d PCA projection fit on the catalog",
            "random": "a seeded random permutation of the catalog per task",
        },
        "sets": {SET: {"n_units": N_UNITS, "unit": "one test-split Claude session (one task per session)"}},
        "question": "Does a hyperbolic hierarchical index of rules, decisions and skills put the gold item of a real "
                    "agent task in its top 10 more often than flat bge-m3 cosine?",
        "purpose": "model_selection",
        "hypotheses": [
            hyp("H1", "The hyperbolic hierarchical index finds the first gold item in its top 10 more often than flat "
                      "cosine.", "hyp_tree", "flat",
                ["a gain comes from the tree, not the hyperbolic geometry (H5)",
                 "a loss comes from a wrong landing node in the descent (H7) or from the dimension cut (H6)",
                 "the hierarchy is imposed (Ward clusters and decision domains), so a loss is not evidence about a "
                 "real-edge hierarchy (the hierarchy gate failed)", LABEL_BIAS]),
            hyp("H2", "The hyperbolic index beats BM25 on the same outcome.", "hyp_tree", "bm25",
                ["lexical overlap between briefs and decision texts favours BM25", LABEL_BIAS]),
            hyp("H3", "The hyperbolic index beats an LLM that reads the whole catalog.", "hyp_tree", "llm_catalog",
                ["the cheapest Claude tier underrates the LLM approach", LABEL_BIAS]),
            hyp("H4", "The hyperbolic index beats the popularity control.", "hyp_tree", "popularity",
                ["a few frequent items dominate the gold"]),
            hyp("H5", "The Poincare geometry adds hit@10 beyond the explicit tree it encodes.", "hyp_tree", "tree_cos",
                [f"the two arms share the tree, the descent and the landing node; hyp_tree orders the items by distance "
                 f"in the {c['ball_dim']}-d ball, tree_cos by hop count with {c['dim']}-d cosine ties, so a loss can "
                 "come from the ball's lower dimension and not from the hyperbolic metric"]),
            hyp("H6", f"Reducing the embedding from 1024 to {c['dim']} dimensions loses hit@10.", "flat_pca", "flat",
                ["the PCA may keep the retrieval signal"], op="<", crit=">="),
            hyp("H7", "The tree descent and the ball ranking add hit@10 over cosine in the same reduced space.",
                "hyp_tree", "flat_pca",
                ["a loss is tree pruning: the gold item is outside the landing node's group", LABEL_BIAS]),
        ],
        "premise_audit": [
            premise("bge-m3 embeddings are distinct and meaningful for catalog and task texts, and the cached item "
                    "vectors come from the recorded model digest", "local Ollama bge-m3",
                    "collapse, paraphrase, bilingual, duplicate-vector and nearest-neighbour checks; a fresh embedding "
                    "of all items equals the cache", "premise_embeddings.md", d),
            premise("the ball positions keep every child at tau from its parent, preserve the tree order, fit float32 "
                    "and match the pg_hyperbolic distance", "retrieval.py build_ball",
                    "children at tau, order agreement, largest norm, pg_hyperbolic cross-check, hermetic tests",
                    "premise_tree.md", d),
            premise("the gold labels mark items an agent should read and the texts do not name them",
                    "tasks.jsonl", "automated leak scan for names, subjects and subject fragments on key v3, and three "
                    "20-task hand checks", "premise_key_handcheck.md", d),
            premise("the test split was unseen when the design was fixed", "results directory and embedding caches",
                    "listing of results and caches, tuning on dev only, split by session, the review's label-only reads",
                    "premise_heldout.md", d),
            premise("the primary units are independent sessions", "tasks.jsonl test split",
                    "unit sampling rule, gold concentration, sessions in both splits, near-duplicate pairs",
                    "premise_units.md", d),
            premise("no arm reads the test labels", "experiments/catalog_retrieval code",
                    "code read of every arm", "premise_arm_code.md", d),
        ],
        "answer_key": {"path": "tasks.jsonl", "sha256": sha(d / "tasks.jsonl"),
                       "version": f"key v3 (2026-11-25); catalog.jsonl sha256 {sha(d / 'catalog.jsonl')}"},
        "power": {"exploratory": True,
                  "reason": "144 independent sessions per arm cannot support a confirmatory claim: with the declared "
                            "threshold 0 a true difference of 0 gives FAIL with probability 0.027 (no n reaches FAIL "
                            "power 0.80), and one contrast has PASS power 0.41 at a true effect of 0.10, 0.73 at 0.15 "
                            "and 0.94 at 0.20 (ngc_eval.stats power, two_proportion, baseline 0.50); the joint AND "
                            "over the baseline and three controls is lower. The study reports estimates with 95% "
                            "intervals and no gate PASS or FAIL"},
        "baseline": {"arm": "flat", "description": "flat cosine on the full bge-m3 embedding, the simplest semantic method"},
        "control": {"arms": ["bm25", "llm_catalog", "popularity"],
                    "description": "the other simple methods (lexical BM25, an LLM reading the catalog) and a "
                                   "popularity prior: the treatment must beat the best of them"},
        "cost": {"unit": "USD per 1000 queries, marginal API spend (local CPU and GPU count as 0; every embedding arm "
                         "also pays one bge-m3 query embedding of about 0.34 s on the GPU, BM25 pays none)",
                 "per_arm": {"hyp_tree": 0.0, "flat": 0.0, "bm25": 0.0, "llm_catalog": LLM_USD_PER_1000,
                             "popularity": 0.0, "tree_cos": 0.0, "flat_pca": 0.0, "random": 0.0},
                 "source": "dev pilot on key v3, results/dev/llm_catalog_v3.jsonl (Haiku 5.5 total_cost_usd per call)"},
    }
    if review:
        rec["design_review"] = {"reviewer": "reviewer (Claude Code agent)", "verdict": verdict, "path": review,
                                "sha256": sha(d / review)}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study_dir")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--review")
    ap.add_argument("--verdict")
    a = ap.parse_args()
    if A.FROZEN is None:
        raise SystemExit("run_arms.FROZEN is not set: freeze the configuration from the dev grid first")
    d = Path(a.study_dir)
    json.dump(build(d, a.commit, a.review, a.verdict), open(d / "record.json", "w"), indent=1)
    print(d / "record.json")


if __name__ == "__main__":
    main()
