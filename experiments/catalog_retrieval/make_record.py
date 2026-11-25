#!/usr/bin/env python3
"""WORK-2283: write the ngc-eval study record (schema ngc-eval/study-record/v1) for the study directory.

  python3 make_record.py <study_dir> [--review design_review.md --verdict MERGE]

Every sha256 is computed from the files in <study_dir>; nothing is typed by hand.
"""
import argparse
import hashlib
import json
from pathlib import Path

STUDY = "catalog-retrieval-hyperbolic-01"
SET = "test_sessions"
DATA = ("test split, one task per session (smallest sha256 of 'work-2283-unit:' + task id), first gold item in the "
        "top 10; key v2")


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


def build(d, review=None, verdict=None):
    rec = {
        "schema": "ngc-eval/study-record/v1",
        "study_id": STUDY,
        "title": "Hyperbolic hierarchical retrieval of rules, decisions and skills for agent tasks (WORK-2283)",
        "kind": "exploratory",
        "model": "none",
        "prereg_ref": "design.md in this directory (cohesion-gated-taxonomy docs/catalog_retrieval/design.md), "
                      "written before any test-split run",
        "notes": "Retrieval methods are under study, not a model. Fixed components: bge-m3 embeddings (local Ollama) "
                 "for flat, flat_pca, hyp_tree and tree_cos; Claude Haiku 5.5 (claude-haiku-5-5) for the "
                 "llm_catalog control only. Gold is rule-based (real Skill calls, real owner_decision citations), no "
                 "model labels. Exploratory: 146 independent sessions cannot reach power 0.80 (design.md Power).",
        "arms": ["hyp_tree", "flat", "bm25", "llm_catalog", "popularity", "tree_cos", "flat_pca", "random"],
        "arm_specs": {
            "hyp_tree": "Sarkar-constructed Poincare-disk catalog tree (tau 6.921, seriated siblings), 128-d PCA query, "
                        "beam-6 descent, rank by Poincare distance to the nearest landing node rounded to 2 decimals, "
                        "ties by 128-d cosine",
            "flat": "cosine similarity on the full 1024-d bge-m3 vectors",
            "bm25": "Okapi BM25 with k1 1.5 and b 0.75 on name plus first 40 description words",
            "llm_catalog": "Claude Haiku 5.5 reads the whole catalog as its system prompt and returns 10 ids",
            "popularity": "items ordered by gold frequency in the dev split",
            "tree_cos": "the hyp_tree tree and beam-6 descent ranked by explicit tree hop distance, ties by 128-d cosine",
            "flat_pca": "cosine similarity on the 128-d PCA projection fit on the catalog",
            "random": "a seeded random permutation of the catalog per task",
        },
        "sets": {SET: {"n_units": 146, "unit": "one test-split Claude session (one task per session)"}},
        "question": "Does a hyperbolic hierarchical index of rules, decisions and skills put the gold item of a real "
                    "agent task in its top 10 more often than flat bge-m3 cosine?",
        "purpose": "model_selection",
        "hypotheses": [
            hyp("H1", "The hyperbolic hierarchical index finds the first gold item in its top 10 more often than flat "
                      "cosine.", "hyp_tree", "flat",
                ["a gain comes from the tree, not the hyperbolic geometry (H5)",
                 "a loss comes from wrong branch choices in the descent or from the dimension cut (H6)"]),
            hyp("H2", "The hyperbolic index beats BM25 on the same outcome.", "hyp_tree", "bm25",
                ["lexical overlap between briefs and decision texts favours BM25"]),
            hyp("H3", "The hyperbolic index beats an LLM that reads the whole catalog.", "hyp_tree", "llm_catalog",
                ["the cheapest Claude tier underrates the LLM approach"]),
            hyp("H4", "The hyperbolic index beats the popularity control.", "hyp_tree", "popularity",
                ["a few frequent items dominate the gold"]),
            hyp("H5", "The Poincare geometry adds hit@10 beyond the explicit tree it encodes.", "hyp_tree", "tree_cos",
                ["with exact tree encoding the two rankings differ only inside a ring"]),
            hyp("H6", "Reducing the embedding from 1024 to 128 dimensions loses hit@10.", "flat_pca", "flat",
                ["PCA keeps 77% of the variance and may keep the retrieval signal"], op="<", crit=">="),
        ],
        "premise_audit": [
            premise("bge-m3 embeddings are distinct and meaningful for catalog and task texts", "local Ollama bge-m3",
                    "collapse, paraphrase, bilingual, duplicate-vector and nearest-neighbour checks", "premise_embeddings.md", d),
            premise("the Sarkar positions preserve the catalog tree order and exact ties", "retrieval.py sarkar",
                    "order agreement, float64 tie spread before and after rounding, ring gap, hermetic tests",
                    "premise_tree.md", d),
            premise("the gold labels mark items an agent should read and the texts do not name them",
                    "tasks.jsonl", "automated leak scan and two 20-task hand checks", "premise_key_handcheck.md", d),
            premise("the test split was unseen when the design was fixed", "results directory and embedding caches",
                    "listing of results and caches, tuning on dev only, split by session", "premise_heldout.md", d),
            premise("the primary units are independent sessions", "tasks.jsonl test split",
                    "unit sampling rule, gold concentration, sessions in both splits", "premise_units.md", d),
            premise("no arm reads the test labels", "experiments/catalog_retrieval code",
                    "code read of every arm", "premise_arm_code.md", d),
        ],
        "answer_key": {"path": "tasks.jsonl", "sha256": sha(d / "tasks.jsonl"),
                       "version": f"key v2 (2026-11-25); catalog.jsonl sha256 {sha(d / 'catalog.jsonl')}"},
        "power": {"exploratory": True,
                  "reason": "146 independent sessions per arm: FAIL power 0.80 needs at least 169 units at margin 0.15 "
                            "and PASS power 0.80 needs a true effect near 0.30 (n-for-power, baseline 0.58)"},
        "baseline": {"arm": "flat", "description": "flat cosine on the full bge-m3 embedding, the simplest semantic method"},
        "control": {"arms": ["bm25", "llm_catalog", "popularity"],
                    "description": "the other simple methods (lexical BM25, an LLM reading the catalog) and a "
                                   "popularity prior: the treatment must beat the best of them"},
        "cost": {"unit": "USD per 1000 queries, marginal API spend (local CPU and GPU count as 0)",
                 "per_arm": {"hyp_tree": 0.0, "flat": 0.0, "bm25": 0.0, "llm_catalog": 7.56, "popularity": 0.0,
                             "tree_cos": 0.0, "flat_pca": 0.0, "random": 0.0},
                 "source": "dev pilot, results/dev/summary.json (Haiku 5.5 total_cost_usd per call)"},
    }
    if review:
        rec["design_review"] = {"reviewer": "reviewer (Claude Code agent)", "verdict": verdict, "path": review,
                                "sha256": sha(d / review)}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study_dir")
    ap.add_argument("--review")
    ap.add_argument("--verdict")
    a = ap.parse_args()
    d = Path(a.study_dir)
    json.dump(build(d, a.review, a.verdict), open(d / "record.json", "w"), indent=1)
    print(d / "record.json")


if __name__ == "__main__":
    main()
