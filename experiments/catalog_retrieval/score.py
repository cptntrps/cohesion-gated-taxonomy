#!/usr/bin/env python3
"""WORK-2283 scoring (fixed before the test run).

  python3 score.py <data_dir> <split> --arms-file arms_dim128_beam6.jsonl [--results-out results.json] [--mlflow]

Primary (the study record): one task per session, the one with the smallest sha256("work-2283-unit:" + tid);
field hit10 = the first gold item is in the arm's top 10. Raw counts of sessions go to results.json in the
ngc-eval study-record shape (verdicts and intervals are computed by ngc_eval.method score-record, never here).
Secondary (estimate only): every task of the split; hit5, hit10, MRR, recall5, recall10; 95% intervals by the
session-cluster bootstrap of ngc_eval.stats; paired per-task differences of hyp_tree minus each arm with the same
bootstrap; the same by kind (skill, decision) and by task type (skill_call, security_review, decision_brief);
latency (median, p95 ms) and cost per 1000 queries.
An arm with more than 10% failed rows did not run (its hypotheses are not_run); failed rows below that count as
misses.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "src/products/ngc-eval"))
from ngc_eval import stats  # noqa: E402

TREAT, BASE = "hyp_tree", "flat"
CONTROLS = ("bm25", "llm_catalog", "popularity")
SECONDARY = (("H5", TREAT, "tree_cos"), ("H6", "flat_pca", "flat"))
METRICS = ("hit5", "hit10", "rr", "recall5", "recall10")
MISSING_LIMIT = 0.10


def unit_task(tasks):
    pick = {}
    for t in tasks:
        h = hashlib.sha256(f"work-2283-unit:{t['tid']}".encode()).hexdigest()
        if t["session"] not in pick or h < pick[t["session"]][0]:
            pick[t["session"]] = (h, t["tid"])
    return {tid for _, tid in pick.values()}


def load_rows(d, split, arms_file):
    key = [json.loads(l) for l in open(d / "tasks.jsonl") if json.loads(l)["split"] == split]
    tids = {t["tid"] for t in key}
    rows = {}
    for f in (d / "results" / split / arms_file, d / "results" / split / "llm_catalog.jsonl"):
        if not f.exists():
            continue
        for l in open(f):
            r = json.loads(l)
            if r["tid"] in tids:
                rows.setdefault(r["arm"], {})[r["tid"]] = r
    return key, rows


def primary_counts(key, rows):
    units = unit_task(key)
    out, status = {}, {}
    for arm, by in rows.items():
        got = [by.get(t) for t in sorted(units)]
        failed = sum(1 for r in got if r is None or r.get("failed"))
        status[arm] = {"n": len(units), "failed": failed}
        if failed / len(units) > MISSING_LIMIT:
            out[arm] = None
            continue
        out[arm] = [sum(int(r["hit10"]) for r in got if r and not r.get("failed")), len(units)]
    return out, status, len(units)


def record_results(counts):
    res = {}
    pairs = [("H1", TREAT, BASE)] + [(f"H{i + 2}", TREAT, c) for i, c in enumerate(CONTROLS)] + list(SECONDARY)
    for hid, a, b in pairs:
        if counts.get(a) is None or counts.get(b) is None:
            bad = [x for x in (a, b) if counts.get(x) is None]
            res[hid] = {"not_run": f"arm(s) {bad} missing or above the 10% failed-row limit"}
        else:
            res[hid] = {"kind": "difference", "a": counts[a], "b": counts[b]}
    return res


def task_type(t):
    if t["kind"] == "skill":
        return "skill_call"
    return "security_review" if t["text"].startswith("Review this change for security") else "decision_brief"


def secondary(key, rows, kind=None, ttype=None):
    tasks = [t for t in key if (kind is None or t["kind"] == kind) and (ttype is None or task_type(t) == ttype)]
    sess = [t["session"] for t in tasks]
    out = {}
    for arm, by in rows.items():
        m = {}
        for met in METRICS:
            v = [float((by.get(t["tid"]) or {}).get(met, 0.0)) for t in tasks]
            m[met] = {"est": float(np.mean(v)), "ci95": stats.cluster_bootstrap_ci(v, sess)}
        ms = [r["ms"] for r in by.values() if "ms" in r and not r.get("failed")]
        m["latency_ms"] = {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95))} if ms else None
        costs = [r.get("cost_usd", 0.0) for r in by.values() if not r.get("failed")]
        m["usd_per_1000"] = 1000 * float(np.mean(costs)) if costs else 0.0
        m["failed"] = sum(1 for t in tasks if (by.get(t["tid"]) or {"failed": True}).get("failed"))
        out[arm] = m
    diffs = {}
    if TREAT in rows:
        for arm in rows:
            if arm == TREAT:
                continue
            for met in ("hit10", "rr"):
                v = [float(rows[TREAT].get(t["tid"], {}).get(met, 0.0)) - float(rows[arm].get(t["tid"], {}).get(met, 0.0))
                     for t in tasks]
                diffs[f"{TREAT}-{arm}:{met}"] = {"est": float(np.mean(v)), "ci95": stats.cluster_bootstrap_ci(v, sess)}
    return {"n_tasks": len(tasks), "n_sessions": len(set(sess)), "arms": out, "paired_diffs": diffs}


def log_mlflow(split, summary, meta):
    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:8591")
    mlflow.set_experiment("work-2283-catalog-retrieval")
    for arm, m in summary["all"]["arms"].items():
        with mlflow.start_run(run_name=f"{split}:{arm}"):
            mlflow.log_params({"split": split, "arm": arm, **{k: v for k, v in meta.items() if k != "split"}})
            for met in METRICS:
                mlflow.log_metric(met, m[met]["est"])
                mlflow.log_metric(f"{met}_ci_lo", m[met]["ci95"][0])
                mlflow.log_metric(f"{met}_ci_hi", m[met]["ci95"][1])
            if m["latency_ms"]:
                mlflow.log_metric("latency_ms_median", m["latency_ms"]["median"])
            mlflow.log_metric("cost_usd_per_1000", m["usd_per_1000"])
            p = summary["primary_counts"].get(arm)
            if p:
                mlflow.log_metric("primary_hit10_sessions", p[0] / p[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("split", choices=("dev", "test"))
    ap.add_argument("--arms-file", default="arms_dim128_beam6.jsonl")
    ap.add_argument("--results-out")
    ap.add_argument("--mlflow", action="store_true")
    a = ap.parse_args()
    d = Path(a.data_dir)
    key, rows = load_rows(d, a.split, a.arms_file)
    counts, status, n_units = primary_counts(key, rows)
    summary = {"split": a.split, "n_units": n_units, "primary_counts": counts, "arm_status": status,
               "all": secondary(key, rows), "by_kind": {k: secondary(key, rows, k) for k in ("skill", "decision")},
               "by_type": {k: secondary(key, rows, ttype=k) for k in ("skill_call", "security_review", "decision_brief")}}
    out = d / "results" / a.split / "summary.json"
    json.dump(summary, open(out, "w"), indent=1)
    if a.results_out:
        json.dump(record_results(counts), open(a.results_out, "w"), indent=1)
    if a.mlflow:
        meta = json.load(open(d / "results" / a.split / a.arms_file.replace(".jsonl", ".meta.json")))
        log_mlflow(a.split, summary, meta)
    print(f"units={n_units}")
    for arm, m in summary["all"]["arms"].items():
        c = counts.get(arm)
        print(f"{arm:11s} primary={c} hit10={m['hit10']['est']:.3f} {np.round(m['hit10']['ci95'], 3)} "
              f"mrr={m['rr']['est']:.3f} recall10={m['recall10']['est']:.3f} "
              f"ms={m['latency_ms']['median'] if m['latency_ms'] else None} usd/1k={m['usd_per_1000']:.2f}")


if __name__ == "__main__":
    main()
