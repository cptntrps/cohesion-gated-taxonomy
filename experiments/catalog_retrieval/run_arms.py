#!/usr/bin/env python3
"""WORK-2283: run the retrieval arms on one split and write per-task results.

  python3 run_arms.py <data_dir> <split> [--dim D --beam B] [--arms a,b,...] [--tune]

Item text (every arm sees the same): "<name>: <first ITEM_WORDS words of the description>".
Task text: the key's text (already redacted and truncated). Embeddings: bge-m3 via the local Ollama.
Per task and arm the result row holds the rank of the first gold item, hit@5, hit@10, RR (rank cut at 50),
recall@5 and recall@10 over all gold, and the ranking latency. The LLM arm is llm_arm.py.
"""
import argparse
import hashlib
import json
import math
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

import retrieval as R

EMBED_MODEL = "bge-m3:latest"
OLLAMA = "http://127.0.0.1:11434/api/embed"
ITEM_WORDS = 40
CUT = 50
ARMS = ("bm25", "flat", "flat_pca", "hyp_tree", "tree_cos", "popularity", "random")


def item_text(c):
    name = c["name"] if c["kind"] == "decision" else c["name"].replace("-", " ").replace("_", " ")
    return f"{name}: {' '.join(c['description'].split()[:ITEM_WORDS])}"


def embed(texts, cache):
    if cache.exists():
        z = np.load(cache)
        if z["sha"].item() == hashlib.sha256("\x00".join(texts).encode()).hexdigest():
            return z["V"]
    out = []
    for i in range(0, len(texts), 16):
        req = urllib.request.Request(OLLAMA, data=json.dumps({"model": EMBED_MODEL, "input": texts[i:i + 16]}).encode(),
                                     headers={"Content-Type": "application/json"})
        out += json.load(urllib.request.urlopen(req, timeout=300))["embeddings"]
    V = np.array(out, dtype=np.float32)
    if len(V) != len(texts) or not np.isfinite(V).all():
        raise RuntimeError("embedding failed: row count or values are wrong")
    np.savez(cache, V=V, sha=hashlib.sha256("\x00".join(texts).encode()).hexdigest())
    return V


def load(d):
    cat = [json.loads(l) for l in open(d / "catalog.jsonl")]
    tasks = [json.loads(l) for l in open(d / "tasks.jsonl")]
    return cat, tasks


def popularity_scores(cat, tasks, d):
    """Gold frequency in the DEV split only (no test labels), ties by kind share in dev."""
    pos = {c["id"]: i for i, c in enumerate(cat)}
    cnt = Counter(g for t in tasks if t["split"] == "dev" for g in t["gold"])
    kind_share = Counter(t["kind"] for t in tasks if t["split"] == "dev")
    return np.array([cnt.get(c["id"], 0) + 1e-3 * kind_share.get(c["kind"], 0) for c in cat], dtype=float)


def score_row(t, arm, ranking, gidx, ms):
    fr = R.first_rank(ranking[:CUT], [gidx[0]]) if gidx[0] in set(int(i) for i in ranking[:CUT]) else None
    return {"tid": t["tid"], "session": t["session"], "kind": t["kind"], "arm": arm, "first_rank": fr,
            "hit5": int(fr is not None and fr <= 5), "hit10": int(fr is not None and fr <= 10),
            "rr": (1.0 / fr) if fr else 0.0, "recall5": R.recall_at(ranking, gidx, 5),
            "recall10": R.recall_at(ranking, gidx, 10), "ms": ms, "top10": [int(i) for i in ranking[:10]]}


def run(d, split, dim, beam, arms, write=True):
    cat, tasks = load(d)
    pos = {c["id"]: i for i, c in enumerate(cat)}
    tasks = [t for t in tasks if t["split"] == split]
    texts = [item_text(c) for c in cat]
    Vi = R.unit(embed(texts, d / "emb_items.npz"))
    Vq = R.unit(embed([t["text"] for t in tasks], d / f"emb_tasks_{split}.npz"))
    kinds, groups = [c["kind"] for c in cat], [c["group"] for c in cat]
    tree = R.build_tree(kinds, groups, Vi)
    tau = R.sala_tau(tree)
    idx = R.HypIndex(tree, Vi, dim=dim, tau=tau, beam=beam)
    bm = R.BM25(texts)
    pop = popularity_scores(cat, load(d)[1], d)
    rows = []
    for t, qv in zip(tasks, Vq):
        gidx = [pos[g] for g in t["gold"]]
        for arm in arms:
            t0 = time.perf_counter()
            if arm == "bm25":
                rk = R.rank_by(bm.scores(t["text"]))
            elif arm == "flat":
                rk = R.rank_by(Vi @ qv)
            elif arm == "flat_pca":
                rk = R.rank_by(idx.Xd @ idx.reduce(qv))
            elif arm == "hyp_tree":
                rk, _ = idx.rank(qv, "poincare")
            elif arm == "tree_cos":
                rk, _ = idx.rank(qv, "hops")
            elif arm == "popularity":
                rk = R.rank_by(pop)
            elif arm == "random":
                seed = int(hashlib.sha256(f"work-2283:{t['tid']}".encode()).hexdigest()[:8], 16)
                rk = np.random.default_rng(seed).permutation(len(cat))
            ms = (time.perf_counter() - t0) * 1000
            rows.append(score_row(t, arm, rk, gidx, ms))
    meta = {"split": split, "dim": dim, "beam": beam, "tau": tau, "tree_nodes": len(tree), "n_items": len(cat),
            "n_tasks": len(tasks), "order_agreement": R.order_agreement(tree, idx.pos),
            "pca_explained": float(idx.pca.explained_variance_ratio_.sum())}
    if write:
        out = d / "results" / split
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"arms_dim{dim}_beam{beam}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        json.dump(meta, open(out / f"arms_dim{dim}_beam{beam}.meta.json", "w"), indent=1)
    return rows, meta


def summary(rows):
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    return {a: {"n": len(v), "hit10": np.mean([r["hit10"] for r in v]), "hit5": np.mean([r["hit5"] for r in v]),
                "mrr": np.mean([r["rr"] for r in v]), "recall10": np.mean([r["recall10"] for r in v]),
                "ms": np.median([r["ms"] for r in v])} for a, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("split", choices=("dev", "test"))
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--tune", action="store_true", help="dev only: grid over dim and beam for hyp_tree")
    a = ap.parse_args()
    d = Path(a.data_dir)
    if a.tune:
        if a.split != "dev":
            raise SystemExit("tuning runs on dev only")
        grid = {}
        for dim in (16, 32, 64, 128):
            for beam in (1, 2, 3, 4, 6):
                rows, meta = run(d, "dev", dim, beam, ["hyp_tree", "tree_cos", "flat_pca"], write=False)
                s = summary(rows)
                grid[f"dim{dim}_beam{beam}"] = {k: v["hit10"] for k, v in s.items()}
                print(dim, beam, {k: round(v["hit10"], 3) for k, v in s.items()}, flush=True)
        json.dump(grid, open(d / "results" / "dev_tuning_grid.json", "w"), indent=1)
        return
    rows, meta = run(d, a.split, a.dim, a.beam, a.arms.split(","))
    print(json.dumps(meta, indent=1))
    for arm, s in summary(rows).items():
        print(f"{arm:11s} n={s['n']} hit10={s['hit10']:.3f} hit5={s['hit5']:.3f} mrr={s['mrr']:.3f} "
              f"recall10={s['recall10']:.3f} ms={s['ms']:.2f}")


if __name__ == "__main__":
    main()
