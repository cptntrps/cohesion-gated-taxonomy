#!/usr/bin/env python3
"""WORK-2283: run the retrieval arms on one split and write per-task results.

  python3 run_arms.py <data_dir> dev --tune                 the dev grid and the selection rule (choose)
  python3 run_arms.py <data_dir> <split> --frozen           the frozen configuration (the only one for test)
  python3 run_arms.py <data_dir> dev [--dim D --ball_dim K --tau T --beam B --land_all 0|1 --k1 X --b Y]

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


def check_embedder(d, texts, n=16):
    """The live embedding model must be the one that made the item cache: embed n item texts again and compare."""
    cached = R.unit(np.load(d / "emb_items.npz")["V"][:n])
    req = urllib.request.Request(OLLAMA, data=json.dumps({"model": EMBED_MODEL, "input": texts[:n]}).encode(),
                                 headers={"Content-Type": "application/json"})
    fresh = R.unit(np.array(json.load(urllib.request.urlopen(req, timeout=300))["embeddings"], dtype=np.float32))
    worst = float(np.min(np.sum(cached * fresh, axis=1)))
    if worst < 0.9999:
        raise SystemExit(f"the live embedding model does not reproduce the item cache (smallest cosine {worst:.6f})")
    return worst


def load(d):
    cat = [json.loads(l) for l in open(d / "catalog.jsonl")]
    tasks = [json.loads(l) for l in open(d / "tasks.jsonl")]
    return cat, tasks


def check_frozen_key(d):
    """A test-split step reads <data_dir>/tasks.jsonl and catalog.jsonl: they must be the frozen copies that the
    registered study record names (study/record.json answer_key)."""
    rec = json.load(open(d / "study" / "record.json"))
    for name, want in (("tasks.jsonl", rec["answer_key"]["sha256"]),
                       ("catalog.jsonl", hashlib.sha256((d / "study" / "catalog.jsonl").read_bytes()).hexdigest())):
        got = hashlib.sha256((d / name).read_bytes()).hexdigest()
        if got != want or want not in json.dumps(rec["answer_key"]):
            raise SystemExit(f"{name} is not the frozen key of the study record ({got[:12]} != {want[:12]})")


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


DEFAULT = {"dim": 128, "ball_dim": 32, "tau": 2.5, "beam": 6, "land_all": 0, "k1": 1.5, "b": 0.75}
# The configuration of the test run: the output of choose() on results/dev_tuning_grid_v4.json (dev split, key v3),
# fixed before the design review of amendment v2. A test run takes no other configuration.
FROZEN = {"dim": 256, "ball_dim": 256, "tau": 1.5, "beam": 24, "land_all": 0, "k1": 2.5, "b": 0.75}


def tree_cells():
    """(dim, ball_dim, tau) of the dev grid. The base grid, plus the cells past its best region: the best base
    cells have ball_dim = dim, the largest dim and the smallest tau, so the grid goes on to dim 256 and tau 1.0
    with ball_dim = dim."""
    base = [(dim, ball, tau) for dim in (64, 128) for ball in (16, 32, 64, 128) if ball <= dim
            for tau in (1.5, 2.5, 3.5)]
    past = [(64, 64, 1.0), (128, 128, 1.0)] + [(256, 256, tau) for tau in (1.0, 1.5, 2.5, 3.5)]
    return base + past


def choose(grid):
    """The dev-only selection rule. hyp_tree: the cell with the highest dev hit@10 whose ball fits float32; ties go
    to the higher dev MRR, then the smaller beam, ball_dim and dim (the cheaper index), then tau nearest 2.5,
    then land_all 0. BM25: the highest dev hit@10; ties to the higher MRR, then the cell nearest (1.5, 0.75)."""
    cells = [v for v in grid["tree"].values() if v["fits_float32_ball"]]
    t = min(cells, key=lambda v: (-v["hyp_tree"]["hit10"], -v["hyp_tree"]["mrr"], v["cfg"]["beam"],
                                  v["cfg"]["ball_dim"], v["cfg"]["dim"], abs(v["cfg"]["tau"] - 2.5),
                                  v["cfg"]["land_all"]))
    bm = min(grid["bm25"].values(), key=lambda v: (-v["hit10"], -v["mrr"],
                                                   abs(v["cfg"]["k1"] - 1.5) + abs(v["cfg"]["b"] - 0.75)))
    return {**{k: t["cfg"][k] for k in ("dim", "ball_dim", "tau", "beam", "land_all")},
            "k1": bm["cfg"]["k1"], "b": bm["cfg"]["b"]}


def cfg_name(cfg):
    return "_".join(f"{k}{cfg[k]}" for k in sorted(cfg))


def model_digest():
    tags = json.load(urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=30))["models"]
    return next(m["digest"] for m in tags if m["name"] == EMBED_MODEL)


def run(d, split, cfg, arms, write=True, digest=None):
    cat, tasks = load(d)
    pos = {c["id"]: i for i, c in enumerate(cat)}
    tasks = [t for t in tasks if t["split"] == split]
    texts = [item_text(c) for c in cat]
    Vi = R.unit(embed(texts, d / "emb_items.npz"))
    Vq = R.unit(embed([t["text"] for t in tasks], d / f"emb_tasks_{split}.npz"))
    kinds, groups = [c["kind"] for c in cat], [c["group"] for c in cat]
    tree = R.build_tree(kinds, groups, Vi)
    idx = R.HypIndex(tree, Vi, dim=cfg["dim"], ball_dim=cfg["ball_dim"], tau=cfg["tau"], beam=cfg["beam"],
                      land_all=bool(cfg["land_all"]))
    bm = R.BM25(texts, k1=cfg["k1"], b=cfg["b"])
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
            row = score_row(t, arm, rk, gidx, ms)
            row["gold_fragment"] = bool(t.get("gold_fragment"))
            rows.append(row)
    r = np.linalg.norm(idx.pos, axis=1)
    meta = {"split": split, **cfg, "tree_nodes": len(tree), "n_items": len(cat), "n_tasks": len(tasks),
            "order_agreement": R.order_agreement(tree, idx.pos), "max_ball_norm": float(r.max()),
            "fits_float32_ball": bool(r.max() < 1 - 1e-7),
            "pca_explained": float(idx.pca.explained_variance_ratio_.sum()),
            "embed_model": EMBED_MODEL, "embed_digest": digest}
    if write:
        out = d / "results" / split
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"arms_{cfg_name(cfg)}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        json.dump(meta, open(out / f"arms_{cfg_name(cfg)}.meta.json", "w"), indent=1)
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
    for k, v in DEFAULT.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--tune", action="store_true", help="dev only: grid for hyp_tree, tree_cos, flat_pca and bm25")
    ap.add_argument("--frozen", action="store_true", help="use the frozen configuration (FROZEN)")
    a = ap.parse_args()
    d = Path(a.data_dir)
    digest = model_digest()
    cfg = dict(FROZEN) if a.frozen else {k: getattr(a, k) for k in DEFAULT}
    if a.tune:
        if a.split != "dev":
            raise SystemExit("tuning runs on dev only")
        out = d / "results" / "dev_tuning_grid_v4.json"
        grid = json.load(open(out)) if out.exists() else {"digest": digest, "tree": {}, "bm25": {}}
        if grid["digest"] != digest:
            raise SystemExit("the grid file comes from another embedding model digest: move it away and run again")
        for dim, ball_dim, tau in tree_cells():
            for land_all in (0, 1):
                for beam in (1, 2, 4, 6, 8, 12, 16, 24, 32):
                    c = dict(DEFAULT, dim=dim, ball_dim=ball_dim, tau=tau, beam=beam, land_all=land_all)
                    if cfg_name(c) in grid["tree"]:        # a finished cell of an earlier pass of this grid
                        continue
                    rows, meta = run(d, "dev", c, ["hyp_tree", "tree_cos", "flat_pca"], write=False, digest=digest)
                    s = summary(rows)
                    grid["tree"][cfg_name(c)] = {"cfg": c, "fits_float32_ball": meta["fits_float32_ball"],
                                                 **{k: {"hit10": float(v["hit10"]), "mrr": float(v["mrr"])}
                                                    for k, v in s.items()}}
                    print(cfg_name(c), {k: round(float(v["hit10"]), 3) for k, v in s.items()}, flush=True)
        for k1 in (0.9, 1.2, 1.5, 2.0, 2.5, 3.0):
            for b in (0.3, 0.5, 0.75, 0.9, 1.0):
                if f"k1{k1}_b{b}" in grid["bm25"]:
                    continue
                c = dict(DEFAULT, k1=k1, b=b)
                rows, _ = run(d, "dev", c, ["bm25"], write=False, digest=digest)
                s = summary(rows)["bm25"]
                grid["bm25"][f"k1{k1}_b{b}"] = {"cfg": c, "hit10": float(s["hit10"]), "mrr": float(s["mrr"])}
        grid["chosen"] = choose(grid)
        json.dump(grid, open(out, "w"), indent=1)
        print(json.dumps(grid["chosen"], indent=1))
        return
    if a.split == "test":
        check_frozen_key(d)
        if cfg != FROZEN:
            raise SystemExit(f"the test split runs only with the frozen configuration {FROZEN}")
        if (d / "results" / "test" / f"arms_{cfg_name(cfg)}.jsonl").exists():
            raise SystemExit("the test split has been run: it is scored once, and its result file is kept")
        print("embedder check, smallest cosine to the item cache:", check_embedder(d, [item_text(c) for c in load(d)[0]]))
    rows, meta = run(d, a.split, cfg, a.arms.split(","), digest=digest)
    print(json.dumps(meta, indent=1))
    for arm, s in summary(rows).items():
        print(f"{arm:11s} n={s['n']} hit10={s['hit10']:.3f} hit5={s['hit5']:.3f} mrr={s['mrr']:.3f} "
              f"recall10={s['recall10']:.3f} ms={s['ms']:.2f}")


if __name__ == "__main__":
    main()
