#!/usr/bin/env python3
"""Assign every reel card to 1-3 topics of reel_topics_v1.json with a local model.

Read-only over reel_extract; output to ~/data/reel-kind-emergence/ and MLflow.
Every answer is validated against the vocabulary; an invalid batch is retried one
reel at a time and any reel still invalid is recorded as an error, never guessed.
Cross-method check: purity of the clusters-B topics under these labels.
"""
import datetime, glob, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from reels_emerge import DOOR, NAME_MODEL, SEED, post, psql_json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.expanduser("~/data/reel-kind-emergence")
BATCH = 10

VOCAB = json.load(open(os.path.join(HERE, "reel_topics_v1.json")))
LEAVES = {}
for t in VOCAB["topics"]:
    if t["sub"]:
        for s in t["sub"]:
            LEAVES[s["topic"]] = (t["topic"], s["label"], s["description"])
    else:
        LEAVES[t["topic"]] = (t["topic"], t["label"], t["description"])

SQL = """select json_agg(json_build_object('shortcode', shortcode, 'creator', creator, 'gist', gist))
         from reel_extract.card where gist is not null and origin_class <> 'fixture'"""

PROMPT = """Assign each saved Instagram reel to the SUBJECT topics it is about. Use only these topic ids:
{vocab}

Rules: 1 to 3 topics per reel, the main subject first. Judge the subject, not the form
(a meme about AI is ai_agents or coding_repos; a joke about Brazilian life is brazil_culture).
Reels:
{reels}

Answer with JSON only: {{"assignments": [{{"id": <reel number>, "topics": ["<topic id>", ...]}}, ...]}}"""


def vocab_txt():
    return "\n".join(f"- {k}: {d}" for k, (_, _, d) in LEAVES.items())


def ask(batch):
    reels = "\n".join(f"{i + 1}. {c['gist']}" for i, c in enumerate(batch))
    r = post("/api/chat", {"model": NAME_MODEL, "stream": False, "format": "json",
                           "options": {"temperature": 0, "seed": SEED, "num_ctx": 8192},
                           "messages": [{"role": "user", "content": PROMPT.format(vocab=vocab_txt(), reels=reels)}]})
    out = json.loads(r["message"]["content"]).get("assignments", [])
    got = {}
    for a in out:
        ts = [t for t in a.get("topics", []) if t in LEAVES][:3]
        if len(ts) > 1 and "unknown" in ts:  # unknown is valid only as the single topic
            a["topics"] = [t for t in a["topics"] if t != "unknown"]
            ts = [t for t in ts if t != "unknown"]
        if isinstance(a.get("id"), int) and 1 <= a["id"] <= len(batch) and ts and len(ts) == len(a.get("topics", [])[:3]):
            got[batch[a["id"] - 1]["shortcode"]] = ts
    return got


def assign(batch):
    try:
        got = ask(batch)
    except Exception:
        got = {}
    errors = {}
    for c in batch:
        if c["shortcode"] in got:
            continue
        try:
            one = ask([c])
            if c["shortcode"] in one:
                got[c["shortcode"]] = one[c["shortcode"]]
            else:
                errors[c["shortcode"]] = "invalid answer"
        except Exception as e:
            errors[c["shortcode"]] = f"{type(e).__name__}: {e}"
    return got, errors


def cluster_purity(labels):
    f = sorted(glob.glob(os.path.join(OUT_DIR, "topics-B-*.json")))[-1]
    tree = json.load(open(f))["tree"]
    rows = []
    for n in tree:
        prim = [LEAVES[labels[m][0]][0] for m in n["members"] if m in labels]
        if not prim:
            continue
        vals, cnt = np.unique(prim, return_counts=True)
        rows.append((n["naming"]["label"], len(prim), str(vals[cnt.argmax()]), cnt.max() / len(prim)))
    total = sum(r[1] for r in rows)
    return sum(r[1] * r[3] for r in rows) / total, rows


def main():
    t0 = time.time()
    cards = psql_json(SQL)
    batches = [cards[i:i + BATCH] for i in range(0, len(cards), BATCH)]
    labels, errors = {}, {}
    with ThreadPoolExecutor(2) as ex:
        for got, err in ex.map(assign, batches):
            labels.update(got)
            errors.update(err)
    purity, rows = cluster_purity(labels)
    n_top = {}
    for ts in labels.values():
        n_top[LEAVES[ts[0]][0]] = n_top.get(LEAVES[ts[0]][0], 0) + 1
    metrics = {"n_cards": len(cards), "labelled": len(labels), "errors": len(errors),
               "mean_topics_per_reel": round(float(np.mean([len(v) for v in labels.values()])), 3),
               "clusterB_purity_under_llm_primary_top": round(purity, 4),
               "seconds": round(time.time() - t0, 1), "cost_usd": 0.0}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"topic-assign-{datetime.datetime.now():%Y-%m-%dT%H%M%S}.json")
    json.dump({"vocab": VOCAB["version"], "model": NAME_MODEL, "metrics": metrics, "labels": labels,
               "errors": errors, "cluster_rows": rows}, open(path, "w"), indent=1, ensure_ascii=False)
    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:8591")
    mlflow.set_experiment("reel-kind-emergence")
    with mlflow.start_run(run_name=f"topic-assign-{VOCAB['version']}-{NAME_MODEL}") as r:
        mlflow.log_params({"vocab": VOCAB["version"], "name_model": NAME_MODEL, "batch": BATCH, "seed": SEED, "door": DOOR})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(path)
        rid = r.info.run_id
    print(json.dumps({"mlflow_run": rid, "out": path, **metrics}, indent=1))
    print("primary top-level topic:", dict(sorted(n_top.items(), key=lambda x: -x[1])))
    for lab, n, top, p in rows:
        print(f"  B:{lab:22.22} n={n:4d} dominant={top:18} share={p:.2f}")
    if errors:
        print("errors:", list(errors.items())[:5])
        sys.exit(1)


if __name__ == "__main__":
    main()
