#!/usr/bin/env python3
"""Anchored kind assignment for reel cards: nearest kind anchor by cosine, no LLM.

Read-only over reel_extract. Local embeddings through the Barn Dance door.
Reference = the jev classifier's latest kind on cards it did NOT put in `other`
(not gold; it measures whether embedding can replace the LLM call for this job).

Goal (declared before the run): agreement with jev >= 0.85 on those cards
(majority-class baseline = share of the largest kind). Below that, the anchor
method does not replace the classifier.

Anchors:
  text      - embedding of "label: description" for each kind (zero-shot)
  prototype - centroid of the cards jev gave that kind, scored 5-fold so a card
              never sits in the anchor that scores it
"""
import json, sys, time

import numpy as np

from reels_emerge import DOOR, SEED, post, psql_json

EMBED_MODEL = "nomic-embed-text"
PREFIX = "classification: "

SQL = """
with top as (
  select distinct on (shortcode) shortcode, label
  from reel_extract.card_suggestion where field = 'kind'
  order by shortcode, created_at desc, p_model desc)
select json_agg(json_build_object('shortcode', c.shortcode, 'gist', c.gist, 'jev', t.label))
from top t join reel_extract.card c using (shortcode)
where c.gist is not null and c.origin_class <> 'fixture'
"""
KINDS_SQL = """select json_agg(json_build_object('kind', kind, 'label', label, 'description', description))
               from reel_extract.kind_option where active and kind <> 'other'"""


def embed(texts):
    vecs = []
    for i in range(0, len(texts), 64):
        vecs += post("/api/embed", {"model": EMBED_MODEL,
                                    "input": [PREFIX + t.lower()[:1500] for t in texts[i:i + 64]]})["embeddings"]
    A = np.array(vecs, dtype=np.float32)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def main():
    t0 = time.time()
    cards = psql_json(SQL)
    kinds = psql_json(KINDS_SQL)
    names = [k["kind"] for k in kinds]
    V = embed([c["gist"] for c in cards])
    A_text = embed([f"{k['label']}: {k['description'] or k['label']}" for k in kinds])

    jev = np.array([c["jev"] for c in cards])
    ref = np.where(jev != "other")[0]
    other = np.where(jev == "other")[0]
    y = jev[ref]
    majority = max(np.mean(y == k) for k in set(y))

    pred_text = np.array(names)[np.argmax(V[ref] @ A_text.T, 1)]
    acc_text = float(np.mean(pred_text == y))

    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, 5, len(ref))
    pred_proto = np.empty(len(ref), dtype=object)
    for f in range(5):
        tr, te = ref[fold != f], ref[fold == f]
        ks = sorted(set(jev[tr]))
        P = np.stack([unit(V[tr][jev[tr] == k].mean(0)) for k in ks])
        pred_proto[fold == f] = np.array(ks)[np.argmax(V[te] @ P.T, 1)]
    acc_proto = float(np.mean(pred_proto == y))

    per_kind = {k: {"n": int(np.sum(y == k)),
                    "text": round(float(np.mean(pred_text[y == k] == k)), 3),
                    "prototype": round(float(np.mean(pred_proto[y == k] == k)), 3)} for k in sorted(set(y))}
    other_text = dict(zip(*np.unique(np.array(names)[np.argmax(V[other] @ A_text.T, 1)], return_counts=True)))
    metrics = {"n_ref": len(ref), "n_other": len(other), "majority_baseline": round(float(majority), 4),
               "acc_text_anchor": round(acc_text, 4), "acc_prototype_anchor": round(acc_proto, 4),
               "seconds": round(time.time() - t0, 1), "cost_usd": 0.0}

    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:8591")
    mlflow.set_experiment("reel-kind-emergence")
    with mlflow.start_run(run_name="anchor-nearest-nomic") as run:
        mlflow.log_params({"method": "nearest_anchor", "embed_model": EMBED_MODEL, "prefix": PREFIX,
                           "reference": "jev latest kind, non-other", "folds": 5, "seed": SEED, "door": DOOR})
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"per_kind": per_kind, "other_text_anchor": {k: int(v) for k, v in other_text.items()}},
                        "anchor_detail.json")
        print(json.dumps({"mlflow_run": run.info.run_id, **metrics, "per_kind": per_kind,
                          "other_by_text_anchor": {k: int(v) for k, v in other_text.items()}}, indent=1))


if __name__ == "__main__":
    main()
