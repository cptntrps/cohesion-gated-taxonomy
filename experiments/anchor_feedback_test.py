"""Condition E: the anchor plus a human who picks which clauses match it in the company's own sense.

Pre-registered 2026-10-25 BEFORE the run. Owner's correction after A-D: "the idea is that once you set an anchor you
should pick which clauses match that anchor in your company lingo". A-D tested the machine alone; E tests the
machine plus a few human picks (relevance feedback).

  human       simulated: answers yes iff the clause's hidden LEDGAR type is the anchor's type (the label stands in
              for "what we call this here"). Exact type, no synonym families.
  anchors     the same 20 random LEDGAR names (seed 2026)
  loop        round 1 shows the 10 clauses closest to the embedded name; the human ticks yes/no. A logistic
              regression (768-d embeddings, class_weight balanced) learns ticked yes vs ticked no plus 1,000 random
              unticked clauses as presumed negatives. Each next round shows the 10 unjudged clauses it rates highest.
              No positive ticked yet: the next round keeps following the name ranking. Budgets: 10, 20, 40 picks.
  assign      every clause gets each anchor's probability; it joins the best anchor if that is >= 0.5. An anchor with
              no ticked positive has no model and stays empty.
  placebos    "patent prosecution", "crop insurance": the simulated human answers no to every candidate (no
              LEDGAR type is either), so each must end empty.

Gates at 20 picks per anchor (400 human decisions for 20 concepts):
  F1  median precision (exact type)                                    >= 0.70
  F2  median recall                                                     >= 0.50
  F3  clauses of the 80 unanchored types pulled into some anchor       <= 5%
  F4  placebos end empty                                                2 of 2
Diagnostics, not gates: the same at 10 and 40 picks; a random-candidate control (same budget, candidates drawn at
random instead of ranked by the anchor) to measure what the anchor name itself contributes. Judged clauses count
toward precision and recall like any other clause.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from anchor_test import N_REAL, SEED
from cohesion_taxonomy.build_tree import load
from cohesion_taxonomy.cluster import embed_names

ROUND = 10
BUDGETS = [10, 20, 40]
N_BG = 1000
PLACEBOS = ["patent prosecution", "crop insurance"]


def teach(V, name_vec, truth, budget, rng, ranked=True):
    """Return a fitted model (or None) after `budget` simulated human picks."""
    judged, ticks = [], []
    order = np.argsort(-(V @ name_vec)) if ranked else rng.permutation(len(V))
    model = None
    while len(judged) < budget:
        seen = set(judged)
        if model is None or not ranked:
            cand = [i for i in order if i not in seen][:ROUND]
        else:
            p = model.predict_proba(V)[:, 1]
            p[judged] = -1
            cand = np.argsort(-p)[:ROUND].tolist()
        for i in cand:
            judged.append(int(i)); ticks.append(bool(truth[i]))
        if any(ticks):
            bg = rng.choice(np.setdiff1d(np.arange(len(V)), judged), size=N_BG, replace=False)
            X = np.vstack([V[judged], V[bg]])
            y = np.r_[np.array(ticks, int), np.zeros(N_BG, int)]
            model = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0).fit(X, y)
    return model


def evaluate(V, labels, gold, real, budget, ranked):
    rng = np.random.default_rng(SEED)
    names = embed_names(real)
    P = np.zeros((len(V), len(real)))
    for ai, name in enumerate(real):
        truth = labels == gold.index(name)
        m = teach(V, names[ai], truth, budget, rng, ranked)
        P[:, ai] = m.predict_proba(V)[:, 1] if m is not None else 0.0
    best, joined = P.argmax(1), P.max(1) >= 0.5
    per = []
    for ai, name in enumerate(real):
        gi = gold.index(name)
        got = np.where(joined & (best == ai))[0]
        tp = int((labels[got] == gi).sum())
        per.append({"anchor": name, "joined": int(len(got)), "precision": round(tp / len(got), 3) if len(got) else None,
                    "recall": round(tp / int((labels == gi).sum()), 3)})
    precs = [p["precision"] for p in per if p["precision"] is not None]
    un = ~np.isin(labels, [gold.index(g) for g in real])
    return {"budget": budget, "ranked": ranked, "per_anchor": per,
            "median_precision": round(float(np.median(precs)), 3) if precs else 0.0,
            "median_recall": round(float(np.median([p["recall"] for p in per])), 3),
            "empty": sum(p["joined"] == 0 for p in per),
            "collateral_frac": round(float((joined & un).sum() / un.sum()), 4)}


def main():
    texts, labels, gold, V = load()
    real = sorted(np.random.default_rng(SEED).choice(gold, size=N_REAL, replace=False).tolist())
    res = {"round": ROUND, "anchors": real, "runs": []}
    for budget in BUDGETS:
        for ranked in (True, False):
            r = evaluate(V, labels, gold, real, budget, ranked)
            res["runs"].append(r)
            print(f"picks {budget:3d} {'anchor-ranked' if ranked else 'random       '}  precision {r['median_precision']:.3f}"
                  f"  recall {r['median_recall']:.3f}  collateral {r['collateral_frac']:.4f}  empty {r['empty']}", flush=True)
    rng = np.random.default_rng(SEED)
    pv = embed_names(PLACEBOS)
    res["placebos"] = {p: teach(V, pv[i], np.zeros(len(V), bool), 20, rng) is None for i, p in enumerate(PLACEBOS)}
    main_run = next(r for r in res["runs"] if r["budget"] == 20 and r["ranked"])
    res["gates"] = {
        "F1_precision": "PASS" if main_run["median_precision"] >= 0.70 else "FAIL",
        "F2_recall": "PASS" if main_run["median_recall"] >= 0.50 else "FAIL",
        "F3_collateral": "PASS" if main_run["collateral_frac"] <= 0.05 else "FAIL",
        "F4_placebos_empty": "PASS" if all(res["placebos"].values()) else "FAIL",
    }
    json.dump(res, open("anchor_feedback_results.json", "w"), indent=1)
    print(json.dumps(res["gates"], indent=1))
    for p in main_run["per_anchor"]:
        print(f"  {p['anchor']:24s} joined {p['joined']:5d} precision {p['precision']} recall {p['recall']}")


if __name__ == "__main__":
    main()
