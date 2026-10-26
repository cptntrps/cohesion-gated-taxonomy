"""Condition E scored by validity, not by agreement with LEDGAR's single heading label.

Pre-registered 2026-10-25 BEFORE any judging. Owner: "the question must not be if it found the same exact as the
gold, but if it found valid ones". LEDGAR gives each clause one heading-derived type; a clause can validly belong to
an anchor whose name is not its heading (a Waivers clause is a valid "No Waivers" clause).

  python3 anchor_validity.py prep    rebuild condition E at 20 picks (same seeds), write validity_items.json (blind,
                                     shuffled: id, anchor, clause) and validity_key.json (private: kind, gold type)
  (judge)                            a frontier-model judge, blind to kind and gold, writes validity_verdicts.json
                                     {id: "valid" | "invalid"}: would a legal team filing clauses under the anchor
                                     accept this clause there? judge the clause's function, not what it mentions;
                                     unclear = invalid
  python3 anchor_validity.py score   gates below

Items per anchor: up to 8 "disputed" (assigned to the anchor, gold type differs), 2 "gold" (assigned, gold type
matches; calibration positives), 2 "random" (random clauses of a type that is none of the 20 anchors; calibration
negatives). Seed 2026.

Gates:
  V0  judge trust: accepts >= 90% of gold items AND rejects >= 90% of random items (else V1 is not reported as a result)
  V1  median validity precision >= 0.70, where per anchor
        validity precision = (gold-matching members + disputed members x judged-valid rate of its disputed sample)
                             / all members
Recall stays measured against the labels (validity can only raise precision; the number of valid clauses the anchor
missed is unknown).
"""
import json
import sys

import numpy as np

from anchor_feedback_test import teach
from anchor_test import N_REAL, SEED
from cohesion_taxonomy.build_tree import load
from cohesion_taxonomy.cluster import embed_names

N_DISPUTED, N_GOLD, N_RANDOM = 8, 2, 2


def assignments(V, labels, gold, real):
    rng = np.random.default_rng(SEED)
    names = embed_names(real)
    P = np.zeros((len(V), len(real)))
    for ai, name in enumerate(real):
        m = teach(V, names[ai], labels == gold.index(name), 20, rng, True)
        P[:, ai] = m.predict_proba(V)[:, 1] if m is not None else 0.0
    return P.argmax(1), P.max(1) >= 0.5


def prep():
    texts, labels, gold, V = load()
    real = sorted(np.random.default_rng(SEED).choice(gold, size=N_REAL, replace=False).tolist())
    best, joined = assignments(V, labels, gold, real)
    rng = np.random.default_rng(SEED)
    other = np.where(~np.isin(labels, [gold.index(g) for g in real]))[0]
    items, key, members = [], {}, {}
    for ai, name in enumerate(real):
        got = np.where(joined & (best == ai))[0]
        gi = gold.index(name)
        disputed, matched = got[labels[got] != gi], got[labels[got] == gi]
        members[name] = {"total": int(len(got)), "gold_match": int(len(matched)), "disputed": int(len(disputed))}
        picks = [("disputed", i) for i in rng.choice(disputed, min(N_DISPUTED, len(disputed)), replace=False)] + \
                [("gold", i) for i in rng.choice(matched, min(N_GOLD, len(matched)), replace=False)] + \
                [("random", i) for i in rng.choice(other, N_RANDOM, replace=False)]
        for kind, i in picks:
            items.append({"anchor": name, "clause": texts[i].strip()[:1500], "_kind": kind, "_gold": gold[labels[i]]})
    order = rng.permutation(len(items))
    blind = []
    for n, k in enumerate(order):
        it = items[k]
        iid = f"v{n:03d}"
        blind.append({"id": iid, "anchor": it["anchor"], "clause": it["clause"]})
        key[iid] = {"kind": it["_kind"], "gold": it["_gold"], "anchor": it["anchor"]}
    json.dump(blind, open("validity_items.json", "w"), indent=1)
    json.dump({"key": key, "members": members}, open("validity_key.json", "w"), indent=1)
    print(f"{len(blind)} items written; kinds:", {k: sum(v["kind"] == k for v in key.values()) for k in ("disputed", "gold", "random")})


def score():
    key = json.load(open("validity_key.json"))
    verdict = json.load(open("validity_verdicts.json"))
    K, members = key["key"], key["members"]
    missing = [i for i in K if i not in verdict]
    if missing:
        raise SystemExit(f"{len(missing)} items have no verdict, e.g. {missing[:5]}")
    ok = lambda i: verdict[i].strip().lower() == "valid"
    gold_acc = np.mean([ok(i) for i, v in K.items() if v["kind"] == "gold"])
    rand_rej = np.mean([not ok(i) for i, v in K.items() if v["kind"] == "random"])
    per = []
    for name, m in members.items():
        ds = [ok(i) for i, v in K.items() if v["anchor"] == name and v["kind"] == "disputed"]
        rate = float(np.mean(ds)) if ds else 0.0
        vp = (m["gold_match"] + m["disputed"] * rate) / m["total"] if m["total"] else None
        lp = m["gold_match"] / m["total"] if m["total"] else None
        per.append({"anchor": name, "members": m["total"], "label_precision": None if lp is None else round(lp, 3),
                    "disputed_valid_rate": round(rate, 3), "disputed_judged": len(ds),
                    "validity_precision": None if vp is None else round(vp, 3)})
    vps = [p["validity_precision"] for p in per if p["validity_precision"] is not None]
    res = {"judge_accepts_gold": round(float(gold_acc), 3), "judge_rejects_random": round(float(rand_rej), 3),
           "per_anchor": per, "median_validity_precision": round(float(np.median(vps)), 3),
           "median_label_precision": round(float(np.median([p["label_precision"] for p in per if p["label_precision"] is not None])), 3)}
    res["gates"] = {"V0_judge_trust": "PASS" if gold_acc >= 0.9 and rand_rej >= 0.9 else "FAIL",
                    "V1_validity_precision": "PASS" if res["median_validity_precision"] >= 0.70 else "FAIL"}
    json.dump(res, open("anchor_validity_results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_anchor"}, indent=1))
    for p in per:
        print(f"  {p['anchor']:24s} members {p['members']:5d} label {p['label_precision']}  disputed valid "
              f"{p['disputed_valid_rate']} ({p['disputed_judged']})  -> validity {p['validity_precision']}")


if __name__ == "__main__":
    {"prep": prep, "score": score}[sys.argv[1]]()
