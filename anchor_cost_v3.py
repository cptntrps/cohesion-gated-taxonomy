"""v3: anchor + picks at zero LLM calls vs an LLM reading every clause (docs/PREREG_anchor_cost_v3.md).

  python3 anchor_cost_v3.py prep    test set S, Nomen variants R20 R40 R80 U40 U80 -> v3_nomen.json; v3_llm_items.json
  (LLM baseline writes v3_llm_b{1..8}.json: {id: term or "None"})
  python3 anchor_cost_v3.py pool    pooled pairs + calibration, blind -> v3_pairs.json, private v3_pairs_key.json
  (judges J1 J2 write v3_pv_J{1,2}_b{n}.json: {id: "valid" | "invalid"})
  python3 anchor_cost_v3.py score   H5 H6 H7 -> anchor_cost_v3_results.json
"""
import glob
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

from anchor_test import family_of
from build_tree import load
from cluster import embed_names

SEED = 7777
N_TEST = 2000
ROUND, N_BG = 10, 1000
VARIANTS = [("R20", 20, "ranked"), ("R40", 40, "ranked"), ("R80", 80, "ranked"), ("U40", 40, "uncertain"), ("U80", 80, "uncertain")]
BOOT = 2000


def teach(V, allowed, name_vec, truth, budget, rng, mode):
    judged, ticks, model = [], [], None
    order = [i for i in np.argsort(-(V @ name_vec)) if allowed[i]]
    while len(judged) < budget:
        seen = set(judged)
        if model is None:
            cand = [i for i in order if i not in seen][:ROUND]
        else:
            p = model.predict_proba(V)[:, 1]
            p[~allowed] = -1
            p[judged] = -1
            if mode == "ranked":
                cand = np.argsort(-p)[:ROUND].tolist()
            else:
                top = np.argsort(-p)[:ROUND // 2].tolist()
                unc = np.abs(p - 0.5); unc[p < 0] = 9; unc[top] = 9
                cand = top + np.argsort(unc)[:ROUND - len(top)].tolist()
        for i in cand:
            judged.append(int(i)); ticks.append(bool(truth[i]))
        if any(ticks):
            pool = np.where(allowed)[0]
            bg = rng.choice(np.setdiff1d(pool, judged), size=N_BG, replace=False)
            X = np.vstack([V[judged], V[bg]])
            y = np.r_[np.array(ticks, int), np.zeros(N_BG, int)]
            model = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X, y)
    return model


def prep():
    texts, labels, gold, V = load()
    terms = json.load(open("v2_key.json"))["anchors"]
    rng = np.random.default_rng(SEED)
    S = np.sort(rng.choice(len(V), size=N_TEST, replace=False))
    allowed = np.ones(len(V), bool); allowed[S] = False
    names = embed_names(terms)
    preds = {}
    for vname, budget, mode in VARIANTS:
        vr = np.random.default_rng(SEED)
        P = np.zeros((N_TEST, len(terms)))
        for ti, t in enumerate(terms):
            m = teach(V, allowed, names[ti], labels == gold.index(t), budget, vr, mode)
            if m is not None:
                P[:, ti] = m.predict_proba(V[S])[:, 1]
        best, ok = P.argmax(1), P.max(1) >= 0.5
        preds[vname] = [terms[best[k]] if ok[k] else None for k in range(N_TEST)]
        print(vname, "assigned", int(ok.sum()), "of", N_TEST, flush=True)
    sid = [f"s{k:04d}" for k in range(N_TEST)]
    json.dump({"terms": terms, "S": S.tolist(), "ids": sid, "gold": [gold[labels[i]] for i in S], "preds": preds},
              open("v3_nomen.json", "w"))
    json.dump({"terms": terms, "items": [{"id": sid[k], "clause": texts[i].strip()[:1200]} for k, i in enumerate(S)]},
              open("v3_llm_items.json", "w"), indent=1)


def pool():
    texts, labels, gold, V = load()
    N = json.load(open("v3_nomen.json"))
    terms, S, ids, glab = N["terms"], N["S"], N["ids"], N["gold"]
    llm = {}
    for f in sorted(glob.glob("v3_llm_b*.json")):
        llm.update(json.load(open(f)))
    miss = [i for i in ids if i not in llm]
    if miss:
        raise SystemExit(f"LLM baseline missing {len(miss)} ids, e.g. {miss[:5]}")
    norm = {t.lower(): t for t in terms}
    llm_pred = [norm.get(str(llm[i]).strip().lower()) for i in ids]
    systems = dict(N["preds"]); systems["LLM"] = llm_pred
    pairs = {}
    for k in range(len(ids)):
        cand = {systems[s][k] for s in systems if systems[s][k]}
        if glab[k] in terms:
            cand.add(glab[k])
        for t in cand:
            pairs[(k, t)] = {"kind": "pool", "label_match": glab[k] == t,
                             "proposed_by": [s for s in systems if systems[s][k] == t]}
    rng = np.random.default_rng(SEED)
    pos = [p for p, v in pairs.items() if v["label_match"]]
    for j in rng.choice(len(pos), size=min(40, len(pos)), replace=False):
        pairs[pos[j]]["calib"] = "positive"
    negs = []
    while len(negs) < 40:
        k = int(rng.integers(len(ids))); t = terms[int(rng.integers(len(terms)))]
        if t != glab[k] and glab[k] not in family_of(t) and (k, t) not in pairs:
            negs.append((k, t))
    for p in negs:
        pairs[p] = {"kind": "calib_negative", "label_match": False, "proposed_by": []}
    keys = list(pairs)
    order = rng.permutation(len(keys))
    blind, key = [], {}
    for n, o in enumerate(order):
        k, t = keys[o]
        pid = f"p{n:04d}"
        blind.append({"id": pid, "anchor": t, "clause": texts[S[k]].strip()[:1500]})
        key[pid] = {"k": k, "term": t, **pairs[keys[o]]}
    json.dump(blind, open("v3_pairs.json", "w"), indent=1)
    json.dump({"key": key, "llm_pred": llm_pred, "llm_none": sum(p is None for p in llm_pred)}, open("v3_pairs_key.json", "w"))
    print(f"{len(blind)} pairs ({len(negs)} calibration negatives, {sum(v.get('calib') == 'positive' for v in pairs.values())} positives)")


def kappa(a, b):
    a, b = np.array(a, float), np.array(b, float)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def score():
    N = json.load(open("v3_nomen.json")); PK = json.load(open("v3_pairs_key.json"))
    key, terms, n = PK["key"], N["terms"], len(N["ids"])
    systems = dict(N["preds"]); systems["LLM"] = PK["llm_pred"]
    J = {}
    for j in ("J1", "J2"):
        J[j] = {}
        for f in sorted(glob.glob(f"v3_pv_{j}_b*.json")):
            J[j].update(json.load(open(f)))
        miss = [p for p in key if p not in J[j]]
        if miss:
            raise SystemExit(f"{j} missing {len(miss)} verdicts, e.g. {miss[:5]}")
    ok = {j: {p: J[j][p].strip().lower() == "valid" for p in key} for j in J}
    valid = {(v["k"], v["term"]): ok["J1"][p] and ok["J2"][p] for p, v in key.items() if v["kind"] == "pool"}
    calib = {j: {"accept_pos": float(np.mean([ok[j][p] for p, v in key.items() if v.get("calib") == "positive"])),
                 "reject_neg": float(np.mean([not ok[j][p] for p, v in key.items() if v["kind"] == "calib_negative"]))} for j in J}
    pool_ids = [p for p, v in key.items() if v["kind"] == "pool"]
    kap = kappa([ok["J1"][p] for p in pool_ids], [ok["J2"][p] for p in pool_ids])

    def metrics(sample):
        out = {}
        tot_valid = sum(valid[(k, t)] for (k, t) in valid if k in sample)
        for s, pred in systems.items():
            prop = [(k, pred[k]) for k in sample if pred[k]]
            hit = sum(valid.get(p, False) for p in prop)
            out[s] = {"proposed": len(prop), "valid": hit, "precision": hit / len(prop) if prop else 0.0,
                      "recall": hit / tot_valid if tot_valid else 0.0}
        return out
    from collections import Counter
    full = metrics(list(range(n)))
    rng = np.random.default_rng(SEED)
    diffs = {s: {"precision": [], "recall": []} for s in systems if s != "LLM"}
    for _ in range(BOOT):
        smp = Counter(rng.integers(0, n, n).tolist())
        sample = [k for k, c in smp.items() for _ in range(c)]
        m = metrics(sample)
        for s in diffs:
            for x in ("precision", "recall"):
                diffs[s][x].append(m[s][x] - m["LLM"][x])
    ci = {s: {x: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] for x, v in d.items()} for s, d in diffs.items()}
    h7 = all(c["accept_pos"] >= 0.9 and c["reject_neg"] >= 0.9 for c in calib.values()) and kap >= 0.60
    budgets = {"R20": 20, "R40": 40, "R80": 80, "U40": 40, "U80": 80}
    match = sorted([(budgets[s], s) for s in ci if budgets[s] <= 80 and ci[s]["recall"][0] >= -0.05])
    res = {"calibration": calib, "kappa": round(kap, 3), "metrics": full, "ci_vs_llm": ci,
           "llm_none": PK["llm_none"], "valid_pairs_in_pool": int(sum(valid.values())),
           "cost_per_60k": {s: {"llm_calls": 60000 if s == "LLM" else 0, "human_picks": 0 if s == "LLM" else budgets[s] * 20} for s in systems}}
    res["hypotheses"] = {
        "H7_judges": "PASS" if h7 else "FAIL",
        "H5_precision_R20": ("PASS" if ci["R20"]["precision"][0] >= -0.05 else "FAIL") if h7 else "NOT ESTABLISHED",
        "H6_recall": (("PASS: " + match[0][1]) if match else "FAIL") if h7 else "NOT ESTABLISHED",
    }
    json.dump(res, open("anchor_cost_v3_results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"prep": prep, "pool": pool, "score": score}[sys.argv[1]]()
