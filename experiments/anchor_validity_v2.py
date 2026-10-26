"""Anchor + human picks scored by validity, v2: the protocol in docs/PREREG_anchor_validity_v2.md.

  python3 anchor_validity_v2.py prep     assignments for E and A on 20 out-of-sample anchors, blind items, private key
  (judges J1 and J2 each write v2_verdicts_J{1,2}.json: {id: "valid" | "invalid"} for EVERY item)
  python3 anchor_validity_v2.py score    H1-H4 and diagnostics -> anchor_validity_v2_results.json
"""
import json
import sys

import numpy as np
from scipy.stats import wilcoxon

from anchor_feedback_test import teach
from anchor_test import FAMILIES, N_REAL, assign, family_of
from cohesion_taxonomy.build_tree import load
from cohesion_taxonomy.cluster import TAU, embed_names

SEED_V1, SEED = 2026, 4242
N = {"E_disputed": 15, "A_disputed": 10, "gold": 3, "random": 3, "hard": 3, "near": 10, "far": 2}
BOOT = 2000


def anchors(gold):
    v1 = set(np.random.default_rng(SEED_V1).choice(gold, size=N_REAL, replace=False).tolist())
    pool = [g for g in gold if g not in v1]
    return sorted(np.random.default_rng(SEED).choice(pool, size=N_REAL, replace=False).tolist())


def prep():
    texts, labels, gold, V = load()
    real = anchors(gold)
    rng = np.random.default_rng(SEED)
    names = embed_names(real)
    P = np.zeros((len(V), len(real)))
    for ai, name in enumerate(real):
        m = teach(V, names[ai], labels == gold.index(name), 20, rng, True)
        P[:, ai] = m.predict_proba(V)[:, 1] if m is not None else 0.0
    e_best, e_join = P.argmax(1), P.max(1) >= 0.5
    a_best, a_join = assign(V, real, TAU)
    cents = np.vstack([V[labels == k].mean(0) for k in range(len(gold))])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    outside = np.where(~np.isin(labels, [gold.index(g) for g in real]))[0]

    items, members = [], {}
    pick = lambda arr, k: rng.choice(arr, size=min(k, len(arr)), replace=False) if len(arr) else []
    for ai, name in enumerate(real):
        gi = gold.index(name)
        e_mem = np.where(e_join & (e_best == ai))[0]
        a_mem = np.where(a_join & (a_best == ai))[0]
        fam = {gold.index(x) for x in family_of(name)}
        near_type = [k for k in np.argsort(-(cents @ cents[gi])) if k != gi and k not in fam][0]
        nonmem = np.where(~(e_join & (e_best == ai)))[0]
        pa = P[:, ai]
        near = nonmem[(pa[nonmem] >= 0.2) & (pa[nonmem] < 0.5) & (labels[nonmem] != gi)]
        far = nonmem[(pa[nonmem] < 0.2) & (labels[nonmem] != gi)]
        members[name] = {
            "E_total": int(len(e_mem)), "E_gold": int((labels[e_mem] == gi).sum()), "E_disputed": int((labels[e_mem] != gi).sum()),
            "A_total": int(len(a_mem)), "A_gold": int((labels[a_mem] == gi).sum()), "A_disputed": int((labels[a_mem] != gi).sum()),
            "gold_nonmembers": int(((labels == gi) & ~(e_join & (e_best == ai))).sum()),
            "near_count": int(len(near)), "far_count": int(len(far)), "hard_type": gold[near_type]}
        groups = [("E_disputed", e_mem[labels[e_mem] != gi]), ("A_disputed", a_mem[labels[a_mem] != gi]),
                  ("gold", e_mem[labels[e_mem] == gi]), ("random", outside),
                  ("hard", np.where(labels == near_type)[0]), ("near", near), ("far", far)]
        for kind, arr in groups:
            for i in pick(arr, N[kind]):
                items.append({"anchor": name, "clause": texts[int(i)].strip()[:1500], "kind": kind, "gold": gold[labels[int(i)]]})
    order = rng.permutation(len(items))
    blind, key = [], {}
    for n, k in enumerate(order):
        iid = f"w{n:04d}"
        blind.append({"id": iid, "anchor": items[k]["anchor"], "clause": items[k]["clause"]})
        key[iid] = {x: items[k][x] for x in ("anchor", "kind", "gold")}
    json.dump(blind, open("v2_items.json", "w"), indent=1)
    json.dump({"anchors": real, "key": key, "members": members}, open("v2_key.json", "w"), indent=1)
    kinds = {k: sum(v["kind"] == k for v in key.values()) for k in N}
    print(f"anchors {real}\n{len(blind)} items {kinds}")


def kappa(a, b):
    a, b = np.array(a), np.array(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def score():
    K = json.load(open("v2_key.json"))
    key, members, real = K["key"], K["members"], K["anchors"]
    J = {j: json.load(open(f"v2_verdicts_{j}.json")) for j in ("J1", "J2")}
    for j, v in J.items():
        miss = [i for i in key if i not in v]
        if miss:
            raise SystemExit(f"{j}: {len(miss)} items have no verdict, e.g. {miss[:5]}")
    ok = {j: {i: v[i].strip().lower() == "valid" for i in key} for j, v in J.items()}
    both = {i: ok["J1"][i] and ok["J2"][i] for i in key}
    either = {i: ok["J1"][i] or ok["J2"][i] for i in key}
    ids = lambda kind, anchor=None: [i for i, v in key.items() if v["kind"] == kind and (anchor is None or v["anchor"] == anchor)]

    calib = {j: {"accept_gold": float(np.mean([ok[j][i] for i in ids("gold")])),
                 "reject_random": float(np.mean([not ok[j][i] for i in ids("random")])),
                 "accept_hard": float(np.mean([ok[j][i] for i in ids("hard")])),
                 "accept_far": float(np.mean([ok[j][i] for i in ids("far")]))} for j in ok}
    pair = ids("E_disputed") + ids("A_disputed") + ids("near")
    kap = kappa([ok["J1"][i] for i in pair], [ok["J2"][i] for i in pair])

    def per_anchor(verdict, sample=None):
        rows = []
        for name in real:
            m = members[name]
            def rate(kind):
                xs = [verdict[i] for i in ids(kind, name)]
                if sample is not None and xs:
                    xs = sample.choice(xs, size=len(xs), replace=True)
                return float(np.mean(xs)) if len(xs) else 0.0
            e_p = (m["E_gold"] + m["E_disputed"] * rate("E_disputed")) / m["E_total"] if m["E_total"] else None
            a_p = (m["A_gold"] + m["A_disputed"] * rate("A_disputed")) / m["A_total"] if m["A_total"] else None
            valid_mem = (m["E_gold"] + m["E_disputed"] * rate("E_disputed"))
            missed = m["gold_nonmembers"] + m["near_count"] * rate("near")
            rec = valid_mem / (valid_mem + missed) if (valid_mem + missed) else None
            rows.append({"anchor": name, "E_members": m["E_total"], "A_members": m["A_total"],
                         "E_label_precision": round(m["E_gold"] / m["E_total"], 3) if m["E_total"] else None,
                         "E_validity_precision": None if e_p is None else round(e_p, 3),
                         "A_validity_precision": None if a_p is None else round(a_p, 3),
                         "E_validity_recall": None if rec is None else round(rec, 3), "hard_type": m["hard_type"]})
        return rows

    rows = per_anchor(both)
    med = lambda rs, k: float(np.median([r[k] for r in rs if r[k] is not None]))
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(BOOT):
        rs = per_anchor(both, rng)
        pick = rng.choice(len(rs), size=len(rs), replace=True)
        boot.append(med([rs[k] for k in pick], "E_validity_precision"))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    paired = [(r["E_validity_precision"], r["A_validity_precision"]) for r in rows
              if r["E_validity_precision"] is not None and r["A_validity_precision"] is not None]
    diffs = [e - a for e, a in paired]
    p = float(wilcoxon(diffs, alternative="greater").pvalue) if any(diffs) else 1.0
    res = {
        "anchors": real, "calibration": calib, "kappa": round(kap, 3), "per_anchor": rows,
        "E_median_validity_precision": round(med(rows, "E_validity_precision"), 3),
        "E_ci95": [round(float(lo), 3), round(float(hi), 3)],
        "A_median_validity_precision": round(med(rows, "A_validity_precision"), 3),
        "E_median_label_precision": round(med(rows, "E_label_precision"), 3),
        "E_median_validity_recall": round(med(rows, "E_validity_recall"), 3),
        "paired_n": len(paired), "wilcoxon_p": round(p, 5),
        "either_judge_E_median_validity_precision": round(med(per_anchor(either), "E_validity_precision"), 3),
    }
    h3 = all(c["accept_gold"] >= 0.9 and c["reject_random"] >= 0.9 for c in calib.values()) and kap >= 0.60
    res["hypotheses"] = {
        "H3_judges_reliable": "PASS" if h3 else "FAIL",
        "H1_E_valid": ("PASS" if res["E_median_validity_precision"] >= 0.70 and lo >= 0.60 else "FAIL") if h3 else "NOT ESTABLISHED",
        "H2_picks_matter": ("PASS" if p < 0.05 else "FAIL") if h3 else "NOT ESTABLISHED",
        "H4_E_recall": ("PASS" if res["E_median_validity_recall"] >= 0.50 else "FAIL") if h3 else "NOT ESTABLISHED",
    }
    json.dump(res, open("anchor_validity_v2_results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_anchor"}, indent=1))
    for r in rows:
        print(f"  {r['anchor']:26s} E {r['E_members']:5d} label {r['E_label_precision']} valid {r['E_validity_precision']} "
              f"recall {r['E_validity_recall']} | A {r['A_members']:5d} valid {r['A_validity_precision']}")


if __name__ == "__main__":
    {"prep": prep, "score": score}[sys.argv[1]]()
