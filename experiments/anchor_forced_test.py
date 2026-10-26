"""Condition D: a forced-choice judge on what a clause DOES, instead of a leading yes/no per anchor.

Pre-registered 2026-10-25 BEFORE the run (owner: "do it"), after C failed: its yes/no judge shared the anchors'
confusion between what a clause mentions ("This Amendment shall be governed by ...") and what it does (governing law).
One change from C, the judge's question:

  attraction  exactly condition A (20 random LEDGAR names, seed 2026, embed_names, nearest anchor, TAU 0.60)
  judge       gemma3:12b (temperature 0), 15 random members per anchor (seed 2026; all if fewer). Shown one clause
              (first 1,200 chars) and a list: the 20 anchor names plus "Other", shuffled per call (seed 2026).
              Asked which ONE category describes what the clause does (its legal function, not merely what it
              mentions); "Other" if none fits. The reply is matched to a list entry; no match counts as Other.
  accept      >= 70% of the sampled members assigned to their own anchor
  controls    the two capturing placebos ("patent prosecution", "crop insurance"), attracted alone at TAU 0.60;
              their members judged against the 20 real names + the placebo name + "Other"; accepted on the same rule

Gates (same bars as C):
  C1  median true precision (exact gold type) of ACCEPTED anchors                          >= 0.70
  C2  judge decision agrees with truth (true precision >= 0.70 <=> accepted) on            >= 80% of anchors
  C3  capturing placebos rejected                                                           2 of 2
  C4  clauses of other types pulled into accepted anchors                                   <= 5%
  C5  accepted anchors                                                                       >= 5 of 20
If D fails, the line closes: anchors are kept for discovering vocabulary, not for precise assignment.
"""
import json
import urllib.request

import numpy as np

from anchor_test import N_REAL, SEED, assign, family_of
from cohesion_taxonomy.build_tree import load
from cohesion_taxonomy.cluster import TAU

JUDGE = "gemma3:12b"
N_JUDGE = 15
ACCEPT = 0.70
PLACEBOS = ["patent prosecution", "crop insurance"]
SHUF = np.random.default_rng(SEED)


def choose(clause, options):
    opts = list(options) + ["Other"]
    SHUF.shuffle(opts)
    prompt = ("Here is a clause from a contract.\n\n"
              f"Clause:\n{clause.strip()[:1200]}\n\n"
              "Which ONE category best describes what this clause DOES (its legal function), not merely what it "
              "mentions? If none fits, answer Other.\nCategories: " + "; ".join(opts) +
              "\nAnswer with the category name only.")
    req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps({
        "model": JUDGE, "prompt": prompt, "stream": False, "options": {"temperature": 0, "num_predict": 12}
    }).encode(), headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))["response"].strip().lower().strip(" .*\"'")
    hits = [o for o in options if o.lower() in out]
    return max(hits, key=len) if hits else "Other"


def own_rate(name, members, texts, options, rng):
    if len(members) == 0:
        return None, 0, {}
    pick = rng.choice(members, size=min(N_JUDGE, len(members)), replace=False)
    picks = [choose(texts[i], options) for i in pick]
    counts = {}
    for p in picks:
        counts[p] = counts.get(p, 0) + 1
    return picks.count(name) / len(picks), len(picks), counts


def main():
    texts, labels, gold, V = load()
    N = len(labels)
    real = sorted(np.random.default_rng(SEED).choice(gold, size=N_REAL, replace=False).tolist())
    rng = np.random.default_rng(SEED)
    best, joined = assign(V, real, TAU)
    rows, accepted_mask = [], np.zeros(N, bool)
    for ai, name in enumerate(real):
        got = np.where(joined & (best == ai))[0]
        gi = gold.index(name)
        true_p = float((labels[got] == gi).mean()) if len(got) else None
        fam_p = float(np.isin(labels[got], [gold.index(x) for x in family_of(name)]).mean()) if len(got) else None
        rate, n_j, counts = own_rate(name, got, texts, real, rng)
        acc = rate is not None and rate >= ACCEPT
        if acc:
            accepted_mask[got] = True
        rows.append({"anchor": name, "joined": int(len(got)), "true_precision": None if true_p is None else round(true_p, 3),
                     "family_precision": None if fam_p is None else round(fam_p, 3),
                     "recall": round(float((labels[got] == gi).sum() / (labels == gi).sum()), 3),
                     "judged_own": None if rate is None else round(rate, 3), "judged_n": n_j,
                     "judge_choices": counts, "accepted": acc})
        print(f"{name:24s} joined {len(got):5d} true {rows[-1]['true_precision']} judged-own {rows[-1]['judged_own']} "
              f"-> {'ACCEPT' if acc else 'reject'}  {counts}", flush=True)

    pbest, pjoined = assign(V, PLACEBOS, TAU)
    ctrl = []
    for i, name in enumerate(PLACEBOS):
        got = np.where(pjoined & (pbest == i))[0]
        rate, n_j, counts = own_rate(name, got, texts, real + [name], rng)
        ctrl.append({"placebo": name, "joined": int(len(got)), "judged_own": rate, "judged_n": n_j,
                     "judge_choices": counts, "accepted": rate is not None and rate >= ACCEPT})
        print(f"placebo {name:18s} joined {len(got):4d} judged-own {rate} -> "
              f"{'ACCEPT' if ctrl[-1]['accepted'] else 'reject'}  {counts}", flush=True)

    acc_rows = [r for r in rows if r["accepted"]]
    acc_prec = [r["true_precision"] for r in acc_rows]
    agree = [r["accepted"] == (r["true_precision"] is not None and r["true_precision"] >= ACCEPT) for r in rows]
    un_mask = ~np.isin(labels, [gold.index(g) for g in real])
    collateral = float((accepted_mask & un_mask).sum() / un_mask.sum())
    res = {"judge": JUDGE, "n_judge": N_JUDGE, "accept": ACCEPT, "tau": TAU, "anchors": rows, "placebos": ctrl,
           "accepted": [r["anchor"] for r in acc_rows],
           "median_true_precision_accepted": float(np.median(acc_prec)) if acc_prec else None,
           "agreement": round(float(np.mean(agree)), 3), "collateral_frac": round(collateral, 4)}
    res["gates"] = {
        "C1_precision": "PASS" if acc_prec and np.median(acc_prec) >= 0.70 else "FAIL",
        "C2_agreement": "PASS" if res["agreement"] >= 0.80 else "FAIL",
        "C3_placebos": "PASS" if not any(c["accepted"] for c in ctrl) else "FAIL",
        "C4_collateral": "PASS" if collateral <= 0.05 else "FAIL",
        "C5_coverage": "PASS" if len(acc_rows) >= 5 else "FAIL",
    }
    json.dump(res, open("anchor_forced_results.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("gates", "accepted", "median_true_precision_accepted", "agreement",
                                         "collateral_frac")}, indent=1))


if __name__ == "__main__":
    main()
