"""Do human anchors give a taxonomy in the company's own words, or is anchoring a confirmation machine?

Pre-registered 2026-10-25 before any run (owner: "well test it"). LEDGAR's 100 clause-type names are real
corporate vocabulary, so they stand in for a legal team's playbook terms; the hidden labels score them.

Conditions (the production path: cluster.embed_names, cluster.TAU, nearest anchor if cosine >= TAU,
build_taxonomy(flat, no naming) for the discovery side):
  real     20 anchors drawn at random (seed 2026) from the 100 type names
  placebo  10 off-domain fakes + 5 legal-sounding fakes absent from LEDGAR, anchored alone
  mixed    real + placebo together (diagnostic: do fakes steal from real anchors?)

Gates (fixed before the run):
  A1  median anchor precision (joined clauses whose gold type IS the anchor's type)  >= 0.70
  A2  median anchor recall (that type's clauses captured by its anchor)              >= 0.50
  A3  off-domain placebos, total capture                                             <  0.5% of corpus
  A3b each legal-sounding placebo                                                    <= 1% of corpus
  A4  of the 80 unanchored types, those found in discovery (a discovery leaf with that
      gold majority at purity >= 50%) vs found by the unanchored baseline build        >= 90%
  A5  clauses of unanchored types pulled into some anchor                              <= 5%
Diagnostics, not gates: precision by synonym family (map fixed below, before the run), a TAU sweep.

Condition B, pre-registered 2026-10-25 AFTER condition A failed A1, A2, A4, A5 (a separate hypothesis, not a
rescue of A; the PRD's anchor design is "canonical names and optional descriptions or exemplars"):
  exemplar  the same 20 anchors, each given as its name plus 5 example clauses of that type (random, seed 2026),
            as a legal team could supply. Anchor vector = unit mean of the name and the 5 clause vectors.
            Per-anchor join threshold, label-free: the lowest leave-one-out cosine of an exemplar to the mean of
            the other four plus the name. A clause joins its nearest anchor if it clears that anchor's threshold.
            The 100 exemplars are excluded from every score. Same gates A1, A2, A5 (A3 does not apply: a placebo
            has no exemplars; A4 is not re-run, as discovery is untouched by how anchors are formed).
"""
import collections
import json
import time

import numpy as np

from build_tree import load
from cluster import TAU, build_taxonomy, embed_names

SEED = 2026
N_REAL = 20
OFF_DOMAIN = ["fishing rights", "recipe ingredients", "volcanic eruptions", "football tactics",
              "wedding photography", "space exploration", "pet grooming", "medieval castles",
              "jazz improvisation", "skin care routine"]
LEGALISH = ["maritime salvage", "patent prosecution", "data residency", "carbon credits", "crop insurance"]
FAMILIES = [  # LEDGAR near-duplicate types, fixed before the run
    {"Indemnifications", "Indemnity"}, {"Waivers", "No Waivers", "Waiver Of Jury Trials"},
    {"Taxes", "Tax Withholdings", "Withholdings"},
    {"Jurisdictions", "Consent To Jurisdiction", "Submission To Jurisdiction", "Venues"},
    {"Governing Laws", "Applicable Laws", "Compliance With Laws"}, {"Assignments", "Assigns", "Successors"},
    {"Entire Agreements", "Integration", "Agreements"}, {"Amendments", "Modifications"},
    {"Definitions", "Defined Terms", "Interpretations", "Construction"}, {"Terms", "Terminations"},
    {"Expenses", "Costs", "Fees"}, {"Approvals", "Consents", "Authorizations"},
    {"Effective Dates", "Effectiveness"}, {"Enforceability", "Enforcements"},
]
SWEEP = [0.50, 0.55, 0.60, 0.65, 0.70]


def family_of(name):
    for f in FAMILIES:
        if name in f:
            return f
    return {name}


def assign(V, names, tau):
    A = embed_names(names)
    sims = V @ A.T
    best, bestsim = sims.argmax(1), sims.max(1)
    return best, bestsim >= tau


def found_types(nodes, members, labels, gold, root_prefix=None):
    """Gold types that some leaf (optionally under a top-level id prefix) holds as majority at purity >= 50%."""
    parents = {n["parent"] for n in nodes}
    top = {n["id"]: n["id"] for n in nodes if n["depth"] == 1}
    byid = {n["id"]: n for n in nodes}

    def top_of(nid):
        while byid[nid]["depth"] > 1:
            nid = byid[nid]["parent"]
        return nid
    out = set()
    for n in nodes:
        if n["depth"] == 0 or n["id"] in parents or not n["count"]:
            continue
        if root_prefix and not top_of(n["id"]).startswith(root_prefix):
            continue
        if (n.get("purity") or 0) >= 50:
            out.add(n["gold_dominant"])
    return out


def main():
    t0 = time.time()
    texts, labels, gold, V = load()
    N = len(labels)
    rng = np.random.default_rng(SEED)
    real = sorted(rng.choice(gold, size=N_REAL, replace=False).tolist())
    unanchored = [g for g in gold if g not in real]
    print(f"{N} clauses; real anchors: {real}", flush=True)
    res = {"seed": SEED, "tau": TAU, "real_anchors": real, "placebo_off": OFF_DOMAIN, "placebo_legalish": LEGALISH}

    # real anchors: precision, recall, collateral
    best, joined = assign(V, real, TAU)
    per = []
    for ai, name in enumerate(real):
        got = np.where(joined & (best == ai))[0]
        gi = gold.index(name)
        tp = int((labels[got] == gi).sum())
        fam = {gold.index(x) for x in family_of(name)}
        fam_tp = int(np.isin(labels[got], list(fam)).sum())
        total = int((labels == gi).sum())
        top3 = [[gold[l], c] for l, c in collections.Counter(labels[got].tolist()).most_common(3)]
        per.append({"anchor": name, "joined": int(len(got)), "type_size": total,
                    "precision": round(tp / len(got), 3) if len(got) else None,
                    "recall": round(tp / total, 3) if total else None,
                    "family_precision": round(fam_tp / len(got), 3) if len(got) else None, "top3": top3})
    precs = [p["precision"] for p in per if p["precision"] is not None]
    recs = [p["recall"] for p in per]
    un_mask = ~np.isin(labels, [gold.index(g) for g in real])
    collateral = float((joined & un_mask).sum() / un_mask.sum())
    res["real"] = {"per_anchor": per, "median_precision": float(np.median(precs)) if precs else 0.0,
                   "median_recall": float(np.median(recs)), "empty_anchors": sum(p["joined"] == 0 for p in per),
                   "joined_total": int(joined.sum()), "collateral_frac": round(collateral, 4),
                   "median_family_precision": float(np.median([p["family_precision"] for p in per
                                                               if p["family_precision"] is not None]))}

    # placebo alone
    pl = OFF_DOMAIN + LEGALISH
    pbest, pjoined = assign(V, pl, TAU)
    cap = {name: int((pjoined & (pbest == i)).sum()) for i, name in enumerate(pl)}
    top_types = {name: [[gold[l], c] for l, c in collections.Counter(
        labels[pjoined & (pbest == i)].tolist()).most_common(2)] for i, name in enumerate(pl)}
    res["placebo"] = {"capture": cap, "top_types": top_types,
                      "off_domain_total_frac": round(sum(cap[n] for n in OFF_DOMAIN) / N, 5),
                      "legalish_max_frac": round(max(cap[n] for n in LEGALISH) / N, 5)}

    # mixed: do fakes steal from real anchors?
    mbest, mjoined = assign(V, real + pl, TAU)
    res["mixed"] = {"placebo_capture": {name: int((mjoined & (mbest == N_REAL + i)).sum()) for i, name in enumerate(pl)}}

    # TAU sweep (diagnostic)
    A_real, A_pl = embed_names(real), embed_names(pl)
    sr, sp = V @ A_real.T, V @ A_pl.T
    sweep = []
    for tau in SWEEP:
        b, j = sr.argmax(1), sr.max(1) >= tau
        pr, rc = [], []
        for ai, name in enumerate(real):
            got = labels[j & (b == ai)]; gi = gold.index(name)
            if len(got):
                pr.append((got == gi).mean())
            rc.append(((got == gi).sum()) / max(1, (labels == gi).sum()))
        sweep.append({"tau": tau, "median_precision": round(float(np.median(pr)), 3) if pr else None,
                      "median_recall": round(float(np.median(rc)), 3),
                      "placebo_total_frac": round(float((sp.max(1) >= tau).sum() / N), 4)})
    res["tau_sweep"] = sweep

    # discovery: unanchored baseline vs anchored build (flat, structure only)
    log = lambda m: None
    nb, _, mb, _, _ = build_taxonomy(V, texts, labels, gold, anchors=None, geometry="flat", do_naming=False, log=log)
    na, _, ma, _, _ = build_taxonomy(V, texts, labels, gold, anchors=real, geometry="flat", do_naming=False, log=log)
    base_found = found_types(nb, mb, labels, gold) & set(unanchored)
    anch_found = found_types(na, ma, labels, gold, root_prefix="D_") & set(unanchored)
    res["discovery"] = {"baseline_found": sorted(base_found), "anchored_found": sorted(anch_found),
                        "ratio": round(len(anch_found) / max(1, len(base_found)), 3),
                        "lost": sorted(base_found - anch_found), "gained": sorted(anch_found - base_found)}

    # condition B: name + 5 exemplar clauses per anchor, label-free per-anchor threshold
    rngb = np.random.default_rng(SEED)
    names_v = embed_names(real)
    vecs, thr, ex_all = [], [], []
    for ai, name in enumerate(real):
        pool = np.where(labels == gold.index(name))[0]
        ex = rngb.choice(pool, size=min(5, len(pool)), replace=False)
        ex_all += ex.tolist()
        parts = np.vstack([names_v[ai][None], V[ex]])
        c = parts.mean(0); vecs.append(c / np.linalg.norm(c))
        loo = []
        for k in range(1, len(parts)):
            rest = np.delete(parts, k, 0).mean(0); rest /= np.linalg.norm(rest)
            loo.append(float(parts[k] @ rest))
        thr.append(min(loo))
    Ab = np.vstack(vecs); thr = np.array(thr)
    keep = np.ones(N, bool); keep[ex_all] = False
    sb = V @ Ab.T
    bb = sb.argmax(1); jb = (sb.max(1) >= thr[bb]) & keep
    perb = []
    for ai, name in enumerate(real):
        got = np.where(jb & (bb == ai))[0]; gi = gold.index(name)
        tp = int((labels[got] == gi).sum()); total = int(((labels == gi) & keep).sum())
        perb.append({"anchor": name, "threshold": round(float(thr[ai]), 3), "joined": int(len(got)),
                     "precision": round(tp / len(got), 3) if len(got) else None, "recall": round(tp / total, 3),
                     "top3": [[gold[l], c] for l, c in collections.Counter(labels[got].tolist()).most_common(3)]})
    pb = [x["precision"] for x in perb if x["precision"] is not None]
    collb = float((jb & un_mask & keep).sum() / (un_mask & keep).sum())
    res["exemplar"] = {"per_anchor": perb, "median_precision": float(np.median(pb)) if pb else 0.0,
                       "median_recall": float(np.median([x["recall"] for x in perb])), "collateral_frac": round(collb, 4)}
    e = res["exemplar"]
    res["gates_B"] = {"A1_precision": "PASS" if e["median_precision"] >= 0.70 else "FAIL",
                      "A2_recall": "PASS" if e["median_recall"] >= 0.50 else "FAIL",
                      "A5_collateral": "PASS" if e["collateral_frac"] <= 0.05 else "FAIL"}
    print("B (name + 5 exemplars):", res["gates_B"], f"precision {e['median_precision']:.3f} recall {e['median_recall']:.3f} collateral {e['collateral_frac']}")

    g = res["real"]; p = res["placebo"]
    res["gates"] = {
        "A1_precision": "PASS" if g["median_precision"] >= 0.70 else "FAIL",
        "A2_recall": "PASS" if g["median_recall"] >= 0.50 else "FAIL",
        "A3_offdomain": "PASS" if p["off_domain_total_frac"] < 0.005 else "FAIL",
        "A3b_legalish": "PASS" if p["legalish_max_frac"] <= 0.01 else "FAIL",
        "A4_discovery": "PASS" if res["discovery"]["ratio"] >= 0.90 else "FAIL",
        "A5_collateral": "PASS" if g["collateral_frac"] <= 0.05 else "FAIL",
    }
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open("anchor_results.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("gates", "seconds")}, indent=1))
    print(f"median precision {g['median_precision']:.3f}  recall {g['median_recall']:.3f}  "
          f"family precision {g['median_family_precision']:.3f}  empty {g['empty_anchors']}  collateral {g['collateral_frac']}")
    print("placebo capture", p["capture"])
    print("discovery", len(anch_found), "/", len(base_found), "lost", res["discovery"]["lost"])


if __name__ == "__main__":
    main()
