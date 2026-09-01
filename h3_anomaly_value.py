#!/usr/bin/env python3
"""H3: does hyperbolic anomaly scoring add value over Euclidean — tested at the
ITEM level, where anomaly detection actually operates.

CONTRARIAN FRAMING (owner directive 2026-09-01). The prior "redundant with
cohesion" verdict came from a LEAF-level average correlation (misfit_mean -0.73 vs
cohesion +0.74). That was the wrong test for this feature: anomaly detection is an
item-level RANKING problem and its value is in the tail, not the mean. Two signals
with correlated leaf averages can rank the top-40 items completely differently.
This script is written so the hyperbolic path CAN win, and states before running
what vindication looks like.

The innovation, steelmanned: shipped score = geodesic misfit + 0.15*radius in the
Poincare ball (PCA -> tanh). Radius is squashed PCA magnitude = distinctiveness.
Poincare distance amplifies misfit near the boundary, so the score should PREFER
"distinct clause that fits nowhere" (new-concept candidate) over "vague clause
that fits nowhere" (noise). Plain cosine misfit cannot make that distinction.

Ground truth (LEDGAR gold, evaluation-only): for every item in a leaf,
  correct     - its gold type IS the leaf's dominant type
  misassigned - its gold type is dominant in ANOTHER leaf (routing error)
  novel       - its gold type is dominant in NO leaf (the taxonomy lacks the
                concept; the anchor loop's actual quarry)

Rankings compared, identical protocol:
  E1 cosine misfit to own leaf unit-centroid              (Euclidean baseline)
  H1 Poincare geodesic misfit to own leaf ball-centroid   (ball, no radius term)
  H2 shipped: H1 + 0.15 * radius
  E2 ablation: E1 + w * radius  (the distinctiveness IDEA without the ball;
     w chosen so the radius term has the same relative magnitude as in H2)
  R  radius alone (control: is everything just PCA magnitude?)

Declared outcomes, before running:
  VINDICATED  - H2 (or H1) beats E1 by >= +0.03 AUC on novel-vs-rest, or by
                >= +5 points precision-in-top-40 for novel items, AND E2 does not
                close the gap. Then the BALL itself earns its place.
  IDEA-ONLY   - E2 matches H2 (within 0.01 AUC / 2 points P@40) and both beat E1.
                Keep the distinctiveness weighting, drop the hyperbolic machinery.
  REDUNDANT   - E1 matches or beats H2. Re-plumb the panel to cosine misfit.
Also reported either way: top-40 overlap between rankings, AUC for
novel-vs-misassigned separation (the panel's real triage question), and the actual
relative magnitude of the 0.15*radius term (is it cosmetic?).
"""
import json, collections, pathlib, os
import numpy as np

HERE = pathlib.Path(__file__).parent
EMB = pathlib.Path(os.environ.get("LEDGAR_EMB", "ledgar_emb.npz"))
TOPK = 40

def auc(pos_scores, neg_scores, rng, n=200000):
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    p = pos_scores[rng.integers(0, len(pos_scores), n)]
    q = neg_scores[rng.integers(0, len(neg_scores), n)]
    return float((p > q).mean() + 0.5 * (p == q).mean())

def main():
    from datasets import load_dataset
    from hyperbolic import make_ball, _ball_mean, poincare_dist_matrix
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    labels = np.array(ds["label"]); gold = ds.features["label"].names
    V = np.load(EMB)["V"].astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    n = min(len(labels), len(V)); labels, V = labels[:n], V[:n]

    tree = json.load(open(HERE / "tree.json")); members = json.load(open(HERE / "members.json"))
    child_ids = {l["source"] for l in tree["links"]}
    leaves = [nd for nd in tree["nodes"] if nd["depth"] > 0 and nd["id"] not in child_ids and nd["count"] > 0]
    P, _, _ = make_ball(V)
    radius = np.linalg.norm(P, axis=1)

    # dominant gold type per leaf, recomputed from labels (not trusted from json)
    dom = {}
    for nd in leaves:
        idx = np.array([i for i in members[nd["id"]] if i < n], dtype=int)
        if len(idx) == 0: continue
        dom[nd["id"]] = collections.Counter(labels[idx].tolist()).most_common(1)[0][0]
    homed = set(dom.values())
    novel_types = [g for g in range(len(gold)) if g not in homed]
    print(f"{len(leaves)} leaves, {len(homed)} gold types have a home, "
          f"{len(novel_types)} types are NOVEL to the taxonomy")

    items, cls = [], []
    E1 = []; H1 = []
    for nd in leaves:
        idx = np.array([i for i in members[nd["id"]] if i < n], dtype=int)
        if len(idx) == 0: continue
        Vm = V[idx]
        c = Vm.mean(0); c /= np.linalg.norm(c) + 1e-9
        E1.append(1.0 - Vm @ c)
        cen = _ball_mean(P[idx])
        H1.append(poincare_dist_matrix(P[idx], cen[None, :])[:, 0])
        d = dom[nd["id"]]
        for i in idx:
            items.append(i)
            g = labels[i]
            cls.append(0 if g == d else (2 if g in novel_types else 1))
    items = np.array(items); cls = np.array(cls)
    E1 = np.concatenate(E1); H1 = np.concatenate(H1); R = radius[items]
    print(f"{len(items)} leaf items: {int((cls==0).sum())} correct, "
          f"{int((cls==1).sum())} misassigned, {int((cls==2).sum())} novel")

    H2 = H1 + 0.15 * R
    # magnitude honesty: how big is the radius term relative to misfit spread?
    print(f"\nshipped-score anatomy: misfit p50 {np.median(H1):.3f} (spread p90-p10 "
          f"{np.percentile(H1,90)-np.percentile(H1,10):.3f}); 0.15*radius p50 "
          f"{np.median(0.15*R):.4f} (spread {np.percentile(0.15*R,90)-np.percentile(0.15*R,10):.4f})")
    # E2 ablation: give radius the SAME relative weight vs E1's spread as it has vs H1's
    w = 0.15 * (np.percentile(E1,90)-np.percentile(E1,10)) / max(np.percentile(H1,90)-np.percentile(H1,10), 1e-9)
    E2 = E1 + w * R
    print(f"E2 ablation weight on radius: {w:.4f} (matched relative magnitude)")

    rng = np.random.default_rng(0)
    scores = {"E1 cosine misfit": E1, "H1 hyp misfit": H1, "H2 shipped": H2,
              "E2 cos+radius": E2, "R radius alone": R}
    anom = cls > 0
    print(f"\n{'signal':18}{'AUC novel':>10}{'AUC anom':>10}{'AUC nov|mis':>12}{'P@40 nov':>10}{'P@40 anom':>11}")
    results = {}
    for name, s in scores.items():
        a_nov = auc(s[cls == 2], s[cls != 2], rng)
        a_any = auc(s[anom], s[~anom], rng)
        a_sep = auc(s[cls == 2], s[cls == 1], rng)
        top = np.argsort(-s)[:TOPK]
        p_nov = (cls[top] == 2).mean(); p_any = (cls[top] > 0).mean()
        results[name] = dict(auc_novel=round(a_nov,3), auc_anom=round(a_any,3),
                             auc_nov_vs_mis=round(a_sep,3),
                             p40_novel=round(100*p_nov,1), p40_anom=round(100*p_any,1))
        print(f"{name:18}{a_nov:>10.3f}{a_any:>10.3f}{a_sep:>12.3f}{100*p_nov:>9.1f}%{100*p_any:>10.1f}%")

    for a, b in [("H2 shipped", "E1 cosine misfit"), ("H2 shipped", "E2 cos+radius")]:
        ta = set(np.argsort(-scores[a])[:TOPK].tolist()); tb = set(np.argsort(-scores[b])[:TOPK].tolist())
        print(f"top-{TOPK} overlap {a} vs {b}: {len(ta & tb)}/{TOPK}")

    h2, e1, e2 = results["H2 shipped"], results["E1 cosine misfit"], results["E2 cos+radius"]
    if (h2["auc_novel"] - e1["auc_novel"] >= 0.03 or h2["p40_novel"] - e1["p40_novel"] >= 5.0) \
       and (h2["auc_novel"] - e2["auc_novel"] > 0.01):
        verdict = "VINDICATED — the ball itself adds novel-detection value"
    elif (abs(h2["auc_novel"] - e2["auc_novel"]) <= 0.01 or abs(h2["p40_novel"] - e2["p40_novel"]) <= 2.0) \
         and (e2["auc_novel"] - e1["auc_novel"] >= 0.02 or e2["p40_novel"] - e1["p40_novel"] >= 5.0):
        verdict = "IDEA-ONLY — distinctiveness weighting helps; the ball does not"
    elif e1["auc_novel"] >= h2["auc_novel"] - 0.005 and e1["p40_novel"] >= h2["p40_novel"] - 2.0:
        verdict = "REDUNDANT — cosine misfit matches the shipped score"
    else:
        verdict = "MIXED — see numbers; no declared boundary crossed cleanly"
    print(f"\nVERDICT: {verdict}")
    json.dump({"results": results, "verdict": verdict,
               "n_items": int(len(items)), "n_novel": int((cls==2).sum()),
               "n_mis": int((cls==1).sum()), "novel_types": [gold[g] for g in novel_types]},
              open(HERE / "h3_results.json", "w"), indent=1)
    print("wrote h3_results.json")

if __name__ == "__main__":
    main()
