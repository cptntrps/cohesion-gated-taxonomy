#!/usr/bin/env python3
"""H3c: what does ball radius MEAN, and was the shipped anomaly score's sign wrong?

Owner hypothesis (2026-09-01): center vs edge in the PCA ball is ABSTRACT vs
CONCRETE, not vague vs distinctive. If true, (a) the most-central gold types
should be the abstract legal machinery and the most-peripheral the concrete
content clauses, and (b) since h3 showed novel types sit near the center, the
shipped score's +0.15*radius term pointed AWAY from novelty and inverting the
sign should raise novel yield.

Falsifiable goals: (a) is judged by inspection of the printed type lists;
(b) passes if E1 - w*z(radius) beats E1 on P@40 novel by >= 10 points.
Also tested: does inverted radius improve min-dist-to-any-leaf (h3b's winner)?
"""
import json, collections, pathlib, os
import numpy as np

HERE = pathlib.Path(__file__).parent
EMB = pathlib.Path(os.environ.get("LEDGAR_EMB", "ledgar_emb.npz"))

def main():
    from datasets import load_dataset
    from hyperbolic import make_ball
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    labels = np.array(ds["label"]); gold = ds.features["label"].names
    V = np.load(EMB)["V"].astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    n = min(len(labels), len(V)); labels, V = labels[:n], V[:n]
    P, _, _ = make_ball(V); R = np.linalg.norm(P, axis=1)

    mr = [(gold[g], float(R[labels == g].mean()), int((labels == g).sum()))
          for g in range(len(gold)) if (labels == g).sum() >= 100]
    mr.sort(key=lambda t: t[1])
    print("MOST CENTRAL types (low radius):")
    for nm, r, c in mr[:8]: print(f"  {r:.3f}  {nm} (n={c})")
    print("MOST PERIPHERAL types (high radius):")
    for nm, r, c in mr[-8:]: print(f"  {r:.3f}  {nm} (n={c})")

    tree = json.load(open(HERE / "tree.json")); members = json.load(open(HERE / "members.json"))
    child_ids = {l["source"] for l in tree["links"]}
    leaves = [nd for nd in tree["nodes"] if nd["depth"] > 0 and nd["id"] not in child_ids and nd["count"] > 0]
    dom = {}; cents = []
    for nd in leaves:
        idx = np.array([i for i in members[nd["id"]] if i < n], dtype=int)
        if len(idx) == 0: continue
        dom[nd["id"]] = collections.Counter(labels[idx].tolist()).most_common(1)[0][0]
        c = V[idx].mean(0); c /= np.linalg.norm(c) + 1e-9
        cents.append((nd["id"], c, idx))
    homed = set(dom.values())
    items, cls, E1 = [], [], []
    for nid, c, idx in cents:
        E1.append(1.0 - V[idx] @ c)
        d = dom[nid]
        for i in idx:
            items.append(i); g = labels[i]
            cls.append(0 if g == d else (2 if g not in homed else 1))
    items = np.array(items); cls = np.array(cls); E1 = np.concatenate(E1); Ri = R[items]
    C = np.array([c for _, c, _ in cents]); minall = 1 - (V[items] @ C.T).max(1)

    rng = np.random.default_rng(0)
    def auc(p, q, m=200000):
        a = p[rng.integers(0, len(p), m)]; b = q[rng.integers(0, len(q), m)]
        return float((a > b).mean() + 0.5 * (a == b).mean())
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    out = {"central_types": mr[:8], "peripheral_types": mr[-8:], "scores": {}}
    print("\nnovel detection with INVERTED radius (prefer central outliers):")
    for name, s in [("E1 misfit", E1), ("min-dist-any-leaf", minall),
                    ("E1 - 0.5*z(radius)", z(E1) - 0.5 * z(Ri)),
                    ("minall - 0.5*z(radius)", z(minall) - 0.5 * z(Ri))]:
        top = np.argsort(-s)[:40]
        row = dict(auc_novel=round(auc(s[cls == 2], s[cls != 2]), 3),
                   p40_novel=round(100 * (cls[top] == 2).mean(), 1))
        out["scores"][name] = row
        print(f"  {name:24} AUC {row['auc_novel']:.3f}  P@40 novel {row['p40_novel']}%")
    gain = out["scores"]["E1 - 0.5*z(radius)"]["p40_novel"] - out["scores"]["E1 misfit"]["p40_novel"]
    out["verdict"] = ("SIGN WAS WRONG — inverted radius adds >= 10 pts to misfit's novel yield; "
                      "min-dist-any-leaf still best overall") if gain >= 10 else "no material sign effect"
    print("VERDICT:", out["verdict"])
    json.dump(out, open(HERE / "h3c_results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
