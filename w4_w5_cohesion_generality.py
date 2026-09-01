#!/usr/bin/env python3
"""W4+W5 from the validation brief: does the cohesion->purity signal generalize
across corpora, survive a cluster-size control, and hold out of sample?

purity_probe.py measured Spearman +0.74 on LEDGAR leaves only, with in-sample OLS
R^2. Confound: log_size correlates with purity at -0.55 and small clusters are
mechanically tighter, so cohesion may partly proxy size.

Method, per corpus (LEDGAR, 20 Newsgroups, Amazon) and per seed (5 seeds):
  - divisive flat KMeans tree (structure only, no naming, no LLM, no gold)
  - per leaf (size >= MIN_LEAF): cohesion, bimodal_gap, disp_p90, log_size; gold
    purity joined AFTER the fact for scoring only
  - full Spearman(cohesion, purity)
  - PARTIAL Spearman(cohesion, purity | log_size) via rank-residual correlation
  - leave-one-out R^2 of OLS on [cohesion, bimodal_gap, disp_p90, log_size]

Falsifiable goals, declared before running:
  G6 (generality): mean full Spearman >= +0.5 on BOTH non-LEDGAR corpora, else the
      signal is corpus-specific and C1 does not generalize.
  G7 (size confound): mean partial Spearman >= +0.3 on every corpus, else the gate
      is mostly a size heuristic.
  G8 (out of sample): mean LOO R^2 within 0.15 of in-sample R^2, else the
      multivariate fit is overfit bookkeeping.
"""
import json, collections, pathlib, os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, rankdata

HERE = pathlib.Path(__file__).parent
SEEDS = [13, 21, 34, 55, 89]
MIN_LEAF = 40

def divisive_leaves(V, kroot, ksub, min_split, max_depth, seed):
    leaves = []
    def rec(idx, depth):
        if depth >= max_depth or len(idx) < min_split:
            leaves.append(idx); return
        k = min(ksub if depth else kroot, len(idx))
        if k < 2:
            leaves.append(idx); return
        lab = KMeans(k, n_init=4, random_state=seed).fit(V[idx]).labels_
        subs = [idx[lab == c] for c in range(k) if (lab == c).any()]
        if len(subs) < 2:
            leaves.append(idx); return
        for s in subs:
            rec(s, depth + 1)
    rec(np.arange(len(V)), 0)
    return [l for l in leaves if len(l) >= MIN_LEAF]

def leaf_signals(V, idx, seed):
    Vm = V[idx]
    c = Vm.mean(0); cn = np.linalg.norm(c)
    cos = Vm @ (c / (cn + 1e-9))
    dists = np.sqrt(np.clip(2 - 2 * cos, 0, None))
    if len(idx) >= 4:
        km = KMeans(2, n_init=3, random_state=seed).fit(Vm)
        a, b = km.cluster_centers_
        gap = float(1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    else:
        gap = 0.0
    return dict(cohesion=float(cos.mean()), bimodal_gap=gap,
                disp_p90=float(np.percentile(dists, 90)), log_size=float(np.log10(len(idx))))

def purity_of(y, idx):
    cnt = collections.Counter(y[idx].tolist())
    return 100.0 * cnt.most_common(1)[0][1] / len(idx)

def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.polyfit(b, a, 1)
        return a - np.polyval(b1, b)
    r = np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1]
    return float(r)

def loo_r2(X, y):
    n = len(y)
    pred = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool); m[i] = False
        reg = LinearRegression().fit(X[m], y[m])
        pred[i] = reg.predict(X[i:i+1])[0]
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)

def load_ledgar():
    from datasets import load_dataset
    EMBP = pathlib.Path(os.environ.get("LEDGAR_EMB", "ledgar_emb.npz"))
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    labels = np.array(ds["label"])
    V = np.load(EMBP)["V"].astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    n = min(len(labels), len(V))
    return V[:n], labels[:n], dict(kroot=8, ksub=5, min_split=1200, max_depth=3)

def load_ng():
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    texts, targets = d.data, np.array(d.target)
    keep = [i for i, t in enumerate(texts) if len(t.strip()) > 40][:6000]
    V = np.load(HERE / "ng_emb.npz")["V"].astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return V, targets[keep], dict(kroot=8, ksub=4, min_split=120, max_depth=3)

def load_amz():
    from datasets import load_dataset
    ds = load_dataset("ckandemir/amazon-products", split="train")
    names, cats = ds["Product Name"], ds["Category"]
    def top(c):
        for s in ["|", ">", "/"]:
            if s in c: return c.split(s)[0].strip()
        return c.strip()
    byc = collections.defaultdict(list)
    for i, c in enumerate(cats):
        if c and names[i]: byc[top(c)].append(i)
    tops = [t for t, v in sorted(byc.items(), key=lambda kv: -len(kv[1])) if len(v) >= 500][:6]
    idx = []
    for t in tops: idx += byc[t][:500]
    y = np.array([top(cats[i]) for i in idx])
    codes = {c: i for i, c in enumerate(sorted(set(y.tolist())))}
    y = np.array([codes[c] for c in y])
    V = np.load(HERE / "amz_emb.npz")["V"].astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return V, y, dict(kroot=6, ksub=4, min_split=100, max_depth=3)

def main():
    out = {}
    for name, loader in [("ledgar", load_ledgar), ("20ng", load_ng), ("amazon", load_amz)]:
        V, y, cfg = loader()
        rows = []
        for seed in SEEDS:
            leaves = divisive_leaves(V, seed=seed, **cfg)
            sig = [leaf_signals(V, l, seed) for l in leaves]
            pur = np.array([purity_of(y, l) for l in leaves])
            coh = np.array([s["cohesion"] for s in sig])
            ls = np.array([s["log_size"] for s in sig])
            X = np.array([[s["cohesion"], s["bimodal_gap"], s["disp_p90"], s["log_size"]] for s in sig])
            Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
            full = float(spearmanr(coh, pur).correlation)
            part = partial_spearman(coh, pur, ls)
            r2_in = float(LinearRegression().fit(Xz, pur).score(Xz, pur))
            r2_loo = loo_r2(Xz, pur)
            rows.append(dict(seed=seed, n_leaves=len(leaves), full=round(full, 3),
                             partial=round(part, 3), r2_in=round(r2_in, 3), r2_loo=round(r2_loo, 3)))
            print(f"{name} seed {seed}: {len(leaves)} leaves | Spearman {full:+.3f} | "
                  f"partial(|size) {part:+.3f} | R2 in {r2_in:.3f} loo {r2_loo:.3f}", flush=True)
        mf = np.mean([r["full"] for r in rows]); mp = np.mean([r["partial"] for r in rows])
        mi = np.mean([r["r2_in"] for r in rows]); ml = np.mean([r["r2_loo"] for r in rows])
        out[name] = dict(rows=rows, mean_full=round(float(mf), 3), mean_partial=round(float(mp), 3),
                         mean_r2_in=round(float(mi), 3), mean_r2_loo=round(float(ml), 3))
        print(f"{name} MEAN: full {mf:+.3f} | partial {mp:+.3f} | R2 in {mi:.3f} / loo {ml:.3f}\n")

    g6 = "PASS" if (out["20ng"]["mean_full"] >= 0.5 and out["amazon"]["mean_full"] >= 0.5) else "FAIL"
    g7 = "PASS" if all(out[c]["mean_partial"] >= 0.3 for c in out) else "FAIL"
    g8 = "PASS" if all(out[c]["mean_r2_in"] - out[c]["mean_r2_loo"] <= 0.15 for c in out) else "FAIL"
    print(f"G6 generality: {g6} | G7 size-confound: {g7} | G8 out-of-sample fit: {g8}")
    out["verdicts"] = dict(g6=g6, g7=g7, g8=g8)
    json.dump(out, open(HERE / "w4_w5_results.json", "w"), indent=1)
    print("wrote w4_w5_results.json")

if __name__ == "__main__":
    main()
