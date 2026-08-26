#!/usr/bin/env python3
"""Can we detect a low-purity cluster WITHOUT gold labels?

For every leaf, compute UNSUPERVISED quality signals (geometry only) and correlate
them with the gold purity (which the real corpus won't have). If a signal tracks
purity, it is a gold-free quality gate — flag low-purity clusters for human review.

Signals per leaf (no gold used to compute them):
  cohesion       mean cosine of members to their centroid          (tight -> pure?)
  concentration  norm of the mean unit vector                       (same idea)
  bimodal_gap    cosine split of a 2-means on the leaf              (mixed -> impure?)
  disp_p90       90th pct distance-to-centroid (tail spread)
  radius_std     std of Poincare ball radius                        (hyperbolic)
  misfit_mean    mean Poincare geodesic misfit to ball centroid     (hyperbolic)
  misfit_p90     90th pct misfit
  log_size       log10(count)                                       (control)

Goal: some signal |Spearman| >= 0.5 vs purity, AND a multivariate fit R^2 that beats
the best single signal, or we report no reliable gold-free gate.
"""
import os, json, pathlib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).parent
EMB = pathlib.Path(os.environ.get("LEDGAR_EMB", "ledgar_emb.npz"))
MIN_LEAF = 40

def main():
    from hyperbolic import make_ball, _ball_mean, poincare_dist_matrix
    V = np.load(EMB)["V"].astype(np.float32); V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    tree = json.load(open(HERE / "tree.json"))
    members = json.load(open(HERE / "members.json"))
    child_ids = {l["source"] for l in tree["links"]}
    leaves = [n for n in tree["nodes"] if n["depth"] > 0 and n["id"] not in child_ids and n["count"] >= MIN_LEAF]
    P, _, _ = make_ball(V)

    rows = []
    for n in leaves:
        idx = np.array(members[n["id"]], dtype=int)
        Vm = V[idx]
        c = Vm.mean(0); cn = np.linalg.norm(c)
        cu = c / (cn + 1e-9)
        cos = Vm @ cu
        cohesion = float(cos.mean())
        concentration = float(cn)
        dists = np.sqrt(np.clip(2 - 2 * cos, 0, None))     # euclidean on unit sphere
        disp_p90 = float(np.percentile(dists, 90))
        # bimodality: split into 2, cosine gap between sub-centroids
        if len(idx) >= 4:
            km = KMeans(n_clusters=2, n_init=3, random_state=13).fit(Vm)
            a, b = km.cluster_centers_
            bimodal_gap = float(1 - (a @ b) / ((np.linalg.norm(a)*np.linalg.norm(b)) + 1e-9))
        else:
            bimodal_gap = 0.0
        Pm = P[idx]; rad = np.linalg.norm(Pm, axis=1)
        cen = _ball_mean(Pm)
        mf = poincare_dist_matrix(Pm, cen[None, :])[:, 0]
        rows.append(dict(name=n["name"], purity=n["purity"], size=len(idx),
                         cohesion=cohesion, concentration=concentration,
                         bimodal_gap=bimodal_gap, disp_p90=disp_p90,
                         radius_std=float(rad.std()), misfit_mean=float(mf.mean()),
                         misfit_p90=float(np.percentile(mf, 90)), log_size=float(np.log10(len(idx)))))

    # concentration == cohesion (norm of mean unit vector); keep one to avoid collinearity
    signals = ["cohesion", "bimodal_gap", "disp_p90",
               "radius_std", "misfit_mean", "log_size"]
    pur = np.array([r["purity"] for r in rows])
    print(f"{len(rows)} leaves (size>={MIN_LEAF}). Spearman of each unsupervised signal vs GOLD purity:\n")
    print(f"{'signal':14} {'spearman':>9} {'p':>9}")
    print("-" * 34)
    ranked = []
    for s in signals:
        x = np.array([r[s] for r in rows])
        rho, p = spearmanr(x, pur)
        ranked.append((s, rho, p))
    for s, rho, p in sorted(ranked, key=lambda t: -abs(t[1])):
        star = " <=" if abs(rho) >= 0.5 else ""
        print(f"{s:14} {rho:9.3f} {p:9.1e}{star}")

    # multivariate (house rule: pairwise never alone)
    X = np.array([[r[s] for s in signals] for r in rows])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    reg = LinearRegression().fit(X, pur)
    r2 = reg.score(X, pur)
    best_single = max(abs(t[1]) for t in ranked)
    print(f"\nmultivariate OLS R^2 = {r2:.3f}  (best single |rho| = {best_single:.3f})")
    print("standardized coefficients (direction each signal pushes purity):")
    for s, co in sorted(zip(signals, reg.coef_), key=lambda t: -abs(t[1])):
        print(f"  {s:14} {co:+.2f}")

    strong = [t for t in ranked if abs(t[1]) >= 0.5]
    print("\nVERDICT:", ("gold-free gate EXISTS — " + ", ".join(f"{s}({rho:+.2f})" for s, rho, _ in strong))
          if strong else "no single signal reaches |rho|>=0.5 — weak gold-free gate; lean on the multivariate fit.")
    json.dump({"rows": rows, "spearman": {s: rho for s, rho, _ in ranked}, "r2": r2},
              open(HERE / "purity_probe.json", "w"), indent=1)

if __name__ == "__main__":
    main()
