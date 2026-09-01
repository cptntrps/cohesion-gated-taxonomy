#!/usr/bin/env python3
"""W1+W2+W6 from the validation brief: the honest, non-oracle, out-of-sample gate test.

W1 (oracle arm): hybrid_paired_test.py used the gold-majority label as the cluster
prediction. Here every cluster prediction is a REAL LLM call (ask_state on the
cluster's representatives), so the hybrid is fully model-produced end to end.

W2 (post-hoc threshold): the read-fraction is chosen on a CALIBRATION half and the
declared policy is then scored once on the untouched EVAL half, paired against
read-everything on the same items, with a random-gate control at the same budget.

W6 (single seed): the cluster arm is repeated over 5 KMeans seeds (real LLM calls
per seed); per-item reads are seed-independent and cached.

Falsifiable goals, declared before running:
  G1 (W1): at the 20% read budget, real-cf hybrid's paired 95% CI vs read-everything
      has lower bound >= -2.0 points. Below that, claim C2 is FALSIFIED.
  G2 (W2): the calibration-chosen policy, scored on the eval half, keeps CI lower
      bound >= -2.0 AND beats the mean random gate at equal budget. Otherwise the
      cohesion gate does not survive out of sample (C3 falsified).
  G3 (W6): across 5 seeds the real-cf hybrid accuracy at the chosen policy varies
      by <= 2.0 points (max-min). More means the result is seed-fragile.
"""
import os, sys, json, pathlib, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from extraction_test import ground_state
from extraction_validate import ask_state, EMB, K
from cluster import cohesion

HERE = pathlib.Path(__file__).parent
SAMPLE = 400
WORKERS = 8
SEEDS = [13, 21, 34, 55, 89]
FRACS = [0.1, 0.2, 0.3, 0.4, 0.5]
MARGIN = 2.0          # non-inferiority margin, accuracy points
CAL_SEED = 101        # calibration/eval split seed (never used for clustering)

def load():
    from datasets import load_dataset
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    texts = list(ds["text"]); labels = np.array(ds["label"]); gold = ds.features["label"].names
    V = np.load(EMB)["V"].astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    n = min(len(texts), len(V)); texts, labels, V = texts[:n], labels[:n], V[:n]
    gl = np.where(labels == gold.index("Governing Laws"))[0]
    st = [ground_state(texts[i]) for i in gl]
    keep = [j for j, s in enumerate(st) if s]
    idx = gl[keep]; y = np.array([st[j] for j in keep])
    return texts, idx, y, V[idx]

def reps_texts(texts, idx, Vg, members, n=6):
    c = Vg[members].mean(0); c /= np.linalg.norm(c) + 1e-9
    order = members[np.argsort(-(Vg[members] @ c))]
    return [texts[idx[m]] for m in order[:n]]

def boot_ci(diff, rng, n_boot=4000):
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5]) * 100
    return lo, hi

def main():
    texts, idx, y, Vg = load()
    N = len(y)
    print(f"{N} governing-law clauses with regex ground truth", flush=True)

    # ---- per-item reads: one cached set, seed-independent (400 real calls) ----
    rng7 = np.random.default_rng(7)
    samp = rng7.choice(N, SAMPLE, replace=False)
    cache_f = HERE / "w1_read_cache.json"
    if cache_f.exists():
        reads_map = {int(k): v for k, v in json.load(open(cache_f)).items()}
    else:
        reads_map = {}
    todo = [int(i) for i in samp if int(i) not in reads_map]
    if todo:
        print(f"per-item LLM reads: {len(todo)} calls...", flush=True)
        with ThreadPoolExecutor(WORKERS) as ex:
            for i, v in zip(todo, ex.map(lambda i: ask_state([texts[idx[i]]]), todo)):
                reads_map[i] = v
        json.dump(reads_map, open(cache_f, "w"))
    reads = np.array([reads_map[int(i)] for i in samp], dtype=object)
    ys = y[samp]
    read_ok = (reads == ys).astype(int)
    print(f"read-everything on the sample: {100*read_ok.mean():.1f}%", flush=True)

    # ---- calibration / eval split of the sample (W2) ----
    rng_cal = np.random.default_rng(CAL_SEED)
    perm = rng_cal.permutation(SAMPLE)
    cal, ev = perm[:SAMPLE // 2], perm[SAMPLE // 2:]

    cpred_f = HERE / "w1_cluster_preds.json"
    all_cpreds = json.load(open(cpred_f)) if cpred_f.exists() else {}
    per_seed = {}
    rng_b = np.random.default_rng(4242)

    for seed in SEEDS:
        km = KMeans(K, n_init=4, random_state=seed).fit(Vg)
        clusters = [np.where(km.labels_ == c)[0] for c in range(K)]
        skey = str(seed)
        if skey not in all_cpreds:
            print(f"seed {seed}: {K} real cluster-level LLM calls...", flush=True)
            with ThreadPoolExecutor(WORKERS) as ex:
                all_cpreds[skey] = list(ex.map(
                    lambda m: ask_state(reps_texts(texts, idx, Vg, m)), clusters))
            json.dump(all_cpreds, open(cpred_f, "w"), indent=1)
        cvals = all_cpreds[skey]
        cf = np.empty(N, dtype=object); coh = np.zeros(N)
        orc = np.empty(N, dtype=object)
        for c, m in enumerate(clusters):
            cf[m] = cvals[c]
            orc[m] = collections.Counter(y[m].tolist()).most_common(1)[0][0]
            coh[m] = cohesion(Vg[m])
        cf_ok = (cf[samp] == ys).astype(int)
        orc_ok = (orc[samp] == ys).astype(int)
        cohs = coh[samp]
        per_seed[seed] = dict(cf_ok=cf_ok, orc_ok=orc_ok, cohs=cohs)
        print(f"seed {seed}: real cluster-first {100*cf_ok.mean():.1f}% vs oracle arm {100*orc_ok.mean():.1f}%")

    # ---- W1: full-sample paired test at each budget, real vs oracle arm (seed 13) ----
    s13 = per_seed[13]
    print(f"\n== W1 (seed 13, full sample n={SAMPLE}): REAL cluster predictions ==")
    print(f"{'read %':>7}{'calls':>7}{'real hybrid':>13}{'oracle hybrid':>15}{'real-CI vs read-all':>22}{'G1':>5}")
    w1_rows = []
    for frac in FRACS:
        thr = np.quantile(s13["cohs"], frac)
        do_read = s13["cohs"] <= thr
        hyb = np.where(do_read, read_ok, s13["cf_ok"])
        hyb_o = np.where(do_read, read_ok, s13["orc_ok"])
        lo, hi = boot_ci(hyb - read_ok, rng_b)
        g1 = "PASS" if lo >= -MARGIN else "FAIL"
        w1_rows.append(dict(frac=frac, calls=int(do_read.sum()) + K,
                            real=round(100*hyb.mean(), 1), oracle=round(100*hyb_o.mean(), 1),
                            ci=[round(lo, 1), round(hi, 1)], g1=g1))
        print(f"{int(frac*100):>6}%{int(do_read.sum())+K:>7}{100*hyb.mean():>12.1f}%"
              f"{100*hyb_o.mean():>14.1f}%{f'[{lo:+.1f}, {hi:+.1f}]':>22}{g1:>5}")

    # ---- W2: choose policy on calibration, score once on eval (seed 13) ----
    print(f"\n== W2 (seed 13): calibrate on {len(cal)}, evaluate on {len(ev)} ==")
    ra_cal = read_ok[cal].mean()
    chosen = None
    for frac in FRACS:
        thr = np.quantile(s13["cohs"][cal], frac)
        hyb_cal = np.where(s13["cohs"][cal] <= thr, read_ok[cal], s13["cf_ok"][cal])
        print(f"  calib frac {int(frac*100)}%: hybrid {100*hyb_cal.mean():.1f}% vs read-all {100*ra_cal:.1f}%")
        if chosen is None and hyb_cal.mean() >= ra_cal - 0.01:
            chosen = (frac, thr)
    if chosen is None:
        chosen = (0.5, np.quantile(s13["cohs"][cal], 0.5))
    frac, thr = chosen
    do_read_ev = s13["cohs"][ev] <= thr
    hyb_ev = np.where(do_read_ev, read_ok[ev], s13["cf_ok"][ev])
    lo, hi = boot_ci(hyb_ev - read_ok[ev], rng_b)
    n_reads = int(do_read_ev.sum())
    rand_accs = []
    for _ in range(1000):
        pick = rng_b.choice(len(ev), n_reads, replace=False)
        mask = np.zeros(len(ev), dtype=bool); mask[pick] = True
        rand_accs.append(np.where(mask, read_ok[ev], s13["cf_ok"][ev]).mean())
    rand_mean = float(np.mean(rand_accs))
    g2 = "PASS" if (lo >= -MARGIN and hyb_ev.mean() > rand_mean) else "FAIL"
    print(f"  CHOSEN on calibration: read lowest {int(frac*100)}% (cohesion <= {thr:.4f})")
    print(f"  EVAL: hybrid {100*hyb_ev.mean():.1f}% | read-all {100*read_ok[ev].mean():.1f}% | "
          f"CI [{lo:+.1f}, {hi:+.1f}] | random gate at {n_reads} reads: {100*rand_mean:.1f}% | G2 {g2}")

    # ---- W6: seed stability at the chosen policy, real arms ----
    print(f"\n== W6: seed stability at the {int(frac*100)}% policy (real cluster arms) ==")
    accs = []
    for seed in SEEDS:
        s = per_seed[seed]
        t = np.quantile(s["cohs"], frac)
        hyb = np.where(s["cohs"] <= t, read_ok, s["cf_ok"])
        accs.append(100 * hyb.mean())
        print(f"  seed {seed}: hybrid {accs[-1]:.1f}%  (cluster-first alone {100*s['cf_ok'].mean():.1f}%)")
    spread = max(accs) - min(accs)
    g3 = "PASS" if spread <= 2.0 else "FAIL"
    print(f"  spread {spread:.1f} points -> G3 {g3}")

    json.dump({"w1": w1_rows, "read_all": round(100*read_ok.mean(), 1),
               "w2": {"chosen_frac": frac, "eval_hybrid": round(100*hyb_ev.mean(), 1),
                      "eval_read_all": round(100*read_ok[ev].mean(), 1),
                      "ci": [round(lo, 1), round(hi, 1)],
                      "random_gate": round(100*rand_mean, 1), "g2": g2},
               "w6": {"accs": [round(a, 1) for a in accs], "spread": round(spread, 1), "g3": g3}},
              open(HERE / "w1_w2_w6_results.json", "w"), indent=1)
    print("\nwrote w1_w2_w6_results.json")

if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set")
    main()
