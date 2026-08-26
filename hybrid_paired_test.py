#!/usr/bin/env python3
"""HONEST paired test of the headline claim (codex catch, 2026-08-26).

Earlier the hybrid frontier ASSUMED read items are correct -> that was an upper
bound, not a measurement, and the 96.2 vs 95.3 gap was never significance-tested.

Here: on ONE random sample, for every item get BOTH
  (a) cluster-first label (0 extra calls), and
  (b) a real per-item LLM read (read-everything),
then form the gated hybrid (read only low-cohesion clusters, trust the rest) and
run a PAIRED bootstrap on hybrid - read_everything. Reports CI and call counts.
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

def main():
    from datasets import load_dataset
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    texts = list(ds["text"]); labels = np.array(ds["label"]); gold = ds.features["label"].names
    V = np.load(EMB)["V"].astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    n = min(len(texts), len(V)); texts, labels, V = texts[:n], labels[:n], V[:n]
    gl = np.where(labels == gold.index("Governing Laws"))[0]
    st = [ground_state(texts[i]) for i in gl]; keep = [j for j, s in enumerate(st) if s]
    idx = gl[keep]; y = np.array([st[j] for j in keep]); Vg = V[idx]; N = len(y)

    km = KMeans(K, n_init=4, random_state=13).fit(Vg)
    cl = [np.where(km.labels_ == c)[0] for c in range(K)]
    cf = np.empty(N, dtype=object); coh = np.zeros(N)
    for c, m in enumerate(cl):
        cf[m] = collections.Counter(y[m].tolist()).most_common(1)[0][0]   # cluster-first label
        coh[m] = cohesion(Vg[m])

    rng = np.random.default_rng(7)
    samp = rng.choice(N, SAMPLE, replace=False)
    print(f"paired sample n={SAMPLE} of {N}. Running {SAMPLE} per-item LLM reads...", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        reads = np.array(list(ex.map(lambda i: ask_state([texts[idx[i]]]), samp)), dtype=object)

    ys = y[samp]; cfs = cf[samp]; cohs = coh[samp]
    read_ok = (reads == ys).astype(int)          # read-everything correctness, per item
    cf_ok = (cfs == ys).astype(int)              # cluster-first correctness, per item

    print(f"\nread-everything : {100*read_ok.mean():.1f}%  ({SAMPLE} calls on this sample)")
    print(f"cluster-first   : {100*cf_ok.mean():.1f}%  (0 calls on this sample; {K} naming calls total)")

    print(f"\n{'gate: read lowest-cohesion X% ':>32}{'calls':>7}{'hybrid acc':>12}{'vs read-all':>13}{'95% CI of diff':>22}{'sig?':>6}")
    for frac in [0.1, 0.2, 0.3, 0.5]:
        thr = np.quantile(cohs, frac)
        do_read = cohs <= thr
        hyb_ok = np.where(do_read, read_ok, cf_ok)
        diff = hyb_ok - read_ok
        # paired bootstrap on the per-item difference
        boots = np.array([diff[rng.integers(0, SAMPLE, SAMPLE)].mean() for _ in range(4000)])
        lo, hi = np.percentile(boots, [2.5, 97.5]) * 100
        sig = "YES" if (lo > 0 or hi < 0) else "no"
        print(f"{int(frac*100):>30}% {int(do_read.sum()):>7}{100*hyb_ok.mean():>11.1f}%"
              f"{100*diff.mean():>+12.1f}{f'[{lo:+.1f}, {hi:+.1f}]':>22}{sig:>6}")
    print("\nsig = the paired 95% bootstrap CI of (hybrid - read_everything) excludes 0.")
    print("If 'no', the honest wording is 'matches read-everything at a fraction of the calls'.")

if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set")
    main()
