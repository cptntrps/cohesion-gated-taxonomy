#!/usr/bin/env python3
"""B2 generality suite: does cohesion-gated routing generalize beyond the
governing-law case that produced the '5x' number?

Owner correction (2026-09-01): the 5x savings figure was a property of ONE task
(LEDGAR governing law), where near-pure clusters make propagation unusually
cheap. The method's broad claims are:
  B1-gate: cohesion ranks clusters by propagation safety; the cohesion gate
           beats a random gate at equal budget on tasks never used to tune it.
  B2:      gated routing reaches parity with read-everything (paired CI lower
           bound >= -2.0 points) at SOME budget materially below 100%, and the
           parity budget is task-dependent, tracking cluster purity structure.

Tasks (never used for gate development):
  20ng    - assign each post to one of 20 newsgroups (noisy prose, 20-way)
  amazon  - assign each product to one of its 6 top categories (6-way)

Protocol per task, all predictions REAL model calls, same model both arms:
  1. KMeans K clusters on cached local embeddings.
  2. One cluster-level call per cluster (6 central representatives + label list).
  3. n=400 sampled per-item reads (text + label list), cached.
  4. Frontier at read fractions {0,10,20,30,40,50}%, paired bootstrap CI vs
     read-everything on the same sample; random-gate mean at equal budget.
  5. Calibration/holdout: pick the smallest passing fraction on half the
     sample, score once on the other half.

Falsifiers, declared before running:
  F1 (kills B1-gate): cohesion gate <= random gate at every budget on either task.
  F2 (kills B2):      no budget < 100% reaches parity at margin -2.0 on either
                      task, at n=400 power.
  Expected under B2: parity budgets DIFFER across tasks (the ratio is a corpus
  property, not a constant).
"""
import os, sys, json, pathlib, collections, urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from cluster import cohesion

HERE = pathlib.Path(__file__).parent
API = os.environ.get("EXTRACT_API", "https://api.deepseek.com/v1/chat/completions")
MODEL = os.environ.get("EXTRACT_MODEL", "deepseek-v4-flash")
SAMPLE = 400
WORKERS = 8
FRACS = [0.1, 0.2, 0.3, 0.4, 0.5]
MARGIN = 2.0

def chat(prompt, max_tokens=60):
    key = os.environ["DEEPSEEK_API_KEY"]
    body = json.dumps({"model": MODEL, "temperature": 0.0, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]}).encode()
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(API, data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}),
                timeout=120))
            return (r["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            sys.stderr.write(f"  llm retry: {e}\n")
    return ""

def match_label(out, labels):
    ol = out.lower()
    for lb in labels:
        if lb.lower() in ol:
            return lb
    return "NONE"

def ask(texts_batch, labels, kind):
    menu = "\n".join(f"- {lb}" for lb in labels)
    if kind == "cluster":
        head = ("The following texts all belong to ONE category from this list. "
                "Reply with ONLY the category name.\n\nCategories:\n" + menu + "\n\nTexts:\n")
        body = "\n---\n".join(t.strip()[:400] for t in texts_batch)
    else:
        head = ("Assign this text to ONE category from the list. "
                "Reply with ONLY the category name.\n\nCategories:\n" + menu + "\n\nText:\n")
        body = texts_batch[0].strip()[:1200]
    return match_label(chat(head + body), labels)

def load_20ng():
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    texts, targets, names = d.data, d.target, d.target_names
    keep = [i for i, t in enumerate(texts) if len(t.strip()) > 40][:6000]
    texts = [texts[i] for i in keep]
    y = np.array([names[targets[i]] for i in keep], dtype=object)
    V = np.load(HERE / "ng_emb.npz")["V"].astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return texts, y, V, sorted(set(y.tolist())), 40

def load_amazon():
    from datasets import load_dataset
    ds = load_dataset("ckandemir/amazon-products", split="train")
    names, cats, descs = ds["Product Name"], ds["Category"], ds["Description"]
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
    texts = [f"{names[i]}. {descs[i] or ''}" for i in idx]
    y = np.array([top(cats[i]) for i in idx], dtype=object)
    V = np.load(HERE / "amz_emb.npz")["V"].astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return texts, y, V, sorted(set(y.tolist())), 24

def boot_ci(diff, rng, n_boot=4000):
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5]) * 100
    return round(float(lo), 1), round(float(hi), 1)

def run_task(name, loader, out):
    texts, y, V, labels, K = loader()
    N = len(y)
    print(f"\n=== {name}: {N} items, {len(labels)} labels, K={K} ===", flush=True)
    km = KMeans(K, n_init=4, random_state=13).fit(V)
    clusters = [np.where(km.labels_ == c)[0] for c in range(K)]
    def reps(m):
        c = V[m].mean(0); c /= np.linalg.norm(c) + 1e-9
        order = m[np.argsort(-(V[m] @ c))]
        return [texts[i] for i in order[:6]]

    cache_f = HERE / f"b2_{name}_cache.json"
    cache = json.load(open(cache_f)) if cache_f.exists() else {"cluster": None, "reads": {}}
    if not cache["cluster"]:
        print(f"{K} cluster-level calls...", flush=True)
        with ThreadPoolExecutor(WORKERS) as ex:
            cache["cluster"] = list(ex.map(lambda m: ask(reps(m), labels, "cluster"), clusters))
        json.dump(cache, open(cache_f, "w"))
    cf = np.empty(N, dtype=object); coh = np.zeros(N)
    for c, m in enumerate(clusters):
        cf[m] = cache["cluster"][c]; coh[m] = cohesion(V[m])

    rng7 = np.random.default_rng(7)
    samp = rng7.choice(N, min(SAMPLE, N), replace=False)
    todo = [int(i) for i in samp if str(int(i)) not in cache["reads"]]
    if todo:
        print(f"{len(todo)} per-item reads...", flush=True)
        done = 0
        with ThreadPoolExecutor(WORKERS) as ex:
            for i, v in zip(todo, ex.map(lambda i: ask([texts[i]], labels, "item"), todo)):
                cache["reads"][str(i)] = v
                done += 1
                if done % 25 == 0:
                    json.dump(cache, open(cache_f, "w"))
                    print(f"  reads {done}/{len(todo)}", flush=True)
        json.dump(cache, open(cache_f, "w"))
    reads = np.array([cache["reads"][str(int(i))] for i in samp], dtype=object)
    ys = y[samp]; cfs = cf[samp]; cohs = coh[samp]
    read_ok = (reads == ys).astype(int); cf_ok = (cfs == ys).astype(int)
    print(f"read-everything {100*read_ok.mean():.1f}% | cluster-first {100*cf_ok.mean():.1f}% "
          f"(purity ceiling of this clustering)", flush=True)

    rng_b = np.random.default_rng(4242)
    rows = []
    parity_frac = None
    for frac in [0.0] + FRACS:
        thr = np.quantile(cohs, frac) if frac > 0 else -1
        do_read = cohs <= thr
        hyb = np.where(do_read, read_ok, cf_ok)
        lo, hi = boot_ci(hyb - read_ok, rng_b)
        n_reads = int(do_read.sum())
        rand = []
        for _ in range(500):
            pick = rng_b.choice(len(samp), n_reads, replace=False)
            mask = np.zeros(len(samp), dtype=bool); mask[pick] = True
            rand.append(np.where(mask, read_ok, cf_ok).mean())
        parity = lo >= -MARGIN
        if parity and parity_frac is None:
            parity_frac = frac
        rows.append(dict(frac=frac, calls=K + int(round(frac * N)),
                         acc=round(100*hyb.mean(), 1), ci=[lo, hi],
                         random_gate=round(100*float(np.mean(rand)), 1), parity=parity))
        print(f"  read {int(frac*100):>3}%: hybrid {100*hyb.mean():5.1f}% | CI [{lo:+.1f},{hi:+.1f}] "
              f"| random {100*np.mean(rand):5.1f}% | parity {'YES' if parity else 'no'}", flush=True)

    rng_cal = np.random.default_rng(101)
    perm = rng_cal.permutation(len(samp)); cal, ev = perm[:len(samp)//2], perm[len(samp)//2:]
    chosen = None
    for frac in FRACS:
        thr = np.quantile(cohs[cal], frac)
        hyb_cal = np.where(cohs[cal] <= thr, read_ok[cal], cf_ok[cal])
        if hyb_cal.mean() >= read_ok[cal].mean() - MARGIN/100:
            chosen = (frac, thr); break
    if chosen is None: chosen = (0.5, np.quantile(cohs[cal], 0.5))
    frac, thr = chosen
    hyb_ev = np.where(cohs[ev] <= thr, read_ok[ev], cf_ok[ev])
    lo, hi = boot_ci(hyb_ev - read_ok[ev], rng_b)
    oos = dict(chosen_frac=frac, eval_acc=round(100*float(hyb_ev.mean()), 1),
               eval_read_all=round(100*float(read_ok[ev].mean()), 1), ci=[lo, hi],
               parity=bool(lo >= -MARGIN))
    print(f"  OOS: chose {int(frac*100)}% on calibration; eval {oos['eval_acc']}% vs "
          f"read-all {oos['eval_read_all']}% CI [{lo:+.1f},{hi:+.1f}] parity {'YES' if oos['parity'] else 'no'}")

    out[name] = dict(n=N, k=K, labels=len(labels),
                     read_all=round(100*float(read_ok.mean()), 1),
                     cluster_first=round(100*float(cf_ok.mean()), 1),
                     frontier=rows, parity_frac=parity_frac, oos=oos)

def main():
    out = {}
    for name, loader in [("20ng", load_20ng), ("amazon", load_amazon)]:
        run_task(name, loader, out)
    g1 = any(all(r["acc"] <= r["random_gate"] for r in out[t]["frontier"] if r["frac"] > 0) for t in out)
    g2 = any(out[t]["parity_frac"] is None for t in out)
    out["verdicts"] = {
        "F1_gate_dead": bool(g1),
        "F2_no_parity_below_100": bool(g2),
        "parity_fracs": {t: out[t]["parity_frac"] for t in out if t != "verdicts"},
    }
    print("\nVERDICTS:", json.dumps(out["verdicts"]))
    json.dump(out, open(HERE / "b2_results.json", "w"), indent=1)
    print("wrote b2_results.json")

if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set")
    main()
