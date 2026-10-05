#!/usr/bin/env python3
"""Topic (subject) emergence over ALL reel cards: a 2-level tree, local models only.

Hypothesis: embeddings of gists encode subject, not form, so topic clusters carry
structure where kind clusters did not (reels_emerge.py: lift 0.026).
Gate (declared before the run): top-level cohesion lift over a size-matched random
partition >= 0.05. Configs: A = raw gist, B = gist with template words removed.
Baseline: reels_topics.py --baseline (one LLM call proposes topics from a sample).
"""
import datetime, json, os, re, sys, time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cluster import cohesion
from reels_emerge import DOOR, NAME_MODEL, SEED, post, psql_json, random_baseline

OUT_DIR = os.path.expanduser("~/data/reel-kind-emergence")
K_TOP = range(6, 17)
K_SUB = range(2, 6)
SPLIT_AT = 70
MIN_SIZE = 5
N_REPR = 12

SQL = """select json_agg(json_build_object('shortcode', shortcode, 'creator', creator, 'gist', gist))
         from reel_extract.card where gist is not null and origin_class <> 'fixture'"""

TEMPLATE = re.compile(
    r"^(an?|the)\s+(instagram\s+|saved\s+|short\s+|brazilian\s+|portuguese[- ]language\s+|promotional\s+)*"
    r"(reel|carousel|clip|video|post|screenshot)(\s+(by|from)\s+[\w.@]+)?\s*(shows?|showing|features?|featuring|"
    r"lists?|listing|explains?|explaining|presents?|presenting|offers?|offering|displays?|displaying|"
    r"recommends?|describes?|demonstrates?|argues?|says?|claims?|introduces?|introducing)?\s*(that\s+)?",
    re.I)


def strip_template(g, creator):
    g = TEMPLATE.sub("", g.strip())
    if creator:
        g = re.sub(re.escape(creator), "", g, flags=re.I)
    return re.sub(r"\s+", " ", g).strip() or g


def embed(texts):
    vecs = []
    for i in range(0, len(texts), 64):
        vecs += post("/api/embed", {"model": "nomic-embed-text",
                                    "input": ["clustering: " + t.lower()[:1500] for t in texts[i:i + 64]]})["embeddings"]
    A = np.array(vecs, dtype=np.float32)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def best_kmeans(V, ks):
    best, sweep = None, {}
    for k in ks:
        if k >= len(V):
            break
        lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(V)
        if np.bincount(lab).min() < MIN_SIZE:
            continue
        s = float(silhouette_score(V, lab, metric="cosine"))
        sweep[k] = round(s, 4)
        if best is None or s > best[1]:
            best = (k, s, lab)
    return best, sweep


def lift(V, lab):
    sizes = np.bincount(lab).tolist()
    coh = float(np.average([cohesion(V[lab == c]) for c in range(len(sizes))], weights=sizes))
    return coh, random_baseline(V, sizes)


TOPIC_PROMPT = """You build a TOPIC taxonomy for saved Instagram reels. A topic is the SUBJECT a reel is about
(for example AI, travel, cooking, politics), never its form (not "list", "meme", "tutorial").
{context}Below are {n} reel summaries from ONE cluster.
{gists}

Answer with JSON only:
{{"topic": "<snake_case slug>", "label": "<1-3 words>",
  "description": "<one sentence: what subjects belong here>",
  "coherent": <true only if almost all summaries share this subject>,
  "outliers": <how many of the summaries do NOT fit the topic>}}"""


def name(gists, parent=None):
    ctx = f'This cluster is a sub-topic inside the topic "{parent}". Name the narrower subject.\n' if parent else ""
    prompt = TOPIC_PROMPT.format(context=ctx, n=len(gists), gists="\n".join(f"- {g}" for g in gists))
    try:
        r = post("/api/chat", {"model": NAME_MODEL, "stream": False, "format": "json",
                               "options": {"temperature": 0, "seed": SEED},
                               "messages": [{"role": "user", "content": prompt}]})
        out = json.loads(r["message"]["content"])
        if not out.get("topic"):
            raise ValueError(f"no topic: {out}")
        return out, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def node(V, idx, cards, raw):
    Vm = V[idx]
    cen = Vm.mean(0); cen /= np.linalg.norm(cen) + 1e-9
    order = idx[np.argsort(-(Vm @ cen))]
    return {"size": len(idx), "cohesion": round(cohesion(Vm), 4),
            "repr": [raw[i] for i in order[:N_REPR]],
            "creators_top": _top_creators([cards[i]["creator"] for i in idx]),
            "members": [cards[i]["shortcode"] for i in order]}


def _top_creators(cs):
    vals, cnt = np.unique([c or "?" for c in cs], return_counts=True)
    o = np.argsort(-cnt)[:3]
    return [[str(vals[i]), int(cnt[i])] for i in o]


def run(config, cards, mlflow):
    t0 = time.time()
    raw = [c["gist"] for c in cards]
    texts = raw if config == "A" else [strip_template(c["gist"], c["creator"]) for c in cards]
    V = embed(texts)
    (k, sil, lab), sweep = best_kmeans(V, K_TOP)
    coh, base = lift(V, lab)
    tree = []
    for c in range(k):
        idx = np.where(lab == c)[0]
        top = node(V, idx, cards, raw)
        top["naming"], top["naming_error"] = name(top["repr"])
        top["children"] = []
        if len(idx) >= SPLIT_AT:
            res, _ = best_kmeans(V[idx], K_SUB)
            if res:
                _, _, sl = res
                parent = (top["naming"] or {}).get("label")
                for s in range(sl.max() + 1):
                    ch = node(V, idx[sl == s], cards, raw)
                    ch["naming"], ch["naming_error"] = name(ch["repr"], parent)
                    top["children"].append(ch)
        tree.append(top)
    tree.sort(key=lambda n: -n["size"])
    nodes = tree + [ch for n in tree for ch in n["children"]]
    metrics = {"n_cards": len(cards), "k_top": k, "silhouette": round(sil, 4), "cohesion_mean": round(coh, 4),
               "cohesion_random_baseline": round(base, 4), "cohesion_lift": round(coh - base, 4),
               "gate_pass": float(coh - base >= 0.05),
               "n_subtopics": sum(len(n["children"]) for n in tree),
               "naming_failures": sum(1 for n in nodes if n["naming_error"]),
               "incoherent_nodes": sum(1 for n in nodes if (n["naming"] or {}).get("coherent") is False),
               "outliers_reported": sum(int((n["naming"] or {}).get("outliers") or 0) for n in nodes),
               "seconds": round(time.time() - t0, 1), "cost_usd": 0.0}
    params = {"config": config, "text": "gist" if config == "A" else "gist_template_stripped",
              "embed_model": "nomic-embed-text", "name_model": NAME_MODEL, "k_top": f"{K_TOP.start}-{K_TOP.stop - 1}",
              "k_sub": f"{K_SUB.start}-{K_SUB.stop - 1}", "split_at": SPLIT_AT, "min_size": MIN_SIZE,
              "seed": SEED, "door": DOOR}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"topics-{config}-{datetime.datetime.now():%Y-%m-%dT%H%M%S}.json")
    json.dump({"params": params, "metrics": metrics, "k_sweep": sweep, "tree": tree}, open(path, "w"),
              indent=1, ensure_ascii=False)
    with mlflow.start_run(run_name=f"topics-{config}-{NAME_MODEL}") as r:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(path)
        rid = r.info.run_id
    return path, rid, metrics, tree


BASE_PROMPT = """You build a TOPIC taxonomy for saved Instagram reels. A topic is the SUBJECT a reel is about
(for example AI, travel, cooking, politics), never its form (not "list", "meme", "tutorial").
Below are {n} reel summaries, a random sample of all saved reels.
{gists}

Propose 8-14 top-level topics that cover them, each with 0-4 sub-topics where a topic is large.
Answer with JSON only:
{{"topics": [{{"topic": "<slug>", "label": "<1-3 words>", "count": <summaries above that belong>,
              "subtopics": ["<label>", ...]}}]}}"""


def baseline(cards, mlflow, n=120):
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    sample = [cards[i]["gist"] for i in rng.choice(len(cards), n, replace=False)]
    r = post("/api/chat", {"model": NAME_MODEL, "stream": False, "format": "json",
                           "options": {"temperature": 0, "seed": SEED, "num_ctx": 24576},
                           "messages": [{"role": "user", "content": BASE_PROMPT.format(
                               n=n, gists="\n".join(f"- {g}" for g in sample))}]})
    out = json.loads(r["message"]["content"])
    with mlflow.start_run(run_name=f"topics-baseline-direct-{NAME_MODEL}") as run:
        mlflow.log_params({"method": "direct_llm_no_clustering", "name_model": NAME_MODEL, "n_sample": n,
                           "seed": SEED, "door": DOOR})
        mlflow.log_metrics({"n_topics": len(out.get("topics", [])), "seconds": round(time.time() - t0, 1),
                            "cost_usd": 0.0})
        mlflow.log_dict(out, "baseline_topics.json")
        return run.info.run_id, out


def show(tree):
    for n in tree:
        nm = n["naming"] or {}
        print(f"{n['size']:4d} coh={n['cohesion']:.3f} {nm.get('label', 'FAILED ' + str(n['naming_error']))}"
              f"  [coherent={nm.get('coherent')} outliers={nm.get('outliers')}]  creators={n['creators_top']}")
        for ch in n["children"]:
            c = ch["naming"] or {}
            print(f"     {ch['size']:4d} coh={ch['cohesion']:.3f} {c.get('label', 'FAILED')}"
                  f"  [coherent={c.get('coherent')} outliers={c.get('outliers')}]")


def main():
    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:8591")
    mlflow.set_experiment("reel-kind-emergence")
    cards = psql_json(SQL)
    if "--baseline" in sys.argv:
        rid, out = baseline(cards, mlflow)
        print(json.dumps({"mlflow_run": rid, **out}, indent=1, ensure_ascii=False))
        return
    for config in sys.argv[1:] or ["A", "B"]:
        path, rid, metrics, tree = run(config, cards, mlflow)
        print(f"\n=== config {config}  run {rid}  {path}")
        print(json.dumps(metrics))
        show(tree)


if __name__ == "__main__":
    main()
