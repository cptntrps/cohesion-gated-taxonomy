#!/usr/bin/env python3
"""Kind emergence on reel cards that the classifier puts in `other`.

Read-only over reel_extract. Local models only, through the Barn Dance door (:11434):
nomic-embed-text embeds the card gists, KMeans clusters them, and a local LLM names
each cluster or maps it to an existing kind. Every run goes to MLflow
(experiment reel-kind-emergence) and to ~/data/reel-kind-emergence/.

Goal (declared before the run): the clusters carry structure (cohesion lift over a
size-matched random partition > 0.05), and the owner later accepts clusters that cover
>= 50% of the `other` cards. Failure of either means the mechanism does not earn a place.
"""
import datetime, json, os, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cluster import cohesion, multimodality

DOOR = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
NAME_MODEL = os.environ.get("NAME_MODEL", "gemma3:27b")
K_RANGE = range(6, 25)
MIN_SIZE = 4
N_REPR = 10
SEED = 13
OUT_DIR = os.path.expanduser("~/data/reel-kind-emergence")

SQL = """
with top as (
  select distinct on (shortcode) shortcode, label
  from reel_extract.card_suggestion where field = 'kind'
  order by shortcode, created_at desc, p_model desc)
select json_agg(json_build_object('shortcode', c.shortcode, 'creator', c.creator, 'gist', c.gist))
from top t join reel_extract.card c using (shortcode)
where t.label = 'other' and c.gist is not null and c.origin_class <> 'fixture'
"""
KINDS_SQL = """select json_agg(json_build_object('kind', kind, 'label', label, 'description', description)
               order by sort_order) from reel_extract.kind_option where active"""


def psql_json(sql):
    cmd = ["bash", "-c", 'source ~/src/infrastructure/lib/pg.sh && pg_load_dsn && psql "$PG_DSN" -X -At -v ON_ERROR_STOP=1 -c "$0"', sql]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    return json.loads(out) or []


def post(path, body, timeout=600):
    req = urllib.request.Request(DOOR + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def embed(texts):
    # Ollama 0.21.2 does not lowercase for nomic-embed; capitals collapse to [UNK].
    vecs = []
    for i in range(0, len(texts), 64):
        batch = ["clustering: " + t.lower()[:1500] for t in texts[i:i + 64]]
        vecs += post("/api/embed", {"model": EMBED_MODEL, "input": batch})["embeddings"]
    A = np.array(vecs, dtype=np.float32)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def random_baseline(V, sizes, reps=200, seed=SEED):
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(reps):
        perm = rng.permutation(len(V))
        cuts = np.cumsum(sizes)[:-1]
        means.append(np.mean([cohesion(V[g]) for g in np.split(perm, cuts)]))
    return float(np.mean(means))


def choose_k(V):
    best = None
    sweep = {}
    for k in K_RANGE:
        lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(V)
        if np.bincount(lab).min() < MIN_SIZE:
            continue
        s = float(silhouette_score(V, lab, metric="cosine"))
        sweep[k] = round(s, 4)
        if best is None or s > best[1]:
            best = (k, s, lab)
    if best is None:
        sys.exit(f"no k in {list(K_RANGE)} gives clusters of >= {MIN_SIZE} cards")
    return best, sweep


NAME_PROMPT = """You organize saved Instagram reels into kinds. These are the existing kinds:
{kinds}

Below are {n} reel summaries from ONE cluster of reels that the classifier put in "other".
{gists}

Decide if this cluster is one of the existing kinds (not "other"), or a NEW kind.
A new kind must name what the reels ARE (their content type or subject), in the same style as the existing kinds.
Answer with JSON only:
{{"existing_kind": "<slug of an existing kind, or null>",
  "kind": "<snake_case slug for a new kind, or null>",
  "label": "<short display label, 1-3 words>",
  "description": "<one sentence in the style of the existing descriptions>",
  "coherent": <true if the summaries share one kind, false if they are mixed>}}"""


def name_cluster(kinds_txt, gists):
    prompt = NAME_PROMPT.format(kinds=kinds_txt, n=len(gists),
                                gists="\n".join(f"- {g}" for g in gists))
    try:
        r = post("/api/chat", {"model": NAME_MODEL, "stream": False, "format": "json",
                               "options": {"temperature": 0, "seed": SEED},
                               "messages": [{"role": "user", "content": prompt}]})
        out = json.loads(r["message"]["content"])
        if not (out.get("existing_kind") or out.get("kind")):
            raise ValueError(f"no kind in answer: {out}")
        return out, None
    except Exception as e:  # recorded loudly per cluster, never backfilled
        return None, f"{type(e).__name__}: {e}"


DIRECT_PROMPT = """You organize saved Instagram reels into kinds. These are the existing kinds:
{kinds}

Below are {n} reel summaries, a random sample of the reels that the classifier put in "other".
{gists}

Propose the smallest set of NEW kinds that covers these reels, in the same style as the existing kinds.
Answer with JSON only:
{{"kinds": [{{"kind": "<snake_case slug>", "label": "<1-3 words>", "description": "<one sentence>",
             "count": <how many of the summaries above belong to it>}}]}}"""


def baseline_direct(n_sample=80):
    """No-clustering baseline: the same model proposes kinds from a random sample."""
    t0 = time.time()
    cards = psql_json(SQL)
    kinds = [k for k in psql_json(KINDS_SQL) if k["kind"] != "other"]
    rng = np.random.default_rng(SEED)
    sample = [cards[i]["gist"] for i in rng.choice(len(cards), min(n_sample, len(cards)), replace=False)]
    prompt = DIRECT_PROMPT.format(kinds="\n".join(f"- {x['kind']}: {x['description'] or x['label']}" for x in kinds),
                                  n=len(sample), gists="\n".join(f"- {g}" for g in sample))
    r = post("/api/chat", {"model": NAME_MODEL, "stream": False, "format": "json",
                           "options": {"temperature": 0, "seed": SEED, "num_ctx": 16384},
                           "messages": [{"role": "user", "content": prompt}]})
    out = json.loads(r["message"]["content"])
    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:8591")
    mlflow.set_experiment("reel-kind-emergence")
    with mlflow.start_run(run_name=f"baseline-direct-{NAME_MODEL}") as run:
        mlflow.log_params({"method": "direct_llm_no_clustering", "name_model": NAME_MODEL,
                           "n_sample": len(sample), "seed": SEED, "door": DOOR})
        mlflow.log_metrics({"n_kinds": len(out.get("kinds", [])), "seconds": round(time.time() - t0, 1),
                            "cost_usd": 0.0})
        mlflow.log_dict(out, "direct_kinds.json")
        print(json.dumps({"mlflow_run": run.info.run_id, **out}, indent=1, ensure_ascii=False))


def main():
    if "--baseline" in sys.argv:
        return baseline_direct()
    t0 = time.time()
    cards = psql_json(SQL)
    kinds = [k for k in psql_json(KINDS_SQL) if k["kind"] != "other"]
    if not cards:
        sys.exit("no `other` cards found")
    V = embed([c["gist"] for c in cards])
    (k, sil, lab), sweep = choose_k(V)
    sizes = np.bincount(lab).tolist()
    base = random_baseline(V, sizes)

    kinds_txt = "\n".join(f"- {x['kind']}: {x['description'] or x['label']}" for x in kinds)
    clusters = []
    for c in range(k):
        idx = np.where(lab == c)[0]
        Vm = V[idx]
        cen = Vm.mean(0); cen /= np.linalg.norm(cen) + 1e-9
        order = idx[np.argsort(-(Vm @ cen))]
        clusters.append({"cluster": c, "size": len(idx), "cohesion": round(cohesion(Vm), 4),
                         "multimodality": round(multimodality(Vm), 4),
                         "repr": [cards[i]["gist"] for i in order[:N_REPR]],
                         "members": [{"shortcode": cards[i]["shortcode"], "creator": cards[i]["creator"],
                                      "gist": cards[i]["gist"]} for i in order]})
    with ThreadPoolExecutor(2) as ex:
        names = list(ex.map(lambda cl: name_cluster(kinds_txt, cl["repr"]), clusters))
    for cl, (nm, err) in zip(clusters, names):
        cl["naming"], cl["naming_error"] = nm, err
    clusters.sort(key=lambda cl: -cl["size"])

    mean_coh = float(np.average([cl["cohesion"] for cl in clusters], weights=[cl["size"] for cl in clusters]))
    named = [cl for cl in clusters if cl["naming"]]
    to_existing = sum(cl["size"] for cl in named if cl["naming"].get("existing_kind"))
    to_new = sum(cl["size"] for cl in named if not cl["naming"].get("existing_kind"))
    metrics = {"n_cards": len(cards), "k": k, "silhouette": round(sil, 4),
               "cohesion_mean": round(mean_coh, 4), "cohesion_random_baseline": round(base, 4),
               "cohesion_lift": round(mean_coh - base, 4),
               "naming_failures": sum(1 for cl in clusters if cl["naming_error"]),
               "incoherent_clusters": sum(1 for cl in named if cl["naming"].get("coherent") is False),
               "cards_to_existing_kind": to_existing, "cards_to_new_kind": to_new,
               "seconds": round(time.time() - t0, 1), "cost_usd": 0.0}
    params = {"embed_model": EMBED_MODEL, "embed_text": "gist", "embed_prefix": "clustering: ",
              "name_model": NAME_MODEL, "k_range": f"{K_RANGE.start}-{K_RANGE.stop - 1}",
              "min_size": MIN_SIZE, "n_repr": N_REPR, "seed": SEED, "door": DOOR}

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    path = os.path.join(OUT_DIR, f"run-{stamp}.json")
    with open(path, "w") as f:
        json.dump({"params": params, "metrics": metrics, "k_sweep": sweep, "kinds": kinds,
                   "clusters": clusters}, f, indent=1, ensure_ascii=False)

    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:8591")
    mlflow.set_experiment("reel-kind-emergence")
    with mlflow.start_run(run_name=f"kmeans-gist-{NAME_MODEL}") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict({str(a): b for a, b in sweep.items()}, "k_sweep.json")
        mlflow.log_artifact(path)
        run_id = run.info.run_id
    print(json.dumps({"out": path, "mlflow_run": run_id, **metrics}, indent=1))
    for cl in clusters:
        nm = cl["naming"] or {}
        print(f"{cl['size']:4d}  coh={cl['cohesion']:.3f}  "
              f"{(nm.get('existing_kind') and '-> ' + nm['existing_kind']) or nm.get('kind') or 'FAILED: ' + str(cl['naming_error'])}"
              f"  | {nm.get('label', '')} | coherent={nm.get('coherent')}")


if __name__ == "__main__":
    main()
