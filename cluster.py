#!/usr/bin/env python3
"""Shared taxonomy builder: category-blind embedding clustering, LLM names only.

Two modes, unified (the spec's entry model):
- discover: pure top-down divisive KMeans on clause embeddings.
- anchored: human-named anchors seed the top level. Each clause joins its nearest
  anchor if cosine >= TAU, else falls to the discovery pool that KMeans splits.
  Anchor clusters keep the HUMAN name; discovered clusters are LLM-named.
  An empty anchor (no clause clears TAU) still appears, awaiting matches.

Gold labels are used ONLY to score purity, never to cluster or to name.
"""
import json, collections, urllib.request
import numpy as np
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
from naming import name_cluster, MODEL

MAX_DEPTH = 3
KROOT, KSUB = 8, 5
MIN_SPLIT = 1200
N_REPR = 10
SEED = 13
NAME_WORKERS = 8
TAU = 0.60            # cosine gate for a clause to join a human anchor (measured 2026-08-26)
OLLAMA_EMBED = "http://localhost:11434/api/embed"

def embed_names(names):
    """nomic-embed-text, lowercased+normalized — same space as the clause vectors."""
    req = urllib.request.Request(OLLAMA_EMBED, data=json.dumps({
        "model": "nomic-embed-text", "input": [n.lower()[:400] for n in names]
    }).encode(), headers={"Content-Type": "application/json"})
    emb = json.load(urllib.request.urlopen(req, timeout=120))["embeddings"]
    A = np.array(emb, dtype=np.float32)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)

def _purity(members, labels, gold_names):
    if len(members) == 0:
        return None, 0.0, []
    cnt = collections.Counter(labels[members].tolist())
    top_label, top_n = cnt.most_common(1)[0]
    top3 = [[gold_names[l], int(c)] for l, c in cnt.most_common(3)]
    return gold_names[top_label], round(100 * top_n / len(members), 1), top3

def cohesion(Vm):
    """Gold-free cluster quality: mean cosine of members to their unit centroid.
    Measured to correlate with gold purity at Spearman +0.74 (purity_probe.py)."""
    if len(Vm) < 2:
        return 1.0
    c = Vm.mean(0); n = np.linalg.norm(c)
    return float((Vm @ (c / n)).mean()) if n > 1e-9 else 0.0

def multimodality(Vm, sample=500, seed=13):
    """Gold-free 'is this a FALSE MERGE?' signal: cosine silhouette of the best
    2-means split. High => two distinct sub-concepts stuck together (a real seam,
    should split). Low => unimodal/coherent (coarse is fine for EMERGE, not wrong).
    This is the emerge criterion; cohesion/purity are the map criteria."""
    n = len(Vm)
    if n < 20:
        return 0.0
    if n > sample:
        Vm = Vm[np.random.default_rng(seed).choice(n, sample, replace=False)]
    from sklearn.metrics import silhouette_score
    lab = KMeans(n_clusters=2, n_init=3, random_state=seed).fit(Vm).labels_
    if len(set(lab.tolist())) < 2:
        return 0.0
    return float(silhouette_score(Vm, lab, metric="cosine"))

def refine_partition(Vsub, lab, temp=0.0, sweeps=8, frac=0.35, seed=13):
    """Random-walk local search that raises cohesion BEFORE any LLM call.
    Each sweep: recompute unit centroids, test every clause against all centroids,
    and MOVE a random `frac` of the clauses whose cohesion would improve (the
    walk's stochastic step). With temp>0 also move a few non-improving clauses
    (Metropolis explore) to escape local optima. Objective is cohesion only — no
    gold, so it is legal to run before naming. Vectorized: sweeps x (n,k) matmul."""
    lab = lab.copy().astype(int)
    n, d = Vsub.shape
    k = int(lab.max()) + 1
    if k < 2 or n < 8:
        return lab
    rng = np.random.default_rng(seed)
    ar = np.arange(n)
    for _ in range(sweeps):
        cents = np.zeros((k, d))
        for c in range(k):
            m = lab == c
            if m.any():
                cents[c] = Vsub[m].mean(0)
        cents /= (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
        cos = Vsub @ cents.T                       # (n, k) cosine to each centroid
        best = cos.argmax(1)
        improve = np.where((best != lab) & (cos[ar, best] - cos[ar, lab] > 0))[0]
        moved = improve[rng.random(len(improve)) < frac] if len(improve) else improve
        if temp > 0:                               # occasional non-improving hop
            other = np.where(best == lab)[0]
            if len(other):
                hop = other[rng.random(len(other)) < temp]
                alt = rng.integers(0, k, size=len(hop))
                lab[hop] = alt
        if len(moved) == 0 and temp == 0:
            break
        lab[moved] = best[moved]
    # drop any emptied cluster ids to a contiguous range
    uniq = {c: i for i, c in enumerate(sorted(set(lab.tolist())))}
    return np.array([uniq[c] for c in lab], dtype=int)

def build_taxonomy(V, texts, labels, gold_names, anchors=None,
                   geometry="hyperbolic", do_naming=True,
                   refine=False, refine_temp=0.0, mode="map", mm_split=0.10,
                   log=print):
    """anchors: list of human anchor names (strings) or None. geometry:
    'hyperbolic' (Poincare-ball k-means, owner directive) or 'flat' (KMeans).
    do_naming=False builds structure only (for flat-vs-hyperbolic purity compare).
    Returns (nodes, links, members, report)."""
    N = len(texts)
    anchors = [a for a in (anchors or []) if a and a.strip()]

    if geometry == "hyperbolic":
        from hyperbolic import make_ball, hyp_kmeans
        P, _, _ = make_ball(V)
        _base = lambda idx, k: hyp_kmeans(P[idx], k, SEED)
    else:
        _base = lambda idx, k: KMeans(n_clusters=min(k, len(idx)),
                                      random_state=SEED, n_init=4).fit(V[idx]).labels_
    def clus(idx, k):
        lab = _base(idx, k)
        if refine:
            lab = refine_partition(V[idx], lab, temp=refine_temp)   # cohesion-max, pre-LLM
        return lab
    nodes = [{"id": "root", "name": "All LEDGAR clauses", "depth": 0, "count": N,
              "size": N, "parent": None, "samples": [], "human_named": False}]
    links, members, jobs = [], {"root": list(range(N))}, []

    def reps_of(sub):
        c = V[sub].mean(axis=0); c = c / (np.linalg.norm(c) + 1e-9)
        order = np.argsort(-(V[sub] @ c))
        return [texts[sub[i]].strip()[:400] for i in order[:N_REPR]]

    def add_node(nid, sub, depth, parent_id, name, human):
        dom, pur, top3 = _purity(sub, labels, gold_names)
        reps = reps_of(sub) if len(sub) else []
        coh = round(cohesion(V[sub]), 3) if len(sub) else None
        nodes.append({"id": nid, "name": name, "depth": depth, "count": int(len(sub)),
                      "size": int(len(sub)) if len(sub) else 1, "parent": parent_id,
                      "gold_dominant": dom, "purity": pur, "gold_top3": top3,
                      "cohesion": coh, "samples": reps, "human_named": human})
        links.append({"source": parent_id, "target": nid})
        members[nid] = sub.tolist() if hasattr(sub, "tolist") else list(sub)
        if not human and len(sub):
            jobs.append(nid)   # discovered -> LLM names it

    emerge = (mode == "emerge")
    min_sz = 60 if emerge else MIN_SPLIT      # emerge: split down to coherence, not depth
    max_d = 6 if emerge else MAX_DEPTH

    def divide(idx, parent_id, depth, path):
        if depth >= max_d or len(idx) < min_sz:
            return
        # EMERGE: split only where there is a real seam (multimodal); stop when
        # unimodal even if coarse. MAP: always split to depth (maximize purity).
        if emerge and multimodality(V[idx]) < mm_split:
            return
        k = min(KSUB, len(idx))
        if k < 2:
            return
        lab = clus(idx, k)
        subs = [idx[lab == c] for c in range(int(lab.max()) + 1) if (lab == c).any()]
        if len(subs) < 2:
            return
        for c, sub in enumerate(subs):
            npath = path + [c]
            nid = "L%d_%s" % (depth + 1, "_".join(map(str, npath)))
            add_node(nid, sub, depth + 1, parent_id, "...", False)
            divide(sub, nid, depth + 1, npath)

    all_idx = np.arange(N)
    if anchors:
        log(f"anchored rebuild: {len(anchors)} anchors, gate cos>= {TAU}")
        A = embed_names(anchors)
        sims = V @ A.T                       # (N, a)
        best = sims.argmax(axis=1)
        bestsim = sims.max(axis=1)
        joined = bestsim >= TAU
        for ai, name in enumerate(anchors):
            sub = all_idx[(best == ai) & joined]
            nid = "A_%d" % ai
            add_node(nid, sub, 1, "root", name, True)   # human name, fixed
            divide(sub, nid, 1, [ai])
        pool = all_idx[~joined]
        log(f"  {joined.sum()} clauses joined anchors, {len(pool)} to discovery")
        if len(pool) >= 2:
            kd = max(2, KROOT - len(anchors))
            lab = clus(pool, min(kd, len(pool)))
            for c in range(int(lab.max()) + 1):
                sub = pool[lab == c]
                if len(sub) == 0:
                    continue
                nid = "D_%d" % c
                add_node(nid, sub, 1, "root", "...", False)
                divide(sub, nid, 1, [100 + c])
    else:
        log(f"discovery rebuild: no anchors, geometry={geometry}")
        lab = clus(all_idx, KROOT)
        for c in range(int(lab.max()) + 1):
            sub = all_idx[lab == c]
            if len(sub) == 0:
                continue
            nid = "L1_%d" % c
            add_node(nid, sub, 1, "root", "...", False)
            divide(sub, nid, 1, [c])

    # ---- name every DISCOVERED cluster in parallel, with ancestor context ----
    by_id = {nd["id"]: nd for nd in nodes}
    def anc(nid):
        chain, cur = [], by_id[nid]["parent"]
        while cur and cur != "root":
            chain.append(by_id[cur]["name"]); cur = by_id[cur]["parent"]
        return list(reversed(chain))
    def do(nid):
        name, ok = name_cluster(by_id[nid]["samples"], anc(nid))
        return nid, name, ok
    fails = 0
    if do_naming and jobs:
        log(f"naming {len(jobs)} discovered clusters via {MODEL}")
        with ThreadPoolExecutor(max_workers=NAME_WORKERS) as ex:
            for nid, name, ok in ex.map(do, jobs):
                by_id[nid]["name"] = name; fails += not ok

    child_ids = {l["source"] for l in links}
    leaves = [nd for nd in nodes if nd["depth"] > 0 and nd["id"] not in child_ids and nd["count"] > 0]
    lp = [nd["purity"] for nd in leaves]; ls = [nd["count"] for nd in leaves]
    baseline = round(100 * collections.Counter(labels.tolist()).most_common(1)[0][1] / N, 1)
    w_purity = round(float(np.average(lp, weights=ls)), 1) if leaves else 0.0

    # ---- hyperbolic anomaly / novelty: Poincare misfit to own leaf centroid ----
    # This is where curvature earns its place. A clause pushed far from every leaf
    # centroid (large geodesic misfit + large ball radius) fits no known cluster ->
    # a NEW-concept candidate that feeds the human anchor loop. Flat has no such axis.
    anomalies = []
    if geometry == "hyperbolic":
        from hyperbolic import _ball_mean, poincare_dist_matrix
        radius = np.linalg.norm(P, axis=1)
        PER_LEAF = 3   # diversify: at most this many candidates per leaf
        cand = []      # (score, clause_index, leaf)
        for leaf in leaves:
            idx = np.array(members[leaf["id"]], dtype=int)
            if len(idx) == 0:
                continue
            cen = _ball_mean(P[idx])
            d = poincare_dist_matrix(P[idx], cen[None, :])[:, 0]
            score = d + 0.15 * radius[idx]
            for j in np.argsort(-score)[:PER_LEAF]:
                cand.append((float(score[j]), int(idx[j]), float(d[j]), leaf))
        cand.sort(key=lambda c: -c[0])
        for _, i, mf, lf in cand[:40]:
            anomalies.append({
                "idx": i, "text": texts[i].strip()[:400],
                "misfit": round(mf, 3), "radius": round(float(radius[i]), 3),
                "leaf_id": lf["id"], "leaf_name": lf["name"], "gold": gold_names[labels[i]],
            })

    report = {
        "n_provisions": N, "n_gold_types": len(gold_names),
        "baseline_majority_pct": baseline, "leaf_weighted_purity_pct": w_purity,
        "goal_threshold_pct": 2 * baseline,
        "verdict": "PASS" if w_purity >= 2 * baseline else "FAIL",
        "n_leaves": len(leaves), "n_nodes": len(nodes) - 1,
        "distinct_dominant_gold_types": len({nd["gold_dominant"] for nd in leaves}),
        "naming_failures": fails, "n_anchors": len(anchors), "n_anomalies": len(anomalies),
        "max_depth": MAX_DEPTH, "name_model": MODEL, "geometry": geometry,
    }
    return nodes, links, members, report, anomalies
