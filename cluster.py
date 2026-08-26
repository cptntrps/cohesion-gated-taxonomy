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

def build_taxonomy(V, texts, labels, gold_names, anchors=None,
                   geometry="hyperbolic", do_naming=True, log=print):
    """anchors: list of human anchor names (strings) or None. geometry:
    'hyperbolic' (Poincare-ball k-means, owner directive) or 'flat' (KMeans).
    do_naming=False builds structure only (for flat-vs-hyperbolic purity compare).
    Returns (nodes, links, members, report)."""
    N = len(texts)
    anchors = [a for a in (anchors or []) if a and a.strip()]

    if geometry == "hyperbolic":
        from hyperbolic import make_ball, hyp_kmeans
        P, _, _ = make_ball(V)
        clus = lambda idx, k: hyp_kmeans(P[idx], k, SEED)
    else:
        clus = lambda idx, k: KMeans(n_clusters=min(k, len(idx)),
                                     random_state=SEED, n_init=4).fit(V[idx]).labels_
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
        nodes.append({"id": nid, "name": name, "depth": depth, "count": int(len(sub)),
                      "size": int(len(sub)) if len(sub) else 1, "parent": parent_id,
                      "gold_dominant": dom, "purity": pur, "gold_top3": top3,
                      "samples": reps, "human_named": human})
        links.append({"source": parent_id, "target": nid})
        members[nid] = sub.tolist() if hasattr(sub, "tolist") else list(sub)
        if not human and len(sub):
            jobs.append(nid)   # discovered -> LLM names it

    def divide(idx, parent_id, depth, path):
        k = min(KSUB, len(idx))
        if k < 2:
            return
        lab = clus(idx, k)
        for c in range(k):
            sub = idx[lab == c]
            if len(sub) == 0:
                continue
            npath = path + [c]
            nid = "L%d_%s" % (depth + 1, "_".join(map(str, npath)))
            add_node(nid, sub, depth + 1, parent_id, "...", False)
            if len(sub) >= MIN_SPLIT and depth + 1 < MAX_DEPTH:
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
            if len(sub) >= MIN_SPLIT:
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
                if len(sub) >= MIN_SPLIT:
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
            if len(sub) >= MIN_SPLIT:
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
