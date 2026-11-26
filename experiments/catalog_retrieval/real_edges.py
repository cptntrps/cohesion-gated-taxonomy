#!/usr/bin/env python3
"""WORK-2283 hierarchy gate (docs/catalog_retrieval/hierarchy_gate.md): real edges and Gromov delta vs shuffles.

  source ~/src/infrastructure/lib/pg.sh && pg_load_dsn && python3 real_edges.py <study_dir>

Edges come from the catalog's own texts (decisions in ops.owner_decisions, SKILL.md bodies), never from tasks.
"""
import collections
import json
import os
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np

CITE = re.compile(r"owner_decision:([a-z0-9][a-z0-9/_.-]{2,120}[a-z0-9])")
SKILLS = Path.home() / "src/agents/skills"


def names_skill(text, sk):
    s = re.escape(sk)
    return re.search(rf"skills/{s}/|\[{s}\]\(|`{s}` skill|skill `{s}`|\b{s} skill\b|\bskill {s}\b", text) is not None


def extract(cat, decision_rows):
    subj = {c["name"] for c in cat if c["kind"] == "decision"}
    skills = sorted(c["name"] for c in cat if c["kind"] == "skill")
    edges = set()
    for s, text, sup in decision_rows:
        if s not in subj:
            continue
        for c in set(x.rstrip(".,;:)") for x in CITE.findall(text)):
            if c in subj and c != s:
                edges.add((f"decision:{s}", f"decision:{c}", "cites"))
        if sup and sup in subj and sup != s:
            edges.add((f"decision:{s}", f"decision:{sup}", "supersedes"))
        for sk in skills:
            if names_skill(text, sk):
                edges.add((f"decision:{s}", f"skill:{sk}", "names"))
    for sk in skills:
        body = (SKILLS / sk / "SKILL.md").read_text(errors="replace")
        for c in set(x.rstrip(".,;:)") for x in CITE.findall(body)):
            if c in subj:
                edges.add((f"skill:{sk}", f"decision:{c}", "cites"))
        for o in skills:
            if o != sk and names_skill(body, o):
                edges.add((f"skill:{sk}", f"skill:{o}", "refs"))
    return sorted(edges)


def four_point(G, n_quads=20000, seed=13):
    nodes = list(G.nodes)
    rng = np.random.default_rng(seed)
    sp = dict(nx.all_pairs_shortest_path_length(G))
    q = rng.choice(len(nodes), size=(n_quads, 4))
    ds = []
    for a, b, c, e in q:
        x, y, z, w = nodes[a], nodes[b], nodes[c], nodes[e]
        s = sorted([sp[x][y] + sp[z][w], sp[x][z] + sp[y][w], sp[x][w] + sp[y][z]], reverse=True)
        ds.append((s[0] - s[1]) / 2)
    pairs = rng.choice(len(nodes), size=(5000, 2))
    mean_d = float(np.mean([sp[nodes[i]][nodes[j]] for i, j in pairs]))
    return float(np.mean(ds)), float(np.max(ds)), mean_d, nx.diameter(G)


def lcc(G):
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def gate(edges, n_items, replicas=20, seed=13):
    G = nx.Graph()
    G.add_edges_from((a, b) for a, b, _ in edges)
    covered = G.number_of_nodes()
    L = lcc(G)
    real = four_point(L)
    shuf = []
    for r in range(replicas):
        H = G.copy()
        nx.double_edge_swap(H, nswap=10 * H.number_of_edges(), max_tries=200 * H.number_of_edges(), seed=seed + r)
        shuf.append(four_point(lcc(H), seed=seed + r)[0])
    parents = collections.Counter(a for a, b, _ in edges)
    children = collections.Counter(b for a, b, _ in edges)
    res = {
        "edges": len(edges), "by_relation": dict(collections.Counter(k for _, _, k in edges)),
        "items_with_edges": covered, "coverage": covered / n_items,
        "lcc_nodes": L.number_of_nodes(), "lcc_edges": L.number_of_edges(),
        "cycle_rank_per_node": (G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G))
        / max(G.number_of_nodes(), 1),
        "components": nx.number_connected_components(G),
        "real_mean_delta": real[0], "real_max_delta": real[1], "real_mean_distance": real[2], "real_diameter": real[3],
        "real_delta_over_mean_distance": real[0] / real[2],
        "shuffled_mean_delta": shuf, "shuffled_mean": float(np.mean(shuf)),
        "items_cited_by_2plus": sum(1 for v in children.values() if v >= 2),
        "items_citing_2plus": sum(1 for v in parents.values() if v >= 2),
    }
    res["H_G1_coverage_pass"] = res["coverage"] >= 0.5
    res["H_G2_tree_pass"] = real[0] < min(shuf) and real[0] <= 0.8 * res["shuffled_mean"]
    res["gate_pass"] = res["H_G1_coverage_pass"] and res["H_G2_tree_pass"]
    return res


def main(study):
    import psycopg2
    d = Path(study)
    cat = [json.loads(l) for l in open(d / "catalog.jsonl")]
    conn = psycopg2.connect(os.environ["PG_DSN"])
    with conn, conn.cursor() as cur:
        cur.execute("""SELECT DISTINCT ON (o.subject_key) o.subject_key,
                              concat_ws(' ', o.question, o.decision, o.why, o.note, o.evidence::text, o.authority_quote),
                              (SELECT s.subject_key FROM ops.owner_decisions s WHERE s.id = o.supersedes)
                       FROM ops.owner_decisions o ORDER BY o.subject_key, o.id DESC""")
        rows = cur.fetchall()
    edges = extract(cat, rows)
    json.dump(edges, open(d / "real_edges.json", "w"))
    res = gate(edges, len(cat))
    json.dump(res, open(d / "hierarchy_gate_result.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
