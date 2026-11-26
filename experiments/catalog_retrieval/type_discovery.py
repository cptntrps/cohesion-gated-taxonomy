#!/usr/bin/env python3
"""WORK-2283 type discovery on the DEV split (rules in docs/catalog_retrieval/hierarchy_gate.md).

  python3 type_discovery.py <data_dir>

1. Each dev task descends the imposed tree (hyp_tree, dim 128, beam 6). Landing distance = 1 - cosine between the
   reduced query and the best landing node's descriptor. Uncovered = above the 90th percentile of dev landings.
2. Uncovered task vectors are Ward-clustered (distance threshold fixed below); a cluster of >= 3 tasks is a candidate
   type: its sector (members) and a name proposed by Claude Haiku 5.5 from member snippets.
3. Blind judge (local qwen3.8:27b, another family; texts never leave aurora): for each (type name, member) pair the
   judge answers whether the member belongs to the type, default false. Placebo = a candidate name moved to another
   candidate's members (or a fixed off-domain name); known-real = an existing decision domain on its own decisions.
   A candidate passes when its precision is >= 0.65 and the placebo precision is <= 0.20 (anchor-emergence bands).
Nothing is created; a passed candidate is a proposal for the type_decision flow and a Rule F folder.
"""
import hashlib
import json
import random
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

import retrieval as R
import run_arms as A

JUDGE = "qwen3.8:27b"
NAMER = "claude-haiku-5-5"
WARD_THRESHOLD = 1.0
OFF_DOMAIN = "sourdough baking"


def snippet(t, n=60):
    return " ".join(t.split()[:n])


def name_cluster(members):
    prompt = ("These agent tasks fall outside every existing folder of a catalog of skills and owner decisions. "
              "Propose ONE short folder name (2-4 words, lower case) for the kind of work they share, and one "
              "sentence that defines it. JSON only: {\"name\": \"...\", \"definition\": \"...\"}\n\n"
              + "\n".join(f"- {snippet(m)}" for m in members))
    r = subprocess.run(["claude", "-p", "--model", NAMER, "--output-format", "json", "--tools", "", "--setting-sources",
                        "", "--no-session-persistence"], input=prompt, capture_output=True, text=True, timeout=240)
    d = json.loads(r.stdout)
    d = [x for x in d if x.get("type") == "result"][-1] if isinstance(d, list) else d
    m = re.search(r"\{.*\}", d.get("result", ""), re.S)
    return json.loads(m.group(0)) if m else {"name": "unnamed", "definition": ""}


def judge(type_name, definition, text):
    prompt = (f"Folder: \"{type_name}\". {definition}\nDoes the item below belong in this folder? Default to no "
              f"unless it clearly fits. JSON only: {{\"belongs\": true|false}}\n\nITEM: {snippet(text, 80)}")
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=json.dumps(
        {"model": JUDGE, "prompt": prompt, "stream": False, "think": False,
         "options": {"num_ctx": 4096, "temperature": 0}}).encode(), headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))["response"]
    m = re.search(r"\{.*?\}", out, re.S)
    try:
        return bool(json.loads(m.group(0)).get("belongs")) if m else False
    except ValueError:
        return False


def precision(name, definition, texts):
    v = [judge(name, definition, t) for t in texts]
    return sum(v) / len(v) if v else float("nan"), v


def main(d):
    d = Path(d)
    cat, tasks = A.load(d)
    dev = [t for t in tasks if t["split"] == "dev"]
    Vi = R.unit(np.load(d / "emb_items.npz")["V"])
    Vq = R.unit(np.load(d / "emb_tasks_dev.npz")["V"])
    tree = R.build_tree([c["kind"] for c in cat], [c["group"] for c in cat], Vi)
    idx = R.HypIndex(tree, Vi, dim=128, tau=R.sala_tau(tree), beam=6)
    land = []
    for qv in Vq:
        qd = idx.reduce(qv)
        b = idx.descend(qd)
        land.append(1 - float(max(idx.desc[x] @ qd for x in b)))
    land = np.array(land)
    bound = float(np.percentile(land, 90))
    out_idx = [i for i, x in enumerate(land) if x > bound]
    res = {"dev_tasks": len(dev), "bound": bound, "uncovered": len(out_idx), "candidates": []}
    if len(out_idx) >= 3:
        lab = AgglomerativeClustering(n_clusters=None, distance_threshold=WARD_THRESHOLD,
                                      linkage="ward").fit_predict(Vq[out_idx])
        groups = {}
        for i, l in zip(out_idx, lab):
            groups.setdefault(int(l), []).append(i)
        cands = [g for g in groups.values() if len(g) >= 3]
        named = []
        for g in cands:
            texts = [dev[i]["text"] for i in g]
            nm = name_cluster(texts)
            named.append((g, texts, nm))
        rng = random.Random(13)
        for k, (g, texts, nm) in enumerate(named):
            p, votes = precision(nm["name"], nm.get("definition", ""), texts)
            if len(named) > 1:
                other = named[(k + 1) % len(named)][2]
                plac_name, plac_def = other["name"], other.get("definition", "")
            else:
                plac_name, plac_def = OFF_DOMAIN, "bread recipes and fermentation"
            pp, _ = precision(plac_name, plac_def, texts)
            res["candidates"].append({"size": len(g), "tids": [dev[i]["tid"] for i in g], "name": nm["name"],
                                      "definition": nm.get("definition", ""), "precision": p,
                                      "placebo_name": plac_name, "placebo_precision": pp,
                                      "passed": p >= 0.65 and pp <= 0.20})
    # known-real control: three decision domains on 8 of their own decisions each
    doms = {}
    for c in cat:
        if c["kind"] == "decision":
            doms.setdefault(c["group"], []).append(c)
    real = []
    for dom in sorted(doms, key=lambda x: -len(doms[x]))[:3]:
        items = sorted(doms[dom], key=lambda c: hashlib.sha256(c["id"].encode()).hexdigest())[:8]
        p, _ = precision(dom, f"owner decisions about {dom}", [A.item_text(c) for c in items])
        real.append({"domain": dom, "precision": p})
    res["known_real_controls"] = real
    res["passed"] = sum(c["passed"] for c in res["candidates"])
    json.dump(res, open(d / "results" / "dev" / "type_discovery.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "candidates"}, indent=1))
    for c in res["candidates"]:
        print(c["size"], c["name"], "precision", round(c["precision"], 2), "placebo", round(c["placebo_precision"], 2),
              "passed", c["passed"])


if __name__ == "__main__":
    main(sys.argv[1])
