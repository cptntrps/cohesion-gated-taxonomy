#!/usr/bin/env python3
"""Initial LEDGAR taxonomy build (discovery mode, no anchors).

Geometry builds the structure, the LLM only names clusters top-to-bottom.
Writes tree.json (+ report) and members.json (node_id -> provision indices) so
server.py can re-cluster the whole corpus with human anchors on demand.
"""
import json, pathlib, sys, os
import numpy as np

HERE = pathlib.Path(__file__).parent
EMB = pathlib.Path("/home/gui/projects/hyperbolic-experiments/crossover/ledgar_emb.npz")

def load():
    from datasets import load_dataset
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    texts = list(ds["text"])
    gold_names = ds.features["label"].names
    labels = np.array(ds["label"])
    V = np.load(EMB)["V"].astype(np.float32)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    n = min(len(texts), len(V))
    return texts[:n], labels[:n], gold_names, V[:n]

def main():
    from cluster import build_taxonomy
    texts, labels, gold_names, V = load()
    print(f"loaded {len(texts)} provisions", flush=True)

    # honest side-by-side: flat structure purity (no naming, cheap)
    print("\n-- flat comparison (structure only, no LLM) --", flush=True)
    _, _, _, flat = build_taxonomy(V, texts, labels, gold_names, anchors=None,
                                   geometry="flat", do_naming=False,
                                   log=lambda m: print("  "+m, flush=True))
    print(f"  flat leaf purity: {flat['leaf_weighted_purity_pct']}%", flush=True)

    print("\n-- hyperbolic build (named) --", flush=True)
    nodes, links, members, report = build_taxonomy(
        V, texts, labels, gold_names, anchors=None, geometry="hyperbolic",
        log=lambda m: print("  "+m, flush=True))
    report["flat_leaf_purity_pct"] = flat["leaf_weighted_purity_pct"]
    tree = {"nodes": nodes, "links": links, "report": report, "anchors": []}
    json.dump(tree, open(HERE / "tree.json", "w"), indent=1)
    json.dump(members, open(HERE / "members.json", "w"))
    print("\n==== REPORT ====\n" + json.dumps(report, indent=1))
    print(f"\nwrote tree.json ({len(nodes)} nodes) + members.json")

if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set")
    main()
