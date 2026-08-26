#!/usr/bin/env python3
"""Least-words naming test: does a MINIMAL-word prompt name clusters closer to the
canonical (gold) term and collapse duplicates, without performing worse?

Names the SAME leaves two ways (verbose = current, minimal = new), then scores each
name against the leaf's gold dominant type by embedding cosine (gold-free at naming
time; gold only used to score). Also counts distinct names per gold type (lower =
better consolidation). 'Benefits and Expense Reimbursements' vs 'Benefits'.
"""
import os, sys, json, pathlib, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import urllib.request

API = "https://api.deepseek.com/v1/chat/completions"; MODEL = "deepseek-v4-flash"

VERBOSE = ("Below are contract clauses grouped by similar language. Give the group a "
           "short category name (2 to 5 words), the kind of heading a lawyer uses in a "
           "clause library.\n\n{clauses}\n\nReply with ONLY the category name.")
MINIMAL = ("Below are contract clauses grouped by similar language. Give the SHORTEST "
           "canonical category name — 1 to 3 words, the single standard clause-library "
           "term. Do NOT combine multiple topics with 'and'; pick the dominant one. "
           "Prefer the common legal term (e.g. 'Benefits', 'Notices', 'Governing Law').\n\n"
           "{clauses}\n\nReply with ONLY the name.")

def name(prompt, samples):
    key = os.environ["DEEPSEEK_API_KEY"]
    body = json.dumps({"model": MODEL, "temperature": 0.0, "max_tokens": 800, "stream": False,
        "messages": [{"role": "user", "content": prompt.format(
            clauses="\n\n".join("- " + s[:300] for s in samples))}]}).encode()
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(API, data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}), timeout=90))
            out = (r["choices"][0]["message"].get("content") or "").strip().splitlines()
            nm = (out[0] if out else "").strip().strip('"').strip("'")
            if nm:
                return nm[:60]
        except Exception as e:
            sys.stderr.write(f"  retry {e}\n")
    return "[fail]"

def main():
    from cluster import embed_names
    HERE = pathlib.Path(__file__).parent
    tree = json.load(open(HERE / "tree.json"))
    child = {l["source"] for l in tree["links"]}
    leaves = [n for n in tree["nodes"] if n["depth"] > 0 and n["id"] not in child
              and n["count"] > 0 and n.get("samples")]
    leaves = leaves[:80]
    with ThreadPoolExecutor(8) as ex:
        vb = list(ex.map(lambda n: name(VERBOSE, n["samples"]), leaves))
        mn = list(ex.map(lambda n: name(MINIMAL, n["samples"]), leaves))

    golds = [n["gold_dominant"] for n in leaves]
    G = embed_names(golds); Vb = embed_names(vb); Mn = embed_names(mn)
    cos_vb = float(np.mean([Vb[i] @ G[i] for i in range(len(leaves))]))
    cos_mn = float(np.mean([Mn[i] @ G[i] for i in range(len(leaves))]))
    w_vb = float(np.mean([len(x.split()) for x in vb]))
    w_mn = float(np.mean([len(x.split()) for x in mn]))
    has_and_vb = np.mean(["and" in x.lower().split() for x in vb])
    has_and_mn = np.mean(["and" in x.lower().split() for x in mn])

    def distinct_per_gold(names):
        by = collections.defaultdict(set)
        for g, nm in zip(golds, names): by[g].add(nm.lower())
        multi = {g: len(s) for g, s in by.items() if sum(1 for x in golds if x == g) > 1}
        return round(float(np.mean(list(multi.values()))), 2) if multi else 0

    print(f"{len(leaves)} leaves named two ways.\n")
    print(f"{'metric':32}{'verbose':>10}{'minimal':>10}")
    print("-" * 52)
    print(f"{'avg words per name':32}{w_vb:10.2f}{w_mn:10.2f}")
    print(f"{'names containing and':32}{100*has_and_vb:9.0f}%{100*has_and_mn:9.0f}%")
    print(f"{'cosine name<->gold canonical':32}{cos_vb:10.3f}{cos_mn:10.3f}")
    print(f"{'distinct names per repeated gold':32}{distinct_per_gold(vb):10.2f}{distinct_per_gold(mn):10.2f}")
    print("\n--- side by side (gold | verbose | minimal) ---")
    for i in range(min(18, len(leaves))):
        print(f"  {golds[i]:24} | {vb[i]:34} | {mn[i]}")
    print(f"\nGOAL: minimal has fewer words AND >= cosine-to-gold AND fewer distinct names.")
    json.dump({"cos_vb": round(cos_vb,3), "cos_mn": round(cos_mn,3),
               "w_vb": round(w_vb,2), "w_mn": round(w_mn,2)},
              open(HERE / "naming_conciseness.json", "w"), indent=1)

if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set")
    main()
