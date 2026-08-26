#!/usr/bin/env python3
"""When the LLM proposes 'A and B', try A, B, and 'A and B' — pick whichever name
best matches the cluster's content (cosine of the name embedding to the cluster
centroid, gold-free). Test: does this beat forcing a single word, on the two-topic
clusters, measured by closeness to the canonical gold term?
"""
import os, sys, json, pathlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import urllib.request

API = "https://api.deepseek.com/v1/chat/completions"; MODEL = "deepseek-v4-flash"
EMB = pathlib.Path(os.environ.get("LEDGAR_EMB", "ledgar_emb.npz"))
# allow a compound when genuinely two topics; we resolve it afterwards by content
PROMPT = ("Below are contract clauses grouped by similar language. Give the SHORTEST "
          "canonical clause-library name (1-3 words). If the group genuinely covers TWO "
          "distinct concepts, you may answer 'A and B'; otherwise give the single term.\n\n"
          "{clauses}\n\nReply with ONLY the name.")

def propose(samples):
    key = os.environ["DEEPSEEK_API_KEY"]
    body = json.dumps({"model": MODEL, "temperature": 0.0, "max_tokens": 800, "stream": False,
        "messages": [{"role": "user", "content": PROMPT.format(
            clauses="\n\n".join("- " + s[:300] for s in samples))}]}).encode()
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(API, data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}), timeout=90))
            o = (r["choices"][0]["message"].get("content") or "").strip().splitlines()
            nm = (o[0] if o else "").strip().strip('"').strip("'")
            if nm: return nm[:60]
        except Exception as e:
            sys.stderr.write(f" retry {e}\n")
    return "[fail]"

def main():
    from cluster import embed_names
    HERE = pathlib.Path(__file__).parent
    V = np.load(EMB)["V"].astype(np.float32); V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    tree = json.load(open(HERE / "tree.json")); members = json.load(open(HERE / "members.json"))
    child = {l["source"] for l in tree["links"]}
    leaves = [n for n in tree["nodes"] if n["depth"] > 0 and n["id"] not in child
              and n["count"] > 0 and n.get("samples")][:90]
    with ThreadPoolExecutor(8) as ex:
        names = list(ex.map(lambda n: propose(n["samples"]), leaves))

    # centroids of each leaf (nomic space)
    cents = []
    for n in leaves:
        m = np.array(members[n["id"]], dtype=int); c = V[m].mean(0); cents.append(c / (np.linalg.norm(c)+1e-9))
    cents = np.array(cents)

    rows = []
    for i, (n, nm) in enumerate(zip(leaves, names)):
        parts = [p.strip() for p in nm.replace("/", " and ").split(" and ") if p.strip()]
        cands = list(dict.fromkeys(parts + [nm])) if len(parts) > 1 else [nm]
        E = embed_names(cands)
        pick = cands[int(np.argmax(E @ cents[i]))]           # name closest to cluster content
        force1 = parts[0] if len(parts) > 1 else nm          # naive "first word" single
        rows.append((n["gold_dominant"], nm, pick, force1, len(parts) > 1))

    golds = [r[0] for r in rows]; G = embed_names(golds)
    def gcos(names):
        E = embed_names(names); return float(np.mean([E[i] @ G[i] for i in range(len(rows))]))
    comp = [i for i, r in enumerate(rows) if r[4]]
    print(f"{len(rows)} leaves, {len(comp)} proposed as two-topic ('A and B')\n")
    print(f"gold-cosine over ALL leaves:  whole={gcos([r[1] for r in rows]):.3f}  "
          f"pick-best={gcos([r[2] for r in rows]):.3f}  force-first={gcos([r[3] for r in rows]):.3f}")
    if comp:
        gc = embed_names([golds[i] for i in comp])
        def gc2(sel):
            E = embed_names([sel(rows[i]) for i in comp]); return float(np.mean([E[k]@gc[k] for k in range(len(comp))]))
        print(f"gold-cosine on TWO-TOPIC only: whole={gc2(lambda r:r[1]):.3f}  "
              f"pick-best={gc2(lambda r:r[2]):.3f}  force-first={gc2(lambda r:r[3]):.3f}")
    print("\n--- two-topic clusters: gold | proposed | pick-best ---")
    for i in comp[:16]:
        print(f"  {rows[i][0]:22} | {rows[i][1]:34} | {rows[i][2]}")

if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set")
    main()
