#!/usr/bin/env python3
"""Interactive taxonomy server.

Holds the LEDGAR embeddings + texts in memory. Serves the page and re-runs the
WHOLE anchored clustering on demand. The only durable human input is the anchor
name list; geometry + the LLM rebuild everything else.

Endpoints (POST, JSON):
  /rename  {id, name}  -> promote/replace an anchor name, rebuild, return tree
  /anchor  {name}      -> add an empty named anchor, rebuild, return tree
  /unanchor{name}      -> drop an anchor, rebuild, return tree
  /rerun   {}          -> rebuild with current anchors, return tree
  /reset   {}          -> clear anchors, rebuild discovery, return tree
"""
import json, os, sys, threading, pathlib, urllib.request, urllib.parse
import numpy as np
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

HERE = pathlib.Path(__file__).parent
EMB = pathlib.Path(os.environ.get("LEDGAR_EMB", "ledgar_emb.npz"))
LOCK = threading.Lock()
STATE = {}   # V, texts, labels, gold_names, anchors

def load_corpus():
    from datasets import load_dataset
    ds = load_dataset("MAdAiLab/lex_glue_ledgar", split="train")
    texts = list(ds["text"])
    gold_names = ds.features["label"].names
    labels = np.array(ds["label"])
    V = np.load(EMB)["V"].astype(np.float32)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    n = min(len(texts), len(V))
    STATE.update(V=V[:n], texts=texts[:n], labels=labels[:n], gold_names=gold_names)
    tree = json.load(open(HERE / "tree.json"))
    STATE["anchors"] = tree.get("anchors", [])
    STATE["prev_anchors"] = list(STATE["anchors"])
    STATE["lower"] = [t.lower() for t in STATE["texts"]]
    load_leaves(tree)
    print(f"loaded {n} provisions, {len(STATE['anchors'])} anchors", flush=True)

def load_leaves(tree):
    """idx -> name of the leaf that holds the clause (members.json is written by every rebuild)."""
    load_leaves_from(json.load(open(HERE / "members.json")), tree)

def load_leaves_from(members, tree):
    names = {x["id"]: x["name"] for x in tree["nodes"]}
    has_child = {l["source"] for l in tree["links"]}
    leaf_of = np.full(len(STATE["texts"]), -1, dtype=np.int64)
    leaf_names = []
    for nid, idxs in members.items():
        if nid in has_child or nid not in names:
            continue
        leaf_names.append(names[nid])
        li = len(leaf_names) - 1
        arr = np.asarray(idxs, dtype=np.int64)
        arr = arr[arr < len(leaf_of)]
        leaf_of[arr] = li
    STATE["leaf_of"], STATE["leaf_names"] = leaf_of, leaf_names

def embed_query(text):
    req = urllib.request.Request("http://localhost:11434/api/embed", data=json.dumps(
        {"model": "nomic-embed-text", "input": text.lower()}).encode(), headers={"Content-Type": "application/json"})
    a = np.array(json.load(urllib.request.urlopen(req, timeout=60))["embeddings"][0], dtype=np.float32)
    return a / (np.linalg.norm(a) + 1e-9)

def gold_match(q):
    """The benchmark label that names this concept, if one does (used only to score recall; never to rank)."""
    ql = q.lower().strip()
    names = STATE["gold_names"]
    for i, g in enumerate(names):
        if g.lower() == ql or g.lower() == ql + "s" or g.lower().rstrip("s") == ql.rstrip("s"):
            return i
    for i, g in enumerate(names):
        if ql in g.lower() or g.lower() in ql:
            return i
    return None

def score(q, top=25, recall=0.90):
    """Find a concept three ways: semantic ranking (the product), keyword, read everything."""
    V, lower, labels = STATE["V"], STATE["lower"], STATE["labels"]
    N = len(lower)
    sims = V @ embed_query(q)
    order = np.argsort(-sims)
    kw = q.lower().strip()
    km = np.fromiter((kw in t for t in lower), dtype=bool, count=N)
    out = {"query": q, "n": N, "keyword_matches": int(km.sum()), "recall_target": recall}
    gi = gold_match(q)
    if gi is not None:
        gold = labels == gi
        G = int(gold.sum()); need = int(np.ceil(recall * G))
        hit = np.cumsum(gold[order]); k90 = int(np.searchsorted(hit, need) + 1) if need and hit[-1] >= need else None
        kw_hits = int((km & gold).sum()); kw_rec = round(100 * kw_hits / max(G, 1), 1)
        kw_to_90 = int(km.sum()) if kw_rec >= 100 * recall else N
        out.update(gold_label=STATE["gold_names"][gi], gold_count=G, semantic_to_90=k90, semantic_found_at_90=need, keyword_hits=kw_hits,
                   semantic_precision_at_90=round(100 * need / k90, 1) if k90 else None,
                   keyword_recall=kw_rec, keyword_precision=round(100 * kw_hits / max(int(km.sum()), 1), 1),
                   keyword_to_90=kw_to_90, times_fewer=round(kw_to_90 / k90, 1) if k90 else None)
    leaf_of, leaf_names = STATE["leaf_of"], STATE["leaf_names"]
    rows = []
    for i in order[:top]:
        li = int(leaf_of[i]); rows.append({"idx": int(i), "score": round(float(sims[i]), 3),
            "leaf": leaf_names[li] if li >= 0 else None, "text": STATE["texts"][i][:320],
            "keyword_hit": bool(km[i]), "gold": STATE["gold_names"][int(labels[i])]})
    counts = {}
    for i in order[:200]:
        li = int(leaf_of[i]); nm = leaf_names[li] if li >= 0 else "(none)"; counts[nm] = counts.get(nm, 0) + 1
    out["results"] = rows
    out["leaves_in_top200"] = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
    return out

def rebuild():
    from cluster import build_taxonomy
    anchors = STATE["anchors"]
    old_of, old_names = STATE.get("leaf_of"), list(STATE.get("leaf_names", []))
    old_anchors = list(STATE.get("prev_anchors", []))
    nodes, links, members, report, anomalies = build_taxonomy(
        STATE["V"], STATE["texts"], STATE["labels"], STATE["gold_names"],
        anchors=anchors, log=lambda m: print(m, flush=True))
    tree = {"nodes": nodes, "links": links, "report": report,
            "anchors": anchors, "anomalies": anomalies}
    load_leaves_from(members, tree)
    tree["change"] = change_summary(old_of, old_names, old_anchors, anchors, nodes, members)
    STATE["prev_anchors"] = list(anchors)
    json.dump(tree, open(HERE / "tree.json", "w"), indent=1)
    json.dump(members, open(HERE / "members.json", "w"))
    return tree

def change_summary(old_of, old_names, old_anchors, anchors, nodes, members, top=5):
    """What one correction moved: for a new concept, the old groups its clauses came from; for a removed one, where they went."""
    if old_of is None:
        return None
    def named(of, names, idx):
        c = {}
        for i in idx:
            li = int(of[i]); nm = names[li] if li >= 0 else "(no group)"; c[nm] = c.get(nm, 0) + 1
        return sorted(c.items(), key=lambda kv: -kv[1])[:top]
    added = [a for a in anchors if a not in old_anchors]; removed = [a for a in old_anchors if a not in anchors]
    if added:
        name = added[-1]; node = next((n for n in nodes if n["name"] == name and n.get("human_named")), None)
        if node is None:
            return {"kind": "added", "concept": name, "clauses": 0, "from": []}
        idx = np.asarray(members[node["id"]], dtype=np.int64)
        return {"kind": "added", "concept": name, "clauses": int(len(idx)), "from": named(old_of, old_names, idx)}
    if removed and removed[-1] in old_names:
        name = removed[-1]; idx = np.nonzero(old_of == old_names.index(name))[0]
        return {"kind": "removed", "concept": name, "clauses": int(len(idx)), "to": named(STATE["leaf_of"], STATE["leaf_names"], idx)}
    return None

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/score"):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            q = (qs.get("q") or [""])[0].strip()
            if not q:
                return self._json({"error": "empty query"}, 400)
            try:
                return self._json(score(q, top=int((qs.get("top") or ["25"])[0])))
            except Exception as e:
                import traceback; traceback.print_exc()
                return self._json({"error": str(e)}, 500)
        return super().do_GET()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json({"error": f"bad json: {e}"}, 400)
        route = self.path.rstrip("/")
        with LOCK:
            anchors = STATE["anchors"]
            try:
                if route == "/rename":
                    nid, name = req.get("id"), (req.get("name") or "").strip()
                    if not name:
                        return self._json({"error": "empty name"}, 400)
                    tree = json.load(open(HERE / "tree.json"))
                    old = next((x["name"] for x in tree["nodes"] if x["id"] == nid), None)
                    if old in anchors:
                        anchors[anchors.index(old)] = name
                    elif name not in anchors:
                        anchors.append(name)
                elif route == "/anchor":
                    name = (req.get("name") or "").strip()
                    if not name:
                        return self._json({"error": "empty name"}, 400)
                    if name not in anchors:
                        anchors.append(name)
                elif route == "/unanchor":
                    name = (req.get("name") or "").strip()
                    if name in anchors:
                        anchors.remove(name)
                elif route == "/reset":
                    anchors.clear()
                elif route == "/rerun":
                    pass
                else:
                    return self._json({"error": "unknown route"}, 404)
                tree = rebuild()
                return self._json(tree)
            except Exception as e:
                import traceback; traceback.print_exc()
                return self._json({"error": str(e)}, 500)

    def log_message(self, *a):
        pass

def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set")
    if not (HERE / "tree.json").exists():
        sys.exit("run build_tree.py first (need tree.json)")
    load_corpus()
    port = int(os.environ.get("PORT", "8799"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"serving http://127.0.0.1:{port}  (Ctrl-C to stop)", flush=True)
    srv.serve_forever()

if __name__ == "__main__":
    main()
