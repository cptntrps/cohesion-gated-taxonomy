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
import json, os, sys, threading, pathlib
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
    print(f"loaded {n} provisions, {len(STATE['anchors'])} anchors", flush=True)

def rebuild():
    from cluster import build_taxonomy
    anchors = STATE["anchors"]
    nodes, links, members, report, anomalies = build_taxonomy(
        STATE["V"], STATE["texts"], STATE["labels"], STATE["gold_names"],
        anchors=anchors, log=lambda m: print(m, flush=True))
    tree = {"nodes": nodes, "links": links, "report": report,
            "anchors": anchors, "anomalies": anomalies}
    json.dump(tree, open(HERE / "tree.json", "w"), indent=1)
    json.dump(members, open(HERE / "members.json", "w"))
    return tree

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
