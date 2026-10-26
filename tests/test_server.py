import importlib
import json
import threading
import urllib.request

import numpy as np
import pytest

from cohesion_taxonomy import cluster, paths


@pytest.fixture
def live_server(corpus, monkeypatch, tmp_path):
    monkeypatch.setenv("COHESION_TAXONOMY_DATA", str(tmp_path))
    importlib.reload(paths)
    from cohesion_taxonomy import server
    importlib.reload(server)
    V, texts, labels, gold = corpus
    nodes, links, members, report, anomalies = cluster.build_taxonomy(
        V, texts, labels, gold, anchors=None, geometry="flat", do_naming=False, log=lambda m: None)
    tree = {"nodes": nodes, "links": links, "report": report, "anchors": [], "anomalies": anomalies}
    json.dump(tree, open(tmp_path / "tree.json", "w"))
    json.dump(members, open(tmp_path / "members.json", "w"))
    server.STATE.update(V=V, texts=texts, labels=labels, gold_names=gold, anchors=[], prev_anchors=[],
                        lower=[t.lower() for t in texts])
    server.load_leaves_from(members, tree)
    centroid = V[labels == 1].mean(0); centroid /= np.linalg.norm(centroid)
    monkeypatch.setattr(cluster, "embed_names", lambda names: np.vstack([centroid] * len(names)).astype(np.float32))
    monkeypatch.setattr(server, "embed_query", lambda text: centroid.astype(np.float32))
    srv = server.serve("127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base, server, len(texts)
    srv.shutdown()
    monkeypatch.delenv("COHESION_TAXONOMY_DATA")
    importlib.reload(paths)


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_healthz_asserts_what_the_server_produced(live_server):
    base, server, n = live_server
    status, h = _get(base + "/healthz")
    assert status == 200 and h["ok"] is True
    assert h["provisions"] == n and h["clauses_in_a_leaf"] == n and h["leaves"] > 0
    assert h["naming"] is False and h["version"] == server.__version__


def test_state_files_come_from_the_data_dir_and_pages_from_web(live_server):
    base, _, _ = live_server
    status, tree = _get(base + "/tree.json?_=1")
    assert status == 200 and tree["nodes"][0]["id"] == "root"
    with urllib.request.urlopen(base + "/", timeout=30) as r:
        assert r.status == 200 and b"<html" in r.read(2000).lower()
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base + "/extraction.json", timeout=30)
    assert e.value.code == 404                                   # not built in this data dir


def test_score_ranks_the_queried_group_first(live_server):
    base, _, _ = live_server
    status, out = _get(base + "/api/score?q=gold_1&top=10")
    assert status == 200 and out["gold_label"] == "gold_1"
    assert out["semantic_to_90"] is not None and out["semantic_to_90"] <= out["gold_count"] + 5
    assert all(row["gold"] == "gold_1" for row in out["results"][:5])
    status, err = _get(base + "/api/score?q=")
    assert status == 400 and "error" in err


def test_correction_routes_rebuild_and_persist(live_server):
    base, server, n = live_server
    status, tree = _post(base + "/anchor", {"name": "Group one"})
    assert status == 200
    anchor = next(nd for nd in tree["nodes"] if nd["human_named"])
    assert anchor["name"] == "Group one" and tree["anchors"] == ["Group one"]
    assert tree["change"]["kind"] == "added" and tree["change"]["clauses"] > 0
    persisted = json.load(open(paths.DATA / "tree.json"))
    assert persisted["anchors"] == ["Group one"]
    status, tree = _post(base + "/unanchor", {"name": "Group one"})
    assert status == 200 and tree["anchors"] == [] and tree["change"]["kind"] == "removed"
    status, tree = _post(base + "/reset", {})
    assert status == 200 and tree["anchors"] == []
    assert _post(base + "/anchor", {"name": "  "})[0] == 400
    assert _post(base + "/nope", {})[0] == 404
    status, h = _get(base + "/healthz")
    assert h["ok"] is True and h["clauses_in_a_leaf"] == n
