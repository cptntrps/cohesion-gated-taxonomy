import numpy as np

from cohesion_taxonomy import cluster


def test_cohesion_bounds_and_order():
    tight = np.tile(np.array([1.0, 0.0, 0.0]), (10, 1))
    assert cluster.cohesion(tight) == 1.0
    loose = np.eye(3)
    assert 0.0 < cluster.cohesion(loose) < cluster.cohesion(tight)
    assert cluster.cohesion(np.zeros((1, 3))) == 1.0            # a singleton is trivially cohesive
    assert cluster.cohesion(np.array([[1.0, 0], [-1.0, 0]])) == 0.0   # cancelling centroid


def test_multimodality_separates_seam_from_blob(corpus):
    V, _, labels, _ = corpus
    two = V[(labels == 0) | (labels == 1)]
    one = V[labels == 0]
    assert cluster.multimodality(two) > cluster.multimodality(one)
    assert cluster.multimodality(V[:10]) == 0.0                 # too small to judge


def test_refine_partition_never_lowers_cohesion(corpus):
    V, _, labels, _ = corpus
    rng = np.random.default_rng(1)
    noisy = labels.copy()
    flip = rng.choice(len(noisy), 60, replace=False)
    noisy[flip] = (noisy[flip] + 1) % 3
    before = np.mean([cluster.cohesion(V[noisy == c]) for c in range(3)])
    lab = cluster.refine_partition(V, noisy)
    after = np.mean([cluster.cohesion(V[lab == c]) for c in range(int(lab.max()) + 1)])
    assert after >= before
    assert set(lab.tolist()) == set(range(int(lab.max()) + 1))   # contiguous ids


def _check_tree(nodes, links, members, report, n):
    ids = {nd["id"] for nd in nodes}
    assert "root" in ids and len(ids) == len(nodes)
    assert all(l["source"] in ids and l["target"] in ids for l in links)
    child_ids = {l["source"] for l in links}
    leaves = [nd for nd in nodes if nd["depth"] > 0 and nd["id"] not in child_ids and nd["count"] > 0]
    covered = sorted(i for nd in leaves for i in members[nd["id"]])
    assert covered == list(range(n))                               # leaves partition the corpus
    assert report["n_leaves"] == len(leaves)
    assert report["leaf_weighted_purity_pct"] >= report["baseline_majority_pct"]
    assert all(nd["cohesion"] is not None for nd in leaves)


def test_build_taxonomy_flat_without_naming(corpus):
    V, texts, labels, gold = corpus
    nodes, links, members, report, anomalies = cluster.build_taxonomy(
        V, texts, labels, gold, anchors=None, geometry="flat", do_naming=False, log=lambda m: None)
    _check_tree(nodes, links, members, report, len(texts))
    assert anomalies == []
    assert all(nd["name"] == "..." for nd in nodes if nd["id"] != "root")   # nothing named


def test_build_taxonomy_hyperbolic_names_fall_back_without_key(corpus):
    V, texts, labels, gold = corpus
    nodes, links, members, report, anomalies = cluster.build_taxonomy(
        V, texts, labels, gold, anchors=None, geometry="hyperbolic", do_naming=True, log=lambda m: None)
    _check_tree(nodes, links, members, report, len(texts))
    assert report["naming_failures"] == report["n_nodes"]          # no key: every name is a fallback
    assert all(nd["name"] == "[no api key]" for nd in nodes if nd["id"] != "root")
    assert 0 < len(anomalies) <= 40
    assert {"idx", "misfit", "radius", "leaf_id"} <= set(anomalies[0])


def test_anchors_claim_their_group(corpus, monkeypatch):
    V, texts, labels, gold = corpus
    centroid = V[labels == 2].mean(0)
    centroid /= np.linalg.norm(centroid)
    monkeypatch.setattr(cluster, "embed_names", lambda names: np.vstack([centroid] * len(names)).astype(np.float32))
    nodes, links, members, report, _ = cluster.build_taxonomy(
        V, texts, labels, gold, anchors=["Group two"], geometry="flat", do_naming=False, log=lambda m: None)
    anchor = next(nd for nd in nodes if nd["human_named"])
    assert anchor["name"] == "Group two" and anchor["depth"] == 1
    claimed = np.array(members[anchor["id"]])
    assert len(claimed) > 0 and (labels[claimed] == 2).all()        # the anchor pulls only its group
    assert report["n_anchors"] == 1
