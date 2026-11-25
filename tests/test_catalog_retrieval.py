"""Hermetic tests for the WORK-2283 catalog retrieval study (experiments/catalog_retrieval)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "catalog_retrieval"))

import build_key  # noqa: E402
import harvest  # noqa: E402
import retrieval as R  # noqa: E402
from conftest import synthetic_corpus  # noqa: E402


def _tree(V, labels):
    kinds = ["decision"] * len(V)
    groups = [f"g{l}" for l in labels]
    return R.build_tree(kinds, groups, V, max_group=1000)


def test_bm25_prefers_the_matching_document():
    docs = ["leak-triage: rotate a leaked secret", "tutor: explain a topic", "nook: e-reader sync"]
    bm = R.BM25(docs)
    assert int(R.rank_by(bm.scores("a secret leaked in the log, rotate it"))[0]) == 0
    assert bm.scores("zzz unknown words").sum() == 0


def test_rank_by_breaks_ties_then_index():
    assert list(R.rank_by([1, 1, 0], [0.1, 0.9, 5])) == [1, 0, 2]
    assert list(R.rank_by([0, 0, 0])) == [0, 1, 2]


def test_sarkar_children_at_tau_and_root_at_origin():
    V, _, labels, _ = synthetic_corpus(n_groups=3, per_group=8)
    t = _tree(V, labels)
    pos = R.sarkar(t, tau=1.5)
    assert abs(pos[0]) == 0
    for v in range(1, len(t)):
        assert math.isclose(float(R.poincare(pos[t.parent[v]], pos[v])), 1.5, rel_tol=1e-6)
    assert np.all(np.abs(pos) < 1)


def test_sarkar_preserves_tree_distance_order():
    V, _, labels, _ = synthetic_corpus(n_groups=4, per_group=10)
    t = _tree(V, labels)
    low, high = R.sarkar(t, tau=1.5), R.sarkar(t, tau=R.sala_tau(t))
    assert R.order_agreement(t, high) == 1.0
    assert R.order_agreement(t, low) < 1.0       # too small a step lets cousins come closer than siblings
    rho, n = R.dist_fidelity(t, high, n_pairs=800)
    assert n > 500 and rho > 0.7


def test_hyp_index_lands_in_the_query_cluster_and_ranks_it_first():
    V, _, labels, _ = synthetic_corpus(n_groups=3, per_group=20)
    t = _tree(V, labels)
    idx = R.HypIndex(t, V, dim=8, tau=R.sala_tau(t), beam=1)
    q = V[labels == 2].mean(0)
    rank, land = idx.rank(q)
    assert set(labels[rank[:20]]) == {2}
    rank_h, _ = idx.rank(q, metric="hops")
    assert set(labels[rank_h[:20]]) == {2}
    assert t.label[land[0]] == "decision/g2"


def test_metrics():
    ranking = np.array([5, 3, 9, 1])
    assert R.first_rank(ranking, [9, 1]) == 3
    assert R.recall_at(ranking, [9, 1], 3) == 0.5


def test_masking_removes_gold_names():
    t = harvest.mask_skill("Use the leak-triage skill, or /leak-triage, or leak triage.", "leak-triage")
    assert "leak" not in t.lower()
    d = harvest.mask_decisions("Authority: owner_decision:security/x-20261125 and security/x-20261125.",
                               {"security/x-20261125"})
    assert "security/x" not in d and d.count("[decision]") == 2


def test_key_dedupes_templates_and_splits_by_session():
    a = build_key.norm_key("Repair job /home/gui/data/attempts/003932-x/evidence.json now 12")
    b = build_key.norm_key("Repair job /home/gui/data/attempts/001797-y/evidence.json now 7")
    assert a == b
    assert build_key.split_of("s1") == build_key.split_of("s1")
    tasks = [{"tid": str(i), "source": "s", "session": f"s{i}", "at": str(i), "text": f"text number {w}",
              "gold": ["decision:a"]} for i, w in enumerate(["one", "two", "three", "four"])]
    out, stats = build_key.build(tasks, {"decision:a"})
    assert len(out) == build_key.CAP and stats["over_cap"] == 1
