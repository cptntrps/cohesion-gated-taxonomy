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


def test_mobius_add_identities():
    x = np.array([0.3, 0.4])
    assert np.allclose(R.mobius_add(np.zeros(2), x)[0], x)
    assert np.allclose(R.mobius_add(x, -x)[0], 0, atol=1e-12)
    a, b = np.array([0.5, 0.0]), np.array([-0.2, 0.3])
    assert math.isclose(2 * math.atanh(np.linalg.norm(R.mobius_add(-a, b)[0])), float(R.poincare(a, b)), rel_tol=1e-9)


def test_ball_children_at_tau_inside_float32_ball():
    V, _, labels, _ = synthetic_corpus(n_groups=4, per_group=10)
    t = _tree(V, labels)
    idx = R.HypIndex(t, V, dim=8, ball_dim=8, tau=2.5, beam=1)
    for v in range(1, len(t)):
        assert math.isclose(float(R.poincare(idx.pos[t.parent[v]], idx.pos[v])), 2.5, rel_tol=1e-6)
    r = np.linalg.norm(idx.pos, axis=1)
    assert r.max() < 1 - 1e-7          # representable in pg_hyperbolic's float32 vectors
    assert R.order_agreement(t, idx.pos) > 0.95


def test_hyp_index_lands_in_the_query_cluster_and_distance_decides_the_ranks():
    V, _, labels, _ = synthetic_corpus(n_groups=3, per_group=20)
    t = _tree(V, labels)
    idx = R.HypIndex(t, V, dim=8, ball_dim=8, tau=2.5, beam=1)
    q = V[labels == 2].mean(0)
    rank, land = idx.rank(q)
    assert t.label[land[0]] == "decision/g2"
    assert set(labels[rank[:20]]) == {2}
    d = R.poincare(idx.query_point(idx.reduce(q), land[0])[None, :], idx.pos[idx.item_nodes])
    assert list(rank) == list(R.rank_by(-d))          # the Poincare distance alone orders every item
    rank_h, _ = idx.rank(q, metric="hops")
    assert set(labels[rank_h[:20]]) == {2}


def test_distance_matches_pg_hyperbolic():
    """Cross-check with the pg_hyperbolic extension (skips without PG_DSN)."""
    import os
    import pytest
    if not os.environ.get("PG_DSN"):
        pytest.skip("no PG_DSN: pg_hyperbolic cross-check needs the database")
    psycopg2 = pytest.importorskip("psycopg2")
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(20, 8)).astype(np.float32)
    pts *= (rng.uniform(0.1, 0.99, 20) / np.linalg.norm(pts, axis=1))[:, None].astype(np.float32)
    with psycopg2.connect(os.environ["PG_DSN"]) as conn, conn.cursor() as cur:
        for a, b in zip(pts[:10], pts[10:]):
            va, vb = "[" + ",".join(map(str, a)) + "]", "[" + ",".join(map(str, b)) + "]"
            cur.execute("SELECT poincare_distance(%s::vector, %s::vector), hyp_mobius_add(%s::vector, %s::vector)::text",
                        (va, vb, va, vb))
            dist, madd = cur.fetchone()
            assert math.isclose(dist, float(R.poincare(a, b)), rel_tol=1e-5)
            assert np.allclose(np.array(eval(madd)), R.mobius_add(a, b)[0], atol=1e-5)


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


def test_fragment_masking_covers_slug_forms_and_dates():
    subj = "governance/decision-bar-20261019"
    frags = build_key.subject_fragments(subj)
    assert {"decision-bar-20261019", "decision-bar", "decision_bar", "decision bar"} <= frags
    one = build_key.subject_fragments("home/retire-20261120")
    assert "retire-20261120" in one and "retire" not in one      # the bare one-word slug is not masked
    mask = build_key.fragment_masker([subj, "money/medallion-x-20261106"])
    out = mask("Per the decision bar (decision_bar) on 20261019, see decision-bar-20261019.")
    assert "decision bar" not in out.lower() and "decision_bar" not in out and "20261019" not in out
    assert build_key.has_gold_fragment("follow the decision bar", ["decision:" + subj])
    assert not build_key.has_gold_fragment("unrelated text", ["decision:" + subj])


def test_key_drops_jaccard_near_duplicates():
    base = "repair the failed job and prove the repair with evidence from the log tail and the unit status"
    tasks = [{"tid": "a", "source": "s", "session": "s1", "at": "1", "text": base, "gold": ["decision:x"]},
             {"tid": "b", "source": "s", "session": "s2", "at": "2", "text": base + " today", "gold": ["decision:y"]}]
    out, stats = build_key.build(tasks, {"decision:x", "decision:y"})
    assert len(out) == 1 and stats["near_duplicate_jaccard"] == 1


def test_llm_parse_maps_numbers_and_drops_out_of_range():
    import llm_arm
    assert llm_arm.parse('{"ids": [3, "7", 3, 999, "#2"]}', 10) == [3, 7, 2]
    assert llm_arm.parse("I pick 4 then 1", 10) == [4, 1]
    assert llm_arm.parse("no answer", 10) == []
