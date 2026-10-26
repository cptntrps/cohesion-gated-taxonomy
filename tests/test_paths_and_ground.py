import importlib

from cohesion_taxonomy import paths
from cohesion_taxonomy.extraction_ground import ground_state


def test_data_dir_follows_env(monkeypatch, tmp_path):
    monkeypatch.setenv("COHESION_TAXONOMY_DATA", str(tmp_path))
    monkeypatch.delenv("LEDGAR_EMB", raising=False)
    importlib.reload(paths)
    assert paths.DATA == tmp_path
    assert paths.ledgar_emb() == tmp_path / "ledgar_emb.npz"
    assert paths.EXPERIMENT_CACHE == tmp_path / "experiments"
    monkeypatch.delenv("COHESION_TAXONOMY_DATA")
    importlib.reload(paths)
    assert paths.DATA == paths.REPO / "data"
    assert (paths.WEB / "index.html").exists()


def test_ground_state_reads_the_written_state():
    assert ground_state("This Agreement shall be governed by the laws of the State of Delaware.") == "Delaware"
    assert ground_state("governed by the internal laws of the Commonwealth of Massachusetts") == "Massachusetts"
    assert ground_state("The parties agree to arbitrate all disputes.") is None
