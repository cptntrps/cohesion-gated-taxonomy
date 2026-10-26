"""Where the product keeps data that is not source.

Nothing under the repository is runtime state. Embeddings, the built tree and the
extraction result live in the data directory:

  COHESION_TAXONOMY_DATA  (default: <repo>/data, which .gitignore excludes)
    ledgar_emb.npz           corpus embeddings (nomic-embed-text, 60k LEDGAR clauses)
    tree.json, members.json  the current taxonomy (rebuilt on every human correction)
    extraction.json          the extraction demo result
    extraction_validate.json the measured read-everything baseline for the demo
    experiments/             embedding caches the research scripts use
    var/                     logs and before/after snapshots

LEDGAR_EMB overrides the embedding file alone (kept for the research scripts).
"""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
WEB = REPO / "web"
DATA = pathlib.Path(os.environ.get("COHESION_TAXONOMY_DATA", str(REPO / "data")))
EXPERIMENT_CACHE = DATA / "experiments"
STATE_FILES = ("tree.json", "members.json", "extraction.json")


def ledgar_emb() -> pathlib.Path:
    return pathlib.Path(os.environ.get("LEDGAR_EMB", str(DATA / "ledgar_emb.npz")))
