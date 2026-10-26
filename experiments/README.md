# Experiments

One committed script per experiment; the commit message states its finding, including
the negative ones. Result files sit beside the script that wrote them. `docs/RESULTS.md`
is the index; `docs/METHODS.md` says how to reproduce each number.

Run from the repo with the package installed (`.venv/bin/pip install -e .[research]`):

    .venv/bin/python experiments/purity_probe.py

Inputs the scripts read through `cohesion_taxonomy.paths`:

| file | where |
|---|---|
| `ledgar_emb.npz`, `tree.json`, `members.json`, `extraction*.json` | the data directory (`COHESION_TAXONOMY_DATA`) |
| `amz_emb.npz`, `ng_emb.npz`, `amz_big_emb.npz` | `<data dir>/experiments/` (built by the scripts that need them) |
| `*_results.json`, `v2_*`, `v3_*`, `validity_*`, `w*_*`, `h3*` | this directory |

A script that calls a model reads the same `NAMING_API_*` / `DEEPSEEK_API_KEY` variables as
the product; the `anchor_*` and `v2`/`v3` judge scripts also need the keys their
pre-registration names.
