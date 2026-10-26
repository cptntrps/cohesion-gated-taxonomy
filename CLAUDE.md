# Cohesion-Gated Taxonomy (product)

Read `README.md` first: it is the intent brief and holds the measured claims and the
negative results. `docs/ARCHITECTURE.md` explains the layout, `docs/METHODS.md` how to
reproduce every number, `docs/RESULTS.md` the full evidence log.

Rules for agents working here:
- `cohesion_taxonomy/` is the product. `experiments/` holds one committed experiment per
  script; never edit a script to change a published number, add a new script and a new
  result file.
- Runtime state is never in the repo: `cohesion_taxonomy/paths.py` resolves the data
  directory (`COHESION_TAXONOMY_DATA`, on aurora `~/data/cohesion-gated-taxonomy`).
- The suite (`python3 -m pytest -q`) is hermetic: no network, no model, no corpus. It is
  the pre-push gate and CI. A test that needs a live service self-skips.
- Deploy on aurora is `./deploy.sh` (restart the user unit, prove `/healthz` ok). The unit
  is `cohesion-gated-taxonomy.service` on loopback :8580; its secret store is
  `infrastructure-secrets/services/cohesion-gated-taxonomy.enc.env`.
- Steward: double_gui (`registry.product` slug `cohesion-gated-taxonomy`).
