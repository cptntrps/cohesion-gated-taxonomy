# LEDGAR self-evolving clause taxonomy (MVP)

Geometry finds the clusters, the LLM only names them, the human anchors and re-runs.

- **Category-blind clustering** of 60k LEDGAR clauses on clause language alone
  (nomic-embed-text, 768d). No contract type, no gold labels feed the clustering.
- **Hyperbolic cluster-finding** — Poincare-ball k-means (`hyperbolic.py`),
  top-to-bottom, 3 levels. `geometry="flat"` switches to Euclidean KMeans.
- **LLM names only** — DeepSeek `deepseek-v4-flash` names each discovered cluster
  from representative clauses, with the ancestor chain as context (`naming.py`).
- **Human anchors** — rename a cluster or add a named one; the whole corpus
  re-clusters around your anchors (nearest-anchor at cosine >= 0.60, then discover
  the rest). An empty anchor attracts its clauses on the next run.
- **Gold labels score purity POST-HOC only** — never shown to the clusterer or LLM.

## Run
```
export DEEPSEEK_API_KEY=...              # from infrastructure-secrets/services/livingos-crons.enc.env
python3 build_tree.py                     # initial discovery tree -> tree.json + members.json
PORT=8799 python3 server.py               # interactive: serves the UI + rename/anchor/rerun
# open http://127.0.0.1:8799
```

## Measured (60k LEDGAR, discovery mode)
- baseline (majority class) 5.3% · **hyperbolic leaf purity 41.6%** · flat 50.6% · PASS (>= 2x baseline).
- Honest: unsupervised hyperbolic does NOT beat flat here. It needs the trained
  dense-graph regime; this MVP is the flat/anchored product with hyperbolic wired
  in as the swappable cluster-finder.

Files: `hyperbolic.py` geometry · `cluster.py` shared builder · `naming.py` LLM ·
`build_tree.py` batch build · `server.py` interactive · `index.html` React + 3d-force-graph UI.
