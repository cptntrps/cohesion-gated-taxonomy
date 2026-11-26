#!/usr/bin/env python3
"""WORK-2283: persist the frozen answer key in ngc.gold_dataset / ngc.gold_item (gold-dataset gate 7).

  source ~/src/infrastructure/lib/pg.sh && pg_load_dsn && python3 load_gold.py <study_dir>

dev maps to split 'calib' (the table allows calib and test). Idempotent: an existing dataset with the same
source_sha256 is left as it is; a different sha256 under the same id is refused (a changed key is a new version).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import Json, execute_values

DATASET = "catalog-retrieval-work2283-v3"


def main(study):
    d = Path(study)
    key = d / "tasks.jsonl"
    sha = hashlib.sha256(key.read_bytes()).hexdigest()
    tasks = [json.loads(l) for l in open(key)]
    conn = psycopg2.connect(os.environ["PG_DSN"])
    with conn, conn.cursor() as cur:
        cur.execute("SELECT source_sha256 FROM ngc.gold_dataset WHERE dataset_id = %s", (DATASET,))
        row = cur.fetchone()
        if row and row[0] != sha:
            sys.exit(f"{DATASET} exists with sha256 {row[0]}, the key is {sha}: a changed key is a new dataset id")
        if row:
            print(f"{DATASET} already loaded ({sha})")
            return
        cur.execute("""INSERT INTO ngc.gold_dataset (dataset_id, source_url, license, source_sha256, label_kind,
                       description, fetched_at, recorded_at) VALUES (%s, %s, %s, %s, %s, %s, now(), now())""",
                    (DATASET, f"file://{key}", "private owner data (not for publication)", sha,
                     "rule-based real outcome: the skill an agent invoked, or the owner decisions a brief cited",
                     "WORK-2283 catalog retrieval key v3: agent task texts (redacted, gold names masked) with the "
                     "catalog items an agent should read; catalog sha256 "
                     + hashlib.sha256((d / "catalog.jsonl").read_bytes()).hexdigest()
                     + "; caveats in cohesion-gated-taxonomy experiments/catalog_retrieval/DATASET.yaml"))
        execute_values(cur, """INSERT INTO ngc.gold_item (dataset_id, item_id, split, context_id, input, gold, label)
                               VALUES %s""",
                       [(DATASET, t["tid"], "calib" if t["split"] == "dev" else "test", None,
                         Json({"text": t["text"], "session": t["session"], "source": t["source"], "kind": t["kind"],
                               "at": t["at"]}),
                         Json({"gold": t["gold"], "gold_first": t["gold_first"]}), t["gold_first"]) for t in tasks])
        cur.execute("SELECT split, count(*) FROM ngc.gold_item WHERE dataset_id = %s GROUP BY 1 ORDER BY 1", (DATASET,))
        print(DATASET, sha, cur.fetchall())


if __name__ == "__main__":
    main(sys.argv[1])
