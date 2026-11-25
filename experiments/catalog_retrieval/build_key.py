#!/usr/bin/env python3
"""WORK-2283 answer key: redacted tasks -> deduplicated, capped, split tasks.jsonl (frozen by sha256).

Rules (fixed before any arm runs):
  - text is truncated to MAX_CHARS AFTER redaction (redact the full text first).
  - near-duplicates: one task per normalized prefix (lower case, paths, numbers and hex ids removed, 400 chars).
  - at most CAP tasks per first-gold item, in time order, so no template family dominates.
  - split by session (sha256 of seed + session): DEV_SHARE to dev, the rest to test. Tasks from one session
    stay in one split. Dev is for tuning; test is scored once.
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

SEED = "work-2283-v1"  # split seed (unchanged in key v2)
MAX_CHARS = 2000
CAP = 3
DEV_SHARE = 0.25


def norm_key(text):
    t = text.lower()
    t = re.sub(r"\S*/\S*", " ", t)
    t = re.sub(r"[0-9a-f]{6,}|\d+", " ", t)
    t = re.sub(r"\W+", " ", t).strip()
    return t[:400]


def split_of(session):
    h = int(hashlib.sha256(f"{SEED}:{session}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < DEV_SHARE else "test"


def build(tasks, catalog_ids):
    stats = collections.Counter()
    tasks = sorted(tasks, key=lambda t: (t.get("at") or "", t["tid"]))
    seen, per_gold, out = set(), collections.Counter(), []
    for t in tasks:
        gold = [g for g in t["gold"] if g in catalog_ids]
        if not gold:
            stats["gold_not_in_catalog"] += 1
            continue
        k = norm_key(t["text"])
        if k in seen:
            stats["near_duplicate"] += 1
            continue
        seen.add(k)
        if per_gold[gold[0]] >= CAP:
            stats["over_cap"] += 1
            continue
        per_gold[gold[0]] += 1
        out.append({"tid": t["tid"], "source": t["source"], "session": t["session"], "at": t.get("at", ""),
                    "split": split_of(t["session"]), "kind": gold[0].split(":")[0], "text": t["text"][:MAX_CHARS],
                    "gold": gold, "gold_first": gold[0]})
    stats["kept"] = len(out)
    return out, dict(stats)


def main(d):
    d = Path(d)
    bad = collections.Counter()
    def load(p):
        rows = []
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except ValueError:
                bad[p.name] += 1
        return rows
    cat = load(d / "catalog.redacted.jsonl")
    tasks = load(d / "tasks.redacted.jsonl")
    if bad:
        sys.exit(f"redaction broke JSON lines: {dict(bad)}; fix the redactor input, do not drop rows silently")
    key, stats = build(tasks, {c["id"] for c in cat})
    with open(d / "catalog.jsonl", "w") as f:
        for c in cat:
            f.write(json.dumps(c, sort_keys=True) + "\n")
    with open(d / "tasks.jsonl", "w") as f:
        for t in key:
            f.write(json.dumps(t, sort_keys=True) + "\n")
    by = collections.Counter((t["split"], t["kind"]) for t in key)
    stats["by_split_kind"] = {f"{a}/{b}": n for (a, b), n in sorted(by.items())}
    stats["units_by_split"] = {s: len({t["session"] for t in key if t["split"] == s}) for s in ("dev", "test")}
    for name in ("catalog.jsonl", "tasks.jsonl"):
        stats[f"sha256:{name}"] = hashlib.sha256((d / name).read_bytes()).hexdigest()
    json.dump(stats, open(d / "key_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
