#!/usr/bin/env python3
"""WORK-2283 answer key: redacted tasks -> deduplicated, capped, split tasks.jsonl (frozen by sha256).

Rules (fixed before any arm runs):
  - text is truncated to MAX_CHARS AFTER redaction (redact the full text first).
  - masking (key v3): distinctive fragments of every catalog subject key (slug without domain, with and without
    date, hyphen/underscore/space forms) and every 8-digit date token; gold_fragment records whether the text
    held a fragment of its own gold before this masking (the sensitivity stratum).
  - near-duplicates: one task per normalized prefix (lower case, paths, numbers and hex ids removed, 400 chars),
    and no task whose token set has Jaccard >= 0.75 with a kept task.
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

SEED = "work-2283-v1"  # split seed (unchanged in keys v2 and v3)
KEY_VERSION = "v3"
JACCARD_DUP = 0.75
DATE = re.compile(r"\b20\d{6}\b")
MAX_CHARS = 2000
CAP = 3
DEV_SHARE = 0.25


def norm_key(text):
    t = text.lower()
    t = re.sub(r"\S*/\S*", " ", t)
    t = re.sub(r"[0-9a-f]{6,}|\d+", " ", t)
    t = re.sub(r"\W+", " ", t).strip()
    return t[:400]


def subject_fragments(subject):
    """Distinctive fragments of a decision subject key: the slug without its domain, with and without its
    trailing date, in hyphen, underscore and space forms; only fragments of two or more hyphen parts (so a bare
    one-word slug such as "retire" is not masked, but "retire-20261120" is)."""
    slug = subject.split("/")[-1]
    bases = {slug, re.sub(r"-?20\d{6}$", "", slug)}
    out = set()
    for b in bases:
        if len([w for w in b.split("-") if w]) < 2:
            continue
        out |= {b, b.replace("-", "_"), b.replace("-", " ")}
    return out


def fragment_masker(subjects):
    frags = sorted({f.lower() for s in subjects for f in subject_fragments(s)}, key=len, reverse=True)
    pats = {f: re.compile(rf"(?<![\w-]){re.escape(f)}(?![\w-])", re.I) for f in frags}

    def mask(text):
        low = text.lower()
        for f in frags:                     # longest first, so a longer fragment is masked before its parts
            if f in low:
                text = pats[f].sub("[decision]", text)
                low = text.lower()
        return DATE.sub("[date]", text)
    return mask


def has_gold_fragment(text, gold):
    for g in gold:
        if not g.startswith("decision:"):
            continue
        subj = g.split(":", 1)[1]
        slug = subj.split("/")[-1]
        if any(re.search(rf"(?<![\w-]){re.escape(f)}(?![\w-])", text, re.I) for f in subject_fragments(subj)):
            return True
        m = re.search(r"(20\d{6})$", slug)
        if m and m.group(1) in text:
            return True
    return False


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def split_of(session):
    h = int(hashlib.sha256(f"{SEED}:{session}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < DEV_SHARE else "test"


def build(tasks, catalog_ids):
    stats = collections.Counter()
    tasks = sorted(tasks, key=lambda t: (t.get("at") or "", t["tid"]))
    mask = fragment_masker([c.split(":", 1)[1] for c in catalog_ids if c.startswith("decision:")])
    seen, per_gold, out, toks = set(), collections.Counter(), [], []
    for t in tasks:
        gold = [g for g in t["gold"] if g in catalog_ids]
        if not gold:
            stats["gold_not_in_catalog"] += 1
            continue
        frag = has_gold_fragment(t["text"], gold)
        t = dict(t, text=mask(t["text"]))
        k = norm_key(t["text"])
        if k in seen:
            stats["near_duplicate"] += 1
            continue
        tk = set(re.findall(r"\w+", t["text"][:MAX_CHARS].lower()))
        if any(jaccard(tk, o) >= JACCARD_DUP for o in toks):
            stats["near_duplicate_jaccard"] += 1
            continue
        seen.add(k)
        toks.append(tk)
        stats["gold_fragment"] += frag
        if per_gold[gold[0]] >= CAP:
            stats["over_cap"] += 1
            continue
        per_gold[gold[0]] += 1
        out.append({"tid": t["tid"], "source": t["source"], "session": t["session"], "at": t.get("at", ""),
                    "split": split_of(t["session"]), "kind": gold[0].split(":")[0], "text": t["text"][:MAX_CHARS],
                    "gold": gold, "gold_first": gold[0], "gold_fragment": frag})
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
