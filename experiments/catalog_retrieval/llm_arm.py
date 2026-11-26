#!/usr/bin/env python3
"""WORK-2283 LLM arm: Claude Haiku 5.5 reads the whole catalog (same item text as every arm) and picks 10.

  python3 llm_arm.py <data_dir> <split> [--workers 4]

The catalog is the system prompt (a file, cached by the provider across calls), one numbered line per item; the
model answers row numbers, mapped back to catalog ids (long ids were altered by the model in the v2 pilot). The task
is the user message. Every raw reply is stored. A reply with 0 valid picks is a failed row.
Each call runs `claude -p` with no tools, no settings and no session persistence. The accepted vocabulary is the
catalog ids in the prompt, never the gold. A call that fails twice is a failed row (counted, never a miss).
A usage-limit reply is fatal: the run stops loudly.
"""
import argparse
import collections
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_arms as A

MODEL = "claude-haiku-5-5"
INSTR = """You choose which catalog items an agent must read before it does a task.
The catalog lists skills (procedures an agent can load) and owner decisions (rules and decisions the agent must follow).
Each line is: <number> | <name and description>.
Read the task, then return the numbers of the 10 most relevant items, most relevant first, as JSON only:
{"ids": [<number>, ...]}. Use only numbers that appear in the catalog. Do not explain.

CATALOG
"""


def catalog_prompt(cat):
    return INSTR + "\n".join(f"{i} | {A.item_text(c)}" for i, c in enumerate(cat)) + "\n"


def call(sys_file, text, timeout=240):
    r = subprocess.run(["claude", "-p", "--model", MODEL, "--output-format", "json", "--tools", "",
                        "--setting-sources", "", "--no-session-persistence", "--system-prompt-file", str(sys_file)],
                       input=f"TASK\n{text}", capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    d = json.loads(out) if out.startswith(("{", "[")) else {}
    if isinstance(d, list):
        d = [x for x in d if isinstance(x, dict) and x.get("type") == "result"][-1]
    res = str(d.get("result", ""))
    if re.search(r"(?i)usage limit|rate limit|limit reached", res + r.stderr):
        raise SystemExit(f"FATAL usage limit: {res[:200]} {r.stderr[:200]}")
    if r.returncode != 0 or d.get("is_error"):
        raise RuntimeError(f"call failed rc={r.returncode}: {res[:200]} {r.stderr[:200]}")
    return res, d.get("usage", {}), float(d.get("total_cost_usd") or 0.0)


def parse(res, n_items):
    """Catalog row numbers from the reply, in order, unique, in range; the vocabulary is the prompt's numbering."""
    m = re.search(r"\{.*\}", res, re.S)
    got = []
    try:
        got = json.loads(m.group(0)).get("ids", []) if m else []
    except ValueError:
        got = []
    if not got:
        got = re.findall(r"\b\d{1,4}\b", res)
    out = []
    for g in got:
        try:
            i = int(str(g).strip().lstrip("#"))
        except ValueError:
            continue
        if 0 <= i < n_items and i not in out:
            out.append(i)
    return out[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("split", choices=("dev", "test"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d = Path(a.data_dir)
    cat, tasks = A.load(d)
    tasks = [t for t in tasks if t["split"] == a.split]
    pos = {c["id"]: i for i, c in enumerate(cat)}
    sys_file = d / "llm_catalog_prompt.txt"
    sys_file.write_text(catalog_prompt(cat))
    out = d / "results" / a.split / "llm_catalog_v3.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["tid"] for l in open(out)} if out.exists() else set()

    def work(t):
        err = None
        for attempt in range(2):
            t0 = time.perf_counter()
            try:
                res, usage, cost = call(sys_file, t["text"])
                picks = parse(res, len(cat))
                ms = (time.perf_counter() - t0) * 1000
                if not picks:
                    raise ValueError(f"0 valid picks in the reply: {res[:200]!r}")
                # the ranking is the picks only (no padding): an item the model did not pick is not retrieved
                row = A.score_row(t, "llm_catalog", picks, [pos[g] for g in t["gold"]], ms)
                row.update(n_picks=len(picks), usage=usage, cost_usd=cost, failed=False, raw=res)
                return row
            except (RuntimeError, ValueError, subprocess.TimeoutExpired, IndexError) as e:
                err = str(e)[:300]
                time.sleep(5)
        return {"tid": t["tid"], "session": t["session"], "kind": t["kind"], "arm": "llm_catalog", "failed": True,
                "error": err, "hit5": 0, "hit10": 0, "rr": 0.0, "recall5": 0.0, "recall10": 0.0, "first_rank": None}

    todo = [t for t in tasks if t["tid"] not in done]
    print(f"{len(todo)} calls ({len(done)} done)", flush=True)
    with ThreadPoolExecutor(a.workers) as ex, open(out, "a") as f:
        for i, row in enumerate(ex.map(work, todo)):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 20 == 0:
                print(i, row.get("hit10"), row.get("failed"), flush=True)
    rows = [json.loads(l) for l in open(out)]
    ok = [r for r in rows if not r["failed"]]
    picks = sorted(collections.Counter(r.get("n_picks", 0) for r in rows).items())
    print(json.dumps({"n": len(rows), "failed": len(rows) - len(ok), "n_picks": picks,
                      "hit10": sum(r["hit10"] for r in rows) / len(rows),
                      "cost_usd": sum(r.get("cost_usd", 0) for r in ok),
                      "median_ms": sorted(r["ms"] for r in ok)[len(ok) // 2] if ok else None}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
