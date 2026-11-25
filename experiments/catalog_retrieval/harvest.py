#!/usr/bin/env python3
"""WORK-2283 harvest: the catalog (skills + owner decisions/rules) and real agent tasks with rule-based gold.

Catalog
  - skills: every ~/src/agents/skills/<name>/SKILL.md (name, description, body).
  - decisions: one item per ops.owner_decisions subject_key whose newest row is decided or executed
    (rules such as Rule A and Rule G are rows of this table).

Tasks (gold never comes from a model)
  - skill tasks: governance.skill_invocation rows. The text is the last real user or brief message before the
    Skill tool call in the session transcript. A user-typed slash command for that skill is excluded (the user
    named the skill: no retrieval). The gold is the invoked skill.
  - decision tasks: briefs (Agent tool prompts, teammate messages, user messages) in the Claude transcripts that
    cite owner_decision:<subject>. The gold is the cited subjects that are in the catalog.
  - masking: the gold names (skill names, decision citations, subject keys) are removed from the task text.
Every text goes through the canonical redactor (lib/redact.sh) on its full length before it is written.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SKILLS = Path.home() / "src/agents/skills"
PROJECTS = Path.home() / ".claude/projects"
PG = str(Path.home() / "src/infrastructure/bin/pg-exec")
CITE = re.compile(r"owner_decision:([a-z0-9][a-z0-9/_.-]{2,120}[a-z0-9])")
SYS_BLOCK = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
TAG_BLOCK = re.compile(r"<(local-command-stdout|command-message|command-args|local-command-caveat)>.*?</\1>", re.S)
EXCLUDE_TEXT = ("Base directory for this skill", "This session is being continued from a previous conversation",
                "Caveat: The messages below", "[Request interrupted")
THIS_STUDY = "WORK-2283"
# harness notifications (an agent's report, not a task) and batch-judge prompts (citations sit inside the judged
# records, as data): found by the hand check of the v1 key, excluded before any test scoring
NOT_A_TASK = ("<task-notification>", "For each record below")


def pg_rows(sql):
    out = subprocess.run([PG, "-At", "-F", "\t", "-c", sql], check=True, capture_output=True, text=True).stdout
    return [line.split("\t") for line in out.splitlines() if line]


def frontmatter(text):
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return {}, text
    fm, body = m.group(1), m.group(2)
    meta = {}
    for key in ("name", "description"):
        mm = re.search(rf"^{key}:\s*(.*)$", fm, re.M)
        if mm:
            v = mm.group(1).strip().strip('"').strip("'")
            if v in ("|", ">", "|-", ">-"):
                lines = []
                for line in fm[mm.end():].splitlines()[1:]:
                    if line.startswith("  "):
                        lines.append(line.strip())
                    else:
                        break
                v = " ".join(lines)
            meta[key] = v
    return meta, body


def catalog():
    items = []
    for d in sorted(SKILLS.iterdir()):
        f = d / "SKILL.md"
        if not f.is_file():
            continue
        meta, body = frontmatter(f.read_text(errors="replace"))
        items.append({"id": f"skill:{d.name}", "kind": "skill", "name": d.name,
                      "description": meta.get("description", ""), "body": body.strip()[:4000], "group": None})
    rows = pg_rows("""
        SELECT DISTINCT ON (subject_key) subject_key, status, coalesce(domain,'other'),
               regexp_replace(coalesce(headline,''), '\\s+', ' ', 'g'),
               regexp_replace(coalesce(question,''), '\\s+', ' ', 'g'),
               regexp_replace(coalesce(decision,''), '\\s+', ' ', 'g')
        FROM ops.owner_decisions ORDER BY subject_key, id DESC""")
    for subject, status, domain, headline, question, decision in rows:
        if status not in ("decided", "executed"):
            continue
        desc = " ".join(x for x in (headline, decision or question) if x)
        items.append({"id": f"decision:{subject}", "kind": "decision", "name": subject, "description": desc[:1200],
                      "body": "", "group": domain})
    return items


def transcripts():
    idx = {}
    for p in glob.glob(str(PROJECTS / "*/*.jsonl")):
        idx.setdefault(Path(p).stem, p)
    return idx


def msg_text(rec):
    """The human/brief text of a transcript user record, or None (tool results, reminders, meta)."""
    if rec.get("type") != "user" or rec.get("isMeta") or rec.get("isCompactSummary"):
        return None
    c = (rec.get("message") or {}).get("content")
    if isinstance(c, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
            return None
        c = "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    if not isinstance(c, str):
        return None
    t = TAG_BLOCK.sub("", SYS_BLOCK.sub("", c)).strip()
    if not t or t.startswith(EXCLUDE_TEXT) or any(m in t for m in NOT_A_TASK):
        return None
    return t


def mask_skill(text, name):
    variants = {name, name.replace("-", " "), name.replace("-", "_"), name.replace("_", "-")}
    for v in sorted(variants, key=len, reverse=True):
        text = re.sub(rf"(?i)/?\b{re.escape(v)}\b", "[skill]", text)
    return text


def mask_decisions(text, subjects):
    text = re.sub(r"owner_decision:[a-z0-9][a-z0-9/_.-]*", "[decision]", text)
    for s in sorted(subjects, key=len, reverse=True):
        text = text.replace(s, "[decision]")
    return text


REDACT = str(Path.home() / "src/infrastructure/lib/redact.sh")


def redact(text):
    """The canonical redactor on the FULL text (truncation happens later, in build_key.py)."""
    r = subprocess.run(["bash", "-c", f'source "{REDACT}" && redact'], input=text, capture_output=True, text=True,
                       check=True)
    return r.stdout.rstrip("\n") if not text.endswith("\n") else r.stdout


def words(t):
    return len(re.findall(r"\w+", t))


def skill_tasks(cat_ids, idx):
    """Every Skill tool call for an in-catalog skill in any transcript (main sessions and subagents)."""
    logged = {(r[0], r[1].split(":")[-1]) for r in pg_rows(
        "SELECT DISTINCT session_ref, skill_name FROM governance.skill_invocation")}
    files = subprocess.run(["grep", "-rlF", "--include=*.jsonl", '"name":"Skill"', str(PROJECTS)],
                           capture_output=True, text=True).stdout.split()
    out, stats = [], {"files": len(files), "calls": 0, "not_in_catalog": 0, "slash_named": 0, "too_short": 0,
                      "kept": 0, "kept_in_skill_invocation": 0}
    for path in files:
        session = Path(path).parent.parent.name if "/subagents/" in path else Path(path).stem
        last_text, slash_for, seen = None, None, set()
        with open(path, errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("type") == "user":
                    c = (rec.get("message") or {}).get("content")
                    raw = c if isinstance(c, str) else ""
                    if isinstance(c, list):
                        raw = "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
                    m = re.search(r"<command-name>/?([^<]+)</command-name>", raw)
                    if m:
                        slash_for = m.group(1).strip().split(":")[-1]
                        continue
                    t = msg_text(rec)
                    if t:
                        last_text, slash_for = t, None
                    continue
                if rec.get("type") != "assistant":
                    continue
                for b in (rec.get("message") or {}).get("content") or []:
                    if not (isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Skill"):
                        continue
                    base = str((b.get("input") or {}).get("skill", "")).split(":")[-1]
                    stats["calls"] += 1
                    if f"skill:{base}" not in cat_ids:
                        stats["not_in_catalog"] += 1
                        continue
                    if base in seen:
                        continue
                    seen.add(base)
                    if slash_for == base:
                        stats["slash_named"] += 1
                        continue
                    if not last_text or THIS_STUDY in last_text:
                        stats["too_short"] += 1
                        continue
                    text = mask_skill(last_text, base)
                    if words(text) < 6:
                        stats["too_short"] += 1
                        continue
                    stats["kept"] += 1
                    inlog = (session, base) in logged
                    stats["kept_in_skill_invocation"] += inlog
                    out.append({"source": "skill_call", "session": session, "at": rec.get("timestamp", ""),
                                "text": text, "gold": [f"skill:{base}"], "in_skill_invocation": inlog})
    return out, stats


def decision_tasks(cat_ids, idx):
    subjects = {i[len("decision:"):] for i in cat_ids if i.startswith("decision:")}
    hits = subprocess.run(["grep", "-rlF", "--include=*.jsonl", "owner_decision:", str(PROJECTS)],
                          capture_output=True, text=True).stdout.split()
    out, stats = [], {"files": len(hits), "texts_with_cite": 0, "no_catalog_cite": 0, "too_short": 0, "kept": 0}
    for path in hits:
        if "/subagents/" in path:
            continue
        session = Path(path).stem
        with open(path, errors="replace") as fh:
            for line in fh:
                if "owner_decision:" not in line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                texts = []
                t = msg_text(rec)
                if t:
                    texts.append(("message", t))
                if rec.get("type") == "assistant":
                    for b in (rec.get("message") or {}).get("content") or []:
                        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") in ("Agent", "Task"):
                            texts.append(("brief", str((b.get("input") or {}).get("prompt", ""))))
                for kind, t in texts:
                    cites = [c.rstrip(".,;:)") for c in CITE.findall(t)]
                    if not cites:
                        continue
                    stats["texts_with_cite"] += 1
                    gold = []
                    for c in cites:
                        if c in subjects and f"decision:{c}" not in gold:
                            gold.append(f"decision:{c}")
                    if not gold:
                        stats["no_catalog_cite"] += 1
                        continue
                    if THIS_STUDY in t:
                        continue
                    text = mask_decisions(t, subjects)
                    if words(text) < 8:
                        stats["too_short"] += 1
                        continue
                    stats["kept"] += 1
                    out.append({"source": f"decision_{kind}", "session": session, "at": rec.get("timestamp", ""),
                                "text": text, "gold": gold})
    return out, stats


def main(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cat = catalog()
    cat_ids = {i["id"] for i in cat}
    idx = transcripts()
    st, s_stats = skill_tasks(cat_ids, idx)
    dt, d_stats = decision_tasks(cat_ids, idx)
    with open(outdir / "catalog.redacted.jsonl", "w") as f:
        for i in cat:
            i["description"], i["body"] = redact(i["description"]), redact(i["body"])
            f.write(json.dumps(i) + "\n")
    with open(outdir / "tasks.redacted.jsonl", "w") as f:
        for t in st + dt:
            t["text"] = redact(t["text"])
            t["tid"] = hashlib.sha256((t["source"] + t["session"] + t["text"]).encode()).hexdigest()[:16]
            f.write(json.dumps(t) + "\n")
    json.dump({"catalog": {"skill": sum(i["kind"] == "skill" for i in cat),
                           "decision": sum(i["kind"] == "decision" for i in cat)},
               "skill_tasks": s_stats, "decision_tasks": d_stats}, open(outdir / "harvest_stats.json", "w"), indent=1)
    print(open(outdir / "harvest_stats.json").read())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/data/cohesion-gated-taxonomy/catalog_retrieval"))
