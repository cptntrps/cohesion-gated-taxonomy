#!/usr/bin/env python3
"""The LLM's only job: name a cluster from representative clause texts, given the
ancestor name chain as context. Shared by build_tree.py and server.py so the batch
build and the interactive re-run name identically.

The endpoint is any OpenAI-compatible chat completions API:
  NAMING_API_BASE      default https://api.deepseek.com/v1
  NAMING_API_KEY       falls back to DEEPSEEK_API_KEY
  NAMING_MODEL         default deepseek-v4-flash
  NAMING_FALLBACK_MODEL default deepseek-chat (used when the model returns empty content)
On aurora the unit points NAMING_API_BASE at the livingos-api gateway's neutral
DeepSeek lane (dg-chat), so every call lands in api.usage with its cost.
"""
import json, os, sys, urllib.request

API_BASE = os.environ.get("NAMING_API_BASE", "https://api.deepseek.com/v1").rstrip("/")
API = API_BASE + "/chat/completions"
MODEL = os.environ.get("NAMING_MODEL", "deepseek-v4-flash")
FALLBACK_MODEL = os.environ.get("NAMING_FALLBACK_MODEL", "deepseek-chat")


def api_key():
    return os.environ.get("NAMING_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")


def name_cluster(sample_texts, ancestors=None, avoid=None):
    """Return (name, ok). ancestors: list of ancestor names, root-first."""
    key = api_key()
    if not key:
        return "[no api key]", False
    ctx = ""
    if ancestors:
        ctx = ("This group is a narrower sub-category under: "
               + " > ".join(ancestors) + ". Name the sub-category, not the parent.\n\n")
    if avoid:
        ctx += ("Sibling groups are already named: " + ", ".join(avoid) +
                ". Give THIS group a different, more specific name that tells it apart.\n\n")
    clauses = "\n\n".join(f"- {t.strip()[:320]}" for t in sample_texts)
    prompt = (
        ctx +
        "Below are contract clauses that a clustering algorithm grouped together "
        "because their language is similar. Give the group the SHORTEST canonical "
        "category name: 1 to 3 words, the single standard clause-library term. Do NOT "
        "combine multiple topics with 'and' — pick the dominant one. Prefer the common "
        "legal heading (e.g. 'Benefits', 'Notices', 'Governing Law', 'Counterparts').\n\n"
        f"{clauses}\n\n"
        "Reply with ONLY the name. No quotes, no explanation."
    )
    for attempt in range(4):
        try:
            model = MODEL if attempt < 2 else FALLBACK_MODEL
            body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                               "temperature": 0.1, "max_tokens": 4000 if attempt else 1000, "stream": False}).encode()
            req = urllib.request.Request(API, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            out = json.load(urllib.request.urlopen(req, timeout=90))
            choices = out.get("choices") or []
            msg = choices[0]["message"] if choices else {}
            content = msg.get("content") or ""
            if not content.strip() and msg.get("reasoning_content"):   # reasoning model ran out of tokens
                sys.stderr.write(f"  name attempt {attempt}: empty content, retrying\n")
                continue
            name = parse_name(content)
            if name:
                return name, True
        except Exception as e:
            sys.stderr.write(f"  name attempt {attempt} failed: {e}\n")
    return fallback_name(sample_texts), False


def parse_name(content):
    """First non-empty line, unquoted, without a 'Label:' prefix, capped at 60 chars."""
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    name = lines[0] if lines else ""
    name = name.strip('"').strip("'").strip()
    if ":" in name and len(name.split(":")[0]) < 20:
        name = name.split(":", 1)[1].strip()
    return name[:60]


def fallback_name(sample_texts):
    """Last resort, deterministic: the most frequent capitalised heading-like phrase in the samples,
    else the most common content words. Marked (auto) so a human can tell it from a model name."""
    import re, collections
    heads = collections.Counter()
    for t in sample_texts:
        m = re.match(r"\s*(?:section\s*[\d.]+\s*)?([A-Z][A-Za-z' -]{2,40}?)\s*[.:;-]", t.strip())
        if m:
            heads[m.group(1).strip().title()] += 1
    if heads:
        return heads.most_common(1)[0][0][:40] + " (auto)"
    stop = set("the of and to in a or any shall be this that by for with as on at its such other from is are will not".split())
    words = collections.Counter(w for t in sample_texts for w in re.findall(r"[a-z]{4,}", t.lower()) if w not in stop)
    return " ".join(w.title() for w, _ in words.most_common(2)) + " (auto)"
