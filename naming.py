#!/usr/bin/env python3
"""The LLM's only job: name a cluster from representative clause texts, given the
ancestor name chain as context. DeepSeek deepseek-v4-flash. Shared by build_tree.py
and server.py so the batch build and the interactive re-run name identically."""
import json, os, sys, urllib.request

API = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"

def name_cluster(sample_texts, ancestors=None):
    """Return (name, ok). ancestors: list of ancestor names, root-first."""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return "[no api key]", False
    ctx = ""
    if ancestors:
        ctx = ("This group is a narrower sub-category under: "
               + " > ".join(ancestors) + ". Name the sub-category, not the parent.\n\n")
    clauses = "\n\n".join(f"- {t.strip()[:320]}" for t in sample_texts)
    prompt = (
        ctx +
        "Below are contract clauses that a clustering algorithm grouped together "
        "because their language is similar. Give the group a short category name "
        "(2 to 5 words), the kind of heading a lawyer uses in a clause library.\n\n"
        f"{clauses}\n\n"
        "Reply with ONLY the category name. No quotes, no explanation."
    )
    body = json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1, "max_tokens": 1000, "stream": False,
    }).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(API, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            out = json.load(urllib.request.urlopen(req, timeout=90))
            choices = out.get("choices") or []
            content = (choices[0]["message"].get("content") if choices else "") or ""
            lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
            name = lines[0] if lines else ""
            name = name.strip('"').strip("'").strip()
            if ":" in name and len(name.split(":")[0]) < 20:
                name = name.split(":", 1)[1].strip()
            if name:
                return name[:60], True
        except Exception as e:
            sys.stderr.write(f"  name attempt {attempt} failed: {e}\n")
    return "[naming failed]", False
