"""
report_generator.py — Timestamped HTML/JSON report writer.

Reports are saved to a dedicated output directory. Source data is never modified.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from jinja2 import Environment, BaseLoader
    HAS_JINJA = True
except ImportError:
    HAS_JINJA = False


_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{{ title }}</title>
  <style>
    body { font-family: sans-serif; max-width: 900px; margin: 2rem auto; }
    h1 { color: #2c3e50; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }
    th { background: #f4f6f7; }
    .meta { color: #7f8c8d; font-size: 0.85rem; margin-bottom: 1.5rem; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <p class="meta">Generated: {{ timestamp }} | Run ID: {{ run_id }}</p>
  <pre>{{ body }}</pre>
</body>
</html>
"""


def make_run_id() -> str:
    """Return a sortable timestamped run ID string."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def save_json_report(data: dict[str, Any], output_dir: str | Path, filename: str) -> Path:
    """Save a dictionary as a pretty-printed JSON report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fpath = out / filename
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return fpath


def save_html_report(
    title: str,
    body: str,
    output_dir: str | Path,
    filename: str,
    run_id: Optional[str] = None,
) -> Path:
    """Render and save a basic HTML report."""
    if not HAS_JINJA:
        raise ImportError("Install jinja2: pip install jinja2")
    run_id = run_id or make_run_id()
    env = Environment(loader=BaseLoader())
    tmpl = env.from_string(_REPORT_TEMPLATE)
    html = tmpl.render(
        title=title,
        timestamp=datetime.utcnow().isoformat() + "Z",
        run_id=run_id,
        body=body,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fpath = out / filename
    fpath.write_text(html, encoding="utf-8")
    return fpath
