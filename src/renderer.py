from __future__ import annotations
from jinja2 import Template
from pathlib import Path

def render_system_prompt(jinja_path: str, *, today: str, domains: dict | None = None, subs_map: dict | None = None) -> str:
    text = Path(jinja_path).read_text(encoding="utf-8")
    tpl = Template(text)
    return tpl.render(today=today, domains=domains, subs_map=subs_map)

def read_raw_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")
