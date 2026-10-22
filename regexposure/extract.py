"""Extracting what firms do: Item 1 (Business) and Item 1A (Risk Factors).

Locating and cleaning the sections is LLM-guided. Each filing is converted to lines; GPT-5.6
Terra sees the numbered heading candidates and the repeated short lines, and returns where each
section starts and ends in the body of the report (not the table of contents) and which
repeated lines are page headers, footers or page numbers to remove.
Outputs
  data/interim/sections/<cik>/<fiscal_year>_item1.txt, ..._item1a.txt
  data/interim/extract_raw.jsonl  (every model answer, with prompt version)
  data/interim/sections.csv       (one row per firm-year-section, status and text hash)
"""
import re
from collections import Counter

import pandas as pd

from . import llm, prompts
from .textutil import JsonlStore, html_to_lines, read_text, sha256, write_text

HEADING_HINT = re.compile(r"\b(item|part)\s*[0-9ivx]|business|risk\s+factors|properties|legal\s+proceedings|"
                          r"table\s+of\s+contents|signatures|unresolved|cybersecurity", re.I)


def outline(lines: list[str], max_lines: int, max_repeated: int):
    heads = [f"{i}\t{l[:160]}" for i, l in enumerate(lines)
             if len(l) <= 200 and (HEADING_HINT.search(l[:80]) or (l.isupper() and len(l) >= 4))]
    counts = Counter(l for l in lines if len(l) <= 120)
    reps = [f"{c}\t{s}" for s, c in counts.most_common(max_repeated) if c >= 3]
    return heads[:max_lines], reps


def user_message(fy: int, n_lines: int, heads, reps) -> str:
    return (f"Fiscal year: {fy}\nTotal lines in filing: {n_lines}\n\n"
            "HEADING CANDIDATES (line_index<TAB>text):\n" + "\n".join(heads) +
            "\n\nREPEATED LINES (count<TAB>text):\n" + ("\n".join(reps) if reps else "(none)"))


def parse(out: dict) -> dict:
    for k in ("item_1", "item_1a"):
        s = out[k]
        if s["status"] in ("present", "not_applicable") and not (0 <= s["start_line"] < s["end_line"]):
            raise ValueError(f"{k}: invalid line range {s['start_line']}-{s['end_line']}")
    return out


def run(cfg, limit=None):
    filings = pd.read_csv(cfg.paths.interim / "filings.csv", dtype={"cik": str})
    filings = filings[filings["downloaded"]]
    if limit:
        filings = filings.head(limit)
    store = JsonlStore(cfg.paths.interim / "extract_raw.jsonl")
    done = store.done_ids()
    client = llm.get_client(cfg)
    X = cfg.extraction

    todo = [r for r in filings.itertuples() if f"{r.cik}_{r.fiscal_year}" not in done]
    print(f"[extract] {len(todo):,} filings to locate")
    for start in range(0, len(todo), X.chunk_size):          # bounded memory
        requests, meta = {}, {}
        for r in todo[start:start + X.chunk_size]:
            cid = f"{r.cik}_{r.fiscal_year}"
            lines = html_to_lines(read_text(cfg.root / r.path))
            heads, reps = outline(lines, X.max_heading_lines, X.max_repeated_lines)
            requests[cid] = llm.json_request(cfg, "extraction", prompts.EXTRACT_INSTRUCTIONS,
                                             user_message(r.fiscal_year, len(lines), heads, reps),
                                             prompts.EXTRACT_SCHEMA, "section_locations")
            meta[cid] = {"cik": r.cik, "fiscal_year": int(r.fiscal_year), "accession": r.accession,
                         "prompt_version": prompts.EXTRACT_VERSION}
        llm.run_json_requests(cfg, client, requests, store, parse, meta)
        print(f"[extract] {min(start + X.chunk_size, len(todo)):,}/{len(todo):,}")
    write_sections(cfg, filings)


def write_sections(cfg, filings):
    """Cut the located sections out of each filing, drop the model-flagged boilerplate, save text."""
    by_id = {f"{r.cik}_{r.fiscal_year}": r for r in filings.itertuples()}
    latest = {}
    for rec in JsonlStore(cfg.paths.interim / "extract_raw.jsonl").records():
        if rec.get("ok"):
            latest[rec["custom_id"]] = rec
    rows = []
    for cid, rec in latest.items():
        f = by_id.get(cid)
        if f is None:
            continue
        lines = html_to_lines(read_text(cfg.root / f.path))
        drop = set(rec["result"]["boilerplate"])
        for key, name in (("item_1", "item1"), ("item_1a", "item1a")):
            s = rec["result"][key]
            status, text = s["status"], ""
            if status in ("present", "not_applicable"):
                text = "\n".join(l for l in lines[s["start_line"]:s["end_line"]] if l not in drop)
                if status == "present" and len(text) < cfg.extraction.min_section_chars:
                    status = "too_short"
            path = cfg.paths.sections / f.cik / f"{f.fiscal_year}_{name}.txt"
            if status == "present":
                write_text(path, text)
            rows.append({"section_id": f"{f.cik}_{f.fiscal_year}_{name}", "cik": f.cik, "fiscal_year": int(f.fiscal_year),
                         "section": name, "status": status, "n_chars": len(text),
                         "text_path": str(path.relative_to(cfg.root)) if status == "present" else "",
                         "text_sha256": sha256(text) if status == "present" else "",
                         "accession": f.accession, "model": rec["model"], "prompt_version": rec["prompt_version"]})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.paths.interim / "sections.csv", index=False)
    if len(df):
        print("[extract] section status:\n" + df.groupby(["section", "status"]).size().to_string())
