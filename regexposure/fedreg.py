"""Data: every final and proposed rule published in the Federal Register.

For each rule we keep its full text (HTML), title, summary (abstract), publication date and
issuing agencies, using the official Federal Register API (no key required).
Outputs
  data/raw/federal_register/metadata/<year>.jsonl
  data/raw/federal_register/html/<year>/<document_number>.html
  data/raw/federal_register/text/<year>/<document_number>.txt
  data/interim/rules.parquet   (one row per rule)
"""
import calendar
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from .settings import years
from .textutil import html_to_lines, write_text

API = "https://www.federalregister.gov/api/v1/documents.json"
FIELDS = ["document_number", "title", "abstract", "publication_date", "agencies", "type", "action",
          "body_html_url", "html_url", "cfr_references", "docket_ids", "regulation_id_numbers"]


def _get(session, url, params=None, tries=5):
    for k in range(tries):
        try:
            r = session.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** k)
                continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(2 ** k)
    raise RuntimeError(f"failed to fetch {url}")


def list_month(session, year, month, types):
    """All documents of the given types published in one month (monthly queries stay under the API's result cap)."""
    last = calendar.monthrange(year, month)[1]
    params = [("conditions[type][]", t) for t in types]
    params += [("conditions[publication_date][gte]", f"{year}-{month:02d}-01"),
               ("conditions[publication_date][lte]", f"{year}-{month:02d}-{last}"),
               ("per_page", 1000), ("order", "oldest")]
    params += [("fields[]", f) for f in FIELDS]
    out, url = [], API
    while url:
        r = _get(session, url, params if url == API else None).json()
        out += r.get("results", [])
        url = r.get("next_page_url")
    return out


def download(cfg):
    P = cfg.paths.fr
    s = requests.Session()
    s.headers["User-Agent"] = cfg.sec.user_agent
    for y in years(cfg):
        meta_path = P / "metadata" / f"{y}.jsonl"
        if not meta_path.exists():
            docs = []
            for m in range(1, 13):
                docs += list_month(s, y, m, cfg.federal_register.document_types)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                for d in docs:
                    f.write(json.dumps(d) + "\n")
            print(f"[FR] {y}: {len(docs):,} rules listed")
        docs = [json.loads(l) for l in open(meta_path, encoding="utf-8")]

        def fetch(d):
            num = d["document_number"]
            hp, tp = P / "html" / str(y) / f"{num}.html", P / "text" / str(y) / f"{num}.txt"
            if tp.exists():
                return
            if not hp.exists():
                if not d.get("body_html_url"):
                    return
                write_text(hp, _get(s, d["body_html_url"]).text)
            write_text(tp, "\n".join(html_to_lines(hp.read_text(encoding="utf-8", errors="replace"))))

        with ThreadPoolExecutor(cfg.federal_register.download_workers) as ex:
            list(ex.map(fetch, docs))
        print(f"[FR] {y}: full text downloaded")
    build_table(cfg)


def build_table(cfg):
    P = cfg.paths.fr
    rows = []
    for y in years(cfg):
        meta_path = P / "metadata" / f"{y}.jsonl"
        if not meta_path.exists():
            continue
        for line in open(meta_path, encoding="utf-8"):
            d = json.loads(line)
            tp = P / "text" / str(y) / f"{d['document_number']}.txt"
            ags = d.get("agencies") or []
            rows.append({
                "document_number": d["document_number"],
                "title": d.get("title") or "",
                "summary": d.get("abstract") or "",
                "action": d.get("action") or "",
                "type": d.get("type") or "",
                "publication_date": d.get("publication_date"),
                "year": int(str(d.get("publication_date"))[:4]),
                "agencies": "; ".join(a.get("name") or a.get("raw_name") or "" for a in ags),
                "agency_slugs": "; ".join(a.get("slug") or "" for a in ags),
                "text_path": str(tp.relative_to(cfg.root)) if tp.exists() else "",
            })
    df = pd.DataFrame(rows).drop_duplicates("document_number")
    df.to_parquet(cfg.paths.interim / "rules.parquet", index=False)
    print(f"[FR] rules table: {len(df):,} rules, {(df['text_path'] != '').mean():.1%} with full text")
