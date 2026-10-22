"""Data: every 10-K annual report on SEC EDGAR, tied to a firm-year by CIK and fiscal year.

1. EDGAR quarterly form indexes give every CIK that filed a 10-K.
2. Each CIK's submissions record gives the filing's period of report and primary document.
3. Fiscal year = calendar year of the period-of-report date (sec.fiscal_year_rule
   "period_end_year"), or the Compustat convention ("compustat": fiscal years ending
   January-May belong to the previous year).
4. One 10-K per firm-year: the latest original filing (amendments excluded by default).
Outputs
  data/interim/edgar_universe.csv, data/interim/filings.csv
  data/raw/edgar/<cik>/<accession>_<primary document>
"""
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from .settings import years

FULL_INDEX = "https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx"
SUBMISSIONS = "https://data.sec.gov/submissions/{name}"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
IDX_LINE = re.compile(r"^(?P<form>\S[^\s].*?)\s{2,}(?P<name>.+?)\s{2,}(?P<cik>\d+)\s{2,}(?P<date>\d{4}-\d{2}-\d{2})\s{2,}(?P<file>\S+)\s*$")


class Sec:
    """Polite SEC session: declared User-Agent and a global request-rate limit."""

    def __init__(self, cfg):
        if "example" in cfg.sec.user_agent.lower():
            raise ValueError("Set sec.user_agent in config.yaml to 'Name email@domain' (required by the SEC).")
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": cfg.sec.user_agent, "Accept-Encoding": "gzip, deflate"})
        self.gap = 1.0 / cfg.sec.requests_per_second
        self.lock, self.last = threading.Lock(), 0.0

    def get(self, url, tries=6):
        for k in range(tries):
            with self.lock:
                wait = self.last + self.gap - time.time()
                if wait > 0:
                    time.sleep(wait)
                self.last = time.time()
            try:
                r = self.s.get(url, timeout=120)
                if r.status_code == 200:
                    return r
                if r.status_code == 404:
                    return None
            except requests.RequestException:
                pass
            time.sleep(2 ** k)
        raise RuntimeError(f"failed to fetch {url}")


def fiscal_year(report_date: str, filing_date: str, rule: str) -> tuple[int, str]:
    if isinstance(report_date, str) and len(report_date) >= 7:
        y, m = int(report_date[:4]), int(report_date[5:7])
        src = "report_date"
    else:  # no period of report: assume the 10-K covers the fiscal year ending before filing
        y, m = int(filing_date[:4]) - 1, 12
        src = "filing_date"
    if rule == "compustat" and m <= 5:
        y -= 1
    return y, src


def build_universe(cfg, sec):
    ciks = set()
    forms = set(cfg.sec.forms) | ({f + "/A" for f in cfg.sec.forms} if cfg.sec.include_amendments else set())
    for y in years(cfg):
        for q in range(1, 5):
            r = sec.get(FULL_INDEX.format(y=y, q=q))
            if r is None:
                continue
            for line in r.text.splitlines():
                m = IDX_LINE.match(line)
                if m and m["form"].strip() in forms:
                    ciks.add(int(m["cik"]))
        print(f"[EDGAR] {y}: universe {len(ciks):,} CIKs")
    pd.DataFrame({"cik": sorted(ciks)}).to_csv(cfg.paths.interim / "edgar_universe.csv", index=False)
    return sorted(ciks)


def filings_for_cik(cfg, sec, cik: int) -> list[dict]:
    forms = set(cfg.sec.forms) | ({f + "/A" for f in cfg.sec.forms} if cfg.sec.include_amendments else set())
    r = sec.get(SUBMISSIONS.format(name=f"CIK{cik:010d}.json"))
    if r is None:
        return []
    j = r.json()
    blocks = [j["filings"]["recent"]]
    for extra in j["filings"].get("files", []):
        rr = sec.get(SUBMISSIONS.format(name=extra["name"]))
        if rr is not None:
            blocks.append(rr.json())
    out = []
    for b in blocks:
        for i, form in enumerate(b.get("form", [])):
            if form not in forms:
                continue
            fy, src = fiscal_year(b["reportDate"][i], b["filingDate"][i], cfg.sec.fiscal_year_rule)
            out.append({"cik": f"{cik:010d}", "company": j.get("name", ""), "form": form,
                        "accession": b["accessionNumber"][i], "filing_date": b["filingDate"][i],
                        "report_date": b["reportDate"][i], "fiscal_year": fy, "fiscal_year_source": src,
                        "primary_document": b["primaryDocument"][i]})
    return out


def download(cfg):
    sec = Sec(cfg)
    uni = cfg.paths.interim / "edgar_universe.csv"
    ciks = pd.read_csv(uni)["cik"].tolist() if uni.exists() else build_universe(cfg, sec)

    rows = []
    with ThreadPoolExecutor(4) as ex:
        for i, res in enumerate(ex.map(lambda c: filings_for_cik(cfg, sec, c), ciks), 1):
            rows += res
            if i % 1000 == 0:
                print(f"[EDGAR] submissions {i:,}/{len(ciks):,}")
    df = pd.DataFrame(rows)
    df = df[df["fiscal_year"].between(cfg.project.start_year, cfg.project.end_year)]
    df = (df.sort_values(["cik", "fiscal_year", "filing_date"])
            .assign(is_amend=lambda d: d["form"].str.endswith("/A"))
            .sort_values(["cik", "fiscal_year", "is_amend", "filing_date"], ascending=[True, True, True, False])
            .drop_duplicates(["cik", "fiscal_year"]).drop(columns="is_amend"))

    def path_of(r):
        return cfg.paths.edgar / r["cik"] / f"{r['accession']}_{r['primary_document']}"

    def fetch(r):
        p = path_of(r)
        if p.exists() or not r["primary_document"]:
            return
        url = ARCHIVE.format(cik=int(r["cik"]), acc=r["accession"].replace("-", ""), doc=r["primary_document"])
        resp = sec.get(url)
        if resp is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(resp.content)

    recs = df.to_dict("records")
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(fetch, recs))
    df["path"] = [str(path_of(r).relative_to(cfg.root)) for r in recs]
    df["downloaded"] = [path_of(r).exists() for r in recs]
    df.to_csv(cfg.paths.interim / "filings.csv", index=False)
    print(f"[EDGAR] {len(df):,} firm-years, {df['downloaded'].mean():.1%} downloaded")
