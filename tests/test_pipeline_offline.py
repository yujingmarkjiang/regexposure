"""End-to-end test of every step after the downloads, with a fake OpenAI client.

It checks that extraction, embedding, clustering, labeling, scoring, linking and the
figure export fit together. No network or API key is used.  Run: pytest -q
"""
import hashlib
import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from regexposure import domains, embed, extract, link, llm, prompts, score
from regexposure.figures import make as figures
from regexposure.figures.data import load_data
from regexposure.settings import load_config

TOPICS = {
    "health": "drug clinical patients hospital medicare pharmaceutical labeling prescription",
    "transport": "aviation aircraft airline pilots airworthiness vehicles rail freight",
    "environment": "emissions pollution water air hazardous waste climate discharge",
    "finance": "bank capital lending deposits securities liquidity loans credit",
}
DIM = 64


def _vec(text):
    v = np.zeros(DIM, np.float32)
    for w in re.findall(r"[a-z]+", text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 8) % 2 else -1.0
    return (v / max(np.linalg.norm(v), 1e-9)).tolist()


class FakeResponses:
    def create(self, **body):
        name = body["text"]["format"]["name"]
        user = body["input"]
        if name == "section_locations":
            heads = [l.split("\t") for l in user.split("HEADING CANDIDATES (line_index<TAB>text):\n")[1].split("\n\n")[0].splitlines()]
            body_heads = [(int(i), t) for i, t in heads][4:]           # skip the table of contents
            find = lambda pat: next(i for i, t in body_heads if re.match(pat, t, re.I))
            i1, i1a, i1b = find(r"item 1\. "), find(r"item 1a"), find(r"item 1b")
            out = {"item_1": {"status": "present", "start_line": i1, "end_line": i1a},
                   "item_1a": {"status": "present", "start_line": i1a, "end_line": i1b},
                   "boilerplate": ["Page header ACME"]}
        elif name == "cluster_label":
            words = Counter_top(user)
            out = {"label": f"{words.title()} cluster {abs(hash(user)) % 1000}", "description": f"Texts about {words}."}
        else:
            k = len(body["text"]["format"]["schema"]["properties"]["scores"]["required"])
            txt = user.lower()
            out = {"scores": {f"d{i + 1:02d}": (90 if list(TOPICS.values())[i % 4].split()[0] in txt else 0) for i in range(k)}}
        dump = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(out)}]}],
                "usage": {"input_tokens": 10, "output_tokens": 5}}
        return SimpleNamespace(model_dump=lambda: dump)


def Counter_top(text):
    from collections import Counter
    c = Counter(w for w in re.findall(r"[a-z]{5,}", text.lower()) if w not in {"cluster", "examples", "section", "register", "federal"})
    return c.most_common(1)[0][0] if c else "misc"


class FakeEmbeddings:
    def create(self, model, input, dimensions=None):
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=_vec(t)) for i, t in enumerate(input)])


class FakeClient:
    responses, embeddings = FakeResponses(), FakeEmbeddings()


def _tenk(topic, extra):
    words = TOPICS[topic]
    body = ["<p>TABLE OF CONTENTS</p>", "<p>Item 1. Business 3</p>", "<p>Item 1A. Risk Factors 9</p>", "<p>Item 1B. Unresolved Staff Comments 15</p>",
            "<p>Page header ACME</p>", "<p>Item 1. Business</p>"]
    body += [f"<p>We operate in {words} markets. {extra} {words}.</p>" for _ in range(30)]
    body += ["<p>Page header ACME</p>", "<p>Page header ACME</p>", "<p>Item 1A. Risk Factors</p>"]
    body += [f"<p>Risks include {words} regulation and {words} costs.</p>" for _ in range(30)]
    body += ["<p>Item 1B. Unresolved Staff Comments</p>", "<p>None.</p>", "<p>Item 2. Properties</p>"]
    return "<html><body>" + "".join(body) + "</body></html>"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = load_config()
    root, old = tmp_path, c.paths.data
    c.root = root
    for name in vars(c.paths):
        p = root / "data" / getattr(c.paths, name).relative_to(old)
        p.mkdir(parents=True, exist_ok=True)
        setattr(c.paths, name, p)
    c.openai.embedding_dim, c.domains.k, c.domains.pca_dims, c.domains.fit_sample = DIM, 4, 8, 1000
    c.domains.label_examples, c.domains.n_init, c.openai.max_workers = 6, 3, 2
    c.domains.projection.method, c.domains.projection.sample = "tsne", 200
    c.raw["figures"]["fig2d_firms"] = []
    monkeypatch.setattr(llm, "get_client", lambda cfg: FakeClient())

    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        topic = list(TOPICS)[i % 4]
        y = 2013 + i % 3
        tp = root / "data" / "raw" / "fr" / f"{y}-{i:05d}.txt"
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(f"This rule sets {TOPICS[topic]} requirements. " * 20)
        rows.append({"document_number": f"{y}-{i:05d}", "title": f"{topic} rule {i}", "summary": "", "action": "Final rule",
                     "type": "Rule", "publication_date": f"{y}-06-01", "year": y,
                     "agencies": {"health": "Food and Drug Administration", "transport": "Federal Aviation Administration",
                                  "environment": "Environmental Protection Agency", "finance": "Federal Deposit Insurance Corporation"}[topic],
                     "agency_slugs": "", "text_path": str(tp.relative_to(root))})
    pd.DataFrame(rows).to_parquet(c.paths.interim / "rules.parquet", index=False)
    frows = []
    for j in range(12):
        topic = list(TOPICS)[j % 4]
        cik = f"{j + 1:010d}"
        p = root / "data" / "raw" / "edgar" / cik / "x.htm"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_tenk(topic, "Our company " * rng.integers(1, 5)))
        frows.append({"cik": cik, "company": f"Firm {j}", "form": "10-K", "accession": f"0000-{j}", "filing_date": "2016-03-01",
                      "report_date": "2015-12-31", "fiscal_year": 2015, "fiscal_year_source": "report_date",
                      "primary_document": "x.htm", "path": str(p.relative_to(root)), "downloaded": True})
    pd.DataFrame(frows).to_csv(c.paths.interim / "filings.csv", index=False)
    return c


def test_pipeline_offline(cfg):
    extract.run(cfg)
    sec = pd.read_csv(cfg.paths.interim / "sections.csv", dtype={"cik": str})
    assert (sec["status"] == "present").all() and len(sec) == 24
    txt = (cfg.root / sec["text_path"].iloc[0]).read_text()
    assert txt.startswith("Item 1. Business") and "Page header ACME" not in txt and "Item 1A" not in txt

    embed.run(cfg, "rules"); embed.run(cfg, "sections")
    assert len(embed.load(cfg, "rules")[0]) == 80 and len(embed.load(cfg, "sections")[0]) == 24

    domains.run(cfg)
    labels = pd.read_csv(cfg.paths.domains / "cluster_labels.csv")
    assert len(labels) == 4 and labels["label"].is_unique

    score.run(cfg, "rules"); score.run(cfg, "sections")
    rs = pd.read_parquet(cfg.paths.scores / "rule_scores.parquet")
    ss = pd.read_parquet(cfg.paths.scores / "section_scores.parquet")
    assert len(rs) == 80 and len(ss) == 24 and rs["complete"].all()
    raw = [json.loads(l) for l in open(cfg.paths.scores / "rules_raw.jsonl")]
    assert all(r["prompt_version"] and r["text_sha256"] and r["model"] == "gpt-5.6-terra" for r in raw)
    assert all(r["endpoint"] == "/v1/responses" for r in raw)
    emb_log = [json.loads(l) for l in open(cfg.paths.logs / "embeddings_requests.jsonl")]
    assert emb_log and all(r["endpoint"] == "/v1/embeddings" for r in emb_log)

    link.run(cfg)
    fyd = pd.read_parquet(cfg.paths.exposure / "firm_year_domain.parquet")
    fay = pd.read_parquet(cfg.paths.exposure / "firm_agency_year.parquet")
    pairs = pd.read_parquet(cfg.paths.exposure / "pairs" / "fiscal_year=2015.parquet")
    assert len(fyd) == 12 * 4 and fyd["n_rules_window"].eq(80).all()
    keys = prompts.domain_keys(4)
    assert np.allclose(pairs[keys].sum(axis=1), pairs["E_sum"], atol=1e-3)
    # agency totals add up to domain totals (equal split)
    a = fay.groupby("cik")["exposure_sum"].sum(); d = fyd.groupby("cik")["exposure_sum"].sum()
    assert np.allclose(a.loc[d.index], d, rtol=1e-4)
    # exposure is the product of the two ratings
    p = pairs.iloc[0]
    firm = link.firm_profiles(cfg).set_index("cik").loc[p["cik"], keys].to_numpy(float)
    rule = rs.set_index("document_number").loc[p["document_number"], keys].to_numpy(float)
    assert np.allclose(firm * rule / 100, p[keys].to_numpy(float), atol=1e-3)

    # figure inputs are built from the pipeline outputs
    F = cfg.raw["figures"]
    rid = rs["document_number"].tolist()
    F["focal_firm"] = {"cik": "0000000001", "year": 2015}
    F["rule_a"], F["rule_d"], F["rule_b"] = ({"document_number": rid[i], "label": l} for i, l in ((0, "A"), (4, "D"), (1, "B")))
    F["fig2c_firms"] = [{"cik": f"{j:010d}", "year": 2015, "name": f"F{j}"} for j in (1, 2)]
    F["fig2c_rules"] = [{"document_number": rid[i], "name": f"R{i}"} for i in (0, 1)]
    C = figures.figure_config(cfg)
    figures.export_inputs(cfg, C, cfg.paths.figures_input)
    D = load_data(cfg.paths.figures_input)
    assert D.K == 4 and len(D.pair_similarity) >= 80


def test_figures_demo(tmp_path):
    cfg = load_config()
    cfg.paths.figures_input = tmp_path / "fi"
    cfg.paths.figures_input.mkdir()
    figures.run(cfg, demo=True, out=tmp_path / "out", formats=("png",))
    assert (tmp_path / "out" / "Figure1_exposure_measurement.png").exists()
    assert (tmp_path / "out" / "Figure2_text_embeddings.png").exists()
