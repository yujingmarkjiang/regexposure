"""Rating firms and regulations with an LLM.

GPT-5.6 Terra receives one document at a time, with one fixed set of instructions (identical
for every firm section and every rule), and rates the document on all 20 domains (0-100).
- Firms: Item 1 (Business) and Item 1A (Risk Factors) are rated separately, giving each
  firm-year two 20-domain profiles.
- Rules: the model reads the rule's full text. A text longer than scoring.max_input_tokens
  is split into consecutive parts; the rule's score on each domain is the maximum over parts.
Outputs
  data/processed/scores/<corpus>_raw.jsonl     every answer: id, part, scores, usage, model,
                                               prompt version, text hash
  data/processed/scores/rule_scores.parquet    document_number, d01..d20
  data/processed/scores/section_scores.parquet section_id, cik, fiscal_year, section, d01..d20
"""
import pandas as pd

from . import llm, prompts
from .textutil import JsonlStore, n_tokens, read_text, sha256, token_chunks


def _labels(cfg):
    lab = pd.read_csv(cfg.paths.domains / "cluster_labels.csv").sort_values("cluster")
    return lab["label"].tolist(), lab["description"].tolist()


def _documents(cfg, corpus):
    """Yield (doc_id, header, text) for every document to score."""
    if corpus == "rules":
        rules = pd.read_parquet(cfg.paths.interim / "rules.parquet")
        kind = {"Rule": "final rule", "Proposed Rule": "proposed rule"}
        for r in rules[rules["text_path"] != ""].itertuples():
            head = (f"Document: Federal Register {kind.get(r.type, r.type.lower())}\n"
                    f"Title: {r.title}\nAction: {r.action}\nPublished: {r.publication_date}")
            yield r.document_number, head, read_text(cfg.root / r.text_path)
    else:
        sec = pd.read_csv(cfg.paths.interim / "sections.csv", dtype={"cik": str})
        name = {"item1": "Item 1 (Business)", "item1a": "Item 1A (Risk Factors)"}
        for r in sec[sec["status"] == "present"].itertuples():
            yield r.section_id, f"Document: 10-K {name[r.section]}, fiscal year {r.fiscal_year}", read_text(cfg.root / r.text_path)


def run(cfg, corpus, limit=None, dry_run=False):
    assert corpus in ("rules", "sections")
    labels, descs = _labels(cfg)
    k = len(labels)
    instructions = prompts.score_instructions(labels, descs)
    version = prompts.score_version(instructions, k)
    schema = prompts.score_schema(k)
    keys = prompts.domain_keys(k)
    store = JsonlStore(cfg.paths.scores / f"{corpus}_raw.jsonl")
    done = store.done_ids()
    client = None if dry_run else llm.get_client(cfg)
    S = cfg.scoring

    def parse(out):
        sc = out["scores"]
        return {key: int(min(max(int(sc[key]), 0), 100)) for key in keys}

    requests, meta, n_docs, n_tok = {}, {}, 0, 0
    for doc_id, head, text in _documents(cfg, corpus):
        if limit and n_docs >= limit:
            break
        n_docs += 1
        parts = token_chunks(text, S.max_input_tokens, S.part_overlap_tokens, cfg.scoring.tokenizer)
        for i, part in enumerate(parts):
            cid = f"{doc_id}::p{i + 1}of{len(parts)}"
            if cid in done:
                continue
            label = f"{head}\nPart {i + 1} of {len(parts)}" if len(parts) > 1 else head
            user = f"{label}\n\nText:\n{part}"
            if dry_run:
                n_tok += n_tokens(instructions + user, cfg.scoring.tokenizer)
                continue
            requests[cid] = llm.json_request(cfg, "scoring", instructions, user, schema, "domain_scores")
            meta[cid] = {"doc_id": doc_id, "part": i + 1, "n_parts": len(parts), "prompt_version": version,
                         "text_sha256": sha256(text)}
        if len(requests) >= S.chunk_size:
            llm.run_json_requests(cfg, client, requests, store, parse, meta)
            requests, meta = {}, {}
            print(f"[score] {corpus}: {n_docs:,} documents processed")
    if dry_run:
        p = cfg.openai.price_per_million
        cost = n_tok / 1e6 * p.input * (0.5 if cfg.openai.use_batch else 1.0)
        print(f"[score] dry run {corpus}: {n_docs:,} documents, {n_tok:,} input tokens, "
              f"about ${cost:,.0f} input cost ({'batch' if cfg.openai.use_batch else 'standard'} pricing; output extra)")
        return
    llm.run_json_requests(cfg, client, requests, store, parse, meta)
    consolidate(cfg, corpus)


def consolidate(cfg, corpus):
    """One profile per document: maximum over parts of the latest successful answer for each part."""
    labels, _ = _labels(cfg)
    keys = prompts.domain_keys(len(labels))
    latest = {}
    for r in JsonlStore(cfg.paths.scores / f"{corpus}_raw.jsonl").records():
        if r.get("ok"):
            latest[r["custom_id"]] = r
    rows = {}
    complete = {}
    for r in latest.values():
        d = r["doc_id"]
        cur = rows.setdefault(d, {key: 0 for key in keys})
        for key in keys:
            cur[key] = max(cur[key], r["result"][key])
        complete.setdefault(d, set()).add(r["part"])
        rows[d]["n_parts"] = r["n_parts"]
        rows[d]["model"], rows[d]["prompt_version"], rows[d]["text_sha256"] = r["model"], r["prompt_version"], r["text_sha256"]
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "doc_id"
    df = df.reset_index()
    df["complete"] = [len(complete[d]) == n for d, n in zip(df["doc_id"], df["n_parts"])]
    if corpus == "rules":
        df = df.rename(columns={"doc_id": "document_number"})
        df.to_parquet(cfg.paths.scores / "rule_scores.parquet", index=False)
    else:
        parts = df["doc_id"].str.rsplit("_", n=2, expand=True)
        df.insert(1, "cik", parts[0]); df.insert(2, "fiscal_year", parts[1].astype(int)); df.insert(3, "section", parts[2])
        df = df.rename(columns={"doc_id": "section_id"})
        df.to_parquet(cfg.paths.scores / "section_scores.parquet", index=False)
    print(f"[score] {corpus}: {len(df):,} documents scored ({df['complete'].mean():.1%} complete)")
