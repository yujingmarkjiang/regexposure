"""Text embeddings with OpenAI text-embedding-3-large (3,072 dimensions).

Embedded documents: every Federal Register rule (title + full text) and every 10-K Business
and Risk Factors section. Vectors are unit length, so cosine similarity is a dot product.
Outputs (resumable shards)
  data/interim/embeddings/<corpus>_<n>.npy   float32 [n_docs x 3072]
  data/interim/embeddings/<corpus>_<n>.csv   doc_id per row, same order
"""
import numpy as np
import pandas as pd

from . import llm
from .textutil import read_text


def shards(cfg, corpus):
    return sorted(cfg.paths.embeddings.glob(f"{corpus}_*.npy"), key=lambda p: int(p.stem.split("_")[-1]))


def load(cfg, corpus):
    """Return (ids DataFrame, float32 matrix) for corpus 'rules' or 'sections'."""
    ids, mats = [], []
    for p in shards(cfg, corpus):
        ids.append(pd.read_csv(p.with_suffix(".csv"), dtype={"doc_id": str}))
        mats.append(np.load(p, mmap_mode="r"))
    if not ids:
        return pd.DataFrame({"doc_id": []}), np.zeros((0, cfg.openai.embedding_dim), np.float32)
    return pd.concat(ids, ignore_index=True), np.concatenate(mats, axis=0)


def _documents(cfg, corpus):
    if corpus == "rules":
        rules = pd.read_parquet(cfg.paths.interim / "rules.parquet")
        rules = rules[rules["text_path"] != ""]
        for r in rules.itertuples():
            yield r.document_number, f"{r.title}\n\n{read_text(cfg.root / r.text_path)}"
    else:
        sec = pd.read_csv(cfg.paths.interim / "sections.csv", dtype={"cik": str})
        for r in sec[sec["status"] == "present"].itertuples():
            yield r.section_id, read_text(cfg.root / r.text_path)


def run(cfg, corpus):
    assert corpus in ("rules", "sections")
    done = set(load(cfg, corpus)[0]["doc_id"])
    client = llm.get_client(cfg)
    n = len(shards(cfg, corpus))
    buf = []

    def flush():
        nonlocal n, buf
        if not buf:
            return
        vecs = llm.embed_documents(cfg, client, buf)
        keep = [d for d, _ in buf if d in vecs]
        np.save(cfg.paths.embeddings / f"{corpus}_{n}.npy", np.stack([vecs[d] for d in keep]).astype(np.float32))
        pd.DataFrame({"doc_id": keep}).to_csv(cfg.paths.embeddings / f"{corpus}_{n}.csv", index=False)
        print(f"[embed] {corpus}: shard {n} ({len(keep):,} documents)")
        n, buf = n + 1, []

    for doc_id, text in _documents(cfg, corpus):
        if doc_id in done or not text.strip():
            continue
        buf.append((doc_id, text))
        if len(buf) >= cfg.embedding.shard_size:
            flush()
    flush()


def firm_year_vectors(cfg):
    """Firm-year vector u = unit-length mean of its Business and Risk Factors section vectors."""
    ids, M = load(cfg, "sections")
    key = ids["doc_id"].str.rsplit("_", n=1).str[0]                   # <cik>_<fiscal_year>
    out = {}
    for fy_id, idx in key.groupby(key).groups.items():
        v = np.asarray(M[np.asarray(idx)], dtype=np.float32).mean(0)
        out[fy_id] = v / max(np.linalg.norm(v), 1e-12)
    return out
