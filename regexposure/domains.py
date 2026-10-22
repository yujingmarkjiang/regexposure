"""A common language: 20 data-driven regulatory domains.

1. Both corpora (every rule and every 10-K Business / Risk Factors section) are embedded
   with text-embedding-3-large (embed.py).
2. Unsupervised clustering groups the embeddings into K = 20 clusters with no labels supplied:
   spherical k-means (k-means on unit-length vectors, after PCA) fitted on a sample balanced
   across the two corpora, then applied to every document.
3. GPT-5.6 Terra reads example texts from each cluster and writes a short summary label,
   which becomes the domain's name.
Outputs
  data/processed/domains/assignments.parquet   doc_id, source, cluster, distance
  data/processed/domains/cluster_labels.csv    cluster, label, short_label, description, counts
  data/processed/domains/projection.csv        2-D coordinates for Figure 1b and 2d
  data/processed/domains/model.npz             PCA + centroids (to assign new documents)
"""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from . import embed, llm, prompts
from .textutil import JsonlStore, read_text


def _unit(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def _all_docs(cfg):
    rid, R = embed.load(cfg, "rules")
    sid, S = embed.load(cfg, "sections")
    ids = pd.concat([rid.assign(source="FR"), sid.assign(source="10-K")], ignore_index=True)
    return ids, R, S


def _rows(R, S, idx):
    idx = np.asarray(idx)
    nR = len(R)
    a, b = idx[idx < nR], idx[idx >= nR] - nR
    parts = [np.asarray(R[a]) if len(a) else None, np.asarray(S[b]) if len(b) else None]
    out = np.concatenate([p for p in parts if p is not None], axis=0)
    order = np.argsort(np.r_[np.where(idx < nR)[0], np.where(idx >= nR)[0]])
    return out[order]


def cluster(cfg):
    D = cfg.domains
    ids, R, S = _all_docs(cfg)
    rng = np.random.default_rng(D.seed)
    nR, nS = len(R), len(S)
    if D.balance_corpora:
        half = D.fit_sample // 2
        samp = np.r_[rng.choice(nR, min(half, nR), replace=False), nR + rng.choice(nS, min(half, nS), replace=False)]
    else:
        samp = rng.choice(nR + nS, min(D.fit_sample, nR + nS), replace=False)
    X = _unit(_rows(R, S, samp))
    pca = PCA(n_components=D.pca_dims, random_state=D.seed).fit(X)
    Z = _unit(pca.transform(X))
    km = MiniBatchKMeans(n_clusters=D.k, random_state=D.seed, n_init=D.n_init, batch_size=4096, max_iter=300).fit(Z)
    C = _unit(km.cluster_centers_)
    np.savez(cfg.paths.domains / "model.npz", pca_components=pca.components_, pca_mean=pca.mean_, centroids=C)

    labels, dist = np.empty(nR + nS, np.int16), np.empty(nR + nS, np.float32)
    for s in range(0, nR + nS, 50000):
        idx = np.arange(s, min(s + 50000, nR + nS))
        Zc = _unit(pca.transform(_unit(_rows(R, S, idx))))
        sims = Zc @ C.T
        labels[idx], dist[idx] = sims.argmax(1), 1 - sims.max(1)
    out = ids.assign(cluster=labels, distance=dist)
    out.to_parquet(cfg.paths.domains / "assignments.parquet", index=False)
    print(f"[domains] {D.k} clusters; sizes:\n" + out.groupby(["cluster", "source"]).size().unstack(fill_value=0).to_string())
    return out


def _example_text(cfg, doc_id, source, rules, sections, n_chars):
    try:
        if source == "FR":
            r = rules.loc[doc_id]
            return f"[Federal Register rule] {r.title}\n{read_text(cfg.root / r.text_path)[:n_chars]}"
        s = sections.loc[doc_id]
        name = "Business" if s.section == "item1" else "Risk Factors"
        return f"[10-K {name} section]\n{read_text(cfg.root / s.text_path)[:n_chars]}"
    except (KeyError, FileNotFoundError):
        return ""


def label(cfg):
    """GPT-5.6 Terra writes a short summary label for each cluster from example texts."""
    D = cfg.domains
    asg = pd.read_parquet(cfg.paths.domains / "assignments.parquet")
    rules = pd.read_parquet(cfg.paths.interim / "rules.parquet").set_index("document_number")
    sections = pd.read_csv(cfg.paths.interim / "sections.csv", dtype={"cik": str}).set_index("section_id")
    rng = np.random.default_rng(D.seed)
    client = llm.get_client(cfg)
    store = JsonlStore(cfg.paths.domains / "label_raw.jsonl")

    def examples(k):
        g = asg[asg["cluster"] == k]
        picks = []
        for src in ("FR", "10-K"):
            gs = g[g["source"] == src]
            n = D.label_examples // 2
            near = gs.nsmallest(n // 2, "distance")                       # most typical texts
            rest = gs.drop(near.index)
            rand = rest.iloc[rng.choice(len(rest), min(n - len(near), len(rest)), replace=False)] if len(rest) else rest
            picks += [near, rand]
        sel = pd.concat(picks)
        texts = [_example_text(cfg, d, s, rules, sections, D.label_chars_per_example) for d, s in zip(sel["doc_id"], sel["source"])]
        return "\n\n---\n\n".join(t for t in texts if t)

    def request(k, avoid=""):
        return llm.json_request(cfg, "labeling", prompts.LABEL_INSTRUCTIONS.format(avoid=avoid),
                                f"Cluster {k} examples:\n\n{examples(k)}", prompts.LABEL_SCHEMA, "cluster_label")

    reqs = {f"c{k:02d}": request(k) for k in range(D.k)}
    meta = {c: {"prompt_version": prompts.LABEL_VERSION} for c in reqs}
    llm.run_json_requests(cfg, client, reqs, store, lambda o: o, meta)
    got = {r["custom_id"]: r["result"] for r in store.records() if r.get("ok")}

    # labels must be distinct: re-ask for duplicates, telling the model which names are taken
    for _ in range(3):
        seen, dups = {}, []
        for c in sorted(got):
            key = got[c]["label"].strip().lower()
            (dups.append(c) if key in seen else seen.setdefault(key, c))
        if not dups:
            break
        taken = "; ".join(v["label"] for v in got.values())
        for c in dups:
            redo = {f"{c}_r{len(store.records())}": request(int(c[1:]), f"The label must differ from these existing labels: {taken}.")}
            llm.run_json_requests(cfg, client, redo, store, lambda o: o, {k: {"prompt_version": prompts.LABEL_VERSION} for k in redo})
            rec = [r for r in store.records() if r["custom_id"] in redo and r.get("ok")]
            if rec:
                got[c] = rec[-1]["result"]

    counts = asg.groupby(["cluster", "source"]).size().unstack(fill_value=0)
    rows = [{"cluster": k, "label": got[f"c{k:02d}"]["label"].strip(), "short_label": got[f"c{k:02d}"]["label"].strip(),
             "description": got[f"c{k:02d}"]["description"].strip(),
             "n_rules": int(counts.loc[k].get("FR", 0)), "n_sections": int(counts.loc[k].get("10-K", 0))} for k in range(D.k)]
    pd.DataFrame(rows).to_csv(cfg.paths.domains / "cluster_labels.csv", index=False)
    print(pd.DataFrame(rows)[["cluster", "label", "n_rules", "n_sections"]].to_string(index=False))


def projection(cfg, keep_ids=()):
    """2-D projection (UMAP if installed, else t-SNE) of a balanced sample, for Figures 1b and 2d."""
    D = cfg.domains.projection
    ids, R, S = _all_docs(cfg)
    asg = pd.read_parquet(cfg.paths.domains / "assignments.parquet")
    rng = np.random.default_rng(cfg.domains.seed)
    nR = len(R)
    half = D.sample // 2
    samp = np.r_[rng.choice(nR, min(half, nR), replace=False), nR + rng.choice(len(S), min(half, len(S)), replace=False)]
    keep = np.where(ids["doc_id"].isin(list(keep_ids)))[0]
    samp = np.unique(np.r_[samp, keep])
    m = np.load(cfg.paths.domains / "model.npz")
    Z = _unit((_unit(_rows(R, S, samp)) - m["pca_mean"]) @ m["pca_components"].T)
    method = D.method
    if method == "umap":
        try:
            import umap
            XY = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="cosine", random_state=cfg.domains.seed).fit_transform(Z)
        except ImportError:
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        XY = TSNE(n_components=2, perplexity=40, init="pca", random_state=cfg.domains.seed).fit_transform(Z[:, :50])
    out = ids.iloc[samp][["doc_id", "source"]].reset_index(drop=True)
    out["x"], out["y"] = XY[:, 0], XY[:, 1]
    out["cluster"] = asg.set_index("doc_id").loc[out["doc_id"], "cluster"].to_numpy()
    out.round(4).to_csv(cfg.paths.domains / "projection.csv", index=False)
    print(f"[domains] projection of {len(out):,} documents ({method})")


def run(cfg):
    cluster(cfg)
    label(cfg)
    keep = [f"{f['cik']}_{f['year']}_item1" for f in cfg.raw["figures"]["fig2d_firms"]]
    projection(cfg, keep)
