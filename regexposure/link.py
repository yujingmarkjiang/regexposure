"""Linking firms to regulations and building firm-level measures.

Each firm-year is compared with the rules published around that time: rules published in
years [t - window_years_before, t + window_years_after] of fiscal year t.
For every firm f, rule r and domain d:
    Exposure_{f,r,d} = Firm_{f,d} x Rule_{r,d} / 100
Firm_{f,d} combines the firm-year's Business and Risk Factors scores (linking.firm_profile:
"max" by default, or "mean", "business", "risk").
Semantic similarity is the cosine of the firm-year and rule embeddings (text-embedding-3-large);
the firm-year vector is the unit-length mean of its section vectors.
Outputs
  data/processed/exposure/pairs/fiscal_year=<t>.parquet  every firm-rule pair with positive
        exposure (linking.store_pairs): cik, fiscal_year, document_number, cos_sim, E_max, E_sum, d01..d20
  data/processed/exposure/firm_year_domain.parquet       sum and mean of exposure over rules, counts
  data/processed/exposure/firm_agency_year.parquet       exposure by issuing agency
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import embed, prompts


def firm_profiles(cfg) -> pd.DataFrame:
    """Firm_{f,d}: one 20-domain profile per firm-year from its two section profiles."""
    sec = pd.read_parquet(cfg.paths.scores / "section_scores.parquet")
    sec = sec[sec["complete"]]                                  # every part of the section was scored
    keys = [c for c in sec.columns if c.startswith("d") and c[1:].isdigit()]
    how = cfg.linking.firm_profile
    if how in ("business", "risk"):
        want = "item1" if how == "business" else "item1a"
        sec = sec[sec["section"] == want]
        how = "max"
    agg = sec.groupby(["cik", "fiscal_year"])[keys].agg(how)
    return agg.reset_index()


def _agency_matrix(rules: pd.DataFrame, allocation: str):
    lists = rules["agencies"].fillna("").map(lambda s: [a.strip() for a in s.split(";") if a.strip()] or ["Unknown"])
    names = sorted({a for l in lists for a in l})
    pos = {a: i for i, a in enumerate(names)}
    A = np.zeros((len(rules), len(names)), np.float32)
    for i, l in enumerate(lists):
        for a in l:
            A[i, pos[a]] = 1.0 / len(l) if allocation == "equal_split" else 1.0
    return A, names


def run(cfg):
    L = cfg.linking
    labels = pd.read_csv(cfg.paths.domains / "cluster_labels.csv").sort_values("cluster")["label"].tolist()
    keys = prompts.domain_keys(len(labels))
    rules = pd.read_parquet(cfg.paths.interim / "rules.parquet")
    rs = pd.read_parquet(cfg.paths.scores / "rule_scores.parquet")
    rs = rs[rs["complete"]]                                     # every part of the rule was scored
    rules = rules.merge(rs[["document_number"] + keys], on="document_number", how="inner")
    rid, RV = embed.load(cfg, "rules")
    rpos = pd.Series(np.arange(len(rid)), index=rid["doc_id"])
    rules = rules[rules["document_number"].isin(rpos.index)].reset_index(drop=True)
    firms = firm_profiles(cfg)
    fvec = embed.firm_year_vectors(cfg)
    firms = firms[[f"{c}_{y}" in fvec for c, y in zip(firms["cik"], firms["fiscal_year"])]]
    print(f"[link] {len(firms):,} firm-years, {len(rules):,} rules")

    pair_dir = cfg.paths.exposure / "pairs"
    pair_dir.mkdir(exist_ok=True)
    dom_rows, ag_rows = [], []
    for t, F in firms.groupby("fiscal_year"):
        w = rules["year"].between(t - L.window_years_before, t + L.window_years_after).to_numpy()
        if not w.any():
            continue
        Rw = rules[w].reset_index(drop=True)
        G = Rw[keys].to_numpy(np.float32)                                   # rules x domains
        V = np.asarray(RV[rpos.loc[Rw["document_number"]].to_numpy()], np.float32)
        A, ag_names = _agency_matrix(Rw, L.agency_allocation)
        writer = None
        for s in range(0, len(F), L.firm_chunk):
            Fc = F.iloc[s:s + L.firm_chunk]
            Sf = Fc[keys].to_numpy(np.float32)                              # firms x domains
            U = np.stack([fvec[f"{c}_{t}"] for c in Fc["cik"]])
            E = Sf[:, None, :] * G[None, :, :] / 100.0                      # firms x rules x domains
            cos = U @ V.T
            tot = E.sum(-1)                                                 # firms x rules

            esum, npos = E.sum(1), (E > 0).sum(1)
            for i, cik in enumerate(Fc["cik"]):
                for d, lab in enumerate(labels):
                    dom_rows.append((cik, t, keys[d], lab, float(esum[i, d]), float(esum[i, d] / len(Rw)),
                                     int(npos[i, d]), len(Rw)))
            ag = tot @ A                                                    # firms x agencies
            for i, cik in enumerate(Fc["cik"]):
                for j in np.nonzero(ag[i])[0]:
                    ag_rows.append((cik, t, ag_names[j], float(ag[i, j])))

            if L.store_pairs != "none":
                fi, ri = np.nonzero(E.max(-1) > 0) if L.store_pairs == "positive" else np.indices(tot.shape).reshape(2, -1)
                tbl = pd.DataFrame({"cik": Fc["cik"].to_numpy()[fi], "fiscal_year": t,
                                    "document_number": Rw["document_number"].to_numpy()[ri],
                                    "cos_sim": cos[fi, ri].round(4), "E_max": E[fi, ri].max(-1), "E_sum": tot[fi, ri]})
                for d, key in enumerate(keys):
                    tbl[key] = E[fi, ri, d].astype(np.float32)
                table = pa.Table.from_pandas(tbl, preserve_index=False)
                writer = writer or pq.ParquetWriter(pair_dir / f"fiscal_year={t}.parquet", table.schema)
                writer.write_table(table)
        if writer:
            writer.close()
        print(f"[link] fiscal year {t}: {len(F):,} firms x {len(Rw):,} rules")

    pd.DataFrame(dom_rows, columns=["cik", "fiscal_year", "domain_key", "domain", "exposure_sum", "exposure_mean",
                                    "n_rules_positive", "n_rules_window"]).to_parquet(
        cfg.paths.exposure / "firm_year_domain.parquet", index=False)
    pd.DataFrame(ag_rows, columns=["cik", "fiscal_year", "agency", "exposure_sum"]).to_parquet(
        cfg.paths.exposure / "firm_agency_year.parquet", index=False)
    print("[link] wrote firm_year_domain.parquet and firm_agency_year.parquet")


def pair_similarity(cfg, cik: str, year: int, documents=None) -> pd.DataFrame:
    """Cosine similarity of one firm-year with the scored rules in its window (or with given documents).

    Only rules that have domain scores are returned, the same set used by link.run, so a rule
    whose scoring failed never appears as a zero-overlap pair.
    """
    L = cfg.linking
    rules = pd.read_parquet(cfg.paths.interim / "rules.parquet")
    rs = pd.read_parquet(cfg.paths.scores / "rule_scores.parquet", columns=["document_number", "complete"])
    scored = set(rs.loc[rs["complete"], "document_number"])
    rules = rules[rules["document_number"].isin(scored)]
    rid, RV = embed.load(cfg, "rules")
    rpos = pd.Series(np.arange(len(rid)), index=rid["doc_id"])
    if documents is None:
        sel = rules[rules["year"].between(year - L.window_years_before, year + L.window_years_after)]["document_number"]
    else:
        sel = pd.Series(documents)
        sel = sel[sel.isin(scored)]
    sel = sel[sel.isin(rpos.index)]
    u = embed.firm_year_vectors(cfg)[f"{cik}_{year}"]
    sims = np.asarray(RV[rpos.loc[sel].to_numpy()], np.float32) @ u
    return pd.DataFrame({"cik": cik, "year": year, "document_number": sel.to_numpy(), "cos_sim": sims.round(4)})
