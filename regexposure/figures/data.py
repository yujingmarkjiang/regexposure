"""Load and validate the CSV inputs, and compute exposure quantities.

Every plot reads only from the CSV files listed in SCHEMA. Replace the demo
files in data/demo/ with files in the same format to plot real results.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCHEMA = {
    "cluster_labels.csv": ["cluster", "label"],                       # optional: short_label
    "firm_scores.csv": ["cik", "year", "domain", "score"],
    "rule_scores.csv": ["document_number", "domain", "score"],
    "rules.csv": ["document_number", "title", "agencies", "publication_date"],
    "pair_similarity.csv": ["cik", "year", "document_number", "cos_sim"],
    "embedding_projection.csv": ["doc_id", "source", "x", "y", "cluster"],
    "firm_year_exposure.csv": ["cik", "year", "domain", "exposure"],
    "firm_year_agency_exposure.csv": ["cik", "year", "agency", "exposure"],
}
STR_COLS = {"cik": str, "document_number": str, "doc_id": str}


def _read(path, cols):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path, dtype=STR_COLS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} is missing columns {missing}; found {list(df.columns)}")
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    return df


def load_data(data_dir):
    D = SimpleNamespace()
    for fname, cols in SCHEMA.items():
        setattr(D, fname.replace(".csv", ""), _read(os.path.join(data_dir, fname), cols))

    cl = D.cluster_labels.sort_values("cluster").reset_index(drop=True)
    D.domains = cl["label"].tolist()
    D.short = cl["short_label"].fillna(cl["label"]).tolist() if "short_label" in cl else D.domains
    D.K = len(D.domains)
    D.dom_idx = {d: i for i, d in enumerate(D.domains)}

    for name in ("firm_scores", "rule_scores", "firm_year_exposure"):
        unknown = set(getattr(D, name)["domain"]) - set(D.domains)
        if unknown:
            print(f"[warning] {name}.csv has domains not in cluster_labels.csv: {sorted(unknown)[:5]}")

    D.R = (D.rule_scores.pivot_table(index="document_number", columns="domain", values="score", aggfunc="max")
           .reindex(columns=D.domains).fillna(0.0))
    D.F = (D.firm_scores.pivot_table(index=["cik", "year"], columns="domain", values="score", aggfunc="max")
           .reindex(columns=D.domains).fillna(0.0))
    D.titles = D.rules.set_index("document_number")["title"].to_dict()
    return D


def firm_vector(D, cik, year):
    try:
        return D.F.loc[(cik, year)].to_numpy(float)
    except KeyError:
        raise KeyError(f"No firm scores for cik={cik}, year={year} in firm_scores.csv")


def rule_vector(D, doc):
    if doc not in D.R.index:
        raise KeyError(f"No rule scores for document_number={doc} in rule_scores.csv")
    return D.R.loc[doc].to_numpy(float)


def rule_matrix(D, docs):
    return D.R.reindex(docs).fillna(0.0).to_numpy(float)


def focal_pairs(D, cik, year):
    """All rules paired with the focal firm-year: similarity, domain-level and max exposure."""
    p = D.pair_similarity[(D.pair_similarity["cik"] == cik) & (D.pair_similarity["year"] == year)]
    if p.empty:
        raise KeyError(f"No rows in pair_similarity.csv for cik={cik}, year={year}")
    p = p.drop_duplicates("document_number").reset_index(drop=True)
    unscored = ~p["document_number"].isin(D.R.index)
    if unscored.any():
        # a rule without domain scores would otherwise count as zero overlap and distort Fig. 2e-j
        print(f"[warning] {unscored.sum():,} of {len(p):,} rules paired with {cik}/{year} have no domain scores; "
              "they are left out of the similarity panels")
        p = p[~unscored].reset_index(drop=True)
    s = firm_vector(D, cik, year)
    Ed = rule_matrix(D, p["document_number"]) * s / 100.0
    p = p.assign(E_max=Ed.max(1), E_sum=Ed.sum(1))
    return p, Ed


def cos_of(D, cik, year, doc):
    m = D.pair_similarity
    v = m.loc[(m["cik"] == cik) & (m["year"] == year) & (m["document_number"] == doc), "cos_sim"]
    return float(v.iloc[0]) if len(v) else np.nan


def primary_domain(D):
    """Primary (highest-scoring) domain index for every rule, as a Series keyed by document_number."""
    return pd.Series(D.R.to_numpy().argmax(1), index=D.R.index)
