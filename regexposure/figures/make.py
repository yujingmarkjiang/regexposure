"""Figures 1 and 2: build the eight figure-input CSVs from pipeline outputs and render them.

    python run.py figures            # from pipeline outputs (data/processed/...)
    python run.py figures --demo     # simulated demo inputs (layout check only)
"""
from types import SimpleNamespace

import pandas as pd

from .. import link, prompts
from .data import load_data
from .fig1 import make_figure1
from .fig2 import make_figure2
from .simulate import write_demo


def figure_config(cfg):
    """Translate the `figures:` block of config.yaml into the names the plotting code uses."""
    F = cfg.raw["figures"]
    pick = lambda d: {k: (str(v) if k in ("cik", "document_number") else v) for k, v in d.items()}
    return SimpleNamespace(
        SCORING_MODEL=cfg.openai.display_name, LABEL_MODEL=cfg.openai.display_name,
        EMBEDDING_MODEL=cfg.openai.embedding_model, EMBED_DIM=cfg.openai.embedding_dim,
        FOCAL_FIRM=pick(F["focal_firm"]), RULE_A=pick(F["rule_a"]), RULE_D=pick(F["rule_d"]), RULE_B=pick(F["rule_b"]),
        FIG1C_COLUMNS=F["fig1c_columns"], FIG1B_HIGHLIGHT=F["fig1b_highlight"], FIG1B_MAX_POINTS=F["fig1b_max_points"],
        FIG1E_DOMAINS=F.get("fig1e_domains"), FIG1F_YEAR=F["fig1f_year"], FIG1F_TOP=F["fig1f_top"],
        FIG2A_FIRM_EXCERPT=tuple(F["fig2a_firm_excerpt"]), FIG2A_RULE_EXCERPT=tuple(F["fig2a_rule_excerpt"]),
        FIG2C_FIRMS=[(str(f["cik"]), f["year"], f["name"]) for f in F["fig2c_firms"]],
        FIG2C_RULES=[(str(r["document_number"]), r["name"]) for r in F["fig2c_rules"]],
        FIG2D_HIGHLIGHT=F["fig2d_highlight"], FIG2D_MAX_POINTS=F["fig2d_max_points"],
        FIG2D_FIRMS=[(str(f["cik"]), f["year"], f["name"]) for f in F["fig2d_firms"]],
        FIG2F_CUTS=tuple(F["fig2f_cuts"]), FIG2G_THRESHOLD=F["fig2g_threshold"], FIG2H_MAX_POINTS=F["fig2h_max_points"],
        FIG2I_DOMAIN=F["fig2i_domain"], FIG2I_TOL=F["fig2i_tol"], FIG2I_N=F["fig2i_n"], FIG2I_TITLE_CHARS=F["fig2i_title_chars"],
        FIG2J_K=F["fig2j_k"], RULE_SHORT_TITLES={str(k): v for k, v in (F.get("rule_short_titles") or {}).items()},
        DOMAIN_COLORS=F.get("domain_colors") or {},
    )


def export_inputs(cfg, C, out_dir):
    """Write the eight CSVs the plotting code reads, from the pipeline's processed outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(cfg.paths.domains / "cluster_labels.csv").sort_values("cluster")
    labels[["cluster", "label", "short_label"]].to_csv(out_dir / "cluster_labels.csv", index=False)
    keys = prompts.domain_keys(len(labels))
    key2lab = dict(zip(keys, labels["label"]))
    cik, year = C.FOCAL_FIRM["cik"], C.FOCAL_FIRM["year"]

    # firm profiles: focal firm-year plus the firms shown in Fig. 2c and 2d
    prof = link.firm_profiles(cfg)
    want = {(cik, year)} | {(c, y) for c, y, _ in C.FIG2C_FIRMS + C.FIG2D_FIRMS}
    prof = prof[[(c, y) in want for c, y in zip(prof["cik"], prof["fiscal_year"])]]
    long = prof.melt(id_vars=["cik", "fiscal_year"], value_vars=keys, var_name="k", value_name="score")
    long = long[long["score"] > 0].assign(domain=lambda d: d["k"].map(key2lab)).rename(columns={"fiscal_year": "year"})
    long[["cik", "year", "domain", "score"]].to_csv(out_dir / "firm_scores.csv", index=False)

    # similarity: focal firm vs every rule in its window, and the Fig. 2c firm x rule matrix
    named = [C.RULE_A["document_number"], C.RULE_B["document_number"], C.RULE_D["document_number"]]
    sim = [link.pair_similarity(cfg, cik, year), link.pair_similarity(cfg, cik, year, named)]
    mat_rules = [d for d, _ in C.FIG2C_RULES]
    sim += [link.pair_similarity(cfg, c, y, mat_rules) for c, y, _ in C.FIG2C_FIRMS]
    sim = pd.concat(sim).drop_duplicates(["cik", "year", "document_number"])
    sim.to_csv(out_dir / "pair_similarity.csv", index=False)

    # rules that appear anywhere in the figures
    docs = set(sim["document_number"])
    rs = pd.read_parquet(cfg.paths.scores / "rule_scores.parquet")
    rs = rs[rs["complete"] & rs["document_number"].isin(docs)]
    rl = rs.melt(id_vars=["document_number"], value_vars=keys, var_name="k", value_name="score")
    rl = rl[rl["score"] > 0].assign(domain=lambda d: d["k"].map(key2lab))
    rl[["document_number", "domain", "score"]].to_csv(out_dir / "rule_scores.csv", index=False)
    rules = pd.read_parquet(cfg.paths.interim / "rules.parquet")
    rules[rules["document_number"].isin(docs)][["document_number", "title", "agencies", "publication_date"]].to_csv(
        out_dir / "rules.csv", index=False)

    # 2-D projection; firm-years starred in Fig. 2d are placed at their Business section
    proj = pd.read_csv(cfg.paths.domains / "projection.csv", dtype={"doc_id": str})
    alias = []
    for c, y, _ in C.FIG2D_FIRMS:
        row = proj[proj["doc_id"] == f"{c}_{y}_item1"]
        if len(row):
            alias.append(row.assign(doc_id=f"{c}_{y}"))
    pd.concat([proj] + alias).to_csv(out_dir / "embedding_projection.csv", index=False)

    # firm-year exposure by domain and by agency (focal firm)
    fyd = pd.read_parquet(cfg.paths.exposure / "firm_year_domain.parquet")
    fyd = fyd[fyd["cik"] == cik].rename(columns={"fiscal_year": "year", "exposure_sum": "exposure"})
    fyd[["cik", "year", "domain", "exposure"]].to_csv(out_dir / "firm_year_exposure.csv", index=False)
    fay = pd.read_parquet(cfg.paths.exposure / "firm_agency_year.parquet")
    fay = fay[fay["cik"] == cik].rename(columns={"fiscal_year": "year", "exposure_sum": "exposure"})
    fay[["cik", "year", "agency", "exposure"]].to_csv(out_dir / "firm_year_agency_exposure.csv", index=False)
    print(f"[figures] inputs written to {out_dir}")


def run(cfg, demo=False, out=None, formats=("pdf", "svg", "png")):
    C = figure_config(cfg)
    out = out or cfg.root / cfg.figures.output_dir
    if demo:
        data_dir = cfg.paths.figures_input.parent / "figures_demo"
        write_demo(data_dir)
        note = "Simulated values for illustration."
    else:
        data_dir = cfg.paths.figures_input
        export_inputs(cfg, C, data_dir)
        note = None
    D = load_data(data_dir)
    paths = make_figure1(D, C, out, formats, note) + make_figure2(D, C, out, formats, note)
    print("[figures] saved:\n  " + "\n  ".join(map(str, paths)))
