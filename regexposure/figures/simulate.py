"""Write simulated demo data in exactly the format real data must follow.

The demo shows the expected file layout; none of these values are results.
Run:  python make_figures.py --demo
"""
import os

import numpy as np
import pandas as pd

DOMAINS = ["Environment & climate", "Energy & resources", "Technology & cyber", "Data privacy & AI",
           "Finance & securities", "Tax & accounting", "Labor & employment", "Workplace safety",
           "Healthcare & pharma", "Consumer protection", "Product safety", "Antitrust",
           "Trade & sanctions", "Supply chain", "Transportation", "Telecom & media",
           "Agriculture & food", "Real estate & land", "Governance & compliance", "Gov. contracting"]
IDX = {d: i for i, d in enumerate(DOMAINS)}
P_DOM = np.array([.16, .06, .02, .01, .08, .04, .04, .02, .09, .03, .03, .005, .04, .01, .22, .04, .07, .02, .02, .025])
P_DOM = P_DOM / P_DOM.sum()
AGENCY = {0: ["EPA"], 1: ["DOE", "FERC", "BOEM"], 2: ["NIST", "CISA"], 3: ["FTC", "HHS"], 4: ["SEC", "FDIC", "OCC", "CFPB"],
          5: ["IRS"], 6: ["DOL", "EEOC"], 7: ["OSHA", "MSHA"], 8: ["FDA", "CMS"], 9: ["CFPB", "FTC"], 10: ["CPSC", "FDA"],
          11: ["FTC", "DOJ"], 12: ["BIS", "OFAC", "CBP"], 13: ["FDA", "GSA"], 14: ["FAA", "FMCSA", "NHTSA", "USCG"],
          15: ["FCC"], 16: ["USDA", "FDA"], 17: ["HUD", "BLM"], 18: ["SEC"], 19: ["GSA", "DOD"]}

FOCAL = ("0000000001", 2015)
FIRMS = {  # cik: (name, {domain: score})
    "0000000001": ("Pharmaceutical", dict(zip(DOMAINS, [12, 5, 18, 27, 9, 14, 22, 0, 90, 34, 72, 15, 18, 46, 0, 0, 0, 6, 11, 24]))),
    "0000000002": ("Airline", {"Transportation": 91, "Workplace safety": 38, "Environment & climate": 33, "Energy & resources": 24,
                               "Consumer protection": 29, "Labor & employment": 44, "Trade & sanctions": 12}),
    "0000000003": ("Bank", {"Finance & securities": 93, "Data privacy & AI": 47, "Technology & cyber": 41, "Consumer protection": 43,
                            "Governance & compliance": 36, "Tax & accounting": 27}),
    "0000000004": ("Electric utility", {"Energy & resources": 89, "Environment & climate": 84, "Real estate & land": 31,
                                        "Workplace safety": 27, "Gov. contracting": 12}),
    "0000000005": ("Software", {"Technology & cyber": 92, "Data privacy & AI": 76, "Consumer protection": 31,
                                "Trade & sanctions": 21, "Antitrust": 26, "Labor & employment": 17}),
}
NAMED = [  # document_number, title, agency, {domain: score}, cos with focal firm
    ("2015-01234", "Drug labeling requirements", "FDA", {"Healthcare & pharma": 90, "Consumer protection": 55, "Product safety": 76,
                                                        "Supply chain": 12, "Governance & compliance": 6}, 0.61),
    ("2015-02345", "Hospital billing transparency", "CMS", {"Healthcare & pharma": 90, "Consumer protection": 21, "Finance & securities": 14}, 0.47),
    ("2015-03456", "Aviation safety standards", "FAA", {"Transportation": 93, "Workplace safety": 31}, 0.14),
    ("2015-04567", "Capital requirements", "FDIC; OCC", {"Finance & securities": 90, "Governance & compliance": 40}, 0.09),
    ("2015-05678", "Emission standards", "EPA", {"Environment & climate": 92, "Energy & resources": 58}, 0.14),
    ("2015-06789", "Data privacy safeguards", "FTC", {"Data privacy & AI": 92, "Technology & cyber": 55, "Consumer protection": 40}, 0.23),
    ("2014-11111", "Clinical trial registration", "FDA", {"Healthcare & pharma": 90, "Product safety": 30}, 0.59),
    ("2014-22222", "Biosimilar licensure", "FDA", {"Healthcare & pharma": 90, "Antitrust": 20}, 0.57),
    ("2013-33333", "Drug supply-chain security", "FDA", {"Healthcare & pharma": 90, "Supply chain": 70}, 0.54),
    ("2015-44444", "Drug price transparency", "CMS", {"Healthcare & pharma": 90, "Consumer protection": 35}, 0.52),
    ("2014-55555", "Physician fee schedule", "CMS", {"Healthcare & pharma": 90, "Finance & securities": 10}, 0.41),
    ("2013-66666", "Hospice payment update", "CMS", {"Healthcare & pharma": 90}, 0.29),
]
MATRIX = np.array([[0.61, 0.14, 0.09, 0.14, 0.23], [0.08, 0.57, 0.16, 0.33, 0.19], [0.07, 0.13, 0.54, 0.18, 0.29],
                   [0.12, 0.17, 0.21, 0.59, 0.12], [0.18, 0.10, 0.24, 0.09, 0.51]])
MATRIX_RULES = ["2015-01234", "2015-03456", "2015-04567", "2015-05678", "2015-06789"]


def _centers(rng, k=20):
    c = []
    while len(c) < k:
        p = rng.uniform([0.8, 0.8], [9.2, 7.2])
        if all(np.hypot(*(p - q)) > 1.45 for q in c):
            c.append(p)
    return np.array(c)


def write_demo(out_dir, seed=9, n=5831):          # 5,831 simulated + 12 named rules = 5,843
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    hl = IDX["Healthcare & pharma"]

    # cluster labels -------------------------------------------------------------------------
    pd.DataFrame({"cluster": range(20), "label": DOMAINS, "short_label": DOMAINS}).to_csv(f"{out_dir}/cluster_labels.csv", index=False)

    # firm scores ----------------------------------------------------------------------------
    rows = [(cik, FOCAL[1], d, s) for cik, (_, sc) in FIRMS.items() for d, s in sc.items() if s > 0]
    pd.DataFrame(rows, columns=["cik", "year", "domain", "score"]).to_csv(f"{out_dir}/firm_scores.csv", index=False)
    s_focal = np.array([FIRMS[FOCAL[0]][1][d] for d in DOMAINS], float)

    # rule pool -----------------------------------------------------------------------------------
    d = rng.choice(20, n, p=P_DOM)
    g = rng.integers(45, 96, n)
    bad = (d == hl) & (g >= 87) & (g <= 93)          # keep the equal-overlap example clean
    g[bad] = rng.integers(45, 86, bad.sum())
    sec = rng.choice(20, n, p=P_DOM); g2 = rng.integers(5, 40, n); has2 = (rng.random(n) < 0.4) & (sec != d)
    docs = [f"{rng.choice([2013, 2014, 2015])}-9{i:04d}" for i in range(n)]   # never clashes with the named rules
    score_rows, rule_rows = [], []
    E = np.zeros(n)
    for i in range(n):
        score_rows.append((docs[i], DOMAINS[d[i]], int(g[i])))
        e = s_focal[d[i]] * g[i] / 100
        if has2[i]:
            score_rows.append((docs[i], DOMAINS[sec[i]], int(g2[i])))
            e = max(e, s_focal[sec[i]] * g2[i] / 100)
        E[i] = e
        yr = docs[i][:4]
        rule_rows.append((docs[i], f"Simulated rule {i:05d}", rng.choice(AGENCY[d[i]]), f"{yr}-{rng.integers(1, 13):02d}-15"))
    base = 0.075 + rng.gamma(5.0, 0.021, n)
    sims = base + 0.0041 * E + rng.normal(0, 1, n) * (0.028 + 0.00035 * E)
    sims += ((rng.random(n) < 0.04) & (E < 25)) * rng.uniform(0.1, 0.33, n)
    sims = np.clip(sims, 0, 0.8)

    pair_rows = [(FOCAL[0], FOCAL[1], docs[i], round(float(sims[i]), 4)) for i in range(n)]
    for doc, title, ag, sc, cs in NAMED:
        for dm, s in sc.items():
            score_rows.append((doc, dm, s))
        rule_rows.append((doc, title, ag, f"{doc[:4]}-06-15"))
        pair_rows.append((FOCAL[0], FOCAL[1], doc, cs))
    for i, cik in enumerate(FIRMS):
        if cik == FOCAL[0]:
            continue
        for j, doc in enumerate(MATRIX_RULES):
            pair_rows.append((cik, FOCAL[1], doc, MATRIX[i, j]))
    pd.DataFrame(score_rows, columns=["document_number", "domain", "score"]).to_csv(f"{out_dir}/rule_scores.csv", index=False)
    pd.DataFrame(rule_rows, columns=["document_number", "title", "agencies", "publication_date"]).to_csv(f"{out_dir}/rules.csv", index=False)
    pd.DataFrame(pair_rows, columns=["cik", "year", "document_number", "cos_sim"]).to_csv(f"{out_dir}/pair_similarity.csv", index=False)

    # 2-D embedding projection (rules + 10-K sections) ------------------------------------------------
    crng = np.random.default_rng(5)
    C = _centers(crng)
    rot = crng.uniform(0, np.pi, 20); sx = crng.uniform(0.3, 0.55, 20); sy = crng.uniform(0.15, 0.3, 20)
    curve = np.where(crng.random(20) < 0.35, crng.uniform(0.5, 1.1, 20), 0.0)

    def place(k, m):
        z = rng.standard_normal((m, 2)) * [sx[k], sy[k]]
        z[:, 1] += curve[k] * z[:, 0] ** 2
        R = np.array([[np.cos(rot[k]), -np.sin(rot[k])], [np.sin(rot[k]), np.cos(rot[k])]])
        return z @ R.T + C[k]

    proj = []
    all_docs = docs + [x[0] for x in NAMED]
    prim = np.r_[d, [IDX[max(x[3], key=x[3].get)] for x in NAMED]]
    for k in range(20):
        idx = np.where(prim == k)[0]
        if len(idx):
            xy = place(k, len(idx)) + np.array([np.cos(rot[k] + 1.3), np.sin(rot[k] + 1.3)]) * 0.12
            proj += [(all_docs[i], "FR", x, y, k) for i, (x, y) in zip(idx, xy)]
    n10k = rng.multinomial(1400, P_DOM * 0.5 + 0.025)
    for k, m in enumerate(n10k):
        xy = place(k, m) - np.array([np.cos(rot[k] + 1.3), np.sin(rot[k] + 1.3)]) * 0.12
        proj += [(f"{900000000 + 1000 * k + j:010d}_{FOCAL[1]}", "10-K", x, y, k) for j, (x, y) in enumerate(xy)]
    fpos = {"0000000001": (0.72 * C[hl] + 0.28 * C[IDX["Product safety"]], hl),
            "0000000002": (C[IDX["Transportation"]] + [0.15, 0.25], IDX["Transportation"]),
            "0000000003": (C[IDX["Finance & securities"]] + [-0.1, 0.2], IDX["Finance & securities"]),
            "0000000004": (0.7 * C[IDX["Environment & climate"]] + 0.3 * C[IDX["Energy & resources"]], IDX["Environment & climate"]),
            "0000000005": (0.8 * C[IDX["Technology & cyber"]] + 0.2 * C[IDX["Data privacy & AI"]], IDX["Technology & cyber"])}
    for cik, (p, k) in fpos.items():
        proj.append((f"{cik}_{FOCAL[1]}", "10-K", p[0], p[1], k))
    pd.DataFrame(proj, columns=["doc_id", "source", "x", "y", "cluster"]).round(4).to_csv(f"{out_dir}/embedding_projection.csv", index=False)

    # firm-year exposure (focal firm) ----------------------------------------------------------------
    yrs = np.arange(2001, 2021)
    r = np.random.default_rng(13)

    def ar(sd, phi=0.55):
        e = np.zeros(len(yrs))
        for t in range(len(yrs)):
            e[t] = (phi * e[t - 1] if t else 0) + r.normal(0, sd)
        return e
    bump = lambda c, a, wd: a * np.exp(-((yrs - c) / wd) ** 2)
    series = {"Healthcare & pharma": (782 + 13 * (yrs - 2001) + bump(2011, 430, 1.25) + bump(2016.6, 170, 0.7)) * (1 + ar(0.045)),
              "Product safety": (402 + 5.5 * (yrs - 2001) + bump(2014.2, 235, 0.95)) * (1 + ar(0.055)),
              "Consumer protection": (258 + 4.2 * (yrs - 2001) + bump(2009.6, 66, 0.9)) * (1 + ar(0.06)),
              "Data privacy & AI": (28 + 335 / (1 + np.exp(-(yrs - 2016.8) / 0.95)) + bump(2019.6, 40, 0.6)) * (1 + ar(0.07)),
              "Supply chain": (120 + 3 * (yrs - 2001)) * (1 + ar(0.08)),
              "Trade & sanctions": (60 + 2 * (yrs - 2001)) * (1 + ar(0.1))}
    rows = [(FOCAL[0], int(y), dm, round(float(v), 1)) for dm, vals in series.items() for y, v in zip(yrs, vals)]
    pd.DataFrame(rows, columns=["cik", "year", "domain", "exposure"]).to_csv(f"{out_dir}/firm_year_exposure.csv", index=False)

    ag = [("FDA", 1186), ("CMS", 437), ("FTC", 241), ("DEA", 132), ("EPA", 87), ("OSHA", 58), ("CPSC", 31), ("USDA", 19)]
    pd.DataFrame([(FOCAL[0], 2015, a, v) for a, v in ag], columns=["cik", "year", "agency", "exposure"]).to_csv(
        f"{out_dir}/firm_year_agency_exposure.csv", index=False)
    return out_dir
