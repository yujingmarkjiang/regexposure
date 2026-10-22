"""Figure 2: semantic similarity between firm and rule texts (panels a-j)."""
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle

from . import style as S
from .data import cos_of, focal_pairs, primary_domain


def _embedding_schematic(ax, C, rng):
    rows = [(0.95, "10-K, firm $f$", *C.FIG2A_FIRM_EXCERPT, r"\mathbf{u}", "u", S.BLUE),
            (0.32, "FR rule $r$", *C.FIG2A_RULE_EXCERPT, r"\mathbf{v}", "v", S.VERM)]
    cmap = mpl.colormaps["RdBu_r"]
    dim = C.EMBED_DIM
    for yc, hd, l1, l2, vec, sym, col in rows:
        ax.add_patch(FancyBboxPatch((0.02, yc - 0.2), 0.9, 0.42, boxstyle="round,pad=0,rounding_size=0.04", fc="white", ec=col, lw=0.8))
        ax.text(0.07, yc + 0.11, hd, fontsize=6.0, fontweight="bold", va="center", color=col)
        ax.text(0.07, yc - 0.01, l1, fontsize=5.5, va="center", style="italic")
        ax.text(0.07, yc - 0.12, l2, fontsize=5.5, va="center", style="italic")
        S.arrow(ax, 0.95, yc, 1.43, yc)
        vals = np.clip(rng.normal(0, 0.018, 18), -0.05, 0.05)
        x0, cw = 1.48, 0.04
        for i, v in enumerate(vals):
            xi = x0 + i * cw + (0.08 if i >= 15 else 0)
            ax.add_patch(Rectangle((xi, yc - 0.06), cw, 0.12, fc=cmap(0.5 + v / 0.1), ec="white", lw=0.3))
        ax.text(x0 + 15 * cw + 0.04, yc, r"$\cdots$", ha="center", va="center", fontsize=5.5)
        xe = x0 + 18 * cw + 0.08
        ax.add_patch(Rectangle((x0, yc - 0.06), xe - x0, 0.12, fill=False, ec=S.INK, lw=0.5))
        ax.text(x0 + cw / 2, yc - 0.1, f"${sym}_1$", ha="center", va="top", fontsize=5.2)
        ax.text(xe - cw / 2, yc - 0.1, f"${sym}_{{{dim}}}$", ha="center", va="top", fontsize=5.2)
        ax.text((x0 + xe) / 2, yc + 0.09, rf"${vec}\in\mathbb{{R}}^{{{dim}}},\ \|{vec}\|_2=1$", ha="center", va="bottom", fontsize=5.4)
        ax.plot([xe + 0.02, 2.46], [yc, 0.635], color=S.INK, lw=0.6)
    name = C.EMBEDDING_MODEL
    dashes = [i for i, ch in enumerate(name) if ch in "- "]
    cut = min(dashes, key=lambda i: abs(i - len(name) / 2)) if dashes and len(name) > 16 else len(name) - 1
    ax.text(1.19, 0.68, r"$\phi(\cdot)$", ha="center", va="center", fontsize=6.2)
    ax.text(1.19, 0.56, name[:cut + 1].strip(), ha="center", va="center", fontsize=5.0, color=S.GRAY)
    ax.text(1.19, 0.48, name[cut + 1:].strip(), ha="center", va="center", fontsize=5.0, color=S.GRAY)
    ax.add_patch(FancyBboxPatch((2.47, 0.5), 0.46, 0.27, boxstyle="round,pad=0,rounding_size=0.04", fc="#EDEDED", ec=S.INK, lw=0.6))
    ax.text(2.70, 0.635, r"$\cos\theta_{f,r}$", ha="center", va="center", fontsize=6.0)


def _cosine_geometry(ax, cA, cD, cB, labA, labD, labB):
    base = 6
    ang = {"f": base}
    for k, c in (("A", cA), ("D", cD), ("B", cB)):
        ang[k] = base + np.degrees(np.arccos(np.clip(c, -1, 1)))
    ax.add_patch(Arc((0, 0), 2, 2, theta1=-4, theta2=max(100, ang["B"] + 6), color=S.LGRAY, lw=0.6, ls="--"))
    for k, col, lw in [("B", S.GRAY, 0.8), ("D", S.ORANGE, 0.9), ("A", S.VERM, 0.9), ("f", S.BLUE, 1.4)]:
        t = np.radians(ang[k])
        S.arrow(ax, 0, 0, np.cos(t), np.sin(t), color=col, lw=lw, ms=5)
    lab = {"f": ("$f$", S.BLUE), "A": (f"{labA} {cA:.2f}", S.VERM), "D": (f"{labD} {cD:.2f}", S.DGOLD), "B": (f"{labB} {cB:.2f}", S.GRAY)}
    for k, (s, col) in lab.items():
        t = np.radians(ang[k])
        ax.text(1.1 * np.cos(t), 1.1 * np.sin(t), s, color=col, fontsize=5.6, ha="left" if k in ("f", "A") else "center",
                va="center" if k == "f" else "bottom", fontweight="bold" if k == "f" else "normal")
    ax.add_patch(Arc((0, 0), 0.6, 0.6, theta1=ang["f"], theta2=ang["A"], color=S.INK, lw=0.6))
    tm = np.radians((ang["f"] + ang["A"]) / 2)
    ax.text(0.41 * np.cos(tm), 0.41 * np.sin(tm), rf"$\theta_{{f{labA}}}$", fontsize=5.6, ha="center", va="center")
    ax.text(0.65, -0.12, r"$\cos\theta=\dfrac{\mathbf{u}^{\top}\mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|}$", fontsize=6.2, ha="center", va="top")
    ax.set_xlim(-0.08, 1.45); ax.set_ylim(-0.5, 1.33); ax.axis("off")


def make_figure2(D, C, out_dir, formats, note=None):
    S.apply_style()
    W, H = 7.0, 5.12
    fig, ax_in, title = S.new_figure(W, H)
    rng = np.random.default_rng(21)
    TY1, TY2, TY3 = 4.98, 3.22, 1.48
    cik, year = C.FOCAL_FIRM["cik"], C.FOCAL_FIRM["year"]
    A, B, Dd = C.RULE_A, C.RULE_B, C.RULE_D
    cA, cB, cD = (cos_of(D, cik, year, r["document_number"]) for r in (A, B, Dd))
    pairs, Ed = focal_pairs(D, cik, year)
    E, sims = pairs["E_max"].to_numpy(), pairs["cos_sim"].to_numpy()
    eA = pairs.loc[pairs["document_number"] == A["document_number"], "E_max"]
    eD = pairs.loc[pairs["document_number"] == Dd["document_number"], "E_max"]
    eB = pairs.loc[pairs["document_number"] == B["document_number"], "E_max"]
    eA, eD, eB = (float(v.iloc[0]) if len(v) else np.nan for v in (eA, eD, eB))
    xmax_cos = max(0.75, np.ceil(np.nanmax(sims) * 20) / 20)
    K_TOP = C.FIG2J_K

    # (a) ------------------------------------------------------------------------------
    title(0.1, TY1, "a", "Text embedding")
    ax = ax_in(0.05, 3.58, 2.95, 1.28); ax.set_xlim(0, 2.95); ax.set_ylim(0, 1.28); ax.axis("off")
    _embedding_schematic(ax, C, rng)

    # (b) ------------------------------------------------------------------------------
    title(3.12, TY1, "b", "Cosine similarity")
    ax = ax_in(3.2, 3.55, 1.3, 1.32); ax.set_aspect("equal")
    _cosine_geometry(ax, cA, cD, cB, A["label"], Dd["label"], B["label"])

    # (c) firm x rule matrix ---------------------------------------------------------------
    title(4.62, TY1, "c", r"Firm $\times$ rule similarity")
    Smat = np.array([[cos_of(D, fc, fy, rd) for rd, _ in C.FIG2C_RULES] for fc, fy, _ in C.FIG2C_FIRMS])
    ax = ax_in(5.8, 3.83, 1.05, 1.05)
    ax.imshow(np.nan_to_num(Smat), cmap="cividis", vmin=0, vmax=max(0.7, np.nanmax(Smat)))
    for (i, j), v in np.ndenumerate(Smat):
        ax.text(j, i, "\u2013" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=5.0,
                color="black" if v > 0.4 else "white")
    ax.set_yticks(range(len(C.FIG2C_FIRMS))); ax.set_yticklabels([n for *_, n in C.FIG2C_FIRMS], fontsize=5.6)
    ax.set_xticks(range(len(C.FIG2C_RULES)))
    ax.set_xticklabels([n for _, n in C.FIG2C_RULES], fontsize=5.3, rotation=40, ha="right", rotation_mode="anchor")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # (d) embedding map -------------------------------------------------------------------
    title(0.1, TY2, "d", "Embedding map (2-D projection)")
    ax = ax_in(0.2, 1.88, 1.75, 1.2)
    P = D.embedding_projection
    fr = P[P["source"].str.upper().str.startswith("FR")]
    if len(fr) > C.FIG2D_MAX_POINTS:
        fr = fr.sample(C.FIG2D_MAX_POINTS, random_state=3)
    prim = primary_domain(D).reindex(fr["doc_id"]).to_numpy()
    hl = {D.dom_idx[l]: S.domain_color(C, l, i) for i, l in enumerate(C.FIG2D_HIGHLIGHT) if l in D.dom_idx}
    grey = ~np.isin(prim, list(hl))
    ax.scatter(fr["x"].to_numpy()[grey], fr["y"].to_numpy()[grey], s=1.6, c="#C2C2C2", lw=0, alpha=0.7)
    for k, c in hl.items():
        m = prim == k
        ax.scatter(fr["x"].to_numpy()[m], fr["y"].to_numpy()[m], s=1.8, c=c, lw=0, alpha=0.85)
    span = (P["x"].max() - P["x"].min(), P["y"].max() - P["y"].min())
    for fc, fy, nm in C.FIG2D_FIRMS:
        row = P[P["doc_id"] == f"{fc}_{fy}"]
        if row.empty:
            print(f"[warning] firm {fc}_{fy} not found in embedding_projection.csv")
            continue
        px, py = float(row["x"].iloc[0]), float(row["y"].iloc[0])
        ax.scatter(px, py, s=38, marker="*", c=S.INK, ec="white", lw=0.5, zorder=5)
        ax.text(px + 0.022 * span[0], py + 0.015 * span[1], nm, fontsize=5.3, zorder=6,
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.75))
    ax.set_xticks([]); ax.set_yticks([]); ax.margins(0.03)
    ax.set_xlabel("dim. 1", fontsize=6); ax.set_ylabel("dim. 2", fontsize=6)

    # (e) similarity distribution -------------------------------------------------------------
    title(2.1, TY2, "e", "Similarity, one firm vs. all rules")
    ax = ax_in(2.47, 1.88, 1.3, 1.2)
    thr = np.sort(sims)[-min(K_TOP, len(sims))]
    ax.hist(sims, bins=np.linspace(0, xmax_cos, 61), color="#A6A6A6", ec="white", lw=0.15)
    ax.axvspan(thr, xmax_cos, color=S.BLUE, alpha=0.1, lw=0)
    ax.set_yscale("log"); ax.set_xlim(0, xmax_cos)
    ytop = ax.get_ylim()[1] * 1.8
    ax.set_ylim(0.8, ytop)
    for v, s, c in [(cA, A["label"], S.VERM), (cD, Dd["label"], S.DGOLD), (cB, B["label"], S.GRAY)]:
        if not np.isnan(v):
            ax.axvline(v, color=c, lw=0.7)
            ax.text(v, ytop * 1.1, s, color=c, fontsize=5.6, ha="center", va="bottom", fontweight="bold", clip_on=False)
    ax.text(xmax_cos * 0.935, ytop ** 0.68, f"top\n{K_TOP}", fontsize=5.2, color=S.BLUE, ha="center", va="center")
    ax.set_xlabel(rf"$\cos\theta_{{f,r}}$  ($n$ = {len(sims):,} rules)", fontsize=6.3); ax.set_ylabel("Rules (log)", fontsize=6.3)

    # (f) similarity by overlap level -----------------------------------------------------------
    title(3.85, TY2, "f", "Similarity by overlap level")
    ax = ax_in(4.2, 1.88, 1.15, 1.2)
    e1, e2 = C.FIG2F_CUTS
    groups = [E == 0, (E > 0) & (E < e1), (E >= e1) & (E < e2), E >= e2]
    bp = ax.boxplot([sims[m] if m.any() else [np.nan] for m in groups], widths=0.55, patch_artist=True, showfliers=False,
                    whis=(5, 95), medianprops=dict(color=S.INK, lw=0.9), whiskerprops=dict(lw=0.6),
                    capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
    for patch, c in zip(bp["boxes"], ["#E0E0E0", "#C6DBEF", "#6BAED6", S.BLUE]):
        patch.set_facecolor(c)
    ytop_f = xmax_cos * 0.95
    for i, m in enumerate(groups):
        ax.text(i + 1, ytop_f, f"{m.sum():,}", ha="center", fontsize=4.9, color=S.GRAY)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["0", f"1\u2013{e1 - 1}", f"{e1}\u2013{e2 - 1}", f"\u2265{e2}"], fontsize=5.6)
    ax.set_ylim(0, xmax_cos); ax.set_xlabel(r"$\mathrm{max}_d\,E_{f,r,d}$", fontsize=6.3)
    ax.set_ylabel(r"$\cos\theta_{f,r}$", fontsize=6.3)

    # (g) agreement with domain overlap ------------------------------------------------------------
    title(5.45, TY2, "g", "Agreement with overlap")
    ax = ax_in(5.83, 1.88, 1.07, 1.2)
    meaningful = (E >= C.FIG2G_THRESHOLD)[np.argsort(-sims)]
    ks = np.unique(np.round(np.logspace(0, np.log10(len(sims)), 120)).astype(int))
    ax.plot(ks, np.cumsum(meaningful)[ks - 1] / ks, color=S.BLUE, lw=1.1)
    ax.axhline(meaningful.mean(), color=S.GRAY, lw=0.7, ls="--")
    base = meaningful.mean()
    ax.text(1.3, base + 0.04 if base < 0.85 else base - 0.08, "all rules", fontsize=5.2, color=S.GRAY, ha="left")
    ax.axvline(K_TOP, color=S.LGRAY, lw=0.5, ls=":")
    ax.set_xscale("log"); ax.set_xlim(1, len(sims)); ax.set_ylim(0, 1.05)
    ax.set_xticks([t for t in (1, 10, 100, 1000, 10000, 100000) if t <= len(sims)])
    ax.set_xlabel("Top $k$ by similarity", fontsize=6.3)
    ax.set_ylabel(rf"Share with $E\geq{C.FIG2G_THRESHOLD}$", fontsize=6.3)

    # (h) similarity vs overlap -----------------------------------------------------------------------
    title(0.1, TY3, "h", "Similarity vs. domain overlap")
    ax = ax_in(0.47, 0.36, 1.55, 1.0)
    sel = rng.choice(len(E), min(C.FIG2H_MAX_POINTS, len(E)), replace=False)
    xj = E[sel] + np.where(E[sel] == 0, rng.uniform(-1.6, 1.6, sel.size), 0)
    ax.scatter(xj, sims[sel], s=1.8, c="#8F8F8F", alpha=0.3, lw=0)
    emax = max(E.max(), 1)
    edges = np.r_[0.5, np.arange(10, emax + 10, 10)]
    mx, my, ci = [0], [sims[E == 0].mean()], [1.96 * sims[E == 0].std() / np.sqrt(max((E == 0).sum(), 1))]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (E >= lo) & (E < hi)
        if m.sum() > 5:
            mx.append(E[m].mean()); my.append(sims[m].mean()); ci.append(1.96 * sims[m].std() / np.sqrt(m.sum()))
    mx, my, ci = map(np.array, (mx, my, ci))
    ax.fill_between(mx, my - ci, my + ci, color=S.INK, alpha=0.15, lw=0)
    ax.plot(mx, my, color=S.INK, lw=0.9, marker="s", ms=2, label="binned mean, 95% CI")
    for e, c, lab, col, tcol in [(eA, cA, A["label"], S.VERM, S.VERM), (eD, cD, Dd["label"], S.ORANGE, S.DGOLD),
                                  (eB, cB, B["label"], S.GRAY, S.GRAY)]:
        if np.isnan(e) or np.isnan(c):
            continue
        ax.scatter([e], [c], s=14, c=col, ec=S.INK, lw=0.4, zorder=5)
        hi = e > emax * 0.5
        ax.text(e - emax * 0.04 if hi else e + emax * 0.04, c if hi else c - 0.05, lab, fontsize=5.6,
                ha="right" if hi else "left", va="center", fontweight="bold", color=tcol)
    ax.set_xlim(-emax * 0.05, emax * 1.1); ax.set_ylim(0, xmax_cos)
    ax.set_xlabel(r"$\mathrm{max}_d\,E_{f,r,d}$", fontsize=6.3); ax.set_ylabel(r"$\cos\theta_{f,r}$", fontsize=6.3)
    ax.legend(loc="upper left", handlelength=1.1, borderaxespad=0.05, fontsize=5.4)

    # (i) ranking at equal overlap --------------------------------------------------------------------
    dom = D.dom_idx[C.FIG2I_DOMAIN]
    prod = Ed[:, dom]
    docs = pairs["document_number"].to_numpy()
    target = prod[docs == A["document_number"]][0]
    cand = np.where(np.abs(prod - target) <= C.FIG2I_TOL)[0]
    cand = cand[np.argsort(-sims[cand])]
    must = [i for i in cand if docs[i] in (A["document_number"], Dd["document_number"])]
    rest = [i for i in cand if i not in must]
    n_rest = max(C.FIG2I_N - len(must), 0)
    pick = [rest[int(round(j))] for j in np.linspace(0, len(rest) - 1, n_rest)] if rest and n_rest else []
    chosen = sorted(set(must + pick), key=lambda i: -sims[i])
    title(2.3, TY3, "i", f"Ranking at equal overlap ($E$ = {target:.0f})")
    ax = ax_in(3.5, 0.36, 0.92, 1.0)
    yy = np.arange(len(chosen))[::-1]
    labels = []
    for yv, i in zip(yy, chosen):
        doc = docs[i]
        t = str(C.RULE_SHORT_TITLES.get(doc, D.titles.get(doc, doc)))
        if len(t) > C.FIG2I_TITLE_CHARS:   # cut at a word boundary
            t = t[:C.FIG2I_TITLE_CHARS - 1].rsplit(" ", 1)[0].rstrip(" ,;:-") + "\u2026"
        if doc == A["document_number"]:
            t, c = f"{t} ({A['label']})", S.VERM
        elif doc == Dd["document_number"]:
            t, c = f"{t} ({Dd['label']})", S.ORANGE
        else:
            c = "#8C8C8C"
        labels.append((t, c))
        ax.plot([0.2, sims[i]], [yv, yv], color=c, lw=0.8)
        ax.plot(sims[i], yv, "o", color=c, ms=2.8, mec=S.INK, mew=0.3)
    ax.set_yticks(yy); ax.set_yticklabels([t for t, _ in labels], fontsize=5.4)
    for tick, (t, c) in zip(ax.get_yticklabels(), labels):
        if c != "#8C8C8C":
            tick.set_color(S.VERM if c == S.VERM else S.DGOLD); tick.set_fontweight("bold")
    lo = min(0.2, sims[chosen].min() - 0.03) if chosen else 0.2
    ax.set_xlim(lo, min(xmax_cos, sims[chosen].max() + 0.05) if chosen else xmax_cos)
    ax.set_xlabel(r"$\cos\theta_{f,r}$", fontsize=6.3); ax.tick_params(axis="y", length=0)

    # (j) Venn: top-k by each signal --------------------------------------------------------------------
    title(4.72, TY3, "j", f"Top-{K_TOP} rules by each signal")
    ax = ax_in(4.85, 0.3, 2.05, 1.08); ax.set_xlim(0, 2.05); ax.set_ylim(0, 1.08); ax.set_aspect("equal"); ax.axis("off")
    tie = np.random.default_rng(0).random(len(E))
    top_sim = set(np.argsort(-sims)[:K_TOP]); top_ov = set(np.lexsort((tie, -E))[:K_TOP])
    both, only_ov, only_sim = len(top_sim & top_ov), len(top_ov - top_sim), len(top_sim - top_ov)
    ax.add_patch(Circle((0.78, 0.5), 0.42, fc=S.ORANGE, alpha=0.28, ec=S.DGOLD, lw=0.8))
    ax.add_patch(Circle((1.27, 0.5), 0.42, fc=S.BLUE, alpha=0.22, ec=S.BLUE, lw=0.8))
    for xx, v in ((0.58, only_ov), (1.025, both), (1.47, only_sim)):
        ax.text(xx, 0.5, f"{v}", ha="center", va="center", fontsize=7, fontweight="bold")
    ax.text(0.62, 0.99, "domain overlap", ha="center", va="center", fontsize=5.6, color=S.DGOLD)
    ax.text(1.43, 0.99, "similarity", ha="center", va="center", fontsize=5.6, color=S.BLUE)
    ax.text(1.025, 0.02, "both", ha="center", va="center", fontsize=5.4, color=S.GRAY)

    if note:
        S.footnote(fig, note)
    return S.save(fig, out_dir, "Figure2_text_embeddings", formats)
