"""Figure 1: measurement of firm-level regulatory exposure (panels a-f)."""
import numpy as np
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from . import style as S
from .data import cos_of, firm_vector, rule_vector


def _pipeline(ax, D, C, s_f, g_a, series):
    """Panel (a): five illustrated stage boxes."""
    w, y0, h = 1.2, 0.02, 1.33
    g = (6.7 - 5 * w) / 4
    heads = ["Corpora", "Domain discovery", "Domain scoring", "Matching", "Aggregation"]
    foots = [["rules $r$;  firm-years $(f,t)$"],
             [f"$K$ = {D.K} clusters", f"labels: {C.LABEL_MODEL}"],
             [C.SCORING_MODEL, r"$s_{f,d},\ g_{r,d}\in[0,100]$"],
             [r"$E_{f,r,d}=s_{f,d}\,g_{r,d}\,/\,100$"],
             [r"$E_{f,t,d}=\Sigma_{r\in R_t}\,E_{f,r,d}$"]]
    rng = np.random.default_rng(2)
    for i in range(5):
        x = i * (w + g)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc="#F7F7F7", ec="#C4C4C4", lw=0.6))
        ax.add_patch(Circle((x + 0.14, y0 + h - 0.15), 0.075, fc=S.INK, ec="none"))
        ax.text(x + 0.14, y0 + h - 0.153, str(i + 1), color="white", fontsize=6.3, fontweight="bold", ha="center", va="center")
        ax.text(x + 0.26, y0 + h - 0.15, heads[i], fontsize=6.7, fontweight="bold", ha="left", va="center")
        ax.add_patch(Rectangle((x + 0.08, y0 + 0.07), w - 0.16, 0.31, fc="white", ec="none"))
        fl = foots[i]
        for j, line in enumerate(fl):
            yy = y0 + 0.225 if len(fl) == 1 else y0 + 0.295 - j * 0.14
            math = line.startswith("$") and line.endswith("$")
            ax.text(x + w / 2, yy, line, fontsize=6.6 if math else 5.9, ha="center", va="center",
                    fontweight="bold" if line in (C.SCORING_MODEL,) else "normal")
        gx, gy, gw, gh = x + 0.1, y0 + 0.44, w - 0.2, 0.62
        if i < 4:
            S.arrow(ax, x + w + 0.02, y0 + h / 2 + 0.05, x + w + g - 0.02, y0 + h / 2 + 0.05, color="#8C8C8C", lw=1.2, ms=8)
        if i == 0:
            for k in range(3):
                S.doc_icon(ax, gx + 0.08 + 0.05 * k, gy + 0.22 - 0.05 * k, 0.28, 0.36, S.VERM)
                S.doc_icon(ax, gx + 0.56 + 0.05 * k, gy + 0.22 - 0.05 * k, 0.28, 0.36, S.BLUE)
            ax.text(gx + 0.26, gy + 0.03, "FR rules", fontsize=5.6, ha="center", color=S.VERM)
            ax.text(gx + 0.76, gy + 0.03, "10-K (1, 1A)", fontsize=5.6, ha="center", color=S.BLUE)
        elif i == 1:  # schematic cluster glyph (illustrative)
            ax.add_patch(Rectangle((gx, gy), gw, gh, fc="white", ec="#D0D0D0", lw=0.5))
            cols = S.HIGHLIGHT_CYCLE[:6] + [S.LGRAY] * 6
            cen = rng.uniform([gx + 0.1, gy + 0.1], [gx + gw - 0.1, gy + gh - 0.1], (12, 2))
            for (cx, cy), c in zip(cen, cols):
                p = rng.normal([cx, cy], [0.045, 0.03], (14, 2))
                p[:, 0] = np.clip(p[:, 0], gx + 0.02, gx + gw - 0.02)
                p[:, 1] = np.clip(p[:, 1], gy + 0.02, gy + gh - 0.02)
                ax.scatter(p[:, 0], p[:, 1], s=1.4, c=c, lw=0)
        elif i == 2:  # score profiles of the focal firm and rule A
            bw = gw / D.K
            for row, (vals, c, lab) in enumerate([(s_f, S.BLUE, "$s_f$"), (g_a, S.VERM, "$g_r$")]):
                by = gy + gh - 0.3 - row * 0.34
                ax.plot([gx + 0.17, gx + gw], [by, by], color="#9E9E9E", lw=0.4)
                for d, v in enumerate(vals):
                    if v:
                        ax.add_patch(Rectangle((gx + 0.17 + d * bw * 0.83, by), bw * 0.83 * 0.75, v / 100 * 0.26, fc=c, ec="none"))
                ax.text(gx + 0.08, by + 0.08, lab, fontsize=6.2, ha="center", va="center", color=c)
        elif i == 3:  # element-wise product strips
            cw = (gw - 0.2) / D.K
            prod = s_f * g_a / 100
            for row, (vals, cm, lab) in enumerate([(s_f, "Blues", "$s_f$"), (g_a, "Reds", "$g_r$"), (prod, "Oranges", "$E$")]):
                ry = gy + gh - 0.17 - row * 0.24
                cmap = mpl.colormaps[cm]
                for d, v in enumerate(vals):
                    ax.add_patch(Rectangle((gx + 0.2 + d * cw, ry), cw, 0.13, fc=cmap(0.06 + 0.9 * v / 100), ec="white", lw=0.25))
                ax.text(gx + 0.08, ry + 0.065, lab, fontsize=6.2, ha="center", va="center")
                if row:
                    ax.text(gx + 0.2 + D.K / 2 * cw, ry + 0.185, r"$\odot$" if row == 1 else "$=$", fontsize=5.5, ha="center", va="center", color=S.GRAY)
            for d in np.where(prod > 0)[0]:
                ax.add_patch(Rectangle((gx + 0.2 + d * cw, gy + gh - 0.17 - 2 * 0.24), cw, 0.13, fill=False, ec=S.INK, lw=0.4))
        elif i == 4:  # mini time series of the focal firm's exposure
            ax.plot([gx + 0.05, gx + 0.05, gx + gw], [gy + gh, gy + 0.05, gy + 0.05], color=S.GRAY, lw=0.5)
            top = max(np.nanmax(v) for _, v, _ in series) if series else 1
            for _, v, c in series[:3]:
                xs = np.linspace(gx + 0.08, gx + gw - 0.02, len(v))
                ax.plot(xs, gy + 0.07 + v / top * (gh - 0.12), color=c, lw=0.8)


def make_figure1(D, C, out_dir, formats, note=None):
    S.apply_style()
    W, H = 7.0, 7.2
    fig, ax_in, title = S.new_figure(W, H)
    cik, year = C.FOCAL_FIRM["cik"], C.FOCAL_FIRM["year"]
    s_f = firm_vector(D, cik, year)
    g_a = rule_vector(D, C.RULE_A["document_number"])
    g_b = rule_vector(D, C.RULE_B["document_number"])

    # firm-year exposure series (also used by the panel-(a) glyph)
    fy = D.firm_year_exposure[(D.firm_year_exposure["cik"] == cik)]
    piv = fy.pivot_table(index="year", columns="domain", values="exposure", aggfunc="sum").sort_index()
    if len(piv):   # years with no scored 10-K show as gaps, not as zeros joined by a line
        piv = piv.reindex(range(int(piv.index.min()), int(piv.index.max()) + 1))
    doms = C.FIG1E_DOMAINS or piv.mean().sort_values(ascending=False).index[:4].tolist()
    cols = [S.BLUE, S.ORANGE, S.GREEN, S.PURPLE, S.SKY, S.VERM]
    series = [(d, piv[d].to_numpy(float), cols[i % len(cols)]) for i, d in enumerate(doms) if d in piv]

    # (a) ---------------------------------------------------------------------------
    title(0.1, 7.02, "a", "Measurement pipeline")
    ax = ax_in(0.1, 5.5, 6.7, 1.38); ax.set_xlim(0, 6.7); ax.set_ylim(0, 1.38); ax.axis("off")
    _pipeline(ax, D, C, s_f, g_a, series)

    # (b) domain discovery ---------------------------------------------------------------
    title(0.1, 5.22, "b", f"Domain discovery ($K$ = {D.K})")
    ax = ax_in(0.32, 2.55, 2.5, 2.45)
    P = D.embedding_projection
    if len(P) > C.FIG1B_MAX_POINTS:
        P = P.sample(C.FIG1B_MAX_POINTS, random_state=1)
    hl = {D.dom_idx[l]: S.domain_color(C, l, i) for i, l in enumerate(C.FIG1B_HIGHLIGHT) if l in D.dom_idx}
    is_fr = P["source"].str.upper().str.startswith("FR")
    for k in sorted(P["cluster"].unique()):
        m = P["cluster"] == k
        col, a = (hl[k], 0.8) if k in hl else ("#B5B5B5", 0.5)
        ax.scatter(P.loc[m & is_fr, "x"], P.loc[m & is_fr, "y"], s=2.8, marker="o", c=col, alpha=a, lw=0)
        ax.scatter(P.loc[m & ~is_fr, "x"], P.loc[m & ~is_fr, "y"], s=4.0, marker="^", c=col, alpha=a, lw=0)
    x0, x1 = P["x"].min(), P["x"].max(); y0, y1 = P["y"].min(), P["y"].max()
    dx, dy = x1 - x0, y1 - y0
    ax.set_xlim(x0 - 0.03 * dx, x1 + 0.03 * dx); ax.set_ylim(y0 - 0.04 * dy, y1 + 0.1 * dy)
    for k, col in hl.items():
        sub = D.embedding_projection[D.embedding_projection["cluster"] == k]
        top, bot = np.percentile(sub["y"], 97), np.percentile(sub["y"], 3)
        above = top + 0.08 * dy < y1 + 0.1 * dy
        ax.text(np.clip(sub["x"].median(), x0 + 0.12 * dx, x1 - 0.12 * dx), top + 0.02 * dy if above else bot - 0.02 * dy,
                D.domains[k], fontsize=5.7, ha="center", va="bottom" if above else "top",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=0.6, alpha=0.92))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Embedding projection, dim. 1"); ax.set_ylabel("Embedding projection, dim. 2")
    hnd = [Line2D([], [], marker="o", ls="", color=S.INK, ms=3, label="FR rule"),
           Line2D([], [], marker="^", ls="", color=S.INK, ms=3.5, label="10-K section"),
           Line2D([], [], marker="o", ls="", color="#B5B5B5", ms=3.5, label=f"other {D.K - len(hl)} clusters")]
    ax.legend(handles=hnd, loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=3, handletextpad=0.2, columnspacing=0.8)

    # (c) domain-score heat map -------------------------------------------------------------
    title(3.0, 5.22, "c", f"Domain scores ({C.SCORING_MODEL})")
    M = np.c_[s_f, g_a, g_b]
    hi_row = int(np.argmax(s_f * g_a))
    axc = ax_in(4.06, 2.88, 1.2, 2.06)
    axc.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    fmt = (lambda v: f"{v:.0f}") if np.allclose(M, M.round()) else (lambda v: f"{v:.1f}")
    for (i, j), v in np.ndenumerate(M):
        if v:
            axc.text(j, i, fmt(v), ha="center", va="center", fontsize=5.6, color="white" if v > 55 else S.INK)
    axc.set_yticks(range(D.K)); axc.set_yticklabels(D.short, fontsize=5.9)
    axc.set_xticks(range(3)); axc.set_xticklabels(C.FIG1C_COLUMNS, fontsize=5.9)
    axc.xaxis.tick_top(); axc.tick_params(length=0)
    for sp in axc.spines.values():
        sp.set_visible(False)
    axc.set_xticks(np.arange(-0.5, 3), minor=True); axc.set_yticks(np.arange(-0.5, D.K), minor=True)
    axc.grid(which="minor", color="white", lw=0.6); axc.tick_params(which="minor", length=0)
    axc.add_patch(Rectangle((-0.5, hi_row - 0.5), 3, 1, fill=False, ec=S.INK, lw=0.8))

    # (d) exposure heat map -------------------------------------------------------------------
    title(5.4, 5.22, "d", r"Exposure $E_{f,r,d}$")
    Ex = np.c_[s_f * g_a, s_f * g_b] / 100
    axd = ax_in(5.42, 2.88, 0.8, 2.06)
    axd.imshow(Ex, cmap="Oranges", vmin=0, vmax=100, aspect="auto")
    for (i, j), v in np.ndenumerate(Ex):
        if v:
            axd.text(j, i, f"{v:.0f}" if v >= 1 else f"{v:.1f}", ha="center", va="center", fontsize=5.6,
                     color="white" if v > 55 else S.INK)
    axd.set_yticks([]); axd.set_xticks(range(2))
    axd.set_xticklabels([rf"$f\times${C.RULE_A['label']}", rf"$f\times${C.RULE_B['label']}"], fontsize=5.9)
    axd.xaxis.tick_top(); axd.tick_params(length=0)
    for sp in axd.spines.values():
        sp.set_visible(False)
    axd.set_xticks(np.arange(-0.5, 2), minor=True); axd.set_yticks(np.arange(-0.5, D.K), minor=True)
    axd.grid(which="minor", color="white", lw=0.6); axd.tick_params(which="minor", length=0)
    axd.add_patch(Rectangle((-0.5, hi_row - 0.5), 2, 1, fill=False, ec=S.INK, lw=0.8))
    coss = [cos_of(D, cik, year, C.RULE_A["document_number"]), cos_of(D, cik, year, C.RULE_B["document_number"])]
    yb = D.K - 0.5
    for j, (sv, cv) in enumerate(zip(Ex.sum(0), coss)):
        axd.text(j, yb + 0.85, f"{sv:.1f}", ha="center", va="top", fontsize=6, fontweight="bold")
        axd.text(j, yb + 0.85 + D.K * 0.06, "\u2013" if np.isnan(cv) else f"{cv:.2f}", ha="center", va="top", fontsize=6)
    axd.text(1.62, yb + 0.85, r"$\Sigma_d E_{f,r,d}$", ha="left", va="top", fontsize=6)
    axd.text(1.62, yb + 0.85 + D.K * 0.06, r"$\cos\theta_{f,r}$", ha="left", va="top", fontsize=6)
    for cm, lab, x0, wd in [("Blues", "Score", 4.12, 1.08), ("Oranges", "Exposure", 5.42, 0.8)]:
        cax = ax_in(x0, 2.42, wd, 0.06)
        cb = fig.colorbar(mpl.cm.ScalarMappable(Normalize(0, 100), cm), cax=cax, orientation="horizontal", ticks=[0, 50, 100])
        cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.6, length=1.5, width=0.4, pad=1)
        cb.set_label(lab, fontsize=5.8, labelpad=1)

    # (e) firm-year exposure by domain ---------------------------------------------------------
    title(0.1, 1.98, "e", "Firm-year exposure by domain")
    ax = ax_in(0.55, 0.42, 3.55, 1.38)
    for d, v, c in series:
        ax.plot(piv.index, v, color=c, lw=1.1, marker="o", ms=2, label=d)
    ymax = max(np.nanmax(v) for _, v, _ in series) * 1.32
    ax.axvline(C.FIG1F_YEAR, color=S.GRAY, lw=0.6, ls=":")
    ax.text(C.FIG1F_YEAR + 0.2, ymax * 0.8, "panel (f)", fontsize=5.6, color=S.GRAY, va="top")
    ax.set_xlim(piv.index.min() - 0.5, piv.index.max() + 0.5); ax.set_ylim(0, ymax)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel(r"Exposure $E_{f,t,d}$"); ax.set_xlabel("Year $t$")
    ax.legend(loc="upper left", ncol=2, handlelength=1.4, columnspacing=1.0, borderaxespad=0.1)

    # (f) by issuing agency ------------------------------------------------------------------------
    title(4.45, 1.98, "f", f"By issuing agency, $t$ = {C.FIG1F_YEAR}")
    ax = ax_in(5.05, 0.42, 1.65, 1.38)
    ag = D.firm_year_agency_exposure
    ag = (ag[(ag["cik"] == cik) & (ag["year"] == C.FIG1F_YEAR)].groupby("agency")["exposure"].sum()
          .sort_values(ascending=False).head(C.FIG1F_TOP))
    yy = np.arange(len(ag))[::-1]
    ax.barh(yy, ag.values, color=S.BLUE, height=0.62, alpha=0.9)
    xmax = ag.max() * 1.25
    for yv, v in zip(yy, ag.values):
        ax.text(v + xmax * 0.02, yv, f"{v:,.0f}", va="center", fontsize=5.8)
    ax.set_yticks(yy); ax.set_yticklabels(ag.index)
    ax.set_xlim(0, xmax); ax.set_xlabel(r"Exposure $E_{f,t,a}$"); ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    if note:
        S.footnote(fig, note)
    return S.save(fig, out_dir, "Figure1_exposure_measurement", formats)
