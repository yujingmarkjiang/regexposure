"""Shared visual style for the regulatory-exposure figures.

All figures use a 7.0-inch (178 mm) double-column width, 6-7.5 pt type,
the Okabe-Ito colour-blind-safe palette, and embedded TrueType fonts so
text stays editable in Illustrator / Inkscape.
"""
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

# Okabe-Ito palette
BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
DGOLD = "#B07800"          # darker orange, for text on white
INK = "#1A1A1A"
GRAY = "#7F7F7F"
LGRAY = "#BDBDBD"
HIGHLIGHT_CYCLE = [BLUE, GREEN, ORANGE, VERM, PURPLE, SKY, YELLOW]


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2, "legend.frameon": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.labelpad": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "stixsans", "lines.linewidth": 1.0,
    })


def new_figure(width, height):
    """Return (fig, ax_in, title).

    ax_in(x, y, w, h) adds an axes positioned in inches from the bottom-left corner.
    title(x, y, letter, text) writes a bold panel letter followed by a regular title.
    """
    fig = plt.figure(figsize=(width, height))

    def ax_in(x, y, w, h, **kw):
        return fig.add_axes([x / width, y / height, w / width, h / height], **kw)

    def title(x, y, letter, text):
        fig.text(x / width, y / height, f"({letter})", fontsize=7.5, fontweight="bold", ha="left", va="baseline")
        fig.text((x + 0.21) / width, y / height, text, fontsize=7.5, ha="left", va="baseline")

    return fig, ax_in, title


def footnote(fig, text):
    w, h = fig.get_size_inches()
    fig.text((w - 0.05) / w, 0.03 / h, text, fontsize=5.3, color=GRAY, ha="right")


def save(fig, out_dir, name, formats=("pdf", "svg", "png"), dpi=600):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in formats:
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=dpi if ext in ("png", "tif", "tiff", "jpg") else None)
        paths.append(p)
    plt.close(fig)
    return paths


def arrow(ax, x0, y0, x1, y1, color=INK, lw=0.7, ms=6, style="-|>", **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, **kw))


def doc_icon(ax, x, y, w, h, col, fill="white"):
    """Small document glyph with a folded corner and text lines."""
    f = 0.22 * w
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - f), (x + w - f, y + h), (x, y + h)],
                         closed=True, fc=fill, ec=col, lw=0.7))
    ax.plot([x + w - f, x + w - f, x + w], [y + h, y + h - f, y + h - f], color=col, lw=0.6)
    for k in range(4):
        yy = y + h - f - 0.045 - k * 0.045
        ax.plot([x + 0.05, x + w - (0.09 if k == 3 else 0.05)], [yy, yy], color=col, lw=0.6, alpha=0.6)


def domain_color(C, label, i=0):
    """Colour for a highlighted domain: config.DOMAIN_COLORS first, then the palette cycle."""
    return getattr(C, "DOMAIN_COLORS", {}).get(label, HIGHLIGHT_CYCLE[i % len(HIGHLIGHT_CYCLE)])
