"""Shared visual style for the demo video + dark figures.

Registers the bundled fonts (Space Grotesk for headings, JetBrains Mono for mono/body)
and exposes the black terminal/product palette. `apply_dark_rc()` styles matplotlib
plots to read clearly on a black slide. Falls back to default fonts if the TTFs are
missing, so the build never hard-fails.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager as fm

REPO = Path(__file__).resolve().parent.parent
_FONTS = REPO / "assets" / "fonts"

for _ttf in (_FONTS.glob("*.ttf") if _FONTS.exists() else []):
    if _ttf.stem != "SpaceGrotesk-VF":  # skip the variable source
        try:
            fm.fontManager.addfont(str(_ttf))
        except Exception:  # noqa: BLE001
            pass

_FAMILIES = {f.name for f in fm.fontManager.ttflist}
HEAD = "Space Grotesk" if "Space Grotesk" in _FAMILIES else "DejaVu Sans"
MONO = "JetBrains Mono" if "JetBrains Mono" in _FAMILIES else "DejaVu Sans Mono"

# Palette (matches the slides).
BG = "#0A0A0B"
FG = "#FFFFFF"
MUTED = "#9AA0AA"
GRID = "#24242B"
GREEN = "#2BE06B"
MAGENTA = "#FF3DA6"
BLUE = "#5AA9FF"
ORANGE = "#FFB454"
VIOLET = "#C792EA"
TEAL = "#5BE3C9"
SERIES = [BLUE, GREEN, MAGENTA, ORANGE, VIOLET, TEAL]


def apply_dark_rc():
    """Style matplotlib for figures that sit directly on a black slide."""
    mpl.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": MUTED, "axes.titlecolor": FG,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "grid.color": GRID, "grid.alpha": 0.6, "axes.grid": True, "axes.axisbelow": True,
        "font.family": MONO, "font.size": 13,
        "axes.titlesize": 15, "axes.titleweight": "bold", "figure.titleweight": "bold",
        "legend.facecolor": "#141417", "legend.edgecolor": GRID, "legend.framealpha": 0.95,
        "legend.fontsize": 9,
    })
