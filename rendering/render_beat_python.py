"""
Python reimplementation of the single-beat renderer (colour and line arms).

Matches render_study.R exactly: same 1.6 s window, same 40 ms / 200 ms grid,
same y limits, same 448 x 448 output, same nine-level palette anchored at the
rate-correction threshold.

Inputs per recording:
    signal CSV  (QT_coloured_ECG_signal_v3/<name>_.csv)
        time, mv, newcolourVector_nom / _qtcb / _qtcf
    Rpeaks CSV  (QT_coloured_ECG_signal_v3/Rpeaks/<name>_.csv)
        R_peaks_s, anchor_nom / anchor_qtcb / anchor_qtcf

`mv` is already the fully preprocessed signal (baseline removal, wavelet
denoise, ecg_clean, median-filter subtraction) -- the same trace detection ran
on. Nothing further is filtered here.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

# ---- geometry: identical to render_study.R ----------------------------------
FIXED_WINDOW = 1.6        # s, viewport length
PAD          = 0.04       # s before the first R peak
Y_LIM        = (-0.2, 1.0)  # mV, fixed for every image
GRID_MIN     = 0.04       # s, minor gridlines (40 ms)
GRID_MAJ     = 0.20       # s, major gridlines (200 ms)
GRID_Y       = 0.1        # mV, horizontal minor gridlines
SHOW_GRID    = True

IMG_IN   = 2.24           # 2.24 in at 200 dpi = 448 x 448
IMG_DPI  = 200

BAND        = 40.0        # ms per colour level
BANDS_BELOW = 6           # level 1 sits 6 bands below the anchor
BANDS_ABOVE = 2           # level 9 sits 2 bands above (1 when HR >= 85)

# ggplot2 linewidths are in mm; matplotlib's are in points. 1 mm = 2.845 pt.
MM_TO_PT  = 72.27 / 25.4
LW_TRACE  = 0.7  * MM_TO_PT     # geom_line(linewidth = 0.7)
LW_MAJOR  = 0.5  * MM_TO_PT     # theme_grey panel.grid.major
LW_MINOR  = 0.25 * MM_TO_PT     # theme_grey panel.grid.minor
PANEL_BG  = "#EBEBEB"           # theme_grey panel background ("grey92")

# ---- palette: rev(RColorBrewer::brewer.pal(11, "Spectral")) ------------------
SPECTRAL_11 = ["#9E0142", "#D53E4F", "#F46D43", "#FDAE61", "#FEE08B", "#FFFFBF",
               "#E6F598", "#ABDDA4", "#66C2A5", "#3288BD", "#5E4FA2"]
PALETTE = LinearSegmentedColormap.from_list("spectral_r", SPECTRAL_11[::-1], N=256)


def fill_limits(anchor, hr):
    """The gradient limits R uses: [anchor - 240 + 1, anchor + 80 - 1].

    The +1/-1 reproduces scale_fill_gradientn(limits = c(v[1]+1, v[9]-1)).
    At HR >= 85 the scale is truncated to anchor + 40, i.e. levels 1-8.
    """
    bands_above = 1 if hr >= 85 else BANDS_ABOVE
    lo = anchor - BAND * BANDS_BELOW + 1.0
    hi = anchor + BAND * bands_above - 1.0
    return lo, hi


# =============================================================================
#  renderer
# =============================================================================
def render_beat(dat, outfile, style, t0, rr, anchor=None, hr=None,
                colvar="newcolourVector_nom"):
    """One beat in a fixed 1.6 s window.

    style: 'colour' (banded fill), 'grey' (flat fill), 'line' (trace only)

    The data slice runs from 40 ms before the first R peak to 40 ms past the
    NEXT R peak, so each image holds one beat plus the following R peak; the
    viewport then extends to 1.6 s, and the blank space that leaves is what
    makes heart rate geometric.
    """
    lo = t0 - PAD
    hi = t0 + rr + PAD
    d = dat[(dat["time"] >= lo) & (dat["time"] <= hi)]
    if len(d) == 0:
        return False

    t = d["time"].to_numpy()
    y = d["mv"].to_numpy()

    fig, ax = plt.subplots(figsize=(IMG_IN, IMG_IN), dpi=IMG_DPI)
    ax.set_xlim(lo, lo + FIXED_WINDOW)      # coord_cartesian + expand = c(0, 0)
    ax.set_ylim(*Y_LIM)
    ax.set_facecolor(PANEL_BG if SHOW_GRID else "white")

    if SHOW_GRID:
        # grid lines sit on absolute time, as seq(0, 20, 0.04) does in R
        g0 = np.floor(lo / GRID_MIN) * GRID_MIN
        for g in np.arange(g0, lo + FIXED_WINDOW + GRID_MIN, GRID_MIN):
            ax.axvline(g, color="white", lw=LW_MINOR, zorder=0)
        g0 = np.floor(lo / GRID_MAJ) * GRID_MAJ
        for g in np.arange(g0, lo + FIXED_WINDOW + GRID_MAJ, GRID_MAJ):
            ax.axvline(g, color="white", lw=LW_MAJOR, zorder=0)
        for g in np.arange(Y_LIM[0], Y_LIM[1] + GRID_Y, GRID_Y):
            ax.axhline(g, color="white", lw=LW_MINOR, zorder=0)

    if style == "colour":
        # geom_ridgeline_gradient(y = 0, height = mv): fill between the
        # baseline and the trace, coloured by newcolourVector (ms from Q
        # onset). Samples outside the gradient limits are NA in R and are not
        # drawn; the same samples are already NaN in the signal CSV.
        if anchor is None or hr is None:
            raise ValueError("colour style needs anchor and hr")
        cv = d[colvar].to_numpy(dtype=float)
        glo, ghi = fill_limits(anchor, hr)
        norm = (cv - glo) / (ghi - glo)

        keep = np.where(np.isfinite(norm[:-1]) & (norm[:-1] >= 0) & (norm[:-1] <= 1))[0]
        if len(keep):
            quads = np.stack([
                np.column_stack([t[keep],     np.zeros(len(keep))]),
                np.column_stack([t[keep + 1], np.zeros(len(keep))]),
                np.column_stack([t[keep + 1], y[keep + 1]]),
                np.column_stack([t[keep],     y[keep]]),
            ], axis=1)
            ax.add_collection(PolyCollection(
                quads, facecolors=PALETTE(norm[keep]),
                edgecolors="none", linewidths=0, zorder=1))

    elif style == "grey":
        # geom_ribbon over the whole visible trace -- note this is a WIDER
        # region than the colour fill, which is clipped to the anchor window
        ax.fill_between(t, 0, y, color="#999999", lw=0, zorder=1)

    elif style != "line":
        raise ValueError(f"unknown style: {style}")

    ax.plot(t, y, color="black", lw=LW_TRACE, zorder=2, solid_capstyle="butt")

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(outfile, dpi=IMG_DPI, facecolor="white")
    plt.close(fig)
    return True


# =============================================================================
#  batch
# =============================================================================
BASE      = "/content/drive/MyDrive/Colab Notebooks/2026/ECGRDVQ"
SIGNAL_DIR = os.path.join(BASE, "QT_coloured_ECG_signal_v3")
RPEAK_DIR  = os.path.join(SIGNAL_DIR, "Rpeaks")
OUT_ROOT   = os.path.join(BASE, "images", "study_py")

STYLES = {                       # cell name -> (style, colour column, anchor)
    "beat_colour": ("colour", "newcolourVector_nom",  "anchor_nom"),
    "beat_line":   ("line",   None,                   None),
    # "beat_grey":  ("grey",   None,                   None),
}

LEAD_NUM = re.compile(r"^(\d+)")


def run(limit=None):
    for cell in STYLES:
        Path(os.path.join(OUT_ROOT, cell)).mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in os.listdir(SIGNAL_DIR) if f.endswith(".csv"))
    if limit:
        files = files[:limit]
    print(f"{len(files)} signal CSVs")

    n_done = n_clip = 0
    for i, fname in enumerate(files, 1):
        m = LEAD_NUM.match(fname)
        if not m:
            continue
        rec = int(m.group(1))
        png = f"{rec}.png"

        todo = [c for c in STYLES
                if not os.path.exists(os.path.join(OUT_ROOT, c, png))]
        if not todo:
            continue

        rp_path = os.path.join(RPEAK_DIR, fname)
        if not os.path.exists(rp_path):
            continue
        rp = pd.read_csv(rp_path)
        rpk = rp["R_peaks_s"].to_numpy()
        if len(rpk) < 2:
            continue

        t0 = rpk[0]
        rr = rpk[1] - rpk[0]
        hr = float(rp["HR"].iloc[0])
        if rr + 2 * PAD > FIXED_WINDOW:
            n_clip += 1
            print(f"  !! {rec}: beat-1 RR {rr:.3f} s exceeds the "
                  f"{FIXED_WINDOW:.2f} s window")

        dat = pd.read_csv(os.path.join(SIGNAL_DIR, fname))

        for cell in todo:
            style, colvar, anchor_col = STYLES[cell]
            anchor = float(rp[anchor_col].iloc[0]) if anchor_col else None
            render_beat(dat, os.path.join(OUT_ROOT, cell, png),
                        style, t0, rr, anchor=anchor, hr=hr,
                        colvar=colvar or "newcolourVector_nom")
        n_done += 1

        if i % 200 == 0:
            print(f"[{i}/{len(files)}] {n_done} rendered")

    print(f"\ndone: {n_done} recordings | {n_clip} exceeded the window")
    for cell in STYLES:
        d = os.path.join(OUT_ROOT, cell)
        print(f"  {cell:14s} {len(os.listdir(d))}")


if __name__ == "__main__":
    run()
