#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure: generic edit-sensitivity is a shrinkage that depends on the model state.

Each panel is one model. The x-axis is the logit on the unedited image, the y-axis is the logit change
caused by a neutral edit. The four neutral edit types are overlaid, each with its own regression line.
If an edit only added a fixed amount, the lines would be horizontal. They slope downward, and the
slope differs by edit type: that is exactly the term that placebo subtraction cannot remove.

Reads results/c19_preds.npz (cached), so no GPU is needed.
Out -> paper/figures/fig_shrinkage.{png,tiff}
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from p2_source_guard import require_p2_source          # noqa: E402
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "paper", "figures")
DPI = 300
EPS = 1e-6
PLACEBO = [("pb:flip", "flip"), ("pb:jitter", "spatial jitter"),
           ("pb:hue", "hue"), ("pb:noise", "noise")]
COL = {"flip": "#4C72B0", "spatial jitter": "#55A868", "hue": "#C44E52", "noise": "#8172B2"}
MODELS = [("erm", "ERM"), ("slim", "Slim"), ("causal", "Inv")]


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def main():
    require_p2_source(BASE, "fig_shrinkage", "make_fig_shrinkage.py")
    d = np.load(os.path.join(RES, "c19_preds.npz"), allow_pickle=True)
    names = [str(x) for x in d["names"]]
    os.makedirs(FIG, exist_ok=True)

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8,
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)

    for ax, (key, lab) in zip(axes, MODELS):
        L0 = logit(d[f"p0_{key}"])
        pe = d[f"pe_{key}"]
        for col, nm in PLACEBO:
            if col not in names:
                continue
            dL = logit(pe[:, names.index(col)]) - L0
            ax.scatter(L0, dL, s=4, alpha=0.28, color=COL[nm], linewidths=0, rasterized=True)
            a, b = np.polyfit(L0, dL, 1)
            xs = np.linspace(L0.min(), L0.max(), 50)
            r2 = 1 - np.var(dL - (a * L0 + b)) / np.var(dL)
            ax.plot(xs, a * xs + b, color=COL[nm], lw=1.4,
                    label=f"{nm}  slope {a:+.2f} ($R^2$ {r2:.2f})")
        ax.axhline(0, color="0.55", lw=0.7, ls="--", zorder=0)
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("logit at the unedited image")
        ax.legend(fontsize=5.6, frameon=False, loc="lower left", handlelength=1.2)
    axes[0].set_ylabel("change in logit")
    fig.tight_layout(pad=0.5)

    png = os.path.join(FIG, "fig_shrinkage.png")
    tif = os.path.join(FIG, "fig_shrinkage.tiff")
    fig.savefig(png, dpi=DPI, facecolor="white")
    fig.savefig(tif, dpi=DPI, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    w, h = fig.get_size_inches()
    print(f"saved -> {png} and {tif}   {w:.1f} x {h:.1f} in @ {DPI} dpi "
          f"({int(w*DPI)} x {int(h*DPI)} px)")


if __name__ == "__main__":
    main()
