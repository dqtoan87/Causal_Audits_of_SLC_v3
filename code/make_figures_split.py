#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figures for the two split manuscripts.

paper1 (disease side)
  p1_fig1_dag       assumed graph with the verification node S and the alternative direction
  p1_fig2_forest    adjusted contrast against the within-patient contrast, per proxy

paper2 (model and concept side)
  p2_fig1_dag       the part of the graph the model side reads, with the edit operator
  p3_fig1_dag       working graph with the latent disease state Dis, and the ladder
  p2_fig2_tradeoff  footprint balance against concept neutrality over the control library
  fig_shrinkage     produced by make_fig_shrinkage.py, reused unchanged

Run:  .venv/bin/python Causal_XAI_v3/code/make_figures_split.py
"""
from __future__ import annotations

import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_source_guard import require_p2_source, WrongSourceTree      # noqa: E402
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "paper", "figures")
DPI = 300
MIN_W_IN, MAX_W_IN, MAX_H_IN = 789 / 300.0, 2250 / 300.0, 2625 / 300.0
W2, W1 = 7.50, 7.00
TIFF_KW = {"compression": "tiff_lzw"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "axes.linewidth": 0.8,
})
SM, MD, LG = 8, 9, 10
DET, SPUR, SEL = "#3a6ea5", "#7029b8", "#b8860b"
NODE_FC = {"Z": "#fff3d6", "C": "#e7f5e9", "Y": "#fde7e7", "X": "#eef3fb",
           "U": "#f0e7fb", "D": "#e8e8f6", "S": "#fdf0d5", "Ĉ": "#eaf1f8",
           "f": "#eaf1f8", "T": "#f3e7f7"}


def load(n):
    return json.load(open(os.path.join(RES, n)))


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    w_in, h_in = fig.get_size_inches()
    if not (MIN_W_IN - 1e-6 <= w_in <= MAX_W_IN + 1e-6) or h_in > MAX_H_IN + 1e-6:
        raise ValueError(f"{name}: {w_in:.2f}x{h_in:.2f} in outside the allowed range")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
    im = im.convert("RGB")
    im.save(os.path.join(FIG, name + ".tiff"), format="TIFF", dpi=(DPI, DPI), **TIFF_KW)
    im.save(os.path.join(FIG, name + ".png"), format="PNG", dpi=(DPI, DPI))
    return name


def node(ax, key, x, y, rx=0.068, ry=0.086, dashed=False, label=None):
    p = mpatches.Ellipse((x, y), 2 * rx, 2 * ry, fc=NODE_FC[key], ec="k", lw=1.1,
                         ls=(0, (3, 2)) if dashed else "-", zorder=3)
    ax.add_patch(p)
    ax.text(x, y, label or key, ha="center", va="center", fontsize=LG, weight="bold", zorder=4)
    return p


def edge(ax, a, b, rad=0.0, c="k", ls="-"):
    ax.add_patch(mpatches.FancyArrowPatch(
        a.center, b.center, patchA=a, patchB=b, shrinkA=1.5, shrinkB=1.5,
        connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=11,
        color=c, lw=1.3, ls=ls, zorder=2))


# ----------------------------------------------------------------- paper 1, figure 1
def p1_fig1_dag():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 3.85))
    for ax in (a1, a2):
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    Z = node(a1, "Z", 0.14, 0.86); C = node(a1, "C", 0.48, 0.86)
    D = node(a1, "D", 0.84, 0.86, dashed=True)
    X = node(a1, "X", 0.36, 0.52); S = node(a1, "S", 0.62, 0.52)
    Yr = node(a1, "Y", 0.84, 0.20)
    Ch = node(a1, "Ĉ", 0.14, 0.20, rx=0.060, ry=0.076, dashed=True)
    edge(a1, Z, C); edge(a1, C, D); edge(a1, Z, D, rad=-0.20); edge(a1, Z, X); edge(a1, C, X)
    edge(a1, X, Ch, c=DET, ls=":")
    edge(a1, C, S, c=SEL); edge(a1, D, S, c=SEL, rad=0.18)
    edge(a1, S, Yr, c=SEL); edge(a1, D, Yr)
    a1.set_title("(a) Disease side, with verification", fontsize=LG, weight="bold", pad=2)

    Z = node(a2, "Z", 0.15, 0.84); Dl = node(a2, "D", 0.50, 0.84, dashed=True)
    Y = node(a2, "Y", 0.85, 0.84); C = node(a2, "C", 0.50, 0.50); X = node(a2, "X", 0.42, 0.16)
    Ch2 = node(a2, "Ĉ", 0.16, 0.16, rx=0.060, ry=0.076, dashed=True)
    edge(a2, Z, Dl); edge(a2, Dl, Y); edge(a2, Dl, C); edge(a2, C, X); edge(a2, Z, X)
    edge(a2, X, Ch2, c=DET, ls=":")
    a2.set_title("(b) Alternative direction", fontsize=LG, weight="bold", pad=2)

    fig.text(0.5, 0.175, "Z: measured confounders (age, sex, anatomical and acquisition site, skin tone)    "
             "C: lesion concepts    X: image", ha="center", va="center", fontsize=SM)
    fig.text(0.5, 0.132, "D: true disease state (unobserved, dashed)    "
             "Y: recorded malignant label", ha="center", va="center", fontsize=SM)
    fig.text(0.5, 0.089, "Ĉ = m(X): image-derived proxy, a deterministic reading of the image and not a cause of disease",
             ha="center", va="center", fontsize=SM, color=DET)
    fig.text(0.5, 0.046, "S: verification, that is whether the lesion was worked up. Y = D only where S = 1; "
             "elsewhere Y is benign by default.", ha="center", va="center", fontsize=SM, color=SEL)
    fig.text(0.5, 0.010, "C → S is the ascertainment path: appearance influences which lesions are sampled, so the label is not "
             "missing at random.", ha="center", va="center", fontsize=SM, color=SEL)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.22, wspace=0.04)
    return save(fig, "p1_fig1_dag")


# ----------------------------------------------------------------- paper 1, figure 2
LABEL = {"C_color": "Color variegation", "C_colorstd": "Color SD (exploratory)",
         "D_size": "Size", "contrast_LB": "Lesion-skin contrast",
         "A_asymmetry": "Asymmetry", "B_border": "Border irregularity",
         "NEGCTRL_nevi": "Benign proxy"}
ORDER = ["C_color", "C_colorstd", "D_size", "contrast_LB", "A_asymmetry", "B_border", "NEGCTRL_nevi"]


def p1_fig2_forest():
    ate = {c["concept"]: c for c in load("tier1_ate.json")["concepts"]}
    wp = {c["concept"]: c for c in load("c41_within_patient.json")["concepts"]}
    het = load("c47_heterogeneity.json")["concepts"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 3.60), sharey=True,
                                   gridspec_kw={"width_ratios": [1.0, 1.0]})
    ys = np.arange(len(ORDER))[::-1]
    for ax, get, ttl in (
        (ax1, lambda k: (ate[k]["rr"], ate[k]["rr_ci95"]), "(a) Adjusted risk ratio"),
        (ax2, lambda k: (wp[k]["or"], wp[k]["ci_cluster"]), "(b) Within-patient odds ratio"),
    ):
        for y, k in zip(ys, ORDER):
            p, ci = get(k)
            res = ci[0] > 1 or ci[1] < 1
            col = "#1f77b4" if res else "#8c8c8c"
            ax.plot(ci, [y, y], color=col, lw=1.6, solid_capstyle="butt", zorder=2)
            ax.plot([p], [y], "o", ms=5, color=col, zorder=3)
        ax.axvline(1.0, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(0.2, 8.0)
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
        ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
        ax.set_title(ttl, fontsize=LG, weight="bold", pad=4)
        ax.grid(axis="x", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    # site-specific contrasts on panel (b), as open markers
    for y, k in zip(ys, ORDER):
        if k in het:
            a = het[k]["axes"]["site_limb"]
            ax2.plot([a["or_group0"]], [y + 0.24], "s", ms=3.6, mfc="white",
                     mec="#d62728", mew=1.0, zorder=4)
            ax2.plot([a["or_group1"]], [y - 0.24], "^", ms=4.2, mfc="white",
                     mec="#d62728", mew=1.0, zorder=4)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([LABEL[k] for k in ORDER], fontsize=SM)
    ax1.set_ylim(-0.8, len(ORDER) - 0.2)
    ax1.set_xlabel("risk ratio (log scale)", fontsize=MD)
    ax2.set_xlabel("odds ratio (log scale)", fontsize=MD)
    h = [plt.Line2D([], [], color="#1f77b4", marker="o", lw=1.6, ms=5, label="interval excludes 1"),
         plt.Line2D([], [], color="#8c8c8c", marker="o", lw=1.6, ms=5, label="interval covers 1"),
         plt.Line2D([], [], color="#d62728", marker="s", lw=0, mfc="white", ms=3.6, label="head or trunk"),
         plt.Line2D([], [], color="#d62728", marker="^", lw=0, mfc="white", ms=4.2, label="extremities")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=SM,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.235, right=0.985, top=0.92, bottom=0.215, wspace=0.07)
    return save(fig, "p1_fig2_forest")


# ----------------------------------------------------------------- paper 2, figure 1
OKC, BADC, ASSC = "#2f6b3a", "#b3261e", "#8a6d1f"


def _rung(ax, y, x0, x1, label, status, color):
    """One rung of the identification ladder: a labeled arrow between two named quantities."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=color,
                                ls="-" if status != "assumed" else (0, (4, 2)),
                                mutation_scale=12))
    ax.text((x0 + x1) / 2, y + 0.055, label, ha="center", va="bottom", fontsize=SM, color=color)


def p2_fig1_dag():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 4.35),
                                 gridspec_kw={"width_ratios": [1.0, 1.28]})
    for ax in (a1, a2):
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    Z = node(a1, "Z", 0.14, 0.80, rx=0.070, ry=0.098)
    C = node(a1, "C", 0.50, 0.80, rx=0.070, ry=0.098)
    Y = node(a1, "Y", 0.86, 0.80, rx=0.070, ry=0.098)
    X = node(a1, "X", 0.50, 0.40, rx=0.070, ry=0.098)
    U = node(a1, "U", 0.14, 0.40, rx=0.070, ry=0.098)
    T = node(a1, "T", 0.86, 0.40, rx=0.086, ry=0.098, label="$T_A$")
    F = node(a1, "f", 0.86, 0.10, rx=0.070, ry=0.070, dashed=True)
    edge(a1, Z, C); edge(a1, C, Y); edge(a1, Z, Y, rad=-0.22); edge(a1, Z, X); edge(a1, C, X)
    edge(a1, U, X, c=SPUR, ls="--"); edge(a1, Z, U)
    edge(a1, X, T, c=SPUR); edge(a1, T, F, c=DET, ls=":")
    a1.text(0.68, 0.315, "performed edit", ha="center", va="center", fontsize=SM, color=SPUR)
    a1.set_title("(a) The assumed graph", fontsize=LG, weight="bold", pad=2)

    a2.set_title("(b) The identification ladder", fontsize=LG, weight="bold", pad=2)
    box = dict(boxstyle="round,pad=0.28", fc="#eef3fb", ec="0.45", lw=0.9)
    obs = dict(boxstyle="round,pad=0.28", fc="#e7f5e9", ec="0.45", lw=0.9)
    # artifact chain
    a2.text(0.09, 0.86, r"$\theta^{\,\mathrm{edit}}$", ha="center", va="center", fontsize=MD, bbox=obs)
    a2.text(0.72, 0.86, r"$\theta^{\,\mathrm{do}}$", ha="center", va="center", fontsize=MD, bbox=box)
    _rung(a2, 0.86, 0.18, 0.63, "T1  transfer\nassumed, untested", "assumed", ASSC)
    a2.text(0.0, 0.98, "artifact side", fontsize=SM, weight="bold", ha="left", color="0.30")
    # concept chain
    a2.text(0.0, 0.63, "concept side", fontsize=SM, weight="bold", ha="left", color="0.30")
    a2.text(0.09, 0.46, r"$R(\mathrm{edit})$", ha="center", va="center", fontsize=MD, bbox=obs)
    a2.text(0.44, 0.46, r"$g+h$", ha="center", va="center", fontsize=MD, bbox=box)
    a2.text(0.72, 0.46, r"$h$", ha="center", va="center", fontsize=MD, bbox=box)
    a2.text(0.95, 0.46, r"$\theta^{C}$", ha="center", va="center", fontsize=MD, bbox=box)
    _rung(a2, 0.46, 0.19, 0.36, "C1  working form\nassumed; testable\npart challenged", "assumed", ASSC)
    _rung(a2, 0.46, 0.52, 0.67, "C2  identification\nfrom the library\nfails, nonzero branch", "tested", BADC)
    _rung(a2, 0.46, 0.77, 0.90, "C3  intervention\nvalidity\nassumed", "assumed", ASSC)
    fig.text(0.5, 0.190, "Green box: estimated from the audited images.   Amber dashed rung: "
             "assumed, not tested.   Red rung: tested and failed.", ha="center", va="center",
             fontsize=SM, color="0.25")
    fig.text(0.5, 0.145, "Z: measured confounders    C: lesion concepts    X: image    "
             "U: acquisition artifact    Y: recorded label    f: classifier output",
             ha="center", va="center", fontsize=SM)
    fig.text(0.5, 0.100, "The graph gives no directed path U → Y. It does not give $f(T_A(X)) = f(X)$; "
             "that step is the requirement the audit imposes.", ha="center", va="center",
             fontsize=SM, color=SPUR)
    fig.text(0.5, 0.055, "Z → U is what makes artifacts informative about the outcome, so an "
             "accurate classifier has an incentive to read them.", ha="center", va="center",
             fontsize=SM)
    fig.text(0.5, 0.012, "$T_A$ is applied to the image, so the model-side response is estimated "
             r"rather than identified; $\mathrm{do}(U)$ is not available on an archived image.",
             ha="center", va="center", fontsize=SM)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.225, wspace=0.03)
    return save(fig, "p2_fig1_dag")


# ----------------------------------------------------------------- paper 3, figure 1
def p3_fig1_dag():
    """Working graph with the latent disease state Dis, and the identification ladder."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 4.35),
                                 gridspec_kw={"width_ratios": [1.0, 1.28]})
    for ax in (a1, a2):
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    Z = node(a1, "Z", 0.14, 0.86, rx=0.070, ry=0.090)
    D = node(a1, "D", 0.50, 0.86, rx=0.070, ry=0.090, dashed=True, label="Dis")
    Y = node(a1, "Y", 0.86, 0.86, rx=0.070, ry=0.090)
    C = node(a1, "C", 0.50, 0.60, rx=0.070, ry=0.090)
    U = node(a1, "U", 0.14, 0.34, rx=0.070, ry=0.090)
    X = node(a1, "X", 0.50, 0.34, rx=0.070, ry=0.090)
    T = node(a1, "T", 0.86, 0.34, rx=0.086, ry=0.090, label="$T_A$")
    F = node(a1, "f", 0.86, 0.08, rx=0.065, ry=0.065)
    edge(a1, Z, D); edge(a1, D, Y); edge(a1, Z, Y, rad=-0.30); edge(a1, D, C); edge(a1, Z, C)
    edge(a1, C, X); edge(a1, Z, X); edge(a1, Z, U)
    edge(a1, U, X, c=SPUR, ls="--"); edge(a1, X, T, c=SPUR); edge(a1, T, F, c=DET, ls=":")
    a1.text(0.68, 0.255, "performed edit", ha="center", va="center", fontsize=SM, color=SPUR)
    a1.set_title("(a) The working graph", fontsize=LG, weight="bold", pad=2)

    a2.set_title("(b) The identification ladder", fontsize=LG, weight="bold", pad=2)
    box = dict(boxstyle="round,pad=0.28", fc="#eef3fb", ec="0.45", lw=0.9)
    obs = dict(boxstyle="round,pad=0.28", fc="#e7f5e9", ec="0.45", lw=0.9)
    a2.text(0.09, 0.86, r"$\theta^{\,\mathrm{edit}}$", ha="center", va="center", fontsize=MD, bbox=obs)
    a2.text(0.72, 0.86, r"$\theta^{\,\mathrm{do}}$", ha="center", va="center", fontsize=MD, bbox=box)
    _rung(a2, 0.86, 0.18, 0.63, "T1  transfer\nassumed, untested", "assumed", ASSC)
    a2.text(0.0, 0.98, "artifact side", fontsize=SM, weight="bold", ha="left", color="0.30")
    a2.text(0.0, 0.71, "concept side", fontsize=SM, weight="bold", ha="left", color="0.30")
    a2.text(0.06, 0.52, r"$R(e)$", ha="center", va="center", fontsize=MD, bbox=obs)
    a2.text(0.33, 0.52, r"$g+h+q$", ha="center", va="center", fontsize=MD, bbox=box)
    a2.text(0.62, 0.52, r"$g+h$", ha="center", va="center", fontsize=MD, bbox=box)
    a2.text(0.90, 0.52, r"$h^{\ast}$", ha="center", va="center", fontsize=MD, bbox=box)
    a2.text(0.62, 0.31, r"$\tau(M^{\ast})$", ha="center", va="center", fontsize=MD, bbox=box)
    _rung(a2, 0.52, 0.12, 0.24, "C1a\ngranted", "assumed", ASSC)
    _rung(a2, 0.52, 0.43, 0.55, "C1b\ngranted", "assumed", ASSC)
    _rung(a2, 0.52, 0.69, 0.85, "C2  footprint route\nnot satisfied (C0 gates it)\nN  neutrality: untested", "tested", BADC)
    a2.annotate("", xy=(0.545, 0.31), xytext=(0.36, 0.475),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=ASSC, ls=(0, (4, 2)), mutation_scale=12))
    a2.text(0.30, 0.355, "exact control\n(needs N)\nnone certified", ha="center", va="center", fontsize=SM, color=ASSC)
    a2.annotate("", xy=(0.885, 0.475), xytext=(0.70, 0.31),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=ASSC, ls=(0, (4, 2)), mutation_scale=12))
    a2.text(0.80, 0.355, "C1b", ha="left", va="center", fontsize=SM, color=ASSC)
    a2.text(0.06, 0.12, r"$R(e^{\ast})$", ha="center", va="center", fontsize=MD, bbox=obs)
    a2.text(0.90, 0.12, r"$\theta^{C}$", ha="center", va="center", fontsize=MD, bbox=box)
    _rung(a2, 0.12, 0.16, 0.84, "C3  a: assumed, untested\nb: fixed level not met by the edit", "tested", BADC)
    a2.annotate("", xy=(0.975, 0.17), xytext=(0.975, 0.47),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="0.45", ls=(0, (2, 2))))
    a2.text(0.96, 0.255, r"differ by" "\n" r"$g(M(e^{\ast}))$", ha="right", va="center", fontsize=SM, color="0.30")
    fig.text(0.5, 0.190, "Green box: estimated from the audited images.   Amber dashed rung: "
             "granted or assumed, not established.   Red rung: not satisfied by the design.",
             ha="center", va="center", fontsize=SM, color="0.25")
    fig.text(0.5, 0.145, "Z: measured confounders   Dis: disease state (latent)   C: clinical appearance   "
             "X: image   U: artifact   Y: recorded label   f: classifier output",
             ha="center", va="center", fontsize=SM)
    fig.text(0.5, 0.100, "The graph gives no directed path U \u2192 Y. It does not give $f(T_A(X)) = f(X)$; "
             "that step is the requirement the audit imposes.", ha="center", va="center",
             fontsize=SM, color=SPUR)
    fig.text(0.5, 0.055, "C and Y are associated through Dis, not by an effect of C on Y. U \u2190 Z \u2192 Y lets "
             "an artifact carry information about the label.", ha="center", va="center", fontsize=SM)
    fig.text(0.5, 0.012, "$T_A$ is applied to the image, so the model-side response is estimated "
             r"rather than identified; $\mathrm{do}(U)$ is not available on an archived image.",
             ha="center", va="center", fontsize=SM)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.225, wspace=0.03)
    return save(fig, "p3_fig1_dag")


# ----------------------------------------------------------------- paper 2, figure 2
def p2_fig2_tradeoff():
    require_p2_source(BASE, "p2_fig2_tradeoff", "make_figures_split.py")
    d = load("c25_screen_full.json")["design"]
    tgt_lbl = {"contrast@2": "contrast, top dose", "color_var@2": "color variegation, top dose",
               "contrast@0": "contrast, low dose", "color_var@0": "color variegation, low dose"}
    mk = {"contrast@2": "o", "color_var@2": "s", "contrast@0": "^", "color_var@0": "D"}
    fig, ax = plt.subplots(figsize=(W1, 4.05))
    for t, ctrls in d.items():
        xs, ys = [], []
        for k, v in ctrls.items():
            if not isinstance(v, dict) or "max_abs_smd" not in v:
                continue
            xs.append(max(v["max_abs_smd"], 1e-3)); ys.append(max(abs(v["proxy_ratio"]), 1e-3))
        ax.plot(xs, ys, mk[t], ms=4.6, mfc="none", mew=1.0, lw=0, label=tgt_lbl[t])
    ax.axvline(0.10, color="#1f77b4", lw=1.1, ls="--")
    ax.axhline(0.15, color="#d62728", lw=1.1, ls="--")
    ax.add_patch(mpatches.Rectangle((1e-3, 1e-3), 0.10 - 1e-3, 0.15 - 1e-3,
                                    fc="#e8f4ea", ec="none", zorder=0))
    ax.text(0.030, 0.115, "(A1) and the relative part\nof (A2) met: necessary,\nnot sufficient. Empty\nfor every target.",
            fontsize=SM, color="#2f6b3a", va="top", ha="left")
    ax.annotate("rotd: balanced, moves the concept", xy=(0.050, 1.02), xytext=(0.085, 0.36),
                fontsize=SM, arrowprops=dict(arrowstyle="->", lw=0.8, color="0.35"))
    ax.annotate("pixperm_m: neutral, unbalanced", xy=(1.42, 0.0011), xytext=(0.16, 0.0026),
                fontsize=SM, ha="left", arrowprops=dict(arrowstyle="->", lw=0.8, color="0.35"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.025, 5.0); ax.set_ylim(7e-4, 3.0)
    ax.set_xlabel("footprint imbalance, maximum |SMD| over the six components", fontsize=MD)
    ax.set_ylabel("concept movement, |proxy shift| relative to the edit", fontsize=MD)
    ax.grid(color="0.92", lw=0.6); ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False,
               fontsize=SM, handletextpad=0.3, columnspacing=1.1, bbox_to_anchor=(0.53, 0.002))
    ax.text(0.093, 2.7, "(A1) tolerance", fontsize=SM, color="#1f77b4", rotation=90, va="top", ha="right")
    ax.text(0.027, 0.165, "(A2) relative tolerance", fontsize=SM, color="#d62728", ha="left", va="bottom")
    fig.subplots_adjust(left=0.115, right=0.985, top=0.975, bottom=0.185)
    return save(fig, "p2_fig2_tradeoff")


if __name__ == "__main__":
    for fn in (p1_fig1_dag, p1_fig2_forest, p2_fig1_dag, p3_fig1_dag, p2_fig2_tradeoff):
        try:
            print("wrote", fn())
        except WrongSourceTree as e:
            print("skipped %s:\n%s" % (fn.__name__, e))
