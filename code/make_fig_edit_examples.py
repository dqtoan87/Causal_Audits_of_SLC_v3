#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure of the performed edits on two images (reads no model output).

Row 1, audit image ISIC_4114301 (fold 0, malignant): the unedited image with the proxy lesion region of
the occlusion check, the main hair overlay of Table 4, its skin-only and lesion-only versions, the ruler and
its skin-only version.
Row 2, the same image: the contrast edit at the lowest and the highest gain, the color variegation edit at
the highest gain, two library controls of the highest-dose contrast target (the rotated displacement
`rotd`, best balanced, and the permuted lesion pixels `pixperm_m`, most neutral) and the noise placebo.
Row 3, held-out image ISIC_5583714 (fold 1, in the route test's primary set): the unedited image, the
contrast edit at the highest gain and the four registered route-test controls, rebuilt from their
recorded settings. Their change magnitude equals the edit's at every pixel; the labels give their CIELAB
dE and high-frequency change as a share of the edit's.

Both images are released under CC0 in the ISIC Archive (attribution: ACEMID MIA).
Every overlay uses the realization of Table 4, and every control the seed of the library or of the
registered construction, so the panels show the edits that were analysed.

Out -> paper/figures/p3_fig_edits.png and .tiff
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import audit_cohort                       # noqa: E402
import c2_cace_artifact as c2m            # noqa: E402
import c30_disjoint_audit as c30          # noqa: E402
import c12_matched_control as c12         # noqa: E402
import c62_route_test as c62              # noqa: E402
import c70_occlusion_mask_sensitivity as c70   # noqa: E402
from p2_source_guard import require_p2_source  # noqa: E402

import matplotlib                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402

BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "paper", "figures")
AUDIT_ID, ROUTE_ID = "ISIC_4114301", "ISIC_5583714"
c3m, c10 = c12.c3m, c12.c10


def altered(x, e):
    return 100.0 * float(((e - x).abs().amax(0) > 0.02).float().mean())


def to_np(t):
    return t.clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    require_p2_source(BASE, "p3_fig_edits", "make_fig_edit_examples.py")
    imgs, labels, pid, ids = audit_cohort.load()
    ii = list(ids).index(AUDIT_ID)
    x = imgs[ii]
    edits, _ = c2m.precompute_edits(imgs, c30.ART, c30.SEED)          # the realization of Table 4
    e = edits[ii]
    L = c70.region(x, 0.5, 3)
    keep = (~L).float().unsqueeze(0)
    skin = {a: x + (e[a] - x) * keep for a in ("hair", "ruler")}
    lesion_only = x + (e["hair"] - x) * L.float().unsqueeze(0)

    w, sref, lm = c3m.soft_mask(x)
    ce = {(c, k): c3m.make_edit(x, c, c3m.DOSES[k], w, sref, lm) for c in ("contrast", "color_var") for k in (0, 2)}
    t = ce[("contrast", 2)]
    rotd = c12.struct_controls(x, t - x, w, c12.SEED + ii)["rotd"]
    donor = imgs[(ii + 1) % len(imgs)]                                  # as in the library build on 183 images
    fams = c12.make_families(x, w, sref, c12.SEED + ii, donor=donor)
    fn, lo, hi = fams["pixperm_m"]
    pixperm = c12.match_strength(fn, lo, hi, x, float(c12.dpix(x, t).mean()))[0]
    noise = c10.PLACEBOS["noise"](x, w, sref, np.random.RandomState(c12.SEED + ii + 7))

    cons = json.load(open(os.path.join(RES, "c62_construction.json")))
    jj = cons["ids"].index(ROUTE_ID)
    torch.set_num_threads(1)                                            # rebuild as the construction did
    y = c62.audit_cohort._read([ROUTE_ID])[0]
    cx = c62.context(y, "contrast", 2)
    rec = cons["images"][jj]["targets"]["contrast@2"]
    built = c62.build_controls(y, cx, jj, params={c: rec["controls"][c]["param"] for c in c62.CONTROLS})
    em = rec["edit"]

    def rel(c, k):
        return 100.0 * rec["controls"][c]["measures"][k] / em[k]

    rows = [
        [(x, "(a) unedited; proxy\nlesion region outlined", L),
         (e["hair"], "(b) hair, main dose\n%.1f%% of pixels altered" % altered(x, e["hair"]), None),
         (skin["hair"], "(c) (b) kept outside\nthe region: %.1f%%" % altered(x, skin["hair"]), L),
         (lesion_only, "(d) (b) kept inside\nthe region: %.1f%%" % altered(x, lesion_only), L),
         (e["ruler"], "(e) ruler\n%.1f%% of pixels altered" % altered(x, e["ruler"]), None),
         (skin["ruler"], "(f) (e) kept outside\nthe region: %.1f%%" % altered(x, skin["ruler"]), L)],
        [(ce[("contrast", 0)], "(g) contrast edit,\nlowest gain", None),
         (t, "(h) contrast edit,\nhighest gain", None),
         (ce[("color_var", 2)], "(i) color variegation\nedit, highest gain", None),
         (rotd, "(j) control for (h):\nrotated displacement", None),
         (pixperm, "(k) control for (h):\npermuted lesion pixels", None),
         (noise, "(l) noise placebo,\ndefault strength", None)],
        [(y, "(m) held-out image,\nunedited", None),
         (cx["e"], "(n) contrast edit,\nhighest gain", None)] +
        [(built[c][0], "(%s) %s: same |change|;\n$\\Delta E$ %d%%, HF %d%%" % (lab, c, round(rel(c, "lab_dE")), round(rel(c, "hf_change"))), None)
         for lab, c in zip("opqr", c62.CONTROLS)],
    ]
    fig, axes = plt.subplots(3, 6, figsize=(7.2, 4.65), dpi=300)
    for r, row in enumerate(rows):
        for c, (img, title, region) in enumerate(row):
            ax = axes[r, c]
            ax.imshow(to_np(img), interpolation="nearest")
            if region is not None:
                ax.contour(region.numpy().astype(float), levels=[0.5], colors=["#1f9e89"], linewidths=0.45)
            ax.set_title(title, fontsize=5.6, pad=2.0, linespacing=1.05)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.3)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.005, wspace=0.04, hspace=0.36)
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "tiff"):
        fig.savefig(os.path.join(FIG, "p3_fig_edits.%s" % ext), dpi=300)
    info = {"audit_image": AUDIT_ID, "route_image": ROUTE_ID,
            "altered_pct": {"hair": altered(x, e["hair"]), "hair_skin": altered(x, skin["hair"]),
                            "hair_lesion": altered(x, lesion_only), "ruler": altered(x, e["ruler"]),
                            "ruler_skin": altered(x, skin["ruler"])},
            "route_rel_pct": {c: {"lab_dE": rel(c, "lab_dE"), "hf_change": rel(c, "hf_change"),
                                  "mean_abs": rel(c, "mean_abs"), "frac_altered": rel(c, "frac_altered")} for c in c62.CONTROLS}}
    print(json.dumps(info, indent=1))
    print("saved -> %s" % os.path.join(FIG, "p3_fig_edits.png"))


if __name__ == "__main__":
    main()
