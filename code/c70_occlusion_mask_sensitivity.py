#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C70: sensitivity of the occlusion check to the lesion region, and a dose-matched comparison
(exploratory, not pre-specified)

c66_overlay_occlusion.py restores every overlay pixel inside a proxy-defined lesion region: the soft
lesion mask of Section III-C at w >= 0.5, widened by 3 pixels. Both numbers are audit choices, and the
soft mask is not a lesion segmentation. This script varies them.

  1. Region. For each (w threshold, dilation) in CONFIGS: the share of the frame the region covers,
     and the share of the main hair edit's altered pixels that fall inside it.
  2. Skin-only criterion. The skin-only mixture of c66 for each region: marginal 95% patient-clustered
     intervals with 4,000 resamples, and the 36-component max-statistic band with 20,000 resamples,
     exactly as c66 computes them. The reference region (0.5, 3) must reproduce c66.
  3. Dose-matched comparison. Removing the lesion part of an overlay lowers its dose as well as moving
     it. The skin-only heavy hair overlay alters about as many pixels as the full main hair overlay, so
     comparing the two holds the altered-pixel fraction roughly fixed while the location changes. It
     also changes strand count and opacity, so it is a dose-matched comparison, not a same-overlay one.
     Per cell, both mean absolute shifts and the single-operator verdict of the skin-only heavy overlay.

If the environment variable C70_MONTAGE names a directory, a montage of lesion regions is written there
for visual inspection. The montage is not used by any number.

Out -> results/c70_occlusion_mask_sensitivity.json
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as tF

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import audit_cohort                  # noqa: E402
import c2_cace_artifact as c2m       # noqa: E402
import c3_cace_concept as c3m        # noqa: E402
import c30_disjoint_audit as c30     # noqa: E402
import c66_overlay_occlusion as c66  # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c70_occlusion_mask_sensitivity.json")
HELD, THR, MARGIN = c66.HELD, c66.THR, c66.MARGIN
CONFIGS = [(0.5, 3), (0.3, 3), (0.7, 3), (0.5, 0), (0.5, 7), (0.5, 15)]
DEVICE = c30.DEVICE


def region(x, w_thr, dil):
    w, _, _ = c3m.soft_mask(x)
    L = (w >= w_thr).float()[None, None]
    if dil > 0:
        L = tF.max_pool2d(L, 2 * dil + 1, stride=1, padding=dil)
    return L[0, 0] > 0


def altered(a, b):
    return (a - b).abs().amax(0) > THR


def montage(imgs, regs, path, k=16):
    from PIL import Image, ImageDraw
    share = np.array([float(r.float().mean()) for r in regs[(0.5, 3)]])
    order = np.argsort(share)
    pick = list(order[:4]) + list(order[len(order) // 2 - 4: len(order) // 2 + 4]) + list(order[-4:])
    tiles = []
    for i in pick[:k]:
        x = (imgs[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        im = Image.fromarray(x).convert("RGB")
        ov = np.array(im)
        for cfg, col in (((0.5, 3), (0, 255, 0)), ((0.5, 15), (255, 255, 0)), ((0.7, 3), (255, 0, 255))):
            R = regs[cfg][i].numpy()
            edge = R ^ np.roll(R, 1, 0) | R ^ np.roll(R, 1, 1)
            ov[edge] = col
        tiles.append(Image.fromarray(ov))
    W = 4
    canvas = Image.new("RGB", (224 * W, 224 * ((len(tiles) + W - 1) // W)), "white")
    for j, t in enumerate(tiles):
        canvas.paste(t, ((j % W) * 224, (j // W) * 224))
    ImageDraw.Draw(canvas)
    canvas.save(path)


def main():
    t0 = time.time()
    imgs, labels, pid, ids = audit_cohort.load()
    labels, pid = np.asarray(labels), np.asarray(pid)
    edits, _ = c2m.precompute_edits(imgs, c30.ART, c30.SEED)     # the realization of Table 4
    regs = {cfg: [region(x, *cfg) for x in imgs] for cfg in CONFIGS}
    mdir = os.environ.get("C70_MONTAGE")
    if mdir:
        os.makedirs(mdir, exist_ok=True)
        montage(imgs, regs, os.path.join(mdir, "c70_lesion_regions.png"))

    cover = {}
    for cfg in CONFIGS:
        sh = [float(r.float().mean()) for r in regs[cfg]]
        inside, af_skin = {a: [] for a in HELD}, {a: [] for a in HELD}
        for i, x in enumerate(imgs):
            L = regs[cfg][i]
            for a in HELD:
                d = edits[i][a] - x
                ch = d.abs().amax(0) > THR
                inside[a].append(float((ch & L).sum()) / max(int(ch.sum()), 1))
                af_skin[a].append(float(altered(x + d * (~L).float().unsqueeze(0), x).float().mean()))
        cover["%g/%d" % cfg] = {"frame_share_median": float(np.median(sh)), "frame_share_range": [min(sh), max(sh)],
                                "n_full_frame": int(sum(s >= 0.999 for s in sh)),
                                "altered_in_region_mean": {a: float(np.mean(inside[a])) for a in HELD},
                                "altered_frac_skin_only_mean": {a: float(np.mean(af_skin[a])) for a in HELD}}
        print("region %-7s median share %.3f; main hair altered inside %.3f; skin-only altered: heavy %.4f"
              % ("%g/%d" % cfg, np.median(sh), np.mean(inside["hair"]), np.mean(af_skin["hair_heavy"])), flush=True)
    af_full = {a: float(np.mean([float(altered(edits[i][a], imgs[i]).float().mean()) for i in range(len(imgs))]))
               for a in HELD}

    vec = {}
    for bb, trio in c30.TRIO.items():
        for ck, kind, nm in trio:
            f = os.path.join(os.path.dirname(HERE), "checkpoints", ck + ".pt")
            m = c30.make_model(kind).to(DEVICE)
            m.load_state_dict(torch.load(f, map_location=DEVICE)); m.eval()
            p0 = c30.probs(m, imgs)
            key = "%s-%s" % (bb, nm)
            vec[key] = {"full": np.stack([c30.probs(m, [edits[i][a] for i in range(len(imgs))]) - p0 for a in HELD])}
            for cfg in CONFIGS:
                vec[key]["%g/%d" % cfg] = np.stack(
                    [c30.probs(m, [imgs[i] + (edits[i][a] - imgs[i]) * (~regs[cfg][i]).float().unsqueeze(0)
                                   for i in range(len(imgs))]) - p0 for a in HELD])
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            print("  [%s] scored (%.0fs)" % (key, time.time() - t0), flush=True)

    subs = [("all", np.ones(len(labels), bool)), ("malignant", labels == 1), ("benign", labels == 0)]
    ref = json.load(open(os.path.join(RES, "c66_overlay_occlusion.json")))
    out = {"configs": [list(c) for c in CONFIGS], "altered_threshold": THR, "margin_p": MARGIN,
           "altered_frac_full_mean": af_full, "regions": cover, "skin_only": {}, "dose_matched": {}}
    for cfg in CONFIGS:
        name = "%g/%d" % cfg
        keys, vecs, masks, ismag, jv, low = [], [], [], [], [], []
        for key, dd in vec.items():
            d = dd[name]
            mag, sgn = np.abs(d).mean(0), d.mean(0)
            for sname, msk in subs:
                _, lo, hi = c30.boot_ci(mag[msk], pid[msk])
                _, slo, shi = c30.boot_ci(sgn[msk], pid[msk])
                jv.append(c30.combine(c30.verdict_signed(slo, shi, MARGIN), c30.verdict(lo, hi, MARGIN)))
                low.append(lo)
                for comp, vv in (("mag", mag), ("sgn", sgn)):
                    keys.append("%s/%s/%s" % (key, sname, comp)); vecs.append(vv); masks.append(msk)
                    ismag.append(comp == "mag")
        crit, est, lo, hi, ism = c66.band36(vecs, masks, ismag, pid)
        res = {"marginal_joint_counts": {k: jv.count(k) for k in ("exceedance", "equivalence", "indeterminate")},
               "smallest_marginal_lower": float(min(low)), "crit_36": crit,
               "frontier_36": float(lo[ism].min()), "magnitude_exceed_36": int((lo[ism] > MARGIN).sum())}
        if cfg == (0.5, 3):
            rb = ref["band"]["skin_only"]
            res["check_vs_c66"] = {"frontier_abs_diff": abs(res["frontier_36"] - rb["frontier_36"]),
                                   "counts_equal": res["marginal_joint_counts"] == rb["marginal_joint_counts"]}
        out["skin_only"][name] = res
        print("skin-only %-7s marginal %s; band frontier %.4f, %d/18 jointly"
              % (name, res["marginal_joint_counts"], res["frontier_36"], res["magnitude_exceed_36"]), flush=True)

        # dose-matched: skin-only heavy hair against full main hair, per cell
        ih, im_ = HELD.index("hair_heavy"), HELD.index("hair")
        cells, ver = {}, []
        for key, dd in vec.items():
            for sname, msk in subs:
                a = np.abs(dd[name][ih][msk]); b = np.abs(dd["full"][im_][msk])
                mean, lo_, hi_ = c30.boot_ci(a, pid[msk])
                v = c30.verdict(lo_, hi_, MARGIN); ver.append(v)
                cells["%s/%s" % (key, sname)] = {"skin_heavy": float(a.mean()), "skin_heavy_ci": [lo_, hi_],
                                                 "full_main": float(b.mean()), "ratio": float(a.mean() / max(b.mean(), 1e-12)),
                                                 "verdict_skin_heavy": v}
        rat = [c["ratio"] for c in cells.values()]
        out["dose_matched"][name] = {"altered_frac_skin_heavy": cover[name]["altered_frac_skin_only_mean"]["hair_heavy"],
                                     "altered_frac_full_main": af_full["hair"],
                                     "ratio_range": [min(rat), max(rat)], "ratio_median": float(np.median(rat)),
                                     "skin_heavy_verdicts": {k: ver.count(k) for k in ("exceedance", "equivalence", "indeterminate")},
                                     "cells": cells}
        print("   dose-matched: skin-only heavy %.4f vs full main %.4f altered; ratio %.2f to %.2f (median %.2f); %s"
              % (out["dose_matched"][name]["altered_frac_skin_heavy"], af_full["hair"], min(rat), max(rat),
                 np.median(rat), out["dose_matched"][name]["skin_heavy_verdicts"]), flush=True)
    out["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved -> %s (%.0fs)" % (OUT, out["runtime_sec"]))


if __name__ == "__main__":
    main()
