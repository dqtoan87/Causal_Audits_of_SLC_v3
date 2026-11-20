#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C74: the cycle test on images where the magnitude footprint matches exactly  (exploratory, not pre-specified)

The pre-specified gate G1 accepts a control when its four magnitude descriptors equal the concept edit's
within one pixel (altered-pixel fraction) and a relative 1e-5 (the other three). C1a speaks of equal
footprints, so a disagreement read under G1 also assumes that differences below the G1 tolerance do not
move the response. This script removes that step where the data allow it.

  1. Size of the residual mismatch: for every target and control, the largest relative difference of the
     four descriptors, the largest difference in altered pixels, and the number of images on which all four
     descriptors are equal to the concept edit's bit for bit (float64 values of float32 computations).
  2. The cycle test of the plan, with one change: each difference Delta_ab is computed only on the images of
     the run's primary set on which both of its controls match the edit's four descriptors bit for bit.
     As the plan does for a target, a difference is analysed only if at least 30 images qualify. The
     analysed differences share one patient bootstrap (20,000 resamples, the plan's seed) and one
     simultaneous 95% band, judged by the three-way rule at 0.5 logit. A cell disagrees when an analysed
     difference exceeds, agrees when all four are analysed and within, and is indeterminate otherwise.
     Both runs. The same family without the 30-image rule is reported as well.

It reads the registered construction and the stored per-image responses; no model is run.

Out -> results/c74_route_exact_match.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c62_route_test as c62            # noqa: E402  (pre-specified; imported only)
import c67_route_sensitivity as c67     # noqa: E402

RES = c62.RES
OUT = os.path.join(RES, "c74_route_exact_match.json")
CR = c62.CRIT
TKS = c67.TKS
MAG = ["frac_altered", "mean_abs", "support_rms", "lesion_cov"]
KEY = ["erm|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-hue4",
       "slim|color_var@2|D_rad4-rad16", "slim|contrast@0|D_rad4-rad16"]


def exact_masks(cons):
    n = len(cons["ids"])
    ex, resid = {}, {}
    for tk in TKS:
        for c in c62.CONTROLS:
            m = np.zeros(n, bool); rel, pix = [], []
            for i, im in enumerate(cons["images"]):
                t = im["targets"][tk]; cc = t["controls"][c]
                if not cc["g1"]:
                    continue
                e, mm = t["edit"], cc["measures"]
                m[i] = all(mm[k] == e[k] for k in MAG)
                rel.append(max(abs(mm[k] - e[k]) / max(abs(e[k]), 1e-12) for k in MAG[1:]))
                pix.append(abs(mm["frac_altered"] - e["frac_altered"]) * 224 * 224)
            ex[(tk, c)] = m
            resid["%s|%s" % (tk, c)] = {"n_g1": len(rel), "n_exact": int(m.sum()), "max_rel": float(max(rel)),
                                       "median_rel": float(np.median(rel)), "max_pixels": float(max(pix)),
                                       "images_pixels_differ": int(sum(p > 0.5 for p in pix))}
    return ex, resid


def cycle_pairwise(DL, cols, primary, ex, pids, min_images=0):
    col = {c: k for k, c in enumerate(cols)}
    comps, vals, msk, skipped = [], [], [], []
    for tk in TKS:
        for m in c62.MODELS:
            for a, b, _ in c62.PAIRS:
                name = "%s|%s|D_%s-%s" % (m, tk, a, b)
                mk = primary[tk] & ex[(tk, a)] & ex[(tk, b)]
                if mk.sum() < min_images:
                    skipped.append(name); continue
                comps.append(name)
                vals.append(DL[m][:, col["%s|%s" % (tk, b)]] - DL[m][:, col["%s|%s" % (tk, a)]])
                msk.append(mk)
    vals, msk = np.stack(vals), np.stack(msk)
    est, draws = c62.boot(vals, msk, pids, CR["boot"], CR["boot_seed"])
    se = np.nanstd(draws, 0, ddof=1)
    Z = np.abs(draws - est) / np.maximum(se, 1e-12)
    crit = float(np.nanpercentile(np.nanmax(Z, 1), CR["level"]))
    eps = CR["margin_logit"]
    out = {"crit": crit, "components": {}, "cells": {}, "not_analysed": skipped, "n_analysed": len(comps)}
    for k, name in enumerate(comps):
        lo, hi = est[k] - crit * se[k], est[k] + crit * se[k]
        out["components"][name] = {"est": float(est[k]), "sim": [float(lo), float(hi)], "decision": c62.decide(lo, hi, eps),
                                   "n_images": int(msk[k].sum()), "n_patients": int(len(set(pids[msk[k]])))}
    for tk in TKS:
        for m in c62.MODELS:
            ds = [out["components"].get("%s|%s|D_%s-%s" % (m, tk, a, b), {"decision": "not analysed"})["decision"]
                  for a, b, _ in c62.PAIRS]
            out["cells"]["%s|%s" % (m, tk)] = ("routes disagree" if "exceeds" in ds else
                                               "routes agree" if all(d == "within" for d in ds) else "indeterminate")
    cl = list(out["cells"].values())
    out["summary"] = {k: cl.count(k) for k in ("routes disagree", "routes agree", "indeterminate")}
    return out


def main():
    cons = json.load(open(c62.out_path("construction", False)))
    pids = np.asarray(cons["patient_ids"])
    g1 = c67.g1_masks(cons)
    ex, resid = exact_masks(cons)
    out = {"note": "exploratory, not pre-specified; reads stored responses only", "residual_mismatch": resid, "runs": {}}
    for run, suf in (("second_run", "_inv_retrained"), ("first_run", "")):
        r = c67.load_run(suf)
        common = np.all([r["prob"][m][:, 0] >= CR["score_floor"] for m in c62.MODELS], 0)
        primary = {tk: g1[tk] & common for tk in TKS}
        out["runs"][run] = {}
        for tag, mi in (("min30", CR["min_images"]), ("all", 0)):
            res = cycle_pairwise(r["dl"], r["cols"], primary, ex, pids, mi)
            res["key"] = {k: res["components"].get(k) for k in KEY}
            out["runs"][run][tag] = res
            dis = sorted(c for c, v in res["cells"].items() if v == "routes disagree")
            print("[%s, %s] %d differences analysed, crit %.3f %s disagree: %s"
                  % (run, tag, res["n_analysed"], res["crit"], res["summary"], dis))
            for k in KEY:
                v = res["components"].get(k)
                if v is None:
                    print("   %-32s not analysed" % k); continue
                print("   %-32s n=%2d (%2d pts)  %+.2f [%.2f, %.2f]  %s" % (k, v["n_images"], v["n_patients"], v["est"],
                                                                       v["sim"][0], v["sim"][1], v["decision"]))
    mr = max(v["max_rel"] for v in resid.values()); mp = max(v["max_pixels"] for v in resid.values())
    out["largest_relative_mismatch"] = mr; out["largest_pixel_mismatch"] = mp
    print("largest relative mismatch %.2e, largest pixel mismatch %.0f" % (mr, mp))
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
