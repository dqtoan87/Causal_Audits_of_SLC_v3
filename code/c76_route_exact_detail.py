#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C76: the decisive route-test differences under the three analyses, and the exact identity  (exploratory)

It reads the registered construction and the stored per-image responses; no model is run.

  1. For every difference that decides a cell in any analysis, its images, patients, estimate and
     simultaneous 95% interval under (i) the pre-specified gate G1 (the plan's analysis), (ii) the
     bit-exact reanalysis with all 48 differences and (iii) the bit-exact reanalysis with the 30-image rule.
     (ii) and (iii) are recomputed here with the construction of c74 and must reproduce it.
  2. The exact identity. For the audited f, which includes its inference protocol, a response is a fixed
     number. On the images where both controls of a difference match the concept edit's four magnitude
     descriptors bit for bit, C1a for M_mag on that image set and the latent neutrality of both controls
     would make the mean difference exactly zero. The script counts the differences whose mean is not zero
     there and reports the smallest absolute value. This is an algebraic check on a fixed set, not a test.
  3. Leave one patient out for the Inv cell at the highest contrast dose of the second run, on the bit-exact
     family of all 48 differences: the band is recomputed without each patient in turn.

Out -> results/c76_route_exact_detail.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c62_route_test as c62            # noqa: E402  (pre-specified; imported only)
import c67_route_sensitivity as c67     # noqa: E402
import c74_route_exact_match as c74     # noqa: E402

RES = c62.RES
OUT = os.path.join(RES, "c76_route_exact_detail.json")
CR = c62.CRIT
TKS = c67.TKS
EPS = CR["margin_logit"]
KEYS = ["erm|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-hue4",
        "slim|color_var@2|D_rad4-rad16", "slim|contrast@2|D_rad4-rad16", "slim|contrast@0|D_rad4-rad16"]


def family(DL, cols, primary, ex, min_images):
    """The c74 construction, returned as arrays so that patients can be dropped."""
    col = {c: k for k, c in enumerate(cols)}
    names, vals, msk = [], [], []
    for tk in TKS:
        for m in c62.MODELS:
            for a, b, _ in c62.PAIRS:
                mk = primary[tk] & ex[(tk, a)] & ex[(tk, b)]
                if mk.sum() < min_images:
                    continue
                names.append("%s|%s|D_%s-%s" % (m, tk, a, b))
                vals.append(DL[m][:, col["%s|%s" % (tk, b)]] - DL[m][:, col["%s|%s" % (tk, a)]])
                msk.append(mk)
    return names, np.stack(vals), np.stack(msk)


def band(vals, msk, pids):
    est, draws = c62.boot(vals, msk, pids, CR["boot"], CR["boot_seed"])
    se = np.nanstd(draws, 0, ddof=1)
    crit = float(np.nanpercentile(np.nanmax(np.abs(draws - est) / np.maximum(se, 1e-12), 1), CR["level"]))
    return est, est - crit * se, est + crit * se, crit


def entry(k, names, est, lo, hi, msk, pids):
    if k not in names:
        return None
    j = names.index(k)
    return {"n_images": int(msk[j].sum()), "n_patients": int(len(set(pids[msk[j]]))), "est": float(est[j]),
            "sim": [float(lo[j]), float(hi[j])], "decision": c62.decide(lo[j], hi[j], EPS)}


def main():
    cons = json.load(open(c62.out_path("construction", False)))
    pids = np.asarray(cons["patient_ids"])
    g1 = c67.g1_masks(cons)
    ex, _ = c74.exact_masks(cons)
    c74r = json.load(open(os.path.join(RES, "c74_route_exact_match.json")))
    out = {"note": "exploratory; reads stored responses only", "runs": {}}
    for run, suf, evf in (("second_run", "_inv_retrained", "c62_evaluation_inv_retrained.json"),
                          ("first_run", "", "c62_evaluation.json")):
        r = c67.load_run(suf)
        common = np.all([r["prob"][m][:, 0] >= CR["score_floor"] for m in c62.MODELS], 0)
        primary = {tk: g1[tk] & common for tk in TKS}
        ev = json.load(open(os.path.join(RES, evf)))["primary"]
        R = {"decisive": {}, "check_vs_c74": {}}
        # (i) the plan's analysis, from the registered evaluation; images from the gate mask
        gate = {}
        for k in KEYS:
            m, tk, _ = k.split("|")
            c = ev["components"][k]; mk = primary[tk]
            gate[k] = {"n_images": int(mk.sum()), "n_patients": int(len(set(pids[mk]))), "est": c["est"],
                       "sim": c["sim"], "decision": c62.decide(c["sim"][0], c["sim"][1], EPS)}
        fam = {}
        for tag, mi in (("all", 0), ("min30", CR["min_images"])):
            names, vals, msk = family(r["dl"], r["cols"], primary, ex, mi)
            est, lo, hi, crit = band(vals, msk, pids)
            fam[tag] = (names, vals, msk, est, lo, hi, crit)
            dev = max(max(abs(est[j] - c74r["runs"][run][tag]["components"][n]["est"]),
                          abs(lo[j] - c74r["runs"][run][tag]["components"][n]["sim"][0])) for j, n in enumerate(names))
            R["check_vs_c74"][tag] = {"max_abs_difference": float(dev), "crit": crit}
        for k in KEYS:
            R["decisive"][k] = {"gate": gate[k],
                                "bit_exact_all48": entry(k, *[fam["all"][i] for i in (0, 3, 4, 5, 2)], pids),
                                "bit_exact_min30": entry(k, *[fam["min30"][i] for i in (0, 3, 4, 5, 2)], pids)}
        # 2. the exact identity on each difference's bit-exact image set
        names, vals, msk = fam["all"][0], fam["all"][1], fam["all"][2]
        means = np.array([vals[j][msk[j]].mean() for j in range(len(names))])
        per_image_zero = sum(int((vals[j][msk[j]] == 0).sum()) for j in range(len(names)))
        cells_nonzero = sorted({n.rsplit("|", 1)[0] for j, n in enumerate(names) if means[j] != 0})
        R["exact_identity"] = {"n_differences": len(names), "n_nonzero_mean": int((means != 0).sum()),
                               "smallest_abs_mean": float(np.abs(means).min()), "n_abs_mean_above_margin": int((np.abs(means) > EPS).sum()),
                               "per_image_zero_differences": per_image_zero,
                               "per_image_total": int(msk.sum()), "cells_with_a_nonzero_difference": len(cells_nonzero),
                               "images_range": [int(msk.sum(1).min()), int(msk.sum(1).max())]}
        # 3. leave one patient out for the second run's Inv cell, all-48 bit-exact family
        if run == "second_run":
            k = "causal|contrast@2|D_rad4-rad16"; j = names.index(k)
            inc = sorted(set(pids[msk[j]])); lop = []
            for p in inc:
                keep = pids != p
                e3, l3, h3, _ = band(vals[:, keep], msk[:, keep], pids[keep])
                lop.append({"patient": str(p), "est": float(e3[j]), "lower": float(l3[j]), "decision": c62.decide(l3[j], h3[j], EPS)})
            R["lopo_inv_all48"] = {"difference": k, "n_subsets": len(lop), "n_exceeds": sum(q["decision"] == "exceeds" for q in lop),
                                   "lower_min": min(q["lower"] for q in lop), "est_range": [min(q["est"] for q in lop), max(q["est"] for q in lop)],
                                   "subsets": lop}
        out["runs"][run] = R
        print("[%s] reproduces c74: %s" % (run, {t: "%.1e" % v["max_abs_difference"] for t, v in R["check_vs_c74"].items()}))
        for k, v in R["decisive"].items():
            print("   %-32s" % k, " | ".join("%s n=%s (%s) %s" % (a, *(("%d" % e["n_images"], "%d" % e["n_patients"],
                  "%+.2f [%.2f, %.2f] %s" % (e["est"], e["sim"][0], e["sim"][1], e["decision"])) if e else ("-", "-", "not analysed")))
                  for a, e in v.items()))
        print("   exact identity:", R["exact_identity"])
        if "lopo_inv_all48" in R:
            q = R["lopo_inv_all48"]
            print("   LOPO Inv all-48: %d/%d exceed, lower min %.3f" % (q["n_exceeds"], q["n_subsets"], q["lower_min"]))
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
