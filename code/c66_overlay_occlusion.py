#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C66: does the artifact verdict need the overlay to cover the lesion?  (exploratory, not pre-specified)

A hair strand is placed at a random position, so it can cross the lesion. A response to such an
overlay could then come from hidden lesion evidence rather than from the overlay itself. This
script separates the two on the audit cohort of Table 4.

  1. Coverage. For each held-out overlay, the share of the lesion region it alters and the share of
     its altered pixels that fall in the lesion region. The lesion region is the soft lesion mask of
     Section III-C at w >= 0.5, dilated by DIL pixels.
  2. Restricted overlays. The same realization as Table 4 (same strands, same ruler row), with every
     pixel inside the dilated lesion region restored to the unedited image ("skin only"), or with
     every pixel outside it restored ("lesion only").
  3. The artifact criterion of Section III-B on the skin-only mixture and on the lesion-only mixture:
     marginal 95% patient-clustered intervals with 4,000 resamples, as in Table 4, and the
     36-component max-statistic band with 20,000 resamples, as in Section S2.

The full mixture is recomputed as a check: it must reproduce c30_disjoint_audit.json.

Out -> results/c66_overlay_occlusion.json
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

RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c66_overlay_occlusion.json")
HELD = ["hair_light", "hair", "hair_heavy", "ruler"]
W_LESION, DIL, THR = 0.5, 3, 0.02
MARGIN, BOOT_M, BOOT_S, SEED_S = 0.05, 4000, 20000, 41
DEVICE = c30.DEVICE


def lesion_region(x):
    w, _, _ = c3m.soft_mask(x)
    L = (w >= W_LESION).float()[None, None]
    L = tF.max_pool2d(L, 2 * DIL + 1, stride=1, padding=DIL)[0, 0] > 0
    return L


def band36(vecs, masks, ismag, pid):
    """Max-statistic patient bootstrap over all components, as in c56_simultaneous36.py."""
    gi = np.unique(pid, return_inverse=True)[1]
    idx = [np.where(gi == k)[0] for k in range(gi.max() + 1)]
    rng = np.random.RandomState(SEED_S)
    est = np.array([v[m].mean() for v, m in zip(vecs, masks)])
    draws = np.empty((len(vecs), BOOT_S))
    for b in range(BOOT_S):
        sel = np.concatenate([idx[k] for k in rng.randint(0, len(idx), len(idx))])
        for j, (v, m) in enumerate(zip(vecs, masks)):
            draws[j, b] = v[sel[m[sel]]].mean()
    se = draws.std(axis=1, ddof=1)
    Z = np.abs(draws - est[:, None]) / np.maximum(se[:, None], 1e-12)
    crit = float(np.percentile(Z.max(0), 95))
    lo, hi = est - crit * se, est + crit * se
    ismag = np.asarray(ismag)
    return crit, est, lo, hi, ismag


def main():
    t0 = time.time()
    imgs, labels, pid, ids = audit_cohort.load()
    labels, pid = np.asarray(labels), np.asarray(pid)
    edits, change_frac = c2m.precompute_edits(imgs, c30.ART, c30.SEED)   # the realization of Table 4

    cover = {a: {"lesion_share_altered": [], "altered_in_lesion": [], "altered_frac": [],
                 "altered_frac_skin_only": [], "altered_frac_lesion_only": []} for a in HELD}
    region_frac = []
    bg, les = [{} for _ in imgs], [{} for _ in imgs]
    for i, x in enumerate(imgs):
        L = lesion_region(x)
        region_frac.append(float(L.float().mean()))
        for a in HELD:
            e = edits[i][a]
            d = e - x
            ch = d.abs().amax(0) > THR
            nL = max(int(L.sum()), 1)
            cover[a]["lesion_share_altered"].append(float((ch & L).sum()) / nL)
            cover[a]["altered_in_lesion"].append(float((ch & L).sum()) / max(int(ch.sum()), 1))
            cover[a]["altered_frac"].append(float(ch.float().mean()))
            eb = x + d * (~L).float().unsqueeze(0)
            el = x + d * L.float().unsqueeze(0)
            bg[i][a], les[i][a] = eb, el
            cover[a]["altered_frac_skin_only"].append(float(((eb - x).abs().amax(0) > THR).float().mean()))
            cover[a]["altered_frac_lesion_only"].append(float(((el - x).abs().amax(0) > THR).float().mean()))
    cov_sum = {a: {k: {"mean": float(np.mean(v)), "median": float(np.median(v)), "max": float(np.max(v))}
                   for k, v in d.items()} for a, d in cover.items()}
    print("lesion region: median share of the frame %.3f" % np.median(region_frac), flush=True)
    for a in HELD:
        print("  %-10s lesion altered %.4f (mean)  altered in lesion %.3f (mean)  altered frac %.4f"
              % (a, cov_sum[a]["lesion_share_altered"]["mean"], cov_sum[a]["altered_in_lesion"]["mean"],
                 cov_sum[a]["altered_frac"]["mean"]), flush=True)

    ref = json.load(open(os.path.join(RES, "c30_disjoint_audit.json")))
    vec = {}
    models = {}
    for bb, trio in c30.TRIO.items():
        for ck, kind, nm in trio:
            f = os.path.join(os.path.dirname(HERE), "checkpoints", ck + ".pt")
            m = c30.make_model(kind).to(DEVICE)
            m.load_state_dict(torch.load(f, map_location=DEVICE)); m.eval()
            p0 = c30.probs(m, imgs)
            dfull = np.stack([c30.probs(m, [edits[i][a] for i in range(len(imgs))]) - p0 for a in HELD])
            dbg = np.stack([c30.probs(m, [bg[i][a] for i in range(len(imgs))]) - p0 for a in HELD])
            dles = np.stack([c30.probs(m, [les[i][a] for i in range(len(imgs))]) - p0 for a in HELD])
            key = "%s-%s" % (bb, nm)
            vec[key] = {"full": dfull, "skin_only": dbg, "lesion_only": dles}
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            chk = float(np.abs(np.abs(dfull).mean(0).mean() - ref["models"][bb][nm]["all"]["mean"]))
            print("  [%s] scored (%.0fs); full mixture vs Table 4: |diff| %.2e" % (key, time.time() - t0, chk),
                  flush=True)
            models[key] = {"check_full_vs_c30_all": chk}

    subs = [("all", np.ones(len(labels), bool)), ("malignant", labels == 1), ("benign", labels == 0)]
    out = {"lesion_region": {"w_threshold": W_LESION, "dilation_px": DIL, "altered_threshold": THR,
                             "median_frame_share": float(np.median(region_frac)),
                             "range_frame_share": [float(np.min(region_frac)), float(np.max(region_frac))]},
           "coverage": cov_sum, "margin_p": MARGIN, "boot_marginal": BOOT_M, "boot_band": BOOT_S,
           "cells": {}, "band": {}, "checks": models}
    for variant in ("full", "skin_only", "lesion_only"):
        keys, vecs, masks, ismag = [], [], [], []
        cells = {}
        for key, dd in vec.items():
            d = dd[variant]
            mag, sgn = np.abs(d).mean(0), d.mean(0)
            for sname, msk in subs:
                mean, lo, hi = c30.boot_ci(mag[msk], pid[msk])
                smean, slo, shi = c30.boot_ci(sgn[msk], pid[msk])
                v, sv = c30.verdict(lo, hi, MARGIN), c30.verdict_signed(slo, shi, MARGIN)
                cells["%s/%s" % (key, sname)] = {"mean": mean, "ci": [lo, hi], "verdict": v,
                                                 "signed_mean": smean, "signed_ci": [slo, shi],
                                                 "signed_verdict": sv, "joint_verdict": c30.combine(sv, v),
                                                 "per_operator_mean_abs": {a: float(np.abs(d[k][msk]).mean())
                                                                           for k, a in enumerate(HELD)}}
                for comp, vv in (("mag", mag), ("sgn", sgn)):
                    keys.append("%s/%s/%s" % (key, sname, comp)); vecs.append(vv); masks.append(msk)
                    ismag.append(comp == "mag")
        crit, est, lo, hi, ism = band36(vecs, masks, ismag, pid)
        sim = {k: [float(lo[j]), float(hi[j])] for j, k in enumerate(keys)}
        jv = [c["joint_verdict"] for c in cells.values()]
        out["cells"][variant] = cells
        out["band"][variant] = {"crit_36": crit, "frontier_36": float(lo[ism].min()),
                                "magnitude_exceed_36": int((lo[ism] > MARGIN).sum()), "sim36": sim,
                                "marginal_joint_counts": {k: jv.count(k) for k in
                                                          ("exceedance", "equivalence", "indeterminate")},
                                "smallest_marginal_lower": float(min(c["ci"][0] for c in cells.values()))}
        print("%-11s marginal %s; band crit %.3f frontier %.4f, %d/18 exceed jointly"
              % (variant, out["band"][variant]["marginal_joint_counts"], crit, out["band"][variant]["frontier_36"],
                 out["band"][variant]["magnitude_exceed_36"]), flush=True)
        for k, c in cells.items():
            print("   %-22s %.3f [%.3f, %.3f] %s" % (k, c["mean"], c["ci"][0], c["ci"][1], c["joint_verdict"]),
                  flush=True)
    out["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved -> %s (%.0fs)" % (OUT, out["runtime_sec"]))


if __name__ == "__main__":
    main()
