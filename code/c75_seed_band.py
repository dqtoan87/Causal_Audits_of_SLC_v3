#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C75: the 36-component band and its frontier for each realization of the overlay operator  (exploratory)

The headline frontier is a statement for one realization of the random overlays (seed 42), and C50 showed
that the marginal cell verdicts hold under seven realizations. This script asks the joint question: for each
of the same seven realizations, it recomputes the 36-component max-statistic band of C56, with the same
construction, resamples and seed, and reports the critical value, the frontier and the number of cells that
exceed 0.05 jointly. Seed 42 must reproduce c56_simultaneous36.json. The range over seven realizations is
descriptive; it is not an interval for the population of realizations.

Out -> results/c75_seed_band.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c2_cace_artifact as c2m      # noqa: E402
import c30_disjoint_audit as c30    # noqa: E402
import audit_cohort                 # noqa: E402

DEVICE = c30.DEVICE
BASE = os.path.dirname(HERE)
CKPT = os.path.join(BASE, "checkpoints")
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c75_seed_band.json")
HELD = c30.HELD
MARGIN = c30.MARGIN_P
SEEDS = [42, 1, 2, 3, 7, 11, 99]          # the realizations of C50
BOOT, BSEED = 20000, 41                   # the band of C56


def band(vec, labels, pid):
    """The C56 construction, verbatim."""
    subs = [("all", np.ones(len(labels), bool)), ("mal", labels == 1), ("ben", labels == 0)]
    keys, vecs, masks, ismag = [], [], [], []
    for cell, (mag, sgn) in vec.items():
        for sname, msk in subs:
            for comp, v in (("mag", mag), ("sgn", sgn)):
                keys.append("%s/%s/%s" % (cell, sname, comp)); vecs.append(v); masks.append(msk); ismag.append(comp == "mag")
    gi = np.unique(pid, return_inverse=True)[1]
    idx = [np.where(gi == k)[0] for k in range(gi.max() + 1)]
    rng = np.random.RandomState(BSEED)
    est = np.array([v[m].mean() for v, m in zip(vecs, masks)])
    draws = np.empty((len(keys), BOOT))
    for b in range(BOOT):
        sel = np.concatenate([idx[k] for k in rng.randint(0, len(idx), len(idx))])
        for j, (v, m) in enumerate(zip(vecs, masks)):
            draws[j, b] = v[sel[m[sel]]].mean()
    se = draws.std(axis=1, ddof=1)
    Z = np.abs(draws - est[:, None]) / np.maximum(se[:, None], 1e-12)
    ismag = np.array(ismag)
    crit = float(np.percentile(Z.max(0), 95))
    lo, hi = est - crit * se, est + crit * se
    mag_lo = {k.rsplit("/", 1)[0]: float(lo[j]) for j, k in enumerate(keys) if ismag[j]}
    order_ok = all(lo[j] >= max(lo[j + 1], -hi[j + 1]) and hi[j] >= max(-lo[j + 1], hi[j + 1])
                   for j in range(0, len(keys), 2))          # keys alternate mag, sgn within a cell
    return {"crit_36": crit, "frontier_36": float(lo[ismag].min()), "magnitude_exceed_36": int((lo[ismag] > MARGIN).sum()),
            "argmin_cell": min(mag_lo, key=mag_lo.get), "magnitude_lower": mag_lo, "magnitude_bounds_signed": bool(order_ok)}


def main():
    t0 = time.time()
    imgs, labels, pid, _ = audit_cohort.load()
    labels, pid = np.asarray(labels), np.asarray(pid)
    edits = {}
    for s in SEEDS:                          # keep the held-out operators only
        edits[s] = [{a: d[a] for a in HELD} for d in c2m.precompute_edits(imgs, c30.ART, s)[0]]
    print("overlays built (%.0fs)" % (time.time() - t0), flush=True)
    vec = {s: {} for s in SEEDS}
    for bb, trio in c30.TRIO.items():
        for ck, kind, nm in trio:
            m = c30.make_model(kind).to(DEVICE)
            m.load_state_dict(torch.load(os.path.join(CKPT, ck + ".pt"), map_location=DEVICE)); m.eval()
            p0 = c30.probs(m, imgs)
            for s in SEEDS:
                d = np.stack([c30.probs(m, [edits[s][i][a] for i in range(len(imgs))]) - p0 for a in HELD])
                vec[s]["%s-%s" % (bb, nm)] = (np.abs(d).mean(0), d.mean(0))
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            print("  [%s %s] scored (%.0fs)" % (bb, nm, time.time() - t0), flush=True)
    out = {"seeds": SEEDS, "boot": BOOT, "boot_seed": BSEED, "margin_p": MARGIN,
           "note": "exploratory; the C56 band recomputed for each overlay realization; seed 42 reproduces C56", "per_seed": {}}
    for s in SEEDS:
        out["per_seed"][str(s)] = band(vec[s], labels, pid)
        r = out["per_seed"][str(s)]
        print("seed %3d: crit %.3f frontier %.4f (%s) exceed@0.05 %d/18 order %s (%.0fs)"
              % (s, r["crit_36"], r["frontier_36"], r["argmin_cell"], r["magnitude_exceed_36"], r["magnitude_bounds_signed"], time.time() - t0), flush=True)
    ref = json.load(open(os.path.join(RES, "c56_simultaneous36.json")))
    out["check_seed42_vs_c56"] = {"frontier_abs_diff": abs(out["per_seed"]["42"]["frontier_36"] - ref["frontier_36"]),
                                  "crit_abs_diff": abs(out["per_seed"]["42"]["crit_36"] - ref["crit_36"])}
    fr = [v["frontier_36"] for v in out["per_seed"].values()]
    out["frontier_range"] = [min(fr), max(fr)]
    out["exceed_counts"] = [v["magnitude_exceed_36"] for v in out["per_seed"].values()]
    out["runtime_sec"] = round(time.time() - t0, 1)
    print("check vs c56:", out["check_seed42_vs_c56"], "frontier range", out["frontier_range"])
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
