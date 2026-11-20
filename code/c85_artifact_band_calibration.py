#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C85: size of the 36-component artifact band under a known population, by a double bootstrap.

The artifact tier reads its family-wide statement from a studentized max-statistic patient bootstrap over
36 components (c56: six checkpoints x three subgroups x signed and magnitude). Its joint coverage is nominal.
This post hoc check treats the 100 audit patients as the population, so each component's population value
is its estimate on them, and reruns c56's whole band construction on cohorts resampled from that population:
  outer: 2,000 cohorts of 100 patients drawn with replacement, each drawn patient relabelled as a new cluster
  inner: 20,000 patient resamples per cohort, studentized max statistic, 95th percentile, as in c56
It reports
  1. joint two-sided coverage of the 36 population values, and coverage of each component's own interval;
  2. one-sided joint coverage of the 18 magnitude lower ends, which is what the frontier claim uses;
  3. kappa, the factor by which every half-width must grow for 95% joint coverage (two-sided and lower-end),
     and the frontier and exceedance count at 0.05 of c56's band widened by that factor.

The per-image shifts are recomputed exactly as c56 computes them (same cohort, checkpoints, overlays and seed)
and cached in results/c85_artifact_vectors.npz; the original-sample band is checked against c56_simultaneous36.json.
Out -> results/c85_artifact_band_calibration.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c85_artifact_band_calibration.json")
VEC = os.path.join(RES, "c85_artifact_vectors.npz")
MARGIN, INNER, OUTER, SEED_IN, SEED_OUT = 0.05, 20000, 2000, 41, 85
LEVEL = 95


def compute_vectors():
    import torch
    import audit_cohort
    import c2_cace_artifact as c2m
    import c30_disjoint_audit as c30
    import c56_simultaneous36 as c56
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs, labels, pid, _ = audit_cohort.load()
    edits, _ = c2m.precompute_edits(imgs, c30.ART, c30.SEED)
    names, mags, sgns = [], [], []
    for bb, trio in c30.TRIO.items():
        for ck, kind, nm in trio:
            m = c30.make_model(kind).to(dev)
            m.load_state_dict(torch.load(os.path.join(BASE, "checkpoints", ck + ".pt"), map_location=dev)); m.eval()
            p0 = c30.probs(m, imgs)
            d = np.stack([c30.probs(m, [edits[i][a] for i in range(len(imgs))]) - p0 for a in c56.HELD])
            names.append("%s-%s" % (bb, nm)); mags.append(np.abs(d).mean(0)); sgns.append(d.mean(0))
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()
            print("  [%s %s] done" % (bb, nm), flush=True)
    np.savez(VEC, names=np.array(names), mag=np.array(mags), sgn=np.array(sgns),
             labels=np.asarray(labels), pid=np.asarray(pid))


def components(z):
    labels, pid = z["labels"], z["pid"]
    subs = [("all", np.ones(len(labels), bool)), ("mal", labels == 1), ("ben", labels == 0)]
    keys, vals, masks, ismag = [], [], [], []
    for c, nm in enumerate(z["names"]):
        for sname, msk in subs:
            for comp, v in (("mag", z["mag"][c]), ("sgn", z["sgn"][c])):
                keys.append("%s/%s/%s" % (nm, sname, comp)); vals.append(v); masks.append(msk); ismag.append(comp == "mag")
    return keys, np.array(vals), np.array(masks), np.array(ismag), pid


def cluster_sums(vals, masks, gi, n):
    """Per-cluster sums and counts of every component: arrays (components, clusters)."""
    S = np.zeros((len(vals), n)); N = np.zeros((len(vals), n))
    for j in range(len(vals)):
        np.add.at(S[j], gi[masks[j]], vals[j][masks[j]])
        np.add.at(N[j], gi[masks[j]], 1.0)
    return S, N


def band(S, N, rng, inner=INNER, chunk=2000):
    """c56's construction from cluster sums: estimate, bootstrap SE and the studentized max critical value."""
    n = S.shape[1]
    est = S.sum(1) / N.sum(1)
    draws = []
    for b0 in range(0, inner, chunk):
        W = np.stack([np.bincount(rng.randint(0, n, n), minlength=n) for _ in range(min(chunk, inner - b0))]).astype(float)
        draws.append((W @ S.T) / np.maximum(W @ N.T, 1e-12))          # (draws, components)
    D = np.concatenate(draws)
    se = D.std(axis=0, ddof=1)
    Z = np.abs(D - est) / np.maximum(se, 1e-12)
    return est, se, float(np.percentile(Z.max(1), LEVEL))


def main():
    t0 = time.time()
    if not os.path.exists(VEC):
        compute_vectors()
    z = np.load(VEC, allow_pickle=False)
    keys, vals, masks, ismag, pid = components(z)
    gi = np.unique(pid, return_inverse=True)[1]; n = gi.max() + 1
    S, N = cluster_sums(vals, masks, gi, n)

    # original sample: must reproduce c56 (same construction; the resampling scheme differs, so the
    # critical value agrees to Monte Carlo error only)
    est0, se0, crit0 = band(S, N, np.random.RandomState(SEED_IN))
    c56 = json.load(open(os.path.join(RES, "c56_simultaneous36.json")))
    est56 = np.array([c56["components"][k]["est"] for k in keys])
    theta = est0                                                       # population values
    rng = np.random.RandomState(SEED_OUT)
    r2, rlo, cover_each, crits = [], [], [], []
    for o in range(OUTER):
        take = rng.randint(0, n, n)                                    # relabel: each draw is a new cluster
        So, No = S[:, take], N[:, take]
        keep = No.sum(1) > 0
        est, se, crit = band(So, No, np.random.RandomState(SEED_IN + 1 + o))
        hw = crit * np.maximum(se, 1e-12)
        dev = (est - theta) / hw
        r2.append(np.nanmax(np.abs(dev[keep]))); rlo.append(np.nanmax(dev[ismag & keep]))
        cover_each.append(np.abs(dev) <= 1); crits.append(crit)
        if (o + 1) % 200 == 0:
            print("  outer %d/%d  joint so far %.3f  (%.0fs)" % (o + 1, OUTER, np.mean(np.array(r2) <= 1), time.time() - t0), flush=True)
    r2, rlo, crits = np.array(r2), np.array(rlo), np.array(crits)
    cover_each = np.array(cover_each).mean(0)
    k2, klo = float(np.percentile(r2, LEVEL)), float(np.percentile(rlo, LEVEL))

    def widened(kappa):
        lo = est0 - kappa * crit0 * se0
        return float(lo[ismag].min()), int((lo[ismag] > MARGIN).sum())
    out = {"note": ("Post hoc double bootstrap of the 36-component artifact band (c56). Population = the 100 audit "
                    "patients; outer cohorts resample patients with replacement and relabel each draw as a new "
                    "cluster; inner band = c56's studentized max statistic. Lower-end coverage concerns the 18 "
                    "magnitude components, whose lower ends carry the frontier claim."),
           "outer": OUTER, "inner": INNER, "seed_outer": SEED_OUT, "seed_inner": SEED_IN, "level": LEVEL,
           "original_crit": crit0, "c56_crit": c56["crit_36"], "max_abs_est_diff_vs_c56": float(np.abs(est0 - est56).max()),
           "original_frontier": widened(1.0)[0], "c56_frontier": c56["frontier_36"],
           "joint_coverage_two_sided": float(np.mean(r2 <= 1)), "joint_coverage_mag_lower": float(np.mean(rlo <= 1)),
           "min_marginal_coverage": float(cover_each.min()), "median_crit_outer": float(np.median(crits)),
           "kappa_two_sided_95": k2, "kappa_mag_lower_95": klo,
           "frontier_kappa_two_sided": widened(k2)[0], "exceed_at_0p05_kappa_two_sided": widened(k2)[1],
           "frontier_kappa_mag_lower": widened(klo)[0], "exceed_at_0p05_kappa_mag_lower": widened(klo)[1],
           "runtime_sec": round(time.time() - t0, 1)}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
