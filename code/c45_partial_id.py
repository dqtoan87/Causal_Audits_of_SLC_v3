#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C45: does footprint matching select the right controls?

The bias statement in Section III-D requires the control to match the shrinkage coefficient of the
concept edit. That is a property of the response function, not of the perturbation, so it can be
checked directly: fit the same form for every control, then ask whether the concept edit's
coefficient lies within the range the controls span.

The check has power only when the control set is large enough. With one element, min = max and
"not bracketed" is inevitable, not a finding. The script therefore reports three nested sets and
the size of each, so the reader sees how the conclusion depends on the power of the check:

    A1+A3   footprint balance and overlap   -> very small set, the check has almost no power
    A3      overlap only
    all     the whole library

Also reports the |SMD| of the controls that form the edges, for comparison with the 0.10 balance threshold.

Reads results/c12_preds.npz, so no GPU is needed.
Out -> results/c45_bracketing.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import audit_cohort                 # noqa: E402

BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c45_bracketing.json")
EPS = 1e-6
ORDER = [("erm", "ERM"), ("slim", "Slim"), ("causal", "Inv")]
TARGETS = ["contrast@2", "color_var@2", "contrast@0", "color_var@0"]
SEED, BOOT = 11, 2000


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ols(y, x):
    """Δℓ = −α·L₀ + β. Return (α, β, R²)."""
    b, a = np.polyfit(x, y, 1)
    r2 = 1 - np.var(y - (b * x + a)) / np.var(y) if np.var(y) > 0 else float("nan")
    return -b, a, r2


def boot_ab(y, x, groups, n_boot=BOOT, seed=SEED):
    gi = np.unique(groups, return_inverse=True)[1]
    idx = [np.where(gi == k)[0] for k in range(gi.max() + 1)]
    rng = np.random.RandomState(seed)
    A, B = [], []
    for _ in range(n_boot):
        sel = np.concatenate([idx[k] for k in rng.randint(0, len(idx), len(idx))])
        if np.var(x[sel]) <= 0:
            continue
        a, b, _ = ols(y[sel], x[sel]); A.append(a); B.append(b)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return q(A), q(B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=BOOT)
    args = ap.parse_args()

    z = np.load(os.path.join(RES, "c12_preds.npz"), allow_pickle=True)
    names = [str(x) for x in z["names"]]
    img_index = z["img_index"]
    _, _, pid_all, _ = audit_cohort.load()
    pid = np.array(pid_all)[img_index]

    scr = json.load(open(os.path.join(RES, "c25_screen_full.json")))["design"]

    res = {"n_images": int(len(pid)), "n_patients": int(len(set(pid))),
           "balance_tol": 0.10, "overlap_tol": 0.95,
           "note": ("Does the concept edit's response function lie inside the range the controls "
                    "span? Reported over three nested control sets so that the power of the test "
                    "is visible: with one or two admissible controls the test cannot pass by "
                    "construction."),
           "targets": {}}

    print(f"{len(pid)} images / {len(set(pid))} patients\n")
    print("%-14s %-18s %4s  %s" % ("target", "control set", "|C|", "α/β bracketed by model"))
    for t in TARGETS:
        D = {k: v for k, v in scr.get(t, {}).items()
             if isinstance(v, dict) and "max_abs_smd" in v}
        sets = {
            "A1+A3": [c for c, v in D.items()
                      if v["max_abs_smd"] <= 0.10 and v.get("positivity", 1.0) >= 0.95],
            "A3": [c for c, v in D.items() if v.get("positivity", 1.0) >= 0.95],
            "all": list(D)}
        rec = {"sets": {}}
        for lab, C in sets.items():
            C = [c for c in C if f"dl_erm|{t}|{c}" in z]
            cell = {"controls": C, "n": len(C), "models": {}}
            line = []
            for m, nm in ORDER:
                L0 = logit(z[f"p0_{m}"])
                ae, be, _ = ols(z[f"dl_{m}|{t}"], L0)
                AB = {c: ols(z[f"dl_{m}|{t}|{c}"], L0)[:2] for c in C}
                if not AB:
                    line.append("n/a"); continue
                A = [v[0] for v in AB.values()]; B = [v[1] for v in AB.values()]
                ba = bool(min(A) <= ae <= max(A)); bb = bool(min(B) <= be <= max(B))
                d = {c: be - AB[c][1] for c in C}
                lo, hi = min(d, key=d.get), max(d, key=d.get)
                cell["models"][nm] = {
                    "concept_alpha": ae, "concept_beta": be,
                    "control_alpha_range": [min(A), max(A)],
                    "control_beta_range": [min(B), max(B)],
                    "alpha_bracketed": ba, "beta_bracketed": bb,
                    "both": bool(ba and bb),
                    "edge_controls": [lo, hi],
                    "edge_control_smd": [D[lo]["max_abs_smd"], D[hi]["max_abs_smd"]]}
                line.append(f"{'Y' if ba else 'N'}/{'Y' if bb else 'N'}")
            rec["sets"][lab] = cell
            print("%-14s %-18s %4d  %s" % (t, lab, len(C), "  ".join(line)))
        res["targets"][t] = rec
        print()

    summ = {}
    for lab in ["A1+A3", "A3", "all"]:
        both = sum(1 for t in res["targets"].values()
                   for c in t["sets"][lab]["models"].values() if c["both"])
        ns = [t["sets"][lab]["n"] for t in res["targets"].values()]
        smds = [s_ for t in res["targets"].values()
                for c in t["sets"][lab]["models"].values() if c["both"]
                for s_ in c["edge_control_smd"]]
        summ[lab] = {"set_size_range": [min(ns), max(ns)], "cells_bracketed": both,
                     "edge_control_smd_range": [min(smds), max(smds)] if smds else None}
    res["summary"] = summ
    json.dump(res, open(OUT, "w"), indent=1)
    for lab, v in summ.items():
        print("%-6s |C| %s  bracketed %2d/12  edge |SMD| %s"
              % (lab, v["set_size_range"], v["cells_bracketed"],
                 [round(x, 2) for x in v["edge_control_smd_range"]] if v["edge_control_smd_range"] else "n/a"))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
