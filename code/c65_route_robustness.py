#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C65: robustness of the route-test decisions (exploratory; reads stored responses, no rescoring).

The registered route test (c62_route_test.py) decides each model-by-target cell from four differences of
route sums on a simultaneous band. This script checks how firm those decisions are. It uses the per-image
responses that c63_route_exploratory.py stored, and the registered gates, score floor, bootstrap and rule.

  1. Reproduction: the 48 registered differences and their simultaneous band, recomputed from the stored
     responses with the registered seed, must match results/c62_evaluation.json.
  2. Cell frontiers: for each cell, the largest margin at which it would still disagree and the smallest
     margin at which it would agree, read from the registered simultaneous band.
  3. Bootstrap seeds: the band recomputed with 20 other seeds; decisions and decisive bounds per seed.
  4. Leave one patient out: the band recomputed 54 times, each time without one patient's images; the
     summary counts separately the subsets that drop a patient present in the cell.
  5. Cross pairs: the two differences that the registration left out (rad4 against hue16, rad16 against
     hue4), with marginal 95% intervals, reported descriptively.

Reads results/c63_route_scores.npz, results/c62_construction.json and results/c62_evaluation.json.
Out -> results/c65_route_robustness.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c62_route_test as c62         # noqa: E402  (registered bootstrap, rule and constants)

RES = c62.RES
OUT = os.path.join(RES, "c65_route_robustness.json")
TKS = ["%s@%d" % t for t in c62.TARGETS]
EPS = c62.CRIT["margin_logit"]
SEEDS = list(range(63, 83))


def band(vals, masks, pid, seed, B):
    est, draws = c62.boot(vals, masks, pid, B, seed)
    se = np.nanstd(draws, 0, ddof=1)
    Z = np.abs(draws - est) / np.maximum(se, 1e-12)
    crit = float(np.nanpercentile(np.nanmax(Z, 1), c62.CRIT["level"]))
    return est, est - crit * se, est + crit * se, crit, draws


def main():
    t0 = time.time()
    z = np.load(os.path.join(RES, "c63_route_scores.npz"), allow_pickle=True)
    cons = json.load(open(os.path.join(RES, "c62_construction.json")))
    ev = json.load(open(os.path.join(RES, "c62_evaluation.json")))
    cols = [str(c) for c in z["columns"]]; col = {c: k for k, c in enumerate(cols)}
    pids = np.asarray(z["patient_ids"]).astype(str)
    assert list(map(str, z["isic_ids"])) == cons["ids"]
    P = {m: z["prob_" + m] for m in c62.MODELS}; DL = {m: z["dlogit_" + m] for m in c62.MODELS}
    common = np.all([P[m][:, 0] >= c62.CRIT["score_floor"] for m in c62.MODELS], 0)
    n = len(pids)

    def gate(tk):
        return np.array([all(cons["images"][i]["targets"][tk]["controls"][c]["g1"] for c in c62.CONTROLS)
                         for i in range(n)]) & common

    names, vals, masks = [], [], []
    for tk in TKS:
        msk = gate(tk)
        for m in c62.MODELS:
            for a, b, _ in c62.PAIRS:
                names.append("%s|%s|D_%s-%s" % (m, tk, a, b))
                vals.append(DL[m][:, col["%s|%s" % (tk, b)]] - DL[m][:, col["%s|%s" % (tk, a)]]); masks.append(msk)
    vals, masks = np.stack(vals), np.stack(masks)

    # 1. reproduction of the registered band
    est, lo, hi, crit, _ = band(vals, masks, pids, c62.CRIT["boot_seed"], c62.CRIT["boot"])
    reg = ev["primary"]["components"]
    dev = max(max(abs(est[k] - reg[c]["est"]), abs(lo[k] - reg[c]["sim"][0]), abs(hi[k] - reg[c]["sim"][1]))
              for k, c in enumerate(names))
    res = {"note": "exploratory robustness of the registered decisions; no rescoring",
           "reproduction_max_abs_difference": float(dev), "crit_registered": crit, "decisive": {}, "cells": {}}
    print("reproduction: max abs difference %.2e, crit %.4f" % (dev, crit), flush=True)

    # 2. cell frontiers from the registered band
    for tk in TKS:
        for m in c62.MODELS:
            ks = [names.index("%s|%s|D_%s-%s" % (m, tk, a, b)) for a, b, _ in c62.PAIRS]
            exc = [min(abs(lo[k]), abs(hi[k])) if lo[k] * hi[k] > 0 else 0.0 for k in ks]
            wit = [max(abs(lo[k]), abs(hi[k])) for k in ks]
            res["cells"]["%s|%s" % (m, tk)] = {"registered_class": ev["primary"]["cells"]["%s|%s" % (m, tk)]["class"],
                                                "disagrees_below_margin": float(max(exc)),
                                                "agrees_above_margin": float(max(wit))}

    # decisive differences: those that exceed in the registered band
    dec = [k for k, c in enumerate(names) if reg[c].get("decision_sim") == "exceeds"]
    for k in dec:
        res["decisive"][names[k]] = {"est": float(est[k]), "sim_lower_unrounded": float(lo[k]),
                                     "sim_upper_unrounded": float(hi[k]), "seeds": [], "leave_one_patient_out": []}

    # 3. other bootstrap seeds
    seed_crit = []
    for s in SEEDS:
        e2, l2, h2, c2, _ = band(vals, masks, pids, s, c62.CRIT["boot"])
        seed_crit.append(c2)
        for k in dec:
            res["decisive"][names[k]]["seeds"].append({"seed": s, "lower": float(l2[k]), "decision": c62.decide(l2[k], h2[k], EPS)})
    res["seed_crit_range"] = [float(min(seed_crit)), float(max(seed_crit))]
    print("seeds done (%.0fs)" % (time.time() - t0), flush=True)

    # 4. leave one patient out
    up = sorted(set(pids))
    for p in up:
        keep = pids != p
        e3, l3, h3, c3, _ = band(vals[:, keep], masks[:, keep], pids[keep], c62.CRIT["boot_seed"], c62.CRIT["boot"])
        for k in dec:
            res["decisive"][names[k]]["leave_one_patient_out"].append(
                {"patient": p, "in_cell": bool(masks[k][pids == p].any()), "est": float(e3[k]), "lower": float(l3[k]),
                 "decision": c62.decide(l3[k], h3[k], EPS)})
    print("leave-one-patient-out done (%.0fs)" % (time.time() - t0), flush=True)
    for k in dec:
        d = res["decisive"][names[k]]
        d["seed_exceeds"] = sum(q["decision"] == "exceeds" for q in d["seeds"])
        d["seed_lower_range"] = [min(q["lower"] for q in d["seeds"]), max(q["lower"] for q in d["seeds"])]
        d["lopo_exceeds"] = sum(q["decision"] == "exceeds" for q in d["leave_one_patient_out"])
        d["lopo_n"] = len(d["leave_one_patient_out"])
        d["lopo_lower_min"] = min(q["lower"] for q in d["leave_one_patient_out"])
        inc = [q for q in d["leave_one_patient_out"] if q["in_cell"]]
        d["lopo_in_cell_n"] = len(inc)
        d["lopo_in_cell_exceeds"] = sum(q["decision"] == "exceeds" for q in inc)
        d["lopo_in_cell_lower_min"] = min(q["lower"] for q in inc)

    # 5. cross pairs, descriptive
    res["cross_pairs"] = {}
    xn, xv, xm = [], [], []
    for tk in TKS:
        msk = gate(tk)
        for m in c62.MODELS:
            for a, b in (("rad4", "hue16"), ("rad16", "hue4")):
                xn.append("%s|%s|D_%s-%s" % (m, tk, a, b))
                xv.append(DL[m][:, col["%s|%s" % (tk, b)]] - DL[m][:, col["%s|%s" % (tk, a)]]); xm.append(msk)
    xe, xd = c62.boot(np.stack(xv), np.stack(xm), pids, c62.CRIT["boot"], c62.CRIT["boot_seed"])
    xl, xh = np.nanpercentile(xd, 2.5, 0), np.nanpercentile(xd, 97.5, 0)
    for k, c in enumerate(xn):
        res["cross_pairs"][c] = {"est": float(xe[k]), "ci": [float(xl[k]), float(xh[k])],
                                 "marginal_decision": c62.decide(xl[k], xh[k], EPS)}
    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    for k in dec:
        d = res["decisive"][names[k]]
        print("%-28s est %+.3f lower %.4f | seeds exceed %d/%d lower %.3f..%.3f | LOPO exceed %d/%d min lower %.3f | in cell %d/%d min %.3f" % (
            names[k], d["est"], d["sim_lower_unrounded"], d["seed_exceeds"], len(SEEDS), *d["seed_lower_range"],
            d["lopo_exceeds"], d["lopo_n"], d["lopo_lower_min"], d["lopo_in_cell_exceeds"], d["lopo_in_cell_n"], d["lopo_in_cell_lower_min"]))
    print("seed crit range", res["seed_crit_range"])
    for c, v in res["cells"].items():
        print("  %-22s %-16s disagree below %.2f, agree above %.2f" % (c, v["registered_class"], v["disagrees_below_margin"], v["agrees_above_margin"]))
    inv = {c: v for c, v in res["cross_pairs"].items() if c.startswith("causal|")}
    print("Inv cross pairs:", {c: (round(v["est"], 3), [round(x, 3) for x in v["ci"]]) for c, v in inv.items()})
    print("saved ->", OUT, "(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
