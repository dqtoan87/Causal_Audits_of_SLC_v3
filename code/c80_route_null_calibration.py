#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C80: size and power of the cycle test under a known null (exploratory; reads stored responses only).

The pre-specified cycle test (c62_route_test.py) judges 48 differences of route sums on one simultaneous
95% max-statistic band from a patient bootstrap, with a 0.5-logit margin. Its level is nominal. With 52 to
54 patients per target, many clipped predictions and heavy tails, the realized level could differ.

This script checks the band by a double bootstrap in which the truth is known. The analysed sample of one
run is taken as the population: each difference's population value is its plug-in estimate. An outer draw
resamples patients from that population, with replacement, and relabels each drawn patient as a new
cluster. On each outer draw the plan's procedure is applied unchanged: the same masks, the same patient
bootstrap with 20,000 inner resamples, the same studentized max statistic and the same three-way rule.

Shifting every value of one difference by a constant shifts its estimate on every resample by that
constant and leaves its standard error and the critical value unchanged. So one set of outer draws gives
the behaviour of the band under any configuration of population values:

  1. joint coverage of the 48 plug-in values;
  2. family-wise false disagreement when every population difference is exactly zero, the case in which
     the conjunction the test probes holds exactly;
  3. family-wise false disagreement when every population difference sits on the margin, +0.5 with the
     sign of its estimate, the least favourable null for the practical test;
  4. the same for every configuration of signs, with the least favourable one found by coordinate ascent,
     bounded above by one minus the joint coverage;
  5. power for one difference whose population value is v, read on the plan's decisive pairs;
  6. a calibrated band: the factor kappa by which every half-width must grow for 95% joint coverage under
     this population, and the plan's decisions recomputed on the original sample with that band.

The population is the empirical distribution of the analysed patients, so the check concerns the accuracy
of the bootstrap band at this sample size and with these responses. It does not test latent neutrality,
C1a or any other condition of the route test, and a null that holds here is built by recentring, not by
constructing neutral controls.

Reads results/c63_route_scores.npz, results/c63_route_scores_inv_retrained.npz, results/c62_construction.json,
results/c62_evaluation.json and results/c62_evaluation_inv_retrained.json.
Out -> results/c80_route_null_calibration.json, results/c80_route_null_draws_{first,second}.npz
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
OUT = os.path.join(RES, "c80_route_null_calibration.json")
TKS = ["%s@%d" % t for t in c62.TARGETS]
EPS = c62.CRIT["margin_logit"]
B_IN = c62.CRIT["boot"]              # 20,000, as the plan
R_OUT = 2000
SEED_OUT = 80
V_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
RUNS = {"first": ("c63_route_scores.npz", "c62_evaluation.json"),
        "second": ("c63_route_scores_inv_retrained.npz", "c62_evaluation_inv_retrained.json")}


def band(vals, masks, pid, seed):
    est, draws = c62.boot(vals, masks, pid, B_IN, seed)
    se = np.nanstd(draws, 0, ddof=1)
    Z = np.abs(draws - est) / np.maximum(se, 1e-12)
    crit = float(np.nanpercentile(np.nanmax(Z, 1), c62.CRIT["level"]))
    return est, se, crit


def load_run(npz, cons):
    z = np.load(os.path.join(RES, npz), allow_pickle=True)
    cols = [str(c) for c in z["columns"]]; col = {c: k for k, c in enumerate(cols)}
    pids = np.asarray(z["patient_ids"]).astype(str)
    assert list(map(str, z["isic_ids"])) == cons["ids"]
    P = {m: z["prob_" + m] for m in c62.MODELS}; DL = {m: z["dlogit_" + m] for m in c62.MODELS}
    common = np.all([P[m][:, 0] >= c62.CRIT["score_floor"] for m in c62.MODELS], 0)
    n = len(pids)
    names, vals, masks = [], [], []
    for tk in TKS:
        g1 = np.array([all(cons["images"][i]["targets"][tk]["controls"][c]["g1"] for c in c62.CONTROLS)
                       for i in range(n)])
        msk = g1 & common
        for m in c62.MODELS:
            for a, b, _ in c62.PAIRS:
                names.append("%s|%s|D_%s-%s" % (m, tk, a, b))
                vals.append(DL[m][:, col["%s|%s" % (tk, b)]] - DL[m][:, col["%s|%s" % (tk, a)]])
                masks.append(msk)
    return names, np.stack(vals).astype(float), np.stack(masks), pids


def run_one(label, npz, evname, cons):
    t0 = time.time()
    names, vals, masks, pids = load_run(npz, cons)
    reg = json.load(open(os.path.join(RES, evname)))["primary"]["components"]
    est, se, crit = band(vals, masks, pids, c62.CRIT["boot_seed"])
    dev = max(max(abs(est[k] - reg[c]["est"]), abs(est[k] - crit * se[k] - reg[c]["sim"][0]),
                  abs(est[k] + crit * se[k] - reg[c]["sim"][1])) for k, c in enumerate(names))
    print("[%s] reproduction of the plan's band: max abs difference %.2e, crit %.4f" % (label, dev, crit), flush=True)
    theta = est.copy()                                  # plug-in population values
    sgn = np.where(theta >= 0, 1.0, -1.0)
    upids = np.unique(pids); G = len(upids)
    idx_of = {p: np.where(pids == p)[0] for p in upids}
    rng = np.random.RandomState(SEED_OUT)
    K = len(names)
    DEV = np.empty((R_OUT, K)); SE = np.empty((R_OUT, K)); CR = np.empty(R_OUT)
    for r in range(R_OUT):
        draw = upids[rng.randint(0, G, G)]
        rows, lab = [], []
        for j, p in enumerate(draw):
            ii = idx_of[p]; rows.append(ii); lab += ["c%03d" % j] * len(ii)
        rows = np.concatenate(rows)
        e_r, s_r, c_r = band(vals[:, rows], masks[:, rows], np.asarray(lab), SEED_OUT * 1000 + r)
        DEV[r] = e_r - theta; SE[r] = s_r; CR[r] = c_r
        if (r + 1) % 250 == 0:
            print("  [%s] outer %d/%d (%.0fs)" % (label, r + 1, R_OUT, time.time() - t0), flush=True)
    half = CR[:, None] * SE
    cover = np.abs(DEV) <= half                                     # plug-in value inside the band
    np.savez_compressed(os.path.join(RES, "c80_route_null_draws_%s.npz" % label), names=np.array(names),
                        theta=theta, dev=DEV, se=SE, crit=CR)
    # population value 0 for every difference: false disagreement if the band clears the margin
    fd0 = (DEV - half > EPS) | (DEV + half < -EPS)
    # population value on the margin, with the sign of the estimate: false disagreement if the band clears it
    sd = sgn * DEV
    fdm = (sd - half > 0) | (sd + half < -2 * EPS)
    # cells: a cell disagrees falsely when any of its four differences does
    cells = sorted(set(n.rsplit("|", 1)[0] for n in names))
    cidx = {c: [k for k, n in enumerate(names) if n.startswith(c + "|")] for c in cells}
    mcse = lambda p: float(np.sqrt(p * (1 - p) / R_OUT))
    out = {"reproduction_max_abs_difference": float(dev), "crit_plan": crit,
           "n_patients": int(G), "n_outer": R_OUT, "n_inner": B_IN,
           "crit_outer_median": float(np.median(CR)), "crit_outer_range": [float(CR.min()), float(CR.max())],
           "joint_coverage": float(cover.all(1).mean()), "joint_coverage_mcse": mcse(float(cover.all(1).mean())),
           "min_marginal_coverage": float(cover.mean(0).min()),
           "min_marginal_coverage_component": names[int(np.argmin(cover.mean(0)))],
           "fwer_all_zero": float(fd0.any(1).mean()), "fwer_all_zero_mcse": mcse(float(fd0.any(1).mean())),
           "fwer_at_margin": float(fdm.any(1).mean()), "fwer_at_margin_mcse": mcse(float(fdm.any(1).mean())),
           "mean_false_cells_at_margin": float(np.mean([sum(fdm[r, cidx[c]].any() for c in cells)
                                                         for r in range(R_OUT)])),
           "per_component_false_rate_at_margin_max": float(fdm.mean(0).max()),
           "power": {}}
    # one-sided false rates per component: population value +0.5 (up) or -0.5 (down)
    up, dn = DEV - half > 0, DEV + half < 0
    out["one_sided_rate_up_max"] = float(up.mean(0).max())
    out["one_sided_rate_down_max"] = float(dn.mean(0).max())
    out["one_sided_rate_up_max_component"] = names[int(np.argmax(up.mean(0)))]
    out["one_sided_rate_down_max_component"] = names[int(np.argmax(dn.mean(0)))]

    def fwer_sign(s):                           # every population difference at 0.5*s_k
        return float(((s > 0)[None, :] & up | (s < 0)[None, :] & dn).any(1).mean())
    cfg = {"sign_of_estimate": sgn, "against_estimate": -sgn, "all_positive": np.ones(K), "all_negative": -np.ones(K)}
    out["fwer_margin_by_sign"] = {k: fwer_sign(v) for k, v in cfg.items()}
    # coordinate ascent for the least favourable sign configuration
    best = max(cfg.values(), key=fwer_sign).copy()
    for _ in range(5):
        changed = False
        for k in range(K):
            trial = best.copy(); trial[k] = -trial[k]
            if fwer_sign(trial) > fwer_sign(best) + 1e-12:
                best, changed = trial, True
        if not changed:
            break
    out["fwer_margin_worst_found"] = fwer_sign(best)
    out["fwer_margin_upper_bound"] = float(1 - cover.all(1).mean())
    out["joint_coverage_by_model"] = {m: float(cover[:, [k for k, n in enumerate(names) if n.startswith(m + "|")]].all(1).mean())
                                      for m in c62.MODELS}
    out["noncoverage_share_by_component_top"] = sorted(
        [(names[k], float(1 - cover[:, k].mean())) for k in range(K)], key=lambda t: -t[1])[:6]
    # calibrated band: inflate every half-width by kappa until joint coverage reaches 95%
    lo_k, hi_k = 1.0, 3.0
    for _ in range(40):
        mid = 0.5 * (lo_k + hi_k)
        if (np.abs(DEV) <= mid * half).all(1).mean() >= 0.95:
            hi_k = mid
        else:
            lo_k = mid
    kappa = hi_k
    out["kappa_95"] = float(kappa)
    lo_s, hi_s = est - kappa * crit * se, est + kappa * crit * se
    dec_cal = {names[k]: c62.decide(lo_s[k], hi_s[k], EPS) for k in range(K)}
    out["calibrated_decisions_exceeds"] = sorted(k for k, v in dec_cal.items() if v == "exceeds")
    out["calibrated_lower_bounds_of_plan_exceeds"] = {names[k]: float(lo_s[k]) for k in range(K)
                                                      if reg[names[k]].get("decision_sim") == "exceeds"}
    cl = {}
    for c in cells:
        ds = [dec_cal[names[k]] for k in cidx[c]]
        cl[c] = "routes disagree" if "exceeds" in ds else ("routes agree" if all(v == "within" for v in ds) else "indeterminate")
    out["calibrated_cells"] = cl
    out["calibrated_summary"] = {k: list(cl.values()).count(k) for k in ("routes disagree", "routes agree", "indeterminate")}
    # power for a difference with population value v (others irrelevant to its own decision)
    dec = [k for k, c in enumerate(names) if reg[c].get("decision_sim") == "exceeds"]
    pw_all = {}
    for k in range(K):
        pw_all[names[k]] = {str(v): float(np.mean(v + DEV[:, k] - half[:, k] > EPS)) for v in V_GRID}
    for k in dec:
        out["power"][names[k]] = pw_all[names[k]]
    # smallest grid value at which each difference is detected with probability at least 0.8
    v80 = []
    for k in range(K):
        hit = [v for v in V_GRID if pw_all[names[k]][str(v)] >= 0.8]
        v80.append(hit[0] if hit else float("inf"))
    v80 = np.array(v80)
    out["v80_grid"] = V_GRID
    out["v80_median"] = float(np.median(v80[np.isfinite(v80)])) if np.isfinite(v80).any() else None
    out["v80_share_at_most_2"] = float(np.mean(v80 <= 2.0))
    out["v80_share_above_grid"] = float(np.mean(~np.isfinite(v80)))
    out["v80_by_model"] = {m: [float(x) for x in v80[[k for k, n in enumerate(names) if n.startswith(m + "|")]]]
                           for m in c62.MODELS}
    out["power_all"] = pw_all
    out["runtime_sec"] = round(time.time() - t0, 1)
    print("[%s] joint coverage %.3f, FWER all-zero %.3f, FWER at margin %.3f, worst sign %.3f, kappa %.3f, "
          "calibrated %s, median v80 %s (%.0fs)" % (
        label, out["joint_coverage"], out["fwer_all_zero"], out["fwer_at_margin"], out["fwer_margin_worst_found"],
        out["kappa_95"], out["calibrated_summary"], out["v80_median"], out["runtime_sec"]), flush=True)
    return out


def main():
    t0 = time.time()
    cons = json.load(open(os.path.join(RES, "c62_construction.json")))
    res = {"note": ("double bootstrap with the analysed sample as the population; exploratory; checks the "
                    "accuracy of the plan's band, not any condition of the route test"),
           "margin": EPS, "seed_outer": SEED_OUT}
    for label, (npz, ev) in RUNS.items():
        res[label] = run_one(label, npz, ev, cons)
    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved ->", OUT, flush=True)


if __name__ == "__main__":
    main()
