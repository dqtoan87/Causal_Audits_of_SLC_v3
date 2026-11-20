#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C55: the audit graph whose connectivity decides whether the concept component is identified.

Statement 1 of the paper says the concept component is identified from the performed-edit library
exactly when the concept edit's concept value is connected to the null concept value in a bipartite
graph whose two vertex sets are the distinct footprints and the distinct concept shifts realized by
the library, with one edge per performed edit.

The (A1)-(A2) screen checks only the shortest such route, a single control sharing the concept
edit's footprint and moving no concept. A longer route through intermediate edits would identify
the component just as well, so the screen result alone does not settle the question. This script
settles it, by forming all-pairs comparisons over the whole library and reading off the components.

Two graphs have to be kept apart and this script computes on the second while reporting about the
first. G0 is the exact identification graph the theorem is stated on: an edge joins two edits only
when their footprints, or their concept changes, are equal. Exact equality of a six-component
footprint holds for no two distinct constructions, so G0 is disconnected here by inspection and its
computation is not the interesting one. G_eps is a diagnostic graph. It admits a step when the
match is close, and it measures how close the library comes to identification. It carries no
identification claim of its own without a smoothness assumption on the generic response, which the
last part of this script tests and finds unsupported.

Every step of a candidate path is a match that misses by some amount, scored as a multiple of its
own tolerance: a footprint step by its maximum |SMD| over the six components divided by 0.10, a
concept step by the gap between the two proxy shifts relative to the concept edit's own shift,
divided by 0.15. Two summaries of a path are reported. The bottleneck is its worst step, which says
whether any route is admissible under the screen's own tolerances. The total is the sum of its
steps, which is what a Lipschitz bound on the generic response would multiply to give an interval
half-width, so the minimum-total route is the one a partial-identification argument would use.

The length-two route is the placebo estimator, and its bottleneck is the worse of the (A1) and (A2)
multiples of a single control, so this computation contains the screen of Section IV-B and extends
it to chains the screen cannot see. Comparing concept shifts through the image-derived proxy is
generous, since the proxy cannot see a latent move it misses, which makes a negative answer
conservative.

Out -> results/c55_identification_graph.json
Run:  .venv/bin/python Causal_XAI_v3/code/c55_identification_graph.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c12_matched_control as c12   # noqa: E402
import c25_screen_full_sample as c25  # noqa: E402
import audit_cohort                # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c55_identification_graph.json")

COVS = c12.COVS
SMD_TOL = c12.SMD_TOL              # 0.10, the (A1) balance tolerance
NEUTRAL_TOL = c12.NEUTRAL_TOL      # 0.10, absolute proxy movement counted as null
PROXY_TOL = 0.10                   # two edits share a concept vertex within this proxy distance
TARGETS = c12.TARGETS


CACHE = os.path.join(RES, "c55_design_cache.npz")
BOOT = 5000   # route search is re-run inside every resample
C34 = os.path.join(RES, "c34_ctrl_vs_ctrl.json")


def route_split(start, n, fcost, ccost, is_null):
    """Minimum-total route, returning the total split into its footprint and concept parts.

    States carry (edit, next step type, footprint total, concept total). Dijkstra on the sum, with
    the two components tracked alongside so the bound can be written as L_g times one part plus
    kappa*L_h times the other.
    """
    INF = float("inf")
    best = {(start, 0): (0.0, 0.0, 0.0)}
    frontier, seen, reached = [(0.0, 0.0, 0.0, start, 0)], set(), (INF, INF, INF)
    while frontier:
        frontier.sort()
        c, cm, cc, e, k = frontier.pop(0)
        if (e, k) in seen or c >= reached[0]:
            continue
        seen.add((e, k))
        if k == 1 and is_null[e]:
            if c < reached[0]:
                reached = (c, cm, cc)
            continue
        if k == 0:
            for b in range(n):
                if b == e:
                    continue
                c2, m2 = c + fcost[e][b], cm + fcost[e][b]
                if c2 < best.get((b, 1), (INF,))[0]:
                    best[(b, 1)] = (c2, m2, cc)
                    frontier.append((c2, m2, cc, b, 1))
        else:
            for b in range(n):
                if b == e:
                    continue
                c2, k2 = c + ccost[e][b], cc + ccost[e][b]
                if c2 < best.get((b, 0), (INF,))[0]:
                    best[(b, 0)] = (c2, cm, k2)
                    frontier.append((c2, cm, k2, b, 0))
    return reached


def route(start, n, fcost, ccost, is_null, mode):
    """Best alternating path from the concept edit to a null-concept edit.

    mode 'max' returns the minimum over paths of the worst step, a bottleneck shortest path.
    mode 'sum' returns the minimum over paths of the total, an ordinary shortest path. A state is
    an edit together with the type of step that must come next; the walk starts at the concept edit
    needing a footprint step, since that edit supplies the first edge of the bipartite path.
    """
    INF = float("inf")
    comb = max if mode == "max" else (lambda a, b: a + b)
    best = {(start, 0): 0.0}
    seen, frontier, reached = set(), [(0.0, start, 0)], INF
    while frontier:
        frontier.sort()
        c, e, k = frontier.pop(0)
        if (e, k) in seen or c >= reached:
            continue
        seen.add((e, k))
        if k == 1 and is_null[e]:
            reached = min(reached, c)
            continue
        cost, nxt = (fcost, 1) if k == 0 else (ccost, 0)
        row = cost[e]
        for b in range(n):
            if b == e:
                continue
            c2 = comb(c, row[b])
            if c2 < best.get((b, nxt), INF):
                best[(b, nxt)] = c2
                frontier.append((c2, b, nxt))
    return reached


def smd_matrix(mean, var):
    """Max over the six components of |SMD| between every pair of edits."""
    d = mean[:, None, :] - mean[None, :, :]
    sd = np.sqrt(0.5 * (var[:, None, :] + var[None, :, :]))
    with np.errstate(divide="ignore", invalid="ignore"):
        # zero pooled variance: distance 0 only if the means agree (c71_rule_checks finds no such pair)
        z = np.where(sd > 1e-12, np.abs(d) / sd, np.where(d == 0.0, 0.0, np.inf))
    return z.max(axis=2)


def load_design():
    z = np.load(CACHE, allow_pickle=True)
    names = sorted(str(x) for x in z["names"])
    FP = np.stack([np.stack([z["fp_%s_%s" % (a, k)] for k in COVS], 0) for a in names], 0)
    PRX = np.stack([z["prx_%s" % a] for a in names], 0)
    return names, FP, PRX, int(z["n_images"])


def main():
    t0 = time.time()
    print("##### C55 identification graph #####", flush=True)
    if not os.path.exists(CACHE):
        imgs01, labels, _pid, _ids = audit_cohort.load()
        order = np.arange(len(labels))
        print(f"{len(order)} images; rebuilding the edit library", flush=True)
        _design, n = c25.build_design(imgs01, order)
        FPd, PRXd = c25.build_design.FP, c25.build_design.PRX
        nm = sorted(FPd)
        np.savez_compressed(CACHE, names=np.array(nm), n_images=n,
                            **{"fp_%s_%s" % (a, k): np.asarray(FPd[a][k], float)
                               for a in nm for k in COVS},
                            **{"prx_%s" % a: np.asarray(PRXd[a], float) for a in nm})
    names, FP, PRX, n = load_design()
    N, n_img = len(names), FP.shape[2]
    idx = {a: i for i, a in enumerate(names)}
    print(f"library holds {N} distinct performed edits over {n_img} images", flush=True)

    fmean, fvar = FP.mean(axis=2), FP.var(axis=2, ddof=1)
    smd = smd_matrix(fmean, fvar)
    fcost = smd / SMD_TOL
    prx = PRX.mean(axis=1)

    # G0, the exact identification graph the theorem is stated on
    exact_pairs = int(((smd == 0.0).sum() - N) // 2)
    exact_list = [[names[i], names[j]] for i in range(N) for j in range(i + 1, N)
                  if smd[i, j] == 0.0]

    out_targets = {}
    for t in TARGETS:
        ti = idx[t]
        scale = max(abs(prx[ti]), 1e-9)
        ccost = np.abs(prx[:, None] - prx[None, :]) / scale / c12.NEUTRAL_RATIO
        is_null = (np.abs(prx) <= NEUTRAL_TOL) & (np.abs(prx) / scale <= c12.NEUTRAL_RATIO)
        is_null[ti] = False
        r_max = route(ti, N, fcost, ccost, is_null, "max")
        r_sum, sum_M, sum_C = route_split(ti, N, fcost, ccost, is_null)
        two = float(min([fcost[ti][b] for b in range(N) if b != ti and is_null[b]] or [np.inf]))
        out_targets[t] = {
            "bottleneck_over_all_paths": float(r_max),
            "total_mismatch_best_route": float(r_sum),
            "bottleneck_length_two": two,
            "identified_within_tolerance": bool(r_max <= 1.0),
            "n_null_edits": int(is_null.sum()),
            "proxy_shift": float(prx[ti]),
            "total_footprint_part": float(sum_M),
            "total_concept_part": float(sum_C),
            "G0_footprint_vertex_degree": int((smd[ti] == 0.0).sum()),
            "G0_concept_vertex_degree": int(sum(1 for b in range(N)
                                                if abs(prx[b] - prx[ti]) == 0.0)),
        }
        print(f"  {t:14s} bottleneck {r_max:6.2f}x, total mismatch {r_sum:6.2f} tolerance units, "
              f"length-two {two:6.2f}x", flush=True)

    # patient bootstrap of the bottleneck, since one target sits close to the threshold
    z = np.load(CACHE, allow_pickle=True)
    imgs01, labels, pid, _ids = audit_cohort.load()
    pid = np.asarray(pid)
    upat = sorted(set(pid.tolist()))
    by_pat = [np.where(pid == p)[0] for p in upat]
    rng = np.random.RandomState(42)
    boots = {t: [] for t in TARGETS}
    for b in range(BOOT):
        take = np.concatenate([by_pat[i] for i in rng.randint(0, len(by_pat), len(by_pat))])
        fm, fv = FP[:, :, take].mean(axis=2), FP[:, :, take].var(axis=2, ddof=1)
        fc = smd_matrix(fm, fv) / SMD_TOL
        pb = PRX[:, take].mean(axis=1)
        for t in TARGETS:
            ti = idx[t]
            sc = max(abs(pb[ti]), 1e-9)
            cc = np.abs(pb[:, None] - pb[None, :]) / sc / c12.NEUTRAL_RATIO
            isn = (np.abs(pb) <= NEUTRAL_TOL) & (np.abs(pb) / sc <= c12.NEUTRAL_RATIO)
            isn[ti] = False
            boots[t].append(route(ti, N, fc, cc, isn, "max"))
        if (b + 1) % 100 == 0:
            print(f"   ... bootstrap {b+1}/{BOOT} ({time.time()-t0:.0f}s)", flush=True)
    for t in TARGETS:
        v = np.array(boots[t], float)
        out_targets[t]["bottleneck_ci95"] = [float(np.percentile(v, 2.5)),
                                             float(np.percentile(v, 97.5))]
        out_targets[t]["boot_frac_at_or_below_one"] = float((v <= 1.0).mean())
        print(f"  {t:14s} bottleneck 95% CI [{out_targets[t]['bottleneck_ci95'][0]:.2f}, "
              f"{out_targets[t]['bottleneck_ci95'][1]:.2f}], "
              f"P(<=1) = {out_targets[t]['boot_frac_at_or_below_one']:.3f}", flush=True)

    # What Lipschitz constant the control-versus-control comparisons would force. Under C1 with a
    # Lipschitz g, two concept-neutral controls whose footprints sit a distance d apart can differ
    # by at most L*d, so each pair puts a floor under L and the largest floor binds. Distances are
    # expressed in tolerance units, that is max|SMD| divided by 0.10, and L is therefore in logits
    # per tolerance unit. The same units carry the route totals and the interval half-widths.
    pairs34 = json.load(open(C34))["pairs"]
    # bootstrap the footprint distance of every pair so the floor carries uncertainty
    pair_keys = []
    for pr in pairs34:
        a, b = "%s|%s" % (pr["target"], pr["a"]), "%s|%s" % (pr["target"], pr["b"])
        pair_keys.append((a, b) if (a in idx and b in idx) else None)
    dboot = {k: [] for k in range(len(pairs34)) if pair_keys[k]}
    rng2 = np.random.RandomState(7)
    for _ in range(BOOT):
        take = np.concatenate([by_pat[i2] for i2 in rng2.randint(0, len(by_pat), len(by_pat))])
        fm, fv = FP[:, :, take].mean(axis=2), FP[:, :, take].var(axis=2, ddof=1)
        sb = smd_matrix(fm, fv)
        for k, pk in enumerate(pair_keys):
            if pk:
                dboot[k].append(float(sb[idx[pk[0]], idx[pk[1]]]))
    lip = []
    for k, pr in enumerate(pairs34):
        if not pair_keys[k]:
            continue
        a, b = pair_keys[k]
        d_tol = float(smd[idx[a], idx[b]]) / SMD_TOL
        if d_tol <= 1e-9:
            continue
        d_hi = float(np.percentile(dboot[k], 97.5)) / SMD_TOL
        ci = pr.get("ci_patient", [pr["diff"], pr["diff"]])
        num_lo = 0.0 if ci[0] <= 0.0 <= ci[1] else min(abs(ci[0]), abs(ci[1]))
        lip.append({"model": pr["model"], "target": pr["target"], "a": pr["a"], "b": pr["b"],
                    "abs_diff_logits": abs(pr["diff"]),
                    "abs_diff_lower_patient": num_lo,
                    "distance_tolerance_units": d_tol,
                    "distance_upper_tolerance_units": d_hi,
                    "implied_L": abs(pr["diff"]) / d_tol,
                    "implied_L_conservative": num_lo / d_hi})
    Ls = sorted(x["implied_L"] for x in lip)
    Lc = sorted(x["implied_L_conservative"] for x in lip)
    Lmin, Lmed = Ls[-1], float(np.median(Ls))
    worst = max(lip, key=lambda x: x["implied_L"])
    second = sorted(lip, key=lambda x: -x["implied_L"])[1]
    Lcons = Lc[-1]
    # The bound splits by side: half-width = L_g * (footprint total) + kappa*L_h * (concept total).
    # Only L_g is bounded below by the data, so the footprint part is what the audit can assert;
    # the concept part is a free sensitivity parameter. We therefore report the footprint-only
    # half-width, and the break-even concept-side constant at which the interval stops being
    # narrower than the full range of the corrected scores it would qualify.
    # The range is the span of the corrected scores at the four targets on the three models:
    # the point estimate of the concept edit minus the hue placebo, on the images all three
    # models score at 0.1 or above, read from c12 rather than fixed here, so that it tracks the
    # responses of the same audit cohort. (c7 stores [mean, lower, upper] at the top dose only,
    # so it cannot supply the four targets.)
    _c12 = json.load(open(os.path.join(RES, "c12_matched_control.json")))["models"]
    _cells = [float(_m["dcace"][_t]["u:hue"]["mean"]) for _m in _c12.values() for _t in TARGETS]
    assert len(_cells) == 3 * len(TARGETS), len(_cells)
    SCORE_RANGE = float(max(_cells) - min(_cells))
    SCORE_ENDS = [float(min(_cells)), float(max(_cells))]
    print(f"  corrected-score range over {len(_cells)} target cells: "
          f"{min(_cells):.3f} to {max(_cells):.3f}, span {SCORE_RANGE:.3f}", flush=True)
    hw_fp = {t: Lmin * out_targets[t]["total_footprint_part"] for t in TARGETS}
    hw_fp_loo = {t: second["implied_L"] * out_targets[t]["total_footprint_part"] for t in TARGETS}
    hw_fp_cons = {t: Lcons * out_targets[t]["total_footprint_part"] for t in TARGETS}
    # The interval [S - hw, S + hw] has full width 2*hw, so it is narrower than the span of the
    # corrected scores only while hw <= SCORE_RANGE / 2. The half-width criterion is kept as a
    # separate field for transparency; the full-width one is what the paper reports.
    breakeven, breakeven_hw = {}, {}
    for t in TARGETS:
        cpart = out_targets[t]["total_concept_part"]
        rem = SCORE_RANGE / 2.0 - hw_fp[t]
        breakeven[t] = (rem / cpart) if (cpart > 1e-9 and rem > 0) else 0.0
        rem2 = SCORE_RANGE - hw_fp[t]
        breakeven_hw[t] = (rem2 / cpart) if (cpart > 1e-9 and rem2 > 0) else 0.0
    print(f"\n  binding pair {worst['model']}/{worst['target']} {worst['a']}-{worst['b']}: "
          f"{worst['abs_diff_logits']:.2f} logits over "
          f"{worst['distance_tolerance_units']:.2f} scored units -> L_g >= {Lmin:.2f}", flush=True)
    print(f"  second largest floor {second['implied_L']:.2f}, median {Lmed:.2f}, "
          f"conservative floor {Lcons:.2f}", flush=True)
    for t in TARGETS:
        print(f"  {t:14s} route total {out_targets[t]['total_mismatch_best_route']:5.2f} "
              f"= {out_targets[t]['total_footprint_part']:5.2f} footprint "
              f"+ {out_targets[t]['total_concept_part']:5.2f} concept; "
              f"footprint-only half-width {hw_fp[t]:5.2f} logits; "
              f"break-even kappa*L_h {breakeven[t]:.3f}", flush=True)

    rs = [v["bottleneck_over_all_paths"] for v in out_targets.values()]
    out = {
        "n_images": n_img, "n_edits": N, "boot": BOOT, "route_reselected_each_resample": True,
        "smd_tol": SMD_TOL, "neutral_tol": NEUTRAL_TOL, "neutral_ratio": c12.NEUTRAL_RATIO,
        "note": ("Two graphs. G0 is the exact identification graph of the theorem, in which an "
                 "edge needs exact equality; it has no non-trivial edges here. G_eps is a "
                 "diagnostic graph scoring each step as a multiple of its own tolerance. Per "
                 "target we report the bottleneck, which is the worst step of the best route and "
                 "says whether any route is admissible under the screen's tolerances, and the "
                 "total mismatch of the minimum-total route, which a Lipschitz constant would "
                 "multiply to give an interval half-width. The last block asks what Lipschitz "
                 "constant the control-versus-control comparisons would force."),
        "exact_graph_G0": {
            "n_exactly_matched_pairs": exact_pairs, "pairs": exact_list,
            "note": ("Pairs of distinct edits with identical six-component footprints. All of "
                     "them are the same operator entered twice, once as a ladder member and once "
                     "as a matched control. No concept target belongs to such a pair, so every "
                     "concept edit's footprint vertex has degree one in G0 and the second branch "
                     "of Statement 1 applies to each.")},
        "edits": names, "proxy_shift": {a: float(prx[i]) for a, i in idx.items()},
        "per_target": out_targets,
        "min_bottleneck_over_targets": min(rs),
        "max_bottleneck_over_targets": max(rs),
        "any_target_identified": any(v["identified_within_tolerance"] for v in out_targets.values()),
        "lipschitz": {
            "units": ("distances in tolerance units, that is max|SMD| divided by 0.10; L in "
                      "logits per tolerance unit; half-width = L times a route's total mismatch "
                      "in the same units"),
            "pairs": lip, "n_pairs": len(lip),
            "implied_L_binding": Lmin, "implied_L_second_largest": second["implied_L"],
            "implied_L_median": Lmed, "implied_L_quartiles": [float(np.percentile(Ls, 25)),
                                                              float(np.percentile(Ls, 75))],
            "implied_L_conservative_interval_ends": Lcons,
            "worst_pair": worst, "second_pair": second,
            "footprint_only_half_width_at_binding_L": hw_fp,
            "footprint_only_half_width_leave_one_pair_out": hw_fp_loo,
            "footprint_only_half_width_conservative": hw_fp_cons,
            "score_range_logits": SCORE_RANGE,
            "score_range_ends_logits": SCORE_ENDS,
            "score_range_source": ("c12_matched_control.json, models[*].dcace[target]['u:hue'].mean, "
                                   "four targets by three models"),
            "breakeven_kappa_Lh": breakeven,
            "breakeven_kappa_Lh_halfwidth_criterion": breakeven_hw,
            "breakeven_note": ("Concept-side constant at which the full width 2*hw of the "
                               "single-route outer interval stops being narrower than the span of "
                               "the corrected scores. Zero means the footprint side alone already "
                               "makes the interval wider than that span. The second field uses "
                               "hw <= span instead and is kept only for comparison."),
            "binding_L_note": ("L must hold for every pair, so the largest implied value binds. "
                               "The conservative variant divides the lower end of each pair's "
                               "patient-clustered response interval by the upper end of its "
                               "bootstrapped footprint distance.")},
        "runtime_sec": round(time.time() - t0, 1),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nsaved -> {OUT}  ({out['runtime_sec']}s)")


if __name__ == "__main__":
    main()
