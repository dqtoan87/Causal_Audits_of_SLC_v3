#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C90: four post hoc rereads of stored results (no model output is computed, no image is read).

All four were chosen after the results had been seen. Each reads files that earlier scripts wrote.

  (a) Which footprint descriptor blocks the balance screen (A1).
      Reads results/c25_screen_full.json, the per-descriptor SMDs of the 19 library controls at the four
      concept targets on the 183 audit images. Reports, for every control and target, the descriptor that
      attains the maximum |SMD| and the descriptors above the 0.10 tolerance. It then repeats (A1) and
      (A1)+(A2) with CIELAB Delta E removed from the balance check. That repeat is a screen-level
      sensitivity only: removing a descriptor defines another footprint, with its own graph and its own
      concept component, which this script does not build.

  (b) The Lipschitz floor of the generic term, per checkpoint.
      Reads results/c82_image_set_checks.json. g depends on the classifier, so the floor is reported for
      each of the three EfficientNet-B3 checkpoints and for their envelope max_m L^g_m.

  (c) The size of the cycle test over configurations that route sums can produce.
      Reads results/c80_route_null_draws_{first,second}.npz (deviations, standard errors and critical
      values of the 2,000 outer draws). In each model-by-target cell the four differences are built from
      four route sums, so D(rad4-hue4) - D(rad16-hue16) - D(rad4-rad16) + D(hue4-hue16) = 0. c80's
      coordinate ascent moved each difference on its own, which can leave this identity. Here every cell
      is restricted to the configurations at the 0.5-logit margin that satisfy it (six sign patterns per
      cell), and coordinate ascent over cells, from the sign-of-estimate start and from 20 random starts,
      finds the least favourable one. One minus the joint coverage bounds the rate from above for every
      configuration in the shift family, feasible or not.

  (d) Example B5 of the simulation on the structural response R0.
      Rebuilds the first draws of c54_synthetic_scm.py (same seed and order: L0, M_edit, dC_edit) and
      checks the target against the stored one. The near control has footprint 1.02 * M_edit and no
      concept change, so on R0 its route sum is h* + mean(g(M_edit) - g(1.02 M_edit)); the shrinkage
      term cancels. Reports that route sum, the realized ratio |g gap| / d^M (the smallest L' at which the
      structural bound holds), and the noise part of the stored noisy estimate.

Out -> results/c90_posthoc_rereads.json
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c90_posthoc_rereads.json")


def load(n):
    return json.load(open(os.path.join(RES, n)))


def part_a():
    d = load("c25_screen_full.json")
    tol = d["schemes"]["main"]["tol"]            # [A1 SMD, A2 shift, A2 ratio, A3 feasibility]
    out = {"tolerances_main": tol, "targets": {}}
    for t, cs in d["design"].items():
        rows, arg_count, a1, a1_no_de, a12_no_de = {}, {}, [], [], []
        n_over = {}
        for c, v in cs.items():
            if not (isinstance(v, dict) and "smd" in v):
                continue
            s = v["smd"]
            arg = max(s, key=lambda k: abs(s[k]))
            over = sorted(k for k in s if abs(s[k]) > tol[0])
            m5 = max(abs(x) for k, x in s.items() if k != "lab_dE")
            neutral = v["proxy_shift"] <= tol[1] and v["proxy_ratio"] <= tol[2]
            rows[c] = {"max_abs_smd": v["max_abs_smd"], "argmax_descriptor": arg, "descriptors_above_tol": over,
                       "max_abs_smd_without_dE": m5, "passes_A2": bool(neutral)}
            arg_count[arg] = arg_count.get(arg, 0) + 1
            for k in over:
                n_over[k] = n_over.get(k, 0) + 1
            if v["max_abs_smd"] <= tol[0]:
                a1.append(c)
            if m5 <= tol[0]:
                a1_no_de.append(c)
                if neutral:
                    a12_no_de.append(c)
        out["targets"][t] = {"n_controls": len(rows), "argmax_counts": arg_count,
                             "n_controls_above_tol_by_descriptor": n_over,
                             "pass_A1": sorted(a1), "pass_A1_without_dE": sorted(a1_no_de),
                             "pass_A1_A2_without_dE": sorted(a12_no_de), "controls": rows}
    return out


def part_b():
    d = load("c82_image_set_checks.json")["lipschitz_library_floor_58"]
    pairs = d["pairs"]
    out = {"note": "floor of L^g per checkpoint, 58 images of the library protocol; envelope = max over checkpoints",
           "per_model": {}}
    for m in sorted(set(p["model"] for p in pairs)):
        ps = [p for p in pairs if p["model"] == m]
        b = max(ps, key=lambda p: p["implied_L_58"])
        out["per_model"][m] = {"n_pairs": len(ps), "floor": b["implied_L_58"], "binding_target": b["target"],
                               "binding_pair": [b["a"], b["b"]], "gap": b["abs_diff_library_58"],
                               "distance": b["distance_58"],
                               "conservative_floor": max(p["implied_L_conservative"] for p in ps)}
    out["envelope_floor"] = max(v["floor"] for v in out["per_model"].values())
    out["envelope_floor_stored"] = d["L_floor"]
    return out


def part_c():
    order = ["D_rad4-hue4", "D_rad16-hue16", "D_rad4-rad16", "D_hue4-hue16"]
    V = [np.array(s) for s in itertools.product([1, -1], repeat=4) if s[0] - s[1] - s[2] + s[3] == 0]
    out = {"feasible_sign_patterns": [v.tolist() for v in V]}
    stored = load("c80_route_null_calibration.json")
    for run in ("first", "second"):
        z = np.load(os.path.join(RES, "c80_route_null_draws_%s.npz" % run))
        names = [str(n) for n in z["names"]]
        dev, half = z["dev"], z["crit"][:, None] * z["se"]
        th = z["theta"]
        up, dn = dev - half > 0, dev + half < 0
        cells = sorted(set(n.rsplit("|", 1)[0] for n in names))
        cid = {c: [names.index(c + "|" + s) for s in order] for c in cells}
        resid = max(abs(th[i[0]] - th[i[1]] - th[i[2]] + th[i[3]]) for i in cid.values())

        def rate(cfg):
            s = np.zeros(len(names))
            for c, v in cfg.items():
                s[cid[c]] = v
            return float(((s > 0)[None, :] & up | (s < 0)[None, :] & dn).any(1).mean())

        def ascend(cfg):
            for _ in range(20):
                changed = False
                for c in cells:
                    b = max(V, key=lambda v: rate({**cfg, c: v}))
                    if rate({**cfg, c: b}) > rate(cfg) + 1e-12:
                        cfg[c], changed = b, True
                if not changed:
                    break
            return cfg

        start = {c: max(V, key=lambda v: float(np.dot(v, np.sign(th[cid[c]])))) for c in cells}
        r_start = rate(start)
        best = ascend(dict(start))
        r_best = rate(best)
        rng = np.random.RandomState(0)
        for _ in range(20):
            cfg = ascend({c: V[rng.randint(len(V))] for c in cells})
            if rate(cfg) > r_best:
                best, r_best = cfg, rate(cfg)
        cover = (np.abs(dev) <= half).all(1).mean()
        rs = stored[run] if run in stored else stored.get("runs", {}).get(run, {})
        out[run] = {"identity_residual_of_estimates": float(resid),
                    "rate_feasible_nearest_to_sign_of_estimate": r_start,
                    "rate_feasible_least_favourable_found": r_best,
                    "upper_bound_one_minus_joint_coverage": float(1 - cover),
                    "rate_unconstrained_worst_found_c80": rs.get("fwer_margin_worst_found"),
                    "mcse_least_favourable": float(np.sqrt(r_best * (1 - r_best) / dev.shape[0]))}
    return out


def part_d():
    import c54_synthetic_scm as c54
    rng = np.random.RandomState(c54.SEED)
    L0 = rng.normal(0.4, 1.2, c54.N)
    M_edit = np.abs(rng.normal(1.0, 0.18, (c54.N, 6)))
    dC_edit = np.abs(rng.normal(0.60, 0.12, c54.N))
    target = float(np.mean(c54.h(dC_edit)))
    stored = load("c54_synthetic_scm.json")
    b5 = stored["identified_set"]["B5_approximate_with_lipschitz"]
    near = 1.02 * M_edit
    d_near = float(max(abs(c54.smd(near[:, k], M_edit[:, k])) for k in range(6)))
    g_gap = float(np.mean(c54.g(M_edit, L0, c54.ALPHA_EDIT) - c54.g(near, L0, c54.ALPHA_EDIT)))
    s0 = target + g_gap
    out = {"target_rebuilt": target, "target_stored": stored["target_h"],
           "target_matches": bool(abs(target - stored["target_h"]) < 1e-12),
           "distance_rebuilt": d_near, "distance_stored": b5["footprint_distance_max_smd"],
           "route_sum_R0": s0, "g_gap": g_gap, "realized_ratio_L_prime": abs(g_gap) / d_near,
           "route_sum_noisy_stored": b5["point_estimate_from_near_control"],
           "noise_part": b5["point_estimate_from_near_control"] - s0}
    for L in (0.5, 1.0, 2.0):
        out["R0_interval_at_L_%s" % L] = [s0 - L * d_near, s0 + L * d_near]
        out["R0_covers_target_at_L_%s" % L] = bool(abs(s0 - target) <= L * d_near)
    return out


def main():
    res = {"note": "post hoc rereads of stored results; chosen after the results were seen",
           "a_screen_descriptors": part_a(), "b_lipschitz_per_model": part_b(),
           "c_cycle_test_feasible_size": part_c(), "d_b5_structural": part_d()}
    json.dump(res, open(OUT, "w"), indent=1)
    a = res["a_screen_descriptors"]["targets"]
    for t, v in a.items():
        print(t, "argmax", v["argmax_counts"], "A1", v["pass_A1"], "A1 w/o dE", v["pass_A1_without_dE"],
              "A1+A2 w/o dE", v["pass_A1_A2_without_dE"])
    print("L^g floors", {m: round(v["floor"], 3) for m, v in res["b_lipschitz_per_model"]["per_model"].items()})
    for run in ("first", "second"):
        c = res["c_cycle_test_feasible_size"][run]
        print(run, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
    print("B5", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["d_b5_structural"].items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
