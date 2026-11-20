#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C54: synthetic response model with a known concept component.

Why this exists. Every concept-side result in the paper is a negative one measured on real
images, where the target is latent and no ground truth is available. That leaves two questions
the image data cannot answer. Does the placebo estimator recover the target when its stated
conditions hold, so that the screen is not a rule which only ever rejects? And do the
identification statements behave as stated when the mechanism is known by construction?

This script answers both on a simulated response. Nothing here touches images or classifiers;
the response is generated from a stated formula, so the target is exact.

Part A: five regimes, each one control against the same concept edit, reporting the observable
screen (A1)-(A3) and the estimator error against the known target.

Part B: five constructions, each one clause of Statement 1 or Statement 2, plus the interval that
an explicit component-range restriction leaves.

Out -> results/c54_synthetic_scm.json
Run:  .venv/bin/python Causal_XAI_v3/code/c54_synthetic_scm.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT = os.path.join(RES, "c54_synthetic_scm.json")

SEED = 42
N = 400                      # simulated units
COVS = ["frac_altered", "mean_abs", "lab_dE", "support_rms", "lesion_cov", "hf_change"]
SMD_TOL, NEUTRAL_RATIO, OVERLAP_TOL = 0.10, 0.15, 0.95
CLIP = 13.8155               # the clipped-logit bound the audit works on

# response-model coefficients, fixed by hand so that every reported quantity is exact
BETA = np.array([0.9, 1.4, 0.5, -0.3, 0.7, 1.1])   # generic footprint sensitivity, g
GAMMA = 1.60                                        # concept sensitivity, h
ALPHA_EDIT = 0.22                                   # shrinkage on the baseline logit, concept edit


def smd(a, b):
    """Paired standardized mean difference, the same definition the image-side screen uses."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    sd = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    return float((a.mean() - b.mean()) / sd) if sd > 1e-12 else 0.0


def g(M, L0, alpha):
    """Generic edit sensitivity: a footprint term plus shrinkage toward an edit-specific point."""
    return M @ BETA - alpha * L0


def h(dC):
    """Concept component of the response. Nonlinear, so nothing below assumes a linear h."""
    return GAMMA * np.tanh(1.5 * dC)


def screen(M_edit, M_ctrl, prx_edit, prx_ctrl, overlap):
    """The observable admissibility screen of Section III-C, from design quantities only."""
    sm = {c: smd(M_ctrl[:, j], M_edit[:, j]) for j, c in enumerate(COVS)}
    mx = max(abs(v) for v in sm.values())
    ratio = float(np.mean(prx_ctrl) / max(float(np.mean(prx_edit)), 1e-9))
    return {"max_abs_smd": mx, "proxy_ratio": ratio, "overlap": overlap,
            "A1_balance": bool(mx <= SMD_TOL),
            "A2_proxy_neutrality": bool(abs(ratio) <= NEUTRAL_RATIO),
            "A3_overlap": bool(overlap >= OVERLAP_TOL),
            "admissible": bool(mx <= SMD_TOL and abs(ratio) <= NEUTRAL_RATIO
                               and overlap >= OVERLAP_TOL),
            "smd": {k: float(v) for k, v in sm.items()}}


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)
    print("##### C54 synthetic response model #####", flush=True)

    L0 = rng.normal(0.4, 1.2, N)                       # baseline logit of each unit
    M_edit = np.abs(rng.normal(1.0, 0.18, (N, 6)))     # footprint of the concept edit
    dC_edit = np.abs(rng.normal(0.60, 0.12, N))        # true concept change of the concept edit
    prx_edit = dC_edit * rng.normal(1.0, 0.10, N)      # the proxy reads it with error
    zero = np.zeros(N)

    target = float(np.mean(h(dC_edit)))                # what a valid estimator must return
    eta = lambda: rng.normal(0.0, 0.15, N)             # residual, so no regime is noiseless
    R0_edit = g(M_edit, L0, ALPHA_EDIT) + h(dC_edit)  # structural response R0, before the noise
    R_edit = R0_edit + eta()

    regimes = {}
    ctrl_R, ctrl_R0 = {}, {}                           # per-unit control responses, kept for Part D

    def add(name, note, M_c, dC_c, prx_c, alpha_c, overlap=1.0):
        R_c = g(M_c, L0, alpha_c) + h(dC_c) + eta()
        ctrl_R[name], ctrl_R0[name] = R_c, g(M_c, L0, alpha_c) + h(dC_c)
        est = float(np.mean(R_edit) - np.mean(R_c))
        # the same contrast on the structural responses R0 (no noise): what the estimator targets
        est0 = float(np.mean(R0_edit) - np.mean(g(M_c, L0, alpha_c) + h(dC_c)))
        sc = screen(M_edit, M_c, prx_edit, prx_c, overlap)
        regimes[name] = {"note": note, "screen": sc, "estimate": est, "target": target,
                         "error": est - target,
                         "abs_error_frac_of_target": abs(est - target) / abs(target),
                         "estimate_noise_free": est0, "error_noise_free": est0 - target,
                         "abs_error_frac_of_target_noise_free": abs(est0 - target) / abs(target)}
        print(f"  {name:26s} admissible={str(sc['admissible']):5s} "
              f"maxSMD={sc['max_abs_smd']:6.3f} ratio={sc['proxy_ratio']:+.3f} "
              f"est={est:+.3f} target={target:+.3f} err={est-target:+.3f}", flush=True)

    # A control matched by construction reproduces the edit's footprint on the same unit, which is
    # what the balanced members of the image-side library do, so regimes meant to satisfy (A1)
    # reuse M_edit rather than an independent draw of the same law.
    add("R1_valid", "footprint matched, no concept movement, same shrinkage coefficient",
        M_edit.copy(), zero, zero, ALPHA_EDIT)
    add("R2_footprint_mismatch", "control footprint scaled by 1.35, concept untouched",
        1.35 * M_edit, zero, zero, ALPHA_EDIT)
    add("R3_concept_contamination", "footprint matched but the control moves the concept by 40 percent",
        M_edit.copy(), 0.40 * dC_edit, 0.40 * prx_edit, ALPHA_EDIT)
    add("R4_model_state", "footprint matched and concept neutral, shrinkage coefficient differs",
        M_edit.copy(), zero, zero, ALPHA_EDIT + 0.18)
    add("R5_proxy_blind", "footprint matched, proxy reads zero, latent concept still moves",
        M_edit.copy(), 0.35 * dC_edit, zero, ALPHA_EDIT)

    predicted = (ALPHA_EDIT + 0.18 - ALPHA_EDIT) * float(np.mean(L0))
    regimes["R4_model_state"]["predicted_bias_from_formula"] = predicted
    regimes["R4_model_state"]["formula_abs_error"] = abs(predicted - regimes["R4_model_state"]["error"])
    print(f"  bias formula on R4: predicted {predicted:+.3f} against observed "
          f"{regimes['R4_model_state']['error']:+.3f}", flush=True)

    # ---------------------------------------------------------------- Part B
    print("\n  identified set for the concept component", flush=True)
    shifts = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    ctrl_a, ctrl_b = 1.35 * M_edit, 0.60 * M_edit
    near = M_edit * 1.02
    partB = {}

    def sweep(lib_M, lib_dC, shift_g, shift_h, label):
        """Shift g down and h up by delta on the named edits and see what the data still allows."""
        base = {k: float(np.mean(g(lib_M[k], L0, ALPHA_EDIT) + h(lib_dC[k]))) for k in lib_M}
        rows = []
        for d in shifts:
            obs = {}
            for k in lib_M:
                gk = g(lib_M[k], L0, ALPHA_EDIT) - (d if k in shift_g else 0.0)
                hk = h(lib_dC[k]) + (d if k in shift_h else 0.0)
                obs[k] = float(np.mean(gk + hk))
            rows.append({"shift": d,
                         "max_abs_deviation_from_observed": max(abs(obs[k] - base[k]) for k in base),
                         "implied_target": target + d})
        adm = [r["implied_target"] for r in rows if r["max_abs_deviation_from_observed"] < 1e-9]
        print(f"    {label:38s} shifts reproducing the data: {len(adm)}/{len(shifts)}, "
              f"implied target {min(adm):+.3f} to {max(adm):+.3f}", flush=True)
        return {"rows": rows, "n_admissible_shifts": len(adm),
                "max_abs_deviation_over_shifts": max(r["max_abs_deviation_from_observed"] for r in rows),
                "implied_target_range": [min(adm), max(adm)]}

    # B1. an exactly matched neutral control is present, so the length-two route exists
    partB["B1_connected"] = sweep(
        {"concept_edit": M_edit, "ctrl_matched": M_edit.copy(), "ctrl_far": ctrl_a},
        {"concept_edit": dC_edit, "ctrl_matched": zero, "ctrl_far": zero},
        {"concept_edit", "ctrl_matched"}, {"concept_edit"}, "B1 connected, exact match")
    partB["B1_connected"]["point_estimate"] = float(
        np.mean(R_edit) - np.mean(g(M_edit, L0, ALPHA_EDIT) + h(zero) + eta()))

    # B2. no edit shares the concept edit's footprint
    partB["B2_disconnected"] = sweep(
        {"concept_edit": M_edit, "ctrl_far_a": ctrl_a, "ctrl_far_b": ctrl_b},
        {"concept_edit": dC_edit, "ctrl_far_a": zero, "ctrl_far_b": zero},
        {"concept_edit"}, {"concept_edit"}, "B2 disconnected")

    # B3. adding one exact match to B2 closes the shift
    partB["B3_repaired"] = sweep(
        {"concept_edit": M_edit, "ctrl_far_a": ctrl_a, "ctrl_far_b": ctrl_b,
         "ctrl_matched": M_edit.copy()},
        {"concept_edit": dC_edit, "ctrl_far_a": zero, "ctrl_far_b": zero, "ctrl_matched": zero},
        {"concept_edit", "ctrl_matched"}, {"concept_edit"}, "B3 repaired by one exact match")
    est3 = float(np.mean(R_edit) - np.mean(g(M_edit, L0, ALPHA_EDIT) + h(zero) + eta()))
    partB["B3_repaired"]["point_estimate"] = est3
    partB["B3_repaired"]["abs_error"] = abs(est3 - target)

    # B4. a near match is still a distinct vertex, so an unrestricted g gains nothing from it
    partB["B4_approximate_no_smoothness"] = sweep(
        {"concept_edit": M_edit, "ctrl_near": near, "ctrl_far_a": ctrl_a},
        {"concept_edit": dC_edit, "ctrl_near": zero, "ctrl_far_a": zero},
        {"concept_edit"}, {"concept_edit"}, "B4 near match, g unrestricted")
    d_near = float(max(abs(smd(near[:, k], M_edit[:, k])) for k in range(6)))
    partB["B4_approximate_no_smoothness"]["footprint_distance_max_smd"] = d_near

    # B5. the same near match under a Lipschitz constant gives a finite outer interval
    est_near = float(np.mean(R_edit) - np.mean(g(near, L0, ALPHA_EDIT) + h(zero) + eta()))
    b5 = {"footprint_distance_max_smd": d_near, "point_estimate_from_near_control": est_near}
    for L in (0.5, 1.0, 2.0):
        b5["interval_at_L_%s" % L] = [est_near - L * d_near, est_near + L * d_near]
        b5["covers_target_at_L_%s" % L] = bool(abs(est_near - target) <= L * d_near)
    print(f"    {'B5 near match, Lipschitz g':38s} distance {d_near:.3f} SMD, estimate "
          f"{est_near:+.3f}, outer interval at L=1 "
          f"[{b5['interval_at_L_1.0'][0]:+.3f}, {b5['interval_at_L_1.0'][1]:+.3f}]", flush=True)
    partB["B5_approximate_with_lipschitz"] = b5

    # ------------------------------------------------------------ Part C, constructive repair
    # The negative results say a disconnected library identifies nothing. The converse is what an
    # auditor can act on: a library deliberately built so that a route exists recovers the target,
    # and the route need not be the length-two placebo. Here a chain of length four does it, with
    # no control sharing the concept edit's footprint at all.
    print("\n  constructive repair by a designed route", flush=True)
    M_b = np.abs(rng.normal(1.30, 0.20, (N, 6)))    # a second footprint level, shared by two edits
    dC_b = np.abs(rng.normal(0.35, 0.10, N))        # a second concept level, shared by two edits
    # e1 = concept edit at footprint A and concept c*, e1' = bridging edit at footprint A, concept b
    # e2 = edit at footprint B and concept b,       e2' = neutral control at footprint B, concept 0
    lib = {
        "e1  concept edit      (A, c*)": (M_edit, dC_edit),
        "e1' bridge            (A, b )": (M_edit.copy(), dC_b),
        "e2  bridge            (B, b )": (M_b, dC_b.copy()),
        "e2' neutral control   (B, 0 )": (M_b.copy(), zero),
    }
    Rlib = {k: float(np.mean(g(M, L0, ALPHA_EDIT) + h(c) + eta())) for k, (M, c) in lib.items()}
    keys = list(lib)
    S_chain = (Rlib[keys[0]] - Rlib[keys[1]]) + (Rlib[keys[2]] - Rlib[keys[3]])
    # the same route sum on the structural responses R0: equation (9) says it equals the target exactly
    R0lib = {k: float(np.mean(g(M, L0, ALPHA_EDIT) + h(c))) for k, (M, c) in lib.items()}
    S_chain0 = (R0lib[keys[0]] - R0lib[keys[1]]) + (R0lib[keys[2]] - R0lib[keys[3]])
    # no edit in this library shares the concept edit's footprint except its own bridge, and the
    # placebo route does not exist because no neutral control sits at footprint A
    d_placebo = float(max(abs(smd(M_b[:, k], M_edit[:, k])) for k in range(6)))
    # patient-style bootstrap of the chained estimator over units
    boot = []
    for _ in range(2000):
        idx = rng.randint(0, N, N)
        r = {}
        for k, (M, c) in lib.items():
            r[k] = float(np.mean(g(M[idx], L0[idx], ALPHA_EDIT) + h(c[idx])
                                 + rng.normal(0.0, 0.15, N)))
        boot.append((r[keys[0]] - r[keys[1]]) + (r[keys[2]] - r[keys[3]]))
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    partC = {
        "note": ("A library with no footprint-matched neutral control, so the length-two placebo "
                 "route does not exist, but with a bridging edit and a second footprint level "
                 "that make a length-four route. The alternating sum along that route recovers "
                 "the target."),
        "library": {k: {"mean_response": Rlib[k]} for k in keys},
        "nearest_neutral_control_footprint_distance_max_smd": d_placebo,
        "placebo_route_exists": False,
        "chained_route_estimate": S_chain,
        "chained_route_noise_free": S_chain0,
        "chained_route_noise_free_abs_error": abs(S_chain0 - target),
        "target": target,
        "abs_error": abs(S_chain - target),
        "bootstrap_ci95": ci,
        "covers_target": bool(ci[0] <= target <= ci[1]),
        "boot": 2000,
    }
    print(f"    no neutral control within {d_placebo:.2f} SMD of the concept edit's footprint, "
          f"so no length-two route exists", flush=True)
    print(f"    length-four route returns {S_chain:+.3f} against target {target:+.3f}, "
          f"error {abs(S_chain-target):.3f}, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], "
          f"covers target {partC['covers_target']}", flush=True)

    # Component-range restriction. A bounded classifier output does not by itself bound the
    # additive components, so this is an extra assumption rather than a consequence. Requiring the
    # mean generic term and the concept term to stay in [-B, B] confines the shift: with one
    # footprint vertex and one concept vertex in the component, |gbar - delta| <= B and
    # |h* + delta| <= B.
    gbar = float(np.mean(g(M_edit, L0, ALPHA_EDIT)))
    lo, hi = max(gbar - CLIP, -CLIP - target), min(gbar + CLIP, CLIP - target)
    partB["component_range_restriction"] = {
        "bound_B_logits": CLIP, "mean_g_at_edit": gbar, "h_star": target,
        "admissible_shift_range": [lo, hi], "width": hi - lo,
        "implied_target_range": [target + lo, target + hi],
        "note": ("An extra restriction on the components, not implied by a bounded output. The "
                 "component holds one footprint vertex and one concept vertex, so the shift must "
                 "satisfy |gbar - delta| <= B and |h* + delta| <= B.")}
    print(f"    {'component-range restriction':38s} delta in [{lo:+.2f}, {hi:+.2f}], implied target "
          f"{target+lo:+.2f} to {target+hi:+.2f}, width {hi-lo:.1f} logits", flush=True)


    # ---------------------------------------------------------------- Part D
    # R4 alone is a library of two edits with distinct pairs (M, dC), so the additive form holds on
    # it trivially and Statement 1 identifies the library component, R0(e*) - R0(c_R4). That value
    # differs from the generating concept term because the response depends on the shrinkage
    # coefficient, an attribute of the edit the footprint does not record. Putting the R1 and the
    # R4 control in one library gives two edits with the same footprint and the same concept change
    # (zero) whose responses differ, so the additive form (C1a) fails on that library, and the two
    # placebo routes disagree. This adds no draw from the main generator: the per-unit responses are
    # the ones Part A drew, and the unit bootstrap uses its own generator.
    a, b = "R1_valid", "R4_model_state"
    # route sums S_c = R(e*) - R(c), so S_R4 - S_R1 = R(c_R1) - R(c_R4)
    d0 = float(np.mean(ctrl_R0[a] - ctrl_R0[b]))            # structural gap S_R4 - S_R1 on R0
    dd = ctrl_R[a] - ctrl_R[b]                               # per-unit gap with the realized noise
    rng_d = np.random.RandomState(SEED + 54)
    NB_D = 20000
    boots = np.array([dd[rng_d.randint(0, N, N)].mean() for _ in range(NB_D)])
    ci_d = np.percentile(boots, [2.5, 97.5])
    partD = {
        "note": ("R4 alone: the additive form holds trivially on a two-edit library; Statement 1 "
                 "identifies the library component, estimate_noise_free of R4. Library with the R1 "
                 "and the R4 control: same footprint, zero concept change, different responses, so "
                 "C1a fails on the library; the two placebo routes disagree. Unit bootstrap with its "
                 "own generator (seed 42 + 54), percentile interval."),
        "library_component_R4_alone_noise_free": regimes[b]["estimate_noise_free"],
        "generating_concept_term": target,
        "controls": [a, b],
        "both_pass_screens": bool(regimes[a]["screen"]["admissible"] and regimes[b]["screen"]["admissible"]),
        "same_footprint_exactly": True, "same_concept_change": 0.0,
        "route_sum_R1_noise_free": regimes[a]["estimate_noise_free"],
        "route_sum_R4_noise_free": regimes[b]["estimate_noise_free"],
        "route_sum_R4_minus_R1_noise_free": d0,
        "route_sum_R4_minus_R1_noisy": float(dd.mean()),
        "route_sum_R4_minus_R1_noisy_ci95": [float(ci_d[0]), float(ci_d[1])],
        "n_boot": NB_D, "boot_seed": SEED + 54,
        "excludes_zero": bool(ci_d[0] > 0 or ci_d[1] < 0),
        "disagrees_at_margin_0p5": bool(ci_d[0] > 0.5 or ci_d[1] < -0.5),
    }
    print(f"\n  Part D: R1 and R4 controls in one library, S_R4 - S_R1 on R0 {d0:+.3f}, "
          f"noisy {dd.mean():+.3f} [{ci_d[0]:+.3f}, {ci_d[1]:+.3f}]", flush=True)

    out = {
        "n_units": N, "seed": SEED, "clip_bound": CLIP,
        "note": ("Synthetic response with a known concept component. Part A runs the observable "
                 "admissibility screen and the placebo estimator over five control regimes. Part "
                 "B instantiates each clause of Statement 1 and Statement 2, and reports the "
                 "interval left by an explicit component-range restriction."),
        "response_model": {"beta": BETA.tolist(), "gamma": GAMMA, "alpha_edit": ALPHA_EDIT,
                           "form": "dl = M@beta - alpha*L0 + gamma*tanh(1.5*dC) + eta",
                           "noise_free": ("fields ending in _noise_free use the structural response R0 = "
                                          "M@beta - alpha*L0 + gamma*tanh(1.5*dC), to which Statement 1 "
                                          "applies; the other estimates add the realized noise eta")},
        "target_h": target, "regimes": regimes, "identified_set": partB,
        "constructive_repair": partC, "shared_footprint_controls": partD,
        "runtime_sec": round(time.time() - t0, 1),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nsaved -> {OUT}  ({out['runtime_sec']}s)")


if __name__ == "__main__":
    main()
