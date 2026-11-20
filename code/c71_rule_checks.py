#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C71: checks of three reporting rules  (no model is run; reads stored results only)

1. The SMD rule when the pooled variance is zero. Two footprints with zero variance but different means
   are not balanced, so the SMD must then be infinite, not zero. This part counts how often a zero
   denominator occurs: over all pairs of the 88 library edits (183 images and the 83 malignant images, from
   the footprint cache of c55) and between each route-test control and its concept edit (from the
   registered construction). It also runs a unit test of the corrected rule.
2. The score floor of the concept-tier response analyses. In bf16, one image (ISIC_3039686) scores
   0.099 to 0.102 under the Inv model depending on the batch, so the floor keeps 58 images in some scripts
   and 57 in others (c68). The leave-one-family-out fit of Section S6 and the minimum detectable effect of
   Section S4 use 58. We recompute both on the 57-image set, which the full-precision floor keeps.
3. Multiplicity of the concept-activation-vector test (c49): Holm and Benjamini-Hochberg over the twelve
   cells, from the stored p-values.

Out -> results/c71_rule_checks.json
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c71_rule_checks.json")
TOL = 1e-12


def smd_rule(a, b):
    """Corrected rule: zero denominator -> 0 if the means agree, infinite otherwise."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    sd = math.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    d = a.mean() - b.mean()
    if sd > TOL:
        return d / sd
    return 0.0 if d == 0.0 else math.copysign(math.inf, d)


def unit_test():
    assert smd_rule([1, 1, 1], [1, 1, 1]) == 0.0
    assert smd_rule([1, 1, 1], [2, 2, 2]) == -math.inf
    assert smd_rule([2, 2, 2], [1, 1, 1]) == math.inf
    assert abs(smd_rule([0, 1, 2], [1, 2, 3]) + 1.0) < 1e-12
    return True


def zero_denominators_library():
    z = np.load(os.path.join(RES, "c55_design_cache.npz"), allow_pickle=True)
    names = sorted(str(x) for x in z["names"])
    covs = ["frac_altered", "mean_abs", "lab_dE", "support_rms", "lesion_cov", "hf_change"]
    FP = np.stack([np.stack([z["fp_%s_%s" % (a, k)] for k in covs], 0) for a in names], 0)   # [88, 6, 183]
    coh = json.load(open(os.path.join(RES, "audit_cohort.json")))
    lab = np.asarray(coh["target"])
    out = {}
    for sub, idx in (("all_183", np.arange(FP.shape[2])), ("malignant_83", np.where(lab == 1)[0])):
        F = FP[:, :, idx]
        mean, var = F.mean(2), F.var(2, ddof=1)
        sd = np.sqrt(0.5 * (var[:, None, :] + var[None, :, :]))
        iu = np.triu_indices(len(names), 1)
        zs = sd[iu[0], iu[1], :] <= TOL
        diff = np.abs(mean[:, None, :] - mean[None, :, :])[iu[0], iu[1], :] > 0
        out[sub] = {"n_images": int(len(idx)), "pair_components": int(zs.size), "zero_denominators": int(zs.sum()),
                    "zero_denominators_with_different_means": int((zs & diff).sum()),
                    "smallest_pooled_sd": float(sd[iu[0], iu[1], :].min())}
    return out


def zero_denominators_route():
    cons = json.load(open(os.path.join(RES, "c62_construction.json")))
    fine = ["frac_altered", "mean_abs", "lab_dE", "support_rms", "lesion_cov", "hf_change"]
    n_zero, n_tot, smallest = 0, 0, math.inf
    for tk in cons["images"][0]["targets"]:
        for c in cons["images"][0]["targets"][tk]["controls"]:
            rows = [im["targets"][tk] for im in cons["images"] if im["targets"][tk]["controls"][c]["exact_mag"]]
            for k in fine:
                a = [r["controls"][c]["measures"][k] for r in rows]
                b = [r["edit"]["measures"][k] if "measures" in r["edit"] else r["edit"][k] for r in rows]
                sd = math.sqrt(0.5 * (np.var(a, ddof=1) + np.var(b, ddof=1)))
                n_tot += 1; n_zero += sd <= TOL; smallest = min(smallest, sd)
    return {"control_components": n_tot, "zero_denominators": int(n_zero), "smallest_pooled_sd": smallest}


def floor_57():
    import c12_matched_control as c12
    pr = np.load(os.path.join(RES, "c12_preds.npz"), allow_pickle=True)
    ref = json.load(open(os.path.join(RES, "c12_matched_control.json")))
    z = np.load(os.path.join(RES, "c55_design_cache.npz"), allow_pickle=True)
    coh = json.load(open(os.path.join(RES, "audit_cohort.json")))
    ids = np.asarray(coh["isic_id"])
    mal = np.asarray(pr["img_index"])
    sets = json.load(open(os.path.join(RES, "c68_floor_sets.json")))
    keep57 = set(sets["variants"]["c10_batch"]["isic_ids"])
    LADDER = ["L:%s@%d" % (c, j) for c in c12.LADDER_FAM for j in range(len(c12.LADDER_MULT))]
    Mall = {n: np.stack([z["fp_%s_%s" % (n, k)][mal] for k in c12.COVS], 1) for n in LADDER}
    models = ["erm", "slim", "causal"]
    common58 = np.all([pr["p0_%s" % m] >= 0.1 for m in models], 0)
    common57 = common58 & np.isin(ids[mal], list(keep57))
    out = {"n58": int(common58.sum()), "n57": int(common57.sum()),
           "dropped": [str(i) for i in ids[mal][common58 & ~common57]], "lofo": {}, "mde": {}}
    for m in models:
        out["lofo"][m] = {}
        for tag, cm in (("58", common58), ("57", common57)):
            idxc = np.where(cm)[0]
            res = {}
            for held, fams_h in c12.LOFO_GROUPS.items():
                te = [n for n in LADDER if any(n.startswith("L:%s@" % f) for f in fams_h)]
                tr = [n for n in LADDER if n not in te]
                f2 = c12.fit_g(np.stack([Mall[n][idxc] for n in tr], 1),
                               np.stack([pr["dl_%s|%s" % (m, n)][idxc] for n in tr], 1))
                yt = np.stack([pr["dl_%s|%s" % (m, n)][idxc] for n in te], 1)
                yh = np.stack([c12.pred_g(f2, Mall[n][idxc]) for n in te], 1)
                ss = ((yt - yt.mean(1, keepdims=True)) ** 2).sum()
                res[held] = {"r2_within": float(1 - ((yt - yh) ** 2).sum() / max(ss, 1e-12)),
                             "rmse": float(np.sqrt(((yt - yh) ** 2).mean())),
                             "mae": float(np.abs(yt - yh).mean())}
            out["lofo"][m][tag] = res
        out["lofo"][m]["matches_c12_json_on_58"] = max(
            abs(out["lofo"][m]["58"][h]["r2_within"] - ref["models"][m]["g_fit"]["lofo"][h]["r2_within"])
            for h in c12.LOFO_GROUPS) < 1e-9
        # minimum detectable effect of the concept-minus-placebo difference, 2.8 SD / sqrt(n)
        mde = {}
        for tag, cm in (("58", common58), ("57", common57)):
            v = []
            for t in ["contrast@2", "color_var@2", "contrast@0", "color_var@0"]:
                for p in ["flip", "jitter", "hue", "noise"]:
                    d = pr["dl_%s|%s" % (m, t)][cm] - pr["dl_%s|u:%s" % (m, p)][cm]
                    v.append(2.8 * d.std(ddof=1) / math.sqrt(len(d)))
            mde[tag] = [float(min(v)), float(max(v))]
        out["mde"][m] = mde
    out["lofo_all_negative_57"] = all(r["r2_within"] < 0 for m in models for r in out["lofo"][m]["57"].values())
    out["lofo_all_negative_58"] = all(r["r2_within"] < 0 for m in models for r in out["lofo"][m]["58"].values())
    return out


def tcav_multiplicity(alpha=0.05):
    d = json.load(open(os.path.join(RES, "c49_tcav.json")))
    ps = sorted((e["p_vs_random"], m, c) for m, v in d["models"].items() for c, e in v["concepts"].items())
    k = len(ps)
    holm = 0
    for i, (p, _, _) in enumerate(ps):
        if p <= alpha / (k - i):
            holm += 1
        else:
            break
    bh = max([i + 1 for i, (p, _, _) in enumerate(ps) if p <= (i + 1) * alpha / k] or [0])
    return {"n_cells": k, "nominal": sum(p < alpha for p, _, _ in ps), "holm": holm, "benjamini_hochberg": bh,
            "sorted_p": [[p, m, c] for p, m, c in ps]}


def main():
    out = {"unit_test_smd_rule": unit_test(),
           "smd_zero_denominators_library": zero_denominators_library(),
           "smd_zero_denominators_route_controls": zero_denominators_route(),
           "score_floor_57_vs_58": floor_57(),
           "tcav_multiplicity": tcav_multiplicity()}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: (v if k != "score_floor_57_vs_58" else {kk: vv for kk, vv in v.items() if kk != "lofo"})
                      for k, v in out.items()}, indent=1)[:4000])
    for m, v in out["score_floor_57_vs_58"]["lofo"].items():
        print(m, {t: {h: round(r["r2_within"], 2) for h, r in v[t].items()} for t in ("58", "57")},
              "matches c12:", v["matches_c12_json_on_58"])
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
