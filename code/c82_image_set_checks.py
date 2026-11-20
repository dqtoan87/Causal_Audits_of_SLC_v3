#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C82: image set and protocol of every quantity that Statement 1 or Statement 2 combines.  (post hoc)

Statement 1 and Statement 2 hold for one fixed image set S and one inference protocol. The library
footprints are computed on the 183 audit images, the library responses (c12) on the 83 malignant
images in one batch of 88 edits per image, the corrected scores on the 58 malignant images that all
three EfficientNet-B3 models score at 0.1 or above under that protocol, and diagnostic (T) (c34) on 57
images scored in one batch of the needed controls per image. This script asks whether the verdicts
that combine them survive when every quantity is computed on one set and under one protocol.

A. The exact graph G0 on each image set. Two edits share a footprint vertex on S when their six
   descriptors agree on every image of S at double precision. A subset can only merge vertices, so
   the check is whether new shared vertices appear on the 83, 58 or 57 images.
B. The diagnostic graph G_a, the Lipschitz floor on g and the break-even table, all on the 58-image
   set of the library protocol: footprints and proxy shifts restricted to those images, pairwise
   control responses read from c12_preds.npz (one batch of 88 edits per image), and the span of the
   hue-placebo corrected scores on the same images.

Reads results/c55_design_cache.npz, c12_preds.npz, c12_matched_control.json, c34_ctrl_vs_ctrl.json,
c55_identification_graph.json, c68_floor_sets.json and the audit cohort manifest. No model is run.
Out -> results/c82_image_set_checks.json
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c55_identification_graph as c55   # noqa: E402

RES = c55.RES
OUT = os.path.join(RES, "c82_image_set_checks.json")
CACHE = os.path.join(RES, "c82_library83_cache.npz")
BOOT = 5000
MODELS = ["erm", "slim", "causal"]
NAME = {"erm": "ERM", "slim": "Slim", "causal": "Inv"}
TARGETS = c55.TARGETS


def shared_pairs(FP, names, cols):
    """Pairs of edits whose six descriptors agree on every image in cols (double precision)."""
    sub = FP[:, :, cols]
    out = []
    for i, j in itertools.combinations(range(len(names)), 2):
        if np.array_equal(sub[i], sub[j]):
            out.append([names[i], names[j]])
    return out


def graph_on(FP, PRX, cols, idx, names):
    fm, fv = FP[:, :, cols].mean(axis=2), FP[:, :, cols].var(axis=2, ddof=1)
    smd = c55.smd_matrix(fm, fv)
    fcost = smd / c55.SMD_TOL
    prx = PRX[:, cols].mean(axis=1)
    N = len(names)
    res = {}
    for t in TARGETS:
        ti = idx[t]
        scale = max(abs(prx[ti]), 1e-9)
        ccost = np.abs(prx[:, None] - prx[None, :]) / scale / c55.c12.NEUTRAL_RATIO
        is_null = (np.abs(prx) <= c55.NEUTRAL_TOL) & (np.abs(prx) / scale <= c55.c12.NEUTRAL_RATIO)
        is_null[ti] = False
        r_max = c55.route(ti, N, fcost, ccost, is_null, "max")
        r_sum, sM, sC = c55.route_split(ti, N, fcost, ccost, is_null)
        two = float(min([fcost[ti][b] for b in range(N) if b != ti and is_null[b]] or [np.inf]))
        res[t] = {"bottleneck_any_length": float(r_max), "bottleneck_length_two": two,
                  "total": float(r_sum), "footprint_part": float(sM), "concept_part": float(sC)}
    return res, smd


def main():
    names, FP, PRX, n_img = c55.load_design()
    idx = {a: i for i, a in enumerate(names)}
    z = np.load(os.path.join(RES, "c12_preds.npz"), allow_pickle=True)
    mal = np.asarray(z["img_index"])                       # indices of the 83 malignant images in 0..182
    p0 = {m: np.asarray(z["p0_" + m]) for m in MODELS}
    floor = np.all([p0[m] >= 0.1 for m in MODELS], axis=0)  # the library protocol's floor set
    cols58 = mal[floor]
    c68 = json.load(open(os.path.join(RES, "c68_floor_sets.json")))
    import audit_cohort                                      # noqa: E402
    _i, _l, pid, ids = audit_cohort.load()
    ids = [str(x) for x in ids]
    keep57 = set(c68["variants"]["c10_batch"]["isic_ids"])
    cols57 = np.array([i for i in range(n_img) if ids[i] in keep57])
    sets = {"all_183": np.arange(n_img), "malignant_83": mal, "library_floor_58": cols58, "diagnostic_T_floor_57": cols57}
    out = {"note": "post hoc; model-free except for stored responses of c12 and c34", "n_images": {k: int(len(v)) for k, v in sets.items()}}

    # A. exact graph on every set
    A = {}
    for k, cols in sets.items():
        sp = shared_pairs(FP, names, cols)
        A[k] = {"n_shared_footprint_pairs": len(sp), "pairs": sp}
        print("%-22s %3d images: %d shared footprint pairs" % (k, len(cols), len(sp)), flush=True)
    out["G0_shared_pairs"] = A
    base = sorted(map(tuple, A["all_183"]["pairs"]))
    out["G0_same_pairs_on_every_set"] = all(sorted(map(tuple, A[k]["pairs"])) == base for k in A)

    # C. the library as c12 built it: same edits, same images, same order (texture donor = next
    # malignant image), so footprints, proxies and responses refer to the same performed edits.
    imgs01, labels = _i, _l
    assert np.array_equal(np.where(np.asarray(labels) == 1)[0], mal)
    if os.path.exists(CACHE):
        zc = np.load(CACHE, allow_pickle=True)
        assert [str(a) for a in zc["names"]] == names
        FPm, PRXm = zc["FP"], zc["PRX"]
        STRd = {str(a): zc["STR_" + str(a)] for a in zc["str_names"]}
        print("read the library rebuild from", CACHE, flush=True)
    else:
        print("rebuilding the library on the 83 malignant images in c12's order ...", flush=True)
        c55.c25.build_design(imgs01, mal)
        FPd, PRXd, STRd = c55.c25.build_design.FP, c55.c25.build_design.PRX, c55.c25.build_design.STR
        assert sorted(FPd) == names
        FPm = np.stack([np.stack([np.asarray(FPd[a][k], float) for k in c55.COVS], 0) for a in names], 0)
        PRXm = np.stack([np.asarray(PRXd[a], float) for a in names], 0)
        STRd = {a: np.asarray(v, float) for a, v in STRd.items()}
        np.savez_compressed(CACHE, names=np.array(names), FP=FPm, PRX=PRXm, str_names=np.array(sorted(STRd)),
                            **{"STR_" + a: STRd[a] for a in STRd})
    out["library_construction_83"] = {}
    for k, cols in (("malignant_83", np.arange(len(mal))), ("library_floor_58", np.where(floor)[0])):
        sp = shared_pairs(FPm, names, cols)
        same_strength = {}
        for a, b in sp:
            if a in STRd and b in STRd:
                sa, sb = np.asarray(STRd[a])[cols], np.asarray(STRd[b])[cols]
                same_strength["%s=%s" % (a, b)] = bool(np.array_equal(sa, sb))
        out["library_construction_83"][k] = {"n_shared_footprint_pairs": len(sp), "pairs": sp,
                                             "same_family_same_strength_on_every_image": same_strength}
        print("  c12 construction, %-18s %d shared pairs; same-strength checks %s" % (k, len(sp), same_strength), flush=True)
    dd = np.abs(FPm - FP[:, :, mal]).max(axis=1)              # edit x image
    diff_vs_cache = float(np.nanmax(dd))
    out["library_construction_83"]["edits_images_differing_from_183_order_build"] = sorted(
        {"%s @ image %d" % (names[i], int(mal[j])) for i, j in zip(*np.where(dd > 0))})
    out["library_construction_83"]["max_abs_footprint_difference_vs_183_order_build"] = diff_vs_cache
    print("  largest footprint difference against the 183-order build on the same images: %.3g" % diff_vs_cache, flush=True)

    # B. diagnostic graph, Lipschitz floor and break-even on the library floor set, with the
    # footprints of the library's own construction
    gr183, smd183 = graph_on(FP, PRX, np.arange(n_img), idx, names)
    _, smd57 = graph_on(FP, PRX, cols57, idx, names)
    FP, PRX = FPm, PRXm
    cols58 = np.where(floor)[0]
    gr, smd58 = graph_on(FP, PRX, cols58, idx, names)
    out["G_a_library_floor_58"] = gr
    out["G_a_all_183_recomputed"] = gr183
    for t in TARGETS:
        print("  %-14s 58 images: bottleneck %.2f, length-two %.2f, total %.2f = %.2f + %.2f | 183: %.2f, %.2f, %.2f"
              % (t, gr[t]["bottleneck_any_length"], gr[t]["bottleneck_length_two"], gr[t]["total"],
                 gr[t]["footprint_part"], gr[t]["concept_part"], gr183[t]["bottleneck_any_length"],
                 gr183[t]["bottleneck_length_two"], gr183[t]["total"]), flush=True)

    c34 = json.load(open(os.path.join(RES, "c34_ctrl_vs_ctrl.json")))
    neutral = c34["neutral_sets"]
    pairs = []
    for m in MODELS:
        for t in TARGETS:
            for a, b in itertools.combinations(neutral[t], 2):
                ka, kb = "%s|%s" % (t, a), "%s|%s" % (t, b)
                da = np.asarray(z["dl_%s|%s" % (m, ka)])[floor]
                db = np.asarray(z["dl_%s|%s" % (m, kb)])[floor]
                diff = float((da - db).mean())
                d58 = float(smd58[idx[ka], idx[kb]]) / c55.SMD_TOL
                d183 = float(smd183[idx[ka], idx[kb]]) / c55.SMD_TOL
                pairs.append({"model": NAME[m], "target": t, "a": a, "b": b, "abs_diff_library_58": abs(diff),
                              "distance_58": d58, "distance_183": d183, "implied_L_58": abs(diff) / d58})
    # c34's own numbers, with the distance recomputed on its 57 images (183-order build)
    c34_57 = []
    for pr in c34["pairs"]:
        ka, kb = "%s|%s" % (pr["target"], pr["a"]), "%s|%s" % (pr["target"], pr["b"])
        d57 = float(smd57[idx[ka], idx[kb]]) / c55.SMD_TOL
        d183 = float(smd183[idx[ka], idx[kb]]) / c55.SMD_TOL
        c34_57.append({"model": pr["model"], "target": pr["target"], "a": pr["a"], "b": pr["b"],
                       "abs_diff_c34_57": abs(pr["diff"]), "distance_57": d57, "distance_183": d183,
                       "implied_L_57": abs(pr["diff"]) / d57, "implied_L_183": abs(pr["diff"]) / d183})
    # patient bootstrap on the 58 images: response difference and footprint distance of every pair
    pid58 = np.asarray(pid)[mal][floor]
    upat = sorted(set(pid58.tolist()))
    by_pat = [np.where(pid58 == p)[0] for p in upat]
    rng = np.random.RandomState(7)
    keys = [("%s|%s" % (p["target"], p["a"]), "%s|%s" % (p["target"], p["b"]), [m for m in MODELS if NAME[m] == p["model"]][0]) for p in pairs]
    vals = [np.asarray(z["dl_%s|%s" % (m, ka)])[floor] - np.asarray(z["dl_%s|%s" % (m, kb)])[floor] for ka, kb, m in keys]
    bd, bs = [[] for _ in pairs], [[] for _ in pairs]
    sub = FP[:, :, cols58]
    for _ in range(BOOT):
        take = np.concatenate([by_pat[i] for i in rng.randint(0, len(by_pat), len(by_pat))])
        fm, fv = sub[:, :, take].mean(axis=2), sub[:, :, take].var(axis=2, ddof=1)
        sb = c55.smd_matrix(fm, fv)
        for q, (ka, kb, m) in enumerate(keys):
            bd[q].append(float(vals[q][take].mean())); bs[q].append(float(sb[idx[ka], idx[kb]]) / c55.SMD_TOL)
    for q, p in enumerate(pairs):
        lo, hi = np.percentile(bd[q], [2.5, 97.5])
        p["ci_patient"] = [float(lo), float(hi)]
        p["excludes_zero"] = bool(lo > 0 or hi < 0)
        num_lo = 0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi))
        p["implied_L_conservative"] = num_lo / float(np.percentile(bs[q], 97.5))
    Ls = sorted(p["implied_L_58"] for p in pairs)
    w58 = max(pairs, key=lambda x: x["implied_L_58"])
    w57 = max(c34_57, key=lambda x: x["implied_L_57"])
    L58, L57 = w58["implied_L_58"], w57["implied_L_57"]
    out["lipschitz_library_floor_58"] = {"pairs": pairs, "binding": w58, "L_floor": L58,
                                         "median_implied_L": float(np.median(Ls)),
                                         "quartiles_implied_L": [float(np.percentile(Ls, 25)), float(np.percentile(Ls, 75))],
                                         "second_largest_L": float(Ls[-2]),
                                         "conservative_L": float(max(p["implied_L_conservative"] for p in pairs)),
                                         "n_pairs": len(pairs), "n_excluding_zero": int(sum(p["excludes_zero"] for p in pairs)),
                                         "n_patients": len(upat),
                                         "diff_median": float(np.median([p["abs_diff_library_58"] for p in pairs])),
                                         "diff_range": [min(p["abs_diff_library_58"] for p in pairs),
                                                        max(p["abs_diff_library_58"] for p in pairs)]}
    out["lipschitz_c34_on_its_57"] = {"pairs": c34_57, "binding": w57, "L_floor": L57}
    print("  L_g floor, library protocol on 58 images: %.3f (%s %s %s-%s: %.2f logits at %.2f units)"
          % (L58, w58["model"], w58["target"], w58["a"], w58["b"], w58["abs_diff_library_58"], w58["distance_58"]), flush=True)
    print("  L_g floor, c34 responses with distances on its 57 images: %.3f (%s %s %s-%s: %.2f at %.2f; 183: %.2f)"
          % (L57, w57["model"], w57["target"], w57["a"], w57["b"], w57["abs_diff_c34_57"], w57["distance_57"],
             w57["distance_183"]), flush=True)

    c12 = json.load(open(os.path.join(RES, "c12_matched_control.json")))["models"]
    cells = [float(mm["dcace"][t]["u:hue"]["mean"]) for mm in c12.values() for t in TARGETS]
    span = float(max(cells) - min(cells))
    be = {}
    for t in TARGETS:
        hw = L58 * gr[t]["footprint_part"]
        rem = span / 2.0 - hw
        cp = gr[t]["concept_part"]
        be[t] = {"half_width_footprint": hw, "break_even_kappaLh": (rem / cp) if (cp > 1e-9 and rem > 0) else 0.0,
                 "vacuous": bool(rem <= 0)}
        print("  %-14s half-width %.2f, break-even %s" % (t, hw, "vacuous" if rem <= 0 else "%.3f" % be[t]["break_even_kappaLh"]), flush=True)
    out["break_even_library_floor_58"] = {"score_span": span, "score_ends": [min(cells), max(cells)], "targets": be}
    # component range of Consequence 2 with the library responses on the 83 malignant images
    B = float(np.log((1 - 1e-6) / 1e-6))
    rr = [float(np.asarray(z["dl_%s|%s" % (m, t)]).mean()) for m in MODELS for t in TARGETS]
    out["component_range_library_83"] = {"responses_min_max": [min(rr), max(rr)],
                                         "min_share_of_cap_range": (2 * B - max(abs(x) for x in rr)) / (2 * B)}
    print("  component range on 83 images: responses %.2f to %.2f, interval covers at least %.4f of the cap range"
          % (min(rr), max(rr), out["component_range_library_83"]["min_share_of_cap_range"]), flush=True)
    # c55 combines route totals on the 183 images with the c34 responses on 57 images; Statement 2 does not allow that
    out["cross_set_combination_c55"] = {"note": "c55: route totals on 183 images, L_g from c34 responses (57 images) over 183-image distances"}
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
