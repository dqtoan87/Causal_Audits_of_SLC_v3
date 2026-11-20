#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C83: where do the large protocol-dependent score changes of the route test come from?  (exploratory)

c78 found that scoring a route-test image alone instead of in its fixed batch of 25 moves a clipped-logit
score by a median of 0.016 logit, but 2.0% of scores move by more than 0.5 and one jumps from one clipping
bound to the other (27.6 logits). This script asks whether those moves come from an execution error or from
the numerics of the audited networks on inputs where their features are far outside the usual range.

Checks, for the second run's three checkpoints, on the registered route-test images:
  - every model is in eval mode, and the network has no layer whose eval-mode output depends on the batch
    (batch normalization in eval mode uses its running statistics; dropout and stochastic depth are off);
  - four protocols: the plan's batch in bf16 and in fp32, and one image per batch in bf16 and in fp32, so that
    batch context and precision are varied separately;
  - for each pair of protocols, the share of clipped-logit changes above 0.5 logit, split by whether the
    head's unclipped log-odds lie beyond the clip bound in either protocol, and the relative change of the
    unclipped log-odds where they are large;
  - the image behind the largest change, with its log-odds and pooled feature norm under every protocol;
  - the plan's cycle test under each protocol (cells that disagree).

Out -> results/c83_batch_extremes.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c62_route_test as c62             # noqa: E402  (pre-specified; imported only)
import c67_route_sensitivity as c67      # noqa: E402
import c69_route_clip_sensitivity as c69  # noqa: E402

RES = c62.RES
OUT = os.path.join(RES, "c83_batch_extremes.json")
CR = c62.CRIT
TKS = c67.TKS
ETA = 1e-6
BOUND = float(np.log((1 - ETA) / ETA))
PROTOCOLS = [("stack_bf16", False, False), ("stack_fp32", False, True),
             ("single_bf16", True, False), ("single_fp32", True, True)]
PAIRS = [("stack_bf16", "single_bf16", "batch context, bf16"),
         ("stack_fp32", "single_fp32", "batch context, fp32"),
         ("stack_bf16", "stack_fp32", "precision, plan's batch"),
         ("stack_bf16", "single_fp32", "both")]


def batch_dependent_layers(model):
    bad = []
    for name, mod in model.named_modules():
        if mod.training:
            bad.append(name or "<root>")
        if isinstance(mod, (torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.InstanceNorm2d)):
            continue
        if "BatchNorm" in type(mod).__name__ and getattr(mod, "track_running_stats", True) is False:
            bad.append(name + " (batch statistics in eval)")
    return bad


def main():
    t0 = time.time()
    cons = json.load(open(c62.out_path("construction", False)))
    pids = np.asarray(cons["patient_ids"])
    stacks, cols = c69.build_stacks(cons)
    n, k = len(stacks), stacks[0].shape[0]
    r2 = c67.load_run("_inv_retrained")
    print("rebuilt %d images x %d (%.0fs)" % (n, k, time.time() - t0), flush=True)
    Z, F, P = {}, {}, {}
    layers = {}
    for m in c62.MODELS:
        model = c69.load_model(m, c62.audited_models.CKPTS[m])
        layers[m] = batch_dependent_layers(model)
        for v, single, fp32 in PROTOCOLS:
            if single:
                outs = [[c69.score_feat(model, s[j:j + 1], fp32) for j in range(k)] for s in stacks]
                P[(m, v)] = np.stack([np.concatenate([o[0] for o in row]) for row in outs])
                Z[(m, v)] = np.stack([np.concatenate([o[1] for o in row]) for row in outs])
                F[(m, v)] = np.stack([np.concatenate([o[2] for o in row]) for row in outs])
            else:
                outs = [c69.score_feat(model, s, fp32) for s in stacks]
                P[(m, v)] = np.stack([o[0] for o in outs]); Z[(m, v)] = np.stack([o[1] for o in outs])
                F[(m, v)] = np.stack([o[2] for o in outs])
        del model
        if c62.DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("  scored %s (%.0fs); eval-mode issues: %s" % (m, time.time() - t0, layers[m] or "none"), flush=True)
    chk = {m: float(np.abs(P[(m, "stack_bf16")] - r2["prob"][m]).max()) for m in c62.MODELS}
    print("plan protocol vs stored scores, max |dp|:", chk, flush=True)

    g1 = c67.g1_masks(cons)
    common2 = np.all([r2["prob"][m][:, 0] >= CR["score_floor"] for m in c62.MODELS], 0)
    masks = {(m, tk): g1[tk] & common2 for m in c62.MODELS for tk in TKS}
    prim = np.zeros((n, k), bool)
    for tk in TKS:
        msk = g1[tk] & common2
        prim[msk, 0] = True
        for c in ["edit"] + c62.CONTROLS:
            prim[msk, cols.index("%s|%s" % (tk, c))] = True
    L = {key: c62.c3m.logit(v) for key, v in P.items()}
    fmed = {m: float(np.median(F[(m, "stack_bf16")][:, 0])) for m in c62.MODELS}
    out = {"note": "exploratory; second run's checkpoints; clipped logit (5) at eta=1e-6, bound %.4f" % BOUND,
           "check_stack_bf16_vs_stored_max_abs_prob": chk, "eval_mode_or_batch_statistics_issues": layers,
           "median_unedited_feature_norm": fmed, "pairs": {}, "cycle": {}}
    for a, b, label in PAIRS:
        rec = {"label": label}
        d_all, sat_any, big_rel, zmax, fr = [], [], [], [], []
        for m in c62.MODELS:
            d = np.abs(L[(m, a)] - L[(m, b)])[prim]
            za, zb = Z[(m, a)][prim], Z[(m, b)][prim]
            mx = np.maximum(np.abs(za), np.abs(zb))
            fr.append(np.maximum(F[(m, a)][prim], F[(m, b)][prim]) / fmed[m])
            d_all.append(d); sat_any.append(mx > BOUND); zmax.append(mx)
            rel = np.abs(za - zb) / np.maximum(mx, 1e-12)
            big_rel.append(rel[mx > BOUND])
        d, s, zm = np.concatenate(d_all), np.concatenate(sat_any), np.concatenate(zmax)
        br = np.concatenate(big_rel)
        fr = np.concatenate(fr)
        big = d > 0.5
        rec.update({"share_large_moves_feature_norm_over_10x": float((fr[big] > 10).mean()) if big.any() else None,
                    "median_feature_norm_ratio_large_moves": float(np.median(fr[big])) if big.any() else None,
                    "share_all_scores_feature_norm_over_10x": float((fr > 10).mean()),
                    "share_above_0.5_among_feature_norm_at_most_10x": float(big[fr <= 10].mean()),
                    "max_change_among_feature_norm_at_most_10x": float(d[fr <= 10].max())})
        rec.update({"n_scores": int(d.size), "share_above_0.5": float(big.mean()),
                    "n_above_0.5": int(big.sum()),
                    "share_of_large_moves_with_logodds_beyond_bound_in_either": float(s[big].mean()) if big.any() else None,
                    "share_above_0.5_among_unsaturated": float(big[~s].mean()) if (~s).any() else None,
                    "max_change_among_unsaturated": float(d[~s].max()) if (~s).any() else None,
                    "median_change_among_unsaturated": float(np.median(d[~s])) if (~s).any() else None,
                    "share_scores_beyond_bound_in_either": float(s.mean()),
                    "median_relative_logodds_change_beyond_bound": float(np.median(br)) if br.size else None,
                    "q95_relative_logodds_change_beyond_bound": float(np.quantile(br, 0.95)) if br.size else None,
                    "max_change": float(d.max())})
        out["pairs"]["%s_vs_%s" % (a, b)] = rec
        print("   feature norm: large moves >10x median %.3f (median ratio %.1f); all scores >10x %.4f; share>0.5 at <=10x %.5f, max %.3f"
              % (rec["share_large_moves_feature_norm_over_10x"], rec["median_feature_norm_ratio_large_moves"],
                 rec["share_all_scores_feature_norm_over_10x"], rec["share_above_0.5_among_feature_norm_at_most_10x"],
                 rec["max_change_among_feature_norm_at_most_10x"]), flush=True)
        print("%-26s >0.5: %.4f (n=%d); of those beyond bound: %s; unsaturated: share>0.5 %s, max %.3f; rel. log-odds change median %.4f"
              % (label, rec["share_above_0.5"], rec["n_above_0.5"], rec["share_of_large_moves_with_logodds_beyond_bound_in_either"],
                 rec["share_above_0.5_among_unsaturated"], rec["max_change_among_unsaturated"] or -1,
                 rec["median_relative_logodds_change_beyond_bound"] or -1), flush=True)
    # the image behind the largest bf16 batch-context change
    best = None
    for m in c62.MODELS:
        d = np.abs(L[(m, "stack_bf16")] - L[(m, "single_bf16")]) * prim
        i, j = np.unravel_index(np.argmax(d), d.shape)
        if best is None or d[i, j] > best[0]:
            best = (float(d[i, j]), m, int(i), int(j))
    _, m, i, j = best
    out["largest_change"] = {"model": m, "image": cons["ids"][i], "column": cols[j], "change": best[0],
                             "logodds": {v: float(Z[(m, v)][i, j]) for v, _, _ in PROTOCOLS},
                             "feature_norm_over_median_unedited": {v: float(F[(m, v)][i, j] / fmed[m]) for v, _, _ in PROTOCOLS}}
    print("largest change:", out["largest_change"], flush=True)
    for v, _, _ in PROTOCOLS:
        DL = {m: L[(m, v)] - L[(m, v)][:, :1] for m in c62.MODELS}
        r = c67.cycle(DL, cols, masks, pids)
        out["cycle"][v] = {"crit": r["crit"], "summary": r["summary"],
                           "disagreeing_cells": sorted(c for c, x in r["cells"].items() if x == "routes disagree")}
        print("  cycle %-12s %s %s" % (v, r["summary"], out["cycle"][v]["disagreeing_cells"]), flush=True)
    np.savez_compressed(os.path.join(RES, "c83_batch_extremes_scores.npz"), columns=np.array(cols), isic_ids=np.array(cons["ids"]),
                        primary=prim, **{"logodds_%s_%s" % key: v for key, v in Z.items()},
                        **{"featnorm_%s_%s" % key: v for key, v in F.items()})
    out["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
