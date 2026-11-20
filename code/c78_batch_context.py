#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C78: does the score of an edited image depend on the images scored with it?  (exploratory)

The route test scores each unedited image in one bf16 batch together with its own edits and controls, in a
fixed order. The audited f of the route test is therefore protocol-indexed: f_B(e(X_i); B_i), with B_i the
fixed list of images in that batch. This script asks how much that context matters. For the second run's
three checkpoints it rescores every registered route-test image under four protocols:

  stack_bf16   the plan's protocol: one bf16 batch per image with its edits (must reproduce the stored scores)
  single_bf16  every image alone, batch size 1, bf16: a function of the image only, f_1(x)
  single_fp32  every image alone, batch size 1, full precision
  flat32_bf16  bf16 batches of 32 that mix images (as in C67)

For each protocol it reports the change of every clipped-logit score against the plan's protocol (median,
95th percentile, maximum, share above 0.5 logit), and the plan's cycle test on the plan's primary set.

Out -> results/c78_batch_context.json
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
OUT = os.path.join(RES, "c78_batch_context.json")
CR = c62.CRIT
TKS = c67.TKS
KEYS = ["erm|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-hue4",
        "slim|color_var@2|D_rad4-rad16"]


def main():
    t0 = time.time()
    cons = json.load(open(c62.out_path("construction", False)))
    pids = np.asarray(cons["patient_ids"])
    stacks, cols = c69.build_stacks(cons)
    n, k = len(stacks), stacks[0].shape[0]
    flat = torch.cat(stacks)
    r2 = c67.load_run("_inv_retrained")
    print("rebuilt (%.0fs)" % (time.time() - t0), flush=True)
    P = {}
    for m in c62.MODELS:
        model = c62.p1.build_model("erm" if m == "erm" else "slim", c62.p1.CFG["z_dim"]).to(c62.DEVICE)
        model.load_state_dict(torch.load(c62.audited_models.CKPTS[m], map_location=c62.DEVICE)); model.eval()
        P[(m, "stack_bf16")] = np.stack([c67.score(model, s, False)[0] for s in stacks])
        for v, fp32 in (("single_bf16", False), ("single_fp32", True)):
            P[(m, v)] = np.stack([np.concatenate([c67.score(model, s[j:j + 1], fp32)[0] for j in range(k)]) for s in stacks])
        pf = [c67.score(model, flat[j:j + 32], False)[0] for j in range(0, flat.shape[0], 32)]
        P[(m, "flat32_bf16")] = np.concatenate(pf).reshape(n, k)
        del model
        if c62.DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("  scored %s (%.0fs)" % (m, time.time() - t0), flush=True)
    chk = {m: float(np.abs(P[(m, "stack_bf16")] - r2["prob"][m]).max()) for m in c62.MODELS}
    g1 = c67.g1_masks(cons)
    common2 = np.all([r2["prob"][m][:, 0] >= CR["score_floor"] for m in c62.MODELS], 0)
    masks = {(m, tk): g1[tk] & common2 for m in c62.MODELS for tk in TKS}
    prim = np.zeros((n, k), bool)                 # unedited, edits and controls of the primary sets
    for tk in TKS:
        msk = g1[tk] & common2
        prim[msk, 0] = True
        for c in ["edit"] + c62.CONTROLS:
            prim[msk, cols.index("%s|%s" % (tk, c))] = True
    L = {key: c62.c3m.logit(v) for key, v in P.items()}
    out = {"note": "exploratory; second run's checkpoints; scores on the clipped logit scale (5)",
           "check_stack_bf16_vs_stored_max_abs_prob": chk, "protocols": {}}
    ref = None
    for v in ("stack_bf16", "single_bf16", "single_fp32", "flat32_bf16"):
        d = np.concatenate([np.abs(L[(m, v)] - L[(m, "stack_bf16")])[prim] for m in c62.MODELS])
        DL = {m: L[(m, v)] - L[(m, v)][:, :1] for m in c62.MODELS}
        r = c67.cycle(DL, cols, masks, pids)
        if ref is None:
            ref = r
        dd = [abs(r["components"][c]["est"] - ref["components"][c]["est"]) for c in ref["components"]]
        out["protocols"][v] = {"score_change_median": float(np.median(d)), "score_change_q95": float(np.quantile(d, 0.95)),
                               "score_change_max": float(d.max()), "share_above_0.5": float((d > 0.5).mean()),
                               "n_scores": int(d.size), "crit": r["crit"], "summary": r["summary"],
                               "disagreeing_cells": sorted(c for c, x in r["cells"].items() if x == "routes disagree"),
                               "max_abs_change_D": float(max(dd)),
                               "key": {kk: r["components"][kk] for kk in KEYS}}
        q = out["protocols"][v]
        print("%-12s score change median %.4f q95 %.3f max %.2f share>0.5 %.3f | %s %s | max|dD| %.2f"
              % (v, q["score_change_median"], q["score_change_q95"], q["score_change_max"], q["share_above_0.5"],
                 q["summary"], q["disagreeing_cells"], q["max_abs_change_D"]), flush=True)
        for kk in KEYS:
            e = q["key"][kk]
            print("     %-32s %+.2f [%.2f, %.2f] %s" % (kk, e["est"], e["sim"][0], e["sim"][1], e["decision"]), flush=True)
    out["runtime_sec"] = round(time.time() - t0, 1)
    print("stack_bf16 vs stored:", chk)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
