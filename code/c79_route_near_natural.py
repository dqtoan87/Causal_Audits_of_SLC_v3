#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C79: the cycle test on images where both controls of a pair stay near natural images  (exploratory)

C77 showed that, in an independent ImageNet ResNet-50 feature space, the high-dose controls move many images
further from their unedited version than the nearest other unedited image lies. This script keeps, for each
difference, only the images of the run's primary set on which both of its controls move the image by no more
than the 95th percentile of the natural nearest-neighbour distance of C77. The threshold is fixed by the
unedited images alone and reads no model output. The 48 differences are then judged as in C74: one patient
bootstrap (20,000 resamples, the plan's seed), one simultaneous 95% band, the three-way rule at 0.5 logit.
Reported for both runs, with all 48 differences and with the plan's 30-image rule.

Out -> results/c79_route_near_natural.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torchvision.models as tvm

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c62_route_test as c62              # noqa: E402  (pre-specified; imported only)
import c67_route_sensitivity as c67       # noqa: E402
import c69_route_clip_sensitivity as c69  # noqa: E402
import c74_route_exact_match as c74       # noqa: E402
import c77_route_feature_shift as c77     # noqa: E402

RES = c62.RES
OUT = os.path.join(RES, "c79_route_near_natural.json")
CR = c62.CRIT
TKS = c67.TKS
KEY = c74.KEY


def main():
    t0 = time.time()
    cons = json.load(open(c62.out_path("construction", False)))
    pids = np.asarray(cons["patient_ids"])
    stacks, cols = c69.build_stacks(cons)
    net = tvm.resnet50(weights="IMAGENET1K_V2"); net.fc = torch.nn.Identity(); net = net.to(c62.DEVICE).eval()
    F = np.stack([c77.feats(net, s) for s in stacks])
    F0 = F[:, 0]
    D0 = np.linalg.norm(F0[:, None] - F0[None], axis=2); np.fill_diagonal(D0, np.inf)
    q95 = float(np.percentile(D0.min(1), 95))
    ref = json.load(open(os.path.join(RES, "c77_route_feature_shift.json")))
    near = {}
    for tk in TKS:
        for c in c62.CONTROLS:
            near[(tk, c)] = np.linalg.norm(F[:, cols.index("%s|%s" % (tk, c))] - F0, axis=1) <= q95
    print("features done (%.0fs); q95 %.3f (C77 %.3f)" % (time.time() - t0, q95, ref["natural_nn_q95"]), flush=True)
    g1 = c67.g1_masks(cons)
    out = {"note": "exploratory; threshold from unedited images only", "natural_nn_q95": q95,
           "check_vs_c77_q95": abs(q95 - ref["natural_nn_q95"]), "runs": {}}
    for run, suf in (("second_run", "_inv_retrained"), ("first_run", "")):
        r = c67.load_run(suf)
        common = np.all([r["prob"][m][:, 0] >= CR["score_floor"] for m in c62.MODELS], 0)
        primary = {tk: g1[tk] & common for tk in TKS}
        out["runs"][run] = {}
        for tag, mi in (("all", 0), ("min30", CR["min_images"])):
            res = c74.cycle_pairwise(r["dl"], r["cols"], primary, near, pids, mi)
            res["key"] = {k: res["components"].get(k) for k in KEY}
            out["runs"][run][tag] = res
            dis = sorted(c for c, v in res["cells"].items() if v == "routes disagree")
            print("[%s, %s] %d analysed, crit %.3f %s disagree: %s" % (run, tag, res["n_analysed"], res["crit"], res["summary"], dis), flush=True)
            for k in KEY:
                v = res["components"].get(k)
                print("   %-32s %s" % (k, "not analysed" if v is None else "n=%d (%d) %+.2f [%.2f, %.2f] %s" % (
                    v["n_images"], v["n_patients"], v["est"], v["sim"][0], v["sim"][1], v["decision"])), flush=True)
    out["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
