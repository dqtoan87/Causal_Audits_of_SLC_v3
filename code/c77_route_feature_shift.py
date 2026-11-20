#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C77: how far the route-test images move in an independent feature space  (exploratory, model-free)

The audited backbones put very large pooled features on some control images. Those features belong to the
audited networks, so they show an anomaly of those networks, not that an image has left the range of
natural images. This script uses an encoder that no audited checkpoint shares: the pooled feature of the
torchvision ImageNet ResNet-50 (IMAGENET1K_V2), as in the perceptual check of C52.

The route-test images are rebuilt exactly from the registered construction (as C67 and C69 do). For every
image of fold 1 it computes the feature of the unedited image, of the concept edit and of the four controls
at each target. The natural reference is the distance from each unedited image to its nearest other
unedited image of the fold. For each edit and control it reports, on the second run's primary set of the
target: the median shift from its own unedited image as a multiple of the median natural nearest-neighbour
distance, the share of images shifted further than the 95th percentile of those natural distances, and the
largest feature norm as a multiple of the median unedited norm. The threshold is descriptive; nothing here
is a test of positivity.

Out -> results/c77_route_feature_shift.json
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
import c62_route_test as c62            # noqa: E402  (pre-specified; imported only)
import c67_route_sensitivity as c67     # noqa: E402
import c69_route_clip_sensitivity as c69  # noqa: E402

RES = c62.RES
OUT = os.path.join(RES, "c77_route_feature_shift.json")
TKS = c67.TKS


@torch.no_grad()
def feats(net, batch):
    out = []
    for i in range(0, len(batch), 32):
        x = c62.c3m.p2._to_norm(batch[i:i + 32]).to(c62.DEVICE)
        out.append(net(x).flatten(1).float().cpu().numpy())
    return np.concatenate(out)


def main():
    t0 = time.time()
    cons = json.load(open(c62.out_path("construction", False)))
    stacks, cols = c69.build_stacks(cons)
    print("rebuilt %d images x %d columns (%.0fs)" % (len(stacks), len(cols), time.time() - t0), flush=True)
    net = tvm.resnet50(weights="IMAGENET1K_V2"); net.fc = torch.nn.Identity(); net = net.to(c62.DEVICE).eval()
    F = np.stack([feats(net, s) for s in stacks])                     # images x columns x 2048
    print("features done (%.0fs)" % (time.time() - t0), flush=True)
    F0 = F[:, 0]
    D0 = np.linalg.norm(F0[:, None] - F0[None], axis=2); np.fill_diagonal(D0, np.inf)
    nn = D0.min(1)
    ref_med, ref_q95 = float(np.median(nn)), float(np.percentile(nn, 95))
    n0 = np.linalg.norm(F0, axis=1); n0_med = float(np.median(n0))
    g1 = c67.g1_masks(cons)
    r2 = c67.load_run("_inv_retrained")
    common = np.all([r2["prob"][m][:, 0] >= c62.CRIT["score_floor"] for m in c62.MODELS], 0)
    out = {"encoder": "torchvision resnet50 IMAGENET1K_V2, pooled; shares no weights with the audited checkpoints",
           "n_images": len(stacks), "natural_nn_median": ref_med, "natural_nn_q95": ref_q95,
           "unedited_norm_median": n0_med, "unedited_norm_max_ratio": float(n0.max() / n0_med), "targets": {}}
    for tk in TKS:
        msk = g1[tk] & common
        res = {"n_images": int(msk.sum())}
        for c in ["edit"] + c62.CONTROLS:
            j = cols.index("%s|%s" % (tk, c))
            sh = np.linalg.norm(F[:, j] - F0, axis=1)[msk]
            nr = np.linalg.norm(F[:, j], axis=1)[msk] / n0_med
            res[c] = {"shift_median_over_nn": float(np.median(sh) / ref_med), "share_beyond_q95": float((sh > ref_q95).mean()),
                      "norm_ratio_max": float(nr.max()), "norm_ratio_median": float(np.median(nr))}
        out["targets"][tk] = res
        print("%-12s n=%d  " % (tk, res["n_images"]) + "  ".join(
            "%s: shift %.2fx, beyond q95 %.0f%%, norm max %.2f" % (c, res[c]["shift_median_over_nn"], 100 * res[c]["share_beyond_q95"],
                                                                 res[c]["norm_ratio_max"]) for c in ["edit"] + c62.CONTROLS), flush=True)
    allmax = [out["targets"][tk][c]["norm_ratio_max"] for tk in TKS for c in ["edit"] + c62.CONTROLS]
    out["largest_norm_ratio"] = float(max(allmax))
    out["runtime_sec"] = round(time.time() - t0, 1)
    print("natural NN distance median %.2f q95 %.2f; largest norm ratio %.2f (unedited max %.2f)"
          % (ref_med, ref_q95, out["largest_norm_ratio"], out["unedited_norm_max_ratio"]))
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
