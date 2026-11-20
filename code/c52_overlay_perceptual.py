#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C52: how far does the synthetic overlay move the image, and against what scale?

Why. The heaviest assumption of the artifact tier is the transfer from T_U to do(U), and the paper cannot
test it because the cohort has no artifact annotations. But one part can be tested without any
annotation: **the size of the intervention on the image itself**. If the overlay is a perceptually small change
and the model still reacts strongly, that reaction cannot be attributed to "the image being destroyed".

The scale must be self-supplied, not borrowed from value ranges in other literature. So three quantities are measured
with the same ruler and compared with each other:
    (a) distance between original image <-> image with a held-out overlay (hair, ruler)
    (b) distance between original image <-> image with a global photometric edit (vignette, jitter),
        which changes 68-96% of pixels yet the model barely reacts
    (c) distance between two different lesions in the same cohort: the natural scale of the image set

Two rulers, one deep and one shallow, so the conclusion does not depend on the choice of ruler:
    - ResNet-50 features trained on ImageNet, independent of every audited checkpoint
      (this is what learned perceptual metrics approximate: distance in a deep feature space)
    - CIELAB ΔE76 averaged over pixels, purely image-based, with no network

This does not show that the overlay resembles a real artifact. It only quantifies that the intervention
is small on the scale of the image set itself. That is exactly one half of the transfer question, and the only half
that can be answered without paired images.

Out -> results/c52_overlay_perceptual.json
Run:  python Causal_XAI_v3/code/c52_overlay_perceptual.py
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
import c2_cace_artifact as c2m       # noqa: E402
import c3_cace_concept as c3m        # noqa: E402
import c12_matched_control as c12    # noqa: E402  (rgb2lab)
import c30_disjoint_audit as c30     # noqa: E402
import audit_cohort                  # noqa: E402

p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c52_overlay_perceptual.json")
HELD = c30.HELD
GLOBAL = ["vignette", "jitter"]
SEED, N_PAIR = 42, 2000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@torch.no_grad()
def feats(net, batch):
    out = []
    for i in range(0, len(batch), 32):
        x = p2._to_norm(torch.stack(batch[i:i + 32])).to(DEVICE)
        out.append(net(x).flatten(1).float().cpu().numpy())
    return np.concatenate(out)


def lab_de(a, b):
    """Mean per-pixel ΔE76 between two [0,1] images."""
    return float((c12.rgb2lab(b) - c12.rgb2lab(a)).pow(2).sum(0).sqrt().mean())


def main():
    t0 = time.time()
    imgs, labels, pid, _ = audit_cohort.load()
    n = len(imgs)
    log(f"audit set: {n} images")

    per, cf = c2m.precompute_edits(imgs, c30.ART, SEED)
    log("overlays done; % pixels changed: " + " ".join(f"{a}={cf[a]*100:.2f}" for a in HELD + GLOBAL))

    # Extract features with an ImageNet ResNet-50, not an audited checkpoint.
    bb = tvm.resnet50(weights="IMAGENET1K_V2")
    bb.fc = torch.nn.Identity()
    bb = bb.to(DEVICE).eval()
    F0 = feats(bb, imgs)
    nrm = np.linalg.norm(F0, axis=1)

    res = {"n_images": int(n), "seed": SEED, "n_pairs": N_PAIR,
           "feature_extractor": "torchvision resnet50 IMAGENET1K_V2, pooled, independent of the audited checkpoints",
           "note": ("How far the synthetic overlay moves an image, measured on two scales the "
                    "cohort itself supplies: a global photometric edit that alters most pixels, "
                    "and the distance between two different lesions. This bounds the size of the "
                    "intervention as an image change. It does not establish that the overlay "
                    "resembles a real artifact, which needs paired images this cohort lacks."),
           "pixel_change_frac": {a: float(cf[a]) for a in HELD + GLOBAL},
           "edits": {}}

    for a in HELD + GLOBAL:
        Fe = feats(bb, [d[a] for d in per])
        d_feat = np.linalg.norm(Fe - F0, axis=1) / np.maximum(nrm, 1e-9)
        d_lab = np.array([lab_de(imgs[i], per[i][a]) for i in range(n)])
        res["edits"][a] = {
            "feature_distance_median": float(np.median(d_feat)),
            "feature_distance_iqr": [float(np.percentile(d_feat, 25)), float(np.percentile(d_feat, 75))],
            "lab_de_median": float(np.median(d_lab)),
            "pixel_change_frac": float(cf[a])}
        log(f"  {a:12s} feature {np.median(d_feat):.4f}  ΔE {np.median(d_lab):5.2f}  "
            f"pixel {cf[a]*100:5.2f}%")

    # Natural scale: distance between two different lesions in the same cohort.
    rng = np.random.RandomState(SEED)
    i1 = rng.randint(0, n, N_PAIR); i2 = rng.randint(0, n, N_PAIR)
    keep = i1 != i2
    i1, i2 = i1[keep], i2[keep]
    dp_feat = np.linalg.norm(F0[i1] - F0[i2], axis=1) / np.maximum(nrm[i1], 1e-9)
    dp_lab = np.array([lab_de(imgs[a], imgs[b]) for a, b in zip(i1[:300], i2[:300])])
    res["between_lesion"] = {
        "n_pairs": int(len(i1)),
        "feature_distance_median": float(np.median(dp_feat)),
        "feature_distance_iqr": [float(np.percentile(dp_feat, 25)), float(np.percentile(dp_feat, 75))],
        "lab_de_median": float(np.median(dp_lab))}
    log(f"  {'between lesions':12s} feature {np.median(dp_feat):.4f}  ΔE {np.median(dp_lab):5.2f}")

    ref_f = res["between_lesion"]["feature_distance_median"]
    ref_l = res["between_lesion"]["lab_de_median"]
    for a in HELD + GLOBAL:
        e = res["edits"][a]
        e["feature_ratio_to_between_lesion"] = e["feature_distance_median"] / ref_f
        e["lab_ratio_to_between_lesion"] = e["lab_de_median"] / ref_l
    hair = res["edits"]["hair"]
    res["summary"] = {
        "hair_feature_ratio": hair["feature_ratio_to_between_lesion"],
        "hair_lab_ratio": hair["lab_ratio_to_between_lesion"],
        "held_out_feature_ratio_range": [min(res["edits"][a]["feature_ratio_to_between_lesion"] for a in HELD),
                                         max(res["edits"][a]["feature_ratio_to_between_lesion"] for a in HELD)],
        "global_feature_ratio_range": [min(res["edits"][a]["feature_ratio_to_between_lesion"] for a in GLOBAL),
                                       max(res["edits"][a]["feature_ratio_to_between_lesion"] for a in GLOBAL)],
        "hair_vs_global_feature": hair["feature_distance_median"] /
                                  max(res["edits"][a]["feature_distance_median"] for a in GLOBAL)}
    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
