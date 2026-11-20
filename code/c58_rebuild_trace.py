#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C58: layer-by-layer trace of the one cell on which the reimplementation disagrees.

The cell is the ResNet-50 invariance model on benign images. The reimplementation scores its own
checkpoint at 0.038 (within tolerance); this pipeline scores the same checkpoint on the same cohort at
0.043 to 0.051 (indeterminate). This script holds the checkpoint and the images fixed and compares
the two pipelines one layer at a time:

  L1  unedited probabilities: this pipeline's preprocessing and inference against the cached
      probabilities of the reimplementation, image by image;
  L2  per-operator magnitude on benign images: hair at three doses and the ruler, this pipeline's
      overlay generator against the reimplementation's cached edited probabilities;
  L3  the four-operator mixture, with the patient bootstrap run on the same patient identifiers;
  L4  the same mixture with this pipeline's ruler drawn bright (blend value 1.0), which is how the
      reimplementation reads the ruler colour in the specification;
  L5  the same mixture with this pipeline's hair geometry but each strand pixel blended once. This
      pipeline blends point by point along a strand, so a pixel that several consecutive points
      fall on is darkened several times; the reimplementation blends each pixel once;
  L6  the reimplementation's own model loader and inference function applied to this pipeline's
      image tensors and edits, which separates the inference step from the overlays;
  L7  the reimplementation's own image loader, model loader and inference function on its own
      checkpoint, compared with its cached unedited probabilities.

Reads the reimplementation's results/c19_preds.npz and results/audit_cohort.json.
Out -> results/c58_rebuild_trace.json
Run:  python code/c58_rebuild_trace.py --other /path/to/reimplementation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c2_cace_artifact as c2m       # noqa: E402
import c30_disjoint_audit as c30     # noqa: E402
import c51_independent_rebuild as c51  # noqa: E402

BASE = os.path.dirname(HERE)
OUT = os.path.join(BASE, "results", "c58_rebuild_trace.json")
HELD, MARGIN, SEED = c30.HELD, c30.MARGIN_P, 42
TAG = "r50_inv_s42"


def bright_ruler(x, rng):
    return c2m.art_ruler(x, rng, alpha=0.7, dark=1.0)


def hair_single_blend(n, alpha, dark=0.10):
    """This pipeline's strand geometry and random draws, with each touched pixel blended once."""
    def f(x, rng):
        x = x.clone(); _, H, W = x.shape
        for _ in range(n):
            x0, y0 = rng.randint(0, W), rng.randint(0, H)
            ang = rng.uniform(0, 2 * np.pi)
            length = rng.randint(W // 3, W)
            curv = rng.uniform(-0.0025, 0.0025)
            cs, sn = np.cos(ang), np.sin(ang)
            pc, ps = np.cos(ang + np.pi / 2), np.sin(ang + np.pi / 2)
            pts = set()
            for t in range(length):
                px = int(x0 + t * cs + curv * t * t * pc)
                py = int(y0 + t * sn + curv * t * t * ps)
                if 0 <= px < W and 0 <= py < H:
                    pts.add((py, px))
            if pts:
                yy, xx = map(np.array, zip(*pts))
                x[:, yy, xx] = alpha * dark + (1 - alpha) * x[:, yy, xx]
        return x
    return f


def blend_multiplicity(imgs, n, alpha):
    """Mean number of blends per touched pixel under point-by-point blending, seed as in edits."""
    tot, uniq = 0, 0
    for i, x01 in enumerate(imgs):
        rng = np.random.RandomState(SEED + i); _, H, W = x01.shape
        for _ in range(n):
            x0, y0 = rng.randint(0, W), rng.randint(0, H)
            ang = rng.uniform(0, 2 * np.pi); length = rng.randint(W // 3, W)
            curv = rng.uniform(-0.0025, 0.0025)
            cs, sn = np.cos(ang), np.sin(ang); pc, ps = np.cos(ang + np.pi / 2), np.sin(ang + np.pi / 2)
            pts = []
            for t in range(length):
                px = int(x0 + t * cs + curv * t * t * pc); py = int(y0 + t * sn + curv * t * t * ps)
                if 0 <= px < W and 0 <= py < H:
                    pts.append((py, px))
            tot += len(pts); uniq += len(set(pts))
    return tot / max(uniq, 1)


def edits_with(imgs, ruler_fn, hair_fns=None):
    """This pipeline's overlay generator with the same per-image seeding as precompute_edits."""
    out = {a: [] for a in HELD}
    fns = dict(c2m.ARTIFACTS); fns["ruler"] = ruler_fn
    if hair_fns:
        fns.update(hair_fns)
    for i, x01 in enumerate(imgs):
        rng = np.random.RandomState(SEED + i)
        for a in HELD:
            out[a].append(fns[a](x01, rng))
    return out


def summarize(p0, pe, msk, pid):
    per_op = {}
    for a in HELD:
        v = np.abs(pe[a] - p0)[msk]
        e, lo, hi = c30.boot_ci(v, pid[msk])
        per_op[a] = {"magnitude": float(e), "ci": [float(lo), float(hi)]}
    mix = np.mean([np.abs(pe[a] - p0) for a in HELD], 0)[msk]
    e, lo, hi = c30.boot_ci(mix, pid[msk])
    return {"per_operator": per_op,
            "mixture": {"magnitude": float(e), "ci": [float(lo), float(hi)],
                        "verdict": c30.verdict(lo, hi, MARGIN)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--other", required=True, help="root directory of the reimplementation")
    O = ap.parse_args().other
    t0 = time.time()
    z = np.load(os.path.join(O, "results", "c19_preds.npz"), allow_pickle=True)
    ids = [str(i) for i in z["isic_ids"]]
    lab = np.asarray(z["target"]).astype(int)
    pid = np.asarray(z["patient_ids"]).astype(str)
    p0_o = np.asarray(z[f"{TAG}|identity"], float)
    pe_o = {a: np.asarray(z[f"{TAG}|{a}"], float) for a in HELD}
    ben = lab == 0

    imgs = c51.read_images(ids)
    kind = {v[2]: v[1] for v in c30.TRIO["r50"]}["Inv"]
    m = c51.load_ckpt(c30.make_model(kind).to(c30.DEVICE), os.path.join(O, "checkpoints", TAG + ".pt"))
    p0_h = c30.probs(m, imgs)
    pe_h = {a: c30.probs(m, v) for a, v in edits_with(imgs, c2m.ARTIFACTS["ruler"]).items()}
    pe_hb = {a: c30.probs(m, v) for a, v in edits_with(imgs, bright_ruler).items()}

    sys.path.insert(0, os.path.join(O, "code"))
    from cxai import infer as oinf               # noqa: E402
    mo, _ = oinf.load_model(os.path.join(O, "checkpoints", TAG + ".pt"), device=c30.DEVICE)
    stack = lambda L: torch.stack(list(L))
    p0_x = oinf.malignant_prob(mo, stack(imgs), device=c30.DEVICE)
    ed_h = edits_with(imgs, c2m.ARTIFACTS["ruler"])
    pe_x = {a: oinf.malignant_prob(mo, stack(v), device=c30.DEVICE) for a, v in ed_h.items()}
    from cxai import imaging as oimg, paths as opaths   # noqa: E402
    x_o = torch.stack([oimg.load01(oimg.isic_path(i, opaths.ISIC_IMAGES)) for i in ids])
    p0_own = oinf.malignant_prob(mo, x_o, device=c30.DEVICE)
    del mo
    single = {"hair_light": hair_single_blend(6, 0.40), "hair": hair_single_blend(14, 0.60),
              "hair_heavy": hair_single_blend(24, 0.75)}
    pe_hs = {a: c30.probs(m, v) for a, v in edits_with(imgs, c2m.ARTIFACTS["ruler"], single).items()}
    d0 = np.abs(p0_h - p0_o)
    res = {"cell": "ResNet-50 Inv, benign", "checkpoint": TAG, "n_images": len(ids),
           "n_benign": int(ben.sum()), "seed": SEED, "margin_p": MARGIN,
           "L1_unedited_probability_difference": {
               "max_abs": float(d0.max()), "mean_abs": float(d0.mean()),
               "benign_mean_abs": float(d0[ben].mean())},
           "L2_L3_this_pipeline": summarize(p0_h, pe_h, ben, pid),
           "L2_L3_reimplementation_cached": summarize(p0_o, pe_o, ben, pid),
           "L4_this_pipeline_bright_ruler": summarize(p0_h, pe_hb, ben, pid),
           "L5_this_pipeline_single_blend_hair": summarize(p0_h, pe_hs, ben, pid),
           "L6_reimplementation_inference_on_this_pipeline_edits": summarize(p0_x, pe_x, ben, pid),
           "L6_unedited": {
               "reimpl_inference_vs_reimpl_cache_mean_abs": float(np.abs(p0_x - p0_o).mean()),
               "reimpl_inference_vs_this_pipeline_mean_abs": float(np.abs(p0_x - p0_h).mean()),
               "reimpl_inference_vs_this_pipeline_max_abs": float(np.abs(p0_x - p0_h).max())},
           "L7_reimplementation_rerun_vs_its_cache": {
               "mean_abs": float(np.abs(p0_own - p0_o).mean()),
               "max_abs": float(np.abs(p0_own - p0_o).max()),
               "share_over_0.05": float((np.abs(p0_own - p0_o) > 0.05).mean()),
               "rerun_vs_this_pipeline_mean_abs": float(np.abs(p0_own - p0_h).mean())},
           "blends_per_touched_pixel_heavy_hair": blend_multiplicity(imgs, 24, 0.75)}
    b0 = np.argmax(d0)
    res["L1_unedited_probability_difference"].update({
        "worst_image": ids[b0], "worst_label": int(lab[b0]),
        "share_images_over_0.05": float((d0 > 0.05).mean())})
    res["runtime_sec"] = round(time.time() - t0, 1)
    for k in ("L2_L3_this_pipeline", "L2_L3_reimplementation_cached", "L4_this_pipeline_bright_ruler",
              "L5_this_pipeline_single_blend_hair", "L6_reimplementation_inference_on_this_pipeline_edits"):
        r = res[k]
        print(k, {a: round(v["magnitude"], 4) for a, v in r["per_operator"].items()},
              "mixture", round(r["mixture"]["magnitude"], 4), r["mixture"]["ci"], r["mixture"]["verdict"])
    print("L1", res["L1_unedited_probability_difference"]); print("L6", res["L6_unedited"]); print("L7", res["L7_reimplementation_rerun_vs_its_cache"])
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
