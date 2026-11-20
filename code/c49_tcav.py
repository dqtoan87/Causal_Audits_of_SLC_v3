#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C49: TCAV on the same six checkpoints, the same two concepts, and the same audit image set.

Why. The paper's third contribution states that the placebo-subtracted estimator does not identify the
concept component within the constructed intervention class. A natural question follows: what do the
widely used concept-attribution methods return on exactly this data? If they return a confident answer
where the paper's framework reports non-identification, that is precisely what this framework adds,
and it must be measured, not just argued.

TCAV (Kim et al. 2018) measures the directional derivative of the logit along a concept direction in
activation space. It does not intervene on the concept; it reads the joint sensitivity along a linear direction.

Protocol, following the original:
  1. CAV: a linear logistic regression separates activations of concept-edited images from original images,
     at the pooled feature layer. The normal vector v_C is the concept direction. As in the original, it is
     trained N_CAV times on different random half-samples to obtain a distribution of TCAV scores rather than
     a single score.
  2. Directional derivative: S_i = ∇_a logit_malignant(a_i) · v_C at the activation of original image i.
  3. TCAV score = fraction of images with S_i > 0.
  4. Significance test: N_RAND random CAVs from permuted labels, trained the same way. A concept is
     significant when a two-sample t-test between the N_CAV concept scores and the N_RAND random scores rejects
     at level 0.05, the same test the original uses.

     A note on this null, to be reported rather than hidden: the gradient at the feature layer points in nearly
     the same direction for every image, so a random direction gives a dot product of the same sign on almost
     all images, and random TCAV scores pile up at 0 or 1 instead of clustering around 0.5. The null mean is
     still around 0.5 but its variance is very large, so this test has low power here.

Point to keep in mind when reading the results: the CAV is built from the paper's own concept edits, so the
concept direction is taken from the intervention the paper performs. It therefore carries the edit's footprint
as well as its concept change. Accuracy is reported on all activations and on the half not used to fit each
direction.

Out -> results/c49_tcav.json
Run:  python Causal_XAI_v3/code/c49_tcav.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c3_cace_concept as c3m        # noqa: E402  (make_edit, soft_mask, DOSES)
import c30_disjoint_audit as c30     # noqa: E402  (make_model, probs)
import audit_cohort                  # noqa: E402

p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
BASE = os.path.dirname(HERE)
CKPT = os.path.join(BASE, "checkpoints")
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c49_tcav.json")

SEED, N_CAV, N_RAND = 42, 20, 20
CONCEPTS = ["contrast", "color_var"]
DOSE = c3m.DOSES[-1]          # peak dose, the same dose the paper's concept tier uses
MODELS = [("erm", "dj_erm_b3.pt", "EfficientNet-B3 ERM"),
          ("slim", "dj_slim_b3.pt", "EfficientNet-B3 Slim"),
          ("slim", "dj_causal_b3.pt", "EfficientNet-B3 Inv"),
          ("erm_r50", "dj_erm_r50.pt", "ResNet-50 ERM"),
          ("slim_r50", "dj_slim_r50.pt", "ResNet-50 Slim"),
          ("slim_r50", "dj_causal_r50.pt", "ResNet-50 Inv")]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def backbone_of(model, kind):
    """Pooled feature layer: where standard TCAV places the CAV."""
    if kind in ("erm", "slim"):
        return model.backbone
    return model.backbone


def head_of(model, kind):
    """The part from the pooled features to the logit."""
    if kind == "erm":
        return model.head
    if kind == "slim":
        return lambda a: model.disease_head(model.proj_c(a))
    if kind == "erm_r50":
        return model.head
    return lambda a: model.disease_head(model.proj_c(a))


@torch.no_grad()
def activations(model, kind, batch):
    bb = backbone_of(model, kind)
    out = []
    for i in range(0, len(batch), 32):
        x = p2._to_norm(torch.stack(batch[i:i + 32])).to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            out.append(bb(x).float().cpu().numpy())
    return np.concatenate(out)


def grads(model, kind, batch):
    """∇_a logit_malignant at the activation of each original image."""
    bb, hd = backbone_of(model, kind), head_of(model, kind)
    out = []
    for i in range(0, len(batch), 16):
        x = p2._to_norm(torch.stack(batch[i:i + 16])).to(DEVICE)
        with torch.no_grad():
            a = bb(x).float()
        a = a.detach().requires_grad_(True)
        z = hd(a).float()
        s = (z[:, 1] - z[:, 0]).sum()          # malignant-class logit
        g, = torch.autograd.grad(s, a)
        out.append(g.cpu().numpy())
    return np.concatenate(out)


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)
    imgs01, labels, pid, ids = audit_cohort.load()
    log(f"audit set: {len(imgs01)} images, {len(set(pid))} patients")

    # concept-edited images at the peak dose, the same edits as Section III-D
    edited = {}
    for c in CONCEPTS:
        eb = []
        for x in imgs01:
            w, skin, lm = c3m.soft_mask(x)
            eb.append(c3m.make_edit(x, c, DOSE, w, skin, lm))
        edited[c] = eb
        log(f"  built {len(eb)} edited images for concept {c}")

    res = {"n_images": len(imgs01), "n_patients": int(len(set(pid))), "dose": float(DOSE),
           "n_random_cavs": N_RAND, "seed": SEED,
           "note": ("TCAV on the same checkpoints, concepts and audit images as the concept tier. "
                    "The CAV is trained to separate activations of concept-edited images from "
                    "activations of the originals, so the concept direction comes from the very "
                    "intervention the paper performs. Significance is against N_RAND random CAVs from permuted "
                    "labels, the standard TCAV control."),
           "models": {}}

    for kind, fn, label in MODELS:
        path = os.path.join(CKPT, fn)
        if not os.path.exists(path):
            log(f"SKIP {label}: missing {fn}"); continue
        m = c30.make_model(kind).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval()
        A0 = activations(m, kind, imgs01)
        G0 = grads(m, kind, imgs01)
        entry = {"label": label, "feature_dim": int(A0.shape[1]), "concepts": {}}
        for c in CONCEPTS:
            A1 = activations(m, kind, edited[c])
            X = np.concatenate([A1, A0]); y = np.concatenate([np.ones(len(A1)), np.zeros(len(A0))])
            n = len(A0)
            def one(labels_, half):
                cl = LogisticRegression(max_iter=2000, C=1.0).fit(X[half], labels_[half])
                vv = cl.coef_[0]; vv = vv / (np.linalg.norm(vv) + 1e-12)
                rest = np.setdiff1d(np.arange(len(X)), half)     # activations not used to fit this direction
                return (float(((G0 @ vv) > 0).mean()), float(cl.score(X, labels_)),
                        float(cl.score(X[rest], labels_[rest])))
            scores, accs, accs_ho = [], [], []
            for r in range(N_CAV):
                h = rng.permutation(len(X))[: len(X) // 2]
                sc, ac, ho = one(y, h); scores.append(sc); accs.append(ac); accs_ho.append(ho)
            null = []
            for r in range(N_RAND):
                h = rng.permutation(len(X))[: len(X) // 2]
                null.append(one(rng.permutation(y), h)[0])
            scores, null, accs = np.array(scores), np.array(null), np.array(accs)
            t, p = stats.ttest_ind(scores, null, equal_var=False)
            entry["concepts"][c] = {
                "cav_accuracy_mean": float(accs.mean()),
                "cav_accuracy_heldout_mean": float(np.mean(accs_ho)),
                "tcav_mean": float(scores.mean()),
                "tcav_sd": float(scores.std(ddof=1)),
                "tcav_range": [float(scores.min()), float(scores.max())],
                "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
                "null_range": [float(null.min()), float(null.max())],
                "p_vs_random": float(p),
                "significant": bool(p < 0.05),
                "n_cav": N_CAV, "n_random": N_RAND}
            e = entry["concepts"][c]
            log(f"  {label:22s} {c:10s} CAV acc={accs.mean():.3f}  "
                f"TCAV={scores.mean():.3f}±{scores.std(ddof=1):.3f} "
                f"[{scores.min():.2f},{scores.max():.2f}]  "
                f"null={null.mean():.3f}±{null.std(ddof=1):.3f}  p={p:.2e}  "
                f"significant={e['significant']}")
        res["models"][fn.replace(".pt", "")] = entry
        del m
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    allc = [e for v in res["models"].values() for e in v["concepts"].values()]
    res["summary"] = {"n_cells": len(allc),
                      "n_significant": sum(e["significant"] for e in allc),
                      "tcav_mean_range": [min(e["tcav_mean"] for e in allc),
                                          max(e["tcav_mean"] for e in allc)],
                      "cav_accuracy_heldout_range": [min(e["cav_accuracy_heldout_mean"] for e in allc),
                                                     max(e["cav_accuracy_heldout_mean"] for e in allc)],
                      "cav_accuracy_range": [min(e["cav_accuracy_mean"] for e in allc),
                                             max(e["cav_accuracy_mean"] for e in allc)],
                      "null_sd_range": [min(e["null_sd"] for e in allc),
                                        max(e["null_sd"] for e in allc)]}
    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"saved -> {OUT}  ({res['summary']['n_significant']}/{res['summary']['n_cells']} cells significant)")


if __name__ == "__main__":
    main()
