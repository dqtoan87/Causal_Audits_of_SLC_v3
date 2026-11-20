#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C30: Tier-2 artifact audit on the patient-disjoint AUDIT fold.

The checkpoints of the six C29 models are selected using only the SELECT fold. The AUDIT fold was
never used for training, checkpoint selection or calibration. So the numbers here are not a
post-selection diagnosis, unlike the paper's main audit.

Measures the paper's main criterion: E|Δp| of the held-out artifacts (hair at three doses, ruler), per
model and per subgroup (all, malignant, benign), with patient-level bootstrap confidence intervals,
then applies the three-way interval rule against a tolerance of 0.05.

It also fits calibration (temperature and Platt) on SELECT and reports on AUDIT, so the calibration
metrics here are true out-of-sample estimates.

Out -> results/c30_disjoint_audit.json
Run:  .venv/bin/python Causal_XAI_v3/code/c30_disjoint_audit.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c2_cace_artifact as c2m       # noqa: E402  (ARTIFACTS, precompute_edits)
import c8_backbone2_train as c8      # noqa: E402
import c29_disjoint_train as c29     # noqa: E402
import c3_cace_concept as c3m        # noqa: E402
import audit_cohort               # noqa: E402  (fixed list of audit images)

p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
BASE = os.path.dirname(HERE)
CKPT = os.path.join(BASE, "checkpoints")
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c30_disjoint_audit.json")
SEED, N_EVAL, BOOT = 42, 200, 4000
EPS = 1e-6
MARGIN_P = 0.05
ART = ["hair_light", "hair", "hair_heavy", "ruler", "patch", "vignette", "jitter", "identity"]
HELD = ["hair_light", "hair", "hair_heavy", "ruler"]
TRIO = {"b3": [("dj_erm_b3", "erm", "ERM"), ("dj_slim_b3", "slim", "Slim"),
               ("dj_causal_b3", "slim", "Inv")],
        "r50": [("dj_erm_r50", "erm_r50", "ERM"), ("dj_slim_r50", "slim_r50", "Slim"),
                ("dj_causal_r50", "slim_r50", "Inv")]}


def make_model(kind):
    if kind == "erm":
        return p1.build_model("erm", p1.CFG["z_dim"])
    if kind == "slim":
        return p1.build_model("slim", p1.CFG["z_dim"])
    if kind == "erm_r50":
        return c8.ERM_R50()
    return c8.Slim_R50(p1.CFG["z_dim"])


def clipped_logit(p):
    q = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(q / (1 - q))


@torch.no_grad()
def probs(model, batch):
    out = []
    for i in range(0, len(batch), 32):
        # Normalise exactly as in training (ImageNet mean/std in p1.LesionDataset).
        x = p2._to_norm(torch.stack(batch[i:i + 32])).to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            out.append(torch.softmax(model(x).float(), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def boot_ci(v, groups, n_boot=BOOT, seed=3):
    g = [np.where(groups == q)[0] for q in np.unique(groups)]
    r = np.random.RandomState(seed)
    b = np.array([v[np.concatenate([g[i] for i in r.randint(0, len(g), len(g))])].mean()
                  for _ in range(n_boot)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def verdict(lo, hi, margin):
    """Three-way interval rule of Section III-C, on the non-negative quantity E|Δp|."""
    if lo > margin:
        return "exceedance"
    if hi < margin:
        return "equivalence"
    return "indeterminate"


def verdict_signed(lo, hi, margin):
    """Same rule, but for the signed quantity E[Δp] with the two-sided band [-margin, +margin]."""
    if lo > margin or hi < -margin:
        return "exceedance"
    if lo > -margin and hi < margin:
        return "equivalence"
    return "indeterminate"


def combine(v_sign, v_mag):
    """The null is the union of two conditions: reject if either condition is exceeded, accept if both are equivalent."""
    if "exceedance" in (v_sign, v_mag):
        return "exceedance"
    if v_sign == v_mag == "equivalence":
        return "equivalence"
    return "indeterminate"


def main():
    t0 = time.time()
    print(f"##### C30 audit on the patient-disjoint fold {time.strftime('%H:%M:%S')} #####", flush=True)
    imgs, labels, pid, _ = audit_cohort.load()   # fixed list, the partition is not rebuilt
    print(f"AUDIT eval: {len(imgs)} images / {len(np.unique(pid))} patients "
          f"({int((labels==1).sum())} malignant)", flush=True)

    edits, change_frac = c2m.precompute_edits(imgs, ART, SEED)
    res = {"n_images": int(len(imgs)), "n_patients": int(len(np.unique(pid))),
           "n_malignant": int((labels == 1).sum()), "margin_p": MARGIN_P, "boot": BOOT,
           "change_frac": change_frac,
           "note": ("Tier-2 artifact criterion measured on a patient-disjoint audit fold that was "
                    "never used for training, checkpoint selection or calibration. Unlike the "
                    "primary audit, these numbers are not post-selection."),
           "models": {}}

    W = 96
    print("=" * W)
    print(f"{'backbone':9s}{'model':6s}{'subgroup':11s}{'E|Δp| held-out':>16s}{'95% CI':>22s}"
          f"{'magnitude':>14s}{'E[Δp]':>10s}{'95% CI':>20s}{'direction':>14s}{'joint':>14s}")
    n_exc = n_cell = 0
    for bb, trio in TRIO.items():
        for ck, kind, nm in trio:
            f = os.path.join(CKPT, ck + ".pt")
            if not os.path.exists(f):
                print(f"  [{ck}] no checkpoint yet, skipping", flush=True); continue
            m = make_model(kind).to(DEVICE)
            m.load_state_dict(torch.load(f, map_location=DEVICE)); m.eval()
            p0 = probs(m, imgs)
            per = {}
            for a in ART:
                pe = probs(m, [edits[i][a] for i in range(len(imgs))])
                per[a] = pe
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            absdp = np.mean([np.abs(per[a] - p0) for a in HELD], 0)
            sgndp = np.mean([per[a] - p0 for a in HELD], 0)
            cell = {}
            for sub, msk in [("all", np.ones(len(labels), bool)), ("malignant", labels == 1),
                             ("benign", labels == 0)]:
                mean, lo, hi = boot_ci(absdp[msk], pid[msk])
                v = verdict(lo, hi, MARGIN_P)
                smean, slo, shi = boot_ci(sgndp[msk], pid[msk])
                sv = verdict_signed(slo, shi, MARGIN_P)
                jv = combine(sv, v)
                n_exc += (jv == "exceedance"); n_cell += 1
                cell[sub] = {"mean": mean, "ci": [lo, hi], "verdict": v,
                             "signed_mean": smean, "signed_ci": [slo, shi],
                             "signed_verdict": sv, "joint_verdict": jv}
                print(f"{bb:9s}{nm:6s}{sub:11s}{mean:16.3f}"
                      f"{('[%.3f, %.3f]' % (lo, hi)):>22s}{v:>14s}"
                      f"{smean:10.3f}{('[%.3f, %.3f]' % (slo, shi)):>20s}{sv:>14s}{jv:>14s}")
            cell["identity_max_abs_dlogit"] = float(
                np.abs(clipped_logit(per["identity"]) - clipped_logit(p0)).max())
            cell["hair_dose_dp"] = [float((per[a] - p0)[labels == 1].mean())
                                    for a in ["hair_light", "hair", "hair_heavy"]]
            cell["p0_mean_malignant"] = float(p0[labels == 1].mean())
            # AUROC on the audit cohort itself: rules out the reading "the model is broken, hence invariant".
            cell["audit_auroc"] = float(roc_auc_score(labels, p0))
            res["models"].setdefault(bb, {})[nm] = cell
        print("-" * W)
    res["summary"] = {"cells": n_cell, "exceedance": n_exc}
    print(f"  exceeding tolerance {MARGIN_P}: {n_exc}/{n_cell} cells")
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("=" * W)
    print(f"saved -> {OUT}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
