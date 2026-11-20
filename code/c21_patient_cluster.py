#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C21: rerun all Tier-2 inference with a patient-clustered bootstrap.

Why: the 178 evaluation images come from only 90 patients (35 have more than one image, one has
12 images). A bootstrap that resamples images treats dependent observations as independent, so the
confidence intervals are narrower than they should be. C19 checked this for the artifact E|Δp|; this
script applies the same approach to every remaining inferential quantity of Tier 2:

  - A_raw and A_corr (flip, jitter) on the primary set of 178 images and on the malignant group;
  - the antisymmetric contrast ΔCaCE_sym at low and high dose;
  - the shrinkage coefficient α for up/down edits, and the symmetry test H0: α+ = α−;
  - the upper bound of the residual/effect ratio, i.e. the 10/12, 0/12 and 2/12 decision criterion.

Each quantity is computed twice, with image-level and patient-level bootstrap, to see which conclusions change.

Out -> results/c21_patient_cluster.json
Run:  .venv/bin/python Causal_XAI_v3/code/c21_patient_cluster.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c3_cace_concept as c3m       # noqa: E402
import c10_multiplacebo as c10      # noqa: E402
import c12_matched_control as c12   # noqa: E402
import c15_symmetry as c15          # noqa: E402

p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c21_patient_cluster.json")
import audited_models                # noqa: E402  (checkpoints resolved in one place)
import audit_cohort                  # noqa: E402  (frozen audit cohort)
CKPTS = audited_models.CKPTS         # absolute paths to the fixed set
SEED, N_EVAL, BOOT = 42, 200, 2000
ORDER = ["erm", "slim", "causal"]
NAME = {"erm": "ERM", "slim": "Slim", "causal": "Inv"}
TARGETS = ["contrast@0", "contrast@2", "color_var@0", "color_var@2"]
LOWDOSE = ["contrast@0", "color_var@0"]


def boot_idx(n, groups, rng, by_patient):
    """Generate BOOT index sets: by image, or by patient (resampling whole clusters)."""
    out = []
    for _ in range(BOOT):
        if by_patient:
            pick = rng.randint(0, len(groups), len(groups))
            out.append(np.concatenate([groups[k] for k in pick]))
        else:
            out.append(rng.randint(0, n, n))
    return out


def ci(v, idx):
    b = np.array([v[i].mean() for i in idx])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def ols_alpha(dl, L0):
    """Shrinkage coefficient α in Δℓ = −α·L0 + β (α = −slope)."""
    s, _ = np.polyfit(L0, dl, 1)
    return -s


def main():
    t0 = time.time()
    print(f"##### C21 patient-clustered Tier-2 inference {time.strftime('%H:%M:%S')} #####", flush=True)

    imgs01, labels, pid, _ids = audit_cohort.load()   # frozen audit cohort, never rebuilt
    labels = np.array(labels); pid = np.array(pid)
    assert len(pid) == len(imgs01)
    print(f"{len(imgs01)} images from {len(np.unique(pid))} patients "
          f"(max {max(np.bincount(np.unique(pid, return_inverse=True)[1]))} per patient)", flush=True)

    edits, names = c15.build_edits(imgs01)
    print(f"edits/image = {len(names)}  ({time.time()-t0:.0f}s)", flush=True)

    P0, PE = {}, {}
    for m in ORDER:
        model = p1.build_model("erm" if m == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(CKPTS[m], map_location=DEVICE))
        model.eval()
        P0[m], PE[m] = c15.eval_model(model, imgs01, edits, names)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"[{NAME[m]}] done ({time.time()-t0:.0f}s)", flush=True)

    idx = {nm: j for j, nm in enumerate(names)}
    res = {"n_images": int(len(imgs01)), "n_patients": int(len(np.unique(pid))), "boot": BOOT,
           "note": ("Every Tier-2 inferential quantity recomputed under image-level and "
                    "patient-clustered bootstrap. Point estimates are identical by construction; "
                    "only the intervals and the significance labels can differ."),
           "quantities": {}, "symmetry": {}}

    rng = np.random.RandomState(0)
    for scope, msk in [("all", np.ones(len(labels), bool)), ("malignant", labels == 1)]:
        gp = [np.where(pid[msk] == q)[0] for q in np.unique(pid[msk])]
        n = int(msk.sum())
        bi = boot_idx(n, gp, np.random.RandomState(1), False)
        bp = boot_idx(n, gp, np.random.RandomState(1), True)
        for m in ORDER:
            L0 = c3m.logit(P0[m][msk])
            dl = {nm: c3m.logit(PE[m][msk, idx[nm]]) - L0 for nm in names}
            # A_raw: mean of the two concept edits at high dose; both have positive ATE, so sign = +1
            araw = (dl["contrast@2|up"] + dl["color_var@2|up"]) / 2
            for lab, v in [("A_raw", araw),
                           ("A_corr_flip", araw - dl["pb:flip"]),
                           ("A_corr_jitter", araw - dl["pb:jitter"])]:
                mi, li, hi = ci(v, bi); mp, lp, hp = ci(v, bp)
                res["quantities"].setdefault(scope, {}).setdefault(m, {})[lab] = {
                    "mean": mi, "ci_image": [li, hi], "ci_patient": [lp, hp],
                    "sig_image": bool(li > 0 or hi < 0), "sig_patient": bool(lp > 0 or hp < 0)}
            # antisymmetric contrast: up minus down
            for t in TARGETS:
                v = dl[f"{t}|up"] - dl[f"{t}|dn"]
                mi, li, hi = ci(v, bi); mp, lp, hp = ci(v, bp)
                res["quantities"][scope][m]["sym_" + t] = {
                    "mean": mi, "ci_image": [li, hi], "ci_patient": [lp, hp],
                    "sig_image": bool(li > 0 or hi < 0), "sig_patient": bool(lp > 0 or hp < 0)}

    # --- symmetry test and residual ratio, on all 178 images (as in Table XI) ---
    gp_all = [np.where(pid == q)[0] for q in np.unique(pid)]
    bi = boot_idx(len(pid), gp_all, np.random.RandomState(2), False)
    bp = boot_idx(len(pid), gp_all, np.random.RandomState(2), True)
    for m in ORDER:
        L0 = c3m.logit(P0[m])
        for t in TARGETS:
            du = c3m.logit(PE[m][:, idx[f"{t}|up"]]) - L0
            dd = c3m.logit(PE[m][:, idx[f"{t}|dn"]]) - L0
            au, ad = ols_alpha(du, L0), ols_alpha(dd, L0)
            dsym = float((du - dd).mean())
            def diff(ii):                                     # α− − α+ on one index set
                return ols_alpha(dd[ii], L0[ii]) - ols_alpha(du[ii], L0[ii])
            bI = np.array([diff(i) for i in bi]); bP = np.array([diff(i) for i in bp])
            out = {"alpha_up": float(au), "alpha_dn": float(ad), "dcace_sym": dsym,
                   "diff_point": float(ad - au)}
            for lab, b in [("image", bI), ("patient", bP)]:
                lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
                worst = max(abs(lo), abs(hi))
                out["ci_" + lab] = [lo, hi]
                out["reject_" + lab] = bool(lo > 0 or hi < 0)
                out["resid_upper_" + lab] = float(worst * np.abs(L0).mean() / abs(dsym))
            res["symmetry"].setdefault(m, {})[t] = out

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)

    W = 108
    print("=" * W)
    print("A) Concept scores: image-level vs patient-clustered (frozen 183-image audit cohort)")
    print(f"{'model':6s}{'quantity':16s}{'mean':>8s}{'CI image':>22s}{'CI patient':>22s}{'sig':>16s}")
    for m in ORDER:
        for k, v in res["quantities"]["all"][m].items():
            if not k.startswith("A_"):
                continue
            print(f"{NAME[m]:6s}{k:16s}{v['mean']:+8.3f}"
                  f"{('[%+.2f, %+.2f]' % tuple(v['ci_image'])):>22s}"
                  f"{('[%+.2f, %+.2f]' % tuple(v['ci_patient'])):>22s}"
                  f"{('img %s / pat %s' % ('Y' if v['sig_image'] else 'n', 'Y' if v['sig_patient'] else 'n')):>16s}")
    print("-" * W)
    print("B) SYMMETRY TEST AND RESIDUAL BOUND (all 183 images)")
    print(f"{'model':6s}{'target':14s}{'a+ - a-':>10s}{'reject img':>12s}{'reject pat':>12s}"
          f"{'resid up img':>14s}{'resid up pat':>14s}")
    n_rej_i = n_rej_p = n_pass02_i = n_pass02_p = n_pass06_i = n_pass06_p = 0
    for m in ORDER:
        for t in TARGETS:
            v = res["symmetry"][m][t]
            n_rej_i += v["reject_image"]; n_rej_p += v["reject_patient"]
            n_pass02_i += v["resid_upper_image"] <= 0.2; n_pass02_p += v["resid_upper_patient"] <= 0.2
            n_pass06_i += v["resid_upper_image"] <= 0.6; n_pass06_p += v["resid_upper_patient"] <= 0.6
            print(f"{NAME[m]:6s}{t:14s}{v['diff_point']:+10.3f}"
                  f"{('yes' if v['reject_image'] else 'no'):>12s}{('yes' if v['reject_patient'] else 'no'):>12s}"
                  f"{v['resid_upper_image']:14.2f}{v['resid_upper_patient']:14.2f}")
    print("-" * W)
    print(f"  symmetry rejected : image {n_rej_i}/12   patient {n_rej_p}/12")
    print(f"  residual <= 0.2   : image {n_pass02_i}/12   patient {n_pass02_p}/12")
    print(f"  residual <= 0.6   : image {n_pass06_i}/12   patient {n_pass06_p}/12")
    print("=" * W)
    print(f"saved -> {OUT}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
