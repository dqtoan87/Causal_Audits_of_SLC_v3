#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C19: recompute from the raw predictions two numbers that no result file stores.

(1) Fraction of (image, edit, model) predictions that hit the clipping threshold of the clipped logit
    ℓ_eps(p), eps = 1e-6  ->  |ℓ| = 13.8155. The paper cites "27.8%" without a checkable source.
    The script reports the fraction under several set definitions, so the number in the paper states
    which set it refers to.

(2) MDE (minimum detectable effect) for the exact contrast argued in Section V-E: the per-image
    difference between the concept edit and the placebo, i.e. the quantity underlying A_corr, not the
    MDE of a single edit against baseline (that is what c11 stores). Computed on the common set of
    62 images, with the paper's formula:
    MDE = 2.8 · SD / sqrt(n).

Reproduces the c13 pipeline exactly: same split, same 178 images, same edit set, same model_prob.
Out -> results/c19_saturation_mde.json
Run:  .venv/bin/python Causal_XAI_v3/code/c19_saturation_mde.py
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
import audit_cohort               # noqa: E402
import c13_fullpop as c13           # noqa: E402

p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c19_saturation_mde.json")
import audited_models                # noqa: E402  (checkpoints resolved in one place)
CKPTS = audited_models.CKPTS         # absolute paths to the fixed set
SEED, N_EVAL = 42, 200
ORDER = ["erm", "slim", "causal"]
NAME = {"erm": "ERM", "slim": "Slim", "causal": "Inv"}
EPS = 1e-6
BOUND = float(np.log((1 - EPS) / EPS))     # 13.8155


def saturated(p):
    """True if the probability lies outside (eps, 1-eps), i.e. the clipped logit is clamped."""
    return (np.asarray(p) <= EPS) | (np.asarray(p) >= 1 - EPS)


def main():
    t0 = time.time()
    print(f"##### C19 saturation + MDE {time.strftime('%H:%M:%S')} #####", flush=True)
    print(f"clipping bound |l_eps| = {BOUND:.4f}", flush=True)

    imgs01, labels, _pid, _ids = audit_cohort.load()   # fixed evaluation set
    labels = np.array(labels); n = len(imgs01)
    print(f"eval images {n}  malignant {int((labels==1).sum())}  benign {int((labels==0).sum())}",
          flush=True)

    edits, names = c13.build_all_edits(imgs01)
    print(f"edits/image = {len(names)}: {names}  ({time.time()-t0:.0f}s)", flush=True)

    P0, PE = {}, {}
    for m in ORDER:
        model = p1.build_model("erm" if m == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(CKPTS[m], map_location=DEVICE))
        model.eval()
        P0[m], PE[m] = c13.eval_model(model, imgs01, edits, names)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"[{m}] p0(mal)={P0[m][labels==1].mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)

    res = {"n_images": int(n), "n_malignant": int((labels == 1).sum()),
           "n_edits_per_image": int(len(names)), "edit_names": list(names),
           "eps": EPS, "clip_bound": BOUND,
           "note": ("Recomputed from raw predictions. (1) fraction of predictions saturating the "
                    "clipped-logit bound under several set definitions; (2) MDE = 2.8*SD/sqrt(n) on "
                    "the per-image concept-minus-placebo difference, the quantity underlying A_corr."),
           "saturation": {}, "mde_paired_contrast": {}}

    # ---------- (1) saturation fraction ----------
    print("=" * 96)
    print("A) SATURATION OF THE CLIPPED-LOGIT BOUND")
    print(f"{'set':46s}{'n preds':>12s}{'saturated':>12s}{'fraction':>11s}")
    sets = {}
    all_edit, all_base = [], []
    for m in ORDER:
        se = saturated(PE[m]); sb = saturated(P0[m])
        all_edit.append(se); all_base.append(sb)
        sets[f"{NAME[m]}: edited only (178x{len(names)})"] = se
        sets[f"{NAME[m]}: baseline only (178)"] = sb
        sets[f"{NAME[m]}: baseline + edited"] = np.concatenate([sb[:, None], se], axis=1)
    AE = np.stack(all_edit); AB = np.stack(all_base)
    sets["ALL models, edited only"] = AE
    sets["ALL models, baseline only"] = AB
    sets["ALL models, baseline + edited"] = np.concatenate(
        [AB.reshape(len(ORDER), n, 1), AE], axis=2)
    for k, v in sets.items():
        v = np.asarray(v); frac = float(v.mean())
        res["saturation"][k] = {"n": int(v.size), "n_saturated": int(v.sum()), "fraction": frac}
        print(f"{k:46s}{v.size:12d}{int(v.sum()):12d}{100*frac:10.1f}%")

    # by edit type, pooled over the 3 models
    print("-" * 96)
    print("   by edit (pooled over the three models):")
    by_edit = {}
    for j, nm in enumerate(names):
        v = np.stack([saturated(PE[m][:, j]) for m in ORDER])
        by_edit[nm] = {"n": int(v.size), "fraction": float(v.mean())}
        print(f"     {nm:24s}{100*v.mean():8.1f}%")
    res["saturation_by_edit"] = by_edit

    # ---------- (2) MDE on the concept − placebo contrast ----------
    print("=" * 96)
    print("B) MDE = 2.8*SD/sqrt(n) ON THE PER-IMAGE CONCEPT-MINUS-PLACEBO DIFFERENCE (common set)")
    mal = np.where(labels == 1)[0]
    common = mal[np.all([P0[m][mal] >= 0.1 for m in ORDER], axis=0)]
    print(f"   common set: {len(common)} images (malignant, p_full >= 0.1 for all three models)")
    res["n_common"] = int(len(common))
    idx = {nm: j for j, nm in enumerate(names)}
    concepts = [c for c in ["cpt:contrast@2", "cpt:color_var@2", "cpt:contrast@0", "cpt:color_var@0"]
                if c in idx]
    placebos = [c for c in ["pb:flip", "pb:jitter", "pb:hue", "pb:noise"] if c in idx]
    print(f"{'model':7s}{'contrast':22s}{'placebo':12s}{'mean':>9s}{'SD':>9s}{'MDE':>9s}")
    for m in ORDER:
        L0 = c3m.logit(P0[m][common])
        res["mde_paired_contrast"][m] = {}
        for cn in concepts:
            for pn in placebos:
                d = (c3m.logit(PE[m][common, idx[cn]]) - L0) - \
                    (c3m.logit(PE[m][common, idx[pn]]) - L0)
                sd = float(np.std(d, ddof=1)); mde = float(2.8 * sd / np.sqrt(len(d)))
                res["mde_paired_contrast"][m][f"{cn}|{pn}"] = {
                    "mean": float(d.mean()), "sd": sd, "mde": mde, "n": int(len(d))}
                print(f"{NAME[m]:7s}{cn:22s}{pn:12s}{d.mean():9.3f}{sd:9.3f}{mde:9.3f}")
    for m in ORDER:
        v = [x["mde"] for x in res["mde_paired_contrast"][m].values()]
        res["mde_paired_contrast"][m]["_range"] = [float(min(v)), float(max(v))]
        print(f"   {NAME[m]:5s} MDE range over concept x placebo pairs: "
              f"{min(v):.2f} to {max(v):.2f} logits")

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("=" * 96)
    print(f"saved -> {OUT}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
