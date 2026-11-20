#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C89: the artifact tier on the whole audit fold. POST HOC sensitivity, declared as such.

Why. The headline band (c56, Table 1 / Table S43: frontier 0.0422, 17 of 18 cells exceed at 0.05) is
computed on the 183-image cohort: every malignant image of the audit fold plus one draw of 100 benign
images. Its target is the cohort quantity Q_S, conditional on that benign draw and on one overlay
realization. c60 showed that redrawing the benign images moves the frontier (median 0.047, central
95% range 0.032 to 0.066). This script removes the benign draw: it uses every image of the audit fold.

This analysis is post hoc: the headline results were seen before it was written. It never replaces
the headline. All choices below were fixed before the run (also in results/c89_plan.json).

Fixed choices.
  Images      every image of the audit fold, in fold order (frozen_split.load()[2]): 83 malignant
              images and the benign frame of 4,150 images. 199 patients (198 with benign images,
              44 with malignant images).
  Scoring     c60_design_coverage.score_frame, called unchanged. Six checkpoints dj_{erm,slim,causal}_
              {b3,r50}.pt, the four held-out overlays (hair_light, hair, hair_heavy, ruler), ImageNet
              normalisation inside c30.probs, bf16 autocast, batches of 32 within chunks of 128.
              Per image: magnitude = mean over the four overlays of |p(edit) - p(orig)|, signed = mean
              of p(edit) - p(orig).
  Overlays    PRIMARY: c60's rule, RandomState(42 + fold position), the four overlays drawn in order.
              This differs from Table 1, whose rule is RandomState(42 + cohort index). The plan
              (ext01_response_plan.md, AN1) names c60's position-seeded overlays.
              SECONDARY (S-hyb): the 183 cohort images carry their Table 1 realization (vectors of
              results/c85_artifact_vectors.npz, which reproduce c56); the other 4,050 images keep
              c60's rule. This keeps Table 1's overlays wherever Table 1 had an image.
  Cells       6 checkpoints x 3 subgroups x {magnitude, signed} = 36 components.
              mal = mean over the 83 malignant images; ben = mean over the 4,150 benign images;
              all = (83 * mal + 100 * ben) / 183, fixed design weights in every resample, as c60's
              design targets (the cohort design ratio 83 to 100).
              SECONDARY (S-pool): all = image-weighted mean over the 4,233 fold images.
  Band        as c56: studentized max-statistic patient bootstrap over the 36 components, 20,000
              resamples of the 199 patients with all their images, RandomState(41), rng.randint per
              resample; se = s.d. of the draws (ddof 1); crit = 95th percentile of max |Z|;
              simultaneous interval est +- crit * se. Marginal interval = 2.5 and 97.5 percentiles of
              the draws (c40 convention). Also the 18-magnitude-only critical value, as c56 reports.
  Readouts    frontier = smallest simultaneous lower end over the 18 magnitude components; cells
              exceeding 0.05 jointly (simultaneous lower end > 0.05) and marginally (marginal lower
              end > 0.05); signed decisions with c56.decide.
  Rounding    as the paper: lower ends and frontiers rounded down, upper ends rounded up, to 3 dp.
  Checks      dry run: score_frame on the first 32 (and 20) cohort images in cohort order. Its seed
              42 + j then equals Table 1's seed for cohort index j, so the vectors must reproduce
              results/c85_artifact_vectors.npz. The 32-image subset is one batch, as in c56.
              Full run: the fold means of mal and ben must reproduce c60's design targets.

Run:   python code/c89_fullfold_artifact.py --dry-run
       python code/c89_fullfold_artifact.py            (stage 1 scoring is cached; rerun resumes)
Out -> results/c89_fullfold_vectors.npz, results/c89_fullfold_band.json, results/c89_dryrun.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
VEC = os.path.join(RES, "c89_fullfold_vectors.npz")
OUT = os.path.join(RES, "c89_fullfold_band.json")
DRY = os.path.join(RES, "c89_dryrun.json")
MARGIN, BOOT, SEED_BOOT, LEVEL = 0.05, 20000, 41, 95
N_BEN_DESIGN = 100


def fl(x):
    return math.floor(x * 1000) / 1000


def ce(x):
    return math.ceil(x * 1000) / 1000


def guard_norm():
    """Model-side scoring must normalise. Fail loudly if the normalisation constants are not ImageNet's."""
    import torch
    import c30_disjoint_audit as c30
    m, s = c30.p2.MEAN.flatten().cpu(), c30.p2.STD.flatten().cpu()
    assert torch.allclose(m, torch.tensor([0.485, 0.456, 0.406]), atol=1e-6), m
    assert torch.allclose(s, torch.tensor([0.229, 0.224, 0.225]), atol=1e-6), s
    assert c30.DEVICE == "cuda", "GPU expected (bf16 autocast protocol)"


def dry_run():
    guard_norm()
    import c60_design_coverage as c60
    m = json.load(open(os.path.join(RES, "audit_cohort.json")))
    c85 = np.load(os.path.join(RES, "c85_artifact_vectors.npz"))
    out = {"what": "score_frame on leading cohort images in cohort order vs results/c85_artifact_vectors.npz"}
    for n in (32, 20):
        t0 = time.time()
        mag, sgn = c60.score_frame(m["isic_id"][:n])
        rows = {}
        for i, nm in enumerate(c85["names"]):
            assert nm in mag, nm
            rows[str(nm)] = {"max_abs_diff_mag": float(np.abs(mag[nm] - c85["mag"][i, :n]).max()),
                             "max_abs_diff_sgn": float(np.abs(sgn[nm] - c85["sgn"][i, :n]).max())}
        worst = max(max(r.values()) for r in rows.values())
        out["n%d" % n] = {"per_checkpoint": rows, "max_abs_diff": worst, "sec": round(time.time() - t0, 1)}
        print("dry run n=%d: max abs diff vs c85 = %.3g" % (n, worst), flush=True)
    out["criterion"] = "n32 max abs diff <= 1e-6 (c85 is stored as float32)"
    out["pass"] = bool(out["n32"]["max_abs_diff"] <= 1e-6)
    json.dump(out, open(DRY, "w"), indent=1)
    print("dry run pass:", out["pass"], "->", DRY)


def score_stage(ids):
    """Stage 1: score the fold, or reuse the cache if it holds the same id list."""
    h = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    if os.path.exists(VEC):
        z = np.load(VEC)
        if str(z["ids_sha256"]) == h:
            print("stage 1: reusing", VEC, flush=True)
            return list(z["names"]), z["mag"].astype(float), z["sgn"].astype(float), None
    guard_norm()
    import c60_design_coverage as c60
    t0 = time.time()
    mag, sgn = c60.score_frame(ids)
    sec = time.time() - t0
    names = list(mag)
    np.savez(VEC, names=np.array(names), ids=np.array(ids), ids_sha256=np.array(h),
             mag=np.stack([mag[k] for k in names]), sgn=np.stack([sgn[k] for k in names]),
             overlay_rule=np.array("RandomState(42 + fold position), c60.score_frame"))
    print("stage 1: scored %d images in %.0fs -> %s" % (len(ids), sec, VEC), flush=True)
    return names, np.stack([mag[k] for k in names]), np.stack([sgn[k] for k in names]), sec


def band(names, MAG, SGN, lab, pid, all_rule):
    """36-component studentized max-statistic patient bootstrap, as c56, on patient-level sums."""
    import c56_simultaneous36 as c56
    mal, ben = lab == 1, lab == 0
    n_mal, n_all = int(mal.sum()), len(lab)
    uq, gi = np.unique(pid, return_inverse=True)
    P = len(uq)
    # per-patient sums of every per-image vector within each subgroup, and subgroup counts
    V = np.concatenate([MAG, SGN])                      # (12, N): 6 mag then 6 sgn
    Sm = np.stack([np.bincount(gi, weights=v * mal, minlength=P) for v in V], 1)   # (P, 12)
    Sb = np.stack([np.bincount(gi, weights=v * ben, minlength=P) for v in V], 1)
    Cm = np.bincount(gi, weights=mal.astype(float), minlength=P)
    Cb = np.bincount(gi, weights=ben.astype(float), minlength=P)

    def stats(w):
        """w: (..., P) resample counts per patient -> (..., 36) in key order."""
        mm = (w @ Sm) / np.maximum(w @ Cm, 1)[..., None]
        mb = (w @ Sb) / np.maximum(w @ Cb, 1)[..., None]
        if all_rule == "design":
            ma = (n_mal * mm + N_BEN_DESIGN * mb) / (n_mal + N_BEN_DESIGN)
        else:                                            # image-weighted pooled mean
            ma = ((w @ Sm) + (w @ Sb)) / np.maximum(w @ (Cm + Cb), 1)[..., None]
        cols = []
        for j in range(len(names)):
            for g in (ma, mm, mb):
                cols += [g[..., j], g[..., len(names) + j]]   # mag, sgn
        return np.stack(cols, -1)

    keys, ismag = [], []
    for nm in names:
        for s in ("all", "mal", "ben"):
            for comp in ("mag", "sgn"):
                keys.append("%s/%s/%s" % (nm, s, comp)); ismag.append(comp == "mag")
    ismag = np.array(ismag)
    est = stats(np.ones(P))
    rng = np.random.RandomState(SEED_BOOT)
    W = np.empty((BOOT, P))
    for b in range(BOOT):
        W[b] = np.bincount(rng.randint(0, P, P), minlength=P)
    draws = stats(W).T                                   # (36, BOOT)
    se = draws.std(axis=1, ddof=1)
    Z = np.abs(draws - est[:, None]) / np.maximum(se[:, None], 1e-12)
    crit36 = float(np.percentile(Z.max(0), LEVEL))
    crit18 = float(np.percentile(Z[ismag].max(0), LEVEL))
    lo, hi = est - crit36 * se, est + crit36 * se
    mlo, mhi = np.percentile(draws, 2.5, axis=1), np.percentile(draws, 97.5, axis=1)
    lo18 = est - crit18 * se
    fr = float(lo[ismag].min()); fr_key = keys[int(np.flatnonzero(ismag)[np.argmin(lo[ismag])])]
    comps, cells = {}, {}
    for j, k in enumerate(keys):
        comps[k] = {"est": float(est[j]), "se": float(se[j]), "sim36": [float(lo[j]), float(hi[j])],
                    "marginal": [float(mlo[j]), float(mhi[j])],
                    "rounded": {"est": round(float(est[j]), 3), "sim36": [fl(lo[j]), ce(hi[j])],
                                "marginal": [fl(mlo[j]), ce(mhi[j])]}}
        if ismag[j]:
            cells[k[:-4]] = {"est": round(float(est[j]), 3), "sim36": [fl(lo[j]), ce(hi[j])],
                             "marginal": [fl(mlo[j]), ce(mhi[j])],
                             "exceeds_0.05_joint": bool(lo[j] > MARGIN),
                             "exceeds_0.05_marginal": bool(mlo[j] > MARGIN)}
        else:
            comps[k]["decision36"] = c56.decide(lo[j], hi[j], MARGIN, False)
    sd = [comps[k]["decision36"] for k in keys if not k.endswith("mag")]
    return {"n_patients": int(P), "n_components": len(keys), "all_rule": all_rule,
            "crit_36": crit36, "crit_18_magnitude_only": crit18,
            "frontier_36": fr, "frontier_36_rounded_down": fl(fr), "frontier_cell": fr_key,
            "frontier_18": float(lo18[ismag].min()), "frontier_18_rounded_down": fl(float(lo18[ismag].min())),
            "n_exceed_0.05_joint": int((lo[ismag] > MARGIN).sum()),
            "n_exceed_0.05_marginal": int((mlo[ismag] > MARGIN).sum()),
            "n_exceed_0.05_joint_18band": int((lo18[ismag] > MARGIN).sum()),
            "marginal_frontier": float(mlo[ismag].min()), "marginal_frontier_rounded_down": fl(float(mlo[ismag].min())),
            "signed_counts_36": {v: sd.count(v) for v in ("exceeds", "within", "indet.")},
            "cells_magnitude": cells, "components": comps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        return dry_run()
    t0 = time.time()
    import frozen_split
    _, _, df = frozen_split.load()
    df = df.reset_index(drop=True)
    ids = list(df["isic_id"]); lab = df["target"].to_numpy().astype(int)
    pid = df["patient_id"].astype(str).to_numpy()
    print("audit fold: %d images, %d malignant, %d patients" % (len(ids), lab.sum(), len(set(pid))), flush=True)
    names, MAG, SGN, score_sec = score_stage(ids)

    # reproduction check against c60's design targets (same images, same overlay rule, same batches)
    c60r = json.load(open(os.path.join(RES, "c60_design_coverage.json")))["design_targets"]
    diffs = {}
    for j, nm in enumerate(names):
        for comp, v in (("mag", MAG[j]), ("sgn", SGN[j])):
            for s, msk in (("mal", lab == 1), ("ben", lab == 0)):
                k = "%s/%s/%s" % (nm, s, comp)
                diffs[k] = abs(float(v[msk].mean()) - c60r[k])
    repro = {"max_abs_diff_vs_c60_design_targets": max(diffs.values()), "n_compared": len(diffs),
             "pass_1e-6": bool(max(diffs.values()) <= 1e-6)}
    print("check vs c60 design targets: max abs diff %.3g" % repro["max_abs_diff_vs_c60_design_targets"], flush=True)

    t1 = time.time()
    primary = band(names, MAG, SGN, lab, pid, "design")
    print("primary: crit %.3f frontier %.4f exceed joint %d marginal %d (%.0fs)"
          % (primary["crit_36"], primary["frontier_36"], primary["n_exceed_0.05_joint"],
             primary["n_exceed_0.05_marginal"], time.time() - t1), flush=True)
    pooled = band(names, MAG, SGN, lab, pid, "pooled")
    # S-hyb: Table 1 realization on the 183 cohort images (c85 vectors), c60's rule elsewhere
    m = json.load(open(os.path.join(RES, "audit_cohort.json")))
    c85 = np.load(os.path.join(RES, "c85_artifact_vectors.npz"))
    assert list(c85["names"]) == names
    pos = {k: i for i, k in enumerate(ids)}
    cp = np.array([pos[i] for i in m["isic_id"]])
    assert (lab[cp] == np.asarray(m["target"])).all()
    MH, SH = MAG.copy(), SGN.copy()
    MH[:, cp] = c85["mag"]; SH[:, cp] = c85["sgn"]
    hyb = band(names, MH, SH, lab, pid, "design")
    for tag, r in (("pooled", pooled), ("hybrid", hyb)):
        print("%s: crit %.3f frontier %.4f exceed joint %d marginal %d"
              % (tag, r["crit_36"], r["frontier_36"], r["n_exceed_0.05_joint"], r["n_exceed_0.05_marginal"]), flush=True)

    def brief(r):
        return {k: v for k, v in r.items() if k != "components"}

    res = {"analysis": "c89 artifact tier on the whole audit fold (POST HOC sensitivity; does not replace the headline)",
           "plan": "results/c89_plan.json",
           "script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
           "n_images": len(ids), "n_malignant": int(lab.sum()), "n_benign": int((lab == 0).sum()),
           "n_patients": int(len(set(pid))), "n_benign_patients": int(len(set(pid[lab == 0]))),
           "n_malignant_patients": int(len(set(pid[lab == 1]))),
           "overlay_realization": ("PRIMARY: c60 rule, np.random.RandomState(42 + fold position) per image, "
                                   "hair_light, hair, hair_heavy, ruler drawn in that order (c60.score_frame). "
                                   "Table 1 uses RandomState(42 + cohort index) instead; the 183 cohort images "
                                   "therefore carry different overlays here than in Table 1."),
           "all_weighting": ("all = (83 * mal + 100 * ben) / 183 with fixed design weights in every resample "
                             "(c60 design target, cohort design ratio). Secondary: image-weighted pooled mean."),
           "scoring_protocol": "c60.score_frame: six dj_* checkpoints, ImageNet normalisation, bf16 autocast, batch 32 in chunks of 128",
           "band_protocol": "c56: 36-component studentized max-statistic patient bootstrap, 20,000 resamples, seed 41, 95%",
           "rounding": "lower ends and frontiers rounded down, upper ends rounded up, 3 dp",
           "margin_p": MARGIN, "boot": BOOT, "seed_boot": SEED_BOOT,
           "reproduction_check_c60": repro,
           "primary": primary,
           "secondary_pooled_all": brief(pooled),
           "secondary_hybrid_table1_overlays_on_cohort": brief(hyb),
           "reference_headline_c56": {"frontier_36": 0.0422275400273533, "crit_36": 3.118227187654512,
                                      "magnitude_exceed_36": 17, "n_images": 183, "n_patients": 100},
           "reference_c60_redraws": {"frontier_median": 0.04699395912454757,
                                     "frontier_2.5_97.5": [0.03263946613951531, 0.0661805173704362]},
           "scoring_sec": None if score_sec is None else round(score_sec, 1),
           "runtime_sec": round(time.time() - t0, 1)}
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved ->", OUT, "(%.0fs)" % res["runtime_sec"], flush=True)


if __name__ == "__main__":
    main()
