#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C60: coverage of the 36-component band over repeated benign draws of the audit design.

The audit cohort holds every malignant image of the audit fold and 100 benign images drawn at random
from the fold's benign images. The patient bootstrap keeps that draw fixed. This script measures what
the draw does, holding the audit fold's patients fixed.

  1. Score every image of the audit fold that the design can select: all malignant images and every
     benign image (the benign frame), with the six audited checkpoints and the four held-out
     overlays. Overlays are seeded by the image's position in the fold, so an image carries the same
     overlay in every draw.
  2. Define the design target of each of the 36 components on this fold: the malignant subgroup is
     the mean over all malignant images; the benign subgroup is the mean over the whole benign frame;
     the all-images subgroup mixes the two at the design ratio, 83 to 100, as the image-weighted
     design does in expectation.
  3. Redraw the 100 benign images R times, as the design does (uniformly, without replacement, by
     image). For each draw, rebuild the cohort, run the 36-component max-statistic patient bootstrap
     exactly as in C56, and record whether the band covers every design target, the coverage of each
     component, the frontier and the number of cells that exceed at 0.05.

Scope. Stage one of the design, which patients enter the audit fold, is not simulated: the other folds
were used for training and selection. The result is therefore coverage over the benign draw for the
design target of this fold, not coverage for a clinical population.

Out -> results/c60_design_coverage.json
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import audit_cohort                # noqa: E402
import c2_cace_artifact as c2m     # noqa: E402
import c30_disjoint_audit as c30   # noqa: E402
import frozen_split                # noqa: E402

BASE = os.path.dirname(HERE)
OUT = os.path.join(BASE, "results", "c60_design_coverage.json")
HELD = ["hair_light", "hair", "hair_heavy", "ruler"]
MARGIN, N_BEN = 0.05, 100
R_DRAWS, BOOT, LEVEL = 500, 2000, 95
SEED_EDIT, SEED_DRAW, SEED_BOOT = 42, 2026, 41
CHUNK = 128


def score_frame(ids):
    """Per-image magnitude and signed shift of the four-operator mixture, for every checkpoint."""
    models = {}
    for bb, trio in c30.TRIO.items():
        for ck, kind, nm in trio:
            m = c30.make_model(kind).to(c30.DEVICE)
            m.load_state_dict(torch.load(os.path.join(BASE, "checkpoints", ck + ".pt"), map_location=c30.DEVICE))
            models["%s-%s" % (bb, nm)] = m.eval()
    mag = {k: np.empty(len(ids)) for k in models}
    sgn = {k: np.empty(len(ids)) for k in models}
    t0 = time.time()
    for s in range(0, len(ids), CHUNK):
        imgs = audit_cohort._read(ids[s:s + CHUNK])
        ed = []
        for j, x in enumerate(imgs):
            rng = np.random.RandomState(SEED_EDIT + s + j)     # fixed per image of the fold
            ed.append({a: c2m.ARTIFACTS[a](x, rng) for a in HELD})
        for k, m in models.items():
            p0 = c30.probs(m, imgs)
            d = np.stack([c30.probs(m, [e[a] for e in ed]) - p0 for a in HELD])
            mag[k][s:s + len(imgs)] = np.abs(d).mean(0)
            sgn[k][s:s + len(imgs)] = d.mean(0)
        if (s // CHUNK) % 5 == 0:
            print("  scored %d/%d (%.0fs)" % (s + len(imgs), len(ids), time.time() - t0), flush=True)
    return mag, sgn


def band(vecs, masks, pid, rng):
    """Max-statistic patient bootstrap over the components, as in C56. Returns est, se, crit."""
    gi = np.unique(pid, return_inverse=True)[1]
    idx = [np.where(gi == k)[0] for k in range(gi.max() + 1)]
    V = np.stack(vecs); M = np.stack(masks).astype(float)
    est = (V * M).sum(1) / M.sum(1)
    draws = np.empty((len(vecs), BOOT))
    for b in range(BOOT):
        sel = np.concatenate([idx[k] for k in rng.randint(0, len(idx), len(idx))])
        Ms = M[:, sel]
        draws[:, b] = (V[:, sel] * Ms).sum(1) / np.maximum(Ms.sum(1), 1)
    se = draws.std(axis=1, ddof=1)
    Z = np.abs(draws - est[:, None]) / np.maximum(se[:, None], 1e-12)
    return est, se, float(np.percentile(Z.max(0), LEVEL))


def main():
    t0 = time.time()
    _, _, df = frozen_split.load()
    df = df.reset_index(drop=True)
    ids = list(df["isic_id"]); lab = df["target"].to_numpy().astype(int)
    pid_all = df["patient_id"].astype(str).to_numpy()
    mal = np.where(lab == 1)[0]; ben = np.where(lab == 0)[0]
    print("audit fold: %d images, %d malignant, %d benign in the frame" % (len(ids), len(mal), len(ben)), flush=True)
    mag, sgn = score_frame(ids)

    cells = list(mag)
    keys, target = [], []
    for c in cells:
        for comp, v in (("mag", mag[c]), ("sgn", sgn[c])):
            tm, tb = v[mal].mean(), v[ben].mean()
            ta = (len(mal) * tm + N_BEN * tb) / (len(mal) + N_BEN)
            for sub, t in (("all", ta), ("mal", tm), ("ben", tb)):
                keys.append("%s/%s/%s" % (c, sub, comp)); target.append(t)
    target = np.array(target); ismag = np.array([k.endswith("mag") for k in keys])

    rd, rb = np.random.RandomState(SEED_DRAW), np.random.RandomState(SEED_BOOT)
    cover_all, cover_comp, frontier, n_exceed, crits = [], [], [], [], []
    for r in range(R_DRAWS):
        sel = np.r_[mal, rd.choice(ben, N_BEN, replace=False)]
        lab_s, pid_s = lab[sel], pid_all[sel]
        vecs, masks = [], []
        for c in cells:
            for comp, v in (("mag", mag[c]), ("sgn", sgn[c])):
                for sub, msk in (("all", np.ones(len(sel), bool)), ("mal", lab_s == 1), ("ben", lab_s == 0)):
                    vecs.append(v[sel]); masks.append(msk)
        est, se, crit = band(vecs, masks, pid_s, rb)
        lo, hi = est - crit * se, est + crit * se
        inside = (lo <= target) & (target <= hi)
        cover_all.append(bool(inside.all())); cover_comp.append(inside)
        frontier.append(float(lo[ismag].min())); n_exceed.append(int((lo[ismag] > MARGIN).sum()))
        crits.append(crit)
        if r % 50 == 0:
            print("  draw %d: crit %.3f frontier %.4f exceed %d covered %s (%.0fs)"
                  % (r, crit, frontier[-1], n_exceed[-1], cover_all[-1], time.time() - t0), flush=True)

    cc = np.mean(np.stack(cover_comp), 0)
    fr = np.array(frontier); ne = np.array(n_exceed)
    k_ok = int(np.sum(cover_all)); n = len(cover_all)
    mc_se = float(np.sqrt(max(k_ok / n * (1 - k_ok / n), 1e-12) / n))
    res = {"scope": "benign redraw within the audit fold; patients of the fold held fixed",
           "n_fold_images": len(ids), "n_malignant": int(len(mal)), "n_benign_frame": int(len(ben)),
           "n_benign_frame_patients": int(len(set(pid_all[ben]))), "n_benign_per_draw": N_BEN,
           "draws": R_DRAWS, "boot": BOOT, "level": LEVEL, "margin_p": MARGIN,
           "joint_coverage": k_ok / n, "joint_coverage_mc_se": mc_se,
           "component_coverage_min": float(cc.min()), "component_coverage_median": float(np.median(cc)),
           "component_coverage_by_key": {k: float(v) for k, v in zip(keys, cc)},
           "worst_components": [keys[i] for i in np.argsort(cc)[:5]],
           "crit_median": float(np.median(crits)), "crit_range": [float(min(crits)), float(max(crits))],
           "frontier_median": float(np.median(fr)), "frontier_2.5_97.5": [float(np.percentile(fr, 2.5)), float(np.percentile(fr, 97.5))],
           "frontier_min": float(fr.min()), "share_draws_all_18_exceed": float((ne == 18).mean()),
           "share_draws_frontier_above_0.05": float((fr > MARGIN).mean()),
           "design_targets": {k: float(t) for k, t in zip(keys, target)},
           "runtime_sec": round(time.time() - t0, 1)}
    json.dump(res, open(OUT, "w"), indent=1)
    print("joint coverage %.3f (MC se %.3f); component coverage min %.3f median %.3f"
          % (res["joint_coverage"], mc_se, res["component_coverage_min"], res["component_coverage_median"]))
    print("frontier median %.4f, 95%% range %s, min %.4f; all 18 exceed in %.1f%% of draws"
          % (res["frontier_median"], res["frontier_2.5_97.5"], res["frontier_min"], 100 * res["share_draws_all_18_exceed"]))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
