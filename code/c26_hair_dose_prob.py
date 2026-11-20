#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C26: dose response of the "hair" intervention on the probability scale.

The logit scale cannot show a dose response because 93 to 99% of predictions at the two high doses
already hit the clipping bound (Section V-B). This script computes the signed probability shift for
the three hair densities (1.0%, 2.4%, 4.1% of pixels changed) on each model's thresholded malignant
set, with patient-level bootstrap confidence intervals.

It reads cached predictions (results/c19_preds.npz), so it needs neither a GPU nor the raw images;
it only needs the metadata CSV to rebuild the patient_id of the 178 evaluation images, as c32/c33 do.

Out -> results/c26_hair_dose_prob.json   (source of Fig 3d and every dose number in §V-B,
       including the contrasts between dose steps and their confidence intervals)
Run:  python Causal_XAI_v3/code/c26_hair_dose_prob.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c21_patient_cluster as c21   # noqa: E402

BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c26_hair_dose_prob.json")

DOSES = ["art:hair_light", "art:hair", "art:hair_heavy"]
X_PCT = [1.0, 2.4, 4.1]
THRESH, BOOT, SEED = 0.10, 4000, 7
NAME = {"erm": "ERM", "slim": "Slim", "causal": "Inv"}


def boot_ci(v, groups, n_boot=BOOT, seed=SEED):
    """Cluster bootstrap: resample patients, keeping all images of each selected patient."""
    gi = np.unique(groups, return_inverse=True)[1]
    idx = [np.where(gi == k)[0] for k in range(gi.max() + 1)]
    rng = np.random.RandomState(seed)
    pick = rng.randint(0, len(idx), (n_boot, len(idx)))
    b = np.array([v[np.concatenate([idx[k] for k in row])].mean() for row in pick])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    # Patients are read directly from the cache. Rebuilding the partition and resampling would assign
    # patient ids that do not match the images, breaking every cluster bootstrap interval.
    d = np.load(os.path.join(RES, "c19_preds.npz"), allow_pickle=True)
    labels = d["labels"]; names = [str(x) for x in d["names"]]
    pid = np.array([str(x) for x in d["patient_id"]])
    assert len(pid) == len(labels)

    cols = [names.index(x) for x in DOSES]
    res = {"x_pct": X_PCT,
           "note": ("Signed mean probability shift of the hair edit at three densities, on each "
                    "model's thresholded malignant set, with 95% bootstrap intervals over patients.")}
    for m in ["erm", "slim", "causal"]:
        p0, pe = d[f"p0_{m}"], d[f"pe_{m}"]
        sel = (labels == 1) & (p0 >= THRESH)
        rows = []
        for c in cols:
            v = (pe[:, c] - p0)[sel]
            lo, hi = boot_ci(v, pid[sel])
            rows.append({"mean": float(v.mean()), "ci": [lo, hi]})
        # contrasts between dose steps: this is the quantity §V-B cites, so we must
        # bootstrap the per-image difference, not subtract two separately bootstrapped means.
        steps = []
        for lab, (i, j) in [("1.0->2.4", (0, 1)), ("2.4->4.1", (1, 2))]:
            v = (pe[:, cols[j]] - pe[:, cols[i]])[sel]
            lo, hi = boot_ci(v, pid[sel])
            steps.append({"step": lab, "mean": float(v.mean()), "ci": [lo, hi],
                          "excludes_zero": bool((lo > 0) == (hi > 0))})
        res[NAME[m]] = {"n": int(sel.sum()), "dose": rows, "steps": steps}
        print(f"  {NAME[m]:5s} n={int(sel.sum()):3d}  " +
              "  ".join(f"{r['mean']:+.3f}" for r in rows) +
              "   steps " + "  ".join(f"{s['mean']:+.3f}[{s['ci'][0]:+.3f},{s['ci'][1]:+.3f}]"
                                      for s in steps), flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
