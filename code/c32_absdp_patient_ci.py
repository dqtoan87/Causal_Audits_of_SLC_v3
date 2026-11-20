#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C32: the nine E|Δp| cells of the artifact criterion, with a patient-clustered bootstrap.

Why. Table III reads the three-way verdict on E|Δp| for nine cells (three models × three subgroups).
C13 pools 4 edits × N images into one sample and bootstraps on it, i.e. it treats the four edits of
the same image as four independent observations. The interval is therefore too narrow at two levels:
between the edits of one image, and between the images of one patient. This script first averages
the four edits within each image, then resamples by patient, the same unit the paper uses for every
other Tier-2 inferential quantity.

No GPU needed: it rereads the predictions stored in c19_preds.npz.

Out -> results/c32_absdp_patient_ci.json
Run:  .venv/bin/python Causal_XAI_v3/code/c32_absdp_patient_ci.py
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
OUT = os.path.join(RES, "c32_absdp_patient_ci.json")
HELD = ["art:hair_light", "art:hair", "art:hair_heavy", "art:ruler"]
MARGIN, BOOT, SEED = 0.05, 20000, 11
NAME = {"erm": "ERM", "slim": "Slim", "causal": "Inv"}


def boot_ci(v, groups, n_boot=BOOT, seed=SEED):
    """Cluster bootstrap: resample patients, keeping all images of each selected patient."""
    gi = np.unique(groups, return_inverse=True)[1]
    idx = [np.where(gi == k)[0] for k in range(gi.max() + 1)]
    rng = np.random.RandomState(seed)
    pick = rng.randint(0, len(idx), (n_boot, len(idx)))
    b = np.array([v[np.concatenate([idx[k] for k in row])].mean() for row in pick])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def verdict(lo, hi, margin):
    if lo > margin:
        return "exceedance"
    if hi <= margin:
        return "equivalence"
    return "indeterminate"


def main():
    # Patients are read directly from the cache. Rebuilding the partition and resampling would assign
    # patient ids that do not match the images, breaking every cluster bootstrap interval.
    d = np.load(os.path.join(RES, "c19_preds.npz"), allow_pickle=True)
    labels = d["labels"]; names = [str(x) for x in d["names"]]
    pid = np.array([str(x) for x in d["patient_id"]])
    assert len(pid) == len(labels)
    cols = [names.index(x) for x in HELD]

    res = {"n_images": int(len(labels)), "n_patients": int(len(np.unique(pid))),
           "n_malignant": int((labels == 1).sum()), "margin_p": MARGIN, "boot": BOOT,
           "note": ("Mean absolute probability shift over the four held-out artifact edits. "
                    "The four edits are averaged within an image first, then patients are the "
                    "resampling unit, so neither edits of one image nor images of one patient "
                    "are treated as independent."),
           "cells": {}}
    print("model  subgroup    E|dp|   95% CI (patient)      verdict", flush=True)
    for m in NAME:
        per = np.mean([np.abs(d["pe_" + m][:, j] - d["p0_" + m]) for j in cols], 0)
        for sub, mask in [("all", np.ones(len(labels), bool)),
                          ("malignant", labels == 1), ("benign", labels == 0)]:
            lo, hi = boot_ci(per[mask], pid[mask])
            res["cells"].setdefault(NAME[m], {})[sub] = {
                "mean": float(per[mask].mean()), "ci": [lo, hi], "n": int(mask.sum()),
                "verdict": verdict(lo, hi, MARGIN)}
            print("%-6s %-10s %.3f  [%.3f, %.3f]  %s"
                  % (NAME[m], sub, per[mask].mean(), lo, hi, verdict(lo, hi, MARGIN)), flush=True)

    cells = [c for mm in res["cells"].values() for c in mm.values()]
    res["summary"] = {
        "n_cells": len(cells),
        "exceedance": sum(c["verdict"] == "exceedance" for c in cells),
        "min_lower_bound": min(c["ci"][0] for c in cells),
        "max_mean": max(c["mean"] for c in cells),
        "by_margin": {str(t): {"exceedance": sum(c["ci"][0] > t for c in cells),
                               "equivalence": sum(c["ci"][1] <= t for c in cells),
                               "indeterminate": sum(not (c["ci"][0] > t or c["ci"][1] <= t)
                                                    for c in cells)}
                      for t in (0.01, 0.025, 0.05, 0.10)}}
    print("\nsummary:", json.dumps(res["summary"], indent=1), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved ->", OUT, flush=True)


if __name__ == "__main__":
    main()
