#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C88: the artifact criterion of Table 4 for the epoch-1 EfficientNet-B3 Inv checkpoint.  (post hoc, reporting)

The first selection rule saved the EfficientNet-B3 Inv checkpoint at epoch 1, inside the penalty warm-up; it
was retrained after its results had been seen (Supplementary Section S8). This script applies the unchanged
artifact criterion of c30_disjoint_audit.py (same cohort, overlays, seed, bootstrap and tolerance) to that
checkpoint, released as checkpoints/archived_b3inv_ep1/dj_causal_b3.pt, so that its results are reported.
It imports c30 and changes nothing in it.

Out -> results/c88_artifact_inv_epoch1.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c30_disjoint_audit as c30         # noqa: E402
import c2_cace_artifact as c2m           # noqa: E402
import audit_cohort                      # noqa: E402

CK = os.path.join(c30.CKPT, "archived_b3inv_ep1", "dj_causal_b3.pt")
OUT = os.path.join(c30.RES, "c88_artifact_inv_epoch1.json")


def main():
    t0 = time.time()
    imgs, labels, pid, _ = audit_cohort.load()
    edits, change_frac = c2m.precompute_edits(imgs, c30.ART, c30.SEED)
    kind = [k for ck, k, nm in c30.TRIO["b3"] if nm == "Inv"][0]
    m = c30.make_model(kind).to(c30.DEVICE)
    m.load_state_dict(torch.load(CK, map_location=c30.DEVICE)); m.eval()
    p0 = c30.probs(m, imgs)
    per = {a: c30.probs(m, [edits[i][a] for i in range(len(imgs))]) for a in c30.ART}
    absdp = np.mean([np.abs(per[a] - p0) for a in c30.HELD], 0)
    sgndp = np.mean([per[a] - p0 for a in c30.HELD], 0)
    out = {"note": "post hoc reporting; c30 criterion unchanged; epoch-1 EfficientNet-B3 Inv checkpoint",
           "checkpoint_sha256": hashlib.sha256(open(CK, "rb").read()).hexdigest(),
           "margin_p": c30.MARGIN_P, "boot": c30.BOOT, "cells": {}}
    for sub, msk in [("all", np.ones(len(labels), bool)), ("malignant", labels == 1), ("benign", labels == 0)]:
        mean, lo, hi = c30.boot_ci(absdp[msk], pid[msk])
        smean, slo, shi = c30.boot_ci(sgndp[msk], pid[msk])
        v, sv = c30.verdict(lo, hi, c30.MARGIN_P), c30.verdict_signed(slo, shi, c30.MARGIN_P)
        out["cells"][sub] = {"mean": mean, "ci": [lo, hi], "verdict": v, "signed_mean": smean, "signed_ci": [slo, shi],
                             "signed_verdict": sv, "joint_verdict": c30.combine(sv, v)}
        print("%-10s %.3f [%.3f, %.3f] %s | %+.3f [%+.3f, %+.3f] %s" % (sub, mean, lo, hi, v, smean, slo, shi, sv), flush=True)
    out["audit_auroc"] = float(roc_auc_score(labels, p0))
    out["p0_mean_malignant"] = float(p0[labels == 1].mean())
    out["identity_max_abs_dlogit"] = float(np.abs(c30.clipped_logit(per["identity"]) - c30.clipped_logit(p0)).max())
    out["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print("AUROC %.3f; saved -> %s" % (out["audit_auroc"], OUT))


if __name__ == "__main__":
    main()
