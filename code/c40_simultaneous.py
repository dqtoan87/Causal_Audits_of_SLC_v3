#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C40: simultaneous confidence bands for the cells of the artifact criterion.

Why. The paper rules on each cell with a marginal 95% interval and then states "all nine cells" or
"all eighteen cells". With dependent cells, 95% marginal coverage does not give 95% simultaneous
coverage, so the global statement rests on something not yet computed. This script computes the
simultaneous band with a max statistic: on each patient resample, take the largest standardised
deviation over all cells, then use the 95th percentile of that distribution as the shared critical
value.

Done for two sets:
  (a) the nine cells of the primary sample: rereads c19_preds.npz, no GPU needed;
  (b) the eighteen cells of the patient-disjoint cohort: loads six checkpoints and recomputes per image.

Out -> results/c40_simultaneous.json
Run:  .venv/bin/python Causal_XAI_v3/code/c40_simultaneous.py
"""
from __future__ import annotations

import json, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c21_patient_cluster as c21
import c30_disjoint_audit as c30
import audit_cohort               # noqa: E402  (fixed list of audit images)
import c29_disjoint_train as c29
import c2_cace_artifact as c2m

RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c40_simultaneous.json")
MARGIN, BOOT, SEED = 0.05, 20000, 41
HELD = ["hair_light", "hair", "hair_heavy", "ruler"]
p1, p2, DEVICE = c30.p1, c30.p2, c30.DEVICE


def cells_boot(per_model, labels, pid, n_boot=BOOT, seed=SEED):
    """per_model: {cell name: per-image vector}. Return the point estimate, the marginal band and the simultaneous band."""
    keys = list(per_model)
    gi = np.unique(pid, return_inverse=True)[1]
    idx = [np.where(gi == k)[0] for k in range(gi.max() + 1)]
    rng = np.random.RandomState(seed)
    picks = [np.concatenate([idx[k] for k in rng.randint(0, len(idx), len(idx))]) for _ in range(n_boot)]
    est = {k: float(per_model[k].mean()) for k in keys}
    draws = {k: np.array([per_model[k][s].mean() for s in picks]) for k in keys}
    se = {k: float(draws[k].std(ddof=1)) for k in keys}
    marg = {k: (float(np.percentile(draws[k], 2.5)), float(np.percentile(draws[k], 97.5))) for k in keys}
    # max statistic over the standardised cells
    Z = np.stack([np.abs(draws[k] - est[k]) / max(se[k], 1e-12) for k in keys])
    crit = float(np.percentile(Z.max(0), 95))
    sim = {k: (est[k] - crit * se[k], est[k] + crit * se[k]) for k in keys}
    return est, se, marg, sim, crit


def report(tag, est, se, marg, sim, crit, res):
    print("\n--- %s | simultaneous critical value z* = %.3f (marginal 1.96) ---" % (tag, crit), flush=True)
    print("%-16s %7s %20s %22s %s" % ("cell", "E|dp|", "95% marginal", "95% simultaneous", "exceeds 0.05?"), flush=True)
    n_m = n_s = 0
    for k in est:
        m, s2 = marg[k], sim[k]
        n_m += m[0] > MARGIN; n_s += s2[0] > MARGIN
        print("%-16s %7.3f  [%.3f, %.3f]      [%.3f, %.3f]   marginal %s / simultaneous %s"
              % (k, est[k], m[0], m[1], s2[0], s2[1],
                 "yes" if m[0] > MARGIN else "no", "yes" if s2[0] > MARGIN else "no"), flush=True)
    print("  => exceeding threshold: %d/%d by marginal interval, %d/%d by simultaneous band"
          % (n_m, len(est), n_s, len(est)), flush=True)
    res[tag] = {"crit_z": crit, "n_cells": len(est),
                "exceed_marginal": int(n_m), "exceed_simultaneous": int(n_s),
                "cells": {k: {"mean": est[k], "se": se[k], "ci_marginal": list(marg[k]),
                              "ci_simultaneous": list(sim[k])} for k in est}}


def main():
    t0 = time.time()
    print(f"##### C40 simultaneous bands {time.strftime('%H:%M:%S')} #####", flush=True)
    res = {"margin_p": MARGIN, "boot": BOOT,
           "note": ("Simultaneous confidence bands over the artifact-criterion cells, using a "
                    "max-statistic patient bootstrap. Marginal 95% intervals do not give 95% "
                    "simultaneous coverage across dependent cells, so the global statements are "
                    "rechecked against a band that does.")}

    # ---- (a) nine cells of the primary sample ----
    # Patients are read directly from the cache; the partition is not rebuilt at analysis time.
    d = np.load(os.path.join(RES, "c19_preds.npz"), allow_pickle=True)
    labels = d["labels"]; names = [str(x) for x in d["names"]]
    pid = np.array([str(x) for x in d["patient_id"]])
    cols = [names.index("art:" + x) for x in HELD]
    cell = {}
    for m, nm in [("erm", "ERM"), ("slim", "Slim"), ("causal", "Inv")]:
        per = np.mean([np.abs(d["pe_" + m][:, j] - d["p0_" + m]) for j in cols], 0)
        for sub, msk in [("all", np.ones(len(labels), bool)), ("mal", labels == 1), ("ben", labels == 0)]:
            cell["%s/%s" % (nm, sub)] = per[msk]
    # the cells have different masks -> joint resampling by patient, keeping the original indices
    per_full = {}
    for m, nm in [("erm", "ERM"), ("slim", "Slim"), ("causal", "Inv")]:
        per = np.mean([np.abs(d["pe_" + m][:, j] - d["p0_" + m]) for j in cols], 0)
        per_full[nm] = per
    def boot_multi(per_full, labels, pid):
        keys, vecs, masks = [], [], []
        for nm in per_full:
            for sub, msk in [("all", np.ones(len(labels), bool)), ("mal", labels == 1), ("ben", labels == 0)]:
                keys.append("%s/%s" % (nm, sub)); vecs.append(per_full[nm]); masks.append(msk)
        gi = np.unique(pid, return_inverse=True)[1]
        idx = [np.where(gi == k)[0] for k in range(gi.max() + 1)]
        rng = np.random.RandomState(SEED)
        est = {k: float(v[m].mean()) for k, v, m in zip(keys, vecs, masks)}
        draws = {k: [] for k in keys}
        for _ in range(BOOT):
            sel = np.concatenate([idx[k] for k in rng.randint(0, len(idx), len(idx))])
            for k, v, m in zip(keys, vecs, masks):
                s2 = sel[m[sel]]
                draws[k].append(v[s2].mean() if len(s2) else np.nan)
        draws = {k: np.array(v) for k, v in draws.items()}
        se = {k: float(np.nanstd(draws[k], ddof=1)) for k in keys}
        marg = {k: (float(np.nanpercentile(draws[k], 2.5)), float(np.nanpercentile(draws[k], 97.5))) for k in keys}
        Z = np.stack([np.abs(draws[k] - est[k]) / max(se[k], 1e-12) for k in keys])
        crit = float(np.nanpercentile(np.nanmax(Z, 0), 95))
        sim = {k: (est[k] - crit * se[k], est[k] + crit * se[k]) for k in keys}
        return est, se, marg, sim, crit
    report("primary_9", *boot_multi(per_full, labels, pid), res=res)

    # ---- (b) eighteen cells of the patient-disjoint cohort ----
    imgs, lab2, pid2, _ = audit_cohort.load()   # fixed list, the partition is not rebuilt
    edits, _ = c2m.precompute_edits(imgs, c30.ART, c30.SEED)
    per_full2 = {}
    for bb, trio in c30.TRIO.items():
        for ck, kind, nm in trio:
            f = os.path.join(os.path.dirname(HERE), "checkpoints", ck + ".pt")
            m = c30.make_model(kind).to(DEVICE)
            m.load_state_dict(torch.load(f, map_location=DEVICE)); m.eval()
            p0 = c30.probs(m, imgs)
            per = np.mean([np.abs(c30.probs(m, [edits[i][a] for i in range(len(imgs))]) - p0)
                           for a in HELD], 0)
            per_full2["%s-%s" % (bb, nm)] = per
            del m
            if DEVICE == "cuda": torch.cuda.empty_cache()
            print("  [%s %s] xong (%.0fs)" % (bb, nm, time.time() - t0), flush=True)
    report("disjoint_18", *boot_multi(per_full2, lab2, pid2), res=res)

    json.dump(res, open(OUT, "w"), indent=1)
    print("\nsaved -> %s  (%.0fs)" % (OUT, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
