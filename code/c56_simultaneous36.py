#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C56: simultaneous coverage over all 36 components of the artifact criterion.

c40 corrects the eighteen magnitude statistics, which carry every exceedance claim. This script is
the sensitivity check for that choice. It forms the max-statistic over the eighteen magnitude and
the eighteen signed statistics together, on the same patient resamples, and reports

  * the joint critical value over 36 components,
  * the smallest simultaneous lower bound of the magnitude component, which is the frontier of
    tolerances below which every cell exceeds jointly when the family is all 36 components,
  * the signed-component decisions read from the 36-component band.

Same cohort, checkpoints, overlays, seed and replicate count as c40.

Out -> results/c56_simultaneous36.json
"""
from __future__ import annotations

import json, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c30_disjoint_audit as c30
import audit_cohort               # noqa: E402
import c2_cace_artifact as c2m

RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c56_simultaneous36.json")
MARGIN, BOOT, SEED = 0.05, 20000, 41          # as in c40
HELD = ["hair_light", "hair", "hair_heavy", "ruler"]
DEVICE = c30.DEVICE


def decide(lo, hi, eps, magnitude):
    if magnitude:
        return "exceeds" if lo > eps else ("within" if hi < eps else "indet.")
    if lo > eps or hi < -eps:
        return "exceeds"
    if -eps < lo and hi < eps:
        return "within"
    return "indet."


def main():
    t0 = time.time()
    imgs, labels, pid, _ = audit_cohort.load()
    labels, pid = np.asarray(labels), np.asarray(pid)
    edits, _ = c2m.precompute_edits(imgs, c30.ART, c30.SEED)
    vec = {}
    for bb, trio in c30.TRIO.items():
        for ck, kind, nm in trio:
            f = os.path.join(os.path.dirname(HERE), "checkpoints", ck + ".pt")
            m = c30.make_model(kind).to(DEVICE)
            m.load_state_dict(torch.load(f, map_location=DEVICE)); m.eval()
            p0 = c30.probs(m, imgs)
            d = np.stack([c30.probs(m, [edits[i][a] for i in range(len(imgs))]) - p0 for a in HELD])
            vec["%s-%s" % (bb, nm)] = (np.abs(d).mean(0), d.mean(0))
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            print("  [%s %s] done (%.0fs)" % (bb, nm, time.time() - t0), flush=True)

    subs = [("all", np.ones(len(labels), bool)), ("mal", labels == 1), ("ben", labels == 0)]
    keys, vecs, masks, ismag = [], [], [], []
    for cell, (mag, sgn) in vec.items():
        for sname, msk in subs:
            for comp, v in (("mag", mag), ("sgn", sgn)):
                keys.append("%s/%s/%s" % (cell, sname, comp)); vecs.append(v); masks.append(msk)
                ismag.append(comp == "mag")
    gi = np.unique(pid, return_inverse=True)[1]
    idx = [np.where(gi == k)[0] for k in range(gi.max() + 1)]
    rng = np.random.RandomState(SEED)
    est = np.array([v[m].mean() for v, m in zip(vecs, masks)])
    draws = np.empty((len(keys), BOOT))
    for b in range(BOOT):
        sel = np.concatenate([idx[k] for k in rng.randint(0, len(idx), len(idx))])
        for j, (v, m) in enumerate(zip(vecs, masks)):
            s2 = sel[m[sel]]
            draws[j, b] = v[s2].mean()
    se = draws.std(axis=1, ddof=1)
    Z = np.abs(draws - est[:, None]) / np.maximum(se[:, None], 1e-12)
    ismag = np.array(ismag)
    crit36 = float(np.percentile(Z.max(0), 95))
    crit18 = float(np.percentile(Z[ismag].max(0), 95))
    lo36, hi36 = est - crit36 * se, est + crit36 * se
    lo18 = est - crit18 * se
    out = {"boot": BOOT, "seed": SEED, "margin_p": MARGIN, "n_components": len(keys),
           "crit_36": crit36, "crit_18_magnitude_only": crit18,
           "frontier_36": float(lo36[ismag].min()),
           "frontier_18": float(lo18[ismag].min()),
           "magnitude_exceed_36": int((lo36[ismag] > MARGIN).sum()),
           "signed_decisions_36": {}, "components": {}}
    for j, k in enumerate(keys):
        out["components"][k] = {"est": float(est[j]), "se": float(se[j]),
                                "sim36": [float(lo36[j]), float(hi36[j])]}
        if not ismag[j]:
            out["signed_decisions_36"][k] = decide(lo36[j], hi36[j], MARGIN, False)
    vals = list(out["signed_decisions_36"].values())
    out["signed_counts_36"] = {v: vals.count(v) for v in ("exceeds", "within", "indet.")}
    out["runtime_sec"] = round(time.time() - t0, 1)
    print("crit 36 = %.3f  crit 18 (magnitude) = %.3f" % (crit36, crit18))
    print("frontier 36 = %.4f  frontier 18 = %.4f" % (out["frontier_36"], out["frontier_18"]))
    print("magnitude exceed (36 band): %d/18; signed decisions: %s"
          % (out["magnitude_exceed_36"], out["signed_counts_36"]))
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved -> %s (%.0fs)" % (OUT, out["runtime_sec"]))


if __name__ == "__main__":
    main()
