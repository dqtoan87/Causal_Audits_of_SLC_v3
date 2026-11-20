#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C59: rerun the reimplementation's own prediction cache, generator included.

C58 compared the two pipelines layer by layer but read the reimplementation's edited predictions
from its cache. This script regenerates them with the reimplementation's own code: its image
loader, its overlay generator with its own per-edit seeds, its model loader and its inference
function, on its own checkpoints. Nothing of this pipeline enters except the patient bootstrap and
the three-way rule, which are the same functions C51 and C58 use.

For every checkpoint it reports
  (a) the unedited probabilities of the rerun against the cache, and of a second rerun against the
      first (run-to-run variation of the inference step);
  (b) both components of the four-operator mixture on all, malignant and benign images, from the
      cache and from the rerun, with the patient bootstrap, the three-way verdict of each component
      at the 0.05 tolerance and the joint decision.

If the rerun reproduces the cache, the disagreement of C58 comes from the overlay realization. If
the rerun moves away from the cache, the cache is not reproducible from the code and checkpoint that
are on disk.

Nothing is written into the reimplementation's directory.
Out -> results/c59_rebuild_generator.json
Run:  python code/c59_rebuild_generator.py --other /path/to/reimplementation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c30_disjoint_audit as c30     # noqa: E402

BASE = os.path.dirname(HERE)
OUT = os.path.join(BASE, "results", "c59_rebuild_generator.json")
HELD, MARGIN = c30.HELD, c30.MARGIN_P
TAGS = ("b3_erm_s42", "b3_slim_s42", "b3_inv_s42", "r50_erm_s42", "r50_slim_s42", "r50_inv_s42")


def cell(p0, pe, msk, pid):
    """Both components of the four-operator mixture and the joint decision, as in C30."""
    mix = np.mean([np.abs(pe[a] - p0) for a in HELD], 0)[msk]
    e, lo, hi = c30.boot_ci(mix, pid[msk])
    sg = np.mean([pe[a] - p0 for a in HELD], 0)[msk]
    es, los, his = c30.boot_ci(sg, pid[msk])
    vm, vs = c30.verdict(lo, hi, MARGIN), c30.verdict_signed(los, his, MARGIN)
    return {"magnitude": float(e), "ci": [float(lo), float(hi)], "verdict": vm,
            "signed": float(es), "signed_ci": [float(los), float(his)], "signed_verdict": vs,
            "joint_verdict": c30.combine(vs, vm)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--other", required=True, help="root directory of the reimplementation")
    O = ap.parse_args().other
    t0 = time.time()
    sys.path.insert(0, os.path.join(O, "code"))
    from cxai import edits as oed, imaging as oimg, infer as oinf, paths as opaths   # noqa: E402
    import b6_cache_preds as ob6                                                     # noqa: E402

    z = np.load(os.path.join(O, "results", "c19_preds.npz"), allow_pickle=True)
    ids = [str(i) for i in z["isic_ids"]]
    lab = np.asarray(z["target"]).astype(int)
    pid = np.asarray(z["patient_ids"]).astype(str)
    groups = {"all": np.ones(len(ids), bool), "malignant": lab == 1, "benign": lab == 0}

    base = torch.stack([oimg.load01(oimg.isic_path(i, opaths.ISIC_IMAGES)) for i in ids])
    var = {}
    for name in HELD:
        st = [oed.apply_edit(base[i], name, np.random.default_rng(ob6.EDIT_SEED[name] + i))
              for i in range(len(base))]
        var[name] = torch.stack(st)
    pixfrac_same = {a: bool(np.allclose(
        [oed.pixel_change_fraction(base[i], var[a][i]) for i in range(len(base))],
        np.asarray(z[f"pixfrac|{a}"], float))) for a in HELD}

    res = {"n_images": len(ids), "margin_p": MARGIN, "edit_seeds": {a: ob6.EDIT_SEED[a] for a in HELD},
           "pixel_fraction_matches_cache": pixfrac_same, "models": {}}
    for tag in TAGS:
        ck = os.path.join(O, "checkpoints", tag + ".pt")
        m, _ = oinf.load_model(ck, device=c30.DEVICE)
        p0 = oinf.malignant_prob(m, base, device=c30.DEVICE)
        p0b = oinf.malignant_prob(m, base, device=c30.DEVICE)
        pe = {a: oinf.malignant_prob(m, var[a], device=c30.DEVICE) for a in HELD}
        del m; torch.cuda.empty_cache()
        q0 = np.asarray(z[f"{tag}|identity"], float)
        qe = {a: np.asarray(z[f"{tag}|{a}"], float) for a in HELD}
        d0 = np.abs(p0 - q0)
        de = np.concatenate([np.abs(pe[a] - qe[a]) for a in HELD])
        r = {"unedited_rerun_vs_cache": {"mean_abs": float(d0.mean()), "max_abs": float(d0.max()),
                                         "share_over_0.05": float((d0 > 0.05).mean())},
             "edited_rerun_vs_cache_mean_abs": float(de.mean()),
             "unedited_rerun_vs_rerun_max_abs": float(np.abs(p0 - p0b).max()),
             "cache": {g: cell(q0, qe, msk, pid) for g, msk in groups.items()},
             "rerun": {g: cell(p0, pe, msk, pid) for g, msk in groups.items()}}
        res["models"][tag] = r
        print(tag, "p0 rerun-cache mean %.4f max %.4f | rerun-rerun max %.2e" % (
            d0.mean(), d0.max(), r["unedited_rerun_vs_rerun_max_abs"]),
            "| benign cache %.4f %s %s  rerun %.4f %s %s" % (
                r["cache"]["benign"]["magnitude"], r["cache"]["benign"]["verdict"],
                r["cache"]["benign"]["joint_verdict"], r["rerun"]["benign"]["magnitude"],
                r["rerun"]["benign"]["verdict"], r["rerun"]["benign"]["joint_verdict"]), flush=True)
    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("pixel fractions match cache:", pixfrac_same)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
