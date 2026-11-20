#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C50: does the artifact verdict depend on the random realisation of the overlay?

Why. The overlay operator is random: the position, direction, length and curvature of each hair
strand are drawn at random, and the ruler row is drawn row by row. Every model-side number is
therefore a function of one specific draw (seed 42). Section S8 checked numerical reproducibility
(fp32, batch splitting) but not the variation due to the intervention itself, which is a different
and larger source of variation.

This matters because the verdict is a threshold rule: a cell near the tolerance can flip between
"exceeds" and "within tolerance" just because a different set of hair strands was drawn. The only
way to know is to rerun the whole grid over many realisations.

Design. Keep the checkpoints, image group, rule, tolerance and number of bootstrap draws fixed. Change
only the overlay seed. Uses c2m.precompute_edits with the list c30.ART in order, so seed 42 must
reproduce c30_disjoint_audit.json exactly; this is the correctness check of the script.

The original images do not depend on the seed, so p0 is computed once per checkpoint.

Out -> results/c50_overlay_seed.json
Run:  python Causal_XAI_v3/code/c50_overlay_seed.py [--seeds 42,1,2,3,7,11,99]
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
import c2_cace_artifact as c2m      # noqa: E402
import c30_disjoint_audit as c30    # noqa: E402
import audit_cohort                 # noqa: E402

DEVICE = c30.DEVICE
BASE = os.path.dirname(HERE)
CKPT = os.path.join(BASE, "checkpoints")
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c50_overlay_seed.json")
HELD = c30.HELD
MARGIN = c30.MARGIN_P


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,2,3,7,11,99")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    t0 = time.time()

    imgs, labels, pid, _ids = audit_cohort.load()
    log(f"audit cohort: {len(imgs)} images, {len(set(pid))} patients, {int(labels.sum())} malignant")

    # Overlay: generated once per seed, shared by all six checkpoints.
    edits = {}
    for s in seeds:
        per, cf = c2m.precompute_edits(imgs, c30.ART, s)
        edits[s] = ({a: [d[a] for d in per] for a in HELD},
                    {a: cf[a] for a in HELD})
        log(f"  seed {s}: overlays done, altered pixels "
            + " ".join(f"{a}={cf[a]*100:.2f}%" for a in HELD))

    subs = [("all", np.ones(len(labels), bool)),
            ("malignant", labels == 1), ("benign", labels == 0)]
    res = {"n_images": int(len(imgs)), "n_patients": int(len(set(pid))),
           "n_malignant": int(labels.sum()), "seeds": seeds, "margin_p": MARGIN,
           "boot": c30.BOOT, "held_out": HELD,
           "note": ("Same checkpoints, same audit images, same rule, same tolerance and same "
                    "bootstrap. Only the random realisation of the overlay operator changes. "
                    "Seed 42 is the realisation reported in the paper and must reproduce "
                    "c30_disjoint_audit.json exactly."),
           "pixel_change_frac": {str(s): edits[s][1] for s in seeds},
           "cells": {}}

    for bb, trio in c30.TRIO.items():
        for fn, kind, name in trio:
            path = os.path.join(CKPT, fn + ".pt")
            m = c30.make_model(kind).to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval()
            p0 = c30.probs(m, imgs)
            per_seed = {}
            for s in seeds:
                pe = {a: c30.probs(m, edits[s][0][a]) for a in HELD}
                absdp = np.mean([np.abs(pe[a] - p0) for a in HELD], 0)
                sgndp = np.mean([pe[a] - p0 for a in HELD], 0)
                for sub, msk in subs:
                    e, lo, hi = c30.boot_ci(absdp[msk], pid[msk])
                    se, slo, shi = c30.boot_ci(sgndp[msk], pid[msk])
                    v = c30.verdict(lo, hi, MARGIN)
                    sv = c30.verdict_signed(slo, shi, MARGIN)
                    per_seed.setdefault(sub, []).append(
                        {"seed": s, "magnitude": e, "ci": [lo, hi], "magnitude_verdict": v,
                         "signed": se, "signed_ci": [slo, shi], "signed_verdict": sv,
                         "joint_verdict": c30.combine(sv, v)})
            for sub, rows in per_seed.items():
                mags = [r["magnitude"] for r in rows]
                los = [r["ci"][0] for r in rows]
                vj = {r["joint_verdict"] for r in rows}
                res["cells"][f"{bb}|{name}|{sub}"] = {
                    "magnitude_min": float(min(mags)), "magnitude_max": float(max(mags)),
                    "magnitude_at_seed42": float(rows[0]["magnitude"]),
                    "spread_pct_of_seed42": float(100 * (max(mags) - min(mags)) / rows[0]["magnitude"]),
                    "lower_bound_min": float(min(los)),
                    "joint_verdicts": sorted(vj), "verdict_stable": bool(len(vj) == 1),
                    "per_seed": rows}
                log(f"  {bb:4s} {name:5s} {sub:10s} magnitude {min(mags):.4f} to {max(mags):.4f} "
                    f"({100*(max(mags)-min(mags))/rows[0]['magnitude']:.1f}% of seed 42) "
                    f"| smallest lower bound {min(los):.4f} | verdicts {sorted(vj)}")
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    cells = res["cells"]
    res["summary"] = {
        "n_cells": len(cells),
        "n_verdict_stable": sum(c["verdict_stable"] for c in cells.values()),
        "all_exceedance_every_seed": all(c["joint_verdicts"] == ["exceedance"] for c in cells.values()),
        "max_spread_pct": max(c["spread_pct_of_seed42"] for c in cells.values()),
        "min_lower_bound_over_seeds": min(c["lower_bound_min"] for c in cells.values()),
        "margin_ratio_of_min_lower_bound": min(c["lower_bound_min"] for c in cells.values()) / MARGIN}
    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"saved -> {OUT}")
    log(f"  {res['summary']['n_verdict_stable']}/{len(cells)} cells with a verdict stable across "
        f"{len(seeds)} realizations | smallest lower bound {res['summary']['min_lower_bound_over_seeds']:.4f}")


if __name__ == "__main__":
    main()
