#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C42: positive control, showing that the artifact criterion has an acceptance region.

Problem. Across the whole paper, no cell reaches `equivalence`. A criterion that can only reject
has not been shown to discriminate. The identity-edit check returning exactly 0 only validates the
pipeline, not the discriminating power on real interventions.

Design. Lower the hair dose below the levels used in the paper and run the same three-way criterion on
the same patient-disjoint cohort as Table 2. If some cell reaches `equivalence` at a small enough dose, and the verdict
moves gradually to exceedance as the dose increases, then the criterion both has an acceptance region and tracks
intervention strength. That turns the dose-response from a description into a calibration of the criterion itself.

Dose ladder. The function _hair(x, rng, n, alpha) takes the number of strands n and the opacity alpha. The paper's three doses are
(6, 0.40), (14, 0.60), (24, 0.75), corresponding to 1.0%, 2.4% and 4.1% of pixels changed. We add three
clearly lower doses below them.

Out -> results/c42_positive_control.json
Run:  python code/c42_positive_control.py [--boot 4000]
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
import c29_disjoint_train as c29    # noqa: E402
import c30_disjoint_audit as c30    # noqa: E402
import c3_cace_concept as c3m       # noqa: E402
import audit_cohort               # noqa: E402  (fixed list of audit images)

p2, DEVICE = c3m.p2, c3m.DEVICE
BASE = os.path.dirname(HERE)
CKPT = os.path.join(BASE, "checkpoints")
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c42_positive_control.json")

SEED, N_EVAL = c30.SEED, c30.N_EVAL
MARGIN_P = c30.MARGIN_P

# dose ladder: three new levels below, then the paper's three levels
LADDER = [("hair_x1", 1, 0.15), ("hair_x2", 2, 0.25), ("hair_x3", 3, 0.35),
          ("hair_light", 6, 0.40), ("hair", 14, 0.60), ("hair_heavy", 24, 0.75)]


def register_doses():
    """Add the three low doses to the ARTIFACTS table. The other three already exist and keep their definitions."""
    for name, n, a in LADDER:
        if name in c2m.ARTIFACTS:
            continue
        c2m.ARTIFACTS[name] = (lambda n_, a_: (lambda x, rng: c2m._hair(x, rng, n=n_, alpha=a_)))(n, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=c30.BOOT)
    ap.add_argument("--models", default="all", choices=["all", "b3"])
    args = ap.parse_args()

    t0 = time.time()
    print(f"##### C42 positive control {time.strftime('%H:%M:%S')} #####", flush=True)
    register_doses()
    names = [n for n, _, _ in LADDER] + ["ruler", "identity"]
    HELD = ["hair_light", "hair", "hair_heavy", "ruler"]   # the same set as Table 2

    imgs, labels, pid, _ = audit_cohort.load()   # fixed list, the partition is not rebuilt
    print(f"audit cohort: {len(imgs)} images / {len(np.unique(pid))} patients "
          f"({int((labels==1).sum())} malignant)", flush=True)

    edits, change_frac = c2m.precompute_edits(imgs, names, SEED)
    print("pixels changed by dose: " +
          "  ".join(f"{n}={100*change_frac[n]:.2f}%" for n in names), flush=True)

    trio = dict(c30.TRIO) if args.models == "all" else {"b3": c30.TRIO["b3"]}
    res = {"margin_p": MARGIN_P, "boot": args.boot, "seed": SEED,
           "cohort": {"n_images": int(len(imgs)), "n_patients": int(len(np.unique(pid))),
                      "n_malignant": int((labels == 1).sum())},
           "ladder": [{"name": n, "n_strands": s, "alpha": a, "change_frac": change_frac[n]}
                      for n, s, a in LADDER],
           "identity_change_frac": change_frac["identity"],
           "note": ("Calibration of the artifact criterion. The same three-way interval rule and "
                    "the same patient-disjoint audit cohort as the main artifact table, evaluated "
                    "over a dose ladder that extends below the doses reported there. A cell that "
                    "demonstrates equivalence shows the criterion has an acceptance region."),
           "models": {}}

    W = 104
    print("=" * W)
    print(f"{'backbone':9s}{'model':6s}{'dose':11s}{'%px':>7s}{'subgroup':11s}"
          f"{'E|Δp|':>9s}{'95% CI':>22s}{'verdict':>16s}")
    for bb, models in trio.items():
        for ck, kind, nm in models:
            f = os.path.join(CKPT, ck + ".pt")
            if not os.path.exists(f):
                print(f"  [{ck}] missing checkpoint, skipping", flush=True); continue
            m = c30.make_model(kind).to(DEVICE)
            m.load_state_dict(torch.load(f, map_location=DEVICE)); m.eval()
            p0 = c30.probs(m, imgs)
            cells = {}
            for a in names:
                pe = c30.probs(m, [edits[i][a] for i in range(len(imgs))])
                absdp = np.abs(pe - p0)
                cells[a] = {}
                for sub, msk in [("all", np.ones(len(labels), bool)),
                                 ("malignant", labels == 1), ("benign", labels == 0)]:
                    _, lo, hi = c30.boot_ci(absdp[msk], pid[msk], n_boot=args.boot)
                    v = c30.verdict(lo, hi, MARGIN_P)
                    cells[a][sub] = {"mean": float(absdp[msk].mean()), "ci": [lo, hi],
                                     "verdict": v}
                    print(f"{bb:9s}{nm:6s}{a:11s}{100*change_frac[a]:7.2f}{sub:>11s}"
                          f"{absdp[msk].mean():9.4f}{f'[{lo:.4f}, {hi:.4f}]':>22s}{v:>16s}",
                          flush=True)
            # the Table 2 quantity: mean |Δp| over the four held-out edits
            agg = np.mean([np.abs(c30.probs(m, [edits[i][a] for i in range(len(imgs))]) - p0)
                           for a in HELD], 0)
            cells["_heldout_aggregate"] = {}
            for sub, msk in [("all", np.ones(len(labels), bool)),
                             ("malignant", labels == 1), ("benign", labels == 0)]:
                _, lo, hi = c30.boot_ci(agg[msk], pid[msk], n_boot=args.boot)
                cells["_heldout_aggregate"][sub] = {"mean": float(agg[msk].mean()),
                                                    "ci": [lo, hi],
                                                    "verdict": c30.verdict(lo, hi, MARGIN_P)}
            print(f"{bb:9s}{nm:6s}{'AGG(4 held-out)':21s}"
                  f"{agg.mean():9.4f}   <- Table 2 quantity", flush=True)
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            res["models"].setdefault(bb, {})[nm] = cells
            json.dump(res, open(OUT + ".partial", "w"), indent=1)

    # summary: does the criterion have an acceptance region, and where is the transition
    summ = {"equivalence_cells": [], "indeterminate_cells": [], "n_cells": 0}
    for bb, mm in res["models"].items():
        for nm, cells in mm.items():
            for a in names:
                for sub, c in cells[a].items():
                    summ["n_cells"] += 1
                    if c["verdict"] == "equivalence":
                        summ["equivalence_cells"].append(f"{bb}/{nm}/{a}/{sub}")
                    elif c["verdict"] == "indeterminate":
                        summ["indeterminate_cells"].append(f"{bb}/{nm}/{a}/{sub}")
    summ["criterion_has_acceptance_region"] = bool(summ["equivalence_cells"])
    res["summary"] = summ
    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    if os.path.exists(OUT + ".partial"):
        os.remove(OUT + ".partial")

    print("=" * W)
    print(f"cells at equivalence : {len(summ['equivalence_cells'])} / {summ['n_cells']}")
    print(f"cells indeterminate  : {len(summ['indeterminate_cells'])}")
    print("the criterion has an acceptance region" if summ["criterion_has_acceptance_region"]
          else "the criterion does not reach equivalence at any dose tried")
    print(f"saved -> {OUT}  ({res['runtime_sec']}s)")


if __name__ == "__main__":
    main()
