#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C34: the control-versus-control check of test (T), computed in full and saved.

Test (T) states: if two controls are both valid, the CaCE difference between them must be
indistinguishable from 0, since both are taken to measure only the common response to the
perturbation footprint. C12 pairs only valid controls; since no control is valid, C12 forms 0 or 1
pairs and saves no details. The paper reports a count over all formable pairs, so that number needs
a checkable source.

This script forms every pair among the controls the screen classifies as neutral (A2) for the two
targets at high dose, computes the CaCE difference between the two controls on the common set of
62 images, and bootstraps by patient. It builds only the controls it needs, so it is many times
lighter than C12.

Out -> results/c34_ctrl_vs_ctrl.json
Run:  .venv/bin/python Causal_XAI_v3/code/c34_ctrl_vs_ctrl.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c12_matched_control as c12    # noqa: E402
import audit_cohort               # noqa: E402

c3m, c7m, p1, p2 = c12.c3m, c12.c7m, c12.p1, c12.p2
DEVICE, SEED = c12.DEVICE, c12.SEED
BASE = os.path.dirname(HERE)
OUT = os.path.join(BASE, "results", "c34_ctrl_vs_ctrl.json")
BOOT = 20000
NAME = {"erm": "ERM", "slim": "Slim", "causal": "Inv"}
TOP = ["contrast@2", "color_var@2", "contrast@0", "color_var@0"]
# Pairs can only be formed where at least two controls are both neutral (A2) and pass overlap (A3) on
# the same target. The list below is read directly from the C12 screen, restricted to footprint-matched
# families; the uncalibrated placebos (u:hue, u:noise) are not footprint-matched, so they are not candidates for (T).
# Derived from c25_screen_mal.json (main scheme, malignant images, calibrated "_m" families):
# (A2) absolute <= 0.10 and ratio <= 0.15, (A3) overlap >= 0.95. rotax_m fails (A2) at contrast@0
# (ratio 0.163), and pixperm_m fails (A3) at contrast@2 on the malignant images (0.92).
NEUTRAL = {"contrast@2":  ["globrot_m"],
           "color_var@2": ["globrot_m", "pixperm_m"],
           "contrast@0":  ["globrot_m", "pixperm_m", "noise_m"],
           "color_var@0": ["globrot_m", "pixperm_m", "noise_m"]}


def boot_ci_img(v, n_boot=BOOT, seed=9):
    rng = np.random.RandomState(seed)
    b = v[rng.randint(0, len(v), (n_boot, len(v)))].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_ci(v, groups, n_boot=BOOT, seed=9):
    gi = np.unique(groups, return_inverse=True)[1]
    idx = [np.where(gi == k)[0] for k in range(gi.max() + 1)]
    rng = np.random.RandomState(seed)
    b = np.array([v[np.concatenate([idx[k] for k in rng.randint(0, len(idx), len(idx))])].mean()
                  for _ in range(n_boot)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    t0 = time.time()
    print(f"##### C34 control-vs-control {time.strftime('%H:%M:%S')} #####", flush=True)
    imgs01, labels, _pid, _ids = audit_cohort.load()   # fixed evaluation set
    labels = np.array(labels); mal = np.where(labels == 1)[0]
    pid = _pid
    pid_mal = pid[mal]

    ctrls = sorted({c for v in NEUTRAL.values() for c in v})
    names = list(TOP) + [f"{t}|{c}" for t in TOP for c in NEUTRAL[t]]
    edits = []
    for ii in mal:
        x = imgs01[ii]
        w, skin, lm = c3m.soft_mask(x)
        donor = imgs01[mal[(len(edits) + 1) % len(mal)]]
        fams = c12.make_families(x, w, skin, SEED + int(ii), donor=donor)
        d = {}
        for t in TOP:
            cname, k = t.split("@")
            d[t] = c3m.make_edit(x, cname, c3m.DOSES[int(k)], w, skin, lm)
        for t in TOP:
            tgt = float(c12.dpix(x, d[t]).mean())
            for c in NEUTRAL[t]:
                fn, lo, hi = fams[c]
                e, _, _, _ = c12.match_strength(fn, lo, hi, x, tgt)
                d[f"{t}|{c}"] = e
        edits.append(d)
        if (len(edits)) % 20 == 0:
            print("   ... %d/%d images (%.0fs)" % (len(edits), len(mal), time.time() - t0), flush=True)
    print("edits xong (%.0fs)" % (time.time() - t0), flush=True)

    P0, DL = {}, {}
    for m in c12.ORDER:
        model = p1.build_model("erm" if m == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(c12.CKPTS[m],
                                         map_location=DEVICE))
        model.eval()
        p0 = np.empty(len(mal)); dl = {n: np.empty(len(mal)) for n in names}
        for j in range(len(mal)):
            ps = c3m.model_prob(model, torch.stack([imgs01[mal[j]]] + [edits[j][n] for n in names]))
            p0[j] = ps[0]
            for k, n in enumerate(names):
                dl[n][j] = c3m.logit(ps[k + 1]) - c3m.logit(ps[0])
        P0[m], DL[m] = p0, dl
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("[%s] p_full=%.3f (%.0fs)" % (m, p0.mean(), time.time() - t0), flush=True)

    common = (P0["erm"] >= 0.1) & (P0["slim"] >= 0.1) & (P0["causal"] >= 0.1)
    pid_c = pid_mal[common]
    print("common set n=%d / %d patients" % (int(common.sum()), len(np.unique(pid_c))), flush=True)

    res = {"n_common": int(common.sum()), "n_patients_common": int(len(np.unique(pid_c))),
           "boot": BOOT, "neutral_sets": NEUTRAL,
           "note": ("Control-versus-control test of (T). For each model and each top-dose concept "
                    "target, every pair of controls the (A2) screen classifies as concept-neutral. "
                    "Two valid controls should give the same CaCE, so a non-zero difference "
                    "falsifies the footprint-only model that placebo subtraction assumes."),
           "pairs": [], "summary": {}}
    for m in c12.ORDER:
        for t in TOP:
            for a, b in itertools.combinations(NEUTRAL[t], 2):
                v = (DL[m][f"{t}|{a}"] - DL[m][f"{t}|{b}"])[common]
                li, hi_ = boot_ci_img(v)
                lp, hp = boot_ci(v, pid_c)
                res["pairs"].append({"model": NAME[m], "target": t, "a": a, "b": b,
                                     "diff": float(v.mean()),
                                     "ci_image": [li, hi_], "ci_patient": [lp, hp],
                                     "excludes_zero_image": bool(li > 0 or hi_ < 0),
                                     "excludes_zero_patient": bool(lp > 0 or hp < 0)})
    P = res["pairs"]
    res["summary"] = {
        "n_pairs_per_model": {t: len(NEUTRAL[t]) * (len(NEUTRAL[t]) - 1) // 2 for t in TOP},
        "n_comparisons": len(P),
        "excluding_zero_image": sum(p["excludes_zero_image"] for p in P),
        "excluding_zero_patient": sum(p["excludes_zero_patient"] for p in P),
        "max_abs_diff": max(abs(p["diff"]) for p in P),
        "exceptions_patient": [f"{p['model']} {p['target']} {p['a']}-{p['b']}"
                               for p in P if not p["excludes_zero_patient"]]}
    print("\n%d comparisons | excluding 0: %d (image) %d (patient) | max |diff| %.2f"
          % (len(P), res["summary"]["excluding_zero_image"],
             res["summary"]["excluding_zero_patient"], res["summary"]["max_abs_diff"]), flush=True)
    print("exceptions (patient):", res["summary"]["exceptions_patient"], flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved ->", OUT, "(%.0fs)" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
