#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C91: screen (A3) measured on the final images of every control (post hoc; image side only, no model output).

Why. In C25 the feasibility screen (A3) is the share of images on which a control reaches the concept edit's
strength, mean |Delta| within MATCH_TOL = 2%. It was recorded only for controls that run a calibration step
(the bisection reports whether it reached the target). Controls built directly from the concept edit's own
displacement (rotd, sgnd, permd) and the four uncalibrated placebos carry no calibration step, so C25 stored
no measurement for them. This script measures (A3) for every control in the same way, on the edited image
after clipping to the unit cube: the share of images with |mean|Delta_c| - mean|Delta_t|| / mean|Delta_t|
<= 2%, where t is the concept edit. It then recounts the screens of the four threshold schemes with the
measured (A3).

It rebuilds the library with C25's own build_design on the 183 audit images, which replicates C12 exactly,
and checks the rebuild against results/c25_screen_full.json before using it.

Out -> results/c91_feasibility_measured.json
Run:  .venv/bin/python code/c91_feasibility_measured.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c25_screen_full_sample as c25   # noqa: E402
import audit_cohort                     # noqa: E402

c12 = c25.c12
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c91_feasibility_measured.json")
STRUCT = list(c12.CTRL_STRUCT)


def main():
    t0 = time.time()
    print("##### C91 measured feasibility %s #####" % time.strftime("%H:%M:%S"), flush=True)
    imgs01, labels, _pid, _ids = audit_cohort.load()
    order = np.arange(len(labels))
    design, n = c25.build_design(imgs01, order)
    FP, STR = c25.build_design.FP, c25.build_design.STR
    print("rebuilt %d images (%.0fs)" % (n, time.time() - t0), flush=True)

    # 1. the rebuild must equal the stored C25 design
    stored = json.load(open(os.path.join(RES, "c25_screen_full.json")))["design"]
    dmax = 0.0
    for t, cells in design.items():
        for c, v in cells.items():
            if c.startswith("_"):
                continue
            s = stored[t][c]
            dmax = max(dmax, abs(v["max_abs_smd"] - s["max_abs_smd"]), abs(v["proxy_shift"] - s["proxy_shift"]),
                       abs(v["positivity"] - s["positivity"]))
            for k in v["smd"]:
                dmax = max(dmax, abs(v["smd"][k] - s["smd"][k]))
    print("max abs difference from stored c25 design: %.3g" % dmax, flush=True)

    # 2. measured (A3) for every control
    tol = c12.MATCH_TOL
    out = {"note": "post hoc; image side only; (A3) measured on the final, clipped image for every control",
           "match_tol": tol, "n_images": int(n), "rebuild_max_abs_difference_vs_c25": float(dmax),
           "targets": {}, "schemes": {}}
    for t in design:
        mt = np.asarray(FP[t]["mean_abs"], float)
        rows = {}
        for c, v in design[t].items():
            if c.startswith("_"):
                continue
            key = "%s|%s" % (t, c) if not c.startswith("u:") else c
            mc = np.asarray(FP[key]["mean_abs"], float)
            rel = np.abs(mc - mt) / np.maximum(mt, 1e-8)
            kind = ("calibrated" if c in c12.CTRL_FAMILY + c12.CTRL_DOSE else
                    "built from the edit's displacement" if c in STRUCT else "uncalibrated placebo")
            r = {"kind": kind, "a3_measured": float(np.mean(rel <= tol)),
                 "median_rel_dev": float(np.median(rel)), "max_rel_dev": float(np.max(rel)),
                 "a3_stored_c25": float(v["positivity"])}
            if kind == "calibrated":     # the calibration flag of C25 uses the same criterion on the same image
                r["abs_diff_from_c25_share"] = abs(r["a3_measured"] - r["a3_stored_c25"])
            rows[c] = r
        out["targets"][t] = rows

    # 3. recount the screens of the four schemes with measured (A3)
    for name, ts, ta, tr, tp in c25.SCHEMES:
        per = {}
        for t, cells in design.items():
            nb = nn = nf = nv = nb_n = 0
            for c, v in cells.items():
                if c.startswith("_"):
                    continue
                bal = v["max_abs_smd"] <= ts
                neu = v["proxy_shift"] <= ta and abs(v["proxy_ratio"]) <= tr
                fea = out["targets"][t][c]["a3_measured"] >= tp
                nb += bal; nn += neu; nf += fea; nv += (bal and neu and fea); nb_n += (bal and neu)
            per[t] = {"balanced": int(nb), "neutral": int(nn), "feasible_measured": int(nf),
                      "balanced_and_neutral": int(nb_n), "valid": int(nv)}
        out["schemes"][name] = {"tol": [ts, ta, tr, tp], "per_target": per,
                                "valid_total": int(sum(v["valid"] for v in per.values()))}
    out["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    for t, rows in out["targets"].items():
        print(t, {c: round(r["a3_measured"], 3) for c, r in rows.items()
                  if r["kind"] != "calibrated"}, flush=True)
        print("   calibrated |measured - c25|:", {c: round(r["abs_diff_from_c25_share"], 4)
                                                for c, r in rows.items() if r["kind"] == "calibrated"})
    print({k: v["valid_total"] for k, v in out["schemes"].items()})
    print("wrote", OUT, "(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
