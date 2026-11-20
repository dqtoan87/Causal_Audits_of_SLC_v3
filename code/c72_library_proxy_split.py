#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C72: the (A2) proxy screen of the library, split by concept  (no model output; exploratory)

(A2) averages the relative change of the two edited-concept proxies, contrast and color variegation.
An average can pass while one proxy moves well beyond the threshold. This script rebuilds the library on
all 183 audit images exactly as c25_screen_full_sample.py does, records the two relative proxy changes
separately for every edit, and reports for each concept target and control: the mean change of each
proxy, their maximum, and whether the control passes (A2) under the main thresholds when the average is
replaced by the larger of the two proxies, against the same denominator (the concept edit's average
change). That rule can only be stricter. The averaged values must reproduce c25_screen_full.json.

Out -> results/c72_library_proxy_split.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c12_matched_control as c12            # noqa: E402
import c25_screen_full_sample as c25         # noqa: E402
import audit_cohort                          # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c72_library_proxy_split.json")
ABS_TOL, RATIO_TOL = 0.10, 0.15              # main scheme of (A2)
CALLS = []
_orig = c12.rel_proxy_shift


def _recording(base, px):
    CALLS.append([abs(px[c] - base[c]) / (abs(base[c]) + 1e-6) for c in ("contrast", "color_var")])
    return _orig(base, px)


def main():
    t0 = time.time()
    c12.rel_proxy_shift = _recording
    imgs01, labels, _pid, _ids = audit_cohort.load()
    order = np.arange(len(labels))
    design, n = c25.build_design(imgs01, order)
    TARGETS, MATCHED, UNMATCHED = c12.TARGETS, c12.MATCHED, c12.UNMATCHED
    LADDER = ["L:%s@%d" % (c, j) for c in c12.LADDER_FAM for j in range(len(c12.LADDER_MULT))]
    names = list(TARGETS) + ["u:%s" % p for p in UNMATCHED] + \
        ["%s|%s" % (t, c) for t in TARGETS for c in MATCHED] + LADDER
    A = np.asarray(CALLS, float).reshape(n, len(names), 2)          # images x edits x (contrast, color_var)
    per = {nm: A[:, k, :] for k, nm in enumerate(names)}
    ref = json.load(open(os.path.join(RES, "c25_screen_full.json")))
    dmax = 0.0
    out = {"n_images": int(n), "thresholds": {"abs": ABS_TOL, "ratio": RATIO_TOL}, "targets": {}}
    for t in TARGETS:
        own = per[t].mean(0)                                         # concept edit: mean change of each proxy
        cells, n_mean, n_max = {}, 0, 0
        for c in MATCHED + ["u:%s" % p for p in UNMATCHED]:
            key = "%s|%s" % (t, c) if c in MATCHED else c
            m2 = per[key].mean(0)
            avg = float(m2.mean())
            dmax = max(dmax, abs(avg - ref["design"][t][c]["proxy_shift"]))
            ratio_avg = avg / max(float(own.mean()), 1e-6)
            mx = float(m2.max())
            ratio_mx = mx / max(float(own.mean()), 1e-6)       # same denominator as (A2): a stricter rule
            p_mean = bool(avg <= ABS_TOL and ratio_avg <= RATIO_TOL)
            p_max = bool(mx <= ABS_TOL and ratio_mx <= RATIO_TOL)
            n_mean += p_mean; n_max += p_max
            cells[c] = {"contrast": float(m2[0]), "color_var": float(m2[1]), "average": avg, "max": mx,
                        "pass_A2_average": p_mean, "pass_A2_max": p_max,
                        "pass_A1": bool(ref["design"][t][c]["max_abs_smd"] <= 0.10)}
        out["targets"][t] = {"concept_edit": {"contrast": float(own[0]), "color_var": float(own[1])},
                             "n_pass_A2_average": n_mean, "n_pass_A2_max": n_max,
                             "pass_A1_and_A2_max": [c for c, v in cells.items() if v["pass_A1"] and v["pass_A2_max"]],
                             "controls": cells}
        print("%-12s pass (A2): average %d, max %d; edit's own change contrast %.3f color_var %.3f"
              % (t, n_mean, n_max, own[0], own[1]), flush=True)
    out["check_average_vs_c25_max_abs_diff"] = dmax
    out["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print("average vs c25, max |diff| %.2e; saved -> %s (%.0fs)" % (dmax, OUT, out["runtime_sec"]))


if __name__ == "__main__":
    main()
