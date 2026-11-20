#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C86: the cycle test of the first run under the four inference protocols of c83.  (exploratory)

c83 rescored the second run's images with batch context and precision varied separately (the plan's batch
of 25 images or one image per batch, bf16 or fp32). This script applies the same four protocols to the
first run's configuration: ERM, Slim and the epoch-1 Inv checkpoint that the first run used
(checkpoints/archived_b3inv_ep1). The analysed images are the first run's: G1 matches and the plan's
score floor read from the stored first-run probabilities, as in the plan-locked analysis.

Reported per protocol: the critical value, the cells that disagree on that protocol's own simultaneous
band at the 0.5-logit margin, and the estimate and simultaneous lower end of the differences that decide
the first run's three plan-locked disagreements. The plan's protocol must reproduce the stored scores.

Out -> results/c86_batch_context_first_run.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c62_route_test as c62             # noqa: E402  (pre-specified; imported only)
import c67_route_sensitivity as c67      # noqa: E402
import c69_route_clip_sensitivity as c69  # noqa: E402

RES = c62.RES
OUT = os.path.join(RES, "c86_batch_context_first_run.json")
CR = c62.CRIT
TKS = c67.TKS
CK_EP1 = os.path.join(os.path.dirname(HERE), "checkpoints", "archived_b3inv_ep1", "dj_causal_b3.pt")
PROTOCOLS = [("stack_bf16", False, False), ("stack_fp32", False, True),
             ("single_bf16", True, False), ("single_fp32", True, True)]
DECISIVE = ["erm|contrast@2|D_rad4-rad16", "slim|color_var@2|D_rad4-rad16", "slim|contrast@0|D_rad4-rad16"]


def main():
    t0 = time.time()
    cons = json.load(open(c62.out_path("construction", False)))
    pids = np.asarray(cons["patient_ids"])
    stacks, cols = c69.build_stacks(cons)
    n, k = len(stacks), stacks[0].shape[0]
    r1 = c67.load_run("")
    assert r1["cols"] == cols and r1["ids"] == cons["ids"]
    print("rebuilt %d images x %d (%.0fs)" % (n, k, time.time() - t0), flush=True)
    ckpts = {"erm": c62.audited_models.CKPTS["erm"], "slim": c62.audited_models.CKPTS["slim"], "causal": CK_EP1}
    P = {}
    for m in c62.MODELS:
        model = c69.load_model(m, ckpts[m])
        for v, single, fp32 in PROTOCOLS:
            if single:
                P[(m, v)] = np.stack([np.concatenate([c69.score_feat(model, s[j:j + 1], fp32)[0] for j in range(k)])
                                      for s in stacks])
            else:
                P[(m, v)] = np.stack([c69.score_feat(model, s, fp32)[0] for s in stacks])
        del model
        if c62.DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("  scored %s (%.0fs)" % (m, time.time() - t0), flush=True)
    chk = {m: float(np.abs(P[(m, "stack_bf16")] - r1["prob"][m]).max()) for m in c62.MODELS}
    print("plan protocol vs stored first-run scores, max |dp|:", chk, flush=True)

    g1 = c67.g1_masks(cons)
    common1 = np.all([r1["prob"][m][:, 0] >= CR["score_floor"] for m in c62.MODELS], 0)
    masks = {(m, tk): g1[tk] & common1 for m in c62.MODELS for tk in TKS}
    out = {"note": "exploratory; first run's configuration (ERM, Slim, epoch-1 Inv); clipped logit (5); "
                   "first run's analysed images; c62_route_test.py imported unchanged",
           "c62_script_sha256": c62.sha256(c62.SCRIPT),
           "check_stack_bf16_vs_stored_first_run_max_abs_prob": chk,
           "n_per_target": {tk: int((g1[tk] & common1).sum()) for tk in TKS}, "cycle": {}}
    for v, _, _ in PROTOCOLS:
        DL = {m: c62.c3m.logit(P[(m, v)]) - c62.c3m.logit(P[(m, v)][:, :1]) for m in c62.MODELS}
        r = c67.cycle(DL, cols, masks, pids)
        out["cycle"][v] = {"crit": r["crit"], "summary": r["summary"],
                           "disagreeing_cells": sorted(c for c, x in r["cells"].items() if x == "routes disagree"),
                           "decisive": {c: {"est": r["components"][c]["est"], "sim_lower": r["components"][c]["sim"][0],
                                            "decision": r["components"][c]["decision"]} for c in DECISIVE}}
        print("  cycle %-12s crit %.3f %s %s" % (v, r["crit"], r["summary"], out["cycle"][v]["disagreeing_cells"]),
              flush=True)
        for c in DECISIVE:
            d = out["cycle"][v]["decisive"][c]
            print("      %-32s %+.3f [LB %.3f] %s" % (c, d["est"], d["sim_lower"], d["decision"]), flush=True)
    out["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
