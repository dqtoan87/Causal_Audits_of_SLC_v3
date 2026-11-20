#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C67: sensitivity of the route test to the score floor and to the numerics  (exploratory, not pre-specified)

The pre-specified script c62_route_test.py is imported, never modified. Its bootstrap and decision
functions are reused, so the second run's primary analysis is reproduced exactly before any variant
is computed.

Part A, the score floor (no new model output). The plan keeps an image only if all three
EfficientNet-B3 models score its unedited version at 0.1 or above. Retraining the Inv checkpoint
therefore changed the analysed images of ERM and Slim as well. Three post hoc variants:
  per_model   each model's cells use the images that model itself scores at 0.1 or above;
  run1_set    the second run's responses on the images the first run analysed;
  common_set  both runs on the images that pass the floor under both Inv checkpoints.
They read the per-image responses stored by c63_route_exploratory.py for each run.

Part B, the numerics (EfficientNet-B3 trio, held-out images). The plan scores each image's 25 edited
and unedited versions as one bf16 batch and reads the clipped logit (5). We rebuild the same images
from the registered construction and score them
  bf16_stack   as the plan does (must reproduce the stored second-run responses),
  fp32_stack   in full precision,
  bf16_flat32  in bf16, in flat batches of 32 that mix images,
  raw_logodds  the bf16 head's log-odds z_mal - z_ben, without clipping,
and recompute the 48 differences, their band and the cell decisions on the second run's primary set.
It also reports how many predictions sit at the clipping bound, by column, and how far the unclipped
log-odds go beyond it.

Part C repeats Part B for the first run's configuration, with the epoch-1 Inv checkpoint that the
first run used (checkpoints/archived_b3inv_ep1), on the first run's primary set.

Out -> results/c67_route_sensitivity.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c62_route_test as c62         # noqa: E402  (pre-specified; imported only)

RES = c62.RES
OUT = os.path.join(RES, "c67_route_sensitivity.json")
CR = c62.CRIT
TKS = ["%s@%d" % t for t in c62.TARGETS]
ETA = 1e-6


def load_run(suffix):
    d = np.load(os.path.join(RES, "c63_route_scores%s.npz" % suffix), allow_pickle=True)
    return {"cols": [str(c) for c in d["columns"]], "ids": [str(i) for i in d["isic_ids"]],
            "prob": {m: d["prob_" + m] for m in c62.MODELS}, "dl": {m: d["dlogit_" + m] for m in c62.MODELS}}


def g1_masks(cons):
    n = len(cons["ids"])
    return {tk: np.array([all(cons["images"][i]["targets"][tk]["controls"][c]["g1"] for c in c62.CONTROLS)
                          for i in range(n)]) for tk in TKS}


def cycle(DL, cols, masks, pids):
    """The plan's cycle test on the 48 differences. masks[(m, tk)] -> boolean image mask."""
    col = {c: k for k, c in enumerate(cols)}
    comps, vals, msk = [], [], []
    for tk in TKS:
        for m in c62.MODELS:
            for a, b, _ in c62.PAIRS:
                comps.append((m, tk, "%s-%s" % (a, b)))
                vals.append(DL[m][:, col["%s|%s" % (tk, b)]] - DL[m][:, col["%s|%s" % (tk, a)]])
                msk.append(masks[(m, tk)])
    vals, msk = np.stack(vals), np.stack(msk)
    est, draws = c62.boot(vals, msk, pids, CR["boot"], CR["boot_seed"])
    se = np.nanstd(draws, 0, ddof=1)
    Z = np.abs(draws - est) / np.maximum(se, 1e-12)
    crit = float(np.nanpercentile(np.nanmax(Z, 1), CR["level"]))
    eps = CR["margin_logit"]
    res = {"crit": crit, "components": {}, "cells": {}, "n": {}}
    for k, (m, tk, p) in enumerate(comps):
        sl, sh = est[k] - crit * se[k], est[k] + crit * se[k]
        res["components"]["%s|%s|D_%s" % (m, tk, p)] = {"est": float(est[k]), "sim": [float(sl), float(sh)],
                                                       "decision": c62.decide(sl, sh, eps)}
    for tk in TKS:
        for m in c62.MODELS:
            ds = [res["components"]["%s|%s|D_%s-%s" % (m, tk, a, b)]["decision"] for a, b, _ in c62.PAIRS]
            cls = "routes disagree" if "exceeds" in ds else ("routes agree" if all(v == "within" for v in ds)
                                                              else "indeterminate")
            res["cells"]["%s|%s" % (m, tk)] = cls
            res["n"]["%s|%s" % (m, tk)] = [int(masks[(m, tk)].sum()), int(len(set(pids[masks[(m, tk)]])))]
    cl = list(res["cells"].values())
    res["summary"] = {k: cl.count(k) for k in ("routes disagree", "routes agree", "indeterminate")}
    return res


def part_a(cons, pids):
    g1 = g1_masks(cons)
    r1, r2 = load_run(""), load_run("_inv_retrained")
    assert r1["ids"] == cons["ids"] == r2["ids"]
    fl = CR["score_floor"]
    common1 = np.all([r1["prob"][m][:, 0] >= fl for m in c62.MODELS], 0)
    common2 = np.all([r2["prob"][m][:, 0] >= fl for m in c62.MODELS], 0)
    out = {}
    # the second run's primary analysis, reproduced
    rep = cycle(r2["dl"], r2["cols"], {(m, tk): g1[tk] & common2 for m in c62.MODELS for tk in TKS}, pids)
    ev = json.load(open(os.path.join(RES, "c62_evaluation_inv_retrained.json")))["primary"]
    dmax = max(abs(rep["components"][k]["sim"][j] - ev["components"][k]["sim"][j])
               for k in rep["components"] for j in (0, 1))
    assert dmax < 1e-9 and abs(rep["crit"] - ev["crit_simultaneous"]) < 1e-12, dmax
    out["reproduced_second_run"] = {"max_abs_diff_sim_bounds": dmax, "crit": rep["crit"], "cells": rep["cells"]}
    rep1 = cycle(r1["dl"], r1["cols"], {(m, tk): g1[tk] & common1 for m in c62.MODELS for tk in TKS}, pids)
    out["reproduced_first_run"] = {"crit": rep1["crit"], "cells": rep1["cells"], "summary": rep1["summary"]}
    var = {
        "per_model": (r2, {(m, tk): g1[tk] & (r2["prob"][m][:, 0] >= fl) for m in c62.MODELS for tk in TKS}),
        "run1_set": (r2, {(m, tk): g1[tk] & common1 for m in c62.MODELS for tk in TKS}),
        "common_set_second_run": (r2, {(m, tk): g1[tk] & common1 & common2 for m in c62.MODELS for tk in TKS}),
        "common_set_first_run": (r1, {(m, tk): g1[tk] & common1 & common2 for m in c62.MODELS for tk in TKS}),
        "per_model_first_run": (r1, {(m, tk): g1[tk] & (r1["prob"][m][:, 0] >= fl) for m in c62.MODELS for tk in TKS}),
    }
    for name, (run, masks) in var.items():
        r = cycle(run["dl"], run["cols"], masks, pids)
        out[name] = r
        print("[A] %-22s crit %.3f %s" % (name, r["crit"], r["summary"]), flush=True)
        for c, v in r["cells"].items():
            if v != "indeterminate":
                print("      %-18s %s  n=%s" % (c, v, r["n"][c]), flush=True)
    out["n_common_first"] = int(common1.sum()); out["n_common_second"] = int(common2.sum())
    out["n_common_both"] = int((common1 & common2).sum())
    return out, g1, common2


@torch.no_grad()
def score(model, batch, fp32):
    xn = c62.c3m.p2._to_norm(batch).to(c62.DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(c62.DEVICE == "cuda" and not fp32)):
        z = model(xn).float()
    p = torch.softmax(z, 1)[:, 1].cpu().numpy()
    return p, (z[:, 1] - z[:, 0]).cpu().numpy()


def part_b(cons, pids, g1, common2):
    t0 = time.time()
    ids = cons["ids"]
    torch.set_num_threads(1)                 # rebuild exactly as the single-threaded construction did
    imgs = c62.audit_cohort._read(ids)
    stacks = []
    for i, x in enumerate(imgs):
        items = [x]; rec = cons["images"][i]["targets"]
        for (concept, dk), tk in zip(c62.TARGETS, TKS):
            cx = c62.context(x, concept, dk)
            items.append(cx["e"])
            built = c62.build_controls(x, cx, i, params={c: rec[tk]["controls"][c]["param"] for c in c62.CONTROLS})
            for c in c62.CONTROLS:
                items.append(built[c][0])
        w, skin, _ = c62.c3m.soft_mask(x)
        for fn in c62.c10.PLACEBOS.values():
            items.append(fn(x, w, skin, np.random.RandomState(c62.SEED_PLACEBO + i + 7)))
        stacks.append(torch.stack(items))
    torch.set_num_threads(8)
    cols = ["unedited"] + ["%s|%s" % (tk, c) for tk in TKS for c in ["edit"] + c62.CONTROLS] + \
           ["placebo|" + p for p in c62.c10.PLACEBOS]
    n, k = len(stacks), stacks[0].shape[0]
    flat = torch.cat(stacks)                  # image-major order
    r2 = load_run("_inv_retrained")
    P, Z = {}, {}
    for m in c62.MODELS:
        model = c62.p1.build_model("erm" if m == "erm" else "slim", c62.p1.CFG["z_dim"]).to(c62.DEVICE)
        model.load_state_dict(torch.load(c62.audited_models.CKPTS[m], map_location=c62.DEVICE)); model.eval()
        for v, fp32 in (("bf16_stack", False), ("fp32_stack", True)):
            out = [score(model, s, fp32) for s in stacks]
            P[(m, v)] = np.stack([o[0] for o in out]); Z[(m, v)] = np.stack([o[1] for o in out])
        pf, zf = [], []
        for j in range(0, flat.shape[0], 32):
            p, z = score(model, flat[j:j + 32], False); pf.append(p); zf.append(z)
        P[(m, "bf16_flat32")] = np.concatenate(pf).reshape(n, k); Z[(m, "bf16_flat32")] = np.concatenate(zf).reshape(n, k)
        del model
        if c62.DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("[B] %s scored (%.0fs)" % (m, time.time() - t0), flush=True)
    chk = {m: float(np.abs(P[(m, "bf16_stack")] - r2["prob"][m]).max()) for m in c62.MODELS}
    print("[B] bf16_stack vs stored second-run probabilities, max |diff|:", chk, flush=True)
    masks = {(m, tk): g1[tk] & common2 for m in c62.MODELS for tk in TKS}
    DLs = {}
    for v in ("bf16_stack", "fp32_stack", "bf16_flat32"):
        DLs[v] = {m: c62.c3m.logit(P[(m, v)]) - c62.c3m.logit(P[(m, v)][:, :1]) for m in c62.MODELS}
    DLs["raw_logodds"] = {m: Z[(m, "bf16_stack")] - Z[(m, "bf16_stack")][:, :1] for m in c62.MODELS}
    res = {"check_bf16_stack_vs_stored": chk, "variants": {}}
    ref = cycle(DLs["bf16_stack"], cols, masks, pids)
    for v, DL in DLs.items():
        r = cycle(DL, cols, masks, pids)
        dd = [abs(r["components"][c]["est"] - ref["components"][c]["est"]) for c in ref["components"]]
        db = [abs(r["components"][c]["sim"][j] - ref["components"][c]["sim"][j]) for c in ref["components"] for j in (0, 1)]
        changed = [c for c in ref["cells"] if r["cells"][c] != ref["cells"][c]]
        fl = {m: int(((P[(m, v)][:, 0] >= CR["score_floor"]) != (P[(m, "bf16_stack")][:, 0] >= CR["score_floor"])).sum())
              for m in c62.MODELS} if v != "raw_logodds" else None
        res["variants"][v] = {"crit": r["crit"], "summary": r["summary"], "cells": r["cells"],
                              "max_abs_change_D": float(max(dd)), "median_abs_change_D": float(np.median(dd)),
                              "max_abs_change_sim_bound": float(max(db)), "cells_changed": changed,
                              "images_crossing_floor": fl,
                              "components": {c: r["components"][c] for c in r["components"]}}
        print("[B] %-12s crit %.3f %s  max|dD| %.3f  max|d bound| %.3f  changed %s  floor crossings %s"
              % (v, r["crit"], r["summary"], max(dd), max(db), changed, fl), flush=True)
    # saturation at the clipping bound, second run, primary set
    sat = {}
    for m in c62.MODELS:
        p = P[(m, "bf16_stack")]
        at = (p <= ETA) | (p >= 1 - ETA)
        sat[m] = {}
        for tk in TKS:
            msk = masks[(m, tk)]
            sat[m][tk] = {c: float(at[msk, cols.index("%s|%s" % (tk, c)) if c != "unedited" else 0].mean())
                          for c in ["unedited", "edit"] + c62.CONTROLS}
        sat[m]["overall_primary"] = float(np.mean([at[masks[(m, tk)]][:, [cols.index("%s|%s" % (tk, c))
                                                     for c in ["edit"] + c62.CONTROLS]].mean() for tk in TKS]))
    res["saturation_primary"] = sat
    allsh = [v for m in sat for tk in TKS for v in sat[m][tk].values()]
    res["saturation_max_share"] = float(max(allsh))
    print("[B] largest share of predictions at the clipping bound, any model, target and column: %.3f"
          % res["saturation_max_share"], flush=True)
    # the unclipped log-odds: how far beyond the clipping bound do the predictions go?
    raw = {}
    for m in c62.MODELS:
        z = np.abs(Z[(m, "bf16_stack")])
        prim = np.zeros_like(z, dtype=bool)
        for tk in TKS:
            for c in ["edit"] + c62.CONTROLS:
                prim[masks[(m, tk)], cols.index("%s|%s" % (tk, c))] = True
        zp = z[prim]
        raw[m] = {"max_abs_logodds": float(zp.max()), "q99_abs_logodds": float(np.quantile(zp, 0.99)),
                  "share_beyond_clip": float((zp > np.log((1 - ETA) / ETA)).mean()),
                  "share_beyond_100": float((zp > 100).mean())}
    res["raw_logodds_primary"] = raw
    print("[B] unclipped log-odds on the primary set:", raw, flush=True)
    res["runtime_sec"] = round(time.time() - t0, 1)
    res["_P"], res["_cols"], res["_stacks_n"] = P, cols, n
    return res


def part_c(cons, pids, g1, P, cols):
    """The first run's configuration (ERM, Slim and the epoch-1 Inv checkpoint) under the same numerics."""
    t0 = time.time()
    ck = os.path.join(os.path.dirname(HERE), "checkpoints", "archived_b3inv_ep1", "dj_causal_b3.pt")
    ids = cons["ids"]
    torch.set_num_threads(1)
    imgs = c62.audit_cohort._read(ids)
    stacks = []
    for i, x in enumerate(imgs):
        items = [x]; rec = cons["images"][i]["targets"]
        for (concept, dk), tk in zip(c62.TARGETS, TKS):
            cx = c62.context(x, concept, dk)
            items.append(cx["e"])
            built = c62.build_controls(x, cx, i, params={c: rec[tk]["controls"][c]["param"] for c in c62.CONTROLS})
            for c in c62.CONTROLS:
                items.append(built[c][0])
        w, skin, _ = c62.c3m.soft_mask(x)
        for fn in c62.c10.PLACEBOS.values():
            items.append(fn(x, w, skin, np.random.RandomState(c62.SEED_PLACEBO + i + 7)))
        stacks.append(torch.stack(items))
    torch.set_num_threads(8)
    n, k = len(stacks), stacks[0].shape[0]
    flat = torch.cat(stacks)
    model = c62.p1.build_model("slim", c62.p1.CFG["z_dim"]).to(c62.DEVICE)
    model.load_state_dict(torch.load(ck, map_location=c62.DEVICE)); model.eval()
    Q = {}
    for v, fp32 in (("bf16_stack", False), ("fp32_stack", True)):
        Q[v] = np.stack([score(model, s, fp32)[0] for s in stacks])
    pf = [score(model, flat[j:j + 32], False)[0] for j in range(0, flat.shape[0], 32)]
    Q["bf16_flat32"] = np.concatenate(pf).reshape(n, k)
    del model
    r1 = load_run("")
    chk = float(np.abs(Q["bf16_stack"] - r1["prob"]["causal"]).max())
    print("[C] epoch-1 Inv, bf16_stack vs stored first-run probabilities, max |diff|: %.2e" % chk, flush=True)
    fl = CR["score_floor"]
    common1 = (P[("erm", "bf16_stack")][:, 0] >= fl) & (P[("slim", "bf16_stack")][:, 0] >= fl) & (Q["bf16_stack"][:, 0] >= fl)
    masks = {(m, tk): g1[tk] & common1 for m in c62.MODELS for tk in TKS}
    out = {"check_bf16_stack_vs_stored_first_run": chk, "variants": {}}
    ref = None
    for v in ("bf16_stack", "fp32_stack", "bf16_flat32"):
        PP = {"erm": P[("erm", v)], "slim": P[("slim", v)], "causal": Q[v]}
        DL = {m: c62.c3m.logit(PP[m]) - c62.c3m.logit(PP[m][:, :1]) for m in c62.MODELS}
        r = cycle(DL, cols, masks, pids)
        if ref is None:
            ref = r
        changed = [c for c in ref["cells"] if r["cells"][c] != ref["cells"][c]]
        dd = [abs(r["components"][c]["est"] - ref["components"][c]["est"]) for c in ref["components"]]
        out["variants"][v] = {"crit": r["crit"], "summary": r["summary"], "cells": r["cells"],
                              "cells_changed": changed, "max_abs_change_D": float(max(dd))}
        print("[C] %-12s crit %.3f %s  max|dD| %.3f  changed %s" % (v, r["crit"], r["summary"], max(dd), changed),
              flush=True)
    at = (Q["bf16_stack"] <= ETA) | (Q["bf16_stack"] >= 1 - ETA)
    out["saturation_inv_ep1_max_share"] = float(max(at[masks[("causal", tk)], cols.index("%s|%s" % (tk, c))].mean()
                                                    for tk in TKS for c in c62.CONTROLS))
    out["runtime_sec"] = round(time.time() - t0, 1)
    return out


def main():
    t0 = time.time()
    cons = json.load(open(c62.out_path("construction", False)))
    pids = np.asarray(cons["patient_ids"])
    a, g1, common2 = part_a(cons, pids)
    b = part_b(cons, pids, g1, common2)
    P, cols = b.pop("_P"), b.pop("_cols"); b.pop("_stacks_n")
    c = part_c(cons, pids, g1, P, cols)
    out = {"note": "exploratory, not pre-specified; c62_route_test.py imported unchanged",
           "c62_script_sha256": c62.sha256(c62.SCRIPT), "floor": a, "numerics": b, "numerics_first_run": c,
           "runtime_sec": round(time.time() - t0, 1)}
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved -> %s (%.0fs)" % (OUT, out["runtime_sec"]))


if __name__ == "__main__":
    main()
