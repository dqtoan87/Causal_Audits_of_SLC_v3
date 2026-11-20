#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C69: the route test under other clip levels, and a check of the largest log-odds  (exploratory, not pre-specified)

The pre-specified script c62_route_test.py is imported, never modified. The cycle test, the stored
per-image responses and the scorer are reused from c67_route_sensitivity.py.

Part A, the clip level. The plan reads the clipped logit (5) at eta = 1e-6. Clipping the probability at
eta is the same as clipping the head's log-odds z = z_mal - z_ben at +-log((1 - eta)/eta), so every clip
level can be read from z. We rebuild the registered construction, score it as the plan does (one bf16
batch per image) and in full precision, keep z, and for eta in ETAS recompute the 48 differences, their
simultaneous band and the cell decisions on each run's primary set. The share of predictions at the bound
is reported for every control pair of every cell.

Part B, the largest unclipped log-odds: where they occur, whether full precision gives the same values,
and how large the pooled backbone feature is there, against the unedited images.

Part C, the score floor read in full precision instead of bf16, with the plan's responses.

Part D, the probability scale, descriptive only: a difference of mean probability changes is a different
estimand, with no margin fixed in advance. We report which differences have a simultaneous interval that
excludes zero.

Out -> results/c69_route_clip_sensitivity.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c62_route_test as c62           # noqa: E402  (pre-specified; imported only)
import c67_route_sensitivity as c67    # noqa: E402

RES = c62.RES
OUT = os.path.join(RES, "c69_route_clip_sensitivity.json")
CR = c62.CRIT
TKS = c67.TKS
ETAS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
KEY_CELLS = ["erm|contrast@2", "causal|contrast@2", "slim|color_var@2", "slim|contrast@0"]


def bound(eta):
    return float(np.log((1 - eta) / eta))


def build_stacks(cons):
    """Rebuild the registered images exactly as c67 does (single-threaded construction)."""
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
    cols = ["unedited"] + ["%s|%s" % (tk, c) for tk in TKS for c in ["edit"] + c62.CONTROLS] + \
           ["placebo|" + p for p in c62.c10.PLACEBOS]
    return stacks, cols


@torch.no_grad()
def score_feat(model, batch, fp32):
    """Probability, log-odds and the norm of the pooled backbone feature, for one batch."""
    xn = c62.c3m.p2._to_norm(batch).to(c62.DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(c62.DEVICE == "cuda" and not fp32)):
        f = model.backbone(xn)
        z = (model.head(f) if hasattr(model, "head") else model.disease_head(model.proj_c(f))).float()
    p = torch.softmax(z, 1)[:, 1].cpu().numpy()
    return p, (z[:, 1] - z[:, 0]).cpu().numpy(), f.float().norm(dim=1).cpu().numpy()


def load_model(m, ckpt):
    model = c62.p1.build_model("erm" if m == "erm" else "slim", c62.p1.CFG["z_dim"]).to(c62.DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=c62.DEVICE)); model.eval()
    return model


def score_all(stacks, ckpts):
    """ckpts: {name: (arch, path)}. Returns P, Z, F keyed by (name, numerics)."""
    P, Z, F = {}, {}, {}
    for name, (arch, path) in ckpts.items():
        model = load_model(arch, path)
        for v, fp32 in (("bf16", False), ("fp32", True)):
            out = [score_feat(model, s, fp32) for s in stacks]
            P[(name, v)] = np.stack([o[0] for o in out]); Z[(name, v)] = np.stack([o[1] for o in out])
            F[(name, v)] = np.stack([o[2] for o in out])
        del model
        if c62.DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("  scored %s" % name, flush=True)
    return P, Z, F


def comp_summary(r, cell):
    m, tk = cell.split("|")
    return {"%s-%s" % (a, b): {"est": r["components"]["%s|%s|D_%s-%s" % (m, tk, a, b)]["est"],
                               "sim": r["components"]["%s|%s|D_%s-%s" % (m, tk, a, b)]["sim"],
                               "decision": r["components"]["%s|%s|D_%s-%s" % (m, tk, a, b)]["decision"]}
            for a, b, _ in c62.PAIRS}


def pair_saturation(Zr, cols, masks, L):
    """Share of the analysed images on which the prediction under a or under b sits at the bound."""
    col = {c: k for k, c in enumerate(cols)}
    out = {}
    for tk in TKS:
        for m in c62.MODELS:
            msk = masks[(m, tk)]
            at = np.abs(Zr[m]) >= L
            cell = {}
            for c in ["edit"] + c62.CONTROLS:
                cell[c] = float(at[msk, col["%s|%s" % (tk, c)]].mean())
            for a, b, _ in c62.PAIRS:
                ia, ib = col["%s|%s" % (tk, a)], col["%s|%s" % (tk, b)]
                cell["%s-%s" % (a, b)] = float((at[msk, ia] | at[msk, ib]).mean())
            out["%s|%s" % (m, tk)] = cell
    return out


def run_eta(Zr, cols, masks, pids):
    res = {}
    for eta in ETAS:
        L = bound(eta)
        DL = {m: np.clip(Zr[m], -L, L) - np.clip(Zr[m][:, :1], -L, L) for m in c62.MODELS}
        r = c67.cycle(DL, cols, masks, pids)
        res["%g" % eta] = {"bound_logit": L, "crit": r["crit"], "summary": r["summary"], "cells": r["cells"],
                           "key": {c: comp_summary(r, c) for c in KEY_CELLS},
                           "saturation_pairs": pair_saturation(Zr, cols, masks, L),
                           "components": r["components"]}
        dis = [c for c, v in r["cells"].items() if v != "indeterminate"]
        print("   eta %-6g bound %5.2f crit %.3f %s  %s" % (eta, L, r["crit"], r["summary"],
                                                          {c: r["cells"][c] for c in dis}), flush=True)
    return res


def prob_scale(P, cols, masks, pids):
    """Differences of mean probability changes, simultaneous band, descriptive."""
    col = {c: k for k, c in enumerate(cols)}
    comps, vals, msk = [], [], []
    for tk in TKS:
        for m in c62.MODELS:
            for a, b, _ in c62.PAIRS:
                comps.append("%s|%s|D_%s-%s" % (m, tk, a, b))
                vals.append(P[m][:, col["%s|%s" % (tk, b)]] - P[m][:, col["%s|%s" % (tk, a)]])
                msk.append(masks[(m, tk)])
    vals, msk = np.stack(vals), np.stack(msk)
    est, draws = c62.boot(vals, msk, pids, CR["boot"], CR["boot_seed"])
    se = np.nanstd(draws, 0, ddof=1)
    Zs = np.abs(draws - est) / np.maximum(se, 1e-12)
    crit = float(np.nanpercentile(np.nanmax(Zs, 1), CR["level"]))
    out = {k: {"est": float(est[j]), "sim": [float(est[j] - crit * se[j]), float(est[j] + crit * se[j])]}
           for j, k in enumerate(comps)}
    excl = [k for k, v in out.items() if v["sim"][0] > 0 or v["sim"][1] < 0]
    return {"crit": crit, "components": out, "excluding_zero": excl, "n_excluding_zero": len(excl)}


def main():
    t0 = time.time()
    cons = json.load(open(c62.out_path("construction", False)))
    pids = np.asarray(cons["patient_ids"])
    g1 = c67.g1_masks(cons)
    r1, r2 = c67.load_run(""), c67.load_run("_inv_retrained")
    stacks, cols = build_stacks(cons)
    assert cols == r2["cols"] == r1["cols"]
    ck = dict(c62.audited_models.CKPTS)
    ckpts = {"erm": ("erm", ck["erm"]), "slim": ("slim", ck["slim"]), "causal": ("causal", ck["causal"]),
             "causal_ep1": ("causal", os.path.join(os.path.dirname(HERE), "checkpoints", "archived_b3inv_ep1",
                                                   "dj_causal_b3.pt"))}
    P, Z, F = score_all(stacks, ckpts)
    chk = {"second_run": {m: float(np.abs(P[(m, "bf16")] - r2["prob"][m]).max()) for m in c62.MODELS},
           "first_run_inv_ep1": float(np.abs(P[("causal_ep1", "bf16")] - r1["prob"]["causal"]).max())}
    print("bf16 probabilities vs stored, max |diff|:", chk, flush=True)

    runs = {"second_run": {"erm": "erm", "slim": "slim", "causal": "causal"},
            "first_run": {"erm": "erm", "slim": "slim", "causal": "causal_ep1"}}
    stored = {"second_run": r2, "first_run": r1}
    fl = CR["score_floor"]
    out = {"note": "exploratory, not pre-specified; c62_route_test.py imported unchanged",
           "c62_script_sha256": c62.sha256(c62.SCRIPT), "etas": ETAS, "check_bf16_vs_stored": chk, "runs": {}}
    for run, mp in runs.items():
        print("[%s]" % run, flush=True)
        common = np.all([P[(mp[m], "bf16")][:, 0] >= fl for m in c62.MODELS], 0)
        masks = {(m, tk): g1[tk] & common for m in c62.MODELS for tk in TKS}
        res = {"n_per_target": {tk: int((g1[tk] & common).sum()) for tk in TKS}}
        # the plan's own analysis, from the stored responses, as the reference
        ref = c67.cycle(stored[run]["dl"], cols, masks, pids)
        res["plan"] = {"crit": ref["crit"], "summary": ref["summary"], "cells": ref["cells"],
                       "key": {c: comp_summary(ref, c) for c in KEY_CELLS}}
        # z-based reading at the plan's eta: does it reproduce the plan's decisions?
        for v in ("bf16", "fp32"):
            Zr = {m: Z[(mp[m], v)] for m in c62.MODELS}
            print("  numerics %s" % v, flush=True)
            res["eta_" + v] = run_eta(Zr, cols, masks, pids)
        z6 = res["eta_bf16"]["1e-06"]
        res["z_vs_plan_at_1e-6"] = {
            "cells_equal": z6["cells"] == ref["cells"],
            "max_abs_diff_est": float(max(abs(z6["components"][k]["est"] - ref["components"][k]["est"])
                                          for k in ref["components"]))}
        # Part C: the floor read in full precision, plan responses
        common32 = np.all([P[(mp[m], "fp32")][:, 0] >= fl for m in c62.MODELS], 0)
        m32 = {(m, tk): g1[tk] & common32 for m in c62.MODELS for tk in TKS}
        rc = c67.cycle(stored[run]["dl"], cols, m32, pids)
        res["floor_fp32"] = {"images_changed": int((common32 != common).sum()),
                             "n_per_target": {tk: int((g1[tk] & common32).sum()) for tk in TKS},
                             "summary": rc["summary"], "cells": rc["cells"],
                             "key": {c: comp_summary(rc, c) for c in KEY_CELLS}}
        print("  floor in fp32: %d images change; %s" % (res["floor_fp32"]["images_changed"], rc["summary"]),
              flush=True)
        # Part D: probability scale, descriptive
        Pr = {m: P[(mp[m], "bf16")] for m in c62.MODELS}
        res["probability_scale"] = prob_scale(Pr, cols, masks, pids)
        print("  probability scale: %d of 48 simultaneous intervals exclude zero"
              % res["probability_scale"]["n_excluding_zero"], flush=True)
        out["runs"][run] = res

    # Part B: the largest log-odds on the second run's primary set
    common2 = np.all([P[(m, "bf16")][:, 0] >= fl for m in c62.MODELS], 0)
    prim_cols = [cols.index("%s|%s" % (tk, c)) for tk in TKS for c in ["edit"] + c62.CONTROLS]
    ext = {}
    for m in c62.MODELS:
        zb, zf, fb = Z[(m, "bf16")], Z[(m, "fp32")], F[(m, "bf16")]
        prim = np.zeros_like(zb, dtype=bool)
        for tk in TKS:
            for c in ["edit"] + c62.CONTROLS:
                prim[g1[tk] & common2, cols.index("%s|%s" % (tk, c))] = True
        flat = np.argsort(-np.abs(np.where(prim, zb, 0)).ravel())[:10]
        top = []
        for f in flat:
            i, j = np.unravel_index(f, zb.shape)
            top.append({"image": cons["ids"][i], "column": cols[j], "logodds_bf16": float(zb[i, j]),
                        "logodds_fp32": float(zf[i, j]), "feature_norm_bf16": float(fb[i, j]),
                        "feature_norm_unedited": float(fb[i, 0])})
        fn0 = fb[common2, 0]
        ext[m] = {"top10": top,
                  "unedited_feature_norm_median": float(np.median(fn0)),
                  "unedited_logodds_abs_max": float(np.abs(zb[common2, 0]).max()),
                  "corr_abs_logodds_feature_norm": float(np.corrcoef(np.abs(zb[prim]), fb[prim])[0, 1]),
                  "max_abs_logodds_bf16": float(np.abs(zb[prim]).max()),
                  "max_abs_logodds_fp32": float(np.abs(zf[prim]).max()),
                  "max_rel_diff_bf16_fp32_beyond_100": float(np.max(np.abs(zb[prim] - zf[prim])[np.abs(zf[prim]) > 100]
                                                                    / np.abs(zf[prim])[np.abs(zf[prim]) > 100]))
                  if (np.abs(zf[prim]) > 100).any() else None,
                  "columns_of_top10": sorted(set(t["column"] for t in top))}
        print("[B] %s max |z| bf16 %.0f fp32 %.0f; unedited feature norm median %.1f; top columns %s"
              % (m, ext[m]["max_abs_logodds_bf16"], ext[m]["max_abs_logodds_fp32"],
                 ext[m]["unedited_feature_norm_median"], ext[m]["columns_of_top10"]), flush=True)
    out["extreme_logodds_second_run"] = ext
    out["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved -> %s (%.0fs)" % (OUT, out["runtime_sec"]))


if __name__ == "__main__":
    main()
