#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C25: run the validity screen (A1)-(A3) on all 178 images of the primary sample.

Why. C12 builds the control library and measures footprint/neutrality inside the loop `for ii in mal`,
i.e. only on the 78 malignant images. The conclusion "0 of 19 controls valid" is therefore conditioned
on a label-defined subset, while the paper's primary sample is 178 unthresholded images.

The screen, (A1) footprint balance, (A2) proxy neutrality and (A3) overlap, uses only images and
edits, not model outputs, so it runs on all 178 images without a GPU. This script repeats the exact
C12 computation on the full sample and compares it with the 78-image version.

It does not recompute model-dependent quantities (CaCE, control-library range, test T): those still
need the common set of 62 images and a separate model run.

Out -> results/c25_screen_full.json
Run:  .venv/bin/python Causal_XAI_v3/code/c25_screen_full_sample.py [--subset mal|all]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c12_matched_control as c12   # noqa: E402
import audit_cohort                # noqa: E402  (fixed evaluation cohort)

c3m, c7m, c10, p2 = c12.c3m, c12.c7m, c12.c10, c12.p2
RES = os.path.join(os.path.dirname(HERE), "results")
OUT_ALL = os.path.join(RES, "c25_screen_full.json")
OUT_MAL = os.path.join(RES, "c25_screen_mal.json")
SEED = c12.SEED

# the four threshold schemes of Table B1: (name, max |SMD|, absolute |Δproxy|, proxy ratio, overlap)
SCHEMES = [("strict", 0.05, 0.05, 0.10, 0.95),
           ("main", 0.10, 0.10, 0.15, 0.95),
           ("relaxed", 0.20, 0.20, 0.30, 0.90),
           ("very relaxed", 0.30, 0.30, 0.50, 0.85)]


def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y); n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def build_design(imgs01, order):
    """Build edits and measure footprint/neutrality/overlap on the images indexed by `order`.

    Replicates the logic of C12 main() exactly; only the set of iterated images changes. No step uses the model.
    """
    t0 = time.time()
    TARGETS, MATCHED, UNMATCHED = c12.TARGETS, c12.MATCHED, c12.UNMATCHED
    COVS, LADDER_MULT, LADDER_FAM = c12.COVS, c12.LADDER_MULT, c12.LADDER_FAM
    LADDER = [f"L:{c}@{j}" for c in LADDER_FAM for j in range(len(LADDER_MULT))]
    names = list(TARGETS) + [f"u:{p}" for p in UNMATCHED] + \
            [f"{t}|{c}" for t in TARGETS for c in MATCHED] + LADDER

    FP = {n: {k: [] for k in COVS} for n in names}
    IOU = {f"{t}|{c}": [] for t in TARGETS for c in MATCHED}
    IOU.update({f"{t}|u:{p}": [] for t in TARGETS for p in UNMATCHED})
    PRX = {n: [] for n in names}
    POS = {f"{t}|{c}": [] for t in TARGETS for c in c12.CTRL_FAMILY + c12.CTRL_DOSE}
    STR = {f"{t}|{c}": [] for t in TARGETS for c in c12.CTRL_FAMILY + c12.CTRL_DOSE}
    SIG = {f"{t}|warp2_m": [] for t in TARGETS}
    HYB = {t: [] for t in TARGETS}
    done = 0

    for pos_i, ii in enumerate(order):
        x = imgs01[ii]
        w, skin, lm = c3m.soft_mask(x)
        base = c7m.proxy_fixed(x, skin)
        donor = imgs01[order[(pos_i + 1) % len(order)]]
        fams = c12.make_families(x, w, skin, SEED + int(ii), donor=donor)
        d = {}
        for t in TARGETS:
            cname, k = t.split("@")
            d[t] = c3m.make_edit(x, cname, c3m.DOSES[int(k)], w, skin, lm)
        for p in UNMATCHED:
            d[f"u:{p}"] = c10.PLACEBOS[p](x, w, skin, np.random.RandomState(SEED + int(ii) + 7))
        for t in TARGETS:                                        # identical to C12 main()
            tgt_mag = float(c12.dpix(x, d[t]).mean())
            for c, e in c12.struct_controls(x, d[t] - x, w, SEED + int(ii)).items():
                d[f"{t}|{c}"] = e
            tgt_hf = c12.footprint(x, d[t], w)["hf_change"]
            for c in c12.CTRL_FAMILY:
                if c == "hybrid_m":
                    e, bb, th, ok, _ = c12.match_hybrid(fams["_hybrid"], x, w, tgt_mag, tgt_hf)
                    s_ = th; HYB[f"{t}"].append(bb)
                elif c == "warp2_m":
                    e, s_, ok, sg = c12.match_2knob(fams, x, tgt_mag, tgt_hf, w)
                    SIG[f"{t}|warp2_m"].append(int(sg))
                else:
                    fn, lo, hi = fams[c]
                    e, s_, m_, ok = c12.match_strength(fn, lo, hi, x, tgt_mag)
                d[f"{t}|{c}"] = e
                POS[f"{t}|{c}"].append(bool(ok)); STR[f"{t}|{c}"].append(float(s_))
        for t in TARGETS:                                        # opposite dose, bisection on |g'|
            cname, k = t.split("@")
            e, gp, mm_, ok = c12.match_strength(
                lambda gp: c3m.make_edit(x, cname, -gp, w, skin, lm), 0.0, 1.0, x,
                float(c12.dpix(x, d[t]).mean()))
            d[f"{t}|invdose"] = e
            POS[f"{t}|invdose"].append(bool(ok)); STR[f"{t}|invdose"].append(float(gp))
        ref_mag = float(c12.dpix(x, d["contrast@2"]).mean())
        for c in LADDER_FAM:
            fn, lo, hi = fams[c]
            for j, mult in enumerate(LADDER_MULT):
                d[f"L:{c}@{j}"] = c12.match_strength(fn, lo, hi, x, ref_mag * mult)[0]
        for n in names:
            fp = c12.footprint(x, d[n], w)
            for k in COVS:
                FP[n][k].append(fp[k])
            PRX[n].append(c12.rel_proxy_shift(base, c7m.proxy_fixed(d[n], skin)))
        for t in TARGETS:
            for c in MATCHED:
                IOU[f"{t}|{c}"].append(c12.support_iou(x, d[t], d[f"{t}|{c}"]))
            for p in UNMATCHED:
                IOU[f"{t}|u:{p}"].append(c12.support_iou(x, d[t], d[f"u:{p}"]))
        done += 1
        if done % 20 == 0:
            print(f"   ... {done}/{len(order)} images ({time.time()-t0:.0f}s)", flush=True)

    design = {}
    for t in TARGETS:
        design[t] = {}
        prx_concept = float(np.mean(PRX[t]))
        for c in MATCHED + [f"u:{p}" for p in UNMATCHED]:
            key = f"{t}|{c}" if c in MATCHED else f"u:{c[2:]}"
            sm = {k: c12.smd(FP[key][k], FP[t][k]) for k in COVS}
            design[t][c] = {
                "smd": {k: float(v) for k, v in sm.items()},
                "max_abs_smd": float(max(abs(v) for v in sm.values())),
                "proxy_shift": float(np.mean(PRX[key])),
                "proxy_ratio": float(np.mean(PRX[key]) / max(prx_concept, 1e-6)),
                "positivity": float(np.mean(POS[f"{t}|{c}"])) if f"{t}|{c}" in POS else 1.0,
            }
        design[t]["_concept"] = {"footprint": {k: float(np.mean(FP[t][k])) for k in COVS},
                                 "proxy_shift": prx_concept}
        design[t]["_concept_proxy_shift"] = prx_concept
    # stash the per-image footprint and proxy arrays so that a later script can form
    # all-pairs comparisons without repeating the loop's own conventions
    build_design.FP, build_design.PRX, build_design.STR = FP, PRX, STR
    return design, len(order)


def screen(design, tol_smd, tol_abs, tol_ratio, tol_pos):
    """Count the controls that satisfy all three conditions, per concept target."""
    out = {}
    for t, cells in design.items():
        n_bal = n_neu = n_valid = 0
        for c, v in cells.items():
            if c.startswith("_"):
                continue
            bal = v["max_abs_smd"] <= tol_smd
            neu = (v["proxy_shift"] <= tol_abs) and (abs(v["proxy_ratio"]) <= tol_ratio)
            ov = v["positivity"] >= tol_pos
            n_bal += bal; n_neu += neu; n_valid += (bal and neu and ov)
        out[t] = {"balanced": int(n_bal), "neutral": int(n_neu), "valid": int(n_valid)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["mal", "all"], default="all")
    a = ap.parse_args()
    t0 = time.time()
    print(f"##### C25 screen on {a.subset} {time.strftime('%H:%M:%S')} #####", flush=True)

    # Read the frozen cohort, as c12 does. Rebuilding the partition here would give a different
    # 183-image set from the one every other analysis uses, so the note "on 183 evaluation images" would no longer hold.
    imgs01, labels, _pid, _ids = audit_cohort.load()
    labels = np.array(labels)
    order = np.where(labels == 1)[0] if a.subset == "mal" else np.arange(len(labels))
    print(f"{len(order)} images ({int((labels[order]==1).sum())} malignant, "
          f"{int((labels[order]==0).sum())} benign)", flush=True)

    design, n = build_design(imgs01, order)
    res = {"subset": a.subset, "n_images": n, "design": design, "schemes": {}}

    W = 92
    print("=" * W)
    print(f"VALIDITY SCREEN on {n} images")
    print(f"{'scheme':14s}{'|SMD|≤':>8s}{'proxy≤':>9s}{'ratio≤':>8s}{'ovl≥':>7s}"
          f"{'pass balance':>15s}{'pass neutral':>15s}{'VALID':>8s}")
    for nm, ts, ta, tr, tp in SCHEMES:
        sc = screen(design, ts, ta, tr, tp)
        bal = [sc[t]["balanced"] for t in sc]; neu = [sc[t]["neutral"] for t in sc]
        val = [sc[t]["valid"] for t in sc]
        res["schemes"][nm] = {"tol": [ts, ta, tr, tp], "per_target": sc,
                              "valid_total": int(sum(val)), "valid_max": int(max(val))}
        print(f"{nm:14s}{ts:8.2f}{ta:9.2f}{tr:8.2f}{tp:7.2f}"
              f"{('%d to %d' % (min(bal), max(bal))):>15s}"
              f"{('%d to %d' % (min(neu), max(neu))):>15s}"
              f"{('%d' % max(val)):>8s}")
    print("-" * W)
    print("TRADE-OFF between balance and neutrality (Spearman, per target)")
    for t in c12.TARGETS:
        cells = {c: v for c, v in design[t].items() if not c.startswith("_")}
        xs = [v["max_abs_smd"] for v in cells.values()]
        ys = [abs(v["proxy_ratio"]) for v in cells.values()]
        rho = spearman(xs, ys)
        res.setdefault("spearman", {})[t] = float(rho)
        # Fisher-z confidence interval for the rank rho. With n=19 it is very wide, which is why
        # the paper reads the sign of the monotone trend, not the value of the coefficient.
        n_c = len(cells)
        if n_c > 4 and abs(rho) < 1:
            z, se = np.arctanh(rho), 1.06 / np.sqrt(n_c - 3)
            ci = [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]
        else:
            ci = None
        res.setdefault("spearman_ci", {})[t] = ci
        res.setdefault("spearman_n", {})[t] = int(n_c)
        print(f"   {t:14s} n={n_c:2d}  rho = {rho:+.2f}  95% CI "
              + (f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "n/a"))
    out_path = OUT_ALL if a.subset == "all" else OUT_MAL
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print("=" * W)
    print(f"saved -> {out_path}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
