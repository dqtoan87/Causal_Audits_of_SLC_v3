#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C64: observable screens and off-target checks for the route-test controls (model-free, exploratory).

The route test (c62_route_test.py) built its controls for the magnitude-only footprint. This script asks
how the same controls, and the radial and hue controls optimised in its frontier phase, fare under the
six-descriptor footprint and the observable screens of Section III-C. It reads no model output.

Screens, computed as in c25_screen_full_sample.py, over the images on which a control is defined:
  (A1) footprint balance: |SMD| <= 0.10 on each of the six descriptors against the concept edit;
  (A2) proxy neutrality: the mean relative change of the two concept proxies (contrast and colour
       variance) is at most 10%, and at most 15% of the concept edit's own change. The script also
       reports the target and the off-target proxy separately, and a target-only reading of (A2);
  (A3) overlap: the control is defined on at least 95% of the images (G1 for the registered controls,
       a converged neutrality solve for the frontier controls);
  G2   the independent CIELAB check of the route test, for the target concept;
  link the footprint-link score max |SMD| / 0.10. It is a route bottleneck only for a control that
       passes (A2), since a route must end at a concept-neutral edit.
It also reports the off-target concept change of each control: the median relative change of the CIELAB
measure of the other concept, next to that of the concept edit.

Reads results/c62_construction.json and results/c62_frontier.json; rebuilds the frontier controls from
their recorded knobs and checks that the rebuild reproduces the recorded mismatches.
Out -> results/c64_route_screens.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import audit_cohort                  # noqa: E402
import c62_route_test as c62         # noqa: E402  (registered builders and measures)

RES = c62.RES
OUT = os.path.join(RES, "c64_route_screens.json")
TKS = ["%s@%d" % t for t in c62.TARGETS]
FINE = c62.FINE
OTHER = {"contrast": "color_var", "color_var": "contrast"}
SMD_TOL, ABS_TOL, RATIO_TOL, POS_TOL = 0.10, 0.10, 0.15, 0.95


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sd = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    return float((a.mean() - b.mean()) / sd) if sd > 1e-12 else 0.0


def rel_proxy_shift(base, m):
    """Mean relative change of the two concept proxies, as in the library screen."""
    return float(np.mean([abs(m["proxy_" + c] - base["proxy_" + c]) / (abs(base["proxy_" + c]) + 1e-6)
                          for c in ("contrast", "color_var")]))


def screens(rows, concept):
    """rows: list of (unedited, edit, control) measure dicts over the images where the control is defined."""
    sm = {k: smd([r[2][k] for r in rows], [r[1][k] for r in rows]) for k in FINE}
    mx = max(abs(v) for v in sm.values())
    ps_c = float(np.mean([rel_proxy_shift(r[0], r[2]) for r in rows]))
    ps_e = float(np.mean([rel_proxy_shift(r[0], r[1]) for r in rows]))
    ik, ok_ = c62.INDEP[concept], c62.INDEP[OTHER[concept]]
    g2 = [abs(r[2][ik] - r[0][ik]) <= c62.CRIT["independent_rel"] * abs(r[1][ik] - r[0][ik]) for r in rows]
    off_c = [abs(r[2][ok_] - r[0][ok_]) / max(abs(r[0][ok_]), 1e-9) for r in rows]
    off_e = [abs(r[1][ok_] - r[0][ok_]) / max(abs(r[0][ok_]), 1e-9) for r in rows]
    tp, op = "proxy_" + concept, "proxy_" + OTHER[concept]
    rel = lambda r, k, m: abs(r[m][k] - r[0][k]) / (abs(r[0][k]) + 1e-6)
    p_tgt_c = float(np.mean([rel(r, tp, 2) for r in rows])); p_tgt_e = float(np.mean([rel(r, tp, 1) for r in rows]))
    p_off_c = float(np.mean([rel(r, op, 2) for r in rows])); p_off_e = float(np.mean([rel(r, op, 1) for r in rows]))
    a2t = bool(p_tgt_c <= ABS_TOL and p_tgt_c / max(p_tgt_e, 1e-9) <= RATIO_TOL)
    return {"n": len(rows), "smd": sm, "max_abs_smd": mx, "footprint_link_score": mx / SMD_TOL, "A1": bool(mx <= SMD_TOL),
            "target_proxy_shift": p_tgt_c, "offtarget_proxy_shift": p_off_c,
            "edit_target_proxy_shift": p_tgt_e, "edit_offtarget_proxy_shift": p_off_e, "A2_target_only": a2t,
            "proxy_shift": ps_c, "proxy_ratio": ps_c / max(ps_e, 1e-9),
            "A2": bool(ps_c <= ABS_TOL and ps_c / max(ps_e, 1e-9) <= RATIO_TOL),
            "G2_share": float(np.mean(g2)),
            "offtarget_median_rel_control": float(np.median(off_c)),
            "offtarget_median_rel_edit": float(np.median(off_e))}


def rebuild_one(args):
    """Rebuild the frontier controls of one image from their recorded knobs (no model)."""
    i, iid, fr_rec = args
    torch.set_num_threads(1)
    x = audit_cohort._read([iid])[0]
    H, W = x.shape[1:]
    ring = c62.ring_mask(H, W)
    fs = (c62.field(H, W, c62.SEED_F4 + i, 4), c62.field(H, W, c62.SEED_F16 + i, 16),
          c62.field(H, W, c62.SEED_F48 + i, 48))
    out = {}
    for concept, dk in c62.TARGETS:
        tk = "%s@%d" % (concept, dk)
        cx = c62.context(x, concept, dk)
        p0 = c62.tproxy(x, cx); shift = c62.tproxy(cx["e"], cx) - p0; tol = c62.ROOT_TOL * abs(shift)
        mx_, me = c62.measures(x, x, cx, ring), c62.measures(x, cx["e"], cx, ring)
        out[tk] = {"unedited": mx_, "edit": me}
        for mech in ("rad", "hue"):
            kn = fr_rec["targets"][tk][mech]["knobs"]
            if mech == "rad":
                fld = c62.mix_field(fs, kn[0], kn[1])
                tau, _, conv = c62.solve_root(lambda u: c62.tproxy(c62.build_sgn(x, cx, fld, u, kn[2])[0], cx) - p0,
                                              *c62.TAU_RANGE, c62.FRONT["tau_grid"], tol)
                img = c62.build_sgn(x, cx, fld, tau, kn[2])[0]
            else:
                alpha = c62.hue_alpha(c62.mix_field(fs, kn[0], kn[1])) + kn[2]
                t, _, conv = c62.solve_root(lambda t: c62.tproxy(c62.build_dir(x, cx, t, alpha)[0], cx) - p0,
                                            *c62.T_RANGE, c62.FRONT["t_grid"], tol)
                img = c62.build_dir(x, cx, t, alpha)[0]
            m = c62.measures(x, img, cx, ring)
            rel_de = (m["lab_dE"] - me["lab_dE"]) / max(me["lab_dE"], 1e-12)
            rel_hf = (m["hf_change"] - me["hf_change"]) / max(me["hf_change"], 1e-12)
            out[tk][mech] = {"measures": m, "converged": bool(conv), "rel_dE": float(rel_de), "rel_hf": float(rel_hf)}
    return out


def main():
    t0 = time.time()
    con = json.load(open(os.path.join(RES, "c62_construction.json")))
    fr = json.load(open(os.path.join(RES, "c62_frontier.json")))
    ids = con["ids"]
    res = {"note": "model-free and exploratory; screens as in Section III-C", "registered": {}, "frontier": {},
           "tolerances": {"A1_smd": SMD_TOL, "A2_abs": ABS_TOL, "A2_ratio": RATIO_TOL, "A3_share": POS_TOL}}
    # registered controls, from the recorded construction measures
    for concept, dk in c62.TARGETS:
        tk = "%s@%d" % (concept, dk)
        res["registered"][tk] = {}
        for c in c62.CONTROLS:
            rows, n_def = [], 0
            for r in con["images"]:
                q = r["targets"][tk]
                if q["controls"][c]["g1"]:
                    rows.append((q["unedited"], q["edit"], q["controls"][c]["measures"])); n_def += 1
            s = screens(rows, concept)
            s["A3_share"] = n_def / len(ids); s["A3"] = bool(s["A3_share"] >= POS_TOL)
            s["valid"] = bool(s["A1"] and s["A2"] and s["A3"])
            s["valid_target_only"] = bool(s["A1"] and s["A2_target_only"] and s["A3"])
            res["registered"][tk][c] = s
    # frontier controls, rebuilt
    with Pool(min(64, len(ids))) as pool:
        reb = pool.map(rebuild_one, [(i, iid, fr["images"][i]) for i, iid in enumerate(ids)])
    worst = 0.0
    for i, r in enumerate(reb):
        for tk in TKS:
            for mech in ("rad", "hue"):
                rec = fr["images"][i]["targets"][tk][mech]
                worst = max(worst, abs(r[tk][mech]["rel_dE"] - rec["rel_dE"]), abs(r[tk][mech]["rel_hf"] - rec["rel_hf"]))
    res["rebuild_max_abs_difference"] = worst
    for concept, dk in c62.TARGETS:
        tk = "%s@%d" % (concept, dk)
        res["frontier"][tk] = {}
        for mech in ("rad", "hue"):
            rows = [(r[tk]["unedited"], r[tk]["edit"], r[tk][mech]["measures"]) for r in reb if r[tk][mech]["converged"]]
            s = screens(rows, concept)
            s["A3_share"] = len(rows) / len(ids); s["A3"] = bool(s["A3_share"] >= POS_TOL)
            s["valid"] = bool(s["A1"] and s["A2"] and s["A3"])
            s["valid_target_only"] = bool(s["A1"] and s["A2_target_only"] and s["A3"])
            res["frontier"][tk][mech] = s
    # the concept edit's own off-target change, over all images
    res["edit_offtarget"] = {}
    for concept, dk in c62.TARGETS:
        tk = "%s@%d" % (concept, dk)
        ok_ = c62.INDEP[OTHER[concept]]; op = "proxy_" + OTHER[concept]
        q = [r["targets"][tk] for r in con["images"]]
        res["edit_offtarget"][tk] = {
            "median_rel_cielab": float(np.median([abs(x["edit"][ok_] - x["unedited"][ok_]) / max(abs(x["unedited"][ok_]), 1e-9) for x in q])),
            "mean_rel_proxy": float(np.mean([abs(x["edit"][op] - x["unedited"][op]) / (abs(x["unedited"][op]) + 1e-6) for x in q]))}
    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("rebuild reproduces the recorded frontier mismatches to %.2e" % worst)
    for part in ("registered", "frontier"):
        for tk, d in res[part].items():
            for c, s in d.items():
                print("%-10s %-11s %-6s n=%2d maxSMD=%.3f link=%.2f A1=%d A2=%d (shift %.3f ratio %.3f; target %.3f off %.3f) A2t=%d A3=%.3f G2=%.2f off(ctrl/edit)=%.3f/%.3f valid=%d valid_t=%d" % (
                    part, tk, c, s["n"], s["max_abs_smd"], s["footprint_link_score"], s["A1"], s["A2"], s["proxy_shift"], s["proxy_ratio"],
                    s["target_proxy_shift"], s["offtarget_proxy_shift"], s["A2_target_only"],
                    s["A3_share"], s["G2_share"], s["offtarget_median_rel_control"], s["offtarget_median_rel_edit"], s["valid"], s["valid_target_only"]))
    print("edit off-target:", {k: {a: round(b, 3) for a, b in v.items()} for k, v in res["edit_offtarget"].items()})
    print("saved ->", OUT, "(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
