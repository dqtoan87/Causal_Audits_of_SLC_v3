#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C62: a pre-registered test of exact routes on held-out images.

Question. The six-descriptor footprint of Section III-C admits no route in the audit library, and a
pilot on development images could not build a concept-neutral control that matches all six
descriptors on each image. Four of the six descriptors (altered-pixel fraction, mean absolute change,
spatial support, lesion coverage) depend only on the per-pixel change magnitude. Consequence 4 allows
a coarser footprint. Under the magnitude-only footprint M_mag, a control that copies the concept
edit's per-pixel change magnitude, but changes its direction, matches M_mag exactly. Exact routes of
length two can then be built on real images. The price is a stronger working form: C1 must hold for
M_mag. Under C1 for M_mag, and if the controls are latently neutral, all route sums are equal. A
non-zero difference between two route sums therefore rejects that joint package.

Design. Every choice below is fixed in results/c62_preregistration.json, written by --phase register
before any model output on the locked set is read.
  locked set  every malignant image of the selection fold (fold 1), never used by any concept-tier
              analysis. The fold was used to select checkpoints by AUROC on unedited images.
  targets     contrast and colour variegation at the lowest and the highest dose of (7).
  controls    a two-by-two design, direction by patch scale. Every control keeps the concept edit's
              per-pixel change magnitude, so it matches M_mag exactly.
              rad4, rad16  the concept edit's own (radial) change, with its sign reversed on the
                     patches of a smooth random field of scale 4 or 16 px; the patch threshold is
                     solved per image so that the target proxy does not move;
              hue4, hue16  the same magnitude along the hue direction orthogonal to the concept
                     direction, with the hue sign alternating on the patches of the same fields, and a
                     radial share t solved per image so that the target proxy does not move.
              Where a direction would leave the colour gamut, a hue control takes the next direction
              on the circle and, if none fits, the concept edit's own change; a radial control keeps
              the edit's sign there.
  phase 1     construct (no model): exactness of M_mag, neutrality, the fine-footprint mismatch and
              independent concept measures in CIELAB.
  phase 2     evaluate (EfficientNet-B3 trio): route sums, four pairwise differences D per model and
              target (two direction pairs at a fixed scale, two scale pairs at a fixed direction), a
              simultaneous band over all D, the three-way rule with a 0.5-logit margin and the
              frontier of margins. Four uncalibrated placebos are reported alongside.
  frontier    (no model) for each direction, the smallest fine-footprint mismatch reachable when the
              magnitude map and the neutrality are held exact, by optimising three free knobs: the
              mixture of fields at 4, 16 and 48 px, and a tilt (radial) or offset (hue) of the direction.

Usage: --phase register | construct | evaluate | frontier ; --smoke runs on three development images
with untrained heads and writes only *_smoke files.
Out -> results/c62_preregistration.json, c62_construction.json, c62_evaluation.json, c62_frontier.json
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as tF
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import audit_cohort                  # noqa: E402  (image reader with the pipeline's transform)
import audited_models                # noqa: E402  (the three EfficientNet-B3 checkpoints)
import c3_cace_concept as c3m        # noqa: E402  (soft mask, concept edits, model_prob, logit)
import c7_extra_controls as c7m      # noqa: E402  (concept proxies against the original skin reference)
import c10_multiplacebo as c10       # noqa: E402  (the four uncalibrated placebos)
import c12_matched_control as c12    # noqa: E402  (the six footprint descriptors, CIELAB)
import frozen_split                  # noqa: E402

p1, DEVICE = c3m.p1, c3m.DEVICE
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
SCRIPT = os.path.abspath(__file__)

TARGETS = [("contrast", 0), ("contrast", 2), ("color_var", 0), ("color_var", 2)]
CONTROLS = ["rad4", "rad16", "hue4", "hue16"]
PAIRS = [("rad4", "hue4", "direction"), ("rad16", "hue16", "direction"),
         ("rad4", "rad16", "scale"), ("hue4", "hue16", "scale")]
MAG = ["frac_altered", "mean_abs", "support_rms", "lesion_cov"]
FINE = list(c12.COVS)
MODELS = ["erm", "slim", "causal"]
NAME = {"erm": "ERM", "slim": "Slim", "causal": "Inv"}
INDEP = {"contrast": "lab_contrast", "color_var": "lab_variegation"}
CRIT = {
    "exact_frac_abs_pixels": 1.5,     # altered-pixel fraction may differ by at most one pixel
    "exact_rel": 1e-5,                # the other three magnitude descriptors: float32 precision
    "neutral_rel": 0.01,              # |target proxy shift| <= 1% of the concept edit's own shift
    "fallback_max": 0.10,             # hue controls: share of the change magnitude on fallback pixels
    "independent_rel": 0.15,          # CIELAB measure of the target: <= 15% of the edit's own shift
    "min_images": 30,                 # a target is analysed only if at least this many images pass
    "score_floor": 0.1,               # images all three models score at >= 0.1 (as in Section IV-C)
    "margin_logit": 0.5,              # three-way rule on D
    "level": 95, "boot": 20000, "boot_seed": 62,
    "fine_feasible_rel": 1e-3,        # a six-descriptor route needs dE and HF within this
}
T_RANGE, T_GRID = (-0.95, 0.95), 39
TAU_RANGE, TAU_GRID = (-4.0, 4.0), 41
ROOT_ITERS, ROOT_TOL = 50, 1e-3       # bisection, stop at 0.1% of the edit's shift
OFFS = [0.0] + [s * k * np.pi / 8 for k in range(1, 8) for s in (1, -1)] + [np.pi]
FRONT = {"max_nfev": 30, "diff_step": 0.05, "t_grid": 13, "tau_grid": 17}
SEED_F4, SEED_F16, SEED_F48, SEED_PLACEBO = 10000, 20000, 30000, 42


def out_path(name, smoke):
    return os.path.join(RES, "c62_%s%s.json" % (name, "_smoke" if smoke else ""))


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------
# Image sets
# ------------------------------------------------------------------------------------------------
def locked_set():
    _, df_se, _ = frozen_split.load()
    mal = df_se[df_se["target"] == 1].sort_values("isic_id")
    return list(mal["isic_id"]), list(mal["patient_id"].astype(str))


def dev_set(n=3):
    man = json.load(open(os.path.join(RES, "audit_cohort.json")))
    ids, lab, pid = man["isic_id"], man["target"], man["patient_id"]
    sel = [j for j in range(len(ids)) if lab[j] == 1][:n]
    return [ids[j] for j in sel], [str(pid[j]) for j in sel]


# ------------------------------------------------------------------------------------------------
# Geometry of one image and one target
# ------------------------------------------------------------------------------------------------
def ring_mask(H, W, border=0.15):
    bh, bw = max(1, int(H * border)), max(1, int(W * border))
    r = torch.zeros(H, W, dtype=torch.bool)
    r[:bh] = True; r[-bh:] = True; r[:, :bw] = True; r[:, -bw:] = True
    return r


def unit(v, eps=1e-8):
    n = v.pow(2).sum(0, keepdim=True).sqrt()
    return v / n.clamp_min(eps)


def frame(v):
    """Per pixel: unit vector along v, the grey axis projected orthogonal to it, and their cross."""
    vh = unit(v)
    g = torch.full((3, 1, 1), 1.0 / np.sqrt(3.0), dtype=v.dtype)
    a = g - (g * vh).sum(0, keepdim=True) * vh
    alt = torch.tensor([1.0, -1.0, 0.0], dtype=v.dtype).view(3, 1, 1) / np.sqrt(2.0)
    alt = alt - (alt * vh).sum(0, keepdim=True) * vh
    a = torch.where(a.pow(2).sum(0, keepdim=True).sqrt() > 1e-4, a, alt)
    a = unit(a)
    b = torch.stack([vh[1] * a[2] - vh[2] * a[1], vh[2] * a[0] - vh[0] * a[2], vh[0] * a[1] - vh[1] * a[0]], 0)
    return vh, a, b


def context(x, concept, dk):
    w, skin, lm = c3m.soft_mask(x)
    e = c3m.make_edit(x, concept, c3m.DOSES[dk], w, skin, lm)
    ref = skin if concept == "contrast" else lm
    rh, a, b = frame(x - ref.view(3, 1, 1))
    d = e - x
    dh, da, _ = frame(d)
    return {"w": w, "skin": skin, "e": e, "m": d.pow(2).sum(0).sqrt(), "rh": rh, "a": a, "b": b,
            "dh": dh, "da": da, "concept": concept}


def field(H, W, seed, sigma):
    g = torch.Generator().manual_seed(int(seed))
    n = torch.randn(1, 1, H, W, generator=g)
    k = int(3 * sigma)
    xs = torch.arange(-k, k + 1, dtype=torch.float32)
    ker = torch.exp(-xs ** 2 / (2.0 * sigma ** 2)); ker = ker / ker.sum()
    n = tF.conv2d(n, ker.view(1, 1, 1, -1), padding=(0, k))
    n = tF.conv2d(n, ker.view(1, 1, -1, 1), padding=(k, 0))[0, 0]
    return (n - n.mean()) / n.std()


# ------------------------------------------------------------------------------------------------
# The two mechanisms. Both keep the concept edit's per-pixel change magnitude m(p).
# ------------------------------------------------------------------------------------------------
def build_dir(x, cx, t, alpha):
    """Change of size m(p) along cos(alpha) a + sin(alpha) b (orthogonal to the concept direction),
    tilted by t towards the concept direction. alpha is an [H,W] tensor of preferred angles."""
    m = cx["m"].unsqueeze(0)
    st = float(np.sqrt(max(1.0 - t * t, 0.0)))
    Y = torch.stack([x + m * (st * (torch.cos(alpha + o) * cx["a"] + torch.sin(alpha + o) * cx["b"]) + t * cx["rh"])
                     for o in OFFS])
    ok = ((Y >= 0) & (Y <= 1)).all(1)
    anyok = ok.any(0)
    first = ok.float().argmax(0)
    sel = Y.gather(0, first.view(1, 1, *first.shape).expand(1, 3, *first.shape))[0]
    out = torch.where(anyok.unsqueeze(0), sel, cx["e"])
    fb = (~anyok) & (cx["m"] > 0)
    return out, fb


def build_sgn(x, cx, fld, tau, lam=0.0):
    """The concept edit's own change, with its sign reversed where the field is at or below tau.
    lam > 0 tilts every change towards the grey-axis direction orthogonal to it (frontier only)."""
    neg = fld <= tau
    if lam == 0.0:
        y = 2 * x - cx["e"]
        okn = ((y >= 0) & (y <= 1)).all(0)
        out = torch.where((neg & okn).unsqueeze(0), y, cx["e"])
        return out, neg & ~okn
    s = torch.where(neg, -1.0, 1.0).unsqueeze(0)
    y = x + cx["m"].unsqueeze(0) * (s * float(np.sqrt(1 - lam * lam)) * cx["dh"] + lam * cx["da"])
    ok = ((y >= 0) & (y <= 1)).all(0)
    out = torch.where(ok.unsqueeze(0), y, cx["e"])
    return out, (~ok) & (cx["m"] > 0)


def tproxy(img, cx):
    return c7m.proxy_fixed(img, cx["skin"])[cx["concept"]]


def solve_root(f, lo, hi, n_grid, tol_abs, prefer=0.0):
    """Root of f on [lo, hi]: grid search for sign changes, then bisection in the bracket nearest to
    `prefer`. Returns (argument, value, converged)."""
    grid = np.linspace(lo, hi, n_grid)
    vals = [f(g) for g in grid]
    br = [(grid[i], grid[i + 1], vals[i], vals[i + 1]) for i in range(n_grid - 1)
          if vals[i] == 0 or vals[i] * vals[i + 1] < 0]
    if not br:
        i = int(np.argmin(np.abs(vals)))
        return float(grid[i]), float(vals[i]), abs(vals[i]) <= tol_abs
    br.sort(key=lambda q: min(abs(q[0] - prefer), abs(q[1] - prefer)))
    a, b, fa, fb = br[0]
    if fa == 0:
        return float(a), 0.0, True
    for _ in range(ROOT_ITERS):
        c = 0.5 * (a + b); fc = f(c)
        if abs(fc) <= tol_abs:
            return float(c), float(fc), True
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    c, fc = (a, fa) if abs(fa) < abs(fb) else (b, fb)
    return float(c), float(fc), abs(fc) <= tol_abs


# ------------------------------------------------------------------------------------------------
# Measures
# ------------------------------------------------------------------------------------------------
def measures(x, img, cx, ring):
    fp = c12.footprint(x, img, cx["w"])
    px = c7m.proxy_fixed(img, cx["skin"])
    lab = c12.rgb2lab(img); w0 = cx["w"]; den = w0.sum().clamp_min(1e-6)
    lm = (lab * w0).sum((1, 2)) / den
    sk = lab[:, ring].mean(1)
    var = ((((lab - lm.view(3, 1, 1)) ** 2) * w0).sum((1, 2)).sum() / den).sqrt()
    out = {k: float(fp[k]) for k in FINE}
    out.update({"proxy_" + k: float(v) for k, v in px.items()})
    out.update({"lab_contrast": float((lm - sk).pow(2).sum().sqrt()), "lab_variegation": float(var)})
    return out


def judge(mx, me, mc, concept, fb_share, kind, npix):
    """G1 (exact magnitude footprint, neutrality, fallback) and G2 (independent CIELAB neutrality)."""
    exact = abs(mc["frac_altered"] - me["frac_altered"]) <= CRIT["exact_frac_abs_pixels"] / npix
    exact = exact and all(abs(mc[k] - me[k]) <= CRIT["exact_rel"] * max(abs(me[k]), 1e-12) for k in MAG[1:])
    pk = "proxy_" + concept
    shift = me[pk] - mx[pk]
    neutral_rel = abs(mc[pk] - mx[pk]) / max(abs(shift), 1e-12)
    ik = INDEP[concept]
    indep_rel = abs(mc[ik] - mx[ik]) / max(abs(me[ik] - mx[ik]), 1e-12)
    fb_ok = fb_share <= CRIT["fallback_max"] if kind == "hue" else True
    g1 = bool(exact and neutral_rel <= CRIT["neutral_rel"] and fb_ok)
    return {"exact_mag": bool(exact), "neutral_rel": float(neutral_rel), "fallback_share": float(fb_share),
            "g1": g1, "indep_rel": float(indep_rel), "g2": bool(indep_rel <= CRIT["independent_rel"]),
            "rel_dE": float((mc["lab_dE"] - me["lab_dE"]) / max(abs(me["lab_dE"]), 1e-12)),
            "rel_hf": float((mc["hf_change"] - me["hf_change"]) / max(abs(me["hf_change"]), 1e-12))}


def fb_share(cx, fb):
    return float((cx["m"] * fb).sum() / cx["m"].sum().clamp_min(1e-12))


# ------------------------------------------------------------------------------------------------
# Phase 1: construction (no model)
# ------------------------------------------------------------------------------------------------
def hue_alpha(fld):
    """Hue direction b where the field is positive, its opposite elsewhere."""
    return torch.where(fld > 0, torch.tensor(np.pi / 2), torch.tensor(-np.pi / 2))


def build_controls(x, cx, i, params=None):
    """Return {control: (image, fallback mask, parameter, converged)}; with params, rebuild."""
    H, W = x.shape[1:]
    p0 = tproxy(x, cx); shift = tproxy(cx["e"], cx) - p0
    tol = ROOT_TOL * abs(shift)
    out = {}
    for sc, sig, seed in (("4", 4, SEED_F4), ("16", 16, SEED_F16)):
        fld = field(H, W, seed + i, sig)
        name = "rad" + sc
        if params is None:
            tau, _, conv = solve_root(lambda u: tproxy(build_sgn(x, cx, fld, u)[0], cx) - p0, *TAU_RANGE, TAU_GRID, tol)
        else:
            tau, conv = params[name], True
        img, fb = build_sgn(x, cx, fld, tau); out[name] = (img, fb, tau, conv)
        name, alpha = "hue" + sc, hue_alpha(fld)
        if params is None:
            t, _, conv = solve_root(lambda t: tproxy(build_dir(x, cx, t, alpha)[0], cx) - p0, *T_RANGE, T_GRID, tol)
        else:
            t, conv = params[name], True
        img, fb = build_dir(x, cx, t, alpha); out[name] = (img, fb, t, conv)
    return out


def construct_one(args):
    i, iid = args
    torch.set_num_threads(1)
    x = audit_cohort._read([iid])[0]
    ring = ring_mask(*x.shape[1:]); npix = x.shape[1] * x.shape[2]
    rec = {"isic_id": iid, "targets": {}}
    for concept, dk in TARGETS:
        cx = context(x, concept, dk)
        mx, me = measures(x, x, cx, ring), measures(x, cx["e"], cx, ring)
        tr = {"unedited": mx, "edit": me, "controls": {}}
        for name, (img, fb, prm, conv) in build_controls(x, cx, i).items():
            mc = measures(x, img, cx, ring)
            j = judge(mx, me, mc, concept, fb_share(cx, fb), name[:3], npix)
            j.update({"param": prm, "root_converged": bool(conv), "measures": mc})
            tr["controls"][name] = j
        rec["targets"]["%s@%d" % (concept, dk)] = tr
    return rec


def construct(smoke):
    t0 = time.time()
    ids, pids = dev_set() if smoke else locked_set()
    with Pool(min(64, len(ids))) as pool:
        recs = pool.map(construct_one, list(enumerate(ids)))
    summ = {}
    for concept, dk in TARGETS:
        tk = "%s@%d" % (concept, dk)
        s = {}
        for c in CONTROLS:
            J = [r["targets"][tk]["controls"][c] for r in recs]
            s[c] = {"share_exact_mag": float(np.mean([j["exact_mag"] for j in J])),
                    "share_neutral": float(np.mean([j["neutral_rel"] <= CRIT["neutral_rel"] for j in J])),
                    "share_g1": float(np.mean([j["g1"] for j in J])),
                    "share_g2_independent": float(np.mean([j["g2"] for j in J])),
                    "median_fallback_share": float(np.median([j["fallback_share"] for j in J])),
                    "median_abs_rel_dE": float(np.median([abs(j["rel_dE"]) for j in J])),
                    "median_abs_rel_hf": float(np.median([abs(j["rel_hf"]) for j in J])),
                    "median_indep_rel": float(np.median([j["indep_rel"] for j in J]))}
        s["n_all_three_g1"] = int(sum(all(r["targets"][tk]["controls"][c]["g1"] for c in CONTROLS) for r in recs))
        summ[tk] = s
        print(tk, json.dumps({c: {k: round(v, 4) for k, v in s[c].items()} for c in CONTROLS}),
              "all three G1:", s["n_all_three_g1"], flush=True)
    out = {"phase": "construct", "smoke": smoke, "ids": ids, "patient_ids": pids, "criteria": CRIT,
           "script_sha256": sha256(SCRIPT), "summary": summ, "images": recs,
           "runtime_sec": round(time.time() - t0, 1)}
    p = out_path("construction", smoke)
    json.dump(out, open(p, "w"))
    print("saved ->", p, "sha256", sha256(p), "(%.0fs)" % (time.time() - t0), flush=True)


# ------------------------------------------------------------------------------------------------
# Phase 2: evaluation on the audited classifiers
# ------------------------------------------------------------------------------------------------
def boot(vals, masks, pid, B, seed):
    """Patient bootstrap shared by all components. vals, masks: [K, n]. Returns est [K], draws [B, K]."""
    gi = np.unique(pid, return_inverse=True)[1]; G = gi.max() + 1
    V = np.where(masks, vals, 0.0); M = masks.astype(float)
    ps = np.stack([V[:, gi == g].sum(1) for g in range(G)], 1)
    pc = np.stack([M[:, gi == g].sum(1) for g in range(G)], 1)
    rng = np.random.RandomState(seed)
    cnt = np.zeros((B, G))
    np.add.at(cnt, (np.repeat(np.arange(B), G), rng.randint(0, G, B * G)), 1)
    num, den = cnt @ ps.T, cnt @ pc.T
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = np.where(den > 0, num / den, np.nan)
    return V.sum(1) / np.maximum(M.sum(1), 1), draws


def decide(lo, hi, eps):
    if lo > eps or hi < -eps:
        return "exceeds"
    if -eps < lo and hi < eps:
        return "within"
    return "indeterminate"


def evaluate(smoke):
    t0 = time.time()
    if not smoke:
        reg = json.load(open(out_path("preregistration", False)))
        if reg["script_sha256"] != sha256(SCRIPT):
            raise SystemExit("script differs from the registered version; register an amendment first")
    cons = json.load(open(out_path("construction", smoke)))
    ids, pids = cons["ids"], np.asarray(cons["patient_ids"])
    torch.set_num_threads(1)              # rebuild exactly as the single-threaded construction did
    imgs = audit_cohort._read(ids)
    tks = ["%s@%d" % t for t in TARGETS]
    cols, stacks = [], []
    for i, x in enumerate(imgs):
        items = [x]; rec = cons["images"][i]["targets"]
        for (concept, dk), tk in zip(TARGETS, tks):
            cx = context(x, concept, dk)
            items.append(cx["e"])
            built = build_controls(x, cx, i, params={c: rec[tk]["controls"][c]["param"] for c in CONTROLS})
            for c in CONTROLS:
                img = built[c][0]
                mc = c12.footprint(x, img, cx["w"])
                for k in MAG:   # the rebuilt control must reproduce the registered construction
                    ref = rec[tk]["controls"][c]["measures"][k]
                    assert abs(mc[k] - ref) <= 1e-6 * max(abs(ref), 1e-9), (ids[i], tk, c, k)
                items.append(img)
        w, skin, _ = c3m.soft_mask(x)
        for j, fn in enumerate(c10.PLACEBOS.values()):
            items.append(fn(x, w, skin, np.random.RandomState(SEED_PLACEBO + i + 7)))
        stacks.append(torch.stack(items))
    cols = ["unedited"] + ["%s|%s" % (tk, c) for tk in tks for c in ["edit"] + CONTROLS] + \
           ["placebo|" + p for p in c10.PLACEBOS]
    P = {}
    for m in MODELS:
        model = p1.build_model("erm" if m == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        if not smoke:
            model.load_state_dict(torch.load(audited_models.CKPTS[m], map_location=DEVICE))
        model.eval()
        P[m] = np.stack([c3m.model_prob(model, s) for s in stacks])
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("[%s] scored (%.0fs)" % (m, time.time() - t0), flush=True)
    DL = {m: c3m.logit(P[m]) - c3m.logit(P[m][:, :1]) for m in MODELS}
    col = {c: k for k, c in enumerate(cols)}
    common = np.all([P[m][:, 0] >= CRIT["score_floor"] for m in MODELS], 0)

    def run(sel_rule):
        comps, vals, masks, info = [], [], [], {}
        for tk in tks:
            g1 = np.array([all(cons["images"][i]["targets"][tk]["controls"][c]["g1"] for c in CONTROLS)
                           for i in range(len(ids))])
            g2 = np.array([all(cons["images"][i]["targets"][tk]["controls"][c]["g2"] for c in CONTROLS)
                           for i in range(len(ids))])
            msk = {"primary": g1 & common, "independent": g1 & g2 & common, "all_g1": g1}[sel_rule]
            info[tk] = {"n_images": int(msk.sum()), "n_patients": int(len(set(pids[msk]))),
                        "analysed": bool(msk.sum() >= (CRIT["min_images"] if not smoke else 1))}
            for m in MODELS:
                dl = DL[m]
                re = dl[:, col[tk + "|edit"]]
                def add(name, v):
                    comps.append(name); vals.append(v); masks.append(msk)
                add("%s|%s|R_edit" % (m, tk), re)
                for c in CONTROLS:
                    add("%s|%s|R_%s" % (m, tk, c), dl[:, col["%s|%s" % (tk, c)]])
                    add("%s|%s|S_%s" % (m, tk, c), re - dl[:, col["%s|%s" % (tk, c)]])
                for a, b, kind in PAIRS:
                    add("%s|%s|D_%s-%s" % (m, tk, a, b), dl[:, col["%s|%s" % (tk, b)]] - dl[:, col["%s|%s" % (tk, a)]])
                for p in c10.PLACEBOS:
                    add("%s|%s|PC_%s" % (m, tk, p), re - dl[:, col["placebo|" + p]])
        vals, masks = np.stack(vals), np.stack(masks)
        est, draws = boot(vals, masks, pids, CRIT["boot"] if not smoke else 200, CRIT["boot_seed"])
        lo = np.nanpercentile(draws, 2.5, 0); hi = np.nanpercentile(draws, 97.5, 0)
        isD = np.array(["|D_" in c for c in comps])
        okD = isD & np.array([info[c.split("|")[1]]["analysed"] for c in comps])
        se = np.nanstd(draws, 0, ddof=1)
        Z = np.abs(draws[:, okD] - est[okD]) / np.maximum(se[okD], 1e-12)
        crit = float(np.nanpercentile(np.nanmax(Z, 1), CRIT["level"])) if okD.any() else float("nan")
        eps = CRIT["margin_logit"]
        res = {"selection": sel_rule, "targets": info, "crit_simultaneous": crit, "components": {}, "cells": {}}
        for k, c in enumerate(comps):
            r = {"est": float(est[k]), "ci": [float(lo[k]), float(hi[k])]}
            if okD[k]:
                sl, sh = est[k] - crit * se[k], est[k] + crit * se[k]
                r.update({"sim": [float(sl), float(sh)], "decision_sim": decide(sl, sh, eps),
                          "decision_marginal": decide(lo[k], hi[k], eps),
                          "frontier_within": float(max(abs(sl), abs(sh))),
                          "frontier_exceeds": float(min(abs(sl), abs(sh)) if sl * sh > 0 else 0.0)})
            res["components"][c] = r
        for tk in tks:
            if not info[tk]["analysed"]:
                continue
            for m in MODELS:
                ds = {"%s-%s" % (a, b): res["components"]["%s|%s|D_%s-%s" % (m, tk, a, b)]["decision_sim"]
                      for a, b, _ in PAIRS}
                cls = "routes disagree" if "exceeds" in ds.values() else (
                    "routes agree" if all(v == "within" for v in ds.values()) else "indeterminate")
                res["cells"]["%s|%s" % (m, tk)] = {"class": cls, "pairs": ds}
        cl = [v["class"] for v in res["cells"].values()]
        res["summary"] = {k: cl.count(k) for k in ("routes disagree", "routes agree", "indeterminate")}
        for kind in ("direction", "scale"):
            names = ["%s-%s" % (a, b) for a, b, k in PAIRS if k == kind]
            res["summary"][kind + "_pair_exceeds"] = sum(any(v["pairs"][n] == "exceeds" for n in names)
                                                         for v in res["cells"].values())
        return res

    out = {"phase": "evaluate", "smoke": smoke, "criteria": CRIT, "script_sha256": sha256(SCRIPT),
           "construction_sha256": sha256(out_path("construction", smoke)),
           "n_locked": len(ids), "n_common_score_floor": int(common.sum()),
           "unedited_mean_prob": {m: float(P[m][:, 0].mean()) for m in MODELS},
           "primary": run("primary"), "secondary_independent": run("independent"),
           "secondary_all_g1": run("all_g1"), "runtime_sec": round(time.time() - t0, 1)}
    p = out_path("evaluation", smoke)
    json.dump(out, open(p, "w"), indent=1)
    for k in ("primary", "secondary_independent", "secondary_all_g1"):
        print(k, out[k]["summary"], "crit %.3f" % out[k]["crit_simultaneous"], flush=True)
        for c, v in out[k]["cells"].items():
            print("  ", c, v["class"], v["pairs"], flush=True)
    print("saved ->", p, "(%.0fs)" % (time.time() - t0), flush=True)


# ------------------------------------------------------------------------------------------------
# Frontier: how close can the fine footprint get when the magnitude map and neutrality are exact?
# ------------------------------------------------------------------------------------------------
def mix_field(fs, k1, k2):
    """Unit-variance mixture of three fields on a sphere: k1 = k2 = 0 gives the first field."""
    m = np.cos(k1) * fs[0] + np.sin(k1) * (np.cos(k2) * fs[1] + np.sin(k2) * fs[2])
    return (m - m.mean()) / m.std()


def frontier_one(args):
    i, iid, nfev = args
    torch.set_num_threads(1)
    x = audit_cohort._read([iid])[0]
    H, W = x.shape[1:]
    fs = (field(H, W, SEED_F4 + i, 4), field(H, W, SEED_F16 + i, 16), field(H, W, SEED_F48 + i, 48))
    rec = {"isic_id": iid, "targets": {}}
    for concept, dk in TARGETS:
        cx = context(x, concept, dk)
        p0 = tproxy(x, cx); shift = tproxy(cx["e"], cx) - p0; tol = ROOT_TOL * abs(shift)
        fe = c12.footprint(x, cx["e"], cx["w"])

        def resid_of(img):
            fc = c12.footprint(x, img, cx["w"])
            return np.array([(fc["lab_dE"] - fe["lab_dE"]) / max(fe["lab_dE"], 1e-12),
                             (fc["hf_change"] - fe["hf_change"]) / max(fe["hf_change"], 1e-12)])

        def rad_img(kn):          # knobs: field mixture (k1, k2) and a tilt lam towards the grey axis
            fld = mix_field(fs, kn[0], kn[1])
            tau, _, conv = solve_root(lambda u: tproxy(build_sgn(x, cx, fld, u, kn[2])[0], cx) - p0,
                                      *TAU_RANGE, FRONT["tau_grid"], tol)
            return build_sgn(x, cx, fld, tau, kn[2])[0], conv

        def hue_img(kn):          # knobs: field mixture (k1, k2) and an offset delta of the hue direction
            alpha = hue_alpha(mix_field(fs, kn[0], kn[1])) + kn[2]
            t, _, conv = solve_root(lambda t: tproxy(build_dir(x, cx, t, alpha)[0], cx) - p0,
                                    *T_RANGE, FRONT["t_grid"], tol)
            return build_dir(x, cx, t, alpha)[0], conv

        starts = [[0.0, 0.0, 0.0], [np.pi / 2, 0.0, 0.0], [np.pi / 2, np.pi / 2, 0.0]]
        lo, hi = [0.0, 0.0, -0.5], [np.pi / 2, np.pi / 2, 0.5]
        tr = {}
        for name, fn in (("rad", rad_img), ("hue", hue_img)):
            best = None
            for st in starts:
                try:
                    r = least_squares(lambda kn: resid_of(fn(kn)[0]), st, bounds=(lo, hi),
                                      diff_step=FRONT["diff_step"], max_nfev=nfev)
                except Exception as exc:          # a failed start is recorded, not hidden
                    tr.setdefault("errors", []).append("%s: %s" % (name, exc))
                    continue
                if best is None or r.cost < best.cost:
                    best = r
            img, conv = fn(best.x)
            fc = c12.footprint(x, img, cx["w"])
            mag_rel = max(abs(fc[k] - fe[k]) / max(abs(fe[k]), 1e-12) for k in MAG)
            rr = resid_of(img)
            tr[name] = {"knobs": [float(v) for v in best.x], "rel_dE": float(rr[0]), "rel_hf": float(rr[1]),
                        "max_rel_mag": float(mag_rel), "neutral_converged": bool(conv),
                        "fine_feasible": bool(np.all(np.abs(rr) <= CRIT["fine_feasible_rel"]) and conv
                                              and mag_rel <= CRIT["exact_rel"] * 10)}
        rec["targets"]["%s@%d" % (concept, dk)] = tr
    return rec


def frontier(smoke):
    t0 = time.time()
    ids, _ = dev_set(1) if smoke else locked_set()
    nfev = 3 if smoke else FRONT["max_nfev"]
    with Pool(min(64, len(ids))) as pool:
        recs = pool.map(frontier_one, [(i, iid, nfev) for i, iid in enumerate(ids)])
    summ = {}
    for concept, dk in TARGETS:
        tk = "%s@%d" % (concept, dk)
        summ[tk] = {}
        for name in ("rad", "hue"):
            R = [r["targets"][tk][name] for r in recs]
            dE = np.abs([q["rel_dE"] for q in R]); hf = np.abs([q["rel_hf"] for q in R])
            both = np.maximum(dE, hf)
            summ[tk][name] = {"median_abs_rel_dE": float(np.median(dE)), "median_abs_rel_hf": float(np.median(hf)),
                              "quartiles_max_rel": [float(np.percentile(both, 25)), float(np.percentile(both, 75))],
                              "min_max_rel": float(both.min()),
                              "share_fine_feasible": float(np.mean([q["fine_feasible"] for q in R])),
                              "share_within_1pct": float(np.mean(both <= 0.01)),
                              "share_within_10pct": float(np.mean(both <= 0.10))}
        print(tk, json.dumps(summ[tk]), flush=True)
    out = {"phase": "frontier", "smoke": smoke, "settings": FRONT, "criteria": CRIT,
           "script_sha256": sha256(SCRIPT), "summary": summ, "images": recs,
           "runtime_sec": round(time.time() - t0, 1)}
    p = out_path("frontier", smoke)
    json.dump(out, open(p, "w"))
    print("saved ->", p, "(%.0fs)" % (time.time() - t0), flush=True)


# ------------------------------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------------------------------
def register():
    ids, pids = locked_set()
    reg = {
        "title": "Pre-registered test of exact routes under a magnitude-only footprint",
        "registered_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "code/c62_route_test.py", "script_sha256": sha256(SCRIPT),
        "status_at_registration": ("No model output has been computed on any locked image for this test. "
                                   "The construction phase has not been run on the locked set."),
        "locked_set": {"definition": "every malignant image of the selection fold (fold 1) of the frozen split",
                       "n_images": len(ids), "n_patients": len(set(pids)), "isic_ids": ids, "patient_ids": pids,
                       "note": ("Patient-disjoint from the audit cohort and from training. Fold 1 selected the "
                                "checkpoints by AUROC on unedited images; the test compares controls on the same "
                                "images, so that use cancels in each difference.")},
        "pilot": ("All pilots used development images of the audit cohort and read no output of the audited "
                  "checkpoints. (1) On four to six images, contrast targets only: controls holding the target "
                  "proxy fixed matched the four magnitude descriptors exactly but missed CIELAB dE and the "
                  "high-frequency change. (2) With this script on three images: a uniform hue control moved the "
                  "CIELAB contrast by 45 to 120% of the concept edit's own shift, which led to the alternating-hue "
                  "design registered here; the radial controls missed dE by 0.1 to 1.6% and the high-frequency "
                  "change by 5 to 75%. (3) The evaluation code was run once on three development images with "
                  "untrained classification heads, to check the code path only. The pilots are not evidence."),
        "targets": ["%s@%d" % t for t in TARGETS], "controls": CONTROLS,
        "pairs": [{"pair": "%s-%s" % (a, b), "kind": k} for a, b, k in PAIRS],
        "design": ("Two-by-two: change direction (radial, hue) by patch scale (4, 16 px). A direction pair "
                   "holds the field and changes the direction; a scale pair holds the direction and changes "
                   "the field."),
        "footprints": {"coarse_M_mag": MAG, "fine_M": FINE},
        "criteria": CRIT, "root_search": {"t_range": T_RANGE, "t_grid": T_GRID, "tau_range": TAU_RANGE,
                                          "tau_grid": TAU_GRID, "iters": ROOT_ITERS, "tol_rel": ROOT_TOL,
                                          "rule": "bracket nearest to zero"},
        "frontier_settings": FRONT,
        "models": {m: os.path.basename(audited_models.CKPTS[m]) for m in MODELS},
        "response_scale": "clipped logit, eta = 1e-6, as in (5) of the main text",
        "analysis_sets": {"primary": ("images on which all four controls of the target pass G1 and all three "
                                      "models score the unedited image at >= 0.1"),
                          "secondary_independent": ("the primary images on which all four controls also pass "
                                                    "the independent CIELAB check G2"),
                          "secondary_all_g1": "images on which all four controls of the target pass G1"},
        "hypotheses": {
            "H_null": ("Under C1 for M_mag and latent neutrality of the controls, every pairwise difference D of "
                       "route sums is zero."),
            "decision": ("Each D is judged by the three-way rule on a simultaneous 95% max-statistic patient "
                         "bootstrap band over all analysed D components, with margin 0.5 logit; marginal "
                         "intervals and the frontier of margins are reported alongside."),
            "cell_classes": {"routes disagree": "at least one D of the model-target cell exceeds the margin",
                             "routes agree": "all four D of the cell are within the margin",
                             "indeterminate": "otherwise"},
            "construction_failure": "a target with fewer than 30 images passing G1 for all three controls is not analysed",
            "expectation": ("Based on diagnostic (T) on the audit cohort, we expect direction pairs to exceed the "
                            "margin in most cells. We have no expectation for the scale pairs. The report does not "
                            "depend on these expectations.")},
        "interpretation": {
            "routes disagree": ("rejects the joint package of C1 for M_mag and latent neutrality of the controls; "
                                "the ladder halts at C1 in the coarse footprint"),
            "routes agree": ("the route sum is a candidate concept component for M_mag, conditional on C1 and latent "
                             "neutrality, which agreement does not prove; theta^C is still not reached because "
                             "the fixed-level part of C3b fails for edit (7)"),
            "frontier": ("the smallest fine-footprint mismatch reachable within these mechanisms; it bounds what "
                         "these mechanisms can do, not what every control can do"),
            "direction_vs_scale": ("a direction pair that exceeds while the scale pairs stay within indicates that the "
                                   "classifier's generic sensitivity depends on the direction of the colour change, which "
                                   "the magnitude-only footprint omits; an exceeding scale pair indicates dependence on "
                                   "the spatial pattern"),
            "variegation_hue": ("for the variegation targets the hue controls are expected to fail the independent "
                                "check, because alternating hue adds colour variety that the RGB proxy, held fixed, "
                                "does not register; there the direction pairs test C1 jointly with a known neutrality "
                                "failure, and the scale pairs are the cleaner test")},
        "reporting": "All registered outcomes will be reported, whatever their direction.",
        "amendments": []}
    p = out_path("preregistration", False)
    json.dump(reg, open(p, "w"), indent=1)
    print("registered", p, "script sha256", reg["script_sha256"], "locked images", len(ids), "patients", len(set(pids)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["register", "construct", "evaluate", "frontier"], required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.phase == "register":
        register()
    elif a.phase == "construct":
        construct(a.smoke)
    elif a.phase == "evaluate":
        evaluate(a.smoke)
    else:
        frontier(a.smoke)


if __name__ == "__main__":
    main()
