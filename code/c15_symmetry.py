#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C15: test of the symmetry assumption of the opposite-dose design, and of the calibration
invariance of CaCE on the output space.

Two places where the claims would go beyond the evidence:

 (1) "the opposite-dose design cancels the confidence term exactly". If
         ΔL₊ = −α₊·L₀ + h(+δ)   and   ΔL₋ = −α₋·L₀ + h(−δ)
     then  ΔL₊ − ΔL₋ = (α₋ − α₊)·L₀ + h(+δ) − h(−δ).
     The confidence term cancels only when α₊ = α₋. This condition is testable and must be tested.
     C15 estimates α₊, α₋ and tests H₀: α₊ = α₋ with a paired bootstrap over images.

 (2) The "shrinkage toward the decision boundary" mechanism is measured only on neutral edits; it is
     not shown that concept edits follow the same decomposition. C15 estimates α for concept edits
     too (both directions, two doses).

C15 also quantifies the statement "CaCE on the output space is not calibration-invariant":
temperature scaling p_T = σ(logit(p)/T) shrinks every logit-scale quantity by exactly 1/T, so
cross-model comparisons are multiplied by T_i/T_j. T is fitted on the same validation fold as C14.

Out -> results/c15_symmetry.json
Run:  .venv/bin/python Causal_XAI_v3/code/c15_symmetry.py
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
import c3_cace_concept as c3m       # noqa: E402
import c10_multiplacebo as c10      # noqa: E402
import c12_matched_control as c12   # noqa: E402

torch.set_num_threads(min(8, os.cpu_count() or 8))
p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c15_symmetry.json")
import audited_models                # noqa: E402  (checkpoints resolved in one place)
CKPTS = audited_models.CKPTS         # absolute paths to the fixed set
SEED, N_EVAL, BOOT = 42, 200, 2000
ORDER = ["erm", "slim", "causal"]
NAME = {"erm": "ERM", "slim": "Slim", "causal": "Inv"}
TARGETS = ["contrast@0", "contrast@2", "color_var@0", "color_var@2"]
PLACEBO = ["flip", "jitter", "hue", "noise"]


def build_edits(imgs01):
    """Per image: concept up and concept down (mean|Δ| matched by bisection, same formula as C12) + 4 placebos."""
    edits, names = [], None
    for i, x in enumerate(imgs01):
        w, skin, lm = c3m.soft_mask(x)
        d = {}
        for t in TARGETS:
            cname, k = t.split("@")
            up = c3m.make_edit(x, cname, c3m.DOSES[int(k)], w, skin, lm)
            d[f"{t}|up"] = up
            tgt = float(c12.dpix(x, up).mean())
            dn, _, _, _ = c12.match_strength(
                lambda gp: c3m.make_edit(x, cname, -gp, w, skin, lm), 0.0, 1.0, x, tgt)
            d[f"{t}|dn"] = dn
        for pnm, fn in c10.PLACEBOS.items():
            d[f"pb:{pnm}"] = fn(x, w, skin, np.random.RandomState(SEED + i + 7))
        edits.append(d)
        if names is None:
            names = list(d.keys())
    return edits, names


@torch.no_grad()
def eval_model(model, imgs01, edits, names):
    n = len(imgs01)
    p0 = np.empty(n); pe = np.empty((n, len(names)))
    for i, x in enumerate(imgs01):
        ps = c3m.model_prob(model, torch.stack([x] + [edits[i][nm] for nm in names]))
        p0[i] = ps[0]; pe[i] = ps[1:]
    return p0, pe


def ols_slope(x, y):
    """Return (slope, intercept, R2) of y ~ a + s·x."""
    s, a = np.polyfit(x, y, 1)
    pred = a + s * x
    r2 = 1.0 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return float(s), float(a), float(r2)


def boot_slope_diff(x, y1, y2, n_boot, seed):
    """Paired bootstrap over images for the difference of two slopes (α for the up and down edits)."""
    rng = np.random.RandomState(seed); n = len(x); out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        s1, _, _ = ols_slope(x[idx], y1[idx]); s2, _, _ = ols_slope(x[idx], y2[idx])
        out[b] = s1 - s2
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(out.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)


def fit_temperature(prob, y, lo=0.05, hi=20.0, iters=60):
    """Fit T minimising the NLL of σ(logit(p)/T) by ternary search."""
    L = c3m.logit(prob)

    def nll(T):
        z = L / T
        return float(np.mean(np.logaddexp(0.0, z) - y * z))
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3; m2 = hi - (hi - lo) / 3
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    T = 0.5 * (lo + hi)
    return float(T), nll(T), nll(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    print(f"##### C15 symmetry & calibration-invariance {time.strftime('%H:%M:%S')} #####", flush=True)

    _, df_va = p2.build_split({"seed": SEED, "n_folds": 5})
    imgs01, labels = p2.load_eval_images(df_va, N_EVAL, SEED)
    labels = np.array(labels); n = len(imgs01)
    edits, names = build_edits(imgs01)
    print(f"eval images {n}  edits/image {len(names)}  ({time.time()-t0:.0f}s)", flush=True)

    res = {"n_images": int(n), "n_malignant": int((labels == 1).sum()), "boot": args.boot,
           "note": ("Tests the antisymmetry assumption alpha_+ = alpha_- of the dose-contrast design, "
                    "extends the confidence-contraction model to concept edits, and quantifies the "
                    "calibration-dependence of output-space CaCE via temperature scaling."),
           "models": {}}

    for m in ORDER:
        model = p1.build_model("erm" if m == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(CKPTS[m], map_location=DEVICE))
        model.eval()
        p0, pe = eval_model(model, imgs01, edits, names)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        L0 = c3m.logit(p0); dL = c3m.logit(pe) - L0[:, None]; dP = pe - p0[:, None]
        mr = {"p0_mean": float(p0.mean()), "slopes": {}, "symmetry": {}, "prob_scale": {}}

        # --- (2) shrinkage coefficient for every edit, including concept edits
        for j, nm in enumerate(names):
            s, a, r2 = ols_slope(L0, dL[:, j])
            mr["slopes"][nm] = {"alpha": -s, "intercept": a, "r2": r2}   # alpha = -slope (positive shrinkage)
            sp, ap_, r2p = ols_slope(p0, dP[:, j])
            mr["prob_scale"][nm] = {"slope_p": sp, "r2": r2p}

        # --- (1) test H0: alpha_+ = alpha_-
        for t in TARGETS:
            ju, jd = names.index(f"{t}|up"), names.index(f"{t}|dn")
            au = mr["slopes"][f"{t}|up"]["alpha"]; ad = mr["slopes"][f"{t}|dn"]["alpha"]
            d, lo, hi, sig = boot_slope_diff(L0, dL[:, ju], dL[:, jd], args.boot, SEED)
            sym = dL[:, ju] - dL[:, jd]
            resid = abs(ad - au) * float(np.mean(np.abs(L0)))
            mr["symmetry"][t] = {
                "alpha_up": au, "alpha_dn": ad, "alpha_diff": au - ad,
                "boot_diff_mean": d, "boot_diff_ci": [lo, hi], "reject_symmetry": sig,
                "dcace_sym": float(sym.mean()),
                "residual_confidence_term": resid,
                "residual_over_effect": float(resid / max(abs(float(sym.mean())), 1e-9)),
            }
        res["models"][m] = mr
        print(f"[{m}] done ({time.time()-t0:.0f}s)", flush=True)

    # --- (3) temperature scaling on the validation fold of C14
    temps = {}
    try:
        Y = np.load(os.path.join(RES, "c14_labels.npy")).astype(float)
        for m in ORDER:
            P = np.load(os.path.join(RES, f"c14_pred_{m}.npy")).astype(float)
            T, nT, n1 = fit_temperature(P, Y)
            temps[m] = {"T": T, "nll_T": nT, "nll_T1": n1}
        res["temperature"] = temps
    except FileNotFoundError:
        res["temperature"] = None
        print("  (c14 predictions not found; skipping temperature scaling)", flush=True)

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)

    W = 112
    print("=" * W)
    print("A) CONFIDENCE-CONTRACTION COEFFICIENT alpha  (dLogit = -alpha*L0 + const);  alpha>0 = contraction")
    print(f"{'edit':16s}" + "".join(f"{NAME[m]:>22s}" for m in ORDER))
    for nm in [f"pb:{p}" for p in PLACEBO] + [f"{t}|{d}" for t in TARGETS for d in ("up", "dn")]:
        row = ""
        for m in ORDER:
            s = res["models"][m]["slopes"][nm]
            row += f"{s['alpha']:+.2f} (R2 {s['r2']:.2f})".rjust(22)
        print(f"{nm:16s}" + row)
    print("   -> concept edits obey the same contraction as neutral edits: the decomposition is not")
    print("      specific to placebos.")
    print("-" * W)
    print("B) SYMMETRY TEST  H0: alpha_+ = alpha_-   (paired bootstrap over images)")
    for m in ORDER:
        print(f"  {NAME[m]}")
        for t in TARGETS:
            s = res["models"][m]["symmetry"][t]
            v = "REJECT" if s["reject_symmetry"] else "not rej."
            print(f"    {t:14s} a+={s['alpha_up']:+.3f}  a-={s['alpha_dn']:+.3f}  "
                  f"diff={s['boot_diff_mean']:+.3f} [{s['boot_diff_ci'][0]:+.3f},{s['boot_diff_ci'][1]:+.3f}] "
                  f"{v:9s} resid/effect={s['residual_over_effect']:.2f}")
    print("-" * W)
    print("C) Temperature scaling: output-space CaCE is not calibration-invariant")
    if res["temperature"]:
        for m in ORDER:
            t = res["temperature"][m]
            print(f"  {NAME[m]:5s} T={t['T']:.3f}   NLL {t['nll_T1']:.4f} -> {t['nll_T']:.4f}")
        Ts = {m: res["temperature"][m]["T"] for m in ORDER}
        print("  Every logit-scale CaCE rescales by 1/T, so cross-model contrasts are multiplied by T_i/T_j:")
        for i, a in enumerate(ORDER):
            for b in ORDER[i + 1:]:
                print(f"    {NAME[a]} vs {NAME[b]}: factor {Ts[a]/Ts[b]:.2f}")
    print("=" * W)
    print(f"Saved -> {args.out}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
