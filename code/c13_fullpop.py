#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C13: full-population Tier-2 analysis (no p_full threshold).

Problem: the Tier-2 results (C2/C3/C6/C10/C11/C12) are all computed on a filtered set
  {truly malignant images} ∩ {p_full ≥ 0.10}.
This is a model-dependent conditioning event: the retained set differs between models
(78/78 for ERM & Slim, 62/78 for Inv), so the estimand is not a population effect but the
effect within the group each model already scores above the threshold. A threshold-free
analysis is needed to check whether the conclusions are robust to selection conditioning.

C13 reruns all interventions on all 178 evaluation images, with no filtering, and reports by
5 subgroups:
    all (178) | malignant (78) | benign (100) | malignant & p≥0.1 | malignant & p≥0.7
on both scales: probability (Δp) and logit (Δlogit), each in a signed and an absolute-value version.

  E[Δ]  answers "which direction does the intervention push the prediction" (can cancel if signs differ).
  E|Δ|  answers "how much does the intervention move the prediction" (does not cancel). This is the form
        used for the artifact penalty P(f) in Section III-E.

Note on the logit scale off-threshold: the logit is clipped at eps=1e-6 (|logit| ≤ 13.8). For an image the
model already scores at p≈0, a probability change too small to matter clinically still produces a large Δlogit.
That is exactly why the 0.1 threshold exists. So for the benign group, the probability scale is the one to read,
and the logit scale is reported alongside only for comparison.

The edits use exactly the same RNG convention as C2 (artifact: RandomState(SEED+i)) and C10
(placebo: RandomState(SEED+i+7)), so on the subset {malignant, p≥0.1} the results must reproduce
c2/c10. This is the script's self-check.

Out -> results/c13_fullpop.json
Run:  .venv/bin/python Causal_XAI_v3/code/c13_fullpop.py [--n-eval 200] [--boot 2000]
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
import c2_cace_artifact as c2m      # noqa: E402  (ARTIFACTS, COUPLING)
import audit_cohort               # noqa: E402
import c3_cace_concept as c3m       # noqa: E402  (soft_mask, make_edit, model_prob, logit, DOSES)
import c10_multiplacebo as c10      # noqa: E402  (PLACEBOS)

torch.set_num_threads(min(8, os.cpu_count() or 8))
p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c13_fullpop.json")
import audited_models                # noqa: E402  (checkpoints resolved in one place)
CKPTS = audited_models.CKPTS         # absolute paths to the fixed set
SEED, N_EVAL, BOOT = 42, 200, 2000
ORDER = ["erm", "slim", "causal"]
NAME = {"erm": "ERM", "slim": "Slim", "causal": "Inv"}

ART = ["hair_light", "hair", "hair_heavy", "ruler", "patch", "vignette", "jitter", "identity"]
HELDOUT = ["hair_light", "hair", "hair_heavy", "ruler"]
COUPLED = ["patch", "vignette", "jitter"]
CONCEPT = [f"{c}@{k}" for c in ("contrast", "color_var") for k in range(3)]
PLACEBO = ["flip", "jitter", "hue", "noise"]          # placebo names (distinct from the artifact 'jitter')
TOPDOSE = ["contrast@2", "color_var@2"]


def ename(kind, n):
    return f"{kind}:{n}"


def build_all_edits(imgs01):
    """Edit once per image, shared by the 3 models. Keeps the RNG convention of C2 and C10."""
    edits = []
    for i, x01 in enumerate(imgs01):
        d = {}
        rng_a = np.random.RandomState(SEED + i)                      # as in C2
        for a in ART:
            d[ename("art", a)] = c2m.ARTIFACTS[a](x01, rng_a)
        w, skin, lm = c3m.soft_mask(x01)
        for c in ("contrast", "color_var"):
            for k in range(3):
                d[ename("cpt", f"{c}@{k}")] = c3m.make_edit(x01, c, c3m.DOSES[k], w, skin, lm)
        for pnm, fn in c10.PLACEBOS.items():                          # as in C10
            d[ename("pb", pnm)] = fn(x01, w, skin, np.random.RandomState(SEED + i + 7))
        edits.append(d)
    return edits, list(edits[0].keys())


@torch.no_grad()
def eval_model(model, imgs01, edits, names):
    """Return p0 (n,) and pe (n, n_edit), with no filtering at all."""
    n = len(imgs01)
    p0 = np.empty(n); pe = np.empty((n, len(names)))
    for i, x01 in enumerate(imgs01):
        batch = torch.stack([x01] + [edits[i][nm] for nm in names])
        ps = c3m.model_prob(model, batch)
        p0[i] = ps[0]; pe[i] = ps[1:]
    return p0, pe


def summarize(v, boot, seed):
    m, lo, hi = p2.boot_mean_ci(np.asarray(v, float), boot, seed)
    return dict(mean=float(m), ci=[float(lo), float(hi)], sig=bool(lo > 0 or hi < 0), n=int(len(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    print(f"##### C13 full-population Tier-2 {time.strftime('%H:%M:%S')} #####", flush=True)

    imgs01, labels, _pid, _ids = audit_cohort.load()   # fixed evaluation set
    labels = np.array(labels)
    n = len(imgs01)
    print(f"eval images {n}  malignant {int((labels == 1).sum())}  benign {int((labels == 0).sum())}",
          flush=True)

    edits, names = build_all_edits(imgs01)
    print(f"edits/image = {len(names)}  ({time.time()-t0:.0f}s)", flush=True)

    P0, PE = {}, {}
    for m in ORDER:
        model = p1.build_model("erm" if m == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(CKPTS[m], map_location=DEVICE))
        model.eval()
        P0[m], PE[m] = eval_model(model, imgs01, edits, names)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"[{m}] p0(all)={P0[m].mean():.3f}  p0(mal)={P0[m][labels==1].mean():.3f}  "
              f"p0(ben)={P0[m][labels==0].mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)

    res = {
        "n_images": int(n), "n_malignant": int((labels == 1).sum()),
        "n_benign": int((labels == 0).sum()), "boot": args.boot,
        "note": ("Full-population Tier-2, no p_full threshold. Subgroups are defined per model "
                 "where they depend on p_full. Logit is clipped at eps=1e-6."),
        "subgroups": {}, "models": {},
    }

    for m in ORDER:
        p0 = P0[m]; pe = PE[m]
        l0 = c3m.logit(p0); le = c3m.logit(pe)
        dp = pe - p0[:, None]                 # (n, n_edit) signed Δp
        dl = le - l0[:, None]                 # (n, n_edit) signed Δlogit
        subs = {
            "all": np.ones(n, bool),
            "malignant": labels == 1,
            "benign": labels == 0,
            "malignant_p010": (labels == 1) & (p0 >= 0.10),
            "malignant_p070": (labels == 1) & (p0 >= 0.70),
        }
        res["subgroups"][m] = {k: int(v.sum()) for k, v in subs.items()}
        mr = {"p0_mean": {k: float(p0[v].mean()) for k, v in subs.items()}, "edits": {}}

        for j, nm in enumerate(names):
            e = {}
            for sk, sm in subs.items():
                if sm.sum() < 3:
                    continue
                e[sk] = {
                    "dp": summarize(dp[sm, j], args.boot, SEED),
                    "abs_dp": summarize(np.abs(dp[sm, j]), args.boot, SEED),
                    "dlogit": summarize(dl[sm, j], args.boot, SEED),
                    "abs_dlogit": summarize(np.abs(dl[sm, j]), args.boot, SEED),
                }
            mr["edits"][nm] = e

        # ---- artifact penalty: pooled per-image |Δ| over the 4 held-out edits (Section III-E)
        idx_ho = [names.index(ename("art", a)) for a in HELDOUT]
        idx_cp = [names.index(ename("art", a)) for a in COUPLED]
        mr["aggregate"] = {}
        for gname, idx in (("heldout", idx_ho), ("coupled", idx_cp)):
            g = {}
            for sk, sm in subs.items():
                if sm.sum() < 3:
                    continue
                g[sk] = {"abs_dlogit": summarize(np.abs(dl[np.ix_(sm, idx)]).ravel(), args.boot, SEED),
                         "abs_dp": summarize(np.abs(dp[np.ix_(sm, idx)]).ravel(), args.boot, SEED)}
            mr["aggregate"][gname] = g

        # ---- concept scores: A_raw and A_corr under each placebo (top dose, both concepts)
        idx_cpt = [names.index(ename("cpt", c)) for c in TOPDOSE]
        mr["A_raw"] = {}; mr["A_corr"] = {}
        for sk, sm in subs.items():
            if sm.sum() < 3:
                continue
            raw = dl[np.ix_(sm, idx_cpt)].mean(axis=1)          # mean over the 2 concepts, per image
            mr["A_raw"][sk] = summarize(raw, args.boot, SEED)
            mr["A_corr"][sk] = {}
            for pnm in PLACEBO:
                jp = names.index(ename("pb", pnm))
                mr["A_corr"][sk][pnm] = summarize(raw - dl[sm, jp], args.boot, SEED)

        # ---- sign dissociation: concept response vs neutral response, same subgroup
        mr["sign_dissociation"] = {}
        for sk, sm in subs.items():
            if sm.sum() < 3:
                continue
            cpt_mean = float(dl[np.ix_(sm, idx_cpt)].mean())
            neu = [float(dl[sm, names.index(ename("pb", p))].mean()) for p in PLACEBO]
            mr["sign_dissociation"][sk] = {
                "concept_mean": cpt_mean, "neutral_means": dict(zip(PLACEBO, neu)),
                "dissociates": bool(cpt_mean > 0 and max(neu) <= 0)}
        # ---- baseline-confidence dependence of generic edit-sensitivity
        # If g depends on p_full (contraction toward the decision boundary), then g is not a function of the footprint M,
        # and that is the mechanical explanation for why test (T) in C12 fails. The opposite-dose design is immune
        # to this source of dependence because both edit directions start from the same image, with the same p_full.
        def _pear(a, b):
            a = np.asarray(a, float); b = np.asarray(b, float)
            if a.std() < 1e-12 or b.std() < 1e-12:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])

        conf = {}
        qs = np.quantile(p0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for pnm in PLACEBO:
            jp = names.index(ename("pb", pnm))
            byq = []
            for q in range(5):
                sel = (p0 >= qs[q]) & (p0 <= qs[q + 1] if q == 4 else p0 < qs[q + 1])
                byq.append({"p0_mid": float(p0[sel].mean()) if sel.sum() else None,
                            "dlogit": float(dl[sel, jp].mean()) if sel.sum() else None,
                            "n": int(sel.sum())})
            # OLS Δlogit ~ a + k·logit0 : k<0 = contraction toward the decision boundary
            k, a = np.polyfit(l0, dl[:, jp], 1)
            pred = a + k * l0
            ss = 1.0 - ((dl[:, jp] - pred) ** 2).sum() / ((dl[:, jp] - dl[:, jp].mean()) ** 2).sum()
            conf[pnm] = {"pearson_p0_vs_dlogit": _pear(p0, dl[:, jp]),
                         "pearson_logit0_vs_dlogit": _pear(l0, dl[:, jp]),
                         "ols_slope_on_logit0": float(k), "ols_intercept": float(a),
                         "ols_r2": float(ss), "by_p0_quintile": byq}
        mr["confidence_dependence"] = conf
        res["models"][m] = mr

    os.makedirs(RES, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)

    # ------------------------------ report ------------------------------
    W = 118
    print("=" * W)
    print("A) SUBGROUP SIZES (p_full-based subgroups are model-dependent)")
    hdr = f"{'subgroup':22s}" + "".join(NAME[m].rjust(14) for m in ORDER)
    print(hdr)
    for sk in ["all", "malignant", "benign", "malignant_p010", "malignant_p070"]:
        print(f"{sk:22s}" + "".join(str(res['subgroups'][m][sk]).rjust(14) for m in ORDER))

    print("-" * W)
    print("B) HELD-OUT ARTIFACT PENALTY  E|Δlogit|  (paper's P(f)), by subgroup")
    print(hdr)
    for sk in ["all", "malignant", "benign", "malignant_p010", "malignant_p070"]:
        row = ""
        for m in ORDER:
            g = res["models"][m]["aggregate"]["heldout"].get(sk)
            row += (f"{g['abs_dlogit']['mean']:.2f}".rjust(14) if g else "-".rjust(14))
        print(f"{sk:22s}" + row)

    print("-" * W)
    print("C) HELD-OUT ARTIFACT  E|Δp|  (probability scale: the readable one off-threshold)")
    print(hdr)
    for sk in ["all", "malignant", "benign", "malignant_p010", "malignant_p070"]:
        row = ""
        for m in ORDER:
            g = res["models"][m]["aggregate"]["heldout"].get(sk)
            row += (f"{g['abs_dp']['mean']:.3f}".rjust(14) if g else "-".rjust(14))
        print(f"{sk:22s}" + row)

    print("-" * W)
    print("D) HAIR (2.4% px) SIGNED Δp and Δlogit, by subgroup")
    for sk in ["all", "malignant", "benign", "malignant_p010"]:
        print(f"  {sk}")
        for m in ORDER:
            e = res["models"][m]["edits"][ename("art", "hair")].get(sk)
            if not e:
                continue
            print(f"    {NAME[m]:5s} Δp={e['dp']['mean']:+.3f}[{e['dp']['ci'][0]:+.3f},{e['dp']['ci'][1]:+.3f}]"
                  f"  Δlogit={e['dlogit']['mean']:+.2f}[{e['dlogit']['ci'][0]:+.2f},{e['dlogit']['ci'][1]:+.2f}]"
                  f"  sig={e['dp']['sig']}")

    print("-" * W)
    print("E) IDENTITY EDIT (must be exactly 0 on every subgroup)")
    for m in ORDER:
        v = [res["models"][m]["edits"][ename("art", "identity")][sk]["dlogit"]["mean"]
             for sk in ["all", "malignant", "benign"]]
        print(f"  {NAME[m]:5s} " + "  ".join(f"{x:+.1e}" for x in v))

    print("-" * W)
    print("F) CONCEPT SCORE A_raw and A_corr (top dose), by subgroup")
    for sk in ["all", "malignant", "benign", "malignant_p010", "malignant_p070"]:
        print(f"  {sk}")
        for m in ORDER:
            if sk not in res["models"][m]["A_raw"]:
                continue
            r = res["models"][m]["A_raw"][sk]
            cs = " ".join(f"{p}={res['models'][m]['A_corr'][sk][p]['mean']:+.2f}"
                          f"[{'Y' if res['models'][m]['A_corr'][sk][p]['sig'] else '.'}]"
                          for p in PLACEBO)
            print(f"    {NAME[m]:5s} A_raw={r['mean']:+.2f}[{'Y' if r['sig'] else '.'}]  {cs}")

    print("-" * W)
    print("G) SIGN DISSOCIATION (concept > 0 and every neutral <= 0)")
    for sk in ["all", "malignant", "benign", "malignant_p010"]:
        row = " ".join(f"{NAME[m]}={'YES' if res['models'][m]['sign_dissociation'][sk]['dissociates'] else 'no '}"
                       for m in ORDER)
        print(f"  {sk:22s} {row}")
    print("-" * W)
    print("H) IS GENERIC EDIT-SENSITIVITY A FUNCTION OF BASELINE CONFIDENCE?")
    print("   (if yes, g cannot be a function of the edit footprint M -> explains why test (T) fails)")
    for m in ORDER:
        cd = res["models"][m]["confidence_dependence"]
        print(f"  {NAME[m]}")
        for pnm in PLACEBO:
            r = cd[pnm]
            q = "  ".join(f"{b['p0_mid']:.2f}->{b['dlogit']:+.2f}" for b in r["by_p0_quintile"])
            print(f"    {pnm:7s} r(p0,Δlogit)={r['pearson_p0_vs_dlogit']:+.2f}  "
                  f"slope={r[chr(39)+chr(39)] if False else r['ols_slope_on_logit0']:+.2f} R2={r['ols_r2']:.2f}   by p0 quintile: {q}")
    print("=" * W)
    print(f"Saved -> {args.out}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
