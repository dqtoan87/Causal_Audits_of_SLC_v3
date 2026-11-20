#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C47: test of the common odds ratio assumption of the conditional logistic model in C41.

Why. C41 imposes a single beta_k across all information strata. The half-sample refit (C43) speaks
only to the stability of the estimate, not to effect modification: two random halves can look alike
while the effect differs sharply between men and women, or between anatomical sites. Testing effect
modification requires putting interaction terms into the likelihood itself.

Key point on identification. In a conditional model, the main effect of a variable that is constant
within a patient (sex, age) is cancelled by the stratum, but its interaction with the exposure is
not, because the exposure varies within the stratum, so the product x_ij * S_i also varies. Effect
modification by patient characteristics is therefore estimable, even though the main effect is not.

    logit Pr(Y_ij=1 | ...) = a_i + beta*x_ij + delta*(x_ij * S_ij) + gamma' V_ij

delta is the log of the ratio of odds ratios between the two groups. delta = 0 means the common odds
ratio holds on that axis. Reports delta, the cluster sandwich confidence interval, and each group's
own odds ratio derived from (beta, beta+delta).

Four modification axes:
  sex          : constant within patient (1042/1042), estimable only through the interaction
  age_ge60     : almost constant within patient (942/1042)
  ITA_stratum  : skin tone, varies within patient; dichotomised at the median
  site_limb    : anatomical site, varies within patient; limb vs trunk/head

Out -> results/c47_heterogeneity.json
Run:  python Causal_XAI_v3/code/c47_heterogeneity.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from math import erfc, sqrt

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c1_tier1_ate as c1        # noqa: E402
import c41_within_patient as c41  # noqa: E402

BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c47_heterogeneity.json")
GROUP, OUTCOME = c1.GROUP, c1.OUTCOME

PRIMARY = [("C_color", "tbp_lv_norm_color"),
           ("D_size", "tbp_lv_areaMM2"),
           ("contrast_LB", "tbp_lv_deltaLBnorm")]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def modifiers(d):
    """Return the binary modification axes, with the labels of the two groups."""
    out = {}
    sx = d["sex"].astype(str).str.lower()
    out["sex_male"] = (sx == "male").to_numpy().astype(float), ("female", "male")
    age = pd.to_numeric(d["age_approx"], errors="coerce")
    out["age_ge60"] = (age >= 60).to_numpy().astype(float), ("under 60", "60 or older")
    # ITA_stratum la nhan tu phan vi dang chuoi (ITA_Q1_dark .. ITA_Q4_light), khong phai so.
    ita = d["ITA_stratum"].astype(str)
    out["ita_light"] = ita.isin(["ITA_Q3", "ITA_Q4_light"]).to_numpy().astype(float), ("darker half", "lighter half")
    site = d["anatom_site_general"].astype(str).str.lower()
    out["site_limb"] = site.str.contains("extrem").to_numpy().astype(float), ("head or trunk", "extremity")
    return out


def build(df, col, mod_key=None):
    """Design matrix of C41, optionally with an interaction column x * S at position 1."""
    df = df.copy()
    df["age_band"] = pd.to_numeric(df["age_approx"], errors="coerce")
    x = pd.to_numeric(df[col], errors="coerce")
    ok = x.notna()
    q1, q3 = x.quantile(1 / 3), x.quantile(2 / 3)
    T = pd.Series(np.nan, index=df.index)
    T[ok & (x <= q1)] = 0.0
    T[ok & (x >= q3)] = 1.0
    d = df.loc[T.notna()].copy(); d["_T"] = T[T.notna()].to_numpy()
    g = d.groupby(GROUP)[OUTCOME].agg(["sum", "count"])
    keep = g[(g["sum"] > 0) & (g["sum"] < g["count"])].index
    d = d[d[GROUP].isin(keep)].reset_index(drop=True)
    if not len(d):
        return None

    cols = [d["_T"].rename("T")]
    labels = ["T"]
    S = None
    if mod_key is not None:
        S, names = modifiers(d)[mod_key]
        cols.append(pd.Series(d["_T"].to_numpy() * S, name="TxS"))
        labels.append("TxS")
        # the main effect of S is included only when S varies within the stratum; if constant, the stratum
        # has absorbed it and the column is dropped at the variation check below.
        cols.append(pd.Series(S, name="S")); labels.append("S")
    cat = d[c41.WITHIN_CONF].astype("object").fillna("missing")
    du = pd.get_dummies(cat, prefix=c41.WITHIN_CONF, drop_first=True).astype(float)
    cols.append(du); labels += list(du.columns)
    X = pd.concat(cols, axis=1)

    y = d[OUTCOME].to_numpy().astype(float)
    codes, _ = pd.factorize(d[GROUP])
    order = np.argsort(codes, kind="stable")
    X = X.to_numpy().astype(float)[order]; y = y[order]; codes = codes[order]
    Sv = S[order] if S is not None else None
    bounds = np.searchsorted(codes, np.arange(codes.max() + 2))
    strata = [np.arange(bounds[i], bounds[i + 1]) for i in range(codes.max() + 1)]
    strata = [s for s in strata if len(s) > 1]

    var = np.zeros(X.shape[1])
    for s in strata:
        var += X[s].var(0) * len(s)
    drop = np.array([j for j in np.where(var < 1e-10)[0] if j > (1 if mod_key else 0)])
    if len(drop):
        X = np.delete(X, drop, axis=1)
        labels = [l for j, l in enumerate(labels) if j not in set(drop.tolist())]
    return X, y, strata, labels, Sv, d


def fit(X, y, strata):
    beta, r = c41.fit_clogit(X, y, strata)
    se = c41.sandwich_se(beta, X, y, strata)
    return beta, se, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    data = args.data or json.load(open(os.path.join(RES, "tier1_ate.json")))["data"]
    t0 = time.time()
    log(f"Load {data}")
    df = pd.read_csv(data, low_memory=False)
    df["ITA_stratum"] = c1.compute_ita(df)

    res = {"design": ("Conditional logistic regression stratified by patient, with an "
                      "exposure-by-modifier interaction. Main effects of patient-constant "
                      "modifiers are absorbed by the strata; their interaction with the "
                      "exposure is not, because the exposure varies within a stratum. The "
                      "reported delta is the log ratio of odds ratios between modifier groups; "
                      "delta = 0 is the common-odds-ratio restriction the primary model imposes."),
           "modifier_axes": {
               "sex_male": "recorded sex, constant within every patient",
               "age_ge60": "age at or above 60, constant within 942 of 1,042 patients",
               "ita_light": "the lighter half of the individual typology angle quartiles, varies within patient",
               "site_limb": "extremity versus head or trunk, varies within patient"},
           "concepts": {}}

    for cname, col in PRIMARY:
        base = build(df, col, None)
        Xb, yb, sb, _lab, _S, _d = base
        bb, seb, _r = fit(Xb, yb, sb)
        log(f"{cname}: common OR = {np.exp(bb[0]):.3f}  (strata={len(sb)}, cases={int(yb.sum())})")
        entry = {"column": col, "common_or": float(np.exp(bb[0])),
                 "common_log_or": float(bb[0]), "n_strata": int(len(sb)),
                 "n_cases": int(yb.sum()), "axes": {}}
        for key in ["sex_male", "age_ge60", "ita_light", "site_limb"]:
            out = build(df, col, key)
            if out is None:
                continue
            X, y, st, lab, S, d = out
            if "TxS" not in lab:
                entry["axes"][key] = {"error": "interaction has no within-stratum variation"}
                continue
            n1c = int(((S == 1) & (y == 1)).sum()); n0c = int(((S == 0) & (y == 1)).sum())
            v = sum(X[sl][:, lab.index("TxS")].var() * len(sl) for sl in st)
            if min(n0c, n1c) < 10 or v < 1e-10:
                entry["axes"][key] = {"error": "modifier degenerate on this sample",
                                      "n_cases_group0": n0c, "n_cases_group1": n1c,
                                      "within_stratum_variance_of_interaction": float(v)}
                log(f"  {key:10s} bo qua: ca {n0c}/{n1c}, bien thien trong tang {v:.3g}")
                continue
            beta, se, r = fit(X, y, st)
            j = lab.index("TxS")
            b0, d0 = float(beta[0]), float(beta[j])
            s0, sd = float(se[0]), float(se[j])
            z = d0 / sd if np.isfinite(sd) and sd > 0 else np.nan
            # count cases per modification group, to see which axis carries enough information
            n1 = int(((S == 1) & (y == 1)).sum()); n0 = int(((S == 0) & (y == 1)).sum())
            entry["axes"][key] = {
                "or_group0": float(np.exp(b0)), "or_group1": float(np.exp(b0 + d0)),
                "delta_log_or": d0, "se_delta": sd,
                "ratio_of_ors": float(np.exp(d0)),
                "ci_ratio_of_ors": [float(np.exp(d0 - 1.96 * sd)), float(np.exp(d0 + 1.96 * sd))]
                                   if np.isfinite(sd) else None,
                "z": float(z) if np.isfinite(z) else None,
                "p": float(erfc(abs(z) / sqrt(2))) if np.isfinite(z) else None,
                "n_cases_group0": n0, "n_cases_group1": n1,
                "converged": bool(r.success)}
            a = entry["axes"][key]
            log(f"  {key:10s} OR {a['or_group0']:.2f} vs {a['or_group1']:.2f} | "
                f"ratio {a['ratio_of_ors']:.2f} {a['ci_ratio_of_ors']} p={a['p']:.3f} "
                f"| cases {n0}/{n1}")
        res["concepts"][cname] = entry

    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"saved -> {OUT} ({res['runtime_sec']}s)")


if __name__ == "__main__":
    main()
