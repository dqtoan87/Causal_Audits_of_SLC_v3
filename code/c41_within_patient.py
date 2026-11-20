#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C41: Tier 1 with a within-patient design (conditional logistic, stratified by patient_id).

Why. The backdoor adjustment of C1 only controls measured confounding; the E-value only quantifies
exposure to unmeasured confounding, it does not remove it. In ISIC-2024 each patient contributes
hundreds of tiles, and 258 patients have both benign and malignant tiles, carrying 392/393 malignant
cases. Stratifying by patient therefore removes all patient-level confounding, measured or not, at
the cost of exactly 1 malignant case. Metadata check: acquisition site and sex are constant within
each patient (1042/1042), age is constant in 942/1042. All three are absorbed by the design.
Anatomical site and ITA do vary within a patient, so they still need adjustment, and the script does that.

Estimation. Exact conditional logistic likelihood (elementary symmetric polynomials), optimised with
BFGS using the analytic gradient. Standard errors are the cluster sandwich with the patient as the
cluster. Here each stratum is exactly one cluster, so the cluster variance follows directly from the
design and is not an extra correction. A cluster bootstrap (resampling patients) is added as a check.

Reading the estimand correctly. This is a within-patient conditional odds ratio, which differs from the
estimand of Table I (a tile-level standardized risk difference over the whole cohort). With a rare
outcome the OR approximates the RR, so the two are comparable in direction and relative magnitude,
not as a re-estimate of the same number. The cost: the within-patient combinations are very
unbalanced (median 435 tiles, 1 to 2 malignant cases), so the confidence intervals are wider than in Table I.

Out -> results/c41_within_patient.json
Run:  python Causal_XAI_v3/code/c41_within_patient.py [--boot 500] [--data CSV]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c1_tier1_ate as c1   # noqa: E402  (CONCEPTS, compute_ita, data paths)

BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c41_within_patient.json")

GROUP, OUTCOME = c1.GROUP, c1.OUTCOME
# Confounders that vary within a patient. The constant ones (age, sex, attribution) are absorbed by
# the strata and cancel out of the conditional likelihood, so they are not included.
WITHIN_CONF = ["anatom_site_general", "ITA_stratum", "age_band"]
# age_approx is recorded in 5-year steps. It is constant in 942 patients and varies in 100.
# Included in V as dummies: in constant strata the column drops out because it has no within-stratum
# variation; in varying strata it adjusts exactly that variation.
SEED = 42


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------------
# Conditional logistic likelihood
# ----------------------------------------------------------------------------------
def _esp_and_incl(w, k):
    """Return (e_k, p) where e_k = elementary symmetric polynomial of degree k of w, and
    p_i = w_i·e_{k-1}(w without i)/e_k = probability that unit i is among the k selected cases.

    Uses prefix/suffix DP, so the cost is O(n·k²) and unstable division is avoided."""
    n = len(w)
    # prefix[j, i] = e_j(w[:i]);  suffix[j, i] = e_j(w[i:])
    pre = np.zeros((k + 1, n + 1)); pre[0, :] = 1.0
    for i in range(1, n + 1):
        wi = w[i - 1]
        for j in range(min(k, i), 0, -1):
            pre[j, i] = pre[j, i - 1] + wi * pre[j - 1, i - 1]
    suf = np.zeros((k + 1, n + 2)); suf[0, :] = 1.0
    for i in range(n, 0, -1):
        wi = w[i - 1]
        for j in range(min(k, n - i + 1), 0, -1):
            suf[j, i] = suf[j, i + 1] + wi * suf[j - 1, i + 1]
    ek = pre[k, n]
    if not np.isfinite(ek) or ek <= 0:
        return ek, np.zeros(n)
    # e_{k-1}(w without i) = Σ_j pre[j, i-1]·suf[k-1-j, i+1]
    incl = np.empty(n)
    for i in range(1, n + 1):
        s = 0.0
        for j in range(0, k):
            s += pre[j, i - 1] * suf[k - 1 - j, i + 1]
        incl[i - 1] = w[i - 1] * s / ek
    return ek, incl


def clogit_nll(beta, X, y, strata):
    """-log conditional likelihood and gradient, plus per-stratum scores (for the sandwich)."""
    eta = X @ beta
    nll = 0.0
    grad = np.zeros_like(beta)
    scores = []
    for idx in strata:
        e = eta[idx]
        e = e - e.max()                      # invariant: the ratio is homogeneous of degree k in w
        w = np.exp(e)
        yi = y[idx]
        k = int(yi.sum())
        ek, incl = _esp_and_incl(w, k)
        if ek <= 0 or not np.isfinite(ek):
            continue
        nll -= float(e[yi == 1].sum() - np.log(ek))
        g = (X[idx] * (yi - incl)[:, None]).sum(0)   # stratum score
        grad -= g
        scores.append(g)
    return nll, grad, np.array(scores)


def fit_clogit(X, y, strata, beta0=None, maxiter=200):
    p = X.shape[1]
    b0 = np.zeros(p) if beta0 is None else beta0
    f = lambda b: clogit_nll(b, X, y, strata)[:2]
    r = minimize(f, b0, jac=True, method="BFGS",
                 options={"maxiter": maxiter, "gtol": 1e-6})
    return r.x, r


def sandwich_se(beta, X, y, strata):
    """Cluster standard errors. Each stratum = one patient = one independent cluster, so
    V = H⁻¹ (Σ_s g_s g_sᵀ) H⁻¹ is the cluster variance implied by the design."""
    _, _, S = clogit_nll(beta, X, y, strata)
    eps = 1e-5
    p = len(beta)
    H = np.zeros((p, p))
    for j in range(p):
        bp = beta.copy(); bp[j] += eps
        bm = beta.copy(); bm[j] -= eps
        H[:, j] = (clogit_nll(bp, X, y, strata)[1] - clogit_nll(bm, X, y, strata)[1]) / (2 * eps)
    H = 0.5 * (H + H.T)
    try:
        Hinv = np.linalg.pinv(H)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan)
    meat = S.T @ S
    V = Hinv @ meat @ Hinv
    d = np.diag(V)
    return np.sqrt(np.where(d > 0, d, np.nan))


# ----------------------------------------------------------------------------------
# Prepare one concept
# ----------------------------------------------------------------------------------
def prepare(df, col, adjust=True):
    """Cut tertiles on the whole cohort (as in C1), keep Q3 vs Q1, then keep only strata that still
    contain both classes. Homogeneous strata contribute nothing to the conditional likelihood."""
    df = df.copy()
    df["age_band"] = pd.to_numeric(df["age_approx"], errors="coerce")
    x = pd.to_numeric(df[col], errors="coerce")
    ok = x.notna()
    q1, q3 = x.quantile(1 / 3), x.quantile(2 / 3)
    T = pd.Series(np.nan, index=df.index)
    T[ok & (x <= q1)] = 0.0
    T[ok & (x >= q3)] = 1.0
    sub = T.notna()
    d = df.loc[sub].copy(); d["_T"] = T[sub].to_numpy()

    g = d.groupby(GROUP)[OUTCOME].agg(["sum", "count"])
    keep = g[(g["sum"] > 0) & (g["sum"] < g["count"])].index
    d = d[d[GROUP].isin(keep)]
    if len(d) == 0:
        return None

    cols = [d["_T"].rename("T")]
    if adjust:
        cat = d[WITHIN_CONF].astype("object").fillna("missing")
        du = pd.get_dummies(cat, prefix=WITHIN_CONF, drop_first=True).astype(float)
        cols.append(du)
    X = pd.concat(cols, axis=1)

    y = d[OUTCOME].to_numpy().astype(float)
    codes, _ = pd.factorize(d[GROUP])
    order = np.argsort(codes, kind="stable")
    X = X.to_numpy()[order]; y = y[order]; codes = codes[order]
    bounds = np.searchsorted(codes, np.arange(codes.max() + 2))
    strata = [np.arange(bounds[i], bounds[i + 1]) for i in range(codes.max() + 1)]
    strata = [s for s in strata if len(s) > 1]

    # drop columns with no variation within any stratum (not identifiable)
    if X.shape[1] > 1:
        var = np.zeros(X.shape[1])
        for s in strata:
            var += X[s].var(0) * len(s)
        drop = np.where(var < 1e-10)[0]
        drop = drop[drop != 0]
        if len(drop):
            X = np.delete(X, drop, axis=1)
    return X, y, strata, d


def run_concept(df, cname, col, boot, rng):
    prep = prepare(df, col, adjust=True)
    if prep is None:
        return {"concept": cname, "column": col, "error": "no contributing stratum"}
    X, y, strata, d = prep
    n_case = int(y.sum())
    log(f"  {cname:15s} rows={len(y):>7,} strata={len(strata):>4} cases={n_case:>4} cols={X.shape[1]}")

    beta, r = fit_clogit(X, y, strata)
    se = sandwich_se(beta, X, y, strata)
    b, sb = float(beta[0]), float(se[0])
    z = b / sb if np.isfinite(sb) and sb > 0 else np.nan
    from math import erfc, sqrt
    pval = float(erfc(abs(z) / sqrt(2))) if np.isfinite(z) else float("nan")

    # stratification-only version, no adjustment for site/ITA
    Xu, yu, su, _ = prepare(df, col, adjust=False)
    bu, _ = fit_clogit(Xu, yu, su)

    # cluster bootstrap: resample patients (i.e. strata), keeping all of their tiles
    bs = []
    if boot:
        for t in range(boot):
            pick = rng.randint(0, len(strata), len(strata))
            Xb = np.concatenate([X[strata[i]] for i in pick])
            yb = np.concatenate([y[strata[i]] for i in pick])
            off, sb2 = 0, []
            for i in pick:
                n = len(strata[i]); sb2.append(np.arange(off, off + n)); off += n
            try:
                bb, _ = fit_clogit(Xb, yb, sb2, beta0=beta, maxiter=80)
                bs.append(float(bb[0]))
            except Exception:
                pass
            if (t + 1) % 100 == 0:
                log(f"      boot {t+1}/{boot}")
    bs = np.array(bs)

    return {
        "concept": cname, "column": col,
        "n_rows": int(len(y)), "n_strata": int(len(strata)), "n_cases": n_case,
        "n_covariate_cols": int(X.shape[1]),
        "log_or": b, "or": float(np.exp(b)),
        "se_cluster": sb,
        "ci_cluster": [float(np.exp(b - 1.96 * sb)), float(np.exp(b + 1.96 * sb))]
                      if np.isfinite(sb) else None,
        "z": float(z) if np.isfinite(z) else None, "p": pval,
        "or_stratum_only": float(np.exp(bu[0])),
        "boot_n": int(len(bs)),
        "boot_ci": [float(np.exp(np.percentile(bs, 2.5))),
                    float(np.exp(np.percentile(bs, 97.5)))] if len(bs) > 50 else None,
        "converged": bool(r.success),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=c1.ISIC_META_CSV if hasattr(c1, "ISIC_META_CSV") else None)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--no-age", action="store_true",
                    help="drop age_band from V; used to measure how much the age adjustment contributes, "
                         "writes c41_within_patient_noage.json")
    ap.add_argument("--drop-indeterminate", action="store_true",
                    help="exclude the 114 lesions labelled Indeterminate instead of merging them into the negative class; writes c41_within_patient_noindet.json")
    ap.add_argument("--max-patients", type=int, default=0,
                    help="keep only the first N discordant patients (smoke test)")
    args = ap.parse_args()
    data = args.data or json.load(open(os.path.join(RES, "tier1_ate.json")))["data"]

    t0 = time.time()
    log(f"Load {data}")
    df = pd.read_csv(data, low_memory=False)
    if args.max_rows:
        keep = df.groupby(GROUP, group_keys=False).head(args.max_rows // df[GROUP].nunique() + 1)
        df = pd.concat([keep, df[df[OUTCOME] == 1]]).drop_duplicates("isic_id").reset_index(drop=True)
        log(f"  smoke test: {len(df):,} rows")
    out_path = OUT
    if args.no_age:
        # V without age_band: age is still absorbed in the 942 constant-age patients; only the 100
        # patients with varying age are left unadjusted. This is the configuration against which the
        # "reduced by how many percent" figure in the paper is compared.
        global WITHIN_CONF
        WITHIN_CONF = [c for c in WITHIN_CONF if c != "age_band"]
        out_path = os.path.join(RES, "c41_within_patient_noage.json")
        log("  dropped age_band from V")
    if args.drop_indeterminate:
        n0 = len(df)
        df = df[df["iddx_1"].astype(str) != "Indeterminate"].reset_index(drop=True)
        out_path = os.path.join(RES, "c41_within_patient_noindet.json")
        log(f"  excluded {n0-len(df)} Indeterminate lesions; {len(df):,} rows remain")
    df["ITA_stratum"] = c1.compute_ita(df)
    if args.max_patients:
        gg = df.groupby(GROUP)[OUTCOME].agg(["sum", "count"])
        keep = gg[(gg["sum"] > 0) & (gg["sum"] < gg["count"])].index[: args.max_patients]
        df = df[df[GROUP].isin(keep)].reset_index(drop=True)
        log(f"  smoke test: {args.max_patients} discordant patients, {len(df):,} tiles")
    g = df.groupby(GROUP)[OUTCOME].agg(["sum", "count"])
    disc = g[(g["sum"] > 0) & (g["sum"] < g["count"])]
    log(f"  tiles={len(df):,} patients={df[GROUP].nunique()} cases={int(df[OUTCOME].sum())}")
    log(f"  discordant patients={len(disc)} carrying {int(disc['sum'].sum())} cases "
        f"({100*disc['sum'].sum()/max(df[OUTCOME].sum(),1):.1f}% of all)")

    rng = np.random.RandomState(SEED)

    def snapshot(concepts, status):
        json.dump({"phase": "C41_tier1_within_patient", "status": status,
                   "note": "Written incrementally during the run; the final file is c41_within_patient.json.",
                   "concepts": concepts}, open(out_path + ".partial", "w"), indent=1)

    # Pass 1: point estimates and cluster (sandwich) standard errors. This is the main inference:
    # each stratum is exactly one patient, so the cluster variance comes from this fit itself.
    # It runs fast, so the core results are available early even though the bootstrap takes long.
    log("PASS 1/2: point estimates + cluster sandwich (no bootstrap)")
    out = []
    for c in c1.CONCEPTS:
        out.append(run_concept(df, c["name"], c["col"], 0, rng))
        snapshot(out, "pass1_point_estimates")
    log("PASS 1/2 done: point estimates and cluster CIs are in the .partial file")

    # Pass 2: cluster bootstrap as a check on the sandwich. Slow, and no conclusion depends on it.
    if args.boot:
        log(f"PASS 2/2: cluster bootstrap, {args.boot} replicates per concept (check)")
        for i, c in enumerate(c1.CONCEPTS):
            out[i] = run_concept(df, c["name"], c["col"], args.boot, rng)
            snapshot(out, f"pass2_bootstrap_{i+1}_of_{len(c1.CONCEPTS)}")

    # compare direction with Table I (different estimands; compare only direction and relative magnitude)
    try:
        t1 = {x["concept"]: x for x in json.load(open(os.path.join(RES, "tier1_ate.json")))["concepts"]}
        for o in out:
            ref = t1.get(o["concept"], {})
            for k in ("rr", "risk_ratio", "RR"):
                if k in ref:
                    o["marginal_rr_table1"] = ref[k]; break
    except Exception as e:
        log(f"  (could not compare with tier1_ate.json: {e})")

    res = {"phase": "C41_tier1_within_patient",
           "data": data,
           "design": ("Conditional logistic regression stratified by patient_id. Strata absorb every "
                      "patient-level confounder, measured or not, including acquisition site, sex and "
                      "(for 942 of 1042 patients) age, all of which are constant within patient. "
                      "Anatomical site and ITA stratum vary within patient and are adjusted for. "
                      "Standard errors are the cluster sandwich with the patient as the cluster, which "
                      "here coincides with the stratum."),
           "estimand": ("Within-patient conditional odds ratio for the top-versus-bottom tertile of the "
                        "proxy. This is NOT the estimand of Table I, which is a tile-level standardized "
                        "risk difference over the whole cohort. With a rare outcome the odds ratio "
                        "approximates a risk ratio, so the two are comparable in direction and rough "
                        "magnitude only."),
           "n_rows_total": int(len(df)), "n_patients_total": int(df[GROUP].nunique()),
           "n_cases_total": int(df[OUTCOME].sum()),
           "n_discordant_patients": int(len(disc)),
           "n_cases_in_discordant": int(disc["sum"].sum()),
           "within_patient_confounders": WITHIN_CONF,
           "boot": args.boot, "seed": SEED,
           "concepts": out, "runtime_sec": round(time.time() - t0, 1)}
    json.dump(res, open(out_path, "w"), indent=1)
    if os.path.exists(out_path + ".partial"):
        os.remove(out_path + ".partial")
    log(f"saved -> {out_path}  ({res['runtime_sec']}s)")
    print()
    print(f"{'concept':16s} {'OR':>7s} {'95% CI (cluster)':>22s} {'p':>10s} {'OR strat-only':>14s}")
    for o in out:
        if "or" not in o:
            continue
        ci = o["ci_cluster"] or [float("nan")] * 2
        print(f"{o['concept']:16s} {o['or']:7.3f} [{ci[0]:8.3f},{ci[1]:8.3f}] {o['p']:10.2e} "
              f"{o['or_stratum_only']:14.3f}")


if __name__ == "__main__":
    main()
