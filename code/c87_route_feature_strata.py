#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C87: the route test's decisive differences split by how far the controls drive the backbone.  (exploratory)

The route-test flags sit in cells where many rad4 predictions reach the clipping bound, and c83 found that
large protocol-dependent score changes come from inputs whose pooled backbone feature exceeds ten times its
median norm on unedited images. This script asks whether the decisive differences are carried by those
images. It reads stored outputs only: the plan-protocol responses of both runs (c63_route_scores*.npz) and
the pooled feature norms that c83 recorded under the plan's protocol (c83_batch_extremes_scores.npz).
ERM and Slim use the same checkpoints in both runs, so their feature norms apply to both; the epoch-1 Inv
checkpoint decides no first-run flag, so no new model output is needed.

For each decisive difference D_ab = R(b) - R(a) of a flagged cell, on that run's analysed images (G1 and the
plan's score floor), an image is "within range" if the pooled feature of both controls a and b is at most
ten times the model's median unedited norm, and "beyond" otherwise. For each stratum: images, patients,
mean difference and a marginal 95% patient-clustered percentile interval (20,000 resamples), and the share
of the summed difference that the beyond-range images carry. The ten-times
threshold is the one c83 used; it was fixed before this split was computed. Marginal intervals, no
multiplicity correction; the strata are not the plan's analysis.

Out -> results/c87_route_feature_strata.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c87_route_feature_strata.json")
RATIO = 10.0
BOOT, SEED = 20000, 87
MODELS = ["erm", "slim", "causal"]
TKS = ["contrast@0", "contrast@2", "color_var@0", "color_var@2"]
CONTROLS = ["rad4", "rad16", "hue4", "hue16"]
DECISIVE = {"first_run": ["erm|contrast@2|rad4-rad16", "slim|color_var@2|rad4-rad16", "slim|contrast@0|rad4-rad16"],
            "second_run": ["erm|contrast@2|rad4-rad16", "causal|contrast@2|rad4-rad16", "causal|contrast@2|rad4-hue4",
                           "slim|color_var@2|rad4-rad16"]}


def boot_mean(x, pid, rng):
    """Marginal patient-clustered percentile interval of the image mean."""
    ups = np.unique(pid)
    idx = {p: np.where(pid == p)[0] for p in ups}
    s = np.array([x[idx[p]].sum() for p in ups]); n = np.array([len(idx[p]) for p in ups])
    draws = rng.randint(0, len(ups), size=(BOOT, len(ups)))
    m = s[draws].sum(1) / n[draws].sum(1)
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    t0 = time.time()
    pre = json.load(open(os.path.join(RES, "c62_preregistration.json")))
    floor = pre["criteria"]["score_floor"]
    cons = json.load(open(os.path.join(RES, "c62_construction.json")))
    n = len(cons["ids"])
    g1 = {tk: np.array([all(cons["images"][i]["targets"][tk]["controls"][c]["g1"] for c in CONTROLS)
                        for i in range(n)]) for tk in TKS}
    fz = np.load(os.path.join(RES, "c83_batch_extremes_scores.npz"), allow_pickle=True)
    fcols = [str(c) for c in fz["columns"]]
    F = {m: fz["featnorm_%s_stack_bf16" % m] for m in MODELS}
    fmed = {m: float(np.median(F[m][:, 0])) for m in MODELS}
    runs = {"first_run": "", "second_run": "_inv_retrained"}
    out = {"note": "exploratory; stored plan-protocol responses and c83 feature norms; marginal 95%% patient-clustered "
                   "percentile intervals, %d resamples, seed %d; threshold %g x median unedited norm" % (BOOT, SEED, RATIO),
           "median_unedited_feature_norm": fmed, "runs": {}}
    for run, suf in runs.items():
        d = np.load(os.path.join(RES, "c63_route_scores%s.npz" % suf), allow_pickle=True)
        cols = [str(c) for c in d["columns"]]
        assert cols == fcols and [str(i) for i in d["isic_ids"]] == [str(i) for i in fz["isic_ids"]]
        pid = np.asarray(d["patient_ids"])
        common = np.all([d["prob_" + m][:, 0] >= floor for m in MODELS], 0)
        rec = {}
        for key in DECISIVE[run]:
            m, tk, pair = key.split("|")
            a, b = pair.split("-")
            ca, cb = cols.index("%s|%s" % (tk, a)), cols.index("%s|%s" % (tk, b))
            msk = g1[tk] & common
            dl = d["dlogit_" + m]
            diff = dl[:, cb] - dl[:, ca]
            ratio = np.maximum(F[m][:, ca], F[m][:, cb]) / fmed[m]
            r = {"n_all": int(msk.sum())}
            rng = np.random.RandomState(SEED)
            est, ci = boot_mean(diff[msk], pid[msk], rng)
            r["all"] = {"est": est, "ci95": ci}
            for name, sel in (("within", ratio <= RATIO), ("beyond", ratio > RATIO)):
                mm = msk & sel
                if mm.sum() < 2:
                    r[name] = {"n_images": int(mm.sum()), "n_patients": int(len(set(pid[mm])))}
                    continue
                rng = np.random.RandomState(SEED)
                est, ci = boot_mean(diff[mm], pid[mm], rng)
                r[name] = {"n_images": int(mm.sum()), "n_patients": int(len(set(pid[mm]))), "est": est, "ci95": ci,
                           "lower_above_margin": bool(ci[0] > pre["criteria"]["margin_logit"])}
            r["share_beyond"] = float((msk & (ratio > RATIO)).sum() / max(msk.sum(), 1))
            r["share_of_sum_beyond"] = float(diff[msk & (ratio > RATIO)].sum() / diff[msk].sum())
            rec[key] = r
            w, bb = r["within"], r["beyond"]
            print("%-10s %-30s all %+.2f | within n=%d %s | beyond n=%d %s" % (
                run, key, r["all"]["est"], w["n_images"],
                "%+.2f [%+.2f, %+.2f]" % (w["est"], *w["ci95"]) if "est" in w else "-",
                bb["n_images"], "%+.2f [%+.2f, %+.2f]" % (bb["est"], *bb["ci95"]) if "est" in bb else "-"), flush=True)
        out["runs"][run] = rec
    out["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
