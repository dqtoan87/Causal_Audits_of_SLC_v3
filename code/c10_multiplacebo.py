#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C10: decisive test for placebo validity (a) and dose (b).
The whole "corrected alignment / retraction" result rests on one placebo (hue). Tests:
  (a) Multi-placebo: 4 neutral edits of different kinds (hue / flip / spatial-jitter / small-noise).
      If CaCE_placebo converges, generic edit-sensitivity is real and the correction is valid. If it
      diverges, the single-placebo correction is not valid, so the retraction must be withdrawn
      (concept tier inconclusive).
      + corrected alignment A_corr under each placebo: is it still positive [aligned] for every placebo?
  (b) Per-dose corrected: is A_corr at low dose (clean isolation) still positive?
Edits are shared across all models; common set. Out -> results/c10_multiplacebo.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c3_cace_concept as c3m
import c7_extra_controls as c7m
p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
OUT = os.path.join(os.path.dirname(HERE), "results", "c10_multiplacebo.json")
import audited_models                # noqa: E402  (checkpoints resolved in one place)
import audit_cohort                  # noqa: E402  (frozen audit cohort)
CKPTS = audited_models.CKPTS         # absolute paths to the fixed set
SEED, N_EVAL, BOOT = 42, 200, 2000
DOSES = c3m.DOSES   # [0.3,0.6,1.0]


# ---- neutral placebos (within lesion mask via w), aim: preserve concept proxies ----
def pb_hue(x, w, skin, rng):          # color, norm-preserving rotation (extreme color change)
    d = x - skin.view(-1, 1, 1)
    return torch.clamp(x + w.unsqueeze(0) * (d[[2, 0, 1]] - d), 0, 1)

def pb_flip(x, w, skin, rng):         # structural mirror (in-distribution; preserves color set)
    xf = torch.flip(x, dims=[2])
    return torch.clamp(x * (1 - w) + xf * w, 0, 1)

def pb_jitter(x, w, skin, rng):       # small spatial shift of lesion content
    sh = (rng.randint(-6, 7), rng.randint(-6, 7))
    xr = torch.roll(x, shifts=sh, dims=(1, 2))
    return torch.clamp(x * (1 - w) + xr * w, 0, 1)

def pb_noise(x, w, skin, rng):        # small zero-mean noise in mask
    n = torch.from_numpy(rng.normal(0, 0.05, x.shape).astype(np.float32))
    return torch.clamp(x + w.unsqueeze(0) * n, 0, 1)

PLACEBOS = {"hue": pb_hue, "flip": pb_flip, "jitter": pb_jitter, "noise": pb_noise}


def proxychg(x0, xe, w0, skin0):
    """% change in the concept proxy for one edit (fixed ref). A good placebo gives ~0."""
    a, b = c7m.proxy_fixed(x0, skin0), c7m.proxy_fixed(xe, skin0)
    return {k: abs(b[k] - a[k]) / (abs(a[k]) + 1e-6) for k in a}


def boot(v, groups=None):
    """Mean and 95% percentile interval. With groups, patients are the resampling unit and all
    images of a drawn patient enter together, as in every other interval of the paper."""
    v = np.asarray(v, float)
    if groups is None:
        return p2.boot_mean_ci(v, BOOT, SEED)
    u, inv = np.unique(np.asarray(groups), return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(u))]
    rng = np.random.RandomState(SEED)
    bs = [v[np.concatenate([idx[k] for k in rng.randint(0, len(idx), len(idx))])].mean()
          for _ in range(BOOT)]
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    print(f"##### C10 multi-placebo {time.strftime('%H:%M:%S')} #####", flush=True)
    imgs01, labels, _pid, _ids = audit_cohort.load()
    _pid = np.asarray(_pid)   # frozen audit cohort, never rebuilt
    labels = np.array(labels); mal = np.where(labels == 1)[0]

    # edits: concept contrast/color at 3 doses + 4 placebos
    cnames = [f"{c}@{k}" for c in ("contrast", "color_var") for k in range(3)]
    pnames = list(PLACEBOS)
    names = cnames + pnames
    edits, pchg = [], {p: [] for p in pnames}
    for ii in mal:
        x = imgs01[ii]; rng = np.random.RandomState(SEED + int(ii))
        w, skin, lm = c3m.soft_mask(x)
        d = {}
        for c in ("contrast", "color_var"):
            for k in range(3):
                d[f"{c}@{k}"] = c3m.make_edit(x, c, DOSES[k], w, skin, lm)
        for p, fn in PLACEBOS.items():
            e = fn(x, w, skin, np.random.RandomState(SEED + int(ii) + 7))
            d[p] = e
            pc = proxychg(x, e, w, skin)
            pchg[p].append(np.mean([pc["contrast"], pc["color_var"], pc["size"]]))
        edits.append(d)
    pchg_s = {p: float(np.mean(pchg[p])) for p in pnames}
    print("placebo proxy-change (lower=more neutral): " +
          " ".join(f"{p}={100*pchg_s[p]:.1f}%" for p in pnames), flush=True)

    order = ["erm", "slim", "causal"]
    P0, DL = {}, {}
    for m in order:
        model = p1.build_model("erm" if m == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(CKPTS[m], map_location=DEVICE)); model.eval()
        p0 = np.empty(len(mal)); dl = {n: np.empty(len(mal)) for n in names}
        for j, ii in enumerate(mal):
            batch = torch.stack([imgs01[ii]] + [edits[j][n] for n in names])
            ps = c3m.model_prob(model, batch); p0[j] = ps[0]
            for k, n in enumerate(names):
                dl[n][j] = c3m.logit(ps[k + 1]) - c3m.logit(ps[0])
        P0[m] = p0; DL[m] = dl; del model
        if DEVICE == "cuda": torch.cuda.empty_cache()
        print(f"[{m}] p_full={p0.mean():.3f}", flush=True)

    common = (P0["erm"] >= 0.1) & (P0["slim"] >= 0.1) & (P0["causal"] >= 0.1)
    G = _pid[mal][common]
    res = {"n_common": int(common.sum()), "n_patients_common": int(len(set(G.tolist()))),
           "resampling_unit": "patient", "placebo_proxychg": pchg_s, "models": {}}
    for m in order:
        mr = {"placebo_cace": {}, "A_corr_by_placebo": {}, "A_corr_hue_by_dose": {}}
        for p in pnames:
            mr["placebo_cace"][p] = boot(DL[m][p][common], G)
        # corrected alignment (top dose concepts) under each placebo
        for p in pnames:
            cc = boot((DL[m]["contrast@2"] - DL[m][p])[common], G)
            co = boot((DL[m]["color_var@2"] - DL[m][p])[common], G)
            mr["A_corr_by_placebo"][p] = dict(A=float(np.mean([cc[0], co[0]])),
                                              contrast=cc, color=co,
                                              aligned=bool(cc[1] > 0 and co[1] > 0),
                                              anti=bool(cc[2] < 0 and co[2] < 0))
        # per-dose corrected under hue
        for k in range(3):
            cc = boot((DL[m][f"contrast@{k}"] - DL[m]["hue"])[common], G)
            co = boot((DL[m][f"color_var@{k}"] - DL[m]["hue"])[common], G)
            mr["A_corr_hue_by_dose"][f"dose{k}"] = dict(A=float(np.mean([cc[0], co[0]])),
                                                        aligned=bool(cc[1] > 0 and co[1] > 0))
        res["models"][m] = mr
    json.dump(res, open(OUT, "w"), indent=2, default=float)

    print("\n" + "=" * 96)
    print("(a) Placebo CaCE (logit) per model: converge or diverge?")
    print(f"{'model':8s}" + "".join(f"{p:>16s}" for p in pnames))
    for m in order:
        print(f"{m:8s}" + "".join(f"{res['models'][m]['placebo_cace'][p][0]:>16.2f}" for p in pnames))
    print("-" * 96)
    print("(a) Corrected alignment A_corr under each placebo (aligned if >0 & CI>0):")
    print(f"{'model':8s}" + "".join(f"{p:>16s}" for p in pnames))
    for m in order:
        row = f"{m:8s}"
        for p in pnames:
            d = res['models'][m]['A_corr_by_placebo'][p]
            row += f"{d['A']:+.2f}[{'Y' if d['aligned'] else ('A' if d['anti'] else '.')}]".rjust(16)
        print(row)
    print("-" * 96)
    print("(b) A_corr (hue) by dose (dose0=cleanest):")
    for m in order:
        s = " ".join(f"d{k}={res['models'][m]['A_corr_hue_by_dose'][f'dose{k}']['A']:+.2f}"
                     f"[{'Y' if res['models'][m]['A_corr_hue_by_dose'][f'dose{k}']['aligned'] else '.'}]"
                     for k in range(3))
        print(f"  {m:8s} {s}")
    print("=" * 96)
    print("Conclusion: if placebo_cace converges & A_corr>0[Y] for every placebo & every dose -> correction "
          "valid, retraction holds. If it diverges (e.g. flip≈0, hue large) -> correction not valid -> concept "
          "tier inconclusive, must be downgraded.")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
