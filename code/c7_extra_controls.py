#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C7: extra controls for size, placebo, effective independence and outliers.

(a) Size rescue: the size intervention (dilation+paint) is correct; only the proxy is measured
   wrongly (the recomputed skin ref is contaminated). Fix: measure the proxy against the fixed
   original skin reference (skin0). If the manipulation check passes, size is restored as the third
   concept (ATE+ in C1/C5).
(b) Placebo edit: sign-balanced contrast (±1 per pixel), giving the same local/boundary noise as the
   contrast edit but net concept ≈ 0. A faithful model must give CaCE_placebo ≈ 0 (no response to
   the edit artifact).
(c) Effective independence: per-image correlation of CaCE_contrast vs CaCE_color (high means ~1 measurement).
(d) Outlier: jackknife SE + max single-image influence for concept CaCE.

Reuses c3_cace_concept (soft_mask, edit_contrast/color/size, model_prob, logit). Edits are shared
across all models. CaCE on the common set (images with p_full≥0.1 for all 3 models). Out -> results/c7_controls.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c3_cace_concept as c3m
p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
OUT = os.path.join(os.path.dirname(HERE), "results", "c7_controls.json")
import audited_models                # noqa: E402  (checkpoints resolved in one place)
CKPTS = audited_models.CKPTS         # absolute paths to the fixed set
SEED, N_EVAL, BOOT = 42, 200, 2000
SIZE_RADII = [4, 8, 14]
GAIN = c3m.DOSES[2]      # top dose cho contrast/color/placebo
SCALE = 0.22


def proxy_fixed(x01, skin0):
    """Measure the concept proxy against the original skin reference skin0 (membership recomputed on the edited image)."""
    C = x01.shape[0]
    dist = (x01 - skin0.view(C, 1, 1)).pow(2).sum(0).sqrt()
    w = torch.clamp(dist / SCALE, 0, 1)
    den = w.sum().clamp_min(1e-6)
    lesmean = (x01 * w.unsqueeze(0)).sum((1, 2)) / den
    contrast = float((lesmean - skin0).pow(2).sum().sqrt())
    var = ((x01 - lesmean.view(-1, 1, 1)).pow(2) * w.unsqueeze(0)).sum((1, 2)) / den
    return {"contrast": contrast, "color_var": float(var.sum().sqrt()), "size": float(den)}


def edit_placebo(x01, w, skin, g, rng):
    """Smooth, neutral placebo: rotate the colour-deviation vector (x-skin) by a cyclic channel permutation (R->G->B->R).
    Preserves ||x-skin|| exactly per pixel, so the contrast, colour-variance and size proxies stay unchanged; only
    the hue changes (smoothly, no high-frequency noise). A faithful model (responding to the concept) must give CaCE ≈ 0."""
    d = x01 - skin.view(-1, 1, 1)
    d_rot = d[[2, 0, 1], :, :]                 # cyclic channel permutation -> preserves the per-pixel norm
    return torch.clamp(x01 + w.unsqueeze(0) * (d_rot - d), 0, 1)


def boot_ci(v, seed=SEED, n=BOOT):
    return p2.boot_mean_ci(np.asarray(v, float), n, seed)


def main():
    print(f"##### C7 extra controls {time.strftime('%H:%M:%S')} #####", flush=True)
    _, df_va = p2.build_split({"seed": SEED, "n_folds": 5})
    imgs01, labels = p2.load_eval_images(df_va, N_EVAL, SEED)
    labels = np.array(labels); mal = np.where(labels == 1)[0]
    print(f"malignant {len(mal)}", flush=True)

    # ---- precompute edits + manipulation-check proxies (fixed ref) ----
    names = ["contrast", "color_var", "size0", "size1", "size2", "placebo"]
    edits = []                                  # per mal image: {name: tensor}
    mani = {n: {"intended": [], "leak": []} for n in ["size0", "size1", "size2", "placebo"]}
    for ii in mal:
        x = imgs01[ii]
        w, skin, lm = c3m.soft_mask(x)
        base = proxy_fixed(x, skin)
        d = {}
        d["contrast"] = c3m.edit_contrast(x, w, skin, lm, GAIN)
        d["color_var"] = c3m.edit_color_var(x, w, skin, lm, GAIN)
        for k, r in enumerate(SIZE_RADII):
            d[f"size{k}"] = c3m.edit_size(x, w, lm, r)
        rng = np.random.RandomState(SEED + int(ii))
        d["placebo"] = edit_placebo(x, w, skin, GAIN, rng)
        edits.append(d)
        # manipulation checks (fixed ref skin0)
        for k in range(3):
            px = proxy_fixed(d[f"size{k}"], skin)
            mani[f"size{k}"]["intended"].append((px["size"] - base["size"]) / (base["size"] + 1e-6))
            mani[f"size{k}"]["leak"].append(np.mean([abs(px[c]-base[c])/(abs(base[c])+1e-6)
                                                     for c in ("contrast", "color_var")]))
        pp = proxy_fixed(d["placebo"], skin)
        mani["placebo"]["intended"].append(abs(pp["contrast"]-base["contrast"])/(base["contrast"]+1e-6))
        mani["placebo"]["leak"].append(np.mean([abs(pp[c]-base[c])/(abs(base[c])+1e-6)
                                                for c in ("color_var", "size")]))
    mani_s = {n: {"intended": float(np.mean(mani[n]["intended"])),
                  "leak": float(np.mean(mani[n]["leak"]))} for n in mani}
    print("manipulation-check (fixed-ref):")
    for n in mani: print(f"   {n:9s} intended={100*mani_s[n]['intended']:+6.1f}%  leak={100*mani_s[n]['leak']:5.1f}%", flush=True)

    # ---- per-model p0 + logit-diffs ----
    order = ["erm", "slim", "causal"]
    P0, DL = {}, {}
    for m in order:
        model = p1.build_model("erm" if m == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(CKPTS[m], map_location=DEVICE)); model.eval()
        p0 = np.empty(len(mal)); dl = {n: np.empty(len(mal)) for n in names}
        for j, ii in enumerate(mal):
            batch = torch.stack([imgs01[ii]] + [edits[j][n] for n in names])
            ps = c3m.model_prob(model, batch)
            p0[j] = ps[0]
            for k, n in enumerate(names):
                dl[n][j] = c3m.logit(ps[k+1]) - c3m.logit(ps[0])
        P0[m] = p0; DL[m] = dl
        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()
        print(f"[{m}] done p_full={p0.mean():.3f}", flush=True)

    common = (P0["erm"] >= 0.1) & (P0["slim"] >= 0.1) & (P0["causal"] >= 0.1)
    nC = int(common.sum())

    def jackknife_se(v):
        v = np.asarray(v, float); n = len(v); m = v.mean()
        loo = (v.sum() - v) / (n - 1)
        return float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean())**2))), float(np.max(np.abs(loo - m)))

    res = {"n_malignant": int(len(mal)), "n_common": nC, "gain": GAIN, "size_radii": SIZE_RADII,
           "manipulation": mani_s, "models": {}}
    for m in order:
        sz = [boot_ci(DL[m][f"size{k}"][common]) for k in range(3)]
        ct = boot_ci(DL[m]["contrast"][common]); co = boot_ci(DL[m]["color_var"][common])
        pl = boot_ci(DL[m]["placebo"][common])
        # (c) correlation contrast vs color per-image (common)
        r = float(np.corrcoef(DL[m]["contrast"][common], DL[m]["color_var"][common])[0, 1])
        # (d) jackknife on contrast & color
        jc_se, jc_max = jackknife_se(DL[m]["contrast"][common])
        jo_se, jo_max = jackknife_se(DL[m]["color_var"][common])
        A3 = float(np.mean([ct[0], co[0], sz[2][0]]))    # 3-concept alignment (all ATE+)
        # (b) placebo-corrected: concept edit minus neutral edit (per image) -> removes the common edit-sensitivity
        cc = boot_ci((DL[m]["contrast"] - DL[m]["placebo"])[common])
        cco = boot_ci((DL[m]["color_var"] - DL[m]["placebo"])[common])
        A_corr = float(np.mean([cc[0], cco[0]]))
        res["models"][m] = dict(
            size_cace_bydose=[s[0] for s in sz], size_top=sz[2], size_monotonic=bool(sz[0][0] <= sz[1][0] <= sz[2][0]),
            size_aligned=bool(sz[2][1] > 0), size_anti=bool(sz[2][2] < 0),
            contrast=ct, color_var=co, placebo=pl,
            placebo_null=bool(pl[1] <= 0 <= pl[2]),
            A3_three_concept=A3,
            contrast_corr=cc, color_corr=cco, A_placebo_corrected=A_corr,
            corr_aligned=bool(cc[1] > 0 and cco[1] > 0), corr_anti=bool(cc[2] < 0 and cco[2] < 0),
            corr_contrast_color=r,
            jackknife_contrast_se=jc_se, jackknife_contrast_maxinfl=jc_max,
            jackknife_color_se=jo_se, jackknife_color_maxinfl=jo_max,
        )
    json.dump(res, open(OUT, "w"), indent=2, default=float)

    print("\n" + "=" * 90)
    print(f"COMMON n={nC}  | gain={GAIN}")
    print(f"{'model':7s}{'size_top[CI]':>22s}{'mono':>6s}{'placebo[CI]':>20s}{'pl≈0':>6s}{'corr(c,col)':>12s}{'A3':>8s}")
    for m in order:
        d = res["models"][m]
        s = f"{d['size_top'][0]:+.2f}[{d['size_top'][1]:+.2f},{d['size_top'][2]:+.2f}]"
        pl = f"{d['placebo'][0]:+.2f}[{d['placebo'][1]:+.2f},{d['placebo'][2]:+.2f}]"
        print(f"{m:7s}{s:>22s}{('↑' if d['size_monotonic'] else '·'):>6s}{pl:>20s}"
              f"{('YES' if d['placebo_null'] else 'no'):>6s}{d['corr_contrast_color']:>12.2f}{d['A3_three_concept']:>+8.3f}")
    print("=" * 90)
    print("\n★ Placebo-corrected (concept-edit − neutral-edit; removes the common edit-sensitivity):")
    for m in order:
        d = res["models"][m]
        cc, cco = d["contrast_corr"], d["color_corr"]
        flag = "✓ALIGNED" if d["corr_aligned"] else ("✗anti" if d["corr_anti"] else "·n.s.")
        print(f"  {m:7s} contrast={cc[0]:+.2f}[{cc[1]:+.2f},{cc[2]:+.2f}] color={cco[0]:+.2f}"
              f"[{cco[1]:+.2f},{cco[2]:+.2f}]  A_corr={d['A_placebo_corrected']:+.3f} {flag}")
    print("size: expected sign + (ATE+ in C1). placebo: ≈0 = the model does not respond to the edit artifact (good). "
          "high corr = the 2 concepts measure almost the same thing. A3 = alignment of 3 concepts (contrast+color+size).")
    print("jackknife SE / max-influence (contrast | color):")
    for m in order:
        d = res["models"][m]
        print(f"  {m:7s} SE {d['jackknife_contrast_se']:.3f}/{d['jackknife_color_se']:.3f}  "
              f"maxInfl {d['jackknife_contrast_maxinfl']:.3f}/{d['jackknife_color_maxinfl']:.3f}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
