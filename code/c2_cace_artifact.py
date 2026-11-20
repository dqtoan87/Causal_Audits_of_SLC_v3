#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase C2: CaCE artifact (negative control, editing-only counterfactual).

Measures the causal effect of inserting an acquisition artifact (hair/ruler/patch/vignette/jitter)
on the classifier's malignancy probability. It uses an editing-only intervention: do(U=present) on
the same image, with everything else held fixed. Artifacts are not causes of the disease, so a
causally faithful model must give **CaCE_U ≈ 0**. A large CaCE_U means the model relies on a
shortcut. See the design notes, §5.2(A).

ESTIMAND (design notes §5.3):
    CaCE_U(f) = E_x[ f(add_U(x)) − f(x) ]                       # prob scale, signed
    CaCE_logit(f) = E_x[ logit f(add_U(x)) − logit f(x) ]       # logit scale (robust to
                                                                 #  calibration differences)

Honesty (guard against metric coupling):
  - The artifact CF used when training CAUSAL covers {vignette, jitter, patch}. These three are
    coupled (in-distribution) and are not clean evidence for CAUSAL.
  - {hair, ruler} are held out (CAUSAL was never trained to be invariant to them): clean evidence.
  - For ERM/SLIM every artifact is non-tautological (they never saw an artifact CF).
  Each artifact is labelled coupled or held-out; cross-model claims rely on held-out only.

Design:
  - Edit each image once (rng deterministic by index) and share the edit across all 3 models.
    The intervention is identical, which removes rng noise from the model comparison.
  - Measure on truly malignant images with p_full ≥ floor (the clinically important decision),
    and also report all images.
  - Falsification: the identity edit must give CaCE = 0 (pipeline sanity check). Placebo.
  - Bootstrap CI; "faithful" = CI(CaCE_logit) contains 0 (no prediction change is good for a
    negative control).

Loads the 3 cached checkpoints in code/lib/checkpoints (p2_{erm,slim,causal}_s42.pt). No retraining.
Out -> Causal_XAI_v3/results/tier2_cace_artifact.json. Fast (~1-3 min), needs a GPU.

Run:
  .venv/bin/python Causal_XAI_v3/code/c2_cace_artifact.py
  .venv/bin/python Causal_XAI_v3/code/c2_cace_artifact.py --n-eval 240
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# --- load the training infrastructure (vendored: code/lib) -------------------
# The upstream training modules are vendored under code/lib together with the three
# primary EfficientNet-B3 checkpoints, so this tree is self-contained apart from the raw
# image data. Override with the CAUSALXAI_LIB environment variable if they are moved.
XAI_OLD = os.environ.get(
    "CAUSALXAI_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
sys.path.insert(0, XAI_OLD)
import train_XAI_new_v01 as p1       # noqa: E402  (DEVICE, build_model, CKPT_DIR, CFG, ...)
import phase2_causal_xai_v01 as p2   # noqa: E402  (ARTIFACTS, _to_norm, build_split, ...)

DEVICE = p1.DEVICE
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT_JSON = os.path.join(OUT_DIR, "tier2_cace_artifact.json")

P_FLOOR = 0.10
SEED = 42
import audited_models                # noqa: E402  (checkpoint paths resolved in one place)
CKPTS = audited_models.CKPTS         # absolute paths to the frozen checkpoint set

# Coupling labels (artifact CF used to train CAUSAL = vignette+jitter+patch; see phase3 gpu_artifact)
COUPLING = {"hair_light": "held-out", "hair": "held-out", "hair_heavy": "held-out",
            "ruler": "held-out",
            "patch": "train-coupled", "vignette": "train-coupled", "jitter": "train-coupled"}


# --- Held-out artifacts: realistic (thin + semi-transparent alpha blend), avoid destroying the image ---
# (The crude phase2 version draws solid black strokes over the lesion -> off-manifold, so CaCE
#  reflects "image destruction" rather than a shortcut. This version mimics real hair/rulers:
#  thin 1px, semi-transparent.)
def _hair(x, rng, n, alpha, dark=0.10):
    """n thin curved 1px hairs, alpha-blended (semi-transparent), as in real dermoscopy."""
    x = x.clone(); _, H, W = x.shape
    for _ in range(n):
        x0, y0 = rng.randint(0, W), rng.randint(0, H)
        ang = rng.uniform(0, 2 * np.pi)
        length = rng.randint(W // 3, W)
        curv = rng.uniform(-0.0025, 0.0025)
        cs, sn = np.cos(ang), np.sin(ang)
        pc, ps = np.cos(ang + np.pi / 2), np.sin(ang + np.pi / 2)
        for t in range(length):
            px = int(x0 + t * cs + curv * t * t * pc)
            py = int(y0 + t * sn + curv * t * t * ps)
            if 0 <= px < W and 0 <= py < H:
                x[:, py, px] = alpha * dark + (1 - alpha) * x[:, py, px]
    return x


# Dose-response: 3 hair levels -> check that |CaCE| is monotonic (a classic causal signature)
def art_hair_light(x, rng): return _hair(x, rng, n=6, alpha=0.4)
def art_hair(x, rng):       return _hair(x, rng, n=14, alpha=0.6)
def art_hair_heavy(x, rng): return _hair(x, rng, n=24, alpha=0.75)


def art_ruler(x, rng, alpha=0.7, dark=0.0):
    """Thin semi-transparent ruler at the bottom edge (background skin, not over the lesion centre)."""
    x = x.clone(); _, H, W = x.shape
    y = rng.randint(int(0.75 * H), max(int(0.75 * H) + 1, H - 3))
    x[:, y:y + 2, :] = alpha * dark + (1 - alpha) * x[:, y:y + 2, :]
    for tx in range(0, W, 18):                      # thin tick marks
        yt0, yt1 = max(0, y - 4), y + 4
        x[:, yt0:yt1, tx:tx + 1] = alpha * dark + (1 - alpha) * x[:, yt0:yt1, tx:tx + 1]
    return x


def art_jitter(x, rng):
    """Brightness/contrast jitter. Matches gpu_artifact used when training CAUSAL (coupled artifact)."""
    bright = (rng.rand() - 0.5) * 0.3
    contrast = 0.7 + 0.6 * rng.rand()
    return torch.clamp((x - 0.5) * contrast + 0.5 + bright, 0, 1)


def art_identity(x, rng):
    """Falsification: change nothing -> CaCE must be 0."""
    return x.clone()


# Intervention set: realistic held-out (clean, with hair dose-response) + train-matched coupled + identity
ARTIFACTS = {
    "hair_light": art_hair_light, "hair": art_hair, "hair_heavy": art_hair_heavy,  # held-out dose
    "ruler": art_ruler,                                   # held-out
    "patch": p2.art_patch, "vignette": p2.art_vignette,   # coupled (matches CAUSAL training)
    "jitter": art_jitter,                                  # coupled
    "identity": art_identity,                              # sanity (CaCE≈0)
}
HAIR_DOSE = ["hair_light", "hair", "hair_heavy"]          # for the monotonicity check


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


@torch.no_grad()
def model_prob(model, x01_batch):
    """x01_batch: [B,3,H,W] in [0,1]. Returns malignant prob [B]."""
    xn = p2._to_norm(x01_batch).to(DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
        return torch.softmax(model(xn), 1)[:, 1].float().cpu().numpy()


def precompute_edits(imgs01, art_names, seed, thr=0.02):
    """Edit each image once (rng deterministic by index) -> shared by all models.
    Returns (edits, change_frac): edits=list[dict art->tensor]; change_frac[a]=% of pixels changed
    by more than thr (reports how destructive/off-manifold the edit is, design notes §5.4)."""
    edits = []
    cf = {a: [] for a in art_names}
    for i, x01 in enumerate(imgs01):
        rng = np.random.RandomState(seed + i)   # fixed per image
        d = {}
        for a in art_names:
            e = ARTIFACTS[a](x01, rng)
            d[a] = e
            cf[a].append(float((e - x01).abs().amax(0).gt(thr).float().mean()))
        edits.append(d)
    change_frac = {a: float(np.mean(cf[a])) for a in art_names}
    return edits, change_frac


def eval_model(model, imgs01, labels, edits, art_names, cfg):
    model.eval()
    # batch [orig, edit_a1, edit_a2, ...] per image -> 1 forward pass per image
    p0_all = np.empty(len(imgs01))
    dp = {a: [] for a in art_names}      # signed Δp, per kept image
    dl = {a: [] for a in art_names}      # signed Δlogit
    keep_mask = []
    for i, (x01, lab) in enumerate(zip(imgs01, labels)):
        batch = torch.stack([x01] + [edits[i][a] for a in art_names])
        ps = model_prob(model, batch)
        p0 = ps[0]
        p0_all[i] = p0
        keep = (lab == 1) and (p0 >= cfg["p_floor"])
        keep_mask.append(keep)
        if not keep:
            continue
        for j, a in enumerate(art_names):
            pa = ps[j + 1]
            dp[a].append(pa - p0)
            dl[a].append(logit(pa) - logit(p0))
    keep_mask = np.array(keep_mask)
    return p0_all, dp, dl, keep_mask


def summarize(dp, dl, art_names, bn, seed):
    """Per-artifact CaCE + CI + faithful flag."""
    out = {}
    for a in art_names:
        v_p, v_l = np.array(dp[a]), np.array(dl[a])
        p_mean, p_lo, p_hi = p2.boot_mean_ci(v_p, bn, seed)
        l_mean, l_lo, l_hi = p2.boot_mean_ci(v_l, bn, seed)
        ap_mean, ap_lo, ap_hi = p2.boot_mean_ci(np.abs(v_p), bn, seed)
        faithful = bool(l_lo <= 0 <= l_hi)        # logit CI contains 0 = no prediction change = good
        out[a] = dict(
            coupling=COUPLING.get(a, "n/a"),
            cace_dp=p_mean, cace_dp_ci=[p_lo, p_hi],
            cace_logit=l_mean, cace_logit_ci=[l_lo, l_hi],
            abs_dp=ap_mean, abs_dp_ci=[ap_lo, ap_hi],
            faithful=faithful, n=int(len(v_p)),
        )
    return out


def aggregate_group(dp, dl, names, bn, seed):
    """Pool |Δlogit| and |Δp| over a group of artifacts (e.g. held-out) -> summary metric."""
    allp = np.concatenate([np.abs(dp[a]) for a in names]) if names else np.array([])
    alll = np.concatenate([np.abs(dl[a]) for a in names]) if names else np.array([])
    mp = p2.boot_mean_ci(allp, bn, seed)
    ml = p2.boot_mean_ci(alll, bn, seed)
    return dict(abs_dp=mp[0], abs_dp_ci=[mp[1], mp[2]],
                abs_logit=ml[0], abs_logit_ci=[ml[1], ml[2]], n=int(len(allp)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--boot-n", type=int, default=2000)
    ap.add_argument("--p-floor", type=float, default=P_FLOOR)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    cfg = dict(seed=SEED, p_floor=args.p_floor, boot_n=args.boot_n)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"##### PHASE C2: CaCE artifact {time.strftime('%H:%M:%S')} (dev={DEVICE}) #####", flush=True)
    df_tr, df_va = p2.build_split({"seed": SEED, "n_folds": 5})
    imgs01, labels = p2.load_eval_images(df_va, args.n_eval, SEED)
    n_mal = int(sum(labels))
    print(f"eval: {len(imgs01)} images (truly malignant={n_mal})", flush=True)

    art_names = list(ARTIFACTS.keys())
    held = [a for a in art_names if COUPLING.get(a) == "held-out"]
    coup = [a for a in art_names if COUPLING.get(a) == "train-coupled"]
    edits, change_frac = precompute_edits(imgs01, art_names, SEED)
    print(f"interventions: held-out={held} | coupled={coup} | sanity=['identity']", flush=True)
    print("  %pixels changed: " + " ".join(f"{a}={100*change_frac[a]:.1f}%" for a in art_names), flush=True)

    results = {}
    for name, path in CKPTS.items():
        if not os.path.exists(path):
            print(f"[skip] missing {path}"); continue
        model = p1.build_model("erm" if name == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        t0 = time.time()
        p0_all, dp, dl, keep = eval_model(model, imgs01, labels, edits, art_names, cfg)
        per_art = summarize(dp, dl, art_names, args.boot_n, SEED)
        results[name] = dict(
            n_kept=int(keep.sum()),
            pred_mal_prob=float(np.mean(p0_all[[l == 1 for l in labels]])),  # calibration exposure
            per_artifact=per_art,
            heldout=aggregate_group(dp, dl, held, args.boot_n, SEED),
            coupled=aggregate_group(dp, dl, coup, args.boot_n, SEED),
            identity_cace_logit=per_art["identity"]["cace_logit"],
            hair_dose_monotonic=bool(
                abs(per_art["hair_light"]["cace_logit"]) <= abs(per_art["hair"]["cace_logit"])
                <= abs(per_art["hair_heavy"]["cace_logit"])),
        )
        print(f"[{name}] done ({time.time()-t0:.0f}s, n_kept={results[name]['n_kept']}, "
              f"p_mal={results[name]['pred_mal_prob']:.3f})", flush=True)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    out = dict(
        phase="C2_cace_artifact",
        estimand="CaCE_U(f)=E_x[f(add_U(x))-f(x)]; signed prob & logit scale; editing-only CF.",
        note_coupling="artifact-CF train CAUSAL={vignette,jitter,patch} (coupled); "
                      "{hair,ruler}=held-out (clean). ERM/SLIM: all non-tautological.",
        eval=dict(n_images=len(imgs01), n_malignant=n_mal, p_floor=args.p_floor,
                  boot_n=args.boot_n, heldout=held, coupled=coup, change_frac=change_frac),
        ckpt_set=__import__('audited_models').SET_NAME, models=results,
    )
    json.dump(out, open(args.out, "w"), indent=2, default=float)

    # ---- summary table ----
    order = [m for m in ["erm", "slim", "causal"] if m in results]
    print("\n" + "=" * 96)
    print("CaCE ARTIFACT (negative control): faithful = |CaCE|≈0 (logit CI contains 0). "
          "Clean evidence = held-out (hair/ruler)")
    print("=" * 96)
    print(f"{'artifact':12s}{'coupling':14s}" + "".join(f"{m:>22s}" for m in order))
    print(f"{'':26s}" + "".join(f"{'CaCE_logit [CI] faith':>22s}" for m in order))
    print("-" * 96)
    for a in art_names:
        row = f"{a:12s}{COUPLING.get(a,'n/a'):14s}"
        for m in order:
            r = results[m]["per_artifact"][a]
            tag = "✓" if r["faithful"] else "✗"
            row += f"{r['cace_logit']:+.2f}[{r['cace_logit_ci'][0]:+.1f},{r['cace_logit_ci'][1]:+.1f}]{tag:>2s}".rjust(22)
        print(row)
    print("-" * 96)
    print(f"{'HELD-OUT |Δlogit|':26s}" + "".join(
        f"{results[m]['heldout']['abs_logit']:.3f}[{results[m]['heldout']['abs_logit_ci'][0]:.2f},{results[m]['heldout']['abs_logit_ci'][1]:.2f}]".rjust(22)
        for m in order))
    print(f"{'coupled |Δlogit|':26s}" + "".join(
        f"{results[m]['coupled']['abs_logit']:.3f}[{results[m]['coupled']['abs_logit_ci'][0]:.2f},{results[m]['coupled']['abs_logit_ci'][1]:.2f}]".rjust(22)
        for m in order))
    print(f"{'pred_mal_prob (calib)':26s}" + "".join(f"{results[m]['pred_mal_prob']:.3f}".rjust(22) for m in order))
    print(f"{'identity (sanity≈0)':26s}" + "".join(f"{results[m]['identity_cace_logit']:+.4f}".rjust(22) for m in order))
    print(f"{'hair dose monotonic':26s}" + "".join(
        f"{'YES' if results[m]['hair_dose_monotonic'] else 'no':>22s}" for m in order))
    print("=" * 96)
    print("Reading: CaCE_logit > 0 = the artifact pushes the prediction toward malignant (shortcut). "
          "faith ✓ = CI contains 0 (model ignores the artifact = causally faithful). Compare models "
          "on the held-out rows only; coupled rows are in-distribution only (CAUSAL was trained to "
          "avoid them). identity must be ≈0.")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
