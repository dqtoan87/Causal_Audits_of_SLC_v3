#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase C3: CaCE concept (editing-only counterfactual for clinical concepts).

Measures the causal effect of increasing a clinical concept (measurable and editable) on the
classifier's malignancy probability, using an editing-only intervention inside the lesion region.
Unlike C2 (artifacts, which must give ≈0), in C3 the concept is a real disease mechanism (C1 gives
ATE > 0), so a causally faithful model must give **CaCE > 0, monotonic in dose**. See the design
notes, §5.2(B).

Editable concepts (mapped to C1-significant concepts, all with positive ATE):
  - contrast   ↔ tbp_lv_deltaLBnorm  : push the lesion colour away from the background skin (more contrast).
  - color_var  ↔ tbp_lv_norm_color/color_std : amplify colour deviation inside the lesion (multicolour).
  - size       ↔ tbp_lv_areaMM2      : enlarge the central region (zoom) -> larger lesion.

Design (inherited from C2):
  - soft lesion mask = region whose colour differs from the background skin (outer border) -> no
    external segmentation needed.
  - Edit each image once, shared by all 3 models. Dose-response (3 levels per concept) -> monotonicity check.
  - Signed CaCE (prob + logit scale, robust to calibration). Measured on malignant images with p_full ≥ floor.
  - Manipulation check (design notes §5.4): the concept proxy must change in the right direction and
    by enough; cross-leakage is reported.
  - Falsification: identity -> CaCE = 0.

ESTIMAND: CaCE_C(f) = E_x[ f(edit_{C↑}(x)) − f(x) ] (expected > 0 if the model uses the causal concept).

Loads the 3 cached checkpoints (p2_{erm,slim,causal}_s42.pt). No retraining. Needs a GPU.
Out -> Causal_XAI_v3/results/tier2_cace_concept.json.

Run:
  .venv/bin/python Causal_XAI_v3/code/c3_cace_concept.py
  .venv/bin/python Causal_XAI_v3/code/c3_cace_concept.py --n-eval 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# The upstream training modules are vendored under code/lib together with the three
# primary EfficientNet-B3 checkpoints, so this tree is self-contained apart from the raw
# image data. Override with the CAUSALXAI_LIB environment variable if they are moved.
XAI_OLD = os.environ.get(
    "CAUSALXAI_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
sys.path.insert(0, XAI_OLD)
import train_XAI_new_v01 as p1       # noqa: E402
import phase2_causal_xai_v01 as p2   # noqa: E402

DEVICE = p1.DEVICE
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT_JSON = os.path.join(OUT_DIR, "tier2_cace_concept.json")
P_FLOOR = 0.10
SEED = 42
import audited_models                # noqa: E402  (checkpoint paths resolved in one place)
CKPTS = audited_models.CKPTS         # absolute paths to the frozen checkpoint set

# Concept -> C1 map + dose (3 levels). Every concept has positive C1 ATE -> CaCE expected positive.
DOSES = [0.3, 0.6, 1.0]           # gain for contrast/color_var
# size is excluded: the manipulation check showed that size interventions (zoom and dilation) do not
# cleanly increase the area proxy (lesions touching the border contaminate the skin estimate).
# Morphology needs a generative model -> future work.
CONCEPTS = {
    "contrast":  {"c1": "contrast_LB (ATE+)", "doses": DOSES},
    "color_var": {"c1": "C_color/C_colorstd (ATE+)", "doses": DOSES},
}


# ---------------------------------------------------------------------------------------
# Soft lesion mask + concept edits (image in [0,1], [3,H,W])
# ---------------------------------------------------------------------------------------
def soft_mask(x01, border=0.15, scale=0.22):
    """w[H,W]∈[0,1]: degree of "belonging to the lesion" = colour differs from the background skin (outer border). Returns (w, skin[3], les_mean[3])."""
    C, H, W = x01.shape
    bh, bw = max(1, int(H * border)), max(1, int(W * border))
    ring = torch.zeros(H, W, dtype=torch.bool)
    ring[:bh, :] = True; ring[-bh:, :] = True; ring[:, :bw] = True; ring[:, -bw:] = True
    skin = x01[:, ring].mean(1)                              # background skin colour
    dist = (x01 - skin.view(C, 1, 1)).pow(2).sum(0).sqrt()   # colour distance -> [H,W]
    w = torch.clamp(dist / scale, 0, 1)
    den = w.sum().clamp_min(1e-6)
    les_mean = (x01 * w.unsqueeze(0)).sum((1, 2)) / den      # mean lesion colour
    return w, skin, les_mean


def edit_contrast(x01, w, skin, les_mean, g):
    """Push the lesion away from the background skin colour -> higher deltaLB (lesion-vs-skin contrast)."""
    d = (x01 - skin.view(-1, 1, 1)) * g
    return torch.clamp(x01 + w.unsqueeze(0) * d, 0, 1)


def edit_color_var(x01, w, skin, les_mean, g):
    """Amplify colour deviation around the lesion mean -> more multicolour (color variance)."""
    d = (x01 - les_mean.view(-1, 1, 1)) * g
    return torch.clamp(x01 + w.unsqueeze(0) * d, 0, 1)


def edit_size(x01, w, les_mean, radius):
    """Dilate the soft mask by `radius` px and paint the lesion colour into the new ring
    -> directly increases lesion area (geometric intervention, low leakage)."""
    r = int(radius)
    wd = F.max_pool2d(w[None, None], kernel_size=2 * r + 1, stride=1, padding=r)[0, 0]
    ring = torch.clamp(wd - w, 0, 1)                  # newly added region
    out = x01 + ring.unsqueeze(0) * (les_mean.view(-1, 1, 1) - x01)
    return out.clamp(0, 1)


def make_edit(x01, concept, dose, w, skin, les_mean):
    if concept == "contrast":
        return edit_contrast(x01, w, skin, les_mean, dose)
    if concept == "color_var":
        return edit_color_var(x01, w, skin, les_mean, dose)
    if concept == "size":
        return edit_size(x01, w, les_mean, dose)
    raise ValueError(concept)


# ---------------------------------------------------------------------------------------
# Manipulation-check proxies (measure the concept directly on the image, with the same original mask)
# ---------------------------------------------------------------------------------------
def proxies(x01, w, skin):
    """Returns a proxy dict: contrast=||les_mean-skin||; color_var=sqrt(colour var inside lesion); size=Σw."""
    den = w.sum().clamp_min(1e-6)
    les_mean = (x01 * w.unsqueeze(0)).sum((1, 2)) / den
    contrast = float((les_mean - skin).pow(2).sum().sqrt())
    var = ((x01 - les_mean.view(-1, 1, 1)).pow(2) * w.unsqueeze(0)).sum((1, 2)) / den
    color_var = float(var.sum().sqrt())
    return {"contrast": contrast, "color_var": color_var, "size": float(den)}


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


@torch.no_grad()
def model_prob(model, x01_batch):
    xn = p2._to_norm(x01_batch).to(DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
        return torch.softmax(model(xn), 1)[:, 1].float().cpu().numpy()


def precompute(imgs01):
    """Per image: mask + edits {concept_dose -> tensor} + manipulation-check proxy deltas."""
    edits, names = [], []
    # column name: f"{concept}@{k}" with k=0,1,2 (dose level)
    for concept, meta in CONCEPTS.items():
        for k in range(len(meta["doses"])):
            names.append(f"{concept}@{k}")
    mani = {n: {"intended": [], "leak": []} for n in names}
    for x01 in imgs01:
        w, skin, les_mean = soft_mask(x01)
        base_px = proxies(x01, w, skin)
        d = {}
        for concept, meta in CONCEPTS.items():
            for k, dose in enumerate(meta["doses"]):
                e = make_edit(x01, concept, dose, w, skin, les_mean)
                nm = f"{concept}@{k}"
                d[nm] = e
                # manipulation check (size: use the new mask because the geometry changes)
                wp = w if concept != "size" else soft_mask(e)[0]
                sp = skin if concept != "size" else soft_mask(e)[1]
                px = proxies(e, wp, sp)
                rel = (px[concept] - base_px[concept]) / (abs(base_px[concept]) + 1e-6)
                mani[nm]["intended"].append(rel)                  # change in the target concept (expected >0)
                others = [c for c in CONCEPTS if c != concept]
                leak = np.mean([abs(px[c] - base_px[c]) / (abs(base_px[c]) + 1e-6) for c in others])
                mani[nm]["leak"].append(leak)                     # leakage into other concepts
        edits.append(d)
    mani_summary = {n: {"intended": float(np.mean(mani[n]["intended"])),
                        "leak": float(np.mean(mani[n]["leak"]))} for n in names}
    return edits, names, mani_summary


def eval_model(model, imgs01, labels, edits, names, cfg):
    model.eval()
    dp = {n: [] for n in names}
    dl = {n: [] for n in names}
    p0_mal, kept = [], 0
    for i, (x01, lab) in enumerate(zip(imgs01, labels)):
        batch = torch.stack([x01] + [edits[i][n] for n in names])
        ps = model_prob(model, batch)
        p0 = ps[0]
        if lab == 1:
            p0_mal.append(p0)
        if not (lab == 1 and p0 >= cfg["p_floor"]):
            continue
        kept += 1
        for j, n in enumerate(names):
            dp[n].append(ps[j + 1] - p0)
            dl[n].append(logit(ps[j + 1]) - logit(p0))
    return dp, dl, kept, float(np.mean(p0_mal)) if p0_mal else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--boot-n", type=int, default=2000)
    ap.add_argument("--p-floor", type=float, default=P_FLOOR)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    cfg = dict(p_floor=args.p_floor)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"##### PHASE C3: CaCE concept {time.strftime('%H:%M:%S')} (dev={DEVICE}) #####", flush=True)
    df_tr, df_va = p2.build_split({"seed": SEED, "n_folds": 5})
    imgs01, labels = p2.load_eval_images(df_va, args.n_eval, SEED)
    print(f"eval: {len(imgs01)} images (truly malignant={int(sum(labels))})", flush=True)

    edits, names, mani = precompute(imgs01)
    print("manipulation-check (intended Δ% target concept / leak Δ% other concepts):")
    for n in names:
        print(f"   {n:14s} intended={100*mani[n]['intended']:+6.1f}%  leak={100*mani[n]['leak']:5.1f}%", flush=True)

    results = {}
    for name, fn in CKPTS.items():
        path = fn
        if not os.path.exists(path):
            print(f"[skip] missing {path}"); continue
        model = p1.build_model("erm" if name == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        t0 = time.time()
        dp, dl, kept, p0m = eval_model(model, imgs01, labels, edits, names, cfg)
        per = {}
        for n in names:
            m, lo, hi = p2.boot_mean_ci(dp[n], args.boot_n, SEED)
            lm, llo, lhi = p2.boot_mean_ci(dl[n], args.boot_n, SEED)
            per[n] = dict(cace_dp=m, cace_dp_ci=[lo, hi], cace_logit=lm, cace_logit_ci=[llo, lhi],
                          n=len(dp[n]))
        # monotonic in dose + expected sign (>0) at the highest dose, per concept
        concept_eval = {}
        for concept, meta in CONCEPTS.items():
            ks = [f"{concept}@{k}" for k in range(len(meta["doses"]))]
            vals = [per[k]["cace_logit"] for k in ks]
            top = per[ks[-1]]
            concept_eval[concept] = dict(
                cace_by_dose=vals,
                monotonic_pos=bool(vals[0] <= vals[1] <= vals[2]),
                top_logit=top["cace_logit"], top_ci=top["cace_logit_ci"],
                aligned_pos=bool(top["cace_logit_ci"][0] > 0),   # positive and CI excludes 0 = consistent with C1
            )
        results[name] = dict(n_kept=kept, pred_mal_prob=p0m, per_dose=per, per_concept=concept_eval)
        print(f"[{name}] done ({time.time()-t0:.0f}s, n_kept={kept}, p_mal={p0m:.3f})", flush=True)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    out = dict(
        phase="C3_cace_concept",
        estimand="CaCE_C(f)=E_x[f(edit_{C up}(x))-f(x)]; signed prob & logit; editing-only in lesion.",
        concept_map={c: CONCEPTS[c]["c1"] for c in CONCEPTS},
        manipulation_check=mani,
        eval=dict(n_images=len(imgs01), n_malignant=int(sum(labels)), p_floor=args.p_floor,
                  boot_n=args.boot_n, doses=DOSES,
                  excluded="size (manipulation-check failed: not cleanly editable -> future generative CF)"),
        models=results,
    )
    json.dump(out, open(args.out, "w"), indent=2, default=float)

    order = [m for m in ["erm", "slim", "causal"] if m in results]
    print("\n" + "=" * 96)
    print("CaCE CONCEPT: expect CaCE > 0 and monotonic (concepts with positive C1 ATE). aligned = CI(top)>0.")
    print("=" * 96)
    print(f"{'concept':12s}{'C1-map':24s}" + "".join(f"{m:>20s}" for m in order))
    print(f"{'':36s}" + "".join(f"{'top CaCE_logit mono':>20s}" for m in order))
    print("-" * 96)
    for concept in CONCEPTS:
        row = f"{concept:12s}{CONCEPTS[concept]['c1']:24s}"
        for m in order:
            ce = results[m]["per_concept"][concept]
            tag = "↑" if ce["monotonic_pos"] else "·"
            al = "✓" if ce["aligned_pos"] else "✗"
            row += f"{ce['top_logit']:+.2f}[{ce['top_ci'][0]:+.1f},{ce['top_ci'][1]:+.1f}]{tag}{al}".rjust(20)
        print(row)
    print("-" * 96)
    print(f"{'pred_mal_prob':36s}" + "".join(f"{results[m]['pred_mal_prob']:.3f}".rjust(20) for m in order))
    print("=" * 96)
    print("Reading: top CaCE_logit at the highest dose; ↑=monotonic in dose; ✓=positive and CI excludes 0 "
          "(model responds correctly to the C1 causal concept). ✗=not consistent (model ignores or reverses the disease concept).")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
