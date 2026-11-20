#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C12: matched-control analysis.

Open problem after C10/C11: they show that the control choice matters (A_corr changes sign or size
with the placebo), but not which control is causally valid. C12 turns the choice of control from an
*arbitrary choice* into a *testable assumption*, using a matching design.

--------------------------------------------------------------------------------------------------
ESTIMAND & IDENTIFICATION
--------------------------------------------------------------------------------------------------
For concept edit k, the target quantity is the concept-specific component of the model response:

    ΔCaCE_k(f) = CaCE_logit(edit_k) − CaCE_logit(control)                       (difference-in-differences)

Response model: CaCE(edit) = g(M(edit)) + h(concept change), where M(edit) is the perturbation
footprint of the edit in pixel space and g is the generic edit-sensitivity (not concept-specific).
ΔCaCE_k identifies h(·) if and only if the control satisfies:

  (A1) BALANCE / footprint-exchangeability : M(control) = M(edit_k) on every component of M
       -> g cancels in the difference. M has 5 checkable dimensions:
         (1) altered-pixel fraction, (2) mean absolute pixel change, (3) LAB (ΔE76) distance,
         (4) spatial support (RMS radius of the perturbation mass), (5) lesion-region coverage.
  (A2) NEUTRALITY / exclusion : the control does not change the concept proxy (measured against the
       original skin reference, c7.proxy_fixed) -> the control has no path through the concept channel.
  (A3) POSITIVITY / overlap : the control family must reach the concept edit's footprint on (almost)
       every image. If it cannot, overlap is violated, and that control is unusable and excluded.

All three conditions are measured from the images, independently of the model. So "which control is
valid" is an empirical question.

  (T)  Testable implication: if g depends only on M, then two valid controls from different families
       (different mechanism, same M, both neutral) must give equal CaCE: control-vs-control ≈ 0, and
       ΔCaCE must converge across families. This is a falsification test for the control class itself,
       which "control choice matters" does not provide. Conversely, unmatched controls with a different M
       give dispersed ΔCaCE (as in C10).

--------------------------------------------------------------------------------------------------
CONTROL DESIGN
--------------------------------------------------------------------------------------------------
Let d = x − skin (colour deviation from the background skin), Δ_k = edit_k(x) − x.
  * MATCHED-BY-CONSTRUCTION (footprint matched by construction, using Δ_k itself):
      rotd : x + R_120°(Δ_k). Rotates the vector Δ about the gray axis: per-pixel |Δ| is preserved exactly.
      sgnd : x + s⊙Δ_k, s=±1 random per pixel. Per-pixel |Δ| is preserved, net concept ≈ 0.
      permd: x + spatial permutation of Δ_k within the lesion. Same |Δ| histogram, spatial structure destroyed.
  * REARRANGEMENT (only moves real pixels within the lesion, so the multiset of pixel values is preserved;
    hence the contrast/color_var/size proxies are invariant; strength is tunable; never leaves the gamut):
      pixperm_m : random permutation of pixel values within the lesion (each pixel joins with probability q).
                  Exactly neutral (proxyΔ = 0.00%) for every q. Valid control candidate no. 1.
      warp_m    : elastic deformation (nearest sampling) within the lesion. Keeps local structure, with a
                  mechanism entirely different from pixperm, so this pair is the control-vs-control test.
  * FAMILY-MATCHED (C10 placebo families, strength calibrated by bisection to match mean|Δ| of edit_k):
      hue_m   : rotates d about the gray axis by angle θ (Rodrigues). Preserves per-pixel |d|, so it is
                concept-neutral by construction while strength stays tunable. Strength ceiling = 2|d_⊥gray|.
      rotax_m : same mechanism, but the axis is ⊥ to the mean colour-deviation direction, so the ceiling is
                2|d| (twice the reach, needed for high doses). hue_m and rotax_m are the two main candidate
                controls, with different mechanisms.
      flip_m / jitter_m / noise_m : blend α (or σ) with mirror / roll / Gaussian noise inside the mask.
  * UNMATCHED (as in C10, for comparison): hue, flip, jitter, noise at default strength.

Concept targets: contrast@{0,2}, color_var@{0,2} (lowest and highest dose). Each target has its own 10
controls, so 4 concepts + 4 unmatched + 4×10 matched = 48 edits/image, shared by all 3 models.

Note on positivity: a neutral "rearrangement" control can reach at most an amplitude equal to the
intrinsic colour dispersion of the lesion, while the concept edit amplifies the colour deviation (it
creates new colours outside the original distribution). So at high dose the concept edit can go beyond
the reach of every neutral control. Overlap is then violated, and ΔCaCE at that dose is not identified.
This is a result, not a bug: it explains why at high dose one is forced to use footprint-mismatched
controls (and hence reaches conflicting conclusions).

Out -> results/c12_matched_control.json + paper/figures/fig_matched_control.png
Run:  .venv/bin/python Causal_XAI_v3/code/c12_matched_control.py [--n-eval 200] [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c3_cace_concept as c3m       # noqa: E402  (soft_mask, make_edit, model_prob, logit, DOSES)
import audit_cohort               # noqa: E402
import c7_extra_controls as c7m     # noqa: E402  (proxy_fixed: concept proxy against the original skin reference)
import c10_multiplacebo as c10      # noqa: E402  (the 4 original placebos)

torch.set_num_threads(min(8, os.cpu_count() or 8))   # shared machine: avoid thrashing under high load
p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c12_matched_control.json")
FIGDIR = os.path.join(os.path.dirname(HERE), "paper", "figures")
import audited_models                # noqa: E402  (checkpoints are resolved in one place)
CKPTS = audited_models.CKPTS         # absolute paths to the frozen set
SEED, N_EVAL, BOOT = 42, 200, 2000
ORDER = ["erm", "slim", "causal"]
COL = {"erm": "#1f77b4", "slim": "#7f7f7f", "causal": "#d62728"}
NAME = {"erm": "ERM", "slim": "Slim", "causal": "Inv"}

TARGETS = ["contrast@2", "color_var@2", "contrast@0", "color_var@0"]
TOPDOSE = ["contrast@2", "color_var@2"]
CTRL_STRUCT = ["rotd", "sgnd", "permd"]                      # matched-by-construction
# Symmetric dose: the control is not a placebo but the same intervention in the opposite direction.
# It does not satisfy A2 (it is not neutral), and that is intended: the estimand becomes a dose contrast
#   ΔCaCE_sym = CaCE(C↑) − CaCE(C↓) = h(+δ) − h(−δ),
# where g(M) cancels because the two edits are mirror images of each other in pixel space (same
# per-pixel |Δ|, same support, same texture, same ΔE), which no neutral placebo can achieve.
CTRL_DOSE = ["invdose"]
CTRL_FAMILY = ["hybrid_m", "texswap_m", "globrot_m", "warp2_m", "pixperm_m", "warp_m",
               "hue_m", "rotax_m", "flip_m", "jitter_m", "noise_m"]   # family-matched (bisection)
HYBRID_B = np.linspace(0.0, 1.0, 9)      # grid for the texture-swap knob of the 2-knob control
MATCHED = CTRL_STRUCT + CTRL_FAMILY + CTRL_DOSE
UNMATCHED = ["hue", "flip", "jitter", "noise"]               # as in C10
# The 5 required matching covariates + hf_change: a sixth covariate that the leave-one-family-out
# test itself showed to be missing (two controls matched on the first 5 dimensions still differ by
# >10 logits, because they disrupt texture to very different degrees).
COVS = ["frac_altered", "mean_abs", "lab_dE", "support_rms", "lesion_cov", "hf_change"]
COVLAB = {"frac_altered": "altered-pixel frac", "mean_abs": "mean |Δ| pixel",
          "lab_dE": "LAB ΔE76", "support_rms": "spatial support", "lesion_cov": "lesion coverage",
          "hf_change": "high-freq (texture)"}

# Neutral dose ladder: 2 rearrangement families × 5 strength levels (multiples of mean|Δ| of contrast@2).
# Used to estimate the generic sensitivity function g(M) instead of relying on a single control.
SIGMAS = [2, 8, 24]                 # smoothness of the displacement field (px)
LADDER_MULT = [0.25, 0.5, 1.0, 1.5, 2.0]
# 2-D dose ladder (strength × smoothness) to estimate g(M) along the texture axis too, plus a family
# with a different mechanism (pixperm) held out for leave-one-family-out testing of assumption A1.
LADDER_FAM = ["globrot_m", "w2s24", "w2s2", "pixperm_m"]   # covers the texture axis from ~0 to the highest
LOFO_GROUPS = {"globrot": ["globrot_m"], "warp2": ["w2s24", "w2s2"], "pixperm": ["pixperm_m"]}

SMD_TOL = 0.10        # |SMD| <= 0.10 -> balanced (matching convention)
SMD_TOL_OK = 0.25     # 0.10 = ideal; 0.25 = classic "balanced enough" threshold (Rosenbaum-Rubin)
NEUTRAL_TOL = 0.10    # absolute |Δ| of concept proxy <= 10%
NEUTRAL_RATIO = 0.15  # and <= 15% of the proxy shift of the concept edit itself (bounds first-order bias of ΔCaCE)
MATCH_TOL = 0.02      # mean|Δ| matching error <= 2%
POS_TOL = 0.95        # >=95% of images matched -> positivity holds

_GRID = {}


# ==================================================================================================
# Footprint metrics
# ==================================================================================================
def _grid(H, W):
    if (H, W) not in _GRID:
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32), indexing="ij")
        _GRID[(H, W)] = (yy, xx)
    return _GRID[(H, W)]


def rgb2lab(x01):
    """sRGB [0,1] [3,H,W] -> CIELAB (D65). Used for the perceptual distance ΔE76."""
    m = torch.tensor([[0.4124, 0.3576, 0.1805],
                      [0.2126, 0.7152, 0.0722],
                      [0.0193, 0.1192, 0.9505]], dtype=x01.dtype)
    lin = torch.where(x01 > 0.04045, ((x01 + 0.055) / 1.055).clamp_min(1e-8) ** 2.4, x01 / 12.92)
    xyz = torch.einsum("ij,jhw->ihw", m, lin)
    white = torch.tensor([0.95047, 1.0, 1.08883], dtype=x01.dtype).view(3, 1, 1)
    t = xyz / white
    f = torch.where(t > 0.008856, t.clamp_min(1e-8) ** (1.0 / 3.0), 7.787 * t + 16.0 / 116.0)
    return torch.stack([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])], 0)


def hf_map(x, k=5):
    """High-frequency (texture) energy = ‖x − blur_k(x)‖ per pixel."""
    ker = torch.ones(3, 1, k, k, dtype=x.dtype) / (k * k)
    b = torch.nn.functional.conv2d(x.unsqueeze(0), ker, padding=k // 2, groups=3)[0]
    return (x - b).pow(2).sum(0).sqrt()


def dpix(x, e):
    """Per-pixel change magnitude = L2 norm over the 3 colour channels."""
    return (e - x).pow(2).sum(0).sqrt()


def footprint(x, e, w):
    """5 matching covariates describing the pixel-space footprint of an intervention."""
    d = dpix(x, e)
    H, W = d.shape
    tot = d.sum().clamp_min(1e-8)
    yy, xx = _grid(H, W)
    cy, cx = (d * yy).sum() / tot, (d * xx).sum() / tot
    rms = ((d * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / tot).sqrt() / H
    return {"frac_altered": float((d > 0.02).float().mean()),
            "mean_abs": float(d.mean()),
            "lab_dE": float((rgb2lab(e) - rgb2lab(x)).pow(2).sum(0).sqrt().mean()),
            "support_rms": float(rms),
            "lesion_cov": float((d * w).sum() / tot),
            "hf_change": float((hf_map(e) - hf_map(x)).abs().mean())}


def support_iou(x, e1, e2, thr=0.02):
    """Spatial overlap of two edited regions (checks whether the control touches the same pixels)."""
    a, b = dpix(x, e1) > thr, dpix(x, e2) > thr
    u = (a | b).sum().clamp_min(1)
    return float((a & b).sum() / u)


# ==================================================================================================
# Control builders
# ==================================================================================================
def axis_rot(d, k, theta):
    """Rotate colour vector d[3,H,W] about unit axis k[3] by angle theta (Rodrigues). Preserves the per-pixel norm.

    Since the rotation is linear and norm-preserving, applied to every pixel it keeps exactly: ‖lesmean−skin‖
    (contrast proxy), total colour variance inside the lesion (color_var proxy) and ‖x−skin‖ (=> mask/size).
    This is a control that is concept-neutral by construction, while its strength stays tunable via theta.
    """
    cross = torch.stack([k[1] * d[2] - k[2] * d[1],
                         k[2] * d[0] - k[0] * d[2],
                         k[0] * d[1] - k[1] * d[0]], 0)
    kdot = k[0] * d[0] + k[1] * d[1] + k[2] * d[2]
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return d * c + cross * s + k.view(3, 1, 1) * kdot.unsqueeze(0) * (1 - c)


def gray_rot(d, theta):
    """Rotation about the gray axis (1,1,1)/√3: the classic hue rotation (theta=120° = cyclic channel permutation)."""
    return axis_rot(d, torch.full((3,), 1.0 / float(np.sqrt(3.0)), dtype=d.dtype), theta)


def pix_perm_fn(x, w, seed):
    """Permute pixel values inside the lesion; each pixel joins if u_p < q (nested sets).

    The three concept proxies (contrast=‖lesmean−skin‖, color_var, size=Σw) are all functionals of the multiset
    of pixel values (w_p = clamp(‖x_p−skin0‖/scale) goes with that same value), so every permutation preserves them
    exactly for every q. This gives a perfectly neutral control, with tunable strength, that never leaves the
    gamut (every value is a real pixel value of the image).
    """
    rng = np.random.RandomState(seed + 313)
    idx = torch.nonzero(w.reshape(-1) > 0.05, as_tuple=False).reshape(-1)
    # participation probability ~ w: the perturbation mass concentrates in the lesion core, like the concept edit
    # (concept Δ ∝ w·|d|), so that 'lesion coverage' and 'altered fraction' can be balanced.
    u = torch.from_numpy(rng.rand(len(idx)).astype(np.float32)) / w.reshape(-1)[idx].clamp_min(1e-3)
    key = torch.from_numpy(rng.rand(len(idx)).astype(np.float32))
    flat = x.reshape(3, -1)

    def fn(q):
        m = u < q
        sub = idx[m]
        if len(sub) < 2:
            return x.clone()
        out = flat.clone()
        out[:, sub] = flat[:, sub[torch.argsort(key[m])]]
        return out.reshape(x.shape)
    return fn


def globrot_fn(x, skin, nax):
    """Rotate the vector (x−skin) of the whole image about axis nax by angle theta.

    It is a linear, norm-preserving map applied to every pixel (no soft-mask blending), so:
      * the contrast/color_var/size proxies are exactly invariant  -> concept-neutral by construction;
      * the high-frequency map is invariant (‖R(x−blur x)‖ = ‖x−blur x‖) -> texture is not disrupted;
      * |Δ| = 2 sin(θ/2)|d_⊥| with ceiling 2|d| -> it can reach even the highest concept dose.
    Only the gamut clamp introduces deviations (measured: proxy ratio ~1-2%).
    """
    d = x - skin.view(-1, 1, 1)
    return lambda th: torch.clamp(skin.view(-1, 1, 1) + axis_rot(d, nax, th), 0, 1)


def warp2_fn(x, w, seed, sigma):
    """Warp with a displacement field of smoothness sigma px: the second knob controls texture disruption.

    Large sigma -> smooth displacement, texture kept (small hf_change); small sigma -> local shuffling, texture disrupted.
    The two knobs (amplitude, sigma) are what allow matching mean|Δ| and hf_change of the concept edit at the same time.
    """
    C, H, W = x.shape
    rng = np.random.RandomState(seed + 977 + int(sigma))
    g = max(2, int(round(H / max(sigma, 1.0))))
    f = torch.from_numpy(rng.normal(0, 1, (2, 1, g, g)).astype(np.float32))
    f = torch.nn.functional.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    f = f / f.std().clamp_min(1e-6) * w * w
    yy, xx = _grid(H, W)

    def fn(amp):
        ys = (yy + (f[0] * amp).round()).clamp(0, H - 1).long()
        xs = (xx + (f[1] * amp).round()).clamp(0, W - 1).long()
        return x[:, ys.reshape(-1), xs.reshape(-1)].reshape(C, H, W)
    return fn


def warp_fn(x, w, seed):
    """Elastic deformation inside the lesion (nearest sampling along a smooth displacement field, amplitude amp).

    It only moves real pixels -> the colour distribution is nearly preserved (small proxy shift). Its mechanism is
    entirely different from pix-perm (it keeps local structure) -> used as the second control in the control-vs-control test.
    """
    C, H, W = x.shape
    rng = np.random.RandomState(seed + 707)
    f = torch.from_numpy(rng.normal(0, 1, (2, 1, max(2, H // 8), max(2, W // 8))).astype(np.float32))
    f = torch.nn.functional.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    f = f / f.std().clamp_min(1e-6) * w * w      # ~w²: displacement concentrated in the core, border kept intact
    yy, xx = _grid(H, W)

    def fn(amp):
        ys = (yy + (f[0] * amp).round()).clamp(0, H - 1).long()
        xs = (xx + (f[1] * amp).round()).clamp(0, W - 1).long()
        return x[:, ys.reshape(-1), xs.reshape(-1)].reshape(C, H, W)
    return fn


def hf_split(x, k=5):
    """Split the image into a low band (blur) and a high band (texture)."""
    ker = torch.ones(3, 1, k, k, dtype=x.dtype) / (k * k)
    lf = torch.nn.functional.conv2d(x.unsqueeze(0), ker, padding=k // 2, groups=3)[0]
    return lf, x - lf


def texswap_fn(x, w, donor):
    """Texture swap: replace the high-frequency band inside the lesion with that of another image's lesion.

    Since the high band has local mean ~0 and we normalise it to the same energy (then mix in an
    energy-preserving way), this nearly preserves: lesmean (=> contrast proxy), total colour variance
    (=> color_var proxy) and ‖x−skin‖ (=> mask/size). But it does change hf_change, which is exactly the covariate
    that no earlier neutral control could match. The texture comes from a real lesion => in-distribution.
    """
    lf, hf = hf_split(x)
    hfd = hf_split(donor)[1]
    den = w.sum().clamp_min(1e-6)
    sx = ((hf.pow(2).sum(0) * w).sum() / den).sqrt()
    sd = ((hfd.pow(2).sum(0) * w).sum() / den).sqrt()
    hfd = hfd * (sx / sd.clamp_min(1e-8))
    w3 = w.unsqueeze(0)

    def fn(b):
        mix = ((1 - b) * hf + b * hfd) / float(np.sqrt((1 - b) ** 2 + b ** 2))   # energy-preserving
        return torch.clamp(x + w3 * (mix - hf), 0, 1)
    return fn


def hybrid_fn(x, w, skin, nax, donor):
    """Two knobs: (theta) norm-preserving colour rotation + (b) texture swap.

    The theta knob injects perturbation amplitude without touching texture; the b knob injects texture without
    touching the concept proxy. This makes it possible to match mean|Δ| and hf_change of the concept edit at the
    same time, while staying neutral. This is a valid control candidate under all of A1/A2/A3.
    """
    tex = texswap_fn(x, w, donor)
    d = x - skin.view(-1, 1, 1); w3 = w.unsqueeze(0)

    def base(b):
        e = tex(b)
        return lambda th: torch.clamp(e + w3 * (axis_rot(d, nax, th) - d), 0, 1)
    return base


def make_families(x, w, skin, seed, donor=None):
    """Return {name: (fn(strength) -> image, lo, hi)}. Noise/shift are fixed per image => mag is monotone in s."""
    rng = np.random.RandomState(seed)
    noise = torch.from_numpy(rng.normal(0, 1, x.shape).astype(np.float32))
    sh = (int(rng.randint(4, 9)) * (1 if rng.rand() > .5 else -1),
          int(rng.randint(4, 9)) * (1 if rng.rand() > .5 else -1))
    d = x - skin.view(-1, 1, 1)
    w3 = w.unsqueeze(0)
    xf, xr = torch.flip(x, dims=[2]), torch.roll(x, shifts=sh, dims=(1, 2))
    # rotation axis perpendicular to the mean colour-deviation direction of the lesion -> |Δ| reaches ~2|d| (enough
    # for every concept dose, g<=1), whereas hue about the gray axis only reaches 2|d_⊥gray| (may fall short).
    dm = (d * w3).sum((1, 2)) / w.sum().clamp_min(1e-6)
    g3 = torch.full((3,), 1.0 / float(np.sqrt(3.0)), dtype=x.dtype)
    nax = torch.linalg.cross(dm, g3)
    if float(nax.norm()) < 1e-6:
        nax = torch.linalg.cross(dm, torch.tensor([1.0, 0.0, 0.0], dtype=x.dtype))
    nax = nax / nax.norm().clamp_min(1e-8)
    out = {
        # whole-image colour rotation: neutral + keeps texture + enough reach (see globrot_fn)
        "globrot_m": (globrot_fn(x, skin, nax), 0.0, float(np.pi)),
        # Rearrangement controls: only move pixels -> colour distribution (=> concept proxy) preserved
        "pixperm_m": (pix_perm_fn(x, w, seed), 0.0, 1.0),
        "warp_m":    (warp_fn(x, w, seed), 0.0, 30.0),
        # hue rotation about the gray axis: |d| is preserved before the clamp, but soft-mask blending + gamut clamp
        # shift the proxy -> kept in the table as a rejected candidate (illustrates the neutrality criterion).
        "hue_m":    (lambda s: torch.clamp(x + w3 * (gray_rot(d, s) - d), 0, 1), 0.0, float(np.pi)),
        # rotation about the axis ⊥ to the mean colour-deviation direction: also norm-preserving, but twice the reach
        "rotax_m":  (lambda s: torch.clamp(x + w3 * (axis_rot(d, nax, s) - d), 0, 1), 0.0, float(np.pi)),
        "flip_m":   (lambda s: torch.clamp(x + s * w3 * (xf - x), 0, 1), 0.0, 1.0),
        "jitter_m": (lambda s: torch.clamp(x + s * w3 * (xr - x), 0, 1), 0.0, 1.0),
        "noise_m":  (lambda s: torch.clamp(x + s * w3 * noise, 0, 1), 0.0, 0.60),
    }
    for sg in SIGMAS:                       # 2-knob warp: each sigma is a sub-family, selected by texture
        out[f"w2s{sg}"] = (warp2_fn(x, w, seed, sg), 0.0, 40.0)
    if donor is not None:
        out["texswap_m"] = (texswap_fn(x, w, donor), 0.0, 1.0)
        out["_hybrid"] = hybrid_fn(x, w, skin, nax, donor)     # 2 knobs -> handled separately
    return out


def match_strength(fn, lo, hi, x, target, iters=22):
    """Bisection on strength so that mean|Δ| matches target. Returns (edit, s, mag, ok_positivity)."""
    e_hi = fn(hi)
    if float(dpix(x, e_hi).mean()) < target * (1 - MATCH_TOL):
        return e_hi, hi, float(dpix(x, e_hi).mean()), False      # cannot reach -> overlap violated
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(dpix(x, fn(mid)).mean()) < target:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi); e = fn(s); m = float(dpix(x, e).mean())
    return e, s, m, bool(abs(m - target) / max(target, 1e-8) <= MATCH_TOL)


def match_hybrid(base, x, w, tgt_mag, tgt_hf):
    """Sweep the texture-swap knob b; for each b, bisect on theta to match mean|Δ|; pick the b that best matches hf_change."""
    best = None
    for b in HYBRID_B:
        fn = base(float(b))
        e, th, m, ok = match_strength(fn, 0.0, float(np.pi), x, tgt_mag)
        err = abs(footprint(x, e, w)["hf_change"] - tgt_hf) / max(abs(tgt_hf), 1e-9)
        if best is None or err < best[0]:
            best = (err, e, float(b), float(th), ok)
    return best[1], best[2], best[3], best[4], best[0]


def match_2knob(fams, x, tgt_mag, tgt_hf, w):
    """Match mean|Δ| (amplitude bisection) and hf_change (best sigma) at the same time."""
    best = None
    for sg in SIGMAS:
        fn, lo, hi = fams[f"w2s{sg}"]
        e, s_, m, ok = match_strength(fn, lo, hi, x, tgt_mag)
        err = abs(footprint(x, e, w)["hf_change"] - tgt_hf) / max(abs(tgt_hf), 1e-8)
        if best is None or err < best[0]:
            best = (err, e, s_, m, ok, sg)
    return best[1], best[2], best[4], best[5]


def struct_controls(x, delta, w, seed):
    """Controls with footprint matched by construction, built directly from the Δ of the concept edit."""
    rng = np.random.RandomState(seed + 991)
    out = {"rotd": torch.clamp(x + gray_rot(delta, 2 * np.pi / 3), 0, 1)}
    sgn = torch.from_numpy(np.where(rng.rand(1, *delta.shape[1:]) > 0.5, 1.0, -1.0).astype(np.float32))
    out["sgnd"] = torch.clamp(x + sgn * delta, 0, 1)
    idx = torch.nonzero(w.reshape(-1) > 0.05, as_tuple=False).reshape(-1)
    dp = delta.reshape(3, -1).clone()
    if len(idx) > 1:
        perm = idx[torch.from_numpy(rng.permutation(len(idx)))]
        dp[:, idx] = delta.reshape(3, -1)[:, perm]
    out["permd"] = torch.clamp(x + dp.reshape(delta.shape), 0, 1)
    return out


# ==================================================================================================
def boot(v, nboot, seed=SEED):
    return p2.boot_mean_ci(np.asarray(v, float), nboot, seed)


def smd(a, b):
    """Standardized mean difference (paired): (mean_ctrl − mean_concept)/pooled SD."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    sd = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    d = a.mean() - b.mean()
    if sd > 1e-12:
        return float(d / sd)
    # zero pooled variance: balanced only if the means agree (c71_rule_checks finds no such pair here)
    return 0.0 if d == 0.0 else float(np.copysign(np.inf, d))


def rel_proxy_shift(base, px):
    """Relative |Δ| of the concept proxies (contrast, color_var) vs the original image. 0 = neutral."""
    return float(np.mean([abs(px[c] - base[c]) / (abs(base[c]) + 1e-6) for c in ("contrast", "color_var")]))


# ==================================================================================================
# Estimate the generic sensitivity function g(M) and the adjusted ΔCaCE
# --------------------------------------------------------------------------------------------------
# Instead of picking one control and hoping it matches, we estimate g(M) from the whole neutral control
# dose ladder and read g at the exact footprint of the concept edit:  ΔCaCE_adj = CaCE(edit) − ĝ(M(edit)).
# This is the "counterfactual control": the response that a neutral intervention with the same footprint would produce.
# Within-image regression (image means removed), so differing baseline sensitivity across images adds no noise.
# ==================================================================================================
def _feat(M):
    """M[..., 5] -> linear features + the square of mean|Δ| (nonlinear in strength)."""
    return np.concatenate([M, M[..., 1:2] ** 2], -1)


def fit_g(Mtr, ytr):
    """Mtr [nimg, ntrain, 5], ytr [nimg, ntrain] -> (beta, Fbar, ybar, scale) of the within-image regression."""
    F = _feat(Mtr)
    Fbar, ybar = F.mean(1, keepdims=True), ytr.mean(1, keepdims=True)
    X = (F - Fbar).reshape(-1, F.shape[-1]); y = (ytr - ybar).reshape(-1)
    sc = X.std(0); sc[sc < 1e-12] = 1.0
    beta = np.linalg.lstsq(X / sc, y, rcond=None)[0]
    return beta, Fbar[:, 0], ybar[:, 0], sc


def pred_g(fit, M):
    beta, Fbar, ybar, sc = fit
    return ybar + ((_feat(M) - Fbar) / sc) @ beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--boot", type=int, default=BOOT)
    a = ap.parse_args()
    t0 = time.time()
    print(f"##### C12 matched-control {time.strftime('%H:%M:%S')} #####", flush=True)

    imgs01, labels, _pid, _ids = audit_cohort.load()   # fixed evaluation set
    labels = np.array(labels); mal = np.where(labels == 1)[0]
    print(f"malignant {len(mal)}  device={DEVICE}", flush=True)

    # ---------------------------------------------------------------------------------------------
    # 1) Build edits + measure footprint/neutrality (no model needed -> validity is a design property)
    # ---------------------------------------------------------------------------------------------
    LADDER = [f"L:{c}@{j}" for c in LADDER_FAM for j in range(len(LADDER_MULT))]   # 4 families × 5 levels
    names = list(TARGETS) + [f"u:{p}" for p in UNMATCHED] + \
            [f"{t}|{c}" for t in TARGETS for c in MATCHED] + LADDER
    edits = []
    FP = {n: {k: [] for k in COVS} for n in names}       # footprint per edit
    IOU = {f"{t}|{c}": [] for t in TARGETS for c in MATCHED}
    IOU.update({f"{t}|u:{p}": [] for t in TARGETS for p in UNMATCHED})
    PRX = {n: [] for n in names}                        # rel. proxy-concept shift
    POS = {f"{t}|{c}": [] for t in TARGETS for c in CTRL_FAMILY + CTRL_DOSE}   # positivity
    STR = {f"{t}|{c}": [] for t in TARGETS for c in CTRL_FAMILY + CTRL_DOSE}   # calibrated strength
    SIG = {f"{t}|warp2_m": [] for t in TARGETS}                    # chosen sigma (texture match)
    PRXS = {n: [] for n in list(TARGETS) + [f"{t}|invdose" for t in TARGETS]}   # signed proxy shift
    HYB = {t: [] for t in TARGETS}                                 # chosen texture-swap knob b

    for ii in mal:
        x = imgs01[ii]
        w, skin, lm = c3m.soft_mask(x)
        base = c7m.proxy_fixed(x, skin)
        donor = imgs01[mal[(len(edits) + 1) % len(mal)]]                # texture from the next lesion
        fams = make_families(x, w, skin, SEED + int(ii), donor=donor)
        d = {}
        for t in TARGETS:                                     # concept edits
            cname, k = t.split("@")
            d[t] = c3m.make_edit(x, cname, c3m.DOSES[int(k)], w, skin, lm)
        for p in UNMATCHED:                                   # original placebos (C10)
            d[f"u:{p}"] = c10.PLACEBOS[p](x, w, skin, np.random.RandomState(SEED + int(ii) + 7))
        for t in TARGETS:                                     # controls matched to each target
            tgt_mag = float(dpix(x, d[t]).mean())
            for c, e in struct_controls(x, d[t] - x, w, SEED + int(ii)).items():
                d[f"{t}|{c}"] = e
            tgt_hf = footprint(x, d[t], w)["hf_change"]
            for c in CTRL_FAMILY:
                if c == "hybrid_m":                      # 2 knobs: colour rotation + texture swap
                    e, bb, th, ok, _ = match_hybrid(fams["_hybrid"], x, w, tgt_mag, tgt_hf)
                    s = th; HYB[f"{t}"].append(bb)
                elif c == "warp2_m":                     # match strength + texture at the same time
                    e, s, ok, sg = match_2knob(fams, x, tgt_mag, tgt_hf, w)
                    SIG[f"{t}|warp2_m"].append(int(sg))
                else:
                    fn, lo, hi = fams[c]
                    e, s, m, ok = match_strength(fn, lo, hi, x, tgt_mag)
                d[f"{t}|{c}"] = e
                POS[f"{t}|{c}"].append(bool(ok)); STR[f"{t}|{c}"].append(float(s))
        for t in TARGETS:                                      # reversed dose: same operator, opposite direction
            cname, k = t.split("@")                            # match mean|Δ| by bisection on |g'|
            # (needed because the gamut clamp saturates the increasing edit but not the decreasing one -> ~9% off if g'=g)
            e, gp, mm_, ok = match_strength(
                lambda gp: c3m.make_edit(x, cname, -gp, w, skin, lm), 0.0, 1.0, x,
                float(dpix(x, d[t]).mean()))
            d[f"{t}|invdose"] = e
            POS[f"{t}|invdose"].append(bool(ok)); STR[f"{t}|invdose"].append(float(gp))
            for nm in (t, f"{t}|invdose"):
                px = c7m.proxy_fixed(d[nm], skin)
                key = "contrast" if cname == "contrast" else "color_var"
                PRXS[nm].append((px[key] - base[key]) / (abs(base[key]) + 1e-6))
        ref_mag = float(dpix(x, d["contrast@2"]).mean())       # neutral dose ladder (fit g(M))
        for c in LADDER_FAM:
            fn, lo, hi = fams[c]
            for j, mult in enumerate(LADDER_MULT):
                d[f"L:{c}@{j}"] = match_strength(fn, lo, hi, x, ref_mag * mult)[0]
        for n in names:                                       # measure the 5 covariates + neutrality
            fp = footprint(x, d[n], w)
            for k in COVS: FP[n][k].append(fp[k])
            PRX[n].append(rel_proxy_shift(base, c7m.proxy_fixed(d[n], skin)))
        for t in TARGETS:
            for c in MATCHED: IOU[f"{t}|{c}"].append(support_iou(x, d[t], d[f"{t}|{c}"]))
            for p in UNMATCHED: IOU[f"{t}|u:{p}"].append(support_iou(x, d[t], d[f"u:{p}"]))
        edits.append(d)
        if (len(edits) % 10) == 0:
            print(f"   ... {len(edits)}/{len(mal)} images ({time.time()-t0:.0f}s)", flush=True)
    print(f"built {len(names)} edits/image in {time.time()-t0:.0f}s", flush=True)

    # ---- balance + validity (per target x control) ----
    design = {}
    for t in TARGETS:
        design[t] = {}
        prx_concept = float(np.mean(PRX[t]))
        for c in MATCHED + [f"u:{p}" for p in UNMATCHED]:
            key = f"{t}|{c}" if c in MATCHED else f"u:{c[2:]}"
            sm = {k: smd(FP[key][k], FP[t][k]) for k in COVS}
            pos = float(np.mean(POS[f"{t}|{c}"])) if f"{t}|{c}" in POS else 1.0
            neu = float(np.mean(PRX[key]))
            ratio = neu / max(prx_concept, 1e-6)
            bal = float(max(abs(v) for v in sm.values()))
            design[t][c] = {
                "smd": sm, "max_abs_smd": bal, "balanced": bool(bal <= SMD_TOL),
                "proxy_shift": neu, "proxy_ratio": ratio,
                "neutral": bool(neu <= NEUTRAL_TOL and ratio <= NEUTRAL_RATIO),
                "positivity": pos, "overlap_ok": bool(pos >= POS_TOL),
                "support_iou": float(np.mean(IOU[f"{t}|{c}"] if c in MATCHED else IOU[f"{t}|u:{c[2:]}"])),
                "footprint": {k: float(np.mean(FP[key][k])) for k in COVS},
                "strength": float(np.mean(STR[f"{t}|{c}"])) if f"{t}|{c}" in STR else None,
                "sigma_chosen": (float(np.mean(SIG[f"{t}|warp2_m"])) if c == "warp2_m" else None),
                "texswap_b": (float(np.mean(HYB[t])) if c == "hybrid_m" else None),
            }
            design[t][c]["balanced_ok"] = bool(bal <= SMD_TOL_OK)
            design[t][c]["valid"] = bool(design[t][c]["balanced"] and design[t][c]["neutral"]
                                         and design[t][c]["overlap_ok"])
            design[t][c]["valid_ok"] = bool(design[t][c]["balanced_ok"] and design[t][c]["neutral"]
                                            and design[t][c]["overlap_ok"])
        design[t]["invdose"]["antisymmetric"] = None      # filled in once the signed proxy is available
        design[t]["_concept"] = {"footprint": {k: float(np.mean(FP[t][k])) for k in COVS},
                                 "proxy_shift": float(np.mean(PRX[t]))}

    # ---- overlap of the neutral dose ladder: does the concept footprint lie inside the ladder range? ----
    Mall = {n: np.array([FP[n][k] for k in COVS], float).T for n in names}     # [n_img, 5]
    Mlad = np.stack([Mall[n] for n in LADDER], 1)                             # [n_img, n_ladder, 5]
    for t in TARGETS:
        inr = ((Mall[t] >= Mlad.min(1) - 1e-12) & (Mall[t] <= Mlad.max(1) + 1e-12))
        design[t]["_ladder_overlap"] = {k: float(inr[:, i].mean()) for i, k in enumerate(COVS)}
        design[t]["_ladder_overlap"]["all_covs"] = float(inr.all(1).mean())

    # ---------------------------------------------------------------------------------------------
    # 2) CaCE for every edit, 3 models, shared edits, common set p_full>=0.1
    # ---------------------------------------------------------------------------------------------
    P0, DL = {}, {}
    for m in ORDER:
        model = p1.build_model("erm" if m == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(CKPTS[m], map_location=DEVICE))
        model.eval()
        p0 = np.empty(len(mal)); dl = {n: np.empty(len(mal)) for n in names}
        for j in range(len(mal)):
            ps = c3m.model_prob(model, torch.stack([imgs01[mal[j]]] + [edits[j][n] for n in names]))
            p0[j] = ps[0]
            for k, n in enumerate(names):
                dl[n][j] = c3m.logit(ps[k + 1]) - c3m.logit(ps[0])
        P0[m], DL[m] = p0, dl
        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()
        print(f"[{m}] p_full={p0.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # Save per-image responses so the partial-identification analysis (c45) need not rebuild the edits.
    cache = {"names": np.array(names, dtype=object), "img_index": np.array(mal)}
    for m in ORDER:
        cache[f"p0_{m}"] = P0[m]
        for n in names:
            cache[f"dl_{m}|{n}"] = DL[m][n]
    np.savez_compressed(os.path.join(RES, "c12_preds.npz"), **cache)
    print(f"  saved per-image responses -> c12_preds.npz ({len(names)} edits)", flush=True)

    common = (P0["erm"] >= 0.1) & (P0["slim"] >= 0.1) & (P0["causal"] >= 0.1)
    nC = int(common.sum())
    print(f"common set n={nC}", flush=True)

    # ---------------------------------------------------------------------------------------------
    # 3) ΔCaCE, convergence, control-vs-control falsification
    # ---------------------------------------------------------------------------------------------
    res = {"n_malignant": int(len(mal)), "n_common": nC, "n_boot": a.boot,
           "criteria": {"smd_tol": SMD_TOL, "neutral_tol": NEUTRAL_TOL,
                        "neutral_ratio_tol": NEUTRAL_RATIO,
                        "match_tol": MATCH_TOL, "positivity_tol": POS_TOL},
           "design": design, "models": {}}

    for m in ORDER:
        dl = DL[m]
        mm = {"cace": {}, "dcace": {}, "spread": {}, "ctrl_vs_ctrl": {}, "alignment": {}}
        for n in names:
            mn, lo, hi = boot(dl[n][common], a.boot)
            mm["cace"][n] = {"mean": mn, "ci": [lo, hi], "sig": bool(not (lo <= 0 <= hi))}
        for t in TARGETS:
            mm["dcace"][t] = {}
            for c in MATCHED + [f"u:{p}" for p in UNMATCHED]:
                key = f"{t}|{c}" if c in MATCHED else f"u:{c[2:]}"
                mn, lo, hi = boot((dl[t] - dl[key])[common], a.boot)
                mm["dcace"][t][c] = {"mean": mn, "ci": [lo, hi], "sig": bool(not (lo <= 0 <= hi)),
                                     "valid": design[t][c]["valid"], "valid_ok": design[t][c]["valid_ok"]}
            # Convergence: dispersion of ΔCaCE across controls, unmatched vs matched-valid
            un = [mm["dcace"][t][f"u:{p}"]["mean"] for p in UNMATCHED]
            va = [mm["dcace"][t][c]["mean"] for c in MATCHED if design[t][c]["valid_ok"]]
            al = [mm["dcace"][t][c]["mean"] for c in MATCHED]
            mm["spread"][t] = {
                "unmatched": {"sd": float(np.std(un, ddof=1)), "range": float(np.ptp(un)), "n": len(un)},
                "matched_all": {"sd": float(np.std(al, ddof=1)), "range": float(np.ptp(al)), "n": len(al)},
                "matched_valid": ({"sd": float(np.std(va, ddof=1)), "range": float(np.ptp(va)),
                                   "n": len(va)} if len(va) > 1 else
                                  {"sd": None, "range": None, "n": len(va)}),
                "sign_consistent_valid": bool(len(va) > 0 and (all(v > 0 for v in va) or all(v < 0 for v in va))),
            }
            # Falsification: two neutral controls (different mechanism, same strength) must give equal CaCE
            # if the generic response depends only on the footprint. A difference means neutrality + strength
            # matching still do not pin down the control => ΔCaCE is not identified.
            vc = [c for c in MATCHED if design[t][c]["neutral"] and design[t][c]["overlap_ok"]]
            pairs = {}
            for i in range(len(vc)):
                for jj in range(i + 1, len(vc)):
                    mn, lo, hi = boot((dl[f"{t}|{vc[i]}"] - dl[f"{t}|{vc[jj]}"])[common], a.boot)
                    pairs[f"{vc[i]}-{vc[jj]}"] = {"mean": mn, "ci": [lo, hi],
                                                  "null_ok": bool(lo <= 0 <= hi)}
            mm["ctrl_vs_ctrl"][t] = {"controls": vc, "pairs": pairs,
                                     "all_null": bool(len(pairs) > 0 and all(p["null_ok"] for p in pairs.values())),
                                     "n_pairs": len(pairs)}
        # alignment (2 high-dose concepts) under valid controls vs under unmatched placebos
        for c in MATCHED + [f"u:{p}" for p in UNMATCHED]:
            per = {}
            for t in TOPDOSE:
                key = f"{t}|{c}" if c in MATCHED else f"u:{c[2:]}"
                per[t] = boot((dl[t] - dl[key])[common], a.boot)
            mm["alignment"][c] = {
                "A": float(np.mean([per[t][0] for t in TOPDOSE])),
                "per_concept": {t: {"mean": per[t][0], "ci": [per[t][1], per[t][2]]} for t in TOPDOSE},
                "aligned": bool(all(per[t][1] > 0 for t in TOPDOSE)),
                "anti": bool(all(per[t][2] < 0 for t in TOPDOSE)),
                "valid": all(design[t][c]["valid_ok"] for t in TOPDOSE),
            }

        # ---- Adjusted ΔCaCE: read ĝ(M) at the exact footprint of the concept edit -------------------------
        idxc = np.where(common)[0]
        Mtr = np.stack([Mall[n][idxc] for n in LADDER], 1)
        ytr = np.stack([dl[n][idxc] for n in LADDER], 1)
        fit = fit_g(Mtr, ytr)
        mm["g_adjusted"] = {}
        for t in TARGETS:
            dadj = dl[t][idxc] - pred_g(fit, Mall[t][idxc])
            mn, lo, hi = boot(dadj, a.boot)
            mm["g_adjusted"][t] = {"mean": mn, "ci": [lo, hi], "sig": bool(not (lo <= 0 <= hi)),
                                   "ladder_overlap": design[t]["_ladder_overlap"]["all_covs"]}
        mm["g_adjusted"]["A"] = float(np.mean([mm["g_adjusted"][t]["mean"] for t in TOPDOSE]))
        mm["g_adjusted"]["aligned"] = bool(all(mm["g_adjusted"][t]["ci"][0] > 0 for t in TOPDOSE))
        # Leave-one-family-out: is g really a function of the footprint? (tests assumption A1)
        lofo = {}
        for held, fams_h in LOFO_GROUPS.items():
            te = [n for n in LADDER if any(n.startswith(f"L:{f}@") for f in fams_h)]
            tr = [n for n in LADDER if n not in te]
            f2 = fit_g(np.stack([Mall[n][idxc] for n in tr], 1), np.stack([dl[n][idxc] for n in tr], 1))
            yt = np.stack([dl[n][idxc] for n in te], 1)
            yh = np.stack([pred_g(f2, Mall[n][idxc]) for n in te], 1)
            ss = ((yt - yt.mean(1, keepdims=True)) ** 2).sum()
            lofo[held] = {"r2_within": float(1 - ((yt - yh) ** 2).sum() / max(ss, 1e-12)),
                          "rmse": float(np.sqrt(((yt - yh) ** 2).mean())),
                          "bias": float((yt - yh).mean())}
        mm["g_fit"] = {"lofo": lofo, "n_train_edits": len(LADDER), "n_img": int(len(idxc))}
        res["models"][m] = mm
    json.dump(res, open(OUT, "w"), indent=2, default=float)

    # ---------------------------------------------------------------------------------------------
    # 4) Print tables
    # ---------------------------------------------------------------------------------------------
    ALLC = MATCHED + [f"u:{p}" for p in UNMATCHED]
    line = "=" * 118
    print("\n" + line)
    print("A) FOOTPRINT & VALIDITY (design property, model-independent): target = contrast@2")
    t = "contrast@2"
    cf = design[t]["_concept"]["footprint"]
    print(f"{'control':10s}" + "".join(f"{COVLAB[k][:13]:>15s}" for k in COVS) +
          f"{'IoU':>7s}{'maxSMD':>8s}{'proxyΔ':>8s}{'ratio':>7s}{'overlap':>9s}{'VALID':>7s}")
    print(f"{'CONCEPT':10s}" + "".join(f"{cf[k]:>15.4f}" for k in COVS) +
          f"{1.0:>7.2f}{'-':>8s}{design[t]['_concept']['proxy_shift']*100:>7.0f}%"
          f"{'1.00':>7s}{'-':>9s}{'(ref)':>7s}")
    for c in ALLC:
        d = design[t][c]
        print(f"{c:10s}" + "".join(f"{d['footprint'][k]:>15.4f}" for k in COVS) +
              f"{d['support_iou']:>7.2f}{d['max_abs_smd']:>8.2f}{d['proxy_shift']*100:>7.0f}%"
              f"{d['proxy_ratio']:>7.2f}{d['positivity']*100:>8.0f}%{('YES' if d['valid'] else '-'):>7s}")
    print("-" * 118)
    print("B) ΔCaCE = CaCE_concept − CaCE_control  (logit; [] = 95% CI; * = CI excludes 0)")
    for t in TOPDOSE:
        print(f"  target {t}:")
        print(f"    {'control':10s}{'valid':>7s}" + "".join(f"{NAME[m]:>26s}" for m in ORDER))
        for c in ALLC:
            row = f"    {c:10s}{('Y' if design[t][c]['valid'] else '.'):>7s}"
            for m in ORDER:
                d = res["models"][m]["dcace"][t][c]
                row += f"{d['mean']:+.2f}[{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]{'*' if d['sig'] else ' '}".rjust(26)
            print(row)
        print(f"    {'-'*100}")
        for m in ORDER:
            s = res["models"][m]["spread"][t]
            mv = s["matched_valid"]
            print(f"    spread {NAME[m]:5s} unmatched SD={s['unmatched']['sd']:.2f} range={s['unmatched']['range']:.2f}"
                  f"  |  matched-VALID SD={('%.2f' % mv['sd']) if mv['sd'] is not None else 'n/a'}"
                  f" range={('%.2f' % mv['range']) if mv['range'] is not None else 'n/a'} (n={mv['n']})"
                  f"  sign-consistent={'YES' if s['sign_consistent_valid'] else 'no'}")
    print("-" * 118)
    print("C) FALSIFICATION: control-vs-control (two valid controls from different families must be ≈ 0):")
    for m in ORDER:
        for t in TOPDOSE:
            cc = res["models"][m]["ctrl_vs_ctrl"][t]
            worst = max(cc["pairs"].items(), key=lambda kv: abs(kv[1]["mean"]), default=(None, None))
            print(f"  {NAME[m]:5s} {t:12s} n_pairs={cc['n_pairs']:2d} all_null={'YES' if cc['all_null'] else 'NO'}"
                  + (f"  worst {worst[0]}={worst[1]['mean']:+.2f}[{worst[1]['ci'][0]:+.2f},{worst[1]['ci'][1]:+.2f}]"
                     if worst[0] else ""))
    print("-" * 118)
    print("D) ALIGNMENT A = mean ΔCaCE (contrast@2, color_var@2) per control:")
    print(f"  {'control':10s}{'valid':>7s}" + "".join(f"{NAME[m]:>14s}" for m in ORDER))
    for c in ALLC:
        row = f"  {c:10s}{('Y' if res['models']['erm']['alignment'][c]['valid'] else '.'):>7s}"
        for m in ORDER:
            al = res["models"][m]["alignment"][c]
            row += f"{al['A']:+.2f}[{'Y' if al['aligned'] else ('A' if al['anti'] else '.')}]".rjust(14)
        print(row)
    print("-" * 118)
    print("E) ΔCaCE adjusted by ĝ(footprint): neutral counterfactual control at the exact concept footprint")
    print("   (neutral dose ladder: %s × %s levels; A1 tested by leave-one-family-out)"
          % ("/".join(LADDER_FAM), len(LADDER_MULT)))
    for m in ORDER:
        g = res["models"][m]["g_fit"]["lofo"]
        print(f"  {NAME[m]:5s} LOFO R2(within-image): " +
              "  ".join(f"{k}={v['r2_within']:+.2f} (rmse {v['rmse']:.2f}, bias {v['bias']:+.2f})"
                        for k, v in g.items()))
    print(f"  {'target':12s}{'overlap':>9s}" + "".join(f"{NAME[m]:>24s}" for m in ORDER))
    for t in TARGETS:
        row = f"  {t:12s}{design[t]['_ladder_overlap']['all_covs']*100:>8.0f}%"
        for m in ORDER:
            d = res["models"][m]["g_adjusted"][t]
            row += f"{d['mean']:+.2f}[{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]{'*' if d['sig'] else ' '}".rjust(24)
        print(row)
    print("  A_adj (mean of the 2 high-dose concepts): " +
          "  ".join(f"{NAME[m]}={res['models'][m]['g_adjusted']['A']:+.2f}"
                    f"[{'aligned' if res['models'][m]['g_adjusted']['aligned'] else 'n.s.'}]" for m in ORDER))
    print("-" * 118)
    print("F) PRIMARY ESTIMATE: best-justified control (globrot_m) vs ĝ(M) adjustment:")
    print(f"  {'model':6s}{'target':13s}{'ΔCaCE(globrot_m)':>26s}{'ΔCaCE_adj':>26s}{'agree':>7s}")
    prim = {}
    for m in ORDER:
        prim[m] = {}
        for t in TOPDOSE:
            a1 = res["models"][m]["dcace"][t]["globrot_m"]; a2 = res["models"][m]["g_adjusted"][t]
            ag = bool(a1["ci"][0] <= a2["ci"][1] and a2["ci"][0] <= a1["ci"][1])
            prim[m][t] = {"control": a1, "adjusted": a2, "agree": ag}
            print(f"  {NAME[m]:6s}{t:13s}"
                  f"{a1['mean']:+.2f}[{a1['ci'][0]:+.2f},{a1['ci'][1]:+.2f}]".rjust(26)
                  + f"{a2['mean']:+.2f}[{a2['ci'][0]:+.2f},{a2['ci'][1]:+.2f}]".rjust(26)
                  + f"{('YES' if ag else 'no'):>7s}")
    print("-" * 118)
    print("G) ΔCaCE bounds over the whole control library (min..max): uncertainty due to control choice:")
    bounds = {}
    for m in ORDER:
        bounds[m] = {}
        for t in TARGETS:
            vs = [res["models"][m]["dcace"][t][c]["mean"] for c in ALLC]
            nb = [res["models"][m]["dcace"][t][c]["mean"] for c in ALLC
                  if design[t][c]["neutral"] and design[t][c]["overlap_ok"]]
            bounds[m][t] = {"all": [min(vs), max(vs)], "neutral_only": ([min(nb), max(nb)] if nb else None)}
            print(f"  {NAME[m]:6s}{t:13s} all controls [{min(vs):+.2f}, {max(vs):+.2f}]"
                  + (f"   neutral-only [{min(nb):+.2f}, {max(nb):+.2f}]" if nb else "   neutral-only: n/a"))
    res["bounds"] = bounds
    res["identification"] = {
        "any_valid_control": {t: [c for c in ALLC if design[t][c]["valid"]] for t in TARGETS},
        "any_valid_ok_control": {t: [c for c in ALLC if design[t][c]["valid_ok"]] for t in TARGETS},
        "neutral_controls": {t: [c for c in ALLC if design[t][c]["neutral"]] for t in TARGETS},
        "lofo_passes": {m: bool(all(v["r2_within"] > 0 for v in res["models"][m]["g_fit"]["lofo"].values()))
                        for m in ORDER},
        "ctrl_vs_ctrl_null": {m: {t: res["models"][m]["ctrl_vs_ctrl"][t]["all_null"] for t in TOPDOSE}
                              for m in ORDER},
    }
    res["ladder_mag"] = {n: float(np.mean(FP[n]["mean_abs"])) for n in LADDER}
    res["proxy_signed"] = {n: float(np.mean(v)) for n, v in PRXS.items()}
    for t in TARGETS:                                  # A2' : the reversed intervention must flip the proxy sign
        up, dn = res["proxy_signed"][t], res["proxy_signed"][f"{t}|invdose"]
        design[t]["invdose"]["antisymmetric"] = bool(up > 0 > dn)
        design[t]["invdose"]["valid_sym"] = bool(design[t]["invdose"]["balanced_ok"]
                                                 and design[t]["invdose"]["overlap_ok"]
                                                 and up > 0 > dn)
    res["primary"] = {"control": "globrot_m", "by_model": prim,
                      "validity": {t: design[t]["globrot_m"] for t in TOPDOSE}}
    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print("-" * 118)
    print("H) SYMMETRIC-DOSE DESIGN: the control is the same intervention in the opposite direction, not a placebo")
    print("   ΔCaCE_sym = CaCE(C↑) − CaCE(C↓): g(M) cancels because the two edits are mirror images in pixel space.")
    print(f"   {'target':13s}{'maxSMD':>8s}{'proxy ↑':>10s}{'proxy ↓':>10s}" + "".join(f"{NAME[m]:>24s}" for m in ORDER))
    sym = {}
    for t in TARGETS:
        g = design[t]["invdose"]
        row = (f"   {t:13s}{g['max_abs_smd']:>8.2f}{res['proxy_signed'][t]*100:>9.0f}%"
               f"{res['proxy_signed'][t+'|invdose']*100:>9.0f}%")
        for m in ORDER:
            e = res["models"][m]["dcace"][t]["invdose"]; sym.setdefault(m, {})[t] = e
            row += f"{e['mean']:+.2f}[{e['ci'][0]:+.2f},{e['ci'][1]:+.2f}]{'*' if e['sig'] else ' '}".rjust(24)
        print(row)
    print("   A_sym (mean of the 2 high-dose concepts): " + "  ".join(
        f"{NAME[m]}={np.mean([sym[m][t]['mean'] for t in TOPDOSE]):+.2f}"
        f"[{'aligned' if all(sym[m][t]['ci'][0] > 0 for t in TOPDOSE) else 'n.s.'}]" for m in ORDER))
    res["symmetric_dose"] = {"by_model": {m: {t: sym[m][t] for t in TARGETS} for m in ORDER},
                             "balance": {t: design[t]["invdose"] for t in TARGETS}}
    json.dump(res, open(OUT, "w"), indent=2, default=float)
    print(line)
    print("READING RULE: a control is causally valid only if it passes all three:")
    print("  (A1) balance: |SMD| <= 0.10 (0.25 acceptable) on all 6 footprint covariates;")
    print("  (A2) neutrality: concept proxy shift <= 10% absolute and <= 15% of the concept edit's own shift;")
    print("  (A3) overlap: >= 95% of images can be calibrated to the exact strength of the concept edit.")
    print("  Test (T): neutral controls with different mechanisms must give equal CaCE (section C), and ĝ(M)")
    print("  must generalise to an unseen family (LOFO R2 > 0, section E). If (T) fails, the generic response")
    print("  is not a function of the measured footprint -> no control identifies ΔCaCE, and the bounds in")
    print("  section G are the real uncertainty of every 'placebo-corrected' score.")
    idf = res["identification"]
    print(f"  => controls passing all 3 criteria: {idf['any_valid_ok_control']}")
    print(f"  => LOFO passes: {idf['lofo_passes']}   control-vs-control ≈ 0: {idf['ctrl_vs_ctrl_null']}")

    # ---------------------------------------------------------------------------------------------
    # 5) Figure
    # ---------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    ax = axes[0, 0]                                  # (a) love plot |SMD|
    yy = np.arange(len(ALLC))[::-1]
    mk = dict(zip(COVS, ["o", "s", "^", "D", "v", "P"]))
    cc = dict(zip(COVS, ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]))
    for k in COVS:
        ax.scatter([abs(design["contrast@2"][c]["smd"][k]) for c in ALLC], yy,
                   marker=mk[k], c=cc[k], s=34, label=COVLAB[k], zorder=3)
    ax.axvline(SMD_TOL, color="k", ls="--", lw=1); ax.axvline(SMD_TOL_OK, color="k", ls=":", lw=1)
    ax.set_yticks(yy); ax.set_yticklabels(ALLC, fontsize=8)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("|standardized mean difference| vs concept edit  (dashed 0.10, dotted 0.25)")
    ax.set_title("(a) Covariate balance of each control (target contrast@2)", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    ax.grid(alpha=.3, axis="x")

    ax = axes[0, 1]                                  # (b) ΔCaCE per control, top-dose contrast
    t = TOPDOSE[0]; xs = np.arange(len(ALLC))
    for oi, m in enumerate(ORDER):
        mu = [res["models"][m]["dcace"][t][c]["mean"] for c in ALLC]
        lo = [max(0.0, mu[i] - res["models"][m]["dcace"][t][c]["ci"][0]) for i in range(len(ALLC))]
        hi = [max(0.0, res["models"][m]["dcace"][t][c]["ci"][1] - mu[i]) for i in range(len(ALLC))]
        ax.errorbar(xs + (oi - 1) * 0.22, mu, yerr=[lo, hi], fmt="o", ms=4, lw=1.1,
                    color=COL[m], label=NAME[m])
    ax.axhline(0, color="k", lw=0.8)
    ax.axvspan(-0.5, len(MATCHED) - 0.5, color="#2ca02c", alpha=0.06)
    ax.set_xticks(xs); ax.set_xticklabels(ALLC, rotation=45, ha="right", fontsize=8)
    for i, c in enumerate(ALLC):
        if design[t][c]["valid_ok"]:
            ax.plot(i, ax.get_ylim()[1], marker="v", color="#2ca02c", ms=6, clip_on=False)
    ax.set_title(f"(b) dCaCE by control - {t}  (no control passes the screen)", fontsize=10)
    ax.set_ylabel("ΔCaCE (logit)"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")

    ax = axes[1, 0]                                  # (c) generic sensitivity curve g(magnitude)
    for m in ORDER:
        for fam, ls in zip(LADDER_FAM, ["-", "--", "-.", ":"]):
            nn = [f"L:{fam}@{j}" for j in range(len(LADDER_MULT))]
            ax.plot([res["ladder_mag"][n] for n in nn],
                    [res["models"][m]["cace"][n]["mean"] for n in nn],
                    ls, color=COL[m], lw=1.3, marker="." ,
                    label=f"{NAME[m]} · {fam.replace('_m','')}" if True else None)
        for t2, mkr in zip(TARGETS, ["o", "s", "^", "D"]):
            ax.scatter(design[t2]["_concept"]["footprint"]["mean_abs"], res["models"][m]["cace"][t2]["mean"],
                       marker=mkr, s=52, facecolor="none", edgecolor=COL[m], lw=1.6, zorder=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("perturbation magnitude (mean |Δ| pixel)"); ax.set_ylabel("CaCE (logit)")
    ax.set_title("(c) Generic edit-sensitivity g: neutral dose-ladder (lines) vs concept edits\n"
                 "(open markers: o contrast@2, s color@2, ^ contrast@0, D color@0)", fontsize=9)
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)

    ax = axes[1, 1]                                  # (d) symmetric-dose design (identified)
    xs = np.arange(len(TARGETS))
    for oi, m in enumerate(ORDER):
        mu = [res["models"][m]["dcace"][t2]["invdose"]["mean"] for t2 in TARGETS]
        lo = [max(0.0, mu[i] - res["models"][m]["dcace"][t2]["invdose"]["ci"][0]) for i, t2 in enumerate(TARGETS)]
        hi = [max(0.0, res["models"][m]["dcace"][t2]["invdose"]["ci"][1] - mu[i]) for i, t2 in enumerate(TARGETS)]
        ax.errorbar(xs + (oi - 1) * 0.18, mu, yerr=[lo, hi], fmt="o", ms=5, lw=1.3,
                    color=COL[m], label=NAME[m])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{t2}\n|SMD|={design[t2]['invdose']['max_abs_smd']:.2f}"
                        f"{' ✓' if design[t2]['invdose']['valid_sym'] else ''}" for t2 in TARGETS], fontsize=8)
    ax.set_ylabel("ΔCaCE_sym (logit)")
    ax.set_title("(d) Symmetric-dose design: ΔCaCE_sym = CaCE(C↑) − CaCE(C↓)\n"
                 "the only design that is footprint-balanced at the low dose (check mark) and has full overlap", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")

    fig.suptitle(f"Matched-control analysis: no neutral placebo balances all {len(COVS)} footprint "
                 f"covariates with full overlap; the reversed intervention balances them at the low dose only.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGDIR, "fig_matched_control.png"), dpi=140); plt.close(fig)
    print(f"Saved -> {OUT} + fig_matched_control.png   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
