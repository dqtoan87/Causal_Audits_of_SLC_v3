#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C20: recompute from the raw predictions two numerical-reproducibility numbers that the paper cites
but that no result file stores:

  (a) fp32 recomputation: rerun the same images and edits with bf16 autocast turned off;
  (b) batch composition: rerun with a different batch size and composition.

The compared quantity is the held-out artifact aggregate E|Δℓ_eps| (pooling the 4 edits
hair_light/hair/hair_heavy/ruler, taking the absolute value per image first and then averaging), on
the same set that Table II uses: malignant images with p_full >= 0.1 for each model.

Out -> results/c20_numerical_repro.json
Run:  .venv/bin/python Causal_XAI_v3/code/c20_numerical_repro.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c3_cace_concept as c3m       # noqa: E402
import c13_fullpop as c13           # noqa: E402

p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c20_numerical_repro.json")
import audited_models                # noqa: E402  (checkpoints resolved in one place)
import audit_cohort                  # noqa: E402  (frozen audit cohort)
CKPTS = audited_models.CKPTS         # absolute paths to the fixed set
SEED, N_EVAL = 42, 200
ORDER = ["erm", "slim", "causal"]
NAME = {"erm": "ERM", "slim": "Slim", "causal": "Inv"}
HELDOUT = ["art:hair_light", "art:hair", "art:hair_heavy", "art:ruler"]


@torch.no_grad()
def probs(model, batch, use_bf16):
    """model_prob from c3, but with autocast optionally turned off (fp32)."""
    xn = p2._to_norm(batch).to(DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(use_bf16 and DEVICE == "cuda")):
        return torch.softmax(model(xn), 1)[:, 1].float().cpu().numpy()


def run(model, imgs01, edits, names, use_bf16, mode):
    """mode='per_image' reproduces c13 (1 batch = baseline + every edit of one image);
       mode='chunk32'  pools all (image, edit) pairs and splits them into fixed batches of 32, so batch composition is entirely different."""
    n, k = len(imgs01), len(names)
    p0 = np.zeros(n); pe = np.zeros((n, k))
    if mode == "per_image":
        for i, x01 in enumerate(imgs01):
            batch = torch.stack([x01] + [edits[i][nm] for nm in names])
            ps = probs(model, batch, use_bf16)
            p0[i] = ps[0]; pe[i] = ps[1:]
    else:
        flat, idx = [], []
        for i, x01 in enumerate(imgs01):
            flat.append(x01); idx.append((i, -1))
            for j, nm in enumerate(names):
                flat.append(edits[i][nm]); idx.append((i, j))
        for s in range(0, len(flat), 32):
            ps = probs(model, torch.stack(flat[s:s + 32]), use_bf16)
            for t, v in enumerate(ps):
                i, j = idx[s + t]
                if j < 0: p0[i] = v
                else: pe[i, j] = v
    return p0, pe


def aggregate(p0, pe, names, keep):
    """E|Δℓ| pooled over the 4 held-out edits: |.| per image first, then the mean (as in Section III-E)."""
    cols = [names.index(h) for h in HELDOUT]
    L0 = c3m.logit(p0[keep])
    per = [np.abs(c3m.logit(pe[keep, c]) - L0) for c in cols]
    return float(np.mean(per))


def main():
    t0 = time.time()
    print(f"##### C20 numerical reproducibility {time.strftime('%H:%M:%S')} #####", flush=True)
    imgs01, labels, _pid, _ids = audit_cohort.load()   # frozen audit cohort, never rebuilt
    labels = np.array(labels)
    edits, names = c13.build_all_edits(imgs01)
    print(f"eval {len(imgs01)} images, {len(names)} edits/image  ({time.time()-t0:.0f}s)", flush=True)

    res = {"n_images": int(len(imgs01)), "heldout_edits": HELDOUT,
           "note": ("Held-out artifact aggregate E|dlogit| on each model's thresholded malignant set "
                    "(p_full >= 0.1), under three numerical configurations: the reference bf16 "
                    "per-image batching used throughout, the same in fp32, and bf16 with a different "
                    "batch composition (flat chunks of 32)."),
           "configs": {}}
    configs = [("bf16_per_image", True, "per_image"),
               ("fp32_per_image", False, "per_image"),
               ("bf16_chunk32", True, "chunk32")]
    agg = {c[0]: {} for c in configs}
    keeps = {}

    for m in ORDER:
        model = p1.build_model("erm" if m == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(CKPTS[m], map_location=DEVICE))
        model.eval()
        for cname, bf16, mode in configs:
            p0, pe = run(model, imgs01, edits, names, bf16, mode)
            if cname == "bf16_per_image":                       # held-out set fixed by the reference configuration
                keeps[m] = (labels == 1) & (p0 >= 0.1)
            agg[cname][m] = aggregate(p0, pe, names, keeps[m])
            print(f"  [{NAME[m]:5s}] {cname:16s} n={int(keeps[m].sum()):3d}  "
                  f"E|dlogit| = {agg[cname][m]:.3f}   ({time.time()-t0:.0f}s)", flush=True)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    for cname, _, _ in configs:
        res["configs"][cname] = {m: agg[cname][m] for m in ORDER}
    res["n_kept"] = {m: int(keeps[m].sum()) for m in ORDER}

    print("=" * 92)
    print("Held-out artifact aggregate E|dlogit|. Paper cites: bf16 9.36 / 15.39 / 10.92")
    print(f"{'config':20s}{'ERM':>12s}{'Slim':>12s}{'Inv':>12s}")
    for cname, _, _ in configs:
        print(f"{cname:20s}" + "".join(f"{agg[cname][m]:12.2f}" for m in ORDER))
    print("-" * 92)
    print("change relative to bf16_per_image (paper: fp32 →9.48/15.05/10.87; batch →9.43/15.19/11.00)")
    for cname, _, _ in configs[1:]:
        d = {m: 100 * (agg[cname][m] - agg["bf16_per_image"][m]) / agg["bf16_per_image"][m]
             for m in ORDER}
        res.setdefault("pct_change_vs_reference", {})[cname] = d
        print(f"{cname:20s}" + "".join(f"{d[m]:+11.2f}%" for m in ORDER))
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("=" * 92)
    print(f"saved -> {OUT}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
