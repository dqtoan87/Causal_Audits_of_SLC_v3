#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C68: which audit images pass the score floor, by image identifier  (no new analysis)

Diagnostic (T), the shrinkage fits and the placebo comparison keep the malignant audit images that all
three EfficientNet-B3 models score at 0.1 or above. Each script scores an unedited image in one bf16
batch together with its own edits, so an image near 0.1 can fall on either side of the floor,
depending on the batch. This script records the floor set by image identifier under four ways of
scoring the unedited images:

  c10_batch    the batch c10_multiplacebo.py uses (the image, its six concept edits, four placebos);
  c19_cache    the cached predictions of results/c19_preds.npz, which c12 and c19 read;
  single_bf16  each image alone, in bf16;
  single_fp32  each image alone, in full precision.

It lists the images whose membership changes between them, with their scores.

Out -> results/c68_floor_sets.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import audit_cohort                  # noqa: E402
import audited_models                # noqa: E402
import c3_cace_concept as c3m        # noqa: E402
import c10_multiplacebo as c10       # noqa: E402

p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c68_floor_sets.json")
FLOOR = 0.1
MODELS = ["erm", "slim", "causal"]


@torch.no_grad()
def prob(model, batch, fp32=False):
    xn = p2._to_norm(batch).to(DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda" and not fp32)):
        return torch.softmax(model(xn), 1)[:, 1].float().cpu().numpy()


def main():
    t0 = time.time()
    imgs01, labels, pid, ids = audit_cohort.load()
    labels = np.asarray(labels)
    mal = [i for i in range(len(labels)) if labels[i] == 1]
    # the batches of c10_multiplacebo.py, rebuilt exactly
    batches = []
    for ii in mal:
        x = imgs01[ii]
        w, skin, lm = c3m.soft_mask(x)
        items = [x]
        for c in ("contrast", "color_var"):
            for k in range(3):
                items.append(c3m.make_edit(x, c, c10.DOSES[k], w, skin, lm))
        for p, fn in c10.PLACEBOS.items():
            items.append(fn(x, w, skin, np.random.RandomState(c10.SEED + int(ii) + 7)))
        batches.append(torch.stack(items))
    cache = np.load(os.path.join(RES, "c19_preds.npz"), allow_pickle=True)
    assert list(cache["isic_id"]) == list(ids)
    P = {v: {} for v in ("c10_batch", "c19_cache", "single_bf16", "single_fp32")}
    for m in MODELS:
        model = p1.build_model("erm" if m == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(audited_models.CKPTS[m], map_location=DEVICE)); model.eval()
        P["c10_batch"][m] = np.array([prob(model, b)[0] for b in batches])
        P["single_bf16"][m] = np.array([prob(model, imgs01[ii].unsqueeze(0))[0] for ii in mal])
        P["single_fp32"][m] = np.array([prob(model, imgs01[ii].unsqueeze(0), True)[0] for ii in mal])
        P["c19_cache"][m] = np.asarray(cache["p0_" + m])[mal]
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("[%s] scored (%.0fs)" % (m, time.time() - t0), flush=True)
    mal_ids = [ids[i] for i in mal]
    sets, out = {}, {"floor": FLOOR, "n_malignant": len(mal), "variants": {}}
    for v, pv in P.items():
        keep = np.all([pv[m] >= FLOOR for m in MODELS], 0)
        sets[v] = keep
        out["variants"][v] = {"n_kept": int(keep.sum()),
                              "n_patients": int(len(set(np.asarray(pid)[mal][keep]))),
                              "isic_ids": [mal_ids[j] for j in np.where(keep)[0]]}
        print("%-12s keeps %d images" % (v, keep.sum()), flush=True)
    diff = np.zeros(len(mal), bool)
    for v in sets:
        diff |= sets[v] != sets["c19_cache"]
    out["membership_differs"] = [{"isic_id": mal_ids[j],
                                  "scores": {v: {m: float(P[v][m][j]) for m in MODELS} for v in P}}
                                 for j in np.where(diff)[0]]
    for r in out["membership_differs"]:
        print(r["isic_id"], {v: round(s["causal"], 4) for v, s in r["scores"].items()}, flush=True)
    out["c10_reproduces_n_common"] = int(sets["c10_batch"].sum()) == json.load(
        open(os.path.join(RES, "c10_multiplacebo.json")))["n_common"]
    out["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved -> %s (%.0fs)" % (OUT, out["runtime_sec"]))


if __name__ == "__main__":
    main()
