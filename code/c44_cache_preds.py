#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C44: generate the prediction cache for the fixed evaluation set.

Why. `results/c19_preds.npz` is read by four analyses (c26, c32, c38 and the shrinkage-figure script).
A cache file without a generating script would make the paper non-reproducible at that point. This
script generates it from the same fixed image set and the same edit set that c13 uses, so every
analysis that reads it is consistent.

File contents: labels, names (18 edits), and p0_*, pe_* for the three EfficientNet-B3 models.

Out -> results/c19_preds.npz, and results/c19_preds_manifest.json with the SHA-256 of every checkpoint
       and of the audit-cohort manifest the cache was built from
Run:  python code/c44_cache_preds.py   (code/run_all.sh calls it first)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c13_fullpop as c13          # noqa: E402  (build_all_edits, eval_model, CKPTS, ORDER)
import audit_cohort                # noqa: E402

p1, DEVICE = c13.p1, c13.DEVICE
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c19_preds.npz")
MANIFEST = os.path.join(RES, "c19_preds_manifest.json")


def main():
    t0 = time.time()
    print(f"##### C44 prediction cache {time.strftime('%H:%M:%S')} #####", flush=True)
    imgs01, labels, pid, ids = audit_cohort.load()
    labels = np.array(labels)
    print(f"  {len(imgs01)} images / {len(set(pid))} patients / {int(labels.sum())} malignant", flush=True)

    edits, names = c13.build_all_edits(imgs01)
    print(f"  {len(names)} edits per image ({time.time()-t0:.0f}s)", flush=True)

    out = {"labels": labels, "names": np.array(names, dtype=object),
           "patient_id": np.array(pid, dtype=object), "isic_id": np.array(ids, dtype=object)}
    for m in c13.ORDER:
        model = p1.build_model("erm" if m == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(c13.CKPTS[m], map_location=DEVICE))
        model.eval()
        p0, pe = c13.eval_model(model, imgs01, edits, names)
        out[f"p0_{m}"] = p0
        out[f"pe_{m}"] = pe
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"  [{m}] mean p0 {p0.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(OUT, **out)
    # record what the cache was built from, so a cache built from other weights can be detected
    sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
    man = {"cache": os.path.basename(OUT), "cache_sha256": sha(OUT),
           "checkpoints": {m: {"file": os.path.basename(c13.CKPTS[m]), "sha256": sha(c13.CKPTS[m])}
                           for m in c13.ORDER},
           "audit_cohort_sha256": sha(os.path.join(RES, "audit_cohort.json")),
           "n_images": int(len(imgs01)), "n_edits": int(len(names))}
    json.dump(man, open(MANIFEST, "w"), indent=1)
    print(f"saved -> {OUT} and {MANIFEST}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
