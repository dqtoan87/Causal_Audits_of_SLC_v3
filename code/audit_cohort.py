#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed list of audit images.

Why. Each script used to rebuild the partition with StratifiedGroupKFold and then sample images.
That depends on library versions: the same seed with a different sklearn or pandas version gives
different folds. On this machine the rebuild gives 183 images / 100 patients / 83 malignant cases,
while the original run stored 178 / 90 / 78. The conclusions do not change, but the counts differ
by up to 13%, which is not acceptable for a paper.

Fix. Build once, write the `isic_id` list in exact order, with labels and patient ids, to
`results/audit_cohort.json`, and from then on every script loads directly from that file. Images
are read by id, so nothing depends on any sampling step.

Usage:
    import audit_cohort
    imgs, labels, pid, ids = audit_cohort.load()

Rebuild from scratch (only when you deliberately change the cohort):
    python code/audit_cohort.py --rebuild
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torchvision.transforms as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c3_cace_concept as c3m       # noqa: E402
import frozen_split                 # noqa: E402

p1, p2 = c3m.p1, c3m.p2
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
MANIFEST = os.path.join(RES, "audit_cohort.json")

SEED, N_EVAL = 42, 200


def _read(ids):
    """Read images by id, in order, with the same transform the rest of the pipeline uses."""
    to_t = T.ToTensor()
    size = (p1.CFG["img_size"], p1.CFG["img_size"])
    out = []
    for i in ids:
        img = Image.open(os.path.join(p2.ISIC_IMAGE_DIR, f"{i}.jpg")).convert("RGB")
        out.append(to_t(p1.center_crop_resize(img, size)))
    return out


def build(seed=SEED, n_eval=N_EVAL):
    """Build the cohort from the three-way partition. Replicates the logic of p2.load_eval_images to
    get the true id order, then returns everything the manifest needs."""
    _, _, df_au = frozen_split.load()
    df_au = df_au.reset_index(drop=True)
    k = n_eval // 2
    pos = df_au[df_au["target"] == 1].sample(n=min(k, int((df_au["target"] == 1).sum())),
                                             random_state=seed)
    neg = df_au[df_au["target"] == 0].sample(n=min(k, int((df_au["target"] == 0).sum())),
                                             random_state=seed)
    sel = np.r_[pos.index.values, neg.index.values]
    sub = df_au.loc[sel].reset_index(drop=True)
    return (list(sub["isic_id"]), [int(t) for t in sub["target"]],
            list(sub["patient_id"].astype(str)))


def load(rebuild=False):
    """Return (imgs, labels, pid, ids). Read the manifest if it exists; otherwise build and write it."""
    if rebuild or not os.path.exists(MANIFEST):
        ids, labels, pid = build()
        os.makedirs(RES, exist_ok=True)
        json.dump({"n_images": len(ids), "n_patients": len(set(pid)),
                   "n_malignant": int(sum(labels)), "seed": SEED, "n_eval": N_EVAL,
                   "note": ("Frozen audit cohort. Built once from the three-way patient-disjoint "
                            "split and stored by isic_id so that no later analysis depends on "
                            "reconstructing the fold assignment, which is not stable across "
                            "library versions."),
                   "isic_id": ids, "target": labels, "patient_id": pid},
                  open(MANIFEST, "w"), indent=1)
        print(f"[audit_cohort] building and writing {MANIFEST}", flush=True)
    m = json.load(open(MANIFEST))
    ids = m["isic_id"]; labels = np.array(m["target"]); pid = np.array(m["patient_id"])
    assert len(ids) == len(labels) == len(pid)
    return _read(ids), labels, pid, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    imgs, labels, pid, ids = load(rebuild=a.rebuild)
    print(f"cohort: {len(ids)} images / {len(set(pid))} patients / {int(labels.sum())} malignant")
    print(f"manifest: {MANIFEST}")
    print("first 5 ids:", ids[:5])


if __name__ == "__main__":
    main()
