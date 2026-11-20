#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed three-way partition, stored by isic_id.

Why. `StratifiedGroupKFold` gives different folds across sklearn versions even with the same seed.
On this machine, rebuilding the partition gives train 553 patients / 232 positive and audit 199 / 83,
while the original training run stored 574 / 237 and 202 / 78. The consequence is worse than a
mismatch in counts: the checkpoints were trained under the original partition, whose fold membership
was not stored anywhere, so there is no way to verify that a rebuilt audit cohort is truly
patient-disjoint from the training set. The central claim of the audit depends on exactly that
disjointness.

Fix. Build the partition once, write the three isic_id lists to `results/frozen_split.json`, and
read everything from that file afterwards. Training must be rerun under this partition so that the
checkpoints can be matched to the audit cohort in a verifiable way.

Usage:
    import frozen_split
    df_tr, df_se, df_au = frozen_split.load()

Rebuild (only when you deliberately change the partition; training must then be rerun):
    python code/frozen_split.py --rebuild
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c3_cace_concept as c3m       # noqa: E402

p1 = c3m.p1
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
MANIFEST = os.path.join(RES, "frozen_split.json")
SEED, N_FOLDS = 42, 5


def _build():
    """Original logic: fold0 = audit, fold1 = select, the rest = train, then undersample."""
    df = p1.load_isic_df()
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = [va for _, va in sgkf.split(df, df["target"].values, df["patient_id"].values)]
    audit_i, select_i = folds[0], folds[1]
    train_i = np.setdiff1d(np.arange(len(df)), np.concatenate([audit_i, select_i]))
    df_tr = p1.undersample(df.iloc[train_i], p1.CFG["train_neg_ratio"], SEED * 100)
    df_se = p1.undersample(df.iloc[select_i], p1.CFG["val_neg_ratio"], SEED * 100 + 1)
    df_au = p1.undersample(df.iloc[audit_i], p1.CFG["val_neg_ratio"], SEED * 100 + 2)
    return df_tr, df_se, df_au


def _check(df_tr, df_se, df_au):
    S = [set(d["patient_id"].astype(str)) for d in (df_tr, df_se, df_au)]
    assert not (S[0] & S[1]), "patient leakage train/select"
    assert not (S[0] & S[2]), "patient leakage train/audit"
    assert not (S[1] & S[2]), "patient leakage select/audit"


def load(rebuild=False):
    """Return (df_tr, df_se, df_au) built from the manifest. The patient-disjointness invariant is
    re-asserted on every load, so a broken manifest fails loudly rather than silently."""
    if rebuild or not os.path.exists(MANIFEST):
        df_tr, df_se, df_au = _build()
        _check(df_tr, df_se, df_au)
        os.makedirs(RES, exist_ok=True)
        json.dump({"seed": SEED, "n_folds": N_FOLDS,
                   "note": ("Frozen three-way patient-disjoint split, stored by isic_id. Fold "
                            "assignment from StratifiedGroupKFold is not stable across library "
                            "versions, so it is built once and read from this file thereafter. "
                            "Any checkpoint audited against it must have been trained under it."),
                   "counts": {k: {"n": int(len(d)), "n_patients": int(d["patient_id"].nunique()),
                                  "n_pos": int(d["target"].sum())}
                              for k, d in [("train", df_tr), ("select", df_se), ("audit", df_au)]},
                   "train": list(df_tr["isic_id"]), "select": list(df_se["isic_id"]),
                   "audit": list(df_au["isic_id"])},
                  open(MANIFEST, "w"), indent=1)
        print(f"[frozen_split] building and writing {MANIFEST}", flush=True)

    m = json.load(open(MANIFEST))
    df = p1.load_isic_df()
    by_id = df.set_index("isic_id")
    out = []
    for part in ("train", "select", "audit"):
        ids = m[part]
        missing = [i for i in ids if i not in by_id.index]
        assert not missing, f"{len(missing)} isic_id in the manifest are missing from the metadata"
        out.append(by_id.loc[ids].reset_index())
    _check(*out)
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    df_tr, df_se, df_au = load(rebuild=a.rebuild)
    for nm, d in [("train", df_tr), ("select", df_se), ("audit", df_au)]:
        print("  %-7s n=%5d  patients=%4d  pos=%3d"
              % (nm, len(d), d["patient_id"].nunique(), int(d["target"].sum())))
    print("patient-disjointness invariant: OK")
    print("manifest:", MANIFEST)


if __name__ == "__main__":
    main()
