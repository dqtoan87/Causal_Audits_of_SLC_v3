#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C84: label coding and label provenance of the image sets the audit reads. Reads no model output.

The analyses use the dataset's binary `target`. ISIC-2024 also records a first-level diagnosis
`iddx_1` (Benign, Malignant, Indeterminate) and a full diagnosis `iddx_full`; a full diagnosis other
than the bare "Benign" is a histopathological diagnosis. This script counts, for the whole cohort, the
three stored split frames, the 183-image audit cohort and the 78 malignant images of the route test:
  - how the 114 indeterminate lesions are coded in `target`, and how many each set holds;
  - how many images carry a histopathological diagnosis, by class.

Reads the ISIC-2024 training metadata, results/frozen_split.json and results/audit_cohort.json.
Out -> results/c84_label_provenance.json
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c84_label_provenance.json")
sys.path.insert(0, os.path.join(HERE, "lib"))
from train_XAI_new_v01 import ISIC_META_CSV  # noqa: E402


def summary(d):
    histo = d["iddx_full"] != "Benign"
    return {"n_images": int(len(d)), "n_patients": int(d["patient_id"].nunique()),
            "n_target_1": int(d["target"].sum()),
            "iddx_1_counts": {k: int(v) for k, v in d["iddx_1"].value_counts().items()},
            "n_indeterminate": int((d["iddx_1"] == "Indeterminate").sum()),
            "n_histopathology": int(histo.sum()),
            "n_histopathology_malignant": int((histo & (d["target"] == 1)).sum()),
            "n_histopathology_not_malignant": int((histo & (d["target"] == 0)).sum())}


def main():
    m = pd.read_csv(ISIC_META_CSV, low_memory=False)
    m["patient_id"] = m["patient_id"].fillna("unknown")
    coding = pd.crosstab(m["iddx_1"], m["target"])
    out = {"note": ("Binary target as distributed. A histopathological diagnosis is an iddx_full other "
                    "than the bare 'Benign'. Reads no model output."),
           "target_by_iddx_1": {r: {str(c): int(coding.loc[r, c]) for c in coding.columns} for r in coding.index},
           "indeterminate_coded_as_target_0": bool(((m["iddx_1"] == "Indeterminate") <= (m["target"] == 0)).all()),
           "cohort": summary(m)}
    mi = m.set_index("isic_id")
    split = json.load(open(os.path.join(RES, "frozen_split.json")))
    for nm in ("train", "select", "audit"):
        out["frame_" + nm] = summary(mi.loc[split[nm]])
    aud = json.load(open(os.path.join(RES, "audit_cohort.json")))
    a = mi.loc[aud["isic_id"]]
    out["audit_183"] = summary(a)
    out["audit_183"]["subset_of_audit_frame"] = bool(set(aud["isic_id"]) <= set(split["audit"]))
    for cls, lab in ((1, "malignant"), (0, "benign")):
        out["audit_183_" + lab] = summary(a[a["target"] == cls])
    sel = mi.loc[split["select"]]
    out["route_test_78"] = summary(sel[sel["target"] == 1])
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("target_by_iddx_1", "indeterminate_coded_as_target_0")}, indent=1))
    for k in ("cohort", "frame_train", "frame_select", "frame_audit", "audit_183", "audit_183_malignant",
              "audit_183_benign", "route_test_78"):
        v = out[k]
        print("%-22s n=%6d pat=%5d pos=%4d indet=%3d histo=%5d (mal %4d, other %4d)" % (
            k, v["n_images"], v["n_patients"], v["n_target_1"], v["n_indeterminate"], v["n_histopathology"],
            v["n_histopathology_malignant"], v["n_histopathology_not_malignant"]))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
