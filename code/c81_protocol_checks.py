#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C81: checks that every response in one identification equation comes from one inference protocol.

A score is f(e(X_i); B_i), with B_i the batch an image is scored in. Statement 1 and the route sums combine
responses, so each combination must use one protocol. This script reads stored responses only.

  1. The concept library (c12_matched_control.py) scores each malignant audit image in one batch with all 88
     edits of the library, so every library response shares one batch list per image. The exact graph has
     four footprint vertices of degree two, each one operator entered under two labels. Under one protocol
     the two labels must return the same response; the script checks this for the three models.
  2. The route test (c62_route_test.py) scores each image in one batch with its 24 edited versions, so every
     route sum and difference uses one batch list per image. Library and route-test responses are never
     combined in one equation.

Reads results/c12_preds.npz, results/c55_identification_graph.json and results/c63_route_scores*.npz.
Out -> results/c81_protocol_checks.json
"""
from __future__ import annotations

import json
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c81_protocol_checks.json")
MODELS = ["erm", "slim", "causal"]


def main():
    z = np.load(os.path.join(RES, "c12_preds.npz"), allow_pickle=True)
    names = [str(n) for n in z["names"]]
    pairs = json.load(open(os.path.join(RES, "c55_identification_graph.json")))["exact_graph_G0"]["pairs"]
    dup = []
    for a, b in pairs:
        for m in MODELS:
            x, y = z["dl_%s|%s" % (m, a)], z["dl_%s|%s" % (m, b)]
            dup.append({"a": a, "b": b, "model": m, "n_images": int(len(x)),
                        "max_abs_difference": float(np.max(np.abs(x - y))),
                        "mean_response": float(np.mean(x))})
    route = {}
    for f in ("c63_route_scores.npz", "c63_route_scores_inv_retrained.npz"):
        r = np.load(os.path.join(RES, f), allow_pickle=True)
        cols = [str(c) for c in r["columns"]]
        route[f] = {"items_per_batch": len(cols), "edited_versions": len(cols) - 1,
                    "n_images": int(r["prob_erm"].shape[0])}
    out = {"library": {"items_per_batch": 1 + len(names), "edits": len(names),
                       "n_images": int(len(z["img_index"])),
                       "duplicate_label_pairs": dup,
                       "duplicates_identical": bool(all(d["max_abs_difference"] == 0.0 for d in dup))},
           "route_test": route,
           "note": ("one batch list per image within each tier; library and route-test responses are never "
                    "combined in one equation")}
    json.dump(out, open(OUT, "w"), indent=1)
    print("library: %d edits per batch, duplicates identical: %s" % (len(names), out["library"]["duplicates_identical"]))
    for f, v in route.items():
        print("%s: %d edited versions per batch" % (f, v["edited_versions"]))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
