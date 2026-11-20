#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Record the SHA-256 of every analysis added after the first run of the route test, and of its result files.

The pre-specified route test has its own manifest, results/c62_manifest.sha256 (c62 to c65 and their results).
The scripts listed here were written later, some after the results were seen, so they are exploratory. The
manifest lets a reader check that the published scripts and results are the ones the manuscript describes.
It does not make any of these analyses pre-specified.

Out -> results/posthoc_manifest.sha256   (format of sha256sum: "<hash>  <path relative to the tree>")
Run this after the last change to any listed script or result.
"""
from __future__ import annotations

import glob
import hashlib
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "posthoc_manifest.sha256")
SCRIPTS = ["c63_route_exploratory", "c64_route_screens", "c65_route_robustness", "c66_overlay_occlusion",
           "c67_route_sensitivity", "c68_floor_sets", "c69_route_clip_sensitivity", "c70_occlusion_mask_sensitivity",
           "c71_rule_checks", "c72_library_proxy_split", "c73_route_robustness_table", "c74_route_exact_match",
           "c75_seed_band", "c76_route_exact_detail", "c77_route_feature_shift",
           "c78_batch_context", "c79_route_near_natural", "c80_route_null_calibration",
           "c81_protocol_checks", "c82_image_set_checks", "c83_batch_extremes",
           "c84_label_provenance", "c85_artifact_band_calibration",
           "c86_batch_context_first_run",
           "c87_route_feature_strata", "c88_artifact_inv_epoch1"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []
    for s in SCRIPTS:
        rows.append(os.path.join("code", s + ".py"))
        tag = s.split("_")[0] + "_"
        rows += sorted(os.path.relpath(p, BASE) for p in glob.glob(os.path.join(BASE, "results", tag + "*"))
                       if os.path.isfile(p))
    with open(OUT, "w") as f:
        for r in rows:
            f.write("%s  %s\n" % (sha(os.path.join(BASE, r)), r))
    print("%d files -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
