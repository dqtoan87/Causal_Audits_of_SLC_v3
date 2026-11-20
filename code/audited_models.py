#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The three audited EfficientNet-B3 models, resolved in one single place.

Why. Ten scripts used to declare checkpoint paths on their own. They pointed to a training set under
a partition that can no longer be rebuilt, so one could not verify that the evaluation cohort is
patient-disjoint from the training set. For the artifact tier this is fatal, because the claim there
is a claim about generalisation. For the concept tier it is milder, since no claim there is about
generalisation, but a second-order effect remains: if an evaluation image was in the training set,
its baseline logit is inflated by memorisation, and the whole shrinkage regression uses the baseline
logit as the independent variable.

All analyses now use the models trained under the fixed partition (results/frozen_split.json), so
patient disjointness is verifiable for both tiers.

Override with the environment variable CAUSALXAI_CKPT_SET=legacy to compare against the legacy set.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

_FROZEN = {"erm": os.path.join(BASE, "checkpoints", "dj_erm_b3.pt"),
           "slim": os.path.join(BASE, "checkpoints", "dj_slim_b3.pt"),
           "causal": os.path.join(BASE, "checkpoints", "dj_causal_b3.pt")}

_LEGACY = {"erm": os.path.join(HERE, "lib", "checkpoints", "p2_erm_s42.pt"),
           "slim": os.path.join(HERE, "lib", "checkpoints", "p2_slim_s42.pt"),
           "causal": os.path.join(HERE, "lib", "checkpoints", "p2_causal_s42.pt")}

CKPTS = _LEGACY if os.environ.get("CAUSALXAI_CKPT_SET") == "legacy" else _FROZEN
SET_NAME = "legacy" if CKPTS is _LEGACY else "frozen"


def check():
    missing = [k for k, v in CKPTS.items() if not os.path.exists(v)]
    assert not missing, f"missing checkpoint: {missing}"
    return CKPTS


if __name__ == "__main__":
    print(f"checkpoint set: {SET_NAME}")
    for k, v in check().items():
        print(f"  {k:7s} {v}  ({os.path.getsize(v)/1e6:.0f} MB)")
