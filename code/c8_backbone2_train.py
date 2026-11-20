#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C8: train the ERM/Slim/Inv trio on a second backbone (ResNet50), to test generalisation across
architectures. Same split (seed 42) and same loss/regularizer as the original backbone; only the
backbone changes. Inv = Slim + V-REx(site attribution) + artifact-CF (reuses phase3.train_causal).
Undersample 20:1 => train ~6-7k images => fast. Out -> Causal_XAI_v3/checkpoints/{erm,slim,causal}_r50.pt.

Run (in background, low num_workers to avoid OOM on a shared machine):
  setsid nohup .venv/bin/python Causal_XAI_v3/code/c8_backbone2_train.py > Causal_XAI_v3/logs/c8.log 2>&1 &
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.utils.data import DataLoader

# The upstream training modules are vendored under code/lib together with the three
# primary EfficientNet-B3 checkpoints, so this tree is self-contained apart from the raw
# image data. Override with the CAUSALXAI_LIB environment variable if they are moved.
XAI_OLD = os.environ.get(
    "CAUSALXAI_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
sys.path.insert(0, XAI_OLD)
import train_XAI_new_v01 as p1
import phase2_causal_xai_v01 as p2
import phase3_invariance_v01 as p3

DEVICE = p1.DEVICE
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
SEED = 42
EPOCHS = 12
NUM_WORKERS = 4          # low -> avoids OOM on a shared machine
META = "/mnt/data2/Toan/Causal_ML/data/data_ISIC2024/train-metadata.csv"


# ---- ResNet50 backbone (feat 2048), head structure kept identical to the EfficientNet trio ----
def _bb_r50():
    bb = tvm.resnet50(weights="IMAGENET1K_V2")
    bb.fc = nn.Identity()
    return bb   # output 2048


class ERM_R50(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _bb_r50()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 2))
    def forward(self, x): return self.head(self.backbone(x))


class Slim_R50(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        self.backbone = _bb_r50()
        self.proj_c = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(),
                                    nn.Dropout(0.3), nn.Linear(512, z_dim))
        self.disease_head = nn.Sequential(nn.Linear(z_dim, 256), nn.ReLU(),
                                          nn.Dropout(0.4), nn.Linear(256, 2))
    def forward(self, x): return self.disease_head(self.proj_c(self.backbone(x)))


def class_weight(df_tr):
    n_pos = int(df_tr["target"].sum()); n_neg = len(df_tr) - n_pos
    return torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32)


def main():
    t0 = time.time()
    print(f"##### C8 backbone-2 (ResNet50) {time.strftime('%H:%M:%S')} dev={DEVICE} #####", flush=True)
    df_tr, df_va = p2.build_split({"seed": SEED, "n_folds": 5})
    # env (site attribution) cho Inv
    att = pd.read_csv(META, usecols=["isic_id", "attribution"]).fillna({"attribution": "unknown"})
    envs = sorted(att["attribution"].unique()); emap = {e: i for i, e in enumerate(envs)}
    amap = dict(zip(att["isic_id"], att["attribution"]))
    for d in (df_tr, df_va):
        d["attribution"] = d["isic_id"].map(amap).fillna("unknown")
        d["env"] = d["attribution"].map(emap)
    w = class_weight(df_tr)
    print(f"train {len(df_tr)} (pos {int(df_tr['target'].sum())}) | val {len(df_va)} | envs {len(envs)}", flush=True)

    cfg = dict(p1.CFG); cfg["epochs"] = EPOCHS
    IMG = p1.CFG["img_size"]; IMDIR = p1.ISIC_IMAGE_DIR

    # ----- ERM -----
    f = os.path.join(CKPT_DIR, "erm_r50.pt")
    if not os.path.exists(f):
        p1.set_seed(SEED)
        tr = DataLoader(p1.ISICImgDataset(df_tr, IMDIR, IMG, train=True), cfg["batch_size"],
                        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        va = DataLoader(p1.ISICImgDataset(df_va, IMDIR, IMG, train=False), cfg["batch_size"],
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        m, _ = p1.train_one(ERM_R50(), tr, va, cfg, w, "erm_r50")
        torch.save(m.state_dict(), f); print(f"[erm_r50] saved ({time.time()-t0:.0f}s)", flush=True)
        del m, tr, va; torch.cuda.empty_cache()
    else:
        print("[erm_r50] exists, skip")

    # ----- Slim -----
    f = os.path.join(CKPT_DIR, "slim_r50.pt")
    if not os.path.exists(f):
        p1.set_seed(SEED)
        tr = DataLoader(p1.ISICImgDataset(df_tr, IMDIR, IMG, train=True), cfg["batch_size"],
                        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        va = DataLoader(p1.ISICImgDataset(df_va, IMDIR, IMG, train=False), cfg["batch_size"],
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        m, _ = p1.train_one(Slim_R50(p1.CFG["z_dim"]), tr, va, cfg, w, "slim_r50")
        torch.save(m.state_dict(), f); print(f"[slim_r50] saved ({time.time()-t0:.0f}s)", flush=True)
        del m, tr, va; torch.cuda.empty_cache()
    else:
        print("[slim_r50] exists, skip")

    # ----- Inv (V-REx + artifact-CF) -----
    f = os.path.join(CKPT_DIR, "causal_r50.pt")
    if not os.path.exists(f):
        p1.set_seed(SEED)
        cfg3 = dict(p3.CFG3); cfg3["epochs"] = EPOCHS
        tr = DataLoader(p3.ISICEnvDataset(df_tr, IMDIR, IMG, train=True), cfg["batch_size"],
                        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        va = DataLoader(p3.ISICEnvDataset(df_va, IMDIR, IMG, train=False), cfg["batch_size"],
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        m, _ = p3.train_causal(Slim_R50(p1.CFG["z_dim"]), tr, va, cfg3, w, "causal_r50")
        torch.save(m.state_dict(), f); print(f"[causal_r50] saved ({time.time()-t0:.0f}s)", flush=True)
        del m, tr, va; torch.cuda.empty_cache()
    else:
        print("[causal_r50] exists, skip")

    print(f"DONE all checkpoints -> {CKPT_DIR} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
