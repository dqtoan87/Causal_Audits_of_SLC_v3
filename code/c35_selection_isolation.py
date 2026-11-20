#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C35: isolate the checkpoint-selection effect from training-set size.

Why this is needed. C29/C30 build a three-way patient-disjoint split and measure an artifact magnitude 2.1 to 5.7 times smaller.
But that design changes two things at once: (1) the checkpoint is no longer selected on the audited fold, and
(2) the training set drops from 315 to 237 positive cases. The reduction cannot be attributed to (1) alone.

This script holds (2) fixed and changes only (1):

    split       : the paper's original split, train = 4 folds (315 positive cases), fold 0 = audit
    data        : identical in both arms
    seed/schedule: identical
    epochs      : fixed at 15, no early stopping (early stopping also reads fold 0)

    arm A "leaky"  : checkpoint = epoch with the highest AUROC on fold 0, i.e. the original rule
    arm B "blind"  : checkpoint = final epoch, never references fold 0

Both arms share a single training run, so parameters, batch order and RNG state are the
same; the only difference is which epoch's state is taken. The difference in artifact magnitude between A and B
can therefore be attributed to the checkpoint-selection step itself.

Out -> checkpoints/sel_{leaky,blind}_*.pt and results/c35_selection_isolation.json
Run:  .venv/bin/python Causal_XAI_v3/code/c35_selection_isolation.py
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c2_cace_artifact as c2m       # noqa: E402
import c3_cace_concept as c3m        # noqa: E402
import c30_disjoint_audit as c30     # noqa: E402

p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
sys.path.insert(0, os.path.dirname(p1.__file__))
import phase3_invariance_v01 as p3   # noqa: E402

BASE = os.path.dirname(HERE)
CKPT = os.path.join(BASE, "checkpoints"); os.makedirs(CKPT, exist_ok=True)
OUT = os.path.join(BASE, "results", "c35_selection_isolation.json")
META = "/mnt/data2/Toan/Causal_ML/data/data_ISIC2024/train-metadata.csv"
SEED, EPOCHS, NUM_WORKERS = 42, 15, 0     # /dev/shm 64 MB -> single-process loading
CAUSAL_BATCH = 24                          # the causal job runs two forward passes
N_EVAL, BOOT, MARGIN_P = 200, 4000, 0.05
ART, HELD = c30.ART, c30.HELD
JOBS = [("erm", "erm", "ERM", "plain"), ("slim", "slim", "Slim", "plain"),
        ("causal", "slim", "Inv", "causal")]


def add_env(dfs):
    att = pd.read_csv(META, usecols=["isic_id", "attribution"]).fillna({"attribution": "unknown"})
    envs = sorted(att["attribution"].unique()); emap = {e: i for i, e in enumerate(envs)}
    amap = dict(zip(att["isic_id"], att["attribution"]))
    for d in dfs:
        d["attribution"] = d["isic_id"].map(amap).fillna("unknown")
        d["env"] = d["attribution"].map(emap)


def train_dual(model, tr_loader, va_loader, weight, kind, tag):
    """Train for exactly 15 epochs, return (state at the best epoch on fold 0, state at the final epoch).

    Copies the optimizer/scheduler/loss/AMP/clip of train_one and train_causal exactly, dropping only early
    stopping, because early stopping is itself a decision that reads fold 0.
    """
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=p1.CFG["lr"],
                            weight_decay=p1.CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    use_amp = (DEVICE == "cuda")
    crit = p1.FocalLoss(p1.CFG["focal_gamma"], weight=weight.to(DEVICE)) if kind == "plain" else None
    w_dev = weight.to(DEVICE)
    best_auc, best_state, best_ep, curve = -1.0, None, -1, []

    for ep in range(EPOCHS):
        model.train(); t0 = time.time()
        use_pen = (kind == "causal") and (ep >= p3.CFG3["warmup"])
        for batch in tr_loader:
            if kind == "plain":
                img, y = batch
            else:
                img, y, env = batch; env = env.to(DEVICE, non_blocking=True)
            img = img.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(img)
                if kind == "plain":
                    loss = crit(logits, y)
                else:
                    fps = p3.focal_per_sample(logits, y, w_dev)
                    mean_r, var_r = p3.vrex(fps, env)
                    loss = mean_r
                    if use_pen:
                        logits2 = model(p3.gpu_artifact(img))
                        cf = F.mse_loss(torch.softmax(logits2, 1), torch.softmax(logits, 1))
                        loss = mean_r + p3.CFG3["lambda_rex"] * var_r + p3.CFG3["lambda_cf"] * cf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        yt, ys = (p1.evaluate(model, va_loader) if kind == "plain"
                  else p3.evaluate_env(model, va_loader))
        auc = float(roc_auc_score(yt, ys)) if np.all(np.isfinite(ys)) and len(np.unique(yt)) > 1 \
            else float("nan")
        curve.append(auc)
        print("      [%s] ep%02d/%d  fold0_auc=%.4f  (%.0fs)"
              % (tag, ep + 1, EPOCHS, auc, time.time() - t0), flush=True)
        if np.isfinite(auc) and auc > best_auc:
            best_auc, best_ep = auc, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state, last_state, best_ep, best_auc, curve


def audit(state, kind, imgs, edits, labels, pid):
    """Artifact criterion on the same Tier-2 evaluation set of fold 0."""
    m = (p1.build_model("erm" if kind == "erm" else "slim", p1.CFG["z_dim"])).to(DEVICE)
    m.load_state_dict(state); m.eval()
    p0 = c30.probs(m, imgs)
    per = {a: c30.probs(m, [edits[i][a] for i in range(len(imgs))]) for a in ART}
    del m
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    absdp = np.mean([np.abs(per[a] - p0) for a in HELD], 0)
    out = {"identity_max_abs_dlogit": float(np.max(np.abs(
        c30.clipped_logit(per["identity"]) - c30.clipped_logit(p0)))),
        "p0_mean_malignant": float(p0[labels == 1].mean()),
        "hair_dose_dp": [float((per[a] - p0)[labels == 1].mean())
                         for a in ["hair_light", "hair", "hair_heavy"]]}
    for sub, msk in [("all", np.ones(len(labels), bool)), ("malignant", labels == 1),
                     ("benign", labels == 0)]:
        mean, lo, hi = c30.boot_ci(absdp[msk], pid[msk], BOOT)
        out[sub] = {"mean": mean, "ci": [lo, hi], "verdict": c30.verdict(lo, hi, MARGIN_P)}
    return out


def main():
    t0 = time.time()
    print(f"##### C35 isolating the checkpoint-selection effect {time.strftime('%H:%M:%S')} #####",
          flush=True)
    df_tr, df_va = p2.build_split({"seed": SEED, "n_folds": 5})
    df_va = df_va.reset_index(drop=True)
    add_env([df_tr, df_va])
    print("  TRAIN %d images / %d patients / %d positive cases   (original split, size not reduced)"
          % (len(df_tr), df_tr["patient_id"].nunique(), int(df_tr["target"].sum())), flush=True)
    print("  FOLD0 %d images / %d patients / %d positive cases   (both the audit set and where arm A selects)"
          % (len(df_va), df_va["patient_id"].nunique(), int(df_va["target"].sum())), flush=True)

    imgs, labels = p2.load_eval_images(df_va, N_EVAL, SEED)
    labels = np.array(labels)
    pos = df_va[df_va["target"] == 1].sample(n=int((labels == 1).sum()), random_state=SEED)
    neg = df_va[df_va["target"] == 0].sample(n=int((labels == 0).sum()), random_state=SEED)
    pid = df_va.loc[np.r_[pos.index.values, neg.index.values], "patient_id"].values
    print("  EVAL  %d images / %d patients (%d malignant), the same images as the paper's post-selection audit"
          % (len(imgs), len(np.unique(pid)), int((labels == 1).sum())), flush=True)
    edits, change_frac = c2m.precompute_edits(imgs, ART, SEED)

    w = p1.class_weight(df_tr) if hasattr(p1, "class_weight") else None
    if w is None:
        import c8_backbone2_train as c8
        w = c8.class_weight(df_tr)
    IMG, IMDIR = p1.CFG["img_size"], p1.ISIC_IMAGE_DIR
    res = {"n_train_images": int(len(df_tr)), "n_train_pos": int(df_tr["target"].sum()),
           "n_eval_images": int(len(imgs)), "n_eval_patients": int(len(np.unique(pid))),
           "epochs": EPOCHS, "boot": BOOT, "margin_p": MARGIN_P, "change_frac": change_frac,
           "note": ("Checkpoint-selection leakage isolated from training-set size. Both arms share "
                    "one training run on the original split (315 malignant training cases) and are "
                    "audited on the same fold-0 evaluation images. Arm 'leaky' takes the epoch with "
                    "the highest AUROC on that same fold; arm 'blind' takes the final epoch and "
                    "never reads it. No early stopping, since that also consults fold 0."),
           "models": {}}

    # Training here is fully deterministic (p1.set_seed before each model, no early stopping),
    # so when both checkpoints exist they can be reloaded instead of retraining 15 epochs. The
    # per-epoch AUROC curve is produced only during training, so this mode reads it from the existing JSON.
    reuse = os.environ.get("C35_REUSE_CKPT") == "1"
    prev = {}
    if reuse and os.path.exists(OUT):
        with open(OUT) as fh:
            prev = json.load(fh).get("models", {})

    for name, kind, nm, mode in JOBS:
        f_leaky = os.path.join(CKPT, f"sel_leaky_{name}.pt")
        f_blind = os.path.join(CKPT, f"sel_blind_{name}.pt")
        if reuse and os.path.exists(f_leaky) and os.path.exists(f_blind) and nm in prev:
            print(f"\n[{name}] reloading checkpoints, skipping training", flush=True)
            best = torch.load(f_leaky, map_location=DEVICE)
            last = torch.load(f_blind, map_location=DEVICE)
            curve = prev[nm]["auc_curve_on_fold0"]
            best_ep, best_auc = prev[nm]["leaky_epoch"], prev[nm]["leaky_auc"]
        else:
            print(f"\n[{name}] training ({mode}, fixed 15 epochs) ...", flush=True)
            p1.set_seed(SEED)
            ds = p1.ISICImgDataset if mode == "plain" else p3.ISICEnvDataset
            bs = p1.CFG["batch_size"] if mode == "plain" else CAUSAL_BATCH
            tr = DataLoader(ds(df_tr, IMDIR, IMG, train=True), bs, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
            va = DataLoader(ds(df_va, IMDIR, IMG, train=False), bs, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
            best, last, best_ep, best_auc, curve = train_dual(
                p1.build_model(kind, p1.CFG["z_dim"]), tr, va, w, mode, name)
            torch.save(best, f_leaky)
            torch.save(last, f_blind)
            del tr, va
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"[{name}] leaky = epoch {best_ep} (fold0 AUROC {best_auc:.4f}); "
              f"blind = epoch {EPOCHS} (fold0 AUROC {curve[-1]:.4f})  ({time.time()-t0:.0f}s)",
              flush=True)
        res["models"][nm] = {"auc_curve_on_fold0": curve, "leaky_epoch": best_ep,
                            "leaky_auc": best_auc, "blind_epoch": EPOCHS,
                            "blind_auc": curve[-1],
                            "leaky": audit(best, kind, imgs, edits, labels, pid),
                            "blind": audit(last, kind, imgs, edits, labels, pid)}

    print("\n" + "=" * 92)
    print("Checkpoint-selection effect, training-set size held at 315 positive cases")
    print("%-6s%-11s%14s%14s%9s   %s" % ("model", "subgroup", "leaky E|Δp|", "blind E|Δp|",
                                          "ratio", "verdict (leaky / blind)"))
    ratios = []
    for nm, v in res["models"].items():
        for sub in ("all", "malignant", "benign"):
            a, b = v["leaky"][sub], v["blind"][sub]
            r = a["mean"] / b["mean"]; ratios.append(r)
            print("%-6s%-11s%14.3f%14.3f%9.2f   %s / %s"
                  % (nm, sub, a["mean"], b["mean"], r, a["verdict"], b["verdict"]))
    cells = [c for v in res["models"].values() for arm in ("leaky", "blind")
             for c in (v[arm]["all"], v[arm]["malignant"], v[arm]["benign"])]
    res["summary"] = {"n_cells": len(cells),
                      "exceedance": sum(c["verdict"] == "exceedance" for c in cells),
                      "ratio_min": float(min(ratios)), "ratio_max": float(max(ratios)),
                      "ratio_malignant": {nm: v["leaky"]["malignant"]["mean"]
                                          / v["blind"]["malignant"]["mean"]
                                          for nm, v in res["models"].items()}}
    print("\nleaky/blind ratio: %.2f to %.2f | exceedance %d/%d cells"
          % (res["summary"]["ratio_min"], res["summary"]["ratio_max"],
             res["summary"]["exceedance"], res["summary"]["n_cells"]))
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved -> %s  (%.0fs)" % (OUT, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
