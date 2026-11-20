#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C46: the artifact criterion on a second cohort, PAD-UFES-20.

Why. The artifact results rest on a single cohort, ISIC-2024, i.e. tiles cropped from 3D total-body
photography. The obvious question is whether the results are specific to that cohort or to that acquisition
modality. PAD-UFES-20 consists of clinical smartphone photos, a clearly different modality; it has
patient IDs, so a patient-level split is possible, and a 47% malignancy rate, so no undersampling is needed.

Three steps, each reusing the exact conventions of the primary cohort.
  1. Three-way patient-level partition, fixed by img_id, written to a manifest.
  2. Train ERM and Slim with the same backbone, loss and schedule. Inv is not trained because
     this cohort has no environment variable equivalent to acquisition site.
  3. Apply the same overlay family, the same dose ladder and the same three-way interval rule.

Three things to read alongside the results. Synthetic hair is less physically plausible on clinical images than on
dermoscopy images, while a ruler is more plausible. Lesions are not center-cropped as in TBP tiles. And the
baseline probability sits in a completely different range because prevalence is 47% rather than 0.1%, which interacts
directly with the shrinkage law.

Out -> results/c46_padufes_artifact.json, results/padufes_split.json
Run:  python code/c46_padufes_artifact.py [--epochs 15] [--boot 4000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c2_cace_artifact as c2m      # noqa: E402  (ARTIFACTS, precompute_edits, _hair)
import c3_cace_concept as c3m       # noqa: E402
import c30_disjoint_audit as c30    # noqa: E402  (boot_ci, verdict)

p1, DEVICE = c3m.p1, c3m.DEVICE


def probs(model, batch):
    """Score a batch the way PadDataset feeds the trainer, on [0,1] pixels.

    The primary cohort trains through ISICImgDataset, which applies the ImageNet mean and
    standard deviation, so c30.probs normalises to match it. PadDataset does not normalise:
    it hands the trainer ToTensor output. Scoring this cohort through c30.probs would
    therefore evaluate a model on an input distribution it never saw. Rankings survive that
    mismatch, so AUROC is unaffected, but the probabilities move and E|df| is a statement
    about probabilities, which is why this cohort needs its own scorer.
    """
    out = []
    for i in range(0, len(batch), 32):
        x = torch.stack(batch[i:i + 32]).to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            out.append(torch.softmax(model(x).float(), 1)[:, 1].detach().cpu().numpy())
    return np.concatenate(out)

BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
CKPT = os.path.join(BASE, "checkpoints")
OUT = os.path.join(RES, "c46_padufes_artifact.json")
SPLIT = os.path.join(RES, "padufes_split.json")

ROOT = "/mnt/data2/Toan/Causal_ML_XAI/data/PAD-UFES-20"
MALIGNANT = {"BCC", "SCC", "MEL"}
SEED, N_FOLDS, MARGIN_P = 42, 5, 0.05
HELD = ["hair_light", "hair", "hair_heavy", "ruler"]
LADDER = [("hair_x1", 1, 0.15), ("hair_x2", 2, 0.25), ("hair_x3", 3, 0.35),
          ("hair_light", 6, 0.40), ("hair", 14, 0.60), ("hair_heavy", 24, 0.75)]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def img_path(img_id):
    for p in ("imgs_part_1", "imgs_part_2", "imgs_part_3"):
        f = os.path.join(ROOT, p, p, img_id)
        if os.path.exists(f):
            return f
    raise FileNotFoundError(img_id)


def load_meta():
    m = pd.read_csv(os.path.join(ROOT, "metadata.csv"))
    m["target"] = m["diagnostic"].isin(MALIGNANT).astype(int)
    return m[["patient_id", "lesion_id", "img_id", "diagnostic", "target"]]


def build_split():
    """Three patient-disjoint parts, fixed by img_id. Same template as the primary cohort."""
    if os.path.exists(SPLIT):
        return json.load(open(SPLIT))
    m = load_meta()
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = [va for _, va in sgkf.split(m, m["target"].values, m["patient_id"].values)]
    audit_i, select_i = folds[0], folds[1]
    train_i = np.setdiff1d(np.arange(len(m)), np.concatenate([audit_i, select_i]))
    parts = {k: m.iloc[v] for k, v in
             [("train", train_i), ("select", select_i), ("audit", audit_i)]}
    S = [set(d["patient_id"]) for d in parts.values()]
    assert not (S[0] & S[1]) and not (S[0] & S[2]) and not (S[1] & S[2]), "patient leakage"
    out = {"seed": SEED, "n_folds": N_FOLDS,
           "note": ("Frozen three-way patient-disjoint split of PAD-UFES-20, stored by img_id so "
                    "that the fold assignment is never rebuilt."),
           "counts": {k: {"n": int(len(d)), "n_patients": int(d["patient_id"].nunique()),
                          "n_pos": int(d["target"].sum())} for k, d in parts.items()}}
    for k, d in parts.items():
        out[k] = list(d["img_id"])
    json.dump(out, open(SPLIT, "w"), indent=1)
    log(f"new partition -> {SPLIT}")
    return out


class PadDataset(Dataset):
    def __init__(self, df, size, train):
        self.ids = list(df["img_id"]); self.y = list(df["target"])
        self.size = size; self.train = train; self.to_t = T.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(img_path(self.ids[i])).convert("RGB")
        img = p1.center_crop_resize(img, (self.size, self.size))
        x = self.to_t(img)
        if self.train:
            if np.random.rand() < 0.5:
                x = torch.flip(x, [2])
            if np.random.rand() < 0.5:
                x = torch.rot90(x, int(np.random.randint(4)), [1, 2])
        return x, int(self.y[i])


def raw_images(df, size):
    to_t = T.ToTensor()
    return [to_t(p1.center_crop_resize(Image.open(img_path(i)).convert("RGB"), (size, size)))
            for i in df["img_id"]]


def train_model(kind, df_tr, df_se, epochs, size, batch, lr, wd, patience):
    model = p1.build_model(kind, p1.CFG["z_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    npos = int(df_tr["target"].sum()); nneg = len(df_tr) - npos
    w = torch.tensor([1.0, nneg / max(npos, 1)], dtype=torch.float32).to(DEVICE)
    tr = DataLoader(PadDataset(df_tr, size, True), batch, shuffle=True, num_workers=0)
    se = DataLoader(PadDataset(df_se, size, False), batch, shuffle=False, num_workers=0)
    best, best_state, bad = -1.0, None, 0
    from sklearn.metrics import roc_auc_score
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                ce = F.cross_entropy(model(x), y, weight=w, reduction="none")
                loss = ((1 - torch.exp(-ce)) ** p1.CFG["focal_gamma"] * ce).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval(); ps, ys = [], []
        with torch.no_grad():
            for x, y in se:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                    ps.append(torch.softmax(model(x.to(DEVICE)).float(), 1)[:, 1].cpu().numpy())
                ys.append(y.numpy())
        auc = roc_auc_score(np.concatenate(ys), np.concatenate(ps))
        log(f"      ep{ep+1:02d}/{epochs} select-AUC={auc:.4f}")
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"      early stop @ ep{ep+1}"); break
    model.load_state_dict(best_state)
    return model, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--use-saved", action="store_true",
                    help="audit the checkpoints already in checkpoints/ instead of training new "
                         "ones; the reported magnitudes belong to those weights, and training is "
                         "not reproducible run to run, so this is how the reported numbers are "
                         "regenerated")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    log("##### C46 PAD-UFES-20 artifact criterion #####")
    sp = build_split()
    for k, v in sp["counts"].items():
        log(f"  {k:7s} n={v['n']:5d}  patients={v['n_patients']:5d}  malignant={v['n_pos']:4d}")

    m = load_meta().set_index("img_id")
    dfs = {k: m.loc[sp[k]].reset_index() for k in ("train", "select", "audit")}
    size = p1.CFG["img_size"]

    # register the three low doses, keep the primary cohort's three doses unchanged
    for name, n, a in LADDER:
        if name not in c2m.ARTIFACTS:
            c2m.ARTIFACTS[name] = (lambda n_, a_: (lambda x, rng: c2m._hair(x, rng, n=n_, alpha=a_)))(n, a)
    names = [n for n, _, _ in LADDER] + ["ruler", "identity"]

    log("loading audit images ...")
    imgs = raw_images(dfs["audit"], size)
    labels = dfs["audit"]["target"].to_numpy()
    pid = dfs["audit"]["patient_id"].to_numpy()
    log(f"  audit: {len(imgs)} images / {len(np.unique(pid))} patients / {int(labels.sum())} malignant")
    edits, cf = c2m.precompute_edits(imgs, names, SEED)
    log("pixels changed: " + "  ".join(f"{n}={100*cf[n]:.2f}%" for n in names))

    res = {"cohort": "PAD-UFES-20", "margin_p": MARGIN_P, "boot": args.boot, "seed": SEED,
           "weights": ("saved" if args.use_saved else "retrained"),
           "split_counts": sp["counts"], "change_frac": cf,
           "n_audit": int(len(imgs)), "n_patients": int(len(np.unique(pid))),
           "n_malignant": int(labels.sum()),
           "note": ("Artifact criterion transported to a second cohort of clinical smartphone "
                    "images. Same overlay family, same dose ladder, same three-way interval rule "
                    "and same tolerance as the primary cohort."),
           "models": {}}

    for kind, nm in [("erm", "ERM"), ("slim", "Slim")]:
        saved = os.path.join(CKPT, f"pad_{kind}.pt")
        if args.use_saved:
            log(f"[{nm}] auditing saved weights: {saved}")
            model = p1.build_model(kind, p1.CFG["z_dim"]).to(DEVICE)
            model.load_state_dict(torch.load(saved, map_location=DEVICE))
            auc = float("nan")
        else:
            log(f"[{nm}] training ...")
            model, auc = train_model(kind, dfs["train"], dfs["select"], args.epochs, size,
                                     args.batch, p1.CFG["lr"], p1.CFG["weight_decay"],
                                     p1.CFG["patience"])
            torch.save(model.state_dict(), saved)
        model.eval()
        p0 = probs(model, imgs)
        cells = {"select_auc": float(auc), "p0_mean_malignant": float(p0[labels == 1].mean())}
        for a in names:
            pe = probs(model, [edits[i][a] for i in range(len(imgs))])
            ad = np.abs(pe - p0)
            cells[a] = {}
            for sub, msk in [("all", np.ones(len(labels), bool)),
                             ("malignant", labels == 1), ("benign", labels == 0)]:
                _, lo, hi = c30.boot_ci(ad[msk], pid[msk], n_boot=args.boot)
                cells[a][sub] = {"mean": float(ad[msk].mean()), "ci": [lo, hi],
                                 "verdict": c30.verdict(lo, hi, MARGIN_P)}
        agg = np.mean([np.abs(probs(model, [edits[i][a] for i in range(len(imgs))]) - p0)
                       for a in HELD], 0)
        cells["_heldout_aggregate"] = {}
        for sub, msk in [("all", np.ones(len(labels), bool)),
                         ("malignant", labels == 1), ("benign", labels == 0)]:
            _, lo, hi = c30.boot_ci(agg[msk], pid[msk], n_boot=args.boot)
            cells["_heldout_aggregate"][sub] = {"mean": float(agg[msk].mean()), "ci": [lo, hi],
                                                "verdict": c30.verdict(lo, hi, MARGIN_P)}
            log(f"  {nm} {sub:10s} E|Δp|={agg[msk].mean():.4f} [{lo:.4f},{hi:.4f}] "
                f"{cells['_heldout_aggregate'][sub]['verdict']}")
        res["models"][nm] = cells
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        json.dump(res, open(OUT + ".partial", "w"), indent=1)

    v = [res["models"][nm]["_heldout_aggregate"][g]["verdict"]
         for nm in res["models"] for g in ("all", "malignant", "benign")]
    res["summary"] = {"n_cells": len(v), "exceedance": v.count("exceedance"),
                      "equivalence": v.count("equivalence"),
                      "indeterminate": v.count("indeterminate")}
    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    if os.path.exists(OUT + ".partial"):
        os.remove(OUT + ".partial")
    log(f"summary: {res['summary']}  -> {OUT}  ({res['runtime_sec']:.0f}s)")


if __name__ == "__main__":
    main()
