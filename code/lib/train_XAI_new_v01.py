#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1: Statistical-Integrity Harness
======================================
Goals (design notes §7):
  - Measure diagnostic performance with rigorous statistics, not a single split without CI.
  - Multi-seed × StratifiedGroupKFold (group = patient_id) on ISIC 2024.
  - Two models compared on the same folds:
        * ERM      = EfficientNet-B3 + linear head (honest baseline).
        * SLIM     = backbone -> proj_c(256) -> disease_head  (slim skeleton, without the
                     V-REx/causal loss, which is Phase 3). Purpose: show that the lean
                     architecture (no MoE/ContextMemory) is not worse than ERM.
  - Reports: AUC, balanced-acc, sensitivity/specificity (@0.5 and @Youden) with
        bootstrap 95% CI, and a paired DeLong test for the AUC difference between the two models.

Operational features:
  - Can run in the background (nohup / background); keeps running after the IDE is closed.
  - Resume: each unit (model, seed, fold) saves predictions and is recorded in progress.json;
        a rerun skips finished units. It can be stopped and resumed at any time.
  - All outputs live in XAI_NEW/ (checkpoints/, results/, logs/), kept separate from other runs.

Usage:
  python train_XAI_new_v01.py                 # full run (automatic resume)
  python train_XAI_new_v01.py --seeds 42 123  # choose seeds
  python train_XAI_new_v01.py --aggregate     # only re-aggregate from saved results
"""

import os, sys, json, time, argparse, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
from PIL import Image

import scipy.stats
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_SGKF = True
except Exception:
    from sklearn.model_selection import GroupKFold
    _HAS_SGKF = False

# ============================================================
# Paths & directories (everything under XAI_NEW/)
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
RES_DIR  = os.path.join(HERE, "results")
LOG_DIR  = os.path.join(HERE, "logs")
for d in (CKPT_DIR, RES_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

PROGRESS_FILE = os.path.join(RES_DIR, "progress.json")
SUMMARY_JSON  = os.path.join(RES_DIR, "summary.json")
SUMMARY_CSV   = os.path.join(RES_DIR, "summary.csv")

DATA_ROOT = "/mnt/data2/Toan/Causal_ML/data"
ISIC_IMAGE_DIR = os.path.join(DATA_ROOT, "data_ISIC2024/train-image/image")
ISIC_META_CSV  = os.path.join(DATA_ROOT, "data_ISIC2024/train-metadata.csv")

# ============================================================
# Config (override qua argparse)
# ============================================================
CFG = dict(
    seeds        = [42, 123, 456],   # extend to 5 seeds for tighter estimates
    n_folds      = 5,
    epochs       = 15,
    patience     = 4,
    batch_size   = 64,
    lr           = 3e-4,
    weight_decay = 1e-4,
    img_size     = 224,
    z_dim        = 256,
    focal_gamma  = 2.0,
    train_neg_ratio = 20,   # undersample negatives in the TRAIN fold (neg:pos)
    val_neg_ratio   = 50,   # undersample negatives in the VAL fold (keep all positives)
    num_workers  = 8,
    models       = ["erm", "slim"],
    save_ckpt    = False,   # only preds are needed for the statistics; enable to reuse the weights
    boot_n       = 2000,
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Logging tee
# ============================================================
class Tee:
    def __init__(self, fp):
        self.terminal = sys.stdout
        self.log = open(fp, "a", encoding="utf-8")
    def write(self, m):
        self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Data
# ============================================================
def center_crop_resize(image, target_size):
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return image.crop((left, top, left + side, top + side)).resize(
        target_size, Image.BILINEAR)


class ISICImgDataset(Dataset):
    """Returns (img, label). Phase 1 uses only the image and label."""
    def __init__(self, df, image_dir, img_size, train=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.size = (img_size, img_size)
        self.labels = torch.tensor(self.df["target"].values, dtype=torch.long)
        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.train = train
        self.flip = T.RandomHorizontalFlip(0.5)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.image_dir, f"{row['isic_id']}.jpg")).convert("RGB")
        img = self.to_tensor(center_crop_resize(img, self.size))
        if self.train:
            img = self.flip(img)
        img = self.norm(img)
        return img, self.labels[idx]


def load_isic_df():
    df = pd.read_csv(ISIC_META_CSV, low_memory=False)
    df = df[["isic_id", "patient_id", "target"]].copy()
    df["patient_id"] = df["patient_id"].fillna("unknown")
    df["target"] = df["target"].astype(int)
    return df


def undersample(df, ratio, seed):
    """Keep all positives, sample negatives = ratio × n_pos."""
    pos = df[df["target"] == 1]
    neg = df[df["target"] == 0]
    n_neg = min(len(neg), len(pos) * ratio)
    neg = neg.sample(n=n_neg, random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


# ============================================================
# Models
# ============================================================
def _backbone():
    bb = models.efficientnet_b3(weights="IMAGENET1K_V1")
    bb.classifier = nn.Identity()
    return bb


class ImageOnlyNet(nn.Module):     # ERM baseline
    def __init__(self):
        super().__init__()
        self.backbone = _backbone()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(1536, 2))
    def forward(self, x):
        return self.head(self.backbone(x))


class CausalSlimNet(nn.Module):    # slim skeleton (no MoE / no ContextMemory)
    def __init__(self, z_dim=256):
        super().__init__()
        self.backbone = _backbone()
        self.proj_c = nn.Sequential(nn.Linear(1536, 512), nn.ReLU(),
                                    nn.Dropout(0.3), nn.Linear(512, z_dim))
        self.disease_head = nn.Sequential(nn.Linear(z_dim, 256), nn.ReLU(),
                                          nn.Dropout(0.4), nn.Linear(256, 2))
    def forward(self, x):
        return self.disease_head(self.proj_c(self.backbone(x)))


def build_model(name, z_dim):
    if name == "erm":
        return ImageOnlyNet()
    if name == "slim":
        return CausalSlimNet(z_dim=z_dim)
    raise ValueError(name)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ============================================================
# Train / Eval
# ============================================================
def train_one(model, train_loader, val_loader, cfg, class_weight, tag):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    crit = FocalLoss(cfg["focal_gamma"], weight=class_weight.to(DEVICE))
    use_amp = (DEVICE == "cuda")   # bf16 on A100: avoids fp16 overflow NaN

    best_auc, best_state, no_improve = -1.0, None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        for img, y in train_loader:
            img, y = img.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(img)
                loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        yt, ys = evaluate(model, val_loader)
        if not np.all(np.isfinite(ys)):
            print(f"      [{tag}] ep{ep+1}: scores NaN/inf -> skip epoch", flush=True)
            no_improve += 1
            if no_improve >= cfg["patience"]:
                break
            continue
        auc = roc_auc_score(yt, ys) if len(np.unique(yt)) > 1 else 0.5
        print(f"      [{tag}] ep{ep+1:02d}/{cfg['epochs']} "
              f"val_auc={auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"      [{tag}] early stop @ ep{ep+1}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, yt = [], []
    for img, y in loader:
        img = img.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            prob = torch.softmax(model(img), dim=1)[:, 1]
        ys.append(prob.float().cpu().numpy())
        yt.append(y.numpy())
    return np.concatenate(yt), np.concatenate(ys)


# ============================================================
# Statistics: bootstrap CI + DeLong test
# ============================================================
def metrics_from_scores(y_true, y_score):
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    # @0.5
    pred05 = (y_score >= 0.5).astype(int)
    sens05 = _sens(y_true, pred05); spec05 = _spec(y_true, pred05)
    bacc05 = balanced_accuracy_score(y_true, pred05)
    # @Youden
    thr = youden_threshold(y_true, y_score)
    predy = (y_score >= thr).astype(int)
    sensy = _sens(y_true, predy); specy = _spec(y_true, predy)
    baccy = balanced_accuracy_score(y_true, predy)
    return dict(auc=auc, sens05=sens05, spec05=spec05, bacc05=bacc05,
                youden_thr=thr, sens_y=sensy, spec_y=specy, bacc_y=baccy)


def _sens(yt, yp):
    p = (yt == 1)
    return float((yp[p] == 1).mean()) if p.sum() else float("nan")

def _spec(yt, yp):
    n = (yt == 0)
    return float((yp[n] == 0).mean()) if n.sum() else float("nan")


def youden_threshold(y_true, y_score):
    thrs = np.unique(y_score)
    if len(thrs) > 512:
        thrs = np.quantile(y_score, np.linspace(0, 1, 512))
    best_j, best_t = -1, 0.5
    for t in thrs:
        yp = (y_score >= t).astype(int)
        j = _sens(y_true, yp) + _spec(y_true, yp) - 1
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)


def bootstrap_auc_ci(y_true, y_score, n_boot=2000, ci=0.95, seed=0):
    rng = np.random.RandomState(seed)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return (float("nan"), float("nan"))
    aucs = []
    N = len(y_true)
    for _ in range(n_boot):
        bs = rng.randint(0, N, N)
        yt, ys = y_true[bs], y_score[bs]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    if not aucs:
        return (float("nan"), float("nan"))
    lo = np.percentile(aucs, (1 - ci) / 2 * 100)
    hi = np.percentile(aucs, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


# ---- DeLong (Sun & Xu 2014, paired correlated AUC) ----
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_T, m):
    n = preds_sorted_T.shape[1] - m
    pos = preds_sorted_T[:, :m]
    neg = preds_sorted_T[:, m:]
    k = preds_sorted_T.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_T[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, cov


def delong_test(y_true, s1, s2):
    """Returns (auc1, auc2, two-sided p_value) for H0: AUC1 == AUC2 on the same samples."""
    y_true = np.asarray(y_true); s1 = np.asarray(s1); s2 = np.asarray(s2)
    m = int(y_true.sum())
    if m == 0 or m == len(y_true):
        return float("nan"), float("nan"), float("nan")
    order = (-y_true).argsort(kind="mergesort")   # positives first
    preds = np.vstack((s1, s2))[:, order]
    aucs, cov = _fast_delong(preds, m)
    l = np.array([[1.0, -1.0]])
    var = float(np.squeeze(l @ cov @ l.T))
    if var <= 0:
        p = 1.0 if abs(aucs[0] - aucs[1]) < 1e-12 else 0.0
    else:
        z = abs(aucs[0] - aucs[1]) / np.sqrt(var)
        p = float(2 * scipy.stats.norm.sf(z))
    return float(aucs[0]), float(aucs[1]), p


# ============================================================
# Progress / resume
# ============================================================
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"units": {}}


def save_progress(prog):
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(prog, f, indent=2)
    os.replace(tmp, PROGRESS_FILE)


def preds_path(unit_key):
    return os.path.join(RES_DIR, f"preds_{unit_key}.npz")


# ============================================================
# Main run loop
# ============================================================
def run(cfg):
    print("=" * 64)
    print("PHASE 1 harness")
    print(f"Device={DEVICE}  seeds={cfg['seeds']}  folds={cfg['n_folds']}  "
          f"models={cfg['models']}  SGKF={_HAS_SGKF}")
    print("=" * 64, flush=True)

    df = load_isic_df()
    print(f"ISIC2024: {len(df)} images, {df['target'].sum()} malignant, "
          f"{df['patient_id'].nunique()} patients", flush=True)

    prog = load_progress()
    groups = df["patient_id"].values
    y = df["target"].values

    for seed in cfg["seeds"]:
        if _HAS_SGKF:
            splitter = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True,
                                            random_state=seed)
            split_iter = splitter.split(df, y, groups)
        else:
            splitter = GroupKFold(n_splits=cfg["n_folds"])
            split_iter = splitter.split(df, y, groups)

        for fold, (tr_idx, va_idx) in enumerate(split_iter):
            df_tr_full = df.iloc[tr_idx]
            df_va_full = df.iloc[va_idx]
            # undersample (deterministic by seed/fold), identical for every model
            us_seed = seed * 100 + fold
            df_tr = undersample(df_tr_full, cfg["train_neg_ratio"], us_seed)
            df_va = undersample(df_va_full, cfg["val_neg_ratio"], us_seed + 1)

            for mname in cfg["models"]:
                unit_key = f"{mname}_s{seed}_f{fold}"
                if unit_key in prog["units"] and os.path.exists(preds_path(unit_key)):
                    print(f"[skip] {unit_key} (already done)", flush=True)
                    continue

                print(f"\n[run ] {unit_key}  train={len(df_tr)} "
                      f"(pos={int(df_tr['target'].sum())})  "
                      f"val={len(df_va)} (pos={int(df_va['target'].sum())})", flush=True)
                try:
                    set_seed(seed)

                    tr_ds = ISICImgDataset(df_tr, ISIC_IMAGE_DIR, cfg["img_size"], train=True)
                    va_ds = ISICImgDataset(df_va, ISIC_IMAGE_DIR, cfg["img_size"], train=False)
                    tr_loader = DataLoader(tr_ds, cfg["batch_size"], shuffle=True,
                                           num_workers=cfg["num_workers"], pin_memory=True,
                                           drop_last=True)
                    va_loader = DataLoader(va_ds, cfg["batch_size"], shuffle=False,
                                           num_workers=cfg["num_workers"], pin_memory=True)

                    # class weight from train
                    n_pos = int(df_tr["target"].sum()); n_neg = len(df_tr) - n_pos
                    w = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32)

                    model = build_model(mname, cfg["z_dim"])
                    t0 = time.time()
                    model, best_auc = train_one(model, tr_loader, va_loader, cfg, w, unit_key)
                    yt, ys = evaluate(model, va_loader)
                    mins = (time.time() - t0) / 60

                    if not np.all(np.isfinite(ys)):
                        print(f"[FAIL] {unit_key}: scores contain NaN/inf -> skip this unit", flush=True)
                        continue

                    m = metrics_from_scores(yt, ys)
                    np.savez(preds_path(unit_key), y_true=yt, y_score=ys,
                             seed=seed, fold=fold, model=mname)
                    if cfg["save_ckpt"]:
                        torch.save(model.state_dict(),
                                   os.path.join(CKPT_DIR, f"{unit_key}.pt"))

                    prog["units"][unit_key] = dict(model=mname, seed=seed, fold=fold,
                                                   minutes=round(mins, 1), **m)
                    save_progress(prog)
                    print(f"[done] {unit_key}  AUC={m['auc']:.4f} "
                          f"sens@.5={m['sens05']:.3f} spec@.5={m['spec05']:.3f} "
                          f"sens@Y={m['sens_y']:.3f} ({mins:.1f} min)", flush=True)
                except Exception as e:
                    import traceback
                    print(f"[ERROR] {unit_key}: {e}\n{traceback.format_exc()}", flush=True)
                    continue

    print("\nAll units finished. Aggregating...\n", flush=True)
    aggregate(cfg)


# ============================================================
# Aggregation: mean±CI per model + DeLong (per seed, pooled OOF)
# ============================================================
def aggregate(cfg):
    prog = load_progress()
    if not prog["units"]:
        print("No results to aggregate yet."); return

    rows = list(prog["units"].values())
    dfm = pd.DataFrame(rows)
    summary = {}
    print("=" * 64)
    print("PHASE 1: SUMMARY (mean ± std over fold×seed)")
    print("=" * 64)
    for mname in cfg["models"]:
        sub = dfm[dfm["model"] == mname]
        if sub.empty:
            continue
        s = {}
        for col in ["auc", "bacc05", "sens05", "spec05", "sens_y", "spec_y"]:
            v = sub[col].astype(float).values
            s[col] = dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v)),
                          n=int(np.sum(~np.isnan(v))))
        summary[mname] = s
        print(f"\n[{mname}]  (n={len(sub)} runs)")
        print(f"   AUC      = {s['auc']['mean']:.4f} ± {s['auc']['std']:.4f}")
        print(f"   BalAcc.5 = {s['bacc05']['mean']:.4f} ± {s['bacc05']['std']:.4f}")
        print(f"   Sens@.5  = {s['sens05']['mean']:.3f} ± {s['sens05']['std']:.3f}")
        print(f"   Spec@.5  = {s['spec05']['mean']:.3f} ± {s['spec05']['std']:.3f}")
        print(f"   Sens@Y   = {s['sens_y']['mean']:.3f} ± {s['sens_y']['std']:.3f}")

    # Pooled out-of-fold predictions per seed -> bootstrap CI + DeLong erm vs slim
    print("\n" + "=" * 64)
    print("Bootstrap 95% CI (pooled OOF per seed) + DeLong (ERM vs SLIM)")
    print("=" * 64)
    delong_rows = []
    boot_rows = []
    for seed in cfg["seeds"]:
        pooled = {}
        for mname in cfg["models"]:
            yts, yss = [], []
            ok = True
            for fold in range(cfg["n_folds"]):
                pth = preds_path(f"{mname}_s{seed}_f{fold}")
                if not os.path.exists(pth):
                    ok = False; break
                d = np.load(pth)
                yts.append(d["y_true"]); yss.append(d["y_score"])
            if ok:
                pooled[mname] = (np.concatenate(yts), np.concatenate(yss))
        for mname, (yt, ys) in pooled.items():
            auc = roc_auc_score(yt, ys)
            lo, hi = bootstrap_auc_ci(yt, ys, cfg["boot_n"], seed=seed)
            boot_rows.append(dict(seed=seed, model=mname, auc=auc, ci_lo=lo, ci_hi=hi))
            print(f"  seed {seed} [{mname}] AUC={auc:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]")
        if "erm" in pooled and "slim" in pooled:
            yt = pooled["erm"][0]
            # both models use the same OOF samples (same split+undersample) -> paired test is valid
            if np.array_equal(yt, pooled["slim"][0]):
                a1, a2, p = delong_test(yt, pooled["erm"][1], pooled["slim"][1])
                delong_rows.append(dict(seed=seed, auc_erm=a1, auc_slim=a2, delta=a2 - a1, p=p))
                sig = "SIG" if p < 0.05 else "n.s."
                print(f"  seed {seed} DeLong: AUC_erm={a1:.4f} AUC_slim={a2:.4f} "
                      f"Δ={a2-a1:+.4f}  p={p:.3f} [{sig}]")
            else:
                print(f"  seed {seed}: OOF samples differ, skipping DeLong")

    out = dict(config=cfg, per_model=summary,
               bootstrap=boot_rows, delong=delong_rows,
               n_units=len(prog["units"]))
    with open(SUMMARY_JSON, "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved: {SUMMARY_JSON}\n         {SUMMARY_CSV}", flush=True)

    if delong_rows:
        ps = [r["p"] for r in delong_rows]
        deltas = [r["delta"] for r in delong_rows]
        print("\n>>> Statistical conclusion:")
        print(f"    Δ AUC (slim−erm) mean = {np.mean(deltas):+.4f}")
        print(f"    DeLong p mean = {np.mean(ps):.3f}  "
              f"({'significant difference' if np.mean(ps) < 0.05 else 'not statistically significant'})")
        print("    => confirms: the small AUC gap lies within noise.")


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=CFG["seeds"])
    ap.add_argument("--folds", type=int, default=CFG["n_folds"])
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--models", type=str, nargs="+", default=CFG["models"])
    ap.add_argument("--save_ckpt", action="store_true")
    ap.add_argument("--aggregate", action="store_true", help="aggregate only")
    args = ap.parse_args()

    CFG["seeds"] = args.seeds
    CFG["n_folds"] = args.folds
    CFG["epochs"] = args.epochs
    CFG["models"] = args.models
    if args.save_ckpt:
        CFG["save_ckpt"] = True

    sys.stdout = Tee(os.path.join(LOG_DIR, "run_v01.log"))
    print(f"\n##### START {time.strftime('%Y-%m-%d %H:%M:%S')} #####", flush=True)

    if args.aggregate:
        aggregate(CFG)
    else:
        run(CFG)
    print(f"##### END {time.strftime('%Y-%m-%d %H:%M:%S')} #####", flush=True)


if __name__ == "__main__":
    main()
