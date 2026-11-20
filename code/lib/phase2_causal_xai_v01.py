#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2: Causal XAI with substantive metrics (Module B)
=======================================================
Uses metrics that cannot be optimised directly (unlike tautological metrics such as CF=0,
Causal%=100%). They are measured directly at the image level and always compared with the
ERM baseline:

  (1) Deletion AUC  (necessity):  progressively delete important pixels (by GradCAM) ->
        prob must drop quickly. Low AUC = the attribution is truly necessary.
  (2) Insertion AUC (sufficiency): progressively insert important pixels into a blurred base ->
        prob must rise quickly. High AUC = the attribution is sufficient to explain.
  (3) Artifact-Δ (robustness):    add acquisition artifacts (hair/ruler/patch/
        vignette) -> |Δ prob|. Low = the model ignores the acquisition confounder
        (this is a substantive debiasing test, unlike a metadata counterfactual).

Compares SLIM (slim skeleton, no MoE/ContextMemory) with ERM. Note: Phase 2 does not yet use
V-REx (Phase 3), so SLIM ≈ ERM on artifact-Δ is expected. The goal of Phase 2 is to build
the metrics and obtain baseline numbers against which Phase 3 shows improvement.

Operation: can run in the background; caches checkpoints and JSON results so it can resume.
All outputs go to XAI_NEW/.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_XAI_new_v01 as p1   # reuse Phase 1 model/data/training

CKPT_DIR = p1.CKPT_DIR
RES_DIR  = p1.RES_DIR
LOG_DIR  = p1.LOG_DIR
DEVICE   = p1.DEVICE
ISIC_IMAGE_DIR = p1.ISIC_IMAGE_DIR

RESULTS_FILE = os.path.join(RES_DIR, "phase2_results.json")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

CFG2 = dict(
    seed        = 42,
    n_folds     = 5,
    train_epochs= 12,
    n_eval      = 120,     # number of XAI evaluation images (balanced mal/benign)
    di_steps    = 21,      # number of deletion/insertion steps
    blur_sigma  = 11,      # blurred baseline for insertion/deletion
    boot_n      = 2000,
    models      = ["erm", "slim"],
)


# ============================================================
# GradCAM (hook on backbone.features, shared by ERM and SLIM)
# ============================================================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.acts = None
        self.grads = None
        target = model.backbone.features          # last conv block of EfficientNet
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o): self.acts = o
    def _bwd(self, m, gi, go): self.grads = go[0]

    @torch.enable_grad()
    def __call__(self, x, cls=1):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[:, cls].sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1))                       # [B,h,w]
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:],
                            mode="bilinear", align_corners=False).squeeze(1)
        cam = cam - cam.amin((1, 2), keepdim=True)
        cam = cam / (cam.amax((1, 2), keepdim=True) + 1e-8)
        return cam.detach()


# ============================================================
# Deletion / Insertion AUC
# ============================================================
def _gaussian_blur(x, sigma):
    k = int(sigma) | 1
    return T.functional.gaussian_blur(x, kernel_size=[k, k], sigma=[float(sigma)] * 2)


@torch.no_grad()
def deletion_insertion(model, x_norm, cam, steps, sigma):
    """x_norm: [3,H,W], normalised. cam: [H,W]. Returns (del_auc, ins_auc)."""
    H, W = cam.shape
    order = torch.argsort(cam.flatten(), descending=True)   # most to least important
    base = _gaussian_blur(x_norm.unsqueeze(0), sigma).squeeze(0)
    fracs = np.linspace(0, 1, steps)

    del_imgs, ins_imgs = [], []
    for fr in fracs:
        n = int(fr * H * W)
        mask = torch.zeros(H * W, device=x_norm.device)
        if n > 0:
            mask[order[:n]] = 1.0
        mask = mask.view(1, H, W)
        del_imgs.append(x_norm * (1 - mask) + base * mask)      # delete important pixels
        ins_imgs.append(base * (1 - mask) + x_norm * mask)      # insert important pixels
    del_b = torch.stack(del_imgs); ins_b = torch.stack(ins_imgs)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
        pd = torch.softmax(model(del_b.to(DEVICE)), 1)[:, 1].float().cpu().numpy()
        pi = torch.softmax(model(ins_b.to(DEVICE)), 1)[:, 1].float().cpu().numpy()
    return float(np.trapezoid(pd, fracs)), float(np.trapezoid(pi, fracs))


# ============================================================
# Artifact interventions (on the [0,1] image, then normalised)
# ============================================================
def _to_norm(x01):
    return (x01 - MEAN.to(x01.device)) / STD.to(x01.device)


def art_hair(x, rng, n=12):
    x = x.clone(); _, H, W = x.shape
    for _ in range(n):
        x0, y0 = rng.randint(0, W), rng.randint(0, H)
        x1, y1 = np.clip([x0 + rng.randint(-W//2, W//2),
                          y0 + rng.randint(-H//2, H//2)], 0, [W-1, H-1])
        steps = max(abs(int(x1)-x0), abs(int(y1)-y0), 1)
        xs = np.linspace(x0, int(x1), steps).astype(int)
        ys = np.linspace(y0, int(y1), steps).astype(int)
        for px, py in zip(xs, ys):
            for dy in (-1, 0, 1):
                yy = np.clip(py + dy, 0, H-1)
                x[:, yy, px] = 0.05      # dark hair strand
    return x


def art_ruler(x, rng):
    x = x.clone(); _, H, W = x.shape
    y = rng.randint(0, H)
    x[:, max(0, y-1):y+2, :] = 0.0
    for tx in range(0, W, 20):
        x[:, max(0, y-5):y+5, tx:tx+2] = 0.0
    return x


def art_patch(x, rng):
    x = x.clone(); _, H, W = x.shape
    s = H // 6
    x[0, :s, :s] = 0.9; x[1, :s, :s] = 0.2; x[2, :s, :s] = 0.2   # coloured corner patch
    return x


def art_vignette(x, rng):
    _, H, W = x.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                            indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)
    v = torch.clamp(1.0 - 0.7 * torch.relu(r - 0.6), 0, 1).to(x.device)
    return x * v


ARTIFACTS = {"hair": art_hair, "ruler": art_ruler,
             "patch": art_patch, "vignette": art_vignette}


@torch.no_grad()
def artifact_delta(model, x01, rng):
    """Returns dict {artifact: |Δ prob malignant|}."""
    def prob(img01):
        xn = _to_norm(img01).unsqueeze(0).to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            return float(torch.softmax(model(xn), 1)[0, 1])
    p0 = prob(x01)
    out = {}
    for name, fn in ARTIFACTS.items():
        out[name] = abs(prob(fn(x01, rng)) - p0)
    return out


# ============================================================
# Data: split seed 42, train + fetch eval images (raw [0,1])
# ============================================================
def build_split(cfg):
    df = p1.load_isic_df()
    sgkf = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True,
                                random_state=cfg["seed"])
    tr_idx, va_idx = next(sgkf.split(df, df["target"].values, df["patient_id"].values))
    df_tr = p1.undersample(df.iloc[tr_idx], p1.CFG["train_neg_ratio"], cfg["seed"]*100)
    df_va = p1.undersample(df.iloc[va_idx], p1.CFG["val_neg_ratio"], cfg["seed"]*100+1)
    return df_tr, df_va


def get_or_train(mname, df_tr, df_va, cfg):
    ckpt = os.path.join(CKPT_DIR, f"p2_{mname}_s{cfg['seed']}.pt")
    model = p1.build_model(mname, p1.CFG["z_dim"]).to(DEVICE)
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"[ckpt] loading {ckpt}", flush=True)
        return model
    print(f"[train] {mname} (no ckpt yet) ...", flush=True)
    tr_ds = p1.ISICImgDataset(df_tr, ISIC_IMAGE_DIR, p1.CFG["img_size"], train=True)
    va_ds = p1.ISICImgDataset(df_va, ISIC_IMAGE_DIR, p1.CFG["img_size"], train=False)
    tr_loader = DataLoader(tr_ds, p1.CFG["batch_size"], shuffle=True,
                           num_workers=p1.CFG["num_workers"], pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, p1.CFG["batch_size"], shuffle=False,
                           num_workers=p1.CFG["num_workers"], pin_memory=True)
    n_pos = int(df_tr["target"].sum()); n_neg = len(df_tr) - n_pos
    w = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32)
    tcfg = dict(p1.CFG); tcfg["epochs"] = cfg["train_epochs"]
    p1.set_seed(cfg["seed"])
    model, _ = p1.train_one(model, tr_loader, va_loader, tcfg, w, f"p2_{mname}")
    torch.save(model.state_dict(), ckpt)
    print(f"[train] saved {ckpt}", flush=True)
    return model


def load_eval_images(df_va, n_eval, seed):
    """Fetch n_eval raw [0,1] images (balanced mal/benign)."""
    rng = np.random.RandomState(seed)
    pos = df_va[df_va["target"] == 1]
    neg = df_va[df_va["target"] == 0]
    k = n_eval // 2
    pos = pos.sample(n=min(k, len(pos)), random_state=seed)
    neg = neg.sample(n=min(k, len(neg)), random_state=seed)
    sel = (np.r_[pos.index.values, neg.index.values])
    sub = df_va.loc[sel].reset_index(drop=True)
    to_t = T.ToTensor()
    imgs, labels = [], []
    for _, row in sub.iterrows():
        img = Image.open(os.path.join(ISIC_IMAGE_DIR, f"{row['isic_id']}.jpg")).convert("RGB")
        img = p1.center_crop_resize(img, (p1.CFG["img_size"], p1.CFG["img_size"]))
        imgs.append(to_t(img))
        labels.append(int(row["target"]))
    return imgs, labels


# ============================================================
# Bootstrap CI for the mean
# ============================================================
def boot_mean_ci(vals, n_boot=2000, seed=0):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    bs = [vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(n_boot)]
    return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ============================================================
def evaluate_model(model, imgs01, labels, cfg):
    model.eval()
    cam = GradCAM(model)
    del_aucs, ins_aucs = [], []
    art = {k: [] for k in ARTIFACTS}
    rng = np.random.RandomState(cfg["seed"])
    t0 = time.time()
    for i, x01 in enumerate(imgs01):
        xn = _to_norm(x01).to(DEVICE)
        m = cam(xn.unsqueeze(0), cls=1)[0]                       # [H,W]
        da, ia = deletion_insertion(model, xn, m, cfg["di_steps"], cfg["blur_sigma"])
        del_aucs.append(da); ins_aucs.append(ia)
        d = artifact_delta(model, x01.to(DEVICE), rng)
        for k in ARTIFACTS:
            art[k].append(d[k])
        if (i + 1) % 30 == 0:
            print(f"      ...{i+1}/{len(imgs01)} images ({time.time()-t0:.0f}s)", flush=True)
    res = {}
    res["deletion_auc"] = boot_mean_ci(del_aucs, cfg["boot_n"], cfg["seed"])
    res["insertion_auc"] = boot_mean_ci(ins_aucs, cfg["boot_n"], cfg["seed"])
    all_art = np.concatenate([art[k] for k in ARTIFACTS])
    res["artifact_delta_overall"] = boot_mean_ci(all_art, cfg["boot_n"], cfg["seed"])
    res["artifact_delta_by_type"] = {k: float(np.mean(art[k])) for k in ARTIFACTS}
    return res


def main():
    sys.stdout = p1.Tee(os.path.join(LOG_DIR, "phase2_v01.log"))
    cfg = CFG2
    print(f"\n##### PHASE 2 START {time.strftime('%Y-%m-%d %H:%M:%S')} #####", flush=True)
    print(f"Device={DEVICE}  seed={cfg['seed']}  n_eval={cfg['n_eval']}", flush=True)

    results = {}
    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))

    df_tr, df_va = build_split(cfg)
    print(f"train={len(df_tr)} (pos={int(df_tr['target'].sum())})  "
          f"val={len(df_va)} (pos={int(df_va['target'].sum())})", flush=True)
    imgs01, labels = load_eval_images(df_va, cfg["n_eval"], cfg["seed"])
    print(f"eval images: {len(imgs01)} (mal={sum(labels)})", flush=True)

    for mname in cfg["models"]:
        if mname in results:
            print(f"[skip] {mname} (results already exist)", flush=True)
            continue
        print(f"\n=== Model: {mname} ===", flush=True)
        model = get_or_train(mname, df_tr, df_va, cfg)
        res = evaluate_model(model, imgs01, labels, cfg)
        results[mname] = res
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print(f"[done] {mname}: "
              f"del_auc={res['deletion_auc'][0]:.3f} "
              f"ins_auc={res['insertion_auc'][0]:.3f} "
              f"artifactΔ={res['artifact_delta_overall'][0]:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---- Comparison report ----
    print("\n" + "=" * 64)
    print("PHASE 2, Module B: Causal XAI (vs ERM baseline)")
    print("=" * 64)
    hdr = f"{'metric':<22}" + "".join(f"{m:>16}" for m in cfg["models"])
    print(hdr)
    for key, better in [("deletion_auc", "lower is better (necessity)"),
                        ("insertion_auc", "higher is better (sufficiency)"),
                        ("artifact_delta_overall", "lower is better (robust)")]:
        row = f"{key:<22}"
        for m in cfg["models"]:
            if m in results:
                v, lo, hi = results[m][key]
                row += f"{v:.3f}[{lo:.3f},{hi:.3f}]".rjust(16)
        print(row + f"   <- {better}")
    print("\nArtifact-Δ by type:")
    for m in cfg["models"]:
        if m in results:
            print(f"  {m}: " + "  ".join(
                f"{k}={results[m]['artifact_delta_by_type'][k]:.4f}" for k in ARTIFACTS))
    print(f"\nSaved: {RESULTS_FILE}")
    print(f"##### PHASE 2 END {time.strftime('%Y-%m-%d %H:%M:%S')} #####", flush=True)


if __name__ == "__main__":
    main()
