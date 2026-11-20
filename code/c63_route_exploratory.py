#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C63: exploratory follow-up of the registered route test (not pre-registered).

The registered test (c62_route_test.py) rejects, in three model-by-target cells, the joint package of
C1 for the magnitude-only footprint and latent neutrality of the controls. All three rejections come
from the scale pair rad4-rad16. At the two highest-dose targets most images fail the independent
CIELAB check of neutrality (G2), so the registered result cannot attribute the rejection to C1 alone.

This script repeats each pairwise comparison on the images on which both controls of the pair pass G1
and G2, and on which all three models score the unedited image at 0.1 or above. Everything else
follows the registered rule: the same constructions, the same models and response scale, the patient
bootstrap, a simultaneous 95% band over all analysed differences, and the 0.5-logit margin. A pair is
analysed when at least 30 images qualify. The analysis was chosen after the registered results were
seen, so it is exploratory.

It also stores every per-image response, so later analyses need no rescoring.

Reads results/c62_construction.json.
Out -> results/c63_route_exploratory.json, results/c63_route_scores.npz
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import audit_cohort                  # noqa: E402
import audited_models                # noqa: E402
import c3_cace_concept as c3m        # noqa: E402
import c10_multiplacebo as c10       # noqa: E402
import c12_matched_control as c12    # noqa: E402
import c62_route_test as c62         # noqa: E402  (registered constructions, bootstrap and rule)

p1, DEVICE = c3m.p1, c3m.DEVICE
RES = c62.RES
OUT = os.path.join(RES, "c63_route_exploratory.json")
NPZ = os.path.join(RES, "c63_route_scores.npz")
MIN_IMAGES = 30


def main():
    t0 = time.time()
    cons = json.load(open(os.path.join(RES, "c62_construction.json")))
    ids, pids = cons["ids"], np.asarray(cons["patient_ids"])
    torch.set_num_threads(1)
    imgs = audit_cohort._read(ids)
    tks = ["%s@%d" % t for t in c62.TARGETS]
    stacks = []
    for i, x in enumerate(imgs):
        items = [x]; rec = cons["images"][i]["targets"]
        for (concept, dk), tk in zip(c62.TARGETS, tks):
            cx = c62.context(x, concept, dk)
            items.append(cx["e"])
            built = c62.build_controls(x, cx, i, params={c: rec[tk]["controls"][c]["param"] for c in c62.CONTROLS})
            for c in c62.CONTROLS:
                img = built[c][0]
                mc = c12.footprint(x, img, cx["w"])
                for k in c62.MAG:
                    ref = rec[tk]["controls"][c]["measures"][k]
                    assert abs(mc[k] - ref) <= 1e-6 * max(abs(ref), 1e-9), (ids[i], tk, c, k)
                items.append(img)
        w, skin, _ = c3m.soft_mask(x)
        for fn in c10.PLACEBOS.values():
            items.append(fn(x, w, skin, np.random.RandomState(c62.SEED_PLACEBO + i + 7)))
        stacks.append(torch.stack(items))
    cols = ["unedited"] + ["%s|%s" % (tk, c) for tk in tks for c in ["edit"] + c62.CONTROLS] + \
           ["placebo|" + p for p in c10.PLACEBOS]
    col = {c: k for k, c in enumerate(cols)}
    P = {}
    for m in c62.MODELS:
        model = p1.build_model("erm" if m == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(audited_models.CKPTS[m], map_location=DEVICE))
        model.eval()
        P[m] = np.stack([c3m.model_prob(model, s) for s in stacks])
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("[%s] scored (%.0fs)" % (m, time.time() - t0), flush=True)
    DL = {m: c3m.logit(P[m]) - c3m.logit(P[m][:, :1]) for m in c62.MODELS}
    np.savez_compressed(NPZ, columns=np.array(cols), isic_ids=np.array(ids), patient_ids=pids,
                        **{"prob_" + m: P[m] for m in c62.MODELS}, **{"dlogit_" + m: DL[m] for m in c62.MODELS})
    common = np.all([P[m][:, 0] >= c62.CRIT["score_floor"] for m in c62.MODELS], 0)

    def gate(tk, c, key):
        return np.array([cons["images"][i]["targets"][tk]["controls"][c][key] for i in range(len(ids))])

    comps, vals, masks, info = [], [], [], []
    for tk in tks:
        for a, b, kind in c62.PAIRS:
            msk = gate(tk, a, "g1") & gate(tk, b, "g1") & gate(tk, a, "g2") & gate(tk, b, "g2") & common
            n, npat = int(msk.sum()), int(len(set(pids[msk])))
            for m in c62.MODELS:
                d = DL[m][:, col["%s|%s" % (tk, b)]] - DL[m][:, col["%s|%s" % (tk, a)]]
                comps.append("%s|%s|D_%s-%s" % (m, tk, a, b)); vals.append(d); masks.append(msk)
                info.append({"kind": kind, "n_images": n, "n_patients": npat, "analysed": n >= MIN_IMAGES})
    vals, masks = np.stack(vals), np.stack(masks)
    est, draws = c62.boot(vals, masks, pids, c62.CRIT["boot"], c62.CRIT["boot_seed"])
    lo, hi = np.nanpercentile(draws, 2.5, 0), np.nanpercentile(draws, 97.5, 0)
    ok = np.array([q["analysed"] for q in info])
    se = np.nanstd(draws, 0, ddof=1)
    Z = np.abs(draws[:, ok] - est[ok]) / np.maximum(se[ok], 1e-12)
    crit = float(np.nanpercentile(np.nanmax(Z, 1), c62.CRIT["level"]))
    eps = c62.CRIT["margin_logit"]
    res = {"note": "exploratory, chosen after the registered results were seen",
           "rule": "pairs on images where both controls pass G1 and G2 and all models score >= 0.1",
           "min_images": MIN_IMAGES, "crit_simultaneous": crit, "n_analysed": int(ok.sum()),
           "registered_construction_sha256": c62.sha256(os.path.join(RES, "c62_construction.json")),
           "components": {}}
    for k, c in enumerate(comps):
        r = dict(info[k]); r.update({"est": float(est[k]), "ci": [float(lo[k]), float(hi[k])]})
        if ok[k]:
            sl, sh = est[k] - crit * se[k], est[k] + crit * se[k]
            r.update({"sim": [float(sl), float(sh)], "decision_sim": c62.decide(sl, sh, eps),
                      "decision_marginal": c62.decide(lo[k], hi[k], eps)})
        res["components"][c] = r
    dec = [v["decision_sim"] for v in res["components"].values() if "decision_sim" in v]
    res["summary"] = {d: dec.count(d) for d in ("exceeds", "within", "indeterminate")}
    for kind in ("direction", "scale"):
        sub = [v for v in res["components"].values() if v["kind"] == kind and "decision_sim" in v]
        res["summary"][kind] = {d: sum(v["decision_sim"] == d for v in sub) for d in ("exceeds", "within", "indeterminate")}
    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print("crit %.3f, analysed %d of %d" % (crit, ok.sum(), len(ok)))
    for c, v in res["components"].items():
        if "decision_sim" in v:
            print("  %-34s n=%2d/%2d  D=%+.3f  sim[%+.3f,%+.3f]  %s" % (c, v["n_images"], v["n_patients"], v["est"],
                                                                     *v["sim"], v["decision_sim"]))
    print("summary", res["summary"])
    print("saved ->", OUT, "(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
